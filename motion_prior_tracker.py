"""Class-independent, initialized single-object tracking with motion priors.

One update is one uniformly sampled frame. A detector can provide the initial
normalized box; this component estimates that selected object's subsequent state.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from motion_models import make_motion_model
from tracking_types import (
    TargetSpec, TrackerConfig, TrackObservation, clamp_norm, integer, validate_bbox,
)
from template_matching import (
    crop_at, validate_image, multi_scale_match, dense_scale_match, full_image_search,
)


def imread(path):
    """Read a BGR image, including Windows paths containing Chinese characters."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except (OSError, cv2.error) as exc:
        raise ValueError(f'Cannot read image: {path}') from exc
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


class MotionPriorTracker:
    """One selected target per instance; no bird-specific class logic."""

    def __init__(self, target_spec=None, config=None):
        self.target = target_spec or TargetSpec()
        self.config = config or TrackerConfig()
        self.initialized = False
        self.frame_index = -1
        self.lost_frames = 0
        self.search_radius = (self.target.search_radius if self.config.search_radius is None
                              else self.config.search_radius)
        self.fine_margin = (self.target.fine_margin if self.config.fine_margin is None
                            else self.config.fine_margin)

    def initialize(self, image, bbox_xywh=None):
        validate_image(image)
        bbox = validate_bbox(self.target.bbox_xywh if bbox_xywh is None else bbox_xywh)
        h, w = image.shape[:2]
        bbox = clamp_norm(*bbox)
        center = np.array([bbox[0] * w, bbox[1] * h])
        size = np.array([bbox[2] * w, bbox[3] * h])
        if np.any(size < 3 - 1e-7):
            raise ValueError('initial target needs at least three pixels in each dimension')
        padding = self.target.template_padding
        target_template, target_origin = crop_at(image, center - size/2 - padding, size + 2*padding)
        ctx_size = np.maximum(20., size * (1 + 2 * self.target.context_margin))
        context_template, context_origin = crop_at(image, center - ctx_size/2, ctx_size)
        self.motion_model = make_motion_model(self.config.motion_model_name, center)
        self.target_template = target_template.copy()
        self.original_target_template = target_template.copy()
        self.context_template = context_template.copy()
        self.original_context_template = context_template.copy()
        self.target_anchor = center - target_origin
        self.context_anchor = center - context_origin
        self.ref_th, self.ref_tw = target_template.shape[:2]
        self.ctx_h, self.ctx_w = context_template.shape[:2]
        self.reference_size = size
        self.object_template = self._object_patch(image, center, 1.)
        self.original_object_template = self.object_template.copy()
        self.image_size = (w, h)
        self.image_shape = image.shape
        self.smooth_scale = 1.
        self.frame_index = 0
        self.lost_frames = 0
        self.initialized = True
        return self._observation(center, 1., 'ref')

    def _object_patch(self, image, center, scale):
        size = tuple(int(max(1, round(v * scale))) for v in self.reference_size)
        # Box coordinates describe pixel edges; OpenCV samples pixel centers.
        return cv2.getRectSubPix(image, size, tuple(float(v-.5) for v in center))

    def _observation(self, center, score, mode):
        w, h = self.image_size
        size = self.reference_size * self.smooth_scale
        box = clamp_norm(center[0]/w, center[1]/h, size[0]/w, size[1]/h)
        return TrackObservation(
            frame_index=self.frame_index, bbox_xywh=tuple(box),
            center_px=(box[0]*w, box[1]*h),
            velocity_px=tuple(float(v) for v in self.motion_model.velocity),
            score=float(np.clip(score, 0, 1)), mode=mode,
            lost_frames=self.lost_frames, class_id=self.target.class_id,
            image_size=self.image_size,
        )

    def _locate(self, image, search_center, radius, original=False):
        """A candidate requires BOTH context and target evidence."""
        context = self.original_context_template if original else self.context_template
        target = self.original_target_template if original else self.target_template
        extent = np.array([self.ctx_w, self.ctx_h])
        origin = search_center - self.context_anchor * max(self.config.context_scales) - radius
        size = extent * max(self.config.context_scales) + 2*radius
        region, region_origin = crop_at(image, origin, size)
        context_score, x, y, context_scale = multi_scale_match(
            region, context, self.ctx_w, self.ctx_h, self.config.context_scales)
        if context_score <= self.target.match_thresh:
            return None
        context_origin = region_origin + [x, y]
        approximate = context_origin + self.context_anchor * context_scale
        template_size = np.array([self.ref_tw, self.ref_th])
        fine_origin = approximate - self.target_anchor * max(self.target.scale_range) - self.fine_margin
        fine_size = template_size * max(self.target.scale_range) + 2*self.fine_margin
        fine_region, fine_origin = crop_at(image, fine_origin, fine_size)
        lo, hi = self.target.scale_range
        scales = np.unique(np.linspace(lo, hi, self.config.n_scales))
        target_score, tx, ty, scale = dense_scale_match(
            fine_region, target, self.ref_tw, self.ref_th, scales)
        if target_score <= self.target.target_thresh:
            return None
        measured_center = fine_origin + [tx, ty] + self.target_anchor * scale
        object_size = self.reference_size * scale
        if np.any(measured_center - object_size/2 < 0) or np.any(measured_center + object_size/2 > self.image_size):
            return None
        reference = self.original_object_template if original else self.object_template
        patch = self._object_patch(image, measured_center, scale)
        patch = cv2.resize(patch, (reference.shape[1], reference.shape[0])).astype(float)
        reference = reference.astype(float)
        energy = float(np.sqrt(np.sum(patch**2) * np.sum(reference**2)))
        squared_error = (patch-reference)**2
        if energy > 1e-8:
            object_score = 1 - float(np.sum(squared_error)) / energy
        else:
            # Near-black compression/interpolation noise gets a small absolute
            # tolerance. The full 255-level range would accept missing black
            # objects against unrelated medium-intensity padded backgrounds.
            object_score = 1 - float(np.sqrt(np.mean(squared_error))) / 16.
        # Padded background must never be the sole evidence for object presence.
        # Squared-difference similarity also works for solid-color object crops.
        if object_score <= self.target.target_thresh:
            return None
        return measured_center, scale, min(context_score, target_score, object_score), context_scale

    def _refresh_templates(self, image, center, target_scale, context_scale):
        rate = self.config.template_ema
        if rate == 0:
            return
        for name, anchor, size, scale in (
            ('target_template', self.target_anchor, [self.ref_tw, self.ref_th], target_scale),
            ('context_template', self.context_anchor, [self.ctx_w, self.ctx_h], context_scale),
        ):
            crop, _ = crop_at(image, center - anchor * scale, np.asarray(size) * scale)
            if crop.size:
                refreshed = cv2.resize(crop, tuple(size))
                setattr(self, name, cv2.addWeighted(getattr(self, name), 1-rate, refreshed, rate, 0))
        patch = self._object_patch(image, center, target_scale)
        refreshed = cv2.resize(patch, (self.object_template.shape[1], self.object_template.shape[0]))
        self.object_template = cv2.addWeighted(self.object_template, 1-rate, refreshed, rate, 0)

    def update(self, image, frame_index=None):
        if not self.initialized:
            raise ValueError('tracker must be initialized before update')
        validate_image(image)
        if image.shape != self.image_shape:
            raise ValueError('image dimensions/channels changed; initialize a new sequence')
        index = self.frame_index + 1 if frame_index is None else frame_index
        integer(index, 'frame_index', 1)
        if index != self.frame_index + 1:
            raise ValueError('frame_index must advance by one; use uniformly sampled frames')
        prediction = np.asarray(self.motion_model.predict(), dtype=float)
        candidate = self._locate(image, prediction, self.search_radius)
        if candidate is None and (self.lost_frames + 1) % self.config.reacq_interval == 0:
            score, x, y, _ = full_image_search(
                image, self.original_context_template, self.ctx_w, self.ctx_h,
                self.config.reacq_scale)
            if score > self.target.match_thresh * 0.7:
                candidate = self._locate(
                    image, np.array([x, y]) + self.context_anchor,
                    max(self.fine_margin, int(2/self.config.reacq_scale)), original=True)
        if candidate is None:
            center = self.motion_model.advance()
            self.lost_frames += 1
            mode, score = 'predicted', 0.
        else:
            measurement, scale, score, context_scale = candidate
            center = self.motion_model.update(measurement)
            self.smooth_scale = (self.config.scale_ema * scale
                                 + (1-self.config.scale_ema) * self.smooth_scale)
            mode = 'recovered' if self.lost_frames else 'matched'
            self.lost_frames = 0
            if score >= self.config.template_update_thresh:
                self._refresh_templates(image, measurement, scale, context_scale)
        self.frame_index = int(index)
        return self._observation(center, score, mode)

    def track_sequence(self, images, telemetry_path=None):
        """Process an iterable; an initialized tracker consumes every new frame."""
        observations = []
        for image in images:
            observations.append(self.update(image) if self.initialized else self.initialize(image))
        if not observations:
            raise ValueError('images must contain at least one frame')
        path = telemetry_path if telemetry_path is not None else self.config.telemetry_path
        if path is not None:
            self.write_jsonl(observations, path)
        return observations

    @staticmethod
    def write_jsonl(observations, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream:
            for observation in observations:
                stream.write(json.dumps(observation.to_dict(), ensure_ascii=False, allow_nan=False) + '\n')

    @staticmethod
    def write_yolo(observation, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{observation.class_id} " + ' '.join(f'{v:.6f}' for v in observation.bbox_xywh) + '\n', encoding='utf-8')


def process_folder(folder_path, output_base, target_spec=None, config=None, telemetry_path=None):
    from sequence_io import process_folder as run_folder
    return run_folder(folder_path, output_base, target_spec, config, telemetry_path)


def main(argv=None):
    from tracking_cli import main as run_cli
    return run_cli(argv)


if __name__ == '__main__':
    raise SystemExit(main())
