"""Generic template tracker with pluggable motion priors."""
import argparse
import glob
import json
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
try:
    import cv2
except ImportError:  # array-only use does not require OpenCV
    cv2 = None
from motion_models import make_motion_model

INPUT_BASE = r"C:\Users\Lenovo\Desktop\标注1"
OUTPUT_BASE = r"C:\Users\Lenovo\Desktop\标注1_annotations"
BIRD_CLASS = 15
MOTION_MODEL = "ConstantVelocity"
CONTEXT_MARGIN, SEARCH_RADIUS, BIRD_PAD, FINE_MARGIN = 2, 200, 4, 80
N_SCALES, SCALE_RANGE, SCALE_EMA = 50, (0.5, 1.5), 0.35
MATCH_THRESH, BIRD_THRESH, TEMPLATE_EMA = 0.35, 0.15, 0.3
TARGET_THRESH, REACQ_INTERVAL, REACQ_SCALE = BIRD_THRESH, 10, 0.25
DENSE_SCALES = np.linspace(*SCALE_RANGE, N_SCALES)


def _validate_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if bbox is None or len(bbox) != 4:
        raise ValueError("bbox_xywh must contain four normalized values")
    values = tuple(float(v) for v in bbox)
    cx, cy, w, h = values
    if not all(np.isfinite(values)) or w <= 0 or h <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and w <= 1 and h <= 1):
        raise ValueError("bbox_xywh must contain finite normalized values and positive dimensions")
    return values


@dataclass
class TargetSpec:
    class_id: int = 0
    bbox_xywh: Optional[Sequence[float]] = None
    initial_bbox_xywh: Optional[Sequence[float]] = None
    template_padding: int = BIRD_PAD
    context_margin: float = CONTEXT_MARGIN
    search_radius: int = SEARCH_RADIUS
    fine_margin: int = FINE_MARGIN
    scale_range: Tuple[float, float] = SCALE_RANGE
    match_thresh: float = MATCH_THRESH
    target_thresh: float = TARGET_THRESH

    def __post_init__(self):
        if self.bbox_xywh is None:
            self.bbox_xywh = self.initial_bbox_xywh
        if self.bbox_xywh is not None:
            self.bbox_xywh = _validate_bbox(self.bbox_xywh)
        if int(self.class_id) < 0 or self.template_padding < 0 or self.search_radius < 0 or self.fine_margin < 0:
            raise ValueError("class_id and dimensions must be non-negative")
        if self.scale_range[0] <= 0 or self.scale_range[1] < self.scale_range[0]:
            raise ValueError("scale_range must be positive and ordered")


@dataclass
class TrackerConfig:
    motion_model_name: str = MOTION_MODEL
    template_ema: float = TEMPLATE_EMA
    reacq_interval: int = REACQ_INTERVAL
    reacq_scale: float = REACQ_SCALE
    context_scales: Tuple[float, ...] = (0.88, 0.94, 1.0, 1.06, 1.12)
    telemetry_path: Optional[str] = None
    yolo_output: bool = False
    search_radius: Optional[int] = None
    fine_margin: Optional[int] = None

    @property
    def model_name(self):
        return self.motion_model_name


@dataclass(frozen=True)
class TrackObservation:
    frame_index: int
    bbox_xywh: Tuple[float, float, float, float]
    center_px: Tuple[float, float]
    velocity_px: Tuple[float, float]
    score: float
    mode: str
    lost_frames: int
    class_id: int
    image_size: Tuple[int, int]

    def to_dict(self):
        return {"frame_index": int(self.frame_index), "bbox_xywh": [float(v) for v in self.bbox_xywh],
                "center_px": [float(v) for v in self.center_px], "velocity_px": [float(v) for v in self.velocity_px],
                "score": float(self.score), "mode": self.mode, "lost_frames": int(self.lost_frames),
                "class_id": int(self.class_id), "image_size": [int(v) for v in self.image_size]}


def imread(path):
    if cv2 is None:
        raise RuntimeError("opencv-python is required to read image paths")
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def clamp_norm(cx, cy, bw, bh):
    return [max(0.0003, min(0.9997, float(cx))), max(0.0003, min(0.9997, float(cy))),
            max(0.0002, min(0.9998, float(bw))), max(0.0002, min(0.9998, float(bh)))]


def _resize(image, size):
    width, height = map(int, size)
    if cv2 is not None:
        return cv2.resize(image, (width, height))
    ys = np.linspace(0, image.shape[0] - 1, height).astype(int)
    xs = np.linspace(0, image.shape[1] - 1, width).astype(int)
    return image[np.ix_(ys, xs)]


def _match_template(search, template):
    if search.size == 0 or template.size == 0:
        return -1.0, (0, 0)
    if cv2 is not None:
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, value, _, loc = cv2.minMaxLoc(result)
        return float(value), loc
    a = search.mean(2) if search.ndim == 3 else search
    b = template.mean(2) if template.ndim == 3 else template
    th, tw = b.shape[:2]
    if th > a.shape[0] or tw > a.shape[1]:
        return -1.0, (0, 0)
    b = b.astype(float) - b.mean(); nb = np.linalg.norm(b)
    if nb < 1e-9:
        return 0.0, (0, 0)
    best, loc = -1.0, (0, 0)
    for y in range(a.shape[0] - th + 1):
        for x in range(a.shape[1] - tw + 1):
            p = a[y:y + th, x:x + tw].astype(float); p -= p.mean(); den = np.linalg.norm(p) * nb
            score = float((p * b).sum() / den) if den > 1e-9 else 0.0
            if score > best:
                best, loc = score, (x, y)
    return best, loc


def multi_scale_match(search_region, template, base_w, base_h, scales):
    best = (-1.0, 0, 0, 1.0)
    sh, sw = search_region.shape[:2]
    for scale in scales:
        tw, th = max(3, int(base_w * scale)), max(3, int(base_h * scale))
        if tw > sw or th > sh:
            continue
        score, (x, y) = _match_template(search_region, _resize(template, (tw, th)))
        if score > best[0]:
            best = (score, x, y, float(scale))
    return best


def dense_scale_match(search_region, template, base_w, base_h, scales=None):
    scales = DENSE_SCALES if scales is None else np.asarray(scales)
    scores, best_score, best_idx, best_loc = [], -1.0, 0, (0, 0)
    sh, sw = search_region.shape[:2]
    for i, scale in enumerate(scales):
        tw, th = max(3, int(base_w * scale)), max(3, int(base_h * scale))
        if tw > sw or th > sh:
            scores.append(-1.0); continue
        score, loc = _match_template(search_region, _resize(template, (tw, th)))
        scores.append(score)
        if score > best_score:
            best_score, best_idx, best_loc = score, i, loc
    interp = float(scales[best_idx]) if len(scales) else 1.0
    if 0 < best_idx < len(scales) - 1:
        s0, s1, s2 = scores[best_idx - 1:best_idx + 2]; denom = 2 * (2 * s1 - s0 - s2)
        if s1 > s0 and s1 > s2 and abs(denom) > 1e-8:
            interp += float((s2 - s0) / denom) * float(scales[1] - scales[0])
    return best_score, best_loc[0], best_loc[1], interp


def full_image_search(img, template, base_w, base_h, search_scale):
    H, W = img.shape[:2]
    small = _resize(img, (max(1, W * search_scale), max(1, H * search_scale)))
    tmpl = _resize(template, (max(3, base_w * search_scale), max(3, base_h * search_scale)))
    score, (x, y) = _match_template(small, tmpl)
    return (score, int(x / search_scale), int(y / search_scale), 1.0) if score > MATCH_THRESH * 0.7 else (score, 0, 0, 1.0)


class MotionPriorTracker:
    def __init__(self, target_spec: TargetSpec, config: Optional[TrackerConfig] = None):
        self.target, self.config = target_spec, config or TrackerConfig()
        self.initialized, self.frame_index, self.lost_frames = False, -1, 0

    def initialize(self, image, bbox_xywh=None):
        if image is None or not hasattr(image, "shape") or image.ndim < 2:
            raise ValueError("image must be a non-empty array")
        H, W = image.shape[:2]; bbox = _validate_bbox(bbox_xywh if bbox_xywh is not None else self.target.bbox_xywh)
        if bbox_xywh is not None:
            self.target.bbox_xywh = bbox
        cx, cy, bw, bh = bbox; px_cx, px_cy = cx * W, cy * H
        self.image_size, self.ref_bbox = (W, H), bbox
        self.target_w, self.target_h = max(3, int(bw * W)), max(3, int(bh * H)); pad = int(self.target.template_padding)
        x1, y1 = max(0, int(px_cx - self.target_w / 2) - pad), max(0, int(px_cy - self.target_h / 2) - pad)
        x2, y2 = min(W, int(px_cx + self.target_w / 2) + pad), min(H, int(px_cy + self.target_h / 2) + pad)
        self.target_template = image[y1:y2, x1:x2].copy()
        self.ref_th, self.ref_tw = self.target_template.shape[:2]
        self.ctx_w, self.ctx_h = max(20, int(self.target_w * (1 + 2 * self.target.context_margin))), max(20, int(self.target_h * (1 + 2 * self.target.context_margin)))
        cx1, cy1 = max(0, min(int(px_cx - self.ctx_w / 2), W - self.ctx_w)), max(0, min(int(px_cy - self.ctx_h / 2), H - self.ctx_h))
        self.target_offset = np.array([px_cx - (cx1 + self.ctx_w / 2), px_cy - (cy1 + self.ctx_h / 2)])
        self.context_template = image[cy1:cy1 + self.ctx_h, cx1:cx1 + self.ctx_w].copy()
        self.original_context_template = self.context_template.copy()
        self.motion_model = make_motion_model(self.config.motion_model_name, [cx1 + self.ctx_w / 2, cy1 + self.ctx_h / 2])
        self.smooth_scale, self.frame_index, self.lost_frames, self.initialized = 1.0, 0, 0, True
        return self._observation(0, bbox, [px_cx, px_cy], 1.0, "ref")

    def _observation(self, index, bbox, center, score, mode):
        velocity = np.asarray(self.motion_model.velocity, dtype=float)
        return TrackObservation(index, tuple(clamp_norm(*bbox)), (float(center[0]), float(center[1])), (float(velocity[0]), float(velocity[1])), float(max(0.0, score)), mode, self.lost_frames, int(self.target.class_id), tuple(self.image_size[::-1]))

    def update(self, image, frame_index=None):
        if not self.initialized:
            raise ValueError("tracker must be initialized before update")
        H, W = image.shape[:2]; idx = self.frame_index + 1 if frame_index is None else int(frame_index)
        pred = np.asarray(self.motion_model.predict(), dtype=float)
        search_radius = self.config.search_radius if self.config.search_radius is not None else self.target.search_radius
        fine_margin = self.config.fine_margin if self.config.fine_margin is not None else self.target.fine_margin
        sx1, sy1 = max(0, int(pred[0] - self.ctx_w / 2 - search_radius)), max(0, int(pred[1] - self.ctx_h / 2 - search_radius))
        sx2, sy2 = min(W, int(pred[0] + self.ctx_w / 2 + search_radius)), min(H, int(pred[1] + self.ctx_h / 2 + search_radius))
        context_score, lx, ly, cscale = multi_scale_match(image[sy1:sy2, sx1:sx2], self.context_template, self.ctx_w, self.ctx_h, self.config.context_scales)
        center, out_w, out_h, target_score, mode = pred + self.target_offset, self.target_w, self.target_h, -1.0, "predicted"
        matched = False
        if context_score >= self.target.match_thresh:
            ctx_x, ctx_y = sx1 + lx, sy1 + ly; approx = np.array([ctx_x + self.ctx_w * cscale / 2 + self.target_offset[0] * cscale, ctx_y + self.ctx_h * cscale / 2 + self.target_offset[1] * cscale])
            fx1, fy1, fx2, fy2 = max(0, int(approx[0] - fine_margin)), max(0, int(approx[1] - fine_margin)), min(W, int(approx[0] + fine_margin)), min(H, int(approx[1] + fine_margin))
            target_score, tx, ty, raw = dense_scale_match(image[fy1:fy2, fx1:fx2], self.target_template, self.ref_tw, self.ref_th, np.linspace(*self.target.scale_range, N_SCALES))
            if target_score >= self.target.target_thresh:
                matched = True; self.smooth_scale = max(self.target.scale_range[0], min(self.target.scale_range[1], self.config.template_ema * raw + (1 - self.config.template_ema) * self.smooth_scale)); out_w, out_h = self.target_w * self.smooth_scale, self.target_h * self.smooth_scale
                center = np.array([fx1 + tx + self.ref_tw * raw / 2, fy1 + ty + self.ref_th * raw / 2]); self.motion_model.update(np.array([ctx_x + self.ctx_w * cscale / 2, ctx_y + self.ctx_h * cscale / 2])); mode = "recovered" if self.lost_frames else "matched"; self.lost_frames = 0
        if not matched and self.lost_frames > 0 and self.config.reacq_interval > 0 and self.lost_frames % self.config.reacq_interval == 0:
            original = getattr(self, "original_context_template", self.context_template)
            reacq_score, rx, ry, rs = full_image_search(image, original, self.ctx_w, self.ctx_h, self.config.reacq_scale)
            if reacq_score >= self.target.match_thresh * 0.7:
                fx1, fy1 = max(0, rx - fine_margin), max(0, ry - fine_margin)
                fx2, fy2 = min(W, rx + self.ctx_w + fine_margin), min(H, ry + self.ctx_h + fine_margin)
                score2, lx2, ly2, scale2 = multi_scale_match(image[fy1:fy2, fx1:fx2], original, self.ctx_w, self.ctx_h, self.config.context_scales)
                if score2 >= context_score:
                    ctx_x, ctx_y = fx1 + lx2, fy1 + ly2
                    center = np.array([ctx_x + self.ctx_w * scale2 / 2 + self.target_offset[0] * scale2, ctx_y + self.ctx_h * scale2 / 2 + self.target_offset[1] * scale2])
                    context_score = score2
                    self.motion_model.update(np.array([ctx_x + self.ctx_w * scale2 / 2, ctx_y + self.ctx_h * scale2 / 2]))
                    self.lost_frames = 0
                    mode = "recovered"
                    matched = True
        if not matched:
            self.motion_model.advance(); self.lost_frames += 1
        self.frame_index, self.image_size = idx, (W, H)
        return self._observation(idx, (center[0] / W, center[1] / H, out_w / W, out_h / H), center, max(context_score, target_score), mode)

    def track_sequence(self, images: Iterable, telemetry_path=None):
        images = list(images)
        if not images:
            raise ValueError("images must contain at least one frame")
        observations = [self.initialize(images[0])] if not self.initialized else []
        observations.extend(self.update(image, i) for i, image in enumerate(images[1:], 1))
        path = telemetry_path or self.config.telemetry_path
        if path: self.write_jsonl(observations, path)
        return observations

    @staticmethod
    def write_jsonl(observations, path):
        with open(path, "w", encoding="utf-8") as stream:
            for observation in observations: stream.write(json.dumps(observation.to_dict(), ensure_ascii=False) + "\n")

    def write_yolo(self, observation, path):
        with open(path, "w", encoding="utf-8") as stream: stream.write(f"{observation.class_id} {' '.join(f'{v:.6f}' for v in observation.bbox_xywh)}\n")


def process_folder(folder_path, output_base, target_spec=None, config=None, telemetry_path=None):
    folder_name = os.path.basename(folder_path); images = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    if not images: print(f"  [{folder_name}] No images, skipping"); return None
    first = os.path.splitext(os.path.basename(images[0]))[0]; ref_txt = os.path.join(folder_path, first + ".txt")
    if not os.path.exists(ref_txt): print(f"  [{folder_name}] No reference txt, skipping"); return None
    with open(ref_txt, encoding="utf-8") as stream: parts = stream.readline().strip().split()
    if len(parts) < 5: raise ValueError(f"invalid reference label: {ref_txt}")
    spec = target_spec or TargetSpec(class_id=int(parts[0]), bbox_xywh=[float(v) for v in parts[1:5]])
    tracker = MotionPriorTracker(spec, config or TrackerConfig(telemetry_path=telemetry_path)); observations = tracker.track_sequence([imread(path) for path in images])
    out_folder = os.path.join(output_base, folder_name); os.makedirs(out_folder, exist_ok=True)
    for path, observation in zip(images, observations): tracker.write_yolo(observation, os.path.join(out_folder, os.path.splitext(os.path.basename(path))[0] + ".txt"))
    if telemetry_path: tracker.write_jsonl(observations, telemetry_path)
    widths, heights = [o.bbox_xywh[2] for o in observations], [o.bbox_xywh[3] for o in observations]
    return {"folder": folder_name, "total": len(observations), "matched": sum(o.mode == "matched" for o in observations), "predicted": sum(o.mode == "predicted" for o in observations), "recovered": sum(o.mode == "recovered" for o in observations), "match_rate": sum(o.mode == "matched" for o in observations) / len(observations), "elapsed": 0.0, "fps": 0.0, "w_mean": float(np.mean(widths)), "w_std": float(np.std(widths)), "h_mean": float(np.mean(heights)), "h_std": float(np.std(heights)), "w_unique": len(set(round(v, 8) for v in widths)), "h_unique": len(set(round(v, 8) for v in heights)), "frame_diffs": float(np.mean(np.abs(np.diff(widths)))) if len(widths) > 1 else 0.0}


def main():
    parser = argparse.ArgumentParser(description="Motion-Prior Tracker"); parser.add_argument("--input", default=INPUT_BASE); parser.add_argument("--output", default=OUTPUT_BASE); parser.add_argument("--model", default=MOTION_MODEL); args = parser.parse_args()
    for folder in sorted(path for path in glob.glob(os.path.join(args.input, "*")) if os.path.isdir(path)): process_folder(folder, args.output, config=TrackerConfig(motion_model_name=args.model))


if __name__ == "__main__": main()
