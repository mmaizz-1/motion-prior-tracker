"""Regression tests use real OpenCV images, with known object geometry."""
import json

import cv2
import numpy as np
import pytest

from motion_prior_tracker import (
    MotionPriorTracker, TargetSpec, TrackerConfig, process_folder,
)


def scene(x=50, y=35, visible=True, width=180, height=100):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    if visible:
        patch = np.random.default_rng(82).integers(40, 255, (12, 16, 3), dtype=np.uint8)
        image[y:y + 12, x:x + 16] = patch
    return image


def tracker(**kwargs):
    return MotionPriorTracker(
        TargetSpec(class_id=8, bbox_xywh=(58/180, 41/100, 16/180, 12/100),
                   context_margin=0.5, search_radius=15, fine_margin=12,
                   scale_range=(1., 1.), match_thresh=0.8, target_thresh=0.8),
        TrackerConfig(**kwargs),
    )


def test_observation_image_size_and_center_share_width_height_convention():
    observation = tracker().initialize(scene())
    assert observation.image_size == (180, 100)
    assert observation.center_px == pytest.approx((58, 41))


def test_initialized_sequence_consumes_every_new_frame():
    t = tracker()
    t.initialize(scene())
    observations = t.track_sequence(iter([scene(53), scene(56)]))
    assert [o.frame_index for o in observations] == [1, 2]
    assert observations[-1].center_px == pytest.approx((64, 41), abs=1)


def test_long_loss_never_reports_recovery_on_blank_frame():
    t = tracker(reacq_interval=1)
    t.initialize(scene())
    t.update(scene(53))
    for count in range(1, 7):
        observation = t.update(scene(visible=False))
        assert observation.mode == 'predicted'
        assert observation.lost_frames == count
        assert observation.score == 0


def test_recovers_object_outside_local_search_after_occlusion():
    t = tracker(reacq_interval=1, reacq_scale=0.5)
    t.initialize(scene())
    t.update(scene(visible=False))
    recovered = t.update(scene(135))
    assert recovered.mode == 'recovered'
    assert recovered.center_px == pytest.approx((143, 41), abs=1)
    assert recovered.lost_frames == 0


def test_large_reference_context_is_clipped_to_image_before_matching():
    image = scene()
    t = MotionPriorTracker(TargetSpec(bbox_xywh=(.5, .5, .9, .9)))
    t.initialize(image)
    result = t.update(image)
    assert result.mode == 'matched'
    assert result.center_px == pytest.approx((90, 50), abs=2)


def test_offscreen_prediction_keeps_bbox_and_pixel_center_consistent():
    t = tracker()
    t.initialize(scene())
    t.motion_model.update([1000., 41.])
    result = t.update(scene(visible=False))
    cx, cy, w, h = result.bbox_xywh
    assert 0 <= cx - w/2 <= cx + w/2 <= 1
    assert 0 <= cy - h/2 <= cy + h/2 <= 1
    assert result.center_px == pytest.approx((cx * 180, cy * 100))
    assert result.mode == 'predicted'


@pytest.mark.parametrize('kwargs', [dict(reacq_interval=0), dict(template_ema=2),
                                  dict(reacq_scale=0), dict(context_scales=())])
def test_bad_configuration_fails_at_construction(kwargs):
    with pytest.raises(ValueError):
        TrackerConfig(**kwargs)


@pytest.mark.parametrize('image', [None, np.empty((0, 20, 3), dtype=np.uint8)])
def test_empty_update_rejected_without_advancing_state(image):
    t = tracker()
    t.initialize(scene())
    with pytest.raises(ValueError):
        t.update(image)
    assert t.frame_index == 0


def test_telemetry_writer_creates_parents_and_writes_finite_json(tmp_path):
    path = tmp_path / 'nested' / 'track.jsonl'
    observations = tracker().track_sequence([scene(), scene(53)], telemetry_path=path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len(observations) == 2
    assert rows[1]['image_size'] == [180, 100]
    assert rows[1]['center_px'] == pytest.approx([61, 41], abs=1)


def test_folder_override_preserves_reference_and_measures_runtime(tmp_path):
    folder = tmp_path / 'source' / '目标'
    folder.mkdir(parents=True)
    for index, x in [(1, 50), (2, 53)]:
        cv2.imencode('.png', scene(x))[1].tofile(str(folder / f'{index}.png'))
    (folder / '1.txt').write_text('7 0.3222222222 0.41 0.0888888889 0.12\n')
    stats = process_folder(folder, tmp_path / 'out', target_spec=TargetSpec(class_id=3))
    assert stats['total'] == 2
    assert stats['elapsed'] > 0 and stats['fps'] > 0
    assert stats['matched'] + stats['predicted'] == stats['total']
    assert (tmp_path / 'out' / '目标' / '1.txt').read_text().startswith('3 ')


def test_background_match_without_target_does_not_reacquire():
    background = np.random.default_rng(92).integers(0, 70, (100, 180, 3), dtype=np.uint8)
    reference = background.copy()
    reference[35:47, 50:66] = scene()[35:47, 50:66]
    t = MotionPriorTracker(
        TargetSpec(bbox_xywh=(58/180, .41, 16/180, .12), target_thresh=.9),
        TrackerConfig(reacq_interval=1),
    )
    t.initialize(reference)
    for _ in range(3):
        result = t.update(background)
        assert result.mode == 'predicted'
        assert result.score == 0


def test_predicted_box_keeps_last_measured_scale_and_excludes_padding():
    t = MotionPriorTracker(
        TargetSpec(bbox_xywh=(58/180, .41, 16/180, .12), context_margin=.5,
                   scale_range=(1., 2.)),
        TrackerConfig(scale_ema=1., template_ema=0., context_scales=(1., 2.)),
    )
    t.initialize(scene())
    large = np.zeros((100, 180, 3), dtype=np.uint8)
    large[29:53, 42:74] = cv2.resize(scene()[35:47, 50:66], (32, 24))
    observed = t.update(large)
    assert observed.mode == 'matched'
    assert observed.bbox_xywh[2:] == pytest.approx((32/180, .24), abs=.015)
    predicted = t.update(scene(visible=False))
    assert predicted.mode == 'predicted'
    assert predicted.bbox_xywh[2:] == observed.bbox_xywh[2:]


def test_stale_index_and_resized_image_fail_before_advancing():
    t = tracker()
    t.initialize(scene())
    for frame_index in (0, 2, -1):
        with pytest.raises(ValueError):
            t.update(scene(), frame_index=frame_index)
    with pytest.raises(ValueError):
        t.update(scene(width=200))
    assert t.frame_index == 0


def test_spatially_constant_template_never_claims_perfect_match():
    from template_matching import multi_scale_match
    template = np.full((10, 10, 3), (20, 100, 200), dtype=np.uint8)
    score, _, _, _ = multi_scale_match(scene(), template, 10, 10, [1.])
    assert score <= 0


def test_default_threshold_does_not_recover_from_background_noise():
    background = np.random.default_rng(92).integers(0, 70, (100, 180, 3), dtype=np.uint8)
    reference = background.copy()
    reference[35:47, 50:66] = scene()[35:47, 50:66]
    t = MotionPriorTracker(TargetSpec(bbox_xywh=(58/180, .41, 16/180, .12)),
                           TrackerConfig(reacq_interval=1))
    t.initialize(reference)
    t.update(np.zeros_like(reference))
    result = t.update(background)
    assert result.mode == 'predicted'
    assert result.lost_frames == 2


def test_scale_growth_search_includes_left_and_top_with_zero_margin():
    texture = np.random.default_rng(6).integers(30, 245, (20, 20, 3), dtype=np.uint8)
    first = np.zeros((160, 200, 3), dtype=np.uint8)
    first[70:90, 90:110] = texture
    enlarged = np.zeros_like(first)
    enlarged[60:100, 80:120] = cv2.resize(texture, (40, 40))
    t = MotionPriorTracker(
        TargetSpec(bbox_xywh=(.5, .5, .1, .125), context_margin=0.,
                   template_padding=0, scale_range=(1., 2.), fine_margin=0),
        TrackerConfig(context_scales=(1., 2.), scale_ema=1., template_ema=0.),
    )
    t.initialize(first)
    result = t.update(enlarged)
    assert result.mode == 'matched'
    assert result.center_px == pytest.approx((100, 80), abs=1)
    assert result.bbox_xywh[2:] == pytest.approx((.2, .25), abs=.01)


@pytest.mark.parametrize('size', [3, 4])
def test_padding_cannot_validate_missing_tiny_object(size):
    background = np.random.default_rng(92).integers(0, 180, (100, 180, 3), dtype=np.uint8)
    reference = background.copy()
    reference[35:35+size, 50:50+size] = np.random.default_rng(82).integers(160, 255, (size, size, 3), dtype=np.uint8)
    t = MotionPriorTracker(TargetSpec(bbox_xywh=((50+size/2)/180, (35+size/2)/100, size/180, size/100)),
                           TrackerConfig(reacq_interval=1))
    t.initialize(reference)
    t.update(np.zeros_like(reference))
    assert t.update(background).mode == 'predicted'


def test_sub_three_pixel_target_rejected_as_insufficient_template_support():
    t = MotionPriorTracker(TargetSpec(bbox_xywh=(.5, .5, 2/180, 2/100)))
    with pytest.raises(ValueError, match='three pixels'):
        t.initialize(scene())


def test_direct_folder_api_does_not_overwrite_reference_with_telemetry(tmp_path):
    folder = tmp_path / 'source'
    folder.mkdir()
    cv2.imencode('.png', scene())[1].tofile(str(folder/'1.png'))
    label = folder/'1.txt'
    label.write_text('7 0.32222222 0.41 0.08888889 0.12\n')
    before = label.read_bytes()
    with pytest.raises(ValueError):
        process_folder(folder, tmp_path/'out', telemetry_path=label)
    assert label.read_bytes() == before
    assert not (tmp_path/'out').exists()


def test_duplicate_image_stems_rejected_before_output(tmp_path):
    folder = tmp_path/'source'
    folder.mkdir()
    for extension in ('.jpg', '.png'):
        cv2.imencode(extension, scene())[1].tofile(str(folder/f'1{extension}'))
    (folder/'1.txt').write_text('7 0.32222222 0.41 0.08888889 0.12\n')
    with pytest.raises(ValueError, match='stem'):
        process_folder(folder, tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_black_object_can_match_but_missing_black_object_cannot():
    image = np.full((100, 180, 3), 220, dtype=np.uint8)
    image[35:47, 50:66] = 0
    t = MotionPriorTracker(TargetSpec(bbox_xywh=(58/180, .41, 16/180, .12), scale_range=(1., 1.)),
                           TrackerConfig(template_ema=0., reacq_interval=1))
    t.initialize(image)
    assert t.update(image).mode == 'matched'
    moved = np.full_like(image, 220)
    moved[35:47, 53:69] = 0
    result = t.update(moved)
    assert result.mode == 'matched'
    assert result.center_px == pytest.approx((61, 41), abs=1)
    assert t.update(np.full_like(image, 220)).mode == 'predicted'


@pytest.mark.parametrize('size', [3, 4, 8])
def test_black_object_removed_from_textured_background_is_not_a_match(size):
    background = np.random.default_rng(92).integers(0, 180, (100, 180, 3), dtype=np.uint8)
    reference = background.copy()
    reference[35:35+size, 50:50+size] = 0
    t = MotionPriorTracker(TargetSpec(bbox_xywh=((50+size/2)/180, (35+size/2)/100, size/180, size/100)),
                           TrackerConfig(template_ema=0., reacq_interval=1))
    t.initialize(reference)
    assert t.update(reference).mode == 'matched'
    assert t.update(background).mode == 'predicted'
