"""Image sequence I/O. Decode one frame at a time, including 4K frames."""
from dataclasses import replace
from pathlib import Path
import json
import re
from time import perf_counter

import numpy as np

from tracking_types import TargetSpec, TrackerConfig


def image_paths(folder):
    paths = [p for p in Path(folder).iterdir()
             if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
    stems = [p.stem.casefold() for p in paths]
    if len(set(stems)) != len(stems):
        raise ValueError(f'Duplicate image stem in {folder}; YOLO outputs would collide')
    def key(path):
        return [int(part) if part.isdigit() else part.lower()
                for part in re.split(r'(\d+)', path.name)]
    return sorted(paths, key=key)


def process_folder(folder_path, output_base, target_spec=None, config=None, telemetry_path=None):
    from motion_prior_tracker import MotionPriorTracker, imread
    folder = Path(folder_path)
    paths = image_paths(folder)
    if not paths:
        print(f'[{folder.name}] No images, skipping')
        return None
    if target_spec is None or target_spec.bbox_xywh is None:
        reference = paths[0].with_suffix('.txt')
        if not reference.exists():
            print(f'[{folder.name}] No reference txt, skipping')
            return None
        rows = [line.split() for line in reference.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0]) != 5:
            raise ValueError(f'{reference}: need exactly one YOLO reference target')
        try:
            label = TargetSpec(class_id=int(rows[0][0]), bbox_xywh=tuple(map(float, rows[0][1:])))
        except (ValueError, TypeError) as exc:
            raise ValueError(f'Invalid reference label: {reference}') from exc
        target_spec = label if target_spec is None else replace(target_spec, bbox_xywh=label.bbox_xywh)
    config = config or TrackerConfig()
    tracker = MotionPriorTracker(target_spec, config)
    output = Path(output_base) / folder.name
    source_path, output_path = folder.resolve(), output.resolve()
    if output_path == source_path or source_path in output_path.parents or output_path in source_path.parents:
        raise ValueError('output must not overwrite source annotations')
    telemetry = telemetry_path if telemetry_path is not None else config.telemetry_path
    if telemetry is not None:
        destination = Path(telemetry).resolve()
        if destination == source_path or source_path in destination.parents:
            raise ValueError('telemetry must not overwrite source annotations')
        reserved = {output_path / (p.stem + '.txt') for p in paths}
        reserved.add(output_path / '_tracking_stats.json')
        if destination in reserved:
            raise ValueError('telemetry conflicts with generated output')
    output.mkdir(parents=True, exist_ok=True)
    observations = []
    started = perf_counter()
    for path in paths:
        image = imread(path)
        observation = tracker.update(image) if tracker.initialized else tracker.initialize(image)
        observations.append(observation)
        tracker.write_yolo(observation, output / (path.stem + '.txt'))
    elapsed = perf_counter() - started
    if telemetry is not None:
        tracker.write_jsonl(observations, telemetry)
    widths = [o.bbox_xywh[2] for o in observations]
    heights = [o.bbox_xywh[3] for o in observations]
    measured = sum(o.measurement_valid for o in observations)
    stats = {
        'folder': folder.name, 'total': len(observations),
        'matched': measured, 'predicted': len(observations)-measured,
        'recovered': sum(o.mode == 'recovered' for o in observations),
        'match_rate': measured / len(observations),
        'elapsed': elapsed, 'fps': len(observations)/elapsed if elapsed else 0.,
        'w_mean': float(np.mean(widths)), 'w_std': float(np.std(widths)),
        'h_mean': float(np.mean(heights)), 'h_std': float(np.std(heights)),
        'w_unique': len(set(round(v, 8) for v in widths)),
        'h_unique': len(set(round(v, 8) for v in heights)),
        'frame_diffs': float(np.mean(np.abs(np.diff(widths)))) if len(widths) > 1 else 0.,
    }
    (output / '_tracking_stats.json').write_text(json.dumps(stats, indent=2, allow_nan=False), encoding='utf-8')
    return stats
