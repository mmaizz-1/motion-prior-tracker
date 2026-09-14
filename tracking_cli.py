"""Command-line entry point for initialized single-object image tracking."""
import argparse
import json
from pathlib import Path
import re
import sys
import tempfile

from motion_models import make_motion_model
from sequence_io import image_paths, process_folder
from tracking_types import TargetSpec, TrackerConfig


def build_parser():
    parser = argparse.ArgumentParser(
        description='Track one initialized target per image sequence. The first image needs one YOLO reference label.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', required=True, type=Path,
                        help='one image sequence folder, or a parent containing sequence folders')
    parser.add_argument('--output', required=True, type=Path,
                        help='output root; labels are written to OUTPUT/sequence_name')
    parser.add_argument('--model', default=TrackerConfig().motion_model_name,
                        help='ConstantVelocity, KalmanFilter, or ConstantAcceleration')
    parser.add_argument('--class-id', type=int, default=None,
                        help='override label class ID; omitted preserves each reference label class')
    parser.add_argument('--telemetry', type=Path, default=None,
                        help='aggregate JSONL path with sequence, source, and image_name identifiers')
    defaults = TargetSpec()
    for name, value_type, help_text in (
        ('search_radius', int, 'coarse search radius in pixels'),
        ('fine_margin', int, 'fine search margin in pixels'),
        ('match_thresh', float, 'context similarity threshold, between 0 and 1'),
        ('target_thresh', float, 'target similarity threshold, between 0 and 1'),
    ):
        parser.add_argument('--' + name.replace('_', '-'), type=value_type,
                            default=getattr(defaults, name), help=help_text)
    defaults = TrackerConfig()
    for name, value_type, help_text in (
        ('reacq_interval', int, 'attempt global reacquisition every N lost frames'),
        ('reacq_scale', float, 'image reduction factor for global search, in (0, 1]'),
        ('template_ema', float, 'template update rate, between 0 and 1'),
        ('template_update_thresh', float, 'minimum pair similarity for template updates, between 0 and 1'),
        ('scale_ema', float, 'box scale smoothing rate, between 0 and 1'),
        ('n_scales', int, 'number of fine-search scales'),
    ):
        parser.add_argument('--' + name.replace('_', '-'), type=value_type,
                            default=getattr(defaults, name), help=help_text)
    return parser


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r'(\d+)', path.name)]


def _inside(path, directory):
    return path == directory or directory in path.parents


def _reference_target(paths, args):
    reference = paths[0].with_suffix('.txt')
    if not reference.is_file():
        raise ValueError(f'Missing initial YOLO reference label: {reference}')
    rows = [line.split() for line in reference.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(rows) != 1 or len(rows[0]) != 5:
        raise ValueError(f'{reference}: expected exactly one target: class_id cx cy width height')
    try:
        original = TargetSpec(class_id=int(rows[0][0]), bbox_xywh=tuple(map(float, rows[0][1:])))
        return TargetSpec(
            class_id=original.class_id if args.class_id is None else args.class_id,
            bbox_xywh=original.bbox_xywh, search_radius=args.search_radius,
            fine_margin=args.fine_margin, match_thresh=args.match_thresh,
            target_thresh=args.target_thresh)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'{reference}: {exc}') from exc


def _prepare(args):
    source = args.input.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise ValueError(f'Input folder does not exist: {source}')
    if image_paths(source):
        folders = [source]
    else:
        folders = sorted((path for path in source.iterdir() if path.is_dir() and image_paths(path)),
                         key=_natural_key)
    if not folders:
        raise ValueError(f'No image frames found in {source}; supported: jpg, jpeg, png, bmp')
    config = TrackerConfig(
        motion_model_name=args.model, reacq_interval=args.reacq_interval,
        reacq_scale=args.reacq_scale, template_ema=args.template_ema,
        template_update_thresh=args.template_update_thresh,
        scale_ema=args.scale_ema, n_scales=args.n_scales)
    make_motion_model(args.model, [0., 0.])  # Validate the model before any writes.
    plans = [(folder, image_paths(folder)) for folder in folders]
    sources = [folder.resolve() for folder in folders]
    reserved = [output / '_tracking_stats.json']
    for folder, paths in plans:
        target_folder = (output / folder.name).resolve()
        if any(_inside(target_folder, entry) or _inside(entry, target_folder) for entry in sources):
            raise ValueError('Output must not overlap source folders or overwrite source annotations')
        reserved.extend(target_folder / (path.stem + '.txt') for path in paths)
        reserved.extend([target_folder / '_tracking_stats.json', target_folder / '_track.jsonl'])
    if any(_inside(path.resolve(), entry) for path in reserved for entry in sources):
        raise ValueError('Output must not overwrite source annotations')
    telemetry = args.telemetry.resolve() if args.telemetry is not None else None
    if telemetry is not None:
        if _inside(telemetry, source) or any(_inside(telemetry, entry) for entry in sources):
            raise ValueError('Telemetry path must be outside source folders')
        if telemetry in {path.resolve() for path in reserved}:
            raise ValueError('Telemetry path conflicts with generated label or tracking output')
        if telemetry.exists() and not telemetry.is_file():
            raise ValueError(f'Telemetry path must name a file: {telemetry}')
    prepared = [(folder, paths, _reference_target(paths, args)) for folder, paths in plans]
    return output, telemetry, config, prepared


def _aggregate_telemetry(telemetry, output, plans):
    """Replace the previous complete run, without appending duplicates on reruns."""
    telemetry.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=telemetry.parent,
                                         prefix='.tracking-', suffix='.jsonl', delete=False) as destination:
            temporary_path = Path(destination.name)
            for folder, paths, _ in plans:
                lines = (output / folder.name / '_track.jsonl').read_text(encoding='utf-8').splitlines()
                if len(lines) != len(paths):
                    raise ValueError(f'Telemetry frame count mismatch for {folder}')
                for path, line in zip(paths, lines):
                    observation = json.loads(line)
                    observation.update(sequence=folder.name, source=str(folder.resolve()), image_name=path.name)
                    destination.write(json.dumps(observation, ensure_ascii=False, allow_nan=False) + '\n')
        temporary_path.replace(telemetry)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        output, telemetry, config, plans = _prepare(args)
        summaries = []
        for folder, _, target in plans:
            per_sequence = output / folder.name / '_track.jsonl' if telemetry is not None else None
            stats = process_folder(folder, output, target, config, per_sequence)
            if stats is None or stats['total'] == 0:
                raise ValueError(f'No frames processed for {folder}')
            summaries.append(stats)
        if telemetry is not None:
            _aggregate_telemetry(telemetry, output, plans)
        (output / '_tracking_stats.json').write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Processed {sum(row["total"] for row in summaries)} frames in {len(summaries)} sequence(s). Output: {output}')
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f'Tracking failed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
