"""CLI integration: real image decoding, labels, ordering, and output safety."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from motion_prior_tracker import main


def make_sequence(folder, class_id=7, names=('frame1.png', 'frame2.png', 'frame10.png')):
    folder.mkdir(parents=True)
    patch = np.random.default_rng(12).integers(30, 240, (12, 12, 3), dtype=np.uint8)
    for step, name in enumerate(names):
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        image[24:36, 26+step:38+step] = patch
        encoded = cv2.imencode(Path(name).suffix, image)[1]
        encoded.tofile(folder / name)
    (folder / Path(names[0]).with_suffix('.txt')).write_text(
        f'{class_id} 0.4 0.46875 0.15 0.1875\n', encoding='utf-8')
    return folder


def records(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_class_override_outputs_nonbird_labels_and_telemetry(tmp_path):
    source = make_sequence(tmp_path / 'source')
    output, telemetry = tmp_path / 'output', tmp_path / 'tracks.jsonl'
    assert main(['--input', str(source), '--output', str(output), '--class-id', '42',
                 '--model', 'ConstantAcceleration', '--n-scales', '3',
                 '--telemetry', str(telemetry)]) == 0
    assert (source / 'frame1.txt').read_text().startswith('7 ')
    assert len(list((output / 'source').glob('frame*.txt'))) == 3
    assert all(path.read_text().startswith('42 ') for path in (output / 'source').glob('frame*.txt'))
    obs = records(telemetry)
    assert [row['class_id'] for row in obs] == [42, 42, 42]
    assert obs[0]['image_size'] == [80, 64]
    assert obs[0]['measurement_valid'] is True
    assert len(json.loads((output / '_tracking_stats.json').read_text())) == 1


@pytest.mark.parametrize('extra', [[], ['--search-radius', '12', '--match-thresh', '0.45']])
def test_default_class_preserves_reference_with_target_tuning(tmp_path, extra):
    source = make_sequence(tmp_path / 'sequence', class_id=17)
    output = tmp_path / 'output'
    assert main(['--input', str(source), '--output', str(output), '--n-scales', '3', *extra]) == 0
    assert (output / 'sequence' / 'frame2.txt').read_text().startswith('17 ')


def test_multiple_sequences_aggregate_all_frames_in_natural_order_and_repeat_safely(tmp_path):
    source = tmp_path / 'sequences'
    make_sequence(source / 'seq2', class_id=2)
    make_sequence(source / 'seq10', class_id=10)
    output, telemetry = tmp_path / 'output', tmp_path / 'tracks.jsonl'
    args = ['--input', str(source), '--output', str(output), '--n-scales', '3',
            '--telemetry', str(telemetry)]
    for _ in range(2):
        assert main(args) == 0
        obs = records(telemetry)
        assert [row['sequence'] for row in obs] == ['seq2'] * 3 + ['seq10'] * 3
        assert [row['image_name'] for row in obs] == ['frame1.png', 'frame2.png', 'frame10.png'] * 2
        assert [row['frame_index'] for row in obs] == [0, 1, 2, 0, 1, 2]
        assert {row['source'] for row in obs} == {str((source / 'seq2').resolve()), str((source / 'seq10').resolve())}
    summary = json.loads((output / '_tracking_stats.json').read_text())
    assert [row['total'] for row in summary] == [3, 3]


@pytest.mark.parametrize('kind', ['missing', 'empty', 'missing_label'])
def test_invalid_input_returns_nonzero_with_actionable_error(tmp_path, capsys, kind):
    source = tmp_path / 'input'
    if kind == 'empty':
        source.mkdir()
    elif kind == 'missing_label':
        make_sequence(source)
        (source / 'frame1.txt').unlink()
    assert main(['--input', str(source), '--output', str(tmp_path / 'output')]) != 0
    assert capsys.readouterr().err.strip()


@pytest.mark.parametrize('unsafe', ['source_output', 'source_telemetry', 'nested_source_telemetry'])
def test_unsafe_paths_rejected_before_any_output_writes(tmp_path, unsafe):
    source = make_sequence(tmp_path / 'source')
    before = (source / 'frame1.txt').read_bytes()
    output, telemetry = tmp_path / 'output', tmp_path / 'tracks.jsonl'
    if unsafe == 'source_output':
        output = tmp_path
    elif unsafe == 'source_telemetry':
        telemetry = source / 'frame1.txt'
    else:
        telemetry = source / 'nested' / 'tracks.jsonl'
    assert main(['--input', str(source), '--output', str(output), '--telemetry', str(telemetry)]) != 0
    assert (source / 'frame1.txt').read_bytes() == before
    assert not (source / 'frame2.txt').exists()
    assert not (source / 'nested').exists()
    assert not (tmp_path / 'output').exists()


def test_help_explains_initialization_and_telemetry(capsys):
    with pytest.raises(SystemExit) as result:
        main(['--help'])
    assert result.value.code == 0
    text = capsys.readouterr().out.lower()
    assert '--telemetry' in text and '--class-id' in text and 'initial' in text


def test_template_update_threshold_accepts_conservative_configuration(tmp_path):
    source = make_sequence(tmp_path / 'source')
    output = tmp_path / 'output'
    assert main(['--input', str(source), '--output', str(output),
                 '--template-update-thresh', '1.0', '--n-scales', '3']) == 0
    assert (output / 'source' / 'frame10.txt').exists()


@pytest.mark.parametrize('extra', [
    ['--n-scales', '0'], ['--class-id', '-1'], ['--match-thresh', '2'],
    ['--model', 'Unknown'],
])
def test_invalid_configuration_does_not_create_outputs(tmp_path, extra):
    source = make_sequence(tmp_path / 'source')
    output = tmp_path / 'output'
    assert main(['--input', str(source), '--output', str(output), *extra]) != 0
    assert not output.exists()


@pytest.mark.parametrize('relative', ['source/frame2.txt', 'source/_track.jsonl', '_tracking_stats.json'])
def test_aggregate_cannot_overwrite_other_generated_outputs(tmp_path, relative):
    source = make_sequence(tmp_path / 'source')
    output = tmp_path / 'output'
    assert main(['--input', str(source), '--output', str(output),
                 '--telemetry', str(output / relative)]) != 0
    assert not output.exists()
