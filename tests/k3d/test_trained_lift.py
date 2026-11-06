"""Trained lift calls the same backbone write used by the estimator."""
from pathlib import Path
import csv
import json
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/flowm_bench'))
from k3d.trained_lift import pooled_tokens
from k3d import trained_lift
from algorithms.mem_wm.backbones.kwm3d import KineticBlockworldModel


def test_trained_pooled_write_matches_estimator_imprint():
    torch.set_num_threads(8)
    model = KineticBlockworldModel(arm='a', embed_dim=32, map_proc_encoder_depth=1, dec_depth=1,
                                  num_heads=4, input_shape=(3,32,32), img_patch_size=8).eval()
    frames = torch.rand(2,3,32,32)
    with torch.inference_mode():
        tokens = pooled_tokens(model, frames)
        imprint, box, support = model.observation_imprint(frames)
        expected = tokens @ model.P.T
    torch.testing.assert_close(expected, imprint[:,:,support].transpose(1,2), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('stage', ['gate', 'all'])
def test_c_has_no_trained_pooled_write(capsys, stage):
    with pytest.raises(SystemExit) as error:
        trained_lift.main(['--checkpoint', 'unused.ckpt', '--arm', 'c', '--stage', stage])
    assert error.value.code == 2
    assert 'no trained pooled observation write' in capsys.readouterr().err
    with pytest.raises(ValueError, match='no trained pooled observation write'):
        trained_lift.run('unused.ckpt', 'c', stage=stage)


@pytest.mark.parametrize('stage', ['gate', 'all'])
def test_c_cannot_bypass_gate_rejection_through_patch_dispatch(monkeypatch, capsys, stage):
    monkeypatch.setattr(sys, 'argv', ['patch_motion', '--checkpoint', 'unused.ckpt', '--arm', 'c', '--stage', stage])
    with pytest.raises(SystemExit) as error:
        trained_lift.dispatch_checkpoint('patch')
    assert error.value.code == 2
    assert 'no trained pooled observation write' in capsys.readouterr().err


@pytest.mark.parametrize('stage', ['lift', 'patch'])
@pytest.mark.parametrize('dispatch', [False, True])
def test_c_accepts_observation_lift_and_patch(monkeypatch, tmp_path, stage, dispatch):
    calls = []
    monkeypatch.setattr(trained_lift, 'run', lambda *args: calls.append(args))
    argv = ['--checkpoint', 'unused.ckpt', '--arm', 'c', '--output-dir', str(tmp_path)]
    if dispatch:
        monkeypatch.setattr(sys, 'argv', [stage, *argv])
        assert trained_lift.dispatch_checkpoint(stage)
    else:
        trained_lift.main([*argv, '--stage', stage])
    assert calls == [('unused.ckpt', 'c', 200, stage, 32, tmp_path)]


def test_c_lift_calls_stage1_observation_lift(monkeypatch):
    # Stage-1 observation_lift asserts the released 256-channel gate structure.
    model = KineticBlockworldModel(arm='c', embed_dim=256, map_proc_encoder_depth=1, dec_depth=1,
                                  num_heads=8, input_shape=(3,32,32), img_patch_size=8).eval()
    frames = torch.rand(2,3,32,32)
    calls = []
    observation_lift = trained_lift.observation_lift

    def record(model_arg, frames_arg):
        assert model_arg is model and frames_arg is frames
        result = observation_lift(model_arg, frames_arg)
        calls.append(result)
        return result

    monkeypatch.setattr(trained_lift, 'observation_lift', record)
    with torch.inference_mode():
        actual = trained_lift.lift_tokens(model, frames, 'c')
    assert len(calls) == 1 and actual is calls[0]['tokens']


def test_patch_pairs_match_stage1_released_cache():
    from k3d import lift

    references = {k: trained_lift.RESULTS / f'followup_patch_tokens_k{k}_pairs.csv'
                  for k in trained_lift.PATCH_KS}
    if not (lift.SCRATCH / 'lifts').is_dir() or not all(p.is_file() for p in references.values()):
        pytest.skip('local stage-1 lift cache or pairs CSVs are absent')
    paths = trained_lift.clips('k3d_v_validation', 200)
    paths = [paths[i] for i in (0, 57, 199) if lift.cache_path(paths[i]).is_file()]
    if not paths:
        pytest.skip('local stage-1 lift cache has none of the selected clips')
    selected = {trained_lift.clip_id(path) for path in paths}
    expected = {}
    for k, reference in references.items():
        with reference.open(newline='') as f:
            expected[k] = [r for r in csv.DictReader(f) if r['clip'] in selected]
    actual = {k: [] for k in references}
    for path in paths:
        cached = lift.read_cache(path)
        labels = trained_lift.truth(path)
        for k in references:
            actual[k].extend(trained_lift.patch_pairs(path, cached['tokens'], cached['actions'],
                                                     labels, k, 'observation-only tokens'))
    for k in references:
        assert expected[k], f'no stage-1 reference pairs for K={k}'
        # All 15 columns, including pair identities, velocities and flags; no recomputed reference.
        assert [{key: str(value) for key, value in row.items()} for row in actual[k]] == expected[k]


@pytest.mark.parametrize('checkpoint_arg', [['--checkpoint', 'unused.ckpt'], ['--checkpoint=unused.ckpt']])
def test_stage1_dispatch_accepts_validation_split_without_changing_argv(monkeypatch, checkpoint_arg):
    argv = ['lift', *checkpoint_arg, '--arm', 'a_oracle', '--split', 'k3d_v_validation', '--clips', '2']
    monkeypatch.setattr(sys, 'argv', argv.copy())
    calls = []
    monkeypatch.setattr(trained_lift, 'run', lambda *args: calls.append(args))
    assert trained_lift.dispatch_checkpoint('lift')
    assert calls == [('unused.ckpt', 'a_oracle', 2, 'lift', 8, None)]
    assert sys.argv == argv


@pytest.mark.parametrize('options,message', [
    (['--split', 'k3d_int_validation'], 'require --split k3d_v_validation'),
    (['--ks', '32', '--sigmas', '2'], 'Stage-1 options are not supported with --checkpoint'),
    (['--stage', 'gate'], 'dispatches trained-lift --stage lift')])
def test_stage1_dispatch_rejects_incompatible_options_before_loading(monkeypatch, capsys, options, message):
    monkeypatch.setattr(sys, 'argv', ['lift', '--checkpoint', 'unused.ckpt', '--arm', 'a_oracle', *options])
    monkeypatch.setattr(trained_lift, 'run', lambda *args: pytest.fail('must reject before loading'))
    with pytest.raises(SystemExit) as error:
        trained_lift.dispatch_checkpoint('lift')
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_stage1_dispatch_leaves_released_path_alone(monkeypatch):
    argv = ['lift', '--split', 'k3d_int_validation']
    monkeypatch.setattr(sys, 'argv', argv.copy())
    assert not trained_lift.dispatch_checkpoint('lift')
    assert sys.argv == argv


@pytest.fixture
def synthetic_patch(monkeypatch, tmp_path):
    path = tmp_path / '0' / '0000_rgb.mp4'
    scratch = tmp_path / 'scratch'
    output = tmp_path / 'reported'
    reference = tmp_path / 'stage1.csv'
    expected = dict(clip='0/0000', frame=1, block=0)
    trained_lift.write_rows(reference, [expected, dict(clip='other/clip', frame=1, block=0)])
    monkeypatch.setattr(trained_lift, 'SCRATCH', scratch)
    monkeypatch.setattr(trained_lift, 'STAGE1_PAIRS', reference)
    monkeypatch.setattr(trained_lift, 'checkpoint_hash', lambda _: 'synthetic-sha')
    monkeypatch.setattr(trained_lift, 'load_checkpoint_model', lambda *args: object())
    monkeypatch.setattr(trained_lift, 'clips', lambda *args: [path])
    actions = torch.zeros(140, dtype=torch.long)
    actions[1] = 3
    monkeypatch.setattr(trained_lift, 'ReadOnlyClips', lambda _: [dict(videos=torch.zeros(140,1), conds=actions)])
    labels = dict(block_pos_agent=torch.tensor([[[8., 0.], [100., 100.]]]).expand(140,-1,-1),
                  block_speed=torch.full((140,2), .5), block_vel_world=torch.zeros(140,2,3),
                  agent_dir=torch.zeros(140))
    labels['block_vel_world'][:,:,0] = .5
    monkeypatch.setattr(trained_lift, 'truth', lambda _: labels)
    token = torch.randn(1,177,256, generator=torch.Generator().manual_seed(40))
    calls = []

    def lift(model, frames, arm):
        calls.append((len(frames), arm))
        return token.expand(len(frames),-1,-1)

    monkeypatch.setattr(trained_lift, 'lift_tokens', lift)
    return dict(path=path, output=output, reference=reference, expected=expected, calls=calls, scratch=scratch)


def test_multi_k_patch_writes_stage1_columns_and_summaries(synthetic_patch):
    from k3d.patch_motion import summarize
    fixture = synthetic_patch
    output = trained_lift.run('synthetic.ckpt', 'c', n=1, stage='patch', batch_size=140,
                              output_dir=fixture['output'])
    assert fixture['calls'] == [(140, 'c')]  # One lift reused by all three channel counts.
    info = json.loads((output / 'patch_run.json').read_text())
    assert info['Ks'] == [8, 32, 256] and info['pairs'] == 1 and info['pair_set_matches_stage1'] is True
    assert info['clips'] == 1 and info['frames'] == 140 and info['arm'] == 'c'
    assert info['sha256'] == 'synthetic-sha'
    assert info['status'] == 'ok'
    assert not {'k', 'sigma', 'residual_mode', 'truth'} & info.keys()
    assert info['checkpoint'] == str(Path('synthetic.ckpt').resolve())
    assert 'observation_lift' in info['lift_definition']
    for k in (8, 32, 256):
        with (output / f'patch_k{k}_pairs.csv').open() as f:
            pairs = list(csv.DictReader(f))
        assert len(pairs) == 1
        assert list(pairs[0]) == ['source','k','clip','frame','block','speed','distance','cell_x',
                                  'vx','vy','true_vx','true_vy','error','usable','boundary']
        row = pairs[0]
        assert (row['clip'], row['frame'], row['block']) == ('0/0000', '1', '0')
        assert row['true_vx'] == '0.5' and row['true_vy'] == '0.0'
        numeric = {key: float(row[key]) for key in ('speed','distance','cell_x','error')}
        numeric.update(usable=row['usable'] == 'True', boundary=row['boundary'] == 'True')
        expected = summarize([numeric], row['source'], k)
        with (output / f'patch_k{k}_summary.csv').open() as f:
            summaries = list(csv.DictReader(f))
        assert summaries == [{key: str(value) for key, value in r.items()} for r in expected]
    cached = torch.load(Path(info['cache']) / '0_0000.pt', weights_only=True)
    assert cached['tokens'].dtype == torch.float16
    assert cached['tokens'].shape == (140,177,256)
    assert not list(output.glob('gate*'))


def test_patch_stage_refuses_different_pair_set(synthetic_patch):
    fixture = synthetic_patch
    trained_lift.write_rows(fixture['reference'], [dict(clip='0/0000', frame=2, block=0)])
    with pytest.raises(ValueError, match='pair set differs from stage 1') as error:
        trained_lift.run('synthetic.ckpt', 'c', n=1, stage='patch', batch_size=140,
                         output_dir=fixture['output'])
    assert not list(fixture['output'].glob('*.csv'))
    info = json.loads((fixture['output'] / 'patch_run.json').read_text())
    assert info['status'] == 'failed' and info['error'] == str(error.value)
    assert info['arm'] == 'c' and info['sha256'] == 'synthetic-sha'
    assert info['stage'] == 'patch' and 'observation_lift' in info['lift_definition']


@pytest.mark.parametrize('change', ['missing', 'extra', 'duplicate', 'unknown_clip'])
def test_pair_set_rejects_missing_extra_duplicate_and_unknown_clips(synthetic_patch, change):
    fixture = synthetic_patch
    rows = [fixture['expected']]
    paths = [fixture['path']]
    if change == 'missing':
        rows = []
    elif change == 'extra':
        rows += [dict(clip='0/0000', frame=2, block=0)]
    elif change == 'duplicate':
        rows *= 2
    else:
        paths += [fixture['path'].with_name('unknown_rgb.mp4')]
    with pytest.raises(ValueError, match='pair set differs from stage 1'):
        trained_lift.check_pair_set(rows, paths)


def test_c_lift_assertion_is_reported_without_fallback(monkeypatch, synthetic_patch):
    fixture = synthetic_patch

    def fail(*args):
        raise AssertionError('node mismatch')

    monkeypatch.setattr(trained_lift, 'lift_tokens', fail)
    with pytest.raises(AssertionError, match='node mismatch'):
        trained_lift.run('synthetic.ckpt', 'c', n=1, stage='patch', output_dir=fixture['output'])
    info = json.loads((fixture['output'] / 'patch_run.json').read_text())
    assert info['status'] == 'failed' and 'node mismatch' in info['error']
    assert not list(fixture['output'].glob('*.csv'))
    assert not list(fixture['scratch'].rglob('*.pt'))
