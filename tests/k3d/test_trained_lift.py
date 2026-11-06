"""Trained lift calls the same backbone write used by the estimator."""
from pathlib import Path
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


def test_c_has_no_trained_pooled_write(capsys):
    with pytest.raises(SystemExit) as error:
        trained_lift.main(['--checkpoint', 'unused.ckpt', '--arm', 'c'])
    assert error.value.code == 2
    assert 'no trained pooled observation write' in capsys.readouterr().err
    with pytest.raises(ValueError, match='no trained pooled observation write'):
        trained_lift.run('unused.ckpt', 'c')


@pytest.mark.parametrize('checkpoint_arg', [['--checkpoint', 'unused.ckpt'], ['--checkpoint=unused.ckpt']])
def test_stage1_dispatch_accepts_validation_split_without_changing_argv(monkeypatch, checkpoint_arg):
    argv = ['lift', *checkpoint_arg, '--arm', 'a_oracle', '--split', 'k3d_v_validation', '--clips', '2']
    monkeypatch.setattr(sys, 'argv', argv.copy())
    calls = []
    monkeypatch.setattr(trained_lift, 'run', lambda *args: calls.append(args))
    assert trained_lift.dispatch_checkpoint('lift')
    assert calls == [('unused.ckpt', 'a_oracle', 2, 'lift', 8)]
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
