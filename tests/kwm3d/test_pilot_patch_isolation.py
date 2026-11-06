"""Reported patch motion must be invisible to the registered pilot decision."""
import ast
import builtins
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from scripts import kwm3d_pilot_stats as stats
from scripts import kwm3d_pilot_patch_report as report
from test_pilot_stats import dump

ROOT = Path(__file__).resolve().parents[2]


def test_registered_statistics_have_no_patch_dependency():
    tree = ast.parse((ROOT / 'scripts/kwm3d_pilot_stats.py').read_text())
    forbidden = ('patch', 'trained_lift', 'k3d_trained_lifts')
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert not any(word in node.value.lower() for word in forbidden), node.value
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or '')
            assert not any(word in name for name in names for word in ('k3d', 'trained_lift', 'patch_motion', 'kwm3d_pilot_patch_report'))


def record_opens(monkeypatch, opened):
    """Record all Python file entry points, including torch's deserializer."""
    def wrap(function):
        def recorded(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                opened.append(os.fsdecode(path))
            return function(path, *args, **kwargs)
        return recorded

    monkeypatch.setattr(builtins, 'open', wrap(builtins.open))
    monkeypatch.setattr(io, 'open', wrap(io.open))
    monkeypatch.setattr(Path, 'open', wrap(Path.open))
    monkeypatch.setattr(torch, 'load', wrap(torch.load))


@pytest.mark.parametrize('c,a,b,expected', [(1., .8, 2., 'PASS'), (1., .95, 2., 'FAIL'),
                                         (1., .8, .5, 'INCONCLUSIVE')])
def test_statistics_never_open_diagnostic_files(monkeypatch, tmp_path, c, a, b, expected):
    dumps = [dump(value) for value in (c, a, b)]
    results = []
    for case in ('clean', 'zero', 'extreme', 'garbage'):
        directory = tmp_path / case
        directory.mkdir()
        for name, data in zip(('c', 'a', 'baseline'), dumps):
            torch.save(data, directory / f'{name}.pt')
        if case != 'clean':
            payload = b'\x00\xff\x81garbage' if case == 'garbage' else (
                f'group,bin,pairs,usable,median,p90,within_025,boundary_fraction\n'
                f'all,all,44990,44990,{0 if case == "zero" else 99},999,1,0\n').encode()
            (directory / 'pilot_patch_motion.md').write_bytes(payload)
            for arm in ('c', 'a_oracle'):
                target = directory / 'patch_motion' / arm
                target.mkdir(parents=True)
                for k in (8, 32, 256):
                    for kind in ('summary', 'pairs'):
                        (target / f'patch_k{k}_{kind}.csv').write_bytes(payload)
        opened = []
        with monkeypatch.context() as patch:
            patch.chdir(directory)
            record_opens(patch, opened)
            stats.main(['--c', 'c.pt', '--a_oracle', 'a.pt', '--baseline', 'baseline.pt',
                        '--output', 'pilot_stats_140.md'])
            verdict = stats.compare(*dumps)[1]
        assert opened, 'the recorder must observe the dump reads and statistics write'
        assert {Path(path).name for path in opened} == {'c.pt', 'a.pt', 'baseline.pt', 'pilot_stats_140.md'}
        assert not any('patch' in path.lower() for path in opened)
        results.append(((directory / 'pilot_stats_140.md').read_bytes(), verdict))
    assert all(result == results[0] for result in results)
    assert results[0][1] == expected


def write_summaries(directory, prefix, median=.636):
    directory.mkdir(parents=True, exist_ok=True)
    for k in report.KS:
        rows = [dict(source='synthetic', k=k, group=group, bin=name, pairs=44990, usable=44989,
                     median=median, p90=1.23456789, within_025=.121234567, boundary_fraction=.151234567)
                for group, bins in report.GROUPS for name in bins]
        with (directory / f'{prefix}_k{k}_summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


@pytest.fixture
def released(tmp_path):
    directory = tmp_path / 'released'
    write_summaries(directory, 'followup_patch_tokens')
    return directory


def make_run(directory, arm, median):
    write_summaries(directory, 'patch', median)
    info = dict(checkpoint=f'/checkpoints/{arm}/step-006000.ckpt', sha256=arm+'-sha256', arm=arm,
                lift_definition='stage 1 tokens' if arm == 'c' else 'V-pooled observation_write',
                clips=200, frames=140, Ks=[8,32,256], pairs=44990, pair_set_matches_stage1=True)
    (directory / 'patch_run.json').write_text(json.dumps(info))
    return info


def test_report_released_only_and_missing_lines(released, tmp_path, capsys):
    output = tmp_path / 'report.md'
    report.main(['--released-results', str(released), '--missing', 'C=final checkpoint missing',
                 '--missing', 'A-oracle=chain stopped early', '--output', str(output)])
    text = output.read_text()
    assert text.splitlines()[0] == report.NOTICE
    assert 'C: test not run — final checkpoint missing' in text
    assert 'A-oracle: test not run — chain stopped early' in text
    assert text.count('| frozen released lift (stage 1) |') == 3*(1+3+4)
    assert 'Speed bins' in text and 'Distance bins' in text
    assert not any(word in text for word in ('PASS', 'FAIL', 'INCONCLUSIVE'))
    assert not capsys.readouterr().out


def test_report_two_runs_and_provenance(released, tmp_path):
    c, a = tmp_path / 'c', tmp_path / 'a_oracle'
    infos = [make_run(c, 'c', 0.), make_run(a, 'a_oracle', 99.)]
    output = tmp_path / 'report.md'
    report.main(['--released-results', str(released), '--run', f'C={c}', '--run', f'A-oracle={a}',
                 '--output', str(output)])
    text = output.read_text()
    for name in ('frozen released lift (stage 1)', 'C', 'A-oracle'):
        assert text.count(f'| {name} |') == 24
    main = text.split('Speed bins')[0]
    assert main.count('| frozen released lift (stage 1) |') == 3
    assert main.count('| C |') == main.count('| A-oracle |') == 3
    assert main.index('| frozen released') < main.index('| C |') < main.index('| A-oracle |')
    for info in infos:
        assert info['checkpoint'] in text and info['sha256'] in text and info['lift_definition'] in text
    assert text.count('pair_set_matches_stage1: true') == 2
    assert '| 44990 | 44989 | 0.000000 | 1.234568 | 0.121235 | 0.151235 |' in text
    assert '| 44990 | 44989 | 99.000000 |' in text
    assert not any(word in text for word in ('PASS', 'FAIL', 'INCONCLUSIVE'))


@pytest.mark.parametrize('value,expected', [(0, '0.000000'), (.123456789, '0.123457'),
    (99, '99.000000'), (1e-9, '0.000000'), (float('inf'), 'inf'), (-float('inf'), '-inf'), (float('nan'), 'n/a')])
def test_report_number_formatting(value, expected):
    assert report.number(value) == expected


def test_report_cli_preserves_statistics_and_rejects_statistics_destination(released, tmp_path):
    sentinels = {tmp_path / f'pilot_stats_{horizon}.md': f'original {horizon}\n'.encode() for horizon in (140,280)}
    for path, content in sentinels.items():
        path.write_bytes(content)
    command = [sys.executable, str(ROOT / 'scripts/kwm3d_pilot_patch_report.py'),
               '--released-results', str(released), '--missing', 'C=checkpoint missing', '--output']
    output = tmp_path / 'pilot_patch_motion.md'
    result = subprocess.run([*command, str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert output.read_text().startswith(report.NOTICE)
    for name in ('pilot_stats_140.md', 'pilot_stats_280.md', 'pilot_stats_custom.md'):
        result = subprocess.run([*command, str(tmp_path / name)], capture_output=True, text=True)
        assert result.returncode != 0
        assert 'must not name a pilot_stats file' in result.stderr
    for path, content in sentinels.items():
        assert path.read_bytes() == content
    assert not (tmp_path / 'pilot_stats_custom.md').exists()


def test_report_has_no_torch_or_decision_import():
    tree = ast.parse((ROOT / 'scripts/kwm3d_pilot_patch_report.py').read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module)
    assert set(imports) <= {'argparse', 'csv', 'json', 'pathlib', 'numpy'}
