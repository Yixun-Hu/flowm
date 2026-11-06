import numpy as np
import pytest
import torch
from algorithms.mem_wm.backbones.kwm3d.per_clip import NAMES
from scripts.kwm3d_pilot_stats import compare, bootstrap, read_dump, markdown, sign_disagreement, main, reference_rows


def dump(value, counts=None):
    if counts is None:
        counts = (3,) * 1000
    count = torch.tensor(counts)[:, None].expand(-1, len(NAMES)).clone()
    return dict(metric_names=list(NAMES), paths=[f"{i}.mp4" for i in range(len(counts))],
                dataset_indices=torch.arange(len(counts)), context_frames=70, n_frames=140,
                squared_error=count.double()*torch.as_tensor(value).double(), element_count=count,
                block_pixel_count=count[:, 1:].clone() // 3)


@pytest.mark.parametrize("c,a,b,verdict", [(1., .8, 2., "PASS"), (1., .95, 2., "FAIL"),
    (1., .8, .5, "INCONCLUSIVE"), (1., .8, .8, "INCONCLUSIVE"),
    (1., .9, 2., "PASS"), (1., 1., 2., "FAIL"), (0., 0., 1., "FAIL")])
def test_verdict(c, a, b, verdict):
    rows, actual = compare(dump(c), dump(a), dump(b))
    assert actual == verdict
    assert len(rows) == len(NAMES)
    assert all(row['paired'] == 1000 for row in rows)


def test_pooled_is_pixel_weighted_but_difference_is_clip_weighted():
    c, a, b = [dump(v, (3, 30, 300)) for v in (1., 1., 2.)]
    a['squared_error'] *= torch.tensor([.1, .5, .9])[:, None]
    rows, verdict = compare(c, a, b, smoke=True)
    assert verdict is None
    expected = (3*.1+30*.5+300*.9)/333
    assert rows[0]['a_oracle'] == pytest.approx(expected)
    assert rows[0]['paired_mean'] == pytest.approx(-.5)


def test_zero_regions_excluded_and_empty_secondary_nan():
    c, a, b = dump(1., (0, 3, 6)), dump(.8, (0, 3, 6)), dump(2., (0, 3, 6))
    for d in (c, a, b):
        d['element_count'][:, -1] = 0
        d['squared_error'][:, -1] = 0
        d['block_pixel_count'][:, -1] = 0
    rows, verdict = compare(c, a, b, smoke=True)
    assert verdict is None and rows[1]['excluded'] == 1 and rows[1]['paired'] == 2
    assert rows[-1]['excluded'] == 3 and np.isnan(rows[-1]['ci_high'])
    with pytest.raises(ValueError, match='undefined'):
        compare(dump(1., (0, 0)), dump(.8, (0, 0)), dump(2., (0, 0)), smoke=True)


def test_ci_and_margin_are_both_required():
    counts = (3,)*999 + (30000,)
    c, a, b = [dump(v, counts) for v in (1., 1., 2.)]
    # Big improvement in the large clip, but the other clips deteriorate.
    a['squared_error'] *= torch.tensor([2.]*999 + [.2])[:, None]
    rows, verdict = compare(c, a, b)
    assert rows[1]['reduction'] > .1 and rows[1]['ci_high'] > 0 and verdict == 'FAIL'


def test_bootstrap_seed_and_percentile():
    x = np.array([-.1, -.2, .03])
    rng = np.random.default_rng(20261105)
    means = x[rng.integers(0, len(x), (10000, len(x)))].mean(1)
    assert bootstrap(x) == tuple(np.percentile(means, [2.5, 97.5]))
    assert bootstrap(x) == bootstrap(x)


@pytest.mark.parametrize('field', ['paths', 'dataset_indices', 'element_count', 'n_frames'])
def test_mismatch_rejected(field):
    c, a, b = dump(1.), dump(.8), dump(2.)
    if field == 'paths':
        a[field] = list(reversed(a[field]))
    elif field == 'n_frames':
        a[field] = 280
    else:
        a[field][0] += 1
    with pytest.raises(AssertionError):
        compare(c, a, b)


def test_dump_roundtrip_and_table(tmp_path):
    c, a, b = dump(1.), dump(.8), dump(2.)
    torch.save(c, tmp_path/'per_clip.pt')
    restored = read_dump(tmp_path)
    rows, verdict = compare(restored, a, b)
    text = markdown(rows, verdict, 140, c, b)
    assert '**PASS**' in text and 'Mean A−C' in text and 'Excluded' in text
    assert 'seed variance' in text


def test_opposite_horizon_sign_is_report_only():
    c, a, c2, a2 = dump(1.), dump(.8), dump(1.), dump(1.2)
    c2['n_frames'] = a2['n_frames'] = 280
    assert sign_disagreement(c, a, c2, a2)
    a2['squared_error'] = a['squared_error'].clone()
    assert not sign_disagreement(c, a, c2, a2)


@pytest.mark.parametrize('kind', ['reversed', 'different', 'indices', 'full_count', 'block_count', 'pixels'])
@pytest.mark.parametrize('smoke', [False, True])
def test_baseline_mismatch_rejected(kind, smoke):
    c, a, b = dump(1.), dump(.8), dump(2.)
    if kind == 'reversed':
        # Swapping paths alone also mismatches the dataset indices in smoke mode.
        b['paths'] = list(reversed(b['paths']))
    elif kind == 'different':
        b['paths'][0] = 'not-an-arm-clip.mp4'
    elif kind == 'indices':
        b['dataset_indices'][0] += 1000
    elif kind == 'full_count':
        b['element_count'][0, 0] += 1
    elif kind == 'block_count':
        b['element_count'][0, 1] += 3
        b['block_pixel_count'][0, 0] += 1
    else:
        b['block_pixel_count'][0, 0] += 1
    with pytest.raises(AssertionError):
        compare(c, a, b, smoke=smoke)


@pytest.mark.parametrize('n', [999, 1001])
@pytest.mark.parametrize('which', ['arms', 'baseline', 'all'])
def test_wrong_clip_count_rejected(n, which):
    c, a, b = dump(1.), dump(.8), dump(2.)
    if which in ('arms', 'all'):
        c, a = dump(1., (3,)*n), dump(.8, (3,)*n)
    if which in ('baseline', 'all'):
        b = dump(2., (3,)*n)
    with pytest.raises(AssertionError, match='exactly 1,000'):
        compare(c, a, b)


def subset(d, ids):
    return dict(d, paths=[d['paths'][i] for i in ids], **{
        k: d[k][ids] for k in ('dataset_indices', 'squared_error', 'element_count', 'block_pixel_count')})


def test_smoke_subsets_baseline_by_path_and_suppresses_verdict(tmp_path):
    c, a, b = dump(1.), dump(.8), dump(2.)
    b['squared_error'][999] *= 3
    c, a = subset(c, [999, 2]), subset(a, [999, 2])
    rows, verdict = compare(c, a, b, smoke=True)
    assert rows[1]['baseline'] == 4.0  # Selected clips only, in arm order.
    assert verdict is None
    paths = []
    for name, data in (('c', c), ('a', a), ('b', b)):
        path = tmp_path / f'{name}.pt'
        torch.save(data, path)
        paths.append(str(path))
    output = tmp_path / 'smoke.md'
    main(['--c', paths[0], '--a_oracle', paths[1], '--baseline', paths[2],
          '--smoke', '--output', str(output)])
    text = output.read_text()
    assert text.startswith('SMOKE') and 'baseline clips used: 2 (source: 1000)' in text
    assert all(word not in text for word in ('PASS', 'FAIL', 'INCONCLUSIVE', 'Pilot 140:'))


def test_reordered_baseline_requires_smoke():
    c, a, b = dump(1.), dump(.8), subset(dump(2.), list(reversed(range(1000))))
    with pytest.raises(AssertionError, match='clips/order'):
        compare(c, a, b)
    assert compare(c, a, b, smoke=True)[1] is None


@pytest.mark.parametrize('value,expected', [(.8, 'PASS'), (.95, 'FAIL'), (2.1, 'INCONCLUSIVE')])
def test_references_never_change_verdict(tmp_path, value, expected):
    # Extreme references straddle both arms; neither is allowed into the decision.
    inputs = dict(c=dump(2.), a=dump(value*2), b=dump(1.9 if expected == 'INCONCLUSIVE' else 3.),
                  gray=dump(0.), context_mean=dump(4.), last_flat=dump(.01))
    for name, data in inputs.items():
        torch.save(data, tmp_path / f'{name}.pt')
    args = ['--c', str(tmp_path/'c.pt'), '--a_oracle', str(tmp_path/'a.pt'),
            '--baseline', str(tmp_path/'b.pt'), '--output', str(tmp_path/'stats.md')]
    main(args)
    plain = (tmp_path/'stats.md').read_text()
    for name in ('gray', 'context_mean', 'last_flat'):
        args += ['--reference', f'{name}={tmp_path / (name + ".pt")}']
    main(args)
    reported = (tmp_path/'stats.md').read_text()
    assert plain.splitlines()[0] == reported.splitlines()[0] == f'Pilot 140: **{expected}**'
    assert reported.startswith(plain)
    assert 'reference, not part of the criterion' in reported
    rows = reference_rows(inputs['c'], inputs['a'], {'gray': inputs['gray'], 'context_mean': inputs['context_mean']})
    assert {r['metric'] for r in rows} == {'mse', 'block_mse'}
    assert all(not r['c_below'] and not r['a_oracle_below'] for r in rows if r['reference'] == 'gray')
    if value < 2:
        assert all(r['c_below'] and r['a_oracle_below'] for r in rows if r['reference'] == 'context_mean')
