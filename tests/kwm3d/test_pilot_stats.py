import numpy as np
import pytest
import torch
from algorithms.mem_wm.backbones.kwm3d.per_clip import NAMES
from scripts.kwm3d_pilot_stats import compare, bootstrap, read_dump, markdown, sign_disagreement


def dump(value, counts=(3, 30, 300)):
    count = torch.tensor(counts)[:, None].expand(-1, len(NAMES)).clone()
    return dict(metric_names=list(NAMES), paths=[f"{i}.mp4" for i in range(len(counts))],
                dataset_indices=torch.arange(len(counts)), context_frames=70, n_frames=140,
                squared_error=count.double()*torch.as_tensor(value).double(), element_count=count)


@pytest.mark.parametrize("c,a,b,verdict", [(1., .8, 2., "PASS"), (1., .95, 2., "FAIL"),
    (1., .8, .5, "INCONCLUSIVE"), (1., .8, .8, "INCONCLUSIVE"),
    (1., .9, 2., "PASS"), (1., 1., 2., "FAIL"), (0., 0., 1., "FAIL")])
def test_verdict(c, a, b, verdict):
    rows, actual = compare(dump(c), dump(a), dump(b))
    assert actual == verdict
    assert len(rows) == len(NAMES)
    assert all(row['paired'] == 3 for row in rows)


def test_pooled_is_pixel_weighted_but_difference_is_clip_weighted():
    c, a, b = dump(1.), dump(1.), dump(2.)
    a['squared_error'] *= torch.tensor([.1, .5, .9])[:, None]
    rows, _ = compare(c, a, b)
    expected = (3*.1+30*.5+300*.9)/333
    assert rows[0]['a_oracle'] == pytest.approx(expected)
    assert rows[0]['paired_mean'] == pytest.approx(-.5)


def test_zero_regions_excluded_and_empty_secondary_nan():
    c, a, b = dump(1., (0, 3, 6)), dump(.8, (0, 3, 6)), dump(2., (0, 3, 6))
    for d in (c, a, b):
        d['element_count'][:, -1] = 0
        d['squared_error'][:, -1] = 0
    rows, verdict = compare(c, a, b)
    assert verdict == 'PASS' and rows[1]['excluded'] == 1 and rows[1]['paired'] == 2
    assert rows[-1]['excluded'] == 3 and np.isnan(rows[-1]['ci_high'])
    with pytest.raises(ValueError, match='undefined'):
        compare(dump(1., (0, 0)), dump(.8, (0, 0)), dump(2., (0, 0)))


def test_ci_and_margin_are_both_required():
    c, a, b = dump(1.), dump(1.), dump(2.)
    # Big improvement in the large clip, but the other clips deteriorate.
    a['squared_error'] *= torch.tensor([2., 2., .2])[:, None]
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
