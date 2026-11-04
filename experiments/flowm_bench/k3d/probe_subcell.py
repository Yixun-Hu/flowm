"""Nested, clip-grouped ridge probes of sub-cell offset and positive controls."""
import argparse
import time
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, balanced_accuracy_score
from .common import clips, clip_id, truth, SCRATCH, write_csv, log
from .geometry import rounded_cells, visible
from .lift import read_cache, tokens_to_box

ALPHAS = [1., 100., 10000.]


def extract(paths, variant):
    features, offsets, integers, colors, groups, distances, turns = [], [], [], [], [], [], []
    for i, path in enumerate(paths):
        data, labels = read_cache(path), truth(path)
        ts = torch.arange(0, 140, 10)
        pos = labels["block_pos_agent"][ts]
        box, _ = tokens_to_box(data[variant][ts].float())
        padded = torch.nn.functional.pad(box, (1, 1, 1, 1))
        for t, n in visible(pos).nonzero().tolist():
            rc = rounded_cells(pos[t, n])
            row, col = int(rc[0])-7, int(rc[1])-16
            # Cell-major 3x3 neighbourhood, zero padding includes cone holes.
            patch = padded[t, :, row:row+3, col:col+3].permute(1, 2, 0).flatten()
            features.append(patch.numpy())
            offsets.append((pos[t, n] - torch.floor(pos[t, n] + .5)).numpy())
            integers.append([col, row])
            colors.append(labels["block_colors"][n])
            groups.append(i)
            distances.append(float(pos[t, n, 0]))
            action = int(data["actions"][ts[t]])
            turns.append(int(ts[t]) > 0 and action in (0, 1))
    classes, color = np.unique(colors, return_inverse=True)
    return (np.asarray(features), np.asarray(offsets), np.asarray(integers), color,
            np.asarray(groups), np.asarray(distances), classes, np.asarray(turns))


def run_probe(paths, variant):
    start = time.perf_counter()
    x, y, integer, color, group, distance, classes, turns = extract(paths, variant)
    # Thousands of correlated latent features make alpha=1 ill-conditioned in
    # float32. Fit the linear probe in float64; lift/estimator remain fp32.
    x = x.astype(np.float64)
    log(f"probe {variant}: {len(y)} blocks, {len(paths)} clips, classes={classes.tolist()}")
    targets = np.column_stack([y, integer, np.eye(len(classes))[color]])
    pred, shuffled = np.zeros_like(targets), np.zeros_like(y)
    fold_id = np.zeros(len(y), dtype=int)
    fold_rows = []
    for fold, (train, test) in enumerate(GroupKFold(5).split(x, groups=group)):
        rng = np.random.default_rng(20261104 + fold)
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=fold).split(train, groups=group[train]))
        inner, val = train[a], train[b]
        scale = StandardScaler().fit(x[inner])
        xi, xv = scale.transform(x[inner]), scale.transform(x[val])
        # Independent permutations stay entirely within each split; true test
        # labels are never used to fit or tune a probe, including the control.
        yi = np.column_stack([targets[inner], y[rng.permutation(inner)]])
        yv_shuf = y[rng.permutation(val)]
        scores = []
        for alpha in ALPHAS:
            fit = Ridge(alpha=alpha, solver="cholesky").fit(xi, yi)
            pp = fit.predict(xv)
            scores.append([r2_score(y[val], pp[:, :2]), r2_score(integer[val], pp[:, 2:4]),
                           accuracy_score(color[val], pp[:, 4:-2].argmax(1)), r2_score(yv_shuf, pp[:, -2:])])
        choices = np.argmax(scores, axis=0)
        alpha = np.r_[np.repeat(ALPHAS[choices[0]], 2), np.repeat(ALPHAS[choices[1]], 2),
                      np.repeat(ALPHAS[choices[2]], len(classes)), np.repeat(ALPHAS[choices[3]], 2)]
        scale = StandardScaler().fit(x[train])
        yy = np.column_stack([targets[train], y[rng.permutation(train)]])
        fit = Ridge(alpha=alpha, solver="cholesky").fit(scale.transform(x[train]), yy)
        pp = fit.predict(scale.transform(x[test]))
        pred[test], shuffled[test] = pp[:, :-2], pp[:, -2:]
        fold_id[test] = fold
        # Shuffled held-out labels are independently permuted within this fold.
        control_target = y[rng.permutation(test)]
        for axis in range(2):
            fold_rows.append(dict(variant=variant, fold=fold, axis=axis, n=len(test),
                                  r2=r2_score(y[test, axis], pred[test, axis]),
                                  shuffled_r2=r2_score(control_target[:, axis], shuffled[test, axis]),
                                  alpha=ALPHAS[choices[0]], shuffled_alpha=ALPHAS[choices[3]],
                                  integer_alpha=ALPHAS[choices[1]], color_alpha=ALPHAS[choices[2]]))
        # Preserve the exact held-out shuffled target for pooled scoring.
        if fold == 0:
            shuffled_targets = np.zeros_like(y)
        shuffled_targets[test] = control_target
        log(f"probe {variant} fold={fold}: R2={r2_score(y[test], pred[test,:2], multioutput='raw_values')}")
    rows = []
    bins = [("all", np.ones(len(y), dtype=bool)),
            ("1-4", (distance >= 1) & (distance < 5)),
            ("5-8", (distance >= 5) & (distance < 9)),
            ("9-12", (distance >= 9) & (distance < 13))]
    bins = [(name, step_type, take & step_mask) for name, take in bins
            for step_type, step_mask in (("all", np.ones(len(y), dtype=bool)), ("nonturn", ~turns), ("turn", turns))]
    for name, step_type, take in bins:
        if take.sum() < 2:
            continue
        for axis in range(2):
            score = r2_score(y[take, axis], pred[take, axis])
            control = r2_score(shuffled_targets[take, axis], shuffled[take, axis])
            rows.append(dict(variant=variant, distance=name, step_type=step_type, axis=("forward", "right")[axis], n=int(take.sum()),
                             r2=score, shuffled_r2=control, integer_r2=r2_score(integer[take, axis], pred[take, 2+axis]),
                             color_accuracy=accuracy_score(color[take], pred[take, 4:].argmax(1)),
                             color_balanced_accuracy=balanced_accuracy_score(color[take], pred[take, 4:].argmax(1)),
                             passed=score >= .25 and score >= 5 * control,
                             seconds=time.perf_counter()-start))
    per_clip = []
    for i, path in enumerate(paths):
        take = group == i
        count = int(take.sum())
        for axis in range(2):
            per_clip.append(dict(variant=variant, clip=clip_id(path), fold=int(fold_id[take][0]) if count else -1, axis=axis,
                                 n=count, r2=r2_score(y[take, axis], pred[take, axis]) if count > 1 else float("nan"),
                                 squared_error=float(((y[take, axis]-pred[take, axis])**2).sum()),
                                 target_sum=float(y[take, axis].sum()), target_sumsq=float((y[take, axis]**2).sum()),
                                 shuffled_r2=r2_score(shuffled_targets[take, axis], shuffled[take, axis]) if count > 1 else float("nan"),
                                 integer_r2=r2_score(integer[take, axis], pred[take, 2+axis]) if count > 1 else float("nan"),
                                 color_accuracy=accuracy_score(color[take], pred[take, 4:].argmax(1)) if count else float("nan")))
    np.savez_compressed(SCRATCH / f"probe_{variant}_predictions.npz", target=y, pred=pred, shuffled=shuffled,
                        shuffled_target=shuffled_targets, group=group, fold=fold_id, distance=distance, turns=turns)
    write_csv(f"probe_{variant}_summary.csv", rows)
    write_csv(f"probe_{variant}_folds.csv", fold_rows)
    write_csv(f"probe_{variant}_clips.csv", per_clip)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clips", type=int, default=200)
    p.add_argument("--variants", nargs="+", default=["tokens", "h_new"])
    args = p.parse_args()
    for variant in args.variants:
        run_probe(clips("k3d_v_validation", args.clips), variant)


if __name__ == "__main__":
    main()
