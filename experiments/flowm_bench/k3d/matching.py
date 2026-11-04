"""Euclidean Hungarian matching, as in the 2D gate, with the requested .5 miss cutoff."""
import numpy as np
from scipy.optimize import linear_sum_assignment

SPEED_BINS = [(0.25, 0.5), (0.5, 0.75), (0.75, 1.000001)]


def match_velocities(estimates, truth, cutoff=0.5):
    estimates = np.asarray(estimates, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    error = np.full(len(truth), np.nan)
    assignment = np.full(len(truth), -1, dtype=int)
    distance = np.linalg.norm(truth[:, None] - estimates[None], axis=-1)
    if len(estimates) and len(truth):
        rows, cols = linear_sum_assignment(distance)
        error[rows] = distance[rows, cols]
        assignment[rows] = cols
    miss = (~np.isfinite(error)) | (error > cutoff)
    false = (distance.min(0) > cutoff) if len(truth) else np.ones(len(estimates), dtype=bool)
    return dict(error=error, miss=miss, false=false, assignment=assignment)


def lower_median(values):
    x = np.sort(np.asarray(values))
    return float(x[(len(x)-1)//2]) if len(x) else float("nan")


def nearest_integer(truth):
    nodes = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]])
    return np.linalg.norm(np.asarray(truth)[:, None] - nodes[None], axis=-1).min(1)


def summary(error, miss):
    finite = np.asarray(error)[np.isfinite(error)]
    return dict(truths=len(error), assigned=len(finite), median=lower_median(finite),
                p90=float(np.quantile(finite, .9)) if len(finite) else float("nan"),
                miss_rate=float(np.mean(miss)) if len(miss) else float("nan"))
