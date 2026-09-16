"""dF/F responsiveness metrics for xarray data.

"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr
from scipy.stats import ks_2samp, skew

from statresp import _DEFAULT_COND, _labels, _rsq_from_labels, _slice

# ---------------------------------------------------------------------------
# Responsiveness & selectivity
# ---------------------------------------------------------------------------
    
def response_rsq(
    f: xr.DataArray,
    cond_coords: str | Sequence[str] | None = None,
    window: tuple[float, float] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> float:
    """R² — fraction of dF/F variance explained by the visual stimulus.

    Subtracts the within-condition trial mean then computes the fraction of
    total variance that remains.  Matches the MATLAB ``response_rsq``:
    residuals are computed per unique (direction, SF, TF) combination so
    that different stimulus conditions are never pooled.

    Threshold from the paper: R² > 0.15.  Use ``window=None`` (default) to
    include all frames, matching the MATLAB pipeline.

    Parameters
    ----------
    f : xr.DataArray, dims (trial, time)
    cond_coords : str or list[str] or None
        Trial coordinate(s) defining stimulus conditions.  Defaults to
        ``["direction", "sf", "tf"]``.
    window : (t_start, t_end) or None
        Response epoch in seconds.  ``None`` uses the full time axis.
    """
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    fw = _slice(f, window, time_dim)
    data = fw.values.astype(float)
    labels = _labels(fw, cond_coords, trial_dim)
    return _rsq_from_labels(data, labels)


def selectivity(
    f: xr.DataArray,
    window: tuple[float, float] | None = None,
    cond_coords: str | Sequence[str] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> float:
    """Lifetime sparseness — skewness of mean responses across conditions.

    Matches the MATLAB pipeline: time-average within *window* first, then
    average over trials per condition, then compute skewness across all
    unique (direction, SF, TF) conditions.

    Higher values indicate sparser, more selective responses.

    Parameters
    ----------
    f : xr.DataArray, dims (trial, time)
    window : (t_start, t_end) or None
        Response epoch in seconds (use the moving-phase window).
    cond_coords : str or list[str] or None
        Defaults to ``["direction", "sf", "tf"]``.
    """
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    fw = _slice(f, window, time_dim)
    # Average over time first (matching MATLAB mean(respMat(window,...), 1))
    trial_means = fw.mean(dim=time_dim)  # (trial,)
    labels = _labels(trial_means, cond_coords, trial_dim)
    data = trial_means.values.astype(float)
    # One scalar per unique condition (matching MATLAB mean(..., 5) then skewness)
    cond_means = np.array([np.nanmean(data[labels == c]) for c in np.unique(labels)])
    valid = cond_means[np.isfinite(cond_means)]
    return float(skew(valid)) if len(valid) > 0 else float("nan")


# ---------------------------------------------------------------------------
# Statistical testing
# ---------------------------------------------------------------------------

def ks_test(
    f: xr.DataArray,
    group_coord: str,
    groups: Sequence | None = None,
    window: tuple[float, float] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> tuple[float, float]:
    """Two-sample K-S test on per-trial mean dF/F between two trial groups (D, p).

    `group_coord` is a trial coordinate with exactly two levels
    (e.g. "stim_type" with values stim / shutter). Pass `groups` to fix
    which two levels to compare. Returns (nan, nan) if either group < 2.
    """
    fw = _slice(f, window, time_dim)
    per_trial = np.nanmean(fw.values, axis=1)          # (trial,)
    g = np.asarray(f[group_coord].values)
    if groups is None:
        groups = np.unique(g)
    if len(groups) != 2:
        raise ValueError(f"expected 2 groups, got {list(groups)}")
    a = per_trial[g == groups[0]]
    b = per_trial[g == groups[1]]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    D, p = ks_2samp(a, b)
    return float(D), float(p)


def perm_test(
    f: xr.DataArray,
    group_coord: str,
    groups: Sequence | None = None,
    window: tuple[float, float] | None = None,
    n_perm: int = 1000,
    seed: int | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> tuple[float, float, np.ndarray]:
    """Permutation test on the difference of means between two trial groups (obs, p, null).

    `group_coord` is a trial coordinate with exactly two levels (e.g.
    "stim_type" with values stim / shutter) — same convention as
    `ks_test`. Averages each trial over *window* first, then
    obs = mean(group[0]) - mean(group[1]) on those per-trial means.
    Shuffles which trial belongs to which group `n_perm` times,
    recomputing the same difference each time to build a null
    distribution, and reports where `obs` falls in it. Two-sided
    p = (1 + #{|null| >= |obs|}) / (n_perm + 1).
    """
    fw = _slice(f, window, time_dim)
    trial_means = fw.mean(dim=time_dim).values.astype(float)  # (trial,)
    g = np.asarray(f[group_coord].values)
    if groups is None:
        groups = np.unique(g)
    if len(groups) != 2:
        raise ValueError(f"expected 2 groups, got {list(groups)}")

    def mean_diff(labels: np.ndarray) -> float:
        a = trial_means[labels == groups[0]]
        b = trial_means[labels == groups[1]]
        return float(np.nanmean(a) - np.nanmean(b))

    obs = mean_diff(g)
    rng = np.random.default_rng(seed)
    null = np.array([mean_diff(rng.permutation(g)) for _ in range(n_perm)])
    p = (1.0 + np.sum(np.abs(null) >= abs(obs))) / (n_perm + 1.0)
    return float(obs), float(p), null


def fdr(pvals, method: str = "bh") -> np.ndarray:
    """FDR-adjusted q-values. method="bh" (Benjamini-Hochberg) or "storey"."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return q
    ranks = np.empty(n, int)
    order = np.argsort(pv)
    ranks[order] = np.arange(1, n + 1)
    pi0 = min(1.0, np.mean(pv > 0.5) / 0.5) if method == "storey" else 1.0
    qv = pi0 * n * pv / ranks
    qs = qv[order]
    qs = np.minimum.accumulate(qs[::-1])[::-1]      # step-up
    qv[order] = np.clip(qs, 0, 1)
    q[ok] = qv
    return q
