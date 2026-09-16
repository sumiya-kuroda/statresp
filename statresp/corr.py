"""Pairwise similarity metrics for xarray dF/F data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr
from scipy.stats import pearsonr

from statresp import _DEFAULT_COND, _cond_avg, _labels, _slice

# ---------------------------------------------------------------------------
# Pairwise similarity
# ---------------------------------------------------------------------------

def cosangle(
    f1: xr.DataArray,
    f2: xr.DataArray,
    window: tuple[float, float] | None = None,
    cond_coords: str | Sequence[str] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> float:
    """Cosine similarity of condition-averaged response vectors.

    The response vector is the condition-averaged dF/F within *window*,
    flattened across all conditions and time points.  Returns 0.0 if either
    vector has zero norm.

    Parameters
    ----------
    f1, f2 : xr.DataArray, dims (trial, time)
    window : (t_start, t_end) or None
    cond_coords : str or list[str] or None
        Defaults to ``["direction", "sf", "tf"]``.
    """
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    v1 = np.nan_to_num(_cond_avg(f1, window, cond_coords, trial_dim, time_dim).ravel())
    v2 = np.nan_to_num(_cond_avg(f2, window, cond_coords, trial_dim, time_dim).ravel())
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(v1 @ v2 / denom) if denom != 0.0 else 0.0


def signal_corr(
    f1: xr.DataArray,
    f2: xr.DataArray,
    window: tuple[float, float] | None = None,
    cond_coords: str | Sequence[str] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> tuple[float, float]:
    """Pearson r of condition-averaged response vectors (r, p).

    Correlates the tuning curves (condition means within *window*).
    NaN entries are dropped pairwise.

    Parameters
    ----------
    f1, f2 : xr.DataArray, dims (trial, time)
    window : (t_start, t_end) or None
    cond_coords : str or list[str] or None
        Defaults to ``["direction", "sf", "tf"]``.
    """
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    v1 = _cond_avg(f1, window, cond_coords, trial_dim, time_dim).ravel()
    v2 = _cond_avg(f2, window, cond_coords, trial_dim, time_dim).ravel()
    ok = ~(np.isnan(v1) | np.isnan(v2))
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r, p = pearsonr(v1[ok], v2[ok])
    return float(r), float(p)


def noise_corr(
    f1: xr.DataArray,
    f2: xr.DataArray,
    window: tuple[float, float] | None = None,
    cond_coords: str | Sequence[str] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> tuple[float, float]:
    """Pearson r of within-condition residuals (noise correlation, r, p).

    Subtracts the within-condition mean from each neuron independently,
    then correlates the residuals across all trials × time points.

    Parameters
    ----------
    f1, f2 : xr.DataArray, dims (trial, time)
    window : (t_start, t_end) or None
    cond_coords : str or list[str] or None
        Defaults to ``["direction", "sf", "tf"]``.
    """
    if cond_coords is None:
        cond_coords = _DEFAULT_COND

    def _resid(f: xr.DataArray) -> np.ndarray:
        fw = _slice(f, window, time_dim)
        data = fw.values.astype(float)
        labels = _labels(fw, cond_coords, trial_dim)
        res = data.copy()
        for c in np.unique(labels):
            m = labels == c
            res[m] -= np.nanmean(data[m], axis=0, keepdims=True)
        return res.ravel()

    r1, r2 = _resid(f1), _resid(f2)
    ok = ~(np.isnan(r1) | np.isnan(r2))
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r, p = pearsonr(r1[ok], r2[ok])
    return float(r), float(p)


def trial_corr(
    f1: xr.DataArray,
    f2: xr.DataArray,
    window: tuple[float, float] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> tuple[float, float]:
    """Pearson r of per-trial mean responses (signal + noise, r, p).

    Averages each trial over the response window to get one value per trial,
    then correlates across trials.  Captures both signal and noise correlation.

    Parameters
    ----------
    f1, f2 : xr.DataArray, dims (trial, time)
    window : (t_start, t_end) or None
    """
    v1 = np.nanmean(_slice(f1, window, time_dim).values, axis=1)
    v2 = np.nanmean(_slice(f2, window, time_dim).values, axis=1)
    ok = ~(np.isnan(v1) | np.isnan(v2))
    if ok.sum() < 3:
        return float("nan"), float("nan")
    r, p = pearsonr(v1[ok], v2[ok])
    return float(r), float(p)
