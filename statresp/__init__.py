from __future__ import annotations
from typing import Sequence

import numpy as np
import xarray as xr

_DEFAULT_COND = ["direction", "sf", "tf"]

def _slice(f: xr.DataArray, window: tuple[float, float] | None,
           time_dim: str = "time") -> xr.DataArray:
    if window is None:
        return f
    t = f[time_dim].values
    return f.isel({time_dim: (t >= window[0]) & (t < window[1])})


def _labels(f: xr.DataArray, cond_coords: str | Sequence[str],
            trial_dim: str = "trial") -> np.ndarray:
    if isinstance(cond_coords, str):
        return f[cond_coords].values
    # Join coordinate values into a string per trial so that numpy never
    # misinterprets a list of equal-length tuples as a 2-D array.
    parts = [f[c].values.astype(str) for c in cond_coords]
    return np.array(["|".join(v) for v in zip(*parts)])


def _cond_avg(f: xr.DataArray, window: tuple[float, float] | None,
              cond_coords: str | Sequence[str] | None = None,
              trial_dim: str = "trial",
              time_dim: str = "time") -> np.ndarray:
    """Condition-averaged responses, shape (n_conditions, n_time)."""
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    fw = _slice(f, window, time_dim)
    labels = _labels(fw, cond_coords, trial_dim)
    data = fw.values
    return np.array([np.nanmean(data[labels == c], axis=0)
                     for c in np.unique(labels)])


def _rsq_from_labels(data: np.ndarray, labels: np.ndarray) -> float:
    """FVE given precomputed condition labels (shape (trial, ...))."""
    total_var = float(np.nanvar(data))
    if total_var == 0.0:
        return 0.0
    res = data.copy()
    for c in np.unique(labels):
        m = labels == c
        res[m] -= np.nanmean(data[m], axis=0, keepdims=True)
    return float((total_var - np.nanvar(res)) / total_var)


def _cond_avg_all(
    cube: xr.DataArray,
    window: tuple[float, float] | None,
    cond_coords: str | Sequence[str],
    trial_dim: str,
    time_dim: str,
    cell_dim: str,
) -> np.ndarray:
    """Condition-averaged responses per cell, shape (n_cell, n_conditions * n_time)."""
    fw = _slice(cube, window, time_dim)
    labels = _labels(fw, cond_coords, trial_dim)
    das = [
        fw.isel({trial_dim: labels == c}).mean(trial_dim, skipna=True)
        for c in np.unique(labels)
    ]
    avg = xr.concat(das, dim="condition").transpose(cell_dim, "condition", time_dim)
    return avg.values.reshape(avg.sizes[cell_dim], -1)


def _resid_all(
    cube: xr.DataArray,
    window: tuple[float, float] | None,
    cond_coords: str | Sequence[str],
    trial_dim: str,
    time_dim: str,
    cell_dim: str,
) -> np.ndarray:
    """Within-condition residuals per cell, shape (n_cell, n_trial * n_time)."""
    fw = _slice(cube, window, time_dim).transpose(cell_dim, trial_dim, time_dim)
    labels = _labels(fw, cond_coords, trial_dim)
    data = fw.values.astype(float)
    res = data.copy()
    for c in np.unique(labels):
        m = labels == c
        res[:, m, :] -= np.nanmean(data[:, m, :], axis=1, keepdims=True)
    return res.reshape(res.shape[0], -1)

from statresp import metrics
from statresp import corr_torch
from statresp.fit_tuning import fit_tuning, TuningFit

__all__ = ["metrics", "corr_torch", "fit_tuning", "TuningFit"]
__version__ = "0.1.0"
