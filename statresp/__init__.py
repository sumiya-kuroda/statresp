"""statresp — dF/F responsiveness and selectivity metrics for calcium imaging.

Python 3.12 port of the analysis code from:

    Znamenskiy et al. (2024). Functional specificity of recurrent inhibition
    in visual cortex. *Neuron* 112(6): 991–1000.
    https://doi.org/10.1016/j.neuron.2024.01.008

Quickstart (xarray / NWB / pynapple)
-------------------------------------
.. code-block:: python

    import statresp.metrics as m
    import statresp.corr as corr

    r2  = m.response_rsq(f, cond_coords="direction", window=(0.0, 2.0))
    sp  = m.selectivity(f, window=(0.0, 2.0))
    sim = corr.cosangle(f1, f2, window=(0.0, 2.0))
    r, p = corr.signal_corr(f1, f2, window=(0.0, 2.0))
    r, p = corr.noise_corr(f1, f2, window=(0.0, 2.0))
    r, p = corr.trial_corr(f1, f2, window=(0.0, 2.0))
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Internal helpers shared by statresp.metrics and statresp.corr
# ---------------------------------------------------------------------------

_DEFAULT_COND = ["direction", "sf", "tf"]


def _slice(f: xr.DataArray, window: tuple[float, float] | None,
           time_dim: str = "time") -> xr.DataArray:
    if window is None:
        return f
    t = f[time_dim].values
    return f.isel({time_dim: (t >= window[0]) & (t <= window[1])})


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


from statresp import metrics
from statresp import corr
from statresp.fit_tuning import fit_tuning, TuningFit

__all__ = ["metrics", "corr", "fit_tuning", "TuningFit"]
__version__ = "0.1.0"
