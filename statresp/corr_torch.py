"""
Pairwise similarity metrics for xarray dF/F data with PyTorch.

NaN handling differs from ``statresp.corr``: the ``*_corr_matrix`` functions
drop NaN samples *listwise* (a sample is dropped for every cell if it is NaN
for any cell), so all pairs share the same samples.  ``statresp.corr`` drops
NaNs *pairwise*, using every sample valid for both cells of a pair.  Results
therefore match only when the data contain no NaNs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import scipy.stats
import torch
import xarray as xr

from statresp import _DEFAULT_COND, _cond_avg_all, _resid_all

# ---------------------------------------------------------------------------
# Shared internals
# ---------------------------------------------------------------------------

def _device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _corr_matrix(v, device=None, method="pearson"):
    """
    Full pairwise correlation matrix for rows of *v* (n_cell, n_samples).
    """
    if method == "spearman":
        v = scipy.stats.rankdata(v, axis=1)
    if v.shape[1] < 3:
        raise ValueError(
            f"Need at least 3 samples per row to compute correlation, got {v.shape[1]}."
        )
    x = torch.as_tensor(v, dtype=torch.float32, device=_device(device))
    # torch.corrcoef returns a 0-d tensor for a single row; keep (n_cell, n_cell)
    return np.atleast_2d(torch.corrcoef(x).cpu().numpy().astype(np.float64))


def _cosine_matrix(v: np.ndarray, device: str | torch.device | None = None) -> np.ndarray:
    """
    Full pairwise cosine-similarity matrix (inner dot product) for rows of *v* (n_cell, n_samples).
    """
    x = torch.as_tensor(v, dtype=torch.float32, device=_device(device))
    norm = x.norm(dim=1, keepdim=True)
    zero = (norm.squeeze(1) == 0).cpu().numpy()
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)
    xn = x / norm
    sim = (xn @ xn.T).clamp(-1.0, 1.0).cpu().numpy().astype(np.float64)

    np.fill_diagonal(sim, 1.0)
    if zero.any():
        sim[zero, :] = np.nan
        sim[:, zero] = np.nan
        # cannot calculate dot product for cells with zero norm
    return sim

# ---------------------------------------------------------------------------
# Batched pairwise similarity
# ---------------------------------------------------------------------------

def cosangle_matrix(
    x: xr.DataArray,
    window: tuple[float, float] | None = None,
    cond_coords: str | Sequence[str] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
    cell_dim: str = "cell",
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Batched :func:`statresp.corr.cosangle`: full (n_cell, n_cell) cosine-similarity matrix.

    Parameters
    ----------
    x : xr.DataArray, dims (cell, trial, time) in any order
    window : (t_start, t_end) or None
    cond_coords : str or list[str] or None
        Defaults to ``["direction", "sf", "tf"]``.
    device : torch device or None
        Defaults to CUDA if available, else CPU.
    """
    assert x.ndim == 3
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    v = np.nan_to_num(_cond_avg_all(x, window, cond_coords, trial_dim, time_dim, cell_dim))
    return _cosine_matrix(v, device)


def signal_corr_matrix(
    x: xr.DataArray,
    window: tuple[float, float] | None = None,
    cond_coords: str | Sequence[str] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
    cell_dim: str = "cell",
    device: str | torch.device | None = None,
    method: str = "pearson",
    verbose: bool = False,
) -> np.ndarray:
    """Batched :func:`statresp.corr.signal_corr`: full (n_cell, n_cell) correlation matrix
    (Pearson or Spearman).

    Correlates condition-averaged tuning curves.  NaN samples (conditions
    that are NaN for *any* cell) are dropped listwise across the whole
    population — see the module docstring for how this differs from the
    pairwise-complete NaN handling in ``statresp.corr``.

    Parameters
    ----------
    x : xr.DataArray, dims (cell, trial, time) in any order
    window : (t_start, t_end) or None
    cond_coords : str or list[str] or None
        Defaults to ``["direction", "sf", "tf"]``.
    device : torch device or None
        Defaults to CUDA if available, else CPU.
    method : {"pearson", "spearman"}
        "spearman" ranks each cell's tuning curve before correlating.
    """
    assert x.ndim == 3
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    v = _cond_avg_all(x, window, cond_coords, trial_dim, time_dim, cell_dim)
    ok = ~np.isnan(v).any(axis=0)
    if verbose:
        print(ok.sum(), "of", ok.size, "conditions kept")
        print(np.isnan(v).sum(axis=1))
    return _corr_matrix(v[:, ok], device, method)


def noise_corr_matrix(
    x: xr.DataArray,
    window: tuple[float, float] | None = None,
    cond_coords: str | Sequence[str] | None = None,
    trial_dim: str = "trial",
    time_dim: str = "time",
    cell_dim: str = "cell",
    device: str | torch.device | None = None,
    method: str = "pearson",
    verbose: bool = False,
) -> np.ndarray:
    """Batched :func:`statresp.corr.noise_corr`: full (n_cell, n_cell) correlation matrix
    (Pearson or Spearman).

    Correlates within-condition residuals (each cell's own within-condition
    mean subtracted first).  NaN samples are dropped listwise across the
    whole population — see the module docstring.

    Parameters
    ----------
    x : xr.DataArray, dims (cell, trial, time) in any order
    window : (t_start, t_end) or None
    cond_coords : str or list[str] or None
        Defaults to ``["direction", "sf", "tf"]``.
    device : torch device or None
        Defaults to CUDA if available, else CPU.
    method : {"pearson", "spearman"}
        "spearman" ranks each cell's residuals before correlating.
    """
    assert x.ndim == 3
    if cond_coords is None:
        cond_coords = _DEFAULT_COND
    v = _resid_all(x, window, cond_coords, trial_dim, time_dim, cell_dim)
    ok = ~np.isnan(v).any(axis=0)
    if verbose:
        print(ok.sum(), "of", ok.size, "samples (trial × time) kept")
        print(np.isnan(v).sum(axis=1))
    return _corr_matrix(v[:, ok], device, method)


def total_corr_matrix(
    x: xr.DataArray,
    time_dim: str = "time",
    cell_dim: str = "cell",
    device: str | torch.device | None = None,
    method: str = "pearson",
    verbose: bool = False
) -> np.ndarray:
    """Batched :func:`statresp.corr.total_corr`: full (n_cell, n_cell) correlation matrix.

    Correlates each pair of cells' activity traces across time. NaN time
    points (NaN for *any* cell) are dropped listwise across the whole
    population.

    Parameters
    ----------
    x : xr.DataArray, dims (cell, time) in any order
    time_dim, cell_dim : str
        Names of the time and cell dimensions.
    device : torch device or None
        Defaults to CUDA if available, else CPU.
    method : {"pearson", "spearman"}
        "spearman" ranks each cell's trace over time before correlating.
    verbose : bool
        Print how many time points survive NaN dropping and NaN counts per cell.
    """
    assert x.ndim == 2
    v = x.transpose(cell_dim, time_dim).values
    ok = ~np.isnan(v).any(axis=0)
    if verbose:
        print(ok.sum(), "of", ok.size, "time points kept")
        print(np.isnan(v).sum(axis=1))
    return _corr_matrix(v[:, ok], device, method)
