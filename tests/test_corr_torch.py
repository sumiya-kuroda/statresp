"""Correctness tests for statresp.corr_torch against per-cell numpy/scipy reference calculations."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats
import xarray as xr

from statresp import _cond_avg, _labels, _slice, corr_torch

WINDOW = (0.0, 2.0)
COND = ["direction", "sf", "tf"]

# The constant cell makes np.corrcoef divide by zero (-> NaN), which is expected here.
pytestmark = pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")


def _make_cube(n_cell=5, n_trial=24, n_time=15, seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(-0.5, 2.0, n_time)
    direction = rng.choice([0, 90, 180, 270], size=n_trial)
    sf = rng.choice([0.02, 0.04], size=n_trial)
    tf = rng.choice([1, 2], size=n_trial)
    data = rng.normal(size=(n_cell, n_trial, n_time))
    data[0] = 0.0  # constant/zero cell -> exercises the zero-variance/zero-norm edge case
    return xr.DataArray(
        data,
        dims=("cell", "trial", "time"),
        coords={
            "cell": np.arange(n_cell),
            "trial": np.arange(n_trial),
            "time": time,
            "direction": ("trial", direction),
            "sf": ("trial", sf),
            "tf": ("trial", tf),
        },
    )


# ---------------------------------------------------------------------------
# Reference implementations, one cell at a time
# ---------------------------------------------------------------------------

def _ref_tuning(cube):
    """(n_cell, n_cond * n_time) condition-averaged responses, built cell by cell."""
    return np.stack([
        _cond_avg(cube.isel(cell=i), WINDOW, COND).ravel() for i in range(cube.sizes["cell"])
    ])


def _ref_resid(cube):
    """(n_cell, n_trial * n_time) within-condition residuals, built cell by cell."""
    rows = []
    for i in range(cube.sizes["cell"]):
        fw = _slice(cube.isel(cell=i), WINDOW)
        data = fw.values.astype(float)
        labels = _labels(fw, COND)
        res = data.copy()
        for c in np.unique(labels):
            m = labels == c
            res[m] -= np.nanmean(data[m], axis=0, keepdims=True)
        rows.append(res.ravel())
    return np.stack(rows)


def _trial_means(cube, window=WINDOW):
    """(cell, trial, time) -> (cell, trial): per-trial mean over the window."""
    return _slice(cube, window).mean("time", skipna=True)


def _assert_close(R, expected):
    np.testing.assert_allclose(R, expected, atol=1e-4, equal_nan=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_signal_corr_matrix_matches_reference():
    cube = _make_cube()
    R = corr_torch.signal_corr_matrix(cube, window=WINDOW, device="cpu")
    _assert_close(R, np.corrcoef(_ref_tuning(cube)))


def test_signal_corr_matrix_spearman():
    cube = _make_cube().isel(cell=slice(1, None))  # drop the constant cell
    R = corr_torch.signal_corr_matrix(cube, window=WINDOW, device="cpu", method="spearman")
    _assert_close(R, scipy.stats.spearmanr(_ref_tuning(cube), axis=1).statistic)


def test_noise_corr_matrix_matches_reference():
    cube = _make_cube()
    R = corr_torch.noise_corr_matrix(cube, window=WINDOW, device="cpu")
    _assert_close(R, np.corrcoef(_ref_resid(cube)))


def test_cosangle_matrix_matches_reference():
    cube = _make_cube()
    R = corr_torch.cosangle_matrix(cube, window=WINDOW, device="cpu")
    v = np.nan_to_num(_ref_tuning(cube))[1:]  # cell 0 has zero norm, checked below
    vn = v / np.linalg.norm(v, axis=1, keepdims=True)
    _assert_close(R[1:, 1:], vn @ vn.T)


def test_cosangle_matrix_zero_norm_row_is_nan():
    R = corr_torch.cosangle_matrix(_make_cube(), window=WINDOW, device="cpu")
    assert np.all(np.isnan(R[0, :]))
    assert np.all(np.isnan(R[:, 0]))
    assert np.allclose(np.diag(R)[1:], 1.0)


def test_total_corr_matrix_raw_traces():
    rng = np.random.default_rng(2)
    x = xr.DataArray(rng.normal(size=(4, 50)), dims=("cell", "time"))
    R = corr_torch.total_corr_matrix(x, device="cpu")
    _assert_close(R, np.corrcoef(x.values))


def test_total_corr_matrix_dim_order_and_names():
    rng = np.random.default_rng(3)
    x = xr.DataArray(rng.normal(size=(30, 4)), dims=("frame", "roi"))  # (time, cell) order
    R = corr_torch.total_corr_matrix(x, time_dim="frame", cell_dim="roi", device="cpu")
    _assert_close(R, np.corrcoef(x.values.T))


def test_total_corr_matrix_diagonal_and_edge_cases():
    R = corr_torch.total_corr_matrix(_trial_means(_make_cube()), time_dim="trial", device="cpu")
    assert np.allclose(np.diag(R)[1:], 1.0)  # non-constant cells perfectly self-correlated
    assert np.isnan(R[0, 0])  # constant cell -> NaN


def test_total_corr_matrix_listwise_nan_deletion():
    rng = np.random.default_rng(1)
    n_cell, n_trial, n_time = 3, 10, 8
    time = np.linspace(0.0, 2.0, n_time)
    data = rng.normal(size=(n_cell, n_trial, n_time))
    data[1, 3, :] = np.nan  # cell 1 missing on trial 3
    cube = xr.DataArray(
        data,
        dims=("cell", "trial", "time"),
        coords={"cell": np.arange(n_cell), "trial": np.arange(n_trial), "time": time},
    )

    means = _trial_means(cube)
    R = corr_torch.total_corr_matrix(means, time_dim="trial", device="cpu")

    v = means.values  # (cell, trial)
    ok = ~np.isnan(v).any(axis=0)  # listwise: trial 3 dropped for every cell
    assert ok.sum() == n_trial - 1
    expected = np.corrcoef(v[0, ok], v[2, ok])[0, 1]
    assert R[0, 2] == pytest.approx(expected, abs=1e-4)


def test_too_few_samples_raises():
    x = xr.DataArray(np.ones((3, 2)), dims=("cell", "time"))
    with pytest.raises(ValueError, match="at least 3 samples"):
        corr_torch.total_corr_matrix(x, device="cpu")


def test_single_cell_returns_2d():
    cube = _make_cube(n_cell=2).isel(cell=[1])
    R = corr_torch.signal_corr_matrix(cube, window=WINDOW, device="cpu")
    assert R.shape == (1, 1)
    assert R[0, 0] == pytest.approx(1.0, abs=1e-5)
