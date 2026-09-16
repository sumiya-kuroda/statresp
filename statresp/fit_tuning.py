"""Kim et al. (2014) direction × SF/TF tuning model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.optimize import least_squares


@dataclass
class TuningFit:
    """Result of :func:`fit_tuning`.

    Attributes
    ----------
    params : np.ndarray, shape (10,)
        Fitted model parameters — see module-level table for index↔symbol
        mapping.
    r : float
        R² goodness of fit on per-trial residuals:
        ``(var(resp) − var(resp − pred)) / var(resp)``.
    success : bool
        ``True`` when ``scipy.optimize.least_squares`` terminated with
        status ≥ 1 (converged).
    """

    params: np.ndarray
    r: float
    success: bool


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _ang(delta: np.ndarray) -> np.ndarray:
    """Map angle differences (degrees) onto [0, 180] (matching MATLAB ang_dir)."""
    return np.minimum(np.minimum(np.abs(delta),
                                 np.abs(delta + 360.0)),
                      np.abs(delta - 360.0))


def _model(
    params: np.ndarray,
    dirs: np.ndarray,
    sfs: np.ndarray,
    tfs: np.ndarray,
) -> np.ndarray:
    """Evaluate predicted responses for arrays of (direction, sf, tf) points.

    Parameters are indexed as described in the module docstring.
    *sfs* and *tfs* must already be in log₂ units.
    """
    b, rmax, sf_pref, tf_pref, alpha, sigma_speed, sigma_orth, th_pref, sd, q = params

    # SF/TF: clockwise rotation then 2-D Gaussian
    alpha_rad = np.deg2rad(alpha)
    cos_a, sin_a = np.cos(alpha_rad), np.sin(alpha_rad)
    dsf = sfs - sf_pref
    dtf = tfs - tf_pref
    # Rotation matches MATLAB: rot = [cos, sin; -sin, cos]
    dorth  = cos_a * dsf + sin_a * dtf   # first row  (vec(1,:) in MATLAB)
    dspeed = -sin_a * dsf + cos_a * dtf  # second row (vec(2,:) in MATLAB)
    sigma_speed = max(sigma_speed, 1e-8)
    sigma_orth  = max(sigma_orth,  1e-8)
    G_sftf = np.exp(-dspeed ** 2 / (2.0 * sigma_speed ** 2)
                    - dorth  ** 2 / (2.0 * sigma_orth  ** 2))

    # Direction: double Gaussian
    if sd == 0.0:
        sd = 1e-8
    G_dir = (np.exp(-_ang(dirs - th_pref)         ** 2 / (2.0 * sd ** 2)) +
             q * np.exp(-_ang(dirs + 180.0 - th_pref) ** 2 / (2.0 * sd ** 2)))

    return b + rmax * G_dir * G_sftf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_tuning(
    f: xr.DataArray,
    dir_coords: str = "direction",
    sf_coords: str = "sf",
    tf_coords: str = "tf",
    window: tuple[float, float] | None = None,
    log2_sf_tf: bool = True,
    trial_dim: str = "trial",
    time_dim: str = "time",
    n_iter: int = 10,
    seed: int = 666,
) -> TuningFit:
    """Fit the Kim et al. (2014) direction × SF/TF tuning model.

    Replicates the MATLAB ``sftfParamFit`` / ``extractRoi`` pipeline:

    1. Average dF/F within *window* → one scalar per trial.
    2. (Optionally) log₂-transform SF and TF coordinates.
    3. Fit the model *n_iter* times from random starting points within the
       parameter bounds (seeded RNG for reproducibility).
    4. Return the fit with the lowest residual sum.

    Parameters
    ----------
    f : xr.DataArray, dims (trial, time)
        dF/F traces with *dir_coords*, *sf_coords*, *tf_coords* as
        non-index coordinates on *trial_dim*.
    dir_coords : str
        Coordinate with direction values in **degrees** (0–360).
    sf_coords : str
        Coordinate with spatial-frequency values.
    tf_coords : str
        Coordinate with temporal-frequency values.
    window : (t_start, t_end) or None
        Response epoch in seconds (inclusive).  ``None`` = full axis.
    log2_sf_tf : bool
        If ``True`` (default), apply ``log₂`` to SF and TF coordinates
        before fitting, matching the MATLAB pipeline.
    trial_dim, time_dim : str
        Dimension names.
    n_iter : int
        Number of random restarts.  Default 10 (matches original pipeline).
    seed : int
        NumPy RNG seed.  Default 666 (matches ``rng(666)`` in MATLAB).

    Returns
    -------
    TuningFit
        Best-fit parameters, R² goodness of fit, and convergence flag.

    Examples
    --------
    >>> fit = fit_tuning(f, window=(0.0, 2.0))
    >>> fit.params[7]   # theta_pref (degrees)
    135.0
    >>> fit.r           # R² goodness of fit
    0.91
    """
    # 1. Time-average within the response window → one scalar per trial
    if window is not None:
        t = f[time_dim].values
        f = f.isel({time_dim: (t >= window[0]) & (t <= window[1])})
    trial_means = f.mean(dim=time_dim)  # (n_trials,)

    resp = trial_means.values.astype(float)
    dirs = trial_means[dir_coords].values.astype(float)
    sfs  = trial_means[sf_coords].values.astype(float)
    tfs  = trial_means[tf_coords].values.astype(float)

    # 2. Log₂-transform SF and TF (matching MATLAB: SF = log2(stimParam(:,2)))
    if log2_sf_tf:
        sfs = np.log2(np.where(sfs > 0, sfs, np.nan))
        tfs = np.log2(np.where(tfs > 0, tfs, np.nan))

    # Drop NaN rows
    ok = np.isfinite(resp) & np.isfinite(sfs) & np.isfinite(tfs)
    resp, dirs, sfs, tfs = resp[ok], dirs[ok], sfs[ok], tfs[ok]

    if len(resp) == 0:
        raise ValueError("No finite response values found in the specified window.")

    # 3. Parameter bounds (matching MATLAB sftfParamFit defaults + extractRoi
    #    constraint: ub[1] = 2 * max(condition means))
    from collections import defaultdict
    buckets: dict[tuple[float, float, float], list[float]] = defaultdict(list)
    for d, s, t, r in zip(dirs, sfs, tfs, resp):
        buckets[(float(d), float(s), float(t))].append(r)
    cond_means = np.array([float(np.mean(v)) for v in buckets.values()])
    max_cond_mean = float(cond_means.max()) if len(cond_means) else float(resp.max())

    sf_lo, sf_hi = float(np.nanmin(sfs)), float(np.nanmax(sfs))
    tf_lo, tf_hi = float(np.nanmin(tfs)), float(np.nanmax(tfs))

    lb = np.array([2.0 * resp.min(),  0.0,     sf_lo - 1, tf_lo - 1,  0.0,  0.25, 0.25,   0.0,   1.0, 0.0])
    ub = np.array([      resp.max(),  2.0 * max_cond_mean, sf_hi + 1, tf_hi + 1, 90.0,  4.0,  4.0, 360.0, 180.0, 1.0])

    # 4. Random restarts (matching MATLAB: beta0 = lb + rand(1,10).*(ub-lb))
    rng = np.random.default_rng(seed)
    starts = lb + rng.random((n_iter, 10)) * (ub - lb)
    # Initialise sf_pref / tf_pref from marginal peaks (approximates MATLAB gauss1 fit)
    sf_peak = sfs[np.argmax(resp)]
    tf_peak = tfs[np.argmax(resp)]
    starts[:, 2] = np.clip(sf_peak, lb[2], ub[2])
    starts[:, 3] = np.clip(tf_peak, lb[3], ub[3])

    # 5. Fit
    def residuals(params: np.ndarray) -> np.ndarray:
        return _model(params, dirs, sfs, tfs) - resp

    best_result = None
    best_cost = np.inf

    for p0 in starts:
        try:
            res = least_squares(
                residuals, p0, bounds=(lb, ub), method="trf",
                xtol=1e-12, ftol=1e-12, max_nfev=int(1e5),
            )
            if res.cost < best_cost:
                best_cost = res.cost
                best_result = res
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("All fitting attempts failed.")

    # 6. Goodness of fit: R² = (var(resp) − var(resp − pred)) / var(resp)
    #    Matches MATLAB: r = (var(resp) - var(resp-pred)) / var(resp)
    pred = _model(best_result.x, dirs, sfs, tfs)
    total_var = float(np.var(resp))
    r_sq = float((total_var - np.var(resp - pred)) / total_var) if total_var > 0 else float("nan")

    return TuningFit(params=best_result.x, r=r_sq, success=best_result.status >= 1)
