"""
Statistical tests for xarray dF/F data.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import xarray as xr
from scipy.stats import ks_2samp, false_discovery_control
from zetapy import zetatstest, zetatstest2

from statresp import _slice

SEED = 2026
np.random.seed(SEED)

_ALTERNATIVES = ("two-sided", "greater", "less")


def _check_alternative(alternative: str) -> None:
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater' or 'less', got {alternative!r}"
        )


def ks_test(
    f: xr.DataArray,
    group_coord: str,
    groups: Sequence | None = None,
    window: tuple[float, float] | None = None,
    alternative: str = "two-sided",
    trial_dim: str = "trial",
    time_dim: str = "time",
) -> tuple[float, float]:
    """Two-sample K-S test on per-trial mean dF/F between two trial groups (D, p).

    `group_coord` is a trial coordinate with exactly two levels
    (e.g. "stim_type" with values stim / shutter). Pass `groups` to fix
    which two levels to compare. Returns (nan, nan) if either group < 2.

    `alternative` is "two-sided", "greater" (groups[0] tends to be larger
    than groups[1]) or "less" (groups[0] tends to be smaller), the same
    convention as `perm_test`. Note this is the reverse of scipy's
    `ks_2samp` naming, which refers to the CDFs: a larger group has a
    *lower* CDF.
    """
    _check_alternative(alternative)
    fw = _slice(f, window, time_dim)
    trial_means = fw.mean(dim=time_dim).values.astype(float)  # (trial,)
    g = np.asarray(f[group_coord].values)
    if groups is None:
        groups = np.unique(g)
    if len(groups) != 2:
        raise ValueError(f"expected 2 groups, got {list(groups)}")
    a = trial_means[g == groups[0]]
    b = trial_means[g == groups[1]]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    scipy_alt = {"two-sided": "two-sided", "greater": "less", "less": "greater"}[alternative]
    D, p = ks_2samp(a, b, alternative=scipy_alt)
    return float(D), float(p)


def perm_test(
    f: xr.DataArray,
    group_coord: str,
    groups: Sequence | None = None,
    window: tuple[float, float] | None = None,
    n_perm: int = 1000,
    alternative: str = "two-sided",
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
    distribution, and reports where `obs` falls in it.

    `alternative` is "two-sided" (p from #{|null| >= |obs|}), "greater"
    (groups[0] mean larger: #{null >= obs}) or "less" (#{null <= obs});
    p = (1 + count) / (n_perm + 1). Without `groups`, levels are sorted,
    so pass `groups` explicitly for a one-sided test.
    """
    _check_alternative(alternative)
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
    rng = np.random.default_rng(SEED)
    null = np.array([mean_diff(rng.permutation(g)) for _ in range(n_perm)])
    if alternative == "greater":
        hits = null >= obs
    elif alternative == "less":
        hits = null <= obs
    else:
        hits = np.abs(null) >= abs(obs)
    p = (1.0 + np.sum(hits)) / (n_perm + 1.0)
    return float(obs), float(p), null


def circperm_test(
    trace: xr.DataArray,
    event_times: Sequence[float],
    window: tuple[float, float] = (0.0, 1.0),
    baseline: tuple[float, float] | None = None,
    n_perm: int = 1000,
    min_shift: float | None = None,
    alternative: str = "two-sided",
    time_dim: str = "time",
) -> tuple[float, float, np.ndarray]:
    """Circular-shift permutation test for an event-locked response (obs, p, null).

    `trace` is one cell's continuous dF/F, dims (time,), sampled at a
    constant rate. For each event the response is the mean of `trace` in
    [t_event + window[0], t_event + window[1]), minus the mean in
    [t_event + baseline[0], t_event + baseline[1]) if `baseline` is given;
    obs is the mean response over events. Events whose windows run past
    either end of the trace are dropped.

    The null circularly shifts the trace against the events by a random
    offset in [min_shift, T - min_shift] (T = trace duration), `n_perm`
    times. Unlike shuffling samples or trials, this keeps the trace's
    autocorrelation (calcium decay, slow drift) and the spacing between
    events, so it doesn't overstate significance for autocorrelated dF/F.
    `min_shift` (seconds) defaults to the span covered by `window` and
    `baseline`, so a shifted event can't land on its own response.

    `alternative` is "two-sided", "greater" or "less". Two-sided compares
    distances from the null mean. p = (1 + #{null as or more extreme}) / (n_perm + 1).
    """
    assert trace.ndim == 1, "trace must be 1-D (time,)"
    _check_alternative(alternative)
    t = trace[time_dim].values.astype(float)
    y = trace.values.astype(float)
    n = y.size
    dt = float(np.median(np.diff(t)))

    def offsets(w: tuple[float, float]) -> np.ndarray:
        # samples in [w[0], w[1]) relative to the event sample
        k = np.arange(int(np.ceil(w[0] / dt - 1e-9)), int(np.ceil(w[1] / dt - 1e-9)))
        if k.size == 0:
            raise ValueError(f"window {w} contains no samples at dt = {dt:g} s")
        return k

    win = offsets(window)
    base = offsets(baseline) if baseline is not None else None
    lo = min(win[0], base[0]) if base is not None else win[0]
    hi = max(win[-1], base[-1]) if base is not None else win[-1]

    ev = np.searchsorted(t, np.asarray(event_times, dtype=float))
    ev = ev[(ev + lo >= 0) & (ev + hi < n)]
    if ev.size == 0:
        raise ValueError("no events with windows fully inside the trace")

    def stat(e: np.ndarray) -> float:
        # e: event sample indices; wraps around the ends (circular)
        r = np.nanmean(y[(e[:, None] + win) % n], axis=1)
        if base is not None:
            r = r - np.nanmean(y[(e[:, None] + base) % n], axis=1)
        return float(np.nanmean(r))

    if min_shift is None:
        min_shift = (hi - lo + 1) * dt
    k_min = int(np.ceil(min_shift / dt))
    if n - 2 * k_min < 1:
        raise ValueError(
            f"min_shift = {min_shift:g} s leaves no allowed shifts for a {n * dt:g} s trace"
        )

    obs = stat(ev)
    rng = np.random.default_rng(SEED)
    # shifting the trace by +k is the same as shifting the events by -k
    shifts = rng.integers(k_min, n - k_min + 1, size=n_perm)
    null = np.array([stat(ev - k) for k in shifts])

    if alternative == "greater":
        hits = null >= obs
    elif alternative == "less":
        hits = null <= obs
    else:
        mu = null.mean()
        hits = np.abs(null - mu) >= abs(obs - mu)
    p = (1.0 + np.sum(hits)) / (n_perm + 1.0)
    return float(obs), float(p), null


def zeta_test(
    trace: xr.DataArray,
    event_times: Sequence[float],
    max_duration: float | None = None,
    n_resample: int = 100,
    jitter_size: float = 2.0,
    time_dim: str = "time",
) -> tuple[float, float]:
    """Time-series ZETA test for an event-locked response (zeta, p).

    Wraps `zetapy.zetatstest` (Montijn et al. 2021, eLife). `trace` is one
    cell's continuous dF/F, dims (time,), same input as `circperm_test`.
    ZETA looks for any deviation of the event-locked activity from what
    jittered event times give, over [t_event, t_event + max_duration), so
    it needs no response window or baseline and catches responses of any
    shape or latency. `max_duration` defaults to the shortest gap between
    events. zeta is the responsiveness z-score (> 2 is significant).

    The null jitters each event by up to +/- jitter_size * max_duration.
    Events whose jittered window could run past either end of the trace
    are dropped (zetapy would error on them).

    zetapy jitters events with NumPy's global RNG; it is seeded with SEED
    here and restored afterwards, so results are reproducible.
    """
    assert trace.ndim == 1, "trace must be 1-D (time,)"
    t = trace[time_dim].values.astype(float)
    y = trace.values.astype(float)
    ev = np.sort(np.asarray(event_times, dtype=float))
    if max_duration is None:
        max_duration = float(np.min(np.diff(ev))) if ev.size > 1 else float(t[-1] - ev[0])
    jitter = jitter_size * max_duration
    ev = ev[(ev - jitter >= t[0]) & (ev + jitter + max_duration <= t[-1])]
    if ev.size == 0:
        raise ValueError("no events with jittered windows fully inside the trace")

    p, data = zetatstest(
        t, y, ev, max_duration=max_duration, resampling_number=n_resample,
        jitter_size=jitter_size,
    )
    return float(data["zeta_score"]), float(p)


def zeta_two_test(
    trace: xr.DataArray,
    events: xr.DataArray,
    group_coord: str,
    groups: Sequence | None = None,
    max_duration: float | None = None,
    n_resample: int = 250,
    time_dim: str = "time",
) -> tuple[float, float]:
    """Two-sample time-series ZETA test between two event groups (zeta, p).

    Wraps `zetapy.zetatstest2` (Montijn et al. 2021, eLife). Same question
    as `ks_test` / `perm_test` (does the response to groups[0] differ from
    the response to groups[1]?), but on the continuous trace: `trace` is one
    cell's dF/F, dims (time,), and `events` holds the event times, dims
    (event,), with `group_coord` as an event coordinate with two levels
    (e.g. "stim_type" with values stim / shutter). Pass `groups` to fix
    which two levels to compare.

    ZETA compares the event-locked activity of the two groups over
    [t_event, t_event + max_duration) without averaging it into one number
    per trial, so it picks up differences in shape or latency too. Set
    `max_duration` to the response window (e.g. 2-3 s for GCaMP6s); it
    defaults to the shortest gap between events within a group. Events
    whose window runs past either end of the trace are dropped. Returns
    (nan, nan) if either group has < 3 events. zeta is the z-score
    (> 2 is significant).

    zetapy shuffles with NumPy's global RNG; it is seeded with SEED here
    and restored afterwards, so results are reproducible.
    """
    assert trace.ndim == 1, "trace must be 1-D (time,)"
    t = trace[time_dim].values.astype(float)
    y = trace.values.astype(float)
    ev = np.asarray(events.values, dtype=float)
    g = np.asarray(events[group_coord].values)
    if groups is None:
        groups = np.unique(g)
    if len(groups) != 2:
        raise ValueError(f"expected 2 groups, got {list(groups)}")
    a = np.sort(ev[g == groups[0]])
    b = np.sort(ev[g == groups[1]])
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")
    if max_duration is None:
        max_duration = float(min(np.min(np.diff(a)), np.min(np.diff(b))))
    a = a[(a >= t[0]) & (a + max_duration <= t[-1])]
    b = b[(b >= t[0]) & (b + max_duration <= t[-1])]
    if len(a) < 3 or len(b) < 3:
        return float("nan"), float("nan")

    p, data = zetatstest2(
        t, y, a, t, y, b, max_duration=max_duration, resampling_number=n_resample
    )
    return float(data["zeta_score"]), float(p)


def fdr(pvals, method: str = "bh") -> np.ndarray:
    """FDR-adjusted q-values. method="bh" (Benjamini-Hochberg) or "storey"."""
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    pv = p[ok]
    n = pv.size
    if n == 0:
        return q
    if method == "bh":
        q[ok] = false_discovery_control(pv, method="bh")
        return q
    ranks = np.empty(n, int)
    order = np.argsort(pv)
    ranks[order] = np.arange(1, n + 1)
    pi0 = min(1.0, max(np.mean(pv > 0.5) / 0.5, 1.0 / n))
    qv = pi0 * n * pv / ranks
    qs = qv[order]
    qs = np.minimum.accumulate(qs[::-1])[::-1]      # step-up
    qv[order] = np.clip(qs, 0, 1)
    q[ok] = qv
    return q
