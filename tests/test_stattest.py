import numpy as np
import xarray as xr
from scipy.stats import f_oneway

from statresp.stattest import anova_test


def _make_cube(seed: int = 0) -> xr.DataArray:
    """(ucid, trial, time) cube: cell 0 tuned to direction, cell 1 flat."""
    rng = np.random.default_rng(seed)
    dirs = np.repeat([0, 90, 180, 270], 10)
    sfs = np.tile([0.04, 0.08], 20)
    tfs = np.full(40, 2.0)
    time = np.arange(-0.5, 2.0, 0.1)
    n_trial, n_time = len(dirs), len(time)

    tuned = (dirs == 90).astype(float)[:, None] * (time >= 0)[None, :]
    data = rng.normal(0, 0.3, size=(2, n_trial, n_time))
    data[0] += 2.0 * tuned
    return xr.DataArray(
        data,
        dims=("ucid", "trial", "time"),
        coords={"ucid": [10, 11], "time": time,
                "direction": ("trial", dirs), "sf": ("trial", sfs), "tf": ("trial", tfs)},
    )


def test_anova_single_matches_scipy():
    cube = _make_cube()
    f = cube.isel(ucid=0)
    F, p = anova_test(f, window=(0, 2))

    fw = f.isel(time=(f.time.values >= 0) & (f.time.values < 2))
    tm = fw.mean("time").values
    keys = [f"{d}|{s}|{t}" for d, s, t in zip(f.direction.values, f.sf.values, f.tf.values)]
    keys = np.array(keys)
    F_ref, p_ref = f_oneway(*[tm[keys == k] for k in np.unique(keys)])

    assert isinstance(F, float) and isinstance(p, float)
    np.testing.assert_allclose([F, p], [F_ref, p_ref])


def test_anova_batched_matches_loop():
    cube = _make_cube()
    F, p = anova_test(cube, window=(0, 2))
    assert F.shape == p.shape == (2,)
    for i in range(cube.sizes["ucid"]):
        F_i, p_i = anova_test(cube.isel(ucid=i), window=(0, 2))
        np.testing.assert_allclose([F[i], p[i]], [F_i, p_i])


def test_anova_tuned_vs_flat():
    _, p = anova_test(_make_cube(), window=(0, 2))
    assert p[0] < 1e-6
    assert p[1] > 0.01
