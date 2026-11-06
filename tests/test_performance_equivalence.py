"""Optimierte Pfade müssen den Referenzpfaden entsprechen (bitgleich bzw. auf Rundungsniveau)."""
import numpy as np
import scipy.fft as sfft

from hgr import cooling as co
from hgr import hydro as hy
from hgr.grid import Grid3D
from hgr.kernels import center_weights_rfft
from hgr.schroedinger import drift, kick, kick_inplace
from hgr.units import gas_mass_density


def _random_state(n, seed=0):
    rng = np.random.default_rng(seed)
    rho = 1 + 0.4 * rng.random((n, n, n))
    return hy.cons_from_prim(rho, 0.3 * rng.standard_normal((n, n, n)), 0.2 * rng.standard_normal((n, n, n)),
                             0.1 * rng.standard_normal((n, n, n)), 1 + 0.3 * rng.random((n, n, n)), 5 / 3)


def test_hydro_sweeps_bitwise_identical():
    for ax in range(3):
        U1 = _random_state(12, ax)
        U2 = U1.copy()
        n1 = hy.sweep(U1, ax, 0.01, 1 / 12, 5 / 3, "vanleer")
        n2 = hy.sweep_reference(U2, ax, 0.01, 1 / 12, 5 / 3, "vanleer")
        assert n1 == n2 and np.array_equal(U1, U2)


def test_cooling_numba_matches_numpy():
    cp = {"X_H": 0.76, "x_H2": 5e-4, "T_floor_K": 57.2}
    rng = np.random.default_rng(1)
    rho = gas_mass_density(1.0, 1.2) * np.exp(2 * rng.standard_normal((6, 6, 6)))
    for model, Tmax in (("piecewise_historical", 3e4), ("primordial_h2", 1e6)):
        T = np.exp(rng.uniform(np.log(50), np.log(Tmax), rho.shape))
        U1 = hy.cons_from_prim(rho, 0.01, 0, 0, co.pressure_from_T(rho, T, 1.2), 5 / 3)
        U2 = U1.copy()
        co.apply_cooling(U1, 0.3, model, 5 / 3, 1.2, None, cparams=cp)
        co.apply_cooling_reference(U2, 0.3, model, 5 / 3, 1.2, None, cparams=cp)
        assert np.max(np.abs(U1[4] / U2[4] - 1)) < 1e-12


def test_kick_inplace_and_drift_cache():
    g = Grid3D(4.0, 16)
    rng = np.random.default_rng(2)
    psi = rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape)
    phi = rng.standard_normal(g.shape)
    a = kick(psi, phi, 0.3, 0.02)
    b = psi.copy()
    kick_inplace(b, phi, 0.3, 0.02)
    assert np.max(np.abs(a - b)) < 1e-12
    d1 = drift(psi.copy(), g, 0.5, 0.02, 1)
    d2 = drift(psi.copy(), g, 0.5, 0.02, 1)  # aus dem Cache
    ref = sfft.ifftn(sfft.fftn(psi) * np.exp(-0.5j * 0.02 * 0.5 * g.k2_full))
    assert np.array_equal(d1, d2) and np.max(np.abs(d1 - ref)) < 1e-12


def test_center_value_from_spectrum():
    n = 16
    f = np.random.default_rng(3).standard_normal((n, n, n))
    val = np.real(np.sum(sfft.rfftn(f) * center_weights_rfft(n)))
    assert abs(val - f[n // 2, n // 2, n // 2]) < 1e-12


def test_phibar_at_arbitrary_point():
    from hgr.coupled import Branch, TrailingBoxcar
    n = 16
    g = Grid3D(4.0, n)
    f = np.random.default_rng(4).standard_normal((n, n, n))
    b = Branch(1, np.zeros((5, n, n, n)), None, f, sfft.rfftn(f), TrailingBoxcar(0.0), g, 1)
    b.phibar_hat = sfft.rfftn(f)
    for ci in [(8, 8, 8), (3, 11, 5), (0, 15, 9)]:
        assert abs(b.phibar_at(ci) - f[ci]) < 1e-12
