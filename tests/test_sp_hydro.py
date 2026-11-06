"""Schnelle Regressionstests für SP-Solver (P2) und Hydro/Kühlung (P3)."""
import numpy as np

from hgr import cooling as co
from hgr import hydro as hy
from hgr.grid import Grid3D
from hgr.schroedinger import mass, potential, step_kdk
from hgr.soliton import ground_state_profile, upsample_spectral
from hgr.units import G_CODE, AxionUnits, gas_mass_density

A = AxionUnits(1.0).hbar_over_m


def test_ground_state_literature_values():
    p = ground_state_profile()
    assert abs(p.mu - 25.91) < 0.02 and abs(p.u0 + 0.6495) < 1e-3


def test_kdk_norm_conservation():
    g = Grid3D(8.0, 16)
    rng = np.random.default_rng(0)
    psi = (1e4 * np.exp(-g.r**2) * (1 + 0.1 * rng.standard_normal(g.shape))).astype(complex)
    phi = potential(psi, None, g, G_CODE, 1)
    m0 = mass(psi, g)
    for _ in range(20):
        psi, phi = step_kdk(psi, phi, None, g, 0.05, A, G_CODE, 1)
    assert abs(mass(psi, g) / m0 - 1) < 1e-13


def test_upsample_is_exact_for_bandlimited():
    g1, g2 = Grid3D(8.0, 16), Grid3D(8.0, 32)
    k = 2 * np.pi / 8.0
    f = lambda gr: np.exp(1j * k * gr.mesh_sparse[0]) * np.cos(2 * k * gr.mesh_sparse[1]) + 0 * gr.r  # noqa: E731
    assert np.max(np.abs(upsample_spectral(f(g1), 32, 1) - f(g2))) < 1e-13
    assert np.max(np.abs(upsample_spectral(f(g1), 48, 1) - f(Grid3D(8.0, 48)))) < 1e-13


def test_hllc_consistency_uniform_flow():
    U = hy.cons_from_prim(np.full((8, 8, 8), 1.3), 0.4, -0.2, 0.1, 0.7, 5 / 3)
    U0 = U.copy()
    hy.step(U, 0.01, 1 / 8, 5 / 3, "vanleer", 0)
    assert np.max(np.abs(U - U0)) < 1e-14


def test_hydro_axis_symmetry_and_conservation():
    n = 16
    x = (np.arange(n) + 0.5) / n
    res = []
    for ax in range(3):
        sh = [n if i == ax else 4 for i in range(3)]
        xs = x.reshape([-1 if i == ax else 1 for i in range(3)])
        rho = (1 + 0.5 * np.sin(2 * np.pi * xs)) * np.ones(sh)
        U = hy.cons_from_prim(rho, 0, 0, 0, rho**1.4, 1.4)
        T0 = hy.totals(U, 1.0)
        for s in range(5):
            hy.step(U, hy.cfl_dt(U, 1 / n, 1.4, 0.4), 1 / n, 1.4, "vanleer", s)
        assert np.allclose(hy.totals(U, 1.0), T0, rtol=1e-13, atol=1e-13)
        res.append(np.moveaxis(U[0], ax, 0)[:, 0, 0])
    assert np.max(np.abs(res[0] - res[1])) < 1e-14 and np.max(np.abs(res[0] - res[2])) < 1e-14


def test_cooling_historical_drops_to_300K():
    rho = np.full((2, 2, 2), gas_mass_density(10.0, 1.2))
    U = hy.cons_from_prim(rho, 0, 0, 0, co.pressure_from_T(rho, 1e3, 1.2), 5 / 3)
    co.apply_cooling(U, 1e-6, "piecewise_historical", 5 / 3, 1.2, None)
    assert abs(co.temperature(U, 5 / 3, 1.2)[0, 0, 0] - 300.0) < 1e-6
