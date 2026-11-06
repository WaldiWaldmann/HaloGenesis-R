import numpy as np

from hgr import analytic as an
from hgr.grid import Grid3D
from hgr.poisson import gradient_spectral, poisson_residual, solve_poisson_periodic
from hgr.units import G_CODE

W = 1


def test_single_mode_exact():
    L = 8.0
    g = Grid3D(L, 32)
    x, y, z = g.mesh_sparse
    k = 2 * np.pi / L * np.array([1.0, 0.0, 2.0])
    ph = k[0] * x + k[2] * z + 0 * y
    rho = 1e8 + 1e6 * np.cos(ph)
    phi = solve_poisson_periodic(rho, g, G_CODE, W)
    ex = -4 * np.pi * G_CODE * 1e6 * np.cos(ph) / (k @ k)
    assert np.max(np.abs(phi - ex)) / np.max(np.abs(ex)) < 1e-12
    gx = gradient_spectral(phi, g, W)[0]
    gx_ex = 4 * np.pi * G_CODE * 1e6 * k[0] * np.sin(ph) / (k @ k)
    assert np.max(np.abs(gx - gx_ex)) / np.max(np.abs(gx_ex)) < 1e-12


def test_zero_mean_and_constant_invariance():
    g = Grid3D(8.0, 32)
    rho = an.gaussian_density(g.r, 1e8, 0.5)
    p1 = solve_poisson_periodic(rho, g, G_CODE, W)
    p2 = solve_poisson_periodic(rho + 1e9, g, G_CODE, W)
    assert abs(p1.mean()) < 1e-14 * np.max(np.abs(p1))
    assert np.max(np.abs(p1 - p2)) < 1e-12 * np.max(np.abs(p1))


def test_periodic_background_correction():
    L = 8.0
    g = Grid3D(L, 64)
    rho = an.gaussian_density(g.r, 1e8, 0.4)
    phi = solve_poisson_periodic(rho, g, G_CODE, W)
    d = (
        phi
        - an.gaussian_potential_isolated(g.r, 1e8, 0.4, G_CODE)
        - an.periodic_background_potential(g.r, g.integrate(rho) / L**3, G_CODE)
    )
    inner = g.r < L / 4
    d -= d[inner].mean()
    scale = G_CODE * 1e8 * np.sqrt(2 / np.pi) / 0.4
    assert np.max(np.abs(d[inner])) / scale < 1e-3


def test_spectral_residual_roundoff():
    g = Grid3D(8.0, 32)
    rho = an.gaussian_density(g.r, 1e8, 0.6)
    phi = solve_poisson_periodic(rho, g, G_CODE, W)
    assert poisson_residual(phi, rho, g, G_CODE, "spectral", W) < 1e-12


def test_grid_origin_on_node():
    g = Grid3D(8.0, 16)
    assert g.x1d[g.center_index] == 0.0
    assert g.r[g.center_index, g.center_index, g.center_index] == 0.0
