import numpy as np

from hgr import observables as ob
from hgr.coupled import TrailingBoxcar
from hgr.grid import Grid3D


def test_aperture_volume():
    g = Grid3D(8.0, 64)
    w = ob.aperture_weights(g, 1.0)
    assert abs(w.sum() * g.cell_volume / (4 / 3 * np.pi) - 1) < 1e-2


def test_flux_identity_for_uniform_translation():
    g = Grid3D(8.0, 32)
    w = ob.aperture_weights(g, 1.5)
    U = np.zeros((5,) + g.shape)
    U[0], U[1] = 2.0, 2.0 * 0.3
    assert abs(ob.flux_rate(U, ob.grad_weights(w, g.dx), g.cell_volume)) < 1e-12


def test_occupancy_interpolates_crossings():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    q = np.exp(np.array([1.0, -1.0, -1.0, 1.0]))
    frac, iv = ob.occupancy(t, q, 1.0)
    assert np.allclose(iv, [(0.5, 2.5)]) and np.isclose(frac, 2 / 3)


def test_o3_bounds_and_undefined():
    rng = np.random.default_rng(0)
    sh = (8, 8, 8)
    rho, w = rng.random(sh) + 0.1, np.ones(sh)
    dg = tuple(rng.standard_normal(sh) for _ in range(3))
    assert np.isclose(ob.o3_coherence(rho, dg, dg, w), 1.0)
    assert np.isclose(ob.o3_coherence(rho, dg, tuple(-x for x in dg), w), -1.0)
    assert np.isnan(ob.o3_coherence(rho, dg, tuple(0 * x for x in dg), w))


def test_filter_constant_exact_and_tau0_identity():
    f = TrailingBoxcar(2.0)
    for t0, t1 in [(0, 0.3), (0.3, 1.1), (1.1, 1.5), (1.5, 2.9)]:
        f.push(t0, t1, np.array([4.0 * (t1 - t0)]))
    assert np.isclose(f.mean(2.9, np.array([4.0]))[0], 4.0, rtol=1e-14)
    assert TrailingBoxcar(0.0).mean(1.0, np.array([7.0]))[0] == 7.0
