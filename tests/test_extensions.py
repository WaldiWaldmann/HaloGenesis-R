import copy

import numpy as np

from hgr import cooling as co
from hgr import hydro as hy
from hgr.config import load_config
from hgr.grid import Grid3D
from hgr.observables import truelove_number
from hgr.setup import gas_initial_condition, random_phase_halo
from hgr.units import G_CODE, gas_mass_density

CFG = load_config("configs/physical_fb.yaml")


def test_jeans_floor_exact_and_idempotent():
    g = Grid3D(8.0, 16)
    rho = gas_mass_density(10.0, 1.2) * np.exp(2 * np.random.default_rng(0).standard_normal(g.shape))
    U = hy.cons_from_prim(rho, 0, 0, 0, co.pressure_from_T(rho, 300.0, 1.2), 5 / 3)
    hy.apply_jeans_floor(U, 5 / 3, G_CODE, g.dx, 4.0)
    assert truelove_number(U, 5 / 3, G_CODE, g.dx) <= 0.25 * (1 + 1e-11)
    e2, m2 = hy.apply_jeans_floor(U, 5 / 3, G_CODE, g.dx, 4.0)
    assert e2 == 0.0 and not m2.any()


def test_floor_off_is_noop():
    U = hy.cons_from_prim(np.ones((4, 4, 4)), 0, 0, 0, 1e-30, 5 / 3)
    U0 = U.copy()
    hy.apply_jeans_floor(U, 5 / 3, G_CODE, 0.1, 0.0)
    assert np.array_equal(U, U0)


def test_primordial_zero_below_floor_and_positive_above():
    cp = CFG["gas"]["cooling_params"]
    assert co.primordial_rate_cgs(1.0, cp["T_floor_K"] - 1, cp["x_H2"], cp["T_floor_K"]) == 0.0
    assert co.primordial_rate_cgs(1.0, 1e3, cp["x_H2"], cp["T_floor_K"]) > 0.0


def test_halo_mass_and_determinism():
    g = Grid3D(8.0, 24)
    h1 = random_phase_halo(g, 1e9, 0.03, 1.5, 0.0196, 7, 1)
    h2 = random_phase_halo(g, 1e9, 0.03, 1.5, 0.0196, 7, 1)
    assert np.array_equal(h1, h2)
    assert abs(np.sum(np.abs(h1) ** 2) * g.cell_volume / 1e9 - 1) < 1e-12


def test_seed_changes_gas_ic():
    g = Grid3D(8.0, 8)
    c2 = copy.deepcopy(CFG)
    c2["meta"]["seed"] += 1
    assert not np.array_equal(gas_initial_condition(CFG, g, 1.0), gas_initial_condition(c2, g, 1.0))


def test_halo_same_realization_for_all_N():
    a = 0.0196
    g1, g2 = Grid3D(8.0, 32), Grid3D(8.0, 64)
    h1 = random_phase_halo(g1, 1e9, 0.03, 1.5, a, 7, 1, n_ref=32)
    h2 = random_phase_halo(g2, 1e9, 0.03, 1.5, a, 7, 1, n_ref=32)
    c = np.corrcoef(np.abs(h1).ravel(), np.abs(h2[::2, ::2, ::2]).ravel())[0, 1]
    assert c > 0.999
