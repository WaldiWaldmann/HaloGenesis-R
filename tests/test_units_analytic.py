import numpy as np
import pytest
import astropy.constants as const
import astropy.units as u
from scipy.integrate import quad

from hgr import analytic as an
from hgr.config import load_config
from hgr.units import G_CODE, KMS_TO_CODE, AxionUnits, jeans_mass, sound_speed


def test_G_code_against_SI():
    ref = const.G.value * const.M_sun.value * u.Myr.to(u.s) ** 2 / u.kpc.to(u.m) ** 3
    assert np.isclose(G_CODE, ref, rtol=1e-12)


def test_hbar_over_m_scaling():
    a1, a2 = AxionUnits(1.0), AxionUnits(2.0)
    assert np.isclose(a1.hbar_over_m / a2.hbar_over_m, 2.0, rtol=1e-14)
    assert np.isclose(a1.hbar_over_m, 0.019607, rtol=1e-4)


def test_sound_speed_research_plan():
    assert np.isclose(sound_speed(1e3, 1.2, 5 / 3) / KMS_TO_CODE, 3.38, rtol=2e-3)


def test_jeans_mass_density_scaling():
    cs = 0.003
    assert np.isclose(jeans_mass(cs, 50.0) / jeans_mass(cs, 1.0), 50**-0.5, rtol=1e-14)


def test_schive_shape_integral_vs_quad():
    for x in [0.3, 1.0, 3.0, 10.0]:
        ref, _ = quad(lambda s: s**2 * (1 + an.SCHIVE_A * s**2) ** -8, 0, x)
        assert np.isclose(an._schive_shape_integral(x), ref, rtol=1e-10)
    tot, _ = quad(lambda s: s**2 * (1 + an.SCHIVE_A * s**2) ** -8, 0, np.inf)
    assert np.isclose(an.SCHIVE_SHAPE_TOTAL, tot, rtol=1e-10)


def test_schive_mass_radius_roundtrip():
    rc = an.schive_core_radius_from_mass(1.0, 2.92e8)
    assert np.isclose(an.schive_total_mass(1.0, rc), 2.92e8, rtol=1e-12)
    m22 = an.schive_m22_for(2.92e8, 0.5)
    assert np.isclose(an.schive_core_radius_from_mass(m22, 2.92e8), 0.5, rtol=1e-12)


def test_config_rejects_missing_key(tmp_path):
    import yaml
    from pathlib import Path

    cfg = yaml.safe_load(Path("configs/default.yaml").read_text(encoding="utf-8"))
    del cfg["gas"]["T_K"]
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(p)


def test_config_parses_exponent_floats():
    cfg = load_config("configs/default.yaml")
    assert isinstance(cfg["axion"]["soliton_mass_msun"], float)
    assert cfg["axion"]["soliton_mass_msun"] == 2.92e8
