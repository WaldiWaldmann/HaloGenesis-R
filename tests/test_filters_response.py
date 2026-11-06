import numpy as np
import pytest

from hgr import response as rp
from hgr.filters import KERNELS, make_filter, mean_lag, transfer_function
from hgr.grid import Grid3D
from hgr.setup import fibonacci_probes
from hgr.units import AxionUnits, KMS_TO_CODE


def _feed_linear(kind, tau, slope=0.7, t_end=40.0, dt=0.05):
    f = make_filter(kind, tau)
    f.reset(0.0, np.array([0.0]))
    t = 0.0
    while t < t_end - 1e-12:
        t1 = t + dt
        f.push(t, t1, np.array([0.5 * slope * (t1**2 - t**2)]), np.array([slope * t1]))
        t = t1
    return f.value(t, np.array([slope * t]))[0], slope * t


@pytest.mark.parametrize("kind", KERNELS)
def test_mean_lag_on_linear_ramp(kind):
    tau = 4.0
    y, x = _feed_linear(kind, tau)
    assert abs((x - y) / 0.7 - mean_lag(kind, tau)) < 1e-6 * tau + 5e-3


def test_transfer_function_limits():
    w = np.array([1e-9])
    for kind in KERNELS:
        assert abs(transfer_function(kind, w, 3.0)[0] - 1) < 1e-6
    assert np.allclose(transfer_function("delay_half_tau", np.array([2.0]), 1.0), np.exp(-1j))


@pytest.mark.parametrize("kind", KERNELS)
def test_tau_zero_is_identity(kind):
    f = make_filter(kind, 0.0)
    f.reset(0.0, np.array([5.0]))
    f.push(0.0, 1.0, np.array([5.5]), np.array([6.0]))
    assert f.value(1.0, np.array([6.0]))[0] == 6.0


def test_granule_spectrum_normalisation():
    a, sig = AxionUnits(1).hbar_over_m, 30 * KMS_TO_CODE
    k = np.linspace(0, 25, 40001)
    assert abs(np.trapezoid(4 * np.pi * k**2 / (2 * np.pi) ** 3 * rp.spatial_spectrum(k, sig, a), k) - 1) < 1e-6
    w = np.linspace(-3, 3, 60001)
    assert (
        abs(np.trapezoid(rp.granule_spectrum(2.0, w, sig, a), w) / (2 * np.pi) / rp.spatial_spectrum(2.0, sig, a) - 1)
        < 1e-6
    )


def test_susceptibility_static_and_resonance_sign():
    k, cs = 3.0, 0.05
    chi0 = rp.susceptibility(k, 0.0, cs, 0.0, 0.0)
    assert np.isclose(chi0, -1 / cs**2)
    ws = cs * k
    assert rp.susceptibility(k, 0.5 * ws, cs, 0.0, 0.0).real < 0 < rp.susceptibility(k, 2 * ws, cs, 0.0, 0.0).real


def test_point_statistics_regimes():
    a, sig = AxionUnits(1).hbar_over_m, 30 * KMS_TO_CODE
    st = rp.ResponseSetup(
        sigma=sig, a=a, rho_dm=3e7, rho_g=8e6, cs=0.1 * sig, damping=0.05, kmin=2 * np.pi / 8, highpass=6.0
    )
    cold = rp.point_statistics(st.with_mach(0.05), lags=[0.0], nk=80)
    hot = rp.point_statistics(st.with_mach(4.0), lags=[0.0], nk=80)
    assert cold["r0"] < -0.3 and hot["r0"] > 0.5 and hot["eta"] < cold["eta"]


def test_fibonacci_probes_on_shells():
    g = Grid3D(8.0, 64)
    idx, rad = fibonacci_probes(g, [2.0, 3.0], 48)
    r = np.linalg.norm((idx - g.center_index) * g.dx, axis=1)
    assert len(np.unique(idx, axis=0)) == len(idx) and np.max(np.abs(r - rad)) < g.dx
