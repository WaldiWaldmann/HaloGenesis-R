from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .filters import transfer_function
from .units import G_CODE, KMS_TO_CODE, AxionUnits, sound_speed


def granule_spectrum(k, omega, sigma: float, a: float):
    k = np.asarray(k, float)
    w = np.asarray(omega, float)
    b = a**2 / (4 * sigma**2)
    C = 8 * np.pi**1.5 * b**1.5
    s2 = np.maximum(0.5 * sigma**2 * k**2, 1e-300)
    gauss = np.exp(-0.5 * w**2 / s2) / np.sqrt(2 * np.pi * s2)
    return C * np.exp(-b * k**2) * 2 * np.pi * gauss


def spatial_spectrum(k, sigma: float, a: float):
    b = a**2 / (4 * sigma**2)
    return 8 * np.pi**1.5 * b**1.5 * np.exp(-b * np.asarray(k, float) ** 2)


def temporal_width(k, sigma: float):
    return sigma * np.asarray(k, float) / np.sqrt(2)


def susceptibility(k, omega, cs: float, rho_g: float, damping: float, G: float = G_CODE):
    k = np.asarray(k, float)
    w = np.asarray(omega, float)
    ws2 = cs**2 * k**2 - 4 * np.pi * G * rho_g
    return -(k**2) / (ws2 - w**2 + 1j * damping * w)


def transfer(k, omega, cs: float, rho_g: float, rho_dm: float, damping: float, H=1.0, G: float = G_CODE):
    k = np.asarray(k, float)
    w = np.asarray(omega, float)
    ws2 = cs**2 * k**2 - 4 * np.pi * G * rho_g
    return 4 * np.pi * G * rho_dm * H / (ws2 - w**2 + 1j * damping * w)


def highpass_centered(omega, window: float):
    if window <= 0:
        return np.ones_like(np.asarray(omega, float))
    x = 0.5 * np.asarray(omega, float) * window
    return 1.0 - np.sinc(x / np.pi)


@dataclass(frozen=True)
class ResponseSetup:
    sigma: float
    a: float
    rho_dm: float
    rho_g: float
    cs: float
    damping: float
    kmin: float
    filter_kind: str = "trailing_boxcar"
    tau: float = 0.0
    highpass: float = 0.0
    self_gravity: bool = True

    @property
    def mach(self) -> float:
        return self.cs / self.sigma

    def with_mach(self, M: float) -> "ResponseSetup":
        return replace(self, cs=M * self.sigma)


def _omega_grid(s: float, wr: float, gamma: float, n_core: int = 601, n_res: int = 401) -> np.ndarray:
    parts = [np.linspace(-8 * s, 8 * s, n_core)]
    if wr > 0 and wr < 12 * s:
        dw = max(30 * gamma, 1e-3 * wr)
        for c in (wr, -wr):
            parts.append(np.linspace(c - dw, c + dw, n_res))
    return np.unique(np.concatenate(parts))


def point_statistics(st: ResponseSetup, lags=None, nk: int = 200) -> dict:
    kstar = 2 * st.sigma / st.a
    k = np.geomspace(st.kmin, max(8 * kstar, 2 * st.kmin), nk)
    lags = np.linspace(0, 20, 81) if lags is None else np.asarray(lags, float)
    rg = st.rho_g if st.self_gravity else 0.0
    C = np.zeros((nk, len(lags)), complex)
    VG = np.zeros(nk)
    VD = np.zeros(nk)
    for i, kk in enumerate(k):
        s = temporal_width(kk, st.sigma)
        ws2 = st.cs**2 * kk**2 - 4 * np.pi * G_CODE * rg
        w = _omega_grid(s, np.sqrt(ws2) if ws2 > 0 else 0.0, st.damping)
        P = granule_spectrum(kk, w, st.sigma, st.a) * np.abs(highpass_centered(w, st.highpass)) ** 2
        T = transfer(kk, w, st.cs, rg, st.rho_dm, st.damping, transfer_function(st.filter_kind, w, st.tau))
        VD[i] = np.trapezoid(P, w) / (2 * np.pi)
        VG[i] = np.trapezoid(np.abs(T) ** 2 * P, w) / (2 * np.pi)
        C[i] = np.trapezoid((T * P)[None, :] * np.exp(1j * np.outer(lags, w)), w, axis=1) / (2 * np.pi)
    wk = 4 * np.pi * k**2 / (2 * np.pi) ** 3
    cross = np.trapezoid(wk[:, None] * C, k, axis=0)
    var_g = float(np.trapezoid(wk * VG, k))
    var_d = float(np.trapezoid(wk * VD, k))
    r = np.real(cross) / np.sqrt(var_g * var_d)
    j = int(np.argmax(np.abs(r)))
    with np.errstate(invalid="ignore", divide="ignore"):
        per_k = {"k": k, "eta_k": VG / VD, "r0_k": np.real(C[:, 0]) / np.sqrt(VG * VD)}
    return {
        "r0": float(r[0]),
        "r_max": float(r[j]),
        "lag_at_max": float(lags[j]),
        "eta": var_g / var_d,
        "r_lag": r,
        "lags": lags,
        "var_g": var_g,
        "var_dm": var_d,
        "per_k": per_k,
        "mach": st.mach,
    }


def jeans_mach(st: ResponseSetup) -> float:
    return float(np.sqrt(4 * np.pi * G_CODE * st.rho_g) / (st.sigma * st.kmin))


def mach_scan(st: ResponseSetup, machs, lags=None) -> dict:
    out = {"mach": np.asarray(machs, float), "r0": [], "r_max": [], "lag": [], "eta": []}
    for M in machs:
        res = point_statistics(st.with_mach(M), lags=lags)
        out["r0"].append(res["r0"]), out["r_max"].append(res["r_max"])
        out["lag"].append(res["lag_at_max"]), out["eta"].append(res["eta"])
    return {k_: np.asarray(v) if isinstance(v, list) else v for k_, v in out.items()}


def filter_sensitivity(st: ResponseSetup, machs, tau: float, kind: str = "trailing_boxcar") -> dict:
    base = mach_scan(replace(st, tau=0.0), machs, lags=[0.0])
    filt = mach_scan(replace(st, tau=tau, filter_kind=kind), machs, lags=[0.0])
    return {"mach": base["mach"], "d_eta_rel": filt["eta"] / base["eta"] - 1, "d_r0": filt["r0"] - base["r0"]}


def setup_from_run_meta(
    meta: dict,
    rho_dm: float,
    rho_g: float,
    box_kpc: float,
    damping: float,
    highpass: float,
    tau: float | None = None,
    filter_kind: str | None = None,
) -> ResponseSetup:
    a = AxionUnits(meta["m22"]).hbar_over_m
    cs = sound_speed(meta["T_K"], meta["mu"], meta["gamma"])
    return ResponseSetup(
        sigma=meta["halo_sigma_kms"] * KMS_TO_CODE,
        a=a,
        rho_dm=rho_dm,
        rho_g=rho_g,
        cs=cs,
        damping=damping,
        kmin=2 * np.pi / box_kpc,
        filter_kind=filter_kind or meta.get("filter_kind", "trailing_boxcar"),
        tau=meta["tau"] if tau is None else tau,
        highpass=highpass,
    )
