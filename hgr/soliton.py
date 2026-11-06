"""Numerischer SP-Grundzustand (ersetzt den empirischen Schive-Fit).

Dimensionslos (ξ = r/L₀, τ = t a/L₀², ψ = A χ) lautet das System
    i∂_τ χ = −½∇²χ + Vχ,  ∇²V = |χ|².
Stationär χ = f(ξ)e^{−iετ}, U := V − ε, f(0) = 1:
    f'' + (2/ξ) f' = 2 U f,   U'' + (2/ξ) U' = f².
U(0) wird per Bisektion so gewählt, dass f knotenfrei gegen 0 fällt (Grundzustand).

Rückskalierung (Skalensymmetrie der SP-Gleichung): bei Masse M und a = ħ/m
    μ = 4π∫f²ξ²dξ,  L₀ = a²μ/(4πG M),  A = a/(√(4πG) L₀²),  ψ(r) = A f(r/L₀).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.fft as sfft
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import interp1d

from .grid import Grid3D
from .schroedinger import drift, kick, mass, potential


@dataclass(frozen=True)
class GroundStateProfile:
    xi: np.ndarray       # dimensionslose Radien
    f: np.ndarray        # Profil, f(0) = 1
    u0: float            # U(0)
    mu: float            # 4π∫f²ξ²dξ
    xi_c: float          # f²(ξ_c) = 1/2  (Kernradius)


def _rhs(xi, y):
    f, fp, U, Up = y
    return [fp, 2 * U * f - 2 * fp / xi, Up, f * f - 2 * Up / xi]


def _shoot(u0: float, xi_max: float):
    xi0 = 1e-6
    y0 = [1 + u0 * xi0**2 / 3, 2 * u0 * xi0 / 3, u0 + xi0**2 / 6, xi0 / 3]
    ev_zero = lambda xi, y: y[0]  # noqa: E731
    ev_zero.terminal, ev_zero.direction = True, -1
    ev_up = lambda xi, y: y[1]  # noqa: E731
    ev_up.terminal, ev_up.direction = True, 1
    sol = solve_ivp(_rhs, (xi0, xi_max), y0, method="DOP853", rtol=1e-12, atol=1e-14,
                    events=(ev_zero, ev_up), dense_output=True)
    if sol.t_events[0].size:
        return "low", sol
    if sol.t_events[1].size:
        return "high", sol
    return "none", sol


def ground_state_profile() -> GroundStateProfile:
    lo, hi = -3.0, 0.0
    assert _shoot(lo, 50)[0] == "low" and _shoot(hi, 50)[0] == "high"
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        kind, _ = _shoot(mid, 50)
        if kind == "low":
            lo = mid
        elif kind == "high":
            hi = mid
        else:
            raise RuntimeError("Shooting ohne Ereignis – xi_max zu klein")
        if hi - lo < 1e-15:
            break
    _, sol = _shoot(hi, 50)  # dreht nach oben -> bis zum Minimum gültig
    xi_m = sol.t_events[1][0]
    xi_in = np.linspace(1e-6, xi_m, 4000)
    f_in, _, U_in, _ = sol.sol(xi_in)
    # Asymptotik f ~ C e^{−kξ}/ξ mit k = √(2U(ξ_m)) für ξ > ξ_m
    k = np.sqrt(2 * U_in[-1])
    xi_out = np.linspace(xi_m, xi_m + 40.0 / k, 2000)[1:]
    f_out = f_in[-1] * (xi_m / xi_out) * np.exp(-k * (xi_out - xi_m))
    xi = np.concatenate([xi_in, xi_out])
    f = np.concatenate([f_in, f_out])
    mu = 4 * np.pi * trapezoid(f**2 * xi**2, xi)
    xi_c = float(interp1d(f[: len(xi_in)][::-1] ** 2, xi_in[::-1])(0.5))
    return GroundStateProfile(xi, f, hi, mu, xi_c)


def physical_scales(prof: GroundStateProfile, mass_msun: float, a: float, G: float) -> dict[str, float]:
    L0 = a**2 * prof.mu / (4 * np.pi * G * mass_msun)
    A = a / (np.sqrt(4 * np.pi * G) * L0**2)
    return {"L0": L0, "A": A, "rc": prof.xi_c * L0, "rho0": A**2, "T0": L0**2 / a}


def radial_psi(prof: GroundStateProfile, r: np.ndarray, sc: dict) -> np.ndarray:
    return sc["A"] * np.interp(np.asarray(r) / sc["L0"], prof.xi, prof.f, right=0.0)


def hamiltonian_residual(psi: np.ndarray, phi: np.ndarray, grid: Grid3D, a: float, workers: int) -> float:
    """‖Hψ − ⟨H⟩ψ‖ / ‖(a/2)∇²ψ‖: Eigenzustands-Residuum des diskreten Hamiltonoperators,
    normiert auf den kinetischen Term (⟨H⟩ selbst kann wegen ⟨Φ⟩ = 0 nahe null liegen)."""
    lap = sfft.ifftn(-grid.k2_full * sfft.fftn(psi, workers=workers), workers=workers)
    hpsi = -0.5 * a * lap + (phi / a) * psi
    e = np.vdot(psi, hpsi).real / np.vdot(psi, psi).real
    return float(np.linalg.norm(hpsi - e * psi) / np.linalg.norm(0.5 * a * lap))


def upsample_spectral(psi: np.ndarray, n_new: int, workers: int) -> np.ndarray:
    """Bandbegrenzte Interpolation per Fourier-Zero-Padding (Nyquist-Mode wird verworfen).
    Beide Gitter beginnen bei x = −L/2 -> für jedes n_new ≥ n exakt (bandbegrenztes Trig.-Polynom)."""
    n = psi.shape[0]
    if n_new < n:
        raise ValueError(f"n_new={n_new} < n={n}: Downsampling nicht vorgesehen")
    pk = sfft.fftn(psi, workers=workers)
    out = np.zeros((n_new,) * 3, dtype=np.complex128)
    h = n // 2
    idx_old = np.r_[0:h, n - h + 1:n]
    idx_new = np.r_[0:h, n_new - h + 1:n_new]
    out[np.ix_(idx_new, idx_new, idx_new)] = pk[np.ix_(idx_old, idx_old, idx_old)]
    return sfft.ifftn(out, workers=workers) * (n_new / n) ** 3


def _polish(psi, grid, mass_msun, a, G, dtaus, workers, rho_b):
    """Imaginäre Zeit (gleicher Split-Operator, t → −iτ), Masse nach jedem Teilschritt exakt M."""
    phi = potential(psi, rho_b, grid, G, workers)
    for dtau in dtaus:
        psi = kick(psi, phi, -0.5j * dtau, a)          # exp(−Φ dτ/(2a))
        psi = drift(psi, grid, -1j * dtau, a, workers)   # exp(−a k² dτ/2)
        psi *= np.sqrt(mass_msun / mass(psi, grid))
        phi = potential(psi, rho_b, grid, G, workers)
        psi = kick(psi, phi, -0.5j * dtau, a)
        psi *= np.sqrt(mass_msun / mass(psi, grid))
        phi = potential(psi, rho_b, grid, G, workers)
    return psi, phi


def embed_ground_state(grid: Grid3D, mass_msun: float, a: float, G: float, polish: dict,
                       workers: int, rho_b: np.ndarray | None = None):
    """Soliton im Gitterzentrum als Grundzustand des DISKRETEN PERIODISCHEN Systems.

    1) Shooting-Profil auf grobes Gitter (polish['coarse_n']) legen,
    2) imaginäre Zeit mit geometrisch fallendem dτ (entfernt Boxeffekt + Splitting-Bias),
    3) spektral aufs Zielgitter interpolieren, 4) kurze Feinpolitur.
    dτ-Angaben in Einheiten der Soliton-Zeitskala T₀ = L₀²/a. rho_b nur auf Zielgitter erlaubt."""
    prof = ground_state_profile()
    sc = physical_scales(prof, mass_msun, a, G)
    T0 = sc["T0"]
    psi_direct = radial_psi(prof, grid.r, sc).astype(np.complex128)
    psi_direct *= np.sqrt(mass_msun / mass(psi_direct, grid))
    res_direct = hamiltonian_residual(psi_direct, potential(psi_direct, rho_b, grid, G, workers), grid, a, workers)

    nc = polish["coarse_n"]
    if nc and nc < grid.n:
        if rho_b is not None:
            raise NotImplementedError("Grobgitter-Politur mit Gas: erst auf Zielgitter (P4)")
        gc = Grid3D(grid.box, nc)
        psi_c = radial_psi(prof, gc.r, sc).astype(np.complex128)
        psi_c *= np.sqrt(mass_msun / mass(psi_c, gc))
        d0, d1 = polish["coarse_dtau_T0"]
        psi_c, _ = _polish(psi_c, gc, mass_msun, a, G, np.geomspace(d0, d1, polish["coarse_steps"]) * T0,
                           workers, None)
        psi = upsample_spectral(psi_c, grid.n, workers)
        psi *= np.sqrt(mass_msun / mass(psi, grid))
    else:
        psi = psi_direct
    d0, d1 = polish["fine_dtau_T0"]
    dt_f = np.geomspace(d0, d1, polish["fine_steps"]) * T0 if polish["fine_steps"] else []
    psi, phi = _polish(psi, grid, mass_msun, a, G, dt_f, workers, rho_b)
    info = {"profile": prof, "scales": sc, "residual_direct": res_direct,
            "residual_final": hamiltonian_residual(psi, phi, grid, a, workers)}
    return psi, phi, info
