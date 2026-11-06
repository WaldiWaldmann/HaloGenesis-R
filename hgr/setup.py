"""Gemeinsame Initialisierung für alle gekoppelten Läufe (eine Quelle -> keine Konfundierung)."""
from __future__ import annotations

import numpy as np

from . import cooling as co
from . import hydro as hy
from .coupled import CouplingParams
from .grid import Grid3D
from .units import G_CODE, KMS_TO_CODE, AxionUnits, gas_mass_density


def polish_cfg(cfg: dict) -> dict:
    sp = cfg["sp"]
    return {"coarse_n": sp["polish_coarse_n"], "coarse_steps": sp["polish_coarse_steps"],
            "coarse_dtau_T0": sp["polish_coarse_dtau_T0"], "fine_steps": sp["polish_fine_steps"],
            "fine_dtau_T0": sp["polish_fine_dtau_T0"]}


def coupling_params(cfg: dict, tau: float) -> CouplingParams:
    gp, tp = cfg["gas"], cfg["time"]
    return CouplingParams(a=AxionUnits(cfg["axion"]["m22"]).hbar_over_m, G=G_CODE, gamma=gp["gamma"],
                          mu=gp["mu"], T0=gp["T_K"], cooling_model=gp["cooling_model"],
                          limiter=cfg["hydro"]["limiter"], cfl=tp["cfl_hydro"],
                          sp_kin_factor=tp["sp_kinetic_dt_factor"], sp_phase_max=tp["sp_potential_phase_max_rad"],
                          tau=tau, workers=cfg["compute"]["fft_workers"], grav_cfl=tp["grav_cfl"],
                          ff_fraction=tp["freefall_dt_fraction"], dt_max=cfg["p5"]["dt_max_myr"],
                          jeans_NJ=cfg["hydro"]["jeans_floor_NJ"], cooling_params=dict(gp["cooling_params"]))


def gas_initial_condition(cfg: dict, grid: Grid3D, n_cm3: float) -> np.ndarray:
    """Homogenes Gas + geseedetes weißes Rauschen auf ρ (Druck homogen), v = initial_velocity (isotrop 0)."""
    gp = cfg["gas"]
    if gp["initial_velocity_kms"] != 0.0:
        raise NotImplementedError("nur ruhendes Gas implementiert")
    rng = np.random.default_rng(cfg["meta"]["seed"])
    rho0 = gas_mass_density(n_cm3, gp["mu"])
    rho = rho0 * (1.0 + gp["perturbation_rel_amplitude"] * rng.standard_normal(grid.shape))
    p = co.pressure_from_T(np.full(grid.shape, rho0), gp["T_K"], gp["mu"])
    z = np.zeros(grid.shape)
    _ = KMS_TO_CODE
    return hy.cons_from_prim(rho, z, z, z, p, gp["gamma"])


# ------------------------------------------------------------ B4: Halo --
def random_phase_halo(grid: Grid3D, mass: float, sigma_v: float, r_core: float, a: float, seed: int,
                      workers: int) -> np.ndarray:
    """Granularer Halo als Zufallsphasen-Wellenfeld (vgl. Lin et al. 2018; Yavetz et al. 2022):
    ψ̂(k) ∝ exp(−a²k²/(4σ²))·(g₁ + i g₂)  ⇒  |ψ̂|² ∝ Maxwell-Verteilung f(v = a k);
    ψ = √(ρ_env) χ/⟨|χ|²⟩^½ mit ρ_env ∝ 1/(1+(r/r_h)²); Masse exakt normiert.
    Kein Gleichgewicht: wird vor der Kopplung per SP vorrelaxiert (setup.halo_relax_myr)."""
    import scipy.fft as sfft
    rng = np.random.default_rng(seed)
    amp = np.exp(-(a**2) * grid.k2_full / (4.0 * sigma_v**2))
    z = amp * (rng.standard_normal(grid.shape) + 1j * rng.standard_normal(grid.shape))
    chi = sfft.ifftn(z, workers=workers)
    chi /= np.sqrt(np.mean(np.abs(chi) ** 2))
    psi = np.sqrt(1.0 / (1.0 + (grid.r / r_core) ** 2)) * chi
    return psi * np.sqrt(mass / (np.sum(np.abs(psi) ** 2) * grid.cell_volume))


def build_axion_field(cfg: dict, grid: Grid3D, workers: int):
    """ψ(t=0) für Zweig A=1 gemäß cfg['setup']['kind']. Gibt (psi, info) zurück."""
    from .schroedinger import mass, potential, stable_dt, step_kdk
    from .soliton import embed_ground_state
    a = AxionUnits(cfg["axion"]["m22"]).hbar_over_m
    Ms = cfg["axion"]["soliton_mass_msun"]
    psi, _, ginfo = embed_ground_state(grid, Ms, a, G_CODE, polish_cfg(cfg), workers)
    info = {"residual": ginfo["residual_final"], "kind": cfg["setup"]["kind"]}
    S = cfg["setup"]
    if S["kind"] == "isolated":
        return psi, info
    if S["kind"] != "soliton_halo":
        raise ValueError(f"unbekanntes setup.kind {S['kind']}")
    halo = random_phase_halo(grid, S["halo_mass_ratio"] * Ms, S["halo_sigma_kms"] * KMS_TO_CODE,
                             S["halo_core_kpc"], a, S["halo_seed"], workers)
    psi = psi + halo
    phi = potential(psi, None, grid, G_CODE, workers)
    tp, t, c = cfg["time"], 0.0, grid.center_index
    phic = [float(phi[c, c, c])]
    while t < S["halo_relax_myr"] - 1e-12:
        dt = min(stable_dt(grid, phi, a, tp["sp_kinetic_dt_factor"], tp["sp_potential_phase_max_rad"]),
                 S["halo_relax_myr"] - t)
        psi, phi = step_kdk(psi, phi, None, grid, dt, a, G_CODE, workers)
        t += dt
        phic.append(float(phi[c, c, c]))
    info.update({"M_total": mass(psi, grid), "relax_myr": t,
                 "phi_c_rel_std_relax": float(np.std(phic) / abs(np.mean(phic))),
                 "t_dB_myr": a / (S["halo_sigma_kms"] * KMS_TO_CODE) ** 2})
    return psi, info
