from __future__ import annotations

import numpy as np

from . import cooling as co
from . import hydro as hy
from .coupled import CouplingParams
from .grid import Grid3D
from .units import G_CODE, KMS_TO_CODE, AxionUnits, gas_mass_density


def polish_cfg(cfg: dict) -> dict:
    sp = cfg["sp"]
    return {
        "coarse_n": sp["polish_coarse_n"],
        "coarse_steps": sp["polish_coarse_steps"],
        "coarse_dtau_T0": sp["polish_coarse_dtau_T0"],
        "fine_steps": sp["polish_fine_steps"],
        "fine_dtau_T0": sp["polish_fine_dtau_T0"],
    }


def coupling_params(cfg: dict, tau: float) -> CouplingParams:
    gp, tp = cfg["gas"], cfg["time"]
    return CouplingParams(
        a=AxionUnits(cfg["axion"]["m22"]).hbar_over_m,
        G=G_CODE,
        gamma=gp["gamma"],
        mu=gp["mu"],
        T0=gp["T_K"],
        cooling_model=gp["cooling_model"],
        limiter=cfg["hydro"]["limiter"],
        cfl=tp["cfl_hydro"],
        sp_kin_factor=tp["sp_kinetic_dt_factor"],
        sp_phase_max=tp["sp_potential_phase_max_rad"],
        tau=tau,
        workers=cfg["compute"]["fft_workers"],
        grav_cfl=tp["grav_cfl"],
        ff_fraction=tp["freefall_dt_fraction"],
        dt_max=cfg["p5"]["dt_max_myr"],
        jeans_NJ=cfg["hydro"]["jeans_floor_NJ"],
        cooling_params=dict(gp["cooling_params"]),
        filter_kind=filter_kind_from_cfg(cfg),
    )


def gas_initial_condition(cfg: dict, grid: Grid3D, n_cm3: float) -> np.ndarray:
    gp = cfg["gas"]
    if gp["initial_velocity_kms"] != 0.0:
        raise NotImplementedError("only gas at rest is implemented")
    rng = np.random.default_rng(cfg["meta"]["seed"])
    rho0 = gas_mass_density(n_cm3, gp["mu"])
    rho = rho0 * (1.0 + gp["perturbation_rel_amplitude"] * rng.standard_normal(grid.shape))
    p = co.pressure_from_T(np.full(grid.shape, rho0), gp["T_K"], gp["mu"])
    z = np.zeros(grid.shape)
    _ = KMS_TO_CODE
    return hy.cons_from_prim(rho, z, z, z, p, gp["gamma"])


def random_phase_halo(
    grid: Grid3D,
    mass: float,
    sigma_v: float,
    r_core: float,
    a: float,
    seed: int,
    workers: int,
    n_ref: int | None = None,
) -> np.ndarray:
    import scipy.fft as sfft

    rng = np.random.default_rng(seed)
    if n_ref is None:
        n_ref = grid.n
    if grid.n < n_ref:
        raise ValueError(f"N={grid.n} < halo_ref_n={n_ref}")
    gref = Grid3D(grid.box, n_ref)
    amp = np.exp(-(a**2) * gref.k2_full / (4.0 * sigma_v**2))
    zr = amp * (rng.standard_normal(gref.shape) + 1j * rng.standard_normal(gref.shape))
    h = n_ref // 2
    io = np.r_[0:h, n_ref - h + 1 : n_ref]
    inew = np.r_[0:h, grid.n - h + 1 : grid.n]
    z = np.zeros(grid.shape, dtype=np.complex128)
    z[np.ix_(inew, inew, inew)] = zr[np.ix_(io, io, io)]
    chi = sfft.ifftn(z, workers=workers)
    chi /= np.sqrt(np.mean(np.abs(chi) ** 2))
    psi = np.sqrt(1.0 / (1.0 + (grid.r / r_core) ** 2)) * chi
    return psi * np.sqrt(mass / (np.sum(np.abs(psi) ** 2) * grid.cell_volume))


def build_axion_field(cfg: dict, grid: Grid3D, workers: int):
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
        raise ValueError(f"unknown setup.kind {S['kind']}")
    halo = random_phase_halo(
        grid,
        S["halo_mass_ratio"] * Ms,
        S["halo_sigma_kms"] * KMS_TO_CODE,
        S["halo_core_kpc"],
        a,
        S["halo_seed"],
        workers,
        n_ref=S["halo_ref_n"],
    )
    psi = psi + halo
    phi = potential(psi, None, grid, G_CODE, workers)
    tp, t, c = cfg["time"], 0.0, grid.center_index
    phic = [float(phi[c, c, c])]
    while t < S["halo_relax_myr"] - 1e-12:
        dt = min(
            stable_dt(grid, phi, a, tp["sp_kinetic_dt_factor"], tp["sp_potential_phase_max_rad"]),
            S["halo_relax_myr"] - t,
        )
        psi, phi = step_kdk(psi, phi, None, grid, dt, a, G_CODE, workers)
        t += dt
        phic.append(float(phi[c, c, c]))
    info.update(
        {
            "M_total": mass(psi, grid),
            "relax_myr": t,
            "phi_c_rel_std_relax": float(np.std(phic) / abs(np.mean(phic))),
            "t_dB_myr": a / (S["halo_sigma_kms"] * KMS_TO_CODE) ** 2,
        }
    )
    return psi, info


def filter_kind_from_cfg(cfg: dict) -> str:
    k = cfg["filter"]["kernel"]
    from .filters import KERNELS

    if k not in KERNELS:
        raise ValueError(f"filter.kernel = {k!r}; allowed: {KERNELS}")
    return k


def fibonacci_probes(grid: Grid3D, radii_kpc: list, n_per_shell: int) -> tuple[np.ndarray, np.ndarray]:
    out, rad = [], []
    gold = np.pi * (3.0 - np.sqrt(5.0))
    c = grid.center_index
    for R in radii_kpc:
        for i in range(n_per_shell):
            zc = 1 - 2 * (i + 0.5) / n_per_shell
            rc = np.sqrt(1 - zc * zc)
            ph = gold * i
            v = np.array([rc * np.cos(ph), rc * np.sin(ph), zc]) * R
            idx = tuple(int(round(c + x / grid.dx)) % grid.n for x in v)
            out.append(idx)
            rad.append(R)
    arr = np.array(out)
    _, keep = np.unique(arr, axis=0, return_index=True)
    keep = np.sort(keep)
    return arr[keep], np.array(rad)[keep]
