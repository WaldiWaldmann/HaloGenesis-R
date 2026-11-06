from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .grid import Grid3D
from .poisson import solve_poisson_periodic


def kick(psi: np.ndarray, phi: np.ndarray, dt: float, a: float) -> np.ndarray:
    return psi * np.exp(-1j * phi * (dt / a))


_DRIFT_CACHE: dict = {}


def kick_inplace(psi: np.ndarray, phi: np.ndarray, dt: float, a: float) -> None:
    from .kernels import kick_inplace as _k

    _k(psi.reshape(-1), phi.reshape(-1), dt / a)


def drift(psi: np.ndarray, grid: Grid3D, dt: float, a: float, workers: int) -> np.ndarray:
    key = (grid.n, grid.box, complex(dt), a)
    fac = _DRIFT_CACHE.get(key)
    if fac is None:
        _DRIFT_CACHE.clear()
        fac = np.exp(-0.5j * a * dt * grid.k2_full)
        _DRIFT_CACHE[key] = fac
    psi_k = sfft.fftn(psi, workers=workers, overwrite_x=True)
    psi_k *= fac
    return sfft.ifftn(psi_k, workers=workers, overwrite_x=True)


def potential(psi: np.ndarray, rho_b: np.ndarray | None, grid: Grid3D, G: float, workers: int) -> np.ndarray:
    rho = np.abs(psi) ** 2 if rho_b is None else np.abs(psi) ** 2 + rho_b
    return solve_poisson_periodic(rho, grid, G, workers)


def step_kdk(
    psi: np.ndarray,
    phi: np.ndarray,
    rho_b: np.ndarray | None,
    grid: Grid3D,
    dt: float,
    a: float,
    G: float,
    workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    psi = psi.copy()
    kick_inplace(psi, phi, 0.5 * dt, a)
    psi = drift(psi, grid, dt, a, workers)
    phi = potential(psi, rho_b, grid, G, workers)
    kick_inplace(psi, phi, 0.5 * dt, a)
    return psi, phi


def stable_dt(grid: Grid3D, phi: np.ndarray, a: float, kin_factor: float, phase_max: float) -> float:
    dt_kin = kin_factor * grid.dx**2 / a
    pmax = float(np.max(np.abs(phi)))
    dt_pot = phase_max * a / pmax if pmax > 0 else np.inf
    return min(dt_kin, dt_pot)


def mass(psi: np.ndarray, grid: Grid3D) -> float:
    return float(np.sum(np.abs(psi) ** 2) * grid.cell_volume)


def energies(psi: np.ndarray, phi: np.ndarray, grid: Grid3D, a: float, workers: int) -> dict[str, float]:
    psi_k = sfft.fftn(psi, workers=workers)
    n3 = psi.size
    kin = 0.5 * a**2 * float(np.sum(grid.k2_full * np.abs(psi_k) ** 2)) * grid.cell_volume / n3
    pot = 0.5 * float(np.sum(phi * np.abs(psi) ** 2)) * grid.cell_volume
    return {"kin": kin, "pot": pot, "tot": kin + pot}


def center_of_mass(psi: np.ndarray, grid: Grid3D) -> np.ndarray:
    rho = np.abs(psi) ** 2
    out = np.empty(3)
    L = grid.box
    for ax in range(3):
        prof = rho.sum(axis=tuple(i for i in range(3) if i != ax))
        ang = 2 * np.pi * (grid.x1d + L / 2) / L
        z = np.sum(prof * np.exp(1j * ang))
        out[ax] = (np.angle(z) % (2 * np.pi)) * L / (2 * np.pi) - L / 2
    return out
