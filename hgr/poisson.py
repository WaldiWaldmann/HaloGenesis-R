from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .grid import Grid3D


def solve_poisson_periodic(rho: np.ndarray, grid: Grid3D, G: float, workers: int) -> np.ndarray:
    return solve_poisson_periodic_hat(rho, grid, G, workers)[0]


def solve_poisson_periodic_hat(rho: np.ndarray, grid: Grid3D, G: float, workers: int):
    if rho.shape != grid.shape:
        raise ValueError(f"rho.shape {rho.shape} != grid.shape {grid.shape}")
    if rho.dtype != np.float64:
        raise TypeError(f"rho must be float64, got {rho.dtype}")
    if not np.all(np.isfinite(rho)):
        raise FloatingPointError("rho contains NaN/Inf")
    delta = rho - rho.mean()
    delta_hat = sfft.rfftn(delta, workers=workers)
    k2 = grid.k2_r
    phi_hat = np.zeros_like(delta_hat)
    nz = k2 > 0.0
    phi_hat[nz] = -4.0 * np.pi * G * delta_hat[nz] / k2[nz]
    return sfft.irfftn(phi_hat, s=grid.shape, workers=workers), phi_hat


def gradient_spectral(phi: np.ndarray, grid: Grid3D, workers: int) -> tuple[np.ndarray, ...]:
    return gradient_from_hat(sfft.rfftn(phi, workers=workers), grid, workers)


def gradient_from_hat(phi_hat: np.ndarray, grid: Grid3D, workers: int) -> tuple[np.ndarray, ...]:
    out = []
    n = grid.n
    for axis, k in enumerate(grid.k_r):
        kk = np.array(k, copy=True)
        if axis < 2:
            idx = [slice(None)] * 3
            idx[axis] = n // 2
            kk[tuple(idx)] = 0.0
        else:
            kk[..., -1] = 0.0
        out.append(sfft.irfftn(1j * kk * phi_hat, s=grid.shape, workers=workers))
    return tuple(out)


def laplacian_spectral(phi: np.ndarray, grid: Grid3D, workers: int) -> np.ndarray:
    phi_hat = sfft.rfftn(phi, workers=workers)
    return sfft.irfftn(-grid.k2_r * phi_hat, s=grid.shape, workers=workers)


def laplacian_fd2(phi: np.ndarray, grid: Grid3D) -> np.ndarray:
    out = -6.0 * phi
    for ax in range(3):
        out = out + np.roll(phi, 1, axis=ax) + np.roll(phi, -1, axis=ax)
    return out / grid.dx**2


def poisson_residual(phi: np.ndarray, rho: np.ndarray, grid: Grid3D, G: float, laplacian: str, workers: int) -> float:
    src = 4.0 * np.pi * G * (rho - rho.mean())
    norm = np.linalg.norm(src)
    if norm == 0.0:
        raise ZeroDivisionError("δρ ≡ 0: residual undefined")
    if laplacian == "spectral":
        lap = laplacian_spectral(phi, grid, workers)
    elif laplacian == "fd2":
        lap = laplacian_fd2(phi, grid)
    else:
        raise ValueError(f"unknown Laplacian: {laplacian}")
    return float(np.linalg.norm(lap - src) / norm)
