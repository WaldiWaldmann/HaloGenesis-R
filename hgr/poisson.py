"""Spektraler Poisson-Löser auf periodischem Gitter (Research Plan Gl. 3, 10, 20).

    ∇²Φ = 4πG (ρ − ⟨ρ⟩),   Φ̂(k) = −4πG δρ̂(k)/|k|²  (k ≠ 0),   Φ̂(0) := 0  ⇒ ⟨Φ⟩ = 0.

Die Subtraktion des Mittelwerts ist für die Lösbarkeit des periodischen Problems
zwingend (Kompatibilitätsbedingung). Das Ergebnis ist NICHT das Potential eines
isolierten Systems: Nahe des Ursprungs gilt für eine kompakte Masse M in der Box

    Φ_per(r) = Φ_iso(r) − (2π/3) G ρ̄ r² + C + O(r⁴/L⁵),   ρ̄ = M/L³,

weil die abgezogene Hintergrunddichte −ρ̄ ein harmonisches Zusatzpotential
erzeugt; der l=2-Anteil der periodischen Bilder verschwindet aus kubischer
Symmetrie. Dieser Ausdruck wird in P1 numerisch geprüft.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft

from .grid import Grid3D


def solve_poisson_periodic(rho: np.ndarray, grid: Grid3D, G: float, workers: int) -> np.ndarray:
    """Φ aus ρ [M_sun/kpc³]; Ergebnis in kpc²/Myr² mit ⟨Φ⟩ = 0."""
    return solve_poisson_periodic_hat(rho, grid, G, workers)[0]


def solve_poisson_periodic_hat(rho: np.ndarray, grid: Grid3D, G: float, workers: int):
    """Wie solve_poisson_periodic, gibt zusätzlich Φ̂ (rfftn-Spektrum) zurück -> spart Vorwärts-FFTs."""
    if rho.shape != grid.shape:
        raise ValueError(f"rho.shape {rho.shape} != grid.shape {grid.shape}")
    if rho.dtype != np.float64:
        raise TypeError(f"rho muss float64 sein, ist {rho.dtype}")
    if not np.all(np.isfinite(rho)):
        raise FloatingPointError("rho enthält NaN/Inf")
    delta = rho - rho.mean()
    delta_hat = sfft.rfftn(delta, workers=workers)
    k2 = grid.k2_r
    phi_hat = np.zeros_like(delta_hat)
    nz = k2 > 0.0
    phi_hat[nz] = -4.0 * np.pi * G * delta_hat[nz] / k2[nz]
    # Nullmode: Konvention ⟨Φ⟩ = 0 (phi_hat[0,0,0] bleibt 0)
    return sfft.irfftn(phi_hat, s=grid.shape, workers=workers), phi_hat


def gradient_spectral(phi: np.ndarray, grid: Grid3D, workers: int) -> tuple[np.ndarray, ...]:
    """∇Φ spektral. Nyquist-Moden werden für die ungerade Ableitung genullt
    (sonst entsteht ein nicht-reeller, mehrdeutiger Anteil)."""
    return gradient_from_hat(sfft.rfftn(phi, workers=workers), grid, workers)


def gradient_from_hat(phi_hat: np.ndarray, grid: Grid3D, workers: int) -> tuple[np.ndarray, ...]:
    """∇Φ aus bereits bekanntem rfftn-Spektrum (keine Vorwärts-FFT)."""
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
    """7-Punkt-Laplace zweiter Ordnung, periodisch."""
    out = -6.0 * phi
    for ax in range(3):
        out = out + np.roll(phi, 1, axis=ax) + np.roll(phi, -1, axis=ax)
    return out / grid.dx**2


def poisson_residual(phi: np.ndarray, rho: np.ndarray, grid: Grid3D, G: float,
                     laplacian: str, workers: int) -> float:
    """Normiertes Residuum ε_P = ||∇²_h Φ − 4πG δρ||₂ / ||4πG δρ||₂  (Research Plan Gl. 20).

    Der Wert hängt von der Wahl des diskreten Laplace-Operators ab und MUSS
    zusammen mit dieser Wahl berichtet werden."""
    src = 4.0 * np.pi * G * (rho - rho.mean())
    norm = np.linalg.norm(src)
    if norm == 0.0:
        raise ZeroDivisionError("δρ ≡ 0: Residuum undefiniert")
    if laplacian == "spectral":
        lap = laplacian_spectral(phi, grid, workers)
    elif laplacian == "fd2":
        lap = laplacian_fd2(phi, grid)
    else:
        raise ValueError(f"unbekannter Laplace-Operator: {laplacian}")
    return float(np.linalg.norm(lap - src) / norm)
