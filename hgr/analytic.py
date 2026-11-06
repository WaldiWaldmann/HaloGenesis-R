"""Analytische Referenzlösungen für Validierung und P0-Abschätzungen.

Achtung Evidenzklasse: Das Schive-Profil ist ein EMPIRISCHER Fit an SP-Simulationen
(Schive, Chiueh & Broadhurst 2014, Nature Phys. 10, 496). In P2 wird es durch den
numerisch berechneten SP-Grundzustand ersetzt; hier dient es nur für Abschätzungen
und als glatte Testquelle für den Poisson-Löser.
"""
from __future__ import annotations

import numpy as np
from scipy.special import beta, erf, hyp2f1

# ----------------------------------------------------------------- Gauß -----

def gaussian_density(r: np.ndarray, mass: float, sigma: float) -> np.ndarray:
    return mass / (2.0 * np.pi * sigma**2) ** 1.5 * np.exp(-0.5 * (r / sigma) ** 2)


def gaussian_potential_isolated(r: np.ndarray, mass: float, sigma: float, G: float) -> np.ndarray:
    """Φ_iso(r) = −GM erf(r/(√2σ))/r, mit stetigem Grenzwert bei r=0."""
    r = np.asarray(r, dtype=np.float64)
    out = np.empty_like(r)
    small = r < 1e-12 * sigma
    out[small] = -G * mass * np.sqrt(2.0 / np.pi) / sigma
    rs = r[~small]
    out[~small] = -G * mass * erf(rs / (np.sqrt(2.0) * sigma)) / rs
    return out


def periodic_background_potential(r: np.ndarray, rho_mean: float, G: float) -> np.ndarray:
    """Potential der abgezogenen Hintergrunddichte −ρ̄: −(2π/3) G ρ̄ r²  (s. poisson.py)."""
    return -(2.0 * np.pi / 3.0) * G * rho_mean * np.asarray(r) ** 2


def periodic_background_accel(r: np.ndarray, rho_mean: float, G: float) -> np.ndarray:
    """Radiale Zusatzbeschleunigung g_r = −∂_r Φ_bg = +(4π/3) G ρ̄ r (nach außen)."""
    return (4.0 * np.pi / 3.0) * G * rho_mean * np.asarray(r)


# -------------------------------------------------------- Schive-Profil -----
SCHIVE_A = 0.091            # Profil-Konstante (Schive et al. 2014)
SCHIVE_RHO_C_MSUN_PC3 = 1.9e-2  # ρ_c = 1.9e-2 m22^-2 (r_c/kpc)^-4  M_sun/pc³ (a=1)
PC3_PER_KPC3 = 1.0e9


def schive_central_density(m22: float, rc_kpc: float) -> float:
    """ρ_c in M_sun/kpc³."""
    return SCHIVE_RHO_C_MSUN_PC3 * PC3_PER_KPC3 / (m22**2 * rc_kpc**4)


def schive_density(r: np.ndarray, m22: float, rc_kpc: float) -> np.ndarray:
    rho_c = schive_central_density(m22, rc_kpc)
    return rho_c * (1.0 + SCHIVE_A * (np.asarray(r) / rc_kpc) ** 2) ** -8


def _schive_shape_integral(x: np.ndarray) -> np.ndarray:
    """F(x) = ∫_0^x s² (1 + a s²)^-8 ds = x³/3 · ₂F₁(8, 3/2; 5/2; −a x²)."""
    x = np.asarray(x, dtype=np.float64)
    return x**3 / 3.0 * hyp2f1(8.0, 1.5, 2.5, -SCHIVE_A * x**2)


SCHIVE_SHAPE_TOTAL = SCHIVE_A**-1.5 * beta(1.5, 6.5) / 2.0  # F(∞)


def schive_enclosed_mass(r: np.ndarray, m22: float, rc_kpc: float) -> np.ndarray:
    rho_c = schive_central_density(m22, rc_kpc)
    return 4.0 * np.pi * rho_c * rc_kpc**3 * _schive_shape_integral(np.asarray(r) / rc_kpc)


def schive_total_mass(m22: float, rc_kpc: float) -> float:
    return 4.0 * np.pi * schive_central_density(m22, rc_kpc) * rc_kpc**3 * SCHIVE_SHAPE_TOTAL


def schive_core_radius_from_mass(m22: float, mass_msun: float) -> float:
    """Aus M_tot ∝ 1/(m22² r_c) folgt r_c eindeutig."""
    if mass_msun <= 0:
        raise ValueError("Masse muss > 0 sein")
    const = 4.0 * np.pi * SCHIVE_RHO_C_MSUN_PC3 * PC3_PER_KPC3 * SCHIVE_SHAPE_TOTAL
    return const / (m22**2 * mass_msun)


def schive_m22_for(mass_msun: float, rc_kpc: float) -> float:
    """m22, bei dem M_tot und r_c gleichzeitig gelten."""
    const = 4.0 * np.pi * SCHIVE_RHO_C_MSUN_PC3 * PC3_PER_KPC3 * SCHIVE_SHAPE_TOTAL
    return float(np.sqrt(const / (mass_msun * rc_kpc)))
