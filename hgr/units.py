"""Einheitensystem von HaloGenesis-R.

Code-Einheiten:  Länge = kpc,  Zeit = Myr,  Masse = M_sun.
Daraus:          Geschwindigkeit = kpc/Myr (~977.8 km/s), Dichte = M_sun/kpc^3.

Alle Umrechnungen laufen über astropy.units / astropy.constants (CODATA/IAU),
es gibt keine handkodierten Naturkonstanten. Myr = 1e6 julianische Jahre.
"""
from __future__ import annotations

from dataclasses import dataclass

import astropy.constants as const
import astropy.units as u
import numpy as np

L_UNIT = u.kpc
T_UNIT = u.Myr
M_UNIT = u.Msun
V_UNIT = L_UNIT / T_UNIT
RHO_UNIT = M_UNIT / L_UNIT**3

# Gravitationskonstante in Code-Einheiten [kpc^3 / (M_sun Myr^2)]
G_CODE: float = const.G.to_value(L_UNIT**3 / (M_UNIT * T_UNIT**2))
# 1 km/s in kpc/Myr
KMS_TO_CODE: float = (1.0 * u.km / u.s).to_value(V_UNIT)
# Masse des Wasserstoffatoms (1H, Atommasse 1.00782503207 u)
M_H = 1.00782503207 * const.u


def _check_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} muss endlich und > 0 sein, ist {value!r}")


@dataclass(frozen=True)
class AxionUnits:
    """Axion-abhängige Größen. m22 = m_a / (1e-22 eV)."""

    m22: float

    def __post_init__(self) -> None:
        _check_positive("m22", self.m22)

    @property
    def mass(self) -> u.Quantity:
        return (self.m22 * 1e-22 * u.eV / const.c**2).to(u.kg)

    @property
    def hbar_over_m(self) -> float:
        """hbar/m_a in kpc^2/Myr. Einziger Parameter der SP-Gleichung in Code-Einheiten."""
        return (const.hbar / self.mass).to_value(L_UNIT**2 / T_UNIT)

    @property
    def compton_time_myr(self) -> float:
        """hbar/(m_a c^2). Im nichtrelativistischen SP-Limes herausfaktorisiert!"""
        return (const.hbar / (self.mass * const.c**2)).to_value(T_UNIT)

    def de_broglie_time_myr(self, v_code: float) -> float:
        """Kohärenzzeit hbar/(m v^2) (Hui et al. 2017) für Geschwindigkeit v [kpc/Myr]."""
        _check_positive("v_code", v_code)
        return self.hbar_over_m / v_code**2

    def de_broglie_wavelength_kpc(self, v_code: float) -> float:
        """lambda_dB = 2 pi hbar/(m v)."""
        _check_positive("v_code", v_code)
        return 2.0 * np.pi * self.hbar_over_m / v_code


# ------------------------------------------------------------------ Gas -----
def gas_mass_density(n_cm3: float, mu: float) -> float:
    """rho_b = n mu m_H  in M_sun/kpc^3."""
    _check_positive("n_cm3", n_cm3)
    _check_positive("mu", mu)
    return (n_cm3 * u.cm**-3 * mu * M_H).to_value(RHO_UNIT)


def sound_speed(T_K: float, mu: float, gamma: float) -> float:
    """Adiabatische Schallgeschwindigkeit c_s = sqrt(gamma k_B T/(mu m_H)) in kpc/Myr."""
    _check_positive("T_K", T_K)
    _check_positive("mu", mu)
    _check_positive("gamma", gamma)
    cs = np.sqrt(gamma * const.k_B * T_K * u.K / (mu * M_H))
    return cs.to_value(V_UNIT)


# ---------------------------------------------------------- Jeans etc. -----
def jeans_length(cs: float, rho: np.ndarray | float) -> np.ndarray | float:
    """lambda_J = c_s sqrt(pi/(G rho))  (Research Plan Gl. 14)."""
    return cs * np.sqrt(np.pi / (G_CODE * np.asarray(rho)))


def jeans_mass(cs: np.ndarray | float, rho: np.ndarray | float) -> np.ndarray | float:
    """M_J = pi^{5/2}/6 * c_s^3 / (G^{3/2} rho^{1/2})  (Research Plan Gl. 15)."""
    return (np.pi**2.5 / 6.0) * np.asarray(cs) ** 3 / (G_CODE**1.5 * np.sqrt(np.asarray(rho)))


def free_fall_time(rho: np.ndarray | float) -> np.ndarray | float:
    """t_ff = sqrt(3 pi / (32 G rho))."""
    return np.sqrt(3.0 * np.pi / (32.0 * G_CODE * np.asarray(rho)))


def code_velocity_to_kms(v_code: float) -> float:
    return v_code / KMS_TO_CODE
