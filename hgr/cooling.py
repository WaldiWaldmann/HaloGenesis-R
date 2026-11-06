"""Kühlung (Research Plan Gl. 9, Term Λ; Originalarbeit Abschn. 4.3.4).

Modelle:
  "none"                  adiabatisch
  "isothermal"            T wird nach jedem Schritt auf T0 gesetzt (unendlich schnelle Kühlung+Heizung)
  "piecewise_historical"  Λ(T) EXAKT wie im Code-Ausschnitt der Originalarbeit [erg cm³ s⁻¹]:
                            T > 1e4:        1e-22 √T
                            300 < T ≤ 1e4:  1e-26 exp(−400/T) T^3.5
                            T ≤ 300:        0
                          Kühlrate pro Volumen n²Λ, n = ρ/(μ m_H).
Achtung: Die historischen Koeffizienten liegen um viele Größenordnungen über realistischer
primordialer Kühlung (P3-Report quantifiziert t_cool). Wegen extremer Steifheit wird
implizit integriert: Rückwärts-Euler e_neu + dt·n²Λ(T(e_neu)) = e_alt, gelöst per
vektorisierter Bisektion im Intervall [e(T=1 K), e_alt] (f ist dort vorzeichenwechselnd,
also existiert eine Wurzel; Bisektion findet sie auch bei Sprüngen von Λ).
"""
from __future__ import annotations

import astropy.constants as const
import astropy.units as u
import numpy as np

from .hydro import pressure
from .units import M_H, RHO_UNIT, T_UNIT, V_UNIT

# n²Λ: (erg cm⁻³ s⁻¹) -> Code-Energiedichte pro Zeit [M_sun kpc⁻¹ Myr⁻³]
_RATE_TO_CODE = (u.erg / u.cm**3 / u.s).to(u.Msun / (u.kpc * T_UNIT**3))
_RHO_TO_GRAMCM3 = (1.0 * RHO_UNIT).to_value(u.g / u.cm**3)


def temperature(U: np.ndarray, gamma: float, mu: float) -> np.ndarray:
    """T = μ m_H P/(ρ k_B)."""
    fac = (mu * M_H * V_UNIT**2 / const.k_B).to_value(u.K)
    return fac * pressure(U, gamma) / U[0]


def pressure_from_T(rho: np.ndarray, T: np.ndarray | float, mu: float) -> np.ndarray:
    fac = (mu * M_H * V_UNIT**2 / const.k_B).to_value(u.K)
    return rho * np.asarray(T) / fac


def number_density_cm3(rho: np.ndarray, mu: float) -> np.ndarray:
    return rho * _RHO_TO_GRAMCM3 / (mu * M_H.to_value(u.g))


def lambda_historical(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    hi = 1e-22 * np.sqrt(T)
    mid = 1e-26 * np.exp(-400.0 / np.maximum(T, 1e-30)) * T**3.5
    return np.where(T > 1e4, hi, np.where(T > 300.0, mid, 0.0))


def cooling_time_myr(rho: np.ndarray, T: np.ndarray, mu: float, gamma: float) -> np.ndarray:
    """t_cool = e_int/(n²Λ) mit e_int = P/(γ−1); ∞ wo Λ = 0."""
    n = number_density_cm3(rho, mu)
    rate = n**2 * lambda_historical(T) * _RATE_TO_CODE
    e = pressure_from_T(rho, T, mu) / (gamma - 1.0)
    with np.errstate(divide="ignore"):
        return np.where(rate > 0, e / rate, np.inf)


# ------------------------------------------------- B6: primordiale Kühlung --
# H-Atomkühlung im Stoßionisations-Gleichgewicht (CIE), Raten nach Cen (1992) / Katz, Weinberg &
# Hernquist (1996, Tab. 1–2), plus H₂-Kühlung im Niedrigdichte-Limes nach Galli & Palla (1998).
# Vereinfachungen (deklariert): reiner Wasserstoff für Ionisation, feste H₂-Fraktion x_H2, μ konstant,
# keine Compton-Kühlung, Λ = 0 unterhalb T_floor (CMB-Temperatur). GP98-Fit gültig 10 K < T < 10⁴ K
# und n deutlich unter der kritischen Dichte (~10³–10⁴ cm⁻³) – die Läufe berichten max n.
# HINWEIS: Koeffizienten wurden aus den zitierten Arbeiten übernommen; vor Veröffentlichung gegen die
# Originaltabellen prüfen (Checkliste im README).

def _cen_gauss(T):
    return 1.0 / (1.0 + np.sqrt(T / 1e5))


def h_cie_ionized_fraction(T: np.ndarray) -> np.ndarray:
    """x = Γ_coll/(Γ_coll + α_rec) für reinen Wasserstoff im CIE."""
    T = np.maximum(np.asarray(T, dtype=np.float64), 1.0)
    gam = 5.85e-11 * np.sqrt(T) * np.exp(-157809.1 / T) * _cen_gauss(T)
    alpha = 8.40e-11 / np.sqrt(T) * (T / 1e3) ** -0.2 / (1.0 + (T / 1e6) ** 0.7)
    return gam / (gam + alpha)


def lambda_h2_gp98_low_density(T: np.ndarray) -> np.ndarray:
    """Λ_H2(n→0) [erg cm³ s⁻¹] pro H₂-Molekül und H-Atom (Galli & Palla 1998, Fit 10–10⁴ K)."""
    lt = np.log10(np.clip(np.asarray(T, dtype=np.float64), 10.0, 1e4))
    return 10.0 ** (-103.0 + 97.59 * lt - 48.05 * lt**2 + 10.80 * lt**3 - 0.9032 * lt**4)


def primordial_rate_cgs(n_H: np.ndarray, T: np.ndarray, x_H2: float, T_floor: float) -> np.ndarray:
    """Kühlrate pro Volumen [erg cm⁻³ s⁻¹]. n_H = Wasserstoff-Kerndichte."""
    T = np.asarray(T, dtype=np.float64)
    Tc = np.maximum(T, 1.0)
    x = h_cie_ionized_fraction(Tc)
    ne, nH0, nHp = x * n_H, (1 - x) * n_H, x * n_H
    g = _cen_gauss(Tc)
    exc = 7.50e-19 * np.exp(-118348.0 / Tc) * g * ne * nH0                 # Linienanregung
    ion = 1.27e-21 * np.sqrt(Tc) * np.exp(-157809.1 / Tc) * g * ne * nH0   # Stoßionisation
    rec = 8.70e-27 * np.sqrt(Tc) * (Tc / 1e3) ** -0.2 / (1 + (Tc / 1e6) ** 0.7) * ne * nHp
    ff = 1.42e-27 * 1.3 * np.sqrt(Tc) * ne * nHp                            # Bremsstrahlung (g_ff=1.3)
    h2 = np.where(Tc < 1e4, x_H2 * n_H * n_H * lambda_h2_gp98_low_density(Tc), 0.0)
    return np.where(T > T_floor, exc + ion + rec + ff + h2, 0.0)


def _rate_code(model: str, rho: np.ndarray, T: np.ndarray, mu: float, cparams: dict | None) -> np.ndarray:
    if model == "piecewise_historical":
        return number_density_cm3(rho, mu) ** 2 * lambda_historical(T) * _RATE_TO_CODE
    if model == "primordial_h2":
        if cparams is None:
            raise ValueError("primordial_h2 braucht cooling_params")
        n_H = cparams["X_H"] * rho * _RHO_TO_GRAMCM3 / M_H.to_value(u.g)
        return primordial_rate_cgs(n_H, T, cparams["x_H2"], cparams["T_floor_K"]) * _RATE_TO_CODE
    raise ValueError(f"unbekanntes Kühlmodell {model}")


def cooling_time_model_myr(model: str, rho: np.ndarray, T: np.ndarray, mu: float, gamma: float,
                           cparams: dict | None) -> np.ndarray:
    rate = _rate_code(model, np.asarray(rho), np.asarray(T), mu, cparams)
    e = pressure_from_T(np.asarray(rho), T, mu) / (gamma - 1.0)
    with np.errstate(divide="ignore"):
        return np.where(rate > 0, e / rate, np.inf)


def apply_cooling(U: np.ndarray, dt: float, model: str, gamma: float, mu: float, T0: float | None,
                  *, cparams: dict | None = None) -> float:
    """In-place. Gibt abgestrahlte Energie (Code-Einheiten, pro Volumen summiert) zurück.
    Numba-Pfad (parallel, Frühabbruch, Überspringen nicht kühlender Zellen); Referenz: apply_cooling_reference."""
    if model not in ("piecewise_historical", "primordial_h2"):
        return apply_cooling_reference(U, dt, model, gamma, mu, T0, cparams=cparams)
    from .kernels import cooling_backward_euler
    if model == "primordial_h2" and cparams is None:
        raise ValueError("primordial_h2 braucht cooling_params")
    ekin = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    e_old = U[4] - ekin
    tfac = (mu * M_H * V_UNIT**2 / const.k_B).to_value(u.K) * (gamma - 1.0)
    mHg = M_H.to_value(u.g)
    cp = cparams or {"X_H": 0.0, "x_H2": 0.0, "T_floor_K": 0.0}
    e_new = cooling_backward_euler(U[0].ravel(), e_old.ravel(), dt, 0 if model == "piecewise_historical" else 1,
                                   tfac, 1.0 / tfac, _RHO_TO_GRAMCM3 / (mu * mHg), cp["X_H"] * _RHO_TO_GRAMCM3 / mHg,
                                   cp["x_H2"], cp["T_floor_K"], _RATE_TO_CODE).reshape(e_old.shape)
    U[4] = ekin + e_new
    return float(np.sum(e_old - e_new))


def apply_cooling_reference(U: np.ndarray, dt: float, model: str, gamma: float, mu: float, T0: float | None,
                            *, cparams: dict | None = None) -> float:
    """Ursprünglicher NumPy-Pfad (Referenz für Äquivalenztests)."""
    if model == "none":
        return 0.0
    ekin = 0.5 * (U[1] ** 2 + U[2] ** 2 + U[3] ** 2) / U[0]
    e_old = U[4] - ekin
    if model == "isothermal":
        if T0 is None:
            raise ValueError("isothermal braucht T0")
        e_new = pressure_from_T(U[0], T0, mu) / (gamma - 1.0)
    elif model in ("piecewise_historical", "primordial_h2"):
        lo = pressure_from_T(U[0], 1.0, mu) / (gamma - 1.0)
        lo = np.minimum(lo, e_old)
        hi = e_old.copy()
        tfac = (mu * M_H * V_UNIT**2 / const.k_B).to_value(u.K) * (gamma - 1.0)
        for _ in range(64):  # Bisektion: Intervall schrumpft um 2^-64
            mid = 0.5 * (lo + hi)
            f = mid - e_old + dt * _rate_code(model, U[0], tfac * mid / U[0], mu, cparams)
            pos = f > 0
            hi = np.where(pos, mid, hi)
            lo = np.where(pos, lo, mid)
        e_new = hi
    else:
        raise ValueError(f"unbekanntes Kühlmodell {model}")
    U[4] = ekin + e_new
    return float(np.sum(e_old - e_new))
