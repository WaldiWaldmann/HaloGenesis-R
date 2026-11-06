# P2 – Schrödinger-Poisson & Soliton-Grundzustand

**Gesamtergebnis:** BESTANDEN

Schwellen vorab in `validation_p2` festgelegt. Soliton: numerischer SP-Grundzustand (Shooting + imaginäre Zeit im periodischen Gitter).

## S0

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| U(0) (Shooting) | -0.6495 | — | info | dimensionslos |
| μ = 4π∫f²ξ²dξ | 25.91 | — | info | Literatur ≈ 25.9 (gleiche Skalierung) |
| ξ_c (f²=1/2) | 1.299 | — | info |  |
| r_c numerisch [kpc] | 0.7842 | — | info |  |
| r_c Schive-Fit [kpc] | 0.7539 | — | info | empirischer Fit |
| r_c numerisch / Schive | 1.04 | — | info | Fit-Genauigkeit ~ wenige % |
| ρ(0) numerisch [M_sun/kpc³] | 5.125e+07 | — | info |  |
| Soliton-Zeitskala T0 = L0²/a [Myr] | 18.58 | — | info |  |
| Residuum direkt interpoliert | 0.09675 | — | info | Boxeffekt + Interpolation |
| Residuum nach Politur | 0.001793 | — | info |  |
| Zeit Grundzustand [s] | 5.541 | — | info |  |

## S1

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| max|ρ(0,t)/ρ(0,0) − 1| über 40.0 Myr | 0.002114 | 0.005 | ✅ |  |
| max Normdrift | 7.105e-14 | 1e-12 | ✅ |  |
| max rel. Energiedrift | 2.201e-08 | 0.001 | ✅ |  |
| max Schwerpunktsverschiebung [kpc] | 3.437e-14 | — | info | dx = 0.125 kpc |
| Zeitschritt dt [Myr] | 0.1325 | — | info |  |
| Schritte | 302 | — | info |  |

## S2

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| max|ρ−ρ_exakt|/max ρ | 5.739e-07 | 1e-05 | ✅ | v = 15.06 km/s, σ(t)/σ0 = 2.40 |

## S3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| max |x_com − v t| [kpc] | 1.421e-14 | 0.01 | ✅ | v = 15.06 km/s |
| max |y_com|,|z_com| [kpc] | 1.599e-14 | 0.01 | ✅ |  |
| max |ρ_max(t)/ρ_max(0) − 1| | 0.003987 | 0.005 | ✅ | Galilei-Invarianz: Form bleibt erhalten |

## S4

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| beobachtete Ordnung (Richardson, 3 Lösungen) | 2 | 2 | ✅ |  |
| beobachtete Ordnung (gegen feinste Lösung) | 2.072 | — | info |  |
| ‖ψ−ψ_ref‖/‖ψ_ref‖ bei dt=0.5263 Myr | 1.505e-05 | — | info |  |
| ‖ψ−ψ_ref‖/‖ψ_ref‖ bei dt=0.2632 Myr | 3.581e-06 | — | info |  |
| ‖ψ−ψ_ref‖/‖ψ_ref‖ bei dt=0.1316 Myr | 7.137e-07 | — | info |  |
