# P0 – Parameter-Manifest & Konsistenzprüfung (HGR-physfb-v0.1)

**Provenienz:** NEU GEWAEHLT (Rekonstruktion, Okt. 2026) - keine historischen Eingabewerte

Evidenzklasse aller Zahlen: *abgeleitet* aus der Config (keine Simulation). Soliton-Größen nutzen den empirischen Schive-Fit; P2 ersetzt ihn durch den numerischen SP-Grundzustand.

## Kennzahlen

| Größe | Wert | Einheit | Klasse | Abbildung |
|---|---|---|---|---|
| m22 | 1 | - | gewählt | — |
| hbar/m_a | 0.01961 | kpc²/Myr | abgeleitet | — |
| Soliton-Masse M_s | 2.92e+08 | M_sun | historisch übernommen | p0_profiles |
| Kernradius r_c(M_s, m22) [Schive-Fit] | 0.7539 | kpc | abgeleitet | p0_profiles |
| m22 für M_s UND r_c=0.5 kpc | 1.228 | - | abgeleitet | p0_timescales_m22 |
| zentrale Soliton-Dichte rho_c | 5.881e+07 | M_sun/kpc³ | abgeleitet | p0_profiles |
| M_s(<r_c) | 7.031e+07 | M_sun | abgeleitet | p0_profiles |
| v_circ(r_c) | 20.03 | km/s | abgeleitet | p0_timescales |
| Phi(0) isoliert | 1418 | (km/s)² | abgeleitet | p0_profiles |
| t* = sqrt(r_c³/GM) | 18.06 | Myr | abgeleitet | p0_timescales |
| t_ff(rho_c) | 33.37 | Myr | abgeleitet | p0_timescales |
| t_dB = hbar/(m v²) | 46.74 | Myr | abgeleitet | p0_timescales |
| lambda_dB | 6.015 | kpc | abgeleitet | p0_timescales |
| t_Compton = hbar/(m c²) | 2.086e-07 | Myr | abgeleitet | p0_timescales |
| rho_b (n, mu) | 8.012e+06 | M_sun/kpc³ | abgeleitet | p0_profiles |
| rho_b / rho_c(Soliton) | 0.1362 | - | abgeleitet | p0_profiles |
| c_s | 3.385 | km/s | abgeleitet | p0_timescales |
| lambda_J (Gas) | 1.022 | kpc | abgeleitet | p0_QJ |
| M_J (Gas, t=0) | 4.479e+06 | M_sun | abgeleitet | p0_QJ |
| t_ff (Gas) | 90.4 | Myr | abgeleitet | p0_timescales |
| t_sound = r_c/c_s | 217.8 | Myr | abgeleitet | p0_timescales |
| Gasmasse in Box | 4.102e+09 | M_sun | abgeleitet | — |
| periodische Kraftkorrektur bei r_c | 0.01456 | - | abgeleitet | P1 |
| Q_J(R=0.25 kpc, t=0) | 8.542 | - | abgeleitet | p0_QJ |
| epsilon(R=0.25 kpc) = M_gas/M_sol | 0.1429 | - | abgeleitet | p0_epsilon |
| Q_J(R=0.5 kpc, t=0) | 1.068 | - | abgeleitet | p0_QJ |
| epsilon(R=0.5 kpc) = M_gas/M_sol | 0.1641 | - | abgeleitet | p0_epsilon |
| Q_J(R=1.0 kpc, t=0) | 0.1335 | - | abgeleitet | p0_QJ |
| epsilon(R=1.0 kpc) = M_gas/M_sol | 0.2686 | - | abgeleitet | p0_epsilon |
| [N=64] dx | 0.125 | kpc | abgeleitet | p0_resolution |
| [N=64] Zellen pro r_c | 6.031 | - | abgeleitet | p0_resolution |
| [N=64] dt_SP | 0.1328 | Myr | abgeleitet | p0_cost |
| [N=64] dt_hydro (CFL, v_esc) | 0.6474 | Myr | abgeleitet | p0_cost |
| [N=64] SP-Schritte bis t_end | 225.9 | - | abgeleitet | p0_cost |
| [N=64] Speicher Kern (A=0+A=1) | 0.08598 | GB | abgeleitet | p0_cost |
| [N=64] Filterpuffer tau=3.0 Myr | 0.02097 | GB | abgeleitet | p0_cost |
| [N=128] dx | 0.0625 | kpc | abgeleitet | p0_resolution |
| [N=128] Zellen pro r_c | 12.06 | - | abgeleitet | p0_resolution |
| [N=128] dt_SP | 0.0332 | Myr | abgeleitet | p0_cost |
| [N=128] dt_hydro (CFL, v_esc) | 0.3237 | Myr | abgeleitet | p0_cost |
| [N=128] SP-Schritte bis t_end | 903.5 | - | abgeleitet | p0_cost |
| [N=128] Speicher Kern (A=0+A=1) | 0.6879 | GB | abgeleitet | p0_cost |
| [N=128] Filterpuffer tau=3.0 Myr | 0.3355 | GB | abgeleitet | p0_cost |
| [N=256] dx | 0.03125 | kpc | abgeleitet | p0_resolution |
| [N=256] Zellen pro r_c | 24.13 | - | abgeleitet | p0_resolution |
| [N=256] dt_SP | 0.008301 | Myr | abgeleitet | p0_cost |
| [N=256] dt_hydro (CFL, v_esc) | 0.1618 | Myr | abgeleitet | p0_cost |
| [N=256] SP-Schritte bis t_end | 3614 | - | abgeleitet | p0_cost |
| [N=256] Speicher Kern (A=0+A=1) | 5.503 | GB | abgeleitet | p0_cost |
| [N=256] Filterpuffer tau=3.0 Myr | 5.1 | GB | abgeleitet | p0_cost |

## Abgleich mit der Originalarbeit

| Größe | berichtet | rekonstruiert | Konsequenz |
|---|---|---|---|
| Schallgeschwindigkeit | 2.62 km/s | 3.385 km/s | Arbeit Gl. (11) rechnet falsch; Research Plan korrigiert auf 3.38 km/s. |
| Freifallzeit Soliton | 0.3 Myr | 33.37 Myr | Faktor ~111: Gas kann sich in t_sim=1.2 Myr nicht 50x verdichten. |
| Simulationsdauer vs. Jeans-Dauer | t_sim=1.2 Myr, T_J=5.8 Myr | T_J > t_sim | Logisch unmöglich in einem Lauf; Research Plan nimmt 20 Myr an. |
| Mittelungsfenster | 1e+03 yr | dt_SP(N=128) = 33205 yr | Fenster < Zeitschritt ⇒ Filter war numerisch wirkungslos (≡ instantan). |
| Relevante Fluktuationszeit | tau_Phi = 0.10 yr (Arbeit Gl. 9) | t_dB = 46.7 Myr | Compton-Oszillation ist im SP-Limes eliminiert. |
| Massenverhältnis epsilon | 1e-03–1e-02 | 0.164 (R=0.5 kpc) | Gas-Eigengravitation ist nicht störungstheoretisch. |
| min M_J/M_gas | 5.8e-03 | 1.07e+00 schon bei t=0 | Q<1 ist ohne Soliton bereits erfüllt ⇒ Diagnostik muss differenziell sein (ΔO2). |
| Kernradius bei M_s, m22=1 | 0.5 kpc | 0.754 kpc | Konsistent nur für m22 ≈ 1.23. |

## Abbildungen

2D: `figs/`, 3D statisch: `figs3d/*.png`, 3D interaktiv: `figs3d/*.html` (im Browser öffnen, offline lauffähig).