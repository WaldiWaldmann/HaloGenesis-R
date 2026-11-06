# P5_N48_tau3_n0.27_t10_soliton_halo – Gematchter Lauf A=1/A=0 und Observablen O1–O3

**Gesamtergebnis:** 4 Test(s) NICHT bestanden

N = 48, t_end = 10.0 Myr, τ = 3.0 Myr, n = 0.27 cm⁻³. Aperturen für Tests vorab deklariert (`analysis.apertures_kpc`). Auflösungs- und Ablationsrobustheit (Falsifikationsmatrix) folgt in P6.

## Lauf

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| Status: regulär beendet | 10 | — | info | t_final [Myr] |
| N / τ [Myr] / n [cm⁻³] | 48 | — | info | τ = 3.0, n = 0.27 |
| Makroschritte | 41 | — | info |  |
| ausgelassene Visualisierungsradien (< dx) | 2 | — | info | 0.1, 0.15 |
| Rechenzeit [s] | 13.48 | — | info |  |
| Grundzustands-Residuum | 0.001793 | — | info |  |
| Setup: soliton_halo | 1.205e+09 | — | info | M_Axion gesamt [M_sun]; Vorrelaxation 20.0 Myr, σ(Φ_c)/|Φ_c| = 0.00819, t_dB = 20.8 Myr |
| Potentialfluktuation σ(Φ_c)/|⟨Φ_c⟩| (A=1) | 0.01906 | — | info | isolierter Grundzustand ≈ 0 -> Filter wirkungslos; Halo > 0 |
| max|Φ̄_c − Φ_c|/|⟨Φ_c⟩| (A=1) | 0.01839 | — | info |  |
| max Jeans-Boden-Massenanteil in R_p (A=1 / A=0) | 0 | — | info | A=0: 0; N_J = 0.0 |
| max n_H (A=1) [cm⁻³] | 0.474 | — | info | GP98-Niedrigdichte-Limes: n ≲ 10³ cm⁻³ |
| lokale Rückfälle 1. Ordnung (Hydro) | 0 | — | info |  |
| ε_filter A=1 (Gl. 45), E_ref = E_int(0) | 0.001974 | — | info |  |
| vom Jeans-Boden injizierte Energie / E_int(0) (A=1) | 0 | — | info |  |
| max Truelove-Zahl J = dx/λ_J (A=1, A=0) | 0.3945 | 0.25 | ❌ | J > 1/4: künstliche Fragmentation möglich -> Ergebnis nicht auflösungsrobust |

## O1

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1(R=0.25) bei t_final | 0.2001 | — | info |  |
| max O1(R=0.25) | 0.2001 | — | info |  |
| Fluss-Gegenprobe R=0.25, A=1: max|ΔM − ∫F dt|/max|ΔM| | 0.0118 | 0.05 | ✅ | zweiter, unabhängiger Rechenweg (Gl. 22) |
| Fluss-Gegenprobe R=0.25, A=0: max|ΔM − ∫F dt|/max|ΔM| | 2.429 | 0.05 | ❌ | zweiter, unabhängiger Rechenweg (Gl. 22) |
| O1(R=0.5) bei t_final | 0.1974 | — | info |  |
| max O1(R=0.5) | 0.1974 | — | info |  |
| Fluss-Gegenprobe R=0.5, A=1: max|ΔM − ∫F dt|/max|ΔM| | 0.02529 | 0.05 | ✅ | zweiter, unabhängiger Rechenweg (Gl. 22) |
| Fluss-Gegenprobe R=0.5, A=0: max|ΔM − ∫F dt|/max|ΔM| | 1.258 | 0.05 | ❌ | zweiter, unabhängiger Rechenweg (Gl. 22) |
| O1(R=1.0) bei t_final | 0.1355 | — | info |  |
| max O1(R=1.0) | 0.1355 | — | info |  |
| Fluss-Gegenprobe R=1.0, A=1: max|ΔM − ∫F dt|/max|ΔM| | 0.02007 | 0.05 | ✅ | zweiter, unabhängiger Rechenweg (Gl. 22) |
| Fluss-Gegenprobe R=1.0, A=0: max|ΔM − ∫F dt|/max|ΔM| | 0.8133 | 0.05 | ❌ | zweiter, unabhängiger Rechenweg (Gl. 22) |

## O2

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O2(A=1) Zeitanteil min_R Q < 1 | 1 | — | info | Intervalle: [0.00, 10.00] |
| min Q (A=1) | 0.01791 | — | info |  |
| O2(A=0) Zeitanteil min_R Q < 1 | 1 | — | info | Intervalle: [0.00, 10.00] |
| min Q (A=0) | 0.02195 | — | info |  |
| ΔO2 = O2(1) − O2(0) | 0 | — | info | Hypothese: > 0 |

## O3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O3(R=0.5) Mittel für t ≤ 2.0 Myr | 0.9999 | 0.9 | ✅ | Gl. 30: O3 → 1 früh |
| Impulsbudget früh: max ‖δv − ∫δg dt‖/‖δv‖ | 0.001334 | — | info | Rest = Druck/Advektion R(t) |
| O3(R=0.25) zeitgemittelt | 0.9952 | — | info |  |
| O3(R=0.5) zeitgemittelt | 0.9962 | — | info |  |
| O3(R=1.0) zeitgemittelt | 0.9978 | — | info |  |

## Vergleich

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| C_ρ Aperturmittel R=0.25 (A=1), max | 1.222 | — | info | Arbeit: ~50.0 |
| C_ρ Aperturmittel R=0.25 bei t≈1.20 Myr | 1.003 | — | info |  |
| C_ρ Einzelzelle Zentrum (A=1), max | 1.214 | — | info | NICHT gleichwertig zu aufgelöster Dichte (Research Plan §2) |
| Dauer min_R Q<1 (A=1) [Myr] | 10 | — | info | Arbeit: 5.8 Myr |
