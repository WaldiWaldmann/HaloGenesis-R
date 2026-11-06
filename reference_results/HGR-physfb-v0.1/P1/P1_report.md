# P1 – Verifikation Einheiten/Gitter/Poisson (HGR-physfb-v0.1)

**Gesamtergebnis:** BESTANDEN

Schwellen wurden vor dem Lauf in der Config festgelegt (`validation_p1`).

## T0

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| G_code [kpc³/(Msun Myr²)] | 4.499e-12 | 1e-12 | ✅ | SI-Gegenrechnung 4.498502e-12 |
| hbar/m (m22=1) [kpc²/Myr] | 0.01961 | 1e-10 | ✅ | SI-Gegenrechnung 1.960695e-02 |
| g*(r_c) [m/s²] | 1.628e-10 | 0.005 | ✅ | Research Plan: 1.63e-10 |
| v*(r_c) [km/s] | 50.12 | 0.005 | ✅ | Research Plan: 50.1 |
| t*(r_c) [Myr] | 9.755 | 0.005 | ✅ | Research Plan: 9.75 |
| c_s(10³K) [km/s] | 3.385 | 0.005 | ✅ | Research Plan: 3.38 |

## T1

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| max|ΔΦ|/max|Φ| | 4.322e-15 | 1e-12 | ✅ |  |
| max|Δ∂xΦ|/max|∂xΦ| | 4.297e-14 | 1e-12 | ✅ |  |
| ε_P spektral | 8.25e-14 | 1e-12 | info | nur informativ |

## T2

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| rel. Fehler MIT Hintergrundkorrektur (r<L/4) | 0.0005104 | 0.001 | ✅ |  |
| rel. Fehler OHNE Korrektur (r<L/4) | 0.003687 | — | info | zeigt: periodische Box ≠ isoliertes System |
| ε_P spektral | 1.891e-13 | — | info |  |
| ε_P FD2 | 0.004365 | — | info | gleiche Lösung, anderer Laplace -> anderer Wert |

## T3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| L=4.0: median rel. Kraftfehler korrigiert (2dx≤r≤L/4) | 0.003453 | 0.01 | ✅ |  |
| L=4.0: max rel. Kraftfehler korrigiert | 0.02578 | — | info |  |
| L=4.0: rel. Abweichung vom ISOLIERTEN Soliton bei r_c | 0.1123 | — | info | physikalischer Boxeffekt |
| L=4.0: Masse im Gitter / M_s | 0.9634 | — | info | Abschneiden/Überlapp |
| L=8.0: median rel. Kraftfehler korrigiert (2dx≤r≤L/4) | 0.001263 | 0.01 | ✅ |  |
| L=8.0: max rel. Kraftfehler korrigiert | 0.01358 | — | info |  |
| L=8.0: rel. Abweichung vom ISOLIERTEN Soliton bei r_c | 0.01465 | — | info | physikalischer Boxeffekt |
| L=8.0: Masse im Gitter / M_s | 0.9998 | — | info | Abschneiden/Überlapp |
| L=16.0: median rel. Kraftfehler korrigiert (2dx≤r≤L/4) | 0.000972 | 0.01 | ✅ |  |
| L=16.0: max rel. Kraftfehler korrigiert | 0.01246 | — | info |  |
| L=16.0: rel. Abweichung vom ISOLIERTEN Soliton bei r_c | 0.00183 | — | info | physikalischer Boxeffekt |
| L=16.0: Masse im Gitter / M_s | 1 | — | info | Abschneiden/Überlapp |

## T4

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| σ=0.3: rel. Fehler N=16 vs N_ref=256 | 0.03189 | — | info |  |
| σ=0.3: ε_P FD2 N=16 | 0.2441 | — | info |  |
| σ=0.3: rel. Fehler N=32 vs N_ref=256 | 7.625e-06 | — | info |  |
| σ=0.3: ε_P FD2 N=32 | 0.06601 | — | info |  |
| σ=0.3: rel. Fehler N=64 vs N_ref=256 | 4.181e-16 | — | info |  |
| σ=0.3: ε_P FD2 N=64 | 0.01726 | — | info |  |
| σ=0.3: rel. Fehler N=128 vs N_ref=256 | 4.181e-16 | — | info |  |
| σ=0.3: ε_P FD2 N=128 | 0.004365 | — | info |  |
| beobachtete Ordnung ε_P FD2 (letzte Verfeinerung) | 1.983 | 2 | ✅ | erwartet 2 für glatte Quelle |

## T5

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| konstante Dichte addiert | 2.936e-16 | 1e-12 | ✅ |  |
| Linearität | 3.522e-16 | 1e-12 | ✅ |  |
| Translation (ganzzahlig) | 4.403e-16 | 1e-12 | ✅ |  |
| |⟨Φ⟩| | 0 | 1e-12 | ✅ |  |
| Symmetrie x↔y | 2.936e-16 | 1e-12 | ✅ |  |

## T6

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| Sekunden/Poisson N=32 (1 Threads) | 0.0006474 | — | info |  |
| Sekunden/Poisson N=64 (1 Threads) | 0.005175 | — | info |  |
| Sekunden/Poisson N=96 (1 Threads) | 0.02139 | — | info |  |
| Sekunden/Poisson N=128 (1 Threads) | 0.07136 | — | info |  |
| Sekunden/Poisson N=192 (1 Threads) | 0.3019 | — | info |  |
| Sekunden/Poisson N=256 (1 Threads) | 0.7912 | — | info |  |
