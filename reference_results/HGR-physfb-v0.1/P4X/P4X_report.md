# P4X – Modellerweiterungen B4–B7

**Gesamtergebnis:** BESTANDEN

Schwellen vorab in `validation_p4x`.

## X1

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| isolated: σ(Φ_c)/|⟨Φ_c⟩| über 30.0 Myr | 0.0001717 | — | info |  |
| isolated: max|Φ̄_τ=3 − Φ|/|⟨Φ⟩| im Zentrum | 4.614e-05 | — | info |  |
| soliton_halo: σ(Φ_c)/|⟨Φ_c⟩| über 30.0 Myr | 0.09573 | — | info |  |
| soliton_halo: max|Φ̄_τ=3 − Φ|/|⟨Φ⟩| im Zentrum | 0.02281 | — | info |  |
| Halo: Kohärenzzeit t_dB = a/σ² [Myr] | 20.83 | — | info |  |
| Halo: Gesamtmasse Axion [M_sun] | 1.205e+09 | — | info |  |
| Halo: Autokorrelationszeit von Φ_c (1/e) [Myr] | 6.858 | — | info | vergleiche t_dB |
| Fluktuationsfaktor Halo / isoliert | 557.4 | 10 | ✅ |  |

## X2

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| J vor Boden | 1159 | — | info |  |
| J nach Boden (≤ 1/N_J) | 0.25 | 0.25 | ✅ |  |
| gebodete Zellen [%] | 94.53 | — | info |  |
| zweite Anwendung: injizierte Energie (idempotent = 0) | 0 | 0 | ✅ |  |

## X3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| Λ unter T_floor | 0 | 0 | ✅ |  |
| Ordnung implizite Kühlung (n=0.1, T=2000 K) | 0.9957 | 1 | ✅ | T_ref = 1444 K nach 25.4 Myr |
| t_cool n=0.27, T=300 K: primordial [Myr] | 1551 | — | info | historisch: inf Myr |
| t_cool n=0.27, T=1000 K: primordial [Myr] | 101.4 | — | info | historisch: 1.15e-10 Myr |
| t_cool n=0.27, T=10000 K: primordial [Myr] | 57.86 | — | info | historisch: 2.53e-13 Myr |
| t_cool n=10, T=300 K: primordial [Myr] | 41.88 | — | info | historisch: inf Myr |
| t_cool n=10, T=1000 K: primordial [Myr] | 2.737 | — | info | historisch: 3.1e-12 Myr |
| t_cool n=10, T=10000 K: primordial [Myr] | 1.562 | — | info | historisch: 6.83e-15 Myr |

## X4

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| gleicher Seed -> bitgleich | 1 | 1 | ✅ |  |
| anderer Seed -> verschieden | 1 | 1 | ✅ |  |
