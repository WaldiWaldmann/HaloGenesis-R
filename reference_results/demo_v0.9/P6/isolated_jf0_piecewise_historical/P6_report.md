# P6 – Auflösung, Ablation, Sensitivität, Falsifikationsmatrix

**Gesamtergebnis:** BESTANDEN

Regeln (vorab, Config `p6`): O1/ΔlnQ robust ⇔ Vorzeichen über alle N stabil UND |Wert(N_max)| > |Wert(N_max) − Wert(N_vorletzt)|; ein Zeitpunkt zählt nur als aufgelöst, wenn J ≤ 1/4 bis dahin im feinsten Lauf. O2 original gilt als gesättigt, wenn O2(A=0) ≥ Sättigungsschwelle. Matrix: `figs/p6_falsification_matrix.png`.

| Zeile | t=5 | t=10 | t=20 | t=30 |
|---|---|---|---|---|
| O1 (Konzentration) | robust | — | — | — |
| ΔlnQ (Jeans, kontinuierlich) | robust | — | — | — |
| O2 original | gesättigt | — | — | — |
| O3 (Kohärenz) | robust | — | — | — |
| Numerik: Truelove | robust | — | — | — |
| Numerik: Fluss-Bilanz | robust | — | — | — |

## Zeitverfeinerung (N=64, dt_max 0.25 → 0.125 Myr)

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 bei t=5: e_Zeit | 8.314e-07 | — | info | Werte 0.02586 / 0.02586 |
| ΔlnQ bei t=5: e_Zeit | 1.412e-07 | — | info | Werte -0.03874 / -0.03874 |

## Seeds (N=32, 2 Realisierungen)

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| σ_Seeds O1 bei t=5 | 1.208e-06 | — | info | E_sample (Gl. 34); Werte: 0.0305, 0.0305 |
| σ_Seeds ΔlnQ bei t=5 | 5.365e-07 | — | info | E_sample (Gl. 34); Werte: -0.0457, -0.0457 |

## Läufe

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| gefundene P5-Läufe | 6 | — | info | N32/τ10/n0.27; N32/τ3/n0.27; N32/τ3/n0.27; N48/τ3/n0.27; N64/τ3/n0.27; N64/τ3/n0.27 |
| Auflösungsreihe (τ, n = Basis) | 3 | — | info | N = 32, 48, 64 |

## Auflösung t=5 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[32, 48, 64] | 0.02586 | — | info | 0.0305, 0.0259, 0.0259; Fehlerschätzung 7.9e-05; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[32, 48, 64] | -0.03874 | — | info | -0.0457, -0.0389, -0.0387; Fehlerschätzung 0.00013; p_obs nan |
| max J bis t (N=64) | 0.2268 | 0.25 | ✅ |  |

## Auflösung t=10 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[32, 48, 64] | nan | — | info | nan, nan, nan; Fehlerschätzung nan; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[32, 48, 64] | nan | — | info | nan, nan, nan; Fehlerschätzung nan; p_obs nan |
| max J bis t (N=64) | 0.2283 | 0.25 | ✅ |  |

## Auflösung t=20 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[32, 48, 64] | nan | — | info | nan, nan, nan; Fehlerschätzung nan; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[32, 48, 64] | nan | — | info | nan, nan, nan; Fehlerschätzung nan; p_obs nan |
| max J bis t (N=64) | 0.2283 | 0.25 | ✅ |  |

## Auflösung t=30 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[32, 48, 64] | nan | — | info | nan, nan, nan; Fehlerschätzung nan; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[32, 48, 64] | nan | — | info | nan, nan, nan; Fehlerschätzung nan; p_obs nan |
| max J bis t (N=64) | 0.2283 | 0.25 | ✅ |  |

## O2 original

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| N=32: O2(A=0) / ΔO2 | 1 | 0.95 | info | ΔO2 = 0 |
| N=48: O2(A=0) / ΔO2 | 1 | 0.95 | info | ΔO2 = 0 |
| N=64: O2(A=0) / ΔO2 | 1 | 0.95 | info | ΔO2 = 0 |

## O3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| N=32: frühe Kohärenz | 1 | 0.9 | ✅ |  |
| N=32: Impulsbudget früh | 0.0007757 | 0.05 | ✅ |  |
| N=48: frühe Kohärenz | 1 | 0.9 | ✅ |  |
| N=48: Impulsbudget früh | 0.0006946 | 0.05 | ✅ |  |
| N=64: frühe Kohärenz | 1 | 0.9 | ✅ |  |
| N=64: Impulsbudget früh | 0.0007 | 0.05 | ✅ |  |

## Fluss-Bilanz

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| N=32, A=1: max|ΔM−∫F|/M(0) | 0.003189 | 0.02 | ✅ |  |
| N=32, A=0: max|ΔM−∫F|/M(0) | 3.332e-06 | 0.02 | ✅ |  |
| N=48, A=1: max|ΔM−∫F|/M(0) | 0.001019 | 0.02 | ✅ |  |
| N=48, A=0: max|ΔM−∫F|/M(0) | 6.599e-07 | 0.02 | ✅ |  |
| N=64, A=1: max|ΔM−∫F|/M(0) | 0.0005491 | 0.02 | ✅ |  |
| N=64, A=0: max|ΔM−∫F|/M(0) | 2.849e-06 | 0.02 | ✅ |  |

## Scan tau (N=32)

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 bei tau=3 | 0.03051 | — | info | t5:0.0305, t10:nan, t20:nan, t30:nan |
| O1 bei tau=10 | 0.03051 | — | info | t5:0.0305, t10:nan, t20:nan, t30:nan |
| dlnQ bei tau=3 | -0.04575 | — | info | t5:-0.0457, t10:nan, t20:nan, t30:nan |
| dlnQ bei tau=10 | -0.04574 | — | info | t5:-0.0457, t10:nan, t20:nan, t30:nan |
| O3 bei tau=3 | 1 | — | info | t5:1, t10:nan, t20:nan, t30:nan |
| O3 bei tau=10 | 1 | — | info | t5:1, t10:nan, t20:nan, t30:nan |

## Vergleich (feinster Basislauf)

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| C_ρ Aperturmittel R=0.25 (A=1), max | 1.043 | — | info | Arbeit: ~50.0 |
| C_ρ Aperturmittel R=0.25 bei t≈1.20 Myr | 1.002 | — | info |  |
| C_ρ Einzelzelle Zentrum (A=1), max | 1.045 | — | info | NICHT gleichwertig zu aufgelöster Dichte (Research Plan §2) |
| Dauer min_R Q<1 (A=1) [Myr] | 6 | — | info | Arbeit: 5.8 Myr |
