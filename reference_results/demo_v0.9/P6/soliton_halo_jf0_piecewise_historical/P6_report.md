# P6 – Auflösung, Ablation, Sensitivität, Falsifikationsmatrix

**Gesamtergebnis:** 5 Test(s) NICHT bestanden

Regeln (vorab, Config `p6`): O1/ΔlnQ robust ⇔ Vorzeichen über alle N stabil UND |Wert(N_max)| > |Wert(N_max) − Wert(N_vorletzt)|; ein Zeitpunkt zählt nur als aufgelöst, wenn J ≤ 1/4 bis dahin im feinsten Lauf. O2 original gilt als gesättigt, wenn O2(A=0) ≥ Sättigungsschwelle. Matrix: `figs/p6_falsification_matrix.png`.

| Zeile | t=5 | t=10 | t=20 | t=30 |
|---|---|---|---|---|
| O1 (Konzentration) | — | — | — | — |
| ΔlnQ (Jeans, kontinuierlich) | — | — | — | — |
| O2 original | gesättigt | gesättigt | — | — |
| O3 (Kohärenz) | robust | robust | — | — |
| Numerik: Truelove | verworfen | verworfen | — | — |
| Numerik: Fluss-Bilanz | verworfen | verworfen | — | — |

## Läufe

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| gefundene P5-Läufe | 1 | — | info | N48/τ3/n0.27 |
| Auflösungsreihe (τ, n = Basis) | 1 | — | info | N = 48 |

## Auflösung t=5 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[48] | 0.0754 | — | info | 0.0754; Fehlerschätzung nan; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[48] | -0.1131 | — | info | -0.113; Fehlerschätzung nan; p_obs nan |
| max J bis t (N=48) | 0.3184 | 0.25 | ❌ |  |

## Auflösung t=10 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[48] | 0.2916 | — | info | 0.292; Fehlerschätzung nan; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[48] | -0.4374 | — | info | -0.437; Fehlerschätzung nan; p_obs nan |
| max J bis t (N=48) | 0.3945 | 0.25 | ❌ |  |

## Auflösung t=20 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[48] | nan | — | info | nan; Fehlerschätzung nan; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[48] | nan | — | info | nan; Fehlerschätzung nan; p_obs nan |
| max J bis t (N=48) | 0.3945 | 0.25 | ❌ |  |

## Auflösung t=30 Myr

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| O1 (Konzentration): Werte N=[48] | nan | — | info | nan; Fehlerschätzung nan; p_obs nan |
| ΔlnQ (Jeans, kontinuierlich): Werte N=[48] | nan | — | info | nan; Fehlerschätzung nan; p_obs nan |
| max J bis t (N=48) | 0.3945 | 0.25 | ❌ |  |

## O2 original

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| N=48: O2(A=0) / ΔO2 | 1 | 0.95 | info | ΔO2 = 0 |

## O3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| N=48: frühe Kohärenz | 0.9999 | 0.9 | ✅ |  |
| N=48: Impulsbudget früh | 0.001869 | 0.05 | ✅ |  |

## Fluss-Bilanz

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| N=48, A=1: max|ΔM−∫F|/M(0) | 0.1259 | 0.02 | ❌ |  |
| N=48, A=0: max|ΔM−∫F|/M(0) | 3.073e-06 | 0.02 | ✅ |  |

## Vergleich (feinster Basislauf)

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| C_ρ Aperturmittel R=0.25 (A=1), max | 1.446 | — | info | Arbeit: ~50.0 |
| C_ρ Aperturmittel R=0.25 bei t≈1.20 Myr | 1.006 | — | info |  |
| C_ρ Einzelzelle Zentrum (A=1), max | 1.214 | — | info | NICHT gleichwertig zu aufgelöster Dichte (Research Plan §2) |
| Dauer min_R Q<1 (A=1) [Myr] | 10 | — | info | Arbeit: 5.8 Myr |
