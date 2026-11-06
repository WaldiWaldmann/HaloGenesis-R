# P4 – Kopplung Axion–Gas, Potentialfilter, Lockstep

**Gesamtergebnis:** BESTANDEN

Schwellen vorab in `validation_p4`. Gas reagiert auf Φ̄_τ, Axionen auf Φ.

## C1

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| Filter exakt für konstantes Φ (ungleiche Schritte) | 2.22e-16 | 1e-12 | ✅ |  |
| lineares Φ: rel. Fehler (Vorab-Schwelle 'exakt' war FALSCH begründet) | 0.001577 | — | info | Gl. 42 ist nur für konstantes Φ exakt; Wert bleibt zur Transparenz stehen |
| lineares Φ: Fehler / Schranke βΔt²/(8w) (hergeleitet) | 0.975 | 1 | ✅ |  |
| max Bias / Schranke Lτ/2 (Gl. 43) | 0.9999 | 1 | ✅ | ≤ 1 bestätigt die Schranke |

## C2

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| max|ρ_a(0,t)/ρ_a(0,0) − 1| (Gas vernachlässigbar) | 0.001682 | 0.005 | ✅ |  |

## C3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| max|U(A=1)−U(A=0)|/max|U| nach 10 Schritten (ψ≡0) | 0 | 1e-12 | ✅ | Ablation ändert NUR die Axionquelle |

## C4

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| τ=0.0: max|ΔP_ges| / max Σ|ρv| | 6.063e-11 | 0.01 | ✅ | Splitting-Fehler |
| τ=0.0: Makroschritte / mittlere SP-Teilschritte | 40 | — | info |  |
| τ=3.0: max|ΔP_ges| / max Σ|ρv| | 1.532e-08 | — | info | τ>0: Asymmetrie Axion↔Gas (Research Plan §8.3) |
| τ=3.0: ε_filter (Gl. 45), E_ref = E_int(0) | 0.0003041 | — | info |  |
| τ=3.0: Makroschritte / mittlere SP-Teilschritte | 40 | — | info |  |
