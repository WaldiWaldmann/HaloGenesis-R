# P3 – Hydrodynamik, Gravitationsquellterm, Kühlung

**Gesamtergebnis:** BESTANDEN

Limiter: `vanleer`. Schwellen vorab in `validation_p3` festgelegt.

## H1

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| L1(ρ) N=100 (x/y/z max) | 0.004762 | — | info |  |
| Achsen-Asymmetrie N=100 | 0 | 1e-14 | ✅ | Split-Schema muss richtungsunabhängig sein |
| L1(ρ) N=200 (x/y/z max) | 0.002555 | — | info |  |
| Achsen-Asymmetrie N=200 | 0 | 1e-14 | ✅ | Split-Schema muss richtungsunabhängig sein |
| L1(ρ) N=400 (x/y/z max) | 0.001331 | — | info |  |
| Achsen-Asymmetrie N=400 | 0 | 1e-14 | ✅ | Split-Schema muss richtungsunabhängig sein |
| L1(ρ) N=800 (x/y/z max) | 0.0007336 | — | info |  |
| Achsen-Asymmetrie N=800 | 0 | 1e-14 | ✅ | Split-Schema muss richtungsunabhängig sein |
| L1(ρ) bei N=400 | 0.001331 | 0.005 | ✅ |  |
| beobachtete Ordnung L1 (letzte Verdopplung) | 0.8592 | 0.8 | ✅ | Stoß/Kontakt begrenzen auf ~1 |

## H2

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| rel. Änderung Masse | 0 | 1e-12 | ✅ |  |
| rel. Änderung Impuls x | 5.638e-18 | 1e-12 | ✅ |  |
| rel. Änderung Impuls y | 8.674e-19 | 1e-12 | ✅ |  |
| rel. Änderung Impuls z | 0 | 1e-12 | ✅ |  |
| rel. Änderung Energie | 0 | 1e-12 | ✅ |  |
| lokale Rückfälle auf 1. Ordnung | 0 | — | info |  |

## H3

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| Mode 1 (k/k_J=0.50) rel. Fehler ω² | 0.0001231 | 0.02 | ✅ | instabil (Wachstum) |
| Mode 3 (k/k_J=1.50) rel. Fehler ω² | 0.004325 | 0.02 | ✅ | stabil (Oszillation) |
| Mode 4 (k/k_J=2.00) rel. Fehler ω² | 0.01006 | 0.02 | ✅ | stabil (Oszillation) |

## H4

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| Testtemperatur (isotherm) [K] | 4.634e+04 | — | info | aus vorgegebenem Dichtekontrast |
| max |v|/c_s nach t_end | 0.004319 | 0.05 | ✅ | künstliche Strömung = Rauschboden für O3 |
| max |v| [km/s] | 0.07709 | — | info |  |
| rel. Änderung ρ(0) | -0.001821 | — | info |  |

## H5

| Größe | Wert | Schwelle | Status | Anmerkung |
|---|---|---|---|---|
| n=0.01: T nach 0.1 Myr (implizit) [K] | 300 | — | info |  |
| n=0.01: rel. Abweichung zur Referenz-ODE | 2.22e-16 | 1e-06 | ✅ |  |
| n=0.01: Kühlzeit bei T0=5000 K [s] | 1269 | — | info | historisches Λ |
| n=0.1: T nach 0.1 Myr (implizit) [K] | 300 | — | info |  |
| n=0.1: rel. Abweichung zur Referenz-ODE | 0 | 1e-06 | ✅ |  |
| n=0.1: Kühlzeit bei T0=5000 K [s] | 126.9 | — | info | historisches Λ |
| n=1.0: T nach 0.1 Myr (implizit) [K] | 300 | — | info |  |
| n=1.0: rel. Abweichung zur Referenz-ODE | 2.22e-16 | 1e-06 | ✅ |  |
| n=1.0: Kühlzeit bei T0=5000 K [s] | 12.69 | — | info | historisches Λ |
| n=10.0: T nach 0.1 Myr (implizit) [K] | 300 | — | info |  |
| n=10.0: rel. Abweichung zur Referenz-ODE | 2.22e-16 | 1e-06 | ✅ |  |
| n=10.0: Kühlzeit bei T0=5000 K [s] | 1.269 | — | info | historisches Λ |
| Rückwärts-Euler Ordnung (nicht-steif, T=1e5 K) | 0.9988 | 1 | ✅ | T_ref = 56250 K nach t = 0.0104 Myr |
