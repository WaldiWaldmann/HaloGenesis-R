# HaloGenesis-R

Rekonstruktion von **HaloGenesis** (Axion-Soliton + baryonisches Gas, gefiltertes Potential) mit dem Ziel, die drei Observablen O1–O3 des Research Plans zu testen und die Kennzahlen der Originalarbeit nachzuprüfen. Der Code ist eine **Neuimplementierung** mit eigener Versionskennung und **nicht** der historische Code.

**Stand:** P0–P6 fertig. P0–P4 sind Verifikation (alle Tests bestanden). P5 ist der gematchte Lauf A=1/A=0 mit O1–O3. P6 ist die Kampagne (Auflösung, τ-Ablation, n-Scan) mit Falsifikationsmatrix; die Produktionsläufe laufen lokal.

## Installation (einmalig)

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux, WSL oder macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.12 wird empfohlen. Alle Versionen sind exakt gepinnt.

## Ausführen

```powershell
.\run_all.ps1                                       # Windows: Tests + P0–P5 mit configs/default.yaml
.\run_all.ps1 configs/physical_fb.yaml "P5"         # nur P5 für den physikalischen Lauf
./run_all.sh configs/default.yaml "P4 P5"           # Linux/WSL
python -m scripts.p5_observables --config configs/physical_fb.yaml --N 128 --tau 3
```

**Zwei Laptops:** Laptop 1 rechnet `configs/default.yaml` (Reproduktion, n = 10 cm⁻³), Laptop 2 `configs/physical_fb.yaml` (n = 0,27 cm⁻³, kosmischer Baryonenanteil, sonst identisch). Mit `--N 128` steigt die Laufzeit gegenüber N = 48 etwa um den Faktor 50. Auf einem Kern sind das ~2 h; Numba und FFT nutzen alle Kerne. Die Werte von P5 werden nur berichtet, ein physikalisch negatives Ergebnis bricht nichts ab.

Einzeln:

```
python -m pytest -q                                            # Regressionstests (~1 s)
python -m scripts.p0_manifest --config configs/default.yaml    # P0  (~10 s)
python -m scripts.p1_poisson_validation --config configs/default.yaml   # P1 (~1 min)
python -m scripts.p2_sp_validation --config configs/default.yaml        # P2 (~1 min)
python -m scripts.p3_hydro_validation --config configs/default.yaml     # P3 (~1.5 min)
python -m scripts.p4_coupling_validation --config configs/default.yaml  # P4 (~1 min)
python -m scripts.p5_observables --config configs/default.yaml          # P5 (~2.5 min bei N=48)
```

Mit `--skip-timing` lässt sich der Laufzeit-Benchmark T6 überspringen. Für einen Lauf auf dem zweiten Laptop oder mit eigenen Parametern kopierst du die Config, änderst `meta.run_label` und die gewünschten Werte und startest mit `--config`. Jeder Run landet in einem eigenen Ordner.

## Ausgaben

```
runs/<run_label>/P0|P1/
  P0_report.md / P1_report.md   Tabellen aller Kennzahlen mit Klasse und Bestanden/Info
  *_values.json                 maschinenlesbar
  manifest_*.json               Config, SHA-256 von Config und Quellcode, Versionen, Plattform
  figs/                         2D-Abbildungen (PNG)
  figs3d/*.png                  3D statisch (Flächen, orthogonale Schnitte)
  figs3d/*.html                 3D interaktiv (Browser, offline – plotly.min.js liegt daneben)
```

## Provenienz-Regeln

Alle Eingabewerte in `configs/default.yaml` sind **neu gewählt**. Aus der Originalarbeit übernommene Werte sind mit `# quelle:` markiert. Der Block `historical_reference` dient nur dem Vergleich und wird nie als Eingabe gelesen.

Physikalische Parameter haben **keine Defaults im Code**. Fehlt ein Schlüssel, bricht das Programm ab. Akzeptanzschwellen der Validierung stehen vorab in der Config (`validation_p1`).

## Paketplan

| Paket | Inhalt | Status |
|---|---|---|
| P0 | Parameter-Manifest, Konsistenzprüfung gegen Originalarbeit | ✅ |
| P1 | Einheiten, Gitter, spektraler Poisson-Löser, Validierung T0–T6 | ✅ |
| P2 | SP-Solver (Strang-KDK), numerischer Soliton-Grundzustand (Shooting + imaginäre Zeit) | ✅ |
| P3 | Hydro (MUSCL-Hancock + HLLC, Numba), Kühlung (implizit), Sod/Jeans/Hydrostatik | ✅ |
| P4 | Kopplung, Potentialfilter (Gl. 42), Lockstep A=1/A=0, Gravitations-Zeitschritt | ✅ |
| P5 | Observablen O1–O3 inkl. Fluss- und Impulsbudget-Gegenproben, Truelove-Kriterium | ✅ |
| P6 | Kampagnen-Runner (resume-fähig) + Analysator: Konvergenz, τ-/n-Scan, Falsifikationsmatrix | ✅ (Läufe lokal) |
| P4X | Modellerweiterungen B4–B7 validiert (Halo, Jeans-Boden, primordiale Kühlung, Seeds) | ✅ |
| P6 | Produktionsläufe, Konvergenz, τ- und n-Scan, Vergleich mit Tab. 1 | offen |

## Modulübersicht

| Datei | Zweck |
|---|---|
| `hgr/units.py` | Code-Einheiten kpc/Myr/M_sun über astropy; Jeans-, Schall-, Freifallgrößen |
| `hgr/grid.py` | periodisches 3D-Gitter, Ursprung auf Gitterpunkt, k-Vektoren |
| `hgr/poisson.py` | spektraler Poisson-Löser, spektraler Gradient, normiertes Residuum ε_P |
| `hgr/analytic.py` | Gauß-Kugel, Schive-Profil (empirisch), periodische Hintergrundkorrektur |
| `hgr/viz.py` | 2D-/3D-Visualisierung (matplotlib + plotly) |
| `hgr/config.py` | Config-Validierung, Manifest mit Hashes |
| `hgr/schroedinger.py`, `hgr/soliton.py` | SP-Zeitschritt, Energie/Norm, Grundzustand |
| `hgr/hydro.py`, `hgr/cooling.py`, `hgr/riemann_exact.py` | Finite-Volumen-Hydro, Kühlung, Referenzlöser |
| `hgr/coupled.py`, `hgr/setup.py` | gekoppeltes Lockstep-System, gemeinsame Anfangsbedingungen |
| `hgr/observables.py` | O1–O3, Aperturen, Jeans-Q, Belegungsdauer, Truelove |
| `configs/physical_fb.yaml` | wie default, nur n = 0,27 cm⁻³ (f_b · ρ_a(0)) |

## P6 – Produktionskampagne

```
python -m scripts.p6_campaign --campaign campaigns/laptop1.yaml --dry-run    # Plan + Zeitschätzung
python -m scripts.p6_campaign --campaign campaigns/laptop1.yaml              # Läufe (resume-fähig)
python -m scripts.p6_analyze  --config configs/default.yaml                  # Auswertung -> runs/<label>/P6/
```

Läufe des anderen Laptops kopierst du einfach nach `runs/<run_label>/` und startest die Analyse neu. Die Entscheidungsregeln stehen in `p6` jeder Config und wurden vor den Produktionsläufen festgelegt. `campaigns/demo.yaml` ist nur ein Pipeline-Test von wenigen Minuten.

## Modellerweiterungen (B4–B7) – per CLI/Config schaltbar, Defaults = Reproduktion

| Erweiterung | Schalter | Wirkung |
|---|---|---|
| B4 granularer Halo | `--setup soliton_halo` / `setup.kind` | Soliton + Zufallsphasen-Halo (Maxwell, σ = 30 km/s), 20 Myr SP-Vorrelaxation. Erzeugt echte Potentialfluktuationen; erst damit testet der τ-Scan die Mittelungsidee. |
| B5 Jeans-Boden | `--jeans-floor 4` / `hydro.jeans_floor_NJ` | erzwingt λ_J ≥ 4dx; Kollaps wird ab dort nicht verfolgt. Gebodeter Massenanteil wird berichtet; P6 markiert „bodendominiert“. Sinnvoll erst ab N ≥ 128. |
| B6 primordiale Kühlung | `--cooling primordial_h2` / `gas.cooling_params` | H-Atomkühlung (CIE, Cen 1992/Katz et al. 1996) + H₂ (Galli & Palla 1998), feste x_H2 = 5·10⁻⁴, T_floor = T_CMB(z=20). |
| B7 Seeds | `--seed 11` | weitere Rausch-Realisierung. P6 schätzt σ_Seeds und verlangt \|Wert\| > 2σ. |

P6 wertet jede Variante (Setup × Boden × Kühlung) **getrennt** aus: `runs/<label>/P6/<variante>/`.

**Checkliste vor Veröffentlichung:** Die Koeffizienten der Kühlraten in `hgr/cooling.py` (Cen 1992, Katz et al. 1996 Tab. 1–2, Galli & Palla 1998) gegen die Originaltabellen prüfen.

## Performance (v0.8)

Gemessen am selben P5-Lauf (N = 64, 1 Kern): **18,7 s → 9,2 s**; mit primordialer Kühlung deutlich mehr. Alle Optimierungen sind gegen die alten Pfade getestet (`tests/test_performance_equivalence.py`): Hydro ist bitgleich, alles andere stimmt auf Rundungsniveau überein (≤ 10⁻¹²).

- **Kühlung als Numba-Kernel:** Bisektion pro Zelle mit Frühabbruch; nicht kühlende Zellen werden übersprungen. Das war vorher der größte Hotspot (~45 %).
- **Hydro ohne Transponier-Kopien:** strided Ebenen-Kernel für die x/y-Sweeps (spart 6 Volumenkopien pro Schritt).
- **SP:**
  - Kicks laufen in-place und ohne komplexe `exp`-Temporaries.
  - Der Drift-Phasenfaktor wird gecacht.
  - Zwischen den Teilschritten werden die Halbkicks zu Vollkicks fusioniert (Leapfrog).
- **Filter im Fourierraum:** Φ̂ aus dem Poisson-Löser wird wiederverwendet (spart FFTs), und Φ̄(Zentrum) kommt direkt aus dem Spektrum.
- **Diagnostik:** Aperturen werden auf ihre Bounding-Box zugeschnitten (< 2 % der Zellen), Truelove braucht nur einen Durchlauf, Radialprofile werden dezimiert.

**Weitere Hebel für dich:**
- `compute.fft_backend: "pyfftw"` in der Config, nach `pip install pyfftw`. Typisch 1,3–2× schnellere FFTs; die Ergebnisse unterscheiden sich nur auf Rundungsniveau.
- `p6_campaign ... --jobs 2` startet 2 Läufe parallel mit je halber Kernzahl. Weil FFTs schlecht über viele Kerne skalieren, steigt so meist der Gesamtdurchsatz. Speicher pro Lauf beachten (N = 128 ≈ 1 GB, N = 256 ≈ 8 GB).

## Abbildungs-Leitfaden

- **Ergebnis-Seite pro Lauf:** `runs/<label>/P5_.../figs/p5_DASHBOARD.png`. Sie zeigt Felder, O1, ΔlnQ, O2, O3, Auflösung und Jeans-Boden. Grau schattierte Zeiten sind nicht aufgelöst (J > 1/4), orange schattierte bodendominiert.
- **Gesamturteil:** `runs/<label>/P6/<variante>/figs/p6_falsification_matrix.png`.
- **Validierung:** Die übrigen Abbildungen dokumentieren P0–P4X. 3D-HTML-Isoflächen umschließen 50/80/95 % der Masse.

## v0.9 – Änderungen vor den Produktionsläufen

- **Mitlaufende Aperturen** (`analysis.aperture_center: soliton`, vorab festgelegt): Das Zentrum ist der Schwerpunkt von |ψ|² innerhalb von r_track = 0,75 kpc, iterativ bestimmt, periodisch korrekt und auf eine Zelle gerundet. Es gilt für beide Zweige gleich. Im Halo-Test hätte die feste Apertur O1 um ~30 % unterschätzt. Die Massenbilanz-Gegenprobe nutzt bewusst feste Aperturen.
- **Zeitverfeinerung:** `--dt-max` bzw. `dt_max:` in der Kampagne. P6 addiert den Zeitfehler e_t = |F(dt) − F(dt/2)| zum Ortsfehler in der Robustheitsregel.
- **Bereinigte Kampagnen:**
  - Die isolierten τ-Scans entfallen (dort gilt Φ̄ = Φ); τ = 10 bleibt als Nulltest.
  - Es bleibt 1 Seed-Lauf.
  - Die Kühlung ist entkoppelt (isoliert + primordial).
  - Halo-Varianten sind einzeln zuordenbar.
- **Grafiken:**
  - Massenbilanz relativ zu M(0).
  - dt-Plot mit Snapshot-Markierungen.
  - Snapshots schneiden durch das Soliton-Zentrum.
  - Das Dashboard zeigt die Zentrumsverschiebung.

## Abbildungs-Leitfaden (v0.9.1)

| Datei | Frage, die sie beantwortet |
|---|---|
| `P5_*/figs/p5_SETUP.png` | Wie ist der Lauf aufgebaut? Schema beider Zweige, Filter, Observablen und Parameter, für Bericht und Jury. |
| `P5_*/figs/p5_DASHBOARD.png` | Was passiert in diesem Lauf? Felder als Überdichte δ auf gemeinsamer Skala, O1, ΔlnQ, Q, O3, Auflösung. (g) ist nur dort rot, wo *ausschließlich* mit Soliton Q < 1 gilt. (j) zeigt Φ gegen Φ̄_τ am Soliton-Zentrum und die Soliton-Drift. |
| `P5_*/figs/p5_SNAPSHOTS.png` | Wie entwickelt es sich? Axionfeld ρ_a, δ (A=1/A=0), Wirkung und \|δv\| über die Zeit, je Zeile eine Skala. |
| `P6/<variante>/figs/p6_falsification_matrix.png` | Ist das Ergebnis robust (Auflösung, Zeitschritt, Seeds, Truelove)? |
| `P6/<variante>/figs/p6_HYPOTHESIS_tau.png` | **Kernfrage: Bewirkt die Zeitmittelung (τ) etwas Messbares?** Links Δ(t) gegen das Fehlerband, rechts die Auswertezeiten, unten das Urteil. |

Korrekturen in v0.9.1: Φ_c und Φ̄_c werden am Soliton-Zentrum ausgewertet (vorher am Box-Zentrum), und das Radialprofil (d) ist um das Soliton-Zentrum gemittelt.
