# HaloGenesis-R

Reconstruction and extension of **HaloGenesis**: an ultralight-axion soliton (Schrödinger–Poisson) coupled to
primordial gas (Euler), run as a matched pair with (A=1) and without (A=0) the axion source. The project tests
whether the soliton concentrates gas, whether gas responds to the time-averaged or the instantaneous potential,
and when gas couples to the granular structure of a wave-dark-matter halo.

## Physics and numerics

| Component | Method |
|---|---|
| Axion field | Strang split-step Fourier (KDK), mass-normalised ψ, numerical SP ground state (shooting + imaginary time) |
| Gravity | spectral Poisson solver, periodic box, mean subtracted |
| Gas | dimensionally split MUSCL-Hancock + HLLC (Numba), gravity source terms, optional Jeans floor |
| Cooling | historical piecewise model, primordial H/H₂ (CIE + Galli & Palla 1998), isothermal, none |
| Coupling | gas feels a temporally filtered potential Φ̄_τ (boxcar, zero-lag, delay, EMA), axions feel Φ |
| Halo | soliton + random-phase granular halo, resolution-independent realisation |
| Theory | linear response of gas to granule fluctuations: analytic granule spectrum P_δ(k,ω) and gas susceptibility χ(k,ω) |

## Observables

O1 differential concentration · ΔlnQ / O2 Jeans diagnostics · O3 force–response coherence ·
probe cross-correlations r(ℓ) and response efficiency η vs Mach number M = c_s/σ.
All decision thresholds are pre-declared in the config files.

## Installation

```
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt     # or: pip install -e .[test]
python -m pytest -q
```

Tests run automatically on every push (GitHub Actions, `.github/workflows/tests.yml`).

Optional: `pip install pyfftw` and set `compute.fft_backend: "pyfftw"`.

## Usage

```
./run_all.sh configs/physical_fb.yaml          # validation P0–P4X, P7V (Windows: run_all.ps1)
python -m scripts.p6_campaign --campaign campaigns/laptop2_extensions.yaml
python -m scripts.p6_campaign --campaign campaigns/laptop2_decomposition.yaml
python -m scripts.p6_campaign --campaign campaigns/laptop2_mach_scan.yaml
python -m scripts.p6_analyze --config configs/physical_fb.yaml
python -m scripts.p7_response_analysis --config configs/physical_fb.yaml
```

Campaigns are resumable; `--jobs N` runs simulations in parallel, `--dry-run` prints the plan.

## Stages

| Stage | Script | Content |
|---|---|---|
| P0 | `p0_manifest` | parameter manifest, consistency checks against the original report |
| P1 | `p1_poisson_validation` | units, grid, Poisson solver |
| P2 | `p2_sp_validation` | SP solver, soliton ground state |
| P3 | `p3_hydro_validation` | Sod, conservation, Jeans dispersion, hydrostatics, cooling |
| P4 / P4X | `p4_coupling_validation`, `p4x_extensions_validation` | coupling, filters, halo, Jeans floor, cooling, seeds |
| P5 | `p5_observables` | matched run A=1/A=0, observables, probes, dashboard |
| P6 | `p6_campaign`, `p6_analyze` | campaigns, convergence, τ hypothesis test, delay/smoothing decomposition, falsification matrix |
| P7 | `p7_response_validation`, `p7_response_analysis` | response-theory validation, Mach scan vs theory |

## Key figures

`P5_*/figs/p5_DASHBOARD.png`, `P5_*/figs/p5_SNAPSHOTS.png`,
`P6/<variant>/figs/p6_HYPOTHESIS_tau.png`, `p6_DECOMPOSITION.png`, `p6_falsification_matrix.png`,
`P7_response/figs/p7_MACH_*.png`.

## Layout

```
hgr/        library (units, grid, poisson, schroedinger, soliton, hydro, kernels, cooling,
            coupled, filters, observables, response, setup, viz, config, report)
scripts/    stages P0–P7
tests/      regression and equivalence tests
configs/    default.yaml (reproduction), physical_fb.yaml (physical gas density), demo.yaml
campaigns/  run lists for p6_campaign
```

See `ROADMAP.md` for planned extensions and `CHANGELOG.md` for version history.
