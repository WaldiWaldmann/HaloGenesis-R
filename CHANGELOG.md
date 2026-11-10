# Changelog

## 1.1.0
- HLLC: Roe-averaged (Batten et al. 1997) wave-speed estimates; positivity-preserving, replaces Davis estimates
- fused Numba state kernel (ρ_min, P_min, non-finite count, max signal speed, min P/ρ²) for CFL, physicality checks and Truelove J; removes several N³ temporaries per step
- Poisson: cached 1/k² and ik operators, no boolean-mask copies, NaN check via the k=0 mode
- separable drift phase factor; |ψ|² without sqrt; contiguity guard for in-place kicks
- aperture pressures computed on the aperture crop only
- probes record gas pressure; P7 theory uses the measured effective sound speed; r₀ errors from a moving-block bootstrap
- pyproject.toml, GitHub Actions test workflow
- P5 benchmark (N=64, halo, 8 Myr, 1 core): 35.6 s → 20.8 s; observables change ≤ 4·10⁻⁵ (truncation level)

## 1.0.1
- English codebase, comments and docstrings stripped, black formatting, setup schematic removed

## 1.0.0
- temporal filter library and delay/smoothing decomposition (idea A)
- linear response theory, probes, Mach scan, validation V1–V3 (idea B)
