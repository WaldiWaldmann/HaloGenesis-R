# Usage: .\run_all.ps1 [config] ["P0 P1 P2 P3 P4 P4X P7V"]
$ErrorActionPreference = "Stop"
$cfg = if ($args.Count -gt 0) { $args[0] } else { "configs/physical_fb.yaml" }
$stages = if ($args.Count -gt 1) { $args[1].Split(" ") } else { @("P0","P1","P2","P3","P4","P4X","P7V") }
$mods = @{ P0="scripts.p0_manifest"; P1="scripts.p1_poisson_validation"; P2="scripts.p2_sp_validation"; P3="scripts.p3_hydro_validation"; P4="scripts.p4_coupling_validation"; P4X="scripts.p4x_extensions_validation"; P5="scripts.p5_observables"; P7V="scripts.p7_response_validation" }
python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw "unit tests failed" }
foreach ($s in $stages) {
  python -m $mods[$s] --config $cfg
  if ($LASTEXITCODE -ne 0) { throw "$s failed" }
}
Write-Host "Done. Results in runs\"
