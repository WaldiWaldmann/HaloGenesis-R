#!/usr/bin/env bash
# Usage: ./run_all.sh [config] ["P0 P1 P2 P3 P4 P4X P7V"]
set -euo pipefail
CFG="${1:-configs/physical_fb.yaml}"
STAGES="${2:-P0 P1 P2 P3 P4 P4X P7V}"
declare -A MODS=([P0]=scripts.p0_manifest [P1]=scripts.p1_poisson_validation [P2]=scripts.p2_sp_validation [P3]=scripts.p3_hydro_validation [P4]=scripts.p4_coupling_validation [P4X]=scripts.p4x_extensions_validation [P5]=scripts.p5_observables [P7V]=scripts.p7_response_validation)
python -m pytest -q
for s in $STAGES; do python -m "${MODS[$s]}" --config "$CFG"; done
echo "Done. Results in runs/"
