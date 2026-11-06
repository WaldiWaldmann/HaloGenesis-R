#!/usr/bin/env bash
# HaloGenesis-R – alle bisherigen Pakete (Linux/WSL/macOS)
# Nutzung: ./run_all.sh [config] ["P0 P1 P2 P3 P4 P5"]
set -euo pipefail
CFG="${1:-configs/default.yaml}"
STAGES="${2:-P0 P1 P2 P3 P4 P4X P5}"
declare -A MODS=([P0]=scripts.p0_manifest [P1]=scripts.p1_poisson_validation [P2]=scripts.p2_sp_validation [P3]=scripts.p3_hydro_validation [P4]=scripts.p4_coupling_validation [P4X]=scripts.p4x_extensions_validation [P5]=scripts.p5_observables)
python -m pytest -q
for s in $STAGES; do python -m "${MODS[$s]}" --config "$CFG"; done
echo "Fertig. Ergebnisse unter runs/"
