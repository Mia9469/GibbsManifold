#!/usr/bin/env bash
# P3 cvmax_v2 sweep — absolute-RE-target gate variant.
#
# Standalone launcher for the cvmax_v2 sweep only.  Use this when iterating
# on the v2 algorithm without re-touching null / cauchy / cvmax.  The main
# launcher (p3_sweep.sh) covers all four methods and is what you run for the
# full 144-config matrix; this script is for fast v2 turnaround.
#
# Background
# ----------
# v1 cvmax (`cv_constrained`) uses a warmup-relative gate
#   recon ≤ recon_warmup × budget
# which is almost never closed in practice (recon keeps decreasing past the
# warmup baseline), so v1 effectively reduces to plain `cv` — its RE plateaus
# at ≈0.70 and cannot match null's RE range (0.37–0.60).
#
# v2 cvmax (`cv_constrained_v2`) reinterprets recon_budget as an
# **absolute target RE**: CV is only pushed when the current per-batch RE is
# ≤ recon_budget.  This forces the dictionary to first learn pure
# reconstruction down to the target, only then is CV pressure applied.
#
# This is per-batch self-adaptive curriculum: early batches gate-closed
# (≈null), late batches gate-open (CV active), automatic transition.
#
# Usage
# -----
#   bash p3_sweep_v2.sh                  # sequential, 48 v2 runs
#   bash p3_sweep_v2.sh --parallel 4     # fan out 4 concurrent
#   bash p3_sweep_v2.sh --dry-run        # print commands, don't execute
#   bash p3_sweep_v2.sh --epochs 10      # smoke test
#   bash p3_sweep_v2.sh --seeds 0        # single seed (16 runs)
#   bash p3_sweep_v2.sh --targets 0.4,0.5  # restrict target REs
#   bash p3_sweep_v2.sh --cvrs 0.1,0.2     # restrict cv_ratios
#   bash p3_sweep_v2.sh --reset          # rm old v2 records first (CARE)
#
# Output
# ------
#   results/records/cvmax_v2__lmda5e-3_a0_cv{R}_b{T}__seed{S}.json
#   results/checkpoints/cvmax_v2__...pth
#   results/log/cvmax_v2__...log
#
# After it finishes, run:  python p3_eval.py

set -euo pipefail

# -- Defaults -------------------------------------------------------------
PARALLEL=1
DRY_RUN=0
RESET=0
EPOCHS=30
SEEDS=(0 1 2)

# v2 sweep grid: target RE × cv_ratio
RECON_TARGETS_V2=(0.40 0.50 0.55 0.60)
CVRS_V2=(0.05 0.10 0.20 0.40)

LMDA_BASE=5e-3
PYTHON=${PYTHON:-python}

# -- Argument parsing -----------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)   PARALLEL="$2"; shift 2 ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --reset)      RESET=1; shift ;;
        --epochs)     EPOCHS="$2"; shift 2 ;;
        --seeds)      IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --targets)    IFS=',' read -ra RECON_TARGETS_V2 <<< "$2"; shift 2 ;;
        --cvrs)       IFS=',' read -ra CVRS_V2 <<< "$2"; shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# //' | head -n 40
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")"

mkdir -p results/records results/checkpoints results/log

# -- Optional reset -------------------------------------------------------
if [[ "$RESET" -eq 1 ]]; then
    echo "[reset] removing existing cvmax_v2 records / logs / checkpoints"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry] rm -f results/records/cvmax_v2__*.json"
        echo "[dry] rm -f results/log/cvmax_v2__*.log"
        echo "[dry] rm -f results/checkpoints/cvmax_v2__*.pth"
    else
        rm -f results/records/cvmax_v2__*.json
        rm -f results/log/cvmax_v2__*.log
        rm -f results/checkpoints/cvmax_v2__*.pth
    fi
fi

# -- Helpers --------------------------------------------------------------
run_id_for() {
    # Mirrors p3_train.run_id() exactly so the skip-if-exists check works.
    local cv_ratio="$1" budget="$2" seed="$3"
    printf "cvmax_v2__lmda%g_a0_cv%g_b%g__seed%d" \
        "$LMDA_BASE" "$cv_ratio" "$budget" "$seed"
}

dispatch() {
    local cv_ratio="$1" budget="$2" seed="$3"
    local rid rec
    rid=$(run_id_for "$cv_ratio" "$budget" "$seed")
    rec="results/records/${rid}.json"

    if [[ -f "$rec" ]]; then
        echo "[skip] $rid (record exists)"
        return 0
    fi

    local cmd=("$PYTHON" p3_train.py
        --method cvmax_v2
        --lmda "$LMDA_BASE"
        --alpha_s 0.0
        --cv_ratio "$cv_ratio"
        --recon_budget "$budget"
        --seed "$seed"
        --epochs "$EPOCHS")

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[dry] ${cmd[*]}"
        return 0
    fi

    echo "[run] $rid"
    "${cmd[@]}"
}

# -- Job queue (parallel mode) -------------------------------------------
PIDS=()
wait_for_slot() {
    while [[ ${#PIDS[@]} -ge $PARALLEL ]]; do
        local new_pids=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                new_pids+=("$pid")
            fi
        done
        PIDS=("${new_pids[@]}")
        [[ ${#PIDS[@]} -ge $PARALLEL ]] && sleep 1
    done
}

launch() {
    if [[ $PARALLEL -le 1 ]]; then
        dispatch "$@"
    else
        wait_for_slot
        ( dispatch "$@" ) &
        PIDS+=("$!")
    fi
}

# -- Sweep ----------------------------------------------------------------
START=$(date +%s)
TOTAL=$(( ${#RECON_TARGETS_V2[@]} * ${#CVRS_V2[@]} * ${#SEEDS[@]} ))
echo "=== cvmax_v2 sweep: parallel=$PARALLEL  epochs=$EPOCHS  total=$TOTAL"
echo "    target_RE = ${RECON_TARGETS_V2[*]}"
echo "    cv_ratio  = ${CVRS_V2[*]}"
echo "    seeds     = ${SEEDS[*]}"

for budget in "${RECON_TARGETS_V2[@]}"; do
    for cvr in "${CVRS_V2[@]}"; do
        for seed in "${SEEDS[@]}"; do
            launch "$cvr" "$budget" "$seed"
        done
    done
done

if [[ $PARALLEL -gt 1 ]]; then
    wait
fi

ELAPSED=$(( $(date +%s) - START ))
N_DONE=$(ls results/records/cvmax_v2__*.json 2>/dev/null | wc -l)
echo "=== cvmax_v2 sweep complete in ${ELAPSED}s ==="
echo "    cvmax_v2 records: $N_DONE / $TOTAL"
echo "Next: python p3_eval.py"
