#!/usr/bin/env bash
# P3 matched-baseline sweep launcher.
#
# Trains every (method, knob, seed) configuration once.
#
# Sweep design:
#   - null     : 8 lmda values × 3 seeds = 24 runs
#   - cauchy   : 8 alpha_s values × 3 seeds = 24 runs
#   - cvmax_v2 : cv_constrained_v2 (gates U-update too), same grid = 48 runs
#   Total (default current-v2 sweep): 96 runs
#   - cvmax v1 remains available only through --methods cvmax as a legacy
#     sensitivity and is excluded from the manuscript sweep.
#
# cvmax  (cv_constrained):    warmup-relative budget gate.  In practice this
#                              gate is rarely closed (recon stays below the
#                              warmup baseline), so v1 ≈ plain `cv`.
# cvmax_v2 (cv_constrained_v2): absolute RE-target gate.  recon_budget is
#                              reinterpreted as the target RE; CV only pushes
#                              when current RE ≤ target.  v2 sweeps target RE
#                              over (0.40, 0.50, 0.55, 0.60) which forces the
#                              dictionary to first reach the target RE before
#                              any CV pressure is applied.
# Use --methods cvmax_v2 to add only the v2 runs to an existing sweep.
#
# Each run is independent and writes to results/records/<run_id>.json,
# so re-launches are safe: the script skips a configuration if its
# JSON record already exists (resume support).
#
# Usage
# -----
#   bash p3_sweep.sh                 # sequential, all 96 runs
#   bash p3_sweep.sh --parallel 4    # up to 4 runs concurrently
#   bash p3_sweep.sh --dry-run       # print commands, don't execute
#   bash p3_sweep.sh --epochs 10     # quick smoke test
#   bash p3_sweep.sh --methods cvmax_v2 # restrict to one method
#   bash p3_sweep.sh --methods all      # include legacy cvmax v1 (144 runs)
#
# Output
# ------
#   results/records/<run_id>.json    -- per-run scalar metrics
#   results/checkpoints/<run_id>.pth -- trained model
#   results/log/<run_id>.log         -- console log

set -euo pipefail

# -- Defaults -------------------------------------------------------------
PARALLEL=1            # how many runs to fan out at once (1 = sequential)
DRY_RUN=0
EPOCHS=30             # convergence diagnostic showed ep30 ≈ ep50 (<1% RE drift)
SEEDS=(0 1 2)
METHODS_FILTER="null,cauchy,cvmax_v2"  # current manuscript methods

# -- Sweep grids ----------------------------------------------------------
# null:   8 L1 strengths
LMDAS_NULL=(1e-4 5e-4 1e-3 3e-3 5e-3 1e-2 3e-2 5e-2)

# cauchy: 8 Cauchy weights (lmda fixed at LMDA_BASE)
ALPHAS_CAUCHY=(0.05 0.1 0.3 0.5 1.0 2.0 5.0 10.0)

# cvmax v1 (cv_constrained):  warmup-relative budget × cv_ratio
RECON_BUDGETS=(1.0 1.05 1.10 1.20)
CVRS_CVMAX=(0.05 0.10 0.20 0.40)

# cvmax_v2 (cv_constrained_v2): absolute RE target × cv_ratio.
# recon_budget here is interpreted as a target RE — CV only pushes when
# current RE ≤ this value, forcing the dictionary to first learn recon.
RECON_TARGETS_V2=(0.40 0.50 0.55 0.60)
CVRS_V2=(0.05 0.10 0.20 0.40)

# Common settings (kept identical across methods to ensure matched comparison)
LMDA_BASE=5e-3        # L1 strength when not the sweep knob
PYTHON=${PYTHON:-python}

# -- Argument parsing ------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)
            PARALLEL="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=1; shift ;;
        --epochs)
            EPOCHS="$2"; shift 2 ;;
        --seeds)
            IFS=',' read -ra SEEDS <<< "$2"; shift 2 ;;
        --methods)
            METHODS_FILTER="$2"
            [[ "$METHODS_FILTER" == "all" ]] && METHODS_FILTER=""
            shift 2 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# //' | head -n 35
            exit 0 ;;
        *)
            echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

cd "$(dirname "$0")"

mkdir -p results/records results/checkpoints results/log

# -- Helpers ---------------------------------------------------------------
run_id_for() {
    # Mirrors p3_train.run_id() exactly so the skip-if-exists check works.
    local method="$1" lmda="$2" alpha_s="$3" cv_ratio="$4" budget="$5" seed="$6"
    # printf %g matches Python's :g formatter for the small floats we use.
    printf "%s__lmda%g_a%g_cv%g_b%g__seed%d" \
        "$method" "$lmda" "$alpha_s" "$cv_ratio" "$budget" "$seed"
}

dispatch() {
    # dispatch <method> <lmda> <alpha_s> <cv_ratio> <recon_budget> <seed>
    local method="$1" lmda="$2" alpha_s="$3" cv_ratio="$4" budget="$5" seed="$6"
    local rid
    rid=$(run_id_for "$method" "$lmda" "$alpha_s" "$cv_ratio" "$budget" "$seed")
    local rec="results/records/${rid}.json"

    if [[ -f "$rec" ]]; then
        echo "[skip] $rid (record exists)"
        return 0
    fi

    local cmd=("$PYTHON" p3_train.py
        --method "$method"
        --lmda "$lmda"
        --alpha_s "$alpha_s"
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

# Job queue for parallel mode
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

# -- Method gate -----------------------------------------------------------
want_method() {
    local m="$1"
    [[ -z "$METHODS_FILTER" ]] && return 0
    IFS=',' read -ra arr <<< "$METHODS_FILTER"
    for x in "${arr[@]}"; do [[ "$x" == "$m" ]] && return 0; done
    return 1
}

# -- Sweep loops -----------------------------------------------------------
START=$(date +%s)

echo "=== P3 sweep: parallel=$PARALLEL  epochs=$EPOCHS  seeds=${SEEDS[*]} ==="

# Method 1: null (L1 only).  recon_budget unused → fixed at 1.0 for run-id stability.
if want_method null; then
    for lmda in "${LMDAS_NULL[@]}"; do
        for seed in "${SEEDS[@]}"; do
            launch null "$lmda" 0.0 0.0 1.0 "$seed"
        done
    done
fi

# Method 2: cauchy (L1 + Cauchy prior).  recon_budget unused.
if want_method cauchy; then
    for a in "${ALPHAS_CAUCHY[@]}"; do
        for seed in "${SEEDS[@]}"; do
            launch cauchy "$LMDA_BASE" "$a" 0.0 1.0 "$seed"
        done
    done
fi

# Legacy sensitivity only: cvmax v1 (warmup-relative gate).  This loop is not
# part of the default manuscript sweep; request it explicitly with
# --methods cvmax or --methods all.
if want_method cvmax; then
    for budget in "${RECON_BUDGETS[@]}"; do
        for cvr in "${CVRS_CVMAX[@]}"; do
            for seed in "${SEEDS[@]}"; do
                launch cvmax "$LMDA_BASE" 0.0 "$cvr" "$budget" "$seed"
            done
        done
    done
fi

# Method 4: cvmax_v2 (cv_constrained_v2 — absolute RE-target gate).
# recon_budget here = target RE (CV only fires when current RE ≤ target).
# Sweep target RE over (0.40, 0.50, 0.55, 0.60) to force the dictionary
# down to that RE before CV pressure kicks in.
if want_method cvmax_v2; then
    for budget in "${RECON_TARGETS_V2[@]}"; do
        for cvr in "${CVRS_V2[@]}"; do
            for seed in "${SEEDS[@]}"; do
                launch cvmax_v2 "$LMDA_BASE" 0.0 "$cvr" "$budget" "$seed"
            done
        done
    done
fi

# Drain remaining background jobs
if [[ $PARALLEL -gt 1 ]]; then
    wait
fi

ELAPSED=$(( $(date +%s) - START ))
echo "=== sweep complete in ${ELAPSED}s ==="
echo "Records: $(ls results/records/ | wc -l) JSON files"
echo "Next:    python p3_eval.py"
