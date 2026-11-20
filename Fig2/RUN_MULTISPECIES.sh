#!/usr/bin/env bash
# ============================================================
# RUN_MULTISPECIES.sh
# One-shot runner for the multi-species CV_L analysis.
#
# Usage:
#   bash RUN_MULTISPECIES.sh            # skip already-done steps
#   bash RUN_MULTISPECIES.sh --force    # recompute everything
#
# What this script does (steps are skipped if output already exists):
#   Step 1  install Python dependencies
#   Step 2  compute pvc-11 macaque V1 CV_L (local data, ~1 min)
#   Step 3  compute pvc-8 cat V1 CV_L (local data, ~1 min)
#   Step 4  download + compute Steinmetz 2019 mouse multi-region (~15 min,
#           ~300 MB download)
#   Step 5  compute Allen multi-area mouse (requires allensdk + ~3 GB
#           download on first run, ~20 min)  [OPTIONAL — skipped if no SDK]
#   Step 6  generate FigureS_multispecies.pdf / .png
# ============================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

log()  { echo "[$(date +%H:%M:%S)] $*"; }
done_() { echo "  ✓ $1 — skipping (already exists)"; }
run_()  { log "Running: $1"; python3 "$1" "${@:2}"; }

# ── Step 1: dependencies ──────────────────────────────────────────────────────
log "Step 1/6 — checking Python dependencies"
python3 -m pip install -q numpy scipy matplotlib 2>/dev/null || true
log "  numpy/scipy/matplotlib OK"

# Check allensdk (optional, for Step 5)
HAS_ALLEN=0
python3 -c "import allensdk" 2>/dev/null && HAS_ALLEN=1 || true
[[ $HAS_ALLEN -eq 1 ]] && log "  allensdk detected" \
                        || log "  allensdk NOT found — Step 5 will be skipped"

# ── Step 2: pvc-11 macaque V1 ────────────────────────────────────────────────
log "Step 2/6 — pvc-11 macaque V1 (3 monkeys)"
if [[ $FORCE -eq 1 || ! -f "pvc11_CVL_results.npz" ]]; then
    run_ compute_pvc11_CVL.py
else
    done_ "pvc11_CVL_results.npz"
fi

# ── Step 3: pvc-8 cat V1 ─────────────────────────────────────────────────────
log "Step 3/6 — pvc-8 cat V1"
if [[ $FORCE -eq 1 || ! -f "pvc8_CVL_results.npz" ]]; then
    run_ compute_pvc8_CVL.py
else
    done_ "pvc8_CVL_results.npz"
fi

# ── Step 4: Steinmetz 2019 mouse multi-region ────────────────────────────────
log "Step 4/6 — Steinmetz 2019 mouse (50 brain regions, ~300 MB download)"
if [[ $FORCE -eq 1 || ! -f "steinmetz_CVL_results.npz" ]]; then
    if [[ -f "steinmetz_alldata.npz" ]]; then
        log "  Using cached steinmetz_alldata.npz"
        run_ compute_steinmetz_CVL.py --data-path steinmetz_alldata.npz
    else
        log "  Downloading from OSF (~300 MB)..."
        run_ compute_steinmetz_CVL.py --download
    fi
else
    done_ "steinmetz_CVL_results.npz"
fi

# ── Step 5: Allen multi-area mouse (optional) ─────────────────────────────────
log "Step 5/6 — Allen NPx multi-area mouse (optional)"
if [[ $HAS_ALLEN -eq 0 ]]; then
    log "  SKIPPED (allensdk not installed)"
    log "  To enable: pip install allensdk && bash RUN_MULTISPECIES.sh --force"
else
    if [[ $FORCE -eq 1 || ! -f "allen_multiarea_CVL.npz" ]]; then
        run_ compute_allen_multiarea_CVL.py
    else
        done_ "allen_multiarea_CVL.npz"
    fi
fi

# ── Step 6: generate figure ───────────────────────────────────────────────────
log "Step 6/6 — generating FigureS_multispecies.pdf / .png"
run_ multispecies_CVL_summary.py

log "=== ALL DONE ==="
echo ""
echo "Output files:"
ls -lh FigureS_multispecies.pdf FigureS_multispecies.png 2>/dev/null || true
