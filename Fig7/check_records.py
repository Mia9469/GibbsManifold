"""Diagnostic script: compare final training metrics across three model types."""

import numpy as np
import scipy.io as sio
from scipy import stats
import os

RECORD_DIR = os.path.join(os.path.dirname(__file__), "record")
METRICS = ["CV", "E", "loss", "mu", "sigma", "S"]
RUNS = range(11, 31)  # 11-30
LAST_N = 10  # average over last 10 epochs

# Model types: label -> filename pattern (with {} for run number)
TYPES = {
    "null (_16)":           "record_16_{}.mat",
    "cauchy (_cauchy_16_nl)": "record_cauchy_16_nl_{}.mat",
    "cv (_cv_16_nl)":       "record_cv_16_nl_{}.mat",
}


def load_final_values(pattern, runs, last_n=LAST_N):
    """Load records and return dict of metric -> array of final values (one per run)."""
    results = {m: [] for m in METRICS}
    missing = []
    for r in runs:
        fpath = os.path.join(RECORD_DIR, pattern.format(r))
        if not os.path.exists(fpath):
            missing.append(r)
            continue
        data = sio.loadmat(fpath)
        for m in METRICS:
            vals = data[m].flatten()
            results[m].append(np.mean(vals[-last_n:]))
    if missing:
        print(f"    WARNING: missing runs {missing}")
    return {m: np.array(v) for m, v in results.items()}


# --- Collect data ---
all_data = {}
for label, pat in TYPES.items():
    print(f"Loading {label} ...")
    all_data[label] = load_final_values(pat, RUNS)
    n = len(all_data[label]["CV"])
    print(f"    Loaded {n} runs")

# --- Print summary ---
print("\n" + "=" * 72)
print("FINAL VALUES (mean +/- std over last 10 epochs, across runs 11-30)")
print("=" * 72)

labels = list(TYPES.keys())
for m in METRICS:
    print(f"\n  {m}:")
    for label in labels:
        v = all_data[label][m]
        if len(v) > 0:
            print(f"    {label:30s}  {v.mean():.6f} +/- {v.std():.6f}  (n={len(v)})")
        else:
            print(f"    {label:30s}  NO DATA")

# --- T-tests for key metrics ---
KEY_METRICS = ["CV", "E", "loss"]
print("\n" + "=" * 72)
print("PAIRWISE T-TESTS (Welch's, two-sided)")
print("=" * 72)

pairs = [(labels[0], labels[1]), (labels[0], labels[2]), (labels[1], labels[2])]
for m in KEY_METRICS:
    print(f"\n  {m}:")
    for a, b in pairs:
        va, vb = all_data[a][m], all_data[b][m]
        if len(va) < 2 or len(vb) < 2:
            print(f"    {a} vs {b}: insufficient data")
            continue
        t, p = stats.ttest_ind(va, vb, equal_var=False)
        print(f"    {a:30s} vs {b:30s}  t={t:+.4f}  p={p:.4e}")
