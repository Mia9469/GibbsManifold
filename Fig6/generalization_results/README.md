# Generalization Tests — Fig6 Sparse Coding Models

Three sparse-coding dictionaries are compared: **Null** (L1 only), **SparseNet**
(Cauchy prior), and **CVM-v2** (absolute-RE-gated CV maximization,
`cv_constrained_v2`, `cv_ratio=0.40`, `recon_budget=0.60`).

**Common protocol.** Every dictionary is scored with an *identical* ISTA
inference — same soft-threshold λ = 5e-3 across all models — so differences
reflect the learned basis, not inference-time regularization. Full-data
structured summaries use the **training run** as the unit. Held-out
natural-image LOO-CV uses the **held-out image** as the unit after averaging
6 seeds/fold. Structured OOD summaries additionally report a paired
**pattern-level** directional count.

> Note: absolute RE under this matched, conservative inference is *not*
> comparable to the training-time RE in the main text (`result_new.py`), which
> reuses each model's constraint-aware coefficients. Only the cross-method
> comparison within a panel is meaningful.

---

## Test 1 — Held-out natural images, leave-one-image-out CV  (`held_out_natural/`) ★ primary

`heldout_generalization_test.py --cv loo`. **Leave-one-image-out CV**: each of
the 10 whitened van Hateren / Olshausen images is held out in turn while models
train on the other 9 (6 seeds/fold, averaged → one value per image). The
**image is the unit of replication (n = 10)**. Output dir `record_heldout/` —
the paper's `record/` is untouched. (`--cv split` reproduces the old single
8/2 split.)

| Metric | Null | SparseNet | CVM | Three-model readout |
|--------|------|-----------|-----|---------------------|
| RE ↓   | 0.707 | 0.718 | **0.706** | CVM best in 8/10 folds |
| SSIM ↑ | 0.4105 | 0.4091 | **0.4113** | CVM best in 6/10 folds |
| RAE ↑  | 10.55 | **10.71** | 10.66 | SparseNet mean highest; CVM best in 6/10 folds |
| mean&#124;a&#124; ↓ | 0.0595 | 0.0561 | 0.0594 | tie vs Null |

**Read:** under proper LOO-CV, the effect sizes are small because between-image
variance dominates.  The complete three-model comparison is nevertheless
directionally informative: CVM has the best mean held-out reconstruction
fidelity (RE/SSIM), while SparseNet has slightly higher mean RAE because
it uses lower mean activation.

---

## Test 2 — Oriented / structured patterns  (`structured_patterns/`) ★ primary

`generalization_test.py --patterns-only`. 31 hand-crafted out-of-distribution
patterns (gratings × orientations/frequencies, edges, Gabor bars,
centre-surround, dots), evaluated on the paper's full-data checkpoints (n = 20).

| Metric | Null | SparseNet | CVM | CVM vs Null |
|--------|------|-----------|-----|-------------|
| RE ↓   | 0.8635 | **0.8484** | 0.8539 | −1.1% vs Null (d=-1.7) |
| SSIM ↑ | 0.1924 | **0.2087** | 0.2029 | +5.5% vs Null (d=1.3) |
| RAE ↑  | 87.9 | **90.9** | 89.3 | +1.6% vs Null (d=0.7) |

**Read:** CVM-v2 improves modestly over the shared-L1 Null dictionary on
structured OOD inputs (RE better in **25/31** patterns and SSIM higher in
**22/31** patterns), but the tested Cauchy prior is the strongest run-level
dictionary in this matched-inference OOD regime.  RAE values are inflated by
the near-zero mean activation in this regime and should be interpreted only as
a dimensionless reconstruction--activation score.

---

## Test 3 — 1/f (pink) noise  (`pink_noise/`)  — supplementary / robustness only

`generalization_test.py --pink-noise-only`. Synthetic 1/f images. All three
methods are within ~1% on every metric (Null actually has the lowest RE). This
is a **null result** — useful only to show CVM does not degrade on synthetic
naturalistic OOD input; it is *not* evidence of a CVM advantage. (Note: the
whitened training images have a flattened spectrum, so 1/f input is a spectral
mismatch, i.e. genuinely OOD — not "matched" natural statistics.)

---

## Composite main-text panel

`make_fig6_generalization_panels.py` → `Fig6_generalization_panels.{pdf,png}`
(also copied to `updated/figures/Figure6_generalization.pdf`, referenced as
Fig. \ref{fig:CV-gen} in `result6.tex`). Panels are a 2×3 comparison: rows are
held-out natural images and structured OOD patterns; columns are RE, SSIM, and
RAE. All panels show Null, SparseNet, and CVM side-by-side. Archived JSON files
retain the legacy field name `ICE`; it stores RAE and has no information unit.

---

## Reproduce

```bash
cd Fig6
python heldout_generalization_test.py --cv loo --n-runs 6 --n-epochs 10  # Test 1 LOO-CV
python generalization_test.py --patterns-only                            # Test 2, trains missing CVM-v2 if needed
python make_fig6_generalization_panels.py                               # composite figure
```
