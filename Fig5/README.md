# Fig 5 — Cross-system entropy–cost frontier (composite figure)

Generates **manuscript Fig. 5** (frontier-like organisation across the
three systems on the entropy–cost plane) and **Supp Fig. S7** (R₀
anchor-sensitivity analysis for V1).

## Composite assembly

The three-column main figure is **assembled** from per-system panels
produced by the upstream figure folders:

| Column                | Manuscript panel | Source folder | Produced by                                        |
| --------------------- | ---------------- | ------------- | -------------------------------------------------- |
| Single-neuron LIF     | Fig. 5A          | `Fig3/`       | `figure3_main.m` → `figure5_singleneuron.fig` (in this folder) |
| Recurrent network     | Fig. 5B          | `Fig4/`       | `figure4_main.m` → `figure5_network.fig` (in this folder; spatiotemporal points only) |
| Macaque V1            | Fig. 5C          | `Fig2/`       | `unified_v1_Hrho_ICE.m` (in `Fig2/`) writes V1 (C, I, H, ICE) for the cross-system overlay |

The `.fig` files in this folder are the exported panels, kept here so
the composite can be re-assembled without re-running every upstream
simulation.

## Entry points

| Script                          | Produces                                                |
| ------------------------------- | ------------------------------------------------------- |
| `figS7_anchor_sensitivity.py`   | Supp Fig. S7: V1 frontier under min / median / max-rate anchors |

```bash
python3 figS7_anchor_sensitivity.py
```

## Method

Per-system summary statistics — `(C, I, H, ICE)` along the learning
trajectory (`Fig3/`, `Fig4/`) or per-neuron operating points (`Fig2/`)
— are read from MATLAB / JSON / `.mat` summaries written by the
upstream pipelines.  Each system is calibrated by a single scalar
**R₀** (rate of independent observations per second):

* Single neuron — `R₀ = 1.63` (min-rate anchor; tight trajectory)
* Recurrent network — `R₀ = 1.0` (min-rate anchor)
* V1 — `R₀ = 2.58` (**median-rate anchor**: scatter spans the full
  cost axis, min-rate point is dominated by low-FR outliers)

The Gibbs frontier overlay is

```
H*(C) = (R₀ / ln 2) · [1 + ln((C − 2.4) / 0.37)]   [bits/s]
```

with `C₀ = 2.4 W`, `C₁ = 0.37 W/Hz` (Yu et al. 2018 metabolic cost).

## Outputs

* `figure5_singleneuron.fig` — Fig. 5A panel (from `Fig3/`)
* `figure5_network.fig` — Fig. 5B panel (from `Fig4/`; spatiotemporal points)
* V1 frontier overlay — written by `Fig2/unified_v1_Hrho_ICE.m`
* Anchor-sensitivity panels for Supp Fig. S7

## Dependencies

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`.
