# Fig 5 — Corrected information–proxy-cost projection

Generates **manuscript Fig. 5** (corrected V1 and LIF projections), **manuscript Fig. 6** (the
distribution-level candidate specification test), **Supp Fig. S7**
(R₀ anchor-sensitivity analysis for V1), and **Supp Fig. S18**
(frontier-shape and permutation controls).

## Composite assembly

The two-column main figure is rebuilt directly from corrected caches:

| Column                | Manuscript panel | Source folder | Produced by                                        |
| --------------------- | ---------------- | ------------- | -------------------------------------------------- |
| Single-neuron LIF     | Fig. 5A          | `Fig3/`       | `MI_results_direct_corrected.mat` |
| Macaque V1            | Fig. 5B          | `Fig2/`       | `unified_v1_Hrho_ICE.mat` and `spon_Hr_perN.mat` |

The recurrent-network information result is deliberately not added as a third
Fig. 5 column because it is a population decoder/Fano lower bound, whereas both
Fig. 5 columns use single-unit matched-word Direct-Method information. The
corrected recurrent result is reported in Fig. 4J and its Supplementary audit;
this separation avoids comparing unlike estimands on one calibrated curve.

## Entry points

| Script                          | Produces                                                |
| ------------------------------- | ------------------------------------------------------- |
| `make_corrected_figure5.py`     | Main Fig. 5 from corrected V1 and LIF caches |
| `figS7_anchor_sensitivity.py`   | Supp Fig. S7: V1 frontier under min / median / max-rate anchors |
| `figS20_frontier_model_controls.py` | Supp Fig. S18: leave-one-session-out curve controls and permutation nulls (historical script name retained) |
| `discriminative_maxent_test.py` | Main Fig. 6: continuous candidate test plus assembly of native-count and decoder panels |
| `discrete_count_sensitivity.py` | Native-count observation-model and QC sensitivity with lognormal, gamma, and negative-binomial controls |
| `shared_shape_count_loso.py`    | Native-count leave-one-session-out shared-shape transfer, matched nuisance profiling and model recovery |
| `cost_proxy_sensitivity.py`     | Corrected V1/LIF/recurrent-population cost-coefficient grid and alternative-cost sensitivity |

```bash
python3 make_corrected_figure5.py
python3 figS7_anchor_sensitivity.py
python3 figS20_frontier_model_controls.py
python3 discriminative_maxent_test.py
python3 discrete_count_sensitivity.py --splits 4 --qc both
python3 shared_shape_count_loso.py --splits 8
python3 cost_proxy_sensitivity.py
```

## Method

Corrected `(cost proxy, I((S,J);W), H)` summaries are read from the V1
and LIF `.mat` caches. Each display column is calibrated by one scalar
**R₀**; it is a plotting scale, not a measured observation rate:

* Single neuron — `R₀ = 3.11` (minimum-rate anchor)
* V1 — `R₀ = 2.60` (**median-rate anchor**: scatter spans the full
  cost axis, min-rate point is dominated by low-FR outliers)

The Gibbs frontier overlay is

```
H*(C) = (R₀ / ln 2) · [1 + ln((C − 2.4) / 0.37)]   [bits/s]
```

with calibrated proxy coefficients `C₀ = 2.4`, `C₁ = 0.37` adapted
from Yu et al. (2018). These coefficients are not assigned per-neuron
watt units; the overlay is interpreted only within each system.

The Fig. 6 distribution-level test uses the same five-session V1 pool but
operates on per-neuron spike-count distributions rather than the
information--rate frontier. The continuous analysis compares a mean-scaled
maximum-entropy family `p(r) ∝ exp(-β r^α)` with lognormal and equal-dimensional
gamma alternatives by held-out per-trial log likelihood. The decisive
native-count analysis learns one response-shape parameter per candidate from
four sessions, transfers it unchanged to the fifth, profiles one neuron-level
nuisance parameter for every family on calibration trials, and scores the
disjoint trials. Power-cost exceeds lognormal in 5/5 spontaneous and 3/5 driven
sessions, but gamma and negative binomial score higher on average; negative
binomial is best in all ten condition-by-session folds. Recovery simulations
select the generating family in 85--100% of cases. The result supports a
transferable power-cost-versus-lognormal contrast without uniquely identifying
the Gibbs response law over the complete candidate set.

## Outputs

* `updated/figures/GIBBSMANIFOLD_FIGS/Figure5.pdf` — corrected two-column main figure
* `figure5_corrected_anchors.json` — exact display anchors and sample counts
* Anchor-sensitivity panels for Supp Fig. S7
* `discriminative_maxent_test.json` and `updated/figures/Figure_discriminative.pdf` — main Fig. 6
* `discrete_count_sensitivity.json` — native-count/QC sensitivity for Table S9
* `shared_shape_count_loso.{json,npz}` and `updated/figures/Figure_shared_shape_count_loso.pdf` — shared-shape native-count transfer and recovery
* `cost_proxy_sensitivity.json` and `updated/figures/Figure_cost_proxy_sensitivity.pdf` — 13-by-13 coefficient scan and alternative-cost controls for V1, LIF, and the recurrent-population decoder bound

## Dependencies

Python 3.9.6 core environment from `../environment.yml`.
