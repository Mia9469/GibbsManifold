# Fig 6 — Cross-system overlay on the entropy–cost plane

Generates **manuscript Fig. 5** (frontier-like organisation across
the three systems on the entropy–cost plane) and **Supp Fig. S7**
(R₀ anchor-sensitivity analysis for V1).

## Entry points

| Script                          | Produces                                                |
| ------------------------------- | ------------------------------------------------------- |
| `fig6_dimensionless.py`         | Main Fig. 5: I–C and ICE–C panels for single neuron / recurrent network / V1, with per-system R₀ calibration and the Gibbs frontier overlay |
| `figS8_anchor_sensitivity.py`   | Supp Fig. S7: V1 frontier under min / median / max-rate anchors |

Typical workflow:

```bash
python3 fig6_dimensionless.py        # writes Figure6A_dimensionless.{pdf,png}
python3 figS8_anchor_sensitivity.py
```

## Method

Per-system summary statistics — `(C, I, H, ICE)` values along the
learning trajectory (Fig. 3, Fig. 5) or per-neuron operating points
(Fig. 2) — are read from JSON / `.mat` summaries written by the
per-system pipelines.  Each system is calibrated by a single scalar
**R₀** (rate of independent observations per second):

* Single neuron — `R₀ = 1.63` (min-rate anchor; tight trajectory)
* Recurrent network — `R₀ = 1.0` (min-rate anchor)
* V1 — `R₀ = 2.58` (**median-rate anchor**: scatter spans full cost
  axis, min-rate point is dominated by low-FR outliers)

The Gibbs frontier overlay is

```
H*(C) = (R₀ / ln 2) · [1 + ln((C − 2.4) / 0.37)]   [bits/s]
```

with `C₀ = 2.4 W`, `C₁ = 0.37 W/Hz` from the metabolic cost function
(Yu et al. 2018).

## Outputs

* `Figure6A_dimensionless.pdf` / `.png`
* Per-anchor sensitivity panels for Supp Fig. S7

## Dependencies

Python ≥ 3.10 with `numpy`, `scipy`, `matplotlib`.
