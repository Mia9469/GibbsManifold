# Fig 1 — Theoretical ICE–CV manifold under linear cost

Generates the **manuscript Fig. 1C(iii)** (Gibbs-optimal ICE as a
function of CV under bounded linear cost) and the **Fig. 3G**
Gaussian-channel reference overlay.

## Entry point

```bash
python3 Fig1c.py
```

The script:
1. Sweeps the dimensionless tightness parameter `k = β·r_max` over a
   logarithmic grid.
2. Evaluates the closed-form moments of the truncated exponential
   (Supp. Eqs. 8–13) and the resulting `(CV(k), ICE(k))` curve.
3. Overlays the uniform (`CV ≈ 0.577`) and exponential (`CV = 1`)
   reference limits.

## Outputs

* `Figure1C_panels.pdf` / `.png` — Fig. 1C panels.
* `Figure4G_reference.pdf` / `.png` — reference curve overlaid in main
  Fig. 3G (single-neuron ICE–CV plane).

## Dependencies

`numpy`, `scipy`, `matplotlib` (Python ≥ 3.10).
