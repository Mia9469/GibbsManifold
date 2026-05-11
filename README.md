# GibbsManifold

Code and analysis pipelines accompanying the manuscript:

> **Gibbs Manifold Geometry Unifies Neural Variability, Sparseness, and Metabolic Efficiency.**
> Mingyi Huang, Wei Lin, Anna Wang Roe, Yuguo Yu. *(submitted to Nature Communications)*.

This repository contains the analytical derivations, simulation scripts, and
data-analysis pipelines that generate every main and supplementary figure of
the paper.

---

## Repository layout

Folder names use the **source-code numbering** (`Fig1`–`Fig7`), which evolved
during development.  The mapping to the **manuscript figure numbering** is:

| Folder    | Generates                                                  | Manuscript                  |
| --------- | ---------------------------------------------------------- | --------------------------- |
| `Fig1/`   | Theory illustration: truncated-exponential ICE–CV curves   | Fig. 1C                     |
| `Fig2/`   | Macaque V1/V2 Direct-Method entropy + decoder analysis     | Fig. 2, Fig. S3, Fig. S11   |
| `Fig3/`   | Single-neuron LIF model with inhibitory STDP               | Fig. 3                      |
| `Fig4/`   | Mutual-information rates (frozen-stimulus paradigm)        | feeds Fig. 3 and Fig. 4     |
| `Fig5/`   | Recurrent E–I spiking network with plasticity              | Fig. 4                      |
| `Fig6/`   | Cross-system overlay on the entropy–cost plane             | Fig. 5, Fig. S7             |
| `Fig7/`   | CV-maximising sparse-coding learning rule                  | Fig. 6, Fig. S6, Fig. S8    |
| `updated/`| Manuscript LaTeX sources, bibliography, supplementary      | PDF builds                  |

Each `Fig*/` folder contains a short `README.md` documenting its main
entry-point script(s), required inputs, and produced outputs.

---

## Software requirements

| Language          | Version       | Used by                                     |
| ----------------- | ------------- | ------------------------------------------- |
| Python            | ≥ 3.10        | `Fig1/`, `Fig2/*.py`, `Fig6/`, `Fig7/`      |
| ↳ `numpy`, `scipy`, `matplotlib`            |                                |
| ↳ `scikit-learn`  | ≥ 1.3         | `Fig2/v1_decoder_vs_CV_multi.py` (LDA)      |
| ↳ `pytorch`       | ≥ 2.0         | `Fig7/train_new.py` (GPU-accelerated)       |
| MATLAB            | R2022a+       | `Fig2/*.m`, `Fig3/`, `Fig4/`, `Fig5/`       |
| LaTeX             | TeX Live 2023+ + Springer Nature `sn-jnl` class | `updated/`            |

Install Python deps:
```bash
pip install numpy scipy matplotlib scikit-learn torch
```

---

## Data availability

The macaque V1/V2 recordings analysed in `Fig2/` are publicly available
from the Collaborative Research in Computational Neuroscience (CRCNS)
data portal:

* **`v1v2-1` dataset**: <https://crcns.org/data-sets/vc/v1v2-1>

The five session `.mat` files used by `Fig2/` —

```
105l001p16.mat   106r001p26.mat   106r002p70.mat
107l002p67.mat   107l003p143.mat
```

— are **not** redistributed in this repository (they require CRCNS
account registration).  After registering, download these files into
`Fig2/` and the pipelines run as-is.

---

## Reproducing the figures

From the repository root:

```bash
# Fig. 1C  — theoretical ICE–CV manifold
python3 Fig1/Fig1c.py

# Fig. 2   — V1/V2 Direct-Method analysis (requires CRCNS .mat files)
matlab -batch "cd Fig2; fig2_stats"        # within-condition statistics
python3 Fig2/fig2_styled_panels.py         # paper-style panels (C, D)

# Supplementary Fig. S11 — FR-controlled per-neuron decoder (5-session)
python3 Fig2/v1_decoder_vs_CV_multi.py

# Fig. 3   — single-neuron LIF + inhibitory STDP
matlab -batch "cd Fig3; figure3"
matlab -batch "cd Fig3; figure3_HC_ICE"    # ICE / H–C trajectories

# Fig. 4   — recurrent E–I network
matlab -batch "cd Fig5; Figure5_new"
matlab -batch "cd Fig5; Learning_testentr" # learning-trajectory entropy

# Fig. 5   — cross-system entropy–cost overlay
python3 Fig6/fig6_dimensionless.py
python3 Fig6/figS8_anchor_sensitivity.py   # Supp Fig. S7 anchor sensitivity

# Fig. 6   — CV-max sparse coding
python3 Fig7/train_new.py                  # train all three models (GPU)
python3 Fig7/analyze_models.py             # per-method comparison + panels
bash    Fig7/p3_matched/p3_sweep_v2.sh     # matched-baseline ablation (Supp Fig. S8)
```

Each script writes its outputs (PDFs, JSON summaries, plot data) into
its own `Fig*/` folder.

---

## Citation

```bibtex
@article{HuangGibbsManifold2026,
  title   = {Gibbs Manifold Geometry Unifies Neural Variability,
             Sparseness, and Metabolic Efficiency},
  author  = {Huang, Mingyi and Lin, Wei and Roe, Anna Wang and Yu, Yuguo},
  year    = {2026},
  journal = {(under review at Nature Communications)}
}

@software{HuangGibbsManifoldCode2026,
  title     = {GibbsManifold: analysis and simulation code},
  author    = {Huang, Mingyi and Lin, Wei and Roe, Anna Wang and Yu, Yuguo},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.XXXXXXX}
}
```

A `CITATION.cff` file is provided at the repository root for GitHub's
"Cite this repository" widget.

---

## License

Source code is released under the **MIT License** (see `LICENSE`).
The CRCNS `v1v2-1` data are governed by the CRCNS data-sharing
agreement; please consult the CRCNS portal for those terms before
re-distributing any session file.

---

## Contact

* Analysis pipelines: **Mingyi Huang** — `myhuang20@fudan.edu.cn`
* Conceptual framework: **Yuguo Yu** — `yuyuguo@fudan.edu.cn`
