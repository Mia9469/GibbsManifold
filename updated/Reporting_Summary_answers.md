# Nature Portfolio Reporting Summary — Drafted Answers

**Manuscript:** *Gibbs Manifold Geometry Unifies Neural Variability, Sparseness, and Metabolic Efficiency*
**Authors:** Huang, Lin, Roe, Yu

> **How to use this file**
> Download the *Reporting Summary* PDF from the Nature Communications
> submission system, open it in Acrobat or Preview's form-editing mode,
> and copy each block below into the corresponding field. The headings
> mirror the official template (October 2025 version) so the mapping
> is one-to-one. Items marked **[Confirm]** request a direct confirmation
> from the corresponding author before submission.

---

## 1. Statistics

> *For all statistical analyses, confirm that the following items are present
> in the figure legend, table legend, main text, or Methods section.*

| Item | Status | Where in manuscript |
|------|:------:|---------------------|
| The exact sample size (n) for each experimental group/condition, given as a discrete number and unit of measurement | ☑ | Fig 2 caption (N=159 neurons), Fig S11 caption (N=564 across 5 sessions), Fig 5/6 captions (20 simulations/seeds × 10 epochs), Supp Table S6 |
| A statement on whether measurements were taken from distinct samples or whether the same sample was measured repeatedly | ☑ | "paired across the same N=159 neurons" (Fig 2C); "20 independent simulations" (Fig 4); "20 random seeds" (Fig 6) |
| The statistical test(s) used AND whether they are one- or two-sided. Only common tests should be described solely by name; describe more complex techniques in the Methods section | ☑ | Wilcoxon signed-rank (paired, two-sided), Cliff's δ (effect size), Spearman correlation (raw + partial, two-sided), Fisher z meta-combination, two-sample Welch t-test (two-sided), 10-fold stratified cross-validation. All described in Methods §3.7 and individual figure captions |
| A description of all covariates tested | ☑ | Firing rate (FR) is the covariate in the partial-correlation analyses for Fig 4 and Supp Fig S11 |
| A description of any assumptions or corrections, such as tests of normality and adjustment for multiple comparisons | ☑ | Non-parametric throughout (Spearman, Wilcoxon, Cliff's δ — no normality assumption); 10,000-resample bootstrap for CIs; no multiple-comparison correction beyond per-figure pairwise checks at p < 10⁻⁴ |
| A full description of the statistical parameters including central tendency (e.g. means) or other basic estimates (e.g. regression coefficient) AND variation (e.g. standard deviation) or associated estimates of uncertainty (e.g. confidence intervals) | ☑ | Medians + IQR throughout; bootstrap 95% CIs on paired-difference medians (Supp Table S6); per-seed mean ± SD for sparse-coding comparisons; Spearman ρ with two-sided p-values |
| For null hypothesis testing, the test statistic (e.g. F, t, r) with confidence intervals, effect sizes, degrees of freedom and P value noted. *Give P values as exact values whenever suitable.* | ☑ | All reported (e.g. Fig S11: ρ_partial = +0.68, p = 1.2×10⁻⁷⁸, n = 564; Fisher-combined ρ = +0.72, p = 4.6×10⁻¹⁰⁰; per-session ranges +0.54 to +0.83, all p < 10⁻⁹). Exact p-values given where they exceed numerical floor; otherwise reported as "p < 10⁻ᵏ" |
| For Bayesian analysis, information on the choice of priors and Markov chain Monte Carlo settings | N/A | No Bayesian analyses |
| For hierarchical and complex designs, identification of the appropriate level for tests and full reporting of outcomes | ☑ | Two-level structure (within-session, across-sessions) is explicitly handled in Supp Fig S11: per-session partial Spearman + Fisher-z meta-combination + pooled-N analysis. Both within-FR-quartile and partial-correlation FR controls are reported |
| Estimates of effect sizes (e.g. Cohen's d, Pearson's r), indicating how they were calculated | ☑ | Cliff's δ for all paired V1 spontaneous-vs-evoked comparisons (Supp Table S6); Spearman ρ for monotonic associations; rank-residual partial Spearman for FR-controlled associations |

**Software used for statistical analysis:**
Python ≥ 3.10 with SciPy ≥ 1.10 (`scipy.stats.spearmanr`, `scipy.stats.wilcoxon`, `scipy.stats.norm`, `scipy.stats.t`),
NumPy ≥ 1.25 (rank residualisation, bootstrap),
scikit-learn ≥ 1.3 (`LinearDiscriminantAnalysis`, `StratifiedKFold`, `cross_val_predict`),
MATLAB R2022a (Statistics and Machine Learning Toolbox — bootstrap and Wilcoxon for the V1 pipeline).

---

## 2. Software and code

> *Policy information about availability of computer code.*

**Data collection:**
No data were collected by the authors. Macaque V1/V2 recordings are
publicly available from the CRCNS data-sharing portal: `v1v2-1` dataset
(https://crcns.org/data-sets/vc/v1v2-1), used by permission of the
original investigators.

**Data analysis:**
All custom code is in the GitHub repository
[https://github.com/Mia9469/GibbsManifold](https://github.com/Mia9469/GibbsManifold) (private during peer review;
access granted to editors and reviewers via the URL in the cover letter).

* **Python ≥ 3.10** with NumPy, SciPy, Matplotlib, scikit-learn, PyTorch ≥ 2.0
* **MATLAB R2022a** with the Statistics and Machine Learning Toolbox
* **LaTeX** (TeX Live 2023+) with the Springer Nature `sn-jnl` class

The repository is organised by figure (`Fig1/`–`Fig7/`); a top-level `README.md`
maps folders to manuscript figures, and each subfolder has its own
`README.md` listing entry-point scripts.

Random seed `20260502` is used for the multi-session decoder analysis
(`Fig2/v1_decoder_vs_CV_multi.py`); per-seed seeds for the sparse-coding
training are stored in `Fig7/runs_*` records.

---

## 3. Data

> *Policy information about availability of data.*

**All manuscripts must include a data availability statement.** This statement
appears in the back matter of the main text (Code & Data Availability section)
and reads:

> *"The datasets analysed in this study are publicly available from the
> Collaborative Research in Computational Neuroscience (CRCNS)
> data-sharing repository at* `https://crcns.org/data-sets/vc/v1v2-1`*. The
> specific dataset used is the V1/V2 dataset (v1v2-1)."*

Supplementary Section 3.2 additionally reports cross-dataset validation
using the CRCNS `pvc-11` dataset (Kohn lab) at
`https://crcns.org/data-sets/vc/pvc-11`.

Source Data underlying every quantitative panel of every figure are
provided as an Excel workbook (`SourceData.xlsx`) accompanying this
submission, with one sheet per figure.

---

## 4. Research involving human participants, their data, or biological material

| Item | Status |
|------|:------:|
| Population characteristics | N/A — no human participants |
| Recruitment | N/A |
| Ethics oversight | N/A |
| Note that full information on the approval of the study protocol must also be provided in the manuscript. | — |

---

## 5. Field-specific reporting

**Please select the one below that is the best fit for your research.**
☑ **Life sciences**
☐ Behavioural & social sciences
☐ Ecological, evolutionary & environmental sciences

### Life-sciences study design

> *All studies must disclose on these points even when the disclosure is negative.*

**Sample size**
The sample sizes are determined by dataset availability rather than
*a priori* power calculation. The macaque V1 dataset comprises all
N=159 well-isolated single units from the published CRCNS v1v2-1
session 105l001p16 (used in main-text Fig 2); the multi-session
extension pools N=564 V1 neurons across the five v1v2-1 sessions
(Supp Fig S11). For simulations, 20 independent runs (random seeds)
per condition with 10 epochs each were used, sufficient to achieve
two-sample Welch *t*-test significance at p < 10⁻⁴ for all reported
pairwise comparisons.

**Data exclusions**
No neurons or trials were excluded. The Direct-Method noise-entropy
pipeline reports rho across **all** V1 units (no SNR or firing-rate
filter applied), consistent with the supplementary Section 3.2
within-/cross-dataset robustness analyses.

**Replication**
Replication is built into the experimental design:
(i) Per-neuron decoder accuracy is evaluated via 10-fold stratified
cross-validation;
(ii) The multi-session analysis replicates the within-session
partial-correlation result independently in each of five V1 sessions
(all *p* < 10⁻⁹) and combines them by Fisher z-transform;
(iii) Sparse-coding comparisons use 20 independent random seeds per
method; (iv) Recurrent-network E–I learning is repeated across 20
simulations with independent initialisations.
All replication attempts yielded the expected qualitative direction
(positive partial ρ for CV–accuracy, increasing ICE during learning,
etc.) — no failed replications were excluded.

**Randomization**
For the per-neuron decoder analysis, stratified 10-fold cross-validation
randomly partitions trials within each stimulus class
(`sklearn.model_selection.StratifiedKFold`, random_state=20260502); the
fold assignment is fixed across all 564 neurons so that comparisons
across neurons share the same partition. For sparse-coding training,
random seeds are independently drawn per run (20 seeds × 10 epochs).
Shuffle null analyses (Supp Section 3.2) use independently drawn
circular-shift offsets per trial per neuron, with a fixed master seed
for reproducibility.

**Blinding**
Blinding is not applicable: all reported analyses are deterministic
given the raw spike data and code, and there is no investigator-driven
condition assignment. The CRCNS recordings were collected by independent
investigators (Kohn lab) and were not generated by the authors of this
study.

---

## 6. Reporting for specific materials, systems and methods

> *We require information from authors about some types of materials, experimental
> systems and methods used in many studies. Where applicable, we have asked you to
> provide more information.*

### Materials & experimental systems

| Item | Used in this study? |
|------|:--:|
| Antibodies | ☐ No |
| Eukaryotic cell lines | ☐ No |
| Palaeontology and archaeology | ☐ No |
| Animals and other organisms | ☐ No (we re-analyse publicly available macaque data; no new animal experiments) |
| Clinical data | ☐ No |
| Dual use research of concern | ☐ No |
| Plants | ☐ No |

### Methods

| Item | Used in this study? |
|------|:--:|
| ChIP-seq | ☐ No |
| Flow cytometry | ☐ No |
| MRI-based neuroimaging | ☐ No |

---

## 7. Use of generative AI

Language polishing of an earlier draft was assisted by a large language
model (**Claude, Anthropic**), as declared in the Acknowledgements of
the main manuscript. The LLM was used only for sentence-level
proofreading and clarity edits; all scientific content, derivations,
code, statistical analyses, and conclusions were written, executed,
and verified by the authors.

---

## End of Reporting Summary

> Please copy each block above into the corresponding fillable field of
> the official Reporting Summary PDF. Items marked N/A do not need
> further entry. Save the completed PDF and upload it alongside the
> manuscript, supplementary information, source data, and cover letter
> in the Nature Communications submission system.
