import numpy as np
import scipy.io as scio
import torch
import matplotlib as mpl
mpl.use('Agg')  # non-GUI backend for headless / Windows-server runs
import matplotlib.pyplot as plt
from basis_analysis import linear_independence, orthogonality, decorrelation
import seaborn as sns
from scipy.stats import ttest_ind
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from scipy.io import loadmat
import os

def convert_pvalue_to_asterisks(pvalue):
    if pvalue <= 0.0001:
        return "****"
    elif pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "ns"


# ── Configuration ──────────────────────────────────────────────────────
# Set USE_NEW = True after retraining with train_new.py
USE_NEW = True

# Set USE_CONSTRAINED = True to use the cv_constrained model (Opt3 budget=1.0)
# instead of the unconstrained cv model.
# Set USE_V2 = True to use cv_constrained_v2 (absolute-RE-target gate) which
# overrides USE_CONSTRAINED.  Train with:
#   python train_new.py --cons cv_constrained_v2 --recon-budget 0.6
#                       --cv_ratio 0.4 --reg 5e-3 --n_runs 20 --epoch 10
USE_CONSTRAINED = True
USE_V2 = True   # set True after running the v2 sweep above

if USE_NEW:
    if USE_V2:
        cv_dir = '_cv_v2_new'
    elif USE_CONSTRAINED:
        cv_dir = '_cv_con_new'
    else:
        cv_dir = '_cv_16_nl_new'
    dis = ['_16_new', '_cauchy_16_nl_new', cv_dir]
    test = ['null', 'cauchy', 'cv_m']
    run_start, run_end = 0, 20
    N_EPOCH = 10       # checkpoint saved at last epoch
else:
    dis = ['_16', '_cauchy_16_nl', '_cv_16_nl']
    test = ['null', 'cauchy', 'cv_m']
    run_start, run_end = 11, 31
    N_EPOCH = 100

# Local paths (relative to Fig7/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RECORD_DIR = os.path.join(BASE_DIR, 'record')
DATA_PATH = os.path.join(BASE_DIR, 'src', 'model', 'IMAGES.mat')

# Load natural images for SSIM computation
X = loadmat(DATA_PATH)
X = X['IMAGES']
img_size = X.shape[0]   # 512
n_img = X.shape[2]      # 10

# ── Main analysis ──────────────────────────────────────────────────────
rec = pd.DataFrame(columns=[
    "type", "CV", "CV_lt", "loss", "independence", "SSIM", "RE",
    "ICE", "ICE_ent", "cost"
])

# Accumulate per-neuron mean activations for histogram (legacy panel)
all_neuron_means = {t: [] for t in test}   # test = ['null', 'cauchy', 'cv_m']
# Accumulate POOLED |a| across (patch, neuron) for the Gibbs-exponential check.
# This is the quantity the framework predicts to be approximately exponential
# under linear cost; per-neuron mean (above) instead measures dead/alive
# neuron utilisation and is not the same statistic.
all_activations = {t: [] for t in test}

for n, typ in enumerate(dis):
    model_dir = os.path.join(RECORD_DIR, 'trained_models' + typ)

    for k in range(run_start, run_end):
        # Load training record
        record_path = os.path.join(RECORD_DIR, 'record' + typ + '_' + str(k) + '.mat')
        if not os.path.exists(record_path):
            print(f"[SKIP] record not found: {record_path}")
            continue
        record = scio.loadmat(record_path)

        # Accumulators across images (reset per run)
        sim_acc = 0.0
        re_acc = 0.0        # relative reconstruction error
        ent_acc = 0.0       # total entropy across images
        cost_acc = 0.0
        indep_acc = 0.0
        n_loaded = 0

        for j in range(n_img):
            img = X[:, :, j]

            # Load model checkpoint
            pth_path = os.path.join(model_dir, f'ckpt-{N_EPOCH}-{k}-{j+1}.pth')
            if not os.path.exists(pth_path):
                print(f"[SKIP] model not found: {pth_path}")
                continue

            sparse_net = torch.load(pth_path, map_location='cpu')
            U = sparse_net.U.weight.detach().cpu().numpy()   # [M², K]
            K_neurons = U.shape[1]  # number of neurons = K

            # ── Sparse coefficients ──
            # R: [n_patch, K]
            w = torch.abs(sparse_net.R).detach().numpy()
            mr = np.mean(w, axis=0)   # mean |activation| per neuron [K]
            all_neuron_means[test[n]].append(mr)  # shape [K]
            # Pool the full |a| matrix [n_patch x K] as a flat array for the
            # Gibbs-exponential check (joint distribution across neurons +
            # patches).  Memory is fine: n_patch x K ~ 1e3-1e4 per (image,run).
            all_activations[test[n]].append(w.ravel())

            # ── Entropy of activations (ICE numerator) ──
            # Compute entropy H(|a|) averaged across all K neurons
            ent_neurons = 0.0
            for v in range(K_neurons):
                hist_counts, _ = np.histogram(w[:, v], bins=200, range=(0, 2))
                prob = hist_counts / hist_counts.sum()
                prob = prob[prob > 0]
                ent_neurons += -np.sum(prob * np.log2(prob))
            ent_neurons /= K_neurons  # average entropy per neuron [bits]
            ent_acc += ent_neurons

            # ── Cost = mean activation ā  (Cost 3) ──
            cost_img = np.mean(mr)
            cost_acc += cost_img

            # ── Reconstruction & SSIM ──
            recon = sparse_net.U(sparse_net.R).detach().numpy()  # [n_patch, M²]
            n_patch, patch_size = recon.shape
            patch_w = int(np.sqrt(patch_size))  # 16

            img_rec = np.zeros([img_size, img_size])
            patches_per_row = img_size // patch_w
            for i in range(n_patch):
                patch = np.reshape(recon[i, :], [patch_w, patch_w])
                x_pos = (i % patches_per_row) * patch_w
                y_pos = (i // patches_per_row) * patch_w
                if x_pos + patch_w <= img_size and y_pos + patch_w <= img_size:
                    img_rec[x_pos:x_pos + patch_w, y_pos:y_pos + patch_w] = patch

            sim_val = ssim(img, img_rec, data_range=img_rec.max() - img_rec.min())
            sim_acc += sim_val

            # ── Relative reconstruction error RE = ||img - img_rec||₂ / ||img||₂ ──
            re_val = np.linalg.norm(img - img_rec) / (np.linalg.norm(img) + 1e-12)
            re_acc += re_val

            # ── Independence (decorrelation) ──
            dec = decorrelation(U)  # [K, K], values = 1 - |corr|
            indep_val = np.sum(dec) / (K_neurons * (K_neurons - 1)) * 100
            indep_acc += indep_val

            n_loaded += 1

        if n_loaded == 0:
            print(f"[WARN] No images loaded for type={typ}, run={k}")
            continue

        # ── Average across images ──
        sim_avg = sim_acc / n_loaded
        re_avg = re_acc / n_loaded
        ent_avg = ent_acc / n_loaded
        cost_avg = cost_acc / n_loaded
        indep_avg = indep_acc / n_loaded

        # ── ICE = -log₂(RE) / ā  [bits/activation] ──
        # Under Gaussian noise (high SNR): -log₂(RE) ≈ I(s; ŝ) in bits
        ice = -np.log2(re_avg + 1e-12) / (cost_avg + 1e-12)
        # H(a)/ā: entropy-based (supplementary reference, unchanged)
        ice_ent = ent_avg / (cost_avg + 1e-12)

        # ── Training metrics from record ──
        cv_train = np.mean(record['CV'][0][-10:])
        cv_lt_train = np.mean(record['CV_lt'][0][-10:]) if 'CV_lt' in record else np.nan
        loss_train = np.mean(record['E'][0][-10:]) / np.max(record['E'][0])

        # ── Store one row per (type, run) ──
        rec.loc[len(rec.index)] = [
            test[n],        # type
            cv_train,       # CV population (from training record)
            cv_lt_train,    # CV lifetime (from training record, supp)
            loss_train,     # loss (normalized convergence)
            indep_avg,      # independence (%)
            sim_avg,        # SSIM (averaged over images, kept for reference)
            re_avg,         # RE = relative reconstruction error (fraction)
            ice,            # ICE = -log₂(RE)/ā  [bits/activation]
            ice_ent,        # ICE_ent = H(a)/ā (reference)
            cost_avg,       # cost = mean activation ā
        ]

        print(f"[{test[n]}] run={k}: RE={re_avg:.4f}, SSIM={sim_avg:.4f}, "
              f"ICE(-log2RE/a_bar)={ice:.2f}, ICE(H/a_bar)={ice_ent:.2f}, "
              f"cost={cost_avg:.4f}, indep={indep_avg:.2f}%")


# ── Helpers ────────────────────────────────────────────────────────────

def pval_stars(p):
    if p <= 0.0001: return "****"
    if p <= 0.001:  return "***"
    if p <= 0.01:   return "**"
    if p <= 0.05:   return "*"
    return "ns"


def add_stat_annotation(ax, data, x, y, pairs, order):
    """Draw bracket + stars between each pair, compatible with any seaborn."""
    y_max = data[y].max()
    y_range = data[y].max() - data[y].min()
    h = 0.05 * y_range          # bracket height
    offset = 0.08 * y_range     # gap above tallest box

    pos_map = {name: i for i, name in enumerate(order)}

    for k, (g1, g2) in enumerate(pairs):
        v1 = data.loc[data[x] == g1, y].values
        v2 = data.loc[data[x] == g2, y].values
        if len(v1) < 2 or len(v2) < 2:
            continue
        _, p = ttest_ind(v1, v2)
        stars = pval_stars(p)

        x1, x2 = pos_map[g1], pos_map[g2]
        y_bar = y_max + offset + k * (h + offset)

        ax.plot([x1, x1, x2, x2],
                [y_bar, y_bar + h, y_bar + h, y_bar],
                lw=1.0, color='k')
        ax.text((x1 + x2) / 2, y_bar + h, stars,
                ha='center', va='bottom', fontsize=12)


# ── Plotting ───────────────────────────────────────────────────────────
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams["axes.labelsize"] = 18
palette_map = {'null': '#0073C2FF', 'cauchy': '#0073C2FF', 'cv_m': '#EFC000FF'}
type_order = ['null', 'cauchy', 'cv_m']

box_pairs = [
    ("null", "cauchy"),
    ("null", "cv_m"),
    ("cauchy", "cv_m")
]

# ── Summary table ──────────────────────────────────────────────────────

REPORT_COLS = {
    "RE":           "RE",
    "independence": "Indep (%)",
    "ICE":          "ICE (bits/act)",
    "CV":           "CV_P (train)",
    "CV_lt":        "CV_L (train)",
    "cost":         "mean |a|",
    "SSIM":         "SSIM",
}
label_map = {'null': 'Null', 'cauchy': 'SparseNet', 'cv_m': 'CVM'}
col_w = 18

print("\n" + "=" * 80)
print("  SUMMARY  (mean ± std across runs, n per group shown)")
print("=" * 80)
header = f"{'Model':<14}" + "".join(f"{v:>{col_w}}" for v in REPORT_COLS.values())
print(header)
print("-" * len(header))
for t in type_order:
    sub = rec[rec["type"] == t]
    row = f"{label_map[t]:<10}(n={len(sub):>2})"
    for col in REPORT_COLS:
        vals = sub[col].dropna()
        cell = f"{vals.mean():.3f}±{vals.std():.3f}" if len(vals) > 0 else "—"
        row += f"{cell:>{col_w}}"
    print(row)
print("-" * len(header))

print("\n  Pairwise Welch t-tests")
print(f"  {'Pair':<24}" + "".join(f"{v:>{col_w}}" for v in REPORT_COLS.values()))
for g1, g2 in box_pairs:
    pair_label = f"{label_map[g1]} vs {label_map[g2]}"
    row = f"  {pair_label:<24}"
    for col in REPORT_COLS:
        a = rec.loc[rec["type"] == g1, col].dropna().values
        b = rec.loc[rec["type"] == g2, col].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            from scipy.stats import ttest_ind
            _, p = ttest_ind(a, b, equal_var=False)
            row += f"{pval_stars(p):>{col_w}}"
        else:
            row += f"{'—':>{col_w}}"
    print(row)
print("=" * 80 + "\n")

# Custom y-axis labels (feature name used as fallback)
ylabels = {
    "ICE":          r"ICE $= -\log_2(\mathrm{RE})\,/\,\bar{a}$  (bits/activation)",
    "RE":           "Relative reconstruction error (RE)",
    "SSIM":         "SSIM",
    "CV":           r"Population $\mathrm{CV}_P$",
    "CV_lt":        r"Lifetime $\mathrm{CV}_L$",
    "cost":         r"Mean activation $\bar{a}$",
    "independence": r"Basis independence $\bar{D}_\phi$  (%)",
}

for i, feature in enumerate(["independence", "loss", "CV", "CV_lt",
                              "SSIM", "RE", "ICE", "ICE_ent", "cost"]):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100, facecolor="w")
    sns.boxplot(x="type", y=feature, data=rec, hue="type", order=type_order,
                palette=palette_map, legend=False, ax=ax)

    add_stat_annotation(ax, rec, "type", feature, box_pairs, type_order)

    if feature in ylabels:
        ax.set_ylabel(ylabels[feature])

    fig.tight_layout()
    out_path = os.path.join(BASE_DIR, feature + '_new.pdf')
    fig.savefig(out_path)
    print(f"Saved {out_path}")

# ── Pooled |a| distribution: Gibbs-exponential check (Fig 7C) ──────────
# Plots the density of |a| over all (patch, neuron) pairs on a log-y scale.
# A true linear-cost Gibbs solution is exponential, p(a) = lambda * exp(-lambda a),
# which appears as a straight line on log-y.  Comparing the three models on the
# same log-y density tests which one's coefficient distribution is closest to
# the exponential form predicted by the framework.
hist_colors = {'null': '#868686FF', 'cauchy': '#0073C2FF', 'cv_m': '#EFC000FF'}
hist_labels = {'null': 'Null', 'cauchy': 'Sparse', 'cv_m': 'CV-max'}

# Concatenate all activations per model type
all_a = {}
for t in type_order:
    if len(all_activations[t]) > 0:
        all_a[t] = np.concatenate(all_activations[t])

# Drop exact zeros from soft-thresholding (Sparse) so the surviving
# distribution can be compared on log-y; report the zero fraction
# separately so reviewers see the (small) censored mass.
zero_thresh = 1e-8
zero_frac = {t: float(np.mean(all_a[t] <= zero_thresh)) for t in all_a}
all_a_nz  = {t: all_a[t][all_a[t] > zero_thresh] for t in all_a}

# Shared upper limit at the 99.5th percentile across methods so heavy tails
# don't squash the visible bulk; bins linear from 0 to that.
upper_q = max(np.quantile(all_a_nz[t], 0.995) for t in all_a_nz)
bins = np.linspace(0, upper_q, 80)

fig_hist, ax_hist = plt.subplots(figsize=(7.0, 4.6), dpi=150, facecolor='w')
for t in type_order:
    if t not in all_a_nz:
        continue
    ax_hist.hist(all_a_nz[t], bins=bins, density=True, histtype='step',
                 lw=2.2, color=hist_colors[t],
                 label=f"{hist_labels[t]}  (zeros: {100*zero_frac[t]:.1f}%)")

ax_hist.set_yscale('log')
ax_hist.set_xlabel(r'$|a|$', fontsize=14, labelpad=8)
ax_hist.set_ylabel('Density  (log)', fontsize=14, labelpad=8)
ax_hist.set_title('Pooled coefficient distribution', fontsize=14, pad=12)
ax_hist.legend(fontsize=11, frameon=False, loc='upper right')

# Generous outside-axis padding (log-y: scale by factor; linear x: linear pad)
xpad = 0.05 * upper_q
ax_hist.set_xlim(left=-xpad, right=upper_q + xpad)
y_lo = min(ax_hist.get_ylim()[0], 5e-4)
y_hi = ax_hist.get_ylim()[1] * 1.6
ax_hist.set_ylim(y_lo, y_hi)
ax_hist.tick_params(axis='both', which='major', labelsize=11)
sns.despine(ax=ax_hist, offset=8)

# Annotation 1: CV-max super-exponential tail (yellow region, right side)
ax_hist.annotate(
    'CV-max: heavier tail',
    xy=(0.92, 0.025), xytext=(0.55, 0.6),
    fontsize=11, color='#B8860B',
    arrowprops=dict(arrowstyle='->', color='#B8860B', lw=1.2,
                    connectionstyle='arc3,rad=0.15'))

# Annotation 2: Sparse over-suppression (blue, far below the others in tail)
ax_hist.annotate(
    'SparseNet: Cauchy\nover-suppresses tail',
    xy=(0.85, 0.0028), xytext=(0.30, 0.0015),
    fontsize=11, color='#0073C2',
    arrowprops=dict(arrowstyle='->', color='#0073C2', lw=1.2,
                    connectionstyle='arc3,rad=-0.15'))

# Annotation 3: Null = baseline (left/middle, where it tracks bulk)
ax_hist.annotate(
    'Null: L1 baseline\n(near-exponential)',
    xy=(0.32, 1.0), xytext=(0.45, 4.0),
    fontsize=11, color='0.35',
    arrowprops=dict(arrowstyle='->', color='0.35', lw=1.0,
                    connectionstyle='arc3,rad=-0.1'))

fig_hist.tight_layout()

hist_path = os.path.join(BASE_DIR, 'activation_histogram.pdf')
fig_hist.savefig(hist_path)
print(f"Saved {hist_path}")
print('  zero-coefficient fractions:',
      ', '.join(f'{hist_labels[t]}={100*zero_frac[t]:.2f}%' for t in zero_frac))
print('  pooled |a| means:',
      ', '.join(f'{hist_labels[t]}={np.mean(all_a_nz[t]):.4f}' for t in all_a_nz))

# Legacy per-neuron mean histogram retained as a separate file for reference;
# no longer the main-text Fig 7C panel.
fig_pn, ax_pn = plt.subplots(figsize=(6.0, 4.0), dpi=150, facecolor='w')
all_mr = {t: np.concatenate(all_neuron_means[t]) for t in type_order
          if len(all_neuron_means[t]) > 0}
if all_mr:
    pn_max = max(np.max(v) for v in all_mr.values())
    pn_bins = np.linspace(0, pn_max * 1.05, 60)
    for t in type_order:
        if t in all_mr:
            ax_pn.hist(all_mr[t], bins=pn_bins, density=True, alpha=0.55,
                       color=hist_colors[t], label=hist_labels[t], edgecolor='none')
    ax_pn.set_xlabel('Mean |activation| per neuron', fontsize=13)
    ax_pn.set_ylabel('Density', fontsize=13)
    ax_pn.set_title('Per-neuron mean activation (legacy panel; not used in main text)',
                    fontsize=11)
    ax_pn.legend(fontsize=11, frameon=False)
    ax_pn.set_xlim(left=0)
    sns.despine(ax=ax_pn)
fig_pn.tight_layout()
pn_path = os.path.join(BASE_DIR, 'activation_perneuron_mean.pdf')
fig_pn.savefig(pn_path)
print(f"Saved {pn_path}  (legacy)")

# plt.show()  # disabled: Agg backend; all figures already saved to disk
