"""Redraw S6 from all final-epoch checkpoints in the frozen headline cohort."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.stats import ttest_ind


HERE = Path(__file__).resolve().parent
GROUPS = {"Null": ("_16_new", "non"), "SparseNet": ("_cauchy_16_nl_new", "s"),
          "CVM": ("_cv_v2_new", "cv_constrained_v2")}
METRICS = ("CV_P", "CV_L", "mean_activation")
RUNS = 20
IMAGES = 10
EPOCH = 10


def finite(value, description):
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite {description}")
    return value


def checkpoint_metrics(model):
    r = model.R.detach().cpu()
    u = model.U.weight.detach().cpu()
    if r.shape != (1024, 144) or u.shape != (256, 144):
        raise ValueError(f"Unexpected coefficient/dictionary shapes: {r.shape}, {u.shape}")
    if not torch.isfinite(r).all() or not torch.isfinite(u).all():
        raise ValueError("Non-finite checkpoint coefficients or dictionary")
    with torch.no_grad():
        # Use the trained model's sample-SD convention and its 1e-8 stabilizer.
        values = [model._population_cv(r).item(), model._lifetime_cv(r).item(),
                  r.abs().numpy().mean(axis=0).mean().item()]
    return finite(np.asarray(values), "checkpoint metrics")


def collect(record_dir):
    rows, provenance = [], []
    with (HERE / "sparse_run_statistics.json").open() as stream:
        frozen = json.load(stream)["groups"]
    for label, (suffix, constraint) in GROUPS.items():
        legacy_values = []
        for run in range(RUNS):
            record_path = record_dir / f"record{suffix}_{run}.mat"
            record = loadmat(record_path)
            history = np.stack([np.asarray(record[key]).ravel()
                                for key in ("CV", "CV_lt", "mu")], axis=1)
            if history.shape != (EPOCH, 3):
                raise ValueError(f"Incomplete training history: {record_path}")
            finite(history, str(record_path))
            values = []
            for image in range(1, IMAGES + 1):
                path = record_dir / f"trained_models{suffix}" / f"ckpt-{EPOCH}-{run}-{image}.pth"
                # These are repository-owned full-model checkpoints, not untrusted uploads.
                try:
                    model = torch.load(path, map_location="cpu", weights_only=False)
                except OSError as exc:
                    raise OSError(f"Could not read checkpoint {path}; ensure its iCloud download has completed") from exc
                if model.cons != constraint or not np.isclose(model.lmda, 0.005):
                    raise ValueError(f"Wrong configuration: {path}")
                if label == "CVM" and not (np.isclose(model.recon_budget, 0.6)
                                           and np.isclose(model.cv_ratio, 0.4)):
                    raise ValueError(f"Wrong CVM v2 gate/weight: {path}")
                try:
                    values.append(checkpoint_metrics(model))
                except ValueError as exc:
                    raise ValueError(f"{path}: {exc}") from exc
                provenance.append({"model": label, "run": run, "image": image,
                                   "path": str(path.relative_to(HERE.parent)),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            endpoint = np.mean(values, axis=0)
            np.testing.assert_allclose(endpoint, history[-1], rtol=1e-6, atol=1e-7,
                                       err_msg=f"Checkpoint/history mismatch: {label}, run {run}")
            rows.append({"model": label, "run": run, "epoch": EPOCH, "n_images": IMAGES,
                         **dict(zip(METRICS, endpoint))})
            legacy_values.append([*history[:, :2].mean(axis=0), endpoint[2]])
        legacy_values = np.asarray(legacy_values)
        for i, metric in enumerate(("CV", "CV_lt", "cost")):
            for key, actual in (("mean", legacy_values[:, i].mean()),
                                ("sd", legacy_values[:, i].std(ddof=1))):
                np.testing.assert_allclose(actual, frozen[label][metric][key], rtol=1e-5,
                                           atol=1e-7, err_msg=f"Frozen cohort mismatch: {label}/{metric}/{key}")
        print(f"{label}: 20 runs x 10 checkpoints; frozen-cohort validation passed", flush=True)
    return rows, provenance


def summarize(rows, provenance):
    summary = {"unit": "training run", "epoch": EPOCH, "runs_per_model": RUNS,
               "images_per_run": IMAGES, "estimator": "all metrics from final-epoch coefficients; equal image average within run",
               "cv_convention": "sample standard deviation (ddof=1), mean stabilizer 1e-8; same as SparseNetNew",
               "cohort_validation": "training-average CV and endpoint activation match sparse_run_statistics.json means and SDs",
               "groups": {}, "pairwise": {}, "checkpoints": provenance}
    for label in GROUPS:
        summary["groups"][label] = {}
        for metric in METRICS:
            x = np.array([row[metric] for row in rows if row["model"] == label])
            summary["groups"][label][metric] = {"mean": x.mean(), "sd": x.std(ddof=1), "median": np.median(x)}
    for metric in METRICS:
        summary["pairwise"][metric] = []
        for a, b in combinations(GROUPS, 2):
            x = [row[metric] for row in rows if row["model"] == a]
            y = [row[metric] for row in rows if row["model"] == b]
            test = ttest_ind(x, y, equal_var=False)
            finite(np.array([test.statistic, test.pvalue]), f"Welch test {metric}/{a}/{b}")
            summary["pairwise"][metric].append({"a": a, "b": b, "t": test.statistic,
                                                 "p_raw": test.pvalue, "p_bonferroni": min(1., 3 * test.pvalue)})
    return summary


def draw(rows, destination):
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
                         "font.size": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(5.8, 1.85))
    labels = (r"$\mathrm{CV}_P$", r"$\mathrm{CV}_L$", r"Mean activation $\bar{a}$")
    rng = np.random.default_rng(20260904)
    offsets = rng.uniform(-0.12, 0.12, size=(len(GROUPS), RUNS))
    for panel, (ax, metric, ylabel) in enumerate(zip(axes, METRICS, labels)):
        values = [[row[metric] for row in rows if row["model"] == label] for label in GROUPS]
        ax.boxplot(values, widths=0.52, showfliers=False, patch_artist=True,
                   boxprops={"facecolor": "white", "edgecolor": "#444444", "linewidth": 0.8},
                   medianprops={"color": "black", "linewidth": 1.1},
                   whiskerprops={"color": "#444444", "linewidth": 0.8},
                   capprops={"color": "#444444", "linewidth": 0.8})
        for i, x in enumerate(values):
            ax.scatter(i + 1 + offsets[i], x, s=9, color="#8c8c8c", alpha=0.75,
                       edgecolors="none", zorder=3)
        ax.set_xticks(range(1, 4), list(GROUPS), fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9, labelpad=3)
        ax.tick_params(length=2.5, width=0.7, labelsize=7, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.15)
        ax.text(-0.30, 1.16, chr(65 + panel), transform=ax.transAxes, fontsize=12, va="top")
    fig.subplots_adjust(left=0.095, right=0.99, bottom=0.20, top=0.86, wspace=0.60)
    fig.savefig(destination)
    fig.savefig(destination.with_suffix(".png"), dpi=250)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-dir", type=Path, default=HERE / "record")
    parser.add_argument("--out-dir", type=Path, default=HERE.parent / "outputs/supp_revision_20260904/S6")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows, provenance = collect(args.record_dir)
    summary = summarize(rows, provenance)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "FigureS6_v2_run_data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.out_dir / "FigureS6_v2_statistics.json").open("w") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
    draw(rows, args.out_dir / "FigureS6_v2.pdf")
    print(json.dumps({key: summary[key] for key in ("groups", "pairwise")}, indent=2))


if __name__ == "__main__":
    main()
