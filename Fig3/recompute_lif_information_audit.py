"""Single-neuron cached-count audit; checkpoints contain DIFFERENT input segments.

The primary observation is fixed at 20 ms, before examining decoder outcomes.
Nothing is retrained or written to the MATLAB caches or author artwork.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np
from scipy.io import loadmat
from scipy.special import logsumexp, xlogy
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "Fig3" / "MI_results_direct_corrected.mat"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "final_text_review_20260907" / "lif_corrected"
PRIMARY_WINDOW_MS = 20
WINDOWS_MS = np.array([10, 20, 40, 50, 100])
SEGMENT_MS = 6000
C0, C1 = 2.4, 0.37
SEED = 20260907
N_WORD_NULLS = 20
N_DECODER_NULLS = 20
WORD_LENGTHS = np.array([20, 22, 25, 28, 30, 40, 50, 60, 70, 80, 90, 100])
FIT_RANGES = ((20, 40), (20, 100), (40, 100), (60, 100))
# Candidate zero is the uniform posterior. Ties select it conservatively.
CALIBRATION = [(None, None)] + [
    (strength, temperature)
    for strength in (0.0, 1.0, 4.0, 16.0, 64.0, 256.0)
    for temperature in (0.5, 1.0, 2.0, 4.0)
]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(handle.read()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def preserve_outputs(output_dir: Path) -> None:
    existing = [p for p in output_dir.iterdir() if p.is_file()]
    if existing:
        backup = output_dir / "backups"
        backup.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        with tarfile.open(backup / f"before_rerun_{stamp}.tar.gz", "w:gz") as archive:
            for path in existing:
                archive.add(path, arcname=path.name)


def specification(run_id: str) -> dict:
    return {
        "run_id": run_id,
        "primary_window_ms": PRIMARY_WINDOW_MS,
        "window_sensitivity_ms": WINDOWS_MS.tolist(),
        "primary_window_selection": "20 ms fixed a priori for this audit, matching taumem; never selected by direction or significance",
        "response": "ONE output neuron's integer spike count X in a non-overlapping (jL,(j+1)L] interval",
        "classes": "(template S, relative segment-bin J), uniform prior, K = 20 * (6000 / L_ms)",
        "cv": "CV_X = population SD / mean of counts over the same uniform (S,J,repeat) observation measure, including zero-count windows; ddof=0",
        "rate": "Total original spikes / (20 templates * 10 repeats * 6 seconds), not inverse-ISI event rates",
        "cost": {"C0": C0, "C1": C1, "definition": "2.4 + 0.37 * physical_count_rate_Hz"},
        "outer_cv": "Five fixed folds, test repeat pairs (1,2), (3,4), ..., (9,10); eight complete training repeats per template",
        "inner_cv": "Within the eight outer-training repeats, four leave-two-repeat-out folds; six fitting repeats, two calibration repeats; outer test repeats never used",
        "decoder": "One-dimensional Poisson count likelihood, uniform class prior; class mean shrunk toward fitting-set pooled mean; normalized temperature-scaled posterior",
        "calibration_candidates": [{"pseudo_repeats": s, "temperature": t, "uniform": s is None} for s, t in CALIBRATION],
        "selection": "Maximum mean inner-held-out log posterior across all four calibration folds, then refit on eight outer-training repeats",
        "primary_score": "Signed cross-fitted I_q = log2(K) + mean(log2 q(S,J|X)); not clipped, not maximized against Fano, and not an asymptotic information rate",
        "variational_identity": "I(C;X) - [H(C)+E log2 q(C|X)] = E_X KL_bits(p(C|X)||q(C|X)) >= 0 for fixed q; empirical cross-fitted scores are not confidence bounds",
        "fano": "Plug-in log2(K)-h2(error)-error*log2(K-1); accuracy <= chance returns 0. Accuracy averages uniform random tie-breaking among posterior maximizers",
        "word_nulls_per_segment": N_WORD_NULLS,
        "word_null": "Independent circular shifts of each template/repeat 600-bin binary trace; preserves its count and autocorrelation, but retains template-level rate differences, so not a strict zero-information null",
        "decoder_nulls_per_endpoint": N_DECODER_NULLS,
        "decoder_null": "At fixed 20 ms only, independently permute class positions within each repeat, then refit the complete nested decoder; breaks repeat-aligned S and J while preserving each repeat count marginal",
        "seed": SEED,
        "cache_sha256": sha256(CACHE),
        "simulation_code_sha256": sha256(ROOT / "Fig3" / "figure3_main.m"),
        "audit_code_sha256": sha256(Path(__file__)),
    }


def load_counts() -> tuple[dict, dict[int, np.ndarray], np.ndarray]:
    saved = loadmat(CACHE, squeeze_me=True)
    spikes = saved["Spike_cell"]
    if spikes.shape != (20, 10, 20):
        raise ValueError(f"Unexpected cache shape {spikes.shape}")
    np.testing.assert_array_equal(saved["timewindow"], np.arange(1, 21) * 6)
    counts = {int(w): np.zeros((20, 20, 10, SEGMENT_MS // int(w)), dtype=np.int16)
              for w in WINDOWS_MS}
    physical_rate = np.zeros((20, 20, 10))
    for cp in range(20):
        for s in range(20):
            for r in range(10):
                times = np.asarray(spikes[s, r, cp], float).ravel()
                if np.any(~np.isfinite(times)) or np.any(times <= 0) or np.any(times > SEGMENT_MS):
                    raise ValueError(f"Out-of-segment spike timestamps: {(cp, s, r)}")
                if np.any(np.diff(times) <= 0):
                    raise ValueError("Spike timestamps must be strictly increasing")
                physical_rate[cp, s, r] = times.size / (SEGMENT_MS / 1000)
                for width, array in counts.items():
                    # Right-closed bins also reproduce the saved binary convention.
                    indices = np.ceil(times / width).astype(int) - 1
                    array[cp, s, r] = np.bincount(indices, minlength=SEGMENT_MS // width)
                    if array[cp, s, r].sum() != times.size:
                        raise AssertionError("Binning lost or duplicated spikes")
                np.testing.assert_array_equal(counts[10][cp, s, r] > 0,
                                              np.ravel(saved["Binary_cell"][s, r, cp]).astype(bool))
    return saved, counts, physical_rate


def count_cv(values: np.ndarray, axis=None) -> np.ndarray:
    mean = np.mean(values, axis=axis)
    return np.divide(np.std(values, axis=axis, ddof=0), mean,
                     out=np.full_like(mean, np.nan, dtype=float), where=mean > 0)


def posterior_tables(train: np.ndarray, max_count: int) -> np.ndarray:
    """Log q(class|count), candidate x class x count; only train affects q."""
    k, repeats = train.shape
    totals = train.sum(axis=1)
    global_mean = float(train.mean())
    support = np.arange(max_count + 1)
    tables = [np.full((k, max_count + 1), -np.log(k))]
    for strength, temperature in CALIBRATION[1:]:
        rate = np.maximum((totals + strength * global_mean) / (repeats + strength), 1e-9)
        logits = (np.log(rate)[:, None] * support[None, :] - rate[:, None]) / temperature
        tables.append(logits - logsumexp(logits, axis=0, keepdims=True))
    return np.stack(tables)


def choose_calibration(x: np.ndarray, train_repeats: np.ndarray, max_count: int) -> tuple[int, np.ndarray]:
    scores = np.zeros(len(CALIBRATION))
    classes = np.arange(x.shape[0])[:, None]
    for validation in np.array_split(train_repeats, 4):
        fitting = np.setdiff1d(train_repeats, validation)
        if fitting.size != 6 or validation.size != 2:
            raise AssertionError("Unexpected nested calibration split")
        tables = posterior_tables(x[:, fitting], max_count)
        scores += tables[:, classes, x[:, validation]].mean(axis=(1, 2)) / 4
    return int(np.argmax(scores)), scores


def fano_bits(accuracy: np.ndarray, k: int) -> np.ndarray:
    accuracy = np.asarray(accuracy)
    error = 1 - accuracy
    binary_entropy = -(xlogy(error, error) + xlogy(accuracy, accuracy)) / np.log(2)
    raw = np.log2(k) - binary_entropy - error * np.log2(k - 1)
    return np.where(accuracy > 1 / k + 1e-14, np.maximum(raw, 0), 0)


def decode_counts(counts: np.ndarray, max_count: int) -> dict:
    """counts has template x repeat x time; there is NO neuron axis."""
    s, repeats, bins = counts.shape
    if repeats != 10 or counts.max() > max_count:
        raise ValueError("Unexpected repeat count or count outside fixed support")
    x = counts.transpose(0, 2, 1).reshape(s * bins, repeats)
    k = x.shape[0]
    classes = np.arange(k)[:, None]
    logscore = np.empty_like(x, dtype=float)
    accuracy = np.empty_like(x, dtype=float)
    chosen, inner_scores, normalization_error = [], [], []
    for test in np.array_split(np.arange(repeats), 5):
        train = np.setdiff1d(np.arange(repeats), test)
        if np.intersect1d(train, test).size or train.size != 8:
            raise AssertionError("Repeat leakage")
        best, calibration_scores = choose_calibration(x, train, max_count)
        logq = posterior_tables(x[:, train], max_count)[best]
        normalization_error.append(float(np.max(np.abs(np.exp(logq).sum(axis=0) - 1))))
        logscore[:, test] = (np.log(k) + logq[classes, x[:, test]]) / np.log(2)
        ties = np.isclose(logq, logq.max(axis=0, keepdims=True), rtol=0, atol=1e-12)
        probability_correct = ties / ties.sum(axis=0, keepdims=True)
        accuracy[:, test] = probability_correct[classes, x[:, test]]
        chosen.append(best)
        inner_scores.append(calibration_scores)
    probability = np.bincount(x.ravel(), minlength=max_count + 1) / x.size
    entropy = float(-np.sum(xlogy(probability, probability)) / np.log(2))
    return {
        "logscore_bits_window": float(logscore.mean()),
        "accuracy": float(accuracy.mean()),
        "fano_bits_window": float(fano_bits(accuracy.mean(), k)),
        "count_entropy_plugin_bits_window": entropy,
        "logscore_template_repeat": logscore.reshape(s, bins, repeats).mean(axis=1),
        "accuracy_template_repeat": accuracy.reshape(s, bins, repeats).mean(axis=1),
        "chosen_calibration": np.asarray(chosen),
        "inner_logposterior_scores_nats": np.asarray(inner_scores),
        "posterior_normalization_max_error": max(normalization_error),
    }


def finiteword_rates(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    total, conditional = [], []
    for length in WORD_LENGTHS:
        words = np.lib.stride_tricks.sliding_window_view(binary, length, axis=-1)
        packed = np.ascontiguousarray(np.packbits(words, axis=-1))
        codes = packed.view(np.dtype((np.void, packed.shape[-1]))).reshape(packed.shape[:-1])
        counts = np.unique(codes, return_counts=True)[1]
        p = counts / counts.sum()
        total.append(-np.sum(p * np.log2(p)))
        multiplicity = (codes[:, :, None, :] == codes[:, None, :, :]).sum(axis=2)
        conditional.append(np.log2(binary.shape[1]) - np.log2(multiplicity).mean())
    seconds = WORD_LENGTHS * 0.01
    return np.asarray(total) / seconds, np.asarray(conditional) / seconds


def word_intercepts(total: np.ndarray, conditional: np.ndarray) -> np.ndarray:
    output = []
    for lower, upper in FIT_RANGES:
        keep = (WORD_LENGTHS >= lower) & (WORD_LENGTHS <= upper)
        output.append(np.polyfit(1 / (WORD_LENGTHS[keep] * 0.01),
                                 (total - conditional)[keep], 1)[1])
    return np.asarray(output)


def paired_summary(first: np.ndarray, final: np.ndarray) -> dict:
    delta = np.asarray(final) - np.asarray(first)
    return {"first_mean": float(np.mean(first)), "final_mean": float(np.mean(final)),
            "mean_paired_delta": float(delta.mean()), "paired_deltas": delta.tolist(),
            "n_positive": int(np.sum(delta > 1e-12)), "n_negative": int(np.sum(delta < -1e-12)),
            "n_repeat_aggregates": int(delta.size),
            "scope": "Paired repeat-index aggregates across 20 templates; overlapping decoder training folds and different input segments; no independent training-seed or fixed-input effect inference"}


def load_completed_audit(output_dir: Path = DEFAULT_OUTPUT_DIR) -> tuple[dict, dict]:
    result = json.loads((output_dir / "lif_information_audit.json").read_text())
    spec = json.loads((output_dir / "lif_analysis_prespecification.json").read_text())
    if not result["complete"] or result["run_id"] != spec["run_id"]:
        raise ValueError("LIF audit is incomplete or being regenerated")
    with np.load(output_dir / "lif_information_audit.npz") as data:
        arrays = {key: data[key] for key in data.files}
    if str(arrays["run_id"]) != result["run_id"] or result["specification"]["cache_sha256"] != sha256(CACHE):
        raise ValueError("LIF audit/cache provenance mismatch")
    return result, arrays


def make_figures(output_dir: Path, result: dict, data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    time = data["segment_end_s"]
    primary = int(np.flatnonzero(WINDOWS_MS == PRIMARY_WINDOW_MS)[0])
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(time, data["firing_rate_hz"], "o-", label="Physical count / 6 s")
    ax.plot(time, data["legacy_event_rate_hz"], "--", label="Legacy inverse-ISI rate")
    ax.set(title="A  One output neuron", ylabel="Mean rate (Hz)")
    ax.legend(fontsize=7)
    for j, window in enumerate(WINDOWS_MS):
        axes[0, 1].plot(time, data["count_cv"][:, j], label=f"{window} ms", lw=2 if window == 20 else 1)
        axes[0, 2].plot(time, data["logscore_bits_s"][:, j], label=f"{window} ms", lw=2 if window == 20 else 1)
    axes[0, 1].set(title="B  Matched count-observation CV", ylabel="CV of X, including zeros")
    axes[0, 1].legend(fontsize=7)
    axes[0, 2].set(title="C  Signed held-out log-posterior score", ylabel="Window-normalized score (bits/s)")
    axes[0, 2].axhline(0, color="0.5", lw=.6)
    axes[0, 2].legend(fontsize=7)
    axes[1, 0].plot(time, data["logscore_per_cost"][:, primary], "o-", label="Log-posterior, fixed 20 ms")
    axes[1, 0].plot(time, data["fano_per_cost"][:, primary], "s--", label="Fano plug-in, fixed 20 ms")
    axes[1, 0].set(title="D  Distinct readouts per physical cost", ylabel="Window-normalized score / cost")
    axes[1, 0].legend(fontsize=7)
    paired = result["paired_primary"]["logscore_per_cost"]
    axes[1, 1].scatter(np.arange(1, 11), paired["paired_deltas"])
    axes[1, 1].axhline(0, color="0.5")
    axes[1, 1].set(title="E  Paired endpoint differences", xlabel="Repeat-index aggregate", ylabel="Final - first log score / cost")
    null = data["decoder_null_logscore_bits_window"] / .02
    axes[1, 2].boxplot(null.T, tick_labels=["12 s", "120 s"], showfliers=True)
    axes[1, 2].scatter([1, 2], data["logscore_bits_s"][[0, -1], primary], color="red", label="Observed")
    axes[1, 2].set(title="F  Repeat-wise class-permutation null", ylabel="20-ms log score / 0.02 s")
    axes[1, 2].legend(fontsize=7)
    for i, ax in enumerate(axes.flat):
        ax.spines[["top", "right"]].set_visible(False)
        if i < 4:
            ax.set_xlabel("Segment end time (s)")
    fig.suptitle("Cached LIF audit: different input segments, active plasticity; no fixed-input learning comparison")
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"Figure_lif_count_decoder_audit.{suffix}", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), constrained_layout=True)
    observed = data["legacy_direct_mi_bits_s"]
    null = data["word_null_mi_bits_s"]
    axes[0].plot(time, observed, "o-", label="Preserved Direct Method")
    axes[0].plot(time, null.mean(axis=1), "--", label="Mean circular-shift audit")
    axes[0].fill_between(time, null.min(axis=1), null.max(axis=1), alpha=.2)
    axes[0].set(xlabel="Segment end time (s)", ylabel="Extrapolated bits/s", title="A  Finite-word null diagnostic")
    axes[0].legend(fontsize=7)
    for index in (0, 18):
        axes[1].plot(WORD_LENGTHS * 10, data["word_conditional_rate"][index] * WORD_LENGTHS * .01,
                     "o-", label=f"{time[index]:g} s")
    axes[1].axhline(np.log2(10), color="black", ls="--", label="10-repeat entropy ceiling")
    axes[1].set(xlabel="Word duration (ms)", ylabel="Conditional entropy (bits/word)", title="B  Repeat ceiling")
    axes[1].legend(fontsize=7)
    for index in (0, 18):
        axes[2].plot(np.arange(len(FIT_RANGES)), data["word_fit_range_mi_bits_s"][index], "o-", label=f"{time[index]:g} s")
    axes[2].set_xticks(np.arange(len(FIT_RANGES)), [f"{a}-{b}" for a, b in FIT_RANGES])
    axes[2].set(xlabel="Fitted word lengths (10-ms bins)", ylabel="Extrapolated bits/s", title="C  Prespecified fit-range audit")
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"Figure_lif_finiteword_audit.{suffix}", dpi=200)
    plt.close(fig)


def main(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preserve_outputs(output_dir)
    run_id = datetime.now(timezone.utc).isoformat()
    spec = specification(run_id)
    write_json(output_dir / "lif_analysis_prespecification.json", spec)
    saved, all_counts, all_rates = load_counts()
    retained = np.arange(1, 20)
    time = np.asarray(saved["timewindow"], float)[retained]
    rate_by_template_repeat = all_rates[retained]
    rate = rate_by_template_repeat.mean(axis=(1, 2))
    cost = C0 + C1 * rate
    shape = (19, len(WINDOWS_MS))
    data = {
        "run_id": np.array(run_id), "segment_end_s": time,
        "checkpoint_indices": retained + 1, "windows_ms": WINDOWS_MS,
        "primary_window_ms": np.array(PRIMARY_WINDOW_MS),
        "firing_rate_hz": rate, "cost": cost,
        "firing_rate_template_repeat_hz": rate_by_template_repeat,
        "initialization_firing_rate_hz": np.array(all_rates[0].mean()),
        "count_cv": np.empty(shape), "count_cv_repeat": np.empty(shape + (10,)),
        "logscore_bits_window": np.empty(shape), "fano_bits_window": np.empty(shape),
        "decoder_accuracy": np.empty(shape), "count_entropy_plugin_bits_window": np.empty(shape),
        "logscore_template_repeat": np.empty(shape + (20, 10)),
        "accuracy_template_repeat": np.empty(shape + (20, 10)),
        "chosen_calibration": np.empty(shape + (5,), int),
        "inner_logposterior_scores_nats": np.empty(shape + (5, len(CALIBRATION))),
        "outer_test_repeat_indices": np.arange(10).reshape(5, 2),
        "word_lengths_bins": WORD_LENGTHS,
        "word_fit_ranges_bins": np.asarray(FIT_RANGES),
    }
    normalization_error = 0.0
    for j, width in enumerate(WINDOWS_MS):
        counts = all_counts[int(width)][retained]
        data[f"counts_{width}ms"] = counts
        for cp, segment in enumerate(counts):
            decoded = decode_counts(segment, int(np.ceil(width / 5)) + 1)
            data["count_cv"][cp, j] = count_cv(segment)
            data["count_cv_repeat"][cp, j] = count_cv(segment, axis=(0, 2))
            for key in ("logscore_bits_window", "fano_bits_window", "count_entropy_plugin_bits_window",
                        "logscore_template_repeat", "accuracy_template_repeat", "chosen_calibration",
                        "inner_logposterior_scores_nats"):
                data[key][cp, j] = decoded[key]
            data["decoder_accuracy"][cp, j] = decoded["accuracy"]
            normalization_error = max(normalization_error, decoded["posterior_normalization_max_error"])
        print(f"Decoded all 19 segments at {width} ms; primary remains 20 ms", flush=True)
    seconds = WINDOWS_MS / 1000
    data["logscore_bits_s"] = data["logscore_bits_window"] / seconds
    data["fano_bits_s"] = data["fano_bits_window"] / seconds
    data["logscore_per_cost"] = data["logscore_bits_s"] / cost[:, None]
    data["fano_per_cost"] = data["fano_bits_s"] / cost[:, None]
    for field, key in (("MI_vec", "legacy_direct_mi_bits_s"), ("Hr_vec", "legacy_direct_total_bits_s"),
                       ("Hn_vec", "legacy_direct_conditional_bits_s"), ("ICE_vec", "legacy_direct_information_per_legacy_cost"),
                       ("ra1_grand", "legacy_event_rate_hz"), ("cost1_grand", "legacy_event_cost"),
                       ("cv1_grand", "legacy_mean_within_trace_inverse_isi_cv")):
        data[key] = np.ravel(saved[field])[retained]
    data["direct_method_per_physical_cost_audit_only"] = data["legacy_direct_mi_bits_s"] / cost
    data["word_total_rate"] = np.empty((19, len(WORD_LENGTHS)))
    data["word_conditional_rate"] = np.empty_like(data["word_total_rate"])
    data["word_fit_range_mi_bits_s"] = np.empty((19, len(FIT_RANGES)))
    data["word_null_total_rate"] = np.empty((19, N_WORD_NULLS, len(WORD_LENGTHS)))
    data["word_null_conditional_rate"] = np.empty_like(data["word_null_total_rate"])
    data["word_null_mi_bits_s"] = np.empty((19, N_WORD_NULLS))
    rng = np.random.default_rng(SEED)
    binary = (all_counts[10][retained] > 0).astype(np.uint8)
    for cp, segment in enumerate(binary):
        total, conditional = finiteword_rates(segment)
        data["word_total_rate"][cp], data["word_conditional_rate"][cp] = total, conditional
        intercepts = word_intercepts(total, conditional)
        np.testing.assert_allclose(intercepts[1], data["legacy_direct_mi_bits_s"][cp], rtol=0, atol=1e-8)
        data["word_fit_range_mi_bits_s"][cp] = intercepts
        for null in range(N_WORD_NULLS):
            shifts = rng.integers(1, 600, size=(20, 10, 1))
            shifted = np.take_along_axis(segment, (np.arange(600)[None, None, :] + shifts) % 600, axis=2)
            a, b = finiteword_rates(shifted)
            data["word_null_total_rate"][cp, null] = a
            data["word_null_conditional_rate"][cp, null] = b
            data["word_null_mi_bits_s"][cp, null] = word_intercepts(a, b)[1]
        print(f"Finite-word audit: segment {cp + 2}/20", flush=True)
    data["decoder_null_logscore_bits_window"] = np.empty((2, N_DECODER_NULLS))
    data["decoder_null_fano_bits_window"] = np.empty((2, N_DECODER_NULLS))
    for endpoint, cp in enumerate((0, 18)):
        segment = data["counts_20ms"][cp]
        matrix = segment.transpose(0, 2, 1).reshape(-1, 10)
        for null in range(N_DECODER_NULLS):
            shuffled = np.stack([rng.permutation(matrix[:, r]) for r in range(10)], axis=1)
            decoded = decode_counts(shuffled.reshape(20, 300, 10).transpose(0, 2, 1), 5)
            data["decoder_null_logscore_bits_window"][endpoint, null] = decoded["logscore_bits_window"]
            data["decoder_null_fano_bits_window"][endpoint, null] = decoded["fano_bits_window"]
        print(f"Nested 20-ms decoder null: endpoint {cp + 2}/20", flush=True)
    primary = int(np.flatnonzero(WINDOWS_MS == PRIMARY_WINDOW_MS)[0])
    repeat_rate = rate_by_template_repeat.mean(axis=1)
    repeat_cv = data["count_cv_repeat"][:, primary]
    repeat_logscore = data["logscore_template_repeat"][:, primary].mean(axis=1) / .02
    repeat_fano = fano_bits(data["accuracy_template_repeat"][:, primary].mean(axis=1), 6000) / .02
    # Use the same segment-level denominator as the plotted pooled readout.
    repeat_log_efficiency = repeat_logscore / cost[:, None]
    paired = {
        "physical_rate_hz": paired_summary(repeat_rate[0], repeat_rate[-1]),
        "within_repeat_count_cv": paired_summary(repeat_cv[0], repeat_cv[-1]),
        "logscore_bits_s": paired_summary(repeat_logscore[0], repeat_logscore[-1]),
        "logscore_per_cost": paired_summary(repeat_log_efficiency[0], repeat_log_efficiency[-1]),
        "fano_per_cost_descriptive_repeat_plugin": paired_summary(repeat_fano[0] / cost[0], repeat_fano[-1] / cost[-1]),
    }
    rows = []
    for cp in range(19):
        row = {"checkpoint_index": int(retained[cp] + 1), "segment_start_s": float(time[cp] - 6),
               "segment_end_s": float(time[cp]), "physical_firing_rate_Hz": float(rate[cp]),
               "physical_affine_cost_proxy": float(cost[cp]), "primary_count_CV": float(data["count_cv"][cp, primary])}
        for key in ("logscore_bits_window", "logscore_bits_s", "logscore_per_cost", "fano_bits_window",
                    "fano_bits_s", "fano_per_cost", "decoder_accuracy", "count_entropy_plugin_bits_window"):
            row[f"primary_{key}"] = float(data[key][cp, primary])
        rows.append(row)
    result = {
        "complete": True, "run_id": run_id, "specification": spec,
        "protocol": {
            "n_output_neurons": 1, "saved_shape_template_repeat_segment": [20, 10, 20],
            "retained_segments": 19, "segment_duration_ms": SEGMENT_MS,
            "raw_input_templates_saved": False, "checkpoint_weights_saved": False,
            "same_input_across_repeats": "Supported by current and dev simulation code: InputTemplate is generated outside the repetition loop; original input realizations are not archived in this cache",
            "same_input_across_checkpoints": False,
            "plasticity_frozen_during_measurement": False,
            "checkpoint_interpretation": "Different contiguous six-second portions of each 120-second template while weights keep adapting; each template/repeat starts from common initial weights",
            "comparability": "Within-segment held-out repeat prediction is estimable. Cross-segment changes confound adaptation, changing input content, and history; no matched-input learning effect or frozen-checkpoint comparison is identifiable",
            "missing_replay_inputs": "No recN.mat, entN.mat, archived InputTemplate, or checkpoint synaptic weights found for this LIF cache; no new training or invented population was used",
            "code_evidence": ["Fig3/figure3_main.m:135-151 (template outside repeat loop)",
                              "Fig3/figure3_main.m:172 (reset weights)",
                              "Fig3/figure3_main.m:254-255 (new template segment)",
                              "Fig3/figure3_main.m:269-272,320-323 (active plasticity)"],
        },
        "dimensions_by_window": [{"window_ms": int(w), "time_bins_per_segment": int(6000 // w),
                                  "classes": int(20 * (6000 // w)), "observations_per_segment": int(200 * (6000 // w)),
                                  "outer_fold_training_repeats_per_class": 8, "outer_fold_test_repeats_per_class": 2}
                                 for w in WINDOWS_MS],
        "checkpoint_rows": rows,
        "paired_primary": paired,
        "endpoint_sensitivity": [{"window_ms": int(w), "count_CV_first_final": data["count_cv"][[0, -1], j].tolist(),
                                  **{key + "_first_final": data[key][[0, -1], j].tolist() for key in
                                     ("logscore_bits_s", "logscore_per_cost", "fano_bits_s", "fano_per_cost", "decoder_accuracy")}}
                                 for j, w in enumerate(WINDOWS_MS)],
        "primary_full_trajectory_descriptive": {
            "spearman_count_CV_vs_logscore_per_cost": float(spearmanr(data["count_cv"][:, primary], data["logscore_per_cost"][:, primary]).statistic),
            "spearman_segment_time_vs_logscore_per_cost": float(spearmanr(time, data["logscore_per_cost"][:, primary]).statistic)},
        "finiteword_audit": {
            "legacy_cache_unchanged": True,
            "observed_endpoint_bits_s": data["legacy_direct_mi_bits_s"][[0, -1]].tolist(),
            "circular_shift_endpoint_mean_bits_s": data["word_null_mi_bits_s"][[0, -1]].mean(axis=1).tolist(),
            "circular_shift_endpoint_ranges_bits_s": [[float(x.min()), float(x.max())] for x in data["word_null_mi_bits_s"][[0, -1]]],
            "conditional_entropy_ceiling_bits_word": float(np.log2(10)),
            "longest_word_endpoint_fraction_of_ceiling": (data["word_conditional_rate"][[0, -1], -1] / np.log2(10)).tolist(),
            "fit_ranges_bins": [list(pair) for pair in FIT_RANGES],
            "endpoint_fit_range_bits_s": data["word_fit_range_mi_bits_s"][[0, -1]].tolist(),
            "interpretation": "Preserved extrapolated word estimates are an audit, not substituted for count-decoder information; null subtraction is not a calibrated estimator"},
        "decoder_null": {
            "window_ms": 20, "endpoint_mean_logscore_bits_s": (data["decoder_null_logscore_bits_window"].mean(axis=1) / .02).tolist(),
            "endpoint_logscore_ranges_bits_s": [[float(x.min() / .02), float(x.max() / .02)] for x in data["decoder_null_logscore_bits_window"]],
        },
        "validation": {"posterior_normalization_max_error": normalization_error,
                       "exact_spike_count_conservation_all_windows": True,
                       "saved_binary_cache_reproduced": True,
                       "direct_word_estimates_reproduced_atol": 1e-8},
        "limitations": ["Exactly one output neuron; no population decoding or independent trained-neuron replication",
                        "20-ms primary is fixed regardless of direction; alternative windows are sensitivity results, not selection candidates",
                        "Nested training-only calibration normalizes q and optimizes a proper log score but does not prove population calibration or decoder optimality",
                        "Only ten repeats per class; six for inner fitting, eight for outer fitting, two for outer testing",
                        "Bins within a trajectory are dependent; folds share training data; no bin-level p values or confidence intervals are asserted",
                        "Fano and variational lower-bound formulas hold for population expectations; reported cross-fitted plug-ins are not finite-sample confidence bounds",
                        "Dividing bits/window by window duration gives a window-normalized readout, not an asymptotic information rate",
                        "Different input segments and ongoing plasticity preclude a matched-input causal learning comparison"],
    }
    if normalization_error > 1e-10 or sha256(CACHE) != spec["cache_sha256"]:
        raise AssertionError("Posterior normalization or source-cache stability check failed")
    np.savez_compressed(output_dir / "lif_information_audit.npz", **data)
    make_figures(output_dir, result, data)
    write_json(output_dir / "lif_information_audit.json", result)
    print(json.dumps({"protocol": result["protocol"], "endpoint_sensitivity": result["endpoint_sensitivity"],
                      "paired_primary": paired, "finiteword_audit": result["finiteword_audit"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    main(parser.parse_args().output_dir)
