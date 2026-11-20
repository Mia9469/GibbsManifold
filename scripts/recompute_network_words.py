"""Recompute Figure5J from the saved binary responses, without model training.

Only NumPy and the Python standard library are needed. All paths are relative
to the extracted code archive; output files never replace the supplied inputs.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time

import numpy as np

FIT_LENGTHS = (4, 5, 6, 7, 8, 10, 13, 15)
ALL_LENGTHS = FIT_LENGTHS + (20, 25, 30, 40)


def require_finite(name, values):
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite {name}; refusing to average this checkpoint.")


def entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    counts = counts[counts > 0]
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def marginal_and_conditional(ids):
    """Equal-weight (template, start) conditions, including entropy-zero cells."""
    n_stimuli, n_repeats, n_starts = ids.shape
    maximum = int(ids.max())
    if maximum < 2**16:
        counts = np.bincount(ids.ravel().astype(np.int64))
    else:
        _, counts = np.unique(ids, return_counts=True)
    marginal = entropy_from_counts(counts)
    # Sorted repeat runs give state multiplicities without treating neurons
    # or relative word starts as extra conditional-noise repeats.
    sorted_rows = np.sort(ids.transpose(0, 2, 1).reshape(-1, n_repeats), axis=1)
    edges = np.ones(sorted_rows.shape, dtype=bool)
    edges[:, 1:] = sorted_rows[:, 1:] != sorted_rows[:, :-1]
    starts = np.flatnonzero(edges.ravel())
    multiplicities = np.diff(np.append(starts, sorted_rows.size)).astype(float)
    conditional = np.log2(n_repeats) - np.sum(multiplicities * np.log2(multiplicities)) / (
        n_stimuli * n_starts * n_repeats)
    if -1e-12 < conditional < 0:
        conditional = 0.0
    require_finite("word entropies", [marginal, conditional])
    return marginal, float(conditional)


def encode_words(spikes, maximum_length):
    """Little-endian rolling words; shorter lengths mask the same exact starts."""
    dtype = np.uint16 if maximum_length <= 15 else np.uint64
    words = np.zeros(spikes.shape, dtype=dtype)
    for offset in range(maximum_length):
        words[..., :spikes.shape[-1] - offset] |= (
            spikes[..., offset:].astype(dtype) << offset)
    return words


def estimate(spikes, lengths, dt_bin):
    n_stimuli, n_repeats, n_neurons, n_bins = spikes.shape
    if max(lengths) > n_bins or n_repeats < 2:
        raise ValueError("Invalid word or repeat contract.")
    words = encode_words(spikes, max(lengths))
    total = np.empty((n_neurons, len(lengths)))
    conditional = np.empty_like(total)
    for neuron in range(n_neurons):
        for j, length in enumerate(lengths):
            n_starts = n_bins - length + 1
            ids = words[:, :, neuron, :n_starts] & ((1 << length) - 1)
            total[neuron, j], conditional[neuron, j] = marginal_and_conditional(ids)
    durations = np.asarray(lengths, dtype=float) * dt_bin
    fit_columns = [lengths.index(length) for length in FIT_LENGTHS]
    inverse = 1.0 / durations[fit_columns]
    total_rates = np.polyfit(inverse, (total[:, fit_columns] / durations[fit_columns]).T, 1)[1]
    noise_rates = np.polyfit(inverse, (conditional[:, fit_columns] / durations[fit_columns]).T, 1)[1]
    mi_raw = total_rates - noise_rates
    mean_rate = float(spikes.sum(dtype=np.int64) / (n_stimuli * n_repeats * n_neurons * n_bins * dt_bin))
    require_finite("fitted rates and firing", np.r_[total_rates, noise_rates, mi_raw, mean_rate])
    if mean_rate <= 0:
        raise ValueError("Undefined per-spike ratio at zero ensemble firing.")
    mean_mi = float(np.maximum(mi_raw, 0).mean())
    return {"h_total_words": total, "h_noise_words": conditional,
            "h_rate": total_rates, "h_noise_rate": noise_rates, "mi_raw": mi_raw,
            "mean_mi": mean_mi, "mean_rate": mean_rate, "mi_per_spike": mean_mi / mean_rate,
            "negative_unit_fits": int(np.sum(mi_raw < 0))}


def self_test():
    rng = np.random.default_rng(7)
    spikes = (rng.random((2, 4, 3, 27)) < .12).astype(np.uint8)
    spikes[:, :, 0] = 0
    words = encode_words(spikes, 15)
    from numpy.lib.stride_tricks import sliding_window_view
    for length in (4, 15):
        ids = words[..., :28 - length] & ((1 << length) - 1)
        explicit = np.sum(sliding_window_view(spikes, length, axis=-1) * (1 << np.arange(length)), axis=-1)
        np.testing.assert_array_equal(ids, explicit)
        for neuron in range(3):
            sample = ids[:, :, neuron]
            _, counts = np.unique(sample, return_counts=True)
            expected_total = entropy_from_counts(counts)
            cells = []
            for stimulus in range(2):
                for start in range(sample.shape[-1]):
                    _, counts = np.unique(sample[stimulus, :, start], return_counts=True)
                    cells.append(entropy_from_counts(counts))
            observed = marginal_and_conditional(sample)
            np.testing.assert_allclose(observed, [expected_total, np.mean(cells)], atol=1e-14)
    assert marginal_and_conditional(np.zeros((2, 4, 5), dtype=np.uint16)) == (0.0, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--rasters", type=Path, default=root / "source_data/network_common_input_rasters_packbits.npz")
    parser.add_argument("--reference", type=Path, default=root / "source_data/network_information_results.npz")
    parser.add_argument("--output-dir", type=Path, default=root / "recomputed/network_words")
    parser.add_argument("--stages", type=int, nargs="+", help="One-based checkpoints; default: all29.")
    parser.add_argument("--all-word-lengths", action="store_true", help="Also recompute four longer, non-fitted diagnostic words.")
    args = parser.parse_args()
    self_test()
    lengths = ALL_LENGTHS if args.all_word_lengths else FIT_LENGTHS
    with np.load(args.rasters, allow_pickle=False) as raw, np.load(args.reference, allow_pickle=False) as saved:
        assert raw["packed_spikes_e"].shape == (29, 5, 20, 300, 125)
        assert int(raw["n_bins"]) == 1000 and float(raw["dt_bin"]) == .001
        assert int(raw["seed"]) == 20260905
        assert int(saved["n_fit"]) == 8
        np.testing.assert_array_equal(saved["t_list"], ALL_LENGTHS)
        indices = list(range(29)) if args.stages is None else [stage - 1 for stage in args.stages]
        if len(set(indices)) != len(indices) or any(i < 0 or i >= 29 for i in indices):
            raise ValueError("Checkpoint indices must be unique integers from1 to29.")
        rows, errors = [], []
        started = time.time()
        for i in indices:
            spikes = np.unpackbits(raw["packed_spikes_e"][i], axis=-1, count=1000, bitorder="little")
            value = estimate(spikes, lengths, float(raw["dt_bin"]))
            width = len(lengths)
            np.testing.assert_allclose(value["h_total_words"], saved["h_total_words"][i, :, :width], atol=2e-12, rtol=0)
            np.testing.assert_allclose(value["h_noise_words"], saved["h_noise_words"][i, :, :width], atol=2e-12, rtol=0)
            for key in ("h_rate", "h_noise_rate", "mi_raw"):
                np.testing.assert_allclose(value[key], saved[key][i], atol=2e-9, rtol=0)
            for key in ("mean_mi", "mean_rate", "mi_per_spike"):
                expected = float(saved[key][i])
                np.testing.assert_allclose(value[key], expected, atol=2e-9, rtol=0)
                errors.append(abs(value[key] - expected))
            rows.append({"checkpoint": i + 1, "time_s": 2.0**(i + 1) * 1e-4,
                         "MI_bits_s": value["mean_mi"], "rate_Hz": value["mean_rate"],
                         "estimated_bits_per_spike": value["mi_per_spike"],
                         "negative_unit_fits": value["negative_unit_fits"]})
            print(f"checkpoint{i+1:02d}: MI={value['mean_mi']:.9f}, rate={value['mean_rate']:.9f}, bits/spike={value['mi_per_spike']:.9f}", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "network_from_responses.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {"checkpoints": [row["checkpoint"] for row in rows], "n_neurons": 300,
              "n_templates": 5, "n_repeats": 20, "dt_bin_s": .001,
              "computed_word_lengths_bins": list(lengths), "fit_word_lengths_bins": list(FIT_LENGTHS),
              "zero_entropy_conditions_retained": True, "matched_sliding_starts": True,
              "max_summary_difference": max(errors), "negative_unit_fits": sum(row["negative_unit_fits"] for row in rows),
              "raw_raster_sha256": hashlib.sha256(args.rasters.read_bytes()).hexdigest(),
              "elapsed_s": time.time() - started, "new_simulation_or_training": False}
    (args.output_dir / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: information recomputed from saved responses, not ratios alone.", flush=True)


if __name__ == "__main__":
    main()
