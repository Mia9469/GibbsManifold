#!/usr/bin/env python3
"""Recompute recurrent-network information from saved learning checkpoints.

This supplies a neuron-wise alternative to the historically labelled population
branch in ``figure4_main.m``. That branch hid neuron identity by pooling neurons
with trials; it was neither joint-population MI nor mean single-neuron MI.
The old entropy helper also mismatched total/noise starts and omitted valid
zero-entropy positions. Those are implementation defects. Changing the readout,
0.1-ms to 1-ms word bins, inputs and repeat count are separate analysis choices,
not merely bug fixes. The current protocol:

1. uses the same frozen stimulus ensemble at every checkpoint;
2. generates genuine repeated trials separately for each stimulus;
3. bins model spikes at 1 ms before constructing words;
4. pools all sliding starts for total entropy and uses the identical starts for
   conditional entropy;
5. retains valid zero-entropy conditional positions; and
6. estimates each excitatory neuron separately before population aggregation.

The primary estimand is I((S,J);W), where S is frozen-template identity, J is
the 1 ms word-start index, and W is a single-neuron binary spike word.  The
script saves packed checkpoint rasters so that estimator changes can be audited
without rerunning the stochastic network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numba
import numpy as np
import scipy
from numba import njit
from numpy.lib.stride_tricks import sliding_window_view
from scipy.io import loadmat
from scipy.signal import convolve
from scipy.stats import spearmanr


DEFAULT_SEED = 20260905
DEFAULT_T_LIST = (4, 5, 6, 7, 8, 10, 13, 15, 20, 25, 30, 40)
DEFAULT_N_FIT = 8
DEFAULT_DT_BIN = 0.001
DECODER_WINDOWS_MS = (10, 20, 40, 50, 100)
PRIMARY_DECODER_WINDOW_MS = 20
COST_BASELINE = 2.4
COST_RATE_SLOPE = 0.37


@njit(cache=True)
def runnet_binned(
    dt: float,
    n_time: int,
    bin_factor: int,
    leak: float,
    threshold_e: np.ndarray,
    threshold_i: np.ndarray,
    input_t: np.ndarray,
    feedforward_e: np.ndarray,
    omega_ee: np.ndarray,
    omega_ei: np.ndarray,
    omega_ii: np.ndarray,
    omega_ie: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Numba port of ``runnet.m`` returning 1 ms excitatory rasters.

    Matrix names follow the MATLAB call signature.  At most one excitatory and
    one inhibitory neuron can spike at each 0.1 ms integration step, exactly as
    in the winner-take-all MATLAB implementation.
    """

    np.random.seed(seed)
    n_e = threshold_e.size
    n_i = threshold_i.size
    n_bins = n_time // bin_factor

    spikes_e = np.zeros((n_e, n_bins), dtype=np.uint8)
    voltage_e = np.zeros(n_e)
    voltage_i = np.zeros(n_i)
    refractory_e = np.full(n_e, -1, dtype=np.int16)
    refractory_i = np.full(n_i, -1, dtype=np.int16)
    feedforward_current = np.empty(n_e)

    winner_e = 0
    winner_i = 0
    previous_spike_e = 0
    previous_spike_i = 0

    for t in range(1, n_time):
        for n in range(n_e):
            feedforward_current[n] = dt * (
                feedforward_e[0, n] * input_t[0, t - 1]
                + feedforward_e[1, n] * input_t[1, t - 1]
                + feedforward_e[2, n] * input_t[2, t - 1]
            )

        best_score = -1.0e300
        best_index = 0
        for n in range(n_e):
            voltage_e[n] = (
                (1.0 - leak * dt) * voltage_e[n]
                + feedforward_current[n]
                + previous_spike_e * omega_ee[n, winner_e]
                + previous_spike_i * omega_ie[n, winner_i]
                + 0.001 * np.random.randn()
            )
            score = voltage_e[n] - threshold_e[n] - 0.02 * np.random.randn()
            if score > best_score:
                best_score = score
                best_index = n

        winner_e = best_index
        current_spike_e = 0
        if best_score >= 0.0 and refractory_e[winner_e] < 0:
            refractory_e[winner_e] = 10
            current_spike_e = 1
            spikes_e[winner_e, t // bin_factor] = 1
        for n in range(n_e):
            refractory_e[n] -= 1

        best_score = -1.0e300
        best_index = 0
        for n in range(n_i):
            voltage_i[n] = (
                (1.0 - leak * dt) * voltage_i[n]
                + current_spike_e * omega_ei[n, winner_e]
                + previous_spike_i * omega_ii[n, winner_i]
                + 0.001 * np.random.randn()
            )
            score = voltage_i[n] - threshold_i[n] - 0.02 * np.random.randn()
            if score > best_score:
                best_score = score
                best_index = n

        winner_i = best_index
        current_spike_i = 0
        if best_score >= 0.0 and refractory_i[winner_i] < 0:
            refractory_i[winner_i] = 10
            current_spike_i = 1
        for n in range(n_i):
            refractory_i[n] -= 1

        previous_spike_e = current_spike_e
        previous_spike_i = current_spike_i

    return spikes_e


@njit(cache=True)
def conditional_entropy_from_ids(word_ids: np.ndarray) -> float:
    """Mean entropy across all word starts, including zero-entropy starts."""

    n_repeats, n_windows = word_ids.shape
    codes = np.empty(n_repeats, dtype=np.int64)
    counts = np.empty(n_repeats, dtype=np.int64)
    entropy_sum = 0.0

    for j in range(n_windows):
        n_unique = 0
        for repeat in range(n_repeats):
            code = word_ids[repeat, j]
            found = -1
            for u in range(n_unique):
                if codes[u] == code:
                    found = u
                    break
            if found >= 0:
                counts[found] += 1
            else:
                codes[n_unique] = code
                counts[n_unique] = 1
                n_unique += 1

        entropy = 0.0
        for u in range(n_unique):
            probability = counts[u] / n_repeats
            entropy -= probability * np.log2(probability)
        entropy_sum += entropy

    return entropy_sum / n_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint.mat"))
    parser.add_argument("--legacy-state", type=Path, default=Path("figure4_state.mat"))
    parser.add_argument("--output-dir", type=Path, default=Path("network_information_corrected"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--stimuli", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--null-repeats", type=int, default=5)
    parser.add_argument("--reuse-rasters", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Use 2 stimuli, 4 repeats, and endpoint checkpoints.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_checkpoint(path: Path) -> dict[str, np.ndarray | float | int]:
    fields = [
        "Nx",
        "dt",
        "lambda",
        "NneuronE",
        "NneuronI",
        "ThreshE",
        "ThreshI",
        "A",
        "w",
        "FsE",
        "CsEE",
        "CsEI",
        "CsII",
        "CsIE",
    ]
    raw = loadmat(path, variable_names=fields, squeeze_me=True)
    return {
        "nx": int(raw["Nx"]),
        "dt": float(raw["dt"]),
        "leak": float(raw["lambda"]),
        "n_e": int(raw["NneuronE"]),
        "n_i": int(raw["NneuronI"]),
        "threshold_e": np.ascontiguousarray(raw["ThreshE"], dtype=np.float64),
        "threshold_i": np.ascontiguousarray(raw["ThreshI"], dtype=np.float64),
        "amplitude": float(raw["A"]),
        "kernel": np.ascontiguousarray(raw["w"], dtype=np.float64),
        "feedforward_e": np.ascontiguousarray(raw["FsE"], dtype=np.float64),
        "omega_ee": np.ascontiguousarray(raw["CsEE"], dtype=np.float64),
        "omega_ei": np.ascontiguousarray(raw["CsEI"], dtype=np.float64),
        "omega_ii": np.ascontiguousarray(raw["CsII"], dtype=np.float64),
        "omega_ie": np.ascontiguousarray(raw["CsIE"], dtype=np.float64),
    }


def generate_frozen_stimuli(
    checkpoint: dict[str, np.ndarray | float | int],
    n_stimuli: int,
    n_time: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    stimuli = np.empty((n_stimuli, int(checkpoint["nx"]), n_time), dtype=np.float64)
    kernel = np.asarray(checkpoint["kernel"])
    for stimulus in range(n_stimuli):
        raw = rng.standard_normal((int(checkpoint["nx"]), n_time)) * float(checkpoint["amplitude"])
        for component in range(int(checkpoint["nx"])):
            stimuli[stimulus, component] = (
                convolve(raw[component], kernel, mode="same") + rng.random(n_time)
            )
    return stimuli


def simulate_or_load_rasters(
    checkpoint: dict[str, np.ndarray | float | int],
    stimuli: np.ndarray,
    n_repeats: int,
    stages: np.ndarray,
    seed: int,
    raster_path: Path,
    reuse: bool,
) -> np.ndarray:
    n_stages = len(stages)
    n_stimuli = stimuli.shape[0]
    n_e = int(checkpoint["n_e"])
    n_time = stimuli.shape[-1]
    bin_factor = round(DEFAULT_DT_BIN / float(checkpoint["dt"]))
    if not np.isclose(bin_factor * float(checkpoint["dt"]), DEFAULT_DT_BIN):
        raise ValueError("The 1 ms analysis bin is not an integer multiple of the integration step.")
    n_bins = n_time // bin_factor
    packed_bins = (n_bins + 7) // 8

    if reuse:
        cached = np.load(raster_path, allow_pickle=False)
        expected = (n_stages, n_stimuli, n_repeats, n_e, packed_bins)
        if tuple(cached["packed_spikes_e"].shape) != expected:
            raise ValueError(f"Cached raster shape {cached['packed_spikes_e'].shape} != {expected}")
        return np.asarray(cached["packed_spikes_e"])

    packed = np.empty((n_stages, n_stimuli, n_repeats, n_e, packed_bins), dtype=np.uint8)
    started = time.time()
    for out_index, stage in enumerate(stages):
        stage_started = time.time()
        for stimulus in range(n_stimuli):
            for repeat in range(n_repeats):
                run_seed = seed + 100_000 + int(stage) * 10_000 + stimulus * 100 + repeat
                spikes = runnet_binned(
                    float(checkpoint["dt"]),
                    n_time,
                    bin_factor,
                    float(checkpoint["leak"]),
                    np.asarray(checkpoint["threshold_e"]),
                    np.asarray(checkpoint["threshold_i"]),
                    np.ascontiguousarray(stimuli[stimulus]),
                    np.ascontiguousarray(np.asarray(checkpoint["feedforward_e"])[stage]),
                    np.ascontiguousarray(np.asarray(checkpoint["omega_ee"])[stage]),
                    np.ascontiguousarray(np.asarray(checkpoint["omega_ei"])[stage]),
                    np.ascontiguousarray(np.asarray(checkpoint["omega_ii"])[stage]),
                    np.ascontiguousarray(np.asarray(checkpoint["omega_ie"])[stage]),
                    run_seed,
                )
                packed[out_index, stimulus, repeat] = np.packbits(
                    spikes, axis=-1, bitorder="little"
                )
        elapsed = time.time() - stage_started
        print(f"simulated checkpoint {stage + 1:02d}/{np.max(stages) + 1:02d} in {elapsed:.1f} s", flush=True)

    save_npz_atomic(
        raster_path,
        packed_spikes_e=packed,
        stages=stages,
        n_stimuli=np.array(n_stimuli),
        n_repeats=np.array(n_repeats),
        n_neurons=np.array(n_e),
        n_bins=np.array(n_bins),
        dt_bin=np.array(DEFAULT_DT_BIN),
        seed=np.array(seed),
    )
    print(f"saved packed rasters to {raster_path} ({time.time() - started:.1f} s total)", flush=True)
    return packed


def unpack_stage(packed_stage: np.ndarray, n_bins: int) -> np.ndarray:
    return np.unpackbits(
        packed_stage,
        axis=-1,
        count=n_bins,
        bitorder="little",
    )


def entropy_from_ids(ids: np.ndarray) -> float:
    _, counts = np.unique(ids, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def fit_intercept_and_r2(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual_sum = float(np.sum((y - fitted) ** 2))
    total_sum = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
    return float(intercept), r2


def estimate_matched_direct(
    spikes: np.ndarray,
    t_list: tuple[int, ...] = DEFAULT_T_LIST,
    n_fit: int = DEFAULT_N_FIT,
    dt_bin: float = DEFAULT_DT_BIN,
) -> dict[str, np.ndarray]:
    """Estimate per-neuron I((S,J);W) from [S, repeats, neurons, bins]."""

    n_stimuli, _, n_neurons, n_bins = spikes.shape
    t_values = np.asarray([length for length in t_list if length <= n_bins], dtype=int)
    powers = {length: 1 << np.arange(length, dtype=np.int64) for length in t_values}
    h_total_words = np.zeros((n_neurons, len(t_values)))
    h_noise_words = np.zeros_like(h_total_words)

    for neuron in range(n_neurons):
        neuron_spikes = spikes[:, :, neuron, :]
        for index, length in enumerate(t_values):
            windows = sliding_window_view(neuron_spikes, length, axis=-1)
            word_ids = np.tensordot(windows, powers[length], axes=([-1], [0]))
            h_total_words[neuron, index] = entropy_from_ids(word_ids.ravel())
            h_noise_words[neuron, index] = np.mean(
                [conditional_entropy_from_ids(word_ids[stimulus]) for stimulus in range(n_stimuli)]
            )

    durations = t_values * dt_bin
    inverse_durations = 1.0 / durations
    rate_total = h_total_words / durations
    rate_noise = h_noise_words / durations
    fit_slice = slice(0, min(n_fit, len(t_values)))

    h_rate = np.zeros(n_neurons)
    h_noise_rate = np.zeros(n_neurons)
    total_fit_r2 = np.zeros(n_neurons)
    noise_fit_r2 = np.zeros(n_neurons)
    for neuron in range(n_neurons):
        h_rate[neuron], total_fit_r2[neuron] = fit_intercept_and_r2(
            inverse_durations[fit_slice], rate_total[neuron, fit_slice]
        )
        h_noise_rate[neuron], noise_fit_r2[neuron] = fit_intercept_and_r2(
            inverse_durations[fit_slice], rate_noise[neuron, fit_slice]
        )

    mi_raw = h_rate - h_noise_rate
    return {
        "t_list": t_values,
        "h_total_words": h_total_words,
        "h_noise_words": h_noise_words,
        "h_rate": h_rate,
        "h_noise_rate": h_noise_rate,
        "mi_raw": mi_raw,
        "mi_clipped": np.maximum(mi_raw, 0.0),
        "total_fit_r2": total_fit_r2,
        "noise_fit_r2": noise_fit_r2,
    }


def population_dispersion(spikes: np.ndarray) -> tuple[float, float]:
    counts = spikes.sum(axis=-1).astype(float)
    trial_means = counts.mean(axis=-1)
    trial_sds = counts.std(axis=-1, ddof=1)
    cv = np.divide(trial_sds, trial_means, out=np.full_like(trial_means, np.nan), where=trial_means > 0)
    second_moment = np.mean(counts**2, axis=-1)
    sparseness = 1.0 - np.divide(
        trial_means**2,
        second_moment,
        out=np.zeros_like(trial_means),
        where=second_moment > 0,
    )
    return float(np.nanmean(cv)), float(np.nanmean(sparseness))


def aggregate_stage(spikes: np.ndarray, estimate: dict[str, np.ndarray]) -> dict[str, np.ndarray | float]:
    n_stimuli, n_repeats, n_neurons, n_bins = spikes.shape
    duration = n_bins * DEFAULT_DT_BIN
    firing_rate = spikes.sum(axis=(0, 1, 3)) / (n_stimuli * n_repeats * duration)
    mean_rate = float(np.mean(firing_rate))
    cost = COST_BASELINE + COST_RATE_SLOPE * mean_rate
    mean_mi_raw = float(np.mean(estimate["mi_raw"]))
    mean_mi = float(np.mean(estimate["mi_clipped"]))
    cv_p, sparseness_p = population_dispersion(spikes)
    return {
        "firing_rate": firing_rate,
        "mean_rate": mean_rate,
        "cost": cost,
        "mean_h_rate": float(np.mean(estimate["h_rate"])),
        "mean_h_noise_rate": float(np.mean(estimate["h_noise_rate"])),
        "mean_mi_raw": mean_mi_raw,
        "mean_mi": mean_mi,
        "mi_per_cost": mean_mi / cost,
        "mi_per_spike": mean_mi / mean_rate if mean_rate > 0 else np.nan,
        "mi_per_signalling_cost": mean_mi / (COST_RATE_SLOPE * mean_rate) if mean_rate > 0 else np.nan,
        "cv_p": cv_p,
        "sparseness_p": sparseness_p,
    }


def fano_information_lower_bound(accuracy: float, n_classes: int) -> float:
    """Fano lower bound on I(class; response) for balanced classes."""

    error = float(np.clip(1.0 - accuracy, 0.0, 1.0))
    if error == 0.0:
        binary_entropy = 0.0
    elif error == 1.0:
        binary_entropy = 0.0
    else:
        binary_entropy = -error * np.log2(error) - (1.0 - error) * np.log2(1.0 - error)
    lower_bound = np.log2(n_classes) - binary_entropy - error * np.log2(n_classes - 1)
    return float(max(0.0, lower_bound))


def decode_population_condition(
    spikes: np.ndarray,
    window_ms: int,
    n_folds: int = 5,
) -> tuple[np.ndarray, float]:
    """Decode (stimulus, non-overlapping time window) from population counts.

    Cross-validation holds out complete repeats.  A Poisson naive-Bayes decoder
    is fitted only on the remaining repeats; its held-out error is converted to
    a conservative information lower bound with Fano's inequality.
    """

    n_stimuli, n_repeats, n_neurons, n_bins = spikes.shape
    width = int(round(window_ms / (DEFAULT_DT_BIN * 1000.0)))
    n_windows = n_bins // width
    if n_windows < 2:
        raise ValueError(f"Decoder window {window_ms} ms is too long for {n_bins} bins.")
    counts = (
        spikes[..., : n_windows * width]
        .reshape(n_stimuli, n_repeats, n_neurons, n_windows, width)
        .sum(axis=-1)
        .transpose(0, 1, 3, 2)
        .astype(np.float64)
    )
    n_classes = n_stimuli * n_windows
    n_folds = min(n_folds, n_repeats)
    fold_ids = np.arange(n_repeats) % n_folds
    accuracies = np.zeros(n_folds)

    for fold in range(n_folds):
        test_mask = fold_ids == fold
        train_mask = ~test_mask
        class_rates = counts[:, train_mask].mean(axis=1).reshape(n_classes, n_neurons)
        class_rates = np.maximum(class_rates, 1.0e-3)
        test = (
            counts[:, test_mask]
            .transpose(0, 2, 1, 3)
            .reshape(n_classes * int(np.sum(test_mask)), n_neurons)
        )
        labels = np.repeat(np.arange(n_classes), int(np.sum(test_mask)))
        log_rates = np.log(class_rates)
        if not np.isfinite(test).all() or not np.isfinite(log_rates).all():
            raise FloatingPointError("Non-finite population-decoder input after rate regularization.")
        log_likelihood = np.einsum("ij,kj->ik", test, log_rates, optimize=True)
        log_likelihood -= np.sum(class_rates, axis=1)[None, :]
        if not np.isfinite(log_likelihood).all():
            raise FloatingPointError("Non-finite population-decoder log likelihood.")
        predictions = np.argmax(log_likelihood, axis=1)
        accuracies[fold] = np.mean(predictions == labels)

    return accuracies, fano_information_lower_bound(float(np.mean(accuracies)), n_classes)


def shifted_null(spikes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shifted = np.empty_like(spikes)
    n_bins = spikes.shape[-1]
    minimum_shift = max(1, n_bins // 10)
    for stimulus in range(spikes.shape[0]):
        for repeat in range(spikes.shape[1]):
            shift = int(rng.integers(minimum_shift, n_bins - minimum_shift))
            shifted[stimulus, repeat] = np.roll(spikes[stimulus, repeat], shift, axis=-1)
    return shifted


def legacy_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    names = ["MIE_ST", "MIE_temporal", "PcostE", "MeanPrateE", "x_time"]
    try:
        data = loadmat(path, variable_names=names, squeeze_me=True, struct_as_record=False)
    except Exception as error:
        print(f"warning: legacy state could not be read: {error}", file=sys.stderr)
        return {}
    output: dict[str, np.ndarray] = {}
    for name in names:
        if name in data:
            value = np.asarray(data[name], dtype=float)
            output[name] = np.mean(value, axis=0) if value.ndim == 2 else value
    return output


def analyse(
    packed: np.ndarray,
    stages: np.ndarray,
    n_bins: int,
    n_null_repeats: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict]:
    n_stages = len(stages)
    n_neurons = packed.shape[-2]
    n_words = len(DEFAULT_T_LIST)
    n_decoder_folds = min(5, packed.shape[2])
    decoder_windows = np.asarray(DECODER_WINDOWS_MS, dtype=int)

    arrays: dict[str, np.ndarray] = {
        "stages": stages,
        "h_rate": np.zeros((n_stages, n_neurons)),
        "h_noise_rate": np.zeros((n_stages, n_neurons)),
        "mi_raw": np.zeros((n_stages, n_neurons)),
        "mi_clipped": np.zeros((n_stages, n_neurons)),
        "firing_rate": np.zeros((n_stages, n_neurons)),
        "total_fit_r2": np.zeros((n_stages, n_neurons)),
        "noise_fit_r2": np.zeros((n_stages, n_neurons)),
        "h_total_words": np.zeros((n_stages, n_neurons, n_words)),
        "h_noise_words": np.zeros((n_stages, n_neurons, n_words)),
        "mean_rate": np.zeros(n_stages),
        "cost": np.zeros(n_stages),
        "mean_h_rate": np.zeros(n_stages),
        "mean_h_noise_rate": np.zeros(n_stages),
        "mean_mi_raw": np.zeros(n_stages),
        "mean_mi": np.zeros(n_stages),
        "mi_per_cost": np.zeros(n_stages),
        "mi_per_spike": np.zeros(n_stages),
        "mi_per_signalling_cost": np.zeros(n_stages),
        "cv_p": np.zeros(n_stages),
        "sparseness_p": np.zeros(n_stages),
        "decoder_windows_ms": decoder_windows,
        "decoder_accuracy": np.zeros((n_stages, len(decoder_windows), n_decoder_folds)),
        "decoder_mi_lower_bound_bits_window": np.zeros((n_stages, len(decoder_windows))),
        "decoder_mi_lower_bound_bits_s": np.zeros((n_stages, len(decoder_windows))),
        "decoder_mi_lower_bound_per_cost": np.zeros((n_stages, len(decoder_windows))),
    }

    unpacked_cache: dict[int, np.ndarray] = {}
    for stage_index, stage in enumerate(stages):
        started = time.time()
        spikes = unpack_stage(packed[stage_index], n_bins)
        estimate = estimate_matched_direct(spikes)
        summary = aggregate_stage(spikes, estimate)
        for key in [
            "h_rate",
            "h_noise_rate",
            "mi_raw",
            "mi_clipped",
            "total_fit_r2",
            "noise_fit_r2",
            "h_total_words",
            "h_noise_words",
        ]:
            arrays[key][stage_index] = estimate[key]
        arrays["firing_rate"][stage_index] = summary["firing_rate"]
        for key in [
            "mean_rate",
            "cost",
            "mean_h_rate",
            "mean_h_noise_rate",
            "mean_mi_raw",
            "mean_mi",
            "mi_per_cost",
            "mi_per_spike",
            "mi_per_signalling_cost",
            "cv_p",
            "sparseness_p",
        ]:
            arrays[key][stage_index] = float(summary[key])
        for window_index, window_ms in enumerate(decoder_windows):
            accuracies, information_bound = decode_population_condition(spikes, int(window_ms), n_decoder_folds)
            information_rate_bound = information_bound / (window_ms / 1000.0)
            arrays["decoder_accuracy"][stage_index, window_index] = accuracies
            arrays["decoder_mi_lower_bound_bits_window"][stage_index, window_index] = information_bound
            arrays["decoder_mi_lower_bound_bits_s"][stage_index, window_index] = information_rate_bound
            arrays["decoder_mi_lower_bound_per_cost"][stage_index, window_index] = (
                information_rate_bound / float(summary["cost"])
            )
        if stage_index in {0, n_stages // 2, n_stages - 1}:
            unpacked_cache[stage_index] = spikes
        print(f"estimated checkpoint {stage + 1:02d}: MI={summary['mean_mi']:.3f} bits/s, "
              f"MI/C={summary['mi_per_cost']:.3f} ({time.time() - started:.1f} s)", flush=True)

    diagnostic_indices = sorted(unpacked_cache)
    null_rng = np.random.default_rng(seed + 900_000)
    null_mi = np.zeros((len(diagnostic_indices), n_null_repeats))
    null_ratio = np.zeros_like(null_mi)
    for out_index, stage_index in enumerate(diagnostic_indices):
        spikes = unpacked_cache[stage_index]
        for repeat in range(n_null_repeats):
            null_spikes = shifted_null(spikes, null_rng)
            estimate = estimate_matched_direct(null_spikes)
            summary = aggregate_stage(null_spikes, estimate)
            null_mi[out_index, repeat] = float(summary["mean_mi"])
            null_ratio[out_index, repeat] = float(summary["mi_per_cost"])
            print(
                f"null checkpoint {stages[stage_index] + 1:02d}, repeat {repeat + 1}/{n_null_repeats}",
                flush=True,
            )
    arrays["diagnostic_stage_indices"] = np.asarray(diagnostic_indices, dtype=int)
    arrays["shift_null_mi"] = null_mi
    arrays["shift_null_mi_per_cost"] = null_ratio

    repeat_grid = np.asarray(
        sorted(set(value for value in (8, 12, 16, packed.shape[2]) if value <= packed.shape[2])),
        dtype=int,
    )
    sensitivity_indices = np.asarray([0, n_stages - 1], dtype=int)
    subset_replicates = 3
    sensitivity_mi = np.zeros((len(sensitivity_indices), len(repeat_grid), subset_replicates))
    sensitivity_ratio = np.zeros_like(sensitivity_mi)
    subset_rng = np.random.default_rng(seed + 950_000)
    for endpoint, stage_index in enumerate(sensitivity_indices):
        spikes = unpacked_cache[stage_index]
        for grid_index, repeat_count in enumerate(repeat_grid):
            for replicate in range(subset_replicates):
                if repeat_count == spikes.shape[1]:
                    selected = np.arange(spikes.shape[1])
                else:
                    selected = np.sort(subset_rng.choice(spikes.shape[1], repeat_count, replace=False))
                estimate = estimate_matched_direct(spikes[:, selected])
                summary = aggregate_stage(spikes[:, selected], estimate)
                sensitivity_mi[endpoint, grid_index, replicate] = float(summary["mean_mi"])
                sensitivity_ratio[endpoint, grid_index, replicate] = float(summary["mi_per_cost"])
    arrays["repeat_grid"] = repeat_grid
    arrays["repeat_sensitivity_stage_indices"] = sensitivity_indices
    arrays["repeat_sensitivity_mi"] = sensitivity_mi
    arrays["repeat_sensitivity_mi_per_cost"] = sensitivity_ratio

    first = 0
    final = n_stages - 1
    rho_ratio, p_ratio = spearmanr(stages, arrays["mi_per_cost"])
    rho_cv, p_cv = spearmanr(arrays["cv_p"], arrays["mi_per_cost"])
    primary_decoder_index = int(np.flatnonzero(decoder_windows == PRIMARY_DECODER_WINDOW_MS)[0])
    decoder_ratio = arrays["decoder_mi_lower_bound_per_cost"][:, primary_decoder_index]
    rho_decoder, p_decoder = spearmanr(stages, decoder_ratio)
    report = {
        "estimand": "mean single-neuron I((S,J);W), averaged over all 300 excitatory neurons",
        "sampling": "same 1 ms sliding starts for total and conditional entropy; zero-entropy positions retained",
        "inference_scope": "one saved learned-weight trajectory; stimuli and repeats are test resamples, not training replicates",
        "first_checkpoint": {
            key: float(arrays[key][first])
            for key in ["mean_rate", "cost", "mean_h_rate", "mean_h_noise_rate", "mean_mi", "mi_per_cost", "mi_per_spike", "cv_p"]
        },
        "final_checkpoint": {
            key: float(arrays[key][final])
            for key in ["mean_rate", "cost", "mean_h_rate", "mean_h_noise_rate", "mean_mi", "mi_per_cost", "mi_per_spike", "cv_p"]
        },
        "endpoint_change": {
            key: float(arrays[key][final] - arrays[key][first])
            for key in ["mean_rate", "cost", "mean_mi", "mi_per_cost", "mi_per_spike", "cv_p"]
        },
        "trajectory_spearman": {
            "checkpoint_vs_mi_per_cost_rho": float(rho_ratio),
            "checkpoint_vs_mi_per_cost_p": float(p_ratio),
            "cv_p_vs_mi_per_cost_rho": float(rho_cv),
            "cv_p_vs_mi_per_cost_p": float(p_cv),
            "checkpoint_vs_population_decoder_bound_per_cost_rho": float(rho_decoder),
            "checkpoint_vs_population_decoder_bound_per_cost_p": float(p_decoder),
        },
        "population_decoder": {
            "target": "joint frozen-stimulus identity S and non-overlapping word-start window J",
            "classifier": "Poisson naive Bayes, fivefold cross-validation holding out complete repeats",
            "information_quantity": "Fano lower bound from held-out classification error",
            "primary_window_ms": PRIMARY_DECODER_WINDOW_MS,
            "first_accuracy": float(np.mean(arrays["decoder_accuracy"][first, primary_decoder_index])),
            "final_accuracy": float(np.mean(arrays["decoder_accuracy"][final, primary_decoder_index])),
            "first_mi_lower_bound_bits_s": float(arrays["decoder_mi_lower_bound_bits_s"][first, primary_decoder_index]),
            "final_mi_lower_bound_bits_s": float(arrays["decoder_mi_lower_bound_bits_s"][final, primary_decoder_index]),
            "first_mi_lower_bound_per_cost": float(decoder_ratio[first]),
            "final_mi_lower_bound_per_cost": float(decoder_ratio[final]),
            "window_sensitivity_ms": decoder_windows.tolist(),
            "first_per_cost_by_window": arrays["decoder_mi_lower_bound_per_cost"][first].tolist(),
            "final_per_cost_by_window": arrays["decoder_mi_lower_bound_per_cost"][final].tolist(),
        },
        "fit_quality": {
            "median_total_R2": float(np.nanmedian(arrays["total_fit_r2"])),
            "median_noise_R2": float(np.nanmedian(arrays["noise_fit_r2"])),
            "fraction_total_R2_below_0_95": float(np.mean(arrays["total_fit_r2"] < 0.95)),
            "fraction_noise_R2_below_0_95": float(np.mean(arrays["noise_fit_r2"] < 0.95)),
        },
    }
    return arrays, report


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def training_time(stages: np.ndarray, dt: float) -> np.ndarray:
    return (2.0 ** (stages + 1)) * dt


def plot_audit(arrays: dict[str, np.ndarray], dt: float, output_dir: Path) -> None:
    configure_plotting()
    stages = arrays["stages"]
    x = training_time(stages, dt)
    decoder_index = int(np.flatnonzero(arrays["decoder_windows_ms"] == PRIMARY_DECODER_WINDOW_MS)[0])
    figure, axes = plt.subplots(2, 3, figsize=(10.6, 5.7), constrained_layout=True)

    axis = axes[0, 0]
    axis.plot(x, arrays["mean_mi"], "o-", color="black", ms=3, lw=1)
    axis.set_xscale("log")
    axis.set_xlabel("Training time (s)")
    axis.set_ylabel(r"Mean single-neuron MI rate (bits s$^{-1}$)")
    axis.set_title("A  Corrected matched-word estimate", loc="left", fontweight="bold")

    axis = axes[0, 1]
    axis.plot(x, arrays["mi_per_cost"], "o-", color="black", ms=3, lw=1, label=r"$\mathrm{MI}/\widetilde C$")
    axis.set_xscale("log")
    axis.set_xlabel("Training time (s)")
    axis.set_ylabel(r"$\mathrm{MI}/\widetilde C$ (proxy units)")
    twin = axis.twinx()
    twin.plot(x, arrays["mi_per_spike"], "--", color="#2878b5", lw=1.2, label="bits/spike")
    twin.set_ylabel("Information per spike (bits/spike)", color="#2878b5")
    twin.tick_params(axis="y", colors="#2878b5")
    axis.set_title("B  Total-cost and activity-only yields", loc="left", fontweight="bold")

    axis = axes[0, 2]
    axis.plot(x, arrays["decoder_mi_lower_bound_per_cost"][:, decoder_index], "o-", color="black", ms=3, lw=1)
    axis.set_xscale("log")
    axis.set_xlabel("Training time (s)")
    axis.set_ylabel(r"Population $\mathrm{MI}_{LB}/\widetilde C$")
    axis.set_title("C  Held-out population decoder", loc="left", fontweight="bold")

    axis = axes[1, 0]
    widths = arrays["decoder_windows_ms"]
    axis.plot(widths, arrays["decoder_mi_lower_bound_per_cost"][0], "o-", color="#777777", label="first checkpoint")
    axis.plot(widths, arrays["decoder_mi_lower_bound_per_cost"][-1], "o-", color="#d95f02", label="final checkpoint")
    axis.set_xscale("log")
    axis.set_xticks(widths, [str(value) for value in widths])
    axis.set_xlabel("Population-count window (ms)")
    axis.set_ylabel(r"Population $\mathrm{MI}_{LB}/\widetilde C$")
    axis.set_title("D  Temporal-window sensitivity", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 1]
    repeat_grid = arrays["repeat_grid"]
    sensitivity = arrays["repeat_sensitivity_mi_per_cost"]
    labels = ["first checkpoint", "final checkpoint"]
    colors = ["#777777", "#d95f02"]
    for endpoint in range(sensitivity.shape[0]):
        mean = sensitivity[endpoint].mean(axis=-1)
        low = sensitivity[endpoint].min(axis=-1)
        high = sensitivity[endpoint].max(axis=-1)
        axis.plot(repeat_grid, mean, "o-", color=colors[endpoint], label=labels[endpoint])
        axis.fill_between(repeat_grid, low, high, color=colors[endpoint], alpha=0.15)
    axis.set_xlabel("Repeated trials per stimulus")
    axis.set_ylabel(r"$\mathrm{MI}/\widetilde C$ (proxy units)")
    axis.set_title("E  Repeat-count sensitivity", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)

    axis = axes[1, 2]
    diagnostic_indices = arrays["diagnostic_stage_indices"]
    observed = arrays["mi_per_cost"][diagnostic_indices]
    null = arrays["shift_null_mi_per_cost"]
    positions = np.arange(len(diagnostic_indices))
    axis.scatter(positions - 0.13, observed, color="black", s=28, label="observed", zorder=3)
    for index in range(len(diagnostic_indices)):
        jitter = np.linspace(-0.04, 0.04, null.shape[1])
        axis.scatter(np.full(null.shape[1], positions[index] + 0.13) + jitter, null[index], color="#9e9e9e", s=14)
        axis.plot([positions[index] + 0.08, positions[index] + 0.18], [null[index].mean()] * 2, color="black", lw=1)
    axis.set_xticks(positions, [f"CP {stages[index] + 1}" for index in diagnostic_indices])
    axis.set_ylabel(r"$\mathrm{MI}/\widetilde C$ (proxy units)")
    axis.set_title("F  Independent circular-shift null", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)

    for axis in axes.ravel():
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    axes[0, 1].spines["right"].set_visible(False)

    figure.savefig(output_dir / "network_information_audit.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "network_information_audit.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_manuscript_panel(arrays: dict[str, np.ndarray], dt: float, output_dir: Path) -> None:
    configure_plotting()
    x = training_time(arrays["stages"], dt)
    window_index = int(np.flatnonzero(arrays["decoder_windows_ms"] == PRIMARY_DECODER_WINDOW_MS)[0])
    information_per_cost = arrays["decoder_mi_lower_bound_per_cost"][:, window_index]
    figure, axis = plt.subplots(figsize=(2.30, 1.58))
    axis.plot(x, information_per_cost, color="black", lw=1.35)
    axis.set_xscale("log")
    axis.set_xlabel(r"$t$ (s)")
    axis.set_ylabel(r"$\mathrm{MI}_{LB}/\widetilde C$")
    axis.set_ylim(35.0, 85.0)
    axis.set_yticks([40.0, 60.0, 80.0])
    axis.tick_params(labelsize=10)
    axis.xaxis.label.set_size(11)
    axis.yaxis.label.set_size(11)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    figure.subplots_adjust(left=0.25, right=0.98, bottom=0.29, top=0.97)
    figure.savefig(output_dir / "Figure4J_network_information_corrected.pdf", transparent=True)
    figure.savefig(output_dir / "Figure4J_network_information_corrected.png", dpi=600, transparent=True)
    plt.close(figure)


def plot_decoder_audit(arrays: dict[str, np.ndarray], dt: float, output_dir: Path) -> None:
    configure_plotting()
    x = training_time(arrays["stages"], dt)
    window_index = int(np.flatnonzero(arrays["decoder_windows_ms"] == PRIMARY_DECODER_WINDOW_MS)[0])
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

    axis = axes[0]
    axis.plot(x, arrays["decoder_mi_lower_bound_per_cost"][:, window_index], "o-", color="black", ms=3, lw=1)
    axis.set_xscale("log")
    axis.set_xlabel("Training time (s)")
    axis.set_ylabel(r"$\mathrm{MI}_{LB}/\widetilde C$ (proxy units)")
    axis.set_title("A  Held-out population decoder", loc="left", fontweight="bold")

    axis = axes[1]
    widths = arrays["decoder_windows_ms"]
    axis.plot(widths, arrays["decoder_mi_lower_bound_per_cost"][0], "o-", color="#777777", label="first checkpoint")
    axis.plot(widths, arrays["decoder_mi_lower_bound_per_cost"][-1], "o-", color="#d95f02", label="final checkpoint")
    axis.set_xscale("log")
    axis.set_xticks(widths, [str(value) for value in widths])
    axis.set_xlabel("Population-count window (ms)")
    axis.set_ylabel(r"$\mathrm{MI}_{LB}/\widetilde C$ (proxy units)")
    axis.set_title("B  Temporal-window sensitivity", loc="left", fontweight="bold")
    axis.legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    figure.savefig(output_dir / "network_population_decoder_audit.pdf", bbox_inches="tight")
    figure.savefig(output_dir / "network_population_decoder_audit.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else script_dir / args.checkpoint
    legacy_path = args.legacy_state if args.legacy_state.is_absolute() else script_dir / args.legacy_state
    output_dir = args.output_dir if args.output_dir.is_absolute() else script_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = load_checkpoint(checkpoint_path)
    all_stages = np.arange(np.asarray(checkpoint["feedforward_e"]).shape[0], dtype=int)
    if args.quick:
        stages = np.asarray([0, len(all_stages) - 1], dtype=int)
        n_stimuli = 2
        n_repeats = 4
        n_null_repeats = 1
    else:
        stages = all_stages
        n_stimuli = args.stimuli
        n_repeats = args.repeats
        n_null_repeats = args.null_repeats

    n_time = 10_000
    stimuli = generate_frozen_stimuli(checkpoint, n_stimuli, n_time, args.seed)
    raster_path = output_dir / "network_information_rasters_packbits.npz"
    packed = simulate_or_load_rasters(
        checkpoint,
        stimuli,
        n_repeats,
        stages,
        args.seed,
        raster_path,
        args.reuse_rasters,
    )
    n_bins = n_time // round(DEFAULT_DT_BIN / float(checkpoint["dt"]))
    arrays, report = analyse(packed, stages, n_bins, n_null_repeats, args.seed)

    legacy = legacy_cache(legacy_path)
    for key, value in legacy.items():
        arrays[f"legacy_{key}"] = value

    arrays["dt_bin"] = np.array(DEFAULT_DT_BIN)
    arrays["t_list"] = np.asarray(DEFAULT_T_LIST)
    arrays["n_fit"] = np.array(DEFAULT_N_FIT)
    arrays["seed"] = np.array(args.seed)
    arrays["n_stimuli"] = np.array(n_stimuli)
    arrays["n_repeats"] = np.array(n_repeats)
    save_npz_atomic(output_dir / "network_information_results.npz", **arrays)

    report.update(
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "legacy_state": str(legacy_path) if legacy_path.exists() else None,
            "seed": args.seed,
            "n_stimuli": n_stimuli,
            "n_repeats": n_repeats,
            "n_checkpoints": int(len(stages)),
            "integration_step_s": float(checkpoint["dt"]),
            "analysis_bin_s": DEFAULT_DT_BIN,
            "word_lengths_bins": list(DEFAULT_T_LIST),
            "fit_word_lengths_bins": list(DEFAULT_T_LIST[:DEFAULT_N_FIT]),
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "scipy": scipy.__version__,
                "numba": numba.__version__,
            },
            "legacy_error": (
                "Legacy MIE_ST stacked neurons and trials on one repeat axis, used a different frozen input "
                "at each checkpoint, and evaluated entropy words at the 0.1 ms simulation step."
            ),
        }
    )
    save_json_atomic(output_dir / "network_information_summary.json", report)
    plot_audit(arrays, float(checkpoint["dt"]), output_dir)
    plot_decoder_audit(arrays, float(checkpoint["dt"]), output_dir)
    plot_manuscript_panel(arrays, float(checkpoint["dt"]), output_dir)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
