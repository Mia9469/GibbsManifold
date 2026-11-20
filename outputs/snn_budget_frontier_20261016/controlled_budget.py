"""Actual closed-loop activity control and independently tested waveform readout."""
import json
import time

import numpy as np

from frontier import HERE, ROOT, counts_from_stage, histogram, metrics, sha, snapshot
from network_budget import evaluate, load_checkpoint, generate_frozen_stimuli, runnet_binned


OUT = HERE / "controlled"


def archived_model():
    subset = HERE / "archived_endpoint_weights.npz"
    if subset.exists():
        with np.load(subset, allow_pickle=False) as cached:
            model = {key: (cached[key].item() if cached[key].ndim == 0 else np.ascontiguousarray(cached[key]))
                     for key in cached.files if key != "original_checkpoint_sha256"}
            original_sha = str(cached["original_checkpoint_sha256"])
        return model, {0: 0, 28: 1}, original_sha
    path = ROOT / "Fig4/checkpoint.mat"
    return load_checkpoint(path), {0: 0, 28: 28}, sha(path)


def simulate(model, stage, stimulus, seed, spec):
    return evaluate(model["dt"], model["leak"], model["threshold_e"], model["threshold_i"],
                    np.ascontiguousarray(stimulus), model["feedforward_e"][stage],
                    model["omega_ee"][stage], model["omega_ei"][stage],
                    model["omega_ii"][stage], model["omega_ie"][stage], int(seed),
                    float(spec["target"]), spec["controller_strength"])


def fit_ridge(design, truth, alpha):
    mean_x, mean_y = design.mean(axis=0), truth.mean(axis=0)
    x, y = design - mean_x, truth - mean_y
    gram, cross = np.dot(x.T, x), np.dot(x.T, y)
    eigenvalues, vectors = np.linalg.eigh(gram)
    scale = max(float(np.trace(gram) / len(gram)), 1e-12)
    weights = np.dot(vectors, np.dot(vectors.T, cross) /
                     (np.maximum(eigenvalues, 0)[:, None] + alpha * scale))
    intercept = mean_y - np.dot(mean_x, weights)
    if not np.isfinite(weights).all() or not np.isfinite(intercept).all():
        raise ValueError("Non-finite readout fit")
    return weights, intercept


def predict(weights, intercept, rates):
    result = np.einsum("nk,nt->kt", weights, rates, optimize=False) + intercept[:, None]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite readout prediction")
    return result


def error(truth, estimate):
    return float(np.var(truth - estimate, axis=1).sum() / np.var(truth, axis=1).sum())


def calibrate(rates, truth):
    grid = [1e-4, 1e-3, .01, .1, 1., 10., 100.]
    scores = np.zeros((len(rates), len(grid)))
    for held in range(len(rates)):
        train = [i for i in range(len(rates)) if i != held]
        x = np.concatenate([rates[i] for i in train], axis=1).T
        y = np.concatenate([truth[i] for i in train], axis=1).T
        for j, alpha in enumerate(grid):
            w, b = fit_ridge(x, y, alpha)
            scores[held, j] = error(truth[held], predict(w, b, rates[held]))
    choice = int(np.argmin(scores.mean(axis=0)))
    w, b = fit_ridge(np.concatenate(rates, axis=1).T,
                     np.concatenate(truth, axis=1).T, grid[choice])
    return w, b, {"alpha": grid[choice], "grid": grid,
                  "leave_template_out_errors": scores.tolist()}


def run():
    snapshot()
    spec = json.loads((HERE / "controlled_protocol.json").read_text())
    OUT.mkdir(exist_ok=True)
    print("loading archived weights", flush=True)
    model, stage_map, original_sha = archived_model()
    nt = int(round(spec["duration_s"] / model["dt"]))
    bank = generate_frozen_stimuli(model, spec["calibration_templates"] + spec["test_templates"], nt, spec["seed"])
    calibration_bank, test_bank = bank[:spec["calibration_templates"]], bank[spec["calibration_templates"]:]
    np.savez_compressed(OUT / "frozen_inputs.npz", calibration=calibration_bank, test=test_bank)
    test = evaluate(model["dt"], model["leak"], model["threshold_e"], model["threshold_i"],
                    np.ascontiguousarray(bank[0][:, :200]), model["feedforward_e"][0], model["omega_ee"][0],
                    model["omega_ei"][0], model["omega_ii"][0], model["omega_ie"][0], spec["seed"])[0]
    expected = runnet_binned(model["dt"], 200, 10, model["leak"], model["threshold_e"], model["threshold_i"],
                             np.ascontiguousarray(bank[0][:, :200]), model["feedforward_e"][0], model["omega_ee"][0],
                             model["omega_ei"][0], model["omega_ii"][0], model["omega_ie"][0], spec["seed"])
    np.testing.assert_array_equal(test[:model["n_e"]], expected)
    burn, results = int(round(spec["burn_in_s"] * 1000)), []
    duration = spec["duration_s"] - spec["burn_in_s"]
    started = time.monotonic()
    for target in spec["target_total_spikes_per_second"]:
        spec["target"] = target
        for stage in spec["checkpoint_indices"]:
            tag = f"budget_{target}_checkpoint_{stage + 1:02d}"
            result_path = OUT / (tag + ".json")
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
                print("loaded", tag, flush=True)
                continue
            rates, truth = [], []
            for index, stimulus in enumerate(calibration_bank):
                _, r, x, _ = simulate(model, stage_map[stage], stimulus, spec["seed"] + 1000 + index, spec)
                rates.append(r[:, burn:])
                truth.append(x[:, burn:])
            w, b, readout = calibrate(rates, truth)
            print("readout", tag, "alpha", readout["alpha"], flush=True)
            responses, errors, totals, gain_ranges = [], [], [], []
            for index, stimulus in enumerate(test_bank):
                rr, ee, tt, gg = [], [], [], []
                for repeat in range(spec["noise_repeats_per_template"]):
                    seed = spec["seed"] + 10000 + 100 * index + repeat
                    spikes, r, x, gain = simulate(model, stage_map[stage], stimulus, seed, spec)
                    spikes, r, x, gain = spikes[:, burn:], r[:, burn:], x[:, burn:], gain[burn:]
                    rr.append(spikes)
                    ee.append(error(x, predict(w, b, r)))
                    tt.append(float(spikes.sum()) / duration)
                    gg.append(np.quantile(gain, [0, .5, 1]).tolist())
                responses.append(rr)
                errors.append(ee)
                totals.append(tt)
                gain_ranges.append(gg)
            responses = np.asarray(responses)
            actual = float(np.mean(totals))
            row = {"target_total_rate": target, "checkpoint": stage + 1,
                   "actual_total_rate": actual, "budget_relative_error": abs(actual - target) / target,
                   "budget_within_tolerance": abs(actual - target) / target <= spec["budget_tolerance"],
                   "readout": readout, "waveform_re_mean": float(np.mean(errors)),
                   "waveform_re_by_template_repeat": errors, "rate_by_template_repeat": totals,
                   "gain_ranges_by_template_repeat": gain_ranges, "distributions": []}
            for group, subset in (("all", responses), ("excitatory", responses[:, :, :model["n_e"]]),
                                  ("inhibitory", responses[:, :, model["n_e"]:])):
                for window in [spec["primary_window_ms"]] + spec["sensitivity_windows_ms"]:
                    counts = counts_from_stage(subset, window)
                    row["distributions"].append({"group": group, "window_ms": window,
                        **metrics(histogram(counts, int(np.ceil(window / 1.1))))})
            np.savez_compressed(OUT / (tag + "_responses.npz"),
                                packed_spikes=np.packbits(responses, axis=-1, bitorder="little"),
                                decoder=w, intercept=b, n_bins=np.array(responses.shape[-1]),
                                errors=np.asarray(errors), rates=np.asarray(totals))
            result_path.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            results.append(row)
            print(tag, "rate", actual, "RE", row["waveform_re_mean"],
                  "gap", row["distributions"][0]["deficit_nats"],
                  "elapsed", time.monotonic() - started, flush=True)
    result = {"protocol_sha256": sha(HERE / "controlled_protocol.json"),
              "checkpoint_sha256": original_sha,
              "simulator_matches_existing_port": True, "results": results}
    (OUT / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    snapshot()


if __name__ == "__main__":
    run()
