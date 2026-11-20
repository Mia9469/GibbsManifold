"""Threshold-calibrated SNN endpoint assay, preserving the existing network law."""
from pathlib import Path
import json
import sys
import time

import numpy as np
from numba import njit

from frontier import HERE, ROOT, counts_from_stage, histogram, metrics, sha, snapshot

sys.path.insert(0, str(ROOT))
from Fig4.recompute_network_information import load_checkpoint, generate_frozen_stimuli, runnet_binned


@njit(cache=True)
def evaluate(dt, leak, te, ti, stimulus, fe, ee, ei, ii, ie, seed,
             activity_target=0., controller_strength=.12):
    """Same update/RNG order as runnet_binned, with both populations and readout."""
    np.random.seed(seed)
    ne, ni, nt = len(te), len(ti), stimulus.shape[1]
    factor = int(round(.001 / dt))
    spikes = np.zeros((ne + ni, nt // factor), dtype=np.uint8)
    rates = np.zeros((ne, nt // factor))
    targets = np.zeros((3, nt // factor))
    gains = np.ones(nt // factor)
    ve, vi, r, x = np.zeros(ne), np.zeros(ni), np.zeros(ne), np.zeros(3)
    re, ri = np.full(ne, -1), np.full(ni, -1)
    winner_e, winner_i, previous_e, previous_i = 0, 0, 0, 0
    decay = 1 - leak * dt
    log_gain, gain, block_spikes = 0., 1., 0
    control_period = int(round(.02 / dt))
    for t in range(1, nt):
        best, ke = -1e300, 0
        for n in range(ne):
            drive = dt * (fe[0, n] * stimulus[0, t - 1] + fe[1, n] * stimulus[1, t - 1] + fe[2, n] * stimulus[2, t - 1])
            ve[n] = decay * ve[n] + drive + previous_e * ee[n, winner_e] + previous_i * ie[n, winner_i] + .001 * np.random.randn()
            score = ve[n] - gain * te[n] - .02 * np.random.randn()
            if score > best:
                best, ke = score, n
        winner_e = ke
        current_e = 0
        if best >= 0 and re[ke] < 0:
            current_e, re[ke] = 1, 10
            spikes[ke, t // factor] = 1
        for n in range(ne):
            re[n] -= 1
            r[n] *= decay
        if current_e:
            r[ke] += 1
        best, ki = -1e300, 0
        for n in range(ni):
            vi[n] = decay * vi[n] + current_e * ei[n, ke] + previous_i * ii[n, winner_i] + .001 * np.random.randn()
            score = vi[n] - gain * ti[n] - .02 * np.random.randn()
            if score > best:
                best, ki = score, n
        winner_i = ki
        current_i = 0
        if best >= 0 and ri[ki] < 0:
            current_i, ri[ki] = 1, 10
            spikes[ne + ki, t // factor] = 1
        ri -= 1
        for j in range(3):
            x[j] = decay * x[j] + dt * stimulus[j, t - 1]
        if (t + 1) % factor == 0:
            rates[:, t // factor] = r
            targets[:, t // factor] = x
            gains[t // factor] = gain
        block_spikes += current_e + current_i
        if activity_target > 0 and (t + 1) % control_period == 0:
            log_gain += controller_strength * (block_spikes / (activity_target * .02) - 1)
            log_gain = min(np.log(20.), max(np.log(.03), log_gain))
            gain, block_spikes = np.exp(log_gain), 0
        previous_e, previous_i = current_e, current_i
    return spikes, rates, targets, gains


def simulate(model, stage, stimulus, gain, seed):
    return evaluate(model["dt"], model["leak"], model["threshold_e"] * gain,
                    model["threshold_i"] * gain, np.ascontiguousarray(stimulus),
                    model["feedforward_e"][stage], model["omega_ee"][stage],
                    model["omega_ei"][stage], model["omega_ii"][stage], model["omega_ie"][stage], int(seed))[:3]


def calibration(model, stage, bank, spec, target):
    burn = int(round(spec["burn_in_s"] * 1000))
    length = int(round((spec["duration_s"] - spec["burn_in_s"]) * 1000))
    history = []

    def at(gain, keep=False):
        totals, rs, xs = [], [], []
        for index, stimulus in enumerate(bank):
            spikes, rates, truth = simulate(model, stage, stimulus, gain, spec["seed"] + 1000 + index)
            totals.append(float(spikes[:, burn:].sum()) / (length / 1000))
            if keep:
                rs.append(rates[:, burn:])
                xs.append(truth[:, burn:])
        achieved = float(np.mean(totals))
        history.append({"gain": float(gain), "total_rate": achieved})
        return achieved, rs, xs

    low, high = spec["gain_domain"]
    low_rate, _, _ = at(low)
    high_rate, _, _ = at(high)
    if not high_rate <= target <= low_rate:
        return {"feasible": False, "history": history}, None
    candidates = []
    for _ in range(spec["calibration_iterations"]):
        middle = np.sqrt(low * high)
        rate, _, _ = at(middle)
        candidates.append((abs(rate - target), middle, rate))
        if abs(rate - target) / target <= spec["calibration_tolerance"]:
            break
        if rate > target:
            low = middle
        else:
            high = middle
    _, gain, _ = min(candidates)
    achieved, rates, truth = at(gain, keep=True)
    design = np.concatenate(rates, axis=1).T
    target_signal = np.concatenate(truth, axis=1).T
    decoder = np.linalg.lstsq(design, target_signal, rcond=None)[0]
    return {"feasible": abs(achieved - target) / target <= spec["calibration_tolerance"],
            "gain": float(gain), "achieved_calibration_rate": achieved,
            "relative_calibration_error": abs(achieved - target) / target, "history": history}, decoder


def run():
    snapshot()
    spec = json.loads((HERE / "matched_budget_protocol.json").read_text())
    model = load_checkpoint(ROOT / "Fig4/checkpoint.mat")
    nt = int(round(spec["duration_s"] / model["dt"]))
    bank = generate_frozen_stimuli(model, spec["calibration_templates"] + spec["test_templates"], nt, spec["seed"])
    calibration_bank = bank[:spec["calibration_templates"]]
    test_bank = bank[spec["calibration_templates"]:]
    test_spikes, _, _ = simulate(model, 0, calibration_bank[0][:, :200], 1., spec["seed"])
    expected = runnet_binned(model["dt"], 200, 10, model["leak"], model["threshold_e"], model["threshold_i"],
                            np.ascontiguousarray(calibration_bank[0][:, :200]), model["feedforward_e"][0],
                            model["omega_ee"][0], model["omega_ei"][0], model["omega_ii"][0], model["omega_ie"][0], spec["seed"])
    np.testing.assert_array_equal(test_spikes[:model["n_e"]], expected)
    np.savez_compressed(HERE / "matched_budget_inputs.npz", calibration=calibration_bank, test=test_bank)
    burn = int(round(spec["burn_in_s"] * 1000))
    results = []
    started = time.monotonic()
    for target in spec["target_total_spikes_per_second"]:
        for stage in spec["checkpoint_indices"]:
            tag = f"budget_{target}_checkpoint_{stage + 1:02d}"
            result_path = HERE / (tag + ".json")
            if result_path.exists():
                results.append(json.loads(result_path.read_text()))
                print("loaded", tag, flush=True)
                continue
            cal, decoder = calibration(model, stage, calibration_bank, spec, target)
            row = {"target_total_rate": target, "checkpoint": stage + 1, "calibration": cal}
            print("calibration", tag, cal.get("gain"), cal.get("achieved_calibration_rate"), flush=True)
            if not cal["feasible"]:
                results.append(row)
                result_path.write_text(json.dumps(row, indent=2) + "\n")
                continue
            responses, errors, total_rates = [], [], []
            for index, stimulus in enumerate(test_bank):
                rr, ee, tt = [], [], []
                for repeat in range(spec["noise_repeats_per_template"]):
                    seed = spec["seed"] + 10000 + 100 * index + repeat
                    spikes, rates, truth = simulate(model, stage, stimulus, cal["gain"], seed)
                    spikes, rates, truth = spikes[:, burn:], rates[:, burn:], truth[:, burn:]
                    prediction = decoder.T @ rates
                    re = np.var(truth - prediction, axis=1).sum() / np.var(truth, axis=1).sum()
                    rr.append(spikes)
                    ee.append(float(re))
                    tt.append(float(spikes.sum()))
                responses.append(rr)
                errors.append(ee)
                total_rates.append(tt)
            responses = np.asarray(responses)
            row["actual_test_total_rate"] = float(np.mean(total_rates))
            row["budget_test_relative_error"] = abs(row["actual_test_total_rate"] - target) / target
            row["budget_test_within_tolerance"] = row["budget_test_relative_error"] <= spec["evaluation_budget_tolerance"]
            row["waveform_re_mean"] = float(np.mean(errors))
            row["waveform_re_by_template_repeat"] = errors
            row["rates_by_template_repeat"] = total_rates
            row["distributions"] = []
            for group, subset in (("all", responses), ("excitatory", responses[:, :, :model["n_e"]]),
                                  ("inhibitory", responses[:, :, model["n_e"]:])):
                for window in [spec["primary_window_ms"]] + spec["sensitivity_windows_ms"]:
                    counts = counts_from_stage(subset, window)
                    p = histogram(counts, int(np.ceil(window / 1.1)))
                    row["distributions"].append({"group": group, "window_ms": window, **metrics(p)})
            np.savez_compressed(HERE / (tag + "_responses.npz"),
                                packed_spikes=np.packbits(responses, axis=-1, bitorder="little"),
                                decoder=decoder, n_bins=np.array(1000),
                                rates=np.asarray(total_rates), errors=np.asarray(errors))
            result_path.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
            results.append(row)
            primary = row["distributions"][0]
            print(tag, "test_rate", row["actual_test_total_rate"], "deficit", primary["deficit_nats"],
                  "RE", row["waveform_re_mean"], "elapsed", time.monotonic() - started, flush=True)
    payload = {"protocol_sha256": sha(HERE / "matched_budget_protocol.json"),
               "checkpoint_sha256": sha(ROOT / "Fig4/checkpoint.mat"),
               "simulator_matches_existing_port": True, "results": results}
    (HERE / "matched_budget_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    snapshot()


if __name__ == "__main__":
    run()
