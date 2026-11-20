"""Independent local-plasticity SNNs under a common total-activity constraint."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import time

import numpy as np
from numba import njit
from scipy.signal import convolve

from frontier import HERE, ROOT, counts_from_stage, histogram, metrics, sha, snapshot
from controlled_budget import calibrate, simulate, error, predict
from network_budget import generate_frozen_stimuli


OUT = HERE / "replication"


def initial_model(seed, ne=60, ni=15):
    rng = np.random.default_rng(seed)
    fe = rng.standard_normal((3, ne))
    fe /= np.sqrt((fe ** 2).sum(axis=0))
    ei = .5 * np.tile(np.eye(ni), (1, 4))
    kernel = np.exp(-((np.arange(1, 1001) - 500) ** 2) / (2 * 30. ** 2))
    kernel /= kernel.sum()
    return {"nx": 3, "dt": .0001, "leak": 50., "n_e": ne, "n_i": ni,
            "threshold_e": np.full(ne, .5), "threshold_i": np.full(ni, .5),
            "amplitude": 2000., "kernel": kernel,
            "feedforward_e": fe[None], "omega_ee": (-.02 * np.eye(ne))[None],
            "omega_ei": ei[None], "omega_ii": (-.5 * np.eye(ni))[None],
            "omega_ie": (-.3 * ei.T)[None]}


@njit(cache=True)
def train(fe, ee, ei, ii, ie, bank, seed, target, checkpoints,
          epsf=1e-5, epsr=1e-4, controller=.12):
    np.random.seed(seed)
    ne, ni = fe.shape[1], ei.shape[0]
    ns = len(checkpoints)
    fs, ees, eis = np.zeros((ns, 3, ne)), np.zeros((ns, ne, ne)), np.zeros((ns, ni, ne))
    iis, ies = np.zeros((ns, ni, ni)), np.zeros((ns, ne, ni))
    fs[0], ees[0], eis[0], iis[0], ies[0] = fe, ee, ei, ii, ie
    ve, vi = np.zeros(ne), np.zeros(ni)
    r_e, r_i, bar_e, bar_i = np.zeros(ne), np.zeros(ni), np.zeros(3), np.zeros(ne)
    refractory_e, refractory_i = np.full(ne, -1), np.full(ni, -1)
    winner_e, winner_i, previous_e, previous_i, template = 0, 0, 0, 0, 0
    log_gain, gain, block_spikes, second_spikes = 0., 1., 0, 0
    history = np.zeros((int(checkpoints[-1] // 10000), 2))
    stage, history_index = 1, 0
    for t in range(1, int(checkpoints[-1]) + 1):
        if (t - 1) % 1000 == 0:
            template = np.random.randint(0, bank.shape[0])
        time = (t - 1) % 1000
        for j in range(3):
            bar_e[j] = .97 * bar_e[j] + .0001 * bank[template, j, time]
        for n in range(ne):
            bar_i[n] *= .995
        bar_i[winner_e] += previous_e
        best, ke = -1e300, 0
        for n in range(ne):
            drive = .0001 * (fe[0, n] * bank[template, 0, time]
                             + fe[1, n] * bank[template, 1, time]
                             + fe[2, n] * bank[template, 2, time])
            ve[n] = .995 * ve[n] + drive + previous_e * ee[n, winner_e] + previous_i * ie[n, winner_i] + .002 * np.random.randn()
            score = ve[n] - .5 * gain - .02 * np.random.randn()
            if score > best:
                best, ke = score, n
        winner_e, current_e = ke, 0
        if best >= 0 and refractory_e[ke] < 0:
            current_e, refractory_e[ke] = 1, 10
            for j in range(3):
                fe[j, ke] += epsf * (bar_e[j] - fe[j, ke])
            for n in range(ne):
                ee[n, ke] = max(ee[n, ke] - epsr * (ve[n] + .002 * r_e[n] + ee[n, ke]), 0.)
            ee[ke, ke] = -.02
        refractory_e -= 1
        best, ki = -1e300, 0
        for n in range(ni):
            vi[n] = .995 * vi[n] + current_e * ei[n, ke] + previous_i * ii[n, winner_i] + .002 * np.random.randn()
            score = vi[n] - .5 * gain - .02 * np.random.randn()
            if score > best:
                best, ki = score, n
        winner_i, current_i = ki, 0
        if best >= 0 and refractory_i[ki] < 0:
            current_i, refractory_i[ki] = 1, 10
            for n in range(ne):
                ei[ki, n] = max(ei[ki, n] + epsf * (bar_i[n] - ei[ki, n]), 0.)
            for n in range(ni):
                ii[n, ki] = min(ii[n, ki] - epsr * (vi[n] + .002 * r_i[n] + ii[n, ki] + .002 * (n == ki)), 0.)
            for n in range(ne):
                ie[n, ki] = min(ie[n, ki] - epsr * (ve[n] + .002 * r_e[n] + ie[n, ki]), 0.)
        r_e[ke] += current_e
        r_i[ki] += current_i
        r_e *= .995
        r_i *= .995
        refractory_i -= 1
        block_spikes += current_e + current_i
        second_spikes += current_e + current_i
        if t % 200 == 0:
            log_gain += controller * (block_spikes / (target * .02) - 1)
            log_gain = min(np.log(20.), max(np.log(.03), log_gain))
            gain, block_spikes = np.exp(log_gain), 0
        if t % 10000 == 0:
            history[history_index, 0], history[history_index, 1] = second_spikes, gain
            history_index += 1
            second_spikes = 0
        if stage < ns and t == checkpoints[stage]:
            fs[stage], ees[stage], eis[stage], iis[stage], ies[stage] = fe, ee, ei, ii, ie
            stage += 1
        previous_e, previous_i = current_e, current_i
    return fs, ees, eis, iis, ies, history


def training_bank(model, seed):
    rng = np.random.default_rng(seed + 100000)
    raw = rng.standard_normal((128, 3, 1000)) * 2000
    return np.ascontiguousarray(np.array([
        [convolve(x, model["kernel"], mode="same") for x in sample] for sample in raw
    ]))


def one_case(seed, target):
    spec = json.loads((HERE / "replication_protocol.json").read_text())
    code_hash, protocol_hash = sha(__file__), sha(HERE / "replication_protocol.json")
    tag = f"seed_{seed}_budget_{target}"
    dest, weights_path = OUT / (tag + ".json"), OUT / (tag + "_weights.npz")
    if dest.exists():
        saved = json.loads(dest.read_text())
        assert saved["code_sha256"] == code_hash and saved["protocol_sha256"] == protocol_hash
        return saved
    started = time.monotonic()
    model = initial_model(seed)
    checkpoints = np.asarray(spec["checkpoint_steps"], dtype=np.int64)
    names = ["feedforward_e", "omega_ee", "omega_ei", "omega_ii", "omega_ie"]
    if weights_path.exists():
        cached = np.load(weights_path, allow_pickle=False)
        assert str(cached["code_sha256"]) == code_hash and str(cached["protocol_sha256"]) == protocol_hash
        arrays = [cached[name] for name in names]
        history = cached["training_history"]
    else:
        bank = training_bank(model, seed)
        arrays = train(*(model[name][0].copy() for name in names), bank, int(seed),
                       float(target), checkpoints, spec["feedforward_learning_rate"],
                       spec["recurrent_learning_rate"], spec["controller_strength"])
        history, arrays = arrays[-1], arrays[:-1]
        assert all(np.isfinite(a).all() for a in arrays)
        assert np.all(arrays[2] >= 0) and np.all(arrays[3] <= 0) and np.all(arrays[4] <= 0)
        for ee in arrays[1]:
            off = ee.copy()
            np.fill_diagonal(off, 0)
            assert (off >= 0).all()
            np.testing.assert_allclose(np.diag(ee), -.02)
        np.savez_compressed(weights_path, **dict(zip(names, arrays)), training_history=history,
                            code_sha256=np.array(code_hash), protocol_sha256=np.array(protocol_hash))
    for name, array in zip(names, arrays):
        model[name] = np.ascontiguousarray(array)
    print("trained", tag, f"{time.monotonic() - started:.1f}s", flush=True)
    bank = generate_frozen_stimuli(model, spec["calibration_templates"] + spec["test_templates"],
                                  int(round(spec["duration_s"] / model["dt"])), spec["evaluation_seed"])
    calibration_bank, test_bank = bank[:spec["calibration_templates"]], bank[spec["calibration_templates"]:]
    burn = int(round(spec["burn_in_s"] * 1000))
    duration = spec["duration_s"] - spec["burn_in_s"]
    evaluation = {"target": target, "controller_strength": spec["controller_strength"]}
    results, packed = [], []
    for stage, step in enumerate(checkpoints):
        rates, truth = [], []
        for index, stimulus in enumerate(calibration_bank):
            _, r, x, _ = simulate(model, stage, stimulus, spec["evaluation_seed"] + 1000 + index, evaluation)
            rates.append(r[:, burn:])
            truth.append(x[:, burn:])
        w, b, decoder = calibrate(rates, truth)
        responses, errors, totals = [], [], []
        for index, stimulus in enumerate(test_bank):
            rr, ee, tt = [], [], []
            for repeat in range(spec["noise_repeats_per_template"]):
                test_seed = spec["evaluation_seed"] + 10000 + 100 * index + repeat
                spikes, r, x, _ = simulate(model, stage, stimulus, test_seed, evaluation)
                spikes, r, x = spikes[:, burn:], r[:, burn:], x[:, burn:]
                rr.append(spikes)
                ee.append(error(x, predict(w, b, r)))
                tt.append(float(spikes.sum()) / duration)
            responses.append(rr)
            errors.append(ee)
            totals.append(tt)
        responses = np.asarray(responses)
        actual = float(np.mean(totals))
        row = {"step": int(step), "time_s": float(step * model["dt"]),
               "actual_total_rate": actual, "relative_budget_error": abs(actual - target) / target,
               "budget_within_tolerance": abs(actual - target) / target <= .05,
               "waveform_re": float(np.mean(errors)), "readout_alpha": decoder["alpha"],
               "waveform_re_by_template_repeat": errors, "rate_by_template_repeat": totals,
               "distributions": []}
        for group, subset in (("all", responses), ("excitatory", responses[:, :, :model["n_e"]]),
                              ("inhibitory", responses[:, :, model["n_e"]:])):
            for window in [spec["primary_window_ms"]] + spec["sensitivity_windows_ms"]:
                counts = counts_from_stage(subset, window)
                row["distributions"].append({"group": group, "window_ms": window,
                    **metrics(histogram(counts, int(np.ceil(window / 1.1))))})
        results.append(row)
        packed.append(np.packbits(responses, axis=-1, bitorder="little"))
    result = {"seed": seed, "target_total_rate": target, "code_sha256": code_hash,
              "protocol_sha256": protocol_hash, "network_size": [60,15],
              "training_mean_activity": float(history[10:, 0].mean()),
              "training_budget_error": float(abs(history[10:, 0].mean() - target) / target),
              "all_evaluation_budgets_valid": all(x["budget_within_tolerance"] for x in results),
              "checkpoints": results, "elapsed_seconds": time.monotonic() - started}
    np.savez_compressed(OUT / (tag + "_responses.npz"), packed_spikes=np.asarray(packed),
                        n_bins=np.array(2000), checkpoint_steps=checkpoints)
    dest.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("completed", tag, "RE", results[0]["waveform_re"], "->", results[-1]["waveform_re"],
          "gap", results[0]["distributions"][0]["deficit_nats"], "->", results[-1]["distributions"][0]["deficit_nats"],
          "budget_valid", result["all_evaluation_budgets_valid"],
          f"{time.monotonic() - started:.1f}s", flush=True)
    return result


def validate_kernel():
    model = initial_model(123)
    bank = training_bank(model, 123)[:2]
    names = ["feedforward_e", "omega_ee", "omega_ei", "omega_ii", "omega_ie"]
    source = [model[name][0].copy() for name in names]
    unchanged = train(*(a.copy() for a in source), bank, 123, 480., np.array([0,512]), 0., 0.)
    for before, after in zip(source, unchanged[:5]):
        np.testing.assert_array_equal(before, after[-1])
    updated = train(*(a.copy() for a in source), bank, 123, 480., np.array([0,4096]))
    assert all(np.isfinite(x).all() for x in updated)
    assert not np.array_equal(updated[0][-1], source[0])
    return {"no_plasticity_weights_unchanged": True, "plasticity_updates_finite": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    snapshot()
    OUT.mkdir(exist_ok=True)
    print("kernel_validation", validate_kernel(), flush=True)
    if args.validate_only:
        return
    spec = json.loads((HERE / "replication_protocol.json").read_text())
    cases = [(seed, budget) for seed in spec["training_seeds"] for budget in spec["target_total_spikes_per_second"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one_case, *case): case for case in cases}
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda row: (row["seed"], row["target_total_rate"]))
    (OUT / "results.json").write_text(json.dumps({"protocol_sha256": sha(HERE / "replication_protocol.json"),
                                                 "results": results}, indent=2, allow_nan=False) + "\n")
    snapshot()


if __name__ == "__main__":
    main()
