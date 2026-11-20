"""Recompute saved-response statistics and replay one independent training pair."""
import json

import numpy as np

from frontier import HERE, ROOT, counts_from_stage, histogram, metrics, sha
from controlled_budget import simulate, calibrate, predict, error
from replicate_learning import initial_model, generate_frozen_stimuli


def main():
    spec = json.loads((HERE / "replication_protocol.json").read_text())
    cases = json.loads((HERE / "replication/results.json").read_text())["results"]
    maximum_error = 0.
    for case in cases:
        tag = f"seed_{case['seed']}_budget_{case['target_total_rate']}"
        with np.load(HERE / "replication" / (tag + "_responses.npz"), allow_pickle=False) as cache:
            for index, row in enumerate(case["checkpoints"]):
                spikes = np.unpackbits(cache["packed_spikes"][index], axis=-1,
                                       count=int(cache["n_bins"]), bitorder="little")
                actual = spikes.sum(axis=(2,3)) / 2.
                np.testing.assert_allclose(actual, row["rate_by_template_repeat"], atol=0, rtol=0)
                for item in row["distributions"]:
                    subset = spikes if item["group"] == "all" else (
                        spikes[:, :, :60] if item["group"] == "excitatory" else spikes[:, :, 60:])
                    window = item["window_ms"]
                    counts = counts_from_stage(subset, window)
                    recalculated = metrics(histogram(counts, int(np.ceil(window / 1.1))))
                    for key in ("mean_count", "entropy_nats", "maximum_entropy_nats", "deficit_nats", "cv_pooled"):
                        difference = abs(recalculated[key] - item[key])
                        maximum_error = max(maximum_error, difference)
                        assert difference < 2e-10, (tag, index, key, difference)
        print("raw_response_check", tag, flush=True)
    selected = next(c for c in cases if c["seed"] == spec["training_seeds"][0] and c["target_total_rate"] == 480)
    tag = f"seed_{selected['seed']}_budget_480"
    model = initial_model(selected["seed"])
    with np.load(HERE / "replication" / (tag + "_weights.npz"), allow_pickle=False) as weights:
        for name in ("feedforward_e", "omega_ee", "omega_ei", "omega_ii", "omega_ie"):
            model[name] = np.ascontiguousarray(weights[name])
    bank = generate_frozen_stimuli(model, 16, 25000, spec["evaluation_seed"])
    replay_checks = []
    original = np.load(HERE / "replication" / (tag + "_responses.npz"), allow_pickle=False)
    for index in (0, len(selected["checkpoints"]) - 1):
        rates, truth = [], []
        for j, stimulus in enumerate(bank[:6]):
            _, r, x, _ = simulate(model, index, stimulus, spec["evaluation_seed"] + 1000 + j,
                                  {"target": 480, "controller_strength": .12})
            rates.append(r[:,500:])
            truth.append(x[:,500:])
        w, b, _ = calibrate(rates, truth)
        errors, total = [], []
        for j, stimulus in enumerate(bank[6:]):
            for repeat in range(8):
                spikes, r, x, _ = simulate(model, index, stimulus,
                    spec["evaluation_seed"] + 10000 + 100*j + repeat,
                    {"target": 480, "controller_strength": .12})
                spikes, r, x = spikes[:,500:], r[:,500:], x[:,500:]
                np.testing.assert_array_equal(np.packbits(spikes, axis=-1, bitorder="little"),
                                              original["packed_spikes"][index,j,repeat])
                errors.append(error(x, predict(w,b,r)))
                total.append(spikes.sum() / 2.)
        expected = selected["checkpoints"][index]
        np.testing.assert_allclose(np.mean(errors), expected["waveform_re"], atol=2e-10, rtol=0)
        np.testing.assert_allclose(np.mean(total), expected["actual_total_rate"], atol=0, rtol=0)
        replay_checks.append({"step": expected["step"], "waveform_re": float(np.mean(errors)),
                              "all_80_replayed_rasters_match": True})
        print("independent_replay", index, replay_checks[-1], flush=True)
    # The public numerical check does not depend on private submission files.
    result = {"cases": len(cases), "checkpoints_recomputed": len(cases) * 5,
              "distribution_statistics_recomputed": len(cases) * 5 * 18,
              "maximum_statistic_error": maximum_error,
              "independently_replayed_endpoints": replay_checks,
              "scope": "Archived synthetic responses and endpoint replay; no manuscript-file audit"}
    (HERE / "experiment_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
