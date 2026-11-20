"""Independent checks of the resampling and exact achieved-mean comparisons."""
from collections import Counter
from itertools import combinations
import json

import numpy as np

from sensitivity import BUDGETS, CELLS, HERE, K, LN2, OLD, PROTOCOL, ROOT, SEEDS, STAGES, metrics, sha


def frozen_counts(seed, budget):
    path = OLD / "replication" / f"seed_{seed}_budget_{budget}_responses.npz"
    with np.load(path, allow_pickle=False) as data:
        raster = np.unpackbits(data["packed_spikes"][STAGES], axis=-1, count=2000, bitorder="little")
    # Sum the two halves independently instead of using the analysis reshape.
    return np.stack([raster[..., :1000].sum(axis=-1), raster[..., 1000:].sum(axis=-1)], axis=-1)


def direct_law(counts):
    frequencies = Counter(map(int, np.ravel(counts)))
    p = np.zeros(K + 1)
    for k, n in frequencies.items():
        p[k] = n / counts.size
    return p


def direct_delta(counts):
    endpoints = [metrics(direct_law(a)) for a in counts]
    d = np.array([a["deficit_nats"] / LN2 for a in endpoints])
    g = d / [a["mean_count"] for a in endpoints]
    return np.array([d[1] - d[0], g[1] - g[0]])


def main():
    result = json.loads((HERE / "results.json").read_text())
    assert result["with_replay"]
    assert result["code_sha256"] == sha(HERE / "sensitivity.py")
    assert result["protocol_sha256"] == sha(HERE / "protocol.json")
    for path, expected in result["input_hashes"].items():
        assert sha(ROOT / path) == expected, path
    rng = np.random.default_rng(PROTOCOL["sampling_seed"])
    first = rng.multinomial(8, np.full(8, .125), size=(PROTOCOL["bootstrap_draws"], 10))[0]
    repeat = [np.repeat(np.arange(8), w) for w in first]
    subsets = list(combinations(range(10), 5))
    max_residual, max_gap_error = 0., 0.
    for si, seed in enumerate(SEEDS):
        for bi, budget in enumerate(BUDGETS):
            counts = frozen_counts(seed, budget)
            with np.load(HERE / "sampling_draws.npz", allow_pickle=False) as draws:
                chosen = np.stack([counts[:, t, repeat[t]] for t in range(10)], axis=1)
                np.testing.assert_allclose(direct_delta(chosen), draws["repeat_bootstrap"][0, si, bi],
                                           atol=5e-12, rtol=0)
                subset = subsets[-1]
                np.testing.assert_allclose(direct_delta(counts[:, subset]),
                                           draws["five_template_subset"][-1, si, bi], atol=5e-12, rtol=0)
            replay = json.loads((HERE / "replay" / f"seed_{seed}_budget_{budget}.json").read_text())
            row = next(r for r in result["cases"] if (r["seed"], r["budget"], r["control"]) ==
                       (seed, budget, "replay_mixture"))
            assert row["status"] == "ok"
            entropy = []
            for i, stage in enumerate(replay["stages"]):
                assert stage["bracketed"] and len(stage["attempts"]) == 1
                original = direct_law(counts[i])
                attempt = stage["attempts"][0]
                h = np.array(attempt["histogram"])
                q = np.pad(h / h.sum(), (0, K + 1 - len(h)))
                # Solve the equal-rate mixture using recorded rates, independently
                # of the vectorized count-law metric code.
                weight = (budget - stage["base_rate"]) / (attempt["achieved_total_rate"] - stage["base_rate"])
                p = original * (1 - weight) + q * weight
                measured = np.dot(np.arange(K + 1), p)
                max_residual = max(max_residual, abs(measured - budget / CELLS))
                entropy.append(float(-p[p > 0] @ np.log2(p[p > 0])))
            # At equal mean, H(q*) cancels, so delta D is simply H_initial-H_final.
            exact_delta = entropy[0] - entropy[1]
            max_gap_error = max(max_gap_error, abs(exact_delta - row["deltas"]["D_bits_per_response"]))
            np.testing.assert_allclose(exact_delta / (budget / CELLS), row["deltas"]["G_bits_per_spike"],
                                       atol=5e-12, rtol=0)
    assert max_residual < 2e-11
    report = {"archived_endpoint_count": 48, "verified_input_hashes": len(result["input_hashes"]),
              "direct_repeat_bootstrap_checks": 24, "direct_template_subset_checks": 24,
              "nominal_replay_spike_raster_equality_checks": 48,
              "perturbed_replay_trials": 48 * 80, "exact_nominal_rate_matches": 48,
              "maximum_achieved_rate_residual_spikes_s_cell": max_residual,
              "maximum_equal_cost_entropy_difference_error_bits": max_gap_error,
              "maximum_original_endpoint_error_bits": result["maximum_frozen_D_error_bits"]}
    (HERE / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
