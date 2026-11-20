"""Paired sampling and achieved-cost sensitivity for the archived 1-s endpoint assay."""
import argparse
import csv
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp, xlogy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "outputs/snn_budget_frontier_20261016"
sys.path.insert(0, str(OLD))
from frontier import metrics, sha, thin

PROTOCOL = json.loads((HERE / "protocol.json").read_text())
SEEDS, BUDGETS, STAGES = (PROTOCOL[k] for k in ("seeds", "budgets", "stages"))
CELLS = PROTOCOL["cells"]
K = int(np.ceil(PROTOCOL["window_ms"] / 1.1))
LN2 = np.log(2)
METRICS = ("D_bits_per_response", "G_bits_per_spike")


def reference_entropy(mu):
    """Exact finite geometric entropy in nats, batched over achieved means."""
    mu = np.asarray(mu, dtype=float)
    if np.any(~np.isfinite(mu) | (mu <= 0) | (mu >= K / 2)):
        raise ValueError("Need positive means on the active finite-support branch")
    lo, hi = np.zeros_like(mu), np.log1p(1 / mu)
    for _ in range(48):
        beta = (lo + hi) / 2
        with np.errstate(over="ignore"):
            mean = 1 / np.expm1(beta) - (K + 1) / np.expm1((K + 1) * beta)
        lo, hi = np.where(mean > mu, beta, lo), np.where(mean > mu, hi, beta)
    beta = (lo + hi) / 2
    return beta * mu + np.log(-np.expm1(-(K + 1) * beta)) - np.log(-np.expm1(-beta))


def statistics(p):
    if p.shape[-1] > K + 1 or np.any(p < 0) or not np.allclose(p.sum(axis=-1), 1, atol=1e-12, rtol=0):
        raise ValueError("Invalid count law")
    mu = p @ np.arange(p.shape[-1], dtype=float)
    deficit = (reference_entropy(mu) + np.sum(xlogy(p, p), axis=-1)) / LN2
    if np.any(deficit < -1e-10) or not np.isfinite(deficit).all():
        raise ValueError("Invalid entropy deficit")
    return mu, np.stack([deficit, deficit / mu], axis=-1)


def reweight(p, target):
    k = np.flatnonzero(p)
    if not k[0] < target < k[-1]:
        raise ValueError("Tilt target must be inside observed support")
    logp = np.log(p[k])

    def tilted(lam):
        x = logp + lam * k
        return np.exp(x - logsumexp(x))

    scale = 1.
    while not tilted(-scale) @ k < target < tilted(scale) @ k:
        scale *= 2
    lam = brentq(lambda x: tilted(x) @ k - target, -scale, scale, xtol=1e-14)
    result = np.zeros_like(p)
    result[k] = tilted(lam)
    return result, lam


def mix_at_rate(p, q, target):
    k = np.arange(len(p))
    a, b = float(p @ k), float(q @ k)
    if not min(a, b) <= target <= max(a, b) or a == b:
        raise ValueError("No achieved-rate bracket")
    weight = (target - a) / (b - a)
    return (1 - weight) * p + weight * q, weight


def self_test():
    means = [.01, .1, 1., 3.2, 6.4, 9.6, 25., 100., 450.]
    from frontier import gibbs, entropy
    expected = [entropy(gibbs(mu, K)[0]) for mu in means]
    np.testing.assert_allclose(reference_entropy(means), expected, atol=5e-12, rtol=0)
    p = np.array([.2, .3, .3, .2])
    target = 1.25
    q, _ = reweight(p, target)
    k = np.arange(4)
    np.testing.assert_allclose(q @ k, target, atol=1e-12, rtol=0)
    np.testing.assert_allclose(thin(p, target / (p @ k)) @ k, target, atol=1e-12, rtol=0)
    r, weight = mix_at_rate(p, q, (p @ k + target) / 2)
    np.testing.assert_allclose(r, (p + q) / 2, atol=1e-12, rtol=0)
    assert abs(weight - .5) < 1e-12


def load_histograms():
    prior = json.loads((ROOT / "outputs/elife_reconsideration_20261109/analysis/source_manifest.json").read_text())
    manifest, histograms, targets = {}, {}, []
    for si, seed in enumerate(SEEDS):
        for bi, budget in enumerate(BUDGETS):
            tag = f"seed_{seed}_budget_{budget}"
            folder = OLD / "replication"
            raw, metadata = folder / f"{tag}_responses.npz", folder / f"{tag}.json"
            for path in (raw, metadata):
                name = str(path.relative_to(ROOT))
                manifest[name] = sha(path)
                assert manifest[name] == prior["files"][name], name
            case = json.loads(metadata.read_text())
            with np.load(raw, allow_pickle=False) as cached:
                packed = cached["packed_spikes"]
                assert packed.shape == (5, 10, 8, 75, 250)
                assert int(cached["n_bins"]) == 2000
                spikes = np.unpackbits(packed[STAGES], axis=-1, count=2000, bitorder="little")
            counts = spikes.reshape(2, 10, 8, CELLS, 2, 1000).sum(axis=-1).astype(int)
            for ei, stage in enumerate(STAGES):
                checkpoint = case["checkpoints"][stage]
                np.testing.assert_array_equal(counts[ei].sum(axis=(2, 3)) / 2,
                                              checkpoint["rate_by_template_repeat"])
                target = next(d for d in checkpoint["distributions"]
                              if d["group"] == "all" and d["window_ms"] == 1000)
                targets.append((si, bi, ei, target["mean_count"], target["deficit_nats"] / LN2))
            histograms[si, bi] = [np.bincount(trial, minlength=K + 1)
                                  for trial in counts.reshape(-1, 2 * CELLS)]
    h = np.array([histograms[si, bi] for si in range(len(SEEDS)) for bi in range(len(BUDGETS))])
    h = h.reshape(len(SEEDS), len(BUDGETS), 2, 10, 8, K + 1)
    assert np.all(h.sum(axis=-1) == 2 * CELLS)
    last = int(np.max(np.flatnonzero(h.sum(axis=(0, 1, 2, 3, 4)))))
    h = np.ascontiguousarray(h[..., :last + 1])
    p = h.mean(axis=(3, 4)) / (2 * CELLS)
    mu, values = statistics(p)
    error = 0.
    for si, bi, ei, saved_mu, saved_d in targets:
        np.testing.assert_allclose(mu[si, bi, ei], saved_mu, atol=1e-12, rtol=0)
        error = max(error, abs(values[si, bi, ei, 0] - saved_d))
    assert error < 3e-12
    return h, p, mu, values, manifest, error


def resampled_deltas(h, weights):
    # Each weight acts on one complete 2-s trial, including all neurons and both windows.
    rows = h.transpose(3, 4, 0, 1, 2, 5).reshape(80, -1)
    draws = []
    for w in np.array_split(weights.reshape(-1, 80), max(1, len(weights) // 100)):
        totals = w.astype(np.int64) @ rows
        samples = w.sum(axis=1, keepdims=True) * 2 * CELLS
        p = (totals.reshape(len(w), -1, h.shape[-1]) / samples[..., None])
        assert np.array_equal(totals.reshape(len(w), -1, h.shape[-1]).sum(axis=-1),
                              np.broadcast_to(samples, (len(w), np.prod(h.shape[:3]))))
        p = p.reshape(len(w), *h.shape[:3], h.shape[-1])
        _, values = statistics(p)
        draws.append(values[:, :, :, 1] - values[:, :, :, 0])
    return np.concatenate(draws)


def resampling(h, observed):
    rng = np.random.default_rng(PROTOCOL["sampling_seed"])
    n = PROTOCOL["bootstrap_draws"]
    # Same test bank and seed streams are used across training runs and budgets.
    repeat_weights = rng.multinomial(8, np.full(8, 1 / 8), size=(n, 10))
    repeat_delta = resampled_deltas(h, repeat_weights)
    seed_ids = rng.integers(0, len(SEEDS), size=(n, len(SEEDS)))
    joint = repeat_delta[np.arange(n)[:, None], seed_ids].mean(axis=1)
    leave_one = 1 - np.eye(10, dtype=int)
    template_half = np.array([np.isin(np.arange(10), subset) for subset in combinations(range(10), 5)])
    order = np.argsort(rng.random((PROTOCOL["half_repeat_draws"], 10, 8)), axis=-1)
    half = (order < 4).astype(int)
    half_a = resampled_deltas(h, half)
    half_b = resampled_deltas(h, 1 - half)
    half_avg = (half_a + half_b) / 2
    arrays = {
        "repeat_bootstrap": repeat_delta,
        "seed_bootstrap": observed[seed_ids].mean(axis=1),
        "joint_seed_repeat_bootstrap": joint,
        "leave_one_template": resampled_deltas(h, np.repeat(leave_one[..., None], 8, axis=-1)),
        "five_template_subset": resampled_deltas(h, np.repeat(template_half[..., None], 8, axis=-1)),
        "four_repeat_subset": np.concatenate([half_a, half_b]),
        "half_minus_full": half_avg - observed,
        "half_full_extrapolation": 2 * observed - half_avg,
    }
    summary = {}
    for name, array in arrays.items():
        budget_means = array.mean(axis=1) if array.ndim == 4 else array
        summary[name] = describe_draws(budget_means)
    return arrays, summary


def describe_draws(array):
    return [{metric: {"mean": float(x.mean()), "range": [float(x.min()), float(x.max())],
                      "percentile_95": np.quantile(x, [.025, .975]).tolist(),
                      "fraction_decreasing": float(np.mean(x < 0)),
                      "fraction_increasing": float(np.mean(x > 0)), "draws": len(x)}
             for mi, metric in enumerate(METRICS) for x in [array[:, bi, mi]]}
            for bi in range(len(BUDGETS))]


def cost_controls(p, mu, observed, include_replay, manifest):
    rows = []
    for si, seed in enumerate(SEEDS):
        for bi, budget in enumerate(BUDGETS):
            orig = p[si, bi]
            means = mu[si, bi]
            lower, mid = float(means.min()), float(means.mean())
            retention = lower / means
            thinned = np.stack([thin(q, rate) for q, rate in zip(orig, retention)])
            tilted, lam = zip(*(reweight(q, mid) for q in orig))
            variants = [
                ("original", orig, None, {}),
                ("thinning", thinned, lower, {"retention": retention.tolist()}),
                ("tilt", np.stack(tilted), mid, {"tilt_parameter": list(lam)}),
            ]
            if include_replay:
                path = HERE / "replay" / f"seed_{seed}_budget_{budget}.json"
                replay = json.loads(path.read_text())
                manifest[str(path.relative_to(ROOT))] = sha(path)
                for rel, expected in replay["signature"]["inputs"].items():
                    if rel in manifest:
                        assert manifest[rel] == expected, rel
                    else:
                        assert sha(ROOT / rel) == expected, rel
                        manifest[rel] = expected
                assert replay["signature"]["protocol_sha256"] == sha(HERE / "protocol.json")
                assert replay["signature"]["code_sha256"] == sha(HERE / "replay_cost.py")
                nominal = budget / CELLS
                mixtures, bracket_values = [], []
                for ei, stage in enumerate(replay["stages"]):
                    np.testing.assert_allclose(stage["base_rate"], means[ei] * CELLS, atol=1e-10, rtol=0)
                    if not stage["bracketed"]:
                        break
                    if not stage["attempts"]:
                        mixtures.append(orig[ei])
                        bracket_values.append({"perturbation_weight": 0., "attempts": 0})
                        continue
                    attempt = stage["attempts"][-1]
                    counts = np.array(attempt["histogram"], dtype=float)
                    assert counts.sum() == 12000
                    size = max(len(counts), orig.shape[-1])
                    a = np.pad(orig[ei], (0, size - orig.shape[-1]))
                    b = np.pad(counts / counts.sum(), (0, size - len(counts)))
                    mixture, weight = mix_at_rate(a, b, nominal)
                    mixtures.append(mixture)
                    _, bracket = statistics(np.stack([a, b]))
                    bracket_values.append({"achieved_total_rate": attempt["achieved_total_rate"],
                                           "command": attempt["command"], "perturbation_weight": weight,
                                           "attempts": len(stage["attempts"]), "endpoint_metrics": bracket.tolist()})
                if len(mixtures) == 2:
                    size = max(map(len, mixtures))
                    mix = np.stack([np.pad(q, (0, size - len(q))) for q in mixtures])
                    variants.append(("replay_mixture", mix, nominal, {"brackets": bracket_values}))
                else:
                    rows.append({"seed": seed, "budget": budget, "control": "replay_mixture",
                                 "status": "unbracketed"})
            for name, law, target, detail in variants:
                rates, values = statistics(law)
                if target is not None:
                    np.testing.assert_allclose(rates, target, atol=2e-11, rtol=0)
                delta = values[1] - values[0]
                if name == "original":
                    np.testing.assert_allclose(delta, observed[si, bi], atol=1e-12, rtol=0)
                rows.append({"seed": seed, "budget": budget, "control": name, "status": "ok",
                             "achieved_rate_per_cell": rates.tolist(), "endpoint_metrics": values.tolist(),
                             "deltas": dict(zip(METRICS, delta.tolist())), **detail})
    return rows


def summarize_cost(rows):
    result = {}
    for budget in BUDGETS:
        cases = [r for r in rows if r["budget"] == budget]
        controls = {}
        for name in dict.fromkeys(r["control"] for r in cases):
            all_rows = [r for r in cases if r["control"] == name]
            good = [r for r in all_rows if r["status"] == "ok"]
            controls[name] = {"matched": len(good), "attempted": len(all_rows)}
            for metric in METRICS:
                x = np.array([r["deltas"][metric] for r in good])
                if len(x):
                    controls[name][metric] = {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
                                              "decreasing": int((x < 0).sum()), "increasing": int((x > 0).sum())}
        result[str(budget)] = controls
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-replay", action="store_true")
    args = parser.parse_args()
    self_test()
    h, p, mu, values, manifest, error = load_histograms()
    observed = values[:, :, 1] - values[:, :, 0]
    arrays, sample_summary = resampling(h, observed)
    rows = cost_controls(p, mu, observed, args.with_replay, manifest)
    result = {"protocol_sha256": sha(HERE / "protocol.json"), "code_sha256": sha(__file__),
              "replication_protocol_sha256": sha(OLD / "replication_protocol.json"),
              "finite_support_maximum": K, "maximum_frozen_D_error_bits": error,
              "with_replay": args.with_replay, "metrics": METRICS, "budget_order": BUDGETS,
              "sampling": sample_summary, "cost": summarize_cost(rows), "cases": rows,
              "input_hashes": manifest}
    np.savez_compressed(HERE / "sampling_draws.npz", **arrays)
    (HERE / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (HERE / "cost_by_seed.csv").open("w", newline="") as stream:
        columns = ["seed", "budget", "control", "status", "initial_rate", "learned_rate", *METRICS]
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for r in rows:
            rates = r.get("achieved_rate_per_cell", ["", ""])
            writer.writerow({**{k: r[k] for k in columns[:4]}, "initial_rate": rates[0],
                             "learned_rate": rates[1], **r.get("deltas", {})})
    print(json.dumps({k: result[k] for k in ("maximum_frozen_D_error_bits", "cost")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
