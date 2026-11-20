"""Summarize every prespecified budget/window, using trained seeds as units."""
import csv
from itertools import product
import json

import numpy as np

from frontier import HERE, sha


def sign_flip(values):
    d = np.asarray(values, dtype=float)
    signs = np.asarray(list(product((-1, 1), repeat=len(d))))
    null = np.sum(signs * d[None], axis=1)
    return float(np.mean(np.abs(null) >= abs(d.sum()) - 1e-12))


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.maximum.accumulate((len(order) - np.arange(len(order))) * np.asarray(pvalues)[order])
    result = np.empty(len(order))
    result[order] = np.minimum(1., adjusted)
    return result.tolist()


def distribution(checkpoint, group, window):
    return next(x for x in checkpoint["distributions"] if x["group"] == group and x["window_ms"] == window)


def summary(values, rng):
    values = np.asarray(values, dtype=float)
    bootstrap = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return {"mean_change": float(values.mean()), "sd_change": float(values.std(ddof=1)),
            "training_seed_bootstrap_95pct": np.quantile(bootstrap, [.025,.975]).tolist(),
            "negative_seed_count": int(np.sum(values < 0)), "positive_seed_count": int(np.sum(values > 0)),
            "changes": values.tolist()}


def main():
    spec = json.loads((HERE / "replication_protocol.json").read_text())
    data = json.loads((HERE / "replication/results.json").read_text())["results"]
    expected = {(s,b) for s in spec["training_seeds"] for b in spec["target_total_spikes_per_second"]}
    assert {(row["seed"], row["target_total_rate"]) for row in data} == expected
    assert len(data) == len(expected)
    assert all(row["protocol_sha256"] == sha(HERE / "replication_protocol.json") for row in data)
    rng = np.random.default_rng(20261028)
    rows, budgets, primary_p, reconstruction_p = [], [], [], []
    for budget in spec["target_total_spikes_per_second"]:
        cases = sorted([x for x in data if x["target_total_rate"] == budget], key=lambda x:x["seed"])
        primary_d, re_d = [], []
        for case in cases:
            before, after = case["checkpoints"][0], case["checkpoints"][-1]
            primary_d.append(distribution(after, "all", 1000)["deficit_nats"] - distribution(before, "all", 1000)["deficit_nats"])
            re_d.append(after["waveform_re"] - before["waveform_re"])
            for point in case["checkpoints"]:
                for item in point["distributions"]:
                    rows.append({"seed": case["seed"], "budget": budget, "step": point["step"],
                                 "time_s": point["time_s"], "actual_total_rate": point["actual_total_rate"],
                                 "budget_relative_error": point["relative_budget_error"],
                                 "waveform_re": point["waveform_re"], **item})
        item = {"target_total_rate": budget, "per_neuron_rate": budget / 75,
                "n_independent_training_seeds": len(cases),
                "all_budgets_valid": all(c["all_evaluation_budgets_valid"] for c in cases),
                "maximum_budget_error": max(p["relative_budget_error"] for c in cases for p in c["checkpoints"]),
                "mean_training_budget_error": float(np.mean([c["training_budget_error"] for c in cases])),
                "initial_actual_total_rate": float(np.mean([c["checkpoints"][0]["actual_total_rate"] for c in cases])),
                "final_actual_total_rate": float(np.mean([c["checkpoints"][-1]["actual_total_rate"] for c in cases])),
                "initial_re_mean": float(np.mean([c["checkpoints"][0]["waveform_re"] for c in cases])),
                "final_re_mean": float(np.mean([c["checkpoints"][-1]["waveform_re"] for c in cases])),
                "primary_entropy_change": summary(primary_d, rng), "reconstruction_change": summary(re_d, rng),
                "windows": []}
        primary_p.append(sign_flip(primary_d))
        reconstruction_p.append(sign_flip(re_d))
        for window in [spec["primary_window_ms"]] + spec["sensitivity_windows_ms"]:
            before = [distribution(c["checkpoints"][0], "all", window) for c in cases]
            after = [distribution(c["checkpoints"][-1], "all", window) for c in cases]
            changes = [a["deficit_nats"] - b["deficit_nats"] for a,b in zip(after,before)]
            item["windows"].append({"window_ms": window,
                "initial_deficit_mean": float(np.mean([x["deficit_nats"] for x in before])),
                "final_deficit_mean": float(np.mean([x["deficit_nats"] for x in after])),
                "initial_relative_deficit_mean": float(np.mean([x["relative_deficit"] for x in before])),
                "final_relative_deficit_mean": float(np.mean([x["relative_deficit"] for x in after])),
                "initial_cv_mean": float(np.mean([x["cv_pooled"] for x in before])),
                "final_cv_mean": float(np.mean([x["cv_pooled"] for x in after])),
                **summary(changes, rng)})
        budgets.append(item)
    for i, item in enumerate(budgets):
        item["primary_entropy_change"]["exact_sign_flip_p"] = primary_p[i]
        item["primary_entropy_change"]["holm_adjusted_p"] = holm(primary_p)[i]
        item["reconstruction_change"]["exact_sign_flip_p"] = reconstruction_p[i]
        item["reconstruction_change"]["holm_adjusted_p"] = holm(reconstruction_p)[i]
    with (HERE / "replication_measurements.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {"protocol_sha256": sha(HERE / "replication_protocol.json"),
              "case_count": len(data), "independent_training_seeds": 8, "budgets": budgets,
              "statistical_unit": "Independent training seed; budgets paired within seed",
              "secondary_windows": "Descriptive sensitivity, not a replacement primary test",
              "scope": "Finite-duration reduced-network learning under a specified activity controller; not general neural optimality or measured energy."}
    (HERE / "replication_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
