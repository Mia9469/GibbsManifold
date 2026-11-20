"""Bracket nominal achieved rates by replaying frozen weights, not training."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "outputs/snn_budget_frontier_20261016"
sys.path.insert(0, str(OLD))
from frontier import sha
from controlled_budget import simulate
from replicate_learning import initial_model, generate_frozen_stimuli

SPEC = json.loads((OLD / "replication_protocol.json").read_text())
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
OUT = HERE / "replay"
NAMES = ("feedforward_e", "omega_ee", "omega_ei", "omega_ii", "omega_ie")


def case_inputs(seed, budget):
    tag = f"seed_{seed}_budget_{budget}"
    return [OLD / "replication" / (tag + suffix)
            for suffix in (".json", "_weights.npz", "_responses.npz")]


def setup(seed, paths):
    model = initial_model(seed)
    with np.load(paths[1], allow_pickle=False) as cached:
        assert str(cached["code_sha256"]) == sha(OLD / "replicate_learning.py")
        assert str(cached["protocol_sha256"]) == sha(OLD / "replication_protocol.json")
        for name in NAMES:
            model[name] = np.ascontiguousarray(cached[name])
    bank = generate_frozen_stimuli(model, 16, 25000, SPEC["evaluation_seed"])[6:]
    return model, bank


def trial(model, stage, stimulus, index, repeat, command):
    seed = SPEC["evaluation_seed"] + 10000 + 100 * index + repeat
    spikes, _, _, _ = simulate(model, stage, stimulus, seed,
                                {"target": command, "controller_strength": SPEC["controller_strength"]})
    return spikes[:, 500:]


def run_case(seed, budget):
    paths = case_inputs(seed, budget)
    dest = OUT / f"seed_{seed}_budget_{budget}.json"
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    signature = {"inputs": inputs, "protocol_sha256": sha(HERE / "protocol.json"),
                 "code_sha256": sha(__file__),
                 "simulator_sha256": {name: sha(OLD / name) for name in
                                      ("controlled_budget.py", "network_budget.py", "replicate_learning.py")}}
    if dest.exists():
        data = json.loads(dest.read_text())
        assert data["signature"] == signature
        return data
    model, bank = setup(seed, paths)
    original = json.loads(paths[0].read_text())
    results = []
    with np.load(paths[2], allow_pickle=False) as data:
        packed = data["packed_spikes"]
    for stage in PROTOCOL["stages"]:
        # Verify same simulation, seed, stimuli, burn-in and order on both endpoints.
        check = trial(model, stage, bank[0], 0, 0, budget)
        np.testing.assert_array_equal(np.packbits(check, axis=-1, bitorder="little"), packed[stage, 0, 0])
        base_rate = original["checkpoints"][stage]["actual_total_rate"]
        direction = np.sign(budget - base_rate)
        attempts = []
        if direction:
            for offset in (.05, .10):
                command = budget * (1 + direction * offset)
                counts = []
                for index, stimulus in enumerate(bank):
                    for repeat in range(8):
                        response = trial(model, stage, stimulus, index, repeat, command)
                        counts.append(response.reshape(75, 2, 1000).sum(axis=-1).ravel())
                hist = np.bincount(np.concatenate(counts).astype(int))
                assert hist.sum() == 12000
                rate = float(np.dot(hist, np.arange(len(hist))) / 160)
                attempts.append({"command": float(command), "achieved_total_rate": rate,
                                 "histogram": hist.tolist()})
                if (rate - budget) * direction >= 0:
                    break
        matched = bool(not direction or (attempts[-1]["achieved_total_rate"] - budget) * direction >= 0)
        results.append({"stage": stage, "base_rate": base_rate, "bracketed": matched,
                        "attempts": attempts})
    output = {"seed": seed, "budget": budget, "signature": signature, "stages": results}
    dest.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output


def main():
    OUT.mkdir(exist_ok=True)
    seed, budget = PROTOCOL["seeds"][0], PROTOCOL["budgets"][0]
    model, bank = setup(seed, case_inputs(seed, budget))
    started = time.monotonic()
    trial(model, 0, bank[0], 0, 0, budget)
    print(f"Frozen replay warm-up: {time.monotonic() - started:.2f}s", flush=True)
    cases = [(s, b) for s in PROTOCOL["seeds"] for b in PROTOCOL["budgets"]]
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_case, *case) for case in cases]
        for future in as_completed(futures):
            row = future.result()
            print(row["seed"], row["budget"], [(r["base_rate"], r["bracketed"],
                  [a["achieved_total_rate"] for a in r["attempts"]]) for r in row["stages"]], flush=True)


if __name__ == "__main__":
    main()
