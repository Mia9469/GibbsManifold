"""Verify shared inputs and central frozen-cohort numerical contracts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "outputs/snn_budget_frontier_20261016"
sys.path.insert(0, str(CORE))
from frontier import counts_from_stage, histogram, metrics


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hashes-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text())
    for name, record in manifest["files"].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != record["sha256"]:
            raise ValueError(f"Released input changed or missing: {name}")
    print(f"Verified {len(manifest['files'])} payload hashes", flush=True)
    if args.hashes_only:
        return
    protocol = json.loads((CORE / "replication_protocol.json").read_text())
    assert protocol["training_seeds"] == list(range(20261020, 20261028))
    observed = {budget: [] for budget in (240, 480, 720)}
    maximum_error, distributions = 0.0, 0
    for seed in protocol["training_seeds"]:
        for budget in observed:
            tag = f"seed_{seed}_budget_{budget}"
            saved = json.loads((CORE / "replication" / (tag + ".json")).read_text())
            assert saved["code_sha256"] == digest(CORE / "replicate_learning.py")
            assert saved["protocol_sha256"] == digest(CORE / "replication_protocol.json")
            stages = saved["checkpoints"]
            assert len(stages) == 5 and saved["network_size"] == [60, 15]
            with np.load(CORE / "replication" / (tag + "_responses.npz"), allow_pickle=False) as data:
                assert data["packed_spikes"].shape == (5, 10, 8, 75, 250)
                for stage, packed in zip(stages, data["packed_spikes"]):
                    spikes = np.unpackbits(packed, axis=-1, count=2000, bitorder="little")
                    np.testing.assert_allclose(spikes.sum((2, 3)) / 2,
                                               stage["rate_by_template_repeat"], rtol=0, atol=0)
                    for item in stage["distributions"]:
                        subset = spikes if item["group"] == "all" else (
                            spikes[:, :, :60] if item["group"] == "excitatory" else spikes[:, :, 60:])
                        window = item["window_ms"]
                        actual = metrics(histogram(counts_from_stage(subset, window), int(np.ceil(window / 1.1))))
                        for key in ("mean_count", "entropy_nats", "maximum_entropy_nats", "deficit_nats", "cv_pooled"):
                            error = abs(actual[key] - item[key])
                            maximum_error = max(maximum_error, error)
                            assert error < 2e-10, (tag, key, error)
                        distributions += 1
            initial, learned = stages[0], stages[-1]
            assert learned["waveform_re"] < initial["waveform_re"]
            gaps = []
            for stage in (initial, learned):
                item = next(x for x in stage["distributions"] if x["group"] == "all" and x["window_ms"] == 1000)
                gaps.append(item["deficit_nats"] / np.log(2) / item["mean_count"])
            observed[budget].append(gaps[1] - gaps[0])
    assert len(observed[480]) == len(observed[720]) == 8
    assert np.all(np.asarray(observed[480]) < 0)
    assert np.all(np.asarray(observed[720]) > 0)
    report = {"cases": 24, "checkpoints": 120, "distributions_recomputed": distributions,
              "maximum_statistic_error": maximum_error, "all_24_reconstruction_errors_decrease": True,
              "one_second_gap_seed_deltas_bits_per_spike": observed,
              "scope": "Direct recomputation of archived simulated responses, not new training or population MI"}
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
