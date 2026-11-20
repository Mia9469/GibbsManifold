"""Retrospective, within-cohort population/readout and time-scale checks."""
import csv
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import lfilter
from scipy.special import xlogy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / "outputs/snn_budget_frontier_20261016"
sys.path.insert(0, str(OLD))
from frontier import gibbs, metrics
from replicate_learning import initial_model, generate_frozen_stimuli
from controlled_budget import simulate

SPEC = json.loads((OLD / "replication_protocol.json").read_text())
WINDOWS = (20, 50, 100, 250, 500, 1000)
GROUPS = {"all": slice(None), "excitatory": slice(60), "inhibitory": slice(60, None)}
RNG_SEED = 20261112
PAIRS = np.array(list(combinations(range(60), 2)))[
    np.sort(np.random.default_rng(RNG_SEED).choice(1770, 32, replace=False))]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def ent(p, axis=None):
    return -np.sum(xlogy(p, p), axis=axis) / np.log(2)


def dist(p):
    m = metrics(p)
    d = m["deficit_nats"] / np.log(2)
    return dict(mean=m["mean_count"], D=d, G=d / m["mean_count"], entropy=m["entropy_nats"] / np.log(2))


def hist(x, k):
    h = np.bincount(x.ravel(), minlength=k + 1)
    assert len(h) == k + 1
    return h / x.size


def repeat_product(half, k):
    # For each cell/template/window, average all ordered repeat combinations
    # (including the observed same-repeat pair) exactly, without Monte Carlo.
    x, y = half[..., 0], half[..., 1]
    values = x[:, :, None] + y[:, None, :]
    return hist(values, k)


def target_bank():
    model = initial_model(SPEC["training_seeds"][0])
    stimuli = generate_frozen_stimuli(model, 16, 25000, SPEC["evaluation_seed"])[6:]
    dt = SPEC["dt_s"]
    truth = lfilter([dt], [1, -(1 - model["leak"] * dt)], stimuli[..., :-1], axis=-1)
    truth = np.concatenate((np.zeros((*truth.shape[:-1], 1)), truth), axis=-1)
    truth = truth[..., 9::10][..., 500:]
    _, _, check, _ = simulate(model, 0, stimuli[0], SPEC["evaluation_seed"] + 10000,
                               {"target": 240, "controller_strength": SPEC["controller_strength"]})
    np.testing.assert_allclose(truth[0], check[:, 500:], atol=1e-12, rtol=0)
    return truth.reshape(10, 3, 100, 20).mean(-1).transpose(0, 2, 1)


def heldout_re(x, truth):
    # x: template x repeat x time x 2, truth: template x time x 3.
    # Same count vector for entropy and reconstruction; held-out whole templates.
    target = np.broadcast_to(truth[:, None, :, :], (*x.shape[:3], 3))
    errors = []
    for held in range(10):
        mask = np.arange(10) != held
        a, y = x[mask].reshape(-1, 2), target[mask].reshape(-1, 3)
        ma, my = a.mean(0), y.mean(0)
        a = a - ma
        gram = np.einsum("ni,nj->ij", a, a, optimize=False)
        ridge = .01 * max(np.trace(gram) / 2, 1e-12)
        cross = np.einsum("ni,nj->ij", a, y - my, optimize=False)
        w = np.linalg.solve(gram + ridge * np.eye(2), cross)
        estimate = np.einsum("rti,ij->rtj", x[held] - ma, w, optimize=False) + my
        denom = np.var(target[held], axis=(0, 1)).sum()
        assert np.isfinite(estimate).all() and np.isfinite(denom) and denom > 0
        errors.append(np.var(target[held] - estimate, axis=(0, 1)).sum() / denom)
    return float(np.mean(errors))


def pair_checks(counts, truth):
    k = int(np.ceil(20 / 1.1))
    results = []
    for cells in PAIRS:
        x = counts[:, :, cells, :].transpose(0, 1, 3, 2)
        joint = np.bincount((x[..., 0] * (k + 1) + x[..., 1]).ravel(), minlength=(k + 1)**2)
        joint = joint.reshape(k + 1, k + 1) / (x.size / 2)
        marg = np.stack([joint.sum(1), joint.sum(0)])
        pooled = marg.mean(0)
        mu = np.dot(pooled, np.arange(k + 1))
        q, _ = gibbs(mu, k)
        dp = 2 * (ent(q) - ent(pooled))
        hetero = 2 * ent(pooled) - ent(marg, axis=1).sum()
        tc = ent(marg, axis=1).sum() - ent(joint)
        gap = 2 * ent(q) - ent(joint)
        ref = np.outer(q, q)
        mask = joint > 0
        np.testing.assert_allclose(gap, np.sum(joint[mask] * np.log2(joint[mask] / ref[mask])), atol=1e-11)
        np.testing.assert_allclose(gap, dp + hetero + tc, atol=1e-12)
        results.append(dict(cells=cells.tolist(), pooled=dp, heterogeneity=hetero, dependence=tc,
                            joint_gap=gap, count_readout_RE=heldout_re(x, truth)))
    return results


def run():
    truth = target_bank()
    manifest, records, pairs, timescale, type_terms = {}, [], [], [], []
    for seed in SPEC["training_seeds"]:
        for budget in SPEC["target_total_spikes_per_second"]:
            tag = f"seed_{seed}_budget_{budget}"
            path = OLD / "replication" / f"{tag}_responses.npz"
            meta = OLD / "replication" / f"{tag}.json"
            for p in (path, meta):
                manifest[str(p.relative_to(ROOT))] = sha(p)
            cp = json.loads(meta.read_text())["checkpoints"]
            assert len(cp) == 5
            with np.load(path) as z:
                packed = z["packed_spikes"]
            assert packed.shape == (5, 10, 8, 75, 250)
            for stage in range(5):
                spikes = np.unpackbits(packed[stage], axis=-1, bitorder="little")
                for window in WINDOWS:
                    k = int(np.ceil(window / 1.1))
                    counts = spikes.reshape(10, 8, 75, -1, window).sum(-1, dtype=np.int64)
                    laws = {}
                    for group, sl in GROUPS.items():
                        p = hist(counts[:, :, sl], k)
                        v = dist(p)
                        laws[group] = (p, v)
                        saved = next(d for d in cp[stage]["distributions"] if d["group"] == group and d["window_ms"] == window)
                        np.testing.assert_allclose([v["mean"], v["D"]], [saved["mean_count"], saved["deficit_nats"] / np.log(2)], atol=5e-12)
                        records.append(dict(seed=seed, budget=budget, stage=stage, window=window, group=group,
                                            rate=v["mean"] * 1000 / window, RE=cp[stage]["waveform_re"],
                                            time=cp[stage]["time_s"], target_fraction=cp[stage]["actual_total_rate"] / budget, **v))
                    if stage not in (0, 4):
                        continue
                    p, pooled = laws["all"]
                    (pe, ve), (pi, vi) = laws["excitatory"], laws["inhibitory"]
                    w = 0.8
                    np.testing.assert_allclose(p, w * pe + (1 - w) * pi, atol=1e-14)
                    mix_entropy = pooled["entropy"] - w * ve["entropy"] - (1 - w) * vi["entropy"]
                    frontier = pooled["D"] + pooled["entropy"]
                    offset = frontier - w * (ve["D"] + ve["entropy"]) - (1 - w) * (vi["D"] + vi["entropy"])
                    terms = np.array([w * ve["D"], (1 - w) * vi["D"], offset, -mix_entropy])
                    terms /= pooled["mean"]
                    np.testing.assert_allclose(terms.sum(), pooled["G"], atol=1e-12)
                    type_terms.append(dict(seed=seed, budget=budget, stage=stage, window=window,
                                           e=terms[0], i=terms[1], mean_allocation=terms[2], mixing=terms[3],
                                           inhibitory_spike_share=(1 - w) * vi["mean"] / pooled["mean"]))
                    if window == 20:
                        pairs.extend(dict(seed=seed, budget=budget, stage=stage, **r) for r in pair_checks(counts, truth))
                    if window == 500:
                        half = counts.transpose(0, 2, 3, 1).reshape(10, 75, 2, 2, 8).transpose(0, 1, 2, 4, 3)
                        half = half.reshape(-1, 8, 2)
                        limit = int(np.ceil(1000 / 1.1))
                        a, b = dist(hist(half.sum(-1), limit)), dist(repeat_product(half, limit))
                        np.testing.assert_allclose(a["mean"], b["mean"], atol=1e-12)
                        expected = next(d for d in cp[stage]["distributions"] if d["group"] == "all" and d["window_ms"] == 1000)
                        np.testing.assert_allclose(a["D"], expected["deficit_nats"] / np.log(2), atol=5e-12)
                        centered = half - half.mean(axis=1, keepdims=True)
                        cov = np.mean(centered[..., 0] * centered[..., 1])
                        np.testing.assert_allclose(np.var(half.sum(-1)) -
                            np.var(half[:, :, None, 0] + half[:, None, :, 1]), 2 * cov, atol=1e-12)
                        timescale.append(dict(seed=seed, budget=budget, stage=stage,
                                              observed_G=a["G"], independent_halves_G=b["G"], covariance=cov))
    for name, data in (("counts", records), ("pairs", pairs), ("time_dependence", timescale),
                       ("type_decomposition", type_terms)):
        with (HERE / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (HERE / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Verified {len(records)} archived count summaries; computed {len(pairs)} pair checks and {len(timescale)} temporal controls.")


if __name__ == "__main__":
    run()
