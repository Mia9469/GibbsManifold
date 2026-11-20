"""Equal-cost count-distribution diagnostics for the existing SNN responses."""
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binom


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def entropy(p):
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def gibbs(mean, maximum):
    """Upper-budget optimizer on the active finite-count branch."""
    if maximum < 1 or not 0 <= mean <= maximum / 2:
        raise ValueError(f"Outside the active branch: mean={mean}, K={maximum}")
    k = np.arange(maximum + 1, dtype=float)
    if mean == 0:
        q = np.zeros(maximum + 1)
        q[0] = 1
        return q, float("inf")
    if mean == maximum / 2:
        return np.full(maximum + 1, 1 / (maximum + 1)), 0.0

    def probabilities(beta):
        logq = -beta * k
        return np.exp(logq - logsumexp(logq))

    upper = 1.0
    while np.sum(probabilities(upper) * k) > mean:
        upper *= 2
    beta = brentq(lambda b: np.sum(probabilities(b) * k) - mean,
                  0, upper, xtol=1e-14, rtol=1e-14)
    return probabilities(beta), float(beta)


def metrics(probability):
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1):
        raise ValueError("Invalid count distribution")
    k = np.arange(len(p), dtype=float)
    mean = float(np.sum(p * k))
    q, beta = gibbs(mean, len(p) - 1)
    h, hstar = entropy(p), entropy(q)
    positive = p > 0
    if mean > 0:
        logq = -beta * k - logsumexp(-beta * k)
        kl = float(np.sum(p[positive] * (np.log(p[positive]) - logq[positive])))
    else:
        kl = 0.0
    np.testing.assert_allclose(np.sum(q * k), mean, atol=2e-10, rtol=1e-11)
    np.testing.assert_allclose(hstar - h, kl, atol=2e-10, rtol=1e-9)
    variance = float(np.sum(p * ((k - mean) ** 2)))
    variance_q = float(np.sum(q * ((k - mean) ** 2)))
    return {
        "mean_count": mean, "entropy_nats": h, "maximum_entropy_nats": hstar,
        "deficit_nats": kl, "relative_deficit": kl / hstar if hstar else 0.0,
        "cv_pooled": np.sqrt(variance) / mean if mean else None,
        "cv_gibbs": np.sqrt(variance_q) / mean if mean else None,
        "beta": beta if np.isfinite(beta) else None,
        "zero_fraction": float(p[0]), "support_maximum": len(p) - 1,
    }


def thin(p, retention):
    """Exact count-law thinning; no additional random realization is introduced."""
    if not 0 <= retention <= 1:
        raise ValueError("Thinning cannot increase activity")
    result = np.zeros_like(p, dtype=float)
    for n in np.flatnonzero(p):
        k = np.arange(n + 1)
        result[:n + 1] += p[n] * binom.pmf(k, n, retention)
    result /= result.sum()
    return result


def counts_from_stage(spikes, window_ms):
    if spikes.shape[-1] % window_ms:
        raise ValueError("Window must divide the recorded duration")
    shape = spikes.shape[:-1] + (spikes.shape[-1] // window_ms, window_ms)
    # Keep repeat as the first axis for cluster resampling; pool the fixed bank.
    counts = spikes.reshape(shape).sum(axis=-1)
    return np.moveaxis(counts, 1, 0).reshape(spikes.shape[1], -1)


def histogram(counts, maximum):
    if counts.min() < 0 or counts.max() > maximum:
        raise ValueError("Observed count outside the fixed response support")
    return np.bincount(counts.ravel().astype(np.int64), minlength=maximum + 1) / counts.size


def snapshot():
    source = ROOT / "output/overleaf/GibbsManifold_eLife_AppendixBoxplot_20261011"
    if not source.exists():
        return {}
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.name != ".DS_Store")
    files += [ROOT / "outputs/elife_cover_letter_20261006/cover_letter.tex",
              ROOT / "outputs/elife_cover_letter_20261006/cover_letter.txt"]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in files}
    target = HERE / "protected_inputs.json"
    if target.exists():
        expected = json.loads(target.read_text())
        authorized_path = HERE / "authorized_cover_revision.json"
        if authorized_path.exists():
            authorized = json.loads(authorized_path.read_text())
            assert set(authorized) == {
                "outputs/elife_cover_letter_20261006/cover_letter.tex",
                "outputs/elife_cover_letter_20261006/cover_letter.txt"}
            expected.update(authorized)
        assert expected == manifest, "Protected manuscript changed"
    else:
        target.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_pilot():
    snapshot()
    protocol = json.loads((HERE / "protocol.json").read_text())
    path = ROOT / protocol["pilot_data"]
    cache = np.load(path, allow_pickle=False)
    print({key: list(cache[key].shape) for key in cache.files}, flush=True)
    packed = cache["packed_spikes_e"]
    assert packed.shape[:4] == (29, 5, 20, 300)
    windows = [protocol["primary_window_ms"]] + protocol["sensitivity_windows_ms"]
    rows, endpoints = [], {}
    initial = {}
    for stage in range(len(packed)):
        spikes = np.unpackbits(packed[stage], axis=-1, count=int(cache["n_bins"]), bitorder="little")
        for window in windows:
            counts = counts_from_stage(spikes, window)
            maximum = int(np.ceil(window / 1.1))
            p = histogram(counts, maximum)
            row = {"checkpoint": stage + 1, "window_ms": window,
                   "response_samples": counts.size, **metrics(p)}
            row["rate_hz"] = row["mean_count"] / (window / 1000)
            if stage == 0:
                initial[window] = p
            mean_initial = metrics(initial[window])["mean_count"]
            if row["mean_count"] <= mean_initial:
                control = metrics(thin(initial[window], row["mean_count"] / mean_initial))
                row["activity_only_deficit_nats"] = control["deficit_nats"]
                row["activity_only_relative_deficit"] = control["relative_deficit"]
                row["advantage_over_activity_only_nats"] = control["deficit_nats"] - row["deficit_nats"]
            if stage in (0, len(packed) - 1):
                endpoints[(stage, window)] = counts
            rows.append(row)
        print(f"pilot checkpoint {stage + 1:02d}/29: " + json.dumps(rows[-len(windows)]), flush=True)
    rng = np.random.default_rng(protocol["seed"])
    checks = {}
    for window in windows:
        before, after = endpoints[(0, window)], endpoints[(28, window)]
        maximum = int(np.ceil(window / 1.1))
        before_hist = np.array([np.bincount(x.astype(int), minlength=maximum + 1) for x in before])
        after_hist = np.array([np.bincount(x.astype(int), minlength=maximum + 1) for x in after])
        draws = []
        for _ in range(protocol["bootstrap_draws"]):
            ib = rng.integers(0, len(before), len(before))
            ia = rng.integers(0, len(after), len(after))
            pb = before_hist[ib].sum(axis=0) / before.size
            pa = after_hist[ia].sum(axis=0) / after.size
            mb, ma = metrics(pb), metrics(pa)
            control = metrics(thin(pb, min(1., ma["mean_count"] / mb["mean_count"])))
            draws.append([ma["deficit_nats"] - mb["deficit_nats"],
                          ma["relative_deficit"] - mb["relative_deficit"],
                          control["deficit_nats"] - ma["deficit_nats"]])
        ref = {}
        for name, counts in (("initial", before), ("final", after)):
            p = histogram(counts, maximum)
            q, _ = gibbs(metrics(p)["mean_count"], maximum)
            # Multinomial sampling avoids allocating individual synthetic responses.
            simulated = rng.multinomial(counts.size, q, size=protocol["gibbs_reference_draws"])
            deficits = [metrics(x / counts.size)["deficit_nats"] for x in simulated]
            conservative = np.pad(p, (0, window - maximum))
            ref[name] = {"gibbs_sampling_deficit_mean": float(np.mean(deficits)),
                         "gibbs_sampling_deficit_95pct": np.quantile(deficits, [.025, .975]).tolist(),
                         "conservative_support_deficit": metrics(conservative)["deficit_nats"]}
        checks[str(window)] = {"conditional_repeat_bootstrap_95pct": {
            key: np.quantile(np.asarray(draws)[:, index], [.025, .975]).tolist()
            for index, key in enumerate(("change_deficit_nats", "change_relative_deficit", "final_advantage_over_thinning_nats"))},
            "reference_sampling": ref}
        print("window sensitivity", window, checks[str(window)], flush=True)
    q, _ = gibbs(2., 20)
    alternative = q.copy()
    direction = np.array([-1., 3., -3., 1.])
    step = .8 * min(q[:4][direction < 0] / -direction[direction < 0])
    alternative[:4] += step * direction
    k = np.arange(21)
    np.testing.assert_allclose([alternative.sum(), np.sum(alternative * k), np.sum(alternative * k ** 2)],
                               [q.sum(), np.sum(q * k), np.sum(q * k ** 2)], atol=1e-12)
    protected = snapshot()
    payload = {"protocol_sha256": sha(HERE / "protocol.json"), "raster_sha256": sha(path),
               "rows": rows, "sampling_checks": checks,
               "same_moments_control": {"gibbs": metrics(q), "non_gibbs": metrics(alternative)},
               "protected_inputs_unchanged": True if protected else None}
    (HERE / "pilot_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("Saved", HERE / "pilot_results.json", flush=True)


if __name__ == "__main__":
    run_pilot()
