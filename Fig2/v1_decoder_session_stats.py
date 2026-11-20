"""Cluster-level summary of the five-session V1 decoder decomposition.

This script treats recording session, rather than neuron, as the replication
unit. It summarizes the per-session correlations already produced by
v1_decoder_vs_CV_multi.py with a Fisher-z one-sample t analysis and exact
Wilcoxon/sign sensitivities. No figure is written.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, spearmanr, t as student_t, wilcoxon


HERE = Path(__file__).resolve().parent
INPUT_JSON = HERE / "v1_decoder_vs_CV_multi.json"
INPUT_NPZ = HERE / "v1_decoder_vs_CV_multi.npz"
OUT_JSON = HERE / "v1_decoder_session_stats.json"


def fisher_cluster_summary(values: list[float]) -> dict:
    r = np.asarray(values, float)
    z = np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12))
    n = int(r.size)
    df = n - 1
    mean_z = float(z.mean())
    se_z = float(z.std(ddof=1) / np.sqrt(n))
    t_stat = mean_z / se_z
    crit = float(student_t.ppf(0.975, df))
    ci = np.tanh([mean_z - crit * se_z, mean_z + crit * se_z])
    n_positive = int(np.sum(r > 0))
    return {
        "n_sessions": n,
        "per_session_rho": r.tolist(),
        "median_rho": float(np.median(r)),
        "fisher_mean_rho": float(np.tanh(mean_z)),
        "fisher_mean_rho_ci95_cluster_t": ci.tolist(),
        "fisher_z_t_statistic": float(t_stat),
        "cluster_df": df,
        "fisher_z_t_p_two_sided": float(2 * student_t.sf(abs(t_stat), df)),
        "sessions_with_positive_rho": n_positive,
        "session_wilcoxon_p_two_sided": float(wilcoxon(r).pvalue),
        "session_sign_test_p_two_sided": float(binomtest(n_positive, n, 0.5).pvalue),
    }


def nested_rhos(sessions: list[dict], group: str, metric: str) -> list[float]:
    return [float(session[group][metric]["rho"]) for session in sessions]


def main() -> None:
    source = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    sessions = source["sessions"]
    arrays = np.load(INPUT_NPZ)
    session_names = np.unique(arrays["session"])
    tuning_rhos = [
        float(spearmanr(
            arrays["tuning_cv"][arrays["session"] == session],
            arrays["decoder_accuracy"][arrays["session"] == session],
        ).statistic)
        for session in session_names
    ]

    output = {
        "n_sessions": len(sessions),
        "n_neurons_pooled": int(source["n_pooled"]),
        "pooled_neuron_level_descriptive": source["pooled"],
        "session_level": {
            "raw_total_CV_vs_accuracy": fisher_cluster_summary(
                nested_rhos(sessions, "raw", "cv_vs_acc")
            ),
            "total_CV_vs_accuracy_given_FR": fisher_cluster_summary(
                nested_rhos(sessions, "partial", "cv_acc_given_fr")
            ),
            "FR_vs_accuracy_given_total_CV": fisher_cluster_summary(
                nested_rhos(sessions, "partial", "fr_acc_given_cv")
            ),
            "tuning_CV_vs_accuracy_raw": fisher_cluster_summary(tuning_rhos),
            "total_CV_vs_accuracy_given_FR_and_tuning": fisher_cluster_summary(
                nested_rhos(
                    sessions, "partial", "total_cv_acc_given_fr_and_tuning"
                )
            ),
            "within_stim_noise_CV_vs_accuracy_given_FR": fisher_cluster_summary(
                nested_rhos(
                    sessions, "partial", "within_stim_noise_cv_acc_given_fr"
                )
            ),
        },
        "inference_note": (
            "Session is the unit. Pooled neuron p-values are descriptive. "
            "The earlier sample-size-weighted Fisher test is not used for "
            "cluster-level inference."
        ),
    }
    OUT_JSON.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["session_level"], indent=2))
    print("Wrote", OUT_JSON)


if __name__ == "__main__":
    main()
