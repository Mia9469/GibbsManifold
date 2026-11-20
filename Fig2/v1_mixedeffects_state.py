"""
v1_mixedeffects_state.py
========================
Mixed-effects re-analysis of the macaque V1 spontaneous-vs-evoked CV_L / S_L
state contrast, replacing the within-single-session neuron-level paired test
(which treats N=159 neurons as independent and pools only one session) with a
linear mixed-effects model across all five v1v2-1 sessions.

Model (per metric y in {CV_L, S_L}):
    Delta y = y_spontaneous - y_evoked
    Delta y ~ 1 + (1 | session)
Pairing is therefore handled within neuron before the hierarchical model is
fit, and the fixed intercept estimates the population-level state contrast
while retaining session-level heterogeneity.

Outputs Fig2/v1_mixedeffects_state.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import statsmodels.formula.api as smf
from scipy.stats import binomtest, t as student_t, wilcoxon

HERE = Path(__file__).resolve().parent
SESSIONS = [
    "105l001p16.mat", "106r001p26.mat", "106r002p70.mat",
    "107l002p67.mat", "107l003p143.mat",
]
DRIVEN_BINS = 1280


def per_neuron_cv_s(mat_path: Path):
    """Return per-neuron (CV_L, S_L) for driven and spontaneous, matched 1.28 s."""
    m = sio.loadmat(str(mat_path), squeeze_me=False, struct_as_record=False)
    nd = m["neuralData"][0, 0]
    rasters = nd.spikeRasters
    stim = nd.stim.ravel().astype(int)
    dr, sp = [], []
    for i in range(rasters.shape[0]):
        r = rasters[i, 0]
        if r.shape[1] < DRIVEN_BINS:
            continue
        c = np.asarray(r[:, :DRIVEN_BINS].sum(axis=1)).ravel().astype(float)
        (dr if stim[i] != 0 else sp).append(c)
    if not dr or not sp:
        return None
    D = np.vstack(dr)   # trials x neurons
    S = np.vstack(sp)
    n = D.shape[1]
    out = []
    for j in range(n):
        for X, cond in [(S, "spontaneous"), (D, "evoked")]:
            x = X[:, j]
            mu = x.mean()
            if mu <= 0:
                continue
            cv = x.std(ddof=1) / mu
            N = x.size
            s = (1 - mu ** 2 / np.mean(x ** 2)) / (1 - 1.0 / N)
            out.append({"neuron": j, "cond": cond, "CV_L": cv, "S_L": s})
    return out


def main():
    rows = []
    for k, s in enumerate(SESSIONS):
        p = HERE / s
        if not p.exists():
            continue
        recs = per_neuron_cv_s(p)
        if recs is None:
            continue
        for rec in recs:
            rec["session"] = s.replace(".mat", "")
            rec["uid"] = f"{k}_{rec['neuron']}"
            rows.append(rec)
    df = pd.DataFrame(rows)
    df["cond_evoked"] = (df["cond"] == "evoked").astype(float)

    result = {"n_sessions": int(df["session"].nunique()),
              "n_neuron_condition_rows": int(len(df)),
              "n_neurons": int(df["uid"].nunique())}

    for metric in ["CV_L", "S_L"]:
        wide = (
            df.pivot_table(index=["session", "uid"], columns="cond", values=metric)
            .dropna()
            .reset_index()
        )
        wide["diff"] = wide["spontaneous"] - wide["evoked"]

        md = smf.mixedlm("diff ~ 1", wide, groups=wide["session"])
        mf = md.fit(method="lbfgs", maxiter=500, disp=False)
        beta = float(mf.fe_params["Intercept"])
        se = float(mf.bse_fe["Intercept"])
        p_asymptotic = float(mf.pvalues["Intercept"])
        # Only five recording sessions are independent sampling units.  The
        # default MixedLM z test is anti-conservative here, so use a transparent
        # cluster-df t approximation for the primary fixed-effect inference.
        n_sessions = int(wide["session"].nunique())
        df_clusters = n_sessions - 1
        t_stat = beta / se
        t_crit = float(student_t.ppf(0.975, df_clusters))
        p_cluster_t = float(2 * student_t.sf(abs(t_stat), df_clusters))
        # descriptive medians
        med_sp = float(df.loc[df.cond == "spontaneous", metric].median())
        med_ev = float(df.loc[df.cond == "evoked", metric].median())
        session_diffs_df = (
            wide.groupby("session")["diff"]
            .agg(["count", "mean", "median"])
            .reset_index()
        )
        session_medians = session_diffs_df["median"].to_numpy(float)
        session_means = session_diffs_df["mean"].to_numpy(float)
        positive_medians = int(np.sum(session_medians > 0))
        wilcoxon_p = float(wilcoxon(session_medians, alternative="two-sided").pvalue)
        sign_p = float(
            binomtest(positive_medians, n_sessions, p=0.5, alternative="two-sided").pvalue
        )
        result[metric] = {
            "median_spontaneous": med_sp,
            "median_evoked": med_ev,
            "paired_mean_spont_minus_evoked": float(wide["diff"].mean()),
            "paired_median_spont_minus_evoked": float(wide["diff"].median()),
            "fixed_effect_spont_minus_evoked": beta,
            "fixed_effect_evoked_minus_spont": -beta,
            "se": se,
            "cluster_df": df_clusters,
            "t_statistic": t_stat,
            "ci95_cluster_t": [beta - t_crit * se, beta + t_crit * se],
            "p_value_cluster_t": p_cluster_t,
            "ci95_asymptotic_z": [beta - 1.96 * se, beta + 1.96 * se],
            "p_value_asymptotic_z": p_asymptotic,
            "random_intercept_variance": float(mf.cov_re.iloc[0, 0]),
            "residual_variance": float(mf.scale),
            "converged": bool(mf.converged),
            "session_median_difference_median": float(np.median(session_medians)),
            "session_mean_difference_median": float(np.median(session_means)),
            "sessions_with_positive_median_difference": positive_medians,
            "session_wilcoxon_p_two_sided": wilcoxon_p,
            "session_sign_test_p_two_sided": sign_p,
            "model": "paired neuron difference: y_spontaneous - y_evoked ~ 1 + random intercept by session",
            "inference_note": (
                "The fixed effect is tested against a t distribution with "
                "n_sessions - 1 degrees of freedom; the default asymptotic "
                "MixedLM z result is retained only as a secondary diagnostic."
            ),
            "session_differences": session_diffs_df.to_dict(orient="records"),
        }
        print(f"[{metric}] spont median {med_sp:.3f}, evoked median {med_ev:.3f}; "
              f"mixed-effects spont-evoked = {beta:+.3f} "
              f"(SE {se:.3f}, t({df_clusters})={t_stat:.2f}, "
              f"cluster-df p={p_cluster_t:.4g}; session Wilcoxon p={wilcoxon_p:.4g})")

    out = HERE / "v1_mixedeffects_state.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("Wrote", out)


if __name__ == "__main__":
    main()
