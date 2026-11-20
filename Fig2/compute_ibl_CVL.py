"""compute_ibl_CVL.py

Compute CV_L (lifetime coefficient of variation) for IBL Repeated Site data.

Data source: /Volumes/MimiHDD/ONE/openalyx.internationalbrainlab.org/
Protocol:
  - Spontaneous: gray-screen spontaneousActivity period (after task), 1.5 s windows
  - Evoked:      task trials, spike count in [stim_onset, stim_onset + 0.5 s] window

Only "good" units are included (clusters.metrics label >= 1.0).

Outputs:
  Fig2/ibl_CVL_results.npz   — per-unit arrays + session metadata
  Fig2/ibl_CVL_summary.json  — per-session, per-region summary

Usage:
    python compute_ibl_CVL.py [--ibl-root /path/to/ONE/openalyx...]
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

BASE     = Path(__file__).parent
IBL_ROOT = Path("/Volumes/MimiHDD/ONE/openalyx.internationalbrainlab.org")
CCF_TREE_URL = ("https://raw.githubusercontent.com/cortex-lab/allenCCF/"
                "master/structure_tree_safe_2017.csv")
CCF_TREE_CACHE = BASE / "ccf_structure_tree.csv"

SPON_WIN   = 1.5    # s per spontaneous bin
MIN_WINS   = 20     # minimum windows for CV_L
STIM_WIN   = 0.5    # s post-stim-onset for evoked counts
MIN_TRIALS = 20     # minimum trials for CV_L
GOOD_LABEL = 1.0    # threshold on clusters.metrics.label

# ── CCF ID → acronym (full Allen CCF 2017 tree, loaded on first use) ──────────
_CCF_MAP: dict[int, str] | None = None

def _load_ccf_map() -> dict[int, str]:
    global _CCF_MAP
    if _CCF_MAP is not None:
        return _CCF_MAP
    # try local cache first, then /tmp, then download
    for candidate in [CCF_TREE_CACHE, Path("/tmp/ccf_structure_tree.csv")]:
        if candidate.exists():
            tree = pd.read_csv(str(candidate))
            _CCF_MAP = dict(zip(tree["id"].astype(int), tree["acronym"].astype(str)))
            print(f"  CCF tree loaded ({len(_CCF_MAP)} entries) from {candidate}")
            return _CCF_MAP
    # download
    import urllib.request
    print(f"  Downloading CCF tree from GitHub...")
    urllib.request.urlretrieve(CCF_TREE_URL, str(CCF_TREE_CACHE))
    tree = pd.read_csv(str(CCF_TREE_CACHE))
    _CCF_MAP = dict(zip(tree["id"].astype(int), tree["acronym"].astype(str)))
    print(f"  CCF tree downloaded ({len(_CCF_MAP)} entries)")
    return _CCF_MAP

def get_acronym(ccf_id: int) -> str:
    m = _load_ccf_map()
    return m.get(int(ccf_id), f"area_{ccf_id}")


def bias_corrected_sparseness(r: np.ndarray) -> float:
    N = len(r)
    if N < 2: return np.nan
    mu = r.mean(); mu2 = (r**2).mean()
    return np.nan if mu2 == 0 else (1 - mu**2/mu2) / (1 - 1/N)


def compute_CVL(counts: np.ndarray):
    """Return (cvL, sL, FR) from spike count vector. FR = mean / bin_duration."""
    counts = counts[np.isfinite(counts)]
    if len(counts) < 2 or counts.mean() <= 0:
        return np.nan, np.nan, np.nan
    mu  = counts.mean()
    cvL = counts.std(ddof=0) / mu
    sL  = bias_corrected_sparseness(counts)
    return cvL, sL, mu


def process_probe(ks_path: Path, spon_interval: tuple[float,float],
                  trial_intervals: np.ndarray, good_only: bool = True):
    """
    Compute per-unit CV_L for one probe.

    Returns DataFrame with columns:
      unit_id, cvL_spon, sL_spon, FR_spon,
               cvL_evk,  sL_evk,  FR_evk,
               area_acronym, n_spikes
    """
    spk_t  = np.load(ks_path / "spikes.times.npy")
    spk_cl = np.load(ks_path / "spikes.clusters.npy")
    cl_ch  = np.load(ks_path / "clusters.channels.npy")
    ch_ids = np.load(ks_path / "channels.brainLocationIds_ccf_2017.npy")

    metrics = pd.read_parquet(ks_path / "clusters.metrics.pqt")
    metrics.index = metrics["cluster_id"] if "cluster_id" in metrics.columns else metrics.index

    # cluster → brain area
    def cluster_acronym(cl_idx):
        ch = int(cl_ch[cl_idx]) if cl_idx < len(cl_ch) else 0
        ch = min(ch, len(ch_ids) - 1)
        return get_acronym(int(ch_ids[ch]))

    spon_start, spon_stop = spon_interval
    spon_dur = spon_stop - spon_start
    n_spon_wins = int(np.floor(spon_dur / SPON_WIN))

    rows = []
    unique_clusters = np.unique(spk_cl)

    for cl in unique_clusters:
        cl_int = int(cl)

        # quality filter
        if good_only:
            if cl_int not in metrics.index:
                continue
            lbl = metrics.loc[cl_int, "label"] if "label" in metrics.columns else 1.0
            if lbl < GOOD_LABEL:
                continue

        mask = spk_cl == cl
        st   = spk_t[mask]

        # ── spontaneous CV_L ──────────────────────────────────────────────────
        spon_spk = st[(st >= spon_start) & (st < spon_stop)]
        if n_spon_wins >= MIN_WINS:
            edges  = spon_start + np.arange(n_spon_wins + 1) * SPON_WIN
            counts_s, _ = np.histogram(spon_spk, bins=edges)
            cvL_s, sL_s, fr_s = compute_CVL(counts_s.astype(float))
        else:
            cvL_s = sL_s = fr_s = np.nan

        # ── evoked CV_L ───────────────────────────────────────────────────────
        # count spikes in [trial_start, trial_start + STIM_WIN] for each trial
        if trial_intervals is not None and len(trial_intervals) >= MIN_TRIALS:
            trial_counts = np.array([
                np.sum((st >= t0) & (st < t0 + STIM_WIN))
                for t0 in trial_intervals[:, 0]
            ], dtype=float)
            cvL_e, sL_e, fr_e = compute_CVL(trial_counts)
        else:
            cvL_e = sL_e = fr_e = np.nan

        if np.isnan(cvL_s) and np.isnan(cvL_e):
            continue

        rows.append(dict(
            unit_id=cl_int,
            cvL_spon=cvL_s, sL_spon=sL_s, FR_spon=fr_s / SPON_WIN if not np.isnan(fr_s) else np.nan,
            cvL_evk=cvL_e,  sL_evk=sL_e,  FR_evk=fr_e  / STIM_WIN if not np.isnan(fr_e) else np.nan,
            area_acronym=cluster_acronym(cl_int),
            n_spikes=int(mask.sum()),
        ))

    return pd.DataFrame(rows)


def find_sessions(ibl_root: Path):
    """Yield (lab, subject, date, session_num, session_alf_path, probe_ks_paths)."""
    for lab_dir in sorted(ibl_root.iterdir()):
        if not lab_dir.is_dir() or lab_dir.name.startswith("."): continue
        subj_root = lab_dir / "Subjects"
        if not subj_root.exists(): continue
        for subj in sorted(subj_root.iterdir()):
            if not subj.is_dir() or subj.name.startswith("."): continue
            for date in sorted(subj.iterdir()):
                if not date.is_dir() or date.name.startswith("."): continue
                for num in sorted(date.iterdir()):
                    if not num.is_dir() or num.name.startswith("."): continue
                    alf = num / "alf"
                    if not alf.exists(): continue

                    # collect probe ks paths (structure: probe00/pykilosort/#date#/)
                    probe_paths = []
                    for item in sorted(alf.iterdir()):
                        if not item.name.startswith("probe"): continue
                        for sub in sorted(item.iterdir()):
                            if sub.name.startswith("."): continue
                            # direct: probe00/spikes.times.npy
                            if (sub / "spikes.times.npy").exists():
                                probe_paths.append(sub)
                            else:
                                # one level deeper: probe00/pykilosort/#date#/
                                for subsub in sorted(sub.iterdir()):
                                    if (not subsub.name.startswith(".") and
                                            (subsub / "spikes.times.npy").exists()):
                                        probe_paths.append(subsub)

                    if probe_paths:
                        yield (lab_dir.name, subj.name, date.name, num.name,
                               alf, probe_paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ibl-root", default=str(IBL_ROOT))
    parser.add_argument("--out", default=str(BASE / "ibl_CVL_results.npz"))
    args = parser.parse_args()

    ibl_root = Path(args.ibl_root)
    if not ibl_root.exists():
        print(f"ERROR: IBL root not found: {ibl_root}")
        return

    all_rows = []
    session_summary = []

    for lab, subj, date, num, alf, probe_paths in find_sessions(ibl_root):
        sess_id = f"{lab}/{subj}/{date}"
        print(f"\n── {sess_id} ({len(probe_paths)} probes) ──")

        # ── load passive periods ──────────────────────────────────────────────
        pp_file = alf / "_ibl_passivePeriods.intervalsTable.csv"
        if not pp_file.exists():
            print("  SKIP: no passivePeriods file")
            continue
        pp = pd.read_csv(pp_file)
        # spontaneousActivity column: rows are [start, stop]
        if "spontaneousActivity" not in pp.columns:
            print("  SKIP: no spontaneousActivity column")
            continue
        spon_start = float(pp["spontaneousActivity"].iloc[0])
        spon_stop  = float(pp["spontaneousActivity"].iloc[1])
        spon_dur   = spon_stop - spon_start
        print(f"  Spontaneous: {spon_start:.1f}–{spon_stop:.1f}s ({spon_dur/60:.1f} min)")

        # ── load trial intervals ──────────────────────────────────────────────
        trial_file = alf / "_ibl_trials.intervals_bpod.npy"
        if trial_file.exists():
            trial_ivs = np.load(trial_file)
            print(f"  Trials: {len(trial_ivs)} trials, {trial_ivs[0,0]:.1f}–{trial_ivs[-1,1]:.1f}s")
        else:
            trial_ivs = None
            print("  No trial file — evoked CV_L will be NaN")

        # ── process each probe ────────────────────────────────────────────────
        sess_units = 0
        for ks_path in probe_paths:
            probe_name = ks_path.parent.parent.name
            print(f"  Probe {probe_name}: ", end="", flush=True)
            try:
                df = process_probe(ks_path,
                                   spon_interval=(spon_start, spon_stop),
                                   trial_intervals=trial_ivs)
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            if df.empty:
                print("0 units")
                continue

            df["lab"]      = lab
            df["subject"]  = subj
            df["date"]     = date
            df["probe"]    = probe_name
            df["sess_id"]  = sess_id
            all_rows.append(df)
            sess_units += len(df)
            print(f"{len(df)} units")

            # per-area summary for this probe
            for area, grp in df.groupby("area_acronym"):
                cs = grp["cvL_spon"].dropna()
                ce = grp["cvL_evk"].dropna()
                if len(cs) < 5 and len(ce) < 5: continue
                direction = "spon>evk" if (len(cs)>0 and len(ce)>0 and
                                            np.median(cs) > np.median(ce)) else "evk>=spon"
                print(f"    {area:12s} n={len(grp):3d}  "
                      f"cv_spon={np.nanmedian(cs):.2f}  "
                      f"cv_evk={np.nanmedian(ce):.2f}  {direction}")
                session_summary.append(dict(
                    sess_id=sess_id, lab=lab, area=area,
                    n=len(grp),
                    cv_spon_med=float(np.nanmedian(cs)) if len(cs)>0 else np.nan,
                    cv_evk_med=float(np.nanmedian(ce))  if len(ce)>0 else np.nan,
                    spon_gt_evk=(len(cs)>0 and len(ce)>0 and
                                 np.nanmedian(cs) > np.nanmedian(ce)),
                ))

        print(f"  → {sess_units} total units")

    if not all_rows:
        print("No data found.")
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"TOTAL: {len(df_all)} units across {df_all['sess_id'].nunique()} sessions")

    # ── global spon>evk statistic ─────────────────────────────────────────────
    if session_summary:
        smry = pd.DataFrame(session_summary)
        n_areas = len(smry)
        n_spon_gt = smry["spon_gt_evk"].sum()
        print(f"spon > evoked CV_L: {n_spon_gt}/{n_areas} area×session pairs "
              f"({100*n_spon_gt/n_areas:.0f}%)")

        # by brain region
        by_area = smry.groupby("area")["spon_gt_evk"].agg(["sum","count"])
        by_area["frac"] = by_area["sum"] / by_area["count"]
        print("\nBy brain region:")
        print(by_area.sort_values("frac", ascending=False).to_string())

    # ── save ─────────────────────────────────────────────────────────────────
    out = Path(args.out)
    np.savez(str(out),
             cvL_spon    = df_all["cvL_spon"].values,
             cvL_evk     = df_all["cvL_evk"].values,
             sL_spon     = df_all["sL_spon"].values,
             sL_evk      = df_all["sL_evk"].values,
             FR_spon     = df_all["FR_spon"].values,
             FR_evk      = df_all["FR_evk"].values,
             area        = df_all["area_acronym"].values,
             lab         = df_all["lab"].values,
             sess_id     = df_all["sess_id"].values,
    )
    print(f"\nSaved: {out}")

    json_out = out.with_suffix(".json")
    with open(json_out, "w") as f:
        json.dump(session_summary, f, indent=2, default=str)
    print(f"Saved: {json_out}")

    print(f"\n[IBL] species=mouse N_units={len(df_all)} "
          f"N_sessions={df_all['sess_id'].nunique()} "
          f"N_labs={df_all['lab'].nunique()} "
          f"cv_spon={np.nanmedian(df_all['cvL_spon']):.2f} "
          f"cv_evk={np.nanmedian(df_all['cvL_evk']):.2f} "
          f"direction=spon>evk? {'YES' if np.nanmedian(df_all['cvL_spon'])>np.nanmedian(df_all['cvL_evk']) else 'NO'}")


if __name__ == "__main__":
    main()
