"""compute_steinmetz_CVL.py

Compute CV_L and S_L from the Steinmetz 2019 dataset (mouse, multiple brain regions).

Data format (from NMA alldata.npz, 13 sessions):
  dat[i]['spks']         : (n_neurons, n_trials, n_timebins) — active trials
  dat[i]['spks_passive'] : (n_neurons, n_passive_trials, n_timebins) — passive (visual stim, no task)
  dat[i]['brain_area']   : (n_neurons,) string array
  dat[i]['bin_size']     : 0.01 s (10 ms bins)
  dat[i]['stim_onset']   : 0.5 (bin index for stimulus onset)

For CV_L:
  - 'evoked':      spks, sum a post-stimulus window matched in duration to
                   the pre-stimulus window → one count per trial → CV_L across trials
  - 'spontaneous': spks_passive, bins before stim_onset (0..stim_onset-1) → pre-stim window

Data source: https://osf.io/agvxh/download  (Steinmetz et al. 2019, NMA version)
If already downloaded, pass --data-path.

Saves: Fig2/steinmetz_CVL_results.npz
"""
from __future__ import annotations
import argparse
import subprocess
import shutil
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
DEFAULT_DATA_URL = "https://osf.io/agvxh/download"
DEFAULT_DATA_PATH = BASE / "steinmetz_alldata.npz"
MIN_NEURONS = 10
MIN_TRIALS = 20
AREAS_OF_INTEREST = ["VISp", "VISl", "VISpm", "VISam", "MOs", "ACA", "CA1", "CA3", "DG", "CP", "SSp"]


def bias_corrected_sparseness(r: np.ndarray) -> float:
    N = len(r)
    if N < 2:
        return np.nan
    mu = r.mean()
    mu2 = (r**2).mean()
    if mu2 == 0:
        return np.nan
    return (1.0 - mu**2 / mu2) / (1.0 - 1.0 / N)


def download_data(url: str, out_path: Path) -> Path:
    if out_path.exists():
        print(f"  Using existing: {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
        return out_path
    print(f"  Downloading Steinmetz data (~83 MB) from: {url}")
    curl = shutil.which("curl")
    cmd = [curl, "-L", "--fail", "--continue-at", "-", "--retry", "5",
           "--connect-timeout", "30", "--output", str(out_path), url]
    subprocess.run(cmd, check=True)
    return out_path


def compute_session_area_CVL(session: dict):
    """Compute per-neuron CV_L for all brain areas in one session.

    Returns dict mapping area -> {cvL_spon, cvL_evk, sL_spon, sL_evk, FR_spon, FR_evk}
    """
    spks = session["spks"].astype(float)           # (n_neurons, n_trials, n_timebins)
    spks_passive = session.get("spks_passive")
    brain_area = session["brain_area"]
    bin_size = float(session["bin_size"])           # 0.01 s
    stim_onset_bin = int(session["stim_onset"] / bin_size) if session["stim_onset"] > 1 else \
                     int(session["stim_onset"] * spks.shape[2])  # handle both seconds and fraction

    # stim_onset given in seconds, convert to bin index
    # bin_size=0.01, stim_onset=0.5 -> bin 50
    stim_onset_bin = int(round(float(session["stim_onset"]) / bin_size))
    n_neurons, n_trials, n_timebins = spks.shape

    # Use equal-duration pre- and post-stimulus windows. Count CV depends on
    # integration-window duration, so comparing the 0.5 s baseline with the
    # full 2.0 s post-stimulus epoch would bias Delta CV_L upward.
    evk_stop_bin = min(n_timebins, 2 * stim_onset_bin)
    evk_bins = spks[:, :, stim_onset_bin:evk_stop_bin]   # (n_neurons, n_trials, post_bins)
    evk_counts = evk_bins.sum(axis=2)         # (n_neurons, n_trials)
    evk_dur = (evk_stop_bin - stim_onset_bin) * bin_size

    # Spontaneous (pre-stim): first stim_onset_bin bins of passive trials
    has_passive = spks_passive is not None and hasattr(spks_passive, 'shape') and spks_passive.ndim == 3
    if has_passive:
        sp = spks_passive.astype(float)
        spon_bins = sp[:, :, :stim_onset_bin]   # (n_neurons, n_passive, pre_bins)
        # make sure dimensions match
        if spon_bins.shape[0] != n_neurons:
            has_passive = False
        else:
            spon_counts = spon_bins.sum(axis=2)  # (n_neurons, n_passive)
            spon_dur = stim_onset_bin * bin_size

    results = {}
    unique_areas = np.unique(brain_area)

    for area in unique_areas:
        if area == "root":
            continue
        mask = brain_area == area
        n_area = mask.sum()
        if n_area < MIN_NEURONS:
            continue

        ec = evk_counts[mask]   # (n_area, n_trials)

        cvL_evk_list, sL_evk_list, FR_evk_list = [], [], []
        cvL_spon_list, sL_spon_list, FR_spon_list = [], [], []

        for ni in range(n_area):
            counts_e = ec[ni]
            if len(counts_e) < MIN_TRIALS:
                continue
            mu_e = counts_e.mean()
            if mu_e <= 0:
                continue
            cvL_evk_list.append(counts_e.std(ddof=0) / mu_e)
            sL_evk_list.append(bias_corrected_sparseness(counts_e))
            FR_evk_list.append(mu_e / evk_dur)

            if has_passive:
                counts_s = spon_counts[mask][ni]
                if len(counts_s) < MIN_TRIALS:
                    cvL_spon_list.append(np.nan)
                    sL_spon_list.append(np.nan)
                    FR_spon_list.append(np.nan)
                    continue
                mu_s = counts_s.mean()
                if mu_s <= 0:
                    cvL_spon_list.append(np.nan)
                    sL_spon_list.append(np.nan)
                    FR_spon_list.append(np.nan)
                    continue
                cvL_spon_list.append(counts_s.std(ddof=0) / mu_s)
                sL_spon_list.append(bias_corrected_sparseness(counts_s))
                FR_spon_list.append(mu_s / spon_dur)

        if len(cvL_evk_list) < MIN_NEURONS:
            continue

        entry = dict(
            cvL_evk=np.array(cvL_evk_list),
            sL_evk=np.array(sL_evk_list),
            FR_evk=np.array(FR_evk_list),
        )
        if has_passive and len(cvL_spon_list) == len(cvL_evk_list):
            entry["cvL_spon"] = np.array(cvL_spon_list)
            entry["sL_spon"] = np.array(sL_spon_list)
            entry["FR_spon"] = np.array(FR_spon_list)
        results[area] = entry

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default=str(DEFAULT_DATA_PATH))
    p.add_argument("--download", action="store_true")
    args = p.parse_args()

    data_path = Path(args.data_path)
    if not data_path.exists() or args.download:
        data_path = download_data(DEFAULT_DATA_URL, DEFAULT_DATA_PATH)

    print(f"Loading {data_path}...")
    d = np.load(str(data_path), allow_pickle=True)
    sessions = d["dat"]
    print(f"  {len(sessions)} sessions")

    # Accumulate per area across all sessions
    area_results: dict[str, dict[str, list]] = {}

    for sess_idx, sess in enumerate(sessions):
        sess_results = compute_session_area_CVL(sess)
        for area, res in sess_results.items():
            res["session_id"] = np.full(len(res["cvL_evk"]), sess_idx, dtype=int)
            if area not in area_results:
                area_results[area] = {k: [] for k in res.keys()}
            for k, v in res.items():
                if k in area_results[area]:
                    area_results[area][k].append(v)

    # Concatenate across sessions
    save_dict = {}
    all_cvL_evk, all_cvL_spon = [], []
    all_area_code, all_has_spon = [], []
    area_name_list = []

    for area_idx, (area, arrays) in enumerate(sorted(area_results.items())):
        for k, vlist in arrays.items():
            if vlist:
                merged = np.concatenate(vlist)
                save_dict[f"{k}_{area}"] = merged
        if "cvL_evk" not in save_dict.get(f"cvL_evk_{area}", np.array([])):
            pass  # already saved

        ce = save_dict.get(f"cvL_evk_{area}", np.array([]))
        cs = save_dict.get(f"cvL_spon_{area}", np.array([]))
        if len(ce) == 0:
            continue

        area_name_list.append(area)
        all_cvL_evk.append(ce)
        all_area_code.append(np.full(len(ce), area_idx, dtype=int))
        has_s = len(cs) == len(ce)
        all_has_spon.append(has_s)
        if has_s:
            all_cvL_spon.append(cs)
        else:
            all_cvL_spon.append(np.full(len(ce), np.nan))

        cs_valid = cs[np.isfinite(cs)] if len(cs) > 0 else np.array([])
        print(
            f"[STEINMETZ_{area}] species=mouse region={area} N={len(ce)} "
            f"cv_spon={np.nanmedian(cs_valid):.3f}±{np.nanstd(cs_valid):.3f} "
            f"cv_evk={np.nanmedian(ce):.3f}±{np.nanstd(ce):.3f} "
            f"direction=spon>evk? {'YES' if len(cs_valid)>0 and np.nanmedian(cs_valid) > np.nanmedian(ce) else 'NO/NA'}"
        )

    if all_cvL_evk:
        save_dict["cvL_evk"] = np.concatenate(all_cvL_evk)
        save_dict["cvL_spon"] = np.concatenate(all_cvL_spon)
        save_dict["area_code"] = np.concatenate(all_area_code)
        save_dict["area_names"] = np.array(area_name_list)

        out = BASE / "steinmetz_CVL_results.npz"
        np.savez(str(out), **save_dict)
        print(f"\nSaved: {out}  (total N={len(save_dict['cvL_evk'])})")
    else:
        print("ERROR: no area results computed")


if __name__ == "__main__":
    main()
