"""compute_allen_multiarea_CVL.py

Extend awake_V1_CVL_confirmation.py to ALL visual areas in Allen EcephysSession
715093703.  For each area with >=10 units, compute per-unit CV_L (spon + evoked).

Areas of interest: VISp, VISl, VISal, VISpm, VISam, VISrl

Saves: Fig2/allen_multiarea_CVL.npz
  - per-area arrays named  cvL_spon_<area>, cvL_evk_<area>, etc.
  - flat arrays: cvL_spon, cvL_evk, FR_spon, FR_evk, area_code, unit_id

Usage:
    python compute_allen_multiarea_CVL.py
    python compute_allen_multiarea_CVL.py --direct-download
    python compute_allen_multiarea_CVL.py --nwb-path /path/to/session.nwb
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
import sys
import urllib.request
import warnings
from pathlib import Path

import numpy as np

SESSION_ID = 715093703
SPON_WINDOW_S = 1.50
EVK_WINDOW_S  = 2.00
MIN_TRIALS    = 30
MIN_UNITS_PER_AREA = 10
AREAS_OF_INTEREST = ["VISp", "VISl", "VISal", "VISpm", "VISam", "VISrl"]
ALLEN_NWB_BASE_URL = (
    "https://allen-brain-observatory.s3-us-west-2.amazonaws.com/"
    "visual-coding-neuropixels/ecephys-cache"
)

BASE = Path(__file__).parent


def nwb_url(session_id: int) -> str:
    return (
        f"{ALLEN_NWB_BASE_URL}/session_{session_id}/"
        f"session_{session_id}.nwb"
    )


def direct_download_nwb(session_id: int, cache_dir: str | Path) -> Path:
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"session_{session_id}.nwb"
    url = nwb_url(session_id)
    if out_path.exists():
        print(f"  using existing NWB: {out_path} ({out_path.stat().st_size/1e9:.2f} GB)")
    else:
        print(f"  downloading NWB (~2.9 GB): {url}")
    curl = shutil.which("curl")
    cmd = [curl, "-L", "--fail", "--continue-at", "-", "--retry", "20",
           "--retry-delay", "10", "--retry-all-errors", "--connect-timeout", "30",
           "--output", str(out_path), url]
    subprocess.run(cmd, check=True)
    return out_path


def load_session_allensdk(session_id: int):
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
    except ImportError as e:
        sys.exit(f"allensdk not installed: {e}")
    manifest = Path.home() / "allen-brain-observatory" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest))
    return cache.get_session_data(session_id)


def load_session_from_nwb(nwb_path: str):
    try:
        from allensdk.brain_observatory.ecephys.ecephys_session import EcephysSession
    except ImportError as e:
        sys.exit(f"allensdk required: {e}")
    import h5py
    with h5py.File(nwb_path, "r") as f:
        is_nwb1 = "nwb_version" in f
    if is_nwb1:
        from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysNwb1Api
        api = EcephysNwb1Api(path=nwb_path)
    else:
        from allensdk.brain_observatory.ecephys.ecephys_session_api import EcephysNwbSessionApi
        api = EcephysNwbSessionApi(path=nwb_path)
    return EcephysSession(api=api)


def get_area_col(units):
    for c in ("ecephys_structure_acronym", "structure_acronym", "location"):
        if c in units.columns:
            return c
    raise RuntimeError(f"No area column found. Columns: {list(units.columns)}")


def get_units_for_area(session, area: str):
    units = session.units
    col = get_area_col(units)
    is_area = units[col] == area
    qc = (
        (units["snr"] > 1.0)
        & (units["isi_violations"] < 0.5)
        & (units["firing_rate"] > 0.1)
    )
    keep = is_area & qc
    return units.index[keep].values


def windowize(intervals: np.ndarray, window_s: float) -> np.ndarray:
    wins = []
    for t0, t1 in intervals:
        n = int(np.floor((t1 - t0) / window_s))
        for k in range(n):
            wins.append((t0 + k * window_s, t0 + (k + 1) * window_s))
    return np.asarray(wins)


def count_spikes_per_window(spike_times, windows):
    starts = windows[:, 0]
    stops  = windows[:, 1]
    s = np.searchsorted(spike_times, starts, side="left")
    e = np.searchsorted(spike_times, stops,  side="right")
    return e - s


def get_spontaneous_intervals(session):
    stim = session.stimulus_presentations
    spon = stim[stim["stimulus_name"] == "spontaneous"]
    if len(spon) == 0:
        raise RuntimeError("no spontaneous block")
    return spon[["start_time", "stop_time"]].values


def get_grating_intervals(session):
    stim = session.stimulus_presentations
    drift = stim[stim["stimulus_name"] == "drifting_gratings"]
    if len(drift) == 0:
        drift = stim[stim["stimulus_name"] == "static_gratings"]
        if len(drift) == 0:
            raise RuntimeError("no gratings block")
    return drift[["start_time", "stop_time"]].values


def compute_area_CVL(session, unit_ids, area):
    try:
        spon_iv = get_spontaneous_intervals(session)
        evk_iv  = get_grating_intervals(session)
    except RuntimeError as e:
        print(f"  [{area}] skipping: {e}")
        return None

    spon_wins = windowize(spon_iv, SPON_WINDOW_S)
    evk_wins  = windowize(evk_iv,  EVK_WINDOW_S)

    cvL_spon = np.full(len(unit_ids), np.nan)
    cvL_evk  = np.full(len(unit_ids), np.nan)
    FR_spon  = np.full(len(unit_ids), np.nan)
    FR_evk   = np.full(len(unit_ids), np.nan)

    for i, uid in enumerate(unit_ids):
        st = np.asarray(session.spike_times[uid])
        if st.size < 100:
            continue
        c_spon = count_spikes_per_window(st, spon_wins) / SPON_WINDOW_S
        c_evk  = count_spikes_per_window(st, evk_wins) / EVK_WINDOW_S
        if c_spon.size < MIN_TRIALS or c_evk.size < MIN_TRIALS:
            continue
        mu_s = c_spon.mean(); mu_e = c_evk.mean()
        if mu_s <= 0 or mu_e <= 0:
            continue
        cvL_spon[i] = c_spon.std(ddof=0) / mu_s
        cvL_evk[i]  = c_evk.std(ddof=0) / mu_e
        FR_spon[i] = mu_s
        FR_evk[i]  = mu_e

    keep = np.isfinite(cvL_spon) & np.isfinite(cvL_evk)
    n = int(keep.sum())
    if n < MIN_UNITS_PER_AREA:
        print(f"  [{area}] only {n} valid units — skipping")
        return None
    print(
        f"  [{area}] N={n}  "
        f"CV_L spon median={np.nanmedian(cvL_spon[keep]):.3f}  "
        f"evk median={np.nanmedian(cvL_evk[keep]):.3f}  "
        f"spon>evk: {'YES' if np.nanmedian(cvL_spon[keep]) > np.nanmedian(cvL_evk[keep]) else 'NO'}"
    )
    return {
        "unit_id":  unit_ids[keep],
        "cvL_spon": cvL_spon[keep],
        "cvL_evk":  cvL_evk[keep],
        "FR_spon":  FR_spon[keep],
        "FR_evk":   FR_evk[keep],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--direct-download", action="store_true")
    p.add_argument("--nwb-path", default=None)
    p.add_argument("--cache-dir", default=str(Path.home() / "allen-brain-observatory" / "direct-nwb"))
    args = p.parse_args()

    print("=== compute_allen_multiarea_CVL.py ===")
    if args.nwb_path:
        session = load_session_from_nwb(args.nwb_path)
    elif args.direct_download:
        nwb_path = direct_download_nwb(SESSION_ID, args.cache_dir)
        session = load_session_from_nwb(str(nwb_path))
    else:
        session = load_session_allensdk(SESSION_ID)

    all_cvL_spon, all_cvL_evk = [], []
    all_FR_spon, all_FR_evk = [], []
    all_area_code, all_unit_id = [], []
    per_area = {}

    for area_idx, area in enumerate(AREAS_OF_INTEREST):
        uids = get_units_for_area(session, area)
        print(f"\n[{area}] total QC-passing units: {len(uids)}")
        if len(uids) < MIN_UNITS_PER_AREA:
            print(f"  [{area}] skipping — too few units")
            continue
        result = compute_area_CVL(session, uids, area)
        if result is None:
            continue
        per_area[area] = result
        all_cvL_spon.append(result["cvL_spon"])
        all_cvL_evk.append(result["cvL_evk"])
        all_FR_spon.append(result["FR_spon"])
        all_FR_evk.append(result["FR_evk"])
        all_area_code.append(np.full(len(result["cvL_spon"]), area_idx, dtype=int))
        all_unit_id.append(result["unit_id"])

    if not all_cvL_spon:
        print("ERROR: no areas with sufficient data")
        return

    save_dict = {}
    for area, res in per_area.items():
        for k, v in res.items():
            save_dict[f"{k}_{area}"] = v
    # flat arrays
    save_dict["cvL_spon"]   = np.concatenate(all_cvL_spon)
    save_dict["cvL_evk"]    = np.concatenate(all_cvL_evk)
    save_dict["FR_spon"]    = np.concatenate(all_FR_spon)
    save_dict["FR_evk"]     = np.concatenate(all_FR_evk)
    save_dict["area_code"]  = np.concatenate(all_area_code)
    save_dict["unit_id"]    = np.concatenate(all_unit_id)
    save_dict["area_names"] = np.array(list(per_area.keys()))

    out = BASE / "allen_multiarea_CVL.npz"
    np.savez(str(out), **save_dict)
    print(f"\nSaved: {out}")
    total_n = len(save_dict["cvL_spon"])
    print(f"Total units: {total_n}")

    for area, res in per_area.items():
        cs = res["cvL_spon"]; ce = res["cvL_evk"]
        print(
            f"[ALLEN_{area}] species=mouse region={area} N={len(cs)} "
            f"cv_spon={np.nanmedian(cs):.3f}±{np.nanstd(cs):.3f} "
            f"cv_evk={np.nanmedian(ce):.3f}±{np.nanstd(ce):.3f} "
            f"direction=spon>evk? {'YES' if np.nanmedian(cs) > np.nanmedian(ce) else 'NO'}"
        )
    print("=== done ===")


if __name__ == "__main__":
    main()
