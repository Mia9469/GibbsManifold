"""compute_pvc8_CVL.py

Compute CV_L and S_L for cat V1 neurons from the pvc-8 dataset
(CRCNS.org: pvc-8, cat primary visual cortex, drifting gratings + natural images).

resp_train shape: [n_neurons, n_stimuli, n_repeats, n_timebins]
  - 1ms bins, 106ms per trial
  - No spontaneous epoch available

Outputs: Fig2/pvc8_CVL_results.npz
"""
import numpy as np
import scipy.io as sio
from pathlib import Path

BASE = Path(__file__).parent
PVC8 = BASE / "pvc8" / "data"
RECORDING_IDS = [f"{i:02d}" for i in range(1, 11)]  # 01 through 10
MIN_TRIALS = 30


def bias_corrected_sparseness(r: np.ndarray) -> float:
    N = len(r)
    if N < 2:
        return np.nan
    mu = r.mean()
    mu2 = (r ** 2).mean()
    if mu2 == 0:
        return np.nan
    return (1.0 - mu ** 2 / mu2) / (1.0 - 1.0 / N)


def process_recording(mat_path: Path, rec_id: str):
    """Process one pvc-8 recording file.

    Returns (cvL_evk, sL_evk, FR_evk, rec_ids_array) or None if file missing.
    """
    if not mat_path.exists():
        return None

    d = sio.loadmat(str(mat_path))
    rt = d["resp_train"].astype(float)  # (n_neurons, n_stimuli, n_repeats, n_timebins)
    n_neurons, n_stimuli, n_repeats, n_timebins = rt.shape
    dt_ms = 1.0  # 1ms bins
    trial_duration_s = n_timebins * dt_ms / 1000.0  # seconds

    cvL_list, sL_list, FR_list = [], [], []

    for n in range(n_neurons):
        # spike count per trial: sum over time bins, flatten stimuli × repeats
        counts = rt[n].sum(axis=-1).flatten()  # (n_stimuli * n_repeats,)
        if len(counts) < MIN_TRIALS:
            continue
        mu = counts.mean()
        if mu <= 0:
            continue
        cvL = counts.std(ddof=0) / mu
        sL = bias_corrected_sparseness(counts)
        FR = mu / trial_duration_s

        cvL_list.append(cvL)
        sL_list.append(sL)
        FR_list.append(FR)

    n_valid = len(cvL_list)
    if n_valid == 0:
        return None

    return (
        np.array(cvL_list),
        np.array(sL_list),
        np.array(FR_list),
        np.full(n_valid, int(rec_id), dtype=int),
    )


def main():
    all_cvL, all_sL, all_FR, all_rec = [], [], [], []

    for rec_id in RECORDING_IDS:
        mat_path = PVC8 / f"{rec_id}.mat"
        result = process_recording(mat_path, rec_id)
        if result is None:
            print(f"  [{rec_id}] file not found or no valid neurons — skipping")
            continue
        cvL_e, sL_e, FR_e, rec_ids = result
        print(f"  [{rec_id}] N={len(cvL_e)}  CV_L evk median={np.nanmedian(cvL_e):.3f}")
        all_cvL.append(cvL_e)
        all_sL.append(sL_e)
        all_FR.append(FR_e)
        all_rec.append(rec_ids)

    if not all_cvL:
        print("ERROR: no data found in pvc8/data/")
        return

    cvL_evk = np.concatenate(all_cvL)
    sL_evk = np.concatenate(all_sL)
    FR_evk = np.concatenate(all_FR)
    recording_id = np.concatenate(all_rec)

    out = BASE / "pvc8_CVL_results.npz"
    np.savez(str(out),
             cvL_evk=cvL_evk, sL_evk=sL_evk, FR_evk=FR_evk,
             recording_id=recording_id)
    print(f"\nSaved: {out}")
    print(f"Total N={len(cvL_evk)}")
    print(
        f"[PVC8] species=cat region=V1 N={len(cvL_evk)} "
        f"cv_spon=N/A "
        f"cv_evk={np.nanmedian(cvL_evk):.3f}±{np.nanstd(cvL_evk):.3f} "
        f"direction=spon>evk? N/A (no spontaneous data)"
    )


if __name__ == "__main__":
    main()
