"""compute_pvc11_CVL.py

Compute CV_L and S_L for macaque V1 neurons from the pvc-11 dataset
(CRCNS.org: pvc-11, Ecker et al., macaque V1, 3 monkeys).

Outputs: Fig2/pvc11_CVL_results.npz
"""
import numpy as np
import scipy.io as sio
from pathlib import Path

BASE = Path(__file__).parent
PVC11 = BASE / "pvc11" / "data_and_scripts"
SPON_DIR = PVC11 / "spikes_spontaneous"
GRAT_DIR = PVC11 / "spikes_gratings"

SPON_WINDOW_S = 1.5       # seconds per spontaneous bin
GRAT_DURATION_S = 1.28    # seconds per grating presentation
MIN_WINDOWS = 30          # minimum windows/trials for CV_L


def bias_corrected_sparseness(r: np.ndarray) -> float:
    """Vinje-Gallant bias-corrected Treves-Rolls sparseness.

    S = (1 - mean(r)^2 / mean(r^2)) / (1 - 1/N)
    where N = len(r).  Returns nan if N < 2 or mean(r^2) == 0.
    """
    N = len(r)
    if N < 2:
        return np.nan
    mu = r.mean()
    mu2 = (r ** 2).mean()
    if mu2 == 0:
        return np.nan
    return (1.0 - mu ** 2 / mu2) / (1.0 - 1.0 / N)


def compute_spont_CVL(spont_mat_path: Path):
    """Return (cvL, sL, FR) arrays for spontaneous epoch."""
    d = sio.loadmat(str(spont_mat_path))
    events = d["data"]["EVENTS"][0, 0]          # shape (1, n_neurons)
    n_neurons = events.shape[1]

    cvL_list, sL_list, FR_list = [], [], []

    for i in range(n_neurons):
        spike_times = events[0, i].flatten().astype(float)
        if spike_times.size == 0:
            continue
        total_time = spike_times[-1]
        n_wins = int(np.floor(total_time / SPON_WINDOW_S))
        if n_wins < MIN_WINDOWS:
            continue

        # bin into fixed windows
        edges = np.arange(0, n_wins + 1) * SPON_WINDOW_S
        counts, _ = np.histogram(spike_times, bins=edges)
        counts = counts.astype(float)

        mu = counts.mean()
        if mu <= 0:
            continue

        cvL = counts.std(ddof=0) / mu
        sL = bias_corrected_sparseness(counts)
        FR = mu / SPON_WINDOW_S

        cvL_list.append(cvL)
        sL_list.append(sL)
        FR_list.append(FR)

    return (
        np.array(cvL_list),
        np.array(sL_list),
        np.array(FR_list),
    )


def compute_evoked_CVL(grat_mat_path: Path):
    """Return (cvL, sL, FR) arrays for evoked epoch (drifting gratings).

    EVENTS shape: (n_neurons, n_gratings, n_repeats) cell array of spike-time arrays.
    Each cell contains spike times relative to grating onset (seconds).
    readme: data.EVENTS (num_neurons x num_gratings x num_trials)
    """
    d = sio.loadmat(str(grat_mat_path))
    events = d["data"]["EVENTS"][0, 0]   # shape (n_neurons, n_gratings, n_repeats)
    n_neurons, n_gratings, n_repeats = events.shape

    cvL_list, sL_list, FR_list = [], [], []

    for neu in range(n_neurons):
        # Collect per-trial spike counts across all gratings and repeats
        trial_counts = []
        for g in range(n_gratings):
            for r in range(n_repeats):
                cell = events[neu, g, r]
                if not hasattr(cell, '__len__') or cell.dtype.kind in ('u', 'V'):
                    # empty cell (stored as uint8 empty array)
                    trial_counts.append(0)
                else:
                    st = cell.flatten()
                    # count spikes within [0, GRAT_DURATION_S]
                    n_spk = int(np.sum((st >= 0) & (st <= GRAT_DURATION_S)))
                    trial_counts.append(n_spk)

        counts = np.array(trial_counts, dtype=float)
        if len(counts) < MIN_WINDOWS:
            continue
        mu = counts.mean()
        if mu <= 0:
            continue

        cvL = counts.std(ddof=0) / mu
        sL = bias_corrected_sparseness(counts)
        FR = mu / GRAT_DURATION_S

        cvL_list.append(cvL)
        sL_list.append(sL)
        FR_list.append(FR)

    return (
        np.array(cvL_list),
        np.array(sL_list),
        np.array(FR_list),
    )


def main():
    all_cvL_spon, all_sL_spon, all_FR_spon = [], [], []
    all_cvL_evk, all_sL_evk, all_FR_evk = [], [], []
    all_monkey_id = []

    for monkey_id in [1, 2, 3]:
        print(f"\n--- Monkey {monkey_id} ---")
        spont_path = SPON_DIR / f"spiketimesmonkey{monkey_id}spont.mat"
        grat_path = GRAT_DIR / f"data_monkey{monkey_id}_gratings.mat"

        cvL_s, sL_s, FR_s = compute_spont_CVL(spont_path)
        cvL_e, sL_e, FR_e = compute_evoked_CVL(grat_path)

        # Align by channel count (use min to match up)
        n = min(len(cvL_s), len(cvL_e))
        if n == 0:
            print(f"  WARNING: no valid neurons for monkey {monkey_id}")
            continue
        cvL_s, sL_s, FR_s = cvL_s[:n], sL_s[:n], FR_s[:n]
        cvL_e, sL_e, FR_e = cvL_e[:n], sL_e[:n], FR_e[:n]

        print(f"  N neurons (paired): {n}")
        print(f"  CV_L spon: median={np.nanmedian(cvL_s):.3f}  evk: median={np.nanmedian(cvL_e):.3f}")
        direction = "YES" if np.nanmedian(cvL_s) > np.nanmedian(cvL_e) else "NO"
        print(f"  spon > evk: {direction}")

        all_cvL_spon.append(cvL_s)
        all_sL_spon.append(sL_s)
        all_FR_spon.append(FR_s)
        all_cvL_evk.append(cvL_e)
        all_sL_evk.append(sL_e)
        all_FR_evk.append(FR_e)
        all_monkey_id.append(np.full(n, monkey_id, dtype=int))

    cvL_spon = np.concatenate(all_cvL_spon)
    sL_spon = np.concatenate(all_sL_spon)
    FR_spon = np.concatenate(all_FR_spon)
    cvL_evk = np.concatenate(all_cvL_evk)
    sL_evk = np.concatenate(all_sL_evk)
    FR_evk = np.concatenate(all_FR_evk)
    monkey_id = np.concatenate(all_monkey_id)

    out = BASE / "pvc11_CVL_results.npz"
    np.savez(str(out),
             cvL_spon=cvL_spon, sL_spon=sL_spon, FR_spon=FR_spon,
             cvL_evk=cvL_evk, sL_evk=sL_evk, FR_evk=FR_evk,
             monkey_id=monkey_id)
    print(f"\nSaved: {out}")
    print(f"Total N={len(cvL_spon)}")
    print(
        f"[PVC11] species=macaque region=V1 N={len(cvL_spon)} "
        f"cv_spon={np.nanmedian(cvL_spon):.3f}±{np.nanstd(cvL_spon):.3f} "
        f"cv_evk={np.nanmedian(cvL_evk):.3f}±{np.nanstd(cvL_evk):.3f} "
        f"direction=spon>evk? {'YES' if np.nanmedian(cvL_spon) > np.nanmedian(cvL_evk) else 'NO'}"
    )


if __name__ == "__main__":
    main()
