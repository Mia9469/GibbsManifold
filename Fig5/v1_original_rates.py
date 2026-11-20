"""Aligned original-count V1 rates paired with unchanged entropy estimates."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "final_text_review_20260907" / "v1_corrected"
C0, C1 = 2.4, 0.37
RATE_PROVENANCE = {
    "evoked": "Fig2/v1_CVL_perN.mat: FR_drv, original trial counts / 1.28 s",
    "spontaneous": (
        "Original neuralData.spikeRasters[:, 0] spontaneous trial counts / 1.5 s; "
        "full 1500-bin window matched to the unchanged spontaneous entropy estimate"
    ),
    "cost": "2.4 + 0.37 * original_trial_count_firing_rate_Hz",
    "entropy": "Unchanged unified_v1_Hrho_ICE.mat and spon_Hr_perN.mat estimates",
}


def load_evoked_sessions() -> list[dict]:
    info = loadmat(ROOT / "Fig2" / "unified_v1_Hrho_ICE.mat",
                   squeeze_me=True, struct_as_record=False)
    counts = loadmat(ROOT / "Fig2" / "v1_CVL_perN.mat", squeeze_me=True)
    ids = np.ravel(counts["sess_id"]).astype(int)
    names = [str(name) for name in np.ravel(counts["V1V2_FILES"])]
    fr = np.ravel(counts["FR_drv"]).astype(float)
    cv = np.ravel(counts["cvL_drv"]).astype(float)
    sessions = np.atleast_1d(info["results"].v1v2)
    if len(sessions) != len(names) or not (ids.size == fr.size == cv.size):
        raise ValueError("V1 original-rate and entropy cache sizes disagree")
    if not np.array_equal(np.unique(ids), np.arange(1, len(names) + 1)):
        raise ValueError("Unexpected V1 session identifiers")
    rows = []
    for index, session in enumerate(sessions, start=1):
        if Path(str(session.file)).name != names[index - 1]:
            raise ValueError(f"V1 session order mismatch at session {index}")
        mask = ids == index
        row = {"session": names[index - 1], "session_index": index,
               "fr": fr[mask], "cv": cv[mask]}
        row.update({key: np.ravel(getattr(session, key)).astype(float)
                    for key in ("hr", "hn", "mi", "rho")})
        if {np.size(row[key]) for key in ("fr", "cv", "hr", "hn", "mi", "rho")} != {int(session.n_v1)}:
            raise ValueError(f"Unaligned V1 neurons in {row['session']}")
        row["cost"] = C0 + C1 * row["fr"]
        rows.append(row)
    return rows


def load_spontaneous() -> dict:
    entropy = loadmat(ROOT / "Fig2" / "spon_Hr_perN.mat", squeeze_me=True)
    sessions = load_evoked_sessions()
    ids = np.ravel(entropy["sess_id"]).astype(int)
    h = np.ravel(entropy["hr_spon"]).astype(float)
    expected_ids = np.concatenate([
        np.full(row["fr"].size, row["session_index"], int) for row in sessions
    ])
    if not np.array_equal(ids, expected_ids) or h.size != ids.size:
        raise ValueError("Spontaneous entropy neuron order disagrees with original V1 rates")
    if not np.isclose(float(entropy["DUR_SPON"]), 1.5):
        raise ValueError("Expected a 1.5 s spontaneous entropy window")
    rates, trial_counts = [], []
    for row in sessions:
        raw = loadmat(ROOT / "Fig2" / row["session"],
                      squeeze_me=False, struct_as_record=False)
        rasters = raw["neuralData"][0, 0].spikeRasters[:, 0]
        counts = [np.asarray(raster.sum(axis=1)).ravel().astype(float)
                  for raster in rasters if raster.shape[1] == 1500]
        if not counts or any(count.size != row["fr"].size for count in counts):
            raise ValueError(f"Unaligned spontaneous original counts in {row['session']}")
        rates.append(np.mean(counts, axis=0) / 1.5)
        trial_counts.append(len(counts))
    fr = np.concatenate(rates)
    return {"fr": fr, "hr": h, "cost": C0 + C1 * fr, "session_ids": ids,
            "session_names": [row["session"] for row in sessions],
            "n_trials_per_session": trial_counts}
