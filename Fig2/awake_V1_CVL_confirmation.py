"""awake_V1_CVL_confirmation.py

Confirmation of the spontaneous-vs-evoked CV_L direction on AWAKE mouse V1,
as an anesthesia control for the result reported in the main text
(Fig. 2C, which uses sufentanil-anesthetized macaque V1).

We use the Allen Brain Observatory Visual Coding Neuropixels dataset
(http://observatory.brain-map.org/visualcoding/) because each session
contains both:
  - a stimulus-evoked epoch (drifting gratings), and
  - a spontaneous-activity epoch (~30 min of gray-screen baseline,
    interleaved between stimulus blocks)
allowing a within-session, within-unit paired CV_L contrast.

Workflow:
  1. (one-time, only if downloading through AllenSDK) install allensdk
        pip install allensdk
     A local NWB2 file is read directly with h5py in low-memory mode; AllenSDK
     is not used for --nwb-path or --direct-download analyses.
  2. Run this script.  It downloads ONE session (~2.9 GB) on first run
     and caches it; subsequent runs are seconds.  If AllenSDK download
     hangs, use --direct-download for curl-based resume/retry.
  3. Outputs:
       awake_V1_CVL.npz   - per-unit vectors:
            cvL_spon : CV_L on spontaneous 1.5 s windows
            cvL_evk  : CV_L on drifting-grating 1.5 s windows
            FR_spon, FR_evk : mean firing rates
            unit_id  : Allen unit id
       FigureS15.pdf / .png  -  3-panel supp figure
       summary in stdout

Approx runtime: 5-15 min on first run (dominated by NWB download).
Subsequent runs: <1 min.

If the user does not want to install allensdk, the script also accepts
a path to a previously-downloaded session NWB file (see CLI args at top).

Usage:
    python awake_V1_CVL_confirmation.py
    python awake_V1_CVL_confirmation.py --direct-download --copy-to-manuscript
    python awake_V1_CVL_confirmation.py --direct-download --download-only
    python awake_V1_CVL_confirmation.py --session-id 715093703
    python awake_V1_CVL_confirmation.py --nwb-path /path/to/session.nwb
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import warnings
from collections.abc import Mapping
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

# Default session: a Brain Observatory "functional connectivity" session
# with a clean V1 yield and both spontaneous + drifting-gratings blocks.
DEFAULT_SESSION_ID = 715093703
SPON_WINDOW_S = 1.50           # match the macaque-V1 spontaneous window
EVK_WINDOW_S  = SPON_WINDOW_S  # matched duration avoids count-CV window bias
MIN_TRIALS_PER_COND = 30       # require at least this many trials per cond per unit
ALLEN_NWB_BASE_URL = (
    "https://allen-brain-observatory.s3-us-west-2.amazonaws.com/"
    "visual-coding-neuropixels/ecephys-cache"
)


def nwb_url(session_id: int) -> str:
    """Public S3 HTTPS URL for one Allen Visual Coding NPx NWB file."""
    return (
        f"{ALLEN_NWB_BASE_URL}/session_{session_id}/"
        f"session_{session_id}.nwb"
    )


def remote_size_bytes(url: str) -> int | None:
    """Return Content-Length for a remote file, or None if unavailable."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            size = resp.headers.get("Content-Length")
    except Exception as e:
        warnings.warn(f"Could not query remote file size: {e}")
        return None
    return int(size) if size is not None else None


def direct_download_nwb(session_id: int, cache_dir: str | Path) -> Path:
    """Download NWB with curl resume/retry, bypassing AllenSDK download."""
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"session_{session_id}.nwb"
    url = nwb_url(session_id)
    expected = remote_size_bytes(url)

    if out_path.exists() and expected is not None:
        have = out_path.stat().st_size
        if have == expected:
            print(f"  using existing complete NWB: {out_path}")
            return out_path
        print(
            f"  found partial NWB: {out_path} "
            f"({have / 1e9:.2f} / {expected / 1e9:.2f} GB); resuming"
        )
    elif expected is not None:
        print(f"  downloading NWB ({expected / 1e9:.2f} GB): {url}")
    else:
        print(f"  downloading NWB: {url}")

    curl = shutil.which("curl")
    if curl is None:
        raise RuntimeError("curl is required for --direct-download")

    cmd = [
        curl,
        "-L",
        "--fail",
        "--continue-at", "-",
        "--retry", "20",
        "--retry-delay", "10",
        "--retry-all-errors",
        "--connect-timeout", "30",
        "--output", str(out_path),
        url,
    ]
    print("  running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    if expected is not None:
        have = out_path.stat().st_size
        if have != expected:
            raise RuntimeError(
                f"Incomplete NWB after download: got {have} bytes, "
                f"expected {expected}. Re-run the same command to resume."
            )
    return out_path


def load_session_allensdk(session_id: int):
    """Download + open one EcephysSession from the AllenSDK."""
    try:
        from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
            EcephysProjectCache,
        )
    except ImportError as e:
        sys.exit(
            "allensdk is not installed.  Run:\n"
            "    pip install allensdk\n"
            f"(reported error: {e})"
        )

    manifest_path = (
        Path.home() / "allen-brain-observatory" / "manifest.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    cache = EcephysProjectCache.from_warehouse(manifest=str(manifest_path))
    print(f"  fetching session {session_id} (downloads on first run)...")
    session = cache.get_session_data(session_id)
    return session


def _decode_h5_strings(values: np.ndarray) -> np.ndarray:
    """Decode fixed/vlen byte strings while leaving numeric arrays unchanged."""
    values = np.asarray(values)
    if values.dtype.kind == "S":
        return np.char.decode(values, "utf-8")
    if values.dtype.kind == "O":
        return np.asarray([
            x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x
            for x in values
        ])
    return values


def _read_h5_vector(group: h5py.Group, name: str) -> np.ndarray:
    """Read a simple DynamicTable VectorData column from an NWB2 group."""
    if name not in group:
        raise KeyError(f"Missing NWB column: {group.name}/{name}")
    node = group[name]
    if isinstance(node, h5py.Dataset):
        values = node[()]
    elif isinstance(node, h5py.Group) and "data" in node:
        values = node["data"][()]
    else:
        raise TypeError(f"Unsupported NWB column layout: {node.name}")
    return _decode_h5_strings(values)


class _LazySpikeTimes(Mapping):
    """Mapping from unit id to one on-demand slice of the NWB spike-time array."""

    def __init__(self, units_group: h5py.Group, unit_ids: np.ndarray):
        self._values = units_group["spike_times"]
        self._stops = np.asarray(units_group["spike_times_index"][()], dtype=np.int64)
        self._unit_ids = np.asarray(unit_ids)
        self._row_by_id = {int(uid): i for i, uid in enumerate(self._unit_ids)}

    def __getitem__(self, unit_id: int) -> np.ndarray:
        row = self._row_by_id[int(unit_id)]
        start = 0 if row == 0 else int(self._stops[row - 1])
        stop = int(self._stops[row])
        return np.asarray(self._values[start:stop], dtype=float)

    def __iter__(self):
        return iter(self._row_by_id)

    def __len__(self) -> int:
        return len(self._row_by_id)


class LowMemoryNwbSession:
    """Minimal session interface backed by lazy h5py reads.

    AllenSDK can materialize large NWB tables and be killed by macOS under
    memory pressure. This reader loads only small metadata tables and reads
    spike times for selected units one slice at a time.
    """

    UNIT_COLUMNS = (
        "ecephys_structure_acronym",
        "structure_acronym",
        "location",
        "peak_channel_id",
        "snr",
        "isi_violations",
        "firing_rate",
    )

    def __init__(self, nwb_path: str):
        self.path = str(Path(nwb_path).expanduser().resolve())
        self._file = h5py.File(self.path, "r")
        if "units" not in self._file:
            self.close()
            raise RuntimeError(f"NWB2 units table not found in {self.path}")

        units_group = self._file["units"]
        unit_ids = np.asarray(_read_h5_vector(units_group, "id"), dtype=np.int64)
        columns = {}
        for name in self.UNIT_COLUMNS:
            if name in units_group:
                values = _read_h5_vector(units_group, name)
                if len(values) == len(unit_ids):
                    columns[name] = values
        if not any(
            name in columns
            for name in ("ecephys_structure_acronym", "structure_acronym", "location")
        ):
            columns["ecephys_structure_acronym"] = self._map_unit_brain_areas(
                units_group, unit_ids
            )
        self.units = pd.DataFrame(columns, index=unit_ids)
        self.spike_times = _LazySpikeTimes(units_group, unit_ids)
        self.stimulus_presentations = self._load_stimulus_presentations()

        print(
            f"  low-memory NWB reader: {len(unit_ids)} units, "
            f"{len(self.stimulus_presentations)} stimulus intervals"
        )

    def _map_unit_brain_areas(
        self, units_group: h5py.Group, unit_ids: np.ndarray
    ) -> np.ndarray:
        """Map unit peak-channel ids to electrode structure acronyms."""
        peak_name = next(
            (name for name in ("peak_channel_id", "peak_channel") if name in units_group),
            None,
        )
        electrodes_path = "general/extracellular_ephys/electrodes"
        if peak_name is None or electrodes_path not in self._file:
            raise RuntimeError(
                "Cannot map units to brain areas. Expected a unit peak-channel "
                f"column and /{electrodes_path}.\n"
                f"Available unit columns: {list(units_group.keys())}\n"
                f"Has electrodes table: {electrodes_path in self._file}"
            )

        peak_channels = np.asarray(_read_h5_vector(units_group, peak_name))
        if len(peak_channels) != len(unit_ids):
            raise RuntimeError(
                f"{units_group.name}/{peak_name} has {len(peak_channels)} rows, "
                f"expected {len(unit_ids)}"
            )

        electrodes = self._file[electrodes_path]
        electrode_ids = np.asarray(_read_h5_vector(electrodes, "id"))
        area_name = next(
            (
                name
                for name in (
                    "manual_structure_acronym",
                    "ecephys_structure_acronym",
                    "structure_acronym",
                    "location",
                )
                if name in electrodes
            ),
            None,
        )
        if area_name is None:
            raise RuntimeError(
                "Cannot find a brain-area acronym in the electrodes table.\n"
                f"Available electrode columns: {list(electrodes.keys())}"
            )
        electrode_areas = np.asarray(_read_h5_vector(electrodes, area_name), dtype=object)
        if len(electrode_areas) != len(electrode_ids):
            raise RuntimeError(
                f"{electrodes.name}/{area_name} has {len(electrode_areas)} rows, "
                f"expected {len(electrode_ids)}"
            )

        area_by_id = {
            int(channel_id): area
            for channel_id, area in zip(electrode_ids, electrode_areas)
        }
        mapped = np.asarray([
            None if pd.isna(channel_id) else area_by_id.get(int(channel_id), None)
            for channel_id in peak_channels
        ], dtype=object)

        # DynamicTableRegion values are sometimes stored as electrode row
        # indices rather than the electrode IDs exposed by PyNWB.
        peak_numeric = pd.to_numeric(pd.Series(peak_channels), errors="coerce").to_numpy()
        missing = pd.isna(mapped)
        positional = (
            missing
            & np.isfinite(peak_numeric)
            & (peak_numeric >= 0)
            & (peak_numeric < len(electrode_areas))
        )
        if np.any(positional):
            mapped[positional] = electrode_areas[peak_numeric[positional].astype(int)]

        normalized = np.asarray([
            np.nan if pd.isna(area) or area in ("", "None") else str(area)
            for area in mapped
        ], dtype=object)
        n_mapped = int(np.sum(~pd.isna(normalized)))
        print(
            f"  mapped brain areas via units/{peak_name} -> "
            f"electrodes/{area_name}: {n_mapped} / {len(unit_ids)} units"
        )
        return normalized

    def _load_stimulus_presentations(self) -> pd.DataFrame:
        interval_groups = []
        if "intervals/stimulus_presentations" in self._file:
            interval_groups.append(self._file["intervals/stimulus_presentations"])
        elif "intervals" in self._file:
            interval_groups.extend(
                group
                for name, group in self._file["intervals"].items()
                if name.endswith("_presentations") and isinstance(group, h5py.Group)
            )

        if not interval_groups:
            self.close()
            raise RuntimeError(
                "Cannot find NWB stimulus presentation tables under /intervals."
            )

        tables = []
        for group in interval_groups:
            required = ("start_time", "stop_time")
            missing = [name for name in required if name not in group]
            if missing:
                self.close()
                raise RuntimeError(
                    f"Missing stimulus presentation columns {missing} in {group.name}. "
                    f"Available: {list(group.keys())}"
                )
            table = pd.DataFrame({
                name: _read_h5_vector(group, name)
                for name in required
            })
            if "stimulus_name" in group:
                table["stimulus_name"] = _read_h5_vector(group, "stimulus_name")
            else:
                table["stimulus_name"] = group.name.rsplit("/", 1)[-1].removesuffix(
                    "_presentations"
                )
            tables.append(table)

        return (
            pd.concat(tables, ignore_index=True)
            .sort_values("start_time", kind="stable")
            .reset_index(drop=True)
        )

    def close(self) -> None:
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        self.close()


def load_session_from_nwb(nwb_path: str):
    """Open an Allen NWB2 file without materializing the full session."""
    with h5py.File(nwb_path, "r") as f:
        if "nwb_version" in f:
            raise RuntimeError(
                "The low-memory direct reader supports Allen NWB2 files only. "
                "Use the AllenSDK cache path for NWB1 files."
            )
    return LowMemoryNwbSession(nwb_path)


def get_V1_unit_ids(session) -> np.ndarray:
    """Filter session units to V1/VISp with reasonable QC.

    Column names differ between EcephysProjectCache and EcephysNwbSessionApi
    (NWB v2 direct load), so we probe for each possibility.
    """
    units = session.units

    # Brain-area column: name varies by AllenSDK version / load path
    for area_col in ("ecephys_structure_acronym", "structure_acronym", "location"):
        if area_col in units.columns:
            break
    else:
        raise RuntimeError(
            f"Cannot find brain-area column in units DataFrame.\n"
            f"Available columns: {list(units.columns)}"
        )

    is_v1 = units[area_col] == "VISp"

    # QC columns — snr/isi_violations/firing_rate should exist in both paths,
    # but guard against missing ones so the error is legible.
    missing = [c for c in ("snr", "isi_violations", "firing_rate")
               if c not in units.columns]
    if missing:
        raise RuntimeError(
            f"Missing QC columns: {missing}\n"
            f"Available columns: {list(units.columns)}"
        )

    qc = (
        (units["snr"] > 1.0)
        & (units["isi_violations"] < 0.5)
        & (units["firing_rate"] > 0.1)
    )
    keep = is_v1 & qc
    print(
        f"  V1/VISp units passing QC: "
        f"{int(keep.sum())} / {int(is_v1.sum())} (V1 total) "
        f"/ {len(units)} (session total)"
    )
    return units.index[keep].values


def get_spontaneous_intervals(session) -> np.ndarray:
    """[N x 2] start/stop times (s) of spontaneous (gray-screen) epochs."""
    stim_table = session.stimulus_presentations
    spon = stim_table[stim_table["stimulus_name"] == "spontaneous"]
    if len(spon) == 0:
        raise RuntimeError("no 'spontaneous' stimulus block in this session")
    return spon[["start_time", "stop_time"]].values


def get_grating_intervals(session) -> np.ndarray:
    """[N x 2] start/stop of drifting-grating presentations."""
    stim_table = session.stimulus_presentations
    drift = stim_table[stim_table["stimulus_name"] == "drifting_gratings"]
    if len(drift) == 0:
        # fall back to static gratings if drifting are absent in this session
        drift = stim_table[stim_table["stimulus_name"] == "static_gratings"]
        if len(drift) == 0:
            raise RuntimeError("no gratings stimulus block in this session")
        print("  (using static_gratings as evoked block)")
    return drift[["start_time", "stop_time"]].values


def windowize(intervals: np.ndarray, window_s: float) -> np.ndarray:
    """Split each [start, stop] interval into non-overlapping window_s chunks.

    Returns an [M x 2] array of [start, stop] per window.
    """
    wins = []
    for t0, t1 in intervals:
        n = int(np.floor((t1 - t0) / window_s))
        for k in range(n):
            wins.append((t0 + k * window_s, t0 + (k + 1) * window_s))
    return np.asarray(wins)


def count_spikes_per_window(
    spike_times: np.ndarray, windows: np.ndarray
) -> np.ndarray:
    """Vectorised spike counts for one unit across N windows."""
    starts = windows[:, 0]
    stops = windows[:, 1]
    # binary search on sorted spike times
    s = np.searchsorted(spike_times, starts, side="left")
    e = np.searchsorted(spike_times, stops, side="right")
    return e - s


def compute_per_unit_CVL(session, unit_ids: np.ndarray):
    """Per-unit CV_L on spontaneous vs evoked windows."""
    spon_iv = get_spontaneous_intervals(session)
    evk_iv = get_grating_intervals(session)

    spon_wins = windowize(spon_iv, SPON_WINDOW_S)
    evk_wins = windowize(evk_iv, EVK_WINDOW_S)
    print(
        f"  spontaneous windows: {len(spon_wins)} ({SPON_WINDOW_S} s each)"
    )
    print(f"  evoked windows:      {len(evk_wins)} ({EVK_WINDOW_S} s each)")

    cvL_spon = np.full(len(unit_ids), np.nan)
    cvL_evk = np.full(len(unit_ids), np.nan)
    FR_spon = np.full(len(unit_ids), np.nan)
    FR_evk = np.full(len(unit_ids), np.nan)

    for i, uid in enumerate(unit_ids):
        st = np.asarray(session.spike_times[uid])
        if st.size < 100:
            continue
        c_spon = count_spikes_per_window(st, spon_wins) / SPON_WINDOW_S
        c_evk = count_spikes_per_window(st, evk_wins) / EVK_WINDOW_S
        if (c_spon.size < MIN_TRIALS_PER_COND
                or c_evk.size < MIN_TRIALS_PER_COND):
            continue
        mu_s = c_spon.mean();  mu_e = c_evk.mean()
        if mu_s <= 0 or mu_e <= 0:
            continue
        cvL_spon[i] = c_spon.std(ddof=0) / mu_s
        cvL_evk[i] = c_evk.std(ddof=0) / mu_e
        FR_spon[i] = mu_s
        FR_evk[i] = mu_e

    keep = np.isfinite(cvL_spon) & np.isfinite(cvL_evk)
    print(f"  per-unit CV_L computed for {int(keep.sum())} / {len(unit_ids)} units")
    return {
        "unit_id":  unit_ids[keep],
        "cvL_spon": cvL_spon[keep],
        "cvL_evk":  cvL_evk[keep],
        "FR_spon":  FR_spon[keep],
        "FR_evk":   FR_evk[keep],
    }


def make_figure(R: dict, out_pdf: str, out_png: str):
    spon = R["cvL_spon"];  evk = R["cvL_evk"]
    diff = spon - evk

    col_spon = (0.30, 0.60, 0.90)
    col_evk  = (0.85, 0.33, 0.10)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    # A: per-unit boxplot/violin
    ax = axes[0]
    parts = ax.violinplot(
        [spon, evk], positions=[1, 2], showmedians=True, widths=0.7
    )
    for pc, c in zip(parts["bodies"], [col_spon, col_evk]):
        pc.set_facecolor(c); pc.set_alpha(0.6); pc.set_edgecolor("k")
    ax.set_xticks([1, 2]); ax.set_xticklabels(["spontaneous", "evoked"])
    ax.set_ylabel("per-unit CV_L  (awake mouse V1)")
    ax.set_title("A: per-unit CV_L distributions")
    ax.spines[["top", "right"]].set_visible(False)

    # B: paired scatter
    ax = axes[1]
    ax.scatter(evk, spon, s=22, alpha=0.75,
               color=col_spon, edgecolor="k", linewidth=0.4)
    lim = [0, max(spon.max(), evk.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("CV_L evoked")
    ax.set_ylabel("CV_L spontaneous")
    ax.set_title(f"B: paired (n={len(spon)} V1 units)")
    ax.spines[["top", "right"]].set_visible(False)

    # C: paired differences with bootstrap 95% CI
    ax = axes[2]
    bins = np.linspace(diff.min() - 0.05, diff.max() + 0.05, 31)
    ax.hist(diff, bins=bins, color=(0.6, 0.6, 0.6), edgecolor="k")
    ax.axvline(0, color="k", lw=1)
    rng = np.random.default_rng(0)
    boot = np.array([
        rng.choice(diff, size=diff.size, replace=True).mean()
        for _ in range(10000)
    ])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    ax.axvline(diff.mean(), color="r", lw=2,
               label=f"mean {diff.mean():.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")
    w_stat, p_val = wilcoxon(spon, evk)
    ax.set_xlabel("CV_L spontaneous - evoked")
    ax.set_title(f"C: Wilcoxon W={w_stat:.0f}, p={p_val:.2g}")
    ax.legend(loc="best", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Awake mouse V1 (Allen Visual Coding NPx): matched-window "
        "spontaneous vs evoked CV_L"
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"  wrote {out_pdf} / {out_png}")
    print(
        f"  spontaneous CV_L: median={np.median(spon):.3f} | "
        f"evoked CV_L: median={np.median(evk):.3f} | "
        f"diff median={np.median(diff):.3f} | "
        f"Wilcoxon p={p_val:.2g}"
    )


def copy_to_manuscript(out_stem: str) -> None:
    """Copy generated S15 files into updated/figures for LaTeX."""
    src_dir = Path.cwd()
    dst_dir = Path(__file__).resolve().parents[1] / "updated" / "figures"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        src = src_dir / f"{out_stem}.{ext}"
        if src.exists():
            dst = dst_dir / src.name
            shutil.copy2(src, dst)
            print(f"  copied {src.name} -> {dst}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--session-id", type=int, default=DEFAULT_SESSION_ID,
        help="Allen ecephys session id (default 715093703)"
    )
    p.add_argument(
        "--nwb-path", type=str, default=None,
        help="Path to a previously-downloaded NWB2 file; uses low-memory h5py reads"
    )
    p.add_argument(
        "--direct-download", action="store_true",
        help="Download the NWB directly from Allen S3 with curl resume/retry, "
             "then read it from disk. This bypasses AllenSDK's downloader."
    )
    p.add_argument(
        "--download-only", action="store_true",
        help="With --direct-download, only download/resume the NWB and exit."
    )
    p.add_argument(
        "--cache-dir", type=str,
        default=str(Path.home() / "allen-brain-observatory" / "direct-nwb"),
        help="Directory for direct NWB downloads"
    )
    p.add_argument(
        "--out-stem", type=str, default="FigureS15",
        help="output figure stem (default: FigureS15)"
    )
    p.add_argument(
        "--copy-to-manuscript", action="store_true",
        help="Copy FigureS15.pdf/.png into ../updated/figures after creation"
    )
    args = p.parse_args()

    print("=== awake_V1_CVL_confirmation.py ===")
    if args.nwb_path is not None:
        session = load_session_from_nwb(args.nwb_path)
    elif args.direct_download:
        nwb_path = direct_download_nwb(args.session_id, args.cache_dir)
        if args.download_only:
            print(f"  download complete: {nwb_path}")
            print("=== done ===")
            return
        session = load_session_from_nwb(str(nwb_path))
    else:
        session = load_session_allensdk(args.session_id)
    unit_ids = get_V1_unit_ids(session)
    if len(unit_ids) < 10:
        warnings.warn(
            "Fewer than 10 V1 units in this session.  Consider a different "
            "session_id (e.g., 715093703, 759883607, 762602078).  Proceeding."
        )
    R = compute_per_unit_CVL(session, unit_ids)
    if hasattr(session, "close"):
        session.close()
    np.savez(
        "awake_V1_CVL.npz",
        **R,
        spon_window_s=np.array(SPON_WINDOW_S),
        evk_window_s=np.array(EVK_WINDOW_S),
    )
    print("  wrote awake_V1_CVL.npz")
    make_figure(R, f"{args.out_stem}.pdf", f"{args.out_stem}.png")
    if args.copy_to_manuscript:
        copy_to_manuscript(args.out_stem)
    print("=== done ===")


if __name__ == "__main__":
    main()
