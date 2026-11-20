"""Replace only B-top and E-G in an isolated copy of the author Figure 3."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys

import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf.generic import ContentStream, DecodedStreamObject
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Fig3.recompute_lif_information_audit import load_completed_audit, sha256

SOURCE = ROOT / "updated/figures/GIBBSMANIFOLD_FIGS/Figure3.pdf"
OUTPUT = ROOT / "updated/figures/review_20260907"
BACKUP = ROOT / "outputs/final_text_review_20260907/lif_corrected/backups"
B_TOP = fitz.Rect(5, 180, 242, 325)
RIGHT_X = 648.0
BLUE = "#0072bd"


def leaf_graphics(document: fitz.Document, xref: int) -> bool:
    stream = DecodedStreamObject()
    stream.set_data(document.xref_stream(xref))
    return not any(operator in (b"BT", b"Do")
                   for _, operator in ContentStream(stream, None).operations)


def pdf_figure(fig: plt.Figure) -> fitz.Document:
    buffer = BytesIO()
    fig.savefig(buffer, format="pdf", facecolor="white")
    plt.close(fig)
    return fitz.open(stream=buffer.getvalue(), filetype="pdf")


def base_axis(fig: plt.Figure, twin: bool = False) -> tuple[plt.Axes, plt.Axes | None]:
    ax = fig.add_axes([.255, .27, .53 if twin else .63, .54])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=12, length=3, width=.7)
    ax.set_ylim(1.15, 3.65)
    ax.set_yticks([1.5, 2.5, 3.5])
    ax.set_ylabel(r"$I_q/\widetilde C$", fontsize=15, labelpad=4)
    right = None
    if twin:
        right = ax.twinx()
        right.spines[["top", "left"]].set_visible(False)
        right.spines["right"].set_color(BLUE)
        right.tick_params(axis="y", colors=BLUE, labelsize=12, length=3, width=.7)
        right.set_ylim(2.4, 4.2)
        right.set_yticks([2.5, 3.2, 4.0])
        right.set_ylabel(r"$\mathrm{CV}_X$", color=BLUE, fontsize=15, labelpad=3)
    return ax, right


def make_b(rate: np.ndarray, time: np.ndarray) -> fitz.Document:
    fig = plt.figure(figsize=(B_TOP.width / 72, B_TOP.height / 72))
    ax = fig.add_axes([.23, .25, .65, .60])
    ax.plot(time, rate, "o-", color="black", lw=1.05, ms=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set(xlim=(6, 126), ylim=(0, 10.5), xticks=[12, 60, 120], yticks=[0, 5, 10])
    ax.set_xlabel("Segment end (s)", fontsize=12, labelpad=2)
    ax.set_ylabel("Physical rate (Hz)", fontsize=13, labelpad=3)
    ax.tick_params(labelsize=12, length=3, width=.7)
    return pdf_figure(fig)


def make_right(letter: str, width: float, height: float, time: np.ndarray,
               cv: np.ndarray, score: np.ndarray, ei: np.ndarray, ei_sem: np.ndarray) -> fitz.Document:
    fig = plt.figure(figsize=(width / 72, height / 72))
    fig.text(.012, .98, letter, fontsize=23, va="top")
    ax, right = base_axis(fig, twin=letter != "G")
    if letter == "E":
        fig.text(.53, .965, "20-ms count readout", fontsize=10.5, ha="center", va="top")
        ax.plot(time, score, "o-", color="black", ms=2.5, lw=1.0)
        right.plot(time, cv, "o-", color=BLUE, ms=2.1, lw=1.0)
        ax.set(xlim=(6, 126), xticks=[12, 60, 120])
        ax.set_xlabel("Segment end (s)", fontsize=13, labelpad=3)
    elif letter == "F":
        ax.errorbar(ei, score, xerr=ei_sem, fmt="o", color="black", ms=2.6,
                    elinewidth=.55, capsize=0)
        right.errorbar(ei, cv, xerr=ei_sem, fmt="o", color=BLUE, ms=2.3,
                       elinewidth=.55, capsize=0)
        ax.set(xlim=(1.008, 1.033), xticks=[1.01, 1.03])
        ax.set_xlabel(r"$I_{\mathrm{EX}}/I_{\mathrm{IN}}$ (legacy)", fontsize=12, labelpad=3)
        fig.text(.53, .032, "Horizontal bars: saved descriptive SEM", fontsize=7.7,
                 ha="center", va="bottom")
    else:
        ax.plot(cv, score, "o-", color="black", ms=2.6, lw=1.0)
        ax.set(xlim=(2.43, 4.14), xticks=[2.5, 3.2, 4.0])
        ax.set_xlabel(r"Count $\mathrm{CV}_X$ (20 ms)", fontsize=12.5, labelpad=3)
        ax.annotate("12 s", (cv[0], score[0]), xytext=(6, 3), textcoords="offset points", fontsize=10)
        ax.annotate("120 s", (cv[-1], score[-1]), xytext=(-39, 5), textcoords="offset points", fontsize=10)
        fig.text(.53, .03, "Different input segments; descriptive only", fontsize=7.6,
                 ha="center", va="bottom")
    return pdf_figure(fig)


def main() -> None:
    audit, data = load_completed_audit()
    primary = int(np.flatnonzero(data["windows_ms"] == 20)[0])
    time, rate = data["segment_end_s"], data["firing_rate_hz"]
    cv, score = data["count_cv"][:, primary], data["logscore_per_cost"][:, primary]
    saved = loadmat(ROOT / "Fig3/MI_results_direct_corrected.mat", squeeze_me=True)
    retained = data["checkpoint_indices"] - 1
    ei = np.ravel(saved["EI_grand"])[retained]
    ei_sem = np.ravel(saved["EI_sem"])[retained]
    if not np.all(np.isfinite(np.concatenate([time, rate, cv, score, ei, ei_sem]))):
        raise ValueError("Nonfinite Figure 3 data")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    BACKUP.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source_backup = BACKUP / "Figure3_author_before_composition.pdf"
    if not source_backup.exists():
        shutil.copy2(SOURCE, source_backup)
    for name in ("Figure3.pdf", "Figure3.png", "Figure3_composition_manifest.json"):
        path = OUTPUT / name
        if path.exists():
            shutil.copy2(path, BACKUP / f"{stamp}_{name}")

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12,
                         "pdf.fonttype": 42, "axes.linewidth": .7,
                         "mathtext.fontset": "stix"})
    source_hash = sha256(SOURCE)
    reference = fitz.open(SOURCE)
    original = reference[0]
    source = fitz.open(stream=SOURCE.read_bytes(), filetype="pdf")
    # Clipping hides old text visually but leaves it extractable. Remove only
    # the superseded text in memory before embedding the retained vector art.
    for rect in (fitz.Rect(RIGHT_X, 0, original.rect.width, original.rect.height),
                 fitz.Rect(20, 180, 184, 192), fitz.Rect(5, 192, 184, 325)):
        source[0].add_redact_annot(rect, fill=False, cross_out=False)
    source[0].apply_redactions(images=0, graphics=0, text=0)
    # MuPDF rewrites graphical forms during text redaction. Restore the exact
    # original leaf graphics, without restoring any text or nested text forms.
    graphic_forms = {tuple(round(v, 4) for v in bounds): xref
                     for xref, _, _, bounds in reference[0].get_xobjects()
                     if leaf_graphics(reference, xref)}
    for xref, _, _, bounds in source[0].get_xobjects():
        key = tuple(round(v, 4) for v in bounds)
        if key in graphic_forms and leaf_graphics(source, xref):
            original_xref = graphic_forms[key]
            source.update_object(xref, reference.xref_object(original_xref))
            source.update_stream(xref, reference.xref_stream(original_xref))
    result = fitz.open()
    page = result.new_page(width=original.rect.width, height=original.rect.height)
    preserved = [
        fitz.Rect(0, 0, RIGHT_X, B_TOP.y0),
        fitz.Rect(0, B_TOP.y0, B_TOP.x0, B_TOP.y1),
        fitz.Rect(B_TOP.x1, B_TOP.y0, RIGHT_X, B_TOP.y1),
        fitz.Rect(0, B_TOP.y1, RIGHT_X, original.rect.height),
    ]
    for rect in preserved:
        page.show_pdf_page(rect, source, 0, clip=rect, keep_proportion=False)
    b = make_b(rate, time)
    page.show_pdf_page(B_TOP, b, 0, keep_proportion=False)
    b.close()
    # Preserve the fragments of the existing B label and current legend that
    # share the rectangular replacement area; these are not part of B-top data.
    restored = [fitz.Rect(5, 180, 20, 192), fitz.Rect(184, 304, 242, 325)]
    for rect in restored:
        page.show_pdf_page(rect, source, 0, clip=rect, keep_proportion=False)

    panel_height = original.rect.height / 3
    for index, letter in enumerate("EFG"):
        rect = fitz.Rect(RIGHT_X, index * panel_height, original.rect.width, (index + 1) * panel_height)
        panel = make_right(letter, rect.width, rect.height, time, cv, score, ei, ei_sem)
        page.show_pdf_page(rect, panel, 0, keep_proportion=False)
        panel.close()

    output_pdf = OUTPUT / "Figure3.pdf"
    result.save(output_pdf, garbage=4, deflate=True)
    result.close()
    assembled = fitz.open(output_pdf)
    assembled[0].get_pixmap(dpi=220).save(OUTPUT / "Figure3.png")

    # Verify that the author imagery remains unchanged away from crop edges.
    reference_pixels = original.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    actual_pixels = assembled[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    reference_array = np.frombuffer(reference_pixels.samples, np.uint8).reshape(reference_pixels.height, reference_pixels.width, 3)
    actual_array = np.frombuffer(actual_pixels.samples, np.uint8).reshape(actual_pixels.height, actual_pixels.width, 3)
    preserved_pixel_checks = []
    for rect in preserved:
        inner = fitz.Rect(rect.x0 + 1, rect.y0 + 1, rect.x1 - 1, rect.y1 - 1)
        if inner.is_empty:
            continue
        x0, y0, x1, y1 = [int(value * 2) for value in inner]
        delta = np.abs(reference_array[y0:y1, x0:x1].astype(int) - actual_array[y0:y1, x0:x1].astype(int))
        preserved_pixel_checks.append({"rect": list(inner), "max_channel_difference": int(delta.max()),
                                       "fraction_pixels_different": float(np.mean(np.any(delta > 0, axis=2)))})
    if any(check["max_channel_difference"] > 1 for check in preserved_pixel_checks):
        raise AssertionError(f"Preserved image content changed: {preserved_pixel_checks}")
    if sha256(SOURCE) != source_hash:
        raise AssertionError("Source artwork changed")
    extracted = assembled[0].get_text()
    if any(stale in extracted for stale in ("CVL", "Gaussian ref.", "MI/C")):
        raise AssertionError("Superseded hidden panel text remains")
    manifest = {
        "source": str(SOURCE), "source_sha256": source_hash, "source_backup": str(source_backup),
        "output_pdf": str(output_pdf), "output_png": str(OUTPUT / "Figure3.png"),
        "audit_run_id": audit["run_id"], "primary_window_ms": 20,
        "replaced": ["B-top", "E", "F", "G"],
        "preserved": ["A", "B-bottom currents and E/I trace", "C", "D"],
        "preserved_pixel_checks": preserved_pixel_checks,
        "superseded_hidden_text_removed": True,
        "scope": audit["protocol"]["comparability"],
        "Iq_definition": "Signed nested-repeat-held-out log-posterior information score in bits/window, divided by 0.02 s; not an asymptotic rate or confidence bound",
        "cost_definition": "2.4 + 0.37 * original spike count / six seconds, pooled over templates and repeats",
        "CV_definition": "Population SD / mean of 20-ms counts over all template, relative-bin and repeat observations; includes zeros; not legacy inverse-ISI CV",
        "F_error_bars": "Original EI_sem, horizontal only, descriptive saved summary; no inferential CV or information error bars",
        "segment_end_s": time.tolist(), "physical_rate_Hz": rate.tolist(), "count_CV_20ms": cv.tolist(),
        "Iq_per_cost_20ms": score.tolist(), "legacy_EI_ratio": ei.tolist(), "legacy_EI_SEM": ei_sem.tolist(),
    }
    (OUTPUT / "Figure3_composition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": str(output_pdf), "png": str(OUTPUT / "Figure3.png"),
                      "preserved_pixel_checks": preserved_pixel_checks}, indent=2))


if __name__ == "__main__":
    main()
