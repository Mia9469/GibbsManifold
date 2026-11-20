#!/usr/bin/env python3
"""Restore Figure 4 panel J with the corrected population decoder analysis."""

from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(__file__).with_name("Figure4_corrected_with_withheld_panel.pdf")
PANEL = Path(__file__).with_name("network_information_corrected") / "Figure4J_network_information_corrected.pdf"
OUTPUTS = [
    ROOT / "updated" / "figures" / "GIBBSMANIFOLD_FIGS" / "Figure4_submission.pdf",
    ROOT / "updated" / "figures" / "Figure4_submission.pdf",
]


def compose(output: Path) -> None:
    source = fitz.open(ARCHIVE)
    panel = fitz.open(PANEL)
    source_page = source[0]
    width, height = source_page.rect.width, source_page.rect.height

    target = fitz.open()
    page = target.new_page(width=width, height=height)

    upper = fitz.Rect(0.0, 0.0, width, 261.0)
    page.show_pdf_page(upper, source, 0, clip=upper, keep_proportion=False)

    for crop in (
        fitz.Rect(0.0, 261.0, 157.0, height),
        fitz.Rect(309.0, 261.0, 410.2, height),
        fitz.Rect(410.5, 261.0, width, height),
    ):
        page.show_pdf_page(crop, source, 0, clip=crop, keep_proportion=False)

    page.insert_text(
        (158.5, 280.0),
        "J",
        fontname="Times-Roman",
        fontsize=16.0,
        color=(0, 0, 0),
        overlay=True,
    )
    destination = fitz.Rect(164.0, 270.0, 309.0, 369.0)
    page.show_pdf_page(destination, panel, 0, keep_proportion=True, overlay=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + "_tmp.pdf")
    target.save(temporary, garbage=4, deflate=True)
    temporary.replace(output)

    target.close()
    panel.close()
    source.close()


def main() -> None:
    if not ARCHIVE.exists():
        raise FileNotFoundError(ARCHIVE)
    if not PANEL.exists():
        raise FileNotFoundError(PANEL)
    for output in OUTPUTS:
        compose(output)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
