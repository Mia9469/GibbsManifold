"""Remove the unavailable checkpoint-information panel from Figure 4.

The unaffected vector panels are copied from the corrected archived layout.
Panels K and L are repositioned and relabelled J and K, respectively.
"""
from pathlib import Path
import shutil

import fitz


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "updated" / "figures" / "GIBBSMANIFOLD_FIGS" / "Figure4_submission.pdf"
ARCHIVE = Path(__file__).with_name("Figure4_corrected_with_withheld_panel.pdf")


def main() -> None:
    if not ARCHIVE.exists():
        shutil.copy2(OUT, ARCHIVE)

    source = fitz.open(ARCHIVE)
    src_page = source[0]
    width, height = src_page.rect.width, src_page.rect.height
    target = fitz.open()
    page = target.new_page(width=width, height=height)

    # Preserve the complete unaffected upper block (A--H).
    upper = fitz.Rect(0, 0, width, 261.0)
    page.show_pdf_page(upper, source, 0, clip=upper, keep_proportion=False)

    # Reassemble the unaffected lower panels, omitting the old J panel.
    crops = {
        "I": fitz.Rect(0.0, 261.0, 157.0, height),
        "J": fitz.Rect(309.0, 261.0, 410.2, height),  # old K
        "K": fitz.Rect(410.5, 261.0, width, height),  # old L
    }
    dests = {
        "I": fitz.Rect(20.0, 261.0, 177.0, height),
        "J": fitz.Rect(225.0, 261.0, 326.2, height),
        "K": fitz.Rect(371.0, 261.0, 523.5, height),
    }
    for label in ("I", "J", "K"):
        page.show_pdf_page(dests[label], source, 0, clip=crops[label], keep_proportion=False)

    # Replace the copied old K/L panel letters with the new J/K labels.
    for x, label in ((229.5, "J"), (374.2, "K")):
        page.draw_rect(fitz.Rect(x - 2.5, 261.0, x + 17.0, 285.5),
                       color=None, fill=(1, 1, 1), overlay=True)
        page.insert_text((x, 280.0), label, fontname="Times-Roman",
                         fontsize=16.0, color=(0, 0, 0), overlay=True)

    vector_tmp = OUT.with_name(OUT.stem + "_vector_tmp.pdf")
    target.save(vector_tmp, garbage=4, deflate=True)
    source.close()
    target.close()

    # PDF clipping can leave removed panel text inside an invisible Form XObject.
    # Flatten the assembled page at submission resolution so text extraction and
    # automated preflight see only the panels that are actually displayed.
    vector = fitz.open(vector_tmp)
    vector_page = vector[0]
    pix = vector_page.get_pixmap(
        matrix=fitz.Matrix(600.0 / 72.0, 600.0 / 72.0),
        colorspace=fitz.csRGB,
        alpha=False,
    )
    clean = fitz.open()
    clean_page = clean.new_page(width=width, height=height)
    clean_page.insert_image(clean_page.rect, pixmap=pix, keep_proportion=False)
    final_tmp = OUT.with_name(OUT.stem + "_tmp.pdf")
    clean.save(final_tmp, garbage=4, deflate=True)
    clean.close()
    vector.close()
    vector_tmp.unlink()
    final_tmp.replace(OUT)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
