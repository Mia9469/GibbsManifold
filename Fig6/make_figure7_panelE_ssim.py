"""Build corrected panel E from per-run observations, without modifying artwork."""
import argparse
from pathlib import Path

from make_figure7_submission import ROOT, DEFAULT_OUTPUT, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.input_root, args.output_dir, only="Figure7E_training_SSIM")


if __name__ == "__main__":
    main()
