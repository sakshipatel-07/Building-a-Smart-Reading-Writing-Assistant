"""
Top-level entry point.

Usage:
    python3 pipeline.py                       # run on data/test_samples/*.txt
    python3 pipeline.py path/to/folder         # run on a custom folder
    python3 pipeline.py --text "some raw text" # run on a single string
"""
import sys
import json
from pathlib import Path

from stage5_pipeline.pipeline import process, run_on_folder, print_summary_table

ROOT = Path(__file__).resolve().parent


def main():
    args = sys.argv[1:]

    if args and args[0] == "--text":
        raw = " ".join(args[1:])
        result = process(raw)
        print(json.dumps(result, indent=2, default=str))
        return

    folder = args[0] if args else str(ROOT / "data" / "test_samples")
    output_dir = str(ROOT / "outputs")
    rows = run_on_folder(folder, output_dir)
    print_summary_table(rows)
    print(f"\nFull per-sample JSON output written to: {output_dir}/")


if __name__ == "__main__":
    main()
