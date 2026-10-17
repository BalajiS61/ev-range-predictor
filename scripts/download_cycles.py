"""Download the standard drive-cycle files into data/cycles/."""

from __future__ import annotations

import argparse
from pathlib import Path

from evrange.cycle_library import SOURCES, WLTC_INSTRUCTIONS, download_cycles

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "cycles"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="re-download existing files")
    args = parser.parse_args()

    print(f"Saving cycles to {DATA_DIR}\n")
    status = download_cycles(DATA_DIR, keys=list(SOURCES), overwrite=args.overwrite)
    for key, state in status.items():
        print(f"  {key:<8} {SOURCES[key].filename:<14} {state}")

    failed = [k for k, s in status.items() if s.startswith("failed")]
    if failed:
        print("\nSome downloads failed. Open these links in your browser, press Ctrl+S,")
        print(f"and save each file into {DATA_DIR} with the name shown:")
        for key in failed:
            print(f"  {SOURCES[key].url}  ->  {SOURCES[key].filename}")
    if status.get("wltc3b") == "manual":
        print("\n" + WLTC_INSTRUCTIONS)


if __name__ == "__main__":
    main()
