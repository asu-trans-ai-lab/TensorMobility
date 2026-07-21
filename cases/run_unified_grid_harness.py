from __future__ import annotations

import argparse
from pathlib import Path

from tensormobility.harness.unified_harness import run_unified_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run STB-FTT unified analytical/grid/queue/GMNS harness")
    parser.add_argument("--output", default="outputs/unified_harness")
    args = parser.parse_args()
    summary = run_unified_harness(Path(args.output))
    print(summary)


if __name__ == "__main__":
    main()
