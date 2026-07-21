from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from tensormobility.harness.harness import run_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STB-FTT special-case and activity-DTA harness")
    parser.add_argument("--output", default="outputs/stb_harness")
    args = parser.parse_args()
    summary = run_harness(args.output)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
