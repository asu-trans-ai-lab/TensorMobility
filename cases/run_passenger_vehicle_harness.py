from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from tensormobility.profiles.passenger_vehicle_harness import run_passenger_vehicle_harness


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STB-FTT passenger/vehicle multimodal harness")
    parser.add_argument("--output", default="outputs/passenger_vehicle_harness")
    args = parser.parse_args()
    summary = run_passenger_vehicle_harness(args.output)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
