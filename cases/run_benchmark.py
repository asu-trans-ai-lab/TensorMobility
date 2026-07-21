from __future__ import annotations

import argparse

from tensormobility.harness.benchmark import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the STB-FTT multi-algorithm benchmark")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = run_suite(args.output, seed=args.seed)
    print(summary.to_string(index=False))
    print(f"\nOutputs written to {args.output}")


if __name__ == "__main__":
    main()
