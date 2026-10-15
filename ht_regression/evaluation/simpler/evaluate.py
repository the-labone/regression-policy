"""Evaluate a local GR00T Flow/HT/MSE checkpoint on SimplerEnv."""

import argparse
import json

from .runner import SUITES, GR00TSimplerRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--native-root", required=True, help="GR00T checkout with SimplerEnv installed"
    )
    parser.add_argument(
        "--model-python", required=True, help="GR00T model environment Python"
    )
    parser.add_argument(
        "--simulator-python", required=True, help="Native Simpler environment Python"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Task names without the embodiment prefix; default: entire suite",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        help="Requested per task; default: Bridge 50, Fractal 100",
    )
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument(
        "--n-action-steps", type=int, help="Default: Bridge 4, Fractal 1"
    )
    parser.add_argument("--max-episode-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, help="Optional environment and model seed")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve native VLM processor assets from local cache",
    )
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--task-timeout", type=float, default=14400)
    summary = GR00TSimplerRunner(**vars(parser.parse_args())).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
