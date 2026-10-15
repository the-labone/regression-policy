"""Evaluate GR00T checkpoints using the native RoboCasa-GR1 benchmark collector."""

import argparse
import json

from .runner import RoboCasaGR1Runner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-root", required=True)
    parser.add_argument("--model-python", required=True)
    parser.add_argument("--simulator-python", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Default: 24 tasks saved in checkpoint experiment_cfg/config.yaml",
    )
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--n-action-steps", type=int, default=8)
    parser.add_argument("--max-episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=1800)
    parser.add_argument("--task-timeout", type=float, default=21600)
    result = RoboCasaGR1Runner(**vars(parser.parse_args(argv))).run()
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()
