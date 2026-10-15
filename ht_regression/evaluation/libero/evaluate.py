"""Evaluate PI0.5 or Cosmos checkpoints with their native LIBERO protocols."""

import argparse
import json
from pathlib import Path

from ...adapters.checkpoint.cosmos import load_cosmos_checkpoint
from ...adapters.checkpoint.pi05 import load_pi05_checkpoint
from .common import SUITES
from .cosmos.runner import CosmosLiberoRunner
from .cosmos.server import create_native_service
from .pi05.runner import PI05LiberoRunner


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--checkpoint", required=True)
    shared.add_argument("--output-dir", required=True)
    shared.add_argument("--task-id", type=int, action="append")
    shared.add_argument("--seed", type=int, default=0)
    models = parser.add_subparsers(dest="policy", required=True)

    pi05 = models.add_parser("pi05", parents=[shared], help="Native LeRobot collector")
    pi05.add_argument("--suite", choices=SUITES, action="append")
    pi05.add_argument("--device", default="cuda")
    pi05.add_argument("--n-episodes", type=int, default=100)
    pi05.add_argument("--n-envs", type=int, default=10)
    pi05.add_argument("--n-action-steps", type=int, default=10)
    pi05.add_argument("--num-inference-steps", type=int, default=10)
    pi05.set_defaults(run=_evaluate_pi05)

    cosmos = models.add_parser(
        "cosmos", parents=[shared], help="Native Cosmos LIBERO-10 client"
    )
    cosmos.add_argument("--method", choices=("flow", "ht", "mse"), required=True)
    cosmos.add_argument("--action-stats-path", required=True)
    cosmos.add_argument("--vae-path", required=True)
    cosmos.add_argument("--native-client", required=True)
    cosmos.add_argument("--simulator-python", required=True)
    cosmos.add_argument("--n-episodes", type=int, default=10)
    cosmos.add_argument("--n-envs", type=int, default=8)
    cosmos.add_argument(
        "--deterministic",
        action="store_true",
        help="Disable cuDNN autotuning and require deterministic PyTorch algorithms.",
    )
    cosmos.set_defaults(run=_evaluate_cosmos)
    return parser


def _evaluate_pi05(args):
    bundle = load_pi05_checkpoint(
        args.checkpoint,
        device=args.device,
        n_action_steps=args.n_action_steps,
        num_inference_steps=args.num_inference_steps,
    )
    runner = PI05LiberoRunner(
        bundle.native_policy,
        bundle.preprocessor,
        bundle.postprocessor,
        args.output_dir,
        suites=args.suite or SUITES,
        task_ids=args.task_id,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        seed=args.seed,
        provenance=bundle.provenance,
    )
    return runner.run(bundle.pipeline)


def _evaluate_cosmos(args):
    output = Path(args.output_dir).expanduser().resolve()
    runner = CosmosLiberoRunner(
        output / "rollout",
        native_client=args.native_client,
        simulator_python=args.simulator_python,
        task_ids=args.task_id,
        n_episodes=args.n_episodes,
        n_envs=args.n_envs,
        seed=args.seed,
    )
    output.mkdir(parents=True, exist_ok=False)
    bundle = load_cosmos_checkpoint(
        args.checkpoint,
        method=args.method,
        service_factory=create_native_service,
        action_stats_path=args.action_stats_path,
        vae_path=args.vae_path,
        output_dir=output / "setup",
        seed=args.seed,
        deterministic=args.deterministic,
    )
    return runner.run(bundle)


def main(argv=None):
    args = build_parser().parse_args(argv)
    metrics = args.run(args)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
