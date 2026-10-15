"""Evaluate a training, released HT, or original Diffusion Policy checkpoint."""

import argparse
import json
from pathlib import Path

from .image_runner import RobomimicImageRunner
from .state_runner import RobomimicStateRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--checkpoint-format",
        choices=("native", "release", "diffusion-policy"),
        default="native",
    )
    environment = parser.add_mutually_exclusive_group(required=True)
    environment.add_argument(
        "--env-config",
        help="Environment JSON from configs/evaluation/robomimic; no dataset needed.",
    )
    environment.add_argument(
        "--dataset",
        help="Optional HDF5 for evaluating demonstration initial states.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--weights", choices=("auto", "model", "ema"), default="auto")
    parser.add_argument(
        "--source-robosuite-version",
        help="Required for unversioned env_args; must match the installed simulator.",
    )
    for name in (
        "n-train",
        "n-test",
        "train-start-idx",
        "test-start-seed",
        "max-steps",
        "n-envs",
        "num-workers",
        "n-train-vis",
        "n-test-vis",
        "fps",
        "crf",
    ):
        parser.add_argument("--" + name, type=int)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Policy sampling seed; independent of test reset seeds.",
    )
    parser.add_argument("--train-reset", choices=("state", "model"))
    parser.add_argument(
        "--save-traces", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--worker-timeout", type=float)
    parser.add_argument(
        "--repair-observations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default: follow the environment config or HDF5 geometry-repair marker.",
    )
    args = parser.parse_args(argv)
    if args.env_config is not None:
        if args.n_train not in (None, 0):
            parser.error("--n-train > 0 requires --dataset instead of --env-config.")
        # Standalone evaluation reports test rollouts; training-demo rollouts are
        # available through --dataset with the checkpoint's original settings.
        args.n_train = 0
        args.n_train_vis = 0

    overrides = {
        name: value
        for name, value in vars(args).items()
        if name
        not in {
            "checkpoint",
            "checkpoint_format",
            "dataset",
            "output_dir",
            "device",
            "weights",
        }
        and value is not None
    }

    def report(record):
        print(
            f"{record['split']}/{record['id']}: success={record['success']} max_reward={record['max_reward']:.3f} steps={record['steps']}",
            flush=True,
        )

    if args.checkpoint_format == "release":
        from ht_regression.adapters.checkpoint.robomimic_release import (
            load_released_checkpoint,
        )

        loaded = load_released_checkpoint(
            args.checkpoint, device=args.device, weights=args.weights
        )
        pipeline = loaded.pipeline
        settings = dict(loaded.config["evaluation"])
        settings.update(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            progress=report,
            **overrides,
        )
        runner_class = (
            RobomimicImageRunner
            if pipeline.policy.obs_encoder is not None
            else RobomimicStateRunner
        )
        runner = runner_class(**settings)
    elif args.checkpoint_format == "native":
        from ht_regression.adapters.checkpoint.native import load_native_checkpoint

        loaded = load_native_checkpoint(
            args.checkpoint, device=args.device, weights=args.weights
        )
        pipeline = loaded.pipeline
        policy = pipeline.policy
        encoder = getattr(policy, "obs_encoder", None)
        runner_class = (
            RobomimicImageRunner if encoder is not None else RobomimicStateRunner
        )
        settings = dict(loaded.config["evaluation"])
        settings.pop("_target_", None)
        settings.update(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            obs_keys=loaded.metadata["obs_keys"],
            action_space=loaded.config["dataset"]["action_space"],
            n_obs_steps=policy.n_obs_steps,
            n_action_steps=policy.n_action_steps,
            seed=loaded.training_config["evaluation_seed"],
            progress=report,
        )
        if encoder is not None:
            settings["camera_shapes"] = encoder.camera_shapes
        settings.update(overrides)
        runner = runner_class(**settings)
    else:
        from ht_regression.adapters.checkpoint.diffusion_policy import (
            load_dp_checkpoint,
        )

        loaded = load_dp_checkpoint(
            args.checkpoint, device=args.device, weights=args.weights
        )
        runner_class = (
            RobomimicImageRunner
            if getattr(loaded.policy, "obs_encoder", None) is not None
            else RobomimicStateRunner
        )
        overrides.setdefault("seed", 0)
        runner = runner_class.from_dp_checkpoint(
            loaded,
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            progress=report,
            **overrides,
        )
        pipeline = loaded.make_pipeline()
    metrics = runner.run(pipeline)
    output = Path(args.output_dir).expanduser().resolve()
    (output / "checkpoint.json").write_text(
        json.dumps(
            {
                "path": str(Path(args.checkpoint).expanduser().resolve()),
                "state_key": loaded.state_key,
                "format": args.checkpoint_format,
                "config": loaded.config,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {key: value for key, value in metrics.items() if "sim_" not in key},
            indent=2,
        )
    )
    return metrics


if __name__ == "__main__":
    main()
