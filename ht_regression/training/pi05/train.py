"""Train PI0.5 on LIBERO with the released 30k-step HT recipe."""

import json
from pathlib import Path

from ..native import check_resources, parser


def resolve(args):
    config = json.loads(args.config.read_text())
    policy = config["policy"]
    if (
        policy["loss_type"],
        policy["ht_df"],
        policy["ht_sbias"],
        policy.get("ht_mvt", False),
    ) != ("hetero_t", 2.0, -0.396, False):
        raise ValueError("Expected native df=2 (effective nu=700), bias=-0.396.")
    if (
        config["steps"],
        config["batch_size"],
        policy["chunk_size"],
        config["parallelism"]["dp_replicate"],
    ) != (30000, 4, 50, 8):
        raise ValueError("Expected the saved 30k, global batch 32, horizon 50 recipe.")
    for name in ("info.json", "stats.json"):
        if not (args.data_root / "meta" / name).is_file():
            raise FileNotFoundError(args.data_root / "meta" / name)
    if not (args.base_model / "model.safetensors").is_file():
        raise FileNotFoundError(args.base_model / "model.safetensors")
    base = json.loads((args.base_model / "config.json").read_text())
    if base.get("type") != "pi05" or base.get("loss_type", "flow") != "flow":
        raise ValueError(
            "Start from the original PI0.5 Flow base, not an HT checkpoint."
        )
    config["dataset"]["root"] = str(args.data_root)
    policy["pretrained_path"] = str(args.base_model)
    config["output_dir"] = str(args.output_dir)
    if args.seed is not None:
        config["seed"] = args.seed
    if args.resume_from is None and args.output_dir.exists():
        raise ValueError("Native LeRobot requires a fresh output directory.")
    if args.resume_from is not None:
        checkpoint = args.resume_from.expanduser().resolve(strict=True)
        if not (checkpoint / "train_config.json").is_file():
            raise FileNotFoundError(checkpoint / "train_config.json")
        if not (checkpoint.parent / "training_state").is_dir():
            raise FileNotFoundError(
                "Resume requires training_state beside pretrained_model."
            )
        saved = json.loads((checkpoint / "train_config.json").read_text())
        # Native resume reads this saved file, not the fresh-run JSON below.
        # Reject a different objective/budget rather than silently resuming it.
        for key in (
            "steps",
            "batch_size",
            "optimizer",
            "scheduler",
            "parallelism",
            "accelerator",
            "ema",
        ):
            if saved.get(key) != config[key]:
                raise ValueError(f"Resume checkpoint differs in {key}.")
        for key in (
            "loss_type",
            "ht_df",
            "ht_sbias",
            "chunk_size",
            "n_action_steps",
            "normalization_mapping",
            "freeze_vision_encoder",
            "train_expert_only",
        ):
            if saved["policy"].get(key) != policy[key]:
                raise ValueError(f"Resume checkpoint differs in policy.{key}.")
        if saved["policy"].get("ht_mvt", False):
            raise ValueError("Resume checkpoint uses a different nu encoding.")
        if args.seed is not None and saved["seed"] != args.seed:
            raise ValueError("Resume must retain the saved training seed.")
        saved["dataset"]["root"] = str(args.data_root)
        saved["output_dir"] = str(args.output_dir)
        saved["resume"] = True
        config = saved
    return config


def main(argv=None):
    cli = parser(__doc__, "configs/experiment/pi05/libero_ht.json")
    cli.add_argument(
        "--resume-from",
        type=Path,
        help="Native checkpoint's pretrained_model directory.",
    )
    args = cli.parse_args(argv)
    check_resources(args)
    config = resolve(args)
    if args.check_only:
        print(json.dumps(config, indent=2))
        return
    from .trainer import train

    train(config, resume_from=args.resume_from)


if __name__ == "__main__":
    main()
