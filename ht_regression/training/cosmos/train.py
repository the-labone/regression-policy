"""Train the released Cosmos LIBERO-10 video/direct-action HT recipe."""

import json
import os
import runpy
import sys
from pathlib import Path

from ..native import (
    check_resources,
    configure_environment,
    parser,
    require_world_size,
    write_manifest,
)


def main(argv=None):
    cli = parser(__doc__, "configs/experiment/cosmos/libero10_ht.toml")
    cli.add_argument("--vae-path", type=Path, required=True)
    cli.add_argument("--native-root", type=Path, required=True)
    cli.add_argument(
        "--resume",
        action="store_true",
        help="Resume native state in the same output directory.",
    )
    args = cli.parse_args(argv)
    check_resources(args)
    args.vae_path = args.vae_path.expanduser().resolve(strict=True)
    args.native_root = args.native_root.expanduser().resolve(strict=True)
    import tomllib

    config = tomllib.loads(args.config.read_text())
    if (
        config["trainer"]["max_iter"],
        config["trainer"]["grad_accum_iter"],
        config["model"]["parallelism"]["data_parallel_shard_degree"],
    ) != (2000, 32, 8):
        raise ValueError("Expected native 2000 steps, 32 microbatches, 8 FSDP ranks.")
    if not (args.data_root / "meta" / "info.json").is_file():
        raise FileNotFoundError(args.data_root / "meta" / "info.json")
    if not (args.native_root / "cosmos_framework").is_dir():
        raise FileNotFoundError(args.native_root / "cosmos_framework")
    checkpoint_dir = (
        args.output_dir
        / "runs/cosmos3_action_libero/video_direct_action_sft/libero10_ht/checkpoints"
    )
    if not args.resume and checkpoint_dir.exists():
        raise ValueError(
            "Output contains checkpoints; use --resume or a fresh output directory."
        )
    if args.resume and not (checkpoint_dir / "latest_checkpoint.txt").is_file():
        raise FileNotFoundError("No native checkpoint directory to resume.")
    seed = 42 if args.seed is None else args.seed
    overrides = [
        f"trainer.seed={seed}",
        f"dataloader_train.dataloader.datasets.libero.dataset.episode_shuffle_seed={seed}",
        "checkpoint.verbose=false",
    ]
    if args.check_only:
        print(
            json.dumps(
                dict(config=config, overrides=overrides, resume=args.resume), indent=2
            )
        )
        return
    require_world_size(8)
    configure_environment(
        {
            "HT_ACTION": 1,
            "HT_MVT": 1,
            "HT_DF": 640,
            "HT_SBIAS": 0,
            "HT_BETA": 0,
            "HT_HG_STEPS": 0,
            "HT_NU_START": 0,
            "HT_LEARN_NU": 0,
            "HT_NU_RMS": 0,
            "HT_MSE": 0,
            "DIRECT_LOSS": "ht",
        }
    )
    os.environ.update(
        LIBERO_ROOT=str(args.data_root),
        BASE_CHECKPOINT_PATH=str(args.base_model),
        WAN_VAE_PATH=str(args.vae_path),
        IMAGINAIRE_OUTPUT_ROOT=str(args.output_dir / "runs"),
        DIRECT_AUDIT_DIR=str(args.output_dir / "audit"),
    )
    from .trainer import install

    install()
    write_manifest(
        args.output_dir,
        dict(recipe=config, overrides=overrides),
        sources=[args.config, Path(__file__).with_name("trainer.py")],
    )
    previous_argv, previous_cwd = sys.argv, Path.cwd()
    try:
        os.chdir(args.native_root)
        sys.argv = ["cosmos_train", f"--sft-toml={args.config}", "--", *overrides]
        runpy.run_module("cosmos_framework.scripts.train", run_name="__main__")
    finally:
        sys.argv = previous_argv
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
