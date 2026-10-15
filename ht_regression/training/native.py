"""Small launch/configuration helpers; each model owns its distributed trainer."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def parser(description, default_config):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--config", type=Path, default=Path(default_config))
    result.add_argument("--data-root", type=Path, required=True)
    result.add_argument("--base-model", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int)
    result.add_argument("--check-only", action="store_true")
    return result


def check_resources(args):
    for name in ("config", "data_root", "base_model"):
        path = getattr(args, name).expanduser().resolve(strict=True)
        setattr(args, name, path)
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.seed is not None and not 0 <= args.seed < 2**32:
        raise ValueError("seed must be in [0, 2**32).")


def require_world_size(expected):
    actual = int(os.environ.get("WORLD_SIZE", "1"))
    if actual != expected:
        raise ValueError(f"This recipe requires {expected} ranks; WORLD_SIZE={actual}.")


def configure_environment(values):
    """Reject conflicting experimental environment switches inherited by a job."""
    for name, value in values.items():
        value = str(value)
        if name in os.environ and os.environ[name] != value:
            raise ValueError(f"Conflicting {name}: expected {value!r}.")
        os.environ[name] = value
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def write_manifest(output, config, *, sources=()):
    if int(os.environ.get("RANK", "0")) != 0:
        return
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "release_training.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())

        def semantic(value):
            if isinstance(value, dict):
                return {
                    key: semantic(item)
                    for key, item in value.items()
                    if key not in ("resume", "resume_from_checkpoint")
                }
            if isinstance(value, list):
                return [semantic(item) for item in value]
            return value

        if semantic(previous["config"]) != semantic(config):
            raise ValueError(
                "Existing output directory belongs to a different training recipe."
            )
    record = dict(
        config=config,
        sources={
            str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in sources
        },
    )
    (output / "release_training.json").write_text(json.dumps(record, indent=2) + "\n")
