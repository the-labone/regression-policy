"""Resolve saved GR1/Simpler recipes without replacing modality conventions."""

import json
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from enum import Enum

RECIPES = {
    "gr1": (60000, 512, 928.0, -1.088, 0.2, 1.0),
    "bridge": (20000, 1024, 224.0, -0.4535, 0.8, 1.0),
    "fractal": (20000, 1024, 224.0, -0.5093, 0.5, 1.5),
}


def primitive(value):
    if is_dataclass(value):
        return primitive(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: primitive(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [primitive(item) for item in value]
    return value


def restore_modalities(config):
    from gr00t.data.types import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
    )

    before = primitive(config.data.modality_configs)
    restored = {}
    for tag, modalities in before.items():
        restored[tag] = {}
        for name, modality in modalities.items():
            values = deepcopy(modality)
            if values.get("action_configs") is not None:
                values["action_configs"] = [
                    ActionConfig(
                        rep=ActionRepresentation(group["rep"]),
                        type=ActionType(group["type"]),
                        format=ActionFormat(group["format"]),
                        state_key=group.get("state_key"),
                    )
                    for group in values["action_configs"]
                ]
            restored[tag][name] = ModalityConfig(**values)
    if primitive(restored) != before:
        raise ValueError("Typed modality reconstruction changed the saved recipe.")
    config.data.modality_configs = restored
    return config


def resolve(args):
    cfg = json.loads(args.config.read_text())
    steps, batch, nu, bias, dropout, multiplier = RECIPES[args.task]
    model, training = cfg["model"], cfg["training"]
    expected = dict(
        loss_type="hetero_t",
        ht_mvt=True,
        ht_df=nu,
        ht_sbias=bias,
        state_dropout_prob=dropout,
    )
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"{args.task}: model.{key} must be {value!r}.")
    if (training["max_steps"], training["global_batch_size"], training["num_gpus"]) != (
        steps,
        batch,
        8,
    ):
        raise ValueError(
            "Training budget/batch/world size differs from the saved recipe."
        )
    base = json.loads((args.base_model / "config.json").read_text())
    for key in ("action_horizon", "max_action_dim", "max_state_dim", "select_layer"):
        if base.get(key) != model[key]:
            raise ValueError(f"Base checkpoint architecture mismatch: {key}.")
    if (
        base.get("diffusion_model_cfg", {}).get("num_layers")
        != model["diffusion_model_cfg"]["num_layers"]
    ):
        raise ValueError("Base checkpoint has the wrong action-head depth.")
    if base.get("loss_type", "flow") != "flow":
        raise ValueError("Fresh training starts from the pretrained Flow base.")
    if not any(args.base_model.glob("*.safetensors")):
        raise FileNotFoundError("Base checkpoint weights are missing.")
    for dataset in cfg["data"]["datasets"]:
        paths = [args.data_root / name for name in dataset["dataset_paths"]]
        for path in paths:
            for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "modality.json"):
                if not (path / "meta" / name).is_file():
                    raise FileNotFoundError(path / "meta" / name)
        dataset["dataset_paths"] = [str(path) for path in paths]
    training.update(
        start_from_checkpoint=str(args.base_model),
        output_dir=str(args.output_dir),
        resume_from_checkpoint=args.resume,
    )
    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
    if not args.resume and list(args.output_dir.glob("checkpoint-*")):
        raise ValueError(
            "Output already contains checkpoints; use --resume or a fresh directory."
        )
    if args.resume and not list(
        args.output_dir.glob("checkpoint-*/trainer_state.json")
    ):
        raise FileNotFoundError(
            "No native training checkpoint to resume in output-dir."
        )
    return cfg, multiplier
