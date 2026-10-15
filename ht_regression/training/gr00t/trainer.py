"""Native GR00T optimization, with the historical Fractal sigma LR group."""

import inspect
from pathlib import Path

from ..native import configure_environment, require_world_size, write_manifest
from .config import primitive, restore_modalities


def split_sigma_groups(optimizer, named_parameters, multiplier):
    """Split before scheduler/ZeRO construction; retain native decay membership."""
    if optimizer.state:
        raise ValueError("Sigma LR groups must be created before optimizer updates.")
    names = {id(p): name for name, p in named_parameters if p.requires_grad}
    sigma = {pid for pid, name in names.items() if ".sigma_decoder." in name}
    if not sigma:
        raise ValueError("No sigma decoder in optimizer.")
    original = [p for group in optimizer.param_groups for p in group["params"]]
    if len(original) != len({id(p) for p in original}) or {
        id(p) for p in original
    } != set(names):
        raise ValueError("Optimizer membership differs from trainable parameters.")
    groups = []
    for group in optimizer.param_groups:
        for is_sigma in (False, True):
            params = [p for p in group["params"] if (id(p) in sigma) == is_sigma]
            if params:
                factor = multiplier if is_sigma else 1.0
                new = dict(group, params=params, lr=group["lr"] * factor)
                if "initial_lr" in group:
                    new["initial_lr"] *= factor
                groups.append(new)
    optimizer.param_groups[:] = groups


def train(recipe, *, sigma_lr_multiplier):
    require_world_size(recipe["training"]["num_gpus"])
    configure_environment(
        {
            "GROOT_LEARN_NU": 0,
            "GROOT_MUON": 0,
            "GROOT_FP32_TRAINABLE": 0,
            "GROOT_HUBER": 0,
            "GROOT_HL1": 0,
            "GROOT_NU_RMS": 0,
            "GROOT_FLOW_KMEAN": 1,
            "GROOT_FLOW_NOISE_SCALE": "1.0",
        }
    )
    from gr00t.configs.base_config import Config
    from gr00t.experiment.experiment import run
    from gr00t.experiment.trainer import Gr00tTrainer
    from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as processing
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    if "gr00t_loss" not in inspect.getsource(Gr00tN1d7ActionHead.forward):
        raise RuntimeError("Install the release gr00t-training.patch before training.")
    config = restore_modalities(Config().load_dict(recipe))
    config.validate()
    if primitive(config.model) != recipe["model"]:
        raise ValueError("Native model defaults changed the training recipe.")
    original = Gr00tTrainer.create_optimizer
    original_processor = processing.build_processor

    def build_processor(model_name, loading_kwargs):
        # Some Transformers tokenizer versions make a Hub metadata request even
        # with local_files_only=True. A resolved cache path keeps offline runs
        # local without changing tokenizer assets or preprocessing.
        from huggingface_hub import constants, snapshot_download

        local_only = (
            config.training.transformers_local_files_only
            or loading_kwargs.get("local_files_only", False)
            or constants.HF_HUB_OFFLINE
        )
        if local_only and not Path(model_name).is_dir():
            model_name = snapshot_download(
                model_name,
                revision=loading_kwargs.get("revision"),
                cache_dir=loading_kwargs.get("cache_dir"),
                local_files_only=True,
            )
        return original_processor(model_name, loading_kwargs)

    def create_optimizer(self):
        optimizer = original(self)
        if sigma_lr_multiplier != 1 and not getattr(
            self, "_release_sigma_groups", False
        ):
            if self.lr_scheduler is not None:
                raise RuntimeError("Sigma groups must be split before the scheduler.")
            split_sigma_groups(
                optimizer, self.model.named_parameters(), sigma_lr_multiplier
            )
            self._release_sigma_groups = True
        return optimizer

    write_manifest(
        recipe["training"]["output_dir"],
        dict(recipe=recipe, sigma_lr_multiplier=sigma_lr_multiplier),
        sources=[Path(inspect.getfile(Gr00tN1d7ActionHead)), Path(__file__)],
    )
    Gr00tTrainer.create_optimizer = create_optimizer
    processing.build_processor = build_processor
    try:
        run(config)
    finally:
        Gr00tTrainer.create_optimizer = original
        processing.build_processor = original_processor
