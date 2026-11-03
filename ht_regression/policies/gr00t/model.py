# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Official GR00T backbone/processor with a locally owned regression head."""

from contextvars import ContextVar

from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, get_backbone_cls
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import (
    Gr00tN1d7DataCollator,
    Gr00tN1d7Processor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor, PreTrainedModel

from .config import GR00TConfig
from .native import GR00TNativeActionHead

# The upstream policy constructs AutoModel without forwarding loading kwargs.
# Keep its preprocessing intact while propagating the adapter's offline option.
loading_options = ContextVar("gr00t_loading_options", default=None)


class GR00TNativeModel(Gr00tN1d7):
    config_class = GR00TConfig

    def __init__(self, config, transformers_loading_kwargs=None):
        PreTrainedModel.__init__(self, config)
        options = dict(loading_options.get() or {})
        options.update(transformers_loading_kwargs or {})
        self.backbone = get_backbone_cls(config)(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=options,
        )
        self.action_head = GR00TNativeActionHead(config)
        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=options,
        )

    @classmethod
    def from_pretrained(cls, path, *args, initialize_scale=False, **kwargs):
        """Load all weights strictly; optionally add sigma to a Flow base.

        The exception is explicit and checks the *saved* objective. A damaged
        HT checkpoint can never be mistaken for a fresh Flow initialization.
        """
        output_info = kwargs.pop("output_loading_info", False)
        options = dict(loading_options.get() or {})
        options.update(kwargs.pop("transformers_loading_kwargs", {}) or {})
        for key in ("local_files_only", "cache_dir", "revision", "token"):
            if key in kwargs:
                options[key] = kwargs[key]
        saved = cls.config_class.from_pretrained(path, **options)
        model, info = super().from_pretrained(
            path,
            *args,
            transformers_loading_kwargs=options,
            output_loading_info=True,
            **kwargs,
        )
        missing = set(info.get("missing_keys", []))
        sigma = {
            k for k in model.state_dict() if k.startswith("action_head.sigma_decoder.")
        }
        new_scale = (
            initialize_scale
            and saved.loss_type == "flow"
            and model.config.loss_type == "hetero_t"
            and missing == sigma
            and bool(sigma)
        )
        if (
            (missing and not new_scale)
            or info.get("unexpected_keys")
            or info.get("mismatched_keys")
            or info.get("error_msgs")
        ):
            raise ValueError(f"Checkpoint weight mismatch for {path}: {info}")
        if new_scale:
            # HF's no-init context does not initialize these custom parameters.
            # Construct them after loading, exactly as the original HT trainer.
            decoder = model.action_head._new_decoder()
            parameter = next(model.action_head.action_decoder.parameters())
            model.action_head.sigma_decoder = decoder.to(
                device=parameter.device, dtype=parameter.dtype
            )
            decoder.requires_grad_(model.config.tune_projector)
        return (model, info) if output_info else model


def register_gr00t_models():
    """Opt in to local HT construction for legacy Hugging Face checkpoints."""
    AutoConfig.register(GR00TConfig.model_type, GR00TConfig, exist_ok=True)
    AutoModel.register(GR00TConfig, GR00TNativeModel, exist_ok=True)
    AutoProcessor.register(GR00TConfig, Gr00tN1d7Processor, exist_ok=True)
