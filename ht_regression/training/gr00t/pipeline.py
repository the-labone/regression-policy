"""Local model construction inside the official GR00T data/trainer pipeline."""

from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.model.registry import MODEL_REGISTRY
from gr00t.utils.dist_utils import run_or_wait_on_rank0
from transformers.trainer_utils import get_last_checkpoint

from ...policies.gr00t.config import GR00TConfig
from ...policies.gr00t.model import GR00TNativeModel, register_gr00t_models


class GR00TPipeline(Gr00tN1d7Pipeline):
    model_class = GR00TNativeModel

    def _create_model(self):
        config = self.config
        checkpoint = config.training.start_from_checkpoint
        resume = config.training.resume_from_checkpoint
        if resume:
            if config.training.skip_weight_loading:
                raise ValueError(
                    "Cannot skip checkpoint weights when resuming training."
                )
            checkpoint = (
                get_last_checkpoint(config.training.output_dir)
                if resume is True
                else resume
            )
            if not checkpoint:
                raise ValueError("No checkpoint found for resuming GR00T training.")
            # Validate and restore all model weights before the native Trainer
            # restores its optimizer, scheduler and RNG state. Non-DeepSpeed
            # Trainer otherwise only warns about missing model parameters.
        if checkpoint and not config.training.skip_weight_loading:
            # Architecture comes from the checkpoint; training/objective options
            # come from the recipe, matching the historical fine-tuning path.
            fields = (
                "tune_llm",
                "tune_visual",
                "tune_projector",
                "tune_diffusion_model",
                "tune_vlln",
                "state_dropout_prob",
                "backbone_trainable_params_fp32",
                "load_bf16",
                "loss_type",
                "ht_sbias",
                "ht_df",
                "ht_mvt",
            )
            model = self.model_class.from_pretrained(
                checkpoint,
                **{name: getattr(config.model, name) for name in fields},
                initialize_scale=not resume,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
                **self.transformers_loading_kwargs,
            )
        else:
            model = self.model_class(
                config.model,
                transformers_loading_kwargs=self.transformers_loading_kwargs,
            )
        with run_or_wait_on_rank0(label="final_model_config.json write") as is_rank0:
            if is_rank0:
                (self.save_cfg_dir / "final_model_config.json").write_text(
                    model.config.to_filtered_json()
                )
        return model


def register_gr00t_training():
    register_gr00t_models()
    MODEL_REGISTRY[GR00TConfig] = GR00TPipeline
