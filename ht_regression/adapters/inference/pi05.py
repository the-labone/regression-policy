"""Connect a ht_regression PI0.5 pipeline to LeRobot's unchanged action queue."""

from contextlib import contextmanager
from types import MethodType

import torch


class PI05Adapter:
    """Keep the checkpoint's image preprocessing and external processor contract."""

    def __init__(self, native_policy):
        self.native_policy = native_policy

    def prepare(self, batch):
        from lerobot.utils.constants import (
            OBS_LANGUAGE_ATTENTION_MASK,
            OBS_LANGUAGE_TOKENS,
        )

        images, masks = self.native_policy._preprocess_images(batch)
        prepared = dict(
            images=images,
            image_masks=masks,
            tokens=batch[OBS_LANGUAGE_TOKENS],
            token_mask=batch[OBS_LANGUAGE_ATTENTION_MASK],
        )
        if "action" in batch:
            prepared["action"] = batch["action"]
        return prepared

    @contextmanager
    def bind(self, pipeline):
        """Replace only chunk prediction; restore the instance even after errors.

        The native policy still owns select_action's queue/reset. LeRobot applies
        its original checkpoint pre/postprocessors and environment transforms.
        One native policy cannot be evaluated concurrently by multiple runners.
        """
        native = self.native_policy
        if pipeline.policy.model is not native.model:
            raise ValueError(
                "The pipeline and native facade must share the loaded PI0.5 model."
            )
        if pipeline.policy.n_action_steps != native.config.n_action_steps:
            raise ValueError("Execution chunk does not match the native action queue.")
        if getattr(native, "_ht_regression_bound", False):
            raise RuntimeError(
                "The native policy is already bound to a ht_regression pipeline."
            )
        previous = native.__dict__.get("predict_action_chunk")
        had_override = "predict_action_chunk" in native.__dict__
        modes = [(m, m.training) for m in pipeline.modules()]

        @torch.no_grad()
        def predict_action_chunk(_native, batch, **kwargs):
            if kwargs:
                raise ValueError(
                    "Native sampler/RTC overrides must be configured on the objective."
                )
            return pipeline.predict_action(self.prepare(batch))["action_pred"]

        native._ht_regression_bound = True
        native.predict_action_chunk = MethodType(predict_action_chunk, native)
        try:
            pipeline.eval()
            native.reset()
            yield native
        finally:
            if had_override:
                native.predict_action_chunk = previous
            else:
                del native.predict_action_chunk
            del native._ht_regression_bound
            native.reset()
            for module, mode in modes:
                module.training = mode
