"""Bind the shared Cosmos policy to the native checkpoint inference service."""

from contextlib import contextmanager
from functools import partial

import torch

from ...policies.cosmos import CosmosCondition


@contextmanager
def cosmos_policy_generation(model, policy, *, direct_action):
    """Run the shared network through the native direct-action or Flow service.

    Only this model instance is rebound. The native service still owns image and
    prompt processing, noise seeding, and action unnormalization. No training
    hooks or global framework patches are installed.
    """
    if getattr(model, "_ht_regression_cosmos_bound", False):
        raise RuntimeError("This Cosmos model already has an active adapter.")
    expected = "clean_action" if direct_action else "velocity"
    if policy.prediction_type != expected:
        raise ValueError("Inference recipe does not match the policy parameterization.")

    def denoise(
        net=None, data_batch_packed=None, memory=None, video_temporal_causal=None
    ):
        condition = CosmosCondition(data_batch_packed, memory, video_temporal_causal)
        if not direct_action:
            # Native UniPC owns joint video/action times and state updates.
            return policy.forward_native(condition, network=net)
        condition = policy.encode_condition(condition)
        sample = torch.stack(
            [torch.zeros_like(x) for x in data_batch_packed.action.tokens]
        )
        return policy.forward_modalities(sample, 0, condition, network=net)

    replacements = {"denoise": denoise}
    if direct_action:
        replacements["generate_samples_from_batch"] = partial(
            _generate_direct_actions, model
        )
    missing = object()
    previous = {name: model.__dict__.get(name, missing) for name in replacements}
    model._ht_regression_cosmos_bound = True
    try:
        for name, method in replacements.items():
            setattr(model, name, method)
        yield
    finally:
        for name, method in previous.items():
            if method is missing:
                delattr(model, name)
            else:
                setattr(model, name, method)
        del model._ht_regression_cosmos_bound


@torch.no_grad()
def _generate_direct_actions(
    model,
    data_batch,
    *,
    net=None,
    guidance=1.0,
    num_steps=1,
    seed=0,
    has_negative_prompt=False,
    **kwargs,
):
    if guidance != 1.0 or num_steps != 1 or has_negative_prompt or kwargs:
        raise ValueError("Direct Cosmos evaluation requires one unguided forward.")
    from cosmos_framework.data.generator.action.utils.action_processing import (
        ActionProcessor,
        get_action_processing_records,
    )

    if isinstance(seed, int):
        seed = [seed + i for i in range(len(data_batch[model.input_caption_key]))]
    plans, clean, text, _, noise, _, _, noisy_actions = model._prepare_inference_data(
        data_batch, seed=seed, has_negative_prompt=False
    )
    if (
        not noisy_actions
        or clean.num_vision_items_per_sample is not None
        or clean.x0_tokens_sound is not None
    ):
        raise ValueError("Unsupported Cosmos input layout for direct action inference.")
    network = model.net if net is None else net
    outputs = []
    hook = network.register_forward_hook(
        lambda module, inputs, result: outputs.append(result)
    )
    try:
        model._get_velocity(
            net=network,
            noise_x=noise,
            timestep=torch.full(
                (clean.batch_size,),
                float(
                    model.rectified_flow_video.noise_scheduler.config.num_train_timesteps
                ),
                **model.tensor_kwargs_fp32,
            ),
            text_tokens=text,
            sequence_plans=plans,
            gen_data_clean=clean,
            has_noisy_actions=noisy_actions,
        )
    finally:
        hook.remove()
    if len(outputs) != 1:
        raise RuntimeError(
            f"Expected exactly one network evaluation, got {len(outputs)}."
        )
    records = get_action_processing_records(data_batch)
    # Raw action readout is already the clean prediction. It must never be
    # treated as velocity or passed through x_t - v / UniPC integration.
    return {
        "action": [
            ActionProcessor.postprocess_action(action, record)
            for action, record in zip(outputs[0]["preds_action"], records, strict=True)
        ]
    }
