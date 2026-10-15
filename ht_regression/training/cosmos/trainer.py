"""Native Cosmos video flow + direct action HT training adapters.

Ported from the successful video_direct_action_20260913 recipe. Video noise,
packing, auxiliary loss and trainable modules remain native. Action inputs
are zero at t=0; targets are clean actions, with global first-batch calibration.
"""

import hashlib
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .loss import EPS, NU, bias_for_rms, direct_loss, masked_residual_stats

_INSTALLED = False


def rank0():
    return not dist.is_initialized() or dist.get_rank() == 0


def zero_action_input(pack):
    assert pack.action is not None and pack.sound is None
    assert all(torch.count_nonzero(m).item() == 0 for m in pack.action.condition_mask)
    pack.action.tokens = [torch.zeros_like(x) for x in pack.action.tokens]
    pack.action.timesteps = torch.zeros_like(pack.action.timesteps)


def clean_action_targets(clean, noised):
    """Leave every video field untouched, including its native noise draw."""
    assert clean.x0_tokens_action is not None
    noised.xt_tokens_action = [torch.zeros_like(a) for a in clean.x0_tokens_action]
    noised.vt_target_action = [a.detach().clone() for a in clean.x0_tokens_action]
    noised.epsilon_action = [torch.zeros_like(a) for a in clean.x0_tokens_action]
    return noised


def save_audit(name, record):
    if rank0():
        root = Path(os.environ["DIRECT_AUDIT_DIR"])
        root.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(record, indent=2) + "\n")


def action_loss(model, pred, target, condition_mask, raw_action_dim=None):
    pack = SimpleNamespace(
        action=SimpleNamespace(
            condition_mask=condition_mask, raw_action_dim=raw_action_dim
        )
    )
    head = model.net.llm2action_ht_sigma_head
    if not getattr(model, "_video_direct_calibrated", False) and model._ht_iter > 0:
        # Native resume restored sigma weights together with optimizer/RNG state.
        # Never calibrate a trained head again.
        model._video_direct_calibrated = True
    if not getattr(model, "_video_direct_calibrated", False):
        assert model._ht_iter == 0
        assert torch.count_nonzero(head.weight).item() == 0
        assert torch.count_nonzero(head.bias).item() == 0
        sse, count, _ = masked_residual_stats(
            pred, target, condition_mask, raw_action_dim
        )
        pair = torch.stack([sse, count])
        if dist.is_initialized():
            dist.all_reduce(pair)
        rms = float((pair[0] / pair[1]).sqrt())
        with torch.no_grad():
            head.bias.fill_(bias_for_rms(rms))
        record = {
            "nu": NU,
            "effective_dim": 160,
            "initial_residual_rms": rms,
            "sigma_initial": float(F.softplus(head.bias.float()).item() + EPS),
            "sigma_head_bias": float(head.bias),
            "calibration_samples": int(pair[1] / 160),
            "action_input": "zeros",
            "action_timestep": 0,
            "action_target": "clean action (not velocity)",
            "video_supervision": True,
            "video_training": "native flow matching",
            "video_loss_weight": model.config.rectified_flow_training_config.loss_scale,
            "action_loss_weight": model.config.rectified_flow_training_config.action_loss_weight,
            "initialization": "fresh native LIBERO recipe from Cosmos3-Nano base",
        }
        assert abs(record["sigma_initial"] - rms) < 1e-4
        save_audit("initialization.json", record)
        if rank0():
            print("VIDEO_DIRECT_INIT", json.dumps(record), flush=True)
        model._video_direct_calibrated = True
    loss, sigma = direct_loss(model, pred, target, pack)
    model._video_direct_sigma = float(sigma)
    return loss


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from cosmos_framework.model.generator.mot.cosmos3_vfm_network import (
        Cosmos3VFMNetwork,
    )
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

    original_noise = OmniMoTModel._add_noise_to_input
    original_denoise = OmniMoTModel.denoise
    original_losses = OmniMoTModel._compute_losses
    original_after_backward = OmniMoTModel.on_after_backward

    def add_noise(self, gen_data_clean, packed_sequence, sigmas, **kwargs):
        noised = original_noise(self, gen_data_clean, packed_sequence, sigmas, **kwargs)
        clean_action_targets(gen_data_clean, noised)
        if not getattr(self, "_video_direct_noise_checked", False):
            assert any(torch.count_nonzero(x).item() for x in noised.vt_target_vision)
            record = {
                "native_video_sigmas": sigmas.detach().cpu().tolist(),
                "video_target_rms": [
                    float(v.float().square().mean().sqrt())
                    for v in noised.vt_target_vision
                ],
                "action_target_equals_clean": all(
                    torch.equal(x, y)
                    for x, y in zip(
                        noised.vt_target_action,
                        gen_data_clean.x0_tokens_action,
                        strict=True,
                    )
                ),
                "video_fields_untouched_by_action_adapter": True,
            }
            assert record["action_target_equals_clean"]
            save_audit("training_inputs.json", record)
            self._video_direct_noise_checked = True
        return noised

    def denoise(
        self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None
    ):
        zero_action_input(data_batch_packed)
        # All vision tokens, timesteps, masks and network heads remain native.
        out = original_denoise(
            self,
            net=net,
            data_batch_packed=data_batch_packed,
            memory=memory,
            video_temporal_causal=video_temporal_causal,
        )
        if torch.is_grad_enabled() and not getattr(
            self, "_video_direct_gradient_hooks", False
        ):
            self._video_direct_gradients = {}

            def hook_for(key):
                def check(grad):
                    assert torch.isfinite(grad).all(), f"Nonfinite {key} gradient"
                    value = float(grad.float().norm())
                    assert value > 0, f"Missing {key} supervision"
                    self._video_direct_gradients[key] = value
                    if len(self._video_direct_gradients) == 2:
                        save_audit("first_backward.json", self._video_direct_gradients)
                        if rank0():
                            print(
                                "VIDEO_DIRECT_FIRST_BACKWARD",
                                json.dumps(self._video_direct_gradients),
                                flush=True,
                            )
                    return grad

                return check

            out["preds_action"][0].register_hook(
                hook_for("action_prediction_gradient_norm")
            )
            out["preds_vision"][0].register_hook(
                hook_for("video_prediction_gradient_norm")
            )
            video_params = [p for n, p in self.net.named_parameters() if "llm2vae" in n]
            assert video_params and all(p.requires_grad for p in video_params)
            self._video_direct_gradient_hooks = True
        return out

    def losses(self, *args, **kwargs):
        total, info = original_losses(self, *args, **kwargs)
        assert torch.isfinite(total), "Nonfinite joint objective"
        if rank0() and (self._ht_iter < 2 or self._ht_iter % 50 == 0):
            # Only first microbatch at each logged optimizer update.
            if getattr(self, "_video_direct_logged", -1) != self._ht_iter:
                print(
                    "VIDEO_DIRECT_TRAIN",
                    json.dumps(
                        {
                            "iteration": self._ht_iter,
                            "nu": NU,
                            "video_loss": float(
                                info["flow_matching_loss_vision"].detach()
                            ),
                            "action_ht_loss": float(
                                info["flow_matching_loss_action"].detach()
                            ),
                            "total_loss": float(total.detach()),
                            "sigma": self._video_direct_sigma,
                            "action_input": "zeros",
                            "action_timestep": 0,
                        }
                    ),
                    flush=True,
                )
                self._video_direct_logged = self._ht_iter
        return total, info

    def after_backward(self, iteration=0):
        original_after_backward(self, iteration)
        # Legacy sigma MLP is attached after FSDP: reduce ordinary replicated params.
        if dist.is_initialized() and dist.get_world_size() > 1:
            for param in self.net.llm2action_ht_sigma_head.parameters():
                if not hasattr(param, "device_mesh") and param.grad is not None:
                    dist.all_reduce(param.grad)
                    param.grad.div_(dist.get_world_size())
        if self._ht_iter > 0 and not getattr(
            self, "_video_direct_sigma_sync_checked", False
        ):
            values = torch.cat(
                [
                    p.detach().float().reshape(-1)
                    for p in self.net.llm2action_ht_sigma_head.parameters()
                ]
            )
            if dist.is_initialized():
                reference = values.clone()
                dist.broadcast(reference, src=0)
                assert torch.equal(
                    values, reference
                ), "Sigma parameters diverged across ranks"
            save_audit(
                "first_update_verified.json",
                {
                    "iteration": self._ht_iter,
                    "sigma_parameters_identical_across_ranks": True,
                    "both_prediction_gradients_nonzero": len(
                        self._video_direct_gradients
                    )
                    == 2,
                },
            )
            self._video_direct_sigma_sync_checked = True

    if "DIRECT_AUDIT_DIR" in os.environ:
        preserved = [
            OmniMoTModel.training_step,
            OmniMoTModel._get_training_inputs,
            OmniMoTModel._get_train_noise_level_vision,
            Cosmos3VFMNetwork._decode_vision,
            original_noise,
            original_losses,
        ]
        save_audit(
            "native_source_hashes.json",
            {
                fn.__qualname__: hashlib.sha256(
                    inspect.getsource(fn).encode()
                ).hexdigest()
                for fn in preserved
            },
        )
    OmniMoTModel._add_noise_to_input = add_noise
    OmniMoTModel.denoise = denoise
    OmniMoTModel._compute_losses = losses
    OmniMoTModel._ht_action_loss = action_loss
    OmniMoTModel.on_after_backward = after_backward
    _INSTALLED = True
