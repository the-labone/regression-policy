"""Local HTTP transport and policy binding for Cosmos's simulator process."""

import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import torch

from ....adapters.inference.cosmos import cosmos_policy_generation


@contextmanager
def policy_server(bundle):
    """Serve native requests; close the socket/thread and restore model methods."""
    from cosmos_framework.scripts.action_policy_server_libero import _ActionHandler

    service = CosmosServiceAdapter(bundle)
    with cosmos_policy_generation(
        bundle.service.model, bundle.policy, direct_action=bundle.method != "flow"
    ):
        # server_close joins request handlers before the policy binding is restored.
        with ThreadingHTTPServer(("127.0.0.1", 0), _ActionHandler) as server:
            server.service = service
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{server.server_port}", service
            finally:
                server.shutdown()
                thread.join()


class CosmosServiceAdapter:
    """Keep native batch preprocessing and output transforms behind native HTTP."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.cfg = bundle.service.cfg
        self.errors = []

    def get_info(self):
        return {
            **self.bundle.service.get_info(),
            "ht_regression": self.bundle.provenance,
        }

    def predict_policy_batch(self, requests):
        try:
            result = self.bundle.service.predict_policy_batch(requests)
            actions = torch.as_tensor(result["actions"])
            policy = self.bundle.policy
            expected_shape = (len(requests), policy.horizon, policy.output_action_dim)
            if actions.shape != expected_shape or not torch.isfinite(actions).all():
                raise ValueError(
                    f"Expected finite Cosmos actions with shape {expected_shape}."
                )
            return result
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
            raise

    def predict_policy(self, request):
        return {
            "action": self.predict_policy_batch([request])["actions"][0],
            "video": [],
        }


def create_native_service(
    *, checkpoint, method, action_stats_path, vae_path, output_dir, seed
):
    """Construct the native service with the validated LIBERO-10 recipe."""
    from cosmos_framework.inference.common.args import CheckpointOverrides
    from cosmos_framework.scripts.action_policy_server_libero import (
        ActionModelService,
        ActionServerArgs,
    )

    class EagerArgs(ActionServerArgs):
        def build_setup_overrides(self):
            overrides = super().build_setup_overrides()
            overrides.use_torch_compile = False
            overrides.use_cuda_graphs = False
            return overrides

    # Historical Flow evaluation enables the native compiled model; direct
    # HT/MSE use eager execution to capture exactly one joint-network readout.
    args_type = ActionServerArgs if method == "flow" else EagerArgs
    service = ActionModelService(
        args_type(
            checkpoint=CheckpointOverrides(
                checkpoint_path=str(checkpoint),
                experiment="action_policy_libero_nano",
                experiment_overrides=[f"model.config.tokenizer.vae_path={vae_path}"],
            ),
            output_dir=Path(output_dir),
            action_normalization="quantile_rot",
            action_stats_path=action_stats_path,
            raw_action_dim=10,
            action_chunk_size=16,
            fps=20,
            seed=seed,
            guidance=1.0,
            num_steps=30 if method == "flow" else 1,
            sampler="unipc",
        )
    )
    return service
