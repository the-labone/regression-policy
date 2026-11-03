"""Contracts for the local extension of official GR00T (no VLM download)."""

import importlib
import importlib.util
import math
from copy import deepcopy

import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def test_local_model_construction_is_available():
    name = "ht_regression.policies.gr00t.native"
    assert (
        importlib.util.find_spec(name) is not None
    ), "Local GR00T model construction is missing"


def test_full_model_construction_is_available():
    assert importlib.util.find_spec("ht_regression.policies.gr00t.model") is not None


@pytest.fixture
def model_class(monkeypatch):
    module = importlib.import_module("ht_regression.policies.gr00t.model")

    # Only the external VLM/tokenizer boundary is replaced. Serialization,
    # configuration and action modules use the real implementations.
    monkeypatch.setattr(
        module,
        "get_backbone_cls",
        lambda config: lambda **kwargs: torch.nn.Linear(3, 3),
    )
    monkeypatch.setattr(module, "Gr00tN1d7DataCollator", lambda **kwargs: None)
    return module.GR00TNativeModel


def test_checkpoint_roundtrip_is_strict(model_class, tmp_path):
    model = model_class(small_config(loss_type="hetero_t", ht_mvt=True))
    model.save_pretrained(tmp_path)
    loaded, info = model_class.from_pretrained(tmp_path, output_loading_info=True)
    assert not any(
        info[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys")
    )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "damage", ["missing_sigma", "missing_mean", "unexpected", "mismatched"]
)
def test_incomplete_checkpoint_is_rejected(model_class, tmp_path, damage):
    from safetensors.torch import load_file, save_file

    model_class(small_config(loss_type="hetero_t")).save_pretrained(tmp_path)
    path = tmp_path / "model.safetensors"
    state = load_file(path)
    if damage.startswith("missing"):
        key = (
            "action_head.sigma_decoder.layer1.W"
            if damage == "missing_sigma"
            else "action_head.action_decoder.layer1.W"
        )
        del state[key]
    elif damage == "unexpected":
        state["unknown.weight"] = torch.zeros(1)
    else:
        state["action_head.sigma_decoder.layer1.W"] = torch.zeros(1)
    save_file(state, path, metadata={"format": "pt"})
    with pytest.raises(
        (ValueError, RuntimeError), match="(?i)(checkpoint|mismatch|size)"
    ):
        model_class.from_pretrained(tmp_path, initialize_scale=True)


def test_flow_conversion_initializes_only_scale(model_class, tmp_path):
    flow = model_class(small_config(loss_type="flow"))
    flow.save_pretrained(tmp_path)
    with pytest.raises(ValueError, match="[Cc]heckpoint"):
        model_class.from_pretrained(tmp_path, loss_type="hetero_t")
    ht = model_class.from_pretrained(
        tmp_path, loss_type="hetero_t", initialize_scale=True
    )
    for key, value in flow.state_dict().items():
        torch.testing.assert_close(value, ht.state_dict()[key], rtol=0, atol=0)
    assert torch.isfinite(ht.action_head.sigma_decoder.layer1.W).all()
    assert ht.action_head.sigma_decoder.layer1.W.std() > 0


@pytest.mark.parametrize("task", ["gr1", "bridge", "fractal"])
def test_training_recipe_uses_local_pipeline(task):
    import json
    from pathlib import Path

    from gr00t.configs.base_config import Config
    from gr00t.model.registry import MODEL_REGISTRY

    from ht_regression.policies.gr00t.config import GR00TConfig
    from ht_regression.training.gr00t.config import primitive, restore_modalities
    from ht_regression.training.gr00t.pipeline import (
        GR00TPipeline,
        register_gr00t_training,
    )

    recipe = json.loads(
        (
            Path(__file__).parents[1] / f"configs/experiment/gr00t/{task}_ht.json"
        ).read_text()
    )
    register_gr00t_training()
    config = restore_modalities(Config(model=GR00TConfig()).load_dict(recipe))
    config.validate()
    assert primitive(config.model) == recipe["model"]
    assert MODEL_REGISTRY[type(config.model)] is GR00TPipeline


def test_installer_uses_official_gr00t():
    from scripts.install import REPOS

    assert REPOS["gr00t"] == (
        "https://github.com/NVIDIA/Isaac-GR00T.git",
        "b9955401d50c92a29258732e3ad6ccd579f1bdc0",
    )


def test_installer_replaces_upstream_lock(tmp_path, monkeypatch):
    from scripts import install

    (tmp_path / "uv.lock").write_text("upstream lock")
    commands = []
    installer = install.Installer()
    monkeypatch.setattr(installer, "run", lambda *args, **kwargs: commands.append(args))
    installer.sync(tmp_path, "gr00t", "3.12")
    assert (tmp_path / "uv.lock").read_bytes() == (
        install.ROOT / "environments/gr00t.uv.lock"
    ).read_bytes()
    assert "--locked" in commands[0]


def test_gr1_shipped_object_statistics_survive_cleanup(tmp_path):
    import json

    from gr00t.data.stats import generate_stats

    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(
        json.dumps(
            {"features": {"observation.state": {"dtype": "object", "shape": [2]}}}
        )
    )
    shipped = {"min": [0, 0], "max": [1, 1], "mean": [0.5, 0.5], "std": [0.1, 0.1]}
    (meta / "stats.json").write_text(
        json.dumps({"observation.state": shipped, "removed": shipped})
    )
    generate_stats(tmp_path)
    stats = json.loads((meta / "stats.json").read_text())
    assert stats["observation.state"] == shipped
    assert "removed" not in stats


def test_auto_model_registration_loads_local_ht(model_class, tmp_path):
    from transformers import AutoModel

    from ht_regression.policies.gr00t.model import register_gr00t_models

    model_class(small_config(loss_type="hetero_t")).save_pretrained(tmp_path)
    register_gr00t_models()
    loaded = AutoModel.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(loaded, model_class)
    assert hasattr(loaded.action_head, "sigma_decoder")


@pytest.mark.parametrize("missing_sigma", [False, True])
def test_pipeline_resume_restores_scale_strictly(model_class, tmp_path, missing_sigma):
    from gr00t.configs.base_config import Config
    from safetensors.torch import load_file, save_file

    from ht_regression.training.gr00t.pipeline import GR00TPipeline

    base = tmp_path / "base"
    model_class(small_config(loss_type="flow")).save_pretrained(base)
    output = tmp_path / "run"
    checkpoint = output / "checkpoint-10"
    resumed = model_class(small_config(loss_type="hetero_t"))
    with torch.no_grad():
        for param in resumed.action_head.sigma_decoder.parameters():
            param.fill_(0.42)
    resumed.save_pretrained(checkpoint)
    (checkpoint / "trainer_state.json").write_text('{"global_step": 10}')
    if missing_sigma:
        path = checkpoint / "model.safetensors"
        state = load_file(path)
        del state["action_head.sigma_decoder.layer1.W"]
        save_file(state, path, metadata={"format": "pt"})
    config = Config(model=small_config(loss_type="hetero_t"))
    config.training.start_from_checkpoint = str(base)
    config.training.output_dir = str(output)
    config.training.resume_from_checkpoint = True
    pipeline = GR00TPipeline(config, tmp_path)
    if missing_sigma:
        with pytest.raises(ValueError, match="[Cc]heckpoint"):
            pipeline._create_model()
    else:
        model = pipeline._create_model()
        for key, value in resumed.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)


@pytest.fixture
def native():
    return importlib.import_module("ht_regression.policies.gr00t.native")


def small_config(**overrides):
    from ht_regression.policies.gr00t.config import GR00TConfig

    values = dict(
        backbone_embedding_dim=32,
        hidden_size=32,
        input_embedding_dim=32,
        max_state_dim=5,
        max_action_dim=5,
        action_horizon=3,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=2,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=16,
        use_alternate_vl_dit=False,
        state_dropout_prob=0.0,
        diffusion_model_cfg=dict(
            positional_embeddings=None,
            num_layers=2,
            num_attention_heads=2,
            attention_head_dim=16,
            norm_type="ada_norm",
            dropout=0.0,
            final_dropout=False,
            output_dim=32,
            interleave_self_attention=True,
        ),
    )
    values.update(overrides)
    return GR00TConfig(**values)


def batch(config):
    backbone = BatchFeature(
        dict(
            backbone_features=torch.randn(2, 4, config.backbone_embedding_dim),
            backbone_attention_mask=torch.ones(2, 4, dtype=torch.bool),
            image_mask=torch.ones(2, 4, dtype=torch.bool),
        )
    )
    actions = BatchFeature(
        dict(
            state=torch.randn(2, 1, config.max_state_dim),
            action=torch.randn(2, config.action_horizon, config.max_action_dim),
            action_mask=torch.ones(2, config.action_horizon, config.max_action_dim),
            embodiment_id=torch.zeros(2, dtype=torch.long),
        )
    )
    return backbone, actions


@pytest.mark.parametrize("multivariate", [False, True])
def test_ht_loss_preserves_masked_joint_distribution(native, multivariate):
    config = small_config(
        loss_type="hetero_t", ht_df=4.0, ht_sbias=0.0, ht_mvt=multivariate
    )
    head = native.GR00TNativeActionHead(config).eval()
    with torch.no_grad():
        for decoder in (head.action_decoder, head.sigma_decoder):
            for value in decoder.parameters():
                value.zero_()
    backbone, actions = batch(config)
    actions.action.fill_(1.0)
    actions.action_mask[:, :, -1] = 0
    actions.action[:, :, -1] = 9999.0
    result = head(deepcopy(backbone), deepcopy(actions))
    dimension = 12
    nu = 4.0 if multivariate else 4.0 * dimension
    sigma = math.log(2.0) + 1e-3
    nll = 0.5 * (nu + dimension) * math.log1p(
        dimension / (nu * sigma**2)
    ) + dimension * math.log(sigma)
    assert result["loss"].item() == pytest.approx(2 * nll / (24 + 1e-6), rel=1e-6)
    assert result["action_loss"].shape == (2,)


def test_ht_gradients_reach_mean_scale_and_dit(native):
    head = native.GR00TNativeActionHead(small_config(loss_type="hetero_t", ht_mvt=True))
    backbone, actions = batch(head.config)
    head(backbone, actions)["loss"].backward()
    for module in (head.action_decoder, head.sigma_decoder, head.model):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )


@pytest.mark.parametrize("method", ["hetero_t", "mse", "l1"])
def test_regression_inference_is_one_deterministic_prediction(native, method):
    head = native.GR00TNativeActionHead(small_config(loss_type=method)).eval()
    backbone, actions = batch(head.config)
    del actions["action"]
    calls = []
    handle = head.model.register_forward_hook(lambda *args: calls.append(1))
    first = head.get_action(deepcopy(backbone), deepcopy(actions))["action_pred"]
    assert len(calls) == 1
    torch.manual_seed(999)
    second = head.get_action(deepcopy(backbone), deepcopy(actions))["action_pred"]
    handle.remove()
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_flow_weights_and_forward_match_official_head(native):
    from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead

    config = small_config(loss_type="flow")
    torch.manual_seed(12)
    official = Gr00tN1d7ActionHead(config).eval()
    torch.manual_seed(12)
    local = native.GR00TNativeActionHead(config).eval()
    assert local.state_dict().keys() == official.state_dict().keys()
    for key, value in official.state_dict().items():
        torch.testing.assert_close(value, local.state_dict()[key], rtol=0, atol=0)
    backbone, actions = batch(config)
    torch.manual_seed(87)
    expected = official(deepcopy(backbone), deepcopy(actions))["loss"]
    torch.manual_seed(87)
    actual = local(deepcopy(backbone), deepcopy(actions))["loss"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sigma_freezing_and_legacy_parameter_names(native):
    head = native.GR00TNativeActionHead(
        small_config(loss_type="hetero_t", tune_projector=False)
    )
    assert all(not p.requires_grad for p in head.sigma_decoder.parameters())
    head.train()
    head.set_frozen_modules_to_eval_mode()
    assert not head.sigma_decoder.training
    assert "sigma_decoder.layer1.W" in head.state_dict()
    assert not any("_extra_state" in name for name in head.state_dict())


def test_ht_config_roundtrip_preserves_legacy_default(native, tmp_path):
    from ht_regression.policies.gr00t.config import GR00TConfig

    config = small_config(loss_type="hetero_t", ht_df=224.0, ht_sbias=-0.5)
    assert config.ht_mvt is False
    config.save_pretrained(tmp_path)
    restored = GR00TConfig.from_pretrained(tmp_path, local_files_only=True)
    assert (restored.loss_type, restored.ht_df, restored.ht_sbias, restored.ht_mvt) == (
        "hetero_t",
        224.0,
        -0.5,
        False,
    )
    assert restored.to_filtered_dict()["ht_df"] == 224.0
