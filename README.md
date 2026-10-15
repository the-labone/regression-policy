<h1 align="center">Residual Modeling for Regression Policies</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.12231"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/paper-dark.svg"><source media="(prefers-color-scheme: light)" srcset="assets/readme/paper.svg"><img src="assets/readme/paper.svg" width="105" height="42" alt="Paper"></picture></a>&nbsp;&nbsp;
  <a href="https://projects.thelabone.org/regression-policy/"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/project-page-dark.svg"><source media="(prefers-color-scheme: light)" srcset="assets/readme/project-page.svg"><img src="assets/readme/project-page.svg" width="161" height="42" alt="Project Page"></picture></a>&nbsp;&nbsp;
  <a href="https://huggingface.co/collections/yuchen0187/regressionpolicy"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/models-dark.svg"><source media="(prefers-color-scheme: light)" srcset="assets/readme/models.svg"><img src="assets/readme/models.svg" width="119" height="42" alt="Models"></picture></a>
</p>

<p align="center">
  <a href="assets/teaser.pdf"><img src="assets/teaser.png" width="100%" alt="HT-Policy models heavy-tailed action residuals to close the performance gap with generative policies."></a>
</p>

We revisit the gap between regression and generative policies from the perspective of statistic modeling. Our finding is the performance gap between MSE- and Flow-Policies is attributed to observation-dependent and heavy-tailed residuals. By modeling the heavy tails explicitly, our model (HT-Policy) achieves competitive performance with generative policies, with higher training and inference efficiency.

The first part of this repo provides an easy-to-use [HT-Loss implementation](#ht-loss-usage). It has an inferface similar to PyTorch's `MSELoss`. The second part provides code for our experiments on different benchmarks, starting from [installation](#installation).

<a id="release-plan"></a>

## <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icons/release-plan-dark.svg"><img src="assets/readme/icons/release-plan.svg" width="24" height="28" align="absmiddle" alt=""></picture>&nbsp;&nbsp;Release Plan

- [x] PyTorch implementation of HT-Loss
- [x] RoboMimic training and evaluation code
- [x] VLA evaluation code
- [x] HT-Policy checkpoints for RoboMimic state policies
- [x] HT-Policy checkpoints for VLA/WAM models
- [x] VLA/WAM training code
- [x] HT-Policy checkpoints for RoboMimic vision policies

<a id="ht-policy-demos"></a>

## <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icons/demos-dark.svg"><img src="assets/readme/icons/demos.svg" width="24" height="28" align="absmiddle" alt=""></picture>&nbsp;&nbsp;HT-Policy Demos

<p align="center">
  <img src="assets/videos/peel-note.gif" width="24%" alt="Peel Note — real-world robot demonstration">
  <img src="assets/videos/real-push.gif" width="24%" alt="Push T — real-world robot demonstration">
  <img src="assets/videos/robocasa-gr1.gif" width="24%" alt="Store Bottle — GR1 placing a bottle in a cabinet">
  <img src="assets/videos/tool-hang.gif" width="24%" alt="Tool Hang — simulation demonstration">
  <br>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/readme/demos-labels-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/readme/demos-labels.svg">
    <img src="assets/readme/demos-labels.svg" width="98%" alt="Peel Note and Push T: Real World. Store Bottle and Tool Hang: Simulation.">
  </picture>
</p>

<a id="ht-loss-usage"></a>

## <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icons/ht-loss-dark.svg"><img src="assets/readme/icons/ht-loss.svg" width="24" height="28" align="absmiddle" alt=""></picture>&nbsp;&nbsp;HT-Loss Usage

```bash
# From the repository root
uv pip install ./ht_loss
# pip install ./ht_loss
```

```python
from ht_loss import HTLoss

criterion = HTLoss()  # nu = 4d, where d = T * D
prediction, raw_scale = model(observations)  # [B, T, D], [B, 1]
loss = criterion(prediction, target, raw_scale)
loss.backward()
```

You can add a scale head alongside the action head to predict `raw_scale` from shared features. See [HT-Loss](ht_loss/README.md) for the scale head example and options.

<a id="installation"></a>

## <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icons/installation-dark.svg"><img src="assets/readme/icons/installation.svg" width="24" height="28" align="absmiddle" alt=""></picture>&nbsp;&nbsp;Installation

From this section, we provide codes for training and evaluating HT-Policy on different benchmarks. The first step is to install [uv](https://docs.astral.sh/uv/getting-started/installation/) and clone the repository:

```bash
git clone https://github.com/the-labone/RegressionPolicy.git
cd RegressionPolicy
```

Install the environment you need:

```bash
uv run scripts/install.py robomimic
```

**Options** — replace `robomimic` with your selection:

| Option | Environment |
| --- | --- |
| `robomimic` | RoboMimic training and evaluation |
| `gr1` | GR00T training / evaluation on RoboCasa-GR1 |
| `simpler` | GR00T training / evaluation on SimplerEnv |
| `pi05` | π0.5 training / evaluation on LIBERO |
| `cosmos` | Cosmos training / evaluation on LIBERO |

<a id="training"></a>

## <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icons/training-dark.svg"><img src="assets/readme/icons/training.svg" width="24" height="28" align="absmiddle" alt=""></picture>&nbsp;&nbsp;Training

### RoboMimic

The two examples below show how to train Transformer- and UNet-based policies on RoboMimic. Both architectures support state- and vision-based observations; simply choose the appropriate configuration and dataset for your setup.

#### Prepare dataset

```bash
# RoboMimic Lift PH — state observations
uv run scripts/download_robomimic.py --task lift --split ph --modality state

# RoboMimic Lift PH — image observations
uv run scripts/download_robomimic.py --task lift --split ph --modality vision
```

#### HT-Policy Transformer (State observations)

```bash
uv run --no-sync python -m ht_regression.training.robomimic.train \
  experiment=robomimic/state/lift/ph/ht_transformer_nu200 \
  dataset.path=data/robomimic/lift/ph/low_dim_abs.hdf5
```

#### HT-Policy UNet (Image observations)

```bash
uv run --no-sync python -m ht_regression.training.robomimic.train \
  experiment=robomimic/vision/lift/ph/ht_unet_2d \
  dataset.path=data/robomimic/lift/ph/image_abs.hdf5
```

Vision configurations use `ht_unet_2d` or `ht_transformer_2d`, where
`d` is the prediction horizon times the action dimension. Both backbones use
batch 256, 300k updates, and evaluation every 30k updates for non-Lift tasks.
Lift uses batch 64 and a 500-epoch budget, evaluated every 50 epochs.

| Argument | Options |
| --- | --- |
| `experiment=...` | [Task and backbone configurations](configs/experiment/robomimic) |
| `objective=...` | `ht`, `mse`, `flow`, `mip`, `diffusion` |

### VLA / WAM

#### 1. Download / prepare data

```bash
# GR00T / GR1
uvx --from huggingface-hub hf download nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim --repo-type dataset \
  --include 'LeRobot/gr1_unified.*/*' --local-dir data/gr1

# GR00T / SimplerEnv
uvx --from huggingface-hub hf download IPEC-COMMUNITY/bridge_orig_lerobot --repo-type dataset \
  --local-dir data/bridge
uvx --from huggingface-hub hf download IPEC-COMMUNITY/fractal20220817_data_lerobot --repo-type dataset \
  --local-dir data/fractal

# PI0.5 / LIBERO
uvx --from huggingface-hub hf download HuggingFaceVLA/libero --repo-type dataset --local-dir data/libero

# Cosmos / LIBERO-10
uvx --from huggingface-hub hf download nvidia/LIBERO_LeRobot_v3 --repo-type dataset \
  --include 'libero_10/**' --local-dir data/libero_v3
```

SimplerEnv preprocessing: [Bridge](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md#fine-tune-simpler-env-bridge-dataset-widowx-robot) | [Fractal](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/SimplerEnv/README.md#fine-tune-simpler-env-fractal-dataset-google-robot).

Base weights: [GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) | [PI0.5 LIBERO](https://huggingface.co/lerobot/pi05_libero_finetuned) | [Cosmos](https://github.com/NVIDIA/cosmos-framework/blob/18d90eacbda778e666e6978dc1ab6ba8635400e3/docs/setup.md).

#### 2. Install environment

```bash
# Choose your benchmark
uv run scripts/install.py gr1
uv run scripts/install.py simpler
uv run scripts/install.py pi05
uv run scripts/install.py cosmos
```

#### 3. Train (8 GPUs)

```bash
# GR00T / GR1
uv run scripts/train.py gr1 \
  --data-root data/gr1/LeRobot --base-model /path/to/gr00t_base \
  --output-dir outputs/gr1_ht

# GR00T / Bridge
uv run scripts/train.py simpler --task bridge \
  --data-root data/bridge --base-model /path/to/gr00t_base \
  --output-dir outputs/bridge_ht

# GR00T / Fractal
uv run scripts/train.py simpler --task fractal \
  --data-root data/fractal --base-model /path/to/gr00t_base \
  --output-dir outputs/fractal_ht

# PI0.5 / LIBERO
uv run scripts/train.py pi05 \
  --data-root data/libero --base-model /path/to/pi05_libero_base \
  --output-dir outputs/pi05_ht

# Cosmos / LIBERO-10
uv run scripts/train.py cosmos \
  --data-root data/libero_v3/libero_10 --base-model /path/to/Cosmos3-Nano \
  --vae-path /path/to/Wan2.2_VAE.pth --output-dir outputs/cosmos_ht
```

<a id="evaluation"></a>

## <picture><source media="(prefers-color-scheme: dark)" srcset="assets/readme/icons/evaluation-dark.svg"><img src="assets/readme/icons/evaluation.svg" width="24" height="28" align="absmiddle" alt=""></picture>&nbsp;&nbsp;Evaluation

| Model | Benchmark | Checkpoints |
| --- | --- | --- |
| Transformer / UNet | RoboMimic | [Hugging Face](https://huggingface.co/yuchen0187/RegressionPolicy-RoboMimic) |
| GR00T | RoboCasa-GR1 | [Hugging Face](https://huggingface.co/yuchen0187/RegressionPolicy-GR00T/tree/main/gr1) |
| GR00T | SimplerEnv | [Hugging Face](https://huggingface.co/yuchen0187/RegressionPolicy-GR00T/tree/main/simpler) |
| π0.5 | LIBERO | [Hugging Face](https://huggingface.co/yuchen0187/RegressionPolicy-Pi0.5) |
| Cosmos | LIBERO | [Hugging Face](https://huggingface.co/yuchen0187/RegressionPolicy-Cosmos) |

### Transformer / UNet on RoboMimic

Download and evaluate the released Lift PH HT-Policy with a Transformer backbone:

```bash
uv run scripts/download_checkpoint.py robomimic \
  --path state/ht/lift_ph/ht-transformer

uv run scripts/evaluate.py robomimic \
  --checkpoint checkpoints/robomimic/state/ht/lift_ph/ht-transformer/best_nu100_last10-100.0.pt \
  --checkpoint-format release \
  --env-config configs/evaluation/robomimic/lift.json \
  --device cuda \
  --output-dir evaluation_results/robomimic_lift
```

You can specify the checkpoint to evaluate with `--checkpoint`.

### GR00T on RoboCasa-GR1

```bash
uv run scripts/download_checkpoint.py gr00t \
  --path gr1/checkpoint-60000

uv run scripts/evaluate.py gr1 \
  --checkpoint checkpoints/gr00t/gr1/checkpoint-60000 \
  --output-dir evaluation_results/gr1
```

### GR00T on SimplerEnv

Set `simpler_suite` to `bridge` or `fractal`:

```bash
simpler_suite=bridge
uv run scripts/download_checkpoint.py gr00t \
  --path simpler/${simpler_suite}/checkpoint-20000

uv run scripts/evaluate.py simpler \
  --checkpoint checkpoints/gr00t/simpler/${simpler_suite}/checkpoint-20000 \
  --suite ${simpler_suite} \
  --output-dir evaluation_results/simpler_${simpler_suite}
```

### π0.5 on LIBERO

```bash
uv run scripts/download_checkpoint.py pi05 \
  --path libero/checkpoint-30000/pretrained_model

uv run scripts/evaluate.py pi05 \
  --checkpoint checkpoints/pi05/libero/checkpoint-30000/pretrained_model \
  --output-dir evaluation_results/pi05
```

By default, all four LIBERO suites are evaluated. Add `--suite libero_10` to select one suite.

### Cosmos on LIBERO

```bash
uv run scripts/download_checkpoint.py cosmos --all

uv run scripts/evaluate.py cosmos \
  --checkpoint checkpoints/cosmos/libero_10/iter_000002000 \
  --method ht \
  --action-stats-path checkpoints/cosmos/libero_10/action_stats.json \
  --vae-path checkpoints/cosmos/assets/wan22_vae/Wan2.2_VAE.pth \
  --deterministic \
  --output-dir evaluation_results/cosmos
```

<a id="acknowledgements"></a>

## Acknowledgements

Our code builds on [Much Ado About Noising](https://github.com/simchowitzlabpublic/much-ado-about-noising), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T), [OpenPi](https://github.com/Physical-Intelligence/openpi), and [Cosmos Framework](https://github.com/NVIDIA/cosmos-framework). We thank the authors and contributors for making their code and models publicly available.

<a id="citation"></a>

## Citation

```bibtex
@misc{zhou2026residualmodelingclosesregression,
  title         = {Residual Modeling Closes the Regression and Generative Policy Gap in Robot Learning},
  author        = {Yuchen Zhou and Jiacheng You and Weikang Wan and Weijun Dong and Yang Gao and Jiayuan Mao},
  year          = {2026},
  eprint        = {2610.12231},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2610.12231}
}
```
