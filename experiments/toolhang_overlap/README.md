# Multi-stage MSE with overlapping training phases

This directory and `../toolhang_multistage/` contain the six-stage Tool-Hang
experiment. The four runtime modules are byte-for-byte copies of the frozen
training implementation; `provenance.json` records their hashes and the upstream
commit. Training, checkpointing, EMA, normalization, and rollout use the native
regression-policy code.

The learning objective can be reused for other tasks. The included geometric
router and dataset phase labels are specific to Tool-Hang. Selecting another
task's Hydra configuration does not replace those rules; see the adaptation
instructions below.

## Install and train Tool-Hang

Run from the repository root. Install the pinned Python 3.10 RoboMimic environment:

```bash
uv run scripts/install.py robomimic
```

Supply the original PH ABS state dataset, or obtain it with the repository's
data preparation command:

```bash
uv run --no-sync python scripts/download_robomimic.py \
  --task tool_hang --split ph --modality state
```

Start the original recipe with a new output directory:

```bash
CUDA_VISIBLE_DEVICES=0 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync python -m ht_regression.training.robomimic.train \
  experiment=robomimic/state/tool_hang/ph/mse_multistage6_overlap8 \
  dataset.path=/absolute/path/to/low_dim_abs.hdf5 \
  seed=41 training.evaluation_seed=41 \
  training.output_dir=/absolute/path/to/outputs/toolhang_multistage6_s41
```

The original dataset SHA256 is
`b19746f9e5495a55d064a49db1cdaf1e57fe034dad1ba9efcbc7e934d3f5bf92`.
Compare this hash when reproducing that experiment. It uses 196 training and
4 validation demonstrations, with split seed 42.

The inherited Tool-Hang recipe uses six separate 8-layer, 256-wide Transformers,
two observation frames, horizon 10, eight executed actions starting at index 1,
batch 256, AdamW, EMA, and 2,928,000 optimizer updates. Evaluation and checkpoint
saving happen every 18,300 updates, with 50 test episodes and at most 700 steps
per episode. Boundary overlap is eight states on either side of a transition.

Each process trains the full policy on one GPU. To run independent experiments
on another machine or GPU, select the device and use a distinct output directory.
Changing the training seed is optional. Data split and evaluation seeds can be
held fixed as in the command above.

## What each file does

| File | Responsibility |
| --- | --- |
| `../toolhang_multistage/policy.py` | Current-observation geometric router and six independent expert networks |
| `dataset.py` | Native training windows plus normalized phase memberships near demonstration boundaries |
| `policy.py` | Carries training memberships alongside the normal state condition |
| `objective.py` | Weighted sum of the individual experts' MSE losses; ordinary routed MSE during validation |

Training computes each expert's squared error before applying its membership
weight. It does not average predictions before computing MSE. For each sample,
membership weights sum to one, including when transition bands intersect.
Rollout uses only the current observed state to choose one expert. It does not
use demonstration phase labels, future observations, elapsed time, or simulator
state unavailable in the policy observation.

## Adapting to another task

Keep the Tool-Hang implementation unchanged as a reference. Put the new task's
router and dataset wrapper in a separate experiment package, then add a Hydra
configuration that inherits that task's native state-policy recipe.

1. **Define phase IDs from the actual observation layout.** Replace `stage_ids`
   with a function returning one integer phase per sample from its latest raw
   observation. The provided implementation assumes Tool-Hang's 53D state:
   object44, end-effector position3/quaternion4, and gripper2. Its object offsets,
   insertion/hanging indicators, and metric thresholds are task-specific.
2. **Use the same router for labels and rollout.** The dataset applies it to each
   demonstration independently. The policy applies it after undoing observation
   normalization. A learned router must similarly use available observations
   only; demonstration time or action labels cannot provide rollout switching.
3. **Update all stage-count assumptions together.** In `StageMSEPolicy`, update
   the observation checks, five additional expert copies, thresholds, and
   checkpoint extra-state validation. In the dataset wrapper, update observation
   key checks and the router import. In `boundary_weights`, replace the fixed
   six-column identity and phase range with the new number of stages. The
   MSE objective uses `len(policy.experts)`; the overlap policy wrapper has no
   fixed stage count.
4. **Inherit the new task's own recipe.** Keep its dataset split, normalization,
   action representation, horizon, observation history, optimizer schedule,
   batch size, evaluation horizon, and reset protocol. Set the policy, dataset,
   and objective Hydra targets to the new experiment classes. Do not copy the
   Tool-Hang training budget or geometry thresholds without checking that task.
5. **Check switching and recovery before a long run.** Verify all phases occur
   in training, observation normalization preserves phase IDs, overlap never
   crosses demonstration boundaries, gradients reach the intended experts, and
   rollout works without `expert_weights`. Test checkpoint/EMA reload and a
   short native simulator rollout with the new task's observation layout.

Transport-MH now has a separate [six-stage implementation](../transport_multistage/README.md)
with its own geometry and native Transport recipe.
Phase separation also increases model capacity and does not guarantee Gaussian
residuals; compare the new policy with the same task's native baseline.

## Checks

The unit tests cover routing, recovery, independent expert gradients, chunk
alignment, checkpoint reload, fullgraph compilation, overlap weights, and
inference without training metadata:

```bash
uv run --no-sync python -m unittest \
  experiments.toolhang_multistage.test_policy \
  experiments.toolhang_overlap.test_overlap -v
```

Set `TOOLHANG_DATASET=/absolute/path/to/low_dim_abs.hdf5` to additionally run the
real-data check of native windows, normalization, splits, and phase memberships.
Without this variable, only that integration test is skipped.

Machine-specific job managers, process-stopping launchers, and monitoring scripts
are intentionally not part of this branch. No dataset, checkpoint, or training
output is committed. Run training and evaluation from the repository root so
the `experiments.*` checkpoint classes are importable.
