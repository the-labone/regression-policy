# Transport-MH six-stage overlap MSE

This extends the Tool-Hang experiment to the original robosuite 1.2
TwoArmTransport task with opposed Panda arms and the default table/bin geometry.
It uses the original MH ABS state data, with 59D observations and 20D model-space
actions (14D native actions for two arms). No demonstrations are filtered or edited.
The verified original ABS dataset SHA256 is
`2034f404d1e9dd04514c443f9b2fb2bda99f320b46acd9ffc9983fdbec0f9d95`.

From the repository root, install the pinned environment with
`uv run scripts/install.py robomimic`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --no-sync python -m ht_regression.training.robomimic.train \
  experiment=robomimic/state/transport/mh/mse_multistage6_overlap8 \
  dataset.path=/absolute/path/to/transport/mh/low_dim_abs.hdf5 \
  training.output_dir=/absolute/path/to/outputs/transport_mh_multistage6_s41
```

The recipe inherits Transport-MH's native Transformer settings: seed 41,
294 train / 6 validation demonstrations (split seed 42), two observation frames,
horizon 10, eight executed actions starting at index 1, batch 256, AdamW, EMA,
and **5,976,000 updates**. Validation runs every 747 updates and rollout every
37,350 updates: 6 train + 50 test episodes, at most 700 simulation steps,
28 environments, 8 simulator workers, test seeds starting at 4,200,000.

Six independently parameterized, identically initialized native Transformers
model lid removal, payload pickup, donor transport, handover, receiver transport,
and target placement. Trash clearing can happen concurrently; each expert
predicts both arms jointly. `policy.py` documents the geometric thresholds.
These are geometric phase approximations, not ground-truth grasp labels.

Training uses `../toolhang_overlap/dataset.py::boundary_weights` for an 8-frame
band on each side of each demonstration-local transition. Weights sum to one;
`../toolhang_overlap/objective.py` averages weighted individual expert MSEs.
Validation and rollout use the current observation alone to select one expert;
training metadata never enters the router or Transformer. Routing can reverse
for recovery. The original Tool-Hang runtime modules are unchanged.

More model capacity and geometric partitioning do not guarantee either improved
success or Gaussian residuals. Measure both before making a rebuttal claim.

```bash
TRANSPORT_DATASET=/absolute/path/to/transport/mh/low_dim_abs.hdf5 \
uv run --no-sync python -m unittest experiments.transport_multistage.test_transport -v
```

Run from this checkout for training and checkpoint loading so `experiments.*`
is importable. Use a distinct output directory for every independent job.
