# Transport-MH six-stage MSE

Extend the existing six-expert overlap MSE experiment to original Transport-MH
state demonstrations. This continues the user's authorized experiment series;
the geometry thresholds below are implementation choices, not user selections.

Use the native Transport-MH Transformer recipe, original ABS data, seed 41,
5,976,000 optimizer steps, native episode split, 2 observation frames, horizon
10, 8 executed actions, and 700-step evaluation (6 train + 50 test episodes).
Do not change the simulator, demonstration observations/actions, normalization,
optimizer, scheduling, or the finished Tool-Hang implementation.

## Approach

Reuse six independent native Transformer experts, initialized identically,
and episode-local overlap of 8 frames on each side of observed stage changes.
Keep a normalized sum of expert MSEs per original window. Validation and
rollout use a hard observation-only router. Separate arm-specific routers
would change the policy architecture and treatment of simultaneous actions;
a learned gate would introduce an additional training intervention. Neither
is needed for this extension.

The router is specific to robosuite 1.2 TwoArmTransport, two opposed Panda arms,
default table/bin layout, object41 + robot0 pose/gripper9 + robot1 pose/gripper9.
Stages: lid removal (including concurrent trash clearing), payload pickup,
donor transport, handover, receiver transport, target placement.
From the latest raw observation: source region y < -0.25 m; low payload
height < target-bin base z + 0.12 m; lid still covering if source-region,
lid/payload XY distance < 0.22 m and lid > payload z + 0.08 m. Donor transport
has payload y < -0.12 m. Receiver transport requires donor distance > 0.14 m
and receiver distance < 0.16 m, or payload y > 0.18 m. Target placement
has target/payload XY distance < 0.18 m or payload-in-target flag. Apply
source pickup and covering checks before later-stage cases. No elapsed time,
future action, hidden simulator information, latch, or demonstration ID is used.

All 294 training demonstrations cover every stage. Preliminary training-only
frame fractions: 25.41%, 28.44%, 8.39%, 25.22%, 5.94%, 6.61%; median 5 switches.
This is an interpretable geometric partition, not contact ground truth.
Geometric phases do not guarantee Gaussian residuals. Increased model capacity
remains a confound for causal claims. Report actual success and residuals later.

## Verification

Test routing, recovery and latest-frame dependence; unchanged native windows,
normalization and episode split; weighted-loss gradients and metadata isolation;
checkpoint router compatibility; native configuration parity. Freeze code and
record dataset/environment hashes before training. Smoke-test compiled GPU
updates, native simulator rollout, and EMA checkpoint reload before full launch.
