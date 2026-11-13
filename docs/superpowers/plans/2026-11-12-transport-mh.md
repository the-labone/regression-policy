# Transport-MH Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run original Transport-MH with six-stage overlap MSE.
**Architecture:** Task-specific observation router and dataset labels; reuse the
existing six-stage boundary weighting and MSE objective without modifying them.
**Tech Stack:** PyTorch 2.4, Hydra, original regression-policy and robosuite 1.2.
**Spec:** `docs/superpowers/specs/2026-11-12-transport-mh-design.md`

## Global Constraints

Preserve native Transport-MH training/evaluation settings, original ABS data and
all Tool-Hang code. Six independent Transformers, overlap radius 8, seed 41.

## Review Focus

- Observation order and unnormalization must match simulator rollout.
- No phase labels or future boundary metadata may enter inference.
- Both arms' encoded actions and gripper positions must remain aligned.
- Recovery states must permit routing reversals and no irreversible latch.
- Changed geometry/router must be rejected by dataset/checkpoint validation.

### Task 1: Transport router, data and recipe

Files: `experiments/transport_multistage/{policy.py,dataset.py,test_transport.py,README.md}`;
`configs/experiment/robomimic/state/transport/mh/mse_multistage6_overlap8.yaml`.
Interface: `stage_ids(observation, thresholds=None) -> Tensor[B]`,
`TransportStagePolicy`, `TransportOverlapDataset`.

- [ ] Write failing behavior tests for geometric states, weighted gradients,
  inference metadata isolation, checkpoint compatibility and native data parity.
- [ ] Run `python -m unittest experiments.transport_multistage.test_transport`;
  expected failure because the Transport implementation is missing.
- [ ] Implement router, six-expert policy, dataset labels and inherited config.
- [ ] Run new and existing Tool-Hang tests; expected all applicable tests pass.
- [ ] Commit the task and record test evidence.

### Task 2: Frozen preflight and launch

Files: local run's `prepare.py`, provenance, frozen `code`, smoke report, launcher.
Consumes Task 1's classes and Hydra config; produces live native training run.

- [ ] Compare resolved native and new configs allowing only name and new
  dataset/policy/objective targets and overlap metadata; verify all base source
  and installed package hashes/versions. Audit training/validation stage coverage.
- [ ] Smoke test three compiled GPU updates, native simulator rollout and EMA
  checkpoint load; expected finite losses/gradients, zero simulator failures.
- [ ] Review the change with a fresh reviewer; address material findings.
- [ ] Freeze reviewed source, launch the full native job, confirm metrics advance
  and report actual step and full budget. Publish the extension on the existing
  user-requested experiment branch after checks pass.
