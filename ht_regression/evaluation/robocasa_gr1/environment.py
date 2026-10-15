"""Single GR1 episode and observation histories; no model in simulator workers."""

import os
import random
import sys
from collections import deque

import numpy as np


def create_environment(task, seed):
    if sys.platform == "linux":
        backend = os.environ.setdefault("MUJOCO_GL", "egl")
        if backend in ("egl", "osmesa"):
            os.environ.setdefault("PYOPENGL_PLATFORM", backend)
    import gymnasium as gym
    import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401

    # RoboSuite creates its independent scene Generator in the constructor.
    # RoboCasa's reset(seed) only seeds global np.random, so it cannot replace
    # this constructor seed (or reproduce a scene created with seed=None).
    return gym.make(task, enable_render=True, seed=seed)


class Episode:
    def __init__(
        self,
        factory,
        task,
        seed,
        history_spec,
        language_key,
        max_steps,
        terminate_on_success,
    ):
        self.task, self.seed = task, seed
        self.history_spec, self.language_key = history_spec, language_key
        self.max_steps, self.terminate_on_success = max_steps, terminate_on_success
        self.steps, self.reward, self.success, self.done = 0, 0.0, False, False
        self.terminated = self.truncated = False
        self.env = None
        # The reference environment uses global NumPy RNG as well as reset(seed).
        # Isolate each slot, even when several simulators share a worker process.
        caller = np.random.get_state(), random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)
            self.env = factory(task, seed)
            observation, _ = self.env.reset(seed=seed)
            length = 1 - min(min(indices) for indices in history_spec.values())
            self.history = deque(maxlen=length)
            self._append(observation)
            self.rng = np.random.get_state(), random.getstate()
        except BaseException:
            self.close()
            raise
        finally:
            np.random.set_state(caller[0])
            random.setstate(caller[1])

    def _append(self, observation):
        snapshot = {
            key: np.array(observation[key], copy=True) for key in self.history_spec
        }
        snapshot[self.language_key] = observation[self.language_key]
        self.history.append(snapshot)

    def observation(self):
        result = {
            key: np.stack(
                [
                    self.history[max(0, len(self.history) - 1 + delta)][key]
                    for delta in indices
                ]
            )
            for key, indices in self.history_spec.items()
        }
        result[self.language_key] = self.history[-1][self.language_key]
        return result

    def advance(self, actions):
        if self.done:
            raise RuntimeError("Cannot step a completed episode.")
        if not actions:
            raise ValueError("An action chunk must contain at least one action group.")
        count = next(iter(actions.values())).shape[0]
        if count < 1 or any(
            value.ndim != 2 or len(value) != count for value in actions.values()
        ):
            raise ValueError("Every action group must have shape (chunk, dimension).")
        expected = set(self.env.action_space.spaces)
        if set(actions) != expected:
            raise ValueError(
                f"Action keys differ from environment: expected {sorted(expected)}."
            )
        caller = np.random.get_state(), random.getstate()
        try:
            np.random.set_state(self.rng[0])
            random.setstate(self.rng[1])
            for step in range(min(count, self.max_steps - self.steps)):
                # RoboCasa may consume/mutate its action mapping.
                native = {key: value[step].copy() for key, value in actions.items()}
                for key, value in native.items():
                    if (
                        value.shape != self.env.action_space.spaces[key].shape
                        or not np.isfinite(value).all()
                    ):
                        raise ValueError(f"Invalid physical action for {key}.")
                observation, reward, terminated, truncated, info = self.env.step(native)
                if "success" not in info:
                    raise ValueError("GR1 environment must report info['success'].")
                success = np.asarray(info["success"])
                if (
                    success.size == 0
                    or success.dtype.kind not in "biuf"
                    or not np.isfinite(success).all()
                    or not ((success == 0) | (success == 1)).all()
                ):
                    raise ValueError(
                        "info['success'] must contain boolean or 0/1 values."
                    )
                if not np.isfinite(reward):
                    raise ValueError("Environment reward must be finite.")
                self.steps += 1
                self.reward += float(reward)
                self.success |= bool(success.any())
                self.terminated, self.truncated = bool(terminated), bool(truncated)
                self._append(observation)
                self.done = (
                    self.terminated
                    or self.truncated
                    or self.steps >= self.max_steps
                    or (self.terminate_on_success and self.success)
                )
                if self.done:
                    break
            self.rng = np.random.get_state(), random.getstate()
        finally:
            np.random.set_state(caller[0])
            random.setstate(caller[1])
        return self.observation(), self.done, self.record()

    def record(self):
        return dict(
            task=self.task,
            seed=self.seed,
            success=self.success,
            steps=self.steps,
            reward=self.reward,
            terminated=self.terminated,
            truncated=self.truncated,
            time_limit=self.steps >= self.max_steps,
        )

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
