"""Spawned CPU simulator groups; policy inference stays in the caller process."""

import copy
import multiprocessing as mp
import random
import traceback
from contextlib import ExitStack

import numpy as np
import torch

from .observations import stack_observations


def _rng_state(include_torch=False):
    state = (np.random.get_state(), random.getstate())
    return (*state, torch.get_rng_state().numpy()) if include_torch else state


def _restore_rng(state):
    np.random.set_state(state[0])
    random.setstate(state[1])
    if len(state) == 3:
        torch.set_rng_state(torch.from_numpy(state[2]))


def _worker(connection, runner, episodes, active_count, action_dim, report_rng):
    """Only NumPy arrays cross the pipe; workers never initialize CUDA."""
    try:
        torch.set_num_threads(1)
        with ExitStack() as resources:
            while True:
                command, value = connection.recv()
                if command == "init":
                    _restore_rng(value)
                    slots = runner._create_slots(
                        episodes, active_count, resources, action_dim
                    )
                    connection.send(
                        (
                            "ok",
                            (
                                runner._slot_observations(slots),
                                slots[0]["action_dim"],
                                _rng_state(True),
                            ),
                        )
                    )
                elif command == "step":
                    runner._advance_slots(slots, *value)
                    connection.send(
                        (
                            "ok",
                            (
                                runner._slot_observations(slots),
                                all(s["done"] for s in slots),
                                _rng_state() if report_rng else None,
                            ),
                        )
                    )
                elif command == "finish":
                    connection.send(("ok", runner._finish_slots(slots, active_count)))
                elif command == "close":
                    break
                else:
                    raise ValueError(f"Unknown simulator command: {command}")
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send(("error", traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


class ParallelEnvironments:
    """Keep inference order and RNG consumption equal to serial environment groups.

    Construction/reset is deliberately ordered, forwarding CPU RNG state between
    groups, including Torch state in case environment construction consumes it.
    Stepping is concurrent. Each slot retains its NumPy/Python RNG as in the serial
    runner, and the final slot's state is returned to the parent after each chunk.
    MuJoCo stepping must not consume Torch RNG (the standard RoboMimic path does not).
    Custom env factories must be spawn-pickleable module-level callables.
    """

    def __init__(self, runner, episodes, active_count, action_dim, resources):
        context = mp.get_context("spawn")
        self.connections, self.processes, self.slices = [], [], []
        self.timeout = runner.worker_timeout
        resources.callback(self.close)
        # Progress callbacks may close over training state; do not pickle those.
        worker_runner = copy.copy(runner)
        worker_runner.progress = None
        worker_runner.episodes = []
        state = _rng_state(True)
        count = min(runner.num_workers, len(episodes))
        groups = np.array_split(np.arange(len(episodes)), count)
        for index, group in enumerate(groups):
            start, end = int(group[0]), int(group[-1]) + 1
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker,
                args=(
                    child,
                    worker_runner,
                    episodes[start:end],
                    max(0, min(active_count, end) - start),
                    action_dim,
                    index == count - 1,
                ),
                daemon=True,
            )
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            self.connections.append(parent)
            self.processes.append(process)
            self.slices.append(slice(start, end))
        observations = []
        for index, connection in enumerate(self.connections):
            connection.send(("init", state))
            obs, dimension, state = self._receive(index)
            if action_dim is not None and action_dim != dimension:
                raise ValueError("Worker action dimensions disagree.")
            action_dim = dimension
            observations.append(obs)
        _restore_rng(state)
        self.observations = stack_observations(observations, concatenate=True)
        self.action_dim = action_dim

    def _receive(self, index):
        connection = self.connections[index]
        if not connection.poll(self.timeout):
            raise TimeoutError(
                f"Simulator worker {index} did not respond within {self.timeout}s."
            )
        try:
            status, value = connection.recv()
        except (EOFError, OSError) as exc:
            raise RuntimeError(
                f"Simulator worker {index} exited unexpectedly."
            ) from exc
        if status != "ok":
            raise RuntimeError(f"Simulator worker {index} failed:\n{value}")
        return value

    def step(self, native, model_actions):
        for connection, selection in zip(self.connections, self.slices):
            connection.send(("step", (native[selection], model_actions[selection])))
        responses = [self._receive(i) for i in range(len(self.connections))]
        _restore_rng(responses[-1][2])
        return stack_observations([r[0] for r in responses], concatenate=True), all(
            r[1] for r in responses
        )

    def finish(self):
        for connection in self.connections:
            connection.send(("finish", None))
        return [
            record for i in range(len(self.connections)) for record in self._receive(i)
        ]

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        for connection in self.connections:
            connection.close()
