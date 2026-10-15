"""Spawned GR1 environments with synchronous batches and concurrent chunk stepping."""

import multiprocessing as mp
import os
import traceback
from contextlib import ExitStack

import numpy as np

from .environment import Episode

WORKER_HASH_SEED = "0"


def _start_worker(process):
    # RoboCasa uses hash(class_name) for task seeds and list(set(...)) for object
    # lists. RNG seeds do not control these. Set the hash seed BEFORE the
    # spawned interpreter starts, then restore the caller's environment.
    previous = os.environ.get("PYTHONHASHSEED")
    os.environ["PYTHONHASHSEED"] = WORKER_HASH_SEED
    try:
        process.start()
    finally:
        if previous is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = previous


def _worker(connection, specs):
    try:
        import torch

        torch.set_num_threads(1)
        with ExitStack() as resources:
            episodes = []
            for spec in specs:
                episode = Episode(**spec)
                resources.callback(episode.close)
                episodes.append(episode)
            connection.send((True, [episode.observation() for episode in episodes]))
            while True:
                command, values = connection.recv()
                if command == "close":
                    break
                if command != "step":
                    raise ValueError(f"Unknown command {command}.")
                connection.send(
                    (
                        True,
                        [
                            None if action is None else episode.advance(action)
                            for episode, action in zip(episodes, values)
                        ],
                    )
                )
    except EOFError:
        pass
    except BaseException:
        try:
            connection.send((False, traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


class EnvironmentBatch:
    """Same slot order in serial and parallel modes; workers never receive a policy."""

    def __init__(self, specs, num_workers, timeout):
        self.connections, self.processes, self.groups, self.episodes = [], [], [], []
        self.timeout = timeout
        try:
            if num_workers == 0:
                for spec in specs:
                    self.episodes.append(Episode(**spec))
                self.observations = [episode.observation() for episode in self.episodes]
            else:
                context = mp.get_context("spawn")
                for group in np.array_split(
                    np.arange(len(specs)), min(num_workers, len(specs))
                ):
                    indices = group.tolist()
                    parent, child = context.Pipe()
                    process = context.Process(
                        target=_worker,
                        args=(child, [specs[i] for i in indices]),
                        daemon=True,
                    )
                    try:
                        _start_worker(process)
                    except BaseException:
                        parent.close()
                        child.close()
                        raise
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
                    self.groups.append(indices)
                self.observations = [
                    item for i in range(len(self.groups)) for item in self._receive(i)
                ]
        except BaseException:
            self.close()
            raise

    def _receive(self, index):
        connection = self.connections[index]
        if not connection.poll(self.timeout):
            raise TimeoutError(f"GR1 worker {index} exceeded {self.timeout}s.")
        try:
            ok, result = connection.recv()
        except (EOFError, OSError) as error:
            raise RuntimeError(f"GR1 worker {index} exited unexpectedly.") from error
        if not ok:
            raise RuntimeError(f"GR1 worker failed:\n{result}")
        return result

    def step(self, actions):
        if self.episodes:
            return [
                None if action is None else episode.advance(action)
                for episode, action in zip(self.episodes, actions)
            ]
        for connection, indices in zip(self.connections, self.groups):
            connection.send(("step", [actions[i] for i in indices]))
        return [item for i in range(len(self.groups)) for item in self._receive(i)]

    def close(self):
        for connection in self.connections:
            try:
                connection.send(("close", None))
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        for connection in self.connections:
            connection.close()
        for episode in self.episodes:
            episode.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
