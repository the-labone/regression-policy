"""Batched RoboMimic state rollouts following Diffusion Policy's protocol.

Reference: real-stanford/diffusion_policy (MIT), revision
5ba07ac6661db573af695b419a7947ecb704690f, robomimic_lowdim_runner.py.
See ../../policies/LICENSE.diffusion_policy.
"""

import copy
import json
import logging
import os
import random
import time
from collections import deque
from contextlib import ExitStack, contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import h5py
import numpy as np
import torch

from ht_regression.data.robomimic.actions import check_action_space, decode_actions
from ht_regression.data.robomimic.dataset import demo_names
from ht_regression.data.robomimic.metadata import (
    REPAIRED_OBS_ATTR,
    controller_env_metadata,
    read_env_metadata,
    text,
)
from ht_regression.data.robomimic.observations import encode_observation

from ..artifacts import write_json
from .observations import observation_to_device, stack_observations


def _versions():
    result = {"torch": str(torch.__version__)}
    for package in (
        "robosuite",
        "robomimic",
        "mujoco",
        "mujoco-py",
        "free-mujoco-py",
        "diffusers",
        "numpy",
    ):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            pass
    return result


def _create_env(env_meta, obs_keys, *, seed, render_offscreen, camera_keys=()):
    # No simulator imports at package import time or when inspecting a dataset.
    from .runtime import configure_runtime

    configure_runtime()
    # free-mujoco-py overwrites this variable on import. Preserve the launcher's
    # library search path for subsequently spawned simulators / video encoders.
    library_path = os.environ.get("LD_LIBRARY_PATH")
    try:
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        import robosuite
    finally:
        if library_path is None:
            os.environ.pop("LD_LIBRARY_PATH", None)
        else:
            os.environ["LD_LIBRARY_PATH"] = library_path

    expected = env_meta["env_version"]
    if robosuite.__version__ != expected:
        raise ValueError(
            f"Dataset requires robosuite {expected}; installed {robosuite.__version__}."
        )
    modalities = {"low_dim": list(obs_keys)}
    if camera_keys:
        modalities["rgb"] = list(camera_keys)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modalities)
    env_meta = copy.deepcopy(env_meta)
    if tuple(int(part) for part in robosuite.__version__.split(".")[:2]) >= (1, 5):
        env_meta["env_kwargs"]["seed"] = seed
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=render_offscreen or bool(camera_keys),
        use_image_obs=bool(camera_keys),
    )
    # robomimic 0.2 (used by the DP release) predates EnvRobosuite.version.
    # The imported simulator version was verified above, before construction.
    if not hasattr(env, "version"):
        env.version = robosuite.__version__
    return env


def _close_env(env):
    close = getattr(env, "close", None)
    if not callable(close):
        close = getattr(getattr(env, "env", None), "close", None)
    if callable(close):
        close()


def _is_mujoco_instability(error):
    """Recognize the legacy simulator's QACC warning, not arbitrary worker errors."""
    return (
        type(error).__module__ == "mujoco_py.builder"
        and type(error).__name__ == "MujocoException"
        and str(error).startswith("Got MuJoCo Warning: Nan, Inf or huge value in QACC")
    )


@contextmanager
def _random_state(seed, device):
    """Evaluation must not perturb the caller's training RNG streams."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            np.random.seed(seed)
            random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            for index in cuda_devices:
                torch.cuda.default_generators[index].manual_seed(seed)
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


class RobomimicStateRunner:
    """Serial or process-parallel environments with batched inference and DP history.

    ``policy.predict_action({'obs': (B, To, Do)})['action']`` must return an
    unnormalized execution chunk (B, Ta, Da). This runner neither normalizes nor
    slices it again. All requested episodes contribute to the reported metrics.
    ``n_envs`` controls inference batch size; ``num_workers`` controls CPU simulator
    processes independently. Zero workers selects serial stepping.
    """

    def __init__(
        self,
        dataset_path,
        output_dir,
        obs_keys,
        *,
        action_space,
        n_obs_steps=2,
        n_action_steps=8,
        n_train=10,
        n_test=22,
        train_start_idx=0,
        test_start_seed=10000,
        max_steps=400,
        n_envs=1,
        num_workers=0,
        worker_timeout=300.0,
        seed=0,
        n_train_vis=0,
        n_test_vis=0,
        render_hw=(256, 256),
        render_camera_name="agentview",
        fps=10,
        crf=22,
        save_traces=False,
        train_reset="state",
        repair_observations=None,
        source_robosuite_version=None,
        env_config=None,
        env_factory=None,
        progress=None,
    ):
        self.dataset_path = (
            Path(dataset_path).expanduser().resolve()
            if dataset_path is not None
            else None
        )
        self.env_config = (
            Path(env_config).expanduser().resolve() if env_config is not None else None
        )
        if (self.dataset_path is None) == (self.env_config is None):
            raise ValueError("Supply either env_config or dataset_path.")
        if self.env_config is not None and n_train:
            raise ValueError(
                "Train initial-state rollouts require a dataset; use n_train=0 with env_config."
            )
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.obs_keys = tuple(obs_keys)
        if not self.obs_keys or len(set(self.obs_keys)) != len(self.obs_keys):
            raise ValueError("obs_keys must be nonempty and unique.")
        self.action_space = check_action_space(action_space)
        for name, value in (
            ("n_obs_steps", n_obs_steps),
            ("n_action_steps", n_action_steps),
            ("max_steps", max_steps),
            ("fps", fps),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name, value in (
            ("n_train", n_train),
            ("n_test", n_test),
            ("train_start_idx", train_start_idx),
            ("test_start_seed", test_start_seed),
            ("seed", seed),
            ("n_train_vis", n_train_vis),
            ("n_test_vis", n_test_vis),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if n_train + n_test == 0:
            raise ValueError("Request at least one evaluation episode.")
        if seed >= 2**32 or test_start_seed + n_test > 2**32:
            raise ValueError("Seeds must fit in NumPy's uint32 seed range.")
        n_envs = n_train + n_test if n_envs is None else n_envs
        if type(n_envs) is not int or n_envs < 1:
            raise ValueError("n_envs must be a positive integer.")
        if type(num_workers) is not int or num_workers < 0:
            raise ValueError("num_workers must be a nonnegative integer.")
        if not np.isfinite(worker_timeout) or worker_timeout <= 0:
            raise ValueError("worker_timeout must be finite and positive.")
        self.num_workers, self.worker_timeout = num_workers, worker_timeout
        if train_reset not in ("state", "model"):
            raise ValueError(
                "train_reset must be 'state' (DP) or 'model' (XML plus state)."
            )
        if len(render_hw) != 2 or any(
            type(x) is not int or x < 2 or x % 2 for x in render_hw
        ):
            raise ValueError("render_hw must contain two positive even dimensions.")
        if type(crf) is not int or not 0 <= crf <= 51:
            raise ValueError("crf must be between 0 and 51.")
        self.n_obs_steps, self.n_action_steps = n_obs_steps, n_action_steps
        self.n_train, self.n_test = n_train, n_test
        self.train_start_idx, self.test_start_seed = train_start_idx, test_start_seed
        self.max_steps, self.n_envs, self.seed = max_steps, n_envs, seed
        self.n_train_vis, self.n_test_vis = n_train_vis, n_test_vis
        self.render_hw, self.render_camera_name = tuple(render_hw), render_camera_name
        self.fps, self.crf, self.save_traces = fps, crf, save_traces
        self.train_reset = train_reset
        self.env_factory, self.progress = env_factory or _create_env, progress
        self.episodes = []
        if self.env_config is not None:
            config = json.loads(self.env_config.read_text())
            meta = config["env_meta"]
            marked = config["repair_observations"]
            dimensions = config["obs_dims"]
            if any(
                type(dimensions.get(key)) is not int or dimensions[key] < 1
                for key in self.obs_keys
            ):
                raise ValueError("Environment config lacks dimensions for obs_keys.")
            self.obs_dim = sum(dimensions[key] for key in self.obs_keys)
        else:
            meta, marked = self._read_dataset()
        if meta.get("type") != 1:
            raise ValueError("Only RoboMimic robosuite environments are supported.")
        recorded_version = meta.get("env_version")
        if (
            recorded_version
            and source_robosuite_version
            and recorded_version != source_robosuite_version
        ):
            raise ValueError(
                "source_robosuite_version conflicts with the environment's env_version."
            )
        expected_version = recorded_version or source_robosuite_version
        if not expected_version:
            raise ValueError(
                "Unversioned env_args: supply source_robosuite_version from dataset provenance."
            )
        meta["env_version"] = expected_version
        self.env_meta = controller_env_metadata(meta, self.action_space)
        self.repair_observations = (
            marked if repair_observations is None else bool(repair_observations)
        )
        for index in range(n_test):
            self.episodes.append(
                {
                    "split": "test",
                    "id": test_start_seed + index,
                    "initial": None,
                    "video": index < n_test_vis,
                }
            )

    def _read_dataset(self):
        """Retain demonstration resets for training-time and optional dataset evaluation."""
        with h5py.File(self.dataset_path, "r") as file:
            data = file["data"]
            meta = read_env_metadata(data)
            marked = bool(data.attrs.get(REPAIRED_OBS_ATTR, False))
            reference = data[demo_names(data)[0]]["obs"]
            self.obs_dim = 0
            for key in self.obs_keys:
                if reference[key].ndim != 2:
                    raise ValueError(f"Expected 2D state observations for {key!r}.")
                self.obs_dim += reference[key].shape[-1]
            for index in range(
                self.train_start_idx, self.train_start_idx + self.n_train
            ):
                name = f"demo_{index}"
                demo = data[name]
                initial = {"states": demo["states"][0].copy()}
                if not np.isfinite(initial["states"]).all():
                    raise ValueError(f"Non-finite initial state in {name}.")
                if self.train_reset == "model":
                    initial["model"] = text(demo.attrs["model_file"])
                    if "ep_meta" in demo.attrs:
                        initial["ep_meta"] = text(demo.attrs["ep_meta"])
                self.episodes.append(
                    {
                        "split": "train",
                        "id": index,
                        "initial": initial,
                        "video": index - self.train_start_idx < self.n_train_vis,
                    }
                )
        return meta, marked

    @classmethod
    def from_dp_checkpoint(cls, loaded, *, dataset_path=None, output_dir, **overrides):
        """Restore the original runner protocol, with explicit local overrides."""
        config = copy.deepcopy(loaded.config.get("task", {}).get("env_runner", {}))
        if (
            config.pop("_target_", None)
            != "diffusion_policy.env_runner.robomimic_lowdim_runner.RobomimicLowdimRunner"
        ):
            raise ValueError("Checkpoint does not configure a RoboMimic lowdim runner.")
        if config.pop("n_latency_steps", 0) or config.pop("past_action", False):
            raise ValueError("Latency and past-action conditioning are not supported.")
        config.pop("tqdm_interval_sec", None)
        config.pop("abs_action", None)
        if loaded.action_space is None or loaded.obs_keys is None:
            raise ValueError(
                "Checkpoint must declare action_space and observation ordering."
            )
        config.update(action_space=loaded.action_space, obs_keys=loaded.obs_keys)
        if dataset_path is not None or overrides.get("env_config") is not None:
            config["dataset_path"] = dataset_path
        # Missing n_envs in upstream means all episodes in a single batch.
        config.setdefault("n_envs", None)
        config.update(overrides)
        return cls(output_dir=output_dir, **config)

    def _encode(self, observation):
        return encode_observation(
            observation,
            self.obs_keys,
            self.obs_dim,
            self.env_meta,
            repair_observations=self.repair_observations,
        )

    def _reset_env(self, env, episode):
        if episode["initial"] is not None:
            return env.reset_to(copy.deepcopy(episode["initial"]))
        else:
            seed = episode["id"]
            np.random.seed(seed)
            random.seed(seed)
            # robosuite 1.5 also uses a Generator shared with placement samplers.
            rng = getattr(getattr(env, "env", None), "rng", None)
            if isinstance(rng, np.random.Generator):
                rng.bit_generator.state = type(rng.bit_generator)(seed).state
            return env.reset()

    def _reset(self, env, episode):
        self._reset_env(env, episode)
        # DP's lowdim wrapper queries again; its image wrapper uses the returned
        # observation instead. In legacy robosuite, querying advances sensor time.
        return self._encode(env.get_observation())

    def _video_writer(self, episode):
        import imageio.v2 as imageio

        path = self.output_dir / "media" / f"{episode['split']}_{episode['id']}.mp4"
        path.parent.mkdir(exist_ok=True)
        writer = imageio.get_writer(
            str(path),
            fps=self.fps,
            codec="libx264",
            macro_block_size=1,
            ffmpeg_params=["-crf", str(self.crf), "-threads", "1"],
        )
        return writer, str(path.relative_to(self.output_dir))

    def _render(self, env, writer):
        frame = env.render(
            mode="rgb_array",
            height=self.render_hw[0],
            width=self.render_hw[1],
            camera_name=self.render_camera_name,
        )
        writer.append_data(np.asarray(frame, dtype=np.uint8))

    def _create_slots(self, episodes, active_count, resources, policy_action_dim):
        slots = []
        for index, episode in enumerate(episodes):
            np.random.seed(episode["id"])
            video = episode["video"] and index < active_count
            env = self.env_factory(
                copy.deepcopy(self.env_meta),
                self.obs_keys,
                seed=episode["id"],
                render_offscreen=video,
            )
            resources.callback(_close_env, env)
            if env.version != self.env_meta["env_version"]:
                raise ValueError(
                    f"Environment version {env.version} does not match {self.env_meta['env_version']}."
                )
            if env.action_dimension not in (7, 14):
                raise ValueError("Expected native 7D/14D OSC_POSE controller actions.")
            action_dim = (
                env.action_dimension // 7 * (10 if self.action_space == "abs" else 7)
            )
            if policy_action_dim is not None and policy_action_dim != action_dim:
                raise ValueError(
                    "Policy action dimension does not match the environment/controller."
                )
            obs = self._reset(env, episode)
            writer, video_path = self._video_writer(episode) if video else (None, None)
            if writer is not None:
                resources.callback(writer.close)
                self._render(env, writer)
            slots.append(
                dict(
                    env=env,
                    episode=episode,
                    history=deque(
                        [copy.deepcopy(obs) for _ in range(self.n_obs_steps)],
                        maxlen=self.n_obs_steps,
                    ),
                    action_dim=action_dim,
                    rewards=[],
                    successes=[],
                    done=False,
                    terminated=False,
                    writer=writer,
                    video_path=video_path,
                    numpy_state=np.random.get_state(),
                    python_state=random.getstate(),
                    trace_obs=[obs] if self.save_traces else [],
                    trace_actions=[],
                    trace_model_actions=[],
                )
            )
        return slots

    def _advance_slots(self, slots, native, model_actions):
        render_stride = max(
            int(self.env_meta["env_kwargs"].get("control_freq", 20)) // self.fps, 1
        )
        for index, slot in enumerate(slots):
            np.random.set_state(slot["numpy_state"])
            random.setstate(slot["python_state"])
            for step, command in enumerate(native[index]):
                if slot["done"]:
                    break
                try:
                    raw, reward, done, _ = slot["env"].step(command)
                except Exception as error:
                    if not _is_mujoco_instability(error):
                        raise
                    # Retire this episode using its last finite observation. Do
                    # not retry it or step/render the corrupted simulation again.
                    slot["done"] = True
                    slot["simulation_error"] = str(error)
                    logging.getLogger(__name__).warning(
                        "Counting %s episode %s as failed after %d steps: %s",
                        slot["episode"]["split"],
                        slot["episode"]["id"],
                        len(slot["rewards"]),
                        error,
                    )
                    break
                if not np.isfinite(reward):
                    raise ValueError("Environment returned a non-finite reward.")
                observation = self._encode(raw)
                slot["history"].append(observation)
                slot["rewards"].append(float(reward))
                slot["successes"].append(bool(slot["env"].is_success()["task"]))
                slot["terminated"] = bool(done)
                slot["done"] = bool(done) or len(slot["rewards"]) >= self.max_steps
                if self.save_traces:
                    slot["trace_obs"].append(observation)
                    slot["trace_actions"].append(command.copy())
                    slot["trace_model_actions"].append(
                        model_actions[index, step].copy()
                    )
                if (
                    slot["writer"] is not None
                    and len(slot["rewards"]) % render_stride == 0
                ):
                    self._render(slot["env"], slot["writer"])
            slot["numpy_state"], slot["python_state"] = (
                np.random.get_state(),
                random.getstate(),
            )

    def _finish_slots(self, slots, active_count):
        records = []
        for slot in slots[:active_count]:
            episode = slot["episode"]
            failed = "simulation_error" in slot
            record = dict(
                split=episode["split"],
                id=episode["id"],
                steps=len(slot["rewards"]),
                max_reward=0.0 if failed else max(slot["rewards"]),
                success=False if failed else any(slot["successes"]),
                terminated=slot["terminated"],
                truncated=not slot["terminated"],
                video_path=slot["video_path"],
            )
            if failed:
                record["simulation_error"] = slot["simulation_error"]
            if self.save_traces:
                path = (
                    self.output_dir
                    / "traces"
                    / f"{episode['split']}_{episode['id']}.npz"
                )
                path.parent.mkdir(exist_ok=True)
                observations = stack_observations(slot["trace_obs"])
                trace = (
                    {f"obs/{key}": value for key, value in observations.items()}
                    if isinstance(observations, dict)
                    else {"obs": observations}
                )
                np.savez_compressed(
                    path,
                    **trace,
                    actions=np.asarray(slot["trace_actions"]).reshape(
                        -1, slot["env"].action_dimension
                    ),
                    model_actions=np.asarray(slot["trace_model_actions"]).reshape(
                        -1, slot["action_dim"]
                    ),
                    rewards=np.array(slot["rewards"]),
                    successes=np.array(slot["successes"]),
                )
                record["trace_path"] = str(path.relative_to(self.output_dir))
            records.append(record)
        return records

    @staticmethod
    def _slot_observations(slots):
        return stack_observations(
            [stack_observations(slot["history"]) for slot in slots]
        )

    def _run_batch(self, policy, episodes, active_count, resources):
        from .parallel import ParallelEnvironments

        setup_start = time.perf_counter()
        model_policy = getattr(policy, "policy", policy)
        action_dim = getattr(model_policy, "action_dim", None)
        parallel = None
        if self.num_workers:
            parallel = ParallelEnvironments(
                self, episodes, active_count, action_dim, resources
            )
            histories, action_dim = parallel.observations, parallel.action_dim
        else:
            slots = self._create_slots(episodes, active_count, resources, action_dim)
            histories, action_dim = (
                self._slot_observations(slots),
                slots[0]["action_dim"],
            )
        policy.reset()
        self.timings["setup_seconds"] += time.perf_counter() - setup_start
        done = False
        while not done:
            start = time.perf_counter()
            obs = observation_to_device(histories, policy.device, policy.dtype)
            with torch.no_grad():
                action = policy.predict_action({"obs": obs})["action"]
            expected = (len(episodes), self.n_action_steps, action_dim)
            if not isinstance(action, torch.Tensor) or tuple(action.shape) != expected:
                raise ValueError(f"Policy execution action must have shape {expected}.")
            model_actions = (
                action.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
            native = decode_actions(model_actions, self.action_space)
            self.timings["policy_seconds"] += time.perf_counter() - start
            self.timings["policy_calls"] += 1
            start = time.perf_counter()
            if parallel is not None:
                histories, done = parallel.step(native, model_actions)
            else:
                self._advance_slots(slots, native, model_actions)
                histories = self._slot_observations(slots)
                done = all(slot["done"] for slot in slots)
            self.timings["environment_seconds"] += time.perf_counter() - start
        start = time.perf_counter()
        records = (
            parallel.finish()
            if parallel is not None
            else self._finish_slots(slots, active_count)
        )
        for record in records:
            if self.progress is not None:
                self.progress(record.copy())
        self.timings["finalize_seconds"] += time.perf_counter() - start
        return records

    def run(self, policy):
        """Write metrics.json, episodes.json and config.json; return DP-style metrics."""
        # A composed pipeline exposes the model's metadata through .policy.
        model_policy = getattr(policy, "policy", policy)
        encoder = getattr(model_policy, "obs_encoder", None)
        if getattr(encoder, "camera_shapes", None) != getattr(
            self, "camera_shapes", None
        ):
            raise ValueError("Policy and runner camera modalities disagree.")
        if (
            getattr(encoder, "proprio_keys", None) is not None
            and encoder.proprio_keys != self.obs_keys
        ):
            raise ValueError("Policy and runner proprioception ordering disagree.")
        obs_dim = getattr(
            encoder, "proprio_dim", getattr(model_policy, "obs_dim", self.obs_dim)
        )
        if obs_dim != self.obs_dim:
            raise ValueError("Policy and runner disagree on obs_dim.")
        for name in ("n_obs_steps", "n_action_steps"):
            if getattr(model_policy, name, getattr(self, name)) != getattr(self, name):
                raise ValueError(f"Policy and runner disagree on {name}.")
        device = torch.device(policy.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError(
                "Runner RNG isolation currently supports CPU/CUDA policies."
            )
        if self.output_dir.exists() and any(self.output_dir.iterdir()):
            raise FileExistsError("Evaluation output_dir must be new or empty.")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        self.timings = dict(
            setup_seconds=0.0,
            policy_seconds=0.0,
            environment_seconds=0.0,
            finalize_seconds=0.0,
            policy_calls=0,
        )
        was_training = policy.training
        records = []
        try:
            policy.eval()
            with _random_state(self.seed, device):
                for start in range(0, len(self.episodes), self.n_envs):
                    batch = self.episodes[start : start + self.n_envs]
                    active_count = len(batch)
                    # Keep inference batch shape fixed, as in DP. Padded replicas
                    # never enter metrics, videos, or episode records.
                    batch = batch + [self.episodes[0]] * (self.n_envs - active_count)
                    with ExitStack() as resources:
                        records.extend(
                            self._run_batch(policy, batch, active_count, resources)
                        )
        finally:
            policy.train(was_training)
        metrics = self._episode_metrics(records)
        protocol = self._evaluation_config()
        self.timings["total_seconds"] = time.perf_counter() - started
        for name, value in (
            ("timings", self.timings),
            ("metrics", metrics),
            ("episodes", records),
            ("config", protocol),
        ):
            write_json(self.output_dir / f"{name}.json", value)
        return metrics

    @staticmethod
    def _episode_metrics(records):
        """Aggregate completed requested episodes using the DP score convention."""
        metrics = {}
        for record in records:
            prefix = record["split"] + "/"
            metrics[prefix + f"sim_max_reward_{record['id']}"] = record["max_reward"]
            if record["video_path"] is not None:
                metrics[prefix + f"sim_video_{record['id']}"] = record["video_path"]
        for split in ("train", "test"):
            selected = [record for record in records if record["split"] == split]
            if selected:
                metrics[f"{split}/mean_score"] = float(
                    np.mean([record["max_reward"] for record in selected])
                )
                metrics[f"{split}/success_rate"] = float(
                    np.mean([record["success"] for record in selected])
                )
                metrics[f"{split}/num_episodes"] = len(selected)
                metrics[f"{split}/num_simulation_failures"] = sum(
                    "simulation_error" in record for record in selected
                )
        return metrics

    def _evaluation_config(self):
        """Record the rollout protocol and environment provenance with the scores."""
        protocol = {
            name: getattr(self, name)
            for name in (
                "obs_keys",
                "obs_dim",
                "action_space",
                "n_obs_steps",
                "n_action_steps",
                "n_train",
                "n_test",
                "train_start_idx",
                "test_start_seed",
                "max_steps",
                "n_envs",
                "num_workers",
                "worker_timeout",
                "seed",
                "train_reset",
                "repair_observations",
                "n_train_vis",
                "n_test_vis",
                "render_hw",
                "render_camera_name",
                "fps",
                "crf",
                "save_traces",
            )
        }
        protocol.update(
            dataset_path=str(self.dataset_path) if self.dataset_path else None,
            env_config=str(self.env_config) if self.env_config else None,
            env_meta=self.env_meta,
            versions=_versions(),
            backend="multiprocessing_batched"
            if self.num_workers
            else "synchronous_batched",
            aggregation="all_requested_episodes",
            simulation_instability="QACC warnings count as failed episodes; no retry",
        )
        return protocol
