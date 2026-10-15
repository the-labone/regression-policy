"""Episode-safe RoboMimic state windows with explicit delta/ABS semantics.

Window alignment and normalization follow Diffusion Policy revision
5ba07ac6661db573af695b419a7947ecb704690f; see LICENSE.diffusion_policy.
"""

import copy
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .actions import ActionSpace, check_action_space, encode_actions
from .metadata import (
    REPAIRED_OBS_ATTR,
    controller_env_metadata,
    read_env_metadata,
    resolve_source_action_space,
    text,
    validate_abs_reference_frame,
)
from .normalizer import RobomimicNormalizer
from .observations import (
    encode_observation,
    validate_tool_hang_first_frame,
)


def demo_names(data_group, filter_key: str | None = None) -> list[str]:
    names = [name for name in data_group if re.fullmatch(r"demo_\d+", name)]
    if filter_key is not None:
        names = [text(name) for name in data_group.file[f"mask/{filter_key}"][:]]
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in data_group for name in names)
    ):
        raise ValueError(
            "Expected a nonempty, unique set of valid demonstration names."
        )
    return sorted(names, key=lambda name: int(name.rsplit("_", 1)[1]))


class RobomimicStateDataset(Dataset):
    """Load state observations and actions into RAM, closing the HDF5 immediately.

    Returns unnormalized model-space `obs` (To,D), `action` (H,A), and a boolean
    `action_valid_mask` (H,) identifying real versus edge-repeated action labels.
    DP's original loss includes padded labels; using the mask is an objective choice.
    A sample begins To-1 steps before the current decision. Its first executable
    action is action[To-1]. Windows never cross demonstration boundaries.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        action_space: ActionSpace,
        source_action_space: ActionSpace | None = None,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        obs_keys: tuple[str, ...] | None = None,
        val_ratio: float = 0.0,
        seed: int = 42,
        filter_key: str | None = None,
        normalization: str = "dp",
    ):
        self.path = Path(path).expanduser().resolve()
        self.action_space = check_action_space(action_space)
        if normalization not in ("dp", "minmax"):
            raise ValueError("normalization must be 'dp' or 'minmax'.")
        self.normalization = normalization
        if (
            min(horizon, n_obs_steps, n_action_steps) < 1
            or n_obs_steps - 1 + n_action_steps > horizon
        ):
            raise ValueError(
                "Require positive horizons and To - 1 + n_action_steps <= horizon."
            )
        if not 0 <= val_ratio < 1:
            raise ValueError("val_ratio must lie in [0, 1).")
        self.horizon, self.n_obs_steps, self.n_action_steps = (
            horizon,
            n_obs_steps,
            n_action_steps,
        )
        self.pad_before, self.pad_after = n_obs_steps - 1, n_action_steps - 1
        self._episodes = []
        with h5py.File(self.path, "r") as file:
            data = file["data"]
            source = resolve_source_action_space(data, source_action_space)
            if source != self.action_space:
                raise ValueError(
                    f"File contains {source} actions, requested {self.action_space}. "
                    "Run offline controller-based conversion; changing an action-space flag is not conversion."
                )
            self._env_meta = read_env_metadata(data)
            self._repair_observations = bool(data.attrs.get(REPAIRED_OBS_ATTR, False))
            if self.action_space == "abs":
                validate_abs_reference_frame(self._env_meta)
            self._all_demo_names = demo_names(data, filter_key)
            first_actions = data[self._all_demo_names[0]]["actions"]
            if first_actions.ndim != 2 or first_actions.shape[1] not in (7, 14):
                raise ValueError("Expected native 7D or 14D actions in the HDF5 file.")
            arm_count = first_actions.shape[1] // 7
            default_keys = ("object",) + tuple(
                f"robot{arm}_{key}"
                for arm in range(arm_count)
                for key in ("eef_pos", "eef_quat", "gripper_qpos")
            )
            self.obs_keys = tuple(obs_keys) if obs_keys is not None else default_keys
            if not self.obs_keys or len(set(self.obs_keys)) != len(self.obs_keys):
                raise ValueError("obs_keys must be nonempty and unique.")
            obs_dim = None
            for name in self._all_demo_names:
                demo = data[name]
                native = demo["actions"][:]
                if (
                    native.ndim != 2
                    or native.shape[1] != arm_count * 7
                    or len(native) == 0
                ):
                    raise ValueError(f"{name}: inconsistent or empty action array.")
                obs = {key: demo[f"obs/{key}"][:] for key in self.obs_keys}
                for key, values in obs.items():
                    if (
                        values.ndim != 2
                        or len(values) != len(native)
                        or not np.isfinite(values).all()
                    ):
                        raise ValueError(
                            f"{name}/obs/{key}: expected finite aligned 2D state observations."
                        )
                # Check even if a custom training subset omits EEF pose features.
                if "object" in obs:
                    check_obs = dict(obs)
                    for key in ("robot0_eef_pos", "robot0_eef_quat"):
                        if key not in check_obs and f"obs/{key}" in demo:
                            check_obs[key] = demo[f"obs/{key}"][:]
                    validate_tool_hang_first_frame(check_obs, self._env_meta, name)
                observations = np.concatenate(
                    [obs[key] for key in self.obs_keys], axis=-1
                ).astype(np.float32)
                if obs_dim is not None and observations.shape[-1] != obs_dim:
                    raise ValueError(
                        f"{name}: observation dimension changed between demonstrations."
                    )
                obs_dim = observations.shape[-1]
                self._episodes.append(
                    (observations, encode_actions(native, self.action_space))
                )
        self.obs_dim = obs_dim
        self.action_dim = self._episodes[0][1].shape[-1]
        count = len(self._episodes)
        if val_ratio and count < 2:
            raise ValueError(
                "At least two demonstrations are required for train/validation splitting."
            )
        val_count = min(max(1, round(count * val_ratio)), count - 1) if val_ratio else 0
        self._val_ids = sorted(
            np.random.default_rng(seed)
            .choice(count, size=val_count, replace=False)
            .tolist()
        )
        self._episode_ids = [i for i in range(count) if i not in self._val_ids]
        self.normalizer = RobomimicNormalizer.fit(
            np.concatenate([self._episodes[i][0] for i in self._episode_ids]),
            np.concatenate([self._episodes[i][1] for i in self._episode_ids]),
            self.action_space,
            normalization=self.normalization,
        )
        self._build_windows()
        if not len(self):
            raise ValueError(
                "No training windows: demonstrations are too short for this horizon/padding."
            )

    def _build_windows(self):
        self._windows = [
            (episode, start)
            for episode in self._episode_ids
            for start in range(
                -self.pad_before,
                len(self._episodes[episode][0]) - self.horizon + self.pad_after + 1,
            )
        ]

    @property
    def demo_names(self) -> tuple[str, ...]:
        return tuple(self._all_demo_names[i] for i in self._episode_ids)

    def get_validation_dataset(self):
        validation = copy.copy(self)
        validation._episode_ids = self._val_ids.copy()
        validation._build_windows()
        return validation

    def get_normalizer(self) -> RobomimicNormalizer:
        """Independent copy suitable for moving to a policy's device/checkpoint."""
        return copy.deepcopy(self.normalizer)

    def env_metadata(self) -> dict:
        """Environment metadata for evaluation, with controller mode set explicitly."""
        return controller_env_metadata(self._env_meta, self.action_space)

    def encode_observation(self, observation: dict[str, np.ndarray]) -> np.ndarray:
        """Pack evaluation observations in training order, BEFORE normalization.

        Accepts a single frame or arbitrary leading batch/time dimensions. For
        an explicitly geometry-repaired dataset, apply the same repair online.
        """
        return encode_observation(
            observation,
            self.obs_keys,
            self.obs_dim,
            self._env_meta,
            repair_observations=self._repair_observations,
        )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        episode, start = self._windows[index]
        obs, action = self._episodes[episode]
        indices = np.arange(start, start + self.horizon)
        valid = (indices >= 0) & (indices < len(action))
        clipped = np.clip(indices, 0, len(action) - 1)
        return {
            "obs": torch.from_numpy(obs[clipped[: self.n_obs_steps]].copy()),
            "action": torch.from_numpy(action[clipped].copy()),
            "action_valid_mask": torch.from_numpy(valid),
        }
