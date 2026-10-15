"""Camera history and proprioception using the state dataset's action windows."""

from pathlib import Path

import h5py
import numpy as np
import torch

from .dataset import RobomimicStateDataset, demo_names
from .image_cache import prepare_image_cache
from .observations import encode_rgb


class RobomimicImageDataset(RobomimicStateDataset):
    def __init__(self, path, *, camera_keys, cache_dir=None, obs_keys=None, **kwargs):
        path = Path(path).expanduser().resolve()
        self.camera_keys = tuple(camera_keys)
        if (
            not self.camera_keys
            or len(set(self.camera_keys)) != len(self.camera_keys)
            or "proprio" in self.camera_keys
        ):
            raise ValueError(
                "camera_keys must be nonempty, unique, and exclude proprio."
            )
        with h5py.File(path, "r") as file:
            first = file["data"][demo_names(file["data"])[0]]
            if obs_keys is None:
                arms = first["actions"].shape[1] // 7
                obs_keys = tuple(
                    f"robot{arm}_{key}"
                    for arm in range(arms)
                    for key in ("eef_pos", "eef_quat", "gripper_qpos")
                )
            self.camera_shapes = {}
            for key in self.camera_keys:
                image = first[f"obs/{key}"]
                if image.ndim != 4 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"{key}: expected uint8 RGB (T,H,W,3).")
                self.camera_shapes[key] = (3, *image.shape[1:3])
            self.proprio_shapes = {
                key: first[f"obs/{key}"].shape[-1] for key in obs_keys
            }
        if any(not key.startswith("robot") for key in obs_keys):
            raise ValueError(
                "Visual policies use robot proprioception, not privileged object state."
            )
        super().__init__(path, obs_keys=obs_keys, **kwargs)
        # DP visual normalization is per-field range for pos/qpos and identity
        # for quaternions, unlike the lowdim recipe's single global obs scale.
        if self.normalization == "dp":
            values = np.concatenate([self._episodes[i][0] for i in self._episode_ids])
            offset = 0
            for key, width in self.proprio_shapes.items():
                selection = slice(offset, offset + width)
                if key.endswith("quat"):
                    self.normalizer.obs_scale[selection] = 1
                    self.normalizer.obs_offset[selection] = 0
                elif key.endswith(("pos", "qpos")):
                    low, high = values[:, selection].min(0), values[:, selection].max(0)
                    span = high - low
                    constant = span < 1e-7
                    scale = 2 / np.where(constant, 2, span)
                    bias = np.where(constant, -low, -1 - scale * low)
                    self.normalizer.obs_scale[selection] = torch.from_numpy(scale)
                    self.normalizer.obs_offset[selection] = torch.from_numpy(bias)
                else:
                    raise ValueError(
                        f"DP visual normalization does not define {key}; use minmax."
                    )
                offset += width
        self.cache_path = prepare_image_cache(
            self.path, self._all_demo_names, self.camera_shapes, cache_dir
        )
        self._offsets = np.cumsum([0] + [len(obs) for obs, _ in self._episodes])
        self._images = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None  # Spawned workers reopen memory maps, not copy images.
        return state

    def encode_observation(self, observation):
        return {
            **{
                key: encode_rgb(observation[key], shape)
                for key, shape in self.camera_shapes.items()
            },
            "proprio": super().encode_observation(observation),
        }

    def __getitem__(self, index):
        batch = super().__getitem__(index)
        episode, start = self._windows[index]
        indices = np.clip(
            np.arange(start, start + self.n_obs_steps),
            0,
            len(self._episodes[episode][0]) - 1,
        )
        indices += self._offsets[episode]
        if self._images is None:
            self._images = {
                key: np.load(self.cache_path / f"camera_{i}.npy", mmap_mode="r")
                for i, key in enumerate(self.camera_keys)
            }
        batch["obs"] = {
            **{
                key: torch.from_numpy(array[indices].copy())
                for key, array in self._images.items()
            },
            "proprio": batch["obs"],
        }
        return batch
