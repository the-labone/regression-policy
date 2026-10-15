"""Visual RoboMimic rollouts sharing the state runner's episode protocol."""

import copy
from functools import partial

import h5py

from ht_regression.data.robomimic.dataset import demo_names
from ht_regression.data.robomimic.observations import encode_rgb

from .state_runner import RobomimicStateRunner, _create_env


class RobomimicImageRunner(RobomimicStateRunner):
    def __init__(self, *args, camera_shapes, **kwargs):
        self.camera_shapes = {key: tuple(shape) for key, shape in camera_shapes.items()}
        if not self.camera_shapes or any(
            not key.endswith("_image") or len(shape) != 3 or shape[0] != 3
            for key, shape in self.camera_shapes.items()
        ):
            raise ValueError("Specify RGB camera_shapes keyed by <camera>_image.")
        kwargs.setdefault(
            "env_factory", partial(_create_env, camera_keys=tuple(self.camera_shapes))
        )
        super().__init__(*args, **kwargs)
        if self.dataset_path is not None:
            with h5py.File(self.dataset_path, "r") as file:
                obs = file["data"][demo_names(file["data"])[0]]["obs"]
                for key, shape in self.camera_shapes.items():
                    # Evaluation only needs metadata and initial states. A lowdim
                    # copy of the same demonstrations need not store camera pixels.
                    if key in obs and obs[key].shape[1:] != (shape[1], shape[2], 3):
                        raise ValueError(f"Dataset image shape does not match {key}.")
        self.env_meta["env_kwargs"].update(
            use_object_obs=False,
            camera_names=[key.removesuffix("_image") for key in self.camera_shapes],
            camera_heights=[shape[1] for shape in self.camera_shapes.values()],
            camera_widths=[shape[2] for shape in self.camera_shapes.values()],
        )

    @classmethod
    def from_dp_checkpoint(cls, loaded, *, dataset_path=None, output_dir, **overrides):
        config = copy.deepcopy(loaded.config.get("task", {}).get("env_runner", {}))
        if (
            config.pop("_target_", None)
            != "diffusion_policy.env_runner.robomimic_image_runner.RobomimicImageRunner"
        ):
            raise ValueError("Checkpoint does not configure a RoboMimic image runner.")
        if config.pop("past_action", False) or config.pop("n_latency_steps", 0):
            raise ValueError("Latency and past-action conditioning are not supported.")
        encoder = loaded.policy.obs_encoder
        if encoder is None or loaded.obs_keys is None or loaded.action_space is None:
            raise ValueError(
                "Checkpoint must declare visual observation and action metadata."
            )
        config.pop("tqdm_interval_sec", None)
        config.pop("abs_action", None)
        config.pop("shape_meta", None)
        render_key = config.pop("render_obs_key", "agentview_image")
        config.update(
            camera_shapes=encoder.camera_shapes,
            obs_keys=loaded.obs_keys,
            action_space=loaded.action_space,
            render_camera_name=render_key.removesuffix("_image"),
            render_hw=encoder.camera_shapes[render_key][1:],
        )
        if dataset_path is not None or overrides.get("env_config") is not None:
            config["dataset_path"] = dataset_path
        config.setdefault("n_envs", None)
        config.update(overrides)
        return cls(output_dir=output_dir, **config)

    def _reset(self, env, episode):
        if episode["initial"] is not None:
            # DP fully resets each new visual environment before restoring a
            # demonstration state so that its renderer is initialized.
            env.reset()
        # Re-querying here advances robosuite 1.2's sensor sampling clocks by
        # another physics tick, changing future frames despite identical reset
        # observations. DP's image wrapper consumes the reset return directly.
        return self._encode(self._reset_env(env, episode))

    def _encode(self, observation):
        return {
            **{
                key: encode_rgb(observation[key], shape)
                for key, shape in self.camera_shapes.items()
            },
            "proprio": super()._encode(observation),
        }

    def _evaluation_config(self):
        return {
            **super()._evaluation_config(),
            "camera_shapes": self.camera_shapes,
            "observation_modality": "image",
            "image_layout": "CHW uint8 RGB",
        }
