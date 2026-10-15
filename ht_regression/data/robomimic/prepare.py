"""Prepare explicit delta/ABS HDF5 files without modifying source demonstrations.

True delta-to-ABS conversion queries OSC controller goals at each recorded state,
following DP. It refreshes controller caches even on the first frame and verifies
delta/ABS one-step equivalence before publishing the output file.
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .actions import check_action_space, encode_actions
from .dataset import demo_names
from .metadata import (
    ACTION_SPACE_ATTR,
    REPAIRED_OBS_ATTR,
    SCHEMA_VERSION,
    controller_env_metadata,
    read_env_metadata,
    resolve_source_action_space,
    text,
    validate_abs_reference_frame,
)
from .observations import (
    repair_tool_hang_relative_poses,
    tool_hang_layout,
    validate_tool_hang_first_frame,
)


def _controllers(env):
    """Ordered (robot, arm controller) pairs for one/two single-arm robots."""
    pairs = []
    for robot in env.env.robots:
        if hasattr(robot, "part_controllers"):
            arms = [
                robot.part_controllers[key]
                for key in ("right", "left")
                if key in robot.part_controllers
            ]
            if len(arms) != 1:
                raise ValueError(
                    "Conversion currently supports one arm per robot (including two-robot Transport)."
                )
            controller = arms[0]
        else:
            controller = robot.controller
        if getattr(controller, "impedance_mode", "fixed") != "fixed":
            raise ValueError(
                "Conversion requires fixed-impedance OSC_POSE controllers."
            )
        if getattr(controller, "_goal_update_mode", "achieved") != "achieved":
            raise ValueError(
                "State-local conversion requires achieved-pose delta control."
            )
        pairs.append((robot, controller))
    return pairs


def _reset(env, state, *, model_xml=None, ep_meta=None):
    payload = {"states": state}
    if model_xml is not None:
        payload["model"] = model_xml
    if ep_meta is not None:
        payload["ep_meta"] = ep_meta
    env.reset_to(payload)
    # reset_to forwards the simulator but can leave controller caches stale.
    # Do this on EVERY frame, including frame 0; never skip it in verification.
    for _, controller in _controllers(env):
        controller.update(force=True)


def convert_delta_actions(env, states, actions, *, model_xml=None, ep_meta=None):
    """Read controller-generated world-frame targets; no naive EEF + delta math."""
    encode_actions(actions, "delta")  # Validate native dimensions and finite values.
    if len(states) != len(actions) or not len(actions) or not np.isfinite(states).all():
        raise ValueError(
            "Conversion requires aligned, nonempty finite simulator states."
        )
    output = np.empty_like(actions, dtype=np.float64)
    arms = actions.shape[-1] // 7
    for index, (state, action) in enumerate(zip(states, actions)):
        _reset(
            env,
            state,
            model_xml=model_xml if index == 0 else None,
            ep_meta=ep_meta if index == 0 else None,
        )
        pairs = _controllers(env)
        if len(pairs) != arms:
            raise ValueError(
                "Controller count does not match dataset action dimension."
            )
        for arm, (robot, controller) in enumerate(pairs):
            if getattr(controller, "input_type", "delta") != "delta" or not getattr(
                controller, "use_delta", True
            ):
                raise ValueError("Delta-to-ABS conversion requires a delta controller.")
            command = action[arm * 7 : (arm + 1) * 7]
            robot.control(command, policy_step=True)
            position, orientation = (
                controller.goal_pos.copy(),
                controller.goal_ori.copy(),
            )
            if getattr(controller, "input_ref_frame", "world") == "base":
                position = controller.origin_pos + controller.origin_ori @ position
                orientation = controller.origin_ori @ orientation
            output[index, arm * 7 : arm * 7 + 3] = position
            output[index, arm * 7 + 3 : arm * 7 + 6] = Rotation.from_matrix(
                orientation
            ).as_rotvec()
            output[index, arm * 7 + 6] = command[6]
    return output


def verify_conversion(
    delta_env,
    abs_env,
    states,
    delta_actions,
    abs_actions,
    *,
    model_xml=None,
    ep_meta=None,
    steps=3,
    atol=1e-5,
):
    """Compare one-step simulator states, including the first recorded frame."""
    if steps < 1:
        raise ValueError("At least one verification step is required.")
    indices = np.unique(
        np.linspace(0, len(states) - 1, min(steps, len(states)), dtype=int)
    )
    worst_error = 0.0
    for index in indices:
        for env in (delta_env, abs_env):
            _reset(
                env,
                states[index],
                model_xml=model_xml if index == 0 else None,
                ep_meta=ep_meta if index == 0 else None,
            )
        delta_env.step(delta_actions[index])
        abs_env.step(abs_actions[index])
        error = float(
            np.max(
                np.abs(delta_env.get_state()["states"] - abs_env.get_state()["states"])
            )
        )
        if not np.isfinite(error) or error > atol:
            raise ValueError(
                f"Delta/ABS replay mismatch at frame {index}: max state error {error:.6g} > {atol}."
            )
        worst_error = max(worst_error, error)
    return {"checked_frames": indices.tolist(), "max_state_error": worst_error}


def _create_environments(env_meta, source_robosuite_version):
    # Training-time dataset loading does not import MuJoCo / robosuite.
    import robomimic
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils
    import robosuite

    recorded = env_meta.get("env_version")
    if recorded and source_robosuite_version and recorded != source_robosuite_version:
        raise ValueError("Explicit source robosuite version conflicts with env_args.")
    expected = recorded or source_robosuite_version
    if expected is None:
        raise ValueError(
            "Unversioned source: supply source_robosuite_version from the dataset's provenance."
        )
    if expected != robosuite.__version__:
        raise ValueError(
            f"Source requires robosuite {expected}; installed {robosuite.__version__}. "
            "Use a matching conversion environment; simulator upgrades are not dataset conversion."
        )
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {
            "low_dim": [
                "object",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            ]
        }
    )
    environments = []
    try:
        for space in ("delta", "abs"):
            environments.append(
                EnvUtils.create_env_from_metadata(
                    env_meta=controller_env_metadata(env_meta, space),
                    render=False,
                    render_offscreen=False,
                    use_image_obs=False,
                )
            )
    except BaseException:
        for env in environments:
            env.env.close()
        raise
    return environments, {
        "robosuite": robosuite.__version__,
        "robomimic": robomimic.__version__,
    }


def prepare_dataset(
    source_path,
    output_path,
    *,
    action_space,
    source_action_space=None,
    repair_tool_hang=False,
    source_robosuite_version=None,
    verification_steps=3,
):
    """Copy/convert a file, validate it, then publish a NEW marked HDF5 file.

    Existing source labels may be explicitly declared when there is no marker.
    ABS-to-delta inversion is not provided; use the original delta demonstrations.
    Geometry repair is opt-in and fixes observations only, never ABS action labels.
    """
    check_action_space(action_space)
    source_path, output_path = (
        Path(source_path).expanduser().resolve(),
        Path(output_path).expanduser().resolve(),
    )
    if source_path == output_path or output_path.exists():
        raise FileExistsError("Output must be a new path, distinct from the source.")
    with h5py.File(source_path, "r") as source:
        source_space = resolve_source_action_space(source["data"], source_action_space)
        env_meta = read_env_metadata(source["data"])
        names = demo_names(source["data"])
    if source_space == "abs":
        validate_abs_reference_frame(env_meta)
    if source_space == "abs" and action_space == "delta":
        raise ValueError(
            "Use the original delta dataset; ABS-to-delta conversion is not implicit."
        )
    converting = source_space != action_space
    if converting and verification_steps < 1:
        raise ValueError("Conversion must verify at least the first frame.")
    environments, versions = ([], {})
    if converting:
        environments, versions = _create_environments(
            env_meta, source_robosuite_version
        )
    temporary = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(
            prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
        )
        os.close(fd)
        temporary = Path(name)
        shutil.copy2(source_path, temporary)
        report = {
            "schema_version": SCHEMA_VERSION,
            "source_path": str(source_path),
            "source_action_space": source_space,
            "action_space": action_space,
            "versions": versions,
            "repair_tool_hang": repair_tool_hang,
            "verification": {},
        }
        with h5py.File(temporary, "r+") as output:
            data = output["data"]
            for demo_name in names:
                demo = data[demo_name]
                native = demo["actions"][:]
                encode_actions(native, source_space)
                if converting:
                    if not np.issubdtype(native.dtype, np.floating):
                        raise ValueError(
                            "Conversion requires floating-point action storage."
                        )
                    states = demo["states"][:]
                    model_xml = (
                        text(demo.attrs["model_file"])
                        if "model_file" in demo.attrs
                        else None
                    )
                    ep_meta = (
                        text(demo.attrs["ep_meta"]) if "ep_meta" in demo.attrs else None
                    )
                    converted = convert_delta_actions(
                        environments[0],
                        states,
                        native,
                        model_xml=model_xml,
                        ep_meta=ep_meta,
                    )
                    converted = converted.astype(native.dtype)
                    report["verification"][demo_name] = verify_conversion(
                        environments[0],
                        environments[1],
                        states,
                        native,
                        converted,
                        model_xml=model_xml,
                        ep_meta=ep_meta,
                        steps=verification_steps,
                    )
                    demo["actions"][...] = converted
                for group_name in ("obs", "next_obs"):
                    if group_name not in demo:
                        continue
                    group = demo[group_name]
                    layout = tool_hang_layout(env_meta)
                    if layout is not None:
                        obs = {
                            key: group[key][:]
                            for key in ("object", "robot0_eef_pos", "robot0_eef_quat")
                        }
                        if repair_tool_hang:
                            obs["object"] = repair_tool_hang_relative_poses(obs, layout)
                            group["object"][...] = obs["object"]
                        validate_tool_hang_first_frame(
                            obs, env_meta, f"{demo_name}/{group_name}"
                        )
            data.attrs[ACTION_SPACE_ATTR] = action_space
            data.attrs["praxis_schema_version"] = SCHEMA_VERSION
            data.attrs["praxis_source_env_args"] = json.dumps(env_meta)
            data.attrs["env_args"] = json.dumps(
                controller_env_metadata(env_meta, action_space)
            )
            if repair_tool_hang and tool_hang_layout(env_meta) is not None:
                data.attrs[REPAIRED_OBS_ATTR] = True
            data.attrs["praxis_preparation"] = json.dumps(report)
        # Publish atomically without overwriting a file created concurrently.
        os.link(temporary, output_path)
        return report
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        for env in environments:
            env.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--action-space", choices=("delta", "abs"), required=True)
    parser.add_argument("--source-action-space", choices=("delta", "abs"))
    parser.add_argument("--repair-tool-hang", action="store_true")
    parser.add_argument("--source-robosuite-version")
    parser.add_argument("--verification-steps", type=int, default=3)
    args = parser.parse_args()
    report = prepare_dataset(
        args.input,
        args.output,
        action_space=args.action_space,
        source_action_space=args.source_action_space,
        repair_tool_hang=args.repair_tool_hang,
        source_robosuite_version=args.source_robosuite_version,
        verification_steps=args.verification_steps,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
