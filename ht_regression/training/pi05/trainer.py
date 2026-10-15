"""Use LeRobot's native data, processors, DDP, optimizer and checkpoint loop."""

import inspect
import json
import sys
import tempfile
from pathlib import Path

from ..native import require_world_size, write_manifest


def train(config, *, resume_from=None):
    require_world_size(config["parallelism"]["dp_replicate"])
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.scripts import lerobot_train

    if "scaled_df_nll" not in inspect.getsource(PI05Policy.forward):
        raise RuntimeError("Install the release pi05-training.patch before training.")
    original = lerobot_train.make_optimizer_and_scheduler

    def make_optimizer(cfg, policy):
        result = original(cfg, policy)
        if policy.model.sigma_out_proj is None:
            raise RuntimeError(
                "HT training requires the native random sigma projection."
            )
        write_manifest(
            Path(config["output_dir"]) / "resume_launch"
            if resume_from
            else config["output_dir"],
            config,
            sources=[Path(inspect.getfile(PI05Policy)), Path(__file__)],
        )
        return result

    # LeRobot rejects pre-existing fresh output directories during validation.
    # Put the launch config outside that directory; write provenance afterward.
    with tempfile.TemporaryDirectory(prefix="regression-policy-pi05-") as folder:
        path = Path(folder) / "train_config.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        arguments = [f"--config_path={path}"]
        if resume_from is not None:
            arguments = [
                f"--config_path={Path(resume_from).resolve() / 'train_config.json'}",
                "--resume=true",
                f"--dataset.root={config['dataset']['root']}",
                f"--output_dir={config['output_dir']}",
            ]
        previous_argv = sys.argv
        lerobot_train.make_optimizer_and_scheduler = make_optimizer
        try:
            sys.argv = ["lerobot_train", *arguments]
            lerobot_train.main()
        finally:
            sys.argv = previous_argv
            lerobot_train.make_optimizer_and_scheduler = original
