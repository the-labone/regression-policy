# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Evaluate a checkpoint in its installed environment: uv run scripts/evaluate.py gr1 ..."""

import argparse
import json
import os
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "robomimic": "ht_regression.evaluation.robomimic.evaluate",
    "gr1": "ht_regression.evaluation.robocasa_gr1.evaluate",
    "simpler": "ht_regression.evaluation.simpler.evaluate",
    "pi05": "ht_regression.evaluation.libero.evaluate",
    "cosmos": "ht_regression.evaluation.libero.evaluate",
}


def command_for(benchmark, arguments):
    """Read installer paths and supply defaults without replacing user overrides."""
    profile = ROOT / ".environments" / f"{benchmark}.json"
    install_hint = f"Run `uv run scripts/install.py {benchmark}` first."
    try:
        paths = json.loads(profile.read_text())
        if benchmark in ("gr1", "simpler"):
            python = paths["model_python"]
            defaults = {
                "--native-root": paths["native_root"],
                "--model-python": python,
                "--simulator-python": paths["simulator_python"],
            }
        else:
            python = paths["python"]
            defaults = {}
            if benchmark == "cosmos":
                defaults = {
                    "--native-client": paths["native_client"],
                    "--simulator-python": paths["simulator_python"],
                }
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(
            f"Missing or invalid {benchmark} installation. {install_hint}"
        ) from error
    if not isinstance(python, str) or not Path(python).is_file():
        raise RuntimeError(
            f"The {benchmark} Python environment is unavailable. {install_hint}"
        )

    # Keep the venv executable path: resolving its symlink selects system Python.
    command = [str(Path(python).absolute()), "-m", MODULES[benchmark]]
    if benchmark in ("pi05", "cosmos"):
        command.append(benchmark)
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in arguments):
            command.extend((flag, value))
    return [*command, *arguments]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the resolved command."
    )
    parser.add_argument("benchmark", choices=MODULES)
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Evaluation options; use BENCHMARK --help for the full list.",
    )
    args = parser.parse_args(argv)
    try:
        command = command_for(args.benchmark, args.arguments)
        if args.dry_run:
            print(shlex.join(command))
            return
        env = os.environ.copy()
        env.pop("VIRTUAL_ENV", None)
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        env["PATH"] = f"{Path(command[0]).parent}{os.pathsep}{env.get('PATH', '')}"
        if args.benchmark in ("pi05", "cosmos"):
            env.setdefault("MUJOCO_GL", "egl")
        # Replace the launcher so signals and the evaluator's exit code propagate.
        os.execve(command[0], command, env)
    except (RuntimeError, OSError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
