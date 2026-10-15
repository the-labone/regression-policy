# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Launch a native VLA trainer in its installed environment."""

import argparse
import json
import os
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "robomimic": "robomimic",
    "gr1": "gr00t",
    "simpler": "gr00t",
    "pi05": "pi05",
    "cosmos": "cosmos",
}


def command_for(benchmark, arguments, *, check_only=False):
    try:
        paths = json.loads((ROOT / ".environments" / f"{benchmark}.json").read_text())
        python = (
            paths["model_python"]
            if benchmark in ("gr1", "simpler")
            else paths["python"]
        )
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError(
            f"Run `uv run scripts/install.py {benchmark}` first."
        ) from error
    if not Path(python).is_file():
        raise FileNotFoundError(python)
    module = f"ht_regression.training.{MODULES[benchmark]}.train"
    command = [python]
    if benchmark == "robomimic":
        command += ["-m", module]
        if check_only:
            command += ["--cfg", "job"]
    elif not check_only:
        command += [
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=8",
            "--module",
            module,
        ]
    else:
        command += ["-m", module, "--check-only"]
    if benchmark == "cosmos" and not any(
        a.split("=")[0] == "--native-root" for a in arguments
    ):
        command += ["--native-root", str(ROOT / ".deps/cosmos")]
    if benchmark == "simpler" and not any(
        a.split("=")[0] == "--task" for a in arguments
    ):
        raise ValueError("Simpler requires --task bridge or --task fractal.")
    return [*command, *arguments]


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dry-run", action="store_true")
    cli.add_argument("--check-only", action="store_true")
    cli.add_argument("benchmark", choices=MODULES)
    cli.add_argument("arguments", nargs=argparse.REMAINDER)
    args = cli.parse_args(argv)
    command = command_for(args.benchmark, args.arguments, check_only=args.check_only)
    if args.dry_run:
        print(shlex.join(command))
        return
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env["PATH"] = f"{Path(command[0]).parent}{os.pathsep}{env.get('PATH', '')}"
    os.chdir(ROOT)
    os.execve(command[0], command, env)


if __name__ == "__main__":
    main()
