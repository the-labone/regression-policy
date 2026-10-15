# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Install a benchmark with uv: uv run scripts/install.py gr1."""

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHES = ROOT / "environments" / "patches"
SOURCES = ROOT / ".deps"
ENVS = ROOT / ".environments"

# Source versions used by the evaluation adapters; never follow a moving branch.
REPOS = {
    "robosuite": (
        "https://github.com/cheng-chi/robosuite.git",
        "277ab9588ad7a4f4b55cf75508b44aa67ec171f0",
    ),
    "gr00t": (
        "https://github.com/zyc00/GROOT_HT.git",
        "b861ec9090b841cc2bbae3cdc6b8d2a384a3660b",
    ),
    "gr1": (
        "https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git",
        "4840e671596f93ca03651524b9f72ffb1aadfeff",
    ),
    "simpler": (
        "https://github.com/squarefk/SimplerEnv.git",
        "8a2d286c926c1371927caa7651a412b4cc331756",
    ),
    "pi05": (
        "https://github.com/huggingface/lerobot.git",
        "a16f34c085c9597fcbdb9fde395a3334d78df716",
    ),
    "cosmos": (
        "https://github.com/NVIDIA/cosmos-framework.git",
        "18d90eacbda778e666e6978dc1ab6ba8635400e3",
    ),
    "libero": (
        "https://github.com/Lifelong-Robot-Learning/LIBERO.git",
        "8f1084e3132a39270c3a13ebe37270a43ece2a01",
    ),
}


class Installer:
    def __init__(self, *, dry_run=False, skip_assets=False):
        self.dry_run = dry_run
        self.skip_assets = skip_assets
        self.uv = shutil.which("uv") or "uv"

    def run(self, *args, cwd=ROOT, env=None):
        command = [str(arg) for arg in args]
        print(f"$ {shlex.join(command)}", flush=True)
        if not self.dry_run:
            subprocess.run(command, cwd=cwd, env=env, check=True)

    def environment(self, **updates):
        env = os.environ.copy()
        # Do not inherit uv run's temporary environment or an activated conda env.
        env.pop("VIRTUAL_ENV", None)
        env.pop("UV_PROJECT_ENVIRONMENT", None)
        env.update(GIT_LFS_SKIP_SMUDGE="1", **updates)
        return env

    def checkout(self, name, destination=None, *, recursive=False):
        url, revision = REPOS[name]
        path = destination or SOURCES / name
        env = self.environment()
        if not path.exists():
            if not self.dry_run:
                path.parent.mkdir(parents=True, exist_ok=True)
            self.run(
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                path,
                env=env,
            )
            self.run("git", "-C", path, "checkout", "--detach", revision, env=env)
        elif not self.dry_run:
            actual = subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
            ).strip()
            if actual != revision:
                raise RuntimeError(
                    f"{path} is at {actual}; expected {revision}. "
                    "Move that checkout aside before retrying."
                )
        if recursive:
            self.run(
                "git",
                "-C",
                path,
                "submodule",
                "update",
                "--init",
                "--recursive",
                env=env,
            )
        return path

    def patch(self, source, name):
        patch = PATCHES / name
        if not self.dry_run:
            applied = subprocess.run(
                ["git", "-C", str(source), "apply", "--reverse", "--check", str(patch)],
                capture_output=True,
            )
            if applied.returncode == 0:
                return
        self.run("git", "-C", source, "apply", "--check", patch)
        self.run("git", "-C", source, "apply", patch)

    def patch_series(self, source, names):
        """A later patch can overlap the first; check the final patch first."""
        if not self.dry_run:
            applied = subprocess.run(
                [
                    "git",
                    "-C",
                    str(source),
                    "apply",
                    "--reverse",
                    "--check",
                    str(PATCHES / names[-1]),
                ],
                capture_output=True,
            )
            if applied.returncode == 0:
                return
        for name in names:
            self.patch(source, name)

    def venv(self, name, version):
        path = ENVS / name
        python = path / "bin" / "python"
        if python.exists() and not self.dry_run:
            actual = subprocess.check_output(
                [
                    str(python),
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:2])))",
                ],
                text=True,
            ).strip()
            if actual != version:
                raise RuntimeError(f"{path} uses Python {actual}; expected {version}.")
        else:
            self.run(
                self.uv,
                "venv",
                "--python",
                version,
                "--allow-existing",
                path,
                env=self.environment(),
            )
        return python

    def pip(self, python, *args):
        # The root uv configuration pins RoboMimic's older Torch and NumPy.
        # It must not constrain the other backends' dependency resolution.
        self.run(
            self.uv,
            "pip",
            "install",
            "--no-config",
            "--python",
            python,
            *args,
            env=self.environment(),
        )

    def sync(self, source, name, version, *extras):
        path = ENVS / name
        # GR00T ignores uv.lock in its public checkout. Ship the validated lock
        # here so a fresh clone has the same environment as a cached checkout.
        bundled_lock = ROOT / "environments" / f"{name}.uv.lock"
        if bundled_lock.is_file() and not (source / "uv.lock").exists():
            print(f"Using bundled {name} lockfile", flush=True)
            if not self.dry_run:
                shutil.copyfile(bundled_lock, source / "uv.lock")
        self.run(
            self.uv,
            "sync",
            "--project",
            source,
            "--locked",
            "--inexact",
            "--no-dev",
            "--python",
            version,
            *extras,
            env=self.environment(UV_PROJECT_ENVIRONMENT=str(path)),
        )
        return path / "bin" / "python"

    def expose(self, python, source, name):
        # Simulator environments need native utility imports, not the model's
        # package metadata (GR00T requires Python 3.12; Simpler uses 3.10).
        code = (
            "import pathlib, sysconfig; "
            f"pathlib.Path(sysconfig.get_path('purelib'), {name!r} + '.pth')"
            f".write_text({str(source)!r} + '\\n')"
        )
        self.run(python, "-c", code)

    def package(self, python):
        self.pip(python, "-e", f"{ROOT}[evaluation,download]")

    def configure_libero(self, python, name):
        # HF LIBERO and upstream LIBERO otherwise share ~/.libero/config.yaml,
        # which can point one environment at another installation's assets.
        config_dir = str(ENVS / name / "libero-config")
        startup = (
            f"import os; os.environ.setdefault('LIBERO_CONFIG_PATH', {config_dir!r})\n"
        )
        self.run(
            python,
            "-c",
            "import pathlib, sysconfig; "
            "pathlib.Path(sysconfig.get_path('purelib'), 'regression_policy_libero.pth')"
            f".write_text({startup!r})",
        )
        code = (
            "import os, pathlib; "
            "p = pathlib.Path(os.environ['LIBERO_CONFIG_PATH']); "
            "p.mkdir(parents=True, exist_ok=True); "
            "cfg = p / 'config.yaml'; "
            "fresh = not cfg.exists() or cfg.stat().st_size == 0; "
            "cfg.touch(exist_ok=True) if fresh else None; "
            "from libero.libero import set_libero_default_path; "
            "set_libero_default_path() if fresh else None"
        )
        self.run(
            python,
            "-c",
            code,
            env=self.environment(MUJOCO_GL="egl", LIBERO_CONFIG_PATH=config_dir),
        )

    def robomimic(self):
        source = self.checkout("robosuite")
        self.patch(source, "robosuite-python310.patch")
        self.run(
            self.uv,
            "sync",
            "--locked",
            "--all-extras",
            "--python",
            "3.10",
            env=self.environment(),
        )
        python = ROOT / ".venv/bin/python"
        self.robomimic_runtime(python)
        return {"python": str(python)}

    def robomimic_runtime(self, python):
        # Extract the same Ubuntu 22.04 OSMesa libraries used by Praxis into
        # this checkout; never install packages into the operating system.
        runtime = SOURCES / "robomimic-runtime"
        manifest = json.loads(
            (ROOT / "environments/robomimic-runtime.json").read_text()
        )
        for package, checksum in manifest["packages"].items():
            archive = runtime / Path(package).name
            if not self.dry_run:
                runtime.mkdir(parents=True, exist_ok=True)
                if not archive.is_file():
                    temporary = archive.with_suffix(".partial")
                    try:
                        with (
                            urllib.request.urlopen(
                                manifest["base_url"] + package, timeout=120
                            ) as source,
                            temporary.open("wb") as target,
                        ):
                            shutil.copyfileobj(source, target)
                        if (
                            hashlib.sha256(temporary.read_bytes()).hexdigest()
                            != checksum
                        ):
                            raise RuntimeError(f"Checksum mismatch: {archive.name}")
                        temporary.replace(archive)
                    finally:
                        temporary.unlink(missing_ok=True)
                elif hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
                    raise RuntimeError(f"Checksum mismatch: {archive.name}")
            self.run("dpkg-deb", "--extract", archive, runtime)
        self.run(
            python,
            "-c",
            "import json, pathlib, sys, sysconfig; "
            "mujoco = pathlib.Path(sysconfig.get_path('purelib')) / "
            "'mujoco_py/binaries/linux/mujoco210/bin'; "
            "config = pathlib.Path(sys.prefix) / 'robomimic-runtime.json'; "
            f"config.write_text(json.dumps(dict(root={str(runtime)!r}, mujoco_bin=str(mujoco))))",
        )
        # Build MuJoCo's Cython extension now, before rollout workers race to
        # import it for the first time halfway through a long training run.
        self.run(
            python,
            "-c",
            "from ht_regression.evaluation.robomimic.runtime import configure_runtime; "
            "configure_runtime(); import robosuite, robomimic, mujoco_py; "
            "print('RoboMimic simulator ready:', robosuite.__version__, mujoco_py.get_version())",
        )

    def gr00t(self, benchmark):
        source = self.checkout("gr00t")
        self.patch(source, "gr00t-install.patch")
        self.patch(source, "gr00t-training.patch")
        model = self.sync(source, "gr00t", "3.12")
        self.package(model)
        sim = self.venv(benchmark, "3.12" if benchmark == "gr1" else "3.10")
        if benchmark == "gr1":
            sim_source = self.checkout(
                "gr1", source / "external_dependencies/robocasa-gr1-tabletop-tasks"
            )
            self.pip(
                sim,
                "--torch-backend",
                "cu128",
                "torch==2.9.0",
                "torchvision==0.24.0",
                "numpy==1.26.4",
                "mujoco==3.2.6",
                "gymnasium==0.29.1",
                "av==15.0.0",
                "transformers==4.57.3",
                "msgpack==1.1.0",
                "msgpack-numpy==0.4.8",
                "pydantic",
                "pyzmq",
                "tyro",
                "robosuite @ git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1",
                "-e",
                sim_source,
            )
            if not self.skip_assets:
                assets = sim_source / ".regression-policy-assets-ready"
                if not assets.exists():
                    self.run(
                        sim,
                        sim_source / "robocasa/scripts/download_tabletop_assets.py",
                        "-y",
                    )
                    if not self.dry_run:
                        assets.touch()
        else:
            sim_source = self.checkout(
                "simpler", source / "external_dependencies/SimplerEnv", recursive=True
            )
            self.pip(
                sim,
                "--torch-backend",
                "cu128",
                "--build-constraint",
                ROOT / "environments/simpler-build.txt",
                "torch==2.7.0",
                "torchvision==0.22.0",
                "numpy==1.26.4",
                "gymnasium==0.29.1",
                "json-numpy>=2.1.1",
                "ray==2.48.0",
                "opencv-python-headless==4.10.0.84",
                "tianshou==0.5.1",
                "transformers==4.57.3",
                "diffusers==0.35.1",
                "scipy==1.15.3",
                "pandas==2.2.3",
                "dm-tree==0.1.9",
                "einops==0.8.1",
                "albumentations==1.4.18",
                "msgpack==1.1.0",
                "msgpack-numpy==0.4.8",
                "pydantic",
                "av",
                "pyzmq",
                "tyro",
                "setuptools==80.9.0",
                "-e",
                sim_source / "ManiSkill2_real2sim",
                "-e",
                sim_source,
            )
        self.expose(sim, source, "gr00t")
        self.package(sim)
        return {
            "native_root": str(source),
            "model_python": str(model),
            "simulator_python": str(sim),
        }

    def pi05(self):
        source = self.checkout("pi05")
        self.patch_series(source, ("lerobot-pi05.patch", "pi05-training.patch"))
        python = self.venv("pi05", "3.12")
        self.pip(
            python,
            "--torch-backend",
            "cu128",
            "torch==2.11.0",
            "torchvision==0.26.0",
            "transformers==5.5.4",
            "numpy==2.2.6",
            "-e",
            f"{source}[pi,libero,evaluation]",
        )
        self.package(python)
        self.configure_libero(python, "pi05")
        return {"python": str(python)}

    def cosmos(self):
        source = self.checkout("cosmos")
        self.patch(source, "cosmos-framework.patch")
        # Native action inference imports shared model/config modules from the
        # framework's full runtime, including transformer-engine.
        model = self.sync(
            source, "cosmos", "3.13", "--all-extras", "--group", "cu128-train"
        )
        # TE 2.12 searches nvidia/cudart, but the CUDA 12 wheel installs into
        # nvidia/cuda_runtime. Keep inference usable without a system toolkit.
        self.run(
            model,
            "-c",
            "import pathlib, sysconfig; "
            "root = pathlib.Path(sysconfig.get_path('purelib')) / 'nvidia'; "
            "alias = root / 'cudart'; "
            "alias.symlink_to('cuda_runtime', target_is_directory=True) "
            "if (root / 'cuda_runtime').is_dir() and not alias.exists() else None",
        )
        self.package(model)
        sim_source = self.checkout("libero", source / "LIBERO")
        sim = self.venv("cosmos-libero", "3.10")
        self.pip(
            sim,
            "--torch-backend",
            "cu124",
            "--override",
            ROOT / "environments/libero-overrides.txt",
            "-r",
            sim_source / "requirements.txt",
            "-e",
            sim_source,
            "torch==2.5.1",
            "loguru",
            "requests",
            "scipy",
            "pillow",
            "imageio",
            "imageio-ffmpeg",
        )
        self.expose(sim, source, "cosmos_framework")
        # Upstream LIBERO's setuptools metadata omits its outer namespace.
        # Its native evaluation instructions expose the checkout on PYTHONPATH.
        self.expose(sim, sim_source, "libero")
        self.configure_libero(sim, "cosmos-libero")
        return {
            "python": str(model),
            "simulator_python": str(sim),
            "native_client": str(
                source / "cosmos_framework/simulation/libero/closed_loop_eval.py"
            ),
        }

    def install(self, benchmark):
        if benchmark in ("gr1", "simpler"):
            result = self.gr00t(benchmark)
        else:
            result = getattr(self, benchmark)()
        if not self.dry_run:
            ENVS.mkdir(parents=True, exist_ok=True)
            (ENVS / f"{benchmark}.json").write_text(json.dumps(result, indent=2) + "\n")
            print(f"\nInstalled {benchmark}. Paths: {ENVS / (benchmark + '.json')}")
        return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "benchmark", choices=("robomimic", "gr1", "simpler", "pi05", "cosmos")
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without installing."
    )
    parser.add_argument(
        "--skip-assets",
        action="store_true",
        help="Skip the GR1 simulator asset download.",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and (sys.platform != "linux" or platform.machine() != "x86_64"):
        parser.error("These environments target Linux x86_64 with an NVIDIA GPU.")
    try:
        Installer(dry_run=args.dry_run, skip_assets=args.skip_assets).install(
            args.benchmark
        )
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Installation failed: {error}\n")


if __name__ == "__main__":
    main()
