"""Configure the legacy OSMesa runtime installed for RoboMimic rollouts."""

import ctypes
import json
import os
import sys
from pathlib import Path

_handles = []


def configure_runtime():
    """Use this environment's runtime, leaving externally managed installs alone."""
    config = Path(sys.prefix) / "robomimic-runtime.json"
    if _handles or not config.is_file():
        return
    paths = json.loads(config.read_text())
    runtime = Path(paths["root"])
    libraries = runtime / "usr/lib/x86_64-linux-gnu"
    for name, value in (
        ("CPATH", str(runtime / "usr/include")),
        ("LIBRARY_PATH", str(libraries)),
        ("LD_LIBRARY_PATH", f"{libraries}:{paths['mujoco_bin']}"),
        ("PATH", str(Path(sys.prefix) / "bin")),
    ):
        previous = os.environ.get(name)
        os.environ[name] = f"{value}:{previous}" if previous else value
    # Match Praxis's legacy CPU-rendered simulator. Policy inference uses CUDA.
    os.environ["MUJOCO_PY_FORCE_CPU"] = "1"
    os.environ["MUJOCO_GL"] = "osmesa"
    os.environ.setdefault("PYGLFW_LIBRARY", str(libraries / "libglfw.so.3"))
    # glibc reads LD_LIBRARY_PATH at process startup. Preload OSMesa explicitly
    # so an already-running training process can start its first rollout.
    _handles.append(ctypes.CDLL(str(libraries / "libOSMesa.so.8"), ctypes.RTLD_GLOBAL))
