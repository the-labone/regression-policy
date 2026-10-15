"""Read-only uint8 memory maps shared across DataLoader workers and runs.

Prepare once from HDF5, in bounded batches. Cache identity includes source path,
size/mtime, ordered demonstrations and camera shapes. Publishing is atomic under
a file lock; partial preparations are never reused.
"""

import fcntl
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np


def prepare_image_cache(path, demo_names, camera_shapes, cache_dir=None):
    path = Path(path).resolve()
    stat = path.stat()
    identity = {
        "version": 1,
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "demos": list(demo_names),
        "cameras": camera_shapes,
    }
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]
    parent = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir
        else path.parent / (path.name + ".image_cache")
    )
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / key
    with (parent / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (destination / "manifest.json").is_file():
            return destination
        temporary = Path(tempfile.mkdtemp(prefix=key + ".", dir=parent))
        try:
            with h5py.File(path, "r") as file:
                lengths = [len(file[f"data/{name}/actions"]) for name in demo_names]
                for index, (camera, shape) in enumerate(camera_shapes.items()):
                    array = np.lib.format.open_memmap(
                        temporary / f"camera_{index}.npy",
                        mode="w+",
                        dtype=np.uint8,
                        shape=(sum(lengths), *shape),
                    )
                    offset = 0
                    for name, length in zip(demo_names, lengths):
                        source = file[f"data/{name}/obs/{camera}"]
                        if source.dtype != np.uint8 or source.shape != (
                            length,
                            shape[1],
                            shape[2],
                            3,
                        ):
                            raise ValueError(
                                f"{name}/{camera}: expected aligned uint8 (T,H,W,3) images."
                            )
                        for start in range(0, length, 64):
                            stop = min(start + 64, length)
                            array[offset + start : offset + stop] = source[
                                start:stop
                            ].transpose(0, 3, 1, 2)
                        offset += length
                    array.flush()
                    del array
            (temporary / "manifest.json").write_text(json.dumps(identity, indent=2))
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return destination
