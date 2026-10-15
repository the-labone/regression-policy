# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Download one official ABS dataset without fetching the full multi-task ZIP.

Example: python scripts/download_robomimic.py --task lift --split ph --modality state
Train with dataset.source_action_space=abs and evaluation.source_robosuite_version=1.2.0.
"""

import argparse
import hashlib
import io
import json
import os
import re
import tempfile
import urllib.request
import zipfile
import zlib
from pathlib import Path

ARCHIVES = {
    "state": "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip",
    "vision": "https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip",
}


class HTTPRangeFile(io.RawIOBase):
    """Seekable, bounded-memory HTTP reader for zipfile, including ZIP64 archives."""

    def __init__(self, url, *, block_size=8 * 1024**2):
        super().__init__()
        self.url = url
        self.block_size = block_size
        self.position = 0
        self.size = None
        self.cache_start = 0
        self.cache = self._fetch(0, 0)

    def _fetch(self, start, end):
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            content_range = response.headers.get("Content-Range", "")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
            if response.status != 206 or not match:
                raise ValueError(
                    "Server must support HTTP byte ranges; download aborted."
                )
            first, last, total = map(int, match.groups())
            if (first, last) != (start, end) or (
                self.size is not None and total != self.size
            ):
                raise ValueError("Server returned an unexpected archive byte range.")
            data = response.read(end - start + 2)
            if len(data) != end - start + 1:
                raise ValueError("Incomplete archive byte range.")
            self.size = total
            return data

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=os.SEEK_SET):
        bases = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}
        if whence not in bases:
            raise ValueError("Invalid seek mode.")
        position = bases[whence] + offset
        if position < 0:
            raise ValueError("Negative seek position.")
        self.position = position
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        result = bytearray()
        while size > 0:
            index = self.position - self.cache_start
            if not 0 <= index < len(self.cache):
                self.cache_start = self.position
                self.cache = self._fetch(
                    self.position, min(self.size, self.position + self.block_size) - 1
                )
                index = 0
            count = min(size, len(self.cache) - index)
            result.extend(self.cache[index : index + count])
            self.position += count
            size -= count
        return bytes(result)


def _checksum(path):
    digest, crc, size = hashlib.sha256(), 0, 0
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
            crc = zlib.crc32(block, crc)
            size += len(block)
    return size, crc, digest.hexdigest()


def download(task, split, modality, output_dir, *, dry_run=False, archive_url=None):
    if task not in {"lift", "can", "square", "transport", "tool_hang"}:
        raise ValueError("Unknown RoboMimic task.")
    if split not in {"ph", "mh"} or (task == "tool_hang" and split != "ph"):
        raise ValueError("Unsupported task/split; Tool Hang provides PH only.")
    filename = "low_dim_abs.hdf5" if modality == "state" else "image_abs.hdf5"
    url = archive_url or ARCHIVES[modality]
    member = f"robomimic/datasets/{task}/{split}/{filename}"
    destination = Path(output_dir).expanduser().resolve() / task / split / filename
    with HTTPRangeFile(url) as remote, zipfile.ZipFile(remote) as archive:
        info = archive.getinfo(member)
        report = {
            "url": url,
            "member": member,
            "path": str(destination),
            "compressed_bytes": info.compress_size,
            "bytes": info.file_size,
            "crc32": info.CRC,
            "source_action_space": "abs",
            "source_robosuite_version": "1.2.0",
        }
        if dry_run:
            return report
        if destination.exists():
            size, crc, digest = _checksum(destination)
            if (size, crc) != (info.file_size, info.CRC):
                raise FileExistsError(
                    f"Existing dataset fails archive verification: {destination}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = None
            try:
                digest = hashlib.sha256()
                size = 0
                with tempfile.NamedTemporaryFile(
                    dir=destination.parent, suffix=".partial", delete=False
                ) as output:
                    temporary = Path(output.name)
                    # ZipExtFile validates the member's CRC on reaching EOF.
                    with archive.open(info) as stream:
                        while block := stream.read(8 * 1024**2):
                            output.write(block)
                            digest.update(block)
                            size += len(block)
                if size != info.file_size:
                    raise ValueError(
                        "Extracted dataset size does not match the archive."
                    )
                # Publish without overwriting a concurrently created dataset.
                os.link(temporary, destination)
                digest = digest.hexdigest()
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        report["sha256"] = digest
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("lift", "can", "square", "transport", "tool_hang"),
        default="lift",
    )
    parser.add_argument("--split", choices=("ph", "mh"), default="ph")
    parser.add_argument("--modality", choices=("state", "vision"), default="state")
    parser.add_argument("--output-dir", type=Path, default=Path("data/robomimic"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect download size without extracting data.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            download(
                args.task,
                args.split,
                args.modality,
                args.output_dir,
                dry_run=args.dry_run,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
