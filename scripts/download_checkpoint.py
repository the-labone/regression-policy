# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub==0.25.2"]
# ///
"""Download released RegressionPolicy checkpoints from Hugging Face.

List: uv run scripts/download_checkpoint.py robomimic --list
Download: uv run scripts/download_checkpoint.py robomimic \
    --path state/ht/lift_ph/ht-transformer
GR00T: python scripts/download_checkpoint.py gr00t --path gr1/checkpoint-60000
Cosmos: python scripts/download_checkpoint.py cosmos --all
Pi0.5: python scripts/download_checkpoint.py pi05 \
    --path libero/checkpoint-30000/pretrained_model

Select a full GR00T or Pi0.5 checkpoint directory to include its weights,
processors, normalization statistics, tokenizer, and accompanying license files.
For Cosmos, --all includes the distributed checkpoint, action statistics, and VAE.
"""

import argparse
from pathlib import Path, PurePosixPath

REPOSITORIES = {
    "robomimic": "yuchen0187/RegressionPolicy-RoboMimic",
    "gr00t": "yuchen0187/RegressionPolicy-GR00T",
    "cosmos": "yuchen0187/RegressionPolicy-Cosmos",
    "pi05": "yuchen0187/RegressionPolicy-Pi0.5",
}


def select_files(files, path):
    """Match a repository-relative file or directory, without glob expansion."""
    if path is None:
        return list(files)
    path = path.rstrip("/")
    if not path or path.startswith("/") or ".." in path.split("/"):
        raise ValueError("--path must be a repository-relative file or directory.")
    selected = [
        file
        for file in files
        if file.rfilename == path or file.rfilename.startswith(path + "/")
    ]
    if not selected:
        raise ValueError(f"No files match {path!r}; use --list to see checkpoints.")
    return selected


def checkpoint_paths(files):
    paths = set()
    for file in files:
        path = PurePosixPath(file.rfilename)
        if path.suffix in {".pt", ".pth", ".ckpt"}:
            paths.add(str(path))
        elif path.suffix == ".safetensors":
            paths.add(str(path.parent))
        elif path.name == ".metadata" or path.suffix == ".distcp":
            # Cosmos stores DCP shards in <checkpoint>/model/. Include the
            # checkpoint directory so its metadata and license files travel too.
            directory = path.parent
            if directory.name == "model":
                directory = directory.parent
            paths.add(str(directory))
    return sorted(paths)


def size_label(files):
    if any(file.size is None for file in files):
        return "size unknown"
    size = sum(file.size for file in files)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", choices=REPOSITORIES)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--list", action="store_true", help="List checkpoints.")
    selection.add_argument("--path", help="Repository-relative file or directory.")
    selection.add_argument("--all", action="store_true", help="Download the full repo.")
    parser.add_argument(
        "--revision", default="main", help="Branch, tag, or commit SHA."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Repository download root (default: checkpoints/<repository>).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show selected files without downloading.",
    )
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        parser.error(
            "Run this script with uv: uv run scripts/download_checkpoint.py ..."
        )

    repo_id = REPOSITORIES[args.repository]
    # Resolve once so listing and downloading use the same immutable revision.
    # The Hub client also supports HF_TOKEN and credentials from `hf auth login`.
    info = HfApi().model_info(repo_id, revision=args.revision, files_metadata=True)
    files = sorted(info.siblings, key=lambda file: file.rfilename)
    print(f"Repository: {repo_id}\nRevision: {info.sha}", flush=True)
    if args.list or (args.path is None and not args.all):
        for path in checkpoint_paths(files):
            selected = select_files(files, path)
            print(f"{path}  ({size_label(selected)})")
        print("Use --path FILE_OR_DIRECTORY to download, or --all for the full repo.")
        return

    try:
        selected = select_files(files, args.path)
    except ValueError as error:
        parser.error(str(error))
    output_dir = (args.output_dir or Path("checkpoints") / args.repository).expanduser()
    print(f"Destination: {output_dir.resolve()}")
    print(f"Selected: {len(selected)} files, {size_label(selected)}", flush=True)
    if args.dry_run:
        for file in selected:
            print(file.rfilename)
        return

    destination = snapshot_download(
        repo_id,
        revision=info.sha,
        local_dir=output_dir,
        allow_patterns=[file.rfilename for file in selected],
        max_workers=4,
    )
    print(f"Downloaded to: {Path(destination).resolve()}")
    for path in checkpoint_paths(selected):
        print(f"Checkpoint: {(Path(destination) / path).resolve()}")


if __name__ == "__main__":
    main()
