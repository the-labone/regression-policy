"""Evaluate GR1 with the exact-count diagnostic protocol.

Model/checkpoint construction is separate from the benchmark runner. A factory
receives the JSON kwargs verbatim and must restore the matching processor stats.
For GR00T benchmark comparisons, use the native collector with ht_regression inference;
this CLI uses independent episodes instead of the native autoreset protocol.
"""

import argparse
import importlib
import json

from ..artifacts import write_json
from .diagnostic_runner import GR1DiagnosticRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--factory",
        required=True,
        help="Importable module:function returning (pipeline, processor).",
    )
    parser.add_argument(
        "--factory-kwargs",
        default="{}",
        help="JSON constructor arguments, e.g. checkpoint and device.",
    )
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-envs", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=720)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument(
        "--terminate-on-success", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args(argv)
    module, function = args.factory.split(":", 1)
    kwargs = json.loads(args.factory_kwargs)
    if not isinstance(kwargs, dict):
        raise ValueError("factory-kwargs must be a JSON object.")
    pipeline, processor = getattr(importlib.import_module(module), function)(**kwargs)
    settings = vars(args).copy()
    for key in ("factory", "factory_kwargs", "task"):
        settings.pop(key)
    runner = GR1DiagnosticRunner(processor, tasks=args.task, **settings)
    metrics = runner.run(pipeline)
    write_json(runner.output_dir / "model_source.json", {"factory": args.factory})
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
