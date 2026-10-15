"""Train the released GR00T HT recipe with the native ZeRO-2 trainer."""

import json
import sys

from ..native import check_resources, parser
from .config import RECIPES, resolve


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    # Select the default recipe before parsing the common resource flags.
    import argparse

    selection = argparse.ArgumentParser(add_help=False)
    selection.add_argument("--task", choices=RECIPES, default="gr1")
    task, _ = selection.parse_known_args(argv)
    cli = parser(__doc__, f"configs/experiment/gr00t/{task.task}_ht.json")
    cli.add_argument("--task", choices=RECIPES, default="gr1")
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args(argv)
    check_resources(args)
    config, multiplier = resolve(args)
    if args.check_only:
        print(json.dumps(dict(config=config, sigma_lr_multiplier=multiplier), indent=2))
        return
    from .trainer import train

    train(config, sigma_lr_multiplier=multiplier)


if __name__ == "__main__":
    main()
