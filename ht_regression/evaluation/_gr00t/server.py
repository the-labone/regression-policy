"""Model-environment subprocess serving the native GR00T wire protocol."""

import argparse
from pathlib import Path

from ..artifacts import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embodiment", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    import random

    import numpy as np
    import torch
    import zmq
    from gr00t.policy.gr00t_policy import Gr00tSimPolicyWrapper
    from gr00t.policy.server_client import PolicyServer

    from ...adapters.checkpoint.gr00t import load_gr00t_checkpoint
    from ...adapters.inference.gr00t import gr00t_policy_inference

    # Match the historical model server: seed RNGs without changing backend
    # algorithms. The native simulator independently seeds its collector.
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    bundle = load_gr00t_checkpoint(
        args.checkpoint,
        embodiment=args.embodiment,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    with gr00t_policy_inference(bundle) as native:
        with PolicyServer(
            Gr00tSimPolicyWrapper(native), host="127.0.0.1", port=0
        ) as server:
            endpoint = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
            write_json(
                args.ready_file,
                dict(port=int(endpoint.rsplit(":", 1)[1]), **bundle.provenance),
            )
            server.run()


if __name__ == "__main__":
    main()
