#!/usr/bin/env python3
"""Download and validate the official Meta-Transformer base encoder."""
import argparse
from pathlib import Path

from mpsl.available_datasets.acquisition import ensure_encoder


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights-dir", type=Path, default=Path("checkpoints"),
        help="Checkpoint folder (default: checkpoints)",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Validate existing weights without downloads or writes",
    )
    args = parser.parse_args(argv)
    try:
        path = ensure_encoder(args.weights_dir, allow_download=not args.validate)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"[encoder] Incomplete: {error}", flush=True)
        return 1
    print(f"[encoder] Ready: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
