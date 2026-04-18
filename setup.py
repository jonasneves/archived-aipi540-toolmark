"""Project setup entry point.

Runs the three pipeline steps end-to-end: fetch benchmarks, build features,
train + evaluate all three models. Idempotent — reruns skip completed steps.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))


def run_pipeline(skip_fetch: bool = False, skip_train: bool = False) -> None:
    from scripts import build_features, make_dataset, model

    if not skip_fetch:
        make_dataset.main()
    build_features.main()
    if not skip_train:
        model.main()


def main() -> None:
    parser = argparse.ArgumentParser(description="Toolmark pipeline runner")
    parser.add_argument("--skip-fetch", action="store_true", help="skip dataset fetch")
    parser.add_argument("--skip-train", action="store_true", help="build features only")
    args = parser.parse_args()
    run_pipeline(skip_fetch=args.skip_fetch, skip_train=args.skip_train)


if __name__ == "__main__":
    main()
