#!/usr/bin/env python3
"""CPU-only repair and final merge for the completed collected Qwen run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    from stages.collected_attacks import repair_and_finalize_qwen

    repair_and_finalize_qwen(config, config_path)


if __name__ == "__main__":
    main()
