#!/usr/bin/env python3
"""Quantize Higgs Audio GGUF weights to mostly-Q4_K.

This C++ project entry point reuses the canonical ggbond quantizer so the GGUF
metadata and tensor quantization policy stay identical across Python/C++ trees.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    target = repo_root / "projects/higgs-audio-ggml-py/scripts/cli/quantize_higgs_gguf_q4_k.py"
    if not target.exists():
        raise SystemExit(f"missing canonical quantizer: {target}")

    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
