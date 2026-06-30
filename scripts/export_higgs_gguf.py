#!/usr/bin/env python3
"""Export HiggsAudioV3 HF weights into the mixed GGUF used by this C++ runtime."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
PY_PROJECT = REPO_ROOT / "projects" / "higgs-audio-ggml-py"

if str(PY_PROJECT) not in sys.path:
    sys.path.insert(0, str(PY_PROJECT))

from higgs_audio_ggml_py.convert import export_higgs_to_gguf
from higgs_audio_ggml_py.paths import DEFAULT_GGUF_PATH, MODEL_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(MODEL_ROOT), help="Hugging Face HiggsAudioV3 model directory")
    parser.add_argument("--out", default=str(DEFAULT_GGUF_PATH), help="Output GGUF path")
    parser.add_argument("--manifest", help="Output manifest path; defaults to OUT with .manifest.json suffix")
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16", help="Base export dtype. f16 keeps a.fc.* and a.q.* in f32.")
    args = parser.parse_args()

    gguf_path, manifest_path = export_higgs_to_gguf(args.model_dir, args.out, args.manifest, dtype=args.dtype)
    print(f"wrote GGUF: {gguf_path}")
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
