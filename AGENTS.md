# AGENTS.md

## Mission

`higgs-audio-ggml-cpp` is the C++/GGML runtime target for the HiggsAudioV3
GGUF staged by `projects/higgs-audio-ggml-py`.

The project should preserve the Python staging project's graph, tensor,
metadata, tokenizer, prompt, delay, sampler, and codec decisions while removing
Python, Torch, and ggbond binding dependencies from inference.

## Sources Of Truth

Prefer these in order:

1. `/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf` and metadata.
2. `projects/higgs-audio-ggml-py/docs/vision.md`.
3. `projects/higgs-audio-ggml-py/docs/status.md`.
4. `projects/higgs-audio-ggml-py/higgs_audio_ggml_py/pipeline.py`.
5. `projects/higgs-audio-ggml-py/higgs_audio_ggml_py/runtime.py`.
6. `projects/higgs-audio-ggml-py/higgs_audio_ggml_py/codec.py`.

## Current Boundary

- CPU first.
- CLI and server must route through one shared `HiggsPipeline`.
- First working target is offline zero-shot and reference-audio WAV generation.
- Streaming, batch, CUDA codec, preset voice registry, and MP3 are deferred.
- Tensor shapes must be documented and validated in GGML order
  `(ne0, ne1, ne2, ne3)`.

## GGML Alignment

- Prefer upstream GGML C names and operand order.
- Do not add Torch-style tensor helpers or implicit dimension reversal.
- If source-order host arrays are used, label them as source-order views and
  state their GGML `ne` crossing contract.

## Verification

Initial skeleton smoke:

```bash
cmake -S projects/higgs-audio-ggml-cpp -B projects/higgs-audio-ggml-cpp/build \
  -DHIGGS_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/higgs-audio-ggml-cpp/build -j8
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --help
```
