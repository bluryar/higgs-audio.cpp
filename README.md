# HiggsAudio GGML C++ Runtime

Experimental C++/GGML runtime for HiggsAudioV3 reference-audio TTS.

This project is a runtime target for GGUF files exported by the local
HiggsAudio GGML Python staging project. It is not a model-weight repository and
does not ship generated audio.

## Status

Supported today:

- CLI generation to mono 24 kHz WAV or MP3.
- Reference-audio TTS with `--ref-wav` and `--ref-text`.
- CPU and CUDA backends through GGML.
- Experimental HTTP server with resident runtime, reference-code cache,
  per-request KV slots, a single CUDA executor, and opt-in audio-head
  micro-batch rendezvous.

Not production-ready:

- The server scheduler is experimental.
- True KV decode batching is not implemented.
- Streaming and voice preset management are not implemented.
- RTF is not yet competitive with optimized production TTS servers.

## Requirements

- CMake 3.22+
- C++17 compiler
- Project submodules:
  - `vendor/ggml`, currently tested with ggml `v0.15.3`
  - `third_party/tokenizers.cpp`
  - `third_party/shine`
- CUDA toolkit only for CUDA builds
- A HiggsAudioV3 GGUF model file

The model file is intentionally external. Pass it with `--model`.

MP3 output uses shine, which is distributed under the GNU Library General
Public License version 2. See `third_party/shine/COPYING`.

## Build

CPU:

```bash
git submodule update --init --recursive
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

CUDA:

```bash
git submodule update --init --recursive
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON
cmake --build build-cuda -j
```

Local ggbond development can reuse the workbench GGML checkout:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DHIGGS_ALLOW_WORKBENCH_GGML_FALLBACK=ON
```

## CLI

Export the mixed GGUF used by this runtime:

```bash
PYTHONPATH=/path/to/ggbond/projects/higgs-audio-ggml-py \
uv run --no-project --with torch --with safetensors --with pyyaml python \
  scripts/export_higgs_gguf.py \
  --model-dir /path/to/higgs-audio-v3-tts-4b \
  --out /path/to/higgs-audio-v3-tts-4b-mixed.gguf
```

Reference-audio generation:

```bash
build-cuda/higgs-audio-cli \
  --backend cuda \
  --model /path/to/higgs-audio-v3-tts-4b-mixed.gguf \
  --ref-wav /path/to/reference.wav \
  --ref-text "reference transcript" \
  --text "text to synthesize" \
  --temperature 0 \
  --steps 120 \
  --no-stop-on-eoc \
  --out /tmp/higgs-output.wav
```

Use an `.mp3` suffix to write MP3 through shine:

```bash
build-cuda/higgs-audio-cli \
  --backend cuda \
  --model /path/to/higgs-audio-v3-tts-4b-mixed.gguf \
  --ref-wav /path/to/reference.wav \
  --ref-text "reference transcript" \
  --text "text to synthesize" \
  --temperature 0 \
  --steps 120 \
  --no-stop-on-eoc \
  --out /tmp/higgs-output.mp3
```

Useful self-check:

```bash
build-cuda/higgs-audio-cli \
  --backend cuda \
  --model /path/to/higgs-audio-v3-tts-4b-mixed.gguf \
  --self-check-audio-head-graph \
  --text x \
  --out /tmp/higgs-unused.wav
```

## Server

Start:

```bash
HIGGS_SERVER_AR_SCHEDULER=1 \
HIGGS_REFERENCE_CODES_CACHE_DIR=/tmp/higgs-reference-cache \
build-cuda/higgs-audio-server \
  --backend cuda \
  --model /path/to/higgs-audio-v3-tts-4b-mixed.gguf \
  --port 8080 \
  --workers 2 \
  --queue-size 4 \
  --request-timeout-sec 600 \
  --ar-scheduler
```

Request JSON response:

```bash
curl -sS -X POST http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  --data-binary '{
    "text":"text to synthesize",
    "ref_wav":"/path/to/reference.wav",
    "ref_text":"reference transcript",
    "temperature":0,
    "steps":120,
    "no_stop_on_eoc":true,
    "out_wav":"/tmp/higgs-server-output.wav",
    "response_format":"json"
  }'
```

For binary MP3 responses, set `"response_format":"mp3"`. For JSON responses
that also write MP3 to disk, pass `"out_mp3":"/tmp/higgs-server-output.mp3"`.

Opt-in audio-head rendezvous diagnostic:

```bash
HIGGS_SERVER_AUDIO_HEAD_BATCH=1 HIGGS_SERVER_AUDIO_HEAD_BATCH_WAIT_MS=100 ...
```

This only batches the ordinary audio-head graph. KV decode remains the main
bottleneck.

## Current Performance Baseline

Measured on an RTX 4060 Ti with CUDA, `temperature=0`, `steps=120`,
`--no-stop-on-eoc`, warm reference-code cache, and two concurrent reference
requests:

- Single CUDA executor + server scheduler: about 14.69 s wall for 8.96 s of
  generated audio, average RTF about 1.64.
- Opt-in audio-head rendezvous with a 100 ms diagnostic window: about 14.24 s
  wall for 8.96 s of generated audio, average RTF about 1.59.
- In that run, audio-head batch hits were real
  (`audio_head_batch_calls=61/58`, `audio_head_batch_size_avg=2`), but the
  overall bottleneck remained reference AR KV decode and serialized CUDA work.

Treat these as experimental baselines, not performance claims.

## Known Limitations

- No model weights are included.
- CUDA server path is single-executor for GGML graph safety.
- Audio-head micro-batching is opt-in and diagnostic.
- KV decode is not truly batched across requests.
- Full-prefix fallback/oracle semantics are preserved; do not tune them for
  speed without byte-identical gates.
- Generated audio quality depends on the converted GGUF and reference prompt.

## Roadmap

- Replace wait-window rendezvous with an explicit step-level scheduler.
- Extend batching to larger decode phases only after byte-identical gates pass.
- Investigate true KV decode batching.
- Keep CLI/server output parity with the current safe baseline.
