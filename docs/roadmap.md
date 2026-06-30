# HiggsAudioV3 C++ Roadmap

## M0: Contract And Skeleton

- Write `docs/spec.md` from the Python staging project.
- Add CMake project with GGML fallback for local development.
- Add shared `HiggsPipeline` boundary.
- Add CLI/server executables with final argument surface.
- Status: complete.

## M1: GGUF Loader

- Load GGUF metadata and tensors.
- Validate required metadata:
  `higgs.config.json`, audio codebook count, audio vocab size.
- Create structured text/codec weight views.
- Add `--inspect` or equivalent CLI validation mode.
- Status: inspect/metadata/tensor-frontier validation is implemented and passes
  on the mixed F16/F32 HiggsAudioV3 GGUF. Text/Qwen3 config and frontier tensor
  inventory also pass `--inspect-text`. Full structured executable weight views
  are still part of graph-stage work.

## M2: Prompt And Sampler

- Port prompt builders for `higgstts`, `chatml`, and `boson-chatml`.
- Port delay/reverse-delay helpers.
- Port deterministic and sampled SGLang-style sampler.
- Validate against Python helper outputs.
- Status: special-token loading, `higgstts`/reference prompt builders, fixed
  ChatML/Boson zero-shot prompt rendering, tokenizer metadata inventory,
  compact tokenizer artifact export/inspection, delay/reverse-delay/finalize
  helpers, and deterministic/sampled logits-to-codebook sampler core are
  ported. They pass `--inspect-tokenizer`, `--self-check-tokenizer`,
  `--inspect-tokenizer-artifact`, `--self-check-tokenizer-artifact`,
  `--self-check-tokenizer-bytelevel`, `--self-check-tokenizer-bpe`,
  `--self-check-tokenizers-cpp`, `--self-check-tokenizer-prompt`,
  `--self-check-chatml-prompt`, `--self-check-prompt`, `--self-check-delay`,
  and `--self-check-sampler`. Tokenizers.cpp is validated against the GGUF HF
  tokenizer JSON and public no-reference generation now builds real prompt ids;
  ChatML/Boson reference-audio prompt contract is a post-goal extension.

## M3: Text Graph

- Port fused text/audio embedding overlay.
- Port Qwen3 backbone graph.
- Port fused 8-codebook audio head.
- Validate final audio logits against Python checks.
- Status: embedding graph, real prompt embedding, block attention, block MLP,
  block residual composition, sequential 36-layer backbone, real prompt
  backbone, audio head graph, and real prompt -> audio-head sampler have CPU
  GGML diagnostic gates. A synthetic embedding -> backbone -> audio-head ->
  sampler gate passes. A no-KV autoregressive recompute loop is wired into
  public no-reference generation. KV cache remains.

## M4: Codec Decode

- Port `codes -> quantizer.decode -> a.fc2 -> acoustic_decoder -> waveform`.
- Write mono 24 kHz WAV.
- Validate waveform parity against Python checks.
- Status: `codes -> quantizer.decode/project_out -> a.fc2 -> a.ad.conv1 ->
  a.ad.block.0..4 -> a.ad.conv2` has a CPU GGML diagnostic gate and writes a
  diagnostic mono 24 kHz WAV. `HiggsPipeline::generate` now wires real
  `higgstts` prompt ids through prompt embedding, no-KV autoregressive text
  graph sampling, and codec decode for no-reference requests. KV cache remains.

## M5: Reference Encode

- Port acoustic encoder.
- Port HuBERT semantic branch.
- Port concat + `a.fc` + RVQ encode.
- Validate relaxed native reference encode gate.
- Status: reference encode inventory is ported and passes
  `--inspect-reference-encode`; acoustic encoder conv1, block0..4, and
  acoustic project tail pass
  `--self-check-codec-encode-conv1-graph`,
  `--self-check-codec-encode-block0-graph` ..
  `--self-check-codec-encode-block4-graph`, and
  `--self-check-codec-encode-project-graph`. Semantic encoder entry conv passes
  `--self-check-codec-encode-semantic-conv-graph`; semantic encoder block0/1
  pass `--self-check-codec-encode-semantic-block0-graph` and
  `--self-check-codec-encode-semantic-block1-graph`. HuBERT feature conv0
  passes `--self-check-codec-encode-hubert-fe-conv0-conv-graph` and
  `--self-check-codec-encode-hubert-fe-conv0-graph`; full HuBERT feature
  extractor passes `--self-check-codec-encode-hubert-fe-graph`; HuBERT feature
  projection passes `--self-check-codec-encode-hubert-fp-graph`; HuBERT
  positional convolution passes `--self-check-codec-encode-hubert-pce-graph`.
  HuBERT encoder prelude passes
  `--self-check-codec-encode-hubert-prelude-graph`; HuBERT encoder layer0
  passes `--self-check-codec-encode-hubert-layer0-graph`; HuBERT encoder layers
  0..11 pass `--self-check-codec-encode-hubert-layers-graph`; hidden-state
  mean/downsample passes `--self-check-codec-encode-hubert-mean-graph`. Concat
  `a.fc` passes `--self-check-codec-encode-fc-graph`; RVQ quantizer0 passes
  `--self-check-codec-encode-quantizer0-graph`; full 8-stage RVQ residual loop
  passes `--self-check-codec-encode-quantizers-graph`. Reference WAV
  input/downmix/resample passes `--self-check-reference-wav PATH`; real WAV
  acoustic tail passes `--self-check-reference-acoustic PATH`; real WAV HuBERT
  semantic tail passes `--self-check-reference-semantic PATH`; local reference
  codes pass `--self-check-reference-codes PATH`.

## M6: Offline CLI And Server

- Wire zero-shot generation through `HiggsPipeline`.
- Wire reference-audio generation through `HiggsPipeline`.
- Implement thin `/generate` server route.
- Keep streaming, batch, CUDA codec, preset voices, and MP3 deferred.
- Status: thin `/generate` route calls `HiggsPipeline` for no-reference and
  reference-audio real-prompt-conditioned no-KV AR WAV generation with `text`,
  `ref_wav`, `ref_text`, `steps`, `temperature`, `top_k`, `top_p`, and `seed`.
  CLI/server parse `prompt_format` and `system_prompt`; zero-shot generation
  supports `higgstts`, `chatml`, and `boson-chatml`, while reference-audio
  generation supports the requested `higgstts` path. Streaming, batch, CUDA
  codec, preset voices, MP3, richer JSON, KV cache, and ChatML/Boson reference
  prompts are post-goal work.

## M7: Backend And Request Scheduler

- Add `--backend cpu|cuda` to CLI/server and carry it through
  `HiggsPipeline` options.
- Replace scattered `ggml_backend_cpu_init()` calls with a shared backend
  factory before claiming CUDA execution.
- Add a bounded server request queue with worker threads so `/generate` can
  accept multiple concurrent clients safely.
- Add request timeout handling and cancellation-aware client writes.
- Add KV cache after the backend boundary is centralized; avoid unnecessary
  host/device copies by keeping graph intermediates on the selected backend.
- Status: planned. First slice should keep CPU behavior unchanged while adding
  the public backend option and server scheduler boundary.
