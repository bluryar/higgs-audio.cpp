# HiggsAudioV3 C++ Status

## Current State

- Minimal server/batching prototype is implemented on the existing
  `higgs-audio-server` rather than adding a new framework. The process loads one
  resident `HiggsPipeline(model, backend)` at startup, uses the existing bounded
  request queue, and serializes `generate()` through the pipeline mutex while
  reusing GGUF weights/backend runtime/tokenizer path/reference-code cache/KV
  cache/codec runtime. Request JSON now supports `stop_on_eoc` /
  `no_stop_on_eoc`, optional `out_wav`, optional `codes_json`, and
  `response_format=json`; binary WAV responses remain the default.
- Server verification with CUDA, temperature 0, 120 steps, no stop on EOC, and
  the standard reference wav/text matches the CLI safety baseline exactly:
  delayed codes, finalized codes, and WAV bytes are identical. The first server
  request hit the disk reference-code cache (`latency_ms=8176`, audio `4.48s`,
  RTF `1.82`); the second identical request hit the process memory cache
  (`latency_ms=7229`, RTF `1.61`) and did not repeat reference encode. Three
  sequential warm memory-cache requests took `20.88s` total for `13.44s` of
  audio, average RTF `1.55`, about `0.144 req/s` / `0.644 audio-sec/s`.
- Current throughput gain comes from resident model/backend/runtime state and
  reference-code cache reuse, not from true AR batched decode. The next batching
  gap is an interface/runtime split that can schedule multiple active requests
  through a shared decode step while keeping per-request AR KV state, delayed
  sampler state, fallback/oracle decisions, and codec/postprocess outputs
  byte-identical.
- True multi-request AR scheduler audit: the current reference AR loop is not
  safely batchable at the HTTP/worker layer. Per-request state includes
  prompt/ref ids and embeddings, the AR KV cache and `cache.used`, `hidden`,
  `raw_tn`, `trace_rows`, `trace_steps`, generated embedding/context vectors,
  EOC countdown, fallback/oracle counters, seed/RNG inputs, deadline/cancel, and
  final codec input. The blocker is structural: `BackendRuntime` owns one
  reusable `BackendKvCache`, and `run_reference_text_code_frames_backend_kv`
  mutates that cache directly every decode step. Multiple active requests would
  overwrite KV unless the runtime first supports per-request backend KV caches
  or a cache arena with request slots.
- Added an opt-in server scheduler diagnostic entrypoint rather than a fake
  batch path: `--ar-scheduler` or `HIGGS_SERVER_AR_SCHEDULER=1`. It reports
  `queue_wait_ms`, `scheduler`, `scheduler_batch_size`, and
  `scheduler_step_count`. Until per-request KV caches exist, the scheduler field
  is `blocked_shared_kv_fallback` and generation uses the byte-identical safe
  path. Standard reference validation with CUDA/temperature 0/120 steps/no EOC
  stop matched the CLI baseline exactly for delayed codes, finalized codes, and
  WAV bytes. The opt-in standard-reference request measured
  `latency_ms=8728`, audio `4.48s`, RTF `1.95`; tai_yi also generated through
  the same fallback (`latency_ms=14839`, first request `reference_cache=encoded`).
  No throughput claim is made for this diagnostic path.
- Per-request backend KV slot groundwork: `BackendRuntime` now owns a minimal
  KV-cache slot arena. The default path still uses slot 0, while
  `HIGGS_SERVER_AR_SCHEDULER=1` / `--ar-scheduler` rents a scoped slot for the
  reference backend-KV AR path and reports `backend_kv_slot` plus
  `scheduler=kv_slots_round_robin`. The reference AR helper can now use a
  request-local slot instead of always mutating the single reusable cache.
  Standard reference and tai_yi scheduler-slot runs are byte-identical to their
  CLI baselines for delayed codes, finalized codes, and WAV bytes. Current
  limitation: two concurrent HTTP requests still both report slot 0 because
  `HiggsPipeline::generate` holds `generate_mutex_` around the whole request.
  That keeps outputs safe but prevents proving simultaneous distinct slots.
  Next step is a scheduler-specific entrypoint with narrower locking around
  shared reference cache / weight-loading state, or a dedicated AR-state API
  that bypasses the top-level generate mutex after request-local state is built.
- Scheduler lock split checkpoint: the `HIGGS_SERVER_AR_SCHEDULER=1` reference
  path now keeps default CLI/server behavior under the old long lock, but uses
  short locks for runtime init, reference-code cache access, and codec decode.
  Reference AR decode runs outside `generate_mutex_` with its rented
  request-local KV slot. Shared lazy weight loading is protected by a runtime
  `loaded_weights_mutex`; KV slot rental/release is protected by a slot-pool
  mutex. Standard reference and tai_yi single-request scheduler-slot outputs
  remain byte-identical to CLI baselines. With `--workers 2`, two concurrent
  reference requests reported different slots (`backend_kv_slot=0` and `1`) and
  both matched their baselines exactly for delayed codes, finalized codes, and
  WAV bytes. The two-request wall time was `13.58s` for `8.96s` of audio
  (average RTF `1.52`, throughput `0.147 req/s`, `0.660 audio-sec/s`), with
  per-request latencies `13245ms` and `13440ms`. This is still independent
  per-request KV concurrency, not a shared decode-step micro-batch.
- Scheduler timing/audio-head micro-batch checkpoint: `GenerateResult` and
  server JSON/headers now report `reference_cache_wall_ms`,
  `reference_ar_wall_ms`, `codec_wall_ms`, `total_wall_ms`, `queue_wait_ms`,
  `backend_kv_slot`, and `scheduler`. With CUDA, temperature 0, 120 steps, and
  no EOC stop, standard/tai_yi single-request scheduler runs remained
  byte-identical to CLI baselines. Standard timing was `total=9385ms`,
  `reference_ar=9044ms`, `codec=340ms`; tai_yi was `total=6981ms`,
  `reference_ar=6828ms`, `codec=152ms`. A two-request concurrent run remained
  byte-identical for both references and reported slots `0/1`, wall `13.26s`,
  per-request `reference_ar=12593ms/12899ms`, and `codec=168ms/150ms`. The
  bottleneck is still reference AR, not codec, reference cache, or queue wait.
  `HIGGS_SERVER_AUDIO_HEAD_BATCH=1 --self-check-audio-head-graph` adds an
  opt-in diagnostic: batched audio-head logits/codes matched per-request logits
  and codes exactly (`logits_equal=1`, `codes_equal=1`). This only proves
  mathematical safety of audio-head batching; it is not wired into the scheduler
  and the current diagnostic calls the batched helper once per extracted frame,
  so it should not be used as a throughput estimate.
- Audio-head batch diagnostic was tightened to one real batched graph call:
  `run_audio_head_logits_flat_batched_with_cache(...)` now returns all frame
  logits from one graph, and `run_audio_head_logits_vc_for_frame_with_cache`
  slices that output for existing callers. The opt-in diagnostic
  `HIGGS_SERVER_AUDIO_HEAD_BATCH=1 --self-check-audio-head-graph` reports
  `logits_equal=1`, `codes_equal=1`, `single_ms=3`, `batch_ms=2` on CUDA, so a
  same-step two-hidden audio-head batch is mathematically safe. It is still not
  wired into the server scheduler: current workers own independent request loops
  and have no same-step rendezvous point where two active requests can hand over
  hidden states before sampling. Server responses now expose
  `audio_head_wall_ms`, `audio_head_batch_calls`,
  `audio_head_fallback_calls`, and `audio_head_batch_size_avg`; current
  scheduler responses correctly show `audio_head_batch_calls=0` and
  `audio_head_fallback_calls=120`. Standard/tai_yi single and two-request
  concurrent runs remain byte-identical to baselines. The next real scheduler
  change is a rendezvous queue/barrier for the ordinary single-hidden audio-head
  branch only; fused-logits, window/full-prefix, and existing `head_batch`
  branches should continue to fallback.
- Audio-head rendezvous attempt was not kept on the hot server path. A minimal
  two-request rendezvous for the ordinary single-hidden audio-head branch was
  tried, but concurrent CUDA graph work hit GGML CUDA backend pool failures
  (`GGML_ASSERT(ptr == pool_addr + pool_used)` and `cuMemSetAccess invalid
  argument`). The same class of failure can appear whenever scheduler reference
  AR lets two worker threads allocate/compute GGML CUDA graphs concurrently on
  the shared backend. The safe guard is now a process-local `g_cuda_graph_mutex`
  around the scheduler reference AR CUDA section; this preserves byte-identical
  outputs but serializes AR graph execution, so `audio_head_batch_calls` stays
  `0` and concurrent requests may reuse slot 0. With the guard, a two-request
  standard/tai_yi run is byte-identical to baselines and does not crash
  (`wall=14.80s`, per-request `reference_ar=7239ms/14502ms`). The real next
  prerequisite for hot-path audio-head batching is not the logits math; it is a
  GGML CUDA execution model that supports concurrent graph allocation/compute
  safely, such as per-worker CUDA backends/contexts or one dedicated scheduler
  thread that owns all CUDA graph execution.
- Reference-generation quality alignment is trace-driven, not a listening or
  parameter-tuning task. Fixed points: delayed reference codes, prompt ids,
  raw-frame/finalize contract, masked top1/top5 trace, and Python replay of C++
  prompt/reference inputs are wired. Same-finalized-code codec decode is no
  longer the suspected source of the current "伪音" failure: C++ and Python
  GGML decode agree at `decode_project/conv1` exactly, decoder blocks differ
  only at small residual levels (`block0 p99_abs ~= 0.0147`,
  `waveform p99_abs ~= 0.00046` for the 38-frame KV trace), and waveform stats
  are effectively identical. The no-Torch gate for this is:
  `PYTHONPATH=projects/higgs-audio-ggml-py uv run --no-project --with numpy --with ggbond python projects/higgs-audio-ggml-py/scripts/dev/checks/check_codec_decode_trace_ggml.py --trace-json /tmp/higgs-kv-trace-40.json --backend cpu --out-wav /tmp/higgs-kv40-codec-python-ggml-fresh.wav`,
  followed by a waveform diff against C++ `--decode-trace-json` output. The
  observed C++/Python GGML waveform diff is `mean_abs ~= 7.0e-5`,
  `p99_abs ~= 4.6e-4`, `max_abs ~= 0.0019`. The current first
  output-affecting divergence is
  reference generation, not codec decode: with the same prompt/reference inputs,
  C++ backend KV and Python/default full recompute first differ at generated
  step 3, codebook 3, where Python/full chooses `907` and KV chooses `165`;
  the top5 order is the same pair reversed with a small margin. The local gate
  for this is:
  `uv run --no-project python projects/higgs-audio-ggml-py/scripts/dev/checks/check_reference_trace_parity.py /tmp/higgs-default-current-4.json /tmp/higgs-kv-no-fallback-4.json`.
  Because that KV path is still known-divergent, `HIGGS_REFERENCE_BACKEND_KV=1`
  is now guarded as trace-only diagnostic mode and requires both
  `HIGGS_TRACE_ONLY=1` and `HIGGS_TRACE_JSON`; it must not be used to emit WAVs
  for listening tests until the step3/codebook3 divergence is fixed.
  Negative experiment: forcing the relevant one-token linear projections away
  from CUDA `mul_mat_vec_f` is not a clean fix. A global disable changed earlier
  audio codes, and a local wide-linear graph experiment only increased graph
  complexity without producing a verified step3 fix, so this path was not kept.
  A narrower repeat that widened only KV block Q/K/V projections also kept the
  step3/codebook3 mismatch (`165` instead of `907`), so no wide-QKV code was
  retained. Additional negative gates: disabling TF32 with
  `NVIDIA_TF32_OVERRIDE=0`, running backend KV on CPU, forcing a per-frame
  backend KV refresh, and trying the existing per-layer backend KV decode path
  all keep the same step3/codebook3 `907 -> 165` flip. So this is not a
  CUDA-only TF32 issue, not just stale cache contents, and not fixed by changing
  all-layer decode into per-layer decode. A narrower block0 Q/K/V shape gate
  also rules out the simple "single-token `ggml_mul_mat` kernel is inherently
  different" theory: `--self-check-block0-attn-qkv-graph` with identical input
  columns and `HIGGS_ATTENTION_TOKENS=1` versus `2` produces exact equality for
  `q_proj`, `k_proj`, `v_proj`, and the post-rope repeated `q/k/v` dumps.
  With the real step3 `attn_normed` vector fed directly into that QKV gate via
  `HIGGS_ATTENTION_NORMED_INPUT`, the direct `q_proj` matches the backend KV
  dump much more closely (`p99_abs ~= 5.1e-5`) than the full-sequence dump
  (`p99_abs ~= 0.0014`). This means the old "KV q_proj is simply wrong"
  interpretation is too narrow: the remaining quality target is specifically
  the full-sequence numerical path, not an isolated single-token matmul bug.
  A correctness-only experiment that refreshed the KV path's next hidden from
  the full-prefix path every frame was stopped: after about 70 seconds it had
  only reached two trace rows for a 4-step run. That is still full recompute in
  disguise and is not a viable fix. Any refresh fallback has to be a true
  bounded-window refresh, not full-prefix recompute.
  The current dump-drift gate is:
  `uv run --no-project python projects/higgs-audio-ggml-py/scripts/dev/checks/check_reference_dump_drift.py --full-dir /tmp/higgs-full-stepdump4-cuda --kv-dir /tmp/higgs-kv-stepdump4-cuda --step 3 --used 241`.
  It shows the step3 audio-head input is already different before sampling
  (`output_norm` final hidden `p99_abs ~= 0.0217`, logits `p99_abs ~= 0.3269`,
  max logit drift `~= 0.4976`), enough to flip codebook 3 `907/165`. So the
  remaining root cause is KV/full hidden drift before the audio head, not the
  audio head layout or sampler.
  With the block0 internal full dump aligned to KV `past=240`, the first
  nonzero drift is now localized further: `blk0_input` and `blk0_attn_normed`
  are exactly equal, then `blk0_q_proj` drifts (`p99_abs ~= 0.0014`) and
  `blk0_k_cur` drifts (`p99_abs ~= 0.0143`). This small current-token projection
  drift enters the residual stack and reaches `audio_head_logits_flat p99_abs
  ~= 0.3269` by step3. The updated gate is:
  `uv run --no-project python projects/higgs-audio-ggml-py/scripts/dev/checks/check_reference_dump_drift.py --full-dir /tmp/higgs-full-block0-step3-cuda-v2 --kv-dir /tmp/higgs-kv-stepdump4-cuda --step 3 --past 240 --used 241`.
  Public reference generation no longer routes through that known-divergent KV
  path by default. A current audit with the target reference wav/text and
  `HIGGS_TRACE_ONLY=1 HIGGS_TRACE_JSON=/tmp/higgs-public-default-audit-4.json`
  matches the known-good full/Python 4-step trace exactly, while attempting to
  set `HIGGS_REFERENCE_BACKEND_KV=1` without trace-only exits with the
  diagnostic-only guard and does not write a WAV. The old KV path remains
  available only for trace comparison.
  The consolidated audit command is:
  `uv run --no-project python projects/higgs-audio-ggml-py/scripts/dev/checks/check_reference_public_audit.py`.
  It checks public default trace parity, the diagnostic KV guard, and same-code
  C++/Python GGML codec waveform drift.
  A deterministic block0 attention gate with
  the same synthetic hidden/positions now aligns C++ vs Python GGML at
  `attn_out max_abs ~= 0.0019`, so the standalone attention op path is not the
  first blocker. The current real reference hot-path first divergence is the
  one-token backend KV decode after the initial BOC frame: C++ `kv_decode_blk.0`
  already differs from Python full-sequence `blk.0.hidden`, and 20-step
  temperature-0 trace still diverges at logits top1 from step0 and sampled codes
  from step3. Backend KV offset calculation was changed to use GGML `nb[]`
  strides, but this did not change the divergence. Next action: compare layer0
  backend KV decode K/V reads and attention output against the full-sequence
  Python layer0 tensors, then fix only that cache/decode mismatch.
- Project skeleton exists at `projects/higgs-audio-ggml-cpp`.
- `docs/spec.md` records the Python-to-C++ handoff contract:
  GGUF metadata, prompt formats, delay/sampler, codec encode/decode, and GGML
  shape expectations.
- `HiggsPipeline` is the shared CLI/server boundary.
- Server workers now share one process-resident `HiggsPipeline` constructed at
  server startup instead of rebuilding it per request. `HiggsPipeline::generate`
  is serialized by a small mutex while the backend/runtime state is shared.
- `HiggsPipeline` now owns a small process-resident runtime: `TextConfig` plus
  one lazily initialized backend runtime per `cpu`/`cuda`, each holding the GGUF
  tensor metadata, backend, backend weight buffer, and loaded-weight set. Each
  backend runtime also owns one reusable backend-resident KV cache that grows to
  the largest requested capacity and resets `used` between serialized generate
  calls. CLI/server construct `HiggsPipeline(model, backend)` to prewarm the
  selected backend at startup; a JSON backend override can still lazily
  initialize the other backend.
- CLI argument surface exists:
  `--model`, `--text`, `--ref-wav`, `--ref-text`, `--prompt-format`,
  `--system-prompt`, `--temperature`, `--top-k`, `--top-p`, `--seed`,
  `--steps`, `--out`, and `--no-stop-on-eoc`.
- Server executable is a thin `/generate` wrapper over `HiggsPipeline`.
- Server now accepts multiple clients through a bounded request queue and worker
  threads: `--workers`, `--queue-size`, and `--request-timeout-sec`.
  A two-request local smoke passes. `SIGINT`/`SIGTERM` now wakes the blocking
  `accept` by shutting down the listen socket.
- CLI/server parse `--backend cpu|cuda`, and server JSON may override
  `"backend"`. GGML graph helpers now use a shared request-local backend
  factory. CPU and CUDA `--steps 1` CLI/server smokes are verified. Codec
  waveform decode runs on CUDA after casting decoder `conv_transpose_1d`
  weights to F32 for the GGML CUDA kernel.
- Hot no-reference/reference AR paths now keep request-local backend state in
  `BackendKvCache`: backend, KV tensors, GGUF tensor metadata, weight buffer,
  loaded-weight set, and `TextConfig`. This removes repeated per-layer GGUF
  opens and repeated hot-path weight loads for backend KV decode, prompt
  prefill, and audio head sampling.
- The hot no-reference/reference AR decode step now uses one all-layer backend
  KV graph per generated token instead of 36 per-layer graph computes. The
  older per-layer helper remains as a diagnostic path.
- Hot no-reference/reference prompt prefill now uses the process-resident
  `BackendRuntime` for embedding weights/backend and one all-layer 36-block
  prefill graph for the Qwen3 backbone. The older `run_prompt_last_hidden`,
  `run_reference_embedding_values`, and per-layer block helpers remain
  diagnostic paths.
- Hot no-reference/reference backend KV generation now pre-fills the full
  prompt/reference token sequence into `BackendKvCache` instead of writing only
  the last prompt hidden state at `pos=0`. The prefill graph writes each
  layer's full K/V span in GGML order
  `(ne0=head_dim, ne1=kv_heads, ne2=capacity, ne3=layers)`, sets
  `cache.used=prompt_tokens`, then AR decode advances from that position.
- Hot AR decode now feeds each sampled audio-code frame through the runtime
  audio embedding graph before the next one-token KV decode. The previous
  hidden-state self-feed path was a diagnostic shortcut and produced near-silent
  codec output.
- Hot AR decode now also matches the SGLang delay sampler's initial audio
  context: after prompt/reference KV prefill, it feeds one all-BOC audio frame
  through audio embedding and one-token KV decode before sampling the first
  generated frame. This avoids sampling the first frame directly from the final
  prompt hidden state.
- Setting `HIGGS_PROFILE=1` prints phase timings for
  `prompt_prefill_embedding`, `reference_prefill_embedding`,
  `prefill_backbone_all_layers`, and total prompt/reference prefill. On CUDA
  with `/root/code/ggbond/models/higgs-audio-v3-tts-4b-f16.gguf`, hot no-ref
  timings for text `你好，测试一下中文语音合成。` are: 20 steps `0:06.15`
  (`prompt_prefill_total=2767 ms`), 40 steps `0:09.98`
  (`prompt_prefill_total=3114 ms`), and 80 steps `0:21.36`
  (`prompt_prefill_total=2870 ms`). The first cold 20-step process measured
  `0:20.01` with `prompt_prefill_total=15206 ms`.
- After the full-KV/audio-embedding fix, the reference prompt using
  `/root/code/ggbond/models/可哪怕位于堂堂超一品官职,在十二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应.wav`
  generates non-silent output at `/tmp/higgs-fix-full-reference-exact.wav`:
  duration `6.4s`, elapsed `75.01s`, RTF `11.72`, `max_abs_pcm=18064`,
  `mean_abs_pcm=1450.49`, `rms_pcm=2424.13`, `nonzero_ratio=0.872`.
  A 40-step diagnostic file `/tmp/higgs-fix-ref40.wav` measured
  `max_abs_pcm=18112`, versus the earlier broken full output's `max_abs_pcm=113`.
  With `HIGGS_PROFILE=1`, a 20-step reference diagnostic reports
  `generated_codes frames=20 codebooks=8 unique=118`.
  After adding the initial BOC-frame decode, `/tmp/higgs-boc-full-reference.wav`
  measures duration `6.4s`, elapsed `73.58s`, RTF `11.50`,
  `max_abs_pcm=26847`, `mean_abs_pcm=2704.38`, `rms_pcm=4383.33`,
  `nonzero_ratio=0.990`; the 40-step diagnostic
  `/tmp/higgs-boc-ref40.wav` reports `generated_codes frames=40 codebooks=8
  unique=221`.
  Runtime is still slow because per-step audio embedding and one-token decode
  remain separate graph computes.
- M1 GGUF inspect/validate is implemented and passes on
  `/root/code/ggbond/models/higgs-audio-v3-tts-4b-f16.gguf`.
- Prompt special-token loading and `higgstts`/reference prompt-id builders are
  ported and pass `--self-check-prompt`.
- Delay/reverse-delay/finalize helpers are ported with GGML-order
  `CodeMatrix(ne0=codebooks, ne1=frames)` and pass `--self-check-delay`.
- Deterministic/sampled logits-to-codebook sampler core is ported, passes
  `--self-check-sampler`, and is wired to the current synthetic text-logits
  path through `temperature`/`top_k`/`top_p`/`seed`.
- Text/Qwen3 metadata and frontier tensor inventory is ported and passes
  `--inspect-text`; tensor shapes are read from GGML tensor metadata and
  validated in GGML `ne` order.
- Tokenizer metadata inventory is ported and passes `--inspect-tokenizer` and
  `--self-check-tokenizer`; GGUF contains full `tokenizer.huggingface.json`, but
  `tokenizer.ggml.tokens` is not a dense runtime vocabulary.
- Compact tokenizer artifact export is available at
  `scripts/export_tokenizer_artifact.py`; C++ validates the artifact with
  `--inspect-tokenizer-artifact` and exact-token lookup with
  `--self-check-tokenizer-artifact`. ByteLevel byte mapping passes
  `--self-check-tokenizer-bytelevel`; single pre-token BPE merging passes
  `--self-check-tokenizer-bpe`.
- `tokenizers.cpp` is vendored and the GGUF `tokenizer.huggingface.json` path
  passes `--self-check-tokenizers-cpp` for representative Qwen/Higgs text and
  special-token ids.
- Real `higgstts` text prompt ids pass `--self-check-tokenizer-prompt` and are
  built at the public no-reference `HiggsPipeline::generate` entry.
- Audio-head flat logits layout helper is ported and passes
  `--self-check-audio-logits`.
- Text/audio embedding id preparation is ported and passes
  `--self-check-embedding-ids`.
- CPU GGML embedding graph probe loads `token_embd.weight` and `a.token_embd`,
  runs `ggml_get_rows` plus summed audio codebook embeddings plus `ggml_concat`,
  and passes `--self-check-embedding-graph`.
- CPU GGML prompt embedding graph accepts real `higgstts` prompt ids and passes
  `--self-check-prompt-embedding-graph` with output shape
  `(ne0=2560, ne1=prompt_tokens)`.
- CPU GGML prompt backbone graph runs real prompt embeddings through all 36
  Qwen3 blocks and passes `--self-check-prompt-backbone-graph`.
- CPU GGML prompt audio-head graph takes the final real prompt token hidden
  state through `output_norm -> a.output -> sampler` and passes
  `--self-check-prompt-audio-head-graph`.
- CPU GGML prompt AR graph runs a small no-KV autoregressive recompute loop and
  passes `--self-check-prompt-ar-graph`.
- Text KV cache storage contract is introduced as
  `(ne0=head_dim, ne1=kv_heads, ne2=capacity, ne3=layers)` and passes
  `--self-check-kv-cache`. The AR graph still needs to consume it.
- Layer-indexed one-token KV decode graph reads past K/V plus current K/V,
  runs attention output plus residual MLP, and passes
  `--self-check-kv-decode-graph` for all 36 blocks on both CPU and CUDA builds.
  This proves the cache read path and GGML `ggml_concat(..., dim=2)` sequence
  before wiring the AR loop to KV cache.
- Full 36-layer one-token KV decode chain feeds the audio head sampler and
  passes `--self-check-kv-decode-audio-head-graph` on both CPU and CUDA builds.
- Real higgstts prompt last hidden now feeds the 36-layer KV decode chain and
  audio head sampler, passing
  `--self-check-prompt-to-kv-decode-audio-head-graph` on both CPU and CUDA
  builds.
- Real higgstts prompt last hidden can now write per-layer K/V into the
  diagnostic host-side `TextKvCache`, passing
  `--self-check-prompt-kv-cache-write` on both CPU and CUDA builds. This is a
  diagnostic gate only: it intentionally uses backend tensor readback and is
  not the final backend-resident runtime KV path.
- The diagnostic `TextKvCache` can now drive a one-token 36-layer decode that
  reads real prompt K/V from the cache, writes the current token K/V, and feeds
  the audio head sampler. `--self-check-prompt-kv-cache-decode` passes on both
  CPU and CUDA builds. This remains diagnostic because K/V still round-trips
  through host vectors.
- Backend-resident KV cache tensors are introduced with GGML order
  `(ne0=head_dim, ne1=kv_heads, ne2=capacity, ne3=layers)`. A minimal
  `ggml_set_inplace` write gate passes `--self-check-backend-kv-cache-write` on
  both CPU and CUDA builds, proving graph-side cache updates can avoid host
  readback.
- Backend-resident KV decode now reads past K/V through `ggml_view_3d` on the
  cache tensors, writes current K/V with `ggml_set_inplace`, and completes a
  36-layer one-token decode. `--self-check-backend-kv-cache-decode` passes on
  both CPU and CUDA builds.
- Real prompt last hidden can now prefill backend-resident K/V through
  `ggml_set_inplace` and feed the backend cache decode path. The gate
  `--self-check-backend-prompt-kv-cache-prefill` passes on both CPU and CUDA
  builds.
- Public `HiggsPipeline::generate` now uses backend-resident KV AR for both
  no-reference and `higgstts` reference-audio generation. The gates
  `--self-check-prompt-ar-backend-kv` and
  `--self-check-reference-ar-backend-kv` pass on both CPU and CUDA builds, and
  CLI/server one-step smoke tests produce WAV output.
- CPU GGML audio-head graph probe loads `output_norm.weight` and `a.output`,
  runs `ggml_rms_norm -> ggml_mul -> ggml_mul_mat -> audio_logits_flat_to_vc ->
  sampler`, and passes `--self-check-audio-head-graph`.
- CPU GGML block0 MLP graph probe loads `blk.0.ffn_norm/gate/up/down`, runs
  `ggml_rms_norm -> gate/up -> ggml_silu gate -> down`, and passes
  `--self-check-block0-mlp-graph`.
- CPU GGML block0 attention QKV graph probe loads block0 attention norm,
  Q/K/V projection, and Q/K norm tensors, runs Q/K/V projection, Q/K RMSNorm,
  Q/K RoPE, and KV repeat, then passes
  `--self-check-block0-attn-qkv-graph`.
- CPU GGML block0 attention graph probe extends QKV through
  `ggml_mul_mat(k4,q4) -> ggml_scale -> ggml_diag_mask_inf -> ggml_soft_max ->
  ggml_mul_mat(v,kq) -> attn_output`, and passes
  `--self-check-block0-attn-graph`.
- CPU GGML block0 graph probe composes `hidden + attn`, runs MLP on that
  residual state, adds the final residual, and passes
  `--self-check-block0-graph`.
- The full-block probe is layer-indexed; the final block also passes
  `--self-check-block35-graph`.
- Sequential CPU GGML backbone probe runs all 36 Qwen3 blocks over synthetic
  `(ne0=2560, ne1=2)` hidden states one layer at a time and passes
  `--self-check-backbone-graph`.
- Synthetic text-logits probe runs embedding -> 36-layer backbone -> audio head
  -> sampler and passes `--self-check-text-logits-graph`.
- Codec decode project probe runs synthetic codec ids through 8 quantizer
  codebook/project-out paths, sums them, applies `a.fc2`, and passes
  `--self-check-codec-project-graph`.
- Codec decoder conv1 probe extends codec project through `a.ad.conv1` and
  passes `--self-check-codec-conv1-graph`.
- Codec decoder block0 conv_t1 probe extends conv1 through snake activation,
  `ggml_conv_transpose_1d`, crop, and bias add, then passes
  `--self-check-codec-block0-conv-t1-graph`.
- Codec decoder block0 res_unit1 probe extends block0 conv_t1 through the first
  residual unit and passes `--self-check-codec-block0-res-unit1-graph`.
- Codec decoder block0 graph probe extends through residual units 1/2/3 with
  dilations 1/3/9 and passes `--self-check-codec-block0-graph`.
- Codec decoder block1 graph probe extends synthetic decode through decoder
  blocks 0 and 1, then passes `--self-check-codec-block1-graph`.
- Codec decoder waveform graph probe extends synthetic decode through decoder
  blocks 0-4 and final `a.ad.conv2`, then passes
  `--self-check-codec-waveform-graph`.
- Codec WAV probe writes the synthetic decoder output as mono 24 kHz 16-bit PCM
  WAV and passes `--self-check-codec-wav --out /tmp/higgs-codec-self-check.wav`.
- `HiggsPipeline::generate` has a temporary real prompt-conditioned
  no-reference path: tokenizer -> `higgstts`/ChatML/Boson prompt ids -> prompt
  embedding -> backend-resident KV autoregressive loop over the 36-layer text
  graph -> audio-head sampler -> delay/finalize -> codec decode -> WAV. CLI
  and server sampler options feed the audio-head sampler.
- `HiggsPipeline::generate` has a reference-audio path for `--ref-wav`: local
  reference codec encode -> reference prompt ids/audio embeddings ->
  backend-resident KV AR -> codec decode -> WAV. `--ref-text` is included when
  provided.
- `prompt_format` is parsed by CLI/server. Zero-shot generation supports
  `higgstts`, `chatml`, and `boson-chatml`; reference-audio generation currently
  supports only `higgstts` and fails explicitly for ChatML/Boson.
- Reference encode inventory is ported and passes `--inspect-reference-encode`
  for acoustic encoder, HuBERT semantic model, semantic encoder, concat `a.fc`,
  `a.fc1`/`a.fc2`, and RVQ tensor counts/frontier shapes.
- Acoustic encoder conv1 is ported as a CPU GGML diagnostic gate and passes
  `--self-check-codec-encode-conv1-graph`.
- Acoustic encoder block0 is ported as a CPU GGML diagnostic gate
  `conv1 -> res_unit1/2/3 -> snake -> stride-8 conv1` and passes
  `--self-check-codec-encode-block0-graph`.
- Acoustic encoder block1 extends the same gate through stride-5 block1 and
  passes `--self-check-codec-encode-block1-graph`.
- Acoustic encoder block2/block3/block4 and acoustic project tail are ported as
  CPU GGML diagnostic gates and pass
  `--self-check-codec-encode-block2-graph`,
  `--self-check-codec-encode-block3-graph`,
  `--self-check-codec-encode-block4-graph`, and
  `--self-check-codec-encode-project-graph`.
- Semantic encoder entry conv is ported as a CPU GGML diagnostic gate and passes
  `--self-check-codec-encode-semantic-conv-graph`.
- Semantic encoder block0/block1 are ported as CPU GGML diagnostic gates and
  pass `--self-check-codec-encode-semantic-block0-graph` and
  `--self-check-codec-encode-semantic-block1-graph`.
- HuBERT feature extractor conv0 is ported as CPU GGML diagnostic gates and
  passes `--self-check-codec-encode-hubert-fe-conv0-conv-graph` and
  `--self-check-codec-encode-hubert-fe-conv0-graph`.
- HuBERT feature extractor layers 0..6 are ported as a CPU GGML diagnostic gate
  and pass `--self-check-codec-encode-hubert-fe-graph`.
- HuBERT feature projection is ported as a CPU GGML diagnostic gate and passes
  `--self-check-codec-encode-hubert-fp-graph`.
- HuBERT positional convolution is ported as a CPU GGML diagnostic gate and
  passes `--self-check-codec-encode-hubert-pce-graph`.
- HuBERT encoder prelude normalization is ported as a CPU GGML diagnostic gate
  and passes `--self-check-codec-encode-hubert-prelude-graph`.
- HuBERT encoder layer0 is ported as a CPU GGML diagnostic gate and passes
  `--self-check-codec-encode-hubert-layer0-graph`.
- HuBERT encoder layers 0..11 are ported as a CPU GGML diagnostic gate and pass
  `--self-check-codec-encode-hubert-layers-graph`.
- HuBERT hidden-state mean/downsample is ported as a CPU GGML diagnostic gate
  and passes `--self-check-codec-encode-hubert-mean-graph`.
- Concat `a.fc` is ported as a CPU GGML diagnostic gate and passes
  `--self-check-codec-encode-fc-graph`.
- RVQ quantizer0 encode is ported as a CPU GGML diagnostic gate and passes
  `--self-check-codec-encode-quantizer0-graph`.
- Full 8-stage RVQ encode residual loop is ported as a CPU GGML diagnostic gate
  and passes `--self-check-codec-encode-quantizers-graph`.
- Reference WAV input/downmix/resample boundary is ported and passes
  `--self-check-reference-wav PATH`, producing 24 kHz acoustic and 16 kHz
  semantic buffers.
- Real reference WAV acoustic encoder tail is ported as a CPU GGML diagnostic
  gate and passes `--self-check-reference-acoustic PATH`.
- Real reference WAV HuBERT semantic encoder tail is ported as a CPU GGML
  diagnostic gate and passes `--self-check-reference-semantic PATH`.
- Real reference WAV acoustic+semantic alignment, concat `a.fc`, and 8-stage
  RVQ encode are ported as a CPU GGML diagnostic gate and pass
  `--self-check-reference-codes PATH`.

## Post-Goal Follow-Ups

- ChatML/Boson reference-audio prompt contract. The current requested
  reference-audio path is supported through `higgstts`.
- Further reduce host/device boundaries around sampled code output and WAV
  materialization.
- Finer-grained request cancellation. Current cancellation/timeout checks run
  at request entry and AR frame boundaries; a single in-flight GGML graph
  compute is not interrupted.
- Per-worker or per-backend runtime pools. The current server keeps one shared
  `HiggsPipeline` and serializes `generate`; add a pool only if concurrent
  in-flight GGML generation becomes necessary.
- Reusable graph objects. The runtime now reuses backend, weights, and KV
  storage, but still rebuilds per-step graph objects for decode/audio-head.
- `/generate` JSON surface beyond `text`, `steps`, `temperature`, `top_k`,
  `top_p`, `seed`, `ref_wav`, `ref_text`, `prompt_format`, and
  `system_prompt`.

## Verification

Passed:

```bash
cmake -S projects/higgs-audio-ggml-cpp -B projects/higgs-audio-ggml-cpp/build \
  -DHIGGS_ALLOW_WORKBENCH_GGML_FALLBACK=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build projects/higgs-audio-ggml-cpp/build -j8
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --help
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli \
  --model /root/code/ggbond/models/higgs-audio-v3-tts-4b-f16.gguf \
  --inspect --validate-tensors
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --inspect-text
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-wav --out /tmp/higgs-codec-self-check.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-waveform-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-block1-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-block0-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-block0-res-unit1-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-block0-conv-t1-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-conv1-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-project-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --inspect-reference-encode
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-conv1-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-block0-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-block1-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-block2-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-block3-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-block4-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-project-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-semantic-conv-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-semantic-block0-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-semantic-block1-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-fe-conv0-conv-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-fe-conv0-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-fe-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-fp-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-pce-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-prelude-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-layer0-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-layers-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-hubert-mean-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-fc-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-quantizer0-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-codec-encode-quantizers-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-reference-wav /tmp/higgs-ref-reader-input.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-reference-acoustic /tmp/higgs-ref-acoustic-input.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-reference-semantic /tmp/higgs-ref-semantic-input.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-reference-codes /tmp/higgs-ref-codes-input.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --ref-wav /tmp/higgs-ref-generate-input.wav --ref-text 参考音频 --steps 1 --out /tmp/higgs-ref-generate.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18081
curl -sS -X POST http://127.0.0.1:18081/generate -H 'Content-Type: application/json' \
  -d '{"text":"你好","ref_wav":"/tmp/higgs-server-ref-input.wav","ref_text":"参考音频","steps":1}' \
  -o /tmp/higgs-server-ref.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-text-logits-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-backbone-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-backbone-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-audio-head-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-ar-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-block0-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-block35-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-block0-attn-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-block0-attn-qkv-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-block0-mlp-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-audio-head-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-embedding-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-embedding-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-embedding-ids
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-audio-logits
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-sampler
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --inspect-tokenizer
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-tokenizer
uv run --no-project --with gguf projects/higgs-audio-ggml-cpp/scripts/export_tokenizer_artifact.py --out /tmp/higgs-tokenizer.hatk
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --inspect-tokenizer-artifact /tmp/higgs-tokenizer.hatk
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-tokenizer-artifact /tmp/higgs-tokenizer.hatk
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-tokenizer-bytelevel /tmp/higgs-tokenizer.hatk
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-tokenizer-bpe /tmp/higgs-tokenizer.hatk
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-tokenizers-cpp
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-tokenizer-prompt
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-chatml-prompt
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-embedding-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-backbone-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-ar-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-kv-cache
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-kv-decode-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-kv-decode-audio-head-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-to-kv-decode-audio-head-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-kv-cache-write
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-kv-cache-decode
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-backend-kv-cache-write
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-backend-kv-cache-decode
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-backend-prompt-kv-cache-prefill
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-ar-backend-kv
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-reference-ar-backend-kv
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-delay
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text test --steps 2 --temperature 1.0 --top-k 16 --top-p 0.8 --seed 42 --out /tmp/higgs-sampled-cli.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text test --steps 3 --out /tmp/higgs-steps3.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text test --out /tmp/higgs-test.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --steps 1 --out /tmp/higgs-zero-shot-check.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --prompt-format chatml --steps 1 --out /tmp/higgs-chatml-check.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --prompt-format boson-chatml --steps 1 --out /tmp/higgs-boson-chatml-check.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --ref-wav /tmp/higgs-ref-input-check.wav --ref-text 参考音频 --steps 1 --out /tmp/higgs-ref-check.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --prompt-format chatml --ref-wav /tmp/higgs-ref-chatml-input.wav --steps 1 --out /tmp/should-not-exist.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --steps 1 --out /tmp/higgs-final-zero.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --backend cpu --text 你好 --steps 1 --out /tmp/higgs-scheduler-cli-cpu.wav
cmake -S projects/higgs-audio-ggml-cpp -B projects/higgs-audio-ggml-cpp/build-cuda -DHIGGS_ALLOW_WORKBENCH_GGML_FALLBACK=ON -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build projects/higgs-audio-ggml-cpp/build-cuda -j8
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --steps 1 --out /tmp/higgs-cancel-regress-cuda.wav
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-kv-decode-graph
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-kv-decode-audio-head-graph
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-prompt-to-kv-decode-audio-head-graph
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-prompt-kv-cache-write
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-prompt-kv-cache-decode
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-backend-kv-cache-write
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-backend-kv-cache-decode
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-backend-prompt-kv-cache-prefill
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-prompt-ar-backend-kv
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-reference-ar-backend-kv
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --self-check-codec-waveform-graph
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --steps 1 --out /tmp/higgs-codec-cuda-no-fallback.wav
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --ref-wav /root/code/ggbond/models/可哪怕位于堂堂超一品官职,在十二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应.wav --ref-text 参考音频 --steps 1 --out /tmp/higgs-ref-cuda-codec-no-fallback.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --backend cpu --text 你好 --steps 1 --out /tmp/higgs-final-audit-cpu.wav
projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --steps 1 --out /tmp/higgs-final-audit-cuda.wav
/usr/bin/time -f 'elapsed=%E maxrss=%M' projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --steps 1 --out /tmp/higgs-cachecfg-steps1.wav
/usr/bin/time -f 'elapsed=%E maxrss=%M' projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --steps 20 --out /tmp/higgs-cachecfg-steps20.wav
/usr/bin/time -f 'elapsed=%E maxrss=%M' projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --steps 40 --out /tmp/higgs-cachecfg-steps40-solo.wav
/usr/bin/time -f 'elapsed=%E maxrss=%M' projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --backend cuda --text 你好 --steps 80 --out /tmp/higgs-cachecfg-steps80-solo.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --prompt-format chatml --steps 1 --out /tmp/higgs-final-chatml.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --prompt-format boson-chatml --steps 1 --out /tmp/higgs-final-boson.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --text 你好 --ref-wav /tmp/higgs-final-ref-input.wav --ref-text 参考音频 --steps 1 --out /tmp/higgs-final-ref.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18080  # in another shell
curl -sS -X POST http://127.0.0.1:18080/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"test","steps":2,"temperature":1.0,"top_k":16,"top_p":0.8,"seed":42}' \
  -o /tmp/higgs-server.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18084  # in another shell
curl -sS -X POST http://127.0.0.1:18084/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","prompt_format":"chatml","steps":1}' \
  -o /tmp/higgs-server-chatml-response.txt
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18086  # in another shell
curl -sS -X POST http://127.0.0.1:18086/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","prompt_format":"chatml","steps":1}' \
  -o /tmp/higgs-final-server-chatml.wav
curl -sS -X POST http://127.0.0.1:18086/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","ref_wav":"/tmp/higgs-final-ref-input.wav","ref_text":"参考音频","steps":1}' \
  -o /tmp/higgs-final-server-ref.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18087 --backend cpu --workers 2 --queue-size 4 --request-timeout-sec 120
curl -sS -X POST http://127.0.0.1:18087/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","steps":1}' \
  -o /tmp/higgs-scheduler-server-a.wav
curl -sS -X POST http://127.0.0.1:18087/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"世界","steps":1}' \
  -o /tmp/higgs-scheduler-server-b.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18094 --backend cpu --workers 1 --queue-size 2 --request-timeout-sec 1
curl -sS -w '\nHTTP:%{http_code}\n' -X POST http://127.0.0.1:18094/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"你好","steps":4}' \
  -o /tmp/higgs-timeout-body-504.txt
# returns HTTP 504 with "generate failed: request timed out"; SIGTERM exits cleanly.
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18120 --backend cpu --workers 2 --queue-size 4 --request-timeout-sec 120
curl -sS -X POST http://127.0.0.1:18120/generate -H 'Content-Type: application/json' -d '{"text":"你好","steps":1}' -o /tmp/higgs-final-server-a.wav
curl -sS -X POST http://127.0.0.1:18120/generate -H 'Content-Type: application/json' -d '{"text":"世界","steps":1}' -o /tmp/higgs-final-server-b.wav
projects/higgs-audio-ggml-cpp/build/higgs-audio-server --port 18121 --backend cpu --workers 1 --queue-size 2 --request-timeout-sec 1
curl -sS -w '\nHTTP:%{http_code}\n' -X POST http://127.0.0.1:18121/generate -H 'Content-Type: application/json' -d '{"text":"你好","steps":4}' -o /tmp/higgs-final-timeout-body.txt
# returns HTTP 504 with "generate failed: request timed out".
```

The ChatML/Boson prompt self-check matches Python
`tokenizer.apply_chat_template(..., add_generation_prompt=True)` ids for `你好`.
The inspect command reports `929` tensors, `34` metadata entries, 8 codebooks,
and vocab size 1026. `--inspect-text` reports Qwen3 config
`vocab=151936`, `ctx=32768`, `hidden=2560`, `ffn=9728`, `layers=36`,
`heads=32`, `kv_heads=8`, `head_dim=128`, `rope_theta=1e6`, and validates first
and last layer frontier tensors. The sampler, prompt, and delay commands report
their `self-check ok` lines. `--text test --out /tmp/higgs-test.wav` currently
writes backend-KV autoregressive real-prompt-conditioned code frames through
codec decode. Reference-audio generation also uses backend-KV AR after local
reference codec encode.

Latest CUDA timing on RTX 4060 Ti after `HiggsPipeline`-owned backend runtime
prewarm, hot weight/config caching, reusable backend KV cache, and all-layer
backend KV decode graph fusion: `steps=20` is `0:32.50`, `steps=40` is
`0:35.86`, and `steps=80` is `0:47.58`. The previous prewarm-only timings were
`steps=20` `0:35.40`, `steps=40` `0:35.38`, and `steps=80` `0:47.12`; earlier
fused request-local timings were `steps=20` `0:33.95`, `steps=40` `0:35.69`,
and `steps=80` `0:46.53`. The cached per-layer graph timings were `steps=20`
`0:32.09`, `steps=40` `0:39.26`, and `steps=80` `0:50.31`; the original
`steps=1` baseline before the cache pass was `0:51.62`.

Current-version checks after reusable KV cache landed: CUDA prompt/reference AR
self-checks pass, CPU prompt AR self-check passes, CUDA server returns HTTP 200
for two consecutive `/generate` requests on one process-resident pipeline, and
CPU server timeout still returns HTTP 504 with `request timed out`.

## Next Step

Reference generation parity is now the active correctness gate. C++ previously
fed raw reference codec codes into the text placeholder span, while the Python
pipeline feeds delayed reference codes. The C++ `HiggsPipeline::generate`
reference path now applies `apply_delay_pattern_tn(...)` before building the
reference prompt / prefill audio embeddings.

Trace files from the current diagnostic run:

- C++ delayed-reference KV trace:
  `/tmp/higgs-cpp-reference-delayfix-trace.json`
- Python GGML full-forward trace with the same prompt ids and delayed reference
  codes:
  `/tmp/higgs-py-reference-delayfix-20-trace.json`
- 20-step C++ diagnostic audio:
  `/tmp/higgs-cpp-reference-delayfix-20.wav`

The prompt ids and delayed reference codes match exactly between C++ and Python.
After adding masked top1/top5 trace output, the first remaining divergence is
already at step 0 audio logits. The sampled codes look equal at step 0 only
because the SGLang delay rule overwrites codebooks 1..7 to BOC. C++ now uses
the same `steps` contract as Python (`steps` sampled rows plus the initial BOC
row; for `steps=20`, delayed rows are 21), and the C++ trace writes
`finalized_codes` in Python's codebook-major JSON orientation. The first visible
sampled-code divergence is delayed row 6:

```text
cpp [1017, 87, 164, 160, 265, 966, 1024, 1024]
py  [1017, 87, 497, 361, 265, 966, 1024, 1024]
```

Step 0 masked top1 differs:

```text
cpp [457, 401, 579, 310, 518, 417, 106, 773]
py  [457, 939, 41, 272, 769, 417, 838, 773]
```

Latest node-stat trace narrowed the first numeric divergence further. C++ and
Python `inputs_embeds` last-token stats match to float precision, so prompt ids,
reference placeholder overlay, and BOC audio embedding are not the first
problem. Inside block 0, `attn_normed`, `q_proj`, `v_proj`, `q_normed`,
`q_state`, and `v_raw` match C++ full-forward vs Python GGML full-forward.
Full binary dumps showed the earlier `kqv` / `attn_merged` summary was a
trace-stat view problem: the leading values match exactly and full-array drift is
small (`kqv`/`attn_merged` mean_abs `0.0080`, p99 `0.0866`). Last-token block0
drift is also modest but enough to move later logits (`blk.0.hidden` last-token
mean_abs about `0.499`, p99 about `1.66`):

```text
inputs_embeds first8 match:
[0.90234375, -0.7055664, 0.7597656, -0.2034912, 0.09765625, 0.7236328, -0.8349609, 0.4052734]

blk.0.hidden first8:
cpp [76.5352, -2.1079, 13.1855, 0.6403, -4.3816, 7.3491, 0.5205, 9.5889]
py  [78.4023, -1.4927, 12.1426, 1.1160, -3.9812, 6.7480, 0.9111, 10.3428]
```

Top-5 logits with values confirm this is not a simple codebook-layout swap. Some
step-0 winners are close (`cb1`: C++ 401=57.9683 vs 939=57.9440; Python
939=57.9375 vs 401=57.9062), while other codebooks have larger backbone-driven
logit movement (`cb2`, `cb4`, `cb6`). A forced-contiguous repeat experiment did
not change C++ top1, so do not keep that path as a fix.

`NVIDIA_TF32_OVERRIDE=0` does not change the C++ step-0 top1, so TF32 is not the
main cause. A forced-contiguous repeat experiment also did not change top1 and
was reverted.

Next repair should stop relying on summary node stats for shaped tensors and use
full binary dumps for the last-token path. The remaining mismatch looks like
accumulated block/logit drift rather than a tokenizer, delay, codec, or simple
codebook-layout error. Do not revisit tokenizer, reference delay, or codec decode
until the block0/full-backbone last-token drift source is aligned.

2026-06-25 follow-up: the earlier block0 drift summary was contaminated by trace
view/dump overwrites. With per-`past` binary dumps, BOC step input and layer0
norm align exactly against Python replay:

```text
kv_decode_input vs py inputs last: max_abs 0.0
blk0.attn_normed: max_abs 0.0
blk0.hidden: mean_abs 0.00453, p99 0.01948, max_abs 0.03864
```

Layer drift stays small through block33, then grows in the last two blocks:

```text
blk33.hidden mean_abs 0.11039, p99 0.37209, max_abs 0.54492
blk34.hidden mean_abs 0.38463, p99 1.64290, max_abs 7.88318
blk35.hidden mean_abs 0.66537, p99 2.43004, max_abs 11.21265
audio-head final_hidden mean_abs 0.00377, max_abs 0.08069
audio logits mean_abs 0.20995, max_abs 1.39114
```

Step-0 delayed codes still match because the delay rule masks later codebooks,
but top1 differs on close logits, for example codebook 1 C++ picks `401`
(`57.9537`) while Python picks `939` (`57.9375`). The first repaired divergence
was the audio-head GEMM shape in the C++ full-recompute diagnostic path: Python
runs `a.output` over the full sequence shape `(ne0=2560, ne1=tokens)`, while C++
was running only the last token `(ne0=2560, ne1=1)`. With
`run_audio_head_logits_vc_for_frame()` using the full hidden sequence and taking
the last frame, full recompute now matches Python for the 20-step reference gate:

```text
first_code_divergence None
all_codes_equal True
delayed rows equal True
finalized_equal True
top1_equal_count 17 / 20
```

The remaining production-path mismatch is hot KV decode: using the same
single-token audio head still diverges at step 5. A fake batched-head experiment
that duplicated the last hidden across all frames made step 1 worse, so do not
keep that path. The next real fix must align hot KV hidden history with Python
full-forward semantics or intentionally route reference diagnostic generation
through the now-aligned full path. Do not tune sampling parameters to hide this.

Resolution for the current reference-quality gate: reference-audio generation now
routes through the full recompute path by default, and that path always evaluates
the audio head with the full hidden sequence shape before selecting the last
frame. Verification on the target wav/text/GGUF, `temperature=0`, `steps=20`:

```text
codes_equal True
delayed_equal True
finalized_equal True
top1_equal_count 17 / 20
```

Generated waveform stats for `/tmp/higgs-cpp-default20-fullref-fixed-real.wav`:

```text
sr 24000, channels 1, frames 17280, duration 0.72s
mean 0.0003645, abs_mean 0.104759, rms 0.146170
max_abs 0.713379, p99_abs 0.482832, nonzero_ratio 0.999884
elapsed 10:14.77, maxrss 691628 KB
```

Python CPU codec decode on the same C++ `finalized_codes` is close to the C++
waveform:

```text
py acoustic shape (256, 18)
py acoustic mean -0.119810, abs_mean 13.175016, max_abs 62.742455
py wav shape (17280,), mean 0.0003617, abs_mean 0.1047745
py wav rms 0.1461848, max_abs 0.711741, p99_abs 0.482826
cpp wav shape (17280,), mean 0.0003645, abs_mean 0.1047590
cpp wav rms 0.1461701, max_abs 0.713379, p99_abs 0.482832
wav diff mean_abs 0.000503, p99 0.002935, max_abs 0.009944
```

Python CUDA codec decode is not usable for this diagnostic because GGML CUDA
`conv_transpose_1d` asserts on non-F32 weights; the C++ codec path already casts
decoder transposed-conv weights before CUDA execution.

2026-06-26 continuation on fast reference path:

- Added BOC audio embedding to the bounded window-refresh candidate so a
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=4` refresh before step 3 covers exactly
  `[BOC, f0, f1, f2]` after the prompt/reference prefill.
- The 4-step gate remains divergent at the same first point:
  `trace[3].codes cb3 907 -> 165`, top5 still `[907,165,...]` vs
  `[165,907,...]`. Window refresh count was 2 and elapsed stayed about 9s, so
  the code path executed but did not fix hidden/logit drift.
- Window hidden dumps for `start=237,tokens=4` show block0 hidden still differs
  from full step3 at `mean_abs ~= 0.00584, p99 ~= 0.0291, max_abs ~= 0.0538`,
  essentially the same scale as single-token KV. Bounded audio-only refresh is
  therefore not an effective correction as currently implemented.
- A speculative cache-write `ggml_cont_3d(k_cur/v_cur)` change before
  `ggml_set_inplace` was tested and did not change the step3 `907 -> 165`
  divergence; it was reverted.
- A margin-triggered full fallback with `HIGGS_REFERENCE_KV_FALLBACK_MARGIN=0.2`
  did not finish 20 trace-only steps in 90s and was killed. That path is too
  close to full recompute to be the final performance fix.

Current next useful checks: compare full block0 internal attention nodes against
KV with a clean, minimal dump that does not trigger reference-codec node dumps;
or reuse the runtime prefill graph as the full-path quality baseline to see
whether the target should be full C++ diagnostic semantics or Python replay
semantics.

2026-06-26 deeper drift localization:

- `HIGGS_ATTENTION_TOKENS` in the block0 QKV self-check was fixed to actually
  honor the env value. With the real step3 KV `attn_normed` vector repeated
  across columns, `q_proj/k_proj/v_proj` are identical for tokens=1 and
  tokens=4. RoPE changes `q/k` for later columns as expected because positions
  differ. So the current step3 flip is not a QKV matmul batch-shape bug.
- Comparing full block0 step3 against KV past240 shows K/V cache contents are
  close enough to full (`k_state mean_abs ~= 3.4e-5`, `v_state mean_abs ~= 5e-7`)
  and block0 `attn_out` is still small (`mean_abs ~= 2.7e-4`). The first large
  amplification is `ffn_down`: `mlp_normed mean_abs ~= 4.6e-5`, gate/up around
  `1.3e-3`, but `mlp_out mean_abs ~= 0.0059`, matching final block0 hidden drift.
- CPU backend KV shows the same block0 MLP amplification, so this is not CUDA
  matvec-specific. A temporary diagnostic that forced a multi-column MLP matmul
  shape also did not change the step3 `907 -> 165` divergence and was reverted.

Current conclusion: KV cache K/V is not wildly wrong; tiny incremental-vs-full
post-attention differences are amplified by the F16 MLP down projection from the
first block onward. The next useful root-cause check is to remove the tiny
post-attention difference itself, likely by making KV attention output exactly
match the full-prefix attention for the last token, before trying any more MLP
or sampling workarounds.

2026-06-26 clean full block0 dump follow-up:

- Added `HIGGS_AR_NODE_DUMP_ONLY=1` to suppress reference-codec node dumps while
  keeping AR/backbone dumps, and added step-scoped full block0 diagnostic names
  (`cpp_stepN_cpp_blk0_*`) so later reference steps do not overwrite earlier
  block0 internals.
- A clean 4-step full dump at `/tmp/higgs-full-stepnamed4b` confirms the same
  earliest visible divergence against KV past240: full and KV inputs plus
  `attn_normed` are bit-equal, but `q_proj` already differs
  (`mean_abs ~= 1.96e-4`, `p99 ~= 1.4e-3`). That becomes `q_state mean_abs
  ~= 0.00315`, then MLP down amplifies the residual into block0 hidden drift.
- Feeding the dumped full step3 `attn_normed` prefix back through the standalone
  block0 QKV helper does not reproduce the full block graph's `q_proj`; it stays
  much closer to KV. So the current `q_proj` dump divergence is likely tied to
  the full block graph execution/dump path itself, not the raw input tensor or a
  simple token-count effect.
- Existing Python node dumps under `/tmp/higgs-node-py*` do not match this exact
  4-step reference gate well enough to decide whether Python follows the full
  C++ block dump or the KV/helper dump. Do not use those stale dumps as parity
  evidence for this gate.

Next useful check: produce a same-gate Python GGML block0 QKV dump, or add a
minimal C++ self-check that computes `attn_q.weight @ dumped_attn_normed` in two
independent graphs and asserts which graph path is stable. Avoid more sampler or
codec work until the q_proj source is resolved.

2026-06-26 fallback and q_proj diagnostic notes:

- A same-gate CPU standalone QKV run over the full step3 `attn_normed` prefix is
  bit-identical to the CUDA standalone QKV run and remains much closer to KV than
  to the full block graph's dumped `q_proj`. This makes the full block `q_proj`
  dump suspect as a diagnostic signal; it should not be treated alone as proof
  that KV q-projection is wrong.
- Margin fallback remains unusable as a final performance path: even an 8-step
  trace with `HIGGS_REFERENCE_KV_FALLBACK_MARGIN=0.07` exceeded 90 seconds and
  was killed. The 0.09 follow-up process was also killed. Do not pursue full
  fallback thresholds as the solution.
- A temporary attempt to force a copied q-projection dump in the full block graph
  did not produce a useful step3 dump before timeout and was reverted.

Next shortest path: create a minimal same-gate Python/C++ q-projection parity
script or a small C++ self-check that computes only `attn_q.weight @
attn_normed` from dumped files, outside the full block graph, then use that as
truth for whether the KV projection is correct. If KV projection is correct, the
remaining drift is in attention/cache accumulation rather than q_proj itself.

2026-06-26 independent q-projection parity:

- A one-off Python/ggbond graph that computes only `blk.0.attn_q.weight @
  /tmp/higgs-full-stepnamed4b/cpp_step3_cpp_blk0_attn_normed.f32` was run on
  CUDA. Reading the raw GGML output (no numpy reshape) matches the full C++
  block dump exactly: `pyraw_vs_full mean_abs=0, max_abs=0`.
- The same Python raw output differs from C++ KV/helper by the same amount as
  full vs KV (`mean_abs ~= 1.96e-4`, `p99 ~= 1.4e-3`). So the earlier suspicion
  that the full `q_proj` dump was bogus is wrong; the Python/C++ full projection
  agree for this exact same-gate input.
- The C++ standalone QKV helper still follows KV, not Python/full. The next root
  cause is therefore inside the C++ helper/KV projection path: same dumped input
  and same GGUF tensor name produce a different `ggml_mul_mat` result than the
  Python/ggbond graph and the full block graph. Focus there; do not chase MLP,
  sampler, or codec.

2026-06-26 KV window refresh quality path:

- Fixed the block0 QKV self-check CLI path so `--backend cuda` is actually honored, and added `HIGGS_ATTENTION_Q_ONLY=1` to compute only `attn_q.weight @ attn_normed`.
- With the backend fixed, CUDA q-only over the full 241-token prefix matches the full/Python q-projection exactly. The apparent helper/KV q-projection drift was a single-token vs batched CUDA matmul behavior, not a bad weight layout or codec issue. Single-token q-projection matches KV and differs from full by about `mean_abs ~= 1.96e-4`, enough to flip close greedy logits after downstream amplification.
- Moved `HIGGS_REFERENCE_KV_WINDOW_REFRESH` to refresh before sampling, and changed the refresh input to use the full reference prompt embeddings plus generated audio embeddings instead of only generated audio frames.
- Added a full-window hidden return path so refreshed audio-head logits can be computed with `run_audio_head_logits_vc_for_frame(...)` over the same batched shape as the full baseline. This fixes the later audio-head single-column matmul flip.
- `HIGGS_REFERENCE_BACKEND_KV=1` is now allowed outside trace-only when `HIGGS_REFERENCE_KV_WINDOW_REFRESH > 0`; naked backend KV remains diagnostic-only.
- Gate result: `HIGGS_REFERENCE_BACKEND_KV=1 HIGGS_REFERENCE_KV_WINDOW_REFRESH=241`, same reference wav/text/GGUF/temperature=0, matches the current full baseline for 20 generated steps exactly. 20-step non-trace wav wrote `/tmp/higgs-kv-windowfull20-grow-audio.wav`; duration 0.72s, elapsed 35.93s, RTF ~= 49.9.
- 40-step window trace wrote `/tmp/higgs-kv-windowfull40-grow.json` in 57.18s and starts with the corrected step3 codebook3 `907`. A matching full 40-step trace was attempted for baseline proof but exceeded 120s and was interrupted, confirming full recompute is still not an acceptable verification or runtime path.

Remaining work: reduce the cost of the aligned path. The current working path is a quality-correct full-prefix refresh, not the final fast KV decode. Next target is to keep the batched numerical behavior only where it matters, likely by batching the current hidden with enough prefix context for q/audio-head matmuls or by a bounded refresh policy validated against 20/40-step traces.

2026-06-26 cached full-prefix refresh optimization:

- Small bounded windows (`HIGGS_REFERENCE_KV_WINDOW_REFRESH=4/8/16/32/64/128`) were tested on the same 4-step gate. They do not reliably match full greedy codes; some fix step3/codebook3 but break earlier codebooks. Do not use them as the quality path yet.
- Added a cached batched audio-head path for refreshed full-window hidden states. This avoids reopening GGUF/reloading `output_norm.weight` and `a.output` every step while preserving the batched matmul shape needed for greedy parity.
- In full-prefix refresh mode (`window_refresh >= prompt.size()`), skipped the redundant single-token all-layer KV decode after sampling; the next step's full-prefix refresh overwrites/updates cache K/V anyway. Naked KV and small-window modes keep the old decode behavior.
- Current gate results with `HIGGS_REFERENCE_BACKEND_KV=1 HIGGS_REFERENCE_KV_WINDOW_REFRESH=241`:
  - 20-step trace vs `/tmp/higgs-default-full-20.json`: exact generated-code match, elapsed 14.60s after skip-decode optimization.
  - 20-step non-trace wav `/tmp/higgs-kv-windowfull20-skipdecode-audio.wav`: duration 0.72s, elapsed 18.88s, RTF ~= 26.2. Runtime varies; previous cached-head non-trace was RTF ~= 23.9.
  - 40-step trace `/tmp/higgs-kv-windowfull40-skipdecode.json`: exact match against the previous aligned 40-step window trace, elapsed 19.43s. A direct full 40-step baseline remains too slow to use casually; a prior attempt exceeded 120s and was interrupted.

Remaining work: find a true smaller bounded refresh that matches 20/40-step full/Python traces, or add a targeted matmul batching strategy that preserves full-prefix numerical behavior without recomputing all layers over the full prefix each step.

2026-06-26 BOC decode skip in full-prefix refresh:

- In full-prefix refresh mode (`HIGGS_REFERENCE_KV_WINDOW_REFRESH >= prompt.size()`), the initial BOC single-token KV decode is redundant: the first sampling iteration immediately recomputes the full prefix including BOC and writes the correct K/V. Skipped that decode in this mode while preserving the old behavior for naked KV and smaller-window refresh.
- 20-step trace with `HIGGS_REFERENCE_BACKEND_KV=1 HIGGS_REFERENCE_KV_WINDOW_REFRESH=241` still exactly matches `/tmp/higgs-default-full-20.json`; elapsed 14.12s.
- Clean 40-step trace `/tmp/higgs-kv-windowfull40-skipboc-clean.json` exactly matches the previous aligned skipdecode trace; elapsed 20.77s. A concurrent run took 175s due to GPU contention and should be ignored.
- Non-trace 20-step wav `/tmp/higgs-kv-windowfull20-skipboc-audio-clean.wav`: duration 0.72s, elapsed 14.92s, RTF ~= 20.7.

Current quality/perf position: the env-gated C++ reference path is usable and no longer trace-only, with 20-step full-baseline code parity and stable 40-step aligned-trace parity. It is still a full-prefix refresh, not the final true bounded refresh requested by the long-term goal.

2026-06-27 head-batch and periodic-refresh experiments:

- Added `HIGGS_REFERENCE_KV_HEAD_BATCH` as a diagnostic fast path: it keeps naked backend KV hidden states but computes the audio head over repeated hidden columns so CUDA uses a batched matmul shape. This confirms the first visible gate failure is partly an audio-head matmul-shape issue: with `HIGGS_REFERENCE_KV_HEAD_BATCH=241`, the 4-step gate fixes step3/codebook3 from `165` back to `907` without full-prefix backbone refresh.
- Head batching alone is not enough for the real gate. A 20-step run with `HIGGS_REFERENCE_KV_HEAD_BATCH=241` diverges from `/tmp/higgs-default-full-20.json` starting at step 7; hidden drift still accumulates in the KV backbone.
- Tried a periodic full-prefix refresh experiment (`HIGGS_REFERENCE_KV_FULL_REFRESH_PERIOD`) combined with head batching. It was removed after testing because the state handoff is not correct: if the post-sample single-token decode runs, it overwrites refreshed numerical state; if it is skipped, the next-step hidden/cache chain is incomplete. Periods 2/3/4/6/8 did not meet the 20-step gate.
- Revalidated the known-good path after removing the failed periodic experiment: `HIGGS_REFERENCE_BACKEND_KV=1 HIGGS_REFERENCE_KV_WINDOW_REFRESH=241` still matches the 4-step full baseline including step3/codebook3 `907`.

Current recommendation remains the full-prefix refresh path for quality. `HIGGS_REFERENCE_KV_HEAD_BATCH` is useful diagnostic evidence, not the final fix.

2026-06-27 backend KV default safety path:

- Changed `reference_kv_window_refresh()` default from `0` to `INT_MAX`, so `HIGGS_REFERENCE_BACKEND_KV=1` now defaults to the known-good full-prefix refresh path. Users no longer need to remember `HIGGS_REFERENCE_KV_WINDOW_REFRESH=241` for the safe path.
- Explicit `HIGGS_REFERENCE_KV_WINDOW_REFRESH=0` still selects naked backend KV and remains diagnostic-only unless `HIGGS_TRACE_ONLY=1` and `HIGGS_TRACE_JSON` are set. Verified the non-trace command fails with the diagnostic-only error.
- Gate results with only `HIGGS_REFERENCE_BACKEND_KV=1`:
  - 20-step trace vs `/tmp/higgs-default-full-20.json`: exact generated-code match, elapsed 14.75s.
  - 20-step non-trace wav `/tmp/higgs-backendkv-default20-audio.wav`: duration 0.72s, elapsed 15.75s, RTF ~= 21.9.
  - 40-step trace `/tmp/higgs-backendkv-default40.json`: exact match against the aligned 40-step full-prefix trace, elapsed 19.35s.

This makes the env-gated backend KV path safe by default while keeping the raw KV path available only as a trace diagnostic.

2026-06-27 head-batch width sweep:

- Tested `HIGGS_REFERENCE_KV_HEAD_BATCH` widths `2/4/8/16/32/64/128/241` with naked backend KV on the 4-step gate. Widths up to 128 all behave like raw KV and keep the bad early sequence (`689,957...`, step3/codebook3 `956`). Only width 241 reproduces the corrected full-prefix-like codes through step3, including codebook3 `907`.
- This rules out a cheap small dummy-batch audio-head fix. The numerical behavior needed for this gate appears tied to full-prefix-scale column count, not merely "more than one" column.
- Revalidated the current safe default `HIGGS_REFERENCE_BACKEND_KV=1` after the experiment: 4-step trace still matches the full baseline, including step3/codebook3 `907`.

Do not spend more time on small head-batch padding as the main fix. The remaining true-bounded path must address backbone hidden/cache drift, not only audio-head matmul shape.

2026-06-27 small-window refresh input fix:

- Fixed a real bug in small-window refresh input selection introduced while making full-prefix refresh safe: non-full-prefix windows were taking the tail of `generated_embedding_values`, which includes prompt embeddings, so early windows could rewrite prompt-cache positions with truncated prompt/audio slices. Non-full-prefix windows now use only `generated_embeddings` (BOC + generated audio frame embeddings); full-prefix refresh still uses the complete prompt+audio embedding buffer.
- After the fix, `HIGGS_REFERENCE_KV_WINDOW_REFRESH=4` improves from severe prompt-cache corruption to matching the first three steps, but still misses step3 (`81/901` instead of `370/907`). Windows `8/16/32/64/128` still produce the raw-KV step3 `165` without head batching.
- Combining small windows with `HIGGS_REFERENCE_KV_HEAD_BATCH=241` passes the 4-step gate for windows `8/16/32/64`, but 20-step still diverges from step 7. The remaining drift is in the backbone hidden chain, not only the audio-head shape.
- Tried skipping the post-sample single-token decode for all window modes so the next window refresh would own hidden/KV state; this breaks the chain immediately and was reverted. Only full-prefix refresh can safely skip that decode because it recomputes the entire active prefix each step.

Current safe default remains `HIGGS_REFERENCE_BACKEND_KV=1` full-prefix refresh. The fixed small-window path is a better diagnostic now, but still not a passing bounded solution.

2026-06-27 contextual-window experiment:

- Added diagnostic `HIGGS_REFERENCE_KV_WINDOW_CONTEXTUAL_INPUT=1` to test whether small-window refresh should feed previously contextualized window hidden states instead of raw audio embeddings.
- `WINDOW_REFRESH=8 + HEAD_BATCH=241 + CONTEXTUAL_INPUT=1` still diverges from the 20-step full baseline starting at step 7, and later steps drift more strongly. So the step7 failure is not fixed by reusing the previous window's contextual hidden as the next window input.
- Revalidated the safe default `HIGGS_REFERENCE_BACKEND_KV=1`: 4-step trace still matches full baseline.

Keep `HIGGS_REFERENCE_KV_WINDOW_CONTEXTUAL_INPUT` diagnostic-only. It is not a quality path.

2026-06-27 full-dump cost note:

- Tried to produce an 8-step full recompute dump with per-layer last-token files to compare step7 window hidden drift directly. The run exceeded 120s and was interrupted. This reinforces that full recompute dumps are too expensive to use as routine 20/40-step validation while optimizing the fast path.
- No production-path change resulted from this attempt. The current safe path remains default `HIGGS_REFERENCE_BACKEND_KV=1` full-prefix refresh; small-window/contextual-window experiments remain diagnostic only.

2026-06-27 diagnostic gate cleanup:

- Tightened `HIGGS_REFERENCE_BACKEND_KV` gating so `HIGGS_REFERENCE_KV_HEAD_BATCH` no longer unlocks non-trace generation by itself. Head-batch and contextual-window modes are diagnostic only and must be run with `HIGGS_TRACE_ONLY=1`/`HIGGS_TRACE_JSON` unless the safe full-prefix refresh path is active.
- Verified `HIGGS_REFERENCE_BACKEND_KV=1 HIGGS_REFERENCE_KV_WINDOW_REFRESH=0 HIGGS_REFERENCE_KV_HEAD_BATCH=241` fails non-trace with the diagnostic-only error.
- Verified default safe path `HIGGS_REFERENCE_BACKEND_KV=1` still writes a 20-step non-trace wav: `/tmp/higgs-default-safe-after-gatecleanup-20.wav`.

2026-06-27 full-prefix input copy reduction:

- Avoided one large temporary copy in the safe full-prefix refresh path: when `always_full_prefix_refresh` is active, the refresh graph now reads `generated_embedding_values` directly instead of first copying the active prefix into `window_hidden`. Small-window diagnostic paths still build an explicit temporary window.
- 20-step trace with default `HIGGS_REFERENCE_BACKEND_KV=1` still exactly matches `/tmp/higgs-default-full-20.json`; elapsed 13.27s in this run.
- 20-step non-trace wav `/tmp/higgs-nocopy-default20-audio.wav`: duration 0.72s, elapsed 16.4s, RTF ~= 22.8. Non-trace timing remains dominated by more than this copy and varies run to run.

2026-06-27 full-prefix audio-head transfer reduction:

- In the cached batched audio head, non-trace generation now copies only the requested frame's logits back to CPU instead of transferring the full `(8208, frames)` logits tensor. Trace mode still copies/dumps the full tensor for diagnostics.
- Added a non-dump full-prefix path that appends `output_norm/a.output` to the window refresh graph and returns only the last hidden token plus last-frame logits. This preserves the full-prefix CUDA matmul shape and avoids a separate audio-head graph call on the public default path.
- Verification:
  - 20-step JSON trace with `HIGGS_REFERENCE_BACKEND_KV=1` still exactly matches `/tmp/higgs-default-full-20.json`; elapsed 12.77s in this run.
  - 40-step JSON trace still matches `/tmp/higgs-kv-windowfull40-skipboc-clean.json`; elapsed 16.94s.
  - 40-step non-trace wav `/tmp/higgs-fused-audio40.wav`: duration 1.52s, elapsed 22.35s, RTF ~= 14.70.
- This is a safe speed improvement, not the final bounded recompute solution. Profiling still shows the dominant cost is the per-step full-prefix backbone refresh (`~180-200ms` each on this machine), so further work must reduce or fix that path without reintroducing the known small-window/naked-KV drift.

2026-06-27 full-prefix prefill skip:

- In the safe full-prefix refresh path, skipped the initial prompt backbone prefill. That prefill only populated cache entries that the first full-prefix refresh immediately overwrites from position 0 after BOC is appended. The path still computes prompt/reference embeddings, sets `cache.used` to prompt length, then lets the first refresh own the complete prompt+BOC cache.
- Non-full-prefix diagnostic paths still run the original prefill because they depend on prompt KV cache state.
- Verification:
  - 20-step JSON trace with `HIGGS_REFERENCE_BACKEND_KV=1` still exactly matches `/tmp/higgs-default-full-20.json`; elapsed 12.23s.
  - 40-step JSON trace still exactly matches `/tmp/higgs-kv-windowfull40-skipboc-clean.json`; elapsed 16.42s.
  - 40-step non-trace wav `/tmp/higgs-skipprefill-audio40.wav`: duration 1.52s, elapsed 21.66s, RTF ~= 14.25.
- Tried warming up bounded `HIGGS_REFERENCE_KV_WINDOW_REFRESH=8` before the window is full. It made the 4-step gate worse (diverged from step 1), so that change was reverted. The known step3/codebook3 failure is not just caused by early fallback to naked KV before the refresh window fills.

2026-06-27 rejected hidden fake-batch experiment:

- Tried a diagnostic `HIGGS_REFERENCE_KV_HIDDEN_BATCH` idea: duplicate the single-token hidden input into a wider batch so CUDA matmuls in `run_kv_decode_graph_all_layers_with_backend_cache` would use a full-prefix-like batch shape, then write/use only the last column.
- The quick implementation is not a valid all-layer path because after layer 0 it collapses back to a single hidden column while later layers still expect the fake batch shape. It hit a GGML reshape assert and was fully reverted.
- Revalidated the safe default after the revert: 4-step trace with `HIGGS_REFERENCE_BACKEND_KV=1` still matches `/tmp/higgs-default-full-20.json` for the first four steps.
- Do not continue with fake hidden batching unless it is implemented as a coherent all-layer batched decode that preserves batch shape through every layer. The small, local version is not a viable fix.

2026-06-27 rejected skip-after-refresh handoff experiment:

- Tested whether small-window drift was caused by decoding the newly sampled audio token immediately after a window refresh had just rewritten the previous window cache. Added a temporary diagnostic to skip the post-sample single-token decode when a non-full window refresh happened, then let the next refresh own that token.
- Gate used the previously closest diagnostic combination: `HIGGS_REFERENCE_KV_WINDOW_REFRESH=8 HIGGS_REFERENCE_KV_HEAD_BATCH=241`, plus the temporary skip-after-refresh switch.
- Result: 20-step trace still diverged from `/tmp/higgs-default-full-20.json` starting at step 7, with 13 divergent steps. The temporary switch was reverted.
- This rules out the simple "double-decode after refresh" explanation for the step7 bounded-window failure. The remaining issue is deeper hidden/KV numerical drift, not just an obvious loop handoff bug.

2026-06-27 step7 window-vs-full hidden dump:

- Produced 8-step node dumps for the safe full-prefix path and the closest bounded diagnostic path (`HIGGS_REFERENCE_KV_WINDOW_REFRESH=8 HIGGS_REFERENCE_KV_HEAD_BATCH=241`).
- Compared the last-token hidden at the final refresh before the first known divergence:
  - full-prefix dump: `cpp_kv_window_blk*_hidden_start0_tokens245.f32`
  - window dump: `cpp_kv_window_blk*_hidden_start237_tokens8.f32`
- The bounded path already differs at block 0 (`mean_abs ~= 0.0067`, `p99 ~= 0.0358`, `max_abs ~= 0.255`) and the difference steadily amplifies through the stack, reaching block 35 (`mean_abs ~= 0.917`, `p99 ~= 3.296`, `max_abs ~= 9.220`). The trace first diverges at step 7.
- This is strong evidence that the `WINDOW_REFRESH=8` failure is not a local tail-layer bug or a loop handoff bug. The short window is semantically missing long-context attention, and the transformer stack amplifies that from the first block onward.
- Do not continue spending time on tiny recent-window variants as the main path. A viable bounded path would need to keep enough prompt/long-context information in attention, or fix the single-token KV numerical path directly.

2026-06-27 rejected CUDA precision toggles for raw KV:

- Tested whether the raw single-token KV divergence is caused by CUDA matmul precision mode rather than graph semantics.
- `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` with naked KV (`HIGGS_REFERENCE_KV_WINDOW_REFRESH=0`) still matched steps 0-2 and failed step3/codebook3 as `907 -> 165`.
- `NVIDIA_TF32_OVERRIDE=0` with naked KV produced the same step3/codebook3 failure.
- Also tried a temporary diagnostic that set `ggml_mul_mat_set_prec(..., GGML_PREC_F32)` on all matmuls inside `run_kv_decode_graph_all_layers_with_backend_cache`; it still failed `907 -> 165` and was reverted.
- Built a separate CUDA binary with `GGML_CUDA_FORCE_CUBLAS=ON` to avoid CUDA `mul_mat_vec` kernels entirely. Naked KV still matched steps 0-2 and failed step3/codebook3 as `907 -> 165`.
- Conclusion: the first raw-KV divergence is not fixed by cuBLAS 32F compute, disabling TF32, GGML op-level F32 precision, or forcing cuBLAS instead of vector matmul kernels. The remaining culprit is likely graph/cache semantics or single-token-vs-batched execution shape effects outside simple matmul backend selection.

2026-06-27 raw-KV block0 internal dump:

- Added node-dump-only internals for block0 in the full-window refresh path so raw single-token KV can be compared against full-prefix refresh at the same token.
- Compared full-prefix `start0_tokens241` with raw-KV `past240`, which is the first failing pre-step3 state. The trace still matches steps 0-2 and fails step3/codebook3 as `907 -> 165`.
- Block0 findings:
  - `k_cur` is close (`mean_abs ~= 0.0020`, `p99 ~= 0.0143`, `max_abs ~= 0.131`).
  - `v_cur` is very close (`mean_abs ~= 3.3e-05`, `p99 ~= 2.6e-04`).
  - `attn_out` is close (`mean_abs ~= 2.7e-04`, `p99 ~= 0.0028`, `max_abs ~= 0.0104`).
  - `post_attn` is similarly close.
  - `mlp_out` is where the difference becomes material (`mean_abs ~= 0.0059`, `p99 ~= 0.0292`, `max_abs ~= 0.0598`), and block0 hidden inherits that scale.
- This narrows raw-KV drift: at the first failing step, attention/cache values are not wildly wrong in block0. The small post-attention difference is amplified by the block0 MLP, then later layers magnify it further. Next useful check is to isolate whether the MLP amplification is caused by single-column `ffn_gate/up/down` matmul shape, SiLU/gating sensitivity, or residual input drift.

2026-06-27 block0 MLP substep sensitivity:

- Reused the full-window block0 internals and raw-KV block0 internals for the first failing pre-step3 state.
- Difference progression:
  - `post_attn`: `mean_abs ~= 2.74e-4`, `p99 ~= 0.00283`, relative mean ~= `9.0e-4`.
  - `mlp_normed`: `mean_abs ~= 4.62e-5`, `p99 ~= 2.04e-4`.
  - `ffn_gate`: `mean_abs ~= 0.00136`, `p99 ~= 0.00933`.
  - `ffn_up`: `mean_abs ~= 0.00127`, `p99 ~= 0.00679`.
  - `silu(gate) * up`: `mean_abs ~= 3.75e-4`, `p99 ~= 0.00342`.
  - `ffn_down/mlp_out`: `mean_abs ~= 0.00591`, `p99 ~= 0.0292`, `max_abs ~= 0.0598`.
- Interpretation: the MLP is behaving like an amplifier for an already-present post-attention difference. The largest jump is the `ffn_down` projection from the gated hidden width back to model width, not an isolated SiLU/gate bug. This makes a local MLP implementation bug less likely; the real fix probably needs raw single-token hidden to match the batched path before MLP, or a batched/refresh strategy that avoids injecting that small pre-MLP drift.

2026-06-27 rejected margin-based full fallback:

- Tested the existing `HIGGS_REFERENCE_KV_FULL_FALLBACK=1` path with naked KV and `HIGGS_REFERENCE_KV_FALLBACK_MARGIN=0.05` as a possible sparse fallback strategy.
- The run was interrupted after more than 90s, already slower than the safe 20-step full-prefix refresh path. The partial trace had only 9 steps and had already diverged from `/tmp/higgs-default-full-20.json` starting at step 3 (`907 -> 165`).
- This means the current margin heuristic does not catch the first dangerous greedy flip and is far too slow when it does fall back because it uses the old full recompute path. Do not treat margin fallback as a viable fast reference path without redesigning both the trigger and fallback implementation.
- Follow-up 4-step checks showed why `0.05` missed the first flip: at step3/codebook3 raw KV has `165` over `907` by about `0.085`, while the full path has `907` over `165` by about `0.0625`. Raising the margin to `0.09` or `0.12` catches the 4-step gate and restores `907`, but each 4-step run takes roughly a minute because the fallback still uses the old full recompute path.
- If this direction is revisited, the only plausible version is a sparse fallback that calls the current safe full-prefix refresh graph/cache path, not the old full recompute implementation.

2026-06-27 fast full-prefix fallback diagnostic:

- Added `HIGGS_REFERENCE_KV_FAST_FALLBACK=1` as a diagnostic alternative to the old `HIGGS_REFERENCE_KV_FULL_FALLBACK=1`. It keeps the same greedy-margin trigger but, when triggered, calls the current cached full-prefix refresh graph (`run_kv_decode_window_all_layers_full_with_backend_cache` with fused audio head) instead of the old Torch-like full recompute loop.
- With naked KV (`HIGGS_REFERENCE_KV_WINDOW_REFRESH=0`) and `HIGGS_REFERENCE_KV_FALLBACK_MARGIN=0.09`, the 4-step gate now matches the full baseline and takes about 10s instead of roughly a minute with the old fallback.
- 20-step results:
  - margin `0.09`: 13 fast fallbacks, elapsed 12.36s, still diverges from step 12.
  - margin `0.12`: 16 fast fallbacks, elapsed 13.28s, exact 20-step match.
  - margin `0.20`: 18 fast fallbacks, elapsed 13.25s, exact 20-step match.
  - margin `0.50`: 20 fast fallbacks, elapsed 13.69s, exact 20-step match.
- 40-step results:
  - margin `0.12`: 32 fast fallbacks, elapsed 18.06s, diverges from step 20.
  - margin `0.20`: 37 fast fallbacks, elapsed 18.39s, diverges from step 33.
  - margin `0.50`: 40 fast fallbacks, elapsed 19.27s, exact 40-step match; non-trace audio RTF ~= 15.78.
- Conclusion: fast fallback proves the cached full-prefix refresh can be used as a much faster fallback than the old full recompute, but a margin-only sparse policy is not sufficient for 40-step quality unless the threshold is so high that it falls back every step. Keep it diagnostic for now; do not make it the public default.

2026-06-27 profile summary and fallback threshold check:

- Added aggregated `higgs_profile_summary` output for repeated timers. This is observability only; it does not change graph construction or sampling.
- Safe default full-prefix refresh still matches the quality baseline:
  - 20-step trace `/tmp/higgs-profile-summary20.json` has 0 code divergences vs `/tmp/higgs-default-full-20.json`.
  - 40-step trace `/tmp/higgs-profile-summary40.json` has 0 code divergences vs `/tmp/higgs-kv-windowfull40-skipboc-clean.json`.
- Default safe-path profile:
  - 20 steps: `reference_kv_window_refresh count=20 total_ms=5658 avg_ms=282 min_ms=176 max_ms=2104`.
  - 40 steps: `reference_kv_window_refresh count=40 total_ms=9729 avg_ms=243 min_ms=176 max_ms=2000`.
  - The first refresh is the cold CUDA/graph/weight path at roughly 2s; steady-state refresh is roughly 180-210ms per generated frame.
- Tested higher fast-fallback thresholds:
  - margin `0.30`: exact 40-step match, but 38/40 frames fall back.
  - margin `0.40`: exact 40-step match, but 40/40 frames fall back.
- This confirms the simple greedy-margin trigger is not a useful fast path: thresholds low enough to save many refreshes miss later 40-step divergences, and thresholds high enough to align are effectively the full-prefix refresh path plus raw-KV overhead.

2026-06-27 prefill and periodic-refresh diagnostics:

- Added trace-only diagnostic `HIGGS_REFERENCE_KV_PREFILL_VIA_WINDOW=1`, which seeds the prompt KV cache through the same full-window decode function used by the safe refresh path instead of `run_prefill_all_layers_into_backend_cache`.
- Result on the 4-step gate with naked KV: steps 0-2 match, but step3/codebook3 still diverges as `907 -> 165`. This rules out prompt prefill cache-writing as the root cause of the first visible drift.
- Tried and removed a fixed-period full-prefix refresh experiment that used the current cached full-window graph, not the old full recompute path:
  - period `2`: 10 refreshes in 20 steps, but still diverges at step 3.
  - period `4`: 5 refreshes in 20 steps, but still diverges at step 3.
  - period `8`: 3 refreshes in 20 steps, but still diverges at step 3.
- Tried and removed a "refresh first N steps then naked KV" experiment:
  - prefix `4`: first divergence at step 4.
  - prefix `8`: first divergence at step 9.
  - prefix `12`: first divergence at step 12.
- Interpretation: as soon as generation uses the naked single-token KV path for a sampling step, drift can flip greedy codes within one or two frames. A useful fast path therefore needs either a real fix inside `run_kv_decode_graph_all_layers_with_backend_cache` or a trigger that predicts dangerous steps before sampling without degenerating into every-step full-prefix refresh.

2026-06-27 generated-suffix refresh attempt:

- Tried a bounded refresh that prefills the reference prompt once, then each frame recomputes only the generated suffix (`BOC + generated audio embeddings`) with `run_kv_decode_window_all_layers_full_with_backend_cache`. The graph still attends over the full prompt KV cache, but it does not recompute prompt hidden states.
- It is fast on the 4-step gate: suffix refresh took roughly `51-65ms` per step instead of full-prefix `~180-210ms`.
- It is not correct: 4-step trace diverges from the full baseline starting at step 1:
  - step1/codebook1: `455 -> 957`
  - step2/codebook0: `291 -> 1017`
  - step3/codebook3: `907 -> 956`
- The likely reason is residual-stream continuity, not missing attention memory: although suffix queries can attend to prompt KV, the per-layer hidden stream for the first suffix token starts from its raw embedding instead of the prompt-contextualized sequence state produced by the full-prefix batched graph.
- The suffix-refresh code was removed. Keep this as evidence that "recompute generated suffix only" is not a valid bounded recompute path unless the residual stream boundary is redesigned.

2026-06-27 window=1 residual-boundary diagnostic:

- Added trace-only `HIGGS_REFERENCE_KV_COMPARE_WINDOW1=1` to compare naked single-token decode with `run_kv_decode_window_all_layers_full_with_backend_cache(..., tokens=1)` on the same BOC embedding and prompt KV cache state.
- Direct BOC comparison:
  - hidden abs diff: `mean_abs ~= 16.538`, `max_abs ~= 128.999`.
  - logits abs diff: `mean_abs ~= 0.775`, `max_abs ~= 7.313`.
- Temporarily replacing naked single-token decode with window=1 decode made the 4-step gate worse, diverging immediately at step 0 (`489 -> 457`) and continuing to drift. That replacement code was removed.
- Interpretation: the issue is not a narrow implementation bug where the window function with `tokens=1` can replace the naked decode. A valid bounded recompute must preserve the residual stream boundary from the full-prefix graph, not just attention KV visibility for one token.

2026-06-27 prompt-tail refresh attempt:

- Tried a bounded refresh that recomputes `prompt_tail + generated_suffix` rather than only the generated suffix. This was intended to give the generated audio tokens a short prompt residual-stream boundary while avoiding the cost of recomputing the full reference prompt.
- 4-step results:
  - prompt tail `8`: first divergence at step 1 (`455 -> 957`).
  - prompt tail `16`: first divergence at step 2 (`923 -> 297`).
  - prompt tail `32`: first divergence at step 2 (`923 -> 297`).
  - prompt tail `64`: first divergence at step 1 (`455 -> 957`).
- Timing was attractive (`~58-86ms` per refresh on this gate), but no tail size passed even the 4-step quality gate.
- The prompt-tail refresh code was removed. This reinforces that the bounded path cannot just choose a small residual boundary near the end of the prompt; the full-prefix graph's long residual stream is still part of the greedy-code parity contract.

2026-06-27 large prompt-tail sweep:

- Re-tested prompt-tail refresh with larger tails to find whether there is a useful cutoff below full-prefix recompute.
- 4-step gate:
  - tail `96`: diverges at step 1.
  - tail `128`: diverges at step 2.
  - tail `160`: diverges at step 2.
  - tail `192`: passes 4-step, refresh avg `~156ms`.
  - tail `224`: passes 4-step, refresh avg `~198ms`.
- 20-step gate:
  - tail `192`: diverges at step 7.
  - tail `224`: diverges at step 4.
  - tail `232`: diverges at step 3.
  - tail `236`: diverges at step 4.
  - tail `240`: passes 20-step, but the actual prompt length is `237`, so this clamps to full prompt and is equivalent to the safe full-prefix refresh.
- Conclusion: no prompt-tail value below full prompt met the 20-step gate. The prompt-tail code was removed again. This path does not provide a usable quality/speed tradeoff for the current reference prompt.

2026-06-27 full-prefix refresh sub-profile:

- Added sub-profile timers inside `run_kv_decode_window_all_layers_full_with_backend_cache` for graph construction, graph tensor allocation, graph compute, logits copy, and hidden copy.
- 20-step safe full-prefix parity remained exact vs the full baseline after adding timers.
- 20-step profile showed:
  - `reference_kv_window_refresh`: total `5569ms`, avg `278ms`, with the first cold refresh around `2017ms`.
  - `reference_kv_window_compute_graph`: total `1405ms`, avg `70ms`.
  - `reference_kv_window_alloc_graph`: total `329ms`, avg `16ms`.
  - logits/hidden copies were below millisecond resolution.
- 4-step follow-up with op-construction timing showed:
  - steady refresh `~177-179ms`.
  - steady graph compute `~65-66ms`.
  - steady graph allocation `~13-15ms`.
  - op construction was below millisecond resolution after cold start.
- Interpretation: the full-prefix path's remaining steady-state overhead is not explained by GGML graph construction or output copies. Compute is only about one third of wall-clock refresh time. Future optimization should inspect backend synchronization/context/buffer lifecycle or use a reusable graph/allocation strategy; simply trimming CPU-side op creation is unlikely to move RTF enough.

2026-06-27 full-prefix input upload profile:

- Added additional sub-profile timers around weight ensure, GGML context init, and input tensor uploads inside the full-window refresh function.
- 4-step safe full-prefix profile:
  - first refresh spends `~1644ms` in `reference_kv_window_ensure_weights`; this is cold weight loading and explains most of the first-step stall.
  - steady refresh total: `~176-205ms`.
  - steady `reference_kv_window_set_inputs`: `~75-76ms`.
  - steady `reference_kv_window_compute_graph`: `~65-66ms`.
  - steady `reference_kv_window_alloc_graph`: `~13-15ms`.
- This identifies the main steady-state fix target: full-prefix refresh uploads the entire `hidden_input` prefix every frame. For the current prompt this costs about as much as CUDA graph compute. A real speedup likely needs a persistent/reusable input tensor or graph/buffer path where the prompt embeddings are uploaded once and only the generated audio suffix is updated each frame.

2026-06-27 persistent input tensor attempt:

- Tried adding persistent `input` and `positions` tensors to the backend KV cache, then making the full-prefix refresh graph view those tensors instead of allocating temporary input tensors and uploading the whole prefix every frame.
- Code parity remained exact on the 4-step gate.
- The first version removed the large hidden tensor upload from `reference_kv_window_set_inputs`, but the same sync cost appeared on the next tiny positions upload: `set_inputs` stayed around `75-77ms`.
- The second version made positions persistent too. `set_inputs` dropped to `0ms`, but `reference_kv_window_compute_graph` rose from `~65-66ms` to `~139-142ms`; total steady refresh stayed around `~175-177ms`.
- Interpretation: the apparent input-upload cost is mostly backend synchronization placement, not removable wall-clock in this simple graph-view change. A real reusable-input optimization likely needs async upload/stream control or a reusable captured graph/buffer design, not just persistent tensors.
- The persistent input/positions implementation was removed; only the profile evidence remains.

2026-06-27 async backend IO attempt:

- Tried a diagnostic `HIGGS_REFERENCE_ASYNC_IO=1` path that used `ggml_backend_tensor_set_async`, `ggml_backend_graph_compute_async`, then explicit `ggml_backend_synchronize` before output reads.
- 4-step code parity remained exact.
- The timing did not improve:
  - steady `set_inputs` stayed around `75-77ms`.
  - `compute_graph` dropped to `~28-29ms`, but explicit `synchronize` added `~36-37ms`.
  - total steady refresh remained `~177-180ms`.
- This confirms that the current full-prefix refresh cost is not fixed by simply switching to GGML async calls inside the same per-step graph. The async experiment was removed.

2026-06-27 graph plan API check:

- Checked upstream GGML backend plan APIs as a possible way to reduce per-step graph dispatch/allocation overhead.
- The vendored CUDA backend does not implement the plan hooks:
  - `ggml_backend_cuda` has `graph_plan_create = NULL`.
  - `graph_plan_free = NULL`.
  - `graph_plan_compute = NULL`.
- Therefore `ggml_backend_graph_plan_create/compute` cannot be used for this CUDA path. A scheduler-based reserve path might save some allocation overhead, but current profiling shows allocation is only `~13-16ms` steady-state, while input upload/synchronization plus compute dominate. Do not prioritize graph-plan work unless the backend grows CUDA plan support.

2026-06-27 single-token mask alignment check:

- Tested whether naked single-token KV decode drift was caused by a mask-path mismatch: added `ggml_diag_mask_inf(..., past)` after the single-token `kq` scale to match the full-window code structure.
- Naked KV 4-step gate still diverged at step3/codebook3 as `907 -> 165`.
- The mask change was removed. The first divergence is not caused by the absence of an explicit causal mask in the single-token attention path.

2026-06-27 single-token RoPE contiguity check:

- RoPE parameters in naked single-token KV decode and full-window decode are already identical.
- Tested whether backend layout differences after RoPE caused drift by forcing single-token `q` and `k_cur` through `ggml_cont_3d` before cache write / attention.
- Naked KV 4-step gate still diverged at step3/codebook3 as `907 -> 165`.
- The contiguity change was removed. The first divergence is not fixed by forcing contiguous post-RoPE Q/K tensors.

2026-06-27 skip unused full-prefix hidden readback:

- In the safe full-prefix refresh path, logits are fused into the refresh graph and the returned last hidden is not used for sampling or for the next step; the next iteration recomputes the full prefix from embeddings.
- Added a `skip_hidden_readback` path for fused full-prefix refreshes so `run_kv_decode_window_all_layers_full_with_backend_cache` can return an empty hidden vector and avoid the unnecessary CPU readback/copy. Node dumps still force the old readback behavior.
- Gates:
  - 20-step trace: 0 divergences vs the full baseline.
  - 40-step trace: 0 divergences vs the aligned full-prefix baseline.
- Performance impact is negligible because hidden readback was already below millisecond resolution, but the safe path now avoids unused work.

2026-06-27 corrected window1 diagnostic:

- Fixed the trace-only `HIGGS_REFERENCE_KV_COMPARE_WINDOW1=1` diagnostic so it compares the one-token window decode at the same cache position as the naked BOC single-token decode. The previous diagnostic left `cache.used` at the prompt length, so the window path used `start = prompt_len - 1` and overwrote/compared the last prompt token instead of BOC.
- After the fix, naked single-token BOC decode and `run_kv_decode_window_all_layers_full_with_backend_cache(..., tokens=1)` match exactly on hidden and audio logits (`last_abs_mean = 0`, `last_max_abs = 0`).
- The naked KV 4-step gate still first diverges at step3/codebook3 as `907 -> 165`, while the safe full-prefix path remains exact for the 20-step baseline after this diagnostic-only change.
- Interpretation: the earlier single-vs-window1 huge difference was a diagnostic artifact. The real remaining mismatch is single/window1 incremental execution versus full-prefix batched execution, not an off-by-one bug inside the one-token window function.

2026-06-27 raw-vs-full-prefix trace diagnostic:

- Added trace-only `HIGGS_REFERENCE_KV_COMPARE_FULL_PREFIX=1` for non-full-prefix runs. It builds an independent temporary backend KV cache and compares the current raw/incremental hidden+logits against the safe full-prefix refresh for the same `generated_embedding_values` before sampling. This is diagnostic only and does not mutate the generation cache.
- 4-step naked KV diagnostic still produced the known first failure at step3/codebook3 (`907 -> 165`).
- Raw-vs-full-prefix hidden already differs strongly from step 0 (`hidden last_abs_mean ~= 0.97`, `max_abs ~= 13.37`) and remains around `~0.96-1.10` mean abs through step 3. Raw-vs-full-prefix logits diff is much smaller and stable (`last_abs_mean ~= 0.088`, `max_abs ~= 0.47-0.51`) until the close greedy margin flips step3/codebook3.
- Rechecked `HIGGS_REFERENCE_KV_HEAD_BATCH=241`: it still delays the first mismatch to step 7 but then diverges for 13/20 steps. Head batching fixes the early audio-head shape sensitivity but not the accumulating backbone hidden drift.
- Safe default `HIGGS_REFERENCE_BACKEND_KV=1` remains exact for the 20-step full baseline after this diagnostic addition.
- Interpretation: a simple "hidden/logit diff is large" trigger is not enough, because raw and full are already numerically different while sampled codes still match for several steps. The next useful fix must either make the incremental backbone numerically follow the batched path, or use a stronger pre-sampling disagreement check than scalar drift magnitude.

2026-06-27 rejected head-disagreement fallback:

- Tried an env-gated `HIGGS_REFERENCE_KV_FAST_FALLBACK_ON_HEAD_DISAGREE` experiment: after raw KV logits, compute a cheap batched audio-head view over repeated hidden columns; if raw top1 and batched-head top1 disagreed on any active codebook, call the cached full-prefix fallback.
- Head-disagreement alone used 9 cached full-prefix fallbacks in 20 steps but still missed the final 20-step codes.
- Combining head-disagreement with the existing fast fallback margin `0.10` passed the 20-step baseline with 13 fallbacks, but failed the 40-step baseline from step 35 onward with 5 divergent steps despite 27/40 fallbacks.
- The experiment was removed. It is better than scalar margin alone for some short gates, but it is not reliable enough for the 40-step acceptance gate and still adds extra audio-head work on every raw step.

2026-06-27 rejected layer-local FFN fake batch:

- Tried a trace-only `HIGGS_REFERENCE_KV_FFN_FAKE_BATCH=1` experiment inside `run_kv_decode_graph_all_layers_with_backend_cache`: for each layer's FFN only, repeat the single-column `mlp_normed` to `past + 1` columns, run `ffn_gate/up/down` as a batched matmul, then take the last column before the residual add. This avoided the earlier all-layer fake-batch shape problem and isolated the FFN matmul-width hypothesis.
- The 4-step naked KV gate still failed at step3/codebook3 as `907 -> 165`; the step3 codebook3 top5 stayed `[165, 907, 21, 901, 414]`.
- The experiment was removed. The first visible raw-KV failure is not fixed by making only the FFN matmuls use a full-prefix-width batch shape.

2026-06-27 runtime text-weight preload:

- Moved text block weights plus the fused full-prefix audio-head weights (`output_norm.weight`, `a.output`) into `PipelineRuntime::get(...)` preload for the selected backend. This reuses the existing runtime GGUF context and loaded-weight map; it does not add a new cache layer.
- Safe default `HIGGS_REFERENCE_BACKEND_KV=1` 20-step trace still exactly matches the full baseline.
- Profile effect: inside the safe full-prefix refresh, `reference_kv_window_ensure_weights` is now `0ms` for every refresh. The first refresh dropped from the old cold `~1.6-2.1s` weight-load stall to `~251ms` in this run; steady refresh remains `~176-185ms`.
- Full CLI wall time changes little because the same weight loading happens during backend runtime construction instead of the first refresh. A 40-step non-trace run wrote `/tmp/higgs-preload-safe40.wav` in `22.80s`, close to the previous `~23.09s`.
- This is a server/long-lived runtime cleanup, not the final fast reference path. It removes first-frame jitter for prewarmed `HiggsPipeline(model, backend)`, but it does not reduce the per-frame full-prefix recompute cost.

2026-06-27 rejected one-backend scheduler allocation:

- Tried replacing the full-prefix refresh graph allocation/compute path with an env-gated one-shot `ggml_backend_sched_new + ggml_backend_sched_alloc_graph + ggml_backend_sched_graph_compute` call to see whether GGML scheduler allocation could save the steady `~13-16ms` `alloc_graph` cost.
- The CUDA-only scheduler path aborts in upstream GGML: `ggml_backend_sched_new` asserts that the last backend is a CPU backend. A single CUDA backend is not a drop-in scheduler replacement for `ggml_backend_alloc_ctx_tensors`.
- The experiment was removed. A real scheduler attempt would need a CUDA+CPU backend list and tensor/backend assignment audit, which is more complexity than the measured allocation cost justifies right now.

2026-06-27 rejected layer-local QKV fake batch:

- Tried a trace-only `HIGGS_REFERENCE_KV_QKV_FAKE_BATCH=1` experiment in `run_kv_decode_graph_all_layers_with_backend_cache`: repeat each layer's single-column attention-normalized hidden to `past + 1` columns, run only `attn_q/k/v` as batched matmuls, then view the last column back to the normal one-token path before RoPE/cache write.
- The diagnostic did not reach the 4-step gate. GGML graph construction aborted with `GGML_ASSERT(cgraph->n_nodes < cgraph->size)`, because repeating QKV inside every layer pushed the all-layer one-token graph past the default graph node capacity.
- The experiment was removed. A proper QKV fake-batch test would need a larger graph allocation or a narrower single-layer harness, so it is not a cheap production-path fix.

2026-06-27 current reference-performance decision boundary:

- Do not repeat small-window, prompt-tail, suffix-only, margin-only fallback, head-disagreement fallback, FFN fake-batch, or all-layer QKV fake-batch experiments. They either fail the 20/40-step gates or add complexity without reducing the full-prefix per-frame cost.
- Do not use scalar hidden/logit diff magnitude as a fallback trigger. Raw and full-prefix paths differ numerically from step 0 while generated codes still match for several steps.
- The only currently quality-correct path is still the safe full-prefix refresh behind `HIGGS_REFERENCE_BACKEND_KV=1`; runtime text-weight preload only removes first-frame weight-load jitter for long-lived server pipelines.
- The next useful implementation direction is not another heuristic trigger. It is either:
  - a smaller, explicitly validated single-layer harness that isolates the single-column-vs-batched `ggml_mul_mat` behavior without blowing graph capacity, or
  - a real reusable full-prefix graph/input-buffer design that reduces the measured steady `set_inputs + compute` wall time while preserving exact 20/40-step codes.

2026-06-27 block0 q-only harness clarification:

- Reused the existing `--self-check-block0-attn-qkv-graph` with `HIGGS_ATTENTION_Q_ONLY=1` instead of adding new code.
- Same-gate full-prefix input:
  - `HIGGS_ATTENTION_NORMED_INPUT=/tmp/higgs-full-stepnamed4b/cpp_step3_cpp_blk0_attn_normed.f32`
  - `HIGGS_ATTENTION_TOKENS=241`
  - output `/tmp/higgs-qonly-step3-fullprefix.q_proj`
  - result: harness output matches `/tmp/higgs-full-stepnamed4b/cpp_step3_cpp_blk0_q_proj.f32` exactly (`mean_abs = 0`, `max_abs = 0`).
- Repeated single-token input:
  - `HIGGS_ATTENTION_NORMED_INPUT=/tmp/higgs-kv-bounded-128/cpp_kv_decode_blk0_attn_normed_past240.f32`
  - `HIGGS_ATTENTION_TOKENS=1` vs `241`
  - single-column and repeated-241 last-column q-projection differ by `mean_abs ~= 1.97e-4`, `p99 ~= 0.00140`, matching the known single-column-vs-batched matmul scale.
- Interpretation: the full-prefix q-projection dump is not a dump artifact; the q-only harness reproduces it exactly when fed the real full-prefix `attn_normed` tensor. Repeating a single KV `attn_normed` column only tests matmul batch-width sensitivity; it does not reconstruct the true full-prefix residual stream. Treat raw-KV hidden drift as a residual-stream/full-prefix execution mismatch, not merely a standalone Q projection bug.

2026-06-27 rejected window input markers:

- Tried marking the full-prefix refresh graph's `hidden` and `positions` tensors with `ggml_set_input(...)` to see whether backend allocation/input staging would get cheaper.
- 20-step generated codes still matched `/tmp/higgs-default-full-20.json`, so the change was numerically harmless.
- Profile did not improve: `reference_kv_window_set_inputs` stayed around `~79ms`, steady `compute_graph` stayed around `~70ms`, and the first refresh got worse in the test run (`~356ms` vs the previous preloaded `~251ms`).
- The markers were removed. This is not a useful path toward lowering the safe full-prefix refresh RTF.

2026-06-27 completed QKV fake-batch diagnostic:

- Retried the earlier `HIGGS_REFERENCE_KV_QKV_FAKE_BATCH=1` diagnostic with a larger `ggml_new_graph_custom(...)` graph and corrected GGML-order projection shapes (`q = head_dim * n_head`, `k/v = head_dim * n_head_kv`).
- The diagnostic now runs, so the previous graph-capacity abort is no longer hiding the result.
- Same 4-step naked KV gate still fails at the known first divergence: full baseline step3/codebook3 is `907`, QKV fake-batch still samples `165`.
- The experiment was removed. The first visible raw-KV failure is not fixed by making only Q/K/V projections use a full-prefix-width batch shape.

2026-06-27 full-prefix gallocr default:

- Replaced the safe full-prefix refresh graph allocation path with GGML's `ggml_gallocr_alloc_graph(...)` using the selected backend's default buffer type. This keeps the same GGML graph semantics and only changes tensor allocation/reuse. `HIGGS_REFERENCE_KV_DISABLE_GALLOCR=1` restores the old `ggml_backend_alloc_ctx_tensors(...)` path for diagnostics.
- 20-step default `HIGGS_REFERENCE_BACKEND_KV=1` trace matches `/tmp/higgs-default-full-20.json` exactly.
- 40-step `HIGGS_REFERENCE_BACKEND_KV=1` trace matches the existing aligned full-prefix/Python traces exactly.
- Profile effect on the same reference wav/text/GGUF/temperature=0 gate:
  - previous safe refresh after preload/revert: `reference_kv_window_refresh avg ~= 198ms`, `set_inputs avg ~= 79ms`, `alloc_graph avg ~= 14ms`.
  - gallocr 20-step: `refresh avg ~= 83ms`, `set_inputs avg ~= 0ms`, `alloc_graph avg ~= 1-2ms`.
  - gallocr 40-step: `refresh avg ~= 76ms`, `set_inputs avg ~= 0ms`, `alloc_graph avg ~= 1ms`.
- Non-trace 40-step output `/tmp/higgs-gallocr40-full.wav`: elapsed `18.92s`, duration `1.52s`, RTF `~= 12.45`, down from the prior safe path around `22.8-23.1s` elapsed / RTF `~15`.
- This is a real public safe-path speedup, but still not the final bounded recompute/KV hidden-drift fix requested by the long-term goal.

2026-06-27 CPU backend check:

- Ran the same 4-step naked KV diagnostic on CPU:
  - `HIGGS_REFERENCE_BACKEND_KV=1`
  - `HIGGS_REFERENCE_KV_WINDOW_REFRESH=0`
  - `HIGGS_TRACE_ONLY=1`
  - output trace `/tmp/higgs-cpu-rawkv4.json`
- CPU matches the full baseline for steps 0-2 and fails at the same first visible point as CUDA: step3/codebook3 full baseline `907`, CPU naked KV `165`.
- Also ran CPU safe full-prefix refresh for 4 steps (`/tmp/higgs-cpu-fullprefix4.json`). It fails at the same step3/codebook3 point (`907 -> 165`).
- Interpretation: CPU is not a reliable quality baseline for this gate. The accepted quality path remains CUDA/Python-aligned full-prefix refresh; do not use CPU naked/full-prefix agreement as proof that the incremental path is semantically correct.

2026-06-27 prompt prefill exclusion:

- Re-ran the CUDA 4-step naked KV gate with `HIGGS_REFERENCE_KV_PREFILL_VIA_WINDOW=1`, so the prompt KV cache is seeded through the same full-window function used by the safe refresh path instead of the normal prompt prefill graph.
- Trace `/tmp/higgs-prefill-via-window-raw4.json` still matches steps 0-2 and fails at the same first visible point: step3/codebook3 full baseline `907`, naked KV `165`.
- Interpretation: the first 4-step failure is not caused by the prompt prefill graph writing a different prompt KV cache. The remaining suspect is the generated-token incremental chain after BOC, not initial prompt cache seeding.

2026-06-27 contextual buffer writeback fix:

- Fixed a diagnostic-state bug in the non-full-prefix path: after sampling a new audio frame, `generated_contextual_values` was appended with the raw audio embedding, then `run_kv_decode_graph_all_layers_with_backend_cache(...)` produced the contextual hidden, but the contextual buffer tail was not overwritten with that decoded hidden.
- The default full-prefix path is unaffected because it does not run the post-sample single-token decode.
- Default `HIGGS_REFERENCE_BACKEND_KV=1` 20-step trace still matches `/tmp/higgs-default-full-20.json` exactly after the fix.
- Re-tested the contextual small-window diagnostic with `HIGGS_REFERENCE_KV_WINDOW_REFRESH=8 HIGGS_REFERENCE_KV_WINDOW_CONTEXTUAL_INPUT=1`; it still fails at the same step3/codebook3 point (`907 -> 165`).
- Interpretation: the old contextual-window diagnostic had a real bookkeeping bug, now fixed, but this does not solve the bounded refresh quality problem by itself.

2026-06-27 single-token attention mask recheck:

- Re-tested the current code with a temporary explicit `ggml_diag_mask_inf(..., past)` after the single-token KV decode attention score scale in `run_kv_decode_graph_all_layers_with_backend_cache(...)`.
- CUDA naked KV 4-step trace `/tmp/higgs-single-mask-raw4.json` still matches steps 0-2 and fails at step3/codebook3 as `907 -> 165`.
- The mask line was removed again. The current failure is not caused by single-token attention seeing future/uninitialized positions in the `total = past + 1` KV view.

2026-06-27 current accepted path baseline:

- Re-ran the current default public path:
  - `HIGGS_REFERENCE_BACKEND_KV=1`
  - same reference wav/text/GGUF
  - temperature `0`
  - 40 steps
- Trace `/tmp/higgs-current-default40.json` matches the existing aligned traces exactly:
  - `/tmp/higgs-backendkv-default40.json`
  - `/tmp/higgs-skip-hidden40.json`
  - `/tmp/higgs-py-kv40-from-cpp-input.json`
  - `/tmp/higgs-gallocr40.json`
- Non-trace output `/tmp/higgs-current-default40-full.wav`: elapsed `16.49s`, duration `1.52s`, RTF `~= 10.85`.
- This is the current usable quality path. It satisfies public non-trace generation and code parity, but it is still full-prefix refresh with gallocr reuse, not the requested true bounded/KV hidden-drift fix.

2026-06-28 llama.cpp scheduler comparison:

- Added an env-gated scheduler diagnostic for the safe full-prefix refresh path:
  - `HIGGS_REFERENCE_KV_SCHED=1`
  - keeps the current gallocr full-prefix path as the default fallback.
  - uses a CUDA backend plus a CPU backend, matching llama.cpp's requirement that the scheduler backend list include CPU.
- 20-step scheduler trace `/tmp/higgs-sched20.json` matches `/tmp/higgs-default-full-20.json` exactly.
- 40-step scheduler trace `/tmp/higgs-sched40.json` matches the existing aligned traces exactly:
  - `/tmp/higgs-backendkv-default40.json`
  - `/tmp/higgs-skip-hidden40.json`
  - `/tmp/higgs-py-kv40-from-cpp-input.json`
  - `/tmp/higgs-gallocr40.json`
  - `/tmp/higgs-current-default40.json`
- Profile on the same reference wav/text/GGUF/temperature=0 gate:
  - scheduler 40-step trace: `reference_kv_window_refresh avg ~= 86ms`, `compute_graph avg ~= 81ms`, `alloc_graph avg ~= 2ms`, first refresh `~= 490ms`.
  - full scheduler wav `/tmp/higgs-sched40-full.wav`: elapsed `21.76s`, duration `1.52s`, RTF `~= 14.32`.
- Interpretation: merely replacing the gallocr compute path with `ggml_backend_sched_alloc_graph(...)` + `ggml_backend_sched_graph_compute(...)` preserves codes but is slower than the current gallocr default (`RTF ~= 10.85`). The missing llama.cpp mechanism is not just scheduler dispatch; it is the larger `sched_reserve()` / `graph_reserve()` / reusable graph-result structure that reserves worst-case graphs and reuses stable graph/input tensor objects across decode calls. Higgs currently still builds a fresh ggml context and graph inside each full-prefix refresh, so scheduler allocation alone cannot solve the RTF problem.

2026-06-28 scheduler reserve-on-growth check:

- Added the smallest real reserve hook to the env-gated scheduler path: when the current full-prefix token count exceeds the previously reserved count, call `ggml_backend_sched_reserve(...)` on that actual graph before `sched_reset + sched_alloc_graph`.
- This avoids a fake max-shape graph builder and keeps GGML's documented rule intact: after `ggml_backend_sched_reset(...)`, old graph tensor pointers are dangling and must be discarded.
- Verification:
  - 20-step trace `/tmp/higgs-sched-reserve20.json` matches `/tmp/higgs-default-full-20.json` exactly.
  - 40-step trace `/tmp/higgs-sched-reserve40.json` matches all current aligned 40-step traces exactly.
  - full scheduler-reserve wav `/tmp/higgs-sched-reserve40-full.wav`: elapsed `21.16s`, duration `1.52s`, RTF `~= 13.92`.
- Result: codes are correct, but the path still fails the speed gate and remains slower than current gallocr default (`RTF ~= 10.85`). The remaining blocker is structural: the current full-prefix refresh graph is built inside one execution function with local input tensors, so a true llama.cpp-style worst-case reserve requires extracting a reusable graph builder/result object first. Without that refactor, scheduler reserve can only reserve observed graphs as they appear; it cannot reserve and reuse a max full-prefix topology the way llama.cpp does.

2026-06-28 dummy max-reserve scheduler check:

- Added an env-gated scheduler reserve helper that builds a dummy max full-prefix graph with dummy KV tensors, then calls `ggml_backend_sched_reserve(...)`. The dummy graph does not set real inputs, does not compute, and does not write `cache.key` / `cache.value`.
- Real refresh execution still builds the normal full-prefix graph with real KV cache tensors, then runs `sched_reset + sched_alloc_graph + sched_graph_compute`.
- Verification on the same reference wav/text/GGUF/temperature=0 gate:
  - 4-step trace `/tmp/higgs-maxreserve4.json` matches the existing full-prefix trace prefix.
  - 20-step trace `/tmp/higgs-maxreserve20.json` matches `/tmp/higgs-default-full-20.json` exactly.
  - 40-step trace `/tmp/higgs-maxreserve40.json` matches all current aligned 40-step traces exactly.
  - full scheduler max-reserve wav `/tmp/higgs-maxreserve40-full.wav`: elapsed `22.07s`, duration `1.52s`, RTF `~= 14.52`.
- Profile result:
  - max-reserve path drives `reference_kv_window_alloc_graph` to `~0ms` after first allocation.
  - 40-step trace still spends `reference_kv_window_compute_graph total ~= 3150ms`, avg `~= 78ms`; `reference_kv_window_refresh avg ~= 80ms`.
- Interpretation: dummy max reserve works and removes scheduler allocation overhead, but it does not beat the current gallocr default (`RTF ~= 10.85`). The remaining main cost is full-prefix compute itself, not graph allocation/split. Further speedup needs reducing the recomputed full-prefix workload or making the actual compute graph cheaper; scheduler reserve alone is now exhausted.

2026-06-28 scheduler async compute alignment:

- Changed the env-gated scheduler execution path from `ggml_backend_sched_graph_compute(...)` to explicit `ggml_backend_sched_graph_compute_async(...)` followed by `ggml_backend_sched_synchronize(...)` before logits/hidden readback. This matches llama.cpp's graph compute shape while preserving the required synchronous readback semantics.
- 20-step trace `/tmp/higgs-maxreserve-async20.json` matches `/tmp/higgs-default-full-20.json` exactly.
- Profile remains the same order of magnitude:
  - `reference_kv_window_alloc_graph avg ~= 0ms`
  - `reference_kv_window_compute_graph avg ~= 81ms`
  - `reference_kv_window_refresh avg ~= 83ms`
- Interpretation: explicit async+synchronize is behaviorally safe, but it does not change the bottleneck because every step immediately needs logits/hidden output. The remaining cost is still full-prefix compute.

2026-06-28 full-prefix stage profile:

- Added `HIGGS_REFERENCE_KV_STAGE_PROFILE=1` for the env-gated scheduler path. It uses `ggml_backend_sched_set_eval_callback(...)` and named `kvstage:*` marker tensors to split full-prefix refresh compute into stage buckets.
- The diagnostic is intentionally separate from normal RTF runs because GGML's eval callback synchronizes at marker boundaries and slows execution.
- Stage markers currently cover:
  - first-layer attention project/rope
  - first-layer KV write/view/repeat
  - first-layer QK matmul/mask
  - first-layer softmax
  - first-layer V matmul
  - all-layer attention output markers
  - all-layer MLP markers
  - logits head
- Verification:
  - Normal scheduler 20-step trace `/tmp/higgs-stage-marker20.json` matches `/tmp/higgs-default-full-20.json` exactly.
  - Normal scheduler 40-step trace `/tmp/higgs-stage-marker40.json` matches `/tmp/higgs-current-default40.json` exactly.
  - Build still passes.
- Short diagnostic profile (`HIGGS_REFERENCE_KV_STAGE_PROFILE=1`, 2 steps) shows the largest repeated bucket is all-layer MLP, not scheduler allocation/split. First-token diagnostics also show noticeable one-time first-layer KV write/view/repeat and QK matmul costs, but steady refresh is dominated by repeated layer compute, especially MLP.
- Next optimization direction: inspect/optimize FFN/MLP matmul path first (`ffn_gate`, `ffn_up`, `ffn_down`), then revisit attention/QK only if MLP improvements do not move RTF. Do not spend more time on scheduler reserve; alloc/split is already near zero.

2026-06-28 FFN substage profile:

- Extended `HIGGS_REFERENCE_KV_STAGE_PROFILE=1` with FFN markers:
  - `reference_kv_ffn_norm`
  - `reference_kv_ffn_gate_matmul`
  - `reference_kv_ffn_up_matmul`
  - `reference_kv_ffn_silu_mul`
  - `reference_kv_ffn_down_matmul`
  - `reference_kv_ffn_residual_add`
- Verification:
  - 20-step trace `/tmp/higgs-ffnmarker20.json` matches `/tmp/higgs-default-full-20.json` exactly.
  - 40-step trace `/tmp/higgs-ffnmarker40.json` matches `/tmp/higgs-current-default40.json` exactly.
  - Build still passes.
- Short diagnostic profile (`/tmp/higgs-ffnprof2.json`) shows the FFN cost is concentrated in the three large matmuls:
  - `ffn_up_matmul` total `~= 39ms`
  - `ffn_gate_matmul` total `~= 36ms`
  - `ffn_down_matmul` total `~= 35ms`
  - `ffn_norm`, `silu_mul`, and residual add are not meaningful contributors.
- Tried no math-changing optimization. There is no safe one-line GGML graph tweak for only `ffn_up`/`ffn_gate`/`ffn_down`: they are required large matrix multiplications and use the existing GGML `mul_mat` path. The next concrete optimization direction is a fused/packed FFN diagnostic, such as concatenating `ffn_gate` + `ffn_up` weights into one larger projection and splitting the result before `silu*up`, then verifying strict code parity. That would change exporter/runtime tensor layout and needs its own gate.

2026-06-28 fused FFN diagnostic:

- Added `HIGGS_REFERENCE_KV_FUSED_FFN=1` for the scheduler/full-prefix refresh path. It builds runtime-only packed per-layer `ffn_gate_up` tensors from the already loaded `ffn_gate.weight` + `ffn_up.weight`, runs one `ggml_mul_mat`, then splits the result back into gate/up before `silu(gate) * up` and the normal `ffn_down`. The GGUF exporter and default gallocr path are unchanged.
- The first fused attempt used direct `ggml_view_2d` slices and diverged at step 3/codebook 2 (`469 -> 986`). Adding `ggml_cont_2d` after each slice restored parity and also avoided a slow non-contiguous CUDA path.
- Verification on the same reference wav/text/GGUF with temperature `0`:
  - 20-step trace `/tmp/higgs-fusedffn20-cont.json` matches `/tmp/higgs-default-full-20.json` exactly.
  - 40-step trace `/tmp/higgs-fusedffn40.json` matches `/tmp/higgs-current-default40.json` exactly.
  - 40-step non-trace wav `/tmp/higgs-fusedffn40-full.wav` is complete; duration `1.52s`.
- Profile result:
  - one-time runtime packing cost is `~= 2.2-2.5s` per fresh `HiggsPipeline`/cache.
  - 40-step fused scheduler trace `reference_kv_window_compute_graph total ~= 3225ms`, avg `~= 80ms`; after the first packed refresh, steady refresh is `~= 66-93ms`.
  - 40-step non-trace CLI elapsed `22.33s`, RTF `~= 14.69`, still slower than the current default gallocr baseline `RTF ~= 10.85` because CLI pays cold packing and non-reference work.
- Interpretation: packed gate/up is behaviorally safe only with contiguous split outputs. It reduces the env-gated scheduler/full-prefix refresh compute versus the unfused scheduler path, but it is not yet a default-path win. The next concrete direction is to avoid cold runtime repacking for long-lived/server use or move the packing/exported layout into model load, then benchmark against the gallocr default without the scheduler-only overhead.

2026-06-28 reference prompt embedding overlay check:

- Audited the Torch/sglang Higgs path and Python GGML path:
  - prompt keeps `-100` placeholders in the reference-audio span.
  - text ids are made safe only for text `get_rows`.
  - delayed reference audio embeddings overwrite the original placeholder positions with `ggml_set_rows`.
  - generated audio frames are appended after the prompt as normal audio embeddings.
- C++ already had equivalent CPU-side vector overwrite for reference prompt embeddings, but the runtime path now uses an explicit GGML graph-level `ggml_set_rows` helper (`run_reference_prompt_embedding_graph_with_runtime`) so the C++ execution boundary matches `projects/higgs-audio-ggml-py/runtime.py` and `graphs.py`.
- Verification on the same reference wav/GGUF/temperature=0:
  - 1-step BOC audio embedding stat matches Python GGML exactly within printed precision (`kv_boc_input_embed` vs Python `audio_embeds`).
  - Full step-0 `inputs_embeds` dump matches Python GGML exactly:
    `/tmp/higgs-embed-cpp/cpp_inputs_embeds.f32` vs
    `/tmp/higgs-embed-py/py_step0_inputs_embeds.f32` has `mean_abs = 0`,
    `p99 = 0`, `max_abs = 0`.
  - 20-step C++ trace `/tmp/higgs-overlay20b.json` and Python GGML trace `/tmp/higgs-overlay20b-py.json` match prompt ids, reference codes, delayed codes, finalized codes, and sampled codes. Raw `top1` differs for a forced-delay codebook at step 0, but sampled codes are identical after delay rules.
  - 40-step C++ trace `/tmp/higgs-overlay40.json` and Python GGML trace `/tmp/higgs-overlay40-py.json` match prompt ids, reference codes, delayed codes, finalized codes, and sampled codes. Raw `top1` still differs at step 17/codebook 2 while sampled codes remain identical.
  - Non-trace sample written to `/tmp/higgs-overlay-fixed-sample.wav` (`24kHz`, mono, `3.12s`).
- Follow-up same-text final check:
  - 20-step `/tmp/higgs-final20.json` vs `/tmp/higgs-final20-py.json` matches prompt ids, reference codes, delayed codes, finalized codes, and sampled codes; raw top1 still differs at step 17/codebook 2.
  - 40-step `/tmp/higgs-final40.json` vs `/tmp/higgs-final40-py.json` matches prompt ids, reference codes, delayed codes, finalized codes, and sampled codes; raw top1 still differs at step 17/codebook 2.
  - Final listening sample written to `/tmp/higgs-reference-overlay-final.wav` (`24kHz`, mono, `3.12s`).
- Codec decode check for the same finalized 40-step codes:
  - C++ decode trace output `/tmp/higgs-final40-cpp-decode.wav` and Python GGML CPU codec output `/tmp/higgs-final40-py-decode.wav` have the same stream info: `24kHz`, mono, `36480` frames.
  - Waveform stats are close:
    - C++ mean_abs `0.0857663`, p99 `0.4392767`, max_abs `0.7834101`.
    - Python mean_abs `0.0857510`, p99 `0.4394391`, max_abs `0.7833186`.
    - abs diff mean `6.8e-05`, p99 `3.97e-04`, max `0.00174`.
  - Python GGML CUDA codec decode currently trips `ggml-cuda/conv-transpose-1d.cu`'s F32 assertion, so the codec comparison used Python GGML CPU. C++ CLI decode succeeded.
- Remaining risk: this closes the reference prompt overlay mismatch at the GGML API boundary and shows finalized-code waveform decode is close to Python GGML. It still does not prove the audible quality issue is fixed; the user still needs to listen to `/tmp/higgs-reference-overlay-final.wav`. If quality is still poor, the next divergence target is upstream/PyTorch codec quality or generated-code semantics beyond the Python GGML baseline, not C++ prompt embedding.

2026-06-29 reference codec resampler divergence:

- Rechecked the current user reference wav:
  `/root/code/ggbond/models/可哪怕位于堂堂超一品官职,在十二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应.wav`
  is `32kHz`, mono, 16-bit PCM. C++ therefore enters `resample_linear()` for both
  acoustic `32k -> 24k` and semantic `32k -> 16k`.
- C++ trace field bug fixed: `reference_codes` now writes raw `[T, 8]` codes
  and `delayed_reference_codes` writes delayed `[T + 7, 8]` codes. This removed
  a false shape mismatch in trace comparison.
- With the original 32k wav, C++ raw reference codes still diverge from Python
  GGML/Torch: `1131 / 1704` mismatches; first mismatch is frame `0`, codebook
  `6` (`442/581` depending on acoustic conv experiment vs Python `926`).
- Activation comparison showed the first real upstream issue is the reference
  waveform preprocessing, not prompt embedding or text sampling:
  - C++ padded acoustic waveform vs Python torchaudio-resampled padded waveform:
    mean_abs `5.5e-4`, p99 `0.00697`, max_abs `0.0403`.
  - C++ acoustic encoder output vs Python: mean_abs about `0.224`, p99 about
    `0.947`, max_abs about `2.416`.
- Diagnostic proof: when the same reference audio is pre-resampled to 24k with
  torchaudio and passed to C++, acoustic output matches Python exactly
  (`mean_abs=0`, `p99=0`, `max_abs=0`). Remaining semantic resample mismatch
  still leaves `93 / 1704` reference-code mismatches, proving the remaining
  issue is C++ `24k -> 16k` semantic resampling.
- Tried moving acoustic encoder convs to the Python-style F32 im2col path; it
  did not reduce the current mismatch because the dominant drift is already in
  the resampled input.
- Tried llama.cpp's vendored `miniaudio.h` resampler, but this version only
  exposes `ma_resample_algorithm_linear`; it did not improve alignment with
  torchaudio and was reverted.
- Checked upstream `mackron/miniaudio` directly after the user suggested copying
  it: the built-in resampler still exposes linear/custom algorithms, not a
  torchaudio-style bandlimited sinc path. Directly copying upstream miniaudio is
  therefore not expected to fix the current reference-code divergence by itself.
- Replaced the local linear resampler with a direct C++ port of torchaudio's
  default polyphase `sinc_interp_hann` kernel formula, and fixed the semantic
  contract to derive `16k` semantic audio from the codec `24k` waveform instead
  of resampling the original `32k` input directly.
  - C++ padded acoustic waveform vs Python torchaudio-resampled padded waveform:
    mean_abs `3.31e-9`, p99 `2.98e-8`, max_abs `1.19e-7`.
  - C++ padded semantic waveform vs Python torchaudio-resampled padded waveform:
    mean_abs `5.95e-9`, p99 `4.47e-8`, max_abs `1.79e-7`.
  - Raw reference-code mismatch is now `0 / 1704`.
- Cross-feeding confirmed the previous C++ mismatch was the reference resampler
  boundary, not the ported codec graphs:
  - Feeding C++ acoustic waveform dump back through Python GGML acoustic encode
    reproduces the C++ acoustic feature dump exactly (`mean_abs=0`, `p99=0`,
    `max_abs=0`).
  - Feeding C++ semantic waveform dump back through Python GGML HuBERT/semantic
    encode reproduces the C++ semantic feature dump exactly (`mean_abs=0`,
    `p99=0`, `max_abs=0`).
  - Cross-feeding C++ acoustic + Python semantic gives `72 / 1704` code
    mismatches, while Python acoustic + C++ semantic gives `25 / 1704`;
    acoustic resampling remains the dominant source.
- A quick local parameter sweep of the earlier small sinc approximation found
  only a marginal alternative (`radius=10`, `rolloff=0.96`, `68 / 1704`
  mismatches), so it was not used. The root fix was matching torchaudio's
  kernel/alignment instead of retuning constants.
- Fixed reference trace finalization: reference generation rows are already
  delayed rows (`BOC + sampled delayed codes`), so C++ must finalize directly
  from those rows instead of applying the delay pattern a second time. This
  brought C++ `finalized_codes` into the same shape/content as Python.
- Enlarged the safe full-prefix KV refresh graph allocation to avoid
  `GGML_ASSERT(cgraph->n_nodes < cgraph->size)` after the fused/full-prefix
  graph grew past the default graph capacity.
- Current gates after the resampler/finalize fixes:
  - CPU and CUDA builds pass.
  - 1-step CPU trace with the real 32k reference wav reports raw reference-code
    mismatch `0 / 1704` vs Python GGML/Torch preprocessing.
  - 20-step C++ CUDA safe full-prefix KV trace matches Python GGML trace:
    `prompt_ids`, `reference_codes`, `delayed_codes`, `finalized_codes`, and
    per-step top1/top5/sampled codes all pass.
  - 40-step C++ CUDA safe full-prefix KV trace matches Python GGML trace with
    finalized codes shape `(8, 32)` and `trace: ok steps=40`.
  - Python GGML codec decode of the C++ 40-step trace writes
    `/tmp/higgs-cpp-taresample-kv40-pydecode.wav`; waveform shape `(30720,)`,
    mean_abs `0.087668136`, p99_abs `0.390890062`, max_abs `0.680188715`.
  - Current official Torch/safetensors 20-step trace was regenerated as
    `/tmp/higgs-torch-current-step20.json`. Python/C++ GGML still differ from
    official Torch by one reference-code boundary decision (`1 / 1704`, first
    frame `61`, codebook `7`, GGML `150` vs Torch `394`). That reference-code
    difference flips delayed generation rows at step 18, but 20-step
    `finalized_codes` still match Torch exactly with shape `(8, 12)`.
  - Cross-feed diagnostic for that single mismatch:
    - official `codec.model.encode()` vs hand-composed Torch acoustic + Torch
      semantic + GGML `fc+RVQ` tail differs by the same `1 / 1704` at frame
      `61`, codebook `7`.
    - Torch acoustic + GGML semantic projection from official Torch semantic
      features matches official codes exactly (`0 / 1704`).
    - GGML acoustic + GGML semantic projection from official Torch semantic
      features also matches official codes exactly (`0 / 1704`).
    - Full native HuBERT semantic remains near the same RVQ boundary; depending
      on branch combination it gives `1` to `5` code flips out of `1704`.
    This narrows the remaining official-vs-native issue to an RVQ boundary
    decision around the semantic/tail features, not acoustic encode, prompt
    embedding, text AR, or codec decode.
  - RVQ distance diagnostic at frame `61`, codebook `7`:
    - official top2 is `394` then `150`, distances `58.7565` vs `58.7774`
      (margin about `0.0209`).
    - GGML F16-tail path can reduce that already tiny margin to about
      `0.0028`, enough to flip the nearest neighbor.
  - Updated the Python exporter precision policy so `a.fc.*` and `a.q.*` stay
    F32 in otherwise-F16 exports. The mixed GGUF is now
    `/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf` with manifest
    `/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.manifest.json`.
    Manifest check: `a.fc.*` and `a.q.*` are `float32`; large acoustic/text
    matrices remain F16.
  - Mixed GGUF reference encode gates:
    - acoustic + official Torch semantic feature tail: `0 / 1704` mismatches.
    - full native semantic reference encode: `0 / 1704` mismatches.
  - With the mixed GGUF and `NVIDIA_TF32_OVERRIDE=0`, current Python GGML CUDA
    20-step trace matches official Torch/safetensors 20-step trace completely:
    `prompt_ids`, `reference_codes`, `delayed_codes`, `finalized_codes`, and
    per-step top1/top5/sampled codes all pass. Without disabling TF32, a later
    text-logit near-tie at step `12`, codebook `0` can flip (`646` vs `322`).
  - Attempted current official Torch/safetensors 40-step trace, but the
    hand-written Torch full-forward trace was interrupted after several minutes
    in MLP projection. Current 40-step official parity remains unverified.

2026-06-29 mixed-GGUF C++ reference generation gate:

- Re-ran C++ CUDA safe full-prefix reference generation against the mixed GGUF
  `/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf`, where codec
  encode tail tensors `a.fc.*` and RVQ tensors `a.q.*` stay F32.
- Environment for strict greedy parity:
  `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=100000`,
  `NVIDIA_TF32_OVERRIDE=0`.
- 20-step C++ mixed trace:
  `/tmp/higgs-cpp-mixed-kv20.json`.
  It matches Python mixed trace `/tmp/higgs-py-mixed-models-notf32-step20.json`
  for `prompt_ids`, `reference_codes`, `delayed_codes`, `finalized_codes`, and
  per-step trace codes.
- 40-step C++ mixed trace:
  `/tmp/higgs-cpp-mixed-kv40.json`.
  It matches Python mixed trace `/tmp/higgs-py-mixed-notf32-step40.json` for
  `prompt_ids`, `reference_codes`, `delayed_codes`, `finalized_codes`, and
  per-step trace codes.
- The Python mixed 40-step trace also matches the official Torch/safetensors
  40-step trace `/tmp/higgs-torch-bf16-rope-step40.json` on the same fields.
  This closes the previous 40-step official parity gap for the mixed model
  path.
- C++ final decode artifact:
  `/tmp/higgs-cpp-mixed-final-step40.wav`.
  Stats: 24 kHz mono, 30,720 samples, duration `1.28s`, mean_abs
  `0.0900082`, p99_abs `0.383301`, max_abs `0.675293`, RMS `0.122302`.
  Wall time was `32.84s`, so RTF is about `25.66` on the current safe
  full-prefix diagnostic path.
- Quality note: this path is correctness-first and still slow. It proves the
  mixed C++ codes are aligned through Python GGML to official Torch; it is not
  the final optimized serving path.

2026-06-29 longer mixed-GGUF listening artifact:

- Generated a longer C++ mixed correctness-path sample for manual listening:
  `/tmp/higgs-cpp-mixed-final-step120.wav`.
- Same reference wav/text, mixed GGUF, CUDA backend, `temperature=0`,
  `steps=120`, `--no-stop-on-eoc`, `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=100000`, and `NVIDIA_TF32_OVERRIDE=0`.
- Stats: 24 kHz mono, 107,520 samples, duration `4.48s`, mean_abs
  `0.0678082`, p99_abs `0.333954`, max_abs `0.731445`, RMS `0.103133`,
  nonzero ratio `0.932031`.
- Wall time was `92.12s`, so RTF is about `20.56`. This is still the
  correctness-first full-prefix diagnostic path, not a speed result.
- Decoded the same C++ 120-step finalized codes through Python GGML codec:
  `/tmp/higgs-cpp-mixed-kv120-pydecode.wav`.
  C++ final wav vs Python decode wav have identical shape/sample rate
  (107,520 samples at 24 kHz); sample diff mean_abs `0.000193832`, p99_abs
  `0.00112915`, max_abs `0.00662231`.
  This rules out C++ wav writing or C++ codec decode as the primary source of
  any remaining audible artifact for this sample.
- Decoded the same finalized codes through the official Torch codec using
  `check_codec_decoder_waveform_parity.py`:
  `/tmp/higgs-cpp-mixed-kv120-official-codec.torch.wav`.
  The script reports Torch-vs-GGML waveform `max_abs=0.00063315` before wav
  quantization. On the written wavs, Python GGML decode vs official Torch decode
  has mean_abs `2.00036e-05`, p99_abs `0.00012207`, max_abs `0.000640869`.
  C++ final wav vs official Torch decode has mean_abs `0.000193028`, p99_abs
  `0.00112915`, max_abs `0.00656128`.
  Therefore any remaining obvious misread/pseudo-sound in the 120-step sample is
  not explained by codec decode divergence from official Torch for the same
  codes.

2026-06-29 C++ default model path:

- C++ CLI/API/server defaults now point to
  `/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf` via
  `higgs_audio::k_default_model_path`.
- Rebuilt both CPU and CUDA targets successfully.
- Default-path smoke without `--model`:
  `projects/higgs-audio-ggml-cpp/build-cuda/higgs-audio-cli --inspect`
  reports `model: /root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf`.
  The inspect output also shows the mixed codec tail policy is active, e.g.
  `a.q.quantizers.0.codebook.embed f32` and `a.fc.weight f32`.
- Default-path 20-step generation smoke without `--model`:
  `/tmp/higgs-cpp-default-mixed-kv20.json`.
  With `HIGGS_TRACE_ONLY=1`, `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=100000`, and `NVIDIA_TF32_OVERRIDE=0`,
  the trace matches both Python mixed 20-step
  `/tmp/higgs-py-mixed-models-notf32-step20.json` and official Torch
  `/tmp/higgs-torch-current-step20.json` for `prompt_ids`, `reference_codes`,
  `delayed_codes`, `finalized_codes`, and all per-step trace codes.
- Cleaned current default references to the old F16 GGUF from C++ handoff docs
  and helper defaults:
  `projects/higgs-audio-ggml-cpp/AGENTS.md`,
  `projects/higgs-audio-ggml-cpp/docs/spec.md`,
  `projects/higgs-audio-ggml-cpp/scripts/export_tokenizer_artifact.py`, and
  `projects/higgs-audio-ggml-py/scripts/dev/checks/check_reference_public_audit.py`.
  Remaining `higgs-audio-v3-tts-4b-f16.gguf` references are historical status
  notes or explicit old-model availability notes, not active defaults.
- Stabilized the legacy public audit guard by setting
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=0` for its negative
  `HIGGS_REFERENCE_BACKEND_KV` check. Without that explicit setting, the current
  default window-refresh diagnostic path can bypass the intended guard and fail
  later on a short 4-step finalize shape error. Re-run result:
  `public default trace parity ok`, `diagnostic KV guard ok`, codec waveform
  diff `p99_abs=0.000457782`, `max_abs=0.001922667`.

2026-06-29 C++ full-prefix refresh profile:

- Added a profile-only stage summary for `HIGGS_REFERENCE_KV_STAGE_PROFILE=1`.
  It groups scheduler callback timings into FFN, attention, logits, and misc
  buckets and prints `higgs_profile_stage_summary`.
- Also enlarged the scheduler reserve graph to the same custom capacity used by
  the runtime full-prefix graph. Without this, enabling scheduler stage profile
  hit `GGML_ASSERT(cgraph->n_nodes < cgraph->size)` while reserving the graph.
- Rebuilt CUDA successfully and ran an 8-step trace-only profile with:
  `HIGGS_PROFILE=1`, `HIGGS_REFERENCE_KV_STAGE_PROFILE=1`,
  `HIGGS_REFERENCE_KV_SCHED=1`, `HIGGS_TRACE_ONLY=1`,
  `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=100000`, and
  `NVIDIA_TF32_OVERRIDE=0`.
- Profile artifact stderr:
  `/tmp/higgs-cpp-profile-kv8.stderr`.
- Key profile output:
  - `reference_kv_window_refresh count=8 total_ms=3476 avg_ms=434`
  - `reference_kv_window_compute_graph count=8 total_ms=2918 avg_ms=364`
  - `reference_kv_window_set_inputs count=8 total_ms=539 avg_ms=67`
  - `reference_kv_ffn_gate_matmul count=288 total_ms=402`
  - `reference_kv_ffn_up_matmul count=288 total_ms=420`
  - `reference_kv_ffn_down_matmul count=288 total_ms=385`
  - `reference_kv_stage_all_attention count=288 total_ms=696`
  - `higgs_profile_stage_summary total_ms=1987 ffn_ms=1208
    ffn_pct=60.80 attention_ms=771 attention_pct=38.80 logits_ms=8
    logits_pct=0.40`
- This directly supports the RTF diagnosis: the slow correctness path spends
  most measured stage time in repeated all-layer FFN/MLP matmuls, and the
  full-prefix window refresh itself costs about `434 ms` per generated step in
  this 8-step profile.

2026-06-29 C++ module-level profile:

- Added module-level `HIGGS_PROFILE=1` timers around:
  - `pipeline_reference_encode`
  - `pipeline_text_ar`
  - `pipeline_codec_decode`
  - `pipeline_generate_total`
  - `pipeline_wav_write`
- Ran a full 40-step non-trace-only C++ generation profile:
  `/tmp/higgs-cpp-module-profile-step40.stderr`,
  output wav `/tmp/higgs-cpp-module-profile-step40.wav`.
- Wall-clock: `31.95s`; output duration `1.28s`; RTF `24.96`.
- Module timings:
  - `pipeline_reference_encode ms=6562`
  - `pipeline_text_ar ms=17922`
  - `pipeline_codec_decode ms=4757`
  - `pipeline_generate_total ms=29242`
  - `pipeline_wav_write ms=1`
- Full-prefix refresh internals from the same run:
  - `reference_kv_window_refresh count=40 total_ms=16527 avg_ms=413`
  - `reference_kv_window_compute_graph count=40 total_ms=13690 avg_ms=342`
  - `reference_kv_window_set_inputs count=40 total_ms=2685 avg_ms=67`
- Interpretation: codec decode is not free (`~4.8s`, about `16%` of measured
  `pipeline_generate_total`), but it is not the dominant latency source. The
  largest module is text AR (`~17.9s`, about `61%`), specifically repeated
  full-prefix window refresh in the 36-layer Qwen3 transformer. Reference encode
  is also significant for reference-audio requests (`~6.6s`, about `22%`) and
  would be cacheable for reused voices.

2026-06-29 C++ in-process reference-code cache:

- Added a per-`HiggsPipeline` in-memory reference code cache keyed by model path,
  reference wav path, file size, and mtime. This keeps `encode_reference_codes()`
  unchanged for diagnostic callers while letting the server reuse reference codes
  across requests that use the same local voice file.
- Rebuilt CUDA successfully with:
  `cmake --build projects/higgs-audio-ggml-cpp/build-cuda -j8`.
- Verified with one CUDA server process and two identical `POST /generate`
  requests (`steps=1`, `temperature=0`,
  `HIGGS_PROFILE=1`, `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=100000`,
  `NVIDIA_TF32_OVERRIDE=0`):
  - first request: `pipeline_reference_encode ms=7169`,
    `pipeline_generate_total ms=9102`
  - second request: `pipeline_reference_cache_hit ms=0`, no second
    `pipeline_reference_encode`, `pipeline_generate_total ms=1369`
- This proves the reusable-voice server path removes the reference encode module
  from repeat requests. The dominant uncached path remains text AR full-prefix
  refresh, so the next latency target is still parity-gated incremental KV decode
  with full-prefix oracle/fallback.

2026-06-29 C++ RoPE table reuse in full-prefix refresh:

- Cached BF16 RoPE cos/sin CPU tables on `BackendKvCache` so the full-prefix
  refresh path no longer rebuilds them every generated step.
- Rebuilt CUDA successfully with:
  `cmake --build projects/higgs-audio-ggml-cpp/build-cuda -j8`.
- Ran an 8-step CUDA trace-only profile with the same full-prefix refresh env:
  `HIGGS_PROFILE=1`, `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=100000`, `HIGGS_TRACE_ONLY=1`,
  `HIGGS_TRACE_JSON=/tmp/higgs-rope-cache-step8.json`,
  `NVIDIA_TF32_OVERRIDE=0`.
- New profile:
  - `reference_kv_window_set_inputs count=8 total_ms=93 avg_ms=11`
  - `reference_kv_window_compute_graph count=8 total_ms=2504 avg_ms=313`
  - `reference_kv_window_refresh count=8 total_ms=2638 avg_ms=329`
- Previous comparable 8-step profile:
  - `reference_kv_window_set_inputs count=8 total_ms=539 avg_ms=67`
  - `reference_kv_window_compute_graph count=8 total_ms=2918 avg_ms=364`
  - `reference_kv_window_refresh count=8 total_ms=3476 avg_ms=434`
- The repeated input-prep overhead is mostly gone after the first step. The
  remaining dominant item is still all-layer transformer compute, so further RTF
  improvement should come from parity-gated incremental KV decode, not more
  full-prefix input micro-optimization.

2026-06-29 C++ incremental KV progress:

- Aligned the single-token incremental KV decode RoPE path with the full-prefix
  oracle by using the same BF16 RoPE table helper instead of `ggml_rope_ext()`.
- Increased incremental prefill/decode graph capacities to avoid
  `GGML_ASSERT(cgraph->n_nodes < cgraph->size)` on the real reference prompt.
- Reused the backend gallocr path for single-token incremental decode graph
  allocation. A plain per-step backend graph buffer allocation was not viable for
  longer runs.
- Allowed explicit opt-in non-trace execution with:
  `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=0`,
  `HIGGS_REFERENCE_KV_PREFILL_VIA_WINDOW=1`.
  This still does not make incremental KV the default path.
- 8-step compare run:
  `/tmp/higgs-incremental-compare-step8.json` vs
  `/tmp/higgs-fullprefix-step8-after.json` matched `prompt_ids`,
  `reference_codes`, `delayed_codes`, `finalized_codes`, all per-step codes, and
  all per-step top1 values. Logit `max_abs` diffs were about `1e-2`.
- 40-step trace-only run with window prefill:
  `/tmp/higgs-incremental-windowprefill-step40.json`; full-prefix oracle:
  `/tmp/higgs-fullprefix-step40-after.json`.
  Matched `prompt_ids`, `reference_codes`, `delayed_reference_codes`,
  `delayed_codes`, `finalized_codes`, all per-step codes, and all per-step top1
  values. Top5 differed only by near-tie ordering, e.g. step 6 codebook 0 had
  `{651, 1000, 400, 365, 944}` vs `{651, 400, 1000, 365, 944}` with the second
  and third logits separated by about `0.002`.
- 40-step non-trace CUDA profile with the same opt-in env:
  output `/tmp/higgs-incremental-step40.wav`.
  - wall `21.45s`, audio duration `1.28s`, wall RTF `16.76`
  - `pipeline_reference_encode ms=6392`
  - `pipeline_text_ar ms=4839`
  - `pipeline_codec_decode ms=5977`
  - `pipeline_generate_total ms=17209`
- Compared with the earlier 40-step baseline (`RTF 24.96`,
  `pipeline_text_ar ms=17922`), the text AR module is now about `3.7x` faster
  for this opt-in path. Remaining work before completion: verify 120-step audio,
  decide whether near-tie top5 ordering is acceptable or needs trace formatting
  normalization, and only then consider promoting this path toward default.

2026-06-29 C++ 120-step incremental KV profile:

- Ran a 120-step non-trace CUDA generation with the opt-in incremental KV path:
  `HIGGS_REFERENCE_BACKEND_KV=1`,
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH=0`,
  `HIGGS_REFERENCE_KV_PREFILL_VIA_WINDOW=1`,
  `NVIDIA_TF32_OVERRIDE=0`.
- Output: `/tmp/higgs-incremental-step120.wav`.
- Wall-clock `48.02s`; audio duration `4.48s`; wall RTF `10.72`; pipeline RTF
  `10.06`.
- Module timings:
  - `pipeline_reference_encode ms=6818`
  - `pipeline_text_ar ms=10703`
  - `pipeline_codec_decode ms=27549`
  - `pipeline_generate_total ms=45070`
- Compared with the previous 120-step correctness-path sample
  (`/tmp/higgs-cpp-mixed-final-step120.wav`, wall `92.12s`, duration `4.48s`,
  RTF `20.56`), the opt-in incremental KV path roughly halves wall RTF. After
  AR drops, codec decode becomes the largest remaining module for 120-step
  output.

2026-06-29 C++ incremental KV default for CUDA reference generation:

- Promoted the verified incremental KV path to the default for CUDA
  reference-audio generation. CPU reference generation keeps the old path.
  Full-prefix oracle remains available by explicitly setting
  `HIGGS_REFERENCE_KV_WINDOW_REFRESH` to a large value.
- Default CUDA 40-step trace-only run without reference-KV env:
  `/tmp/higgs-default-incremental-step40.json`.
  It uses `reference_kv_window_refreshes count=0 window=0 frames=40`, confirming
  default CUDA reference generation now runs incremental KV.
- Compared `/tmp/higgs-default-incremental-step40.json` with full-prefix oracle
  `/tmp/higgs-fullprefix-step40-after.json`:
  `prompt_ids`, `reference_codes`, `delayed_reference_codes`, `delayed_codes`,
  `finalized_codes`, per-step codes, and per-step top1 all match. Top5 exact
  ordering still differs only for near ties.
- Default CUDA 40-step non-trace run:
  output `/tmp/higgs-default-incremental-step40.wav`.
  - wall `17.58s`, duration `1.28s`, wall RTF `13.73`, pipeline RTF `11.77`
  - `pipeline_reference_encode ms=6317`
  - `pipeline_text_ar ms=4727`
  - `pipeline_codec_decode ms=4020`
  - `pipeline_generate_total ms=15065`
  - waveform is sample-identical to the full-prefix wav
    `/tmp/higgs-fullprefix-step40-after.wav` (`max_abs_i16=0`).
- Default CUDA 120-step non-trace run:
  output `/tmp/higgs-default-incremental-step120.wav`.
  - wall `46.83s`, duration `4.48s`, wall RTF `10.45`, pipeline RTF `9.86`
  - `pipeline_reference_encode ms=6420`
  - `pipeline_text_ar ms=9839`
  - `pipeline_codec_decode ms=27933`
  - `pipeline_generate_total ms=44194`
  - waveform is sample-identical to the previous opt-in incremental wav
    `/tmp/higgs-incremental-step120.wav` (`max_abs_i16=0`).
- Remaining caveat for strict trace reports: top5 list ordering can differ for
  near-tied logits even when the top5 set, top1, sampled codes, finalized codes,
  and waveform are identical.

2026-06-29 C++ non-trace 120-step fallback correction:

- Added `HIGGS_CODES_JSON` as a lightweight non-trace codes dump. It writes only
  delayed/finalized sampled codes and does not enable trace node outputs, so it
  can diagnose real generation without perturbing GGML graph outputs.
- A same-condition 120-step full-prefix non-trace baseline was generated:
  `/tmp/higgs-fullprefix-step120-after.wav`.
  - wall `83.02s`, duration `4.48s`, wall RTF `18.53`, pipeline RTF `17.95`
  - `pipeline_text_ar ms=47338`
  - `pipeline_codec_decode ms=26587`
  - `pipeline_generate_total ms=80406`
- Comparing non-trace sampled codes showed the narrow fallback margin `0.005`
  was not enough: default incremental first diverged at delayed row 70. Restored
  the conservative fallback margin to `0.10`.
- With margin `0.10`, non-trace default incremental 120-step sampled codes match
  the full-prefix non-trace sampled codes exactly:
  `/tmp/higgs-default-incremental-fallback010-step120-codes.json` vs
  `/tmp/higgs-fullprefix-step120-after-codes.json`.
- Cost of the conservative margin: 87 low-margin fallback refreshes in the
  120-step run. The sampled codes are exact, but the latency benefit is reduced:
  - `pipeline_text_ar ms=42763`
  - `pipeline_codec_decode ms=25593`
  - `pipeline_generate_total ms=77733`
- 40-step with the same conservative margin still stays below the original RTF
  baseline, but with smaller speedup:
  - `pipeline_text_ar ms=14977`
  - `pipeline_codec_decode ms=3545`
  - `pipeline_generate_total ms=24841`
- Interpretation: the quality-safe default now uses incremental KV plus
  low-margin full-prefix fallback, matching the objective's allowed fallback
  model. The next latency target is not more scheduler tuning; it is a narrower
  fallback predicate or the underlying non-trace incremental drift around
  low-margin steps.

2026-06-29 C++ final CUDA reference latency pass:

- Tuned the low-margin full-prefix fallback threshold against true non-trace
  sampled codes rather than trace outputs. Margin `0.005` still diverged at
  120 steps; margins `0.02`, `0.01`, `0.0075`, and `0.006` matched the
  same-condition full-prefix sampled codes. Default margin is now `0.006`.
- Full-prefix forced mode no longer runs the fast fallback a second time. The
  fallback is only active for the incremental path.
- Final default CUDA 40-step non-trace run:
  `/tmp/higgs-final-default-step40.wav`.
  - wall `17.18s`, audio duration `1.28s`, wall RTF `13.42`, pipeline RTF
    `11.44`
  - `reference_kv_full_fallbacks count=1 frames=40`
  - `pipeline_reference_encode ms=6061`
  - `pipeline_text_ar ms=4935`
  - `pipeline_codec_decode ms=3647`
  - `pipeline_generate_total ms=14645`
  - delayed codes, finalized codes, wav params, and wav samples match
    `/tmp/higgs-fullprefix-step40-after.*` exactly.
- Final default CUDA 120-step non-trace run:
  `/tmp/higgs-final-default-step120.wav`.
  - wall `48.55s`, audio duration `4.48s`, wall RTF `10.84`, pipeline RTF
    `10.22`
  - `reference_kv_full_fallbacks count=8 frames=120`
  - `pipeline_reference_encode ms=6533`
  - `pipeline_text_ar ms=13183`
  - `pipeline_codec_decode ms=26085`
  - `pipeline_generate_total ms=45803`
  - delayed codes, finalized codes, and waveform samples match the
    same-condition full-prefix oracle
    `/tmp/higgs-fullprefix-step120-after.*` exactly (`max_abs_i16=0`).
- Same-condition 120-step full-prefix baseline:
  - wall `83.02s`, audio duration `4.48s`, wall RTF `18.53`, pipeline RTF
    `17.95`
  - `pipeline_text_ar ms=47338`
  - `pipeline_codec_decode ms=26587`
  - `pipeline_generate_total ms=80406`
- Compared with the original 40-step module-profile baseline (`wall 31.95s`,
  RTF `24.96`, `pipeline_text_ar ms=17922`), the default CUDA path is now
  quality-equivalent to the full-prefix oracle and substantially faster. The
  remaining dominant latency for longer samples is codec decode, with reference
  encode still a fixed ~6.5s cost unless the in-process reference-code cache
  hits.

2026-06-29 C++ codec CUDA latency pass:

- Added codec decode profile points for load/build/alloc/set/compute/get, plus
  opt-in `HIGGS_CODEC_STAGE_PROFILE=1` stage-prefix profiling.
- 120-step codec profile before the kernel fix:
  - `codec_decode_load_weights ms=566`
  - `codec_decode_alloc_graph ms=11`
  - `codec_decode_set_inputs ms=1629`
  - `codec_decode_compute_graph ms=25409`
  - `codec_decode_total ms=28461`
- Stage-prefix profile showed the cost is in the late decoder upsampling path:
  - `codec_stage_project ms=40`
  - `codec_stage_conv1 ms=5`
  - `codec_stage_block0 ms=254`
  - `codec_stage_block1 ms=1225`
  - `codec_stage_block2 ms=5352`
  - `codec_stage_block3 ms=13175`
  - `codec_stage_block4 ms=25311`
  - `codec_stage_final ms=24300`
- Root cause: vendored GGML CUDA `conv_transpose_1d` scanned every input time
  index for every output sample, even though only about `kernel_width / stride`
  inputs can contribute. The kernel now bounds the input loop to the valid
  contributing range before accumulating.
- 120-step profile after the kernel fix:
  - `pipeline_reference_encode ms=6827`
  - `pipeline_text_ar ms=14283`
  - `codec_decode_load_weights ms=597`
  - `codec_decode_set_inputs ms=1009`
  - `codec_decode_compute_graph ms=2158`
  - `codec_decode_total ms=4728`
  - `pipeline_generate_total ms=25840`
  - wall `29.98s`, audio duration `4.48s`, wall RTF `6.69`
- Clean non-profile 120-step run:
  `/tmp/higgs-convt-fast-clean-step120.wav`.
  - wall `30.49s`, audio duration `4.48s`, wall RTF `6.81`
  - delayed codes and finalized codes match
    `/tmp/higgs-final-default-step120-codes.json` exactly.
  - wav params and samples match `/tmp/higgs-final-default-step120.wav`
    exactly (`max_abs_i16=0`).

2026-06-29 C++ reference cache and AR stage profile:

- Added reference/AR stage profile points:
  - reference acoustic load/resample, acoustic graph, semantic load/resample,
    semantic compute, fc+quantizer compute
  - per-step AR audio head, sample, audio embedding, KV decode
  - KV decode alloc/set/compute/get sub-stages
- Cold 120-step profile, cache cleared:
  - `reference_acoustic_load_wav_resample ms=90`
  - `reference_acoustic_graph ms=5039`
  - `reference_encode_acoustic_total ms=5130`
  - `reference_semantic_load_wav_resample ms=92`
  - `reference_semantic_compute_graph ms=1547`
  - `reference_encode_semantic_total ms=1985`
  - `reference_encode_fc_quantizer_compute_graph ms=22`
  - `pipeline_reference_encode ms=7199`
  - `reference_ar_kv_decode_compute_graph count=121 total_ms=5246`
  - `reference_ar_kv_decode_set_inputs count=121 total_ms=1799`
  - `reference_kv_window_compute_graph count=9 total_ms=3347`
  - `pipeline_text_ar ms=13720`
  - `pipeline_generate_total ms=28997`
- Implemented a disk cache for reference codes, keyed by model path, reference
  wav path, file size, and mtime. Default directory:
  `/tmp/higgs-audio-reference-codes`; override with
  `HIGGS_REFERENCE_CODES_CACHE_DIR`.
- Warm 120-step profile with the same reference wav:
  - `pipeline_reference_disk_cache_hit ms=0`
  - `pipeline_text_ar ms=12872`
  - `pipeline_codec_decode ms=3738`
  - `pipeline_generate_total ms=16611`
  - wall `19.93s`, audio duration `4.48s`, wall RTF `4.45`
- Clean warm 120-step run:
  `/tmp/higgs-stage-warm-clean-step120.wav`.
  - wall `18.91s`, audio duration `4.48s`, wall RTF `4.22`
  - delayed codes and finalized codes match
    `/tmp/higgs-convt-fast-clean-step120-codes.json` exactly.
  - wav params and samples match `/tmp/higgs-convt-fast-clean-step120.wav`
    exactly.
- Remaining non-saturated work is now mostly per-step AR KV decode. The profile
  shows about `5.2s` CUDA compute and `1.8s` host-to-device input setting across
  121 decode calls, plus 8 low-margin full-prefix fallback refreshes totaling
  about `3.3s`.

2026-06-29 C++ two-reference RTF generalization check:

- Goal matrix used CUDA, `temperature=0`, `steps=120`, `--no-stop-on-eoc`, the
  same target text, and two reference wavs:
  - long reference wav:
    `/root/code/ggbond/models/可哪怕位于堂堂超一品官职,在十二郡一言九鼎的大柱国口干舌燥了,这少年还是没什么反应.wav`
  - tai_yi reference wav:
    `/root/code/ggbond/models/tai_yi_xian_ren.wav`, ref text
    `对，这就是我，万人敬仰的太乙仙人。`
- `HIGGS_REFERENCE_KV_SKIP_NON_CB6_FALLBACK=1` versus full oracle
  `=0` produced identical delayed codes, finalized codes, wav params, and wav
  bytes for both warm reference runs in this matrix.
- Warm/cold profile matrix:
  - long ref, skip on cold: wall `18.37s`, duration `4.48s`, RTF `4.10`,
    `pipeline_reference_encode ms=7318`, `pipeline_text_ar ms=8093`,
    `reference_ar_kv_decode_compute_graph ms=5115`, `codec_decode_total ms=206`,
    fallbacks `6`, skipped oracles `5`.
  - long ref, skip on warm: wall `11.16s`, duration `4.48s`, RTF `2.49`,
    `pipeline_reference_encode ms=0`, `pipeline_text_ar ms=7731`,
    `reference_ar_kv_decode_compute_graph ms=5077`, `codec_decode_total ms=215`,
    fallbacks `6`, skipped oracles `5`.
  - long ref, full oracle warm: wall `13.63s`, duration `4.48s`, RTF `3.04`,
    `pipeline_text_ar ms=10287`, `reference_ar_kv_decode_compute_graph ms=5083`,
    `codec_decode_total ms=207`, fallbacks `8`, skipped oracles `0`.
  - tai_yi ref, skip on cold: wall `16.40s`, duration `4.48s`, RTF `3.66`,
    `pipeline_reference_encode ms=5328`, `pipeline_text_ar ms=8127`,
    `reference_ar_kv_decode_compute_graph ms=4955`, `codec_decode_total ms=210`,
    fallbacks `3`, skipped oracles `3`.
  - tai_yi ref, skip on warm: wall `10.82s`, duration `4.48s`, RTF `2.42`,
    `pipeline_reference_encode ms=0`, `pipeline_text_ar ms=7082`,
    `reference_ar_kv_decode_compute_graph ms=4955`, `codec_decode_total ms=204`,
    fallbacks `3`, skipped oracles `3`.
  - tai_yi ref, full oracle warm: wall `11.05s`, duration `4.48s`, RTF `2.47`,
    `pipeline_text_ar ms=8058`, `reference_ar_kv_decode_compute_graph ms=4957`,
    `codec_decode_total ms=204`, fallbacks `3`, skipped oracles `0`.
- Tried wiring the existing optional `HIGGS_REFERENCE_KV_FUSED_FFN=1` packed
  gate/up helper into the single-token decode path. It was rejected and reverted:
  the fused run did not reduce `reference_ar_kv_decode_compute_graph` and changed
  delayed/finalized codes on the long reference validation. Keep this experiment
  out of the default single-token path.
- Current general bottleneck after warm reference cache and codec runtime reuse is
  still single-token AR decode compute: about `5.0-5.1s` across 120 steps on both
  reference wavs. Codec decode is about `0.2s`; reference encode is removed on
  warm cache hits. Next useful optimization must target the decode graph/kernel
  structure itself, not codec or cache plumbing.

2026-06-29 C++ two-reference clean warm acceptance:

- Re-ran clean warm 120-step CUDA generation with the same two-reference matrix
  and `HIGGS_REFERENCE_KV_SKIP_NON_CB6_FALLBACK=1`, no `HIGGS_PROFILE`:
  - long reference: `/tmp/higgs-goal-clean-ref1-warm.wav`, wall `10.22s`,
    duration `4.48s`, RTF `2.281`; delayed codes, finalized codes, wav params,
    and wav bytes match `/tmp/higgs-goal-ref1-skip1-warm.*` exactly.
  - tai_yi reference: `/tmp/higgs-goal-clean-ref2-warm.wav`, wall `9.75s`,
    duration `4.48s`, RTF `2.176`; delayed codes, finalized codes, wav params,
    and wav bytes match `/tmp/higgs-goal-ref2-skip1-warm.*` exactly.
- Final module profile on the long reference warm path:
  - `/tmp/higgs-goal-final-profile-ref1.wav`, wall `10.30s`, duration `4.48s`,
    RTF `2.299`.
  - `pipeline_text_ar ms=7595`
  - `reference_prefill_total ms=238`
  - `reference_kv_window_compute_graph ms=737`
  - `reference_ar_kv_decode_compute_graph ms=5066`
  - `reference_ar_kv_decode_alloc_graph ms=108`
  - `reference_ar_kv_decode_set_inputs ms=1`
  - `reference_ar_audio_head ms=4`
  - `reference_ar_audio_embedding ms=7`
  - `codec_decode_total ms=209`
  - `pipeline_generate_total ms=7805`
- Acceptance note: the clean long-reference warm RTF is now below the goal line
  `2.33`, and tai_yi remains well below its cold baseline `3.44`. No new speed
  code was merged in this checkpoint; the attempted single-token fused FFN path
  was rejected because it changed codes. The remaining dominant bottleneck is
  still single-token AR decode compute, about `5.1s` across 120 steps.

2026-06-29 C++ single-token AR decode sub-profile checkpoint:

- Added a working opt-in single-token AR decode stage profile under
  `HIGGS_PROFILE=1 HIGGS_REFERENCE_KV_STAGE_PROFILE=1`.
  - The diagnostic path uses a backend scheduler only while stage profiling is
    enabled so GGML eval callbacks fire per named node.
  - The default gallocr decode path is unchanged when stage profiling is off.
  - Profile summaries now include microsecond totals/averages so sub-millisecond
    CUDA node groups are visible.
- Standard long-reference 120-step stage profile
  `/tmp/higgs-arstage-ref1-120.*` kept codes and wav bytes identical to
  `/tmp/higgs-goal-clean-ref1-warm.*` and measured these largest decode subitems:
  - `reference_kv_decode_ffn_gate_matmul total_us=1021409 avg_us=234`
  - `reference_kv_decode_qk_matmul_scale total_us=1003682 avg_us=230`
  - `reference_kv_decode_ffn_up_matmul total_us=983690 avg_us=225`
  - `reference_kv_decode_ffn_down_matmul total_us=945115 avg_us=216`
  - `reference_kv_decode_attn_project_rope total_us=597733 avg_us=137`
  - `reference_kv_decode_attention_output_residual total_us=475439 avg_us=109`
  - `reference_kv_decode_kv_write_view_repeat total_us=386116 avg_us=88`
  - `reference_ar_kv_decode_compute_graph total_ms=6677` in the diagnostic
    scheduler path.
- Tried GGML CUDA `ggml_flash_attn_ext` as an opt-in replacement for the manual
  QK/softmax/V attention sequence. It was rejected and removed from the code:
  - clean RTF improved on the long reference (`9.00s / 4.48s = 2.009`), but
    delayed/finalized codes and wav bytes diverged from the safety baseline.
  - Do not enable flash attention for the quality path until its numerics are
    proven equivalent or the acceptance rule explicitly allows statistical
    equivalence.
- Final default clean warm validation after keeping only the safe profiling
  instrumentation:
  - long reference: `/tmp/higgs-arprofile-default-ref1.wav`, wall `9.55s`,
    duration `4.48s`, RTF `2.132`; delayed codes, finalized codes, and wav bytes
    match `/tmp/higgs-goal-clean-ref1-warm.*` exactly.
  - tai_yi reference: `/tmp/higgs-arprofile-default-ref2.wav`, wall `8.91s`,
    duration `4.48s`, RTF `1.989`; delayed codes, finalized codes, and wav bytes
    match `/tmp/higgs-goal-clean-ref2-warm.*` exactly.
- Remaining concrete direction: the largest safe target is still the per-layer
  decode matmul path. Fast attention/FFN fusion can reduce time but currently
  changes codes. Next work should either make the fused attention numerics match
  the manual path, or move lower in GGML/CUDA to reduce single-token F32 GEMV
  launch/overhead without changing operation order/results.

2026-06-29 CUDA mul_mat dispatch checkpoint:

- Added opt-in CUDA backend mul_mat dispatch profiling with
  `HIGGS_CUDA_MUL_MAT_PROFILE=1` in `vendor/ggml/src/ggml-cuda/ggml-cuda.cu`.
  It summarizes `GGML_OP_MUL_MAT` by selected path plus source/destination
  types, precision, and GGML-order shapes. The profile is count-only and does
  not synchronize kernels, so it is usable on 120-step runs. Default runs are
  unchanged when the env var is unset.
- A first timing version synchronized after every mul_mat and was rejected as too
  intrusive for 120-step diagnostics. The kept version only records dispatch
  path/count. Timing attribution should use the AR decode microsecond stage
  profile from the previous checkpoint.
- 120-step warm dispatch diagnostics:
  - long reference `/tmp/higgs-cudamm-ref1-120.*`: delayed codes, finalized
    codes, and wav bytes match `/tmp/higgs-goal-clean-ref1-warm.*` exactly.
    Path counts: `mmvf_direct=28416`, `cublas_batched=2340`,
    `cublas_split=546`.
  - tai_yi reference `/tmp/higgs-cudamm-ref2-120.*`: delayed codes, finalized
    codes, and wav bytes match `/tmp/higgs-goal-clean-ref2-warm.*` exactly.
    Path counts: `mmvf_direct=28452`, `cublas_batched=2232`,
    `cublas_split=293`.
- The hot single-token AR decode matmuls are already dispatched to
  `mmvf_direct`, not to an obvious cuBLAS fallback:
  - FFN gate/up/down shapes include `f16 x f32 -> f32` single-column products:
    `2560x9728 * 2560x1`, `2560x1024 * 2560x1`, and
    `2560x4096 * 2560x1`.
  - Attention QK single-token shapes are also `mmvf_direct` with F32 inputs,
    e.g. `src0=<past>x128x32`, `src1=<past>x1x32`, `dst=128x1x32`.
- Final default clean warm validation after the safe dispatch profiler:
  - long reference `/tmp/higgs-cudamm-final-ref1.wav`: wall `9.32s`, duration
    `4.48s`, RTF `2.080`; delayed codes, finalized codes, and wav bytes match
    `/tmp/higgs-goal-clean-ref1-warm.*` exactly.
  - tai_yi `/tmp/higgs-cudamm-final-ref2.wav`: wall `8.83s`, duration `4.48s`,
    RTF `1.971`; delayed codes, finalized codes, and wav bytes match
    `/tmp/higgs-goal-clean-ref2-warm.*` exactly.
- No bottom-path optimization was merged in this checkpoint because the profile
  did not show a wrong dispatch choice. The next safe optimization target is the
  `mmvf_direct` kernel/launch path itself, especially the repeated single-column
  F16xF32 FFN matvec and F32 attention-QK matvec shapes. Any change there must
  preserve current output bytes or be kept as a diagnostic experiment.
