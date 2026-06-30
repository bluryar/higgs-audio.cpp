# HiggsAudioV3 C++ Runtime Spec

This file is the C++ handoff contract from
`projects/higgs-audio-ggml-py`. It records what the C++ runtime must preserve
before claiming parity with the Python staging graph.

## Scope

Implement an offline CPU-first HiggsAudioV3 runtime with:

- shared `HiggsPipeline` core;
- CLI for zero-shot and reference-audio generation;
- thin HTTP server wrapper over the same pipeline.

Deferred:

- streaming;
- batching;
- CUDA codec execution;
- preset voice registry;
- MP3 output.

## Model Artifacts

Default local paths:

- HF model directory:
  `/root/code/ggbond/models/higgs-audio-v3-tts-4b`
- GGUF:
  `/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf`

Required GGUF metadata:

- `higgs.config.json`
- `higgs.audio.num_codebooks = 8`
- `higgs.audio.vocab_size = 1026`
- `qwen3.vocab_size`
- `qwen3.context_length`
- `qwen3.embedding_length`
- `qwen3.feed_forward_length`
- `qwen3.block_count`
- `qwen3.attention.head_count`
- `qwen3.attention.head_count_kv`
- `qwen3.attention.key_length`
- `qwen3.attention.value_length`
- `qwen3.attention.layer_norm_rms_epsilon`
- `qwen3.rope.dimension_count`
- `qwen3.rope.freq_base`

The runtime must build graph dimensions from metadata/config, not hardcoded
checkpoint assumptions, except for true family constants listed below.

## Family Constants

- codebooks: `8`
- codebook vocab size: `1026`
- payload code ids: `[0, 1023]`
- BOC id: `1024`
- EOC/audio-mask id: `1025`
- audio placeholder text id: `-100`
- codec sample rate: `24000`
- semantic/HuBERT sample rate: `16000`
- codec hop length: `960`
- acoustic pad: `480`
- semantic pad: `160`

## Prompt Contract

Supported public prompt formats:

- `higgstts`: `<|tts|><|text|>{text}<|audio|>`
- `chatml`: local tokenizer chat template with caller-supplied system prompt
- `boson-chatml`: local tokenizer chat template with Boson default system prompt

Reference-audio prompt:

```text
<|tts|>
[<|ref_text|>{ref_text}]
<|ref_audio|>
{-100 repeated delayed_reference_rows}
<|text|>{text}<|audio|>
```

The C++ tokenizer implementation must match local HF tokenizer output for the
same model directory before text graph parity can be claimed.

`--inspect-tokenizer` inventories the GGUF tokenizer metadata:

- `tokenizer.ggml.model=gpt2`
- `qwen3.vocab_size=151936`
- `tokenizer.ggml.tokens=151727`
- `tokenizer.ggml.merges=151387`
- `tokenizer.ggml.added_tokens=84`
- `tokenizer.huggingface.json=11433924` bytes
- `tokenizer.chat_template=2427` bytes

The `tokenizer.ggml.tokens` plus added-token list is not a dense 151936-entry
runtime vocabulary; the full tokenizer source is `tokenizer.huggingface.json`.
Runtime text tokenization should use `tokenizers.cpp` over that JSON.
`--self-check-tokenizers-cpp` loads the JSON directly from GGUF and verifies HF
ids for `hello`, `test`, `我`, `你好`, ` hello`, plus `<|tts|>`.

`scripts/export_tokenizer_artifact.py` writes the current compact artifact
diagnostic artifact format from GGUF tokenizer metadata:

```text
magic: "HATK1\0\0\0"
u32 token_count
u32 merge_count
u32 added_token_count
u32 chat_template_bytes
repeated token strings, dense to qwen3.vocab_size
repeated i32 token types
repeated merge strings
chat template string
```

`--inspect-tokenizer-artifact PATH` validates the header and inventory.
`--self-check-tokenizer-artifact PATH` loads the dense token table and verifies
exact-token lookup for base and Higgs special tokens.
`--self-check-tokenizer-bytelevel PATH` verifies GPT-2 ByteLevel byte mapping
over the artifact for ASCII and UTF-8 bytes. `--self-check-tokenizer-bpe PATH`
verifies single pre-token BPE merging against HF tokenizer outputs for `hello`,
`test`, and `我`. Regex pre-token splitting and chat-template rendering are not
part of the compact artifact path; runtime tokenization uses
`tokenizers.cpp` over `tokenizer.huggingface.json`.

### Prompt Helper Status

Current C++ prompt helpers operate on already-tokenized text ids:

- load `<|tts|>`, `<|text|>`, `<|audio|>`, `<|ref_text|>`, and
  `<|ref_audio|>` ids from `tokenizer.ggml.tokens`;
- tokenize text with `tokenizers.cpp` from GGUF `tokenizer.huggingface.json`;
- build `higgstts` and fixed ChatML/Boson prompt-id layouts;
- build reference prompt-id layout with `-100` audio placeholders.

Validated by:

```bash
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
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-audio-head-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-ar-graph
```

ChatML/Boson zero-shot prompt rendering is implemented for the current
system/user/assistant generation template used by Python
`tokenizer.apply_chat_template(..., add_generation_prompt=True)`.

## Embedding Id Contract

Before `ggml_get_rows`:

- text ids use `(ne0=tokens)` I32;
- `-100` reference placeholders are replaced with `0` for safe text embedding;
- delayed reference audio ids are separately overlaid at the placeholder
  positions;
- each audio codebook id is fused as `id + codebook * 1026` before lookup in
  `a.token_embd`;
- fused audio ids are stored in GGML-order `CodeMatrix(ne0=codebooks,
  ne1=frames)`.

Validated by:

```bash
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-embedding-ids
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-embedding-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-embedding-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-backbone-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-audio-head-graph
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-prompt-ar-graph
```

`--self-check-embedding-graph` is a CPU GGML diagnostic gate. It loads only
`token_embd.weight` and `a.token_embd`, runs `ggml_get_rows`, sums the 8 audio
codebook embeddings, concatenates text/audio embeddings on `dim=1`, and checks
the output shape `(ne0=2560, ne1=2)` plus finite values.
`--self-check-prompt-embedding-graph` reuses the same helper with real
`higgstts` prompt ids and checks `(ne0=2560, ne1=prompt_tokens)`.

`--self-check-audio-head-graph` is a CPU GGML diagnostic gate over synthetic
hidden states. It loads `output_norm.weight` and `a.output`, runs
`ggml_rms_norm -> broadcast ggml_mul -> ggml_mul_mat`, converts the flat
`(ne0=8208, ne1=1)` result with `audio_logits_flat_to_vc`, and exercises the
sampler. It does not include Qwen3 transformer blocks.

`--self-check-block0-mlp-graph` is a CPU GGML diagnostic gate over synthetic
hidden states. It loads `blk.0.ffn_norm.weight`, `blk.0.ffn_gate.weight`,
`blk.0.ffn_up.weight`, and `blk.0.ffn_down.weight`, then runs the Qwen3 MLP
path:

```text
rms_norm(hidden) * ffn_norm
  -> gate = ffn_gate(normed)
  -> up = ffn_up(normed)
  -> ffn_down(silu(gate) * up)
```

It checks output shape `(ne0=2560, ne1=2)` and finite values. Torch parity and
attention/residual/full-block execution remain separate gates.

`--self-check-block0-attn-qkv-graph` is a CPU GGML diagnostic gate over
synthetic hidden states. It loads block0 attention norm, Q/K/V projection, and
Q/K norm tensors, then runs:

```text
rms_norm(hidden) * attn_norm
  -> q/k/v = attn_q/attn_k/attn_v(normed)
  -> reshape q=(ne0=128, ne1=32, ne2=2)
  -> reshape k/v=(ne0=128, ne1=8, ne2=2)
  -> q/k RMSNorm
  -> q/k ggml_rope_ext(mode=NEOX)
  -> repeat k/v to 32 heads
```

It checks Q/K/V output shape `(ne0=128, ne1=32, ne2=2)` and finite values.

`--self-check-block0-attn-graph` extends that gate through the explicit Python
staging attention path:

```text
q4/k4 permute
  -> kq = ggml_mul_mat(k4, q4)
  -> ggml_scale
  -> ggml_diag_mask_inf
  -> ggml_soft_max
  -> kqv = ggml_mul_mat(v_for_mm, kq)
  -> permute + ggml_cont_2d
  -> attn_output
```

It checks output shape `(ne0=2560, ne1=2)` and finite values.

`--self-check-block0-graph` extends the attention and MLP gates through the
Python staging block composition:

```text
attn = qwen3_attention(hidden)
cur = ggml_add(hidden, attn)
mlp = qwen3_mlp(cur)
out = ggml_add(cur, mlp)
```

It loads the block attention and MLP tensors, checks output shape
`(ne0=2560, ne1=2)`, and checks finite values. The same layer-indexed helper
also backs `--self-check-block35-graph`, which validates the final Qwen3 block
uses the same tensor naming and shape contract.

`--self-check-backbone-graph` runs all 36 Qwen3 blocks over synthetic hidden
states with `(ne0=2560, ne1=2)`. `--self-check-prompt-backbone-graph` runs real
`higgstts` prompt embeddings through the same 36-layer path with
`(ne0=2560, ne1=prompt_tokens)`. Both execute one block at a time, loading one
layer's tensors per step and copying each block output into the next block input
to keep the CPU diagnostic memory footprint bounded.
`--self-check-prompt-audio-head-graph` takes the final real prompt token hidden
state through `output_norm -> a.output -> sampler` and checks that one
8-codebook frame is produced. `--self-check-prompt-ar-graph` runs a small
no-KV autoregressive recompute loop: each new frame appends previous sampled raw
audio code embeddings after the prompt, reruns the full backbone, samples the
next frame, then finalizes delayed codec ids. KV cache remains a separate
optimization gate.

`--self-check-text-logits-graph` composes the synthetic text path:

```text
token/audio embeddings
  -> 36-layer Qwen3 backbone
  -> output_norm
  -> a.output
  -> audio_logits_flat_to_vc
  -> sampler
```

It uses the second token hidden state from the synthetic `(ne0=2560, ne1=2)`
embedding graph as the audio decode position and checks that one sampled
8-codebook frame is produced. Real prompt-conditioned no-KV AR sampling is
covered by `--self-check-prompt-ar-graph`; KV cache and reference codec encode
remain separate gates.

## Offline Path

The complete offline path is:

1. Input text plus optional `ref_text` and optional `ref_wav`.
2. Build prompt ids; reference prompts contain `-100` placeholders.
3. Encode reference audio into raw 8-codebook codes when `ref_wav` is present.
4. Apply delay pattern from raw `[T, N]` source view to delayed `[T + N - 1, N]`.
5. Overlay delayed reference code embeddings at `-100` text positions.
6. Run Qwen3 backbone prefill/decode.
7. Project fused 8-codebook audio logits.
8. Run SGLang-style sampler with BOC/EOC handling.
9. Reverse delay and clip to codec payload ids.
10. Decode codec codes to mono 24 kHz waveform.

## Shape Contract

Use GGML order first:

- text ids: `(ne0=tokens)`
- text hidden: `(ne0=hidden, ne1=tokens)`
- generated/delayed codes: `(ne0=codebooks, ne1=frames)`
- raw reference code source view: `[frames, codebooks]`
- delayed reference code source view: `[frames + codebooks - 1, codebooks]`
- codec acoustic latent: `(ne0=frames, ne1=256, ne2=1)` unless graph output is
  explicitly documented otherwise
- semantic latent: `(ne0=frames, ne1=768, ne2=1)`
- audio logits: `(ne0=1026, ne1=8, ne2=frames)`

When loading source-order buffers, document the crossing into GGML `ne`
explicitly. Do not silently reverse dimensions.

## Text Graph Inventory

Current target GGUF reports:

- vocab size: `151936`
- context length: `32768`
- embedding length: `2560`
- feed-forward length: `9728`
- blocks: `36`
- attention heads: `32`
- KV heads: `8`
- key/value length: `128`
- rope dimension count: `128`
- RMS norm epsilon: `1e-6`
- rope freq base: `1e6`

Validated by:

```bash
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --inspect-text
```

Do not infer hidden width from `head_count * key_length`; this model uses
separate config values and GQA.

Current text frontier tensor shapes in GGML `ne` order:

- `token_embd.weight`: `(ne0=2560, ne1=151936)`
- `output_norm.weight`: `(ne0=2560)`
- `a.token_embd`: `(ne0=2560, ne1=8208)`
- `a.output`: `(ne0=2560, ne1=8208)`
- `blk.0.attn_q.weight`: `(ne0=2560, ne1=4096)`
- `blk.0.attn_k.weight`: `(ne0=2560, ne1=1024)`
- `blk.0.attn_v.weight`: `(ne0=2560, ne1=1024)`
- `blk.0.attn_output.weight`: `(ne0=4096, ne1=2560)`
- `blk.0.attn_q_norm.weight`: `(ne0=128)`
- `blk.0.attn_k_norm.weight`: `(ne0=128)`
- `blk.0.ffn_gate.weight`: `(ne0=2560, ne1=9728)`
- `blk.0.ffn_up.weight`: `(ne0=2560, ne1=9728)`
- `blk.0.ffn_down.weight`: `(ne0=9728, ne1=2560)`
- `blk.0.attn_norm.weight`: `(ne0=2560)`
- `blk.0.ffn_norm.weight`: `(ne0=2560)`

`a.output` projects hidden states to `8208 = 8 * 1026`; the runtime must reshape
that axis into `(codebooks=8, vocab=1026)` explicitly when feeding the sampler.

The C++ helper `audio_logits_flat_to_vc` accepts flat audio-head logits in GGML
order `(ne0=8208, ne1=frames)`, selects one frame, and returns source-view
`[vocab=1026, codebooks=8]` for `sample_codes_from_logits_vc`. Validated by:

```bash
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-audio-logits
```

## Sampler Contract

Parameters:

- `temperature`
- `top_k`
- `top_p`
- `seed`
- `steps`
- `stop_on_eoc`

Behavior:

- start generated audio history with a BOC column;
- return delayed codes including the initial BOC column;
- mask ids `[1024, 1024]` according to BOC/EOC rules:
  - payload ids are `[0, 1023]`;
  - EOC is disabled when `stop_on_eoc=false`;
- during the initial delay window, force future codebooks to BOC;
- during EOC winddown, force completed leading codebooks to EOC;
- `temperature <= 0` uses argmax;
- `temperature > 0` uses seeded top-k/top-p sampling.

### Sampler Helper Status

Current C++ sampler helpers operate on one-step source-view logits
`[vocab, codebooks]` and return one code per codebook. They implement:

- payload/BOC/EOC masking;
- argmax for `temperature <= 0`;
- seeded top-k/top-p sampling for `temperature > 0`;
- SGLang BOC delay-window and EOC winddown step rules.

Validated by:

```bash
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-sampler
```

The text graph still needs to provide real logits to these helpers.

## Delay Helper Status

C++ helper layout:

- `CodeMatrix` stores GGML-order generated codes as
  `(ne0=codebooks, ne1=frames)`.
- `apply_delay_pattern_tn` accepts raw source-view codes `[frames, codebooks]`.
- `reverse_delay_pattern_tn` returns raw source-view codes `[frames, codebooks]`.
- `finalize_generated_codes` returns GGML-order codec payload codes
  `(ne0=codebooks, ne1=frames)`.

Validated by:

```bash
projects/higgs-audio-ggml-cpp/build/higgs-audio-cli --self-check-delay
```

## Codec Contract

Decode path:

```text
codes -> quantizer.decode -> a.fc2 -> acoustic_decoder -> waveform
```

Current C++ status:

- `--self-check-codec-project-graph` runs the front of decode over synthetic
  two-frame codec ids:

```text
8 codebook id tensors
  -> ggml_get_rows(a.q.quantizers.N.codebook.embed)
  -> ggml_mul_mat(project_out.weight)
  -> add project_out.bias
  -> sum 8 quantizer outputs
  -> ggml_mul_mat(a.fc2.weight)
  -> add a.fc2.bias
```

It checks the projected decoder input shape `(ne0=256, ne1=2)` and finite
values. Acoustic decoder blocks and waveform output remain separate gates.

- `--self-check-codec-conv1-graph` extends that path through the first acoustic
  decoder convolution:

```text
a.fc2 output
  -> transpose/contiguous view as acoustic input (ne0=frames, ne1=256, ne2=1)
  -> ggml_conv_1d(a.ad.conv1.weight, stride=1, padding=3, dilation=1)
  -> add a.ad.conv1.bias
```

It checks output shape `(ne0=2, ne1=1024, ne2=1)` and finite values. Decoder
upsampling blocks, final conv, and waveform output remain separate gates.

- `--self-check-codec-block0-conv-t1-graph` extends that path through the first
  decoder upsampling op:

```text
a.ad.conv1 output
  -> snake activation using a.ad.block.0.snake1.alpha
  -> ggml_conv_transpose_1d(a.ad.block.0.conv_t1.weight, stride=8)
  -> crop padding window
  -> add a.ad.block.0.conv_t1.bias
```

It checks output shape `(ne0=16, ne1=512, ne2=1)` and finite values.

- `--self-check-codec-block0-res-unit1-graph` extends that path through the
  first residual unit:

```text
a.ad.block.0.conv_t1 output
  -> snake1
  -> ggml_conv_1d(res_unit1.conv1, stride=1, padding=3, dilation=1)
  -> snake2
  -> ggml_conv_1d(res_unit1.conv2, stride=1, padding=0, dilation=1)
  -> residual add
```

It checks output shape `(ne0=16, ne1=512, ne2=1)` and finite values. The full
block0 path is covered by the next diagnostic gate.

- `--self-check-codec-block0-graph` extends that path through all block0
  residual units:

```text
block0 conv_t1 output
  -> res_unit1(dilation=1)
  -> res_unit2(dilation=3)
  -> res_unit3(dilation=9)
```

It checks output shape `(ne0=16, ne1=512, ne2=1)` and finite values. Later
upsampling blocks, final conv, and waveform output remain separate gates.

- `--self-check-codec-block1-graph` extends the synthetic codec decode path
  through decoder blocks 0 and 1:

```text
a.ad.conv1 output
  -> decoder block0(stride=8)
  -> decoder block1(stride=5)
```

It checks output shape `(ne0=80, ne1=256, ne2=1)` and finite values. Decoder
blocks 2-4, final conv, and waveform output remain separate gates.

- `--self-check-codec-waveform-graph` extends the synthetic codec decode path
  through all decoder blocks and the final acoustic decoder convolution:

```text
a.ad.conv1 output
  -> decoder blocks 0..4(strides=8,5,4,2,3)
  -> a.ad.snake1
  -> ggml_conv_1d(a.ad.conv2.weight, stride=1, padding=3, dilation=1)
  -> add a.ad.conv2.bias
```

It checks output shape `(ne0=1920, ne1=1, ne2=1)` and finite values for a
synthetic 2-frame codec input.

- `--self-check-codec-wav --out out.wav` writes that synthetic decoder output as
  mono 24 kHz 16-bit PCM WAV. It is a file-format and decoder-output diagnostic,
  not the public generation path.

The public CLI currently tokenizes `--text` into real prompt ids, then runs a
no-KV autoregressive recompute loop: prompt/audio embeddings -> 36-layer text
graph -> audio-head sampler for each step, followed by codec decode for
no-reference requests. `--text TEXT --out out.wav` checks the shared
prompt-to-WAV boundary, and `--steps` controls the number of decoded codec
frames. `temperature`, `top_k`, `top_p`, and `seed` feed the audio-head sampler.
KV-cache wiring is still deferred, so each step recomputes the full prompt plus
generated audio context.

Encode path:

```text
24 kHz waveform + 16 kHz semantic waveform
  -> acoustic_encoder
  -> HuBERT semantic branch
  -> concat + a.fc
  -> 8-layer RVQ encode
  -> raw codes [frames, codebooks]
```

Reference audio preprocessing belongs at the CLI/server boundary:

- read mono or downmix to mono;
- resample reference waveform to 24 kHz;
- derive semantic waveform at 16 kHz;
- pass both arrays to `HiggsPipeline`.

`--self-check-reference-wav PATH` validates the current WAV input boundary. It
supports RIFF/WAVE PCM 8/16/24/32-bit and float32, downmixes to mono, resamples
one buffer to 24 kHz for acoustic encode, and resamples one buffer to 16 kHz for
the HuBERT semantic branch.

`--self-check-reference-acoustic PATH` validates the real WAV acoustic path. It
loads the reference WAV, uses the 24 kHz buffer, pads short references to at
least one second, applies `CODEC_ACOUSTIC_PAD=480` samples on both sides, then
runs the same acoustic encoder tail as
`--self-check-codec-encode-project-graph`. It checks finite output with channel
shape `(ne1=256, ne2=1)` and prints the acoustic frame count.

`--self-check-reference-semantic PATH` validates the real WAV semantic path. It
loads the reference WAV, uses the 16 kHz buffer, applies
`CODEC_SEMANTIC_PAD=160` samples on both sides, runs HuBERT feature extraction,
feature projection, positional convolution, encoder layer mean/downsample, then
the semantic encoder tail. It checks finite output with channel shape
`(ne1=768, ne2=1)` and prints the semantic frame count.

`--self-check-reference-codes PATH` validates the local reference codec encode
tail. It runs the real WAV acoustic and semantic paths, aligns them with
`min(acoustic_frames, semantic_frames)`, then runs concat `a.fc` and the
8-stage RVQ encode. It checks each code ID is in `[0, 1024)` and prints the
reference code frame count.

`--inspect-reference-encode` validates the current reference encode inventory:

- sample rates: audio `24000`, semantic `16000`;
- downsample factors: codec `320`, semantic branch `2`;
- semantic model config: hidden `768`, layers `12`, heads `12`, FFN `3072`;
- tensor counts: `a.ae.*=110`, `a.sm.*=211`, `a.se.*=13`, `a.fc.*=2`,
  `a.fc1.*=2`, `a.fc2.*=2`, `a.q.*=64`;
- frontier tensors include acoustic encoder, HuBERT feature/encoder, semantic
  encoder, concat `a.fc`, acoustic-only `a.fc1`, decode `a.fc2`, and RVQ
  quantizer weights in GGML `ne` order.

`--self-check-codec-encode-conv1-graph` starts the acoustic encoder graph:

```text
waveform (ne0=samples, ne1=1, ne2=1)
  -> ggml_conv_1d(a.ae.conv1.weight, stride=1, padding=3, dilation=1)
  -> add a.ae.conv1.bias
```

It checks `(ne0=samples, ne1=64, ne2=1)` and finite values for a synthetic
24 kHz waveform.

`--self-check-codec-encode-block0-graph` extends the same acoustic encoder
diagnostic through encoder block0:

```text
conv1 output (ne0=samples, ne1=64, ne2=1)
  -> res_unit1 dilation=1
  -> res_unit2 dilation=3
  -> res_unit3 dilation=9
  -> Snake
  -> ggml_conv_1d(a.ae.block.0.conv1.weight, stride=8, padding=4, dilation=1)
  -> add a.ae.block.0.conv1.bias
```

It checks finite values and channel shape `(ne1=128, ne2=1)` for synthetic
input. This remains a diagnostic gate, not full reference-audio encode.

`--self-check-codec-encode-block1-graph` extends the diagnostic through encoder
block1 using the same residual-unit order and the acoustic encoder stride
contract `(8, 5, 4, 2, 3)`. It checks finite values and channel shape
`(ne1=256, ne2=1)` for synthetic input.

`--self-check-codec-encode-block2-graph`,
`--self-check-codec-encode-block3-graph`, and
`--self-check-codec-encode-block4-graph` continue the same acoustic encoder
contract through strides `4`, `2`, and `3`. The checked channel shapes are
`512`, `1024`, and `2048` respectively.

`--self-check-codec-encode-project-graph` extends the diagnostic through the
acoustic encoder tail:

```text
block4 output (ne1=2048)
  -> Snake(a.ae.snake1.alpha)
  -> ggml_conv_1d(a.ae.conv2.weight, stride=1, padding=1, dilation=1)
  -> add a.ae.conv2.bias
```

It checks finite values and channel shape `(ne1=256, ne2=1)`. This is still
acoustic-only; concat `a.fc` requires the semantic branch and is not part of
this gate.

`--self-check-codec-encode-semantic-conv-graph` starts the semantic encoder
graph after HuBERT hidden-state averaging/downsampling:

```text
semantic features (ne0=frames, ne1=768, ne2=1)
  -> ggml_conv_1d(a.se.conv.weight, stride=1, padding=1, dilation=1)
```

It checks `(ne0=frames, ne1=768, ne2=1)` and finite values for synthetic
semantic features.

`--self-check-codec-encode-semantic-block0-graph` and
`--self-check-codec-encode-semantic-block1-graph` extend that diagnostic through
the two semantic encoder blocks:

```text
semantic conv output
  -> block.res_units.0: ELU -> ggml_conv_1d(padding=1) -> ELU -> ggml_conv_1d(padding=0) -> residual add
  -> block.res_units.1: same
  -> ggml_conv_1d(block.conv.weight, stride=1, padding=1, dilation=1)
  -> add block.conv.bias
```

They check finite values and preserve `(ne1=768, ne2=1)`. HuBERT feature
extraction and HuBERT encoder layers remain separate gates.

`--self-check-codec-encode-hubert-fe-conv0-conv-graph` starts the HuBERT
feature extractor from a padded 16 kHz waveform:

```text
waveform (ne0=samples, ne1=1, ne2=1)
  -> ggml_conv_1d(a.sm.fe.conv_layers.0.conv.weight, stride=5, padding=0, dilation=1)
```

It checks finite values and channel shape `(ne1=512)`.

`--self-check-codec-encode-hubert-fe-conv0-graph` extends that path through the
conv0 normalization and activation used by the Python graph:

```text
conv0 output
  -> reshape for ggml_group_norm(groups=512, eps=1e-5)
  -> affine a.sm.fe.conv_layers.0.layer_norm.{weight,bias}
  -> GELU
```

`--self-check-codec-encode-hubert-fe-graph` runs the full 7-layer HuBERT feature
extractor using strides `(5, 2, 2, 2, 2, 2, 2)`, `padding=0`, `dilation=1`, and
GELU after each convolution. It checks finite values and channel shape
`(ne1=512)`. HuBERT feature projection and encoder layers stay separate gates.

`--self-check-codec-encode-hubert-fp-graph` extends the feature extractor
through HuBERT feature projection:

```text
feature output
  -> transpose + cont_2d to (ne0=512, ne1=frames)
  -> ggml_norm(eps=1e-5) + a.sm.fp.layer_norm.{weight,bias}
  -> ggml_mul_mat(a.sm.fp.projection.weight)
  -> add a.sm.fp.projection.bias
  -> transpose + cont_2d to (ne0=frames, ne1=768)
```

It checks finite values and channel shape `(ne1=768)`. HuBERT positional
convolution and encoder layers stay separate gates.

`--self-check-codec-encode-hubert-pce-graph` validates the HuBERT positional
convolution over synthetic projected hidden states `(ne0=frames, ne1=768)`.
The GGUF stores weight-normalized tensors `a.sm.encoder.pce.conv.pz.w0` and
`w1`; the C++ gate derives the same 16 F16 group kernels as Python, then runs:

```text
hidden groups of 48 channels
  -> ggml_conv_1d(group_kernel, stride=1, padding=64, dilation=1)
  -> crop/view to input frame count
  -> add matching 48-channel bias slice
  -> concat 16 groups on channel dim
  -> hidden + GELU(pos)
```

The temporary group kernels stay F16 because GGML CPU `conv_1d` requires F16
kernel tensors for this path. The gate checks finite values and preserves
`(ne0=frames, ne1=768)`.

`--self-check-codec-encode-hubert-prelude-graph` extends the synthetic PCE gate
through the HuBERT encoder prelude normalization:

```text
hidden + GELU(pos)
  -> transpose + cont_2d to (ne0=768, ne1=frames)
  -> ggml_norm(eps=1e-5) + a.sm.encoder.layer_norm.{weight,bias}
  -> transpose + cont_2d to (ne0=frames, ne1=768)
```

It checks finite values and preserves `(ne0=frames, ne1=768)`. HuBERT encoder
layers remain separate gates.

`--self-check-codec-encode-hubert-layer0-graph` validates one HuBERT encoder
layer over synthetic prelude hidden states:

```text
hidden (ne0=frames, ne1=768)
  -> Q/K/V linear projections + bias
  -> 12-head attention, scale=1/8, softmax
  -> attention output projection + residual
  -> layer_norm
  -> FFN 768 -> 3072 -> GELU -> 768 + residual
  -> final_layer_norm
```

It checks finite values and preserves `(ne0=frames, ne1=768)`. Remaining HuBERT
encoder layers and hidden-state mean/downsample stay separate gates.

`--self-check-codec-encode-hubert-layers-graph` reuses the same layer helper and
runs all 12 HuBERT encoder layers over synthetic prelude hidden states. It
checks finite values and preserves `(ne0=frames, ne1=768)`. Hidden-state
mean/downsample remains a separate gate.

`--self-check-codec-encode-hubert-mean-graph` validates the semantic feature
contract after HuBERT encoder layers:

```text
hidden_sum = prelude_hidden + layer0 + ... + layer11
hidden_mean = hidden_sum / 13
semantic_features = hidden_mean[::2] along frame dim
```

The C++ gate transposes to `(ne0=768, ne1=frames)` before `ggml_get_rows` so the
row IDs select frame positions `0, 2, 4, ...`. It checks finite values and
channel shape `(ne0=768, ne1=ceil(frames/2))` for synthetic hidden input.

`--self-check-codec-encode-fc-graph` validates the post-encoder concat
projection:

```text
acoustic (ne0=frames, ne1=256, ne2=1)
semantic (ne0=frames, ne1=768, ne2=1)
  -> ggml_concat(dim=1) to 1024 channels
  -> reshape_2d(frames, 1024) -> transpose -> cont_2d(1024, frames)
  -> ggml_mul_mat(a.fc.weight)
  -> add a.fc.bias
```

It checks finite values and output shape `(ne0=1024, ne1=frames)` for synthetic
acoustic and semantic inputs.

`--self-check-codec-encode-quantizer0-graph` validates the first RVQ encode
stage:

```text
residual (ne0=1024, ne1=frames)
  -> ggml_mul_mat(a.q.quantizers.0.project_in.weight)
  -> add a.q.quantizers.0.project_in.bias
  -> score = 2 * ggml_mul_mat(codebook.embed, projected) - ||codebook.embed||^2
  -> ggml_argmax(score)
```

The codebook norm is precomputed on CPU from the GGUF `codebook.embed` tensor
and passed as a small F32 tensor. The gate checks output shape `(ne0=frames)`
and code IDs in `[0, 1024)`.

`--self-check-codec-encode-quantizers-graph` extends the same encode path
through all 8 RVQ stages. After each `ggml_argmax`, it runs:

```text
ids
  -> ggml_get_rows(codebook.embed)
  -> ggml_mul_mat(project_out.weight)
  -> add project_out.bias
  -> residual = residual - quantized
```

It checks each codebook output shape `(ne0=frames)` and code IDs in
`[0, 1024)`.

## CLI

Required flags:

- `--model`
- `--text`
- `--ref-wav`
- `--ref-text`
- `--prompt-format higgstts|chatml|boson-chatml`
- `--system-prompt`
- `--temperature`
- `--top-k`
- `--top-p`
- `--seed`
- `--steps`
- `--out`

First output format is mono 24 kHz WAV.

Current `HiggsPipeline::generate` behavior:

- without `--ref-wav`: builds `higgstts`, `chatml`, or `boson-chatml` prompt
  ids, runs no-KV AR recompute, then codec decode;
- with `--ref-wav`: runs local reference codec encode, builds reference prompt
  ids with optional `--ref-text`, overlays reference audio embeddings on the
  `-100` placeholders, runs the same no-KV AR recompute, then codec decode.
- `prompt_format` values other than `higgstts` currently fail explicitly for
  reference-audio generation instead of silently falling back to HiggsTTS.

## Server

The server is a thin wrapper over `HiggsPipeline`. First route:

```text
POST /generate
```

Current JSON fields are `text`, `ref_wav`, `ref_text`, `prompt_format`,
`system_prompt`, `steps`, `temperature`, `top_k`, `top_p`, and `seed`; omitted
values use CLI defaults. The response is raw mono 24 kHz WAV bytes with
`Content-Type: audio/wav`. Streaming remains deferred.

## Python Evidence

Fresh Python checks that define the current C++ target:

- `check_delay_pattern.py`
- `check_text_runtime_smoke.py`
- `check_text_full_logits_diagnostic.py`
- `check_codec_decoder_waveform_parity.py`
- `check_reference_audio_native_tail_encode.py --native-semantic`
- `check_offline_zero_shot_pipeline_smoke.py`
- `check_offline_pipeline_smoke.py`
- `check_offline_sampled_chatml_pipeline_smoke.py`

Known residual risks are post-graph quality/backend policy:

- relaxed acoustic/RVQ drift in native reference encode;
- CUDA codec decode dtype policy;
- listening quality.

## M1 Required Tensor Probe

The C++ loader validates these graph-frontier tensors before deeper graph
ports:

- `token_embd.weight`
- `output_norm.weight`
- `a.token_embd`
- `a.output`
- `blk.0.attn_q.weight`
- `blk.0.ffn_down.weight`
- `a.q.quantizers.0.codebook.embed`
- `a.fc2.weight`
- `a.ad.conv1.weight`
- `a.ae.conv1.weight`
- `a.se.conv.weight`
- `a.fc.weight`
