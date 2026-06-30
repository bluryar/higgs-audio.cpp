#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <csignal>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace higgs_audio {

constexpr int k_num_codebooks = 8;
constexpr int k_codebook_data_size = 1024;
constexpr int k_codebook_vocab_size = 1026;
constexpr int k_boc_id = 1024;
constexpr int k_eoc_id = 1025;
inline constexpr const char * k_default_model_path = "/root/code/ggbond/models/higgs-audio-v3-tts-4b-mixed.gguf";

enum class PromptFormat {
    higgstts,
    chatml,
    boson_chatml,
};

enum class BackendKind {
    cpu,
    cuda,
};

struct GenerateOptions {
    std::string model_path = k_default_model_path;
    std::string text;
    std::string ref_wav;
    std::string ref_text;
    std::string system_prompt;
    BackendKind backend = BackendKind::cpu;
    PromptFormat prompt_format = PromptFormat::higgstts;
    float temperature = 0.0f;
    int top_k = 0;
    float top_p = 1.0f;
    uint32_t seed = 12345;
    int steps = 160;
    bool stop_on_eoc = true;
    const volatile sig_atomic_t * cancel = nullptr;
    std::chrono::steady_clock::time_point deadline{};
};

struct AudioData {
    std::vector<float> samples;
    int sample_rate = 0;
};

struct ReferenceAudioData {
    AudioData original;
    std::vector<float> audio_24k;
    std::vector<float> semantic_16k;
};

struct TensorInfo {
    std::string name;
    std::string type;
    std::vector<int64_t> ne;
    size_t bytes = 0;
};

struct ModelInfo {
    int64_t tensor_count = 0;
    int64_t kv_count = 0;
    int64_t audio_num_codebooks = 0;
    int64_t audio_vocab_size = 0;
    std::string config_json;
    std::vector<TensorInfo> required_tensors;
};

struct SpecialTokenIds {
    int32_t tts = -1;
    int32_t text = -1;
    int32_t audio = -1;
    int32_t ref_text = -1;
    int32_t ref_audio = -1;
};

struct TextConfig {
    int64_t vocab_size = 0;
    int64_t context_length = 0;
    int64_t embedding_length = 0;
    int64_t feed_forward_length = 0;
    int64_t block_count = 0;
    int64_t head_count = 0;
    int64_t head_count_kv = 0;
    int64_t key_length = 0;
    int64_t value_length = 0;
    int64_t rope_dimension_count = 0;
    float rms_norm_eps = 0.0f;
    float rope_freq_base = 0.0f;
    std::vector<TensorInfo> frontier_tensors;
};

struct ReferenceEncodeInfo {
    int64_t sample_rate = 24000;
    int64_t semantic_sample_rate = 16000;
    int64_t downsample_factor = 320;
    int64_t semantic_downsample_factor = 2;
    int64_t semantic_hidden_size = 768;
    int64_t semantic_layer_count = 12;
    int64_t semantic_head_count = 12;
    int64_t semantic_intermediate_size = 3072;
    int64_t acoustic_tensor_count = 0;
    int64_t semantic_model_tensor_count = 0;
    int64_t semantic_encoder_tensor_count = 0;
    int64_t fc_tensor_count = 0;
    int64_t fc1_tensor_count = 0;
    int64_t fc2_tensor_count = 0;
    int64_t quantizer_tensor_count = 0;
    std::vector<TensorInfo> frontier_tensors;
};

struct TokenizerInfo {
    std::string model;
    int64_t vocab_size = 0;
    int64_t token_count = 0;
    int64_t token_type_count = 0;
    int64_t merge_count = 0;
    int64_t added_token_count = 0;
    int64_t huggingface_json_bytes = 0;
    int64_t chat_template_bytes = 0;
    int64_t eos_token_id = -1;
    int64_t padding_token_id = -1;
    SpecialTokenIds special;
};

struct TokenizerArtifactInfo {
    int64_t token_count = 0;
    int64_t merge_count = 0;
    int64_t added_token_count = 0;
    int64_t chat_template_bytes = 0;
    int64_t file_bytes = 0;
};

struct CodeMatrix {
    int codebooks = k_num_codebooks;
    int frames = 0;
    std::vector<int32_t> data;

    int32_t & at(int codebook, int frame);
    int32_t at(int codebook, int frame) const;
};

struct GenerateResult {
    std::vector<float> waveform;
    int sample_rate = 24000;
    CodeMatrix delayed_codes;
    CodeMatrix finalized_codes;
    std::string reference_cache_status;
    int backend_kv_slot = -1;
    long long reference_cache_wall_ms = 0;
    long long reference_ar_wall_ms = 0;
    long long codec_wall_ms = 0;
    long long audio_head_wall_ms = 0;
    int audio_head_batch_calls = 0;
    int audio_head_fallback_calls = 0;
    float audio_head_batch_size_avg = 0.0f;
    long long cuda_executor_wait_ms = 0;
    long long cuda_executor_run_ms = 0;
    int cuda_executor_queue_depth = 0;
    long long total_wall_ms = 0;
    std::string scheduler_mode;
};

struct TextKvCache {
    int64_t layers = 0;
    int64_t head_dim = 0;
    int64_t kv_heads = 0;
    int64_t capacity = 0;
    int64_t used = 0;
    std::vector<float> key;
    std::vector<float> value;
};

class HiggsPipeline {
public:
    explicit HiggsPipeline(std::string model_path);
    HiggsPipeline(std::string model_path, BackendKind backend);

    const std::string & model_path() const;
    GenerateResult generate(const GenerateOptions & options);

private:
    struct ReferenceCodeCacheEntry {
        CodeMatrix codes;
    };

    std::string model_path_;
    std::mutex generate_mutex_;
    std::shared_ptr<void> runtime_;
    std::unordered_map<std::string, ReferenceCodeCacheEntry> reference_code_cache_;
};

ModelInfo inspect_model(const std::string & model_path);
void validate_model_contract(const std::string & model_path);
TextConfig inspect_text_config(const std::string & model_path);
void validate_text_contract(const std::string & model_path);
ReferenceEncodeInfo inspect_reference_encode(const std::string & model_path);
void validate_reference_encode_contract(const std::string & model_path);
TokenizerInfo inspect_tokenizer(const std::string & model_path);
void validate_tokenizer_contract(const std::string & model_path);
TokenizerArtifactInfo inspect_tokenizer_artifact(const std::string & path);
void validate_tokenizer_artifact(const std::string & path);
std::vector<int32_t> encode_exact_token_strings(const std::string & path, const std::vector<std::string> & tokens);
std::vector<int32_t> encode_byte_tokens(const std::string & path, const std::string & text);
std::vector<int32_t> encode_bpe_token_chunk(const std::string & path, const std::string & text);
std::vector<int32_t> encode_text_tokenizers_cpp(const std::string & model_path, const std::string & text);
void self_check_tokenizer_artifact(const std::string & path);
void self_check_tokenizer_bytelevel(const std::string & path);
void self_check_tokenizer_bpe(const std::string & path);
void self_check_tokenizers_cpp(const std::string & model_path);
SpecialTokenIds load_special_token_ids(const std::string & model_path);
std::vector<int32_t> build_higgstts_prompt_ids(const SpecialTokenIds & ids, const std::vector<int32_t> & text_ids);
std::vector<int32_t> build_higgstts_prompt_ids(const std::string & model_path, const std::string & text);
std::vector<int32_t> build_prompt_ids(const std::string & model_path, const GenerateOptions & options);
std::vector<int32_t> build_reference_prompt_ids(const SpecialTokenIds & ids, const std::vector<int32_t> & text_ids, int delayed_ref_rows, const std::vector<int32_t> & ref_text_ids = {});
std::vector<int32_t> safe_text_ids_for_embedding(const std::vector<int32_t> & text_ids);
CodeMatrix fuse_audio_ids_for_embedding(const CodeMatrix & audio_ids);
std::vector<float> audio_logits_flat_to_vc(const std::vector<float> & flat, int frames, int frame);
std::vector<int32_t> sample_codes_from_logits_vc(const std::vector<float> & logits_vc, int vocab, int codebooks, float temperature, int top_k, float top_p, uint32_t & rng_state, bool stop_on_eoc);
void apply_sglang_step_rules(std::vector<int32_t> & codes, int step, int codebooks, int & eoc_countdown, bool stop_on_eoc);
CodeMatrix apply_delay_pattern_tn(const std::vector<int32_t> & raw_tn, int frames, int codebooks);
std::vector<int32_t> reverse_delay_pattern_tn(const CodeMatrix & delayed);
CodeMatrix finalize_generated_codes(const CodeMatrix & delayed, bool trim_bos);
void self_check_delay();
void self_check_prompt(const std::string & model_path);
void self_check_tokenizer_prompt(const std::string & model_path);
void self_check_chatml_prompt(const std::string & model_path);
void self_check_sampler();
void self_check_audio_logits_layout();
void self_check_embedding_ids();
void self_check_embedding_graph(const std::string & model_path);
void self_check_prompt_embedding_graph(const std::string & model_path);
void self_check_audio_head_graph(const std::string & model_path);
void self_check_block0_mlp_graph(const std::string & model_path);
void self_check_block0_attn_qkv_graph(const std::string & model_path, BackendKind backend = BackendKind::cpu);
void self_check_block0_attn_graph(const std::string & model_path);
void self_check_block0_graph(const std::string & model_path);
void self_check_block35_graph(const std::string & model_path);
void self_check_backbone_graph(const std::string & model_path);
void self_check_prompt_backbone_graph(const std::string & model_path);
void self_check_prompt_audio_head_graph(const std::string & model_path);
void self_check_prompt_ar_graph(const std::string & model_path);
void self_check_kv_cache(const std::string & model_path);
void self_check_kv_decode_graph(const std::string & model_path);
void self_check_kv_decode_audio_head_graph(const std::string & model_path);
void self_check_prompt_to_kv_decode_audio_head_graph(const std::string & model_path);
void self_check_prompt_kv_cache_write(const std::string & model_path);
void self_check_prompt_kv_cache_decode(const std::string & model_path);
void self_check_backend_kv_cache_write(const std::string & model_path);
void self_check_backend_kv_cache_decode(const std::string & model_path);
void self_check_backend_prompt_kv_cache_prefill(const std::string & model_path);
void self_check_prompt_ar_backend_kv(const std::string & model_path);
void self_check_reference_ar_backend_kv(const std::string & model_path);
void self_check_text_logits_graph(const std::string & model_path);
void self_check_codec_encode_conv1_graph(const std::string & model_path);
void self_check_codec_encode_block0_graph(const std::string & model_path);
void self_check_codec_encode_block1_graph(const std::string & model_path);
void self_check_codec_encode_block2_graph(const std::string & model_path);
void self_check_codec_encode_block3_graph(const std::string & model_path);
void self_check_codec_encode_block4_graph(const std::string & model_path);
void self_check_codec_encode_project_graph(const std::string & model_path);
void self_check_codec_encode_semantic_conv_graph(const std::string & model_path);
void self_check_codec_encode_semantic_block0_graph(const std::string & model_path);
void self_check_codec_encode_semantic_block1_graph(const std::string & model_path);
void self_check_codec_encode_hubert_fe_conv0_conv_graph(const std::string & model_path);
void self_check_codec_encode_hubert_fe_conv0_graph(const std::string & model_path);
void self_check_codec_encode_hubert_fe_graph(const std::string & model_path);
void self_check_codec_encode_hubert_fp_graph(const std::string & model_path);
void self_check_codec_encode_hubert_pce_graph(const std::string & model_path);
void self_check_codec_encode_hubert_prelude_graph(const std::string & model_path);
void self_check_codec_encode_hubert_layer0_graph(const std::string & model_path);
void self_check_codec_encode_hubert_layers_graph(const std::string & model_path);
void self_check_codec_encode_hubert_mean_graph(const std::string & model_path);
void self_check_codec_encode_fc_graph(const std::string & model_path);
void self_check_codec_encode_quantizer0_graph(const std::string & model_path);
void self_check_codec_encode_quantizers_graph(const std::string & model_path);
int64_t self_check_reference_acoustic(const std::string & model_path, const std::string & wav_path);
int64_t self_check_reference_semantic(const std::string & model_path, const std::string & wav_path);
CodeMatrix encode_reference_codes(const std::string & model_path, const std::string & wav_path);
CodeMatrix self_check_reference_codes(const std::string & model_path, const std::string & wav_path);
void self_check_codec_project_graph(const std::string & model_path);
void self_check_codec_conv1_graph(const std::string & model_path);
void self_check_codec_block0_conv_t1_graph(const std::string & model_path);
void self_check_codec_block0_res_unit1_graph(const std::string & model_path);
void self_check_codec_block0_graph(const std::string & model_path);
void self_check_codec_block1_graph(const std::string & model_path);
void self_check_codec_waveform_graph(const std::string & model_path);
void self_check_codec_wav(const std::string & model_path, const std::string & out_path);
void decode_trace_json_to_wav(const std::string & model_path, const std::string & trace_json_path, const std::string & out_path);
ReferenceAudioData load_reference_wav(const std::string & path);
PromptFormat parse_prompt_format(const std::string & value);
const char * prompt_format_name(PromptFormat value);
BackendKind parse_backend_kind(const std::string & value);
const char * backend_kind_name(BackendKind value);
void write_wav_mono_16(const std::string & path, const std::vector<float> & samples, int sample_rate);
std::vector<uint8_t> encode_mp3_mono(const std::vector<float> & samples, int sample_rate, int bitrate_kbps = 64);
void write_mp3_mono(const std::string & path, const std::vector<float> & samples, int sample_rate, int bitrate_kbps = 64);

} // namespace higgs_audio
