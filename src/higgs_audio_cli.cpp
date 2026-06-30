#include "higgs_audio.h"

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool ends_with(const std::string & value, const char * suffix) {
    const std::string s(suffix);
    return value.size() >= s.size() && value.compare(value.size() - s.size(), s.size(), s) == 0;
}

void usage() {
    std::cout
        << "Usage: higgs-audio-cli --text TEXT --out out.wav|out.mp3 [options]\n"
        << "\nOptions:\n"
        << "  --model PATH\n"
        << "  --backend cpu|cuda\n"
        << "  --ref-wav PATH\n"
        << "  --ref-text TEXT\n"
        << "  --prompt-format higgstts|chatml|boson-chatml\n"
        << "  --system-prompt TEXT\n"
        << "  --temperature FLOAT\n"
        << "  --top-k INT\n"
        << "  --top-p FLOAT\n"
        << "  --seed INT\n"
        << "  --steps INT\n"
        << "  --no-stop-on-eoc\n"
        << "  --decode-trace-json PATH\n"
        << "  --inspect\n"
        << "  --inspect-text\n"
        << "  --inspect-reference-encode\n"
        << "  --inspect-tokenizer\n"
        << "  --inspect-tokenizer-artifact PATH\n"
        << "  --validate-tensors\n"
        << "  --self-check-delay\n"
        << "  --self-check-tokenizer\n"
        << "  --self-check-tokenizer-artifact PATH\n"
        << "  --self-check-tokenizer-bytelevel PATH\n"
        << "  --self-check-tokenizer-bpe PATH\n"
        << "  --self-check-tokenizers-cpp\n"
        << "  --self-check-tokenizer-prompt\n"
        << "  --self-check-chatml-prompt\n"
        << "  --self-check-prompt\n"
        << "  --self-check-sampler\n"
        << "  --self-check-audio-logits\n"
        << "  --self-check-embedding-ids\n"
        << "  --self-check-embedding-graph\n"
        << "  --self-check-prompt-embedding-graph\n"
        << "  --self-check-audio-head-graph\n"
        << "  --self-check-block0-mlp-graph\n"
        << "  --self-check-block0-attn-qkv-graph\n"
        << "  --self-check-block0-attn-graph\n"
        << "  --self-check-block0-graph\n"
        << "  --self-check-block35-graph\n"
        << "  --self-check-backbone-graph\n"
        << "  --self-check-prompt-backbone-graph\n"
        << "  --self-check-prompt-audio-head-graph\n"
        << "  --self-check-prompt-ar-graph\n"
        << "  --self-check-kv-cache\n"
        << "  --self-check-kv-decode-graph\n"
        << "  --self-check-kv-decode-audio-head-graph\n"
        << "  --self-check-prompt-to-kv-decode-audio-head-graph\n"
        << "  --self-check-prompt-kv-cache-write\n"
        << "  --self-check-prompt-kv-cache-decode\n"
        << "  --self-check-backend-kv-cache-write\n"
        << "  --self-check-backend-kv-cache-decode\n"
        << "  --self-check-backend-prompt-kv-cache-prefill\n"
        << "  --self-check-prompt-ar-backend-kv\n"
        << "  --self-check-reference-ar-backend-kv\n"
        << "  --self-check-text-logits-graph\n"
        << "  --self-check-codec-encode-conv1-graph\n"
        << "  --self-check-codec-encode-block0-graph\n"
        << "  --self-check-codec-encode-block1-graph\n"
        << "  --self-check-codec-encode-block2-graph\n"
        << "  --self-check-codec-encode-block3-graph\n"
        << "  --self-check-codec-encode-block4-graph\n"
        << "  --self-check-codec-encode-project-graph\n"
        << "  --self-check-codec-encode-semantic-conv-graph\n"
        << "  --self-check-codec-encode-semantic-block0-graph\n"
        << "  --self-check-codec-encode-semantic-block1-graph\n"
        << "  --self-check-codec-encode-hubert-fe-conv0-conv-graph\n"
        << "  --self-check-codec-encode-hubert-fe-conv0-graph\n"
        << "  --self-check-codec-encode-hubert-fe-graph\n"
        << "  --self-check-codec-encode-hubert-fp-graph\n"
        << "  --self-check-codec-encode-hubert-pce-graph\n"
        << "  --self-check-codec-encode-hubert-prelude-graph\n"
        << "  --self-check-codec-encode-hubert-layer0-graph\n"
        << "  --self-check-codec-encode-hubert-layers-graph\n"
        << "  --self-check-codec-encode-hubert-mean-graph\n"
        << "  --self-check-codec-encode-fc-graph\n"
        << "  --self-check-codec-encode-quantizer0-graph\n"
        << "  --self-check-codec-encode-quantizers-graph\n"
        << "  --self-check-reference-wav PATH\n"
        << "  --self-check-reference-acoustic PATH\n"
        << "  --self-check-reference-semantic PATH\n"
        << "  --self-check-reference-codes PATH\n"
        << "  --self-check-codec-project-graph\n"
        << "  --self-check-codec-conv1-graph\n"
        << "  --self-check-codec-block0-conv-t1-graph\n"
        << "  --self-check-codec-block0-res-unit1-graph\n"
        << "  --self-check-codec-block0-graph\n"
        << "  --self-check-codec-block1-graph\n"
        << "  --self-check-codec-waveform-graph\n"
        << "  --self-check-codec-wav\n"
        << "  --help\n";
}

std::string need_value(int & i, int argc, char ** argv) {
    if (i + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + argv[i]);
    }
    return argv[++i];
}

void print_tensor(const higgs_audio::TensorInfo & tensor) {
    std::cout << "  " << tensor.name << " " << tensor.type;
    if (!tensor.ne.empty()) {
        std::cout << " ne=(";
        for (size_t i = 0; i < tensor.ne.size(); ++i) {
            if (i != 0) {
                std::cout << ",";
            }
            std::cout << tensor.ne[i];
        }
        std::cout << ")";
    }
    std::cout << " " << tensor.bytes << " bytes\n";
}

} // namespace

int main(int argc, char ** argv) {
    higgs_audio::GenerateOptions options;
    std::string out_path;
    bool inspect = false;
    bool inspect_text = false;
    bool inspect_reference_encode = false;
    bool inspect_tokenizer = false;
    std::string inspect_tokenizer_artifact_path;
    bool validate = false;
    bool self_check_delay = false;
    bool self_check_tokenizer = false;
    std::string self_check_tokenizer_artifact_path;
    std::string self_check_tokenizer_bytelevel_path;
    std::string self_check_tokenizer_bpe_path;
    bool self_check_tokenizers_cpp = false;
    bool self_check_tokenizer_prompt = false;
    bool self_check_chatml_prompt = false;
    bool self_check_prompt = false;
    bool self_check_sampler = false;
    bool self_check_audio_logits = false;
    bool self_check_embedding_ids = false;
    bool self_check_embedding_graph = false;
    bool self_check_prompt_embedding_graph = false;
    bool self_check_audio_head_graph = false;
    bool self_check_block0_mlp_graph = false;
    bool self_check_block0_attn_qkv_graph = false;
    bool self_check_block0_attn_graph = false;
    bool self_check_block0_graph = false;
    bool self_check_block35_graph = false;
    bool self_check_backbone_graph = false;
    bool self_check_prompt_backbone_graph = false;
    bool self_check_prompt_audio_head_graph = false;
    bool self_check_prompt_ar_graph = false;
    bool self_check_kv_cache = false;
    bool self_check_kv_decode_graph = false;
    bool self_check_kv_decode_audio_head_graph = false;
    bool self_check_prompt_to_kv_decode_audio_head_graph = false;
    bool self_check_prompt_kv_cache_write = false;
    bool self_check_prompt_kv_cache_decode = false;
    bool self_check_backend_kv_cache_write = false;
    bool self_check_backend_kv_cache_decode = false;
    bool self_check_backend_prompt_kv_cache_prefill = false;
    bool self_check_prompt_ar_backend_kv = false;
    bool self_check_reference_ar_backend_kv = false;
    bool self_check_text_logits_graph = false;
    bool self_check_codec_encode_conv1_graph = false;
    bool self_check_codec_encode_block0_graph = false;
    bool self_check_codec_encode_block1_graph = false;
    bool self_check_codec_encode_block2_graph = false;
    bool self_check_codec_encode_block3_graph = false;
    bool self_check_codec_encode_block4_graph = false;
    bool self_check_codec_encode_project_graph = false;
    bool self_check_codec_encode_semantic_conv_graph = false;
    bool self_check_codec_encode_semantic_block0_graph = false;
    bool self_check_codec_encode_semantic_block1_graph = false;
    bool self_check_codec_encode_hubert_fe_conv0_conv_graph = false;
    bool self_check_codec_encode_hubert_fe_conv0_graph = false;
    bool self_check_codec_encode_hubert_fe_graph = false;
    bool self_check_codec_encode_hubert_fp_graph = false;
    bool self_check_codec_encode_hubert_pce_graph = false;
    bool self_check_codec_encode_hubert_prelude_graph = false;
    bool self_check_codec_encode_hubert_layer0_graph = false;
    bool self_check_codec_encode_hubert_layers_graph = false;
    bool self_check_codec_encode_hubert_mean_graph = false;
    bool self_check_codec_encode_fc_graph = false;
    bool self_check_codec_encode_quantizer0_graph = false;
    bool self_check_codec_encode_quantizers_graph = false;
    std::string self_check_reference_wav_path;
    std::string self_check_reference_acoustic_path;
    std::string self_check_reference_semantic_path;
    std::string self_check_reference_codes_path;
    bool self_check_codec_project_graph = false;
    bool self_check_codec_conv1_graph = false;
    bool self_check_codec_block0_conv_t1_graph = false;
    bool self_check_codec_block0_res_unit1_graph = false;
    bool self_check_codec_block0_graph = false;
    bool self_check_codec_block1_graph = false;
    bool self_check_codec_waveform_graph = false;
    bool self_check_codec_wav = false;
    std::string decode_trace_json_path;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help") {
                usage();
                return 0;
            }
            if (arg == "--model") {
                options.model_path = need_value(i, argc, argv);
            } else if (arg == "--backend") {
                options.backend = higgs_audio::parse_backend_kind(need_value(i, argc, argv));
            } else if (arg == "--text") {
                options.text = need_value(i, argc, argv);
            } else if (arg == "--ref-wav") {
                options.ref_wav = need_value(i, argc, argv);
            } else if (arg == "--ref-text") {
                options.ref_text = need_value(i, argc, argv);
            } else if (arg == "--prompt-format") {
                options.prompt_format = higgs_audio::parse_prompt_format(need_value(i, argc, argv));
            } else if (arg == "--system-prompt") {
                options.system_prompt = need_value(i, argc, argv);
            } else if (arg == "--temperature") {
                options.temperature = std::stof(need_value(i, argc, argv));
            } else if (arg == "--top-k") {
                options.top_k = std::stoi(need_value(i, argc, argv));
            } else if (arg == "--top-p") {
                options.top_p = std::stof(need_value(i, argc, argv));
            } else if (arg == "--seed") {
                options.seed = static_cast<uint32_t>(std::stoul(need_value(i, argc, argv)));
            } else if (arg == "--steps") {
                options.steps = std::stoi(need_value(i, argc, argv));
            } else if (arg == "--out") {
                out_path = need_value(i, argc, argv);
            } else if (arg == "--no-stop-on-eoc") {
                options.stop_on_eoc = false;
            } else if (arg == "--decode-trace-json") {
                decode_trace_json_path = need_value(i, argc, argv);
            } else if (arg == "--inspect") {
                inspect = true;
            } else if (arg == "--inspect-text") {
                inspect_text = true;
            } else if (arg == "--inspect-reference-encode") {
                inspect_reference_encode = true;
            } else if (arg == "--inspect-tokenizer") {
                inspect_tokenizer = true;
            } else if (arg == "--inspect-tokenizer-artifact") {
                inspect_tokenizer_artifact_path = need_value(i, argc, argv);
            } else if (arg == "--validate-tensors") {
                validate = true;
            } else if (arg == "--self-check-delay") {
                self_check_delay = true;
            } else if (arg == "--self-check-tokenizer") {
                self_check_tokenizer = true;
            } else if (arg == "--self-check-tokenizer-artifact") {
                self_check_tokenizer_artifact_path = need_value(i, argc, argv);
            } else if (arg == "--self-check-tokenizer-bytelevel") {
                self_check_tokenizer_bytelevel_path = need_value(i, argc, argv);
            } else if (arg == "--self-check-tokenizer-bpe") {
                self_check_tokenizer_bpe_path = need_value(i, argc, argv);
            } else if (arg == "--self-check-tokenizers-cpp") {
                self_check_tokenizers_cpp = true;
            } else if (arg == "--self-check-tokenizer-prompt") {
                self_check_tokenizer_prompt = true;
            } else if (arg == "--self-check-chatml-prompt") {
                self_check_chatml_prompt = true;
            } else if (arg == "--self-check-prompt") {
                self_check_prompt = true;
            } else if (arg == "--self-check-sampler") {
                self_check_sampler = true;
            } else if (arg == "--self-check-audio-logits") {
                self_check_audio_logits = true;
            } else if (arg == "--self-check-embedding-ids") {
                self_check_embedding_ids = true;
            } else if (arg == "--self-check-embedding-graph") {
                self_check_embedding_graph = true;
            } else if (arg == "--self-check-prompt-embedding-graph") {
                self_check_prompt_embedding_graph = true;
            } else if (arg == "--self-check-audio-head-graph") {
                self_check_audio_head_graph = true;
            } else if (arg == "--self-check-block0-mlp-graph") {
                self_check_block0_mlp_graph = true;
            } else if (arg == "--self-check-block0-attn-qkv-graph") {
                self_check_block0_attn_qkv_graph = true;
            } else if (arg == "--self-check-block0-attn-graph") {
                self_check_block0_attn_graph = true;
            } else if (arg == "--self-check-block0-graph") {
                self_check_block0_graph = true;
            } else if (arg == "--self-check-block35-graph") {
                self_check_block35_graph = true;
            } else if (arg == "--self-check-backbone-graph") {
                self_check_backbone_graph = true;
            } else if (arg == "--self-check-prompt-backbone-graph") {
                self_check_prompt_backbone_graph = true;
            } else if (arg == "--self-check-prompt-audio-head-graph") {
                self_check_prompt_audio_head_graph = true;
            } else if (arg == "--self-check-prompt-ar-graph") {
                self_check_prompt_ar_graph = true;
            } else if (arg == "--self-check-kv-cache") {
                self_check_kv_cache = true;
            } else if (arg == "--self-check-kv-decode-graph") {
                self_check_kv_decode_graph = true;
            } else if (arg == "--self-check-kv-decode-audio-head-graph") {
                self_check_kv_decode_audio_head_graph = true;
            } else if (arg == "--self-check-prompt-to-kv-decode-audio-head-graph") {
                self_check_prompt_to_kv_decode_audio_head_graph = true;
            } else if (arg == "--self-check-prompt-kv-cache-write") {
                self_check_prompt_kv_cache_write = true;
            } else if (arg == "--self-check-prompt-kv-cache-decode") {
                self_check_prompt_kv_cache_decode = true;
            } else if (arg == "--self-check-backend-kv-cache-write") {
                self_check_backend_kv_cache_write = true;
            } else if (arg == "--self-check-backend-kv-cache-decode") {
                self_check_backend_kv_cache_decode = true;
            } else if (arg == "--self-check-backend-prompt-kv-cache-prefill") {
                self_check_backend_prompt_kv_cache_prefill = true;
            } else if (arg == "--self-check-prompt-ar-backend-kv") {
                self_check_prompt_ar_backend_kv = true;
            } else if (arg == "--self-check-reference-ar-backend-kv") {
                self_check_reference_ar_backend_kv = true;
            } else if (arg == "--self-check-text-logits-graph") {
                self_check_text_logits_graph = true;
            } else if (arg == "--self-check-codec-encode-conv1-graph") {
                self_check_codec_encode_conv1_graph = true;
            } else if (arg == "--self-check-codec-encode-block0-graph") {
                self_check_codec_encode_block0_graph = true;
            } else if (arg == "--self-check-codec-encode-block1-graph") {
                self_check_codec_encode_block1_graph = true;
            } else if (arg == "--self-check-codec-encode-block2-graph") {
                self_check_codec_encode_block2_graph = true;
            } else if (arg == "--self-check-codec-encode-block3-graph") {
                self_check_codec_encode_block3_graph = true;
            } else if (arg == "--self-check-codec-encode-block4-graph") {
                self_check_codec_encode_block4_graph = true;
            } else if (arg == "--self-check-codec-encode-project-graph") {
                self_check_codec_encode_project_graph = true;
            } else if (arg == "--self-check-codec-encode-semantic-conv-graph") {
                self_check_codec_encode_semantic_conv_graph = true;
            } else if (arg == "--self-check-codec-encode-semantic-block0-graph") {
                self_check_codec_encode_semantic_block0_graph = true;
            } else if (arg == "--self-check-codec-encode-semantic-block1-graph") {
                self_check_codec_encode_semantic_block1_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-fe-conv0-conv-graph") {
                self_check_codec_encode_hubert_fe_conv0_conv_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-fe-conv0-graph") {
                self_check_codec_encode_hubert_fe_conv0_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-fe-graph") {
                self_check_codec_encode_hubert_fe_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-fp-graph") {
                self_check_codec_encode_hubert_fp_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-pce-graph") {
                self_check_codec_encode_hubert_pce_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-prelude-graph") {
                self_check_codec_encode_hubert_prelude_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-layer0-graph") {
                self_check_codec_encode_hubert_layer0_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-layers-graph") {
                self_check_codec_encode_hubert_layers_graph = true;
            } else if (arg == "--self-check-codec-encode-hubert-mean-graph") {
                self_check_codec_encode_hubert_mean_graph = true;
            } else if (arg == "--self-check-codec-encode-fc-graph") {
                self_check_codec_encode_fc_graph = true;
            } else if (arg == "--self-check-codec-encode-quantizer0-graph") {
                self_check_codec_encode_quantizer0_graph = true;
            } else if (arg == "--self-check-codec-encode-quantizers-graph") {
                self_check_codec_encode_quantizers_graph = true;
            } else if (arg == "--self-check-reference-wav") {
                self_check_reference_wav_path = need_value(i, argc, argv);
            } else if (arg == "--self-check-reference-acoustic") {
                self_check_reference_acoustic_path = need_value(i, argc, argv);
            } else if (arg == "--self-check-reference-semantic") {
                self_check_reference_semantic_path = need_value(i, argc, argv);
            } else if (arg == "--self-check-reference-codes") {
                self_check_reference_codes_path = need_value(i, argc, argv);
            } else if (arg == "--self-check-codec-project-graph") {
                self_check_codec_project_graph = true;
            } else if (arg == "--self-check-codec-conv1-graph") {
                self_check_codec_conv1_graph = true;
            } else if (arg == "--self-check-codec-block0-conv-t1-graph") {
                self_check_codec_block0_conv_t1_graph = true;
            } else if (arg == "--self-check-codec-block0-res-unit1-graph") {
                self_check_codec_block0_res_unit1_graph = true;
            } else if (arg == "--self-check-codec-block0-graph") {
                self_check_codec_block0_graph = true;
            } else if (arg == "--self-check-codec-block1-graph") {
                self_check_codec_block1_graph = true;
            } else if (arg == "--self-check-codec-waveform-graph") {
                self_check_codec_waveform_graph = true;
            } else if (arg == "--self-check-codec-wav") {
                self_check_codec_wav = true;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }

        if (!decode_trace_json_path.empty()) {
            if (out_path.empty()) {
                throw std::invalid_argument("--out is required for --decode-trace-json");
            }
            higgs_audio::decode_trace_json_to_wav(options.model_path, decode_trace_json_path, out_path);
            std::cout << "decoded trace json to " << out_path << "\n";
            return 0;
        }

        if (self_check_codec_wav) {
            if (out_path.empty()) {
                throw std::invalid_argument("--out is required for --self-check-codec-wav");
            }
            higgs_audio::self_check_codec_wav(options.model_path, out_path);
            std::cout << "codec wav self-check wrote " << out_path << "\n";
            return 0;
        }

        if (self_check_codec_waveform_graph) {
            higgs_audio::self_check_codec_waveform_graph(options.model_path);
            std::cout << "codec waveform graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_block1_graph) {
            higgs_audio::self_check_codec_block1_graph(options.model_path);
            std::cout << "codec block1 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_block0_graph) {
            higgs_audio::self_check_codec_block0_graph(options.model_path);
            std::cout << "codec block0 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_block0_res_unit1_graph) {
            higgs_audio::self_check_codec_block0_res_unit1_graph(options.model_path);
            std::cout << "codec block0 res_unit1 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_block0_conv_t1_graph) {
            higgs_audio::self_check_codec_block0_conv_t1_graph(options.model_path);
            std::cout << "codec block0 conv_t1 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_conv1_graph) {
            higgs_audio::self_check_codec_conv1_graph(options.model_path);
            std::cout << "codec conv1 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_project_graph) {
            higgs_audio::self_check_codec_project_graph(options.model_path);
            std::cout << "codec project graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_conv1_graph) {
            higgs_audio::self_check_codec_encode_conv1_graph(options.model_path);
            std::cout << "codec encode conv1 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_block0_graph) {
            higgs_audio::self_check_codec_encode_block0_graph(options.model_path);
            std::cout << "codec encode block0 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_block1_graph) {
            higgs_audio::self_check_codec_encode_block1_graph(options.model_path);
            std::cout << "codec encode block1 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_block2_graph) {
            higgs_audio::self_check_codec_encode_block2_graph(options.model_path);
            std::cout << "codec encode block2 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_block3_graph) {
            higgs_audio::self_check_codec_encode_block3_graph(options.model_path);
            std::cout << "codec encode block3 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_block4_graph) {
            higgs_audio::self_check_codec_encode_block4_graph(options.model_path);
            std::cout << "codec encode block4 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_project_graph) {
            higgs_audio::self_check_codec_encode_project_graph(options.model_path);
            std::cout << "codec encode project graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_semantic_conv_graph) {
            higgs_audio::self_check_codec_encode_semantic_conv_graph(options.model_path);
            std::cout << "codec encode semantic conv graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_semantic_block0_graph) {
            higgs_audio::self_check_codec_encode_semantic_block0_graph(options.model_path);
            std::cout << "codec encode semantic block0 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_semantic_block1_graph) {
            higgs_audio::self_check_codec_encode_semantic_block1_graph(options.model_path);
            std::cout << "codec encode semantic block1 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_fe_conv0_conv_graph) {
            higgs_audio::self_check_codec_encode_hubert_fe_conv0_conv_graph(options.model_path);
            std::cout << "codec encode HuBERT feature conv0 conv graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_fe_conv0_graph) {
            higgs_audio::self_check_codec_encode_hubert_fe_conv0_graph(options.model_path);
            std::cout << "codec encode HuBERT feature conv0 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_fe_graph) {
            higgs_audio::self_check_codec_encode_hubert_fe_graph(options.model_path);
            std::cout << "codec encode HuBERT feature graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_fp_graph) {
            higgs_audio::self_check_codec_encode_hubert_fp_graph(options.model_path);
            std::cout << "codec encode HuBERT feature projection graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_pce_graph) {
            higgs_audio::self_check_codec_encode_hubert_pce_graph(options.model_path);
            std::cout << "codec encode HuBERT positional conv graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_prelude_graph) {
            higgs_audio::self_check_codec_encode_hubert_prelude_graph(options.model_path);
            std::cout << "codec encode HuBERT prelude graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_layer0_graph) {
            higgs_audio::self_check_codec_encode_hubert_layer0_graph(options.model_path);
            std::cout << "codec encode HuBERT layer0 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_layers_graph) {
            higgs_audio::self_check_codec_encode_hubert_layers_graph(options.model_path);
            std::cout << "codec encode HuBERT layers graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_hubert_mean_graph) {
            higgs_audio::self_check_codec_encode_hubert_mean_graph(options.model_path);
            std::cout << "codec encode HuBERT mean graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_fc_graph) {
            higgs_audio::self_check_codec_encode_fc_graph(options.model_path);
            std::cout << "codec encode fc graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_quantizer0_graph) {
            higgs_audio::self_check_codec_encode_quantizer0_graph(options.model_path);
            std::cout << "codec encode quantizer0 graph self-check ok\n";
            return 0;
        }

        if (self_check_codec_encode_quantizers_graph) {
            higgs_audio::self_check_codec_encode_quantizers_graph(options.model_path);
            std::cout << "codec encode quantizers graph self-check ok\n";
            return 0;
        }

        if (!self_check_reference_wav_path.empty()) {
            const higgs_audio::ReferenceAudioData ref = higgs_audio::load_reference_wav(self_check_reference_wav_path);
            std::cout
                << "reference_wav_sample_rate: " << ref.original.sample_rate << "\n"
                << "reference_wav_samples: " << ref.original.samples.size() << "\n"
                << "reference_audio_24k_samples: " << ref.audio_24k.size() << "\n"
                << "reference_semantic_16k_samples: " << ref.semantic_16k.size() << "\n";
            return 0;
        }

        if (!self_check_reference_acoustic_path.empty()) {
            const int64_t frames = higgs_audio::self_check_reference_acoustic(options.model_path, self_check_reference_acoustic_path);
            std::cout << "reference_acoustic_frames: " << frames << "\n";
            return 0;
        }

        if (!self_check_reference_semantic_path.empty()) {
            const int64_t frames = higgs_audio::self_check_reference_semantic(options.model_path, self_check_reference_semantic_path);
            std::cout << "reference_semantic_frames: " << frames << "\n";
            return 0;
        }

        if (!self_check_reference_codes_path.empty()) {
            const higgs_audio::CodeMatrix codes = higgs_audio::self_check_reference_codes(options.model_path, self_check_reference_codes_path);
            std::cout << "reference_code_frames: " << codes.frames << "\n";
            return 0;
        }

        if (self_check_text_logits_graph) {
            higgs_audio::self_check_text_logits_graph(options.model_path);
            std::cout << "text logits graph self-check ok\n";
            return 0;
        }

        if (self_check_backbone_graph) {
            higgs_audio::self_check_backbone_graph(options.model_path);
            std::cout << "backbone graph self-check ok\n";
            return 0;
        }

        if (self_check_prompt_backbone_graph) {
            higgs_audio::self_check_prompt_backbone_graph(options.model_path);
            std::cout << "prompt backbone graph self-check ok\n";
            return 0;
        }

        if (self_check_prompt_audio_head_graph) {
            higgs_audio::self_check_prompt_audio_head_graph(options.model_path);
            std::cout << "prompt audio head graph self-check ok\n";
            return 0;
        }

        if (self_check_prompt_ar_graph) {
            higgs_audio::self_check_prompt_ar_graph(options.model_path);
            std::cout << "prompt AR graph self-check ok\n";
            return 0;
        }

        if (self_check_kv_cache) {
            higgs_audio::self_check_kv_cache(options.model_path);
            std::cout << "KV cache self-check ok\n";
            return 0;
        }

        if (self_check_kv_decode_graph) {
            higgs_audio::self_check_kv_decode_graph(options.model_path);
            std::cout << "KV decode graph self-check ok\n";
            return 0;
        }

        if (self_check_kv_decode_audio_head_graph) {
            higgs_audio::self_check_kv_decode_audio_head_graph(options.model_path);
            std::cout << "KV decode audio head graph self-check ok\n";
            return 0;
        }

        if (self_check_prompt_to_kv_decode_audio_head_graph) {
            higgs_audio::self_check_prompt_to_kv_decode_audio_head_graph(options.model_path);
            std::cout << "prompt-to-KV decode audio head graph self-check ok\n";
            return 0;
        }

        if (self_check_prompt_kv_cache_write) {
            higgs_audio::self_check_prompt_kv_cache_write(options.model_path);
            std::cout << "prompt KV cache write self-check ok\n";
            return 0;
        }

        if (self_check_prompt_kv_cache_decode) {
            higgs_audio::self_check_prompt_kv_cache_decode(options.model_path);
            std::cout << "prompt KV cache decode self-check ok\n";
            return 0;
        }

        if (self_check_backend_kv_cache_write) {
            higgs_audio::self_check_backend_kv_cache_write(options.model_path);
            std::cout << "backend KV cache write self-check ok\n";
            return 0;
        }

        if (self_check_backend_kv_cache_decode) {
            higgs_audio::self_check_backend_kv_cache_decode(options.model_path);
            std::cout << "backend KV cache decode self-check ok\n";
            return 0;
        }

        if (self_check_backend_prompt_kv_cache_prefill) {
            higgs_audio::self_check_backend_prompt_kv_cache_prefill(options.model_path);
            std::cout << "backend prompt KV cache prefill self-check ok\n";
            return 0;
        }

        if (self_check_prompt_ar_backend_kv) {
            higgs_audio::self_check_prompt_ar_backend_kv(options.model_path);
            std::cout << "prompt AR backend KV self-check ok\n";
            return 0;
        }

        if (self_check_reference_ar_backend_kv) {
            higgs_audio::self_check_reference_ar_backend_kv(options.model_path);
            std::cout << "reference AR backend KV self-check ok\n";
            return 0;
        }

        if (self_check_block35_graph) {
            higgs_audio::self_check_block35_graph(options.model_path);
            std::cout << "block35 graph self-check ok\n";
            return 0;
        }

        if (self_check_block0_graph) {
            higgs_audio::self_check_block0_graph(options.model_path);
            std::cout << "block0 graph self-check ok\n";
            return 0;
        }

        if (self_check_block0_attn_graph) {
            higgs_audio::self_check_block0_attn_graph(options.model_path);
            std::cout << "block0 attention graph self-check ok\n";
            return 0;
        }

        if (self_check_block0_attn_qkv_graph) {
            higgs_audio::self_check_block0_attn_qkv_graph(options.model_path, options.backend);
            std::cout << "block0 attention QKV graph self-check ok\n";
            return 0;
        }

        if (self_check_block0_mlp_graph) {
            higgs_audio::self_check_block0_mlp_graph(options.model_path);
            std::cout << "block0 MLP graph self-check ok\n";
            return 0;
        }

        if (self_check_audio_head_graph) {
            higgs_audio::self_check_audio_head_graph(options.model_path);
            std::cout << "audio head graph self-check ok\n";
            return 0;
        }

        if (self_check_embedding_graph) {
            higgs_audio::self_check_embedding_graph(options.model_path);
            std::cout << "embedding graph self-check ok\n";
            return 0;
        }

        if (self_check_prompt_embedding_graph) {
            higgs_audio::self_check_prompt_embedding_graph(options.model_path);
            std::cout << "prompt embedding graph self-check ok\n";
            return 0;
        }

        if (self_check_embedding_ids) {
            higgs_audio::self_check_embedding_ids();
            std::cout << "embedding ids self-check ok\n";
            return 0;
        }

        if (self_check_audio_logits) {
            higgs_audio::self_check_audio_logits_layout();
            std::cout << "audio logits layout self-check ok\n";
            return 0;
        }

        if (self_check_sampler) {
            higgs_audio::self_check_sampler();
            std::cout << "sampler self-check ok\n";
            return 0;
        }

        if (self_check_prompt) {
            higgs_audio::self_check_prompt(options.model_path);
            std::cout << "prompt self-check ok\n";
            return 0;
        }

        if (self_check_tokenizer) {
            higgs_audio::validate_tokenizer_contract(options.model_path);
            std::cout << "tokenizer self-check ok\n";
            return 0;
        }

        if (!self_check_tokenizer_artifact_path.empty()) {
            higgs_audio::self_check_tokenizer_artifact(self_check_tokenizer_artifact_path);
            std::cout << "tokenizer artifact self-check ok\n";
            return 0;
        }

        if (!self_check_tokenizer_bytelevel_path.empty()) {
            higgs_audio::self_check_tokenizer_bytelevel(self_check_tokenizer_bytelevel_path);
            std::cout << "tokenizer byte-level self-check ok\n";
            return 0;
        }

        if (!self_check_tokenizer_bpe_path.empty()) {
            higgs_audio::self_check_tokenizer_bpe(self_check_tokenizer_bpe_path);
            std::cout << "tokenizer BPE self-check ok\n";
            return 0;
        }

        if (self_check_tokenizers_cpp) {
            higgs_audio::self_check_tokenizers_cpp(options.model_path);
            std::cout << "tokenizers.cpp self-check ok\n";
            return 0;
        }

        if (self_check_tokenizer_prompt) {
            higgs_audio::self_check_tokenizer_prompt(options.model_path);
            std::cout << "tokenizer prompt self-check ok\n";
            return 0;
        }

        if (self_check_chatml_prompt) {
            higgs_audio::self_check_chatml_prompt(options.model_path);
            std::cout << "chatml prompt self-check ok\n";
            return 0;
        }

        if (self_check_delay) {
            higgs_audio::self_check_delay();
            std::cout << "delay self-check ok\n";
            return 0;
        }

        if (inspect_text) {
            higgs_audio::validate_text_contract(options.model_path);
            const higgs_audio::TextConfig cfg = higgs_audio::inspect_text_config(options.model_path);
            std::cout
                << "qwen3_vocab_size: " << cfg.vocab_size << "\n"
                << "qwen3_context_length: " << cfg.context_length << "\n"
                << "qwen3_embedding_length: " << cfg.embedding_length << "\n"
                << "qwen3_feed_forward_length: " << cfg.feed_forward_length << "\n"
                << "qwen3_block_count: " << cfg.block_count << "\n"
                << "qwen3_head_count: " << cfg.head_count << "\n"
                << "qwen3_head_count_kv: " << cfg.head_count_kv << "\n"
                << "qwen3_key_length: " << cfg.key_length << "\n"
                << "qwen3_value_length: " << cfg.value_length << "\n"
                << "qwen3_rope_dimension_count: " << cfg.rope_dimension_count << "\n"
                << "qwen3_rms_norm_eps: " << cfg.rms_norm_eps << "\n"
                << "qwen3_rope_freq_base: " << cfg.rope_freq_base << "\n"
                << "text_frontier_tensors:\n";
            for (const higgs_audio::TensorInfo & tensor : cfg.frontier_tensors) {
                print_tensor(tensor);
            }
            return 0;
        }

        if (inspect_tokenizer) {
            higgs_audio::validate_tokenizer_contract(options.model_path);
            const higgs_audio::TokenizerInfo tok = higgs_audio::inspect_tokenizer(options.model_path);
            std::cout
                << "tokenizer_model: " << tok.model << "\n"
                << "tokenizer_vocab_size: " << tok.vocab_size << "\n"
                << "tokenizer_token_count: " << tok.token_count << "\n"
                << "tokenizer_token_type_count: " << tok.token_type_count << "\n"
                << "tokenizer_merge_count: " << tok.merge_count << "\n"
                << "tokenizer_added_token_count: " << tok.added_token_count << "\n"
                << "tokenizer_huggingface_json_bytes: " << tok.huggingface_json_bytes << "\n"
                << "tokenizer_chat_template_bytes: " << tok.chat_template_bytes << "\n"
                << "tokenizer_eos_token_id: " << tok.eos_token_id << "\n"
                << "tokenizer_padding_token_id: " << tok.padding_token_id << "\n"
                << "special_tts: " << tok.special.tts << "\n"
                << "special_text: " << tok.special.text << "\n"
                << "special_audio: " << tok.special.audio << "\n"
                << "special_ref_text: " << tok.special.ref_text << "\n"
                << "special_ref_audio: " << tok.special.ref_audio << "\n";
            return 0;
        }

        if (inspect_reference_encode) {
            higgs_audio::validate_reference_encode_contract(options.model_path);
            const higgs_audio::ReferenceEncodeInfo info = higgs_audio::inspect_reference_encode(options.model_path);
            std::cout
                << "reference_sample_rate: " << info.sample_rate << "\n"
                << "reference_semantic_sample_rate: " << info.semantic_sample_rate << "\n"
                << "reference_downsample_factor: " << info.downsample_factor << "\n"
                << "reference_semantic_downsample_factor: " << info.semantic_downsample_factor << "\n"
                << "reference_semantic_hidden_size: " << info.semantic_hidden_size << "\n"
                << "reference_semantic_layers: " << info.semantic_layer_count << "\n"
                << "reference_semantic_heads: " << info.semantic_head_count << "\n"
                << "reference_semantic_intermediate_size: " << info.semantic_intermediate_size << "\n"
                << "reference_acoustic_tensors: " << info.acoustic_tensor_count << "\n"
                << "reference_semantic_model_tensors: " << info.semantic_model_tensor_count << "\n"
                << "reference_semantic_encoder_tensors: " << info.semantic_encoder_tensor_count << "\n"
                << "reference_fc_tensors: " << info.fc_tensor_count << "\n"
                << "reference_fc1_tensors: " << info.fc1_tensor_count << "\n"
                << "reference_fc2_tensors: " << info.fc2_tensor_count << "\n"
                << "reference_quantizer_tensors: " << info.quantizer_tensor_count << "\n"
                << "reference_frontier_tensors:\n";
            for (const higgs_audio::TensorInfo & tensor : info.frontier_tensors) {
                print_tensor(tensor);
            }
            return 0;
        }

        if (!inspect_tokenizer_artifact_path.empty()) {
            higgs_audio::validate_tokenizer_artifact(inspect_tokenizer_artifact_path);
            const higgs_audio::TokenizerArtifactInfo info = higgs_audio::inspect_tokenizer_artifact(inspect_tokenizer_artifact_path);
            std::cout
                << "tokenizer_artifact_file_bytes: " << info.file_bytes << "\n"
                << "tokenizer_artifact_token_count: " << info.token_count << "\n"
                << "tokenizer_artifact_merge_count: " << info.merge_count << "\n"
                << "tokenizer_artifact_added_token_count: " << info.added_token_count << "\n"
                << "tokenizer_artifact_chat_template_bytes: " << info.chat_template_bytes << "\n";
            return 0;
        }

        if (inspect || validate) {
            if (validate) {
                higgs_audio::validate_model_contract(options.model_path);
            }
            const higgs_audio::ModelInfo info = higgs_audio::inspect_model(options.model_path);
            std::cout
                << "model: " << options.model_path << "\n"
                << "tensors: " << info.tensor_count << "\n"
                << "metadata: " << info.kv_count << "\n"
                << "audio_num_codebooks: " << info.audio_num_codebooks << "\n"
                << "audio_vocab_size: " << info.audio_vocab_size << "\n"
                << "required_tensors:\n";
            for (const higgs_audio::TensorInfo & tensor : info.required_tensors) {
                print_tensor(tensor);
            }
            return 0;
        }

        if (options.text.empty()) {
            throw std::invalid_argument("--text is required");
        }
        if (out_path.empty()) {
            throw std::invalid_argument("--out is required");
        }

        higgs_audio::HiggsPipeline pipeline(options.model_path, options.backend);
        const higgs_audio::GenerateResult result = pipeline.generate(options);
        if (ends_with(out_path, ".mp3")) {
            higgs_audio::write_mp3_mono(out_path, result.waveform, result.sample_rate);
        } else {
            higgs_audio::write_wav_mono_16(out_path, result.waveform, result.sample_rate);
        }
        std::cout << "wrote " << out_path << "\n";
        return 0;
    } catch (const std::exception & err) {
        std::cerr << "higgs-audio-cli: " << err.what() << "\n";
        return 1;
    }
}
