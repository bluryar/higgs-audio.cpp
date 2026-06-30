#include "higgs_audio.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "gguf.h"
extern "C" {
#include "layer3.h"
}
#include "tokenizers_cpp/tokenizer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <deque>
#include <exception>
#include <fstream>
#include <future>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <sys/stat.h>
#include <thread>
#include <unordered_map>
#include <utility>

namespace higgs_audio {
namespace {

struct GgufDeleter {
    void operator()(gguf_context * ctx) const {
        if (ctx != nullptr) {
            gguf_free(ctx);
        }
    }
};

using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;

struct GgmlDeleter {
    void operator()(ggml_context * ctx) const {
        if (ctx != nullptr) {
            ggml_free(ctx);
        }
    }
};

using GgmlPtr = std::unique_ptr<ggml_context, GgmlDeleter>;

struct BackendDeleter {
    void operator()(ggml_backend * backend) const {
        if (backend != nullptr) {
            ggml_backend_free(backend);
        }
    }
};

struct BackendBufferDeleter {
    void operator()(ggml_backend_buffer * buffer) const {
        if (buffer != nullptr) {
            ggml_backend_buffer_free(buffer);
        }
    }
};

struct GgmlGallocrDeleter {
    void operator()(ggml_gallocr * galloc) const {
        if (galloc != nullptr) {
            ggml_gallocr_free(galloc);
        }
    }
};

struct BackendSchedDeleter {
    void operator()(ggml_backend_sched * sched) const {
        if (sched != nullptr) {
            ggml_backend_sched_free(sched);
        }
    }
};

using BackendPtr = std::unique_ptr<ggml_backend, BackendDeleter>;
using BackendBufferPtr = std::unique_ptr<ggml_backend_buffer, BackendBufferDeleter>;
using GgmlGallocrPtr = std::unique_ptr<ggml_gallocr, GgmlGallocrDeleter>;
using BackendSchedPtr = std::unique_ptr<ggml_backend_sched, BackendSchedDeleter>;

thread_local BackendKind g_backend_kind = BackendKind::cpu;
thread_local int g_reference_debug_step = -1;

int reference_kv_window_refresh();
int reference_kv_head_batch();
bool reference_kv_window_contextual_input();
bool reference_kv_gallocr_disabled();
bool reference_kv_sched_enabled();
bool reference_kv_stage_profile_enabled();
bool reference_kv_fused_ffn_enabled();
bool reference_kv_prefill_via_window_enabled();

BackendPtr make_backend() {
    if (g_backend_kind == BackendKind::cpu) {
        BackendPtr backend(ggml_backend_cpu_init());
        if (!backend) {
            throw std::runtime_error("failed to initialize GGML CPU backend");
        }
        return backend;
    }
#ifdef GGML_USE_CUDA
    BackendPtr backend(ggml_backend_cuda_init(0));
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CUDA backend 0");
    }
    return backend;
#else
    throw std::runtime_error("CUDA backend requested, but this binary was built without -DGGML_CUDA=ON");
#endif
}

struct BackendScope {
    explicit BackendScope(BackendKind kind) : previous(g_backend_kind) {
        g_backend_kind = kind;
    }
    ~BackendScope() {
        g_backend_kind = previous;
    }
    BackendKind previous;
};

void throw_if_cancelled(const GenerateOptions & options) {
    if (options.cancel != nullptr && *options.cancel != 0) {
        throw std::runtime_error("request cancelled");
    }
    if (options.deadline != std::chrono::steady_clock::time_point{} && std::chrono::steady_clock::now() >= options.deadline) {
        throw std::runtime_error("request timed out");
    }
}

struct GgufWithTensors {
    GgufPtr gguf;
    GgmlPtr ggml;
};

GgufWithTensors open_gguf_with_tensors(const std::string & path);
bool load_named_tensors_from_gguf(const std::string & path, ggml_context * ggml_ctx, const gguf_context * gguf_ctx, const std::vector<std::string> & names);
std::vector<std::string> all_block_weight_names(const TextConfig & cfg);

struct CudaExecutorRunStats {
    long long wait_ms = 0;
    long long run_ms = 0;
    int queue_depth = 0;
    std::exception_ptr error;
};

class CudaExecutor {
public:
    CudaExecutor() : worker_([this] { run(); }) {}
    ~CudaExecutor() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_ = true;
        }
        cv_.notify_one();
        worker_.join();
    }

    CudaExecutorRunStats run_sync(std::function<void()> fn) {
        Task task;
        task.fn = std::move(fn);
        task.enqueued_at = std::chrono::steady_clock::now();
        auto done = task.done.get_future();
        CudaExecutorRunStats stats;
        {
            std::lock_guard<std::mutex> lock(mu_);
            stats.queue_depth = static_cast<int>(queue_.size()) + (active_ ? 1 : 0);
            task.stats = &stats;
            queue_.push_back(std::move(task));
        }
        cv_.notify_one();
        done.get();
        if (stats.error) {
            std::rethrow_exception(stats.error);
        }
        return stats;
    }

private:
    struct Task {
        std::function<void()> fn;
        std::chrono::steady_clock::time_point enqueued_at;
        CudaExecutorRunStats * stats = nullptr;
        std::promise<void> done;
    };

    void run() {
        for (;;) {
            Task task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&] { return stop_ || !queue_.empty(); });
                if (queue_.empty()) {
                    return;
                }
                task = std::move(queue_.front());
                queue_.pop_front();
                active_ = true;
            }
            const auto start = std::chrono::steady_clock::now();
            task.stats->wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(start - task.enqueued_at).count();
            try {
                task.fn();
            } catch (...) {
                task.stats->error = std::current_exception();
            }
            task.stats->run_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            {
                std::lock_guard<std::mutex> lock(mu_);
                active_ = false;
            }
            task.done.set_value();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<Task> queue_;
    std::thread worker_;
    bool stop_ = false;
    bool active_ = false;
};

CudaExecutor & cuda_executor() {
    static CudaExecutor executor;
    return executor;
}

struct ReferenceArRuntimeStats {
    long long cuda_executor_wait_ms = 0;
    long long cuda_executor_run_ms = 0;
    int cuda_executor_queue_depth = 0;
    long long audio_head_wall_ms = 0;
    int audio_head_batch_calls = 0;
    int audio_head_fallback_calls = 0;
    int audio_head_batch_items = 0;
};

void add_cuda_executor_stats(ReferenceArRuntimeStats * stats, const CudaExecutorRunStats & run) {
    if (stats == nullptr) {
        return;
    }
    stats->cuda_executor_wait_ms += run.wait_ms;
    stats->cuda_executor_run_ms += run.run_ms;
    stats->cuda_executor_queue_depth = std::max(stats->cuda_executor_queue_depth, run.queue_depth);
}

bool server_audio_head_batch_enabled() {
    return std::getenv("HIGGS_SERVER_AUDIO_HEAD_BATCH") != nullptr && std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr;
}

int server_audio_head_batch_wait_ms() {
    const char * value = std::getenv("HIGGS_SERVER_AUDIO_HEAD_BATCH_WAIT_MS");
    if (value == nullptr || *value == '\0') {
        return 2;
    }
    return std::max(0, std::atoi(value));
}

int & active_reference_ar_requests() {
    static int value = 0;
    return value;
}

std::mutex & active_reference_ar_mutex() {
    static std::mutex mu;
    return mu;
}

struct ScopedActiveReferenceArRequest {
    explicit ScopedActiveReferenceArRequest(bool enabled) : enabled(enabled) {
        if (enabled) {
            std::lock_guard<std::mutex> lock(active_reference_ar_mutex());
            ++active_reference_ar_requests();
        }
    }
    ~ScopedActiveReferenceArRequest() {
        if (enabled) {
            std::lock_guard<std::mutex> lock(active_reference_ar_mutex());
            --active_reference_ar_requests();
        }
    }
    bool enabled = false;
};

bool multiple_reference_ar_requests_active() {
    std::lock_guard<std::mutex> lock(active_reference_ar_mutex());
    return active_reference_ar_requests() > 1;
}

struct TensorBlock {
    std::vector<float> values;
    int64_t frames = 0;
    int64_t channels = 0;
};

struct KvDecodeResult {
    std::vector<float> hidden;
    std::vector<float> logits_vc;
};

struct BackendKvCache {
    GgmlPtr ctx;
    BackendBufferPtr buffer;
    ggml_backend * backend = nullptr;
    gguf_context * weights_gguf = nullptr;
    ggml_context * weights_ggml = nullptr;
    std::unordered_map<std::string, bool> * loaded_weights = nullptr;
    std::mutex * loaded_weights_mutex = nullptr;
    TextConfig cfg;
    ggml_tensor * key = nullptr;
    ggml_tensor * value = nullptr;
    int64_t layers = 0;
    int64_t head_dim = 0;
    int64_t kv_heads = 0;
    int64_t capacity = 0;
    int64_t used = 0;
    GgmlGallocrPtr decode_galloc;
    GgmlGallocrPtr window_galloc;
    BackendPtr sched_cpu_backend;
    BackendSchedPtr decode_sched;
    BackendSchedPtr window_sched;
    int64_t window_sched_reserved_tokens = 0;
    std::vector<float> rope_cos_values;
    std::vector<float> rope_sin_values;
    ggml_tensor * rope_cos = nullptr;
    ggml_tensor * rope_sin = nullptr;
    GgmlPtr fused_ffn_ctx;
    BackendBufferPtr fused_ffn_buffer;
    std::vector<ggml_tensor *> fused_ffn_gate_up;
    bool fused_ffn_ready = false;
};

struct BackendRuntime {
    BackendPtr backend;
    GgufPtr weights_gguf;
    GgmlPtr weights_ggml;
    BackendBufferPtr weights_buffer;
    std::unordered_map<std::string, bool> loaded_weights;
    std::mutex loaded_weights_mutex;
    std::mutex kv_cache_slots_mutex;
    std::unique_ptr<BackendKvCache> kv_cache;
    std::vector<std::unique_ptr<BackendKvCache>> kv_cache_slots;
    std::vector<bool> kv_cache_slot_busy;
    const TextConfig * cfg = nullptr;
};

struct ScopedBackendKvCacheSlot {
    BackendRuntime * runtime = nullptr;
    int slot = -1;
    BackendKvCache * cache = nullptr;

    ScopedBackendKvCacheSlot() = default;
    ScopedBackendKvCacheSlot(BackendRuntime & runtime_in, int slot_in, BackendKvCache & cache_in) : runtime(&runtime_in), slot(slot_in), cache(&cache_in) {}
    ScopedBackendKvCacheSlot(const ScopedBackendKvCacheSlot &) = delete;
    ScopedBackendKvCacheSlot & operator=(const ScopedBackendKvCacheSlot &) = delete;
    ScopedBackendKvCacheSlot(ScopedBackendKvCacheSlot && other) noexcept : runtime(other.runtime), slot(other.slot), cache(other.cache) {
        other.runtime = nullptr;
        other.slot = -1;
        other.cache = nullptr;
    }
    ScopedBackendKvCacheSlot & operator=(ScopedBackendKvCacheSlot && other) noexcept {
        if (this != &other) {
            release();
            runtime = other.runtime;
            slot = other.slot;
            cache = other.cache;
            other.runtime = nullptr;
            other.slot = -1;
            other.cache = nullptr;
        }
        return *this;
    }
    ~ScopedBackendKvCacheSlot() {
        release();
    }

    void release();
};

struct PipelineRuntime {
    explicit PipelineRuntime(std::string path) : model_path(std::move(path)), cfg(inspect_text_config(model_path)) {}

    BackendRuntime & get(BackendKind kind) {
        std::unique_ptr<BackendRuntime> & slot = kind == BackendKind::cpu ? cpu : cuda;
        if (!slot) {
            BackendScope scope(kind);
            slot = std::make_unique<BackendRuntime>();
            slot->cfg = &cfg;
            slot->backend = make_backend();
            GgufWithTensors loaded = open_gguf_with_tensors(model_path);
            slot->weights_gguf = std::move(loaded.gguf);
            slot->weights_ggml = std::move(loaded.ggml);
            slot->weights_buffer.reset(ggml_backend_alloc_ctx_tensors(slot->weights_ggml.get(), slot->backend.get()));
            if (!slot->weights_buffer) {
                throw std::runtime_error("failed to allocate pipeline runtime weight tensors");
            }
            std::vector<std::string> names = all_block_weight_names(cfg);
            names.push_back("output_norm.weight");
            names.push_back("a.output");
            if (!load_named_tensors_from_gguf(model_path, slot->weights_ggml.get(), slot->weights_gguf.get(), names)) {
                throw std::runtime_error("failed to preload pipeline runtime text tensors");
            }
            for (const std::string & name : names) {
                slot->loaded_weights[name] = true;
            }
        }
        return *slot;
    }

    std::string model_path;
    TextConfig cfg;
    std::unique_ptr<BackendRuntime> cpu;
    std::unique_ptr<BackendRuntime> cuda;
};

ggml_tensor * repeat_kv_3d(ggml_context * ctx, ggml_tensor * tensor, int64_t repeats) {
    if (repeats == 1) {
        return tensor;
    }
    ggml_tensor * viewed = nullptr;
    if (ggml_is_contiguous(tensor)) {
        viewed = ggml_reshape_4d(ctx, tensor, tensor->ne[0], 1, tensor->ne[1], tensor->ne[2]);
    } else {
        viewed = ggml_view_4d(ctx, tensor, tensor->ne[0], 1, tensor->ne[1], tensor->ne[2], tensor->nb[1], tensor->nb[1], tensor->nb[2], 0);
    }
    ggml_tensor * repeated = ggml_repeat_4d(ctx, viewed, viewed->ne[0], repeats, viewed->ne[2], viewed->ne[3]);
    if (!ggml_is_contiguous(repeated)) {
        repeated = ggml_cont_4d(ctx, repeated, repeated->ne[0], repeated->ne[1], repeated->ne[2], repeated->ne[3]);
    }
    return ggml_reshape_3d(ctx, repeated, repeated->ne[0], repeated->ne[1] * repeated->ne[2], repeated->ne[3]);
}

ggml_tensor * as_4d(ggml_context * ctx, ggml_tensor * tensor) {
    if (ggml_is_contiguous(tensor)) {
        return ggml_reshape_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], 1);
    }
    return ggml_view_4d(ctx, tensor, tensor->ne[0], tensor->ne[1], tensor->ne[2], 1, tensor->nb[1], tensor->nb[2], tensor->nb[2], 0);
}

ggml_tensor * mul_mat_f32(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * out = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(out, GGML_PREC_F32);
    return out;
}

float bf16_rne_to_f32(float value) {
    return ggml_bf16_to_fp32(ggml_fp32_to_bf16(value));
}

std::vector<float> make_bf16_rope_table(const TextConfig & cfg, bool sine) {
    const int64_t half = cfg.key_length / 2;
    std::vector<float> table(static_cast<size_t>(half * cfg.context_length));
    for (int64_t pos = 0; pos < cfg.context_length; ++pos) {
        for (int64_t i = 0; i < half; ++i) {
            const float dim = static_cast<float>(i * 2);
            const float freq = 1.0f / std::pow(cfg.rope_freq_base, dim / static_cast<float>(cfg.key_length));
            const float angle = static_cast<float>(pos) * freq;
            table[static_cast<size_t>(pos * half + i)] = bf16_rne_to_f32(sine ? std::sin(angle) : std::cos(angle));
        }
    }
    return table;
}

ggml_tensor * neox_rope_bf16_cache(ggml_context * ctx, ggml_tensor * x, ggml_tensor * positions, ggml_tensor * cos_table, ggml_tensor * sin_table) {
    const int64_t head_dim = x->ne[0];
    const int64_t n_head = x->ne[1];
    const int64_t n_tokens = x->ne[2];
    const int64_t half = head_dim / 2;
    const size_t item = ggml_element_size(x);
    ggml_tensor * x0 = ggml_view_3d(ctx, x, half, n_head, n_tokens, head_dim * item, head_dim * n_head * item, 0);
    ggml_tensor * x1 = ggml_view_3d(ctx, x, half, n_head, n_tokens, head_dim * item, head_dim * n_head * item, static_cast<size_t>(half) * item);
    ggml_tensor * cos = ggml_get_rows(ctx, cos_table, positions);
    ggml_tensor * sin = ggml_get_rows(ctx, sin_table, positions);
    cos = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, cos, half, 1, n_tokens), half, n_head, n_tokens, 1);
    sin = ggml_repeat_4d(ctx, ggml_reshape_3d(ctx, sin, half, 1, n_tokens), half, n_head, n_tokens, 1);
    cos = ggml_reshape_3d(ctx, cos, half, n_head, n_tokens);
    sin = ggml_reshape_3d(ctx, sin, half, n_head, n_tokens);
    ggml_tensor * y0 = ggml_sub(ctx, ggml_mul(ctx, x0, cos), ggml_mul(ctx, x1, sin));
    ggml_tensor * y1 = ggml_add(ctx, ggml_mul(ctx, x0, sin), ggml_mul(ctx, x1, cos));
    return ggml_concat(ctx, y0, y1, 0);
}

bool profile_enabled() {
    const char * value = std::getenv("HIGGS_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

bool codec_stage_profile_enabled() {
    const char * value = std::getenv("HIGGS_CODEC_STAGE_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

struct ProfileAggregate {
    long long count = 0;
    long long total_ms = 0;
    long long total_us = 0;
    long long min_ms = std::numeric_limits<long long>::max();
    long long max_ms = 0;
    long long min_us = std::numeric_limits<long long>::max();
    long long max_us = 0;
};

std::map<std::string, ProfileAggregate> & profile_aggregates() {
    static std::map<std::string, ProfileAggregate> aggregates;
    return aggregates;
}

void profile_record_elapsed(const char * label, long long elapsed_ms) {
    ProfileAggregate & aggregate = profile_aggregates()[label];
    aggregate.count += 1;
    aggregate.total_ms += elapsed_ms;
    aggregate.total_us += elapsed_ms * 1000;
    aggregate.min_ms = std::min(aggregate.min_ms, elapsed_ms);
    aggregate.max_ms = std::max(aggregate.max_ms, elapsed_ms);
    aggregate.min_us = std::min(aggregate.min_us, elapsed_ms * 1000);
    aggregate.max_us = std::max(aggregate.max_us, elapsed_ms * 1000);
}

void profile_record_elapsed_us(const char * label, long long elapsed_us) {
    ProfileAggregate & aggregate = profile_aggregates()[label];
    aggregate.count += 1;
    aggregate.total_us += elapsed_us;
    aggregate.total_ms += elapsed_us / 1000;
    aggregate.min_us = std::min(aggregate.min_us, elapsed_us);
    aggregate.max_us = std::max(aggregate.max_us, elapsed_us);
    aggregate.min_ms = std::min(aggregate.min_ms, elapsed_us / 1000);
    aggregate.max_ms = std::max(aggregate.max_ms, elapsed_us / 1000);
}

void profile_print_aggregates() {
    if (!profile_enabled()) {
        return;
    }
    long long stage_total_ms = 0;
    long long stage_ffn_ms = 0;
    long long stage_attention_ms = 0;
    long long stage_logits_ms = 0;
    for (const auto & item : profile_aggregates()) {
        const ProfileAggregate & aggregate = item.second;
        const std::string & label = item.first;
        if (label.rfind("reference_kv_ffn_", 0) == 0 || label.rfind("reference_kv_stage_", 0) == 0 || label.rfind("reference_kv_decode_", 0) == 0) {
            stage_total_ms += aggregate.total_ms;
            if (label.find("_ffn_") != std::string::npos) {
                stage_ffn_ms += aggregate.total_ms;
            } else if (
                label.find("_stage_attn") != std::string::npos ||
                label.find("_stage_kv_") != std::string::npos ||
                label.find("_stage_qk_") != std::string::npos ||
                label.find("_stage_softmax") != std::string::npos ||
                label.find("_stage_v_matmul") != std::string::npos ||
                label.find("_stage_all_attention") != std::string::npos ||
                label.find("_decode_attn") != std::string::npos ||
                label.find("_decode_kv_") != std::string::npos ||
                label.find("_decode_qk_") != std::string::npos ||
                label.find("_decode_softmax") != std::string::npos ||
                label.find("_decode_v_matmul") != std::string::npos ||
                label.find("_decode_attention") != std::string::npos) {
                stage_attention_ms += aggregate.total_ms;
            } else if (label.find("_stage_logits_head") != std::string::npos) {
                stage_logits_ms += aggregate.total_ms;
            }
        }
        if (aggregate.count <= 1) {
            continue;
        }
        std::fprintf(stderr,
                     "higgs_profile_summary %s count=%lld total_ms=%lld avg_ms=%lld min_ms=%lld max_ms=%lld total_us=%lld avg_us=%lld min_us=%lld max_us=%lld\n",
                     item.first.c_str(),
                     aggregate.count,
                     aggregate.total_ms,
                     aggregate.total_ms / aggregate.count,
                     aggregate.min_ms,
                     aggregate.max_ms,
                     aggregate.total_us,
                     aggregate.total_us / aggregate.count,
                     aggregate.min_us == std::numeric_limits<long long>::max() ? 0 : aggregate.min_us,
                     aggregate.max_us);
    }
    if (stage_total_ms > 0) {
        const long long stage_misc_ms = stage_total_ms - stage_ffn_ms - stage_attention_ms - stage_logits_ms;
        const auto pct = [stage_total_ms](long long value) -> double {
            return 100.0 * static_cast<double>(value) / static_cast<double>(stage_total_ms);
        };
        std::fprintf(stderr,
                     "higgs_profile_stage_summary total_ms=%lld ffn_ms=%lld ffn_pct=%.2f attention_ms=%lld attention_pct=%.2f logits_ms=%lld logits_pct=%.2f misc_ms=%lld misc_pct=%.2f\n",
                     stage_total_ms,
                     stage_ffn_ms,
                     pct(stage_ffn_ms),
                     stage_attention_ms,
                     pct(stage_attention_ms),
                     stage_logits_ms,
                     pct(stage_logits_ms),
                     stage_misc_ms,
                     pct(stage_misc_ms));
    }
}

struct ScopedProfileTimer {
    explicit ScopedProfileTimer(const char * label) : label(label), start(std::chrono::steady_clock::now()), enabled(profile_enabled()) {}
    ~ScopedProfileTimer() {
        if (!enabled) {
            return;
        }
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        profile_record_elapsed(label, static_cast<long long>(elapsed));
        std::fprintf(stderr, "higgs_profile %s ms=%lld\n", label, static_cast<long long>(elapsed));
    }
    const char * label;
    std::chrono::steady_clock::time_point start;
    bool enabled;
};

struct ReferenceKvStageProfileState {
    std::chrono::steady_clock::time_point start;
};

bool reference_kv_stage_profile_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    const char * prefix = "kvstage:";
    const size_t prefix_len = std::strlen(prefix);
    if (tensor == nullptr || std::strncmp(tensor->name, prefix, prefix_len) != 0) {
        return false;
    }
    auto * state = static_cast<ReferenceKvStageProfileState *>(user_data);
    if (ask) {
        return true;
    }
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - state->start).count();
    const char * label = tensor->name + prefix_len;
    profile_record_elapsed_us(label, static_cast<long long>(elapsed));
    std::fprintf(stderr, "higgs_profile %s us=%lld\n", label, static_cast<long long>(elapsed));
    state->start = now;
    return true;
}

void profile_code_matrix(const char * label, const CodeMatrix & codes) {
    if (!profile_enabled()) {
        return;
    }
    std::vector<int32_t> values = codes.data;
    std::sort(values.begin(), values.end());
    const auto unique_end = std::unique(values.begin(), values.end());
    const int unique = static_cast<int>(unique_end - values.begin());
    std::fprintf(stderr, "higgs_profile %s frames=%d codebooks=%d unique=%d\n", label, codes.frames, codes.codebooks, unique);
}

std::string trace_json_path() {
    const char * value = std::getenv("HIGGS_TRACE_JSON");
    return value == nullptr ? std::string{} : std::string(value);
}

std::string codes_json_path() {
    const char * value = std::getenv("HIGGS_CODES_JSON");
    return value == nullptr ? std::string{} : std::string(value);
}

bool reference_full_diagnostic_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_FULL");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_full_fallback_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_FULL_FALLBACK");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_fast_fallback_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_FAST_FALLBACK");
    return value == nullptr || std::string(value) == "1";
}

bool reference_backend_kv_diagnostic_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_BACKEND_KV");
    if (value == nullptr || std::string(value) != "1") {
        return false;
    }
    if (reference_kv_full_fallback_enabled() || reference_kv_fast_fallback_enabled()) {
        return true;
    }
    if (reference_kv_window_refresh() > 0) {
        return true;
    }
    if (reference_kv_prefill_via_window_enabled()) {
        return true;
    }
    const char * trace_only = std::getenv("HIGGS_TRACE_ONLY");
    if (trace_only == nullptr || std::string(trace_only) != "1" || trace_json_path().empty()) {
        throw std::runtime_error("HIGGS_REFERENCE_BACKEND_KV is diagnostic-only; set HIGGS_TRACE_ONLY=1 and HIGGS_TRACE_JSON to compare traces");
    }
    return true;
}

float reference_kv_fallback_margin() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_FALLBACK_MARGIN");
    return value == nullptr ? 0.006f : std::stof(value);
}

int reference_kv_window_refresh() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_WINDOW_REFRESH");
    return value == nullptr ? 0 : std::stoi(value);
}

int reference_kv_head_batch() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_HEAD_BATCH");
    return value == nullptr ? 1 : std::stoi(value);
}

bool reference_kv_window_contextual_input() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_WINDOW_CONTEXTUAL_INPUT");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_gallocr_disabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_DISABLE_GALLOCR");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_fused_audio_head_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_FUSED_AUDIO_HEAD");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_skip_non_cb6_fallback_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_SKIP_NON_CB6_FALLBACK");
    return value == nullptr || std::string(value) == "1";
}

bool reference_kv_sched_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_SCHED");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_stage_profile_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_STAGE_PROFILE");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_fused_ffn_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_FUSED_FFN");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_prefill_via_window_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_PREFILL_VIA_WINDOW");
    return value == nullptr || std::string(value) == "1";
}

bool reference_kv_compare_window1_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_COMPARE_WINDOW1");
    return value != nullptr && std::string(value) == "1";
}

bool reference_kv_compare_full_prefix_enabled() {
    const char * value = std::getenv("HIGGS_REFERENCE_KV_COMPARE_FULL_PREFIX");
    return value != nullptr && std::string(value) == "1";
}

bool trace_only_enabled() {
    const char * value = std::getenv("HIGGS_TRACE_ONLY");
    return value != nullptr && std::string(value) == "1";
}

std::string attention_dump_path() {
    const char * value = std::getenv("HIGGS_ATTENTION_DUMP");
    return value != nullptr ? std::string(value) : std::string();
}

std::string attention_normed_input_path() {
    const char * value = std::getenv("HIGGS_ATTENTION_NORMED_INPUT");
    return value != nullptr ? std::string(value) : std::string();
}

bool attention_q_only_enabled() {
    const char * value = std::getenv("HIGGS_ATTENTION_Q_ONLY");
    return value != nullptr && std::string(value) == "1";
}

std::string node_dump_dir() {
    const char * value = std::getenv("HIGGS_NODE_DUMP_DIR");
    return value != nullptr ? std::string(value) : std::string();
}

bool ar_node_dump_only() {
    const char * value = std::getenv("HIGGS_AR_NODE_DUMP_ONLY");
    return value != nullptr && std::string(value) == "1";
}

std::string reference_step_node_name(const std::string & name) {
    if (g_reference_debug_step < 0) {
        return name;
    }
    return "cpp_step" + std::to_string(g_reference_debug_step) + "_" + name;
}

struct ReferenceDebugStepScope {
    explicit ReferenceDebugStepScope(int step) : previous(g_reference_debug_step) {
        g_reference_debug_step = step;
    }
    ~ReferenceDebugStepScope() {
        g_reference_debug_step = previous;
    }
    int previous;
};


void dump_node_values(const std::string & name, const std::vector<float> & values) {
    const std::string dir = node_dump_dir();
    if (dir.empty()) {
        return;
    }
    if (ar_node_dump_only() && name.rfind("cpp_reference_", 0) == 0) {
        return;
    }
    std::ofstream out(dir + "/" + name + ".f32", std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open node dump: " + dir + "/" + name + ".f32");
    }
    out.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
}

struct TraceStep {
    std::vector<int32_t> top1;
    std::vector<std::vector<int32_t>> top5;
    std::vector<std::vector<float>> top5_values;
    std::vector<int32_t> codes;
};

struct TraceNodeStat {
    std::vector<int64_t> shape;
    double last_mean = 0.0;
    double last_abs_mean = 0.0;
    double last_max_abs = 0.0;
    std::vector<float> last_first8;
};

std::map<std::string, TraceNodeStat> g_trace_node_stats;

TraceNodeStat make_last_token_stat(const std::vector<float> & values, int64_t features, int64_t tokens) {
    TraceNodeStat stat;
    stat.shape = {features, tokens};
    if (features <= 0 || tokens <= 0 || values.size() != static_cast<size_t>(features * tokens)) {
        return stat;
    }
    const size_t offset = static_cast<size_t>(tokens - 1) * static_cast<size_t>(features);
    double sum = 0.0;
    double abs_sum = 0.0;
    double max_abs = 0.0;
    const int64_t first = std::min<int64_t>(features, 8);
    stat.last_first8.reserve(static_cast<size_t>(first));
    for (int64_t i = 0; i < features; ++i) {
        const double v = values[offset + static_cast<size_t>(i)];
        sum += v;
        abs_sum += std::abs(v);
        max_abs = std::max(max_abs, std::abs(v));
        if (i < first) {
            stat.last_first8.push_back(static_cast<float>(v));
        }
    }
    stat.last_mean = sum / static_cast<double>(features);
    stat.last_abs_mean = abs_sum / static_cast<double>(features);
    stat.last_max_abs = max_abs;
    return stat;
}

TraceNodeStat make_abs_diff_stat(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("trace diff vectors must have the same size");
    }
    std::vector<float> diffs(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        diffs[i] = std::fabs(a[i] - b[i]);
    }
    return make_last_token_stat(diffs, static_cast<int64_t>(diffs.size()), 1);
}

std::vector<int32_t> topk_indices(const std::vector<float> & logits_vc, int vocab, int codebooks, int codebook, int k) {
    std::vector<int32_t> order(static_cast<size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + std::min(k, vocab), order.end(), [&](int a, int b) {
        return logits_vc[static_cast<size_t>(a) * static_cast<size_t>(codebooks) + static_cast<size_t>(codebook)] >
               logits_vc[static_cast<size_t>(b) * static_cast<size_t>(codebooks) + static_cast<size_t>(codebook)];
    });
    order.resize(static_cast<size_t>(std::min(k, vocab)));
    return order;
}

std::vector<float> masked_logits_for_trace(std::vector<float> logits_vc, bool stop_on_eoc) {
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int v = k_codebook_data_size; v < k_eoc_id; ++v) {
            logits_vc[static_cast<size_t>(v) * static_cast<size_t>(k_num_codebooks) + static_cast<size_t>(c)] = -std::numeric_limits<float>::infinity();
        }
        if (!stop_on_eoc) {
            logits_vc[static_cast<size_t>(k_eoc_id) * static_cast<size_t>(k_num_codebooks) + static_cast<size_t>(c)] = -std::numeric_limits<float>::infinity();
        }
    }
    return logits_vc;
}

TraceStep trace_step_from_logits(const std::vector<float> & logits_vc, const std::vector<int32_t> & codes, bool stop_on_eoc) {
    TraceStep step;
    step.codes = codes;
    const std::vector<float> masked = masked_logits_for_trace(logits_vc, stop_on_eoc);
    step.top1.resize(static_cast<size_t>(k_num_codebooks));
    step.top5.resize(static_cast<size_t>(k_num_codebooks));
    step.top5_values.resize(static_cast<size_t>(k_num_codebooks));
    for (int c = 0; c < k_num_codebooks; ++c) {
        step.top5[static_cast<size_t>(c)] = topk_indices(masked, k_codebook_vocab_size, k_num_codebooks, c, 5);
        step.top1[static_cast<size_t>(c)] = step.top5[static_cast<size_t>(c)].front();
        for (int32_t v : step.top5[static_cast<size_t>(c)]) {
            step.top5_values[static_cast<size_t>(c)].push_back(masked[static_cast<size_t>(v) * static_cast<size_t>(k_num_codebooks) + static_cast<size_t>(c)]);
        }
    }
    return step;
}

CodeMatrix delayed_rows_to_matrix(const std::vector<std::vector<int32_t>> & rows) {
    if (rows.empty()) {
        return CodeMatrix{};
    }
    CodeMatrix out;
    out.codebooks = static_cast<int>(rows.front().size());
    out.frames = static_cast<int>(rows.size());
    out.data.resize(static_cast<size_t>(out.frames * out.codebooks));
    for (int f = 0; f < out.frames; ++f) {
        if (rows[static_cast<size_t>(f)].size() != static_cast<size_t>(out.codebooks)) {
            throw std::runtime_error("delayed code row shape mismatch");
        }
        for (int c = 0; c < out.codebooks; ++c) {
            out.at(c, f) = rows[static_cast<size_t>(f)][static_cast<size_t>(c)];
        }
    }
    return out;
}

void write_trace_json(const std::string & path, const std::vector<int32_t> & prompt, const CodeMatrix & ref_codes, const std::vector<std::vector<int32_t>> & rows, const std::vector<TraceStep> & steps, const CodeMatrix & finalized) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open trace json: " + path);
    }
    auto write_vec = [&out](const std::vector<int32_t> & values) {
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) out << ",";
            out << values[i];
        }
        out << "]";
    };
    const std::vector<int32_t> raw_ref = reverse_delay_pattern_tn(ref_codes);
    const int raw_ref_frames = ref_codes.frames - (ref_codes.codebooks - 1);
    out << "{\n\"prompt_ids\":";
    write_vec(prompt);
    out << ",\n\"reference_codes\":[";
    for (int f = 0; f < raw_ref_frames; ++f) {
        if (f) out << ",";
        out << "[";
        for (int c = 0; c < ref_codes.codebooks; ++c) {
            if (c) out << ",";
            out << raw_ref[static_cast<size_t>(f) * static_cast<size_t>(ref_codes.codebooks) + static_cast<size_t>(c)];
        }
        out << "]";
    }
    out << "],\n\"delayed_reference_codes\":[";
    for (int f = 0; f < ref_codes.frames; ++f) {
        if (f) out << ",";
        out << "[";
        for (int c = 0; c < ref_codes.codebooks; ++c) {
            if (c) out << ",";
            out << ref_codes.at(c, f);
        }
        out << "]";
    }
    out << "],\n\"delayed_codes\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i) out << ",";
        write_vec(rows[i]);
    }
    out << "],\n\"trace\":[";
    for (size_t i = 0; i < steps.size(); ++i) {
        if (i) out << ",";
        out << "{\"step\":" << i << ",\"top1\":";
        write_vec(steps[i].top1);
        out << ",\"top5\":[";
        for (size_t c = 0; c < steps[i].top5.size(); ++c) {
            if (c) out << ",";
            write_vec(steps[i].top5[c]);
        }
        out << "],\"top5_values\":[";
        for (size_t c = 0; c < steps[i].top5_values.size(); ++c) {
            if (c) out << ",";
            out << "[";
            for (size_t j = 0; j < steps[i].top5_values[c].size(); ++j) {
                if (j) out << ",";
                out << steps[i].top5_values[c][j];
            }
            out << "]";
        }
        out << "],\"codes\":";
        write_vec(steps[i].codes);
        out << "}";
    }
    out << "],\n\"finalized_codes\":[";
    for (int c = 0; c < finalized.codebooks; ++c) {
        if (c) out << ",";
        out << "[";
        for (int f = 0; f < finalized.frames; ++f) {
            if (f) out << ",";
            out << finalized.at(c, f);
        }
        out << "]";
    }
    out << "],\n\"node_stats\":{";
    bool first_node = true;
    for (const auto & item : g_trace_node_stats) {
        if (!first_node) out << ",";
        first_node = false;
        out << "\n\"" << item.first << "\":{\"shape\":[";
        for (size_t i = 0; i < item.second.shape.size(); ++i) {
            if (i) out << ",";
            out << item.second.shape[i];
        }
        out << "],\"last_mean\":" << item.second.last_mean
            << ",\"last_abs_mean\":" << item.second.last_abs_mean
            << ",\"last_max_abs\":" << item.second.last_max_abs
            << ",\"last_first8\":[";
        for (size_t i = 0; i < item.second.last_first8.size(); ++i) {
            if (i) out << ",";
            out << item.second.last_first8[i];
        }
        out << "]}";
    }
    out << "\n}\n}\n";
}

void write_codes_json(const std::string & path, const std::vector<std::vector<int32_t>> & rows, const CodeMatrix & finalized) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open codes json: " + path);
    }
    auto write_vec = [&out](const std::vector<int32_t> & values) {
        out << "[";
        for (size_t i = 0; i < values.size(); ++i) {
            if (i) out << ",";
            out << values[i];
        }
        out << "]";
    };
    out << "{\"delayed_codes\":[";
    for (size_t i = 0; i < rows.size(); ++i) {
        if (i) out << ",";
        write_vec(rows[i]);
    }
    out << "],\"finalized_codes\":[";
    for (int c = 0; c < finalized.codebooks; ++c) {
        if (c) out << ",";
        out << "[";
        for (int f = 0; f < finalized.frames; ++f) {
            if (f) out << ",";
            out << finalized.at(c, f);
        }
        out << "]";
    }
    out << "]}\n";
}

void flush_partial_trace_json(
    const std::vector<int32_t> & prompt,
    const CodeMatrix & ref_codes,
    const std::vector<std::vector<int32_t>> & rows,
    const std::vector<TraceStep> & steps,
    const std::vector<int32_t> & raw_tn) {
    if (trace_json_path().empty() || raw_tn.empty()) {
        return;
    }
    (void)raw_tn;
    CodeMatrix finalized;
    finalized.codebooks = k_num_codebooks;
    finalized.frames = 0;
    if (rows.size() >= static_cast<size_t>(k_num_codebooks)) {
        finalized = finalize_generated_codes(delayed_rows_to_matrix(rows), true);
    }
    write_trace_json(trace_json_path(), prompt, ref_codes, rows, steps, finalized);
}

size_t kv_cache_offset(const TextKvCache & cache, int64_t layer, int64_t pos, int64_t head, int64_t dim) {
    if (layer < 0 || layer >= cache.layers || pos < 0 || pos >= cache.capacity || head < 0 || head >= cache.kv_heads || dim < 0 || dim >= cache.head_dim) {
        throw std::out_of_range("KV cache index out of range");
    }
    return static_cast<size_t>((((layer * cache.capacity + pos) * cache.kv_heads + head) * cache.head_dim) + dim);
}

TextKvCache make_text_kv_cache(const TextConfig & cfg, int64_t capacity) {
    if (capacity <= 0) {
        throw std::invalid_argument("KV cache capacity must be positive");
    }
    TextKvCache cache;
    cache.layers = cfg.block_count;
    cache.head_dim = cfg.key_length;
    cache.kv_heads = cfg.head_count_kv;
    cache.capacity = capacity;
    const size_t values = static_cast<size_t>(cache.layers * cache.capacity * cache.kv_heads * cache.head_dim);
    cache.key.assign(values, 0.0f);
    cache.value.assign(values, 0.0f);
    return cache;
}

size_t backend_kv_cache_offset_bytes(const BackendKvCache & cache, const ggml_tensor * tensor, int64_t layer, int64_t pos, int64_t head, int64_t dim) {
    if (layer < 0 || layer >= cache.layers || pos < 0 || pos >= cache.capacity || head < 0 || head >= cache.kv_heads || dim < 0 || dim >= cache.head_dim) {
        throw std::out_of_range("backend KV cache index out of range");
    }
    return static_cast<size_t>(layer) * tensor->nb[3] + static_cast<size_t>(pos) * tensor->nb[2] + static_cast<size_t>(head) * tensor->nb[1] + static_cast<size_t>(dim) * tensor->nb[0];
}

BackendKvCache make_backend_kv_cache(const TextConfig & cfg, BackendRuntime & runtime, int64_t capacity) {
    if (capacity <= 0) {
        throw std::invalid_argument("backend KV cache capacity must be positive");
    }
    BackendKvCache cache;
    cache.cfg = cfg;
    cache.layers = cfg.block_count;
    cache.head_dim = cfg.key_length;
    cache.kv_heads = cfg.head_count_kv;
    cache.capacity = capacity;
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    cache.ctx.reset(ggml_init(params));
    if (!cache.ctx) {
        throw std::runtime_error("failed to allocate backend KV cache context");
    }
    cache.key = ggml_new_tensor_4d(cache.ctx.get(), GGML_TYPE_F32, cache.head_dim, cache.kv_heads, cache.capacity, cache.layers);
    cache.value = ggml_new_tensor_4d(cache.ctx.get(), GGML_TYPE_F32, cache.head_dim, cache.kv_heads, cache.capacity, cache.layers);
    cache.rope_cos = ggml_new_tensor_2d(cache.ctx.get(), GGML_TYPE_F32, cfg.key_length / 2, cfg.context_length);
    cache.rope_sin = ggml_new_tensor_2d(cache.ctx.get(), GGML_TYPE_F32, cfg.key_length / 2, cfg.context_length);
    cache.backend = runtime.backend.get();
    cache.weights_gguf = runtime.weights_gguf.get();
    cache.weights_ggml = runtime.weights_ggml.get();
    cache.loaded_weights = &runtime.loaded_weights;
    cache.loaded_weights_mutex = &runtime.loaded_weights_mutex;
    cache.buffer.reset(ggml_backend_alloc_ctx_tensors(cache.ctx.get(), cache.backend));
    if (!cache.key || !cache.value || !cache.rope_cos || !cache.rope_sin || !cache.buffer) {
        throw std::runtime_error("failed to allocate backend KV cache tensors");
    }
    return cache;
}

BackendKvCache & get_backend_kv_cache(BackendRuntime & runtime, int64_t capacity) {
    if (runtime.kv_cache_slots.empty()) {
        runtime.kv_cache_slots.emplace_back();
        runtime.kv_cache_slot_busy.push_back(false);
    }
    if (!runtime.kv_cache_slots[0] || runtime.kv_cache_slots[0]->capacity < capacity) {
        runtime.kv_cache_slots[0] = std::make_unique<BackendKvCache>(make_backend_kv_cache(*runtime.cfg, runtime, capacity));
    }
    runtime.kv_cache = std::move(runtime.kv_cache_slots[0]);
    runtime.kv_cache->used = 0;
    runtime.kv_cache_slots[0] = std::move(runtime.kv_cache);
    return *runtime.kv_cache_slots[0];
}

ScopedBackendKvCacheSlot acquire_backend_kv_cache_slot(BackendRuntime & runtime, int64_t capacity) {
    std::lock_guard<std::mutex> lock(runtime.kv_cache_slots_mutex);
    for (size_t i = 0; i < runtime.kv_cache_slots.size(); ++i) {
        if (runtime.kv_cache_slot_busy[i]) {
            continue;
        }
        if (!runtime.kv_cache_slots[i] || runtime.kv_cache_slots[i]->capacity < capacity) {
            runtime.kv_cache_slots[i] = std::make_unique<BackendKvCache>(make_backend_kv_cache(*runtime.cfg, runtime, capacity));
        }
        runtime.kv_cache_slot_busy[i] = true;
        runtime.kv_cache_slots[i]->used = 0;
        return ScopedBackendKvCacheSlot(runtime, static_cast<int>(i), *runtime.kv_cache_slots[i]);
    }
    runtime.kv_cache_slots.push_back(std::make_unique<BackendKvCache>(make_backend_kv_cache(*runtime.cfg, runtime, capacity)));
    runtime.kv_cache_slot_busy.push_back(true);
    runtime.kv_cache_slots.back()->used = 0;
    return ScopedBackendKvCacheSlot(runtime, static_cast<int>(runtime.kv_cache_slots.size() - 1), *runtime.kv_cache_slots.back());
}

void ScopedBackendKvCacheSlot::release() {
    if (runtime != nullptr && slot >= 0) {
        std::lock_guard<std::mutex> lock(runtime->kv_cache_slots_mutex);
        if (static_cast<size_t>(slot) < runtime->kv_cache_slot_busy.size()) {
            runtime->kv_cache_slot_busy[static_cast<size_t>(slot)] = false;
        }
    }
    runtime = nullptr;
    slot = -1;
    cache = nullptr;
}

bool ensure_backend_cache_weights(const std::string & model_path, BackendKvCache & cache, const std::vector<std::string> & names) {
    std::lock_guard<std::mutex> lock(*cache.loaded_weights_mutex);
    std::vector<std::string> missing;
    for (const std::string & name : names) {
        if (cache.loaded_weights->find(name) == cache.loaded_weights->end()) {
            missing.push_back(name);
        }
    }
    if (missing.empty()) {
        return true;
    }
    if (!load_named_tensors_from_gguf(model_path, cache.weights_ggml, cache.weights_gguf, missing)) {
        return false;
    }
    for (const std::string & name : missing) {
        (*cache.loaded_weights)[name] = true;
    }
    return true;
}

bool ensure_backend_runtime_weights(const std::string & model_path, BackendRuntime & runtime, const std::vector<std::string> & names) {
    std::lock_guard<std::mutex> lock(runtime.loaded_weights_mutex);
    std::vector<std::string> missing;
    for (const std::string & name : names) {
        if (runtime.loaded_weights.find(name) == runtime.loaded_weights.end()) {
            missing.push_back(name);
        }
    }
    if (missing.empty()) {
        return true;
    }
    if (!load_named_tensors_from_gguf(model_path, runtime.weights_ggml.get(), runtime.weights_gguf.get(), missing)) {
        return false;
    }
    for (const std::string & name : missing) {
        runtime.loaded_weights[name] = true;
    }
    return true;
}

void ensure_backend_cache_fused_ffn_weights(BackendKvCache & cache) {
    if (!reference_kv_fused_ffn_enabled() || cache.fused_ffn_ready) {
        return;
    }
    ScopedProfileTimer timer("reference_kv_fused_ffn_pack");
    const TextConfig & cfg = cache.cfg;
    ggml_init_params params{};
    params.mem_size = 4 * 1024 * 1024;
    params.no_alloc = true;
    cache.fused_ffn_ctx.reset(ggml_init(params));
    if (!cache.fused_ffn_ctx) {
        throw std::runtime_error("failed to allocate fused FFN weight context");
    }
    cache.fused_ffn_gate_up.assign(static_cast<size_t>(cfg.block_count), nullptr);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        ggml_tensor * gate = ggml_get_tensor(cache.weights_ggml, (prefix + "ffn_gate.weight").c_str());
        ggml_tensor * up = ggml_get_tensor(cache.weights_ggml, (prefix + "ffn_up.weight").c_str());
        if (gate == nullptr || up == nullptr) {
            throw std::runtime_error("missing FFN gate/up tensor for fused packing");
        }
        if (gate->type != up->type ||
            gate->ne[0] != cfg.embedding_length || up->ne[0] != cfg.embedding_length ||
            gate->ne[1] != cfg.feed_forward_length || up->ne[1] != cfg.feed_forward_length ||
            gate->ne[2] != 1 || up->ne[2] != 1 || gate->ne[3] != 1 || up->ne[3] != 1 ||
            gate->nb[0] != static_cast<int64_t>(ggml_element_size(gate)) ||
            up->nb[0] != static_cast<int64_t>(ggml_element_size(up))) {
            throw std::runtime_error("unsupported FFN gate/up tensor layout for fused packing");
        }
        if (gate->type != GGML_TYPE_F16 && gate->type != GGML_TYPE_F32) {
            throw std::runtime_error("fused FFN packing currently supports only F16/F32 gate/up weights");
        }
        ggml_tensor * packed = ggml_new_tensor_2d(cache.fused_ffn_ctx.get(), gate->type, cfg.embedding_length, cfg.feed_forward_length * 2);
        ggml_set_name(packed, (prefix + "ffn_gate_up.packed").c_str());
        cache.fused_ffn_gate_up[static_cast<size_t>(layer)] = packed;
    }
    cache.fused_ffn_buffer.reset(ggml_backend_alloc_ctx_tensors(cache.fused_ffn_ctx.get(), cache.backend));
    if (!cache.fused_ffn_buffer) {
        throw std::runtime_error("failed to allocate fused FFN weight tensors");
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        ggml_tensor * gate = ggml_get_tensor(cache.weights_ggml, (prefix + "ffn_gate.weight").c_str());
        ggml_tensor * up = ggml_get_tensor(cache.weights_ggml, (prefix + "ffn_up.weight").c_str());
        ggml_tensor * packed = cache.fused_ffn_gate_up[static_cast<size_t>(layer)];
        const size_t gate_bytes = ggml_nbytes(gate);
        const size_t up_bytes = ggml_nbytes(up);
        std::vector<uint8_t> bytes(gate_bytes + up_bytes);
        ggml_backend_tensor_get(gate, bytes.data(), 0, gate_bytes);
        ggml_backend_tensor_get(up, bytes.data() + gate_bytes, 0, up_bytes);
        ggml_backend_tensor_set(packed, bytes.data(), 0, bytes.size());
    }
    cache.fused_ffn_ready = true;
}

void build_reference_ffn_gate_up(ggml_context * ctx, const TextConfig & cfg, BackendKvCache & cache, ggml_context * weights_ctx, int layer, ggml_tensor * mlp_normed, int64_t tokens, ggml_tensor ** gate, ggml_tensor ** up) {
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    if (reference_kv_fused_ffn_enabled()) {
        if (!cache.fused_ffn_ready || cache.fused_ffn_gate_up.size() <= static_cast<size_t>(layer) || cache.fused_ffn_gate_up[static_cast<size_t>(layer)] == nullptr) {
            throw std::runtime_error("fused FFN weights are not ready");
        }
        ggml_tensor * gate_up = mul_mat_f32(ctx, cache.fused_ffn_gate_up[static_cast<size_t>(layer)], mlp_normed);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(gate_up, "kvstage:reference_kv_ffn_gate_up_matmul");
        }
        *gate = ggml_cont_2d(ctx, ggml_view_2d(ctx, gate_up, cfg.feed_forward_length, tokens, gate_up->nb[1], 0), cfg.feed_forward_length, tokens);
        *up = ggml_cont_2d(ctx, ggml_view_2d(ctx, gate_up, cfg.feed_forward_length, tokens, gate_up->nb[1], static_cast<size_t>(cfg.feed_forward_length) * ggml_element_size(gate_up)), cfg.feed_forward_length, tokens);
    } else {
        *gate = mul_mat_f32(ctx, ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(*gate, "kvstage:reference_kv_ffn_gate_matmul");
        }
        *up = mul_mat_f32(ctx, ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(*up, "kvstage:reference_kv_ffn_up_matmul");
        }
    }
}

std::vector<std::string> all_block_weight_names(const TextConfig & cfg) {
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(cfg.block_count * 11));
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        names.push_back(prefix + "attn_norm.weight");
        names.push_back(prefix + "attn_q.weight");
        names.push_back(prefix + "attn_k.weight");
        names.push_back(prefix + "attn_v.weight");
        names.push_back(prefix + "attn_output.weight");
        names.push_back(prefix + "attn_q_norm.weight");
        names.push_back(prefix + "attn_k_norm.weight");
        names.push_back(prefix + "ffn_norm.weight");
        names.push_back(prefix + "ffn_gate.weight");
        names.push_back(prefix + "ffn_up.weight");
        names.push_back(prefix + "ffn_down.weight");
    }
    return names;
}

GgufPtr open_gguf(const std::string & path) {
    gguf_init_params params{};
    params.no_alloc = true;
    GgufPtr ctx(gguf_init_from_file(path.c_str(), params));
    if (!ctx) {
        throw std::runtime_error("failed to open GGUF: " + path);
    }
    return ctx;
}

GgufWithTensors open_gguf_with_tensors(const std::string & path) {
    ggml_context * ggml_ctx = nullptr;
    gguf_init_params params{};
    params.no_alloc = true;
    params.ctx = &ggml_ctx;
    GgufPtr gguf_ctx(gguf_init_from_file(path.c_str(), params));
    if (!gguf_ctx || ggml_ctx == nullptr) {
        if (ggml_ctx != nullptr) {
            ggml_free(ggml_ctx);
        }
        throw std::runtime_error("failed to open GGUF tensors: " + path);
    }
    return {std::move(gguf_ctx), GgmlPtr(ggml_ctx)};
}

bool load_named_tensors_from_gguf(const std::string & path, ggml_context * ggml_ctx, const gguf_context * gguf_ctx, const std::vector<std::string> & names) {
    FILE * file = ggml_fopen(path.c_str(), "rb");
    if (file == nullptr) {
        return false;
    }
    const size_t buf_size = 4 * 1024 * 1024;
    std::vector<uint8_t> buf(buf_size);
    for (const std::string & name : names) {
        const int64_t tid = gguf_find_tensor(gguf_ctx, name.c_str());
        ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, name.c_str());
        if (tid < 0 || tensor == nullptr) {
            fclose(file);
            return false;
        }
        const size_t offset = gguf_get_data_offset(gguf_ctx) + gguf_get_tensor_offset(gguf_ctx, tid);
        if (fseek(file, static_cast<long>(offset), SEEK_SET) != 0) {
            fclose(file);
            return false;
        }
        const size_t nbytes = ggml_nbytes(tensor);
        for (size_t pos = 0; pos < nbytes; pos += buf_size) {
            const size_t n = std::min(buf_size, nbytes - pos);
            if (fread(buf.data(), 1, n, file) != n) {
                fclose(file);
                return false;
            }
            ggml_backend_tensor_set(tensor, buf.data(), pos, n);
        }
    }
    fclose(file);
    return true;
}

int64_t get_i64(const gguf_context * ctx, const char * key) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
    }
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_UINT8:
            return gguf_get_val_u8(ctx, kid);
        case GGUF_TYPE_INT8:
            return gguf_get_val_i8(ctx, kid);
        case GGUF_TYPE_UINT16:
            return gguf_get_val_u16(ctx, kid);
        case GGUF_TYPE_INT16:
            return gguf_get_val_i16(ctx, kid);
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(ctx, kid);
        case GGUF_TYPE_INT32:
            return gguf_get_val_i32(ctx, kid);
        case GGUF_TYPE_UINT64:
            return static_cast<int64_t>(gguf_get_val_u64(ctx, kid));
        case GGUF_TYPE_INT64:
            return gguf_get_val_i64(ctx, kid);
        default:
            throw std::runtime_error(std::string("GGUF metadata key is not integer: ") + key);
    }
}

std::string get_string(const gguf_context * ctx, const char * key) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
    }
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("GGUF metadata key is not string: ") + key);
    }
    return gguf_get_val_str(ctx, kid);
}

float get_f32(const gguf_context * ctx, const char * key) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        throw std::runtime_error(std::string("missing GGUF metadata key: ") + key);
    }
    switch (gguf_get_kv_type(ctx, kid)) {
        case GGUF_TYPE_FLOAT32:
            return gguf_get_val_f32(ctx, kid);
        case GGUF_TYPE_UINT32:
            return static_cast<float>(gguf_get_val_u32(ctx, kid));
        case GGUF_TYPE_INT32:
            return static_cast<float>(gguf_get_val_i32(ctx, kid));
        default:
            throw std::runtime_error(std::string("GGUF metadata key is not float: ") + key);
    }
}

TensorInfo require_tensor(const gguf_context * ctx, const char * name) {
    const int64_t tid = gguf_find_tensor(ctx, name);
    if (tid < 0) {
        throw std::runtime_error(std::string("missing required GGUF tensor: ") + name);
    }
    TensorInfo info;
    info.name = name;
    info.type = ggml_type_name(gguf_get_tensor_type(ctx, tid));
    info.bytes = gguf_get_tensor_size(ctx, tid);
    return info;
}

TensorInfo require_tensor(const gguf_context * ctx, ggml_context * ggml_ctx, const char * name) {
    TensorInfo info = require_tensor(ctx, name);
    const ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, name);
    if (tensor == nullptr) {
        throw std::runtime_error(std::string("missing required GGML tensor metadata: ") + name);
    }
    const int n_dims = ggml_n_dims(tensor);
    for (int i = 0; i < n_dims; ++i) {
        info.ne.push_back(tensor->ne[i]);
    }
    return info;
}

void require_special_token(int32_t id, const char * token) {
    if (id < 0) {
        throw std::runtime_error(std::string("missing tokenizer special token: ") + token);
    }
}

std::string default_chatml_system_prompt(PromptFormat format) {
    if (format == PromptFormat::boson_chatml) {
        return "Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>";
    }
    return "Generate audio following instruction.";
}

std::string render_chatml_prompt(const std::string & system_prompt, const std::string & text) {
    return "<|im_start|>system\n" + system_prompt + "<|im_end|>\n"
        + "<|im_start|>user\n" + text + "<|im_end|>\n"
        + "<|im_start|>assistant\n";
}

} // namespace

std::vector<int32_t> run_synthetic_text_codes(const std::string & model_path, const GenerateOptions & options);
CodeMatrix repeat_code_frame(const std::vector<int32_t> & frame_codes, int output_frames, const char * label);
CodeMatrix run_prompt_text_code_frames(const std::string & model_path, const GenerateOptions & options, int frames, CodeMatrix * delayed_out = nullptr);
CodeMatrix run_prompt_text_code_frames_backend_kv(const std::string & model_path, const GenerateOptions & options, int frames, BackendRuntime & runtime, CodeMatrix * delayed_out = nullptr);
CodeMatrix run_reference_text_code_frames(const std::string & model_path, const GenerateOptions & options, const CodeMatrix & ref_codes, int frames, CodeMatrix * delayed_out = nullptr);
CodeMatrix run_reference_text_code_frames_backend_kv(const std::string & model_path, const GenerateOptions & options, const CodeMatrix & ref_codes, int frames, BackendRuntime & runtime, CodeMatrix * delayed_out = nullptr, int * backend_kv_slot_out = nullptr, ReferenceArRuntimeStats * runtime_stats = nullptr);
CodeMatrix run_synthetic_text_code_frames(const std::string & model_path, const GenerateOptions & options, int frames);
std::vector<float> run_codec_waveform_graph(const std::string & model_path, const std::vector<int32_t> & codes_tn, int frames);
std::vector<float> run_codec_waveform_graph_with_runtime(const std::string & model_path, const std::vector<int32_t> & codes_tn, int frames, BackendRuntime & runtime);
std::vector<float> run_synthetic_codec_waveform_graph(const std::string & model_path);
std::vector<float> run_prompt_last_hidden(const std::string & model_path, const std::string & text, const CodeMatrix * raw_audio, const GenerateOptions & options);
std::vector<float> run_kv_decode_graph_layer_with_backend_cache(const std::string & model_path, int layer, const std::vector<float> & hidden_input, BackendKvCache & cache);
KvDecodeResult run_kv_decode_graph_all_layers_with_backend_cache_ex(const std::string & model_path, const std::vector<float> & hidden_input, BackendKvCache & cache, bool include_audio_logits = false, bool skip_hidden_readback = false);
std::vector<float> run_kv_decode_graph_all_layers_with_backend_cache(const std::string & model_path, const std::vector<float> & hidden_input, BackendKvCache & cache);
std::vector<float> run_kv_decode_window_all_layers_with_backend_cache(const std::string & model_path, const std::vector<float> & hidden_input, int64_t tokens, BackendKvCache & cache);
std::vector<float> run_kv_decode_window_all_layers_full_with_backend_cache(const std::string & model_path, const std::vector<float> & hidden_input, int64_t tokens, BackendKvCache & cache, std::vector<float> * logits_vc_out = nullptr, bool return_last_hidden_only = false, bool skip_hidden_readback = false);
void reserve_kv_decode_window_full_scheduler_graph(const std::string & model_path, BackendKvCache & cache, bool include_logits);
std::vector<float> run_prefill_all_layers_into_backend_cache(const std::string & model_path, const std::vector<float> & hidden_values, BackendKvCache & cache);
void write_backend_kv_cache_layer_from_hidden(const std::string & model_path, int layer, int64_t pos, const std::vector<float> & hidden_input, BackendKvCache & cache);
std::vector<float> run_audio_head_logits_vc_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, BackendKvCache & cache);
std::vector<float> run_audio_head_logits_flat_batched_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, int frames, BackendKvCache & cache);
std::vector<float> run_audio_head_logits_vc_for_frame_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, int frames, int frame, BackendKvCache & cache);
std::vector<int32_t> run_audio_head_sample_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, const GenerateOptions & options, BackendKvCache & cache);
std::vector<float> audio_head_batcher_run(const std::string & model_path, int step, const std::vector<float> & hidden, BackendKvCache & cache, ReferenceArRuntimeStats * stats);

std::string reference_code_cache_key(const std::string & model_path, const std::string & wav_path) {
    struct stat st {};
    if (stat(wav_path.c_str(), &st) != 0) {
        return model_path + "\n" + wav_path + "\nmissing";
    }
    return model_path + "\n" + wav_path + "\n" + std::to_string(static_cast<long long>(st.st_size)) + "\n" + std::to_string(static_cast<long long>(st.st_mtime));
}

std::string reference_code_disk_cache_dir() {
    const char * value = std::getenv("HIGGS_REFERENCE_CODES_CACHE_DIR");
    return value == nullptr ? "/tmp/higgs-audio-reference-codes" : std::string(value);
}

std::string reference_code_disk_cache_path(const std::string & key) {
    std::ostringstream name;
    name << std::hex << std::hash<std::string>{}(key);
    return reference_code_disk_cache_dir() + "/" + name.str() + ".bin";
}

bool read_reference_code_disk_cache(const std::string & key, CodeMatrix & codes) {
    const std::string path = reference_code_disk_cache_path(key);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    char magic[8]{};
    int32_t codebooks = 0;
    int32_t frames = 0;
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char *>(&codebooks), sizeof(codebooks));
    in.read(reinterpret_cast<char *>(&frames), sizeof(frames));
    if (!in || std::memcmp(magic, "HIGGREF1", 8) != 0 || codebooks != k_num_codebooks || frames <= 0) {
        return false;
    }
    std::vector<int32_t> data(static_cast<size_t>(frames * codebooks));
    in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(int32_t)));
    if (!in) {
        return false;
    }
    codes.codebooks = codebooks;
    codes.frames = frames;
    codes.data = std::move(data);
    return true;
}

void write_reference_code_disk_cache(const std::string & key, const CodeMatrix & codes) {
    const std::string dir = reference_code_disk_cache_dir();
    (void) mkdir(dir.c_str(), 0755);
    const std::string path = reference_code_disk_cache_path(key);
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return;
    }
    const int32_t codebooks = codes.codebooks;
    const int32_t frames = codes.frames;
    out.write("HIGGREF1", 8);
    out.write(reinterpret_cast<const char *>(&codebooks), sizeof(codebooks));
    out.write(reinterpret_cast<const char *>(&frames), sizeof(frames));
    out.write(reinterpret_cast<const char *>(codes.data.data()), static_cast<std::streamsize>(codes.data.size() * sizeof(int32_t)));
}

void ensure_backend_kv_rope_tables(BackendKvCache & cache) {
    const size_t expected = static_cast<size_t>((cache.cfg.key_length / 2) * cache.cfg.context_length);
    const bool need_cos = cache.rope_cos_values.size() != expected;
    const bool need_sin = cache.rope_sin_values.size() != expected;
    if (need_cos) {
        cache.rope_cos_values = make_bf16_rope_table(cache.cfg, false);
    }
    if (need_sin) {
        cache.rope_sin_values = make_bf16_rope_table(cache.cfg, true);
    }
    if (need_cos && cache.rope_cos != nullptr) {
        ggml_backend_tensor_set(cache.rope_cos, cache.rope_cos_values.data(), 0, cache.rope_cos_values.size() * sizeof(float));
    }
    if (need_sin && cache.rope_sin != nullptr) {
        ggml_backend_tensor_set(cache.rope_sin, cache.rope_sin_values.data(), 0, cache.rope_sin_values.size() * sizeof(float));
    }
}

HiggsPipeline::HiggsPipeline(std::string model_path) : model_path_(std::move(model_path)), runtime_(std::make_shared<PipelineRuntime>(model_path_)) {}

HiggsPipeline::HiggsPipeline(std::string model_path, BackendKind backend) : HiggsPipeline(std::move(model_path)) {
    std::static_pointer_cast<PipelineRuntime>(runtime_)->get(backend);
}

const std::string & HiggsPipeline::model_path() const {
    return model_path_;
}

GenerateResult HiggsPipeline::generate(const GenerateOptions & options) {
    ScopedProfileTimer total_timer("pipeline_generate_total");
    const auto generate_start = std::chrono::steady_clock::now();
    const bool scheduler_reference_path = std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr && !options.ref_wav.empty();
    std::unique_lock<std::mutex> lock(generate_mutex_, std::defer_lock);
    if (!scheduler_reference_path) {
        lock.lock();
    }
    BackendScope backend_scope(options.backend);
    BackendRuntime * runtime_ptr = nullptr;
    {
        if (scheduler_reference_path) {
            lock.lock();
        }
        runtime_ptr = &std::static_pointer_cast<PipelineRuntime>(runtime_)->get(options.backend);
        if (scheduler_reference_path) {
            lock.unlock();
        }
    }
    BackendRuntime & runtime = *runtime_ptr;
    throw_if_cancelled(options);
    if (options.ref_wav.empty()) {
        CodeMatrix codes;
        CodeMatrix delayed_codes;
        {
            ScopedProfileTimer timer("pipeline_text_ar");
            codes = run_prompt_text_code_frames_backend_kv(model_path_, options, std::max(1, options.steps), runtime, &delayed_codes);
        }
        profile_code_matrix("generated_codes", codes);
        GenerateResult result;
        result.delayed_codes = std::move(delayed_codes);
        result.finalized_codes = codes;
        {
            const auto codec_start = std::chrono::steady_clock::now();
            ScopedProfileTimer timer("pipeline_codec_decode");
            result.waveform = run_codec_waveform_graph_with_runtime(model_path_, codes.data, codes.frames, runtime);
            result.codec_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - codec_start).count();
        }
        result.sample_rate = 24000;
        result.reference_ar_wall_ms = 0;
        result.total_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generate_start).count();
        result.scheduler_mode = "off";
        return result;
    }
    if (options.prompt_format != PromptFormat::higgstts) {
        throw std::runtime_error("reference-audio generation currently supports only higgstts prompt format");
    }
    throw_if_cancelled(options);
    CodeMatrix ref_codes;
    std::string ref_cache_status;
    long long reference_cache_wall_ms = 0;
    {
        const auto ref_cache_start = std::chrono::steady_clock::now();
        if (scheduler_reference_path) {
            lock.lock();
        }
        const std::string ref_cache_key = reference_code_cache_key(model_path_, options.ref_wav);
        auto ref_cached = reference_code_cache_.find(ref_cache_key);
        if (ref_cached != reference_code_cache_.end()) {
            ScopedProfileTimer timer("pipeline_reference_cache_hit");
            ref_codes = ref_cached->second.codes;
            ref_cache_status = "memory";
        } else if (read_reference_code_disk_cache(ref_cache_key, ref_codes)) {
            ScopedProfileTimer timer("pipeline_reference_disk_cache_hit");
            reference_code_cache_[ref_cache_key] = ReferenceCodeCacheEntry{ref_codes};
            ref_cache_status = "disk";
        } else {
            ScopedProfileTimer timer("pipeline_reference_encode");
            BackendScope reference_scope(BackendKind::cpu);
            ref_codes = encode_reference_codes(model_path_, options.ref_wav);
            reference_code_cache_[ref_cache_key] = ReferenceCodeCacheEntry{ref_codes};
            write_reference_code_disk_cache(ref_cache_key, ref_codes);
            ref_cache_status = "encoded";
        }
        if (scheduler_reference_path) {
            lock.unlock();
        }
        reference_cache_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ref_cache_start).count();
    }
    const CodeMatrix delayed_ref_codes = apply_delay_pattern_tn(ref_codes.data, ref_codes.frames, ref_codes.codebooks);
    const int steps = std::max(1, options.steps);
    CodeMatrix codes;
    CodeMatrix delayed_codes;
    int backend_kv_slot = -1;
    long long reference_ar_wall_ms = 0;
    ReferenceArRuntimeStats runtime_stats;
    {
        const auto ar_start = std::chrono::steady_clock::now();
        ScopedProfileTimer timer("pipeline_text_ar");
        const bool use_backend_kv = options.backend == BackendKind::cuda || reference_backend_kv_diagnostic_enabled();
        codes = use_backend_kv
            ? run_reference_text_code_frames_backend_kv(model_path_, options, delayed_ref_codes, steps, runtime, &delayed_codes, &backend_kv_slot, scheduler_reference_path && options.backend == BackendKind::cuda ? &runtime_stats : nullptr)
            : run_reference_text_code_frames(model_path_, options, delayed_ref_codes, steps, &delayed_codes);
        reference_ar_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - ar_start).count();
    }
    profile_code_matrix("generated_codes", codes);
    GenerateResult result;
    result.delayed_codes = std::move(delayed_codes);
    result.finalized_codes = codes;
    result.reference_cache_status = std::move(ref_cache_status);
    result.backend_kv_slot = backend_kv_slot;
    result.reference_cache_wall_ms = reference_cache_wall_ms;
    result.reference_ar_wall_ms = reference_ar_wall_ms;
    result.audio_head_wall_ms = runtime_stats.audio_head_wall_ms;
    result.audio_head_batch_calls = runtime_stats.audio_head_batch_calls;
    result.audio_head_fallback_calls = runtime_stats.audio_head_fallback_calls;
    result.audio_head_batch_size_avg = runtime_stats.audio_head_batch_calls > 0
        ? static_cast<float>(runtime_stats.audio_head_batch_items) / static_cast<float>(runtime_stats.audio_head_batch_calls)
        : 0.0f;
    result.cuda_executor_wait_ms = runtime_stats.cuda_executor_wait_ms;
    result.cuda_executor_run_ms = runtime_stats.cuda_executor_run_ms;
    result.cuda_executor_queue_depth = runtime_stats.cuda_executor_queue_depth;
    result.scheduler_mode = scheduler_reference_path ? (options.backend == BackendKind::cuda ? "cuda_executor" : (backend_kv_slot >= 0 ? "kv_slots_round_robin" : "blocked_shared_kv_fallback")) : "off";
    if (trace_only_enabled()) {
        result.sample_rate = 24000;
        result.total_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generate_start).count();
        return result;
    }
    {
        if (scheduler_reference_path) {
            lock.lock();
        }
        const auto codec_start = std::chrono::steady_clock::now();
        ScopedProfileTimer timer("pipeline_codec_decode");
        if (scheduler_reference_path && options.backend == BackendKind::cuda) {
            const CudaExecutorRunStats stats = cuda_executor().run_sync([&] {
                result.waveform = run_codec_waveform_graph_with_runtime(model_path_, codes.data, codes.frames, runtime);
            });
            result.cuda_executor_wait_ms += stats.wait_ms;
            result.cuda_executor_run_ms += stats.run_ms;
            result.cuda_executor_queue_depth = std::max(result.cuda_executor_queue_depth, stats.queue_depth);
        } else {
            result.waveform = run_codec_waveform_graph_with_runtime(model_path_, codes.data, codes.frames, runtime);
        }
        result.codec_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - codec_start).count();
        if (scheduler_reference_path) {
            lock.unlock();
        }
    }
    result.sample_rate = 24000;
    result.total_wall_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - generate_start).count();
    return result;
}

int32_t & CodeMatrix::at(int codebook, int frame) {
    return data.at(static_cast<size_t>(frame) * static_cast<size_t>(codebooks) + static_cast<size_t>(codebook));
}

int32_t CodeMatrix::at(int codebook, int frame) const {
    return data.at(static_cast<size_t>(frame) * static_cast<size_t>(codebooks) + static_cast<size_t>(codebook));
}

CodeMatrix apply_delay_pattern_tn(const std::vector<int32_t> & raw_tn, int frames, int codebooks) {
    if (frames <= 0 || codebooks <= 0 || static_cast<size_t>(frames * codebooks) != raw_tn.size()) {
        throw std::invalid_argument("raw_tn must have shape [frames, codebooks]");
    }
    CodeMatrix out;
    out.codebooks = codebooks;
    out.frames = frames + codebooks - 1;
    out.data.assign(static_cast<size_t>(out.frames * out.codebooks), k_eoc_id);
    for (int c = 0; c < codebooks; ++c) {
        for (int f = 0; f < c; ++f) {
            out.at(c, f) = k_boc_id;
        }
        for (int f = 0; f < frames; ++f) {
            out.at(c, c + f) = raw_tn[static_cast<size_t>(f) * static_cast<size_t>(codebooks) + static_cast<size_t>(c)];
        }
    }
    return out;
}

std::vector<int32_t> reverse_delay_pattern_tn(const CodeMatrix & delayed) {
    const int frames = delayed.frames - (delayed.codebooks - 1);
    if (delayed.codebooks <= 0 || frames <= 0 || delayed.data.size() != static_cast<size_t>(delayed.frames * delayed.codebooks)) {
        throw std::invalid_argument("delayed must have GGML-order shape (codebooks, frames) with frames >= codebooks");
    }
    std::vector<int32_t> out(static_cast<size_t>(frames * delayed.codebooks));
    for (int c = 0; c < delayed.codebooks; ++c) {
        for (int f = 0; f < frames; ++f) {
            out[static_cast<size_t>(f) * static_cast<size_t>(delayed.codebooks) + static_cast<size_t>(c)] = delayed.at(c, c + f);
        }
    }
    return out;
}

CodeMatrix finalize_generated_codes(const CodeMatrix & delayed, bool trim_bos) {
    const std::vector<int32_t> raw = reverse_delay_pattern_tn(delayed);
    const int raw_frames = delayed.frames - (delayed.codebooks - 1);
    const int start = trim_bos ? 1 : 0;
    CodeMatrix out;
    out.codebooks = delayed.codebooks;
    out.frames = raw_frames > start + 1 ? raw_frames - start - 1 : 0;
    out.data.resize(static_cast<size_t>(out.frames * out.codebooks));
    for (int f = 0; f < out.frames; ++f) {
        for (int c = 0; c < out.codebooks; ++c) {
            int32_t v = raw[static_cast<size_t>(f + start) * static_cast<size_t>(out.codebooks) + static_cast<size_t>(c)];
            v = std::max<int32_t>(0, std::min<int32_t>(k_codebook_data_size - 1, v));
            out.at(c, f) = v;
        }
    }
    return out;
}

void self_check_delay() {
    std::vector<int32_t> raw(static_cast<size_t>(4 * k_num_codebooks));
    for (size_t i = 0; i < raw.size(); ++i) {
        raw[i] = static_cast<int32_t>(i);
    }
    const CodeMatrix delayed = apply_delay_pattern_tn(raw, 4, k_num_codebooks);
    if (delayed.frames != 11 || delayed.codebooks != k_num_codebooks) {
        throw std::runtime_error("delay self-check shape mismatch");
    }
    const std::vector<int32_t> restored = reverse_delay_pattern_tn(delayed);
    if (restored != raw) {
        throw std::runtime_error("delay self-check restore mismatch");
    }
    const CodeMatrix finalized = finalize_generated_codes(delayed, true);
    if (finalized.frames != 2 || finalized.codebooks != k_num_codebooks) {
        throw std::runtime_error("delay self-check finalized shape mismatch");
    }
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int f = 0; f < c; ++f) {
            if (delayed.at(c, f) != k_boc_id) {
                throw std::runtime_error("delay self-check BOC mismatch");
            }
        }
        for (int f = c + 4; f < delayed.frames; ++f) {
            if (delayed.at(c, f) != k_eoc_id) {
                throw std::runtime_error("delay self-check EOC mismatch");
            }
        }
        if (finalized.at(c, 0) != raw[static_cast<size_t>(1 * k_num_codebooks + c)] ||
            finalized.at(c, 1) != raw[static_cast<size_t>(2 * k_num_codebooks + c)]) {
            throw std::runtime_error("delay self-check finalize payload mismatch");
        }
    }
}

ModelInfo inspect_model(const std::string & model_path) {
    const GgufPtr ctx = open_gguf(model_path);

    ModelInfo info;
    info.tensor_count = gguf_get_n_tensors(ctx.get());
    info.kv_count = gguf_get_n_kv(ctx.get());
    info.config_json = get_string(ctx.get(), "higgs.config.json");
    info.audio_num_codebooks = get_i64(ctx.get(), "higgs.audio.num_codebooks");
    info.audio_vocab_size = get_i64(ctx.get(), "higgs.audio.vocab_size");

    const char * required[] = {
        "token_embd.weight",
        "output_norm.weight",
        "a.token_embd",
        "a.output",
        "blk.0.attn_q.weight",
        "blk.0.ffn_down.weight",
        "a.q.quantizers.0.codebook.embed",
        "a.fc2.weight",
        "a.ad.conv1.weight",
        "a.ae.conv1.weight",
        "a.se.conv.weight",
        "a.fc.weight",
    };
    for (const char * name : required) {
        info.required_tensors.push_back(require_tensor(ctx.get(), name));
    }
    return info;
}

void validate_model_contract(const std::string & model_path) {
    const ModelInfo info = inspect_model(model_path);
    if (info.audio_num_codebooks != 8) {
        throw std::runtime_error("unexpected higgs.audio.num_codebooks: " + std::to_string(info.audio_num_codebooks));
    }
    if (info.audio_vocab_size != 1026) {
        throw std::runtime_error("unexpected higgs.audio.vocab_size: " + std::to_string(info.audio_vocab_size));
    }
    if (info.tensor_count <= 0 || info.kv_count <= 0) {
        throw std::runtime_error("GGUF appears empty");
    }
    if (info.config_json.find("higgs_multimodal_qwen3") == std::string::npos) {
        throw std::runtime_error("higgs.config.json does not identify higgs_multimodal_qwen3");
    }
}

TextConfig inspect_text_config(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    const gguf_context * ctx = loaded.gguf.get();
    ggml_context * ggml_ctx = loaded.ggml.get();
    TextConfig cfg;
    cfg.vocab_size = get_i64(ctx, "qwen3.vocab_size");
    cfg.context_length = get_i64(ctx, "qwen3.context_length");
    cfg.embedding_length = get_i64(ctx, "qwen3.embedding_length");
    cfg.feed_forward_length = get_i64(ctx, "qwen3.feed_forward_length");
    cfg.block_count = get_i64(ctx, "qwen3.block_count");
    cfg.head_count = get_i64(ctx, "qwen3.attention.head_count");
    cfg.head_count_kv = get_i64(ctx, "qwen3.attention.head_count_kv");
    cfg.key_length = get_i64(ctx, "qwen3.attention.key_length");
    cfg.value_length = get_i64(ctx, "qwen3.attention.value_length");
    cfg.rms_norm_eps = get_f32(ctx, "qwen3.attention.layer_norm_rms_epsilon");
    cfg.rope_dimension_count = get_i64(ctx, "qwen3.rope.dimension_count");
    cfg.rope_freq_base = get_f32(ctx, "qwen3.rope.freq_base");

    const int last = static_cast<int>(cfg.block_count - 1);
    const std::string last_prefix = "blk." + std::to_string(last) + ".";
    const std::string tensors[] = {
        "token_embd.weight",
        "output_norm.weight",
        "a.token_embd",
        "a.output",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        last_prefix + "attn_q.weight",
        last_prefix + "ffn_down.weight",
    };
    for (const std::string & name : tensors) {
        cfg.frontier_tensors.push_back(require_tensor(ctx, ggml_ctx, name.c_str()));
    }
    return cfg;
}

void validate_text_contract(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    if (cfg.vocab_size <= 0 || cfg.context_length <= 0 || cfg.embedding_length <= 0 || cfg.feed_forward_length <= 0) {
        throw std::runtime_error("invalid qwen3 text dimensions");
    }
    if (cfg.block_count <= 0 || cfg.head_count <= 0 || cfg.head_count_kv <= 0 || cfg.head_count % cfg.head_count_kv != 0) {
        throw std::runtime_error("invalid qwen3 attention head contract");
    }
    if (cfg.key_length <= 0 || cfg.value_length <= 0 || cfg.key_length != cfg.value_length || cfg.rope_dimension_count != cfg.key_length) {
        throw std::runtime_error("invalid qwen3 key/value/rope dimension contract");
    }
    if (cfg.rms_norm_eps <= 0.0f || cfg.rope_freq_base <= 0.0f) {
        throw std::runtime_error("invalid qwen3 norm/rope scalar contract");
    }
    const auto require_ne = [&](const char * name, std::vector<int64_t> expected) {
        const auto it = std::find_if(cfg.frontier_tensors.begin(), cfg.frontier_tensors.end(), [&](const TensorInfo & info) {
            return info.name == name;
        });
        if (it == cfg.frontier_tensors.end() || it->ne != expected) {
            throw std::runtime_error(std::string("unexpected tensor ne for ") + name);
        }
    };
    require_ne("token_embd.weight", {cfg.embedding_length, cfg.vocab_size});
    require_ne("output_norm.weight", {cfg.embedding_length});
    require_ne("a.token_embd", {cfg.embedding_length, k_num_codebooks * k_codebook_vocab_size});
    require_ne("a.output", {cfg.embedding_length, k_num_codebooks * k_codebook_vocab_size});
    require_ne("blk.0.attn_q.weight", {cfg.embedding_length, cfg.head_count * cfg.key_length});
    require_ne("blk.0.attn_k.weight", {cfg.embedding_length, cfg.head_count_kv * cfg.key_length});
    require_ne("blk.0.attn_v.weight", {cfg.embedding_length, cfg.head_count_kv * cfg.value_length});
    require_ne("blk.0.attn_output.weight", {cfg.head_count * cfg.value_length, cfg.embedding_length});
    require_ne("blk.0.attn_q_norm.weight", {cfg.key_length});
    require_ne("blk.0.attn_k_norm.weight", {cfg.key_length});
    require_ne("blk.0.ffn_gate.weight", {cfg.embedding_length, cfg.feed_forward_length});
    require_ne("blk.0.ffn_up.weight", {cfg.embedding_length, cfg.feed_forward_length});
    require_ne("blk.0.ffn_down.weight", {cfg.feed_forward_length, cfg.embedding_length});
    require_ne("blk.0.attn_norm.weight", {cfg.embedding_length});
    require_ne("blk.0.ffn_norm.weight", {cfg.embedding_length});
}

int64_t count_tensors_with_prefix(const gguf_context * ctx, const char * prefix) {
    int64_t count = 0;
    const int64_t n = gguf_get_n_tensors(ctx);
    const size_t prefix_len = std::strlen(prefix);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(ctx, i);
        if (std::strncmp(name, prefix, prefix_len) == 0) {
            ++count;
        }
    }
    return count;
}

ReferenceEncodeInfo inspect_reference_encode(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    const gguf_context * ctx = loaded.gguf.get();
    ggml_context * ggml_ctx = loaded.ggml.get();
    ReferenceEncodeInfo info;
    info.acoustic_tensor_count = count_tensors_with_prefix(ctx, "a.ae.");
    info.semantic_model_tensor_count = count_tensors_with_prefix(ctx, "a.sm.");
    info.semantic_encoder_tensor_count = count_tensors_with_prefix(ctx, "a.se.");
    info.fc_tensor_count = count_tensors_with_prefix(ctx, "a.fc.");
    info.fc1_tensor_count = count_tensors_with_prefix(ctx, "a.fc1.");
    info.fc2_tensor_count = count_tensors_with_prefix(ctx, "a.fc2.");
    info.quantizer_tensor_count = count_tensors_with_prefix(ctx, "a.q.");
    const char * tensors[] = {
        "a.ae.conv1.weight",
        "a.ae.block.0.conv1.weight",
        "a.ae.block.0.res_unit1.conv2.weight",
        "a.ae.conv2.weight",
        "a.sm.fe.conv_layers.0.conv.weight",
        "a.sm.fp.projection.weight",
        "a.sm.encoder.pce.conv.pz.w0",
        "a.sm.encoder.pce.conv.pz.w1",
        "a.sm.encoder.pce.conv.bias",
        "a.sm.encoder.layer_norm.weight",
        "a.sm.encoder.layers.0.attention.q_proj.weight",
        "a.sm.encoder.layers.11.ff.out.weight",
        "a.se.conv.weight",
        "a.se.conv_blocks.0.conv.weight",
        "a.se.conv_blocks.1.res_units.1.conv2.weight",
        "a.fc.weight",
        "a.fc.bias",
        "a.fc1.weight",
        "a.fc2.weight",
        "a.q.quantizers.0.codebook.embed",
        "a.q.quantizers.0.project_in.weight",
        "a.q.quantizers.0.project_out.weight",
    };
    for (const char * name : tensors) {
        info.frontier_tensors.push_back(require_tensor(ctx, ggml_ctx, name));
    }
    return info;
}

void validate_reference_encode_contract(const std::string & model_path) {
    const ReferenceEncodeInfo info = inspect_reference_encode(model_path);
    if (info.sample_rate != 24000 || info.semantic_sample_rate != 16000 || info.downsample_factor != 320 || info.semantic_downsample_factor != 2) {
        throw std::runtime_error("reference encode rate/downsample contract mismatch");
    }
    if (info.semantic_hidden_size != 768 || info.semantic_layer_count != 12 || info.semantic_head_count != 12 || info.semantic_intermediate_size != 3072) {
        throw std::runtime_error("reference encode semantic config mismatch");
    }
    if (info.acoustic_tensor_count != 110 || info.semantic_model_tensor_count != 211 || info.semantic_encoder_tensor_count != 13 ||
        info.fc_tensor_count != 2 || info.fc1_tensor_count != 2 || info.fc2_tensor_count != 2 || info.quantizer_tensor_count != 64) {
        throw std::runtime_error("reference encode tensor inventory mismatch");
    }
    const auto require_ne = [&](const char * name, std::vector<int64_t> expected) {
        const auto it = std::find_if(info.frontier_tensors.begin(), info.frontier_tensors.end(), [&](const TensorInfo & tensor) {
            return tensor.name == name;
        });
        if (it == info.frontier_tensors.end() || it->ne != expected) {
            throw std::runtime_error(std::string("unexpected reference encode tensor ne for ") + name);
        }
    };
    require_ne("a.ae.conv1.weight", {7, 1, 64});
    require_ne("a.sm.fe.conv_layers.0.conv.weight", {10, 1, 512});
    require_ne("a.sm.fp.projection.weight", {512, 768});
    require_ne("a.sm.encoder.pce.conv.pz.w0", {128});
    require_ne("a.sm.encoder.pce.conv.pz.w1", {128, 48, 768});
    require_ne("a.sm.encoder.pce.conv.bias", {768});
    require_ne("a.sm.encoder.layer_norm.weight", {768});
    require_ne("a.sm.encoder.layers.0.attention.q_proj.weight", {768, 768});
    require_ne("a.sm.encoder.layers.11.ff.out.weight", {3072, 768});
    require_ne("a.se.conv.weight", {3, 768, 768});
    require_ne("a.se.conv_blocks.0.conv.weight", {3, 768, 768});
    require_ne("a.se.conv_blocks.1.res_units.1.conv2.weight", {1, 768, 768});
    require_ne("a.fc.weight", {1024, 1024});
    require_ne("a.fc.bias", {1024});
    require_ne("a.fc1.weight", {1024, 768});
    require_ne("a.fc2.weight", {1024, 256});
    require_ne("a.q.quantizers.0.codebook.embed", {64, 1024});
}

int64_t tokenizer_arr_count(const gguf_context * ctx, const char * key, gguf_type type) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return 0;
    }
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, kid) != type) {
        throw std::runtime_error(std::string("unexpected tokenizer metadata type for ") + key);
    }
    return static_cast<int64_t>(gguf_get_arr_n(ctx, kid));
}

int64_t tokenizer_string_bytes(const gguf_context * ctx, const char * key) {
    const int64_t kid = gguf_find_key(ctx, key);
    if (kid < 0) {
        return 0;
    }
    if (gguf_get_kv_type(ctx, kid) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("unexpected tokenizer metadata type for ") + key);
    }
    return static_cast<int64_t>(std::strlen(gguf_get_val_str(ctx, kid)));
}

TokenizerInfo inspect_tokenizer(const std::string & model_path) {
    const GgufPtr ctx = open_gguf(model_path);
    TokenizerInfo info;
    info.model = get_string(ctx.get(), "tokenizer.ggml.model");
    info.vocab_size = get_i64(ctx.get(), "qwen3.vocab_size");
    info.token_count = tokenizer_arr_count(ctx.get(), "tokenizer.ggml.tokens", GGUF_TYPE_STRING);
    info.token_type_count = tokenizer_arr_count(ctx.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32);
    info.merge_count = tokenizer_arr_count(ctx.get(), "tokenizer.ggml.merges", GGUF_TYPE_STRING);
    info.added_token_count = tokenizer_arr_count(ctx.get(), "tokenizer.ggml.added_tokens", GGUF_TYPE_STRING);
    info.huggingface_json_bytes = tokenizer_string_bytes(ctx.get(), "tokenizer.huggingface.json");
    info.chat_template_bytes = tokenizer_string_bytes(ctx.get(), "tokenizer.chat_template");
    info.eos_token_id = get_i64(ctx.get(), "tokenizer.ggml.eos_token_id");
    info.padding_token_id = get_i64(ctx.get(), "tokenizer.ggml.padding_token_id");
    info.special = load_special_token_ids(model_path);
    return info;
}

void validate_tokenizer_contract(const std::string & model_path) {
    const TokenizerInfo info = inspect_tokenizer(model_path);
    if (info.model != "gpt2") {
        throw std::runtime_error("unexpected tokenizer.ggml.model: " + info.model);
    }
    if (info.token_count <= 0 || info.token_count > info.vocab_size || info.added_token_count <= 0) {
        throw std::runtime_error("tokenizer token inventory is invalid");
    }
    if (info.merge_count <= 0 || info.huggingface_json_bytes <= 0 || info.chat_template_bytes <= 0) {
        throw std::runtime_error("tokenizer metadata is incomplete");
    }
    const int32_t specials[] = {info.special.tts, info.special.text, info.special.audio, info.special.ref_text, info.special.ref_audio};
    for (int32_t id : specials) {
        if (id < 0 || id >= info.vocab_size) {
            throw std::runtime_error("tokenizer special token id is out of range");
        }
    }
}

uint32_t read_u32(std::ifstream & in) {
    uint32_t value = 0;
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("unexpected EOF in tokenizer artifact");
    }
    return value;
}

void skip_string(std::ifstream & in) {
    const uint32_t size = read_u32(in);
    in.seekg(size, std::ios::cur);
    if (!in) {
        throw std::runtime_error("unexpected EOF in tokenizer artifact string");
    }
}

std::string read_string(std::ifstream & in) {
    const uint32_t size = read_u32(in);
    std::string out(size, '\0');
    in.read(out.data(), size);
    if (!in) {
        throw std::runtime_error("unexpected EOF in tokenizer artifact string");
    }
    return out;
}

TokenizerArtifactInfo inspect_tokenizer_artifact(const std::string & path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("failed to open tokenizer artifact: " + path);
    }
    TokenizerArtifactInfo info;
    info.file_bytes = static_cast<int64_t>(in.tellg());
    in.seekg(0);
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (std::string(magic, sizeof(magic)) != std::string("HATK1\0\0\0", 8)) {
        throw std::runtime_error("invalid tokenizer artifact magic");
    }
    info.token_count = read_u32(in);
    info.merge_count = read_u32(in);
    info.added_token_count = read_u32(in);
    info.chat_template_bytes = read_u32(in);
    for (int64_t i = 0; i < info.token_count; ++i) {
        skip_string(in);
    }
    in.seekg(info.token_count * static_cast<int64_t>(sizeof(int32_t)), std::ios::cur);
    if (!in) {
        throw std::runtime_error("unexpected EOF in tokenizer artifact token types");
    }
    for (int64_t i = 0; i < info.merge_count; ++i) {
        skip_string(in);
    }
    skip_string(in);
    if (!in) {
        throw std::runtime_error("unexpected EOF in tokenizer artifact chat template");
    }
    return info;
}

std::string merge_key(const std::string & left, const std::string & right) {
    return left + '\n' + right;
}

struct LoadedTokenizerArtifact {
    std::unordered_map<std::string, int32_t> token_to_id;
    std::unordered_map<std::string, int32_t> merge_rank;
};

LoadedTokenizerArtifact load_tokenizer_artifact(const std::string & path, bool with_merges) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open tokenizer artifact: " + path);
    }
    char magic[8]{};
    in.read(magic, sizeof(magic));
    if (std::string(magic, sizeof(magic)) != std::string("HATK1\0\0\0", 8)) {
        throw std::runtime_error("invalid tokenizer artifact magic");
    }
    const uint32_t token_count = read_u32(in);
    const uint32_t merge_count = read_u32(in);
    (void) read_u32(in);
    (void) read_u32(in);
    LoadedTokenizerArtifact artifact;
    artifact.token_to_id.reserve(token_count);
    for (uint32_t i = 0; i < token_count; ++i) {
        const std::string token = read_string(in);
        if (!token.empty()) {
            artifact.token_to_id.emplace(token, static_cast<int32_t>(i));
        }
    }
    in.seekg(token_count * static_cast<uint64_t>(sizeof(int32_t)), std::ios::cur);
    if (!in) {
        throw std::runtime_error("unexpected EOF in tokenizer artifact token types");
    }
    if (with_merges) {
        artifact.merge_rank.reserve(merge_count);
    }
    for (uint32_t i = 0; i < merge_count; ++i) {
        const std::string merge = read_string(in);
        if (!with_merges) {
            continue;
        }
        const size_t space = merge.find(' ');
        if (space == std::string::npos) {
            throw std::runtime_error("invalid tokenizer merge entry");
        }
        artifact.merge_rank.emplace(merge_key(merge.substr(0, space), merge.substr(space + 1)), static_cast<int32_t>(i));
    }
    return artifact;
}

std::unordered_map<std::string, int32_t> load_tokenizer_artifact_token_map(const std::string & path) {
    return load_tokenizer_artifact(path, false).token_to_id;
}

std::vector<int32_t> encode_exact_token_strings(const std::string & path, const std::vector<std::string> & tokens) {
    const std::unordered_map<std::string, int32_t> token_to_id = load_tokenizer_artifact_token_map(path);
    std::vector<int32_t> ids;
    ids.reserve(tokens.size());
    for (const std::string & token : tokens) {
        const auto it = token_to_id.find(token);
        if (it == token_to_id.end()) {
            throw std::runtime_error("tokenizer artifact missing exact token: " + token);
        }
        ids.push_back(it->second);
    }
    return ids;
}

void append_utf8(std::string & out, uint32_t cp) {
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
}

std::array<std::string, 256> byte_level_tokens() {
    std::array<std::string, 256> table{};
    std::vector<int> bs;
    for (int i = '!'; i <= '~'; ++i) {
        bs.push_back(i);
    }
    for (int i = 0xA1; i <= 0xAC; ++i) {
        bs.push_back(i);
    }
    for (int i = 0xAE; i <= 0xFF; ++i) {
        bs.push_back(i);
    }
    std::vector<int> cs = bs;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
            bs.push_back(b);
            cs.push_back(256 + n++);
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        append_utf8(table[static_cast<size_t>(bs[i])], static_cast<uint32_t>(cs[i]));
    }
    return table;
}

std::vector<int32_t> encode_byte_tokens(const std::string & path, const std::string & text) {
    const std::unordered_map<std::string, int32_t> token_to_id = load_tokenizer_artifact_token_map(path);
    const std::array<std::string, 256> table = byte_level_tokens();
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (unsigned char byte : text) {
        const std::string & token = table[static_cast<size_t>(byte)];
        const auto it = token_to_id.find(token);
        if (it == token_to_id.end()) {
            throw std::runtime_error("tokenizer artifact missing byte token");
        }
        ids.push_back(it->second);
    }
    return ids;
}

std::vector<std::string> byte_token_strings(const std::string & text) {
    const std::array<std::string, 256> table = byte_level_tokens();
    std::vector<std::string> tokens;
    tokens.reserve(text.size());
    for (unsigned char byte : text) {
        tokens.push_back(table[static_cast<size_t>(byte)]);
    }
    return tokens;
}

std::vector<int32_t> encode_bpe_token_chunk(const std::string & path, const std::string & text) {
    const LoadedTokenizerArtifact artifact = load_tokenizer_artifact(path, true);
    std::vector<std::string> pieces = byte_token_strings(text);
    while (pieces.size() > 1) {
        int best_rank = std::numeric_limits<int>::max();
        size_t best = pieces.size();
        for (size_t i = 0; i + 1 < pieces.size(); ++i) {
            const auto it = artifact.merge_rank.find(merge_key(pieces[i], pieces[i + 1]));
            if (it != artifact.merge_rank.end() && it->second < best_rank) {
                best_rank = it->second;
                best = i;
            }
        }
        if (best == pieces.size()) {
            break;
        }
        pieces[best] += pieces[best + 1];
        pieces.erase(pieces.begin() + static_cast<std::ptrdiff_t>(best + 1));
    }
    std::vector<int32_t> ids;
    ids.reserve(pieces.size());
    for (const std::string & piece : pieces) {
        const auto it = artifact.token_to_id.find(piece);
        if (it == artifact.token_to_id.end()) {
            throw std::runtime_error("tokenizer artifact missing BPE piece");
        }
        ids.push_back(it->second);
    }
    return ids;
}

void validate_tokenizer_artifact(const std::string & path);

void self_check_tokenizer_artifact(const std::string & path) {
    validate_tokenizer_artifact(path);
    const std::vector<int32_t> ids = encode_exact_token_strings(path, {"!", "A", "<|tts|>", "<|text|>", "<|audio|>"});
    if (ids != std::vector<int32_t>{0, 32, 151667, 151672, 151670}) {
        throw std::runtime_error("tokenizer artifact exact-token self-check mismatch");
    }
}

void self_check_tokenizer_bytelevel(const std::string & path) {
    validate_tokenizer_artifact(path);
    if (encode_byte_tokens(path, " A") != std::vector<int32_t>{220, 32}) {
        throw std::runtime_error("tokenizer byte-level ASCII self-check mismatch");
    }
    if (encode_byte_tokens(path, "我") != std::vector<int32_t>{162, 230, 239}) {
        throw std::runtime_error("tokenizer byte-level UTF-8 self-check mismatch");
    }
}

void self_check_tokenizer_bpe(const std::string & path) {
    validate_tokenizer_artifact(path);
    if (encode_bpe_token_chunk(path, "hello") != std::vector<int32_t>{14990}) {
        throw std::runtime_error("tokenizer BPE hello self-check mismatch");
    }
    if (encode_bpe_token_chunk(path, "test") != std::vector<int32_t>{1944}) {
        throw std::runtime_error("tokenizer BPE test self-check mismatch");
    }
    if (encode_bpe_token_chunk(path, "我") != std::vector<int32_t>{35946}) {
        throw std::runtime_error("tokenizer BPE UTF-8 self-check mismatch");
    }
}

std::vector<int32_t> encode_text_tokenizers_cpp(const std::string & model_path, const std::string & text) {
    const GgufPtr ctx = open_gguf(model_path);
    const tokenizers_cpp::Encoding encoded = tokenizers_cpp::Tokenizer::from_json(get_string(ctx.get(), "tokenizer.huggingface.json")).encode(text, false);
    std::vector<int32_t> ids;
    ids.reserve(encoded.ids.size());
    for (uint32_t id : encoded.ids) {
        ids.push_back(static_cast<int32_t>(id));
    }
    return ids;
}

void self_check_tokenizers_cpp(const std::string & model_path) {
    const GgufPtr ctx = open_gguf(model_path);
    const tokenizers_cpp::Tokenizer tokenizer = tokenizers_cpp::Tokenizer::from_json(get_string(ctx.get(), "tokenizer.huggingface.json"));
    const auto expect = [&](const std::string & text, std::vector<uint32_t> ids) {
        const std::vector<int32_t> got = encode_text_tokenizers_cpp(model_path, text);
        if (got != std::vector<int32_t>(ids.begin(), ids.end())) {
            throw std::runtime_error("tokenizers.cpp encode mismatch for: " + text);
        }
    };
    expect("hello", {14990});
    expect("test", {1944});
    expect("我", {35946});
    expect("你好", {108386});
    expect(" hello", {23811});
    if (tokenizer.token_to_id("<|tts|>").value_or(0) != 151667) {
        throw std::runtime_error("tokenizers.cpp vocabulary contract mismatch");
    }
}

void validate_tokenizer_artifact(const std::string & path) {
    const TokenizerArtifactInfo info = inspect_tokenizer_artifact(path);
    if (info.token_count <= 0 || info.merge_count <= 0 || info.added_token_count <= 0 || info.chat_template_bytes <= 0) {
        throw std::runtime_error("tokenizer artifact contract mismatch");
    }
}

SpecialTokenIds load_special_token_ids(const std::string & model_path) {
    const GgufPtr ctx = open_gguf(model_path);
    const int64_t kid = gguf_find_key(ctx.get(), "tokenizer.ggml.tokens");
    if (kid < 0) {
        throw std::runtime_error("missing GGUF metadata key: tokenizer.ggml.tokens");
    }
    if (gguf_get_kv_type(ctx.get(), kid) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx.get(), kid) != GGUF_TYPE_STRING) {
        throw std::runtime_error("tokenizer.ggml.tokens must be a string array");
    }

    SpecialTokenIds ids;
    const size_t n = gguf_get_arr_n(ctx.get(), kid);
    for (size_t i = 0; i < n; ++i) {
        const std::string token = gguf_get_arr_str(ctx.get(), kid, i);
        if (token == "<|tts|>") {
            ids.tts = static_cast<int32_t>(i);
        } else if (token == "<|text|>") {
            ids.text = static_cast<int32_t>(i);
        } else if (token == "<|audio|>") {
            ids.audio = static_cast<int32_t>(i);
        } else if (token == "<|ref_text|>") {
            ids.ref_text = static_cast<int32_t>(i);
        } else if (token == "<|ref_audio|>") {
            ids.ref_audio = static_cast<int32_t>(i);
        }
    }
    require_special_token(ids.tts, "<|tts|>");
    require_special_token(ids.text, "<|text|>");
    require_special_token(ids.audio, "<|audio|>");
    require_special_token(ids.ref_text, "<|ref_text|>");
    require_special_token(ids.ref_audio, "<|ref_audio|>");
    return ids;
}

std::vector<int32_t> build_higgstts_prompt_ids(const SpecialTokenIds & ids, const std::vector<int32_t> & text_ids) {
    std::vector<int32_t> out;
    out.reserve(text_ids.size() + 3);
    out.push_back(ids.tts);
    out.push_back(ids.text);
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    out.push_back(ids.audio);
    return out;
}

std::vector<int32_t> build_higgstts_prompt_ids(const std::string & model_path, const std::string & text) {
    return build_higgstts_prompt_ids(load_special_token_ids(model_path), encode_text_tokenizers_cpp(model_path, text));
}

std::vector<int32_t> build_prompt_ids(const std::string & model_path, const GenerateOptions & options) {
    if (options.prompt_format == PromptFormat::higgstts) {
        return build_higgstts_prompt_ids(model_path, options.text);
    }
    const std::string system_prompt = options.system_prompt.empty() ? default_chatml_system_prompt(options.prompt_format) : options.system_prompt;
    return encode_text_tokenizers_cpp(model_path, render_chatml_prompt(system_prompt, options.text));
}

std::vector<int32_t> build_reference_prompt_ids(const SpecialTokenIds & ids, const std::vector<int32_t> & text_ids, int delayed_ref_rows, const std::vector<int32_t> & ref_text_ids) {
    if (delayed_ref_rows < 0) {
        throw std::invalid_argument("delayed_ref_rows must be non-negative");
    }
    std::vector<int32_t> out;
    out.reserve(text_ids.size() + ref_text_ids.size() + static_cast<size_t>(delayed_ref_rows) + 5);
    out.push_back(ids.tts);
    if (!ref_text_ids.empty()) {
        out.push_back(ids.ref_text);
        out.insert(out.end(), ref_text_ids.begin(), ref_text_ids.end());
    }
    out.push_back(ids.ref_audio);
    out.insert(out.end(), static_cast<size_t>(delayed_ref_rows), -100);
    out.push_back(ids.text);
    out.insert(out.end(), text_ids.begin(), text_ids.end());
    out.push_back(ids.audio);
    return out;
}

std::vector<int32_t> safe_text_ids_for_embedding(const std::vector<int32_t> & text_ids) {
    std::vector<int32_t> out = text_ids;
    for (int32_t & id : out) {
        if (id == -100) {
            id = 0;
        }
    }
    return out;
}

std::vector<int32_t> placeholder_positions(const std::vector<int32_t> & text_ids) {
    std::vector<int32_t> out;
    for (size_t i = 0; i < text_ids.size(); ++i) {
        if (text_ids[i] == -100) {
            out.push_back(static_cast<int32_t>(i));
        }
    }
    return out;
}

CodeMatrix fuse_audio_ids_for_embedding(const CodeMatrix & audio_ids) {
    if (audio_ids.codebooks != k_num_codebooks || audio_ids.frames < 0 ||
        audio_ids.data.size() != static_cast<size_t>(audio_ids.codebooks * audio_ids.frames)) {
        throw std::invalid_argument("audio_ids must have GGML-order shape (ne0=codebooks, ne1=frames)");
    }
    CodeMatrix out;
    out.codebooks = audio_ids.codebooks;
    out.frames = audio_ids.frames;
    out.data.resize(audio_ids.data.size());
    for (int f = 0; f < audio_ids.frames; ++f) {
        for (int c = 0; c < audio_ids.codebooks; ++c) {
            const int32_t id = audio_ids.at(c, f);
            if (id < 0 || id >= k_codebook_vocab_size) {
                throw std::invalid_argument("audio code id out of embedding range");
            }
            out.at(c, f) = id + c * k_codebook_vocab_size;
        }
    }
    return out;
}

void self_check_prompt(const std::string & model_path) {
    const SpecialTokenIds ids = load_special_token_ids(model_path);
    if (ids.tts != 151667 || ids.audio != 151670 || ids.text != 151672 || ids.ref_audio != 151679 || ids.ref_text != 151680) {
        throw std::runtime_error("prompt self-check special token id mismatch");
    }
    const std::vector<int32_t> text_ids{11, 22};
    const std::vector<int32_t> ref_text_ids{33, 44};
    if (build_higgstts_prompt_ids(ids, text_ids) != std::vector<int32_t>{ids.tts, ids.text, 11, 22, ids.audio}) {
        throw std::runtime_error("prompt self-check higgstts mismatch");
    }
    if (build_reference_prompt_ids(ids, text_ids, 3) != std::vector<int32_t>{ids.tts, ids.ref_audio, -100, -100, -100, ids.text, 11, 22, ids.audio}) {
        throw std::runtime_error("prompt self-check reference mismatch");
    }
    if (build_reference_prompt_ids(ids, text_ids, 3, ref_text_ids) != std::vector<int32_t>{ids.tts, ids.ref_text, 33, 44, ids.ref_audio, -100, -100, -100, ids.text, 11, 22, ids.audio}) {
        throw std::runtime_error("prompt self-check reference text mismatch");
    }
}

void self_check_tokenizer_prompt(const std::string & model_path) {
    if (build_higgstts_prompt_ids(model_path, "你好") != std::vector<int32_t>{151667, 151672, 108386, 151670}) {
        throw std::runtime_error("tokenizer prompt self-check higgstts mismatch");
    }
}

void self_check_chatml_prompt(const std::string & model_path) {
    GenerateOptions options;
    options.text = "你好";
    options.prompt_format = PromptFormat::chatml;
    if (build_prompt_ids(model_path, options) != std::vector<int32_t>{151644, 8948, 198, 31115, 7699, 2701, 7600, 13, 151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198}) {
        throw std::runtime_error("chatml prompt self-check mismatch");
    }
    options.prompt_format = PromptFormat::boson_chatml;
    if (build_prompt_ids(model_path, options) != std::vector<int32_t>{151644, 8948, 198, 31115, 7699, 2701, 7600, 382, 27, 91, 22483, 10986, 4906, 91, 397, 14755, 374, 12433, 504, 264, 11340, 3054, 624, 27, 91, 22483, 10986, 6213, 91, 29, 151645, 198, 151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198}) {
        throw std::runtime_error("boson-chatml prompt self-check mismatch");
    }
}

std::vector<float> audio_logits_flat_to_vc(const std::vector<float> & flat, int frames, int frame) {
    const int fused = k_codebook_vocab_size * k_num_codebooks;
    if (frames <= 0 || frame < 0 || frame >= frames || flat.size() != static_cast<size_t>(fused * frames)) {
        throw std::invalid_argument("flat audio logits must have GGML-order shape (ne0=8208, ne1=frames)");
    }
    std::vector<float> out(static_cast<size_t>(k_codebook_vocab_size * k_num_codebooks));
    const size_t frame_offset = static_cast<size_t>(frame) * static_cast<size_t>(fused);
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int v = 0; v < k_codebook_vocab_size; ++v) {
            out[static_cast<size_t>(v) * k_num_codebooks + static_cast<size_t>(c)] =
                flat[frame_offset + static_cast<size_t>(v) + static_cast<size_t>(c) * k_codebook_vocab_size];
        }
    }
    return out;
}

std::vector<int32_t> sample_codes_from_logits_vc(const std::vector<float> & logits_vc, int vocab, int codebooks, float temperature, int top_k, float top_p, uint32_t & rng_state, bool stop_on_eoc) {
    if (vocab <= 0 || codebooks <= 0 || logits_vc.size() != static_cast<size_t>(vocab * codebooks)) {
        throw std::invalid_argument("logits_vc must have source-view shape [vocab, codebooks]");
    }
    std::vector<int32_t> out(static_cast<size_t>(codebooks));
    std::mt19937 rng(rng_state);
    for (int c = 0; c < codebooks; ++c) {
        std::vector<float> col(static_cast<size_t>(vocab));
        for (int v = 0; v < vocab; ++v) {
            col[static_cast<size_t>(v)] = logits_vc[static_cast<size_t>(v) * static_cast<size_t>(codebooks) + static_cast<size_t>(c)];
        }
        for (int v = k_codebook_data_size; v < std::min(vocab, k_eoc_id); ++v) {
            col[static_cast<size_t>(v)] = -std::numeric_limits<float>::infinity();
        }
        if (!stop_on_eoc && k_eoc_id < vocab) {
            col[static_cast<size_t>(k_eoc_id)] = -std::numeric_limits<float>::infinity();
        }
        if (temperature <= 0.0f) {
            out[static_cast<size_t>(c)] = static_cast<int32_t>(std::max_element(col.begin(), col.end()) - col.begin());
            continue;
        }
        if (top_k > 0 && top_k < vocab) {
            std::vector<int> order(static_cast<size_t>(vocab));
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + top_k, order.end(), [&](int a, int b) { return col[static_cast<size_t>(a)] > col[static_cast<size_t>(b)]; });
            std::vector<float> masked(static_cast<size_t>(vocab), -std::numeric_limits<float>::infinity());
            for (int i = 0; i < top_k; ++i) {
                masked[static_cast<size_t>(order[static_cast<size_t>(i)])] = col[static_cast<size_t>(order[static_cast<size_t>(i)])];
            }
            col.swap(masked);
        }
        const float max_logit = *std::max_element(col.begin(), col.end());
        std::vector<double> probs(static_cast<size_t>(vocab));
        double sum = 0.0;
        for (int v = 0; v < vocab; ++v) {
            const double p = std::exp((static_cast<double>(col[static_cast<size_t>(v)]) - max_logit) / temperature);
            probs[static_cast<size_t>(v)] = p;
            sum += p;
        }
        if (sum <= 0.0 || !std::isfinite(sum)) {
            throw std::runtime_error("sampler probability sum is invalid");
        }
        for (double & p : probs) {
            p /= sum;
        }
        if (top_p < 1.0f) {
            std::vector<int> order(static_cast<size_t>(vocab));
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
            double cdf = 0.0;
            bool remove = false;
            for (size_t i = 0; i < order.size(); ++i) {
                cdf += probs[static_cast<size_t>(order[i])];
                if (remove) {
                    probs[static_cast<size_t>(order[i])] = 0.0;
                }
                remove = cdf > top_p;
            }
        }
        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        out[static_cast<size_t>(c)] = dist(rng);
    }
    rng_state = rng();
    return out;
}

bool greedy_logits_have_small_margin(const std::vector<float> & logits_vc, int vocab, int codebooks, int active_codebooks, float margin) {
    if (vocab <= 0 || codebooks <= 0 || logits_vc.size() != static_cast<size_t>(vocab * codebooks)) {
        throw std::invalid_argument("logits_vc must have source-view shape [vocab, codebooks]");
    }
    active_codebooks = std::max(0, std::min(active_codebooks, codebooks));
    for (int c = 0; c < active_codebooks; ++c) {
        float best = -std::numeric_limits<float>::infinity();
        float second = -std::numeric_limits<float>::infinity();
        for (int v = 0; v < vocab; ++v) {
            float value = logits_vc[static_cast<size_t>(v) * static_cast<size_t>(codebooks) + static_cast<size_t>(c)];
            if (v >= k_codebook_data_size && v < std::min(vocab, k_eoc_id)) {
                value = -std::numeric_limits<float>::infinity();
            }
            if (value > best) {
                second = best;
                best = value;
            } else if (value > second) {
                second = value;
            }
        }
        if (best - second < margin) {
            return true;
        }
    }
    return false;
}

struct GreedyMarginInfo {
    int codebook = -1;
    int top1 = -1;
    int top2 = -1;
    float margin = std::numeric_limits<float>::infinity();
};

GreedyMarginInfo greedy_logits_min_active_margin_info(const std::vector<float> & logits_vc, int vocab, int codebooks, int active_codebooks) {
    if (vocab <= 0 || codebooks <= 0 || logits_vc.size() != static_cast<size_t>(vocab * codebooks)) {
        throw std::invalid_argument("logits_vc must have source-view shape [vocab, codebooks]");
    }
    active_codebooks = std::max(0, std::min(active_codebooks, codebooks));
    GreedyMarginInfo out;
    for (int c = 0; c < active_codebooks; ++c) {
        float best = -std::numeric_limits<float>::infinity();
        float second = -std::numeric_limits<float>::infinity();
        int best_id = -1;
        int second_id = -1;
        for (int v = 0; v < vocab; ++v) {
            float value = logits_vc[static_cast<size_t>(v) * static_cast<size_t>(codebooks) + static_cast<size_t>(c)];
            if (v >= k_codebook_data_size && v < std::min(vocab, k_eoc_id)) {
                value = -std::numeric_limits<float>::infinity();
            }
            if (value > best) {
                second = best;
                second_id = best_id;
                best = value;
                best_id = v;
            } else if (value > second) {
                second = value;
                second_id = v;
            }
        }
        const float cb_margin = best - second;
        if (cb_margin < out.margin) {
            out.codebook = c;
            out.top1 = best_id;
            out.top2 = second_id;
            out.margin = cb_margin;
        }
    }
    return out;
}

float greedy_logits_min_active_margin(const std::vector<float> & logits_vc, int vocab, int codebooks, int active_codebooks) {
    return greedy_logits_min_active_margin_info(logits_vc, vocab, codebooks, active_codebooks).margin;
}

void apply_sglang_step_rules(std::vector<int32_t> & codes, int step, int codebooks, int & eoc_countdown, bool stop_on_eoc) {
    if (codebooks <= 0 || codes.size() != static_cast<size_t>(codebooks)) {
        throw std::invalid_argument("codes must have shape [codebooks]");
    }
    if (step < codebooks) {
        const int next_cb = step + 1;
        for (int c = next_cb; c < codebooks; ++c) {
            codes[static_cast<size_t>(c)] = k_boc_id;
        }
    } else if (eoc_countdown >= 0) {
        for (int c = 0; c < codebooks - eoc_countdown; ++c) {
            codes[static_cast<size_t>(c)] = k_eoc_id;
        }
        --eoc_countdown;
    } else if (stop_on_eoc && codes[0] == k_eoc_id) {
        eoc_countdown = codebooks - 2;
    }
}

void self_check_sampler() {
    std::vector<float> logits(static_cast<size_t>(k_codebook_vocab_size * k_num_codebooks), -100.0f);
    for (int c = 0; c < k_num_codebooks; ++c) {
        logits[static_cast<size_t>(7 + c) * k_num_codebooks + static_cast<size_t>(c)] = 10.0f;
    }
    logits[static_cast<size_t>(1024) * k_num_codebooks] = 99.0f;
    logits[static_cast<size_t>(k_eoc_id) * k_num_codebooks + 1] = 99.0f;
    uint32_t rng_state = 12345;
    std::vector<int32_t> codes = sample_codes_from_logits_vc(logits, k_codebook_vocab_size, k_num_codebooks, 0.0f, 0, 1.0f, rng_state, false);
    if (codes[0] != 7 || codes[1] != 8) {
        throw std::runtime_error("sampler self-check mask/argmax mismatch");
    }
    logits.assign(static_cast<size_t>(k_codebook_vocab_size * k_num_codebooks), -100.0f);
    logits[0] = 3.0f;
    logits[static_cast<size_t>(1) * k_num_codebooks] = 2.0f;
    logits[static_cast<size_t>(2) * k_num_codebooks] = 1.0f;
    rng_state = 1;
    codes = sample_codes_from_logits_vc(logits, k_codebook_vocab_size, k_num_codebooks, 1.0f, 0, 0.7f, rng_state, true);
    if (codes[0] > 1) {
        throw std::runtime_error("sampler self-check top-p shifted mask mismatch");
    }
    int eoc_countdown = -1;
    apply_sglang_step_rules(codes, 0, k_num_codebooks, eoc_countdown, true);
    for (int c = 1; c < k_num_codebooks; ++c) {
        if (codes[static_cast<size_t>(c)] != k_boc_id) {
            throw std::runtime_error("sampler self-check BOC window mismatch");
        }
    }
    codes.assign(static_cast<size_t>(k_num_codebooks), 9);
    codes[0] = k_eoc_id;
    apply_sglang_step_rules(codes, k_num_codebooks, k_num_codebooks, eoc_countdown, true);
    if (eoc_countdown != k_num_codebooks - 2) {
        throw std::runtime_error("sampler self-check EOC countdown start mismatch");
    }
    apply_sglang_step_rules(codes, k_num_codebooks + 1, k_num_codebooks, eoc_countdown, true);
    if (codes[0] != k_eoc_id || codes[1] != k_eoc_id || eoc_countdown != k_num_codebooks - 3) {
        throw std::runtime_error("sampler self-check EOC winddown mismatch");
    }
}

void self_check_audio_logits_layout() {
    const int frames = 2;
    const int fused = k_codebook_vocab_size * k_num_codebooks;
    std::vector<float> flat(static_cast<size_t>(fused * frames), -1.0f);
    for (int f = 0; f < frames; ++f) {
        for (int c = 0; c < k_num_codebooks; ++c) {
            for (int v = 0; v < k_codebook_vocab_size; ++v) {
                flat[static_cast<size_t>(f) * fused + static_cast<size_t>(v) + static_cast<size_t>(c) * k_codebook_vocab_size] =
                    static_cast<float>(100000 * f + 1000 * c + v);
            }
        }
    }
    const std::vector<float> vc = audio_logits_flat_to_vc(flat, frames, 1);
    if (vc.size() != static_cast<size_t>(fused)) {
        throw std::runtime_error("audio logits layout self-check size mismatch");
    }
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int v : {0, 17, k_codebook_vocab_size - 1}) {
            const float expected = static_cast<float>(100000 + 1000 * c + v);
            const float got = vc[static_cast<size_t>(v) * k_num_codebooks + static_cast<size_t>(c)];
            if (got != expected) {
                throw std::runtime_error("audio logits layout self-check value mismatch");
            }
        }
    }
}

void self_check_embedding_ids() {
    const std::vector<int32_t> safe = safe_text_ids_for_embedding({5, -100, 7});
    if (safe != std::vector<int32_t>{5, 0, 7}) {
        throw std::runtime_error("embedding ids self-check safe text mismatch");
    }
    CodeMatrix audio;
    audio.codebooks = k_num_codebooks;
    audio.frames = 2;
    audio.data.resize(static_cast<size_t>(audio.codebooks * audio.frames));
    for (int f = 0; f < audio.frames; ++f) {
        for (int c = 0; c < audio.codebooks; ++c) {
            audio.at(c, f) = f * 17 + c;
        }
    }
    const CodeMatrix fused = fuse_audio_ids_for_embedding(audio);
    if (fused.frames != 2 || fused.codebooks != k_num_codebooks) {
        throw std::runtime_error("embedding ids self-check shape mismatch");
    }
    for (int f = 0; f < fused.frames; ++f) {
        for (int c = 0; c < fused.codebooks; ++c) {
            const int32_t expected = audio.at(c, f) + c * k_codebook_vocab_size;
            if (fused.at(c, f) != expected) {
                throw std::runtime_error("embedding ids self-check fused audio mismatch");
            }
        }
    }
    audio.at(0, 0) = k_codebook_vocab_size;
    try {
        (void) fuse_audio_ids_for_embedding(audio);
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error("embedding ids self-check range guard mismatch");
}

std::vector<float> run_embedding_graph(const std::string & model_path, const std::vector<int32_t> & text_data, const CodeMatrix * raw_audio) {
    if (text_data.empty() && (raw_audio == nullptr || raw_audio->frames == 0)) {
        throw std::invalid_argument("embedding graph requires text or audio ids");
    }
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, {"token_embd.weight", "a.token_embd"})) {
        throw std::runtime_error("failed to load embedding tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate embedding graph context");
    }
    ggml_tensor * out = nullptr;
    ggml_tensor * text_ids = nullptr;
    if (!text_data.empty()) {
        text_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, text_data.size());
        out = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, "token_embd.weight"), text_ids);
    }
    std::vector<ggml_tensor *> audio_id_tensors;
    CodeMatrix fused_audio;
    if (raw_audio != nullptr && raw_audio->frames > 0) {
        fused_audio = fuse_audio_ids_for_embedding(*raw_audio);
        ggml_tensor * audio = nullptr;
        for (int c = 0; c < k_num_codebooks; ++c) {
            ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, fused_audio.frames);
            audio_id_tensors.push_back(ids);
            ggml_tensor * cur = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, "a.token_embd"), ids);
            audio = audio == nullptr ? cur : ggml_add(ctx.get(), audio, cur);
        }
        out = out == nullptr ? audio : ggml_concat(ctx.get(), out, audio, 1);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate embedding graph tensors");
    }
    if (text_ids != nullptr) {
        ggml_backend_tensor_set(text_ids, text_data.data(), 0, text_data.size() * sizeof(int32_t));
    }
    for (size_t c = 0; c < audio_id_tensors.size(); ++c) {
        std::vector<int32_t> ids(static_cast<size_t>(fused_audio.frames));
        for (int f = 0; f < fused_audio.frames; ++f) {
            ids[static_cast<size_t>(f)] = fused_audio.at(static_cast<int>(c), f);
        }
        ggml_backend_tensor_set(audio_id_tensors[c], ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("embedding graph compute failed");
    }
    const int64_t expected_tokens = static_cast<int64_t>(text_data.size()) + (raw_audio == nullptr ? 0 : raw_audio->frames);
    if (out->ne[0] != 2560 || out->ne[1] != expected_tokens) {
        throw std::runtime_error("embedding graph output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float v : values) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("embedding graph output contains non-finite values");
        }
    }
    return values;
}

std::vector<float> run_embedding_graph_with_runtime(const std::string & model_path, const std::vector<int32_t> & text_data, const CodeMatrix * raw_audio, BackendRuntime & runtime) {
    if (text_data.empty() && (raw_audio == nullptr || raw_audio->frames == 0)) {
        throw std::invalid_argument("embedding graph requires text or audio ids");
    }
    if (!ensure_backend_runtime_weights(model_path, runtime, {"token_embd.weight", "a.token_embd"})) {
        throw std::runtime_error("failed to load runtime embedding tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate runtime embedding graph context");
    }
    ggml_context * weights_ctx = runtime.weights_ggml.get();
    ggml_tensor * out = nullptr;
    ggml_tensor * text_ids = nullptr;
    if (!text_data.empty()) {
        text_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, text_data.size());
        out = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, "token_embd.weight"), text_ids);
    }
    std::vector<ggml_tensor *> audio_id_tensors;
    CodeMatrix fused_audio;
    if (raw_audio != nullptr && raw_audio->frames > 0) {
        fused_audio = fuse_audio_ids_for_embedding(*raw_audio);
        ggml_tensor * audio = nullptr;
        for (int c = 0; c < k_num_codebooks; ++c) {
            ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, fused_audio.frames);
            audio_id_tensors.push_back(ids);
            ggml_tensor * cur = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, "a.token_embd"), ids);
            audio = audio == nullptr ? cur : ggml_add(ctx.get(), audio, cur);
        }
        out = out == nullptr ? audio : ggml_concat(ctx.get(), out, audio, 1);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), runtime.backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate runtime embedding graph tensors");
    }
    if (text_ids != nullptr) {
        ggml_backend_tensor_set(text_ids, text_data.data(), 0, text_data.size() * sizeof(int32_t));
    }
    for (size_t c = 0; c < audio_id_tensors.size(); ++c) {
        std::vector<int32_t> ids(static_cast<size_t>(fused_audio.frames));
        for (int f = 0; f < fused_audio.frames; ++f) {
            ids[static_cast<size_t>(f)] = fused_audio.at(static_cast<int>(c), f);
        }
        ggml_backend_tensor_set(audio_id_tensors[c], ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(runtime.backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("runtime embedding graph compute failed");
    }
    const int64_t expected_tokens = static_cast<int64_t>(text_data.size()) + (raw_audio == nullptr ? 0 : raw_audio->frames);
    if (out->ne[0] != runtime.cfg->embedding_length || out->ne[1] != expected_tokens) {
        throw std::runtime_error("runtime embedding graph output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float v : values) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("runtime embedding graph output contains non-finite values");
        }
    }
    return values;
}

std::vector<float> run_reference_prompt_embedding_graph_with_runtime(const std::string & model_path, const std::vector<int32_t> & prompt, const CodeMatrix & ref_codes, BackendRuntime & runtime) {
    const std::vector<int32_t> positions = placeholder_positions(prompt);
    if (positions.size() != static_cast<size_t>(ref_codes.frames)) {
        throw std::runtime_error("reference prompt placeholder count mismatch");
    }
    if (!ensure_backend_runtime_weights(model_path, runtime, {"token_embd.weight", "a.token_embd"})) {
        throw std::runtime_error("failed to load runtime reference embedding tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate runtime reference embedding graph context");
    }
    ggml_context * weights_ctx = runtime.weights_ggml.get();
    const std::vector<int32_t> safe_prompt = safe_text_ids_for_embedding(prompt);
    CodeMatrix fused_ref = fuse_audio_ids_for_embedding(ref_codes);

    ggml_tensor * text_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, safe_prompt.size());
    ggml_tensor * text_embeds = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, "token_embd.weight"), text_ids);
    std::vector<ggml_tensor *> ref_id_tensors;
    ggml_tensor * ref_embeds = nullptr;
    for (int c = 0; c < k_num_codebooks; ++c) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, fused_ref.frames);
        ref_id_tensors.push_back(ids);
        ggml_tensor * cur = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, "a.token_embd"), ids);
        ref_embeds = ref_embeds == nullptr ? cur : ggml_add(ctx.get(), ref_embeds, cur);
    }
    ggml_tensor * position_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, positions.size());
    ggml_tensor * out = ggml_set_rows(ctx.get(), text_embeds, ref_embeds, position_ids);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), runtime.backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate runtime reference embedding graph tensors");
    }
    ggml_backend_tensor_set(text_ids, safe_prompt.data(), 0, safe_prompt.size() * sizeof(int32_t));
    ggml_backend_tensor_set(position_ids, positions.data(), 0, positions.size() * sizeof(int32_t));
    for (size_t c = 0; c < ref_id_tensors.size(); ++c) {
        std::vector<int32_t> ids(static_cast<size_t>(fused_ref.frames));
        for (int f = 0; f < fused_ref.frames; ++f) {
            ids[static_cast<size_t>(f)] = fused_ref.at(static_cast<int>(c), f);
        }
        ggml_backend_tensor_set(ref_id_tensors[c], ids.data(), 0, ids.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(runtime.backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("runtime reference embedding graph compute failed");
    }
    if (out->ne[0] != runtime.cfg->embedding_length || out->ne[1] != static_cast<int64_t>(prompt.size())) {
        throw std::runtime_error("runtime reference embedding graph output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float v : values) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("runtime reference embedding graph output contains non-finite values");
        }
    }
    return values;
}

std::vector<float> run_synthetic_embedding_graph(const std::string & model_path) {
    CodeMatrix raw_audio;
    raw_audio.codebooks = k_num_codebooks;
    raw_audio.frames = 1;
    raw_audio.data.resize(static_cast<size_t>(k_num_codebooks));
    for (int c = 0; c < k_num_codebooks; ++c) {
        raw_audio.at(c, 0) = (c * 17) % k_codebook_data_size;
    }
    return run_embedding_graph(model_path, {151935}, &raw_audio);
}

void self_check_embedding_graph(const std::string & model_path) {
    (void) run_synthetic_embedding_graph(model_path);
}

void self_check_prompt_embedding_graph(const std::string & model_path) {
    const std::vector<int32_t> prompt = build_higgstts_prompt_ids(model_path, "你好");
    const std::vector<float> hidden = run_embedding_graph(model_path, prompt, nullptr);
    if (hidden.size() != static_cast<size_t>(2560 * prompt.size())) {
        throw std::runtime_error("prompt embedding graph output size mismatch");
    }
}

void self_check_audio_head_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, {"output_norm.weight", "a.output"})) {
        throw std::runtime_error("failed to load audio head tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate audio head graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
    ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
    ggml_tensor * logits_flat = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
    if (!trace_json_path().empty()) {
        ggml_set_output(final_hidden);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, logits_flat);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate audio head graph tensors");
    }
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("audio head graph compute failed");
    }
    if (logits_flat->ne[0] != k_num_codebooks * k_codebook_vocab_size || logits_flat->ne[1] != 1) {
        throw std::runtime_error("audio head graph output shape mismatch");
    }
    std::vector<float> logits(static_cast<size_t>(logits_flat->ne[0]));
    ggml_backend_tensor_get(logits_flat, logits.data(), 0, logits.size() * sizeof(float));
    const std::vector<float> vc = audio_logits_flat_to_vc(logits, 1, 0);
    uint32_t rng_state = 12345;
    const std::vector<int32_t> codes = sample_codes_from_logits_vc(vc, k_codebook_vocab_size, k_num_codebooks, 0.0f, 0, 1.0f, rng_state, true);
    if (codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("audio head graph sampled code shape mismatch");
    }
    if (std::getenv("HIGGS_SERVER_AUDIO_HEAD_BATCH") != nullptr) {
        PipelineRuntime pipeline_runtime(model_path);
        BackendRuntime & runtime = pipeline_runtime.get(g_backend_kind);
        BackendKvCache & cache = get_backend_kv_cache(runtime, 4);
        std::vector<float> h0(static_cast<size_t>(cfg.embedding_length));
        std::vector<float> h1(static_cast<size_t>(cfg.embedding_length));
        for (size_t i = 0; i < h0.size(); ++i) {
            h0[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
            h1[i] = static_cast<float>(static_cast<int>((i * 17) % 131) - 65) / 65.0f;
        }
        const auto single0_start = std::chrono::steady_clock::now();
        const std::vector<float> single0 = run_audio_head_logits_vc_with_cache(model_path, h0, cache);
        const auto single0_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - single0_start).count();
        const auto single1_start = std::chrono::steady_clock::now();
        const std::vector<float> single1 = run_audio_head_logits_vc_with_cache(model_path, h1, cache);
        const auto single1_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - single1_start).count();
        std::vector<float> batched;
        batched.reserve(h0.size() + h1.size());
        batched.insert(batched.end(), h0.begin(), h0.end());
        batched.insert(batched.end(), h1.begin(), h1.end());
        const auto batch_start = std::chrono::steady_clock::now();
        const std::vector<float> batch_logits = run_audio_head_logits_flat_batched_with_cache(model_path, batched, 2, cache);
        const auto batch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - batch_start).count();
        const std::vector<float> batch0 = audio_logits_flat_to_vc(batch_logits, 2, 0);
        const std::vector<float> batch1 = audio_logits_flat_to_vc(batch_logits, 2, 1);
        const bool logits_equal = single0 == batch0 && single1 == batch1;
        GenerateOptions sample_options;
        uint32_t s0 = sample_options.seed;
        uint32_t s1 = sample_options.seed;
        uint32_t b0 = sample_options.seed;
        uint32_t b1 = sample_options.seed;
        const std::vector<int32_t> single_codes0 = sample_codes_from_logits_vc(single0, k_codebook_vocab_size, k_num_codebooks, 0.0f, 0, 1.0f, s0, true);
        const std::vector<int32_t> single_codes1 = sample_codes_from_logits_vc(single1, k_codebook_vocab_size, k_num_codebooks, 0.0f, 0, 1.0f, s1, true);
        const std::vector<int32_t> batch_codes0 = sample_codes_from_logits_vc(batch0, k_codebook_vocab_size, k_num_codebooks, 0.0f, 0, 1.0f, b0, true);
        const std::vector<int32_t> batch_codes1 = sample_codes_from_logits_vc(batch1, k_codebook_vocab_size, k_num_codebooks, 0.0f, 0, 1.0f, b1, true);
        const bool codes_equal = single_codes0 == batch_codes0 && single_codes1 == batch_codes1;
        std::fprintf(stderr,
                     "higgs_audio_head_batch_check logits_equal=%d codes_equal=%d single_ms=%lld batch_ms=%lld\n",
                     logits_equal ? 1 : 0,
                     codes_equal ? 1 : 0,
                     static_cast<long long>(single0_ms + single1_ms),
                     static_cast<long long>(batch_ms));
        if (!logits_equal || !codes_equal) {
            throw std::runtime_error("audio head batch diagnostic mismatch");
        }
    }
}

std::vector<float> run_audio_head_logits_vc_for_frame(const std::string & model_path, const std::vector<float> & hidden_values, int frames, int frame) {
    const TextConfig cfg = inspect_text_config(model_path);
    if (frames <= 0 || frame < 0 || frame >= frames || hidden_values.size() != static_cast<size_t>(cfg.embedding_length * frames)) {
        throw std::runtime_error("audio head input hidden shape mismatch");
    }
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, {"output_norm.weight", "a.output"})) {
        throw std::runtime_error("failed to load audio head tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate audio head graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, frames);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
    ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
    ggml_tensor * logits_flat = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
    if (!trace_json_path().empty()) {
        ggml_set_output(final_hidden);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, logits_flat);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate audio head graph tensors");
    }
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("audio head graph compute failed");
    }
    if (logits_flat->ne[0] != k_num_codebooks * k_codebook_vocab_size || logits_flat->ne[1] != frames) {
        throw std::runtime_error("audio head graph output shape mismatch");
    }
    std::vector<float> logits(static_cast<size_t>(logits_flat->ne[0] * logits_flat->ne[1]));
    ggml_backend_tensor_get(logits_flat, logits.data(), 0, logits.size() * sizeof(float));
    if (!trace_json_path().empty()) {
        std::vector<float> final_values(static_cast<size_t>(final_hidden->ne[0] * final_hidden->ne[1]));
        ggml_backend_tensor_get(final_hidden, final_values.data(), 0, final_values.size() * sizeof(float));
        dump_node_values("cpp_audio_head_final_hidden", final_values);
        dump_node_values("cpp_audio_head_logits_flat", logits);
        g_trace_node_stats["final_hidden"] = make_last_token_stat(final_values, cfg.embedding_length, frames);
        g_trace_node_stats["audio_logits_flat"] = make_last_token_stat(logits, k_num_codebooks * k_codebook_vocab_size, frames);
        g_trace_node_stats["audio_logits_vc"] = make_last_token_stat(audio_logits_flat_to_vc(logits, frames, frame), k_num_codebooks * k_codebook_vocab_size, 1);
    }
    return audio_logits_flat_to_vc(logits, frames, frame);
}

std::vector<float> run_audio_head_logits_vc(const std::string & model_path, const std::vector<float> & hidden_values) {
    return run_audio_head_logits_vc_for_frame(model_path, hidden_values, 1, 0);
}

std::vector<int32_t> run_audio_head_sample(const std::string & model_path, const std::vector<float> & hidden_values, const GenerateOptions & options) {
    const std::vector<float> vc = run_audio_head_logits_vc(model_path, hidden_values);
    uint32_t rng_state = options.seed;
    return sample_codes_from_logits_vc(vc, k_codebook_vocab_size, k_num_codebooks, options.temperature, options.top_k, options.top_p, rng_state, options.stop_on_eoc);
}

std::vector<float> run_audio_head_logits_vc_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, BackendKvCache & cache) {
    const TextConfig & cfg = cache.cfg;
    if (hidden_values.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("audio head cached input hidden shape mismatch");
    }
    const std::vector<std::string> names{"output_norm.weight", "a.output"};
    if (!ensure_backend_cache_weights(model_path, cache, names)) {
        throw std::runtime_error("failed to load cached audio head tensors from GGUF");
    }
    ggml_context * weights_ctx = cache.weights_ggml;
    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate cached audio head graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
    ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
    ggml_tensor * logits_flat = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, logits_flat);
    BackendBufferPtr graph_buffer;
    {
        ScopedProfileTimer timer("reference_ar_audio_head_alloc_graph");
        graph_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
    }
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate cached audio head graph tensors");
    }
    {
        ScopedProfileTimer timer("reference_ar_audio_head_set_inputs");
        ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    }
    {
        ScopedProfileTimer timer("reference_ar_audio_head_compute_graph");
        if (ggml_backend_graph_compute(cache.backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("cached audio head graph compute failed");
        }
    }
    std::vector<float> logits(static_cast<size_t>(logits_flat->ne[0]));
    {
        ScopedProfileTimer timer("reference_ar_audio_head_get_logits");
        ggml_backend_tensor_get(logits_flat, logits.data(), 0, logits.size() * sizeof(float));
    }
    if (!trace_json_path().empty()) {
        std::vector<float> final_values(static_cast<size_t>(final_hidden->ne[0]));
        ggml_backend_tensor_get(final_hidden, final_values.data(), 0, final_values.size() * sizeof(float));
        const std::string dump_suffix = "_used" + std::to_string(cache.used);
        dump_node_values("cpp_audio_head_final_hidden" + dump_suffix, final_values);
        dump_node_values("cpp_audio_head_logits_flat" + dump_suffix, logits);
        g_trace_node_stats["final_hidden"] = make_last_token_stat(final_values, cfg.embedding_length, 1);
        g_trace_node_stats["audio_logits_flat"] = make_last_token_stat(logits, static_cast<int64_t>(logits.size()), 1);
        g_trace_node_stats["audio_logits_vc"] = make_last_token_stat(audio_logits_flat_to_vc(logits, 1, 0), static_cast<int64_t>(logits.size()), 1);
    }
    return audio_logits_flat_to_vc(logits, 1, 0);
}

std::vector<float> run_audio_head_logits_flat_batched_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, int frames, BackendKvCache & cache) {
    const TextConfig & cfg = cache.cfg;
    if (frames <= 0 || hidden_values.size() != static_cast<size_t>(cfg.embedding_length * frames)) {
        throw std::runtime_error("cached audio head batched input hidden shape mismatch");
    }
    const std::vector<std::string> names{"output_norm.weight", "a.output"};
    if (!ensure_backend_cache_weights(model_path, cache, names)) {
        throw std::runtime_error("failed to load cached batched audio head tensors from GGUF");
    }
    ggml_context * weights_ctx = cache.weights_ggml;
    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate cached batched audio head graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, frames);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
    ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
    ggml_tensor * logits_flat = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
    if (!trace_json_path().empty()) {
        ggml_set_output(final_hidden);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, logits_flat);
    BackendBufferPtr graph_buffer;
    {
        ScopedProfileTimer timer("reference_ar_audio_head_alloc_graph");
        graph_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
    }
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate cached batched audio head graph tensors");
    }
    {
        ScopedProfileTimer timer("reference_ar_audio_head_set_inputs");
        ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    }
    {
        ScopedProfileTimer timer("reference_ar_audio_head_compute_graph");
        if (ggml_backend_graph_compute(cache.backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("cached batched audio head graph compute failed");
        }
    }
    if (logits_flat->ne[0] != k_num_codebooks * k_codebook_vocab_size || logits_flat->ne[1] != frames) {
        throw std::runtime_error("cached batched audio head graph output shape mismatch");
    }
    std::vector<float> logits(static_cast<size_t>(logits_flat->ne[0] * logits_flat->ne[1]));
    {
        ScopedProfileTimer timer("reference_ar_audio_head_get_logits");
        ggml_backend_tensor_get(logits_flat, logits.data(), 0, logits.size() * sizeof(float));
    }
    if (!trace_json_path().empty()) {
        std::vector<float> final_values(static_cast<size_t>(final_hidden->ne[0] * final_hidden->ne[1]));
        ggml_backend_tensor_get(final_hidden, final_values.data(), 0, final_values.size() * sizeof(float));
        dump_node_values("cpp_audio_head_final_hidden", final_values);
        dump_node_values("cpp_audio_head_logits_flat", logits);
        g_trace_node_stats["final_hidden"] = make_last_token_stat(final_values, cfg.embedding_length, frames);
        g_trace_node_stats["audio_logits_flat"] = make_last_token_stat(logits, k_num_codebooks * k_codebook_vocab_size, frames);
    }
    return logits;
}

std::vector<float> run_audio_head_logits_vc_for_frame_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, int frames, int frame, BackendKvCache & cache) {
    const TextConfig & cfg = cache.cfg;
    if (frames <= 0 || frame < 0 || frame >= frames || hidden_values.size() != static_cast<size_t>(cfg.embedding_length * frames)) {
        throw std::runtime_error("cached audio head batched input hidden shape mismatch");
    }
    const std::vector<std::string> names{"output_norm.weight", "a.output"};
    if (!ensure_backend_cache_weights(model_path, cache, names)) {
        throw std::runtime_error("failed to load cached batched audio head tensors from GGUF");
    }
    ggml_context * weights_ctx = cache.weights_ggml;
    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate cached batched audio head graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, frames);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
    ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
    ggml_tensor * logits_flat = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
    if (!trace_json_path().empty()) {
        ggml_set_output(final_hidden);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, logits_flat);
    BackendBufferPtr graph_buffer;
    {
        ScopedProfileTimer timer("reference_ar_audio_head_alloc_graph");
        graph_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
    }
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate cached batched audio head graph tensors");
    }
    {
        ScopedProfileTimer timer("reference_ar_audio_head_set_inputs");
        ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    }
    {
        ScopedProfileTimer timer("reference_ar_audio_head_compute_graph");
        if (ggml_backend_graph_compute(cache.backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("cached batched audio head graph compute failed");
        }
    }
    if (logits_flat->ne[0] != k_num_codebooks * k_codebook_vocab_size || logits_flat->ne[1] != frames) {
        throw std::runtime_error("cached batched audio head graph output shape mismatch");
    }
    const size_t fused = static_cast<size_t>(k_num_codebooks * k_codebook_vocab_size);
    std::vector<float> logits(fused);
    {
        ScopedProfileTimer timer("reference_ar_audio_head_get_logits");
        ggml_backend_tensor_get(logits_flat, logits.data(), static_cast<size_t>(frame) * fused * sizeof(float), logits.size() * sizeof(float));
    }
    if (!trace_json_path().empty()) {
        std::vector<float> logits_all(static_cast<size_t>(logits_flat->ne[0] * logits_flat->ne[1]));
        ggml_backend_tensor_get(logits_flat, logits_all.data(), 0, logits_all.size() * sizeof(float));
        std::vector<float> final_values(static_cast<size_t>(final_hidden->ne[0] * final_hidden->ne[1]));
        ggml_backend_tensor_get(final_hidden, final_values.data(), 0, final_values.size() * sizeof(float));
        dump_node_values("cpp_audio_head_final_hidden", final_values);
        dump_node_values("cpp_audio_head_logits_flat", logits_all);
        g_trace_node_stats["final_hidden"] = make_last_token_stat(final_values, cfg.embedding_length, frames);
        g_trace_node_stats["audio_logits_flat"] = make_last_token_stat(logits_all, k_num_codebooks * k_codebook_vocab_size, frames);
        g_trace_node_stats["audio_logits_vc"] = make_last_token_stat(audio_logits_flat_to_vc(logits_all, frames, frame), k_num_codebooks * k_codebook_vocab_size, 1);
    }
    return audio_logits_flat_to_vc(logits, 1, 0);
}

class AudioHeadBatcher {
public:
    std::vector<float> run(const std::string & model_path, int step, const std::vector<float> & hidden, BackendKvCache & cache, ReferenceArRuntimeStats * stats) {
        if (!server_audio_head_batch_enabled() || !multiple_reference_ar_requests_active() || !trace_json_path().empty() || !node_dump_dir().empty()) {
            if (stats != nullptr) {
                ++stats->audio_head_fallback_calls;
            }
            return run_single(model_path, hidden, cache, stats);
        }
        Request self;
        self.model_path = model_path;
        self.step = step;
        self.hidden = hidden;
        self.cache = &cache;
        Request * peer = nullptr;
        {
            std::unique_lock<std::mutex> lock(mu_);
            if (waiting_ == nullptr) {
                waiting_ = &self;
                cv_.wait_for(lock, std::chrono::milliseconds(server_audio_head_batch_wait_ms()), [&] { return self.done; });
                if (!self.done && waiting_ == &self) {
                    waiting_ = nullptr;
                }
            } else if (waiting_->model_path == model_path && waiting_->step == step && waiting_->cache != &cache && waiting_->hidden.size() == hidden.size()) {
                peer = waiting_;
                waiting_ = nullptr;
            }
        }
        if (peer != nullptr) {
            std::vector<float> batched;
            batched.reserve(peer->hidden.size() + self.hidden.size());
            batched.insert(batched.end(), peer->hidden.begin(), peer->hidden.end());
            batched.insert(batched.end(), self.hidden.begin(), self.hidden.end());
            try {
                const CudaExecutorRunStats exec_stats = cuda_executor().run_sync([&] {
                    const std::vector<float> flat = run_audio_head_logits_flat_batched_with_cache(model_path, batched, 2, cache);
                    peer->logits_vc = audio_logits_flat_to_vc(flat, 2, 0);
                    self.logits_vc = audio_logits_flat_to_vc(flat, 2, 1);
                });
                add_cuda_executor_stats(stats, exec_stats);
                if (stats != nullptr) {
                    ++stats->audio_head_batch_calls;
                    stats->audio_head_batch_items += 2;
                }
            } catch (...) {
                peer->error = std::current_exception();
                self.error = peer->error;
            }
            {
                std::lock_guard<std::mutex> lock(mu_);
                peer->done = true;
                self.done = true;
            }
            cv_.notify_all();
        }
        if (!self.done) {
            if (stats != nullptr) {
                ++stats->audio_head_fallback_calls;
            }
            return run_single(model_path, hidden, cache, stats);
        }
        if (self.error) {
            std::rethrow_exception(self.error);
        }
        return self.logits_vc;
    }

private:
    struct Request {
        std::string model_path;
        int step = -1;
        std::vector<float> hidden;
        BackendKvCache * cache = nullptr;
        std::vector<float> logits_vc;
        bool done = false;
        std::exception_ptr error;
    };

    std::vector<float> run_single(const std::string & model_path, const std::vector<float> & hidden, BackendKvCache & cache, ReferenceArRuntimeStats * stats) {
        std::vector<float> logits;
        const CudaExecutorRunStats exec_stats = cuda_executor().run_sync([&] {
            logits = run_audio_head_logits_vc_with_cache(model_path, hidden, cache);
        });
        add_cuda_executor_stats(stats, exec_stats);
        return logits;
    }

    std::mutex mu_;
    std::condition_variable cv_;
    Request * waiting_ = nullptr;
};

std::vector<float> audio_head_batcher_run(const std::string & model_path, int step, const std::vector<float> & hidden, BackendKvCache & cache, ReferenceArRuntimeStats * stats) {
    static AudioHeadBatcher batcher;
    return batcher.run(model_path, step, hidden, cache, stats);
}

std::vector<int32_t> run_audio_head_sample_with_cache(const std::string & model_path, const std::vector<float> & hidden_values, const GenerateOptions & options, BackendKvCache & cache) {
    const std::vector<float> vc = run_audio_head_logits_vc_with_cache(model_path, hidden_values, cache);
    uint32_t rng_state = options.seed;
    return sample_codes_from_logits_vc(vc, k_codebook_vocab_size, k_num_codebooks, options.temperature, options.top_k, options.top_p, rng_state, options.stop_on_eoc);
}

void self_check_block0_mlp_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    const std::vector<std::string> names{
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    };
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load block0 MLP tensors from GGUF");
    }

    int tokens = 2;
    if (const char * value = std::getenv("HIGGS_ATTENTION_TOKENS")) {
        tokens = std::stoi(value);
        if (tokens <= 0) {
            throw std::runtime_error("HIGGS_ATTENTION_TOKENS must be positive");
        }
    }
    ggml_init_params params{};
    params.mem_size = 128 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate block0 MLP graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.ffn_norm.weight"), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, norm_weight);
    ggml_tensor * gate = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.ffn_gate.weight"), normed);
    ggml_tensor * up = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.ffn_up.weight"), normed);
    ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.ffn_down.weight"), gated);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate block0 MLP graph tensors");
    }
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length * tokens));
    for (int t = 0; t < tokens; ++t) {
        for (int64_t f = 0; f < cfg.embedding_length; ++f) {
            const int64_t numpy_c_index = f;
            hidden_values[static_cast<size_t>(t) * static_cast<size_t>(cfg.embedding_length) + static_cast<size_t>(f)] =
                static_cast<float>(static_cast<int>(numpy_c_index % 127)) / 63.5f - 1.0f;
        }
    }
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("block0 MLP graph compute failed");
    }
    if (out->ne[0] != cfg.embedding_length || out->ne[1] != tokens) {
        throw std::runtime_error("block0 MLP graph output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float v : values) {
        if (!std::isfinite(v)) {
            throw std::runtime_error("block0 MLP graph output contains non-finite values");
        }
    }
}

void self_check_block0_attn_qkv_graph(const std::string & model_path, BackendKind backend_kind) {
    BackendScope backend_scope(backend_kind);
    const TextConfig cfg = inspect_text_config(model_path);
    if (cfg.head_count <= 0 || cfg.head_count_kv <= 0 || cfg.head_count % cfg.head_count_kv != 0) {
        throw std::runtime_error("block0 attention head config mismatch");
    }
    if (cfg.key_length != cfg.value_length || cfg.key_length != cfg.rope_dimension_count) {
        throw std::runtime_error("block0 attention head dimension mismatch");
    }
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    int tokens = 2;
    if (const char * value = std::getenv("HIGGS_ATTENTION_TOKENS")) {
        tokens = std::stoi(value);
    }
    if (tokens <= 0) {
        throw std::runtime_error("HIGGS_ATTENTION_TOKENS must be positive");
    }

    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    const std::vector<std::string> names{
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
    };
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load block0 attention tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 256 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate block0 attention graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    const bool input_is_normed = !attention_normed_input_path().empty();
    ggml_tensor * normed = hidden;
    if (!input_is_normed) {
        normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
        ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_norm.weight"), cfg.embedding_length, 1);
        normed = ggml_mul(ctx.get(), normed, attn_norm);
    }

    ggml_tensor * q = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_q.weight"), normed);
    ggml_tensor * q_proj = q;
    const auto read_hidden_values = [&]() {
        std::vector<float> values(static_cast<size_t>(cfg.embedding_length * tokens));
        if (input_is_normed) {
            std::ifstream in(attention_normed_input_path(), std::ios::binary);
            if (!in) {
                throw std::runtime_error("failed to open HIGGS_ATTENTION_NORMED_INPUT");
            }
            in.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
            const std::streamsize got = in.gcount();
            if (got == static_cast<std::streamsize>(values.size() * sizeof(float))) {
                // Full prefix supplied in GGML order `(ne0=embedding, ne1=tokens)`.
            } else if (got == static_cast<std::streamsize>(static_cast<size_t>(cfg.embedding_length) * sizeof(float))) {
                const std::vector<float> one(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(cfg.embedding_length));
                for (int t = 1; t < tokens; ++t) {
                    std::copy(one.begin(), one.end(), values.begin() + static_cast<std::ptrdiff_t>(t * cfg.embedding_length));
                }
            } else {
                throw std::runtime_error("failed to read HIGGS_ATTENTION_NORMED_INPUT");
            }
        } else {
            for (int t = 0; t < tokens; ++t) {
                for (int64_t f = 0; f < cfg.embedding_length; ++f) {
                    const int64_t numpy_c_index = f;
                    values[static_cast<size_t>(t) * static_cast<size_t>(cfg.embedding_length) + static_cast<size_t>(f)] =
                        static_cast<float>(static_cast<int>(numpy_c_index % 127)) / 63.5f - 1.0f;
                }
            }
        }
        return values;
    };
    if (attention_q_only_enabled()) {
        ggml_cgraph * graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, q_proj);
        BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!graph_buffer) {
            throw std::runtime_error("failed to allocate block0 q_proj graph tensors");
        }
        std::vector<float> hidden_values = read_hidden_values();
        ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("block0 q_proj graph compute failed");
        }
        if (const std::string path = attention_dump_path(); !path.empty()) {
            const size_t n = static_cast<size_t>(ggml_nelements(q_proj));
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(q_proj, tmp.data(), 0, tmp.size() * sizeof(float));
            std::ofstream part(path + ".q_proj", std::ios::binary);
            if (!part) {
                throw std::runtime_error("failed to open q_proj tensor dump path");
            }
            part.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(float)));
        }
        return;
    }
    ggml_tensor * k = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_k.weight"), normed);
    ggml_tensor * v = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_v.weight"), normed);
    ggml_tensor * k_proj = k;
    ggml_tensor * v_proj = v;
    if (!attention_dump_path().empty()) {
        ggml_set_output(q_proj);
        ggml_set_output(k_proj);
        ggml_set_output(v_proj);
    }
    q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, tokens);
    k = ggml_reshape_3d(ctx.get(), k, head_dim, n_head_kv, tokens);
    v = ggml_reshape_3d(ctx.get(), v, head_dim, n_head_kv, tokens);

    ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_q_norm.weight"), head_dim, 1, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_k_norm.weight"), head_dim, 1, 1);
    q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
    k = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k, cfg.rms_norm_eps), k_norm);
    q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx.get(), k, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    if (!attention_dump_path().empty()) {
        ggml_set_output(q);
        ggml_set_output(k);
        ggml_set_output(v);
    }
    k = repeat_kv_3d(ctx.get(), k, n_rep);
    v = repeat_kv_3d(ctx.get(), v, n_rep);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, q);
    ggml_build_forward_expand(graph, k);
    ggml_build_forward_expand(graph, v);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate block0 attention graph tensors");
    }
    std::vector<float> hidden_values = read_hidden_values();
    std::vector<int32_t> position_values(static_cast<size_t>(tokens));
    std::iota(position_values.begin(), position_values.end(), 0);
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("block0 attention QKV graph compute failed");
    }
    const auto check = [&](ggml_tensor * tensor, const char * name) {
        if (tensor->ne[0] != head_dim || tensor->ne[1] != n_head || tensor->ne[2] != tokens) {
            throw std::runtime_error(std::string(name) + " shape mismatch");
        }
        std::vector<float> values(static_cast<size_t>(tensor->ne[0] * tensor->ne[1] * tensor->ne[2]));
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        for (float value : values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string(name) + " contains non-finite values");
            }
        }
    };
    check(q, "block0 attention q");
    check(k, "block0 attention k");
    check(v, "block0 attention v");
    if (const std::string path = attention_dump_path(); !path.empty()) {
        const auto dump_tensor = [&](const std::string & suffix, ggml_tensor * tensor) {
            const size_t n = static_cast<size_t>(ggml_nelements(tensor));
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(tensor, tmp.data(), 0, tmp.size() * sizeof(float));
            std::ofstream part(path + suffix, std::ios::binary);
            if (!part) {
                throw std::runtime_error("failed to open QKV tensor dump path");
            }
            part.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(float)));
        };
        dump_tensor(".q_proj", q_proj);
        dump_tensor(".k_proj", k_proj);
        dump_tensor(".v_proj", v_proj);
        dump_tensor(".q", q);
        dump_tensor(".k", k);
        dump_tensor(".v", v);
    }
}

void self_check_block0_attn_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    if (cfg.head_count <= 0 || cfg.head_count_kv <= 0 || cfg.head_count % cfg.head_count_kv != 0) {
        throw std::runtime_error("block0 attention head config mismatch");
    }
    if (cfg.key_length != cfg.value_length || cfg.key_length != cfg.rope_dimension_count) {
        throw std::runtime_error("block0 attention head dimension mismatch");
    }
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    int tokens = 2;
    if (const char * value = std::getenv("HIGGS_ATTENTION_TOKENS")) {
        tokens = std::stoi(value);
        if (tokens <= 0) {
            throw std::runtime_error("HIGGS_ATTENTION_TOKENS must be positive");
        }
    }

    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    const std::vector<std::string> names{
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
    };
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load block0 attention tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 512 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate block0 attention graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_norm.weight"), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, attn_norm);

    ggml_tensor * q = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_q.weight"), normed);
    ggml_tensor * k = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_k.weight"), normed);
    ggml_tensor * v = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_v.weight"), normed);
    q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, tokens);
    k = ggml_reshape_3d(ctx.get(), k, head_dim, n_head_kv, tokens);
    v = ggml_reshape_3d(ctx.get(), v, head_dim, n_head_kv, tokens);

    ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_q_norm.weight"), head_dim, 1, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_k_norm.weight"), head_dim, 1, 1);
    q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
    k = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k, cfg.rms_norm_eps), k_norm);
    q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx.get(), k, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    k = repeat_kv_3d(ctx.get(), k, n_rep);
    v = repeat_kv_3d(ctx.get(), v, n_rep);

    ggml_tensor * q4 = ggml_permute(ctx.get(), as_4d(ctx.get(), q), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx.get(), as_4d(ctx.get(), k), 0, 2, 1, 3);
    ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
    kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    kq = ggml_diag_mask_inf(ctx.get(), kq, 0);
    kq = ggml_soft_max(ctx.get(), kq);
    ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), as_4d(ctx.get(), v), 1, 2, 0, 3));
    ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
    ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, tokens);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "blk.0.attn_output.weight"), attn);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate block0 attention graph tensors");
    }
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length * tokens));
    for (int t = 0; t < tokens; ++t) {
        for (int64_t f = 0; f < cfg.embedding_length; ++f) {
            const int64_t numpy_c_index = f * tokens + t;
            hidden_values[static_cast<size_t>(t) * static_cast<size_t>(cfg.embedding_length) + static_cast<size_t>(f)] =
                static_cast<float>(static_cast<int>(numpy_c_index % 127)) / 63.5f - 1.0f;
        }
    }
    std::vector<int32_t> position_values(static_cast<size_t>(tokens));
    std::iota(position_values.begin(), position_values.end(), 0);
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("block0 attention graph compute failed");
    }
    if (out->ne[0] != cfg.embedding_length || out->ne[1] != tokens) {
        throw std::runtime_error("block0 attention output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    if (const std::string path = attention_dump_path(); !path.empty()) {
        std::ofstream dump(path, std::ios::binary);
        if (!dump) {
            throw std::runtime_error("failed to open attention dump path");
        }
        dump.write(reinterpret_cast<const char *>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
        const auto dump_tensor = [&](const std::string & suffix, ggml_tensor * tensor) {
            const size_t n = static_cast<size_t>(ggml_nelements(tensor));
            std::vector<float> tmp(n);
            ggml_backend_tensor_get(tensor, tmp.data(), 0, tmp.size() * sizeof(float));
            std::ofstream part(path + suffix, std::ios::binary);
            if (!part) {
                throw std::runtime_error("failed to open attention tensor dump path");
            }
            part.write(reinterpret_cast<const char *>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(float)));
        };
        dump_tensor(".q", q);
        dump_tensor(".k", k);
        dump_tensor(".v", v);
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("block0 attention output contains non-finite values");
        }
    }
}

std::vector<float> run_block_graph_layer(const std::string & model_path, int layer, const std::vector<float> & hidden_values) {
    const TextConfig cfg = inspect_text_config(model_path);
    if (layer < 0 || layer >= cfg.block_count) {
        throw std::runtime_error("block layer index out of range");
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    if (cfg.head_count <= 0 || cfg.head_count_kv <= 0 || cfg.head_count % cfg.head_count_kv != 0) {
        throw std::runtime_error("block head config mismatch");
    }
    if (cfg.key_length != cfg.value_length || cfg.key_length != cfg.rope_dimension_count) {
        throw std::runtime_error("block head dimension mismatch");
    }
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    if (hidden_values.size() % static_cast<size_t>(cfg.embedding_length) != 0) {
        throw std::runtime_error("block input hidden shape mismatch");
    }
    const int tokens = static_cast<int>(hidden_values.size() / static_cast<size_t>(cfg.embedding_length));
    if (tokens <= 0) {
        throw std::runtime_error("block input token count mismatch");
    }

    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    const std::vector<std::string> names{
        prefix + "attn_norm.weight",
        prefix + "attn_q.weight",
        prefix + "attn_k.weight",
        prefix + "attn_v.weight",
        prefix + "attn_output.weight",
        prefix + "attn_q_norm.weight",
        prefix + "attn_k_norm.weight",
        prefix + "ffn_norm.weight",
        prefix + "ffn_gate.weight",
        prefix + "ffn_up.weight",
        prefix + "ffn_down.weight",
    };
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load block tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 768 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate block graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, attn_norm);

    ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
    ggml_tensor * q_proj = q;
    ggml_tensor * k = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
    ggml_tensor * v = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
    ggml_tensor * v_proj = v;
    q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, tokens);
    k = ggml_reshape_3d(ctx.get(), k, head_dim, n_head_kv, tokens);
    v = ggml_reshape_3d(ctx.get(), v, head_dim, n_head_kv, tokens);

    ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
    q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
    ggml_tensor * q_normed = q;
    k = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k, cfg.rms_norm_eps), k_norm);
    q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx.get(), k, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * k_raw = k;
    ggml_tensor * v_raw = v;
    if (layer == 0 && reference_full_diagnostic_enabled()) {
        ggml_set_output(normed);
        ggml_set_output(q_proj);
        ggml_set_output(v_proj);
        ggml_set_output(q_normed);
        ggml_set_output(k_raw);
        ggml_set_output(v_raw);
    }

    k = repeat_kv_3d(ctx.get(), k, n_rep);
    v = repeat_kv_3d(ctx.get(), v, n_rep);

    ggml_tensor * q4 = ggml_permute(ctx.get(), as_4d(ctx.get(), q), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx.get(), as_4d(ctx.get(), k), 0, 2, 1, 3);
    ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
    kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    kq = ggml_diag_mask_inf(ctx.get(), kq, 0);
    ggml_tensor * kq_masked = kq;
    if (layer == 0 && reference_full_diagnostic_enabled()) {
        ggml_set_output(q);
        ggml_set_output(k);
        ggml_set_output(v);
        ggml_set_output(kq_masked);
    }
    kq = ggml_soft_max(ctx.get(), kq);
    if (layer == 0 && reference_full_diagnostic_enabled()) {
        ggml_set_output(kq);
    }
    ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), as_4d(ctx.get(), v), 1, 2, 0, 3));
    ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
    ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, tokens);
    ggml_tensor * attn_merged = attn;
    ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
    ggml_tensor * cur = ggml_add(ctx.get(), hidden, attn_out);
    if (layer == 0 && reference_full_diagnostic_enabled()) {
        ggml_set_output(kqv);
        ggml_set_output(attn_merged);
        ggml_set_output(attn_out);
        ggml_set_output(cur);
    }

    ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
    ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
    mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
    ggml_tensor * gate = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
    ggml_tensor * up = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
    ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
    ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
    ggml_tensor * out = ggml_add(ctx.get(), cur, mlp);
    if (layer == 0 && reference_full_diagnostic_enabled()) {
        ggml_set_output(mlp_normed);
        ggml_set_output(gate);
        ggml_set_output(up);
        ggml_set_output(gated);
        ggml_set_output(mlp);
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate block graph tensors");
    }
    std::vector<int32_t> position_values(static_cast<size_t>(tokens));
    std::iota(position_values.begin(), position_values.end(), 0);
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("block graph compute failed");
    }
    if (out->ne[0] != cfg.embedding_length || out->ne[1] != tokens) {
        throw std::runtime_error("block output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    if (layer == 0 && reference_full_diagnostic_enabled()) {
        std::vector<float> q_values(static_cast<size_t>(q->ne[0] * q->ne[1] * q->ne[2]));
        std::vector<float> normed_values(static_cast<size_t>(normed->ne[0] * normed->ne[1]));
        std::vector<float> q_proj_values(static_cast<size_t>(head_dim * n_head * tokens));
        std::vector<float> v_proj_values(static_cast<size_t>(head_dim * n_head_kv * tokens));
        std::vector<float> q_normed_values(static_cast<size_t>(q_normed->ne[0] * q_normed->ne[1] * q_normed->ne[2]));
        std::vector<float> k_raw_values(static_cast<size_t>(k_raw->ne[0] * k_raw->ne[1] * k_raw->ne[2]));
        std::vector<float> v_raw_values(static_cast<size_t>(v_raw->ne[0] * v_raw->ne[1] * v_raw->ne[2]));
        std::vector<float> k_values(static_cast<size_t>(k->ne[0] * k->ne[1] * k->ne[2]));
        std::vector<float> v_values(static_cast<size_t>(v->ne[0] * v->ne[1] * v->ne[2]));
        std::vector<float> kq_values(static_cast<size_t>(kq_masked->ne[0] * kq_masked->ne[1] * kq_masked->ne[2] * kq_masked->ne[3]));
        std::vector<float> softmax_values(static_cast<size_t>(kq->ne[0] * kq->ne[1] * kq->ne[2] * kq->ne[3]));
        std::vector<float> kqv_values(static_cast<size_t>(ggml_nelements(kqv)));
        std::vector<float> attn_merged_values(static_cast<size_t>(attn_merged->ne[0] * attn_merged->ne[1]));
        std::vector<float> attn_values(static_cast<size_t>(attn_out->ne[0] * attn_out->ne[1]));
        std::vector<float> cur_values(static_cast<size_t>(cur->ne[0] * cur->ne[1]));
        std::vector<float> mlp_normed_values(static_cast<size_t>(mlp_normed->ne[0] * mlp_normed->ne[1]));
        std::vector<float> gate_values(static_cast<size_t>(gate->ne[0] * gate->ne[1]));
        std::vector<float> up_values(static_cast<size_t>(up->ne[0] * up->ne[1]));
        std::vector<float> gated_values(static_cast<size_t>(gated->ne[0] * gated->ne[1]));
        std::vector<float> mlp_values(static_cast<size_t>(mlp->ne[0] * mlp->ne[1]));
        ggml_backend_tensor_get(normed, normed_values.data(), 0, normed_values.size() * sizeof(float));
        ggml_backend_tensor_get(q_proj, q_proj_values.data(), 0, q_proj_values.size() * sizeof(float));
        ggml_backend_tensor_get(v_proj, v_proj_values.data(), 0, v_proj_values.size() * sizeof(float));
        ggml_backend_tensor_get(q_normed, q_normed_values.data(), 0, q_normed_values.size() * sizeof(float));
        ggml_backend_tensor_get(q, q_values.data(), 0, q_values.size() * sizeof(float));
        ggml_backend_tensor_get(k_raw, k_raw_values.data(), 0, k_raw_values.size() * sizeof(float));
        ggml_backend_tensor_get(v_raw, v_raw_values.data(), 0, v_raw_values.size() * sizeof(float));
        ggml_backend_tensor_get(k, k_values.data(), 0, k_values.size() * sizeof(float));
        ggml_backend_tensor_get(v, v_values.data(), 0, v_values.size() * sizeof(float));
        ggml_backend_tensor_get(kq_masked, kq_values.data(), 0, kq_values.size() * sizeof(float));
        ggml_backend_tensor_get(kq, softmax_values.data(), 0, softmax_values.size() * sizeof(float));
        ggml_backend_tensor_get(kqv, kqv_values.data(), 0, kqv_values.size() * sizeof(float));
        ggml_backend_tensor_get(attn_merged, attn_merged_values.data(), 0, attn_merged_values.size() * sizeof(float));
        ggml_backend_tensor_get(attn_out, attn_values.data(), 0, attn_values.size() * sizeof(float));
        ggml_backend_tensor_get(cur, cur_values.data(), 0, cur_values.size() * sizeof(float));
        ggml_backend_tensor_get(mlp_normed, mlp_normed_values.data(), 0, mlp_normed_values.size() * sizeof(float));
        ggml_backend_tensor_get(gate, gate_values.data(), 0, gate_values.size() * sizeof(float));
        ggml_backend_tensor_get(up, up_values.data(), 0, up_values.size() * sizeof(float));
        ggml_backend_tensor_get(gated, gated_values.data(), 0, gated_values.size() * sizeof(float));
        ggml_backend_tensor_get(mlp, mlp_values.data(), 0, mlp_values.size() * sizeof(float));
        dump_node_values(reference_step_node_name("cpp_blk0_q_state"), q_values);
        dump_node_values(reference_step_node_name("cpp_blk0_attn_normed"), normed_values);
        dump_node_values(reference_step_node_name("cpp_blk0_q_proj"), q_proj_values);
        dump_node_values(reference_step_node_name("cpp_blk0_v_proj"), v_proj_values);
        dump_node_values(reference_step_node_name("cpp_blk0_q_normed"), q_normed_values);
        dump_node_values(reference_step_node_name("cpp_blk0_k_raw"), k_raw_values);
        dump_node_values(reference_step_node_name("cpp_blk0_v_raw"), v_raw_values);
        dump_node_values(reference_step_node_name("cpp_blk0_k_state"), k_values);
        dump_node_values(reference_step_node_name("cpp_blk0_kq"), kq_values);
        dump_node_values(reference_step_node_name("cpp_blk0_kqv"), kqv_values);
        dump_node_values(reference_step_node_name("cpp_blk0_attn_merged"), attn_merged_values);
        dump_node_values(reference_step_node_name("cpp_blk0_attn_out"), attn_values);
        dump_node_values(reference_step_node_name("cpp_blk0_post_attn"), cur_values);
        dump_node_values(reference_step_node_name("cpp_blk0_softmax"), softmax_values);
        dump_node_values(reference_step_node_name("cpp_blk0_v_state"), v_values);
        dump_node_values(reference_step_node_name("cpp_blk0_mlp_normed"), mlp_normed_values);
        dump_node_values(reference_step_node_name("cpp_blk0_mlp_gate"), gate_values);
        dump_node_values(reference_step_node_name("cpp_blk0_mlp_up"), up_values);
        dump_node_values(reference_step_node_name("cpp_blk0_mlp_gated"), gated_values);
        dump_node_values(reference_step_node_name("cpp_blk0_mlp_out"), mlp_values);
        dump_node_values(reference_step_node_name("cpp_blk0_hidden"), values);
        g_trace_node_stats["blk.0.attn_normed"] = make_last_token_stat(normed_values, cfg.embedding_length, tokens);
        g_trace_node_stats["blk.0.q_proj"] = make_last_token_stat(q_proj_values, head_dim * n_head, tokens);
        g_trace_node_stats["blk.0.v_proj"] = make_last_token_stat(v_proj_values, head_dim * n_head_kv, tokens);
        g_trace_node_stats["blk.0.q_normed"] = make_last_token_stat(q_normed_values, head_dim * n_head, tokens);
        g_trace_node_stats["blk.0.q_state"] = make_last_token_stat(q_values, head_dim * n_head, tokens);
        g_trace_node_stats["blk.0.k_raw"] = make_last_token_stat(k_raw_values, head_dim * n_head_kv, tokens);
        g_trace_node_stats["blk.0.v_raw"] = make_last_token_stat(v_raw_values, head_dim * n_head_kv, tokens);
        g_trace_node_stats["blk.0.k_state"] = make_last_token_stat(k_values, head_dim * n_head, tokens);
        g_trace_node_stats["blk.0.v_state"] = make_last_token_stat(v_values, head_dim * n_head, tokens);
        g_trace_node_stats["blk.0.kq"] = make_last_token_stat(kq_values, tokens, tokens * n_head);
        g_trace_node_stats["blk.0.softmax"] = make_last_token_stat(softmax_values, tokens, tokens * n_head);
        g_trace_node_stats["blk.0.kqv"] = make_last_token_stat(kqv_values, static_cast<int64_t>(ggml_nelements(kqv)), 1);
        g_trace_node_stats["blk.0.attn_merged"] = make_last_token_stat(attn_merged_values, head_dim * n_head, tokens);
        g_trace_node_stats["blk.0.attn_out"] = make_last_token_stat(attn_values, cfg.embedding_length, tokens);
        g_trace_node_stats["blk.0.post_attn"] = make_last_token_stat(cur_values, cfg.embedding_length, tokens);
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("block output contains non-finite values");
        }
    }
    return values;
}

void self_check_block_graph_layer(const std::string & model_path, int layer) {
    const TextConfig cfg = inspect_text_config(model_path);
    const int tokens = 2;
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length * tokens));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    (void) run_block_graph_layer(model_path, layer, hidden_values);
}

void self_check_block0_graph(const std::string & model_path) {
    self_check_block_graph_layer(model_path, 0);
}

void self_check_block35_graph(const std::string & model_path) {
    self_check_block_graph_layer(model_path, 35);
}

void self_check_backbone_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    const int tokens = 2;
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length * tokens));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_block_graph_layer(model_path, layer, hidden_values);
    }
    for (float value : hidden_values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("backbone output contains non-finite values");
        }
    }
}

void self_check_prompt_backbone_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    const std::vector<int32_t> prompt = build_higgstts_prompt_ids(model_path, "你好");
    std::vector<float> hidden_values = run_embedding_graph(model_path, prompt, nullptr);
    if (hidden_values.size() != static_cast<size_t>(cfg.embedding_length) * prompt.size()) {
        throw std::runtime_error("prompt backbone embedding shape mismatch");
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_block_graph_layer(model_path, layer, hidden_values);
    }
    for (float value : hidden_values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("prompt backbone output contains non-finite values");
        }
    }
}

std::vector<float> run_prompt_last_hidden(const std::string & model_path, const std::string & text, const CodeMatrix * raw_audio, const GenerateOptions & options) {
    const TextConfig cfg = inspect_text_config(model_path);
    GenerateOptions prompt_options = options;
    prompt_options.text = text;
    const std::vector<int32_t> prompt = build_prompt_ids(model_path, prompt_options);
    std::vector<float> hidden_values = run_embedding_graph(model_path, prompt, raw_audio);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_block_graph_layer(model_path, layer, hidden_values);
    }
    std::vector<float> last_token(static_cast<size_t>(cfg.embedding_length));
    const size_t offset = (prompt.size() - 1) * static_cast<size_t>(cfg.embedding_length);
    std::copy(hidden_values.begin() + static_cast<std::ptrdiff_t>(offset),
              hidden_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
              last_token.begin());
    return last_token;
}

std::vector<float> run_prefill_all_layers_with_runtime(const std::string & model_path, const std::vector<float> & hidden_values, BackendRuntime & runtime) {
    ScopedProfileTimer timer("prefill_backbone_all_layers");
    const TextConfig & cfg = *runtime.cfg;
    if (cfg.head_count <= 0 || cfg.head_count_kv <= 0 || cfg.head_count % cfg.head_count_kv != 0) {
        throw std::runtime_error("prefill head config mismatch");
    }
    if (cfg.key_length != cfg.value_length || cfg.key_length != cfg.rope_dimension_count) {
        throw std::runtime_error("prefill head dimension mismatch");
    }
    if (hidden_values.size() % static_cast<size_t>(cfg.embedding_length) != 0) {
        throw std::runtime_error("prefill input hidden shape mismatch");
    }
    const int64_t tokens = static_cast<int64_t>(hidden_values.size() / static_cast<size_t>(cfg.embedding_length));
    if (tokens <= 0) {
        throw std::runtime_error("prefill input token count mismatch");
    }
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;

    const std::vector<std::string> names = all_block_weight_names(cfg);
    if (!ensure_backend_runtime_weights(model_path, runtime, names)) {
        throw std::runtime_error("failed to load runtime prefill tensors from GGUF");
    }
    ggml_context * weights_ctx = runtime.weights_ggml.get();

    ggml_init_params params{};
    params.mem_size = 4ull * 1024ull * 1024ull * 1024ull;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate runtime prefill graph context");
    }

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * cur_hidden = hidden;
    std::vector<ggml_tensor *> layer_hidden_outputs(static_cast<size_t>(cfg.block_count), nullptr);
    ggml_tensor * layer0_attn_normed = nullptr;
    ggml_tensor * layer0_q_proj = nullptr;
    ggml_tensor * layer0_q_normed = nullptr;
    ggml_tensor * layer0_k_cur = nullptr;
    ggml_tensor * layer0_v_cur = nullptr;
    ggml_tensor * layer0_kq = nullptr;
    ggml_tensor * layer0_softmax = nullptr;
    ggml_tensor * layer0_attn_out = nullptr;
    ggml_tensor * layer0_post_attn = nullptr;
    ggml_tensor * layer0_mlp_normed = nullptr;
    ggml_tensor * layer0_mlp_gate = nullptr;
    ggml_tensor * layer0_mlp_up = nullptr;
    ggml_tensor * layer0_mlp_gated = nullptr;
    ggml_tensor * layer0_mlp_out = nullptr;
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
        normed = ggml_mul(ctx.get(), normed, attn_norm);
        ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
        ggml_tensor * k = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
        ggml_tensor * v = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
        q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, tokens);
        k = ggml_reshape_3d(ctx.get(), k, head_dim, n_head_kv, tokens);
        v = ggml_reshape_3d(ctx.get(), v, head_dim, n_head_kv, tokens);

        ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
        ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
        q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
        k = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k, cfg.rms_norm_eps), k_norm);
        q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx.get(), k, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        k = ggml_cont_3d(ctx.get(), k, head_dim, n_head_kv, tokens);
        v = ggml_cont_3d(ctx.get(), v, head_dim, n_head_kv, tokens);
        k = ggml_reshape_4d(ctx.get(), k, head_dim, 1, n_head_kv, tokens);
        v = ggml_reshape_4d(ctx.get(), v, head_dim, 1, n_head_kv, tokens);
        k = ggml_repeat_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, tokens);
        v = ggml_repeat_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, tokens);
        k = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, tokens), head_dim, n_head, tokens);
        v = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, tokens), head_dim, n_head, tokens);

        ggml_tensor * q4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), q, head_dim, n_head, tokens, 1), 0, 2, 1, 3);
        ggml_tensor * k4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), k, head_dim, n_head, tokens, 1), 0, 2, 1, 3);
        ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
        kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        kq = ggml_diag_mask_inf(ctx.get(), kq, 0);
        kq = ggml_soft_max(ctx.get(), kq);
        ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), v, head_dim, n_head, tokens, 1), 1, 2, 0, 3));
        ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
        ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
        attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, tokens);
        ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
        ggml_tensor * cur = ggml_add(ctx.get(), cur_hidden, attn_out);

        ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
        ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
        mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
        ggml_tensor * gate = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
        ggml_tensor * up = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
        ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
        ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
        cur_hidden = ggml_add(ctx.get(), cur, mlp);
        if (!node_dump_dir().empty()) {
            layer_hidden_outputs[static_cast<size_t>(layer)] = cur_hidden;
            ggml_set_output(cur_hidden);
        }
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), static_cast<size_t>(cfg.block_count * 64 + 16), false);
    ggml_build_forward_expand(graph, cur_hidden);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), runtime.backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate runtime prefill graph tensors");
    }
    std::vector<int32_t> position_values(static_cast<size_t>(tokens));
    std::iota(position_values.begin(), position_values.end(), 0);
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(runtime.backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("runtime prefill graph compute failed");
    }
    if (cur_hidden->ne[0] != cfg.embedding_length || cur_hidden->ne[1] != tokens) {
        throw std::runtime_error("runtime prefill output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(cur_hidden->ne[0] * cur_hidden->ne[1]));
    ggml_backend_tensor_get(cur_hidden, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("runtime prefill output contains non-finite values");
        }
    }
    return values;
}

std::vector<float> run_prompt_last_hidden_with_runtime(const std::string & model_path, const std::string & text, const CodeMatrix * raw_audio, const GenerateOptions & options, BackendRuntime & runtime) {
    ScopedProfileTimer timer("prompt_prefill_total");
    const TextConfig & cfg = *runtime.cfg;
    GenerateOptions prompt_options = options;
    prompt_options.text = text;
    const std::vector<int32_t> prompt = build_prompt_ids(model_path, prompt_options);
    std::vector<float> hidden_values;
    {
        ScopedProfileTimer embedding_timer("prompt_prefill_embedding");
        hidden_values = run_embedding_graph_with_runtime(model_path, prompt, raw_audio, runtime);
    }
    hidden_values = run_prefill_all_layers_with_runtime(model_path, hidden_values, runtime);
    std::vector<float> last_token(static_cast<size_t>(cfg.embedding_length));
    const size_t offset = (prompt.size() - 1) * static_cast<size_t>(cfg.embedding_length);
    std::copy(hidden_values.begin() + static_cast<std::ptrdiff_t>(offset),
              hidden_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
              last_token.begin());
    return last_token;
}

std::vector<float> run_audio_frame_embedding_with_runtime(const std::string & model_path, const std::vector<int32_t> & frame_codes, BackendRuntime & runtime) {
    if (frame_codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("audio frame embedding code shape mismatch");
    }
    CodeMatrix audio;
    audio.codebooks = k_num_codebooks;
    audio.frames = 1;
    audio.data = frame_codes;
    return run_embedding_graph_with_runtime(model_path, {}, &audio, runtime);
}

std::vector<int32_t> run_prompt_audio_head_codes(const std::string & model_path, const std::string & text, const CodeMatrix * raw_audio, const GenerateOptions & options) {
    const std::vector<float> last_token = run_prompt_last_hidden(model_path, text, raw_audio, options);
    return run_audio_head_sample(model_path, last_token, options);
}

std::vector<int32_t> run_prompt_audio_head_codes(const std::string & model_path, const std::string & text, const GenerateOptions & options) {
    return run_prompt_audio_head_codes(model_path, text, nullptr, options);
}

void self_check_prompt_audio_head_graph(const std::string & model_path) {
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    options.temperature = 0.0f;
    options.stop_on_eoc = false;
    const std::vector<int32_t> codes = run_prompt_audio_head_codes(model_path, options.text, options);
    if (codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("prompt audio head sampled code shape mismatch");
    }
    for (int32_t code : codes) {
        if (code < 0 || code >= k_codebook_vocab_size) {
            throw std::runtime_error("prompt audio head sampled code range mismatch");
        }
    }
}

std::vector<int32_t> run_synthetic_text_codes(const std::string & model_path, const GenerateOptions & options) {
    const TextConfig cfg = inspect_text_config(model_path);
    const int tokens = 2;
    std::vector<float> hidden_values = run_synthetic_embedding_graph(model_path);
    if (hidden_values.size() != static_cast<size_t>(cfg.embedding_length * tokens)) {
        throw std::runtime_error("text logits embedding shape mismatch");
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_block_graph_layer(model_path, layer, hidden_values);
    }
    std::vector<float> last_token(static_cast<size_t>(cfg.embedding_length));
    std::copy(
        hidden_values.begin() + static_cast<std::ptrdiff_t>(cfg.embedding_length),
        hidden_values.begin() + static_cast<std::ptrdiff_t>(2 * cfg.embedding_length),
        last_token.begin());
    return run_audio_head_sample(model_path, last_token, options);
}

CodeMatrix repeat_code_frame(const std::vector<int32_t> & frame_codes, int output_frames, const char * label) {
    if (output_frames <= 0) {
        throw std::invalid_argument("steps must be positive");
    }
    const int raw_frames = output_frames;
    if (frame_codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error(std::string(label) + " sampled code shape mismatch");
    }
    std::vector<int32_t> raw_tn(static_cast<size_t>(raw_frames * k_num_codebooks));
    for (int f = 0; f < raw_frames; ++f) {
        std::copy(frame_codes.begin(), frame_codes.end(), raw_tn.begin() + static_cast<std::ptrdiff_t>(f * k_num_codebooks));
    }
    return finalize_generated_codes(apply_delay_pattern_tn(raw_tn, raw_frames, k_num_codebooks), true);
}

CodeMatrix run_prompt_text_code_frames(const std::string & model_path, const GenerateOptions & options, int output_frames, CodeMatrix * delayed_out) {
    if (output_frames <= 0) {
        throw std::invalid_argument("steps must be positive");
    }
    const int raw_frames = output_frames;
    std::vector<int32_t> raw_tn;
    raw_tn.reserve(static_cast<size_t>(raw_frames * k_num_codebooks));
    int eoc_countdown = -1;
    for (int f = 0; f < raw_frames; ++f) {
        throw_if_cancelled(options);
        CodeMatrix previous;
        previous.codebooks = k_num_codebooks;
        previous.frames = f + 1;
        previous.data.resize(static_cast<size_t>((f + 1) * k_num_codebooks));
        for (int c = 0; c < k_num_codebooks; ++c) {
            previous.at(c, 0) = k_boc_id;
        }
        for (int pf = 0; pf < f; ++pf) {
            for (int c = 0; c < k_num_codebooks; ++c) {
                previous.at(c, pf + 1) = raw_tn[static_cast<size_t>(pf * k_num_codebooks + c)];
            }
        }
        std::vector<int32_t> codes = run_prompt_audio_head_codes(model_path, options.text, f == 0 ? nullptr : &previous, options);
        apply_sglang_step_rules(codes, f, k_num_codebooks, eoc_countdown, options.stop_on_eoc);
        raw_tn.insert(raw_tn.end(), codes.begin(), codes.end());
    }
    CodeMatrix delayed = apply_delay_pattern_tn(raw_tn, raw_frames, k_num_codebooks);
    if (delayed_out != nullptr) {
        *delayed_out = delayed;
    }
    return finalize_generated_codes(delayed, true);
}

CodeMatrix run_prompt_text_code_frames_backend_kv(const std::string & model_path, const GenerateOptions & options, int output_frames, BackendRuntime & runtime, CodeMatrix * delayed_out) {
    if (output_frames <= 0) {
        throw std::invalid_argument("steps must be positive");
    }
    const TextConfig & cfg = *runtime.cfg;
    const int raw_frames = output_frames;
    GenerateOptions prompt_options = options;
    prompt_options.text = options.text;
    const std::vector<int32_t> prompt = build_prompt_ids(model_path, prompt_options);
    std::vector<float> prompt_embeddings;
    {
        ScopedProfileTimer embedding_timer("prompt_prefill_embedding");
        prompt_embeddings = run_embedding_graph_with_runtime(model_path, prompt, nullptr, runtime);
    }
    BackendKvCache & cache = get_backend_kv_cache(runtime, static_cast<int64_t>(prompt.size()) + raw_frames + 1);
    std::vector<float> hidden_values = run_prefill_all_layers_into_backend_cache(model_path, prompt_embeddings, cache);
    std::vector<float> hidden(static_cast<size_t>(cfg.embedding_length));
    const size_t offset = (prompt.size() - 1) * static_cast<size_t>(cfg.embedding_length);
    std::copy(hidden_values.begin() + static_cast<std::ptrdiff_t>(offset),
              hidden_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
              hidden.begin());
    hidden = run_audio_frame_embedding_with_runtime(model_path, std::vector<int32_t>(static_cast<size_t>(k_num_codebooks), k_boc_id), runtime);
    hidden = run_kv_decode_graph_all_layers_with_backend_cache(model_path, hidden, cache);
    if (!trace_json_path().empty()) {
        g_trace_node_stats["kv_boc_hidden"] = make_last_token_stat(hidden, cfg.embedding_length, 1);
    }
    cache.used += 1;

    std::vector<int32_t> raw_tn;
    raw_tn.reserve(static_cast<size_t>(raw_frames * k_num_codebooks));
    int eoc_countdown = -1;
    for (int f = 0; f < raw_frames; ++f) {
        throw_if_cancelled(options);
        std::vector<int32_t> codes = run_audio_head_sample_with_cache(model_path, hidden, options, cache);
        apply_sglang_step_rules(codes, f, k_num_codebooks, eoc_countdown, options.stop_on_eoc);
        raw_tn.insert(raw_tn.end(), codes.begin(), codes.end());
        hidden = run_audio_frame_embedding_with_runtime(model_path, codes, runtime);
        hidden = run_kv_decode_graph_all_layers_with_backend_cache(model_path, hidden, cache);
        cache.used += 1;
    }
    CodeMatrix delayed = apply_delay_pattern_tn(raw_tn, raw_frames, k_num_codebooks);
    if (delayed_out != nullptr) {
        *delayed_out = delayed;
    }
    return finalize_generated_codes(delayed, true);
}

std::vector<float> run_reference_embedding_values(const std::string & model_path, const std::vector<int32_t> & prompt, const CodeMatrix & ref_codes, const CodeMatrix * previous) {
    std::vector<float> hidden = run_embedding_graph(model_path, safe_text_ids_for_embedding(prompt), nullptr);
    const std::vector<float> ref_hidden = run_embedding_graph(model_path, {}, &ref_codes);
    dump_node_values("cpp_ref_hidden", ref_hidden);
    size_t ref_index = 0;
    for (size_t token = 0; token < prompt.size(); ++token) {
        if (prompt[token] == -100) {
            if (ref_index >= static_cast<size_t>(ref_codes.frames)) {
                throw std::runtime_error("reference prompt has more placeholders than reference code frames");
            }
            std::copy(
                ref_hidden.begin() + static_cast<std::ptrdiff_t>(ref_index * 2560),
                ref_hidden.begin() + static_cast<std::ptrdiff_t>((ref_index + 1) * 2560),
                hidden.begin() + static_cast<std::ptrdiff_t>(token * 2560));
            ++ref_index;
        }
    }
    if (ref_index != static_cast<size_t>(ref_codes.frames)) {
        throw std::runtime_error("reference prompt placeholder count mismatch");
    }
    if (previous != nullptr && previous->frames > 0) {
        const std::vector<float> prev_hidden = run_embedding_graph(model_path, {}, previous);
        hidden.insert(hidden.end(), prev_hidden.begin(), prev_hidden.end());
    }
    return hidden;
}

std::vector<float> run_reference_embedding_values_with_runtime(const std::string & model_path, const std::vector<int32_t> & prompt, const CodeMatrix & ref_codes, const CodeMatrix * previous, BackendRuntime & runtime) {
    std::vector<float> hidden = run_reference_prompt_embedding_graph_with_runtime(model_path, prompt, ref_codes, runtime);
    if (previous != nullptr && previous->frames > 0) {
        const std::vector<float> prev_hidden = run_embedding_graph_with_runtime(model_path, {}, previous, runtime);
        hidden.insert(hidden.end(), prev_hidden.begin(), prev_hidden.end());
    }
    return hidden;
}

CodeMatrix make_reference_previous_codes(const std::vector<int32_t> & raw_tn, int generated_frames) {
    CodeMatrix previous;
    previous.codebooks = k_num_codebooks;
    previous.frames = generated_frames + 1;
    previous.data.resize(static_cast<size_t>(previous.frames * k_num_codebooks));
    for (int c = 0; c < k_num_codebooks; ++c) {
        previous.at(c, 0) = k_boc_id;
    }
    for (int pf = 0; pf < generated_frames; ++pf) {
        for (int c = 0; c < k_num_codebooks; ++c) {
            previous.at(c, pf + 1) = raw_tn[static_cast<size_t>(pf * k_num_codebooks + c)];
        }
    }
    return previous;
}

std::vector<int32_t> run_reference_audio_head_codes(const std::string & model_path, const GenerateOptions & options, const CodeMatrix & ref_codes, const CodeMatrix * previous) {
    const TextConfig cfg = inspect_text_config(model_path);
    const SpecialTokenIds ids = load_special_token_ids(model_path);
    const std::vector<int32_t> text_ids = encode_text_tokenizers_cpp(model_path, options.text);
    const std::vector<int32_t> ref_text_ids = options.ref_text.empty() ? std::vector<int32_t>{} : encode_text_tokenizers_cpp(model_path, options.ref_text);
    const std::vector<int32_t> prompt = build_reference_prompt_ids(ids, text_ids, ref_codes.frames, ref_text_ids);
    std::vector<float> hidden_values = run_reference_embedding_values(model_path, prompt, ref_codes, previous);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_block_graph_layer(model_path, layer, hidden_values);
    }
    std::vector<float> last_token(static_cast<size_t>(cfg.embedding_length));
    const size_t token_count = prompt.size() + (previous == nullptr ? 0 : static_cast<size_t>(previous->frames));
    const size_t offset = (token_count - 1) * static_cast<size_t>(cfg.embedding_length);
    std::copy(hidden_values.begin() + static_cast<std::ptrdiff_t>(offset),
              hidden_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
              last_token.begin());
    return run_audio_head_sample(model_path, last_token, options);
}

CodeMatrix run_reference_text_code_frames(const std::string & model_path, const GenerateOptions & options, const CodeMatrix & ref_codes, int output_frames, CodeMatrix * delayed_out) {
    if (output_frames <= 0) {
        throw std::invalid_argument("steps must be positive");
    }
    const SpecialTokenIds ids = load_special_token_ids(model_path);
    const std::vector<int32_t> text_ids = encode_text_tokenizers_cpp(model_path, options.text);
    const std::vector<int32_t> ref_text_ids = options.ref_text.empty() ? std::vector<int32_t>{} : encode_text_tokenizers_cpp(model_path, options.ref_text);
    const std::vector<int32_t> prompt = build_reference_prompt_ids(ids, text_ids, ref_codes.frames, ref_text_ids);
    const int raw_frames = output_frames;
    std::vector<int32_t> raw_tn;
    raw_tn.reserve(static_cast<size_t>(raw_frames * k_num_codebooks));
    std::vector<std::vector<float>> generated_embeddings;
    generated_embeddings.reserve(static_cast<size_t>(raw_frames + 1));
    std::vector<std::vector<int32_t>> trace_rows;
    std::vector<TraceStep> trace_steps;
    trace_rows.push_back(std::vector<int32_t>(static_cast<size_t>(k_num_codebooks), k_boc_id));
    int eoc_countdown = -1;
    for (int f = 0; f < raw_frames; ++f) {
        throw_if_cancelled(options);
        ReferenceDebugStepScope debug_step(f);
        CodeMatrix previous;
        previous.codebooks = k_num_codebooks;
        previous.frames = f + 1;
        previous.data.resize(static_cast<size_t>((f + 1) * k_num_codebooks));
        for (int c = 0; c < k_num_codebooks; ++c) {
            previous.at(c, 0) = k_boc_id;
        }
        for (int pf = 0; pf < f; ++pf) {
            for (int c = 0; c < k_num_codebooks; ++c) {
                previous.at(c, pf + 1) = raw_tn[static_cast<size_t>(pf * k_num_codebooks + c)];
            }
        }
        const TextConfig cfg = inspect_text_config(model_path);
        const std::vector<float> hidden_values = run_reference_embedding_values(model_path, prompt, ref_codes, &previous);
        if (f == 0) {
            g_trace_node_stats["inputs_embeds"] = make_last_token_stat(hidden_values, cfg.embedding_length, static_cast<int64_t>(hidden_values.size() / static_cast<size_t>(cfg.embedding_length)));
            dump_node_values("cpp_inputs_embeds", hidden_values);
        }
        std::vector<float> backbone_values = hidden_values;
        if (!node_dump_dir().empty()) {
            const size_t token_count = prompt.size() + static_cast<size_t>(previous.frames);
            const size_t offset = (token_count - 1) * static_cast<size_t>(cfg.embedding_length);
            std::vector<float> input_last(static_cast<size_t>(cfg.embedding_length));
            std::copy(backbone_values.begin() + static_cast<std::ptrdiff_t>(offset),
                      backbone_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
                      input_last.begin());
            dump_node_values("cpp_full_step" + std::to_string(f) + "_input_last", input_last);
        }
        for (int layer = 0; layer < cfg.block_count; ++layer) {
            backbone_values = run_block_graph_layer(model_path, layer, backbone_values);
            if (!node_dump_dir().empty()) {
                const size_t token_count = prompt.size() + static_cast<size_t>(previous.frames);
                const size_t offset = (token_count - 1) * static_cast<size_t>(cfg.embedding_length);
                std::vector<float> layer_last(static_cast<size_t>(cfg.embedding_length));
                std::copy(backbone_values.begin() + static_cast<std::ptrdiff_t>(offset),
                          backbone_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
                          layer_last.begin());
                dump_node_values("cpp_full_step" + std::to_string(f) + "_blk" + std::to_string(layer) + "_last", layer_last);
            }
            if (f == 0 && (layer == 0 || layer == cfg.block_count - 1)) {
                g_trace_node_stats["blk." + std::to_string(layer) + ".hidden"] =
                    make_last_token_stat(backbone_values, cfg.embedding_length, static_cast<int64_t>(backbone_values.size() / static_cast<size_t>(cfg.embedding_length)));
            }
        }
        std::vector<float> last_token(static_cast<size_t>(cfg.embedding_length));
        const size_t token_count = prompt.size() + static_cast<size_t>(previous.frames);
        const size_t offset = (token_count - 1) * static_cast<size_t>(cfg.embedding_length);
        std::copy(backbone_values.begin() + static_cast<std::ptrdiff_t>(offset),
                  backbone_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
                  last_token.begin());
        if (f == 0) {
            g_trace_node_stats["final_hidden"] = make_last_token_stat(last_token, cfg.embedding_length, 1);
        }
        const int audio_head_frames = static_cast<int>(token_count);
        const std::vector<float> logits_vc = run_audio_head_logits_vc_for_frame(model_path, backbone_values, audio_head_frames, audio_head_frames - 1);
        if (f == 0) {
            g_trace_node_stats["audio_logits_flat"] = make_last_token_stat(logits_vc, static_cast<int64_t>(logits_vc.size()), 1);
        }
        uint32_t rng_state = options.seed;
        std::vector<int32_t> codes = sample_codes_from_logits_vc(logits_vc, k_codebook_vocab_size, k_num_codebooks, options.temperature, options.top_k, options.top_p, rng_state, options.stop_on_eoc);
        apply_sglang_step_rules(codes, f, k_num_codebooks, eoc_countdown, options.stop_on_eoc);
        raw_tn.insert(raw_tn.end(), codes.begin(), codes.end());
        trace_rows.push_back(codes);
        trace_steps.push_back(trace_step_from_logits(logits_vc, codes, options.stop_on_eoc));
        flush_partial_trace_json(prompt, ref_codes, trace_rows, trace_steps, raw_tn);
    }
    CodeMatrix delayed = delayed_rows_to_matrix(trace_rows);
    if (delayed_out != nullptr) {
        *delayed_out = delayed;
    }
    const CodeMatrix finalized = finalize_generated_codes(delayed, true);
    write_trace_json(trace_json_path(), prompt, ref_codes, trace_rows, trace_steps, finalized);
    write_codes_json(codes_json_path(), trace_rows, finalized);
    return finalized;
}

CodeMatrix run_reference_text_code_frames_backend_kv(const std::string & model_path, const GenerateOptions & options, const CodeMatrix & ref_codes, int output_frames, BackendRuntime & runtime, CodeMatrix * delayed_out, int * backend_kv_slot_out, ReferenceArRuntimeStats * runtime_stats) {
    if (output_frames <= 0) {
        throw std::invalid_argument("steps must be positive");
    }
    ScopedActiveReferenceArRequest active_request(runtime_stats != nullptr && server_audio_head_batch_enabled());
    const TextConfig & cfg = *runtime.cfg;
    const SpecialTokenIds ids = load_special_token_ids(model_path);
    const std::vector<int32_t> text_ids = encode_text_tokenizers_cpp(model_path, options.text);
    const std::vector<int32_t> ref_text_ids = options.ref_text.empty() ? std::vector<int32_t>{} : encode_text_tokenizers_cpp(model_path, options.ref_text);
    const std::vector<int32_t> prompt = build_reference_prompt_ids(ids, text_ids, ref_codes.frames, ref_text_ids);
    std::vector<float> hidden_values;
    {
        ScopedProfileTimer timer("reference_prefill_total");
        {
            ScopedProfileTimer embedding_timer("reference_prefill_embedding");
            if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
                const CudaExecutorRunStats stats = cuda_executor().run_sync([&] {
                    hidden_values = run_reference_embedding_values_with_runtime(model_path, prompt, ref_codes, nullptr, runtime);
                });
                add_cuda_executor_stats(runtime_stats, stats);
            } else {
                hidden_values = run_reference_embedding_values_with_runtime(model_path, prompt, ref_codes, nullptr, runtime);
            }
        }
    }
    if (!trace_json_path().empty()) {
        g_trace_node_stats["reference_prompt_inputs_embeds"] =
            make_last_token_stat(hidden_values, cfg.embedding_length, static_cast<int64_t>(prompt.size()));
    }
    std::vector<float> generated_embedding_values = hidden_values;
    std::vector<float> generated_contextual_values = hidden_values;
    const int raw_frames = output_frames;
    ScopedBackendKvCacheSlot kv_slot;
    BackendKvCache * slot_cache = nullptr;
    if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
        kv_slot = acquire_backend_kv_cache_slot(runtime, static_cast<int64_t>(prompt.size()) + raw_frames + 1);
        slot_cache = kv_slot.cache;
        if (backend_kv_slot_out != nullptr) {
            *backend_kv_slot_out = kv_slot.slot;
        }
    }
    BackendKvCache & cache = slot_cache != nullptr
        ? *slot_cache
        : get_backend_kv_cache(runtime, static_cast<int64_t>(prompt.size()) + raw_frames + 1);
    cache.used = 0;
    const int window_refresh = reference_kv_window_refresh();
    const bool always_full_prefix_refresh = window_refresh >= static_cast<int>(prompt.size());
    std::vector<std::vector<float>> generated_embeddings;
    generated_embeddings.reserve(static_cast<size_t>(raw_frames + 1));
    if (always_full_prefix_refresh) {
        cache.used = static_cast<int64_t>(prompt.size());
    } else if (reference_kv_prefill_via_window_enabled()) {
        cache.used = static_cast<int64_t>(prompt.size());
        if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
            const CudaExecutorRunStats stats = cuda_executor().run_sync([&] {
                hidden_values = run_kv_decode_window_all_layers_full_with_backend_cache(model_path, hidden_values, static_cast<int64_t>(prompt.size()), cache);
            });
            add_cuda_executor_stats(runtime_stats, stats);
        } else {
            hidden_values = run_kv_decode_window_all_layers_full_with_backend_cache(model_path, hidden_values, static_cast<int64_t>(prompt.size()), cache);
        }
    } else {
        if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
            const CudaExecutorRunStats stats = cuda_executor().run_sync([&] {
                hidden_values = run_prefill_all_layers_into_backend_cache(model_path, hidden_values, cache);
            });
            add_cuda_executor_stats(runtime_stats, stats);
        } else {
            hidden_values = run_prefill_all_layers_into_backend_cache(model_path, hidden_values, cache);
        }
    }
    if (!trace_json_path().empty() && !always_full_prefix_refresh) {
        const size_t prompt_offset = (prompt.size() - 1) * static_cast<size_t>(cfg.embedding_length);
        std::vector<float> prompt_last(static_cast<size_t>(cfg.embedding_length));
        std::copy(hidden_values.begin() + static_cast<std::ptrdiff_t>(prompt_offset),
                  hidden_values.begin() + static_cast<std::ptrdiff_t>(prompt_offset + static_cast<size_t>(cfg.embedding_length)),
                  prompt_last.begin());
        g_trace_node_stats["kv_prefill_last_hidden"] = make_last_token_stat(prompt_last, cfg.embedding_length, 1);
    }
    std::vector<float> hidden(static_cast<size_t>(cfg.embedding_length));
    std::vector<float> fused_logits_vc;
    const size_t offset = (prompt.size() - 1) * static_cast<size_t>(cfg.embedding_length);
    std::copy(hidden_values.begin() + static_cast<std::ptrdiff_t>(offset),
              hidden_values.begin() + static_cast<std::ptrdiff_t>(offset + static_cast<size_t>(cfg.embedding_length)),
    hidden.begin());
        if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
            const CudaExecutorRunStats stats = cuda_executor().run_sync([&] {
                hidden = run_audio_frame_embedding_with_runtime(model_path, std::vector<int32_t>(static_cast<size_t>(k_num_codebooks), k_boc_id), runtime);
            });
            add_cuda_executor_stats(runtime_stats, stats);
        } else {
            hidden = run_audio_frame_embedding_with_runtime(model_path, std::vector<int32_t>(static_cast<size_t>(k_num_codebooks), k_boc_id), runtime);
        }
        generated_embeddings.push_back(hidden);
        generated_embedding_values.insert(generated_embedding_values.end(), hidden.begin(), hidden.end());
        generated_contextual_values.insert(generated_contextual_values.end(), hidden.begin(), hidden.end());
        if (!node_dump_dir().empty()) {
            dump_node_values("cpp_inputs_embeds", generated_embedding_values);
        }
        if (!trace_json_path().empty()) {
            g_trace_node_stats["kv_boc_input_embed"] = make_last_token_stat(hidden, cfg.embedding_length, 1);
        }
    if (!always_full_prefix_refresh) {
        KvDecodeResult boc_decode;
        auto decode_boc = [&] {
            boc_decode = run_kv_decode_graph_all_layers_with_backend_cache_ex(
                model_path,
                hidden,
                cache,
                reference_kv_fused_audio_head_enabled() && trace_json_path().empty(),
                false);
        };
        if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
            const CudaExecutorRunStats stats = cuda_executor().run_sync(decode_boc);
            add_cuda_executor_stats(runtime_stats, stats);
        } else {
            decode_boc();
        }
        hidden = std::move(boc_decode.hidden);
        fused_logits_vc = std::move(boc_decode.logits_vc);
    }
    if (!trace_json_path().empty() && !always_full_prefix_refresh) {
        g_trace_node_stats["kv_boc_hidden"] = make_last_token_stat(hidden, cfg.embedding_length, 1);
    }
    if (!always_full_prefix_refresh && reference_kv_compare_window1_enabled()) {
        if (trace_json_path().empty()) {
            throw std::runtime_error("HIGGS_REFERENCE_KV_COMPARE_WINDOW1 is diagnostic-only; set HIGGS_TRACE_JSON");
        }
        std::vector<float> window_logits_vc;
        const int64_t saved_used = cache.used;
        cache.used = saved_used + 1;
        std::vector<float> window_hidden = run_kv_decode_window_all_layers_full_with_backend_cache(model_path, generated_embeddings.back(), 1, cache, &window_logits_vc, true);
        cache.used = saved_used;
        g_trace_node_stats["kv_boc_window1_hidden"] = make_last_token_stat(window_hidden, cfg.embedding_length, 1);
        g_trace_node_stats["kv_boc_single_vs_window1_hidden_abs_diff"] = make_abs_diff_stat(hidden, window_hidden);
        const std::vector<float> single_logits_vc = run_audio_head_logits_vc_with_cache(model_path, hidden, cache);
        g_trace_node_stats["kv_boc_window1_logits_vc"] = make_last_token_stat(window_logits_vc, static_cast<int64_t>(window_logits_vc.size()), 1);
        g_trace_node_stats["kv_boc_single_logits_vc"] = make_last_token_stat(single_logits_vc, static_cast<int64_t>(single_logits_vc.size()), 1);
        g_trace_node_stats["kv_boc_single_vs_window1_logits_abs_diff"] = make_abs_diff_stat(single_logits_vc, window_logits_vc);
    }
    cache.used += 1;

    std::vector<int32_t> raw_tn;
    raw_tn.reserve(static_cast<size_t>(raw_frames * k_num_codebooks));
    std::vector<std::vector<int32_t>> trace_rows;
    std::vector<TraceStep> trace_steps;
    trace_rows.push_back(std::vector<int32_t>(static_cast<size_t>(k_num_codebooks), k_boc_id));
    int eoc_countdown = -1;
    int full_fallbacks = 0;
    int skipped_fallback_oracles = 0;
    int equivalent_fallbacks = 0;
    int changing_fallbacks = 0;
    int window_refreshes = 0;
    const bool use_full_fallback = reference_kv_full_fallback_enabled();
    const bool use_fast_fallback = !always_full_prefix_refresh && reference_kv_fast_fallback_enabled();
    const float fallback_margin = reference_kv_fallback_margin();
    const int head_batch = reference_kv_head_batch();
    const bool contextual_window_input = reference_kv_window_contextual_input();
    const bool compare_full_prefix = reference_kv_compare_full_prefix_enabled();
    if (compare_full_prefix && (trace_json_path().empty() || always_full_prefix_refresh)) {
        throw std::runtime_error("HIGGS_REFERENCE_KV_COMPARE_FULL_PREFIX is diagnostic-only for non-full-prefix trace runs");
    }
    for (int f = 0; f < raw_frames; ++f) {
        throw_if_cancelled(options);
        std::vector<float> window_full_hidden;
        const bool use_full_prefix_this_step = always_full_prefix_refresh;
        const int effective_window = use_full_prefix_this_step ? static_cast<int>(cache.used) : window_refresh;
        const bool have_window = use_full_prefix_this_step
            ? generated_embedding_values.size() >= static_cast<size_t>(effective_window) * static_cast<size_t>(cfg.embedding_length)
            : static_cast<int>(generated_embeddings.size()) >= effective_window;
        if (effective_window > 0 && have_window) {
            ScopedProfileTimer refresh_timer("reference_kv_window_refresh");
            std::vector<float> window_hidden;
            const std::vector<float> * window_input = &window_hidden;
            if (use_full_prefix_this_step) {
                window_input = &generated_embedding_values;
            } else if (contextual_window_input) {
                const size_t n = static_cast<size_t>(effective_window) * static_cast<size_t>(cfg.embedding_length);
                window_hidden.reserve(n);
                window_hidden.insert(window_hidden.end(), generated_contextual_values.end() - static_cast<std::ptrdiff_t>(n), generated_contextual_values.end());
            } else {
                window_hidden.reserve(static_cast<size_t>(effective_window) * static_cast<size_t>(cfg.embedding_length));
                const size_t first = generated_embeddings.size() - static_cast<size_t>(effective_window);
                for (size_t i = first; i < generated_embeddings.size(); ++i) {
                    window_hidden.insert(window_hidden.end(), generated_embeddings[i].begin(), generated_embeddings[i].end());
                }
            }
            const bool fuse_logits_this_step = use_full_prefix_this_step && node_dump_dir().empty();
            const bool skip_hidden_readback = fuse_logits_this_step;
            auto run_window = [&] {
                window_full_hidden = run_kv_decode_window_all_layers_full_with_backend_cache(model_path, *window_input, effective_window, cache, fuse_logits_this_step ? &fused_logits_vc : nullptr, fuse_logits_this_step, skip_hidden_readback);
            };
            if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
                const CudaExecutorRunStats stats = cuda_executor().run_sync(run_window);
                add_cuda_executor_stats(runtime_stats, stats);
            } else {
                run_window();
            }
            if (!window_full_hidden.empty()) {
                std::copy(window_full_hidden.end() - static_cast<std::ptrdiff_t>(cfg.embedding_length), window_full_hidden.end(), hidden.begin());
                if (window_full_hidden.size() == static_cast<size_t>(effective_window) * static_cast<size_t>(cfg.embedding_length)) {
                    const size_t n = static_cast<size_t>(effective_window) * static_cast<size_t>(cfg.embedding_length);
                    std::copy(window_full_hidden.begin(), window_full_hidden.end(), generated_contextual_values.end() - static_cast<std::ptrdiff_t>(n));
                } else {
                    std::copy(hidden.begin(), hidden.end(), generated_contextual_values.end() - static_cast<std::ptrdiff_t>(cfg.embedding_length));
                }
            }
            ++window_refreshes;
        }
        std::vector<float> logits_vc;
        {
            ScopedProfileTimer timer("reference_ar_audio_head");
            const auto audio_head_start = std::chrono::steady_clock::now();
            if (!fused_logits_vc.empty()) {
                logits_vc = std::move(fused_logits_vc);
                if (runtime_stats != nullptr) {
                    ++runtime_stats->audio_head_fallback_calls;
                }
            } else if (!window_full_hidden.empty()) {
                auto run_head = [&] {
                    logits_vc = run_audio_head_logits_vc_for_frame_with_cache(model_path, window_full_hidden, effective_window, effective_window - 1, cache);
                };
                if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
                    const CudaExecutorRunStats stats = cuda_executor().run_sync(run_head);
                    add_cuda_executor_stats(runtime_stats, stats);
                } else {
                    run_head();
                }
                if (runtime_stats != nullptr) {
                    ++runtime_stats->audio_head_fallback_calls;
                }
            } else if (head_batch > 1) {
                std::vector<float> batched_hidden;
                batched_hidden.reserve(static_cast<size_t>(head_batch) * static_cast<size_t>(cfg.embedding_length));
                for (int i = 0; i < head_batch; ++i) {
                    batched_hidden.insert(batched_hidden.end(), hidden.begin(), hidden.end());
                }
                auto run_head = [&] {
                    logits_vc = run_audio_head_logits_vc_for_frame_with_cache(model_path, batched_hidden, head_batch, head_batch - 1, cache);
                };
                if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
                    const CudaExecutorRunStats stats = cuda_executor().run_sync(run_head);
                    add_cuda_executor_stats(runtime_stats, stats);
                } else {
                    run_head();
                }
                if (runtime_stats != nullptr) {
                    ++runtime_stats->audio_head_fallback_calls;
                }
            } else {
                logits_vc = std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr
                    ? audio_head_batcher_run(model_path, f, hidden, cache, runtime_stats)
                    : run_audio_head_logits_vc_with_cache(model_path, hidden, cache);
            }
            if (runtime_stats != nullptr) {
                runtime_stats->audio_head_wall_ms += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - audio_head_start).count();
            }
        }
        if (compare_full_prefix) {
            BackendKvCache compare_cache = make_backend_kv_cache(cfg, runtime, cache.capacity);
            compare_cache.used = cache.used;
            std::vector<float> full_prefix_logits_vc;
            std::vector<float> full_prefix_hidden = run_kv_decode_window_all_layers_full_with_backend_cache(model_path, generated_embedding_values, cache.used, compare_cache, &full_prefix_logits_vc, true);
            g_trace_node_stats["kv_step" + std::to_string(f) + "_raw_vs_full_hidden_abs_diff"] = make_abs_diff_stat(hidden, full_prefix_hidden);
            g_trace_node_stats["kv_step" + std::to_string(f) + "_raw_vs_full_logits_abs_diff"] = make_abs_diff_stat(logits_vc, full_prefix_logits_vc);
        }
        const int active_codebooks = f < k_num_codebooks ? f + 1 : k_num_codebooks;
        if ((use_full_fallback || use_fast_fallback) && options.temperature <= 0.0f && greedy_logits_have_small_margin(logits_vc, k_codebook_vocab_size, k_num_codebooks, active_codebooks, fallback_margin)) {
            ++full_fallbacks;
            uint32_t raw_rng_state = options.seed;
            std::vector<int32_t> raw_codes = sample_codes_from_logits_vc(logits_vc, k_codebook_vocab_size, k_num_codebooks, options.temperature, options.top_k, options.top_p, raw_rng_state, options.stop_on_eoc);
            int raw_eoc_countdown = eoc_countdown;
            apply_sglang_step_rules(raw_codes, f, k_num_codebooks, raw_eoc_countdown, options.stop_on_eoc);
            const GreedyMarginInfo raw_margin = greedy_logits_min_active_margin_info(logits_vc, k_codebook_vocab_size, k_num_codebooks, active_codebooks);
            if (reference_kv_skip_non_cb6_fallback_enabled() && raw_margin.codebook != 6) {
                ++skipped_fallback_oracles;
                if (profile_enabled()) {
                    std::fprintf(stderr,
                                 "higgs_profile reference_kv_fallback_detail step=%d active_codebooks=%d trigger_codebook=%d raw_top1=%d raw_top2=%d raw_margin=%.9g oracle_skipped=1\n",
                                 f,
                                 active_codebooks,
                                 raw_margin.codebook,
                                 raw_margin.top1,
                                 raw_margin.top2,
                                 static_cast<double>(raw_margin.margin));
                }
            } else {
            auto fallback_start = std::chrono::steady_clock::now();
            std::vector<float> fallback_logits_vc;
            std::vector<float> fallback_hidden;
            if (use_fast_fallback) {
                const int fallback_tokens = static_cast<int>(cache.used);
                fallback_hidden = run_kv_decode_window_all_layers_full_with_backend_cache(model_path, generated_embedding_values, fallback_tokens, cache, &fallback_logits_vc, true);
            } else {
                CodeMatrix previous = make_reference_previous_codes(raw_tn, f);
                fallback_hidden = run_reference_embedding_values(model_path, prompt, ref_codes, &previous);
                for (int layer = 0; layer < cfg.block_count; ++layer) {
                    fallback_hidden = run_block_graph_layer(model_path, layer, fallback_hidden);
                }
                const int audio_head_frames = static_cast<int>(prompt.size()) + previous.frames;
                fallback_logits_vc = run_audio_head_logits_vc_for_frame(model_path, fallback_hidden, audio_head_frames, audio_head_frames - 1);
            }
            const auto fallback_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - fallback_start).count();
            if (profile_enabled()) {
                profile_record_elapsed("reference_kv_fallback_compute", static_cast<long long>(fallback_elapsed));
            }
            uint32_t fallback_rng_state = options.seed;
            std::vector<int32_t> fallback_codes = sample_codes_from_logits_vc(fallback_logits_vc, k_codebook_vocab_size, k_num_codebooks, options.temperature, options.top_k, options.top_p, fallback_rng_state, options.stop_on_eoc);
            int fallback_eoc_countdown = eoc_countdown;
            apply_sglang_step_rules(fallback_codes, f, k_num_codebooks, fallback_eoc_countdown, options.stop_on_eoc);
            int fallback_top1 = -1;
            if (raw_margin.codebook >= 0) {
                float best = -std::numeric_limits<float>::infinity();
                for (int v = 0; v < k_codebook_vocab_size; ++v) {
                    float value = fallback_logits_vc[static_cast<size_t>(v) * static_cast<size_t>(k_num_codebooks) + static_cast<size_t>(raw_margin.codebook)];
                    if (v >= k_codebook_data_size && v < std::min(k_codebook_vocab_size, k_eoc_id)) {
                        value = -std::numeric_limits<float>::infinity();
                    }
                    if (value > best) {
                        best = value;
                        fallback_top1 = v;
                    }
                }
            }
            const bool same_codes = fallback_codes == raw_codes;
            if (same_codes) {
                ++equivalent_fallbacks;
            } else {
                ++changing_fallbacks;
                std::copy(fallback_hidden.end() - static_cast<std::ptrdiff_t>(cfg.embedding_length), fallback_hidden.end(), hidden.begin());
                logits_vc = std::move(fallback_logits_vc);
                if (use_fast_fallback) {
                    ++window_refreshes;
                }
            }
            if (profile_enabled()) {
                std::fprintf(stderr,
                             "higgs_profile reference_kv_fallback_detail step=%d active_codebooks=%d trigger_codebook=%d raw_top1=%d raw_top2=%d raw_margin=%.9g fallback_top1=%d changed_trigger=%d same_codes=%d elapsed_ms=%lld\n",
                             f,
                             active_codebooks,
                             raw_margin.codebook,
                             raw_margin.top1,
                             raw_margin.top2,
                             static_cast<double>(raw_margin.margin),
                             fallback_top1,
                             fallback_top1 != raw_margin.top1 ? 1 : 0,
                             same_codes ? 1 : 0,
                             static_cast<long long>(fallback_elapsed));
            }
            }
        }
        uint32_t rng_state = options.seed;
        std::vector<int32_t> codes;
        {
            ScopedProfileTimer timer("reference_ar_sample");
            codes = sample_codes_from_logits_vc(logits_vc, k_codebook_vocab_size, k_num_codebooks, options.temperature, options.top_k, options.top_p, rng_state, options.stop_on_eoc);
            apply_sglang_step_rules(codes, f, k_num_codebooks, eoc_countdown, options.stop_on_eoc);
        }
        raw_tn.insert(raw_tn.end(), codes.begin(), codes.end());
        trace_rows.push_back(codes);
        trace_steps.push_back(trace_step_from_logits(logits_vc, codes, options.stop_on_eoc));
        flush_partial_trace_json(prompt, ref_codes, trace_rows, trace_steps, raw_tn);
        {
            ScopedProfileTimer timer("reference_ar_audio_embedding");
            auto run_embedding = [&] {
                hidden = run_audio_frame_embedding_with_runtime(model_path, codes, runtime);
            };
            if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
                const CudaExecutorRunStats stats = cuda_executor().run_sync(run_embedding);
                add_cuda_executor_stats(runtime_stats, stats);
            } else {
                run_embedding();
            }
        }
        generated_embeddings.push_back(hidden);
        generated_embedding_values.insert(generated_embedding_values.end(), hidden.begin(), hidden.end());
        generated_contextual_values.insert(generated_contextual_values.end(), hidden.begin(), hidden.end());
        if (!always_full_prefix_refresh) {
            ScopedProfileTimer timer("reference_ar_kv_decode");
            KvDecodeResult decoded;
            auto run_decode = [&] {
                decoded = run_kv_decode_graph_all_layers_with_backend_cache_ex(
                    model_path,
                    hidden,
                    cache,
                    reference_kv_fused_audio_head_enabled() && trace_json_path().empty(),
                    false);
            };
            if (std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr) {
                const CudaExecutorRunStats stats = cuda_executor().run_sync(run_decode);
                add_cuda_executor_stats(runtime_stats, stats);
            } else {
                run_decode();
            }
            hidden = std::move(decoded.hidden);
            fused_logits_vc = std::move(decoded.logits_vc);
            std::copy(hidden.begin(), hidden.end(), generated_contextual_values.end() - static_cast<std::ptrdiff_t>(cfg.embedding_length));
        }
        cache.used += 1;
    }
    if (profile_enabled()) {
        std::fprintf(stderr, "higgs_profile reference_kv_full_fallbacks count=%d frames=%d\n", full_fallbacks, raw_frames);
        std::fprintf(stderr, "higgs_profile reference_kv_skipped_fallback_oracles count=%d frames=%d\n", skipped_fallback_oracles, raw_frames);
        std::fprintf(stderr, "higgs_profile reference_kv_equivalent_fallbacks count=%d changing=%d frames=%d\n", equivalent_fallbacks, changing_fallbacks, raw_frames);
        std::fprintf(stderr, "higgs_profile reference_kv_window_refreshes count=%d window=%d frames=%d\n", window_refreshes, window_refresh, raw_frames);
        profile_print_aggregates();
    }
    CodeMatrix delayed = delayed_rows_to_matrix(trace_rows);
    if (delayed_out != nullptr) {
        *delayed_out = delayed;
    }
    const CodeMatrix finalized = finalize_generated_codes(delayed, true);
    write_trace_json(trace_json_path(), prompt, ref_codes, trace_rows, trace_steps, finalized);
    write_codes_json(codes_json_path(), trace_rows, finalized);
    return finalized;
}

CodeMatrix run_synthetic_text_code_frames(const std::string & model_path, const GenerateOptions & options, int output_frames) {
    return repeat_code_frame(run_synthetic_text_codes(model_path, options), output_frames, "synthetic text");
}

void self_check_kv_cache(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    TextKvCache cache = make_text_kv_cache(cfg, 4);
    if (cache.layers != cfg.block_count || cache.head_dim != cfg.key_length || cache.kv_heads != cfg.head_count_kv || cache.capacity != 4 || cache.used != 0) {
        throw std::runtime_error("KV cache metadata mismatch");
    }
    const size_t expected = static_cast<size_t>(cfg.block_count * 4 * cfg.head_count_kv * cfg.key_length);
    if (cache.key.size() != expected || cache.value.size() != expected) {
        throw std::runtime_error("KV cache storage size mismatch");
    }
    cache.key[kv_cache_offset(cache, 0, 0, 0, 0)] = 1.0f;
    cache.value[kv_cache_offset(cache, 0, 0, 0, 0)] = 2.0f;
    cache.key[kv_cache_offset(cache, cfg.block_count - 1, 3, cfg.head_count_kv - 1, cfg.key_length - 1)] = 3.0f;
    cache.value[kv_cache_offset(cache, cfg.block_count - 1, 3, cfg.head_count_kv - 1, cfg.key_length - 1)] = 4.0f;
    if (cache.key.front() != 1.0f || cache.value.front() != 2.0f || cache.key.back() != 3.0f || cache.value.back() != 4.0f) {
        throw std::runtime_error("KV cache GGML-order offset mismatch");
    }
    cache.used = 2;
    if (cache.used > cache.capacity) {
        throw std::runtime_error("KV cache used/capacity mismatch");
    }
}

std::vector<float> run_kv_decode_graph_layer(const std::string & model_path, int layer, const std::vector<float> & hidden_input) {
    const TextConfig cfg = inspect_text_config(model_path);
    if (layer < 0 || layer >= cfg.block_count) {
        throw std::runtime_error("KV decode layer index out of range");
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    const int past = 1;
    const int total = past + 1;

    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    const std::vector<std::string> names{
        prefix + "attn_norm.weight",
        prefix + "attn_q.weight",
        prefix + "attn_k.weight",
        prefix + "attn_v.weight",
        prefix + "attn_output.weight",
        prefix + "attn_q_norm.weight",
        prefix + "attn_k_norm.weight",
        prefix + "ffn_norm.weight",
        prefix + "ffn_gate.weight",
        prefix + "ffn_up.weight",
        prefix + "ffn_down.weight",
    };
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load KV decode tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 512 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate KV decode graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * past_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, past);
    ggml_tensor * past_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, past);

    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, attn_norm);
    ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
    ggml_tensor * k_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
    ggml_tensor * v_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
    q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, 1);
    k_cur = ggml_reshape_3d(ctx.get(), k_cur, head_dim, n_head_kv, 1);
    v_cur = ggml_reshape_3d(ctx.get(), v_cur, head_dim, n_head_kv, 1);
    ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
    q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
    k_cur = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k_cur, cfg.rms_norm_eps), k_norm);
    q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(ctx.get(), k_cur, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * k = ggml_concat(ctx.get(), past_k, k_cur, 2);
    ggml_tensor * v = ggml_concat(ctx.get(), past_v, v_cur, 2);
    k = ggml_cont_3d(ctx.get(), k, head_dim, n_head_kv, total);
    v = ggml_cont_3d(ctx.get(), v, head_dim, n_head_kv, total);
    k = ggml_reshape_4d(ctx.get(), k, head_dim, 1, n_head_kv, total);
    v = ggml_reshape_4d(ctx.get(), v, head_dim, 1, n_head_kv, total);
    k = ggml_repeat_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, total);
    v = ggml_repeat_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, total);
    k = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, total), head_dim, n_head, total);
    v = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, total), head_dim, n_head, total);

    ggml_tensor * q4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), q, head_dim, n_head, 1, 1), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), k, head_dim, n_head, total, 1), 0, 2, 1, 3);
    ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
    kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    kq = ggml_soft_max(ctx.get(), kq);
    ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), v, head_dim, n_head, total, 1), 1, 2, 0, 3));
    ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
    ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, 1);
    ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
    ggml_tensor * cur = ggml_add(ctx.get(), hidden, attn_out);

    ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
    ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
    mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
    ggml_tensor * gate = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
    ggml_tensor * up = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
    ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
    ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
    ggml_tensor * out = ggml_add(ctx.get(), cur, mlp);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate KV decode graph tensors");
    }
    if (hidden_input.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("KV decode hidden input shape mismatch");
    }
    std::vector<float> past_k_values(static_cast<size_t>(head_dim * n_head_kv * past));
    std::vector<float> past_v_values(static_cast<size_t>(head_dim * n_head_kv * past));
    for (size_t i = 0; i < past_k_values.size(); ++i) {
        past_k_values[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 31.0f;
        past_v_values[i] = static_cast<float>(static_cast<int>(i % 29) - 14) / 29.0f;
    }
    const int32_t pos = past;
    ggml_backend_tensor_set(hidden, hidden_input.data(), 0, hidden_input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, &pos, 0, sizeof(pos));
    ggml_backend_tensor_set(past_k, past_k_values.data(), 0, past_k_values.size() * sizeof(float));
    ggml_backend_tensor_set(past_v, past_v_values.data(), 0, past_v_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("KV decode graph compute failed");
    }
    if (out->ne[0] != cfg.embedding_length || out->ne[1] != 1) {
        throw std::runtime_error("KV decode output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("KV decode output contains non-finite values");
        }
    }
    return values;
}

std::vector<float> run_kv_decode_graph_layer_with_cache(const std::string & model_path, int layer, const std::vector<float> & hidden_input, TextKvCache & cache) {
    const TextConfig cfg = inspect_text_config(model_path);
    if (layer < 0 || layer >= cfg.block_count || cache.used <= 0 || cache.used >= cache.capacity) {
        throw std::runtime_error("KV cache decode layer/cache range mismatch");
    }
    if (hidden_input.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("KV cache decode hidden input shape mismatch");
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    const int64_t past = cache.used;
    const int64_t total = past + 1;

    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    const std::vector<std::string> names{
        prefix + "attn_norm.weight",
        prefix + "attn_q.weight",
        prefix + "attn_k.weight",
        prefix + "attn_v.weight",
        prefix + "attn_output.weight",
        prefix + "attn_q_norm.weight",
        prefix + "attn_k_norm.weight",
        prefix + "ffn_norm.weight",
        prefix + "ffn_gate.weight",
        prefix + "ffn_up.weight",
        prefix + "ffn_down.weight",
    };
    if (!weights_buffer || !load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load KV cache decode tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 512 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * past_k = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, past);
    ggml_tensor * past_v = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, past);

    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, attn_norm);
    ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
    ggml_tensor * k_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
    ggml_tensor * v_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
    q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, 1);
    k_cur = ggml_reshape_3d(ctx.get(), k_cur, head_dim, n_head_kv, 1);
    v_cur = ggml_reshape_3d(ctx.get(), v_cur, head_dim, n_head_kv, 1);
    ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
    q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
    k_cur = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k_cur, cfg.rms_norm_eps), k_norm);
    q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(ctx.get(), k_cur, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * k = ggml_concat(ctx.get(), past_k, k_cur, 2);
    ggml_tensor * v = ggml_concat(ctx.get(), past_v, v_cur, 2);
    k = ggml_cont_3d(ctx.get(), k, head_dim, n_head_kv, total);
    v = ggml_cont_3d(ctx.get(), v, head_dim, n_head_kv, total);
    k = ggml_reshape_4d(ctx.get(), k, head_dim, 1, n_head_kv, total);
    v = ggml_reshape_4d(ctx.get(), v, head_dim, 1, n_head_kv, total);
    k = ggml_repeat_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, total);
    v = ggml_repeat_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, total);
    k = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, total), head_dim, n_head, total);
    v = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, total), head_dim, n_head, total);

    ggml_tensor * q4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), q, head_dim, n_head, 1, 1), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), k, head_dim, n_head, total, 1), 0, 2, 1, 3);
    ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
    kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    kq = ggml_soft_max(ctx.get(), kq);
    ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), v, head_dim, n_head, total, 1), 1, 2, 0, 3));
    ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
    ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, 1);
    ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
    ggml_tensor * cur = ggml_add(ctx.get(), hidden, attn_out);

    ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
    ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
    mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
    ggml_tensor * gate = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
    ggml_tensor * up = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
    ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
    ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
    ggml_tensor * out = ggml_add(ctx.get(), cur, mlp);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    ggml_build_forward_expand(graph, k_cur);
    ggml_build_forward_expand(graph, v_cur);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    std::vector<float> past_k_values(static_cast<size_t>(head_dim * n_head_kv * past));
    std::vector<float> past_v_values(past_k_values.size());
    for (int64_t pos = 0; pos < past; ++pos) {
        for (int64_t head = 0; head < n_head_kv; ++head) {
            for (int64_t dim = 0; dim < head_dim; ++dim) {
                const size_t src = kv_cache_offset(cache, layer, pos, head, dim);
                const size_t dst = static_cast<size_t>((pos * n_head_kv + head) * head_dim + dim);
                past_k_values[dst] = cache.key[src];
                past_v_values[dst] = cache.value[src];
            }
        }
    }
    const int32_t pos = static_cast<int32_t>(past);
    ggml_backend_tensor_set(hidden, hidden_input.data(), 0, hidden_input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, &pos, 0, sizeof(pos));
    ggml_backend_tensor_set(past_k, past_k_values.data(), 0, past_k_values.size() * sizeof(float));
    ggml_backend_tensor_set(past_v, past_v_values.data(), 0, past_v_values.size() * sizeof(float));
    if (!graph_buffer || ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("KV cache decode graph compute failed");
    }

    std::vector<float> values(static_cast<size_t>(cfg.embedding_length));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    std::vector<float> key(static_cast<size_t>(head_dim * n_head_kv));
    std::vector<float> value(key.size());
    ggml_backend_tensor_get(k_cur, key.data(), 0, key.size() * sizeof(float));
    ggml_backend_tensor_get(v_cur, value.data(), 0, value.size() * sizeof(float));
    for (int64_t head = 0; head < n_head_kv; ++head) {
        for (int64_t dim = 0; dim < head_dim; ++dim) {
            const size_t src = static_cast<size_t>(head * head_dim + dim);
            const size_t dst = kv_cache_offset(cache, layer, past, head, dim);
            cache.key[dst] = key[src];
            cache.value[dst] = value[src];
        }
    }
    for (float value_out : values) {
        if (!std::isfinite(value_out)) {
            throw std::runtime_error("KV cache decode output contains non-finite values");
        }
    }
    return values;
}

std::vector<float> run_kv_decode_graph_layer_with_backend_cache(const std::string & model_path, int layer, const std::vector<float> & hidden_input, BackendKvCache & cache) {
    const TextConfig & cfg = cache.cfg;
    if (layer < 0 || layer >= cfg.block_count || cache.used <= 0 || cache.used >= cache.capacity) {
        throw std::runtime_error("backend KV cache decode layer/cache range mismatch");
    }
    if (hidden_input.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("backend KV cache decode hidden input shape mismatch");
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    const int64_t past = cache.used;
    const int64_t total = past + 1;

    ggml_context * weights_ctx = cache.weights_ggml;
    const std::vector<std::string> names{
        prefix + "attn_norm.weight",
        prefix + "attn_q.weight",
        prefix + "attn_k.weight",
        prefix + "attn_v.weight",
        prefix + "attn_output.weight",
        prefix + "attn_q_norm.weight",
        prefix + "attn_k_norm.weight",
        prefix + "ffn_norm.weight",
        prefix + "ffn_gate.weight",
        prefix + "ffn_up.weight",
        prefix + "ffn_down.weight",
    };
    if (!ensure_backend_cache_weights(model_path, cache, names)) {
        throw std::runtime_error("failed to load backend KV cache decode tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 512 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    const size_t layer0 = backend_kv_cache_offset_bytes(cache, cache.key, layer, 0, 0, 0);
    ggml_tensor * past_k = ggml_view_3d(ctx.get(), cache.key, head_dim, n_head_kv, past, cache.key->nb[1], cache.key->nb[2], layer0);
    ggml_tensor * past_v = ggml_view_3d(ctx.get(), cache.value, head_dim, n_head_kv, past, cache.value->nb[1], cache.value->nb[2], layer0);

    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, attn_norm);
    ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
    ggml_tensor * k_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
    ggml_tensor * v_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
    q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, 1);
    k_cur = ggml_reshape_3d(ctx.get(), k_cur, head_dim, n_head_kv, 1);
    v_cur = ggml_reshape_3d(ctx.get(), v_cur, head_dim, n_head_kv, 1);
    ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
    q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
    k_cur = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k_cur, cfg.rms_norm_eps), k_norm);
    q = ggml_rope_ext(ctx.get(), q, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k_cur = ggml_rope_ext(ctx.get(), k_cur, positions, nullptr, head_dim, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    ggml_tensor * key_written = ggml_set_inplace(ctx.get(), cache.key, k_cur, cache.key->nb[1], cache.key->nb[2], cache.key->nb[3], backend_kv_cache_offset_bytes(cache, cache.key, layer, past, 0, 0));
    ggml_tensor * value_written = ggml_set_inplace(ctx.get(), cache.value, v_cur, cache.value->nb[1], cache.value->nb[2], cache.value->nb[3], backend_kv_cache_offset_bytes(cache, cache.value, layer, past, 0, 0));
    ggml_tensor * k = ggml_view_3d(ctx.get(), key_written, head_dim, n_head_kv, total, cache.key->nb[1], cache.key->nb[2], layer0);
    ggml_tensor * v = ggml_view_3d(ctx.get(), value_written, head_dim, n_head_kv, total, cache.value->nb[1], cache.value->nb[2], layer0);
    k = ggml_cont_3d(ctx.get(), k, head_dim, n_head_kv, total);
    v = ggml_cont_3d(ctx.get(), v, head_dim, n_head_kv, total);
    k = ggml_reshape_4d(ctx.get(), k, head_dim, 1, n_head_kv, total);
    v = ggml_reshape_4d(ctx.get(), v, head_dim, 1, n_head_kv, total);
    k = ggml_repeat_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, total);
    v = ggml_repeat_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, total);
    k = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), k, head_dim, n_rep, n_head_kv, total), head_dim, n_head, total);
    v = ggml_reshape_3d(ctx.get(), ggml_cont_4d(ctx.get(), v, head_dim, n_rep, n_head_kv, total), head_dim, n_head, total);

    ggml_tensor * q4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), q, head_dim, n_head, 1, 1), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), k, head_dim, n_head, total, 1), 0, 2, 1, 3);
    ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
    kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
    kq = ggml_soft_max(ctx.get(), kq);
    ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), ggml_reshape_4d(ctx.get(), v, head_dim, n_head, total, 1), 1, 2, 0, 3));
    ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
    ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
    attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, 1);
    ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
    ggml_tensor * cur = ggml_add(ctx.get(), hidden, attn_out);

    ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
    ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
    mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
    ggml_tensor * gate = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
    ggml_tensor * up = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
    ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
    ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
    ggml_tensor * out = ggml_add(ctx.get(), cur, mlp);

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer;
    if (!reference_kv_gallocr_disabled()) {
        if (!cache.window_galloc) {
            cache.window_galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache.backend)));
        }
        if (!cache.window_galloc || !ggml_gallocr_alloc_graph(cache.window_galloc.get(), graph)) {
            throw std::runtime_error("backend KV cache all-layer decode gallocr allocation failed");
        }
    } else {
        graph_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
    }
    const int32_t pos = static_cast<int32_t>(past);
    ggml_backend_tensor_set(hidden, hidden_input.data(), 0, hidden_input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, &pos, 0, sizeof(pos));
    if (!graph_buffer || ggml_backend_graph_compute(cache.backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("backend KV cache decode graph compute failed");
    }
    std::vector<float> values(static_cast<size_t>(cfg.embedding_length));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value_out : values) {
        if (!std::isfinite(value_out)) {
            throw std::runtime_error("backend KV cache decode output contains non-finite values");
        }
    }
    return values;
}

KvDecodeResult run_kv_decode_graph_all_layers_with_backend_cache_ex(const std::string & model_path, const std::vector<float> & hidden_input, BackendKvCache & cache, bool include_audio_logits, bool skip_hidden_readback) {
    const TextConfig & cfg = cache.cfg;
    if (cache.used <= 0 || cache.used >= cache.capacity) {
        throw std::runtime_error("backend KV cache all-layer decode cache range mismatch");
    }
    if (hidden_input.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("backend KV cache all-layer decode hidden input shape mismatch");
    }
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    const int64_t past = cache.used;
    const int64_t total = past + 1;

    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(cfg.block_count * 11 + 2));
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        names.push_back(prefix + "attn_norm.weight");
        names.push_back(prefix + "attn_q.weight");
        names.push_back(prefix + "attn_k.weight");
        names.push_back(prefix + "attn_v.weight");
        names.push_back(prefix + "attn_output.weight");
        names.push_back(prefix + "attn_q_norm.weight");
        names.push_back(prefix + "attn_k_norm.weight");
        names.push_back(prefix + "ffn_norm.weight");
        names.push_back(prefix + "ffn_gate.weight");
        names.push_back(prefix + "ffn_up.weight");
        names.push_back(prefix + "ffn_down.weight");
    }
    if (include_audio_logits) {
        names.push_back("output_norm.weight");
        names.push_back("a.output");
    }
    if (!ensure_backend_cache_weights(model_path, cache, names)) {
        throw std::runtime_error("failed to load backend KV cache all-layer decode tensors from GGUF");
    }
    ggml_context * weights_ctx = cache.weights_ggml;

    ggml_init_params params{};
    params.mem_size = 1024ull * 1024ull * 1024ull;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate backend KV cache all-layer decode graph context");
    }

    ensure_backend_kv_rope_tables(cache);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * rope_cos = cache.rope_cos;
    ggml_tensor * rope_sin = cache.rope_sin;
    ggml_tensor * cur_hidden = hidden;
    ggml_tensor * layer0_hidden = nullptr;
    ggml_tensor * layer0_attn_normed = nullptr;
    ggml_tensor * layer0_q_proj = nullptr;
    ggml_tensor * layer0_v_proj = nullptr;
    ggml_tensor * layer0_q_normed = nullptr;
    ggml_tensor * layer0_q = nullptr;
    ggml_tensor * layer0_k_cur = nullptr;
    ggml_tensor * layer0_v_cur = nullptr;
    ggml_tensor * layer0_k = nullptr;
    ggml_tensor * layer0_v = nullptr;
    ggml_tensor * layer0_kq = nullptr;
    ggml_tensor * layer0_softmax = nullptr;
    ggml_tensor * layer0_attn_out = nullptr;
    ggml_tensor * layer0_post_attn = nullptr;
    ggml_tensor * layer0_mlp_normed = nullptr;
    ggml_tensor * layer0_mlp_gate = nullptr;
    ggml_tensor * layer0_mlp_up = nullptr;
    ggml_tensor * layer0_mlp_gated = nullptr;
    ggml_tensor * layer0_mlp_out = nullptr;
    std::vector<ggml_tensor *> layer_hidden_outputs(static_cast<size_t>(cfg.block_count), nullptr);
    std::vector<ggml_tensor *> layer_post_attn_outputs(static_cast<size_t>(cfg.block_count), nullptr);
    std::vector<ggml_tensor *> layer_mlp_outputs(static_cast<size_t>(cfg.block_count), nullptr);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const size_t layer0 = backend_kv_cache_offset_bytes(cache, cache.key, layer, 0, 0, 0);

        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
        normed = ggml_mul(ctx.get(), normed, attn_norm);
        ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
        ggml_tensor * k_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
        ggml_tensor * v_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_attn_normed = normed;
            layer0_q_proj = q;
            layer0_v_proj = v_cur;
            ggml_set_output(layer0_attn_normed);
            ggml_set_output(layer0_q_proj);
            ggml_set_output(layer0_v_proj);
        }
        q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, 1);
        k_cur = ggml_reshape_3d(ctx.get(), k_cur, head_dim, n_head_kv, 1);
        v_cur = ggml_reshape_3d(ctx.get(), v_cur, head_dim, n_head_kv, 1);
        ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
        ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
        q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_q_normed = q;
            ggml_set_output(layer0_q_normed);
        }
        k_cur = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k_cur, cfg.rms_norm_eps), k_norm);
        q = neox_rope_bf16_cache(ctx.get(), q, positions, rope_cos, rope_sin);
        k_cur = neox_rope_bf16_cache(ctx.get(), k_cur, positions, rope_cos, rope_sin);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(k_cur, "kvstage:reference_kv_decode_attn_project_rope");
        }
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_q = q;
            layer0_k_cur = k_cur;
            layer0_v_cur = v_cur;
            ggml_set_output(layer0_q);
            ggml_set_output(layer0_k_cur);
            ggml_set_output(layer0_v_cur);
        }

        ggml_tensor * key_written = ggml_set_inplace(ctx.get(), cache.key, k_cur, cache.key->nb[1], cache.key->nb[2], cache.key->nb[3], backend_kv_cache_offset_bytes(cache, cache.key, layer, past, 0, 0));
        ggml_tensor * value_written = ggml_set_inplace(ctx.get(), cache.value, v_cur, cache.value->nb[1], cache.value->nb[2], cache.value->nb[3], backend_kv_cache_offset_bytes(cache, cache.value, layer, past, 0, 0));
        ggml_tensor * k = ggml_view_3d(ctx.get(), key_written, head_dim, n_head_kv, total, cache.key->nb[1], cache.key->nb[2], layer0);
        ggml_tensor * v = ggml_view_3d(ctx.get(), value_written, head_dim, n_head_kv, total, cache.value->nb[1], cache.value->nb[2], layer0);
        k = repeat_kv_3d(ctx.get(), k, n_rep);
        v = repeat_kv_3d(ctx.get(), v, n_rep);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(v, "kvstage:reference_kv_decode_kv_write_view_repeat");
        }
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_k = k;
            layer0_v = v;
            ggml_set_output(layer0_k);
            ggml_set_output(layer0_v);
        }

        ggml_tensor * q4 = ggml_permute(ctx.get(), as_4d(ctx.get(), q), 0, 2, 1, 3);
        ggml_tensor * k4 = ggml_permute(ctx.get(), as_4d(ctx.get(), k), 0, 2, 1, 3);
        ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
        kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(kq, "kvstage:reference_kv_decode_qk_matmul_scale");
        }
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_kq = kq;
            ggml_set_output(layer0_kq);
        }
        kq = ggml_soft_max(ctx.get(), kq);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(kq, "kvstage:reference_kv_decode_softmax");
        }
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_softmax = kq;
            ggml_set_output(layer0_softmax);
        }
        ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), as_4d(ctx.get(), v), 1, 2, 0, 3));
        ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
        ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
        attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, 1);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(attn, "kvstage:reference_kv_decode_v_matmul_cont");
        }
        ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
        ggml_tensor * cur = ggml_add(ctx.get(), cur_hidden, attn_out);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(cur, "kvstage:reference_kv_decode_attention_output_residual");
        }
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_attn_out = attn_out;
            layer0_post_attn = cur;
            ggml_set_output(layer0_attn_out);
            ggml_set_output(layer0_post_attn);
        }
        if (!trace_json_path().empty() && (layer == 34 || layer == 35)) {
            layer_post_attn_outputs[static_cast<size_t>(layer)] = cur;
            ggml_set_output(cur);
        }

        ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
        ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
        mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(mlp_normed, "kvstage:reference_kv_decode_ffn_norm");
        }
        ggml_tensor * gate = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(gate, "kvstage:reference_kv_decode_ffn_gate_matmul");
        }
        ggml_tensor * up = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(up, "kvstage:reference_kv_decode_ffn_up_matmul");
        }
        ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(gated, "kvstage:reference_kv_decode_ffn_silu_mul");
        }
        ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(mlp, "kvstage:reference_kv_decode_ffn_down_matmul");
        }
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_mlp_normed = mlp_normed;
            layer0_mlp_gate = gate;
            layer0_mlp_up = up;
            layer0_mlp_gated = gated;
            layer0_mlp_out = mlp;
            ggml_set_output(layer0_mlp_normed);
            ggml_set_output(layer0_mlp_gate);
            ggml_set_output(layer0_mlp_up);
            ggml_set_output(layer0_mlp_gated);
            ggml_set_output(layer0_mlp_out);
        }
        if (!trace_json_path().empty() && (layer == 34 || layer == 35)) {
            layer_mlp_outputs[static_cast<size_t>(layer)] = mlp;
            ggml_set_output(mlp);
        }
        cur_hidden = ggml_add(ctx.get(), cur, mlp);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(cur_hidden, "kvstage:reference_kv_decode_ffn_residual_add");
        }
        if (layer == 0 && !trace_json_path().empty()) {
            layer0_hidden = cur_hidden;
            ggml_set_output(layer0_hidden);
        }
        if (!trace_json_path().empty()) {
            layer_hidden_outputs[static_cast<size_t>(layer)] = cur_hidden;
            ggml_set_output(cur_hidden);
        }
    }

    ggml_tensor * logits_flat = nullptr;
    if (include_audio_logits) {
        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
        ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
        logits_flat = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
    }
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), static_cast<size_t>(cfg.block_count * 128 + 1024), false);
    ggml_build_forward_expand(graph, logits_flat != nullptr ? logits_flat : cur_hidden);
    BackendBufferPtr graph_buffer;
    const bool use_stage_scheduler = reference_kv_stage_profile_enabled();
    {
        ScopedProfileTimer timer("reference_ar_kv_decode_alloc_graph");
        if (use_stage_scheduler) {
            if (!cache.sched_cpu_backend) {
                cache.sched_cpu_backend.reset(ggml_backend_cpu_init());
                if (!cache.sched_cpu_backend) {
                    throw std::runtime_error("backend KV cache stage profile CPU backend init failed");
                }
            }
            if (!cache.decode_sched) {
                ggml_backend_t backends[] = { cache.backend, cache.sched_cpu_backend.get() };
                ggml_backend_buffer_type_t bufts[] = {
                    ggml_backend_get_default_buffer_type(cache.backend),
                    ggml_backend_get_default_buffer_type(cache.sched_cpu_backend.get()),
                };
                const size_t graph_size = std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, static_cast<size_t>(cfg.block_count * 128 + 1024));
                cache.decode_sched.reset(ggml_backend_sched_new(backends, bufts, 2, graph_size, false, true));
                if (!cache.decode_sched) {
                    throw std::runtime_error("backend KV cache stage profile scheduler init failed");
                }
            }
            ggml_backend_sched_reset(cache.decode_sched.get());
            if (!ggml_backend_sched_alloc_graph(cache.decode_sched.get(), graph)) {
                throw std::runtime_error("backend KV cache stage profile scheduler graph allocation failed");
            }
        } else if (!reference_kv_gallocr_disabled()) {
            if (!cache.decode_galloc) {
                cache.decode_galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache.backend)));
            }
            if (!cache.decode_galloc || !ggml_gallocr_alloc_graph(cache.decode_galloc.get(), graph)) {
                throw std::runtime_error("backend KV cache all-layer decode gallocr allocation failed");
            }
        } else {
            graph_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
        }
    }
    const int32_t pos = static_cast<int32_t>(past);
    {
        ScopedProfileTimer timer("reference_ar_kv_decode_set_inputs");
        ggml_backend_tensor_set(hidden, hidden_input.data(), 0, hidden_input.size() * sizeof(float));
        ggml_backend_tensor_set(positions, &pos, 0, sizeof(pos));
    }
    if (!use_stage_scheduler && reference_kv_gallocr_disabled() && !graph_buffer) {
        throw std::runtime_error("backend KV cache all-layer decode graph tensor allocation failed");
    }
    {
        ScopedProfileTimer timer("reference_ar_kv_decode_compute_graph");
        ReferenceKvStageProfileState stage_profile_state{std::chrono::steady_clock::now()};
        if (use_stage_scheduler) {
            ggml_backend_sched_set_eval_callback(cache.decode_sched.get(), reference_kv_stage_profile_callback, &stage_profile_state);
        }
        const ggml_status status = use_stage_scheduler
            ? ggml_backend_sched_graph_compute_async(cache.decode_sched.get(), graph)
            : ggml_backend_graph_compute(cache.backend, graph);
        if (use_stage_scheduler) {
            ggml_backend_sched_set_eval_callback(cache.decode_sched.get(), nullptr, nullptr);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("backend KV cache all-layer decode graph compute failed");
        }
        if (use_stage_scheduler) {
            ggml_backend_sched_synchronize(cache.decode_sched.get());
        }
    }
    KvDecodeResult result;
    if (logits_flat != nullptr) {
        std::vector<float> logits(static_cast<size_t>(k_num_codebooks * k_codebook_vocab_size));
        {
            ScopedProfileTimer timer("reference_ar_kv_decode_get_logits");
            ggml_backend_tensor_get(logits_flat, logits.data(), 0, logits.size() * sizeof(float));
        }
        result.logits_vc = audio_logits_flat_to_vc(logits, 1, 0);
    }
    if (!skip_hidden_readback || !trace_json_path().empty()) {
        result.hidden.resize(static_cast<size_t>(cfg.embedding_length));
        {
            ScopedProfileTimer timer("reference_ar_kv_decode_get_hidden");
            ggml_backend_tensor_get(cur_hidden, result.hidden.data(), 0, result.hidden.size() * sizeof(float));
        }
    }
    if (!trace_json_path().empty()) {
        const std::string dump_suffix = "_past" + std::to_string(past);
        dump_node_values("cpp_kv_decode_input" + dump_suffix, hidden_input);
        if (layer0_hidden != nullptr) {
            std::vector<float> layer0_values(static_cast<size_t>(cfg.embedding_length));
            ggml_backend_tensor_get(layer0_hidden, layer0_values.data(), 0, layer0_values.size() * sizeof(float));
            dump_node_values("cpp_kv_decode_blk0_hidden" + dump_suffix, layer0_values);
            g_trace_node_stats["kv_decode_blk.0.hidden"] = make_last_token_stat(layer0_values, cfg.embedding_length, 1);
        }
        if (layer0_attn_normed != nullptr && layer0_q_proj != nullptr && layer0_v_proj != nullptr && layer0_q_normed != nullptr) {
            std::vector<float> normed_values(static_cast<size_t>(ggml_nelements(layer0_attn_normed)));
            std::vector<float> q_proj_values(static_cast<size_t>(ggml_nelements(layer0_q_proj)));
            std::vector<float> v_proj_values(static_cast<size_t>(ggml_nelements(layer0_v_proj)));
            std::vector<float> q_normed_values(static_cast<size_t>(ggml_nelements(layer0_q_normed)));
            ggml_backend_tensor_get(layer0_attn_normed, normed_values.data(), 0, normed_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_q_proj, q_proj_values.data(), 0, q_proj_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_v_proj, v_proj_values.data(), 0, v_proj_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_q_normed, q_normed_values.data(), 0, q_normed_values.size() * sizeof(float));
            dump_node_values("cpp_kv_decode_blk0_attn_normed" + dump_suffix, normed_values);
            dump_node_values("cpp_kv_decode_blk0_q_proj" + dump_suffix, q_proj_values);
            dump_node_values("cpp_kv_decode_blk0_q_normed" + dump_suffix, q_normed_values);
            g_trace_node_stats["kv_decode_blk.0.attn_normed"] = make_last_token_stat(normed_values, cfg.embedding_length, 1);
            g_trace_node_stats["kv_decode_blk.0.q_proj"] = make_last_token_stat(q_proj_values, head_dim * n_head, 1);
            g_trace_node_stats["kv_decode_blk.0.v_proj"] = make_last_token_stat(v_proj_values, head_dim * n_head_kv, 1);
            g_trace_node_stats["kv_decode_blk.0.q_normed"] = make_last_token_stat(q_normed_values, head_dim * n_head, 1);
        }
        if (layer0_q != nullptr && layer0_k_cur != nullptr && layer0_v_cur != nullptr && layer0_k != nullptr && layer0_v != nullptr &&
            layer0_kq != nullptr && layer0_softmax != nullptr && layer0_attn_out != nullptr && layer0_post_attn != nullptr) {
            std::vector<float> q_values(static_cast<size_t>(ggml_nelements(layer0_q)));
            std::vector<float> k_cur_values(static_cast<size_t>(ggml_nelements(layer0_k_cur)));
            std::vector<float> v_cur_values(static_cast<size_t>(ggml_nelements(layer0_v_cur)));
            std::vector<float> k_values(static_cast<size_t>(ggml_nelements(layer0_k)));
            std::vector<float> v_values(static_cast<size_t>(ggml_nelements(layer0_v)));
            std::vector<float> kq_values(static_cast<size_t>(ggml_nelements(layer0_kq)));
            std::vector<float> softmax_values(static_cast<size_t>(ggml_nelements(layer0_softmax)));
            std::vector<float> attn_values(static_cast<size_t>(ggml_nelements(layer0_attn_out)));
            std::vector<float> post_values(static_cast<size_t>(ggml_nelements(layer0_post_attn)));
            ggml_backend_tensor_get(layer0_q, q_values.data(), 0, q_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_k_cur, k_cur_values.data(), 0, k_cur_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_v_cur, v_cur_values.data(), 0, v_cur_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_k, k_values.data(), 0, k_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_v, v_values.data(), 0, v_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_kq, kq_values.data(), 0, kq_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_softmax, softmax_values.data(), 0, softmax_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_attn_out, attn_values.data(), 0, attn_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_post_attn, post_values.data(), 0, post_values.size() * sizeof(float));
            dump_node_values("cpp_kv_decode_blk0_q_state" + dump_suffix, q_values);
            dump_node_values("cpp_kv_decode_blk0_k_cur" + dump_suffix, k_cur_values);
            dump_node_values("cpp_kv_decode_blk0_v_cur" + dump_suffix, v_cur_values);
            dump_node_values("cpp_kv_decode_blk0_k_state" + dump_suffix, k_values);
            dump_node_values("cpp_kv_decode_blk0_v_state" + dump_suffix, v_values);
            dump_node_values("cpp_kv_decode_blk0_kq" + dump_suffix, kq_values);
            dump_node_values("cpp_kv_decode_blk0_softmax" + dump_suffix, softmax_values);
            dump_node_values("cpp_kv_decode_blk0_attn_out" + dump_suffix, attn_values);
            dump_node_values("cpp_kv_decode_blk0_post_attn" + dump_suffix, post_values);
            g_trace_node_stats["kv_decode_blk.0.q_state"] = make_last_token_stat(q_values, head_dim * n_head, 1);
            g_trace_node_stats["kv_decode_blk.0.k_cur"] = make_last_token_stat(k_cur_values, head_dim * n_head_kv, 1);
            g_trace_node_stats["kv_decode_blk.0.v_cur"] = make_last_token_stat(v_cur_values, head_dim * n_head_kv, 1);
            g_trace_node_stats["kv_decode_blk.0.k_state"] = make_last_token_stat(k_values, head_dim * n_head, total);
            g_trace_node_stats["kv_decode_blk.0.v_state"] = make_last_token_stat(v_values, head_dim * n_head, total);
            g_trace_node_stats["kv_decode_blk.0.kq"] = make_last_token_stat(kq_values, total, n_head);
            g_trace_node_stats["kv_decode_blk.0.softmax"] = make_last_token_stat(softmax_values, total, n_head);
            g_trace_node_stats["kv_decode_blk.0.attn_out"] = make_last_token_stat(attn_values, cfg.embedding_length, 1);
            g_trace_node_stats["kv_decode_blk.0.post_attn"] = make_last_token_stat(post_values, cfg.embedding_length, 1);
        }
        if (layer0_mlp_normed != nullptr && layer0_mlp_gate != nullptr && layer0_mlp_up != nullptr && layer0_mlp_gated != nullptr && layer0_mlp_out != nullptr) {
            std::vector<float> mlp_normed_values(static_cast<size_t>(ggml_nelements(layer0_mlp_normed)));
            std::vector<float> gate_values(static_cast<size_t>(ggml_nelements(layer0_mlp_gate)));
            std::vector<float> up_values(static_cast<size_t>(ggml_nelements(layer0_mlp_up)));
            std::vector<float> gated_values(static_cast<size_t>(ggml_nelements(layer0_mlp_gated)));
            std::vector<float> mlp_values(static_cast<size_t>(ggml_nelements(layer0_mlp_out)));
            ggml_backend_tensor_get(layer0_mlp_normed, mlp_normed_values.data(), 0, mlp_normed_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_mlp_gate, gate_values.data(), 0, gate_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_mlp_up, up_values.data(), 0, up_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_mlp_gated, gated_values.data(), 0, gated_values.size() * sizeof(float));
            ggml_backend_tensor_get(layer0_mlp_out, mlp_values.data(), 0, mlp_values.size() * sizeof(float));
            dump_node_values("cpp_kv_decode_blk0_mlp_normed" + dump_suffix, mlp_normed_values);
            dump_node_values("cpp_kv_decode_blk0_mlp_gate" + dump_suffix, gate_values);
            dump_node_values("cpp_kv_decode_blk0_mlp_up" + dump_suffix, up_values);
            dump_node_values("cpp_kv_decode_blk0_mlp_gated" + dump_suffix, gated_values);
            dump_node_values("cpp_kv_decode_blk0_mlp_out" + dump_suffix, mlp_values);
        }
        for (int layer = 0; layer < cfg.block_count; ++layer) {
            ggml_tensor * layer_tensor = layer_hidden_outputs[static_cast<size_t>(layer)];
            if (layer_tensor == nullptr) {
                continue;
            }
            std::vector<float> layer_values(static_cast<size_t>(cfg.embedding_length));
            ggml_backend_tensor_get(layer_tensor, layer_values.data(), 0, layer_values.size() * sizeof(float));
            dump_node_values("cpp_kv_decode_blk" + std::to_string(layer) + "_hidden" + dump_suffix, layer_values);
        }
        for (int layer : {34, 35}) {
            ggml_tensor * post_tensor = layer_post_attn_outputs[static_cast<size_t>(layer)];
            ggml_tensor * mlp_tensor = layer_mlp_outputs[static_cast<size_t>(layer)];
            if (post_tensor == nullptr || mlp_tensor == nullptr) {
                continue;
            }
            std::vector<float> post_values(static_cast<size_t>(cfg.embedding_length));
            std::vector<float> mlp_values(static_cast<size_t>(cfg.embedding_length));
            ggml_backend_tensor_get(post_tensor, post_values.data(), 0, post_values.size() * sizeof(float));
            ggml_backend_tensor_get(mlp_tensor, mlp_values.data(), 0, mlp_values.size() * sizeof(float));
            dump_node_values("cpp_kv_decode_blk" + std::to_string(layer) + "_post_attn" + dump_suffix, post_values);
            dump_node_values("cpp_kv_decode_blk" + std::to_string(layer) + "_mlp_out" + dump_suffix, mlp_values);
        }
        g_trace_node_stats["kv_decode_blk.35.hidden"] = make_last_token_stat(result.hidden, cfg.embedding_length, 1);
    }
    for (float value_out : result.hidden) {
        if (!std::isfinite(value_out)) {
            throw std::runtime_error("backend KV cache all-layer decode output contains non-finite values");
        }
    }
    return result;
}

std::vector<float> run_kv_decode_graph_all_layers_with_backend_cache(const std::string & model_path, const std::vector<float> & hidden_input, BackendKvCache & cache) {
    return run_kv_decode_graph_all_layers_with_backend_cache_ex(model_path, hidden_input, cache).hidden;
}

void reserve_kv_decode_window_full_scheduler_graph(const std::string & model_path, BackendKvCache & cache, bool include_logits) {
    const TextConfig & cfg = cache.cfg;
    if (!cache.window_sched || cache.window_sched_reserved_tokens >= cache.capacity) {
        return;
    }
    {
        ScopedProfileTimer timer("reference_kv_sched_reserve_ensure_weights");
        std::vector<std::string> names = all_block_weight_names(cfg);
        if (include_logits) {
            names.push_back("output_norm.weight");
            names.push_back("a.output");
        }
        if (!ensure_backend_cache_weights(model_path, cache, names)) {
            throw std::runtime_error("failed to load backend KV cache scheduler reserve tensors from GGUF");
        }
        ensure_backend_cache_fused_ffn_weights(cache);
    }
    ggml_context * weights_ctx = cache.weights_ggml;
    GgmlPtr ctx;
    {
        ScopedProfileTimer timer("reference_kv_sched_reserve_init_context");
        ggml_init_params params{};
        params.mem_size = 1024ull * 1024ull * 1024ull;
        params.no_alloc = true;
        ctx.reset(ggml_init(params));
    }
    if (!ctx) {
        throw std::runtime_error("failed to allocate backend KV cache scheduler reserve graph context");
    }

    const int64_t tokens = cache.capacity;
    const int64_t start = 0;
    const int64_t total = cache.capacity;
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * rope_cos = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim / 2, cfg.context_length);
    ggml_tensor * rope_sin = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim / 2, cfg.context_length);
    ggml_tensor * dummy_key = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, cache.capacity, cfg.block_count);
    ggml_tensor * dummy_value = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, head_dim, n_head_kv, cache.capacity, cfg.block_count);
    ggml_tensor * cur_hidden = hidden;
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const size_t layer0 = static_cast<size_t>(layer) * dummy_key->nb[3];

        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
        normed = ggml_mul(ctx.get(), normed, attn_norm);
        ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
        ggml_tensor * k_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
        ggml_tensor * v_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
        q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, tokens);
        k_cur = ggml_reshape_3d(ctx.get(), k_cur, head_dim, n_head_kv, tokens);
        v_cur = ggml_reshape_3d(ctx.get(), v_cur, head_dim, n_head_kv, tokens);
        ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
        ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
        q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
        k_cur = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k_cur, cfg.rms_norm_eps), k_norm);
        q = neox_rope_bf16_cache(ctx.get(), q, positions, rope_cos, rope_sin);
        k_cur = neox_rope_bf16_cache(ctx.get(), k_cur, positions, rope_cos, rope_sin);

        ggml_tensor * key_written = ggml_set_inplace(ctx.get(), dummy_key, k_cur, dummy_key->nb[1], dummy_key->nb[2], dummy_key->nb[3], layer0);
        ggml_tensor * value_written = ggml_set_inplace(ctx.get(), dummy_value, v_cur, dummy_value->nb[1], dummy_value->nb[2], dummy_value->nb[3], layer0);
        ggml_tensor * k = ggml_view_3d(ctx.get(), key_written, head_dim, n_head_kv, total, dummy_key->nb[1], dummy_key->nb[2], layer0);
        ggml_tensor * v = ggml_view_3d(ctx.get(), value_written, head_dim, n_head_kv, total, dummy_value->nb[1], dummy_value->nb[2], layer0);
        k = repeat_kv_3d(ctx.get(), k, n_rep);
        v = repeat_kv_3d(ctx.get(), v, n_rep);

        ggml_tensor * q4 = ggml_permute(ctx.get(), as_4d(ctx.get(), q), 0, 2, 1, 3);
        ggml_tensor * k4 = ggml_permute(ctx.get(), as_4d(ctx.get(), k), 0, 2, 1, 3);
        ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
        kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        kq = ggml_diag_mask_inf(ctx.get(), kq, static_cast<int>(start));
        kq = ggml_soft_max(ctx.get(), kq);
        ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), as_4d(ctx.get(), v), 1, 2, 0, 3));
        ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
        ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
        attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, tokens);
        ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
        ggml_tensor * cur = ggml_add(ctx.get(), cur_hidden, attn_out);

        ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
        ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
        mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        build_reference_ffn_gate_up(ctx.get(), cfg, cache, weights_ctx, layer, mlp_normed, tokens, &gate, &up);
        ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
        ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
        cur_hidden = ggml_add(ctx.get(), cur, mlp);
    }

    ggml_tensor * output = cur_hidden;
    if (include_logits) {
        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
        ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
        output = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
    }
    ggml_cgraph * graph = nullptr;
    {
        ScopedProfileTimer timer("reference_kv_sched_reserve_build_graph");
        graph = ggml_new_graph_custom(ctx.get(), static_cast<size_t>(cfg.block_count * 128 + 1024), false);
        ggml_build_forward_expand(graph, output);
    }
    {
        ScopedProfileTimer timer("reference_kv_sched_reserve_graph");
        if (!ggml_backend_sched_reserve(cache.window_sched.get(), graph)) {
            throw std::runtime_error("backend KV cache scheduler max graph reserve failed");
        }
    }
    cache.window_sched_reserved_tokens = cache.capacity;
}

std::vector<float> run_kv_decode_window_all_layers_full_with_backend_cache(const std::string & model_path, const std::vector<float> & hidden_input, int64_t tokens, BackendKvCache & cache, std::vector<float> * logits_vc_out, bool return_last_hidden_only, bool skip_hidden_readback) {
    const TextConfig & cfg = cache.cfg;
    if (tokens <= 0 || hidden_input.size() != static_cast<size_t>(cfg.embedding_length * tokens)) {
        throw std::runtime_error("backend KV cache window decode hidden input shape mismatch");
    }
    if (cache.used < tokens || cache.used > cache.capacity) {
        throw std::runtime_error("backend KV cache window decode cache range mismatch");
    }
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;
    const int64_t start = cache.used - tokens;
    const int64_t total = cache.used;

    {
        ScopedProfileTimer timer("reference_kv_window_ensure_weights");
        std::vector<std::string> names = all_block_weight_names(cfg);
        if (logits_vc_out != nullptr) {
            names.push_back("output_norm.weight");
            names.push_back("a.output");
        }
        if (!ensure_backend_cache_weights(model_path, cache, names)) {
            throw std::runtime_error("failed to load backend KV cache window decode tensors from GGUF");
        }
        ensure_backend_cache_fused_ffn_weights(cache);
    }
    ggml_context * weights_ctx = cache.weights_ggml;

    GgmlPtr ctx;
    {
        ScopedProfileTimer timer("reference_kv_window_init_context");
        ggml_init_params params{};
        params.mem_size = 1024ull * 1024ull * 1024ull;
        params.no_alloc = true;
        ctx.reset(ggml_init(params));
    }
    if (!ctx) {
        throw std::runtime_error("failed to allocate backend KV cache window decode graph context");
    }

    const auto construct_start = std::chrono::steady_clock::now();
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * rope_cos = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim / 2, cfg.context_length);
    ggml_tensor * rope_sin = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim / 2, cfg.context_length);
    ggml_tensor * cur_hidden = hidden;
    std::vector<ggml_tensor *> layer_hidden_outputs(static_cast<size_t>(cfg.block_count), nullptr);
    ggml_tensor * layer0_attn_normed = nullptr;
    ggml_tensor * layer0_q_proj = nullptr;
    ggml_tensor * layer0_q_normed = nullptr;
    ggml_tensor * layer0_k_cur = nullptr;
    ggml_tensor * layer0_v_cur = nullptr;
    ggml_tensor * layer0_kq = nullptr;
    ggml_tensor * layer0_softmax = nullptr;
    ggml_tensor * layer0_attn_out = nullptr;
    ggml_tensor * layer0_post_attn = nullptr;
    ggml_tensor * layer0_mlp_normed = nullptr;
    ggml_tensor * layer0_mlp_gate = nullptr;
    ggml_tensor * layer0_mlp_up = nullptr;
    ggml_tensor * layer0_mlp_gated = nullptr;
    ggml_tensor * layer0_mlp_out = nullptr;
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const size_t layer0 = backend_kv_cache_offset_bytes(cache, cache.key, layer, 0, 0, 0);

        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
        normed = ggml_mul(ctx.get(), normed, attn_norm);
        ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
        ggml_tensor * k_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
        ggml_tensor * v_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
        q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, tokens);
        k_cur = ggml_reshape_3d(ctx.get(), k_cur, head_dim, n_head_kv, tokens);
        v_cur = ggml_reshape_3d(ctx.get(), v_cur, head_dim, n_head_kv, tokens);
        ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
        ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
        q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
        k_cur = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k_cur, cfg.rms_norm_eps), k_norm);
        q = neox_rope_bf16_cache(ctx.get(), q, positions, rope_cos, rope_sin);
        k_cur = neox_rope_bf16_cache(ctx.get(), k_cur, positions, rope_cos, rope_sin);
        if (layer == 0 && reference_kv_stage_profile_enabled()) {
            ggml_set_name(k_cur, "kvstage:reference_kv_stage_attn_project_rope");
        }
        if (layer == 0 && !node_dump_dir().empty()) {
            layer0_q_normed = q;
            layer0_k_cur = k_cur;
            layer0_v_cur = v_cur;
            ggml_set_output(layer0_q_normed);
            ggml_set_output(layer0_k_cur);
            ggml_set_output(layer0_v_cur);
        }

        ggml_tensor * key_written = ggml_set_inplace(ctx.get(), cache.key, k_cur, cache.key->nb[1], cache.key->nb[2], cache.key->nb[3], backend_kv_cache_offset_bytes(cache, cache.key, layer, start, 0, 0));
        ggml_tensor * value_written = ggml_set_inplace(ctx.get(), cache.value, v_cur, cache.value->nb[1], cache.value->nb[2], cache.value->nb[3], backend_kv_cache_offset_bytes(cache, cache.value, layer, start, 0, 0));
        ggml_tensor * k = ggml_view_3d(ctx.get(), key_written, head_dim, n_head_kv, total, cache.key->nb[1], cache.key->nb[2], layer0);
        ggml_tensor * v = ggml_view_3d(ctx.get(), value_written, head_dim, n_head_kv, total, cache.value->nb[1], cache.value->nb[2], layer0);
        k = repeat_kv_3d(ctx.get(), k, n_rep);
        v = repeat_kv_3d(ctx.get(), v, n_rep);
        if (layer == 0 && reference_kv_stage_profile_enabled()) {
            ggml_set_name(v, "kvstage:reference_kv_stage_kv_write_view_repeat");
        }

        ggml_tensor * q4 = ggml_permute(ctx.get(), as_4d(ctx.get(), q), 0, 2, 1, 3);
        ggml_tensor * k4 = ggml_permute(ctx.get(), as_4d(ctx.get(), k), 0, 2, 1, 3);
        ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
        kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        kq = ggml_diag_mask_inf(ctx.get(), kq, static_cast<int>(start));
        if (layer == 0 && reference_kv_stage_profile_enabled()) {
            ggml_set_name(kq, "kvstage:reference_kv_stage_qk_matmul_mask");
        }
        if (layer == 0 && !node_dump_dir().empty()) {
            layer0_kq = kq;
            ggml_set_output(layer0_kq);
        }
        kq = ggml_soft_max(ctx.get(), kq);
        if (layer == 0 && reference_kv_stage_profile_enabled()) {
            ggml_set_name(kq, "kvstage:reference_kv_stage_softmax");
        }
        if (layer == 0 && !node_dump_dir().empty()) {
            layer0_softmax = kq;
            ggml_set_output(layer0_softmax);
        }
        ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), as_4d(ctx.get(), v), 1, 2, 0, 3));
        ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
        ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
        attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, tokens);
        if (layer == 0 && reference_kv_stage_profile_enabled()) {
            ggml_set_name(attn, "kvstage:reference_kv_stage_v_matmul");
        }
        ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
        ggml_tensor * cur = ggml_add(ctx.get(), cur_hidden, attn_out);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(cur, "kvstage:reference_kv_stage_all_attention");
        }
        if (layer == 0 && !node_dump_dir().empty()) {
            layer0_attn_out = attn_out;
            layer0_post_attn = cur;
            ggml_set_output(layer0_attn_out);
            ggml_set_output(layer0_post_attn);
        }

        ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
        ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
        mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(mlp_normed, "kvstage:reference_kv_ffn_norm");
        }
        ggml_tensor * gate = nullptr;
        ggml_tensor * up = nullptr;
        build_reference_ffn_gate_up(ctx.get(), cfg, cache, weights_ctx, layer, mlp_normed, tokens, &gate, &up);
        ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(gated, "kvstage:reference_kv_ffn_silu_mul");
        }
        ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(mlp, "kvstage:reference_kv_ffn_down_matmul");
        }
        if (layer == 0 && !node_dump_dir().empty()) {
            layer0_mlp_normed = mlp_normed;
            layer0_mlp_gate = gate;
            layer0_mlp_up = up;
            layer0_mlp_gated = gated;
            layer0_mlp_out = mlp;
            ggml_set_output(layer0_mlp_normed);
            ggml_set_output(layer0_mlp_gate);
            ggml_set_output(layer0_mlp_up);
            ggml_set_output(layer0_mlp_gated);
            ggml_set_output(layer0_mlp_out);
        }
        cur_hidden = ggml_add(ctx.get(), cur, mlp);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(cur_hidden, "kvstage:reference_kv_ffn_residual_add");
        }
        if (!node_dump_dir().empty()) {
            layer_hidden_outputs[static_cast<size_t>(layer)] = cur_hidden;
            ggml_set_output(cur_hidden);
        }
    }

    ggml_tensor * logits_flat = nullptr;
    if (logits_vc_out != nullptr) {
        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * norm_weight = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "output_norm.weight"), cfg.embedding_length, 1);
        ggml_tensor * final_hidden = ggml_mul(ctx.get(), normed, norm_weight);
        logits_flat = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, "a.output"), final_hidden);
        if (reference_kv_stage_profile_enabled()) {
            ggml_set_name(logits_flat, "kvstage:reference_kv_stage_logits_head");
        }
    }
    if (profile_enabled()) {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - construct_start).count();
        profile_record_elapsed("reference_kv_window_construct_ops", static_cast<long long>(elapsed));
        std::fprintf(stderr, "higgs_profile reference_kv_window_construct_ops ms=%lld\n", static_cast<long long>(elapsed));
    }
    ggml_cgraph * graph = nullptr;
    {
        ScopedProfileTimer timer("reference_kv_window_build_graph");
        graph = ggml_new_graph_custom(ctx.get(), static_cast<size_t>(cfg.block_count * 128 + 1024), false);
        ggml_build_forward_expand(graph, logits_flat != nullptr ? logits_flat : cur_hidden);
    }
    BackendBufferPtr graph_buffer;
    const bool use_scheduler = reference_kv_sched_enabled();
    {
        ScopedProfileTimer timer("reference_kv_window_alloc_graph");
        if (use_scheduler) {
            if (!cache.sched_cpu_backend) {
                cache.sched_cpu_backend.reset(ggml_backend_cpu_init());
                if (!cache.sched_cpu_backend) {
                    throw std::runtime_error("backend KV cache scheduler CPU backend init failed");
                }
            }
            if (!cache.window_sched) {
                ggml_backend_t backends[] = { cache.backend, cache.sched_cpu_backend.get() };
                ggml_backend_buffer_type_t bufts[] = {
                    ggml_backend_get_default_buffer_type(cache.backend),
                    ggml_backend_get_default_buffer_type(cache.sched_cpu_backend.get()),
                };
                const size_t graph_size = std::max<size_t>(GGML_DEFAULT_GRAPH_SIZE, static_cast<size_t>(cfg.block_count * 128 + 1024));
                cache.window_sched.reset(ggml_backend_sched_new(backends, bufts, 2, graph_size, false, true));
                if (!cache.window_sched) {
                    throw std::runtime_error("backend KV cache scheduler init failed");
                }
            }
            if (cache.window_sched_reserved_tokens < cache.capacity) {
                reserve_kv_decode_window_full_scheduler_graph(model_path, cache, logits_vc_out != nullptr);
            }
            ggml_backend_sched_reset(cache.window_sched.get());
            if (!ggml_backend_sched_alloc_graph(cache.window_sched.get(), graph)) {
                throw std::runtime_error("backend KV cache scheduler graph allocation failed");
            }
        } else if (!reference_kv_gallocr_disabled()) {
            if (!cache.window_galloc) {
                cache.window_galloc.reset(ggml_gallocr_new(ggml_backend_get_default_buffer_type(cache.backend)));
            }
            if (!cache.window_galloc || !ggml_gallocr_alloc_graph(cache.window_galloc.get(), graph)) {
                throw std::runtime_error("backend KV cache window decode gallocr allocation failed");
            }
        } else {
            graph_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
        }
    }
    std::vector<int32_t> position_values(static_cast<size_t>(tokens));
    for (int64_t i = 0; i < tokens; ++i) {
        position_values[static_cast<size_t>(i)] = static_cast<int32_t>(start + i);
    }
    {
        ScopedProfileTimer timer("reference_kv_window_set_inputs");
        ensure_backend_kv_rope_tables(cache);
        ggml_backend_tensor_set(hidden, hidden_input.data(), 0, hidden_input.size() * sizeof(float));
        ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
        ggml_backend_tensor_set(rope_cos, cache.rope_cos_values.data(), 0, cache.rope_cos_values.size() * sizeof(float));
        ggml_backend_tensor_set(rope_sin, cache.rope_sin_values.data(), 0, cache.rope_sin_values.size() * sizeof(float));
    }
    if (!use_scheduler && reference_kv_gallocr_disabled() && !graph_buffer) {
        throw std::runtime_error("backend KV cache window decode graph tensor allocation failed");
    }
    {
        ScopedProfileTimer timer("reference_kv_window_compute_graph");
        ReferenceKvStageProfileState stage_profile_state{std::chrono::steady_clock::now()};
        const bool use_stage_profile = use_scheduler && reference_kv_stage_profile_enabled();
        if (use_stage_profile) {
            ggml_backend_sched_set_eval_callback(cache.window_sched.get(), reference_kv_stage_profile_callback, &stage_profile_state);
        }
        const ggml_status status = use_scheduler
            ? ggml_backend_sched_graph_compute_async(cache.window_sched.get(), graph)
            : ggml_backend_graph_compute(cache.backend, graph);
        if (use_stage_profile) {
            ggml_backend_sched_set_eval_callback(cache.window_sched.get(), nullptr, nullptr);
        }
        if (status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("backend KV cache window decode graph compute failed");
        }
        if (use_scheduler) {
            ggml_backend_sched_synchronize(cache.window_sched.get());
        }
    }
    if (logits_flat != nullptr) {
        const size_t fused = static_cast<size_t>(k_num_codebooks * k_codebook_vocab_size);
        std::vector<float> logits(fused);
        {
            ScopedProfileTimer timer("reference_kv_window_get_logits");
            ggml_backend_tensor_get(logits_flat, logits.data(), static_cast<size_t>(tokens - 1) * fused * sizeof(float), logits.size() * sizeof(float));
        }
        *logits_vc_out = audio_logits_flat_to_vc(logits, 1, 0);
    }
    const bool need_full_hidden = !return_last_hidden_only || !node_dump_dir().empty();
    if (skip_hidden_readback && !node_dump_dir().empty()) {
        throw std::runtime_error("skip hidden readback is incompatible with node dumps");
    }
    if (skip_hidden_readback) {
        return {};
    }
    std::vector<float> values(static_cast<size_t>(cfg.embedding_length * (need_full_hidden ? tokens : 1)));
    const size_t hidden_offset = static_cast<size_t>(need_full_hidden ? 0 : tokens - 1) * static_cast<size_t>(cfg.embedding_length) * sizeof(float);
    {
        ScopedProfileTimer timer("reference_kv_window_get_hidden");
        ggml_backend_tensor_get(cur_hidden, values.data(), hidden_offset, values.size() * sizeof(float));
    }
    if (!node_dump_dir().empty()) {
        const size_t last_offset = static_cast<size_t>(tokens - 1) * static_cast<size_t>(cfg.embedding_length) * sizeof(float);
        const std::string dump_suffix = "_start" + std::to_string(start) + "_tokens" + std::to_string(tokens);
        auto dump_last_token = [&](const std::string & name, ggml_tensor * tensor, int64_t width) {
            if (tensor == nullptr) {
                return;
            }
            std::vector<float> tensor_values(static_cast<size_t>(width));
            const size_t offset = static_cast<size_t>(tokens - 1) * static_cast<size_t>(width) * sizeof(float);
            ggml_backend_tensor_get(tensor, tensor_values.data(), offset, tensor_values.size() * sizeof(float));
            dump_node_values(name + dump_suffix, tensor_values);
        };
        dump_last_token("cpp_kv_window_blk0_attn_normed", layer0_attn_normed, cfg.embedding_length);
        dump_last_token("cpp_kv_window_blk0_q_proj", layer0_q_proj, head_dim * n_head);
        dump_last_token("cpp_kv_window_blk0_q_normed", layer0_q_normed, head_dim * n_head);
        dump_last_token("cpp_kv_window_blk0_k_cur", layer0_k_cur, head_dim * n_head_kv);
        dump_last_token("cpp_kv_window_blk0_v_cur", layer0_v_cur, head_dim * n_head_kv);
        dump_last_token("cpp_kv_window_blk0_kq", layer0_kq, total * n_head);
        dump_last_token("cpp_kv_window_blk0_softmax", layer0_softmax, total * n_head);
        dump_last_token("cpp_kv_window_blk0_attn_out", layer0_attn_out, cfg.embedding_length);
        dump_last_token("cpp_kv_window_blk0_post_attn", layer0_post_attn, cfg.embedding_length);
        dump_last_token("cpp_kv_window_blk0_mlp_normed", layer0_mlp_normed, cfg.embedding_length);
        dump_last_token("cpp_kv_window_blk0_mlp_gate", layer0_mlp_gate, 9728);
        dump_last_token("cpp_kv_window_blk0_mlp_up", layer0_mlp_up, 9728);
        dump_last_token("cpp_kv_window_blk0_mlp_gated", layer0_mlp_gated, 9728);
        dump_last_token("cpp_kv_window_blk0_mlp_out", layer0_mlp_out, cfg.embedding_length);
        for (int layer = 0; layer < cfg.block_count; ++layer) {
            ggml_tensor * layer_tensor = layer_hidden_outputs[static_cast<size_t>(layer)];
            if (layer_tensor == nullptr) {
                continue;
            }
            std::vector<float> layer_values(static_cast<size_t>(cfg.embedding_length));
            ggml_backend_tensor_get(layer_tensor, layer_values.data(), last_offset, layer_values.size() * sizeof(float));
            dump_node_values("cpp_kv_window_blk" + std::to_string(layer) + "_hidden" + dump_suffix, layer_values);
        }
    }
    return values;
}

std::vector<float> run_kv_decode_window_all_layers_with_backend_cache(const std::string & model_path, const std::vector<float> & hidden_input, int64_t tokens, BackendKvCache & cache) {
    const TextConfig & cfg = cache.cfg;
    std::vector<float> values = run_kv_decode_window_all_layers_full_with_backend_cache(model_path, hidden_input, tokens, cache);
    std::vector<float> last(static_cast<size_t>(cfg.embedding_length));
    std::copy(values.end() - static_cast<std::ptrdiff_t>(cfg.embedding_length), values.end(), last.begin());
    return last;
}

std::vector<float> run_prefill_all_layers_into_backend_cache(const std::string & model_path, const std::vector<float> & hidden_values, BackendKvCache & cache) {
    ScopedProfileTimer timer("prefill_kv_all_layers");
    const TextConfig & cfg = cache.cfg;
    if (cache.used != 0) {
        throw std::runtime_error("backend KV prefill expects an empty cache");
    }
    if (cfg.head_count <= 0 || cfg.head_count_kv <= 0 || cfg.head_count % cfg.head_count_kv != 0) {
        throw std::runtime_error("backend KV prefill head config mismatch");
    }
    if (cfg.key_length != cfg.value_length || cfg.key_length != cfg.rope_dimension_count) {
        throw std::runtime_error("backend KV prefill head dimension mismatch");
    }
    if (hidden_values.size() % static_cast<size_t>(cfg.embedding_length) != 0) {
        throw std::runtime_error("backend KV prefill hidden input shape mismatch");
    }
    const int64_t tokens = static_cast<int64_t>(hidden_values.size() / static_cast<size_t>(cfg.embedding_length));
    if (tokens <= 0 || tokens >= cache.capacity) {
        throw std::runtime_error("backend KV prefill token/capacity mismatch");
    }
    const int64_t head_dim = cfg.key_length;
    const int64_t n_head = cfg.head_count;
    const int64_t n_head_kv = cfg.head_count_kv;
    const int64_t n_rep = n_head / n_head_kv;

    const std::vector<std::string> names = all_block_weight_names(cfg);
    if (!ensure_backend_cache_weights(model_path, cache, names)) {
        throw std::runtime_error("failed to load backend KV prefill tensors from GGUF");
    }
    ggml_context * weights_ctx = cache.weights_ggml;

    ggml_init_params params{};
    params.mem_size = 4ull * 1024ull * 1024ull * 1024ull;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate backend KV prefill graph context");
    }

    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, tokens);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, tokens);
    ggml_tensor * rope_cos = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim / 2, cfg.context_length);
    ggml_tensor * rope_sin = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, head_dim / 2, cfg.context_length);
    ggml_tensor * cur_hidden = hidden;
    ggml_tensor * layer0_k_cur = nullptr;
    ggml_tensor * layer0_v_cur = nullptr;
    ggml_tensor * layer0_k = nullptr;
    ggml_tensor * layer0_v = nullptr;
    std::vector<ggml_tensor *> writes;
    writes.reserve(static_cast<size_t>(cfg.block_count * 2));
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";

        ggml_tensor * normed = ggml_rms_norm(ctx.get(), cur_hidden, cfg.rms_norm_eps);
        ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
        normed = ggml_mul(ctx.get(), normed, attn_norm);
        ggml_tensor * q = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q.weight").c_str()), normed);
        ggml_tensor * k_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
        ggml_tensor * v_cur = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
        q = ggml_reshape_3d(ctx.get(), q, head_dim, n_head, tokens);
        k_cur = ggml_reshape_3d(ctx.get(), k_cur, head_dim, n_head_kv, tokens);
        v_cur = ggml_reshape_3d(ctx.get(), v_cur, head_dim, n_head_kv, tokens);

        ggml_tensor * q_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_q_norm.weight").c_str()), head_dim, 1, 1);
        ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), head_dim, 1, 1);
        q = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), q, cfg.rms_norm_eps), q_norm);
        k_cur = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k_cur, cfg.rms_norm_eps), k_norm);
        q = neox_rope_bf16_cache(ctx.get(), q, positions, rope_cos, rope_sin);
        k_cur = neox_rope_bf16_cache(ctx.get(), k_cur, positions, rope_cos, rope_sin);

        k_cur = ggml_cont_3d(ctx.get(), k_cur, head_dim, n_head_kv, tokens);
        v_cur = ggml_cont_3d(ctx.get(), v_cur, head_dim, n_head_kv, tokens);
        writes.push_back(ggml_set_inplace(ctx.get(), cache.key, k_cur, cache.key->nb[1], cache.key->nb[2], cache.key->nb[3], backend_kv_cache_offset_bytes(cache, cache.key, layer, 0, 0, 0)));
        writes.push_back(ggml_set_inplace(ctx.get(), cache.value, v_cur, cache.value->nb[1], cache.value->nb[2], cache.value->nb[3], backend_kv_cache_offset_bytes(cache, cache.value, layer, 0, 0, 0)));

        ggml_tensor * k = repeat_kv_3d(ctx.get(), k_cur, n_rep);
        ggml_tensor * v = repeat_kv_3d(ctx.get(), v_cur, n_rep);
        if (layer == 0 && !node_dump_dir().empty()) {
            layer0_k_cur = k_cur;
            layer0_v_cur = v_cur;
            layer0_k = k;
            layer0_v = v;
            ggml_set_output(layer0_k_cur);
            ggml_set_output(layer0_v_cur);
            ggml_set_output(layer0_k);
            ggml_set_output(layer0_v);
        }

        ggml_tensor * q4 = ggml_permute(ctx.get(), as_4d(ctx.get(), q), 0, 2, 1, 3);
        ggml_tensor * k4 = ggml_permute(ctx.get(), as_4d(ctx.get(), k), 0, 2, 1, 3);
        ggml_tensor * kq = mul_mat_f32(ctx.get(), k4, q4);
        kq = ggml_scale(ctx.get(), kq, 1.0f / std::sqrt(static_cast<float>(head_dim)));
        kq = ggml_diag_mask_inf(ctx.get(), kq, 0);
        kq = ggml_soft_max(ctx.get(), kq);
        ggml_tensor * v_for_mm = ggml_cont(ctx.get(), ggml_permute(ctx.get(), as_4d(ctx.get(), v), 1, 2, 0, 3));
        ggml_tensor * kqv = mul_mat_f32(ctx.get(), v_for_mm, kq);
        ggml_tensor * attn = ggml_permute(ctx.get(), kqv, 0, 2, 1, 3);
        attn = ggml_cont_2d(ctx.get(), attn, head_dim * n_head, tokens);
        ggml_tensor * attn_out = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_output.weight").c_str()), attn);
        ggml_tensor * cur = ggml_add(ctx.get(), cur_hidden, attn_out);

        ggml_tensor * mlp_normed = ggml_rms_norm(ctx.get(), cur, cfg.rms_norm_eps);
        ggml_tensor * ffn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_norm.weight").c_str()), cfg.embedding_length, 1);
        mlp_normed = ggml_mul(ctx.get(), mlp_normed, ffn_norm);
        ggml_tensor * gate = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_gate.weight").c_str()), mlp_normed);
        ggml_tensor * up = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_up.weight").c_str()), mlp_normed);
        ggml_tensor * gated = ggml_mul(ctx.get(), ggml_silu(ctx.get(), gate), up);
        ggml_tensor * mlp = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "ffn_down.weight").c_str()), gated);
        cur_hidden = ggml_add(ctx.get(), cur, mlp);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), static_cast<size_t>(cfg.block_count * 128 + 1024), false);
    ggml_build_forward_expand(graph, cur_hidden);
    for (ggml_tensor * write : writes) {
        ggml_build_forward_expand(graph, write);
    }
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate backend KV prefill graph tensors");
    }
    std::vector<int32_t> position_values(static_cast<size_t>(tokens));
    std::iota(position_values.begin(), position_values.end(), 0);
    const std::vector<float> cos_values = make_bf16_rope_table(cfg, false);
    const std::vector<float> sin_values = make_bf16_rope_table(cfg, true);
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(positions, position_values.data(), 0, position_values.size() * sizeof(int32_t));
    ggml_backend_tensor_set(rope_cos, cos_values.data(), 0, cos_values.size() * sizeof(float));
    ggml_backend_tensor_set(rope_sin, sin_values.data(), 0, sin_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(cache.backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("backend KV prefill graph compute failed");
    }
    std::vector<float> values(static_cast<size_t>(cur_hidden->ne[0] * cur_hidden->ne[1]));
    ggml_backend_tensor_get(cur_hidden, values.data(), 0, values.size() * sizeof(float));
    if (!node_dump_dir().empty() && layer0_k_cur != nullptr && layer0_v_cur != nullptr && layer0_k != nullptr && layer0_v != nullptr) {
        std::vector<float> k_cur_values(static_cast<size_t>(ggml_nelements(layer0_k_cur)));
        std::vector<float> v_cur_values(static_cast<size_t>(ggml_nelements(layer0_v_cur)));
        std::vector<float> k_values(static_cast<size_t>(ggml_nelements(layer0_k)));
        std::vector<float> v_values(static_cast<size_t>(ggml_nelements(layer0_v)));
        ggml_backend_tensor_get(layer0_k_cur, k_cur_values.data(), 0, k_cur_values.size() * sizeof(float));
        ggml_backend_tensor_get(layer0_v_cur, v_cur_values.data(), 0, v_cur_values.size() * sizeof(float));
        ggml_backend_tensor_get(layer0_k, k_values.data(), 0, k_values.size() * sizeof(float));
        ggml_backend_tensor_get(layer0_v, v_values.data(), 0, v_values.size() * sizeof(float));
        dump_node_values("cpp_prefill_blk0_k_cur", k_cur_values);
        dump_node_values("cpp_prefill_blk0_v_cur", v_cur_values);
        dump_node_values("cpp_prefill_blk0_k_state", k_values);
        dump_node_values("cpp_prefill_blk0_v_state", v_values);
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("backend KV prefill output contains non-finite values");
        }
    }
    cache.used = tokens;
    return values;
}

void run_kv_decode_graph_layer_check(const std::string & model_path, int layer) {
    const TextConfig cfg = inspect_text_config(model_path);
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    (void) run_kv_decode_graph_layer(model_path, layer, hidden_values);
}

void self_check_kv_decode_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        run_kv_decode_graph_layer_check(model_path, layer);
    }
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_kv_decode_graph_layer(model_path, layer, hidden_values);
    }
    if (hidden_values.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("KV decode chain output shape mismatch");
    }
}

void self_check_kv_decode_audio_head_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    GenerateOptions options;
    options.model_path = model_path;
    options.temperature = 0.0f;
    options.stop_on_eoc = false;
    std::vector<float> hidden_values(static_cast<size_t>(cfg.embedding_length));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_kv_decode_graph_layer(model_path, layer, hidden_values);
    }
    const std::vector<int32_t> codes = run_audio_head_sample(model_path, hidden_values, options);
    if (codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("KV decode audio head sampled code shape mismatch");
    }
    for (int32_t code : codes) {
        if (code < 0 || code >= k_codebook_vocab_size) {
            throw std::runtime_error("KV decode audio head sampled code range mismatch");
        }
    }
}

void self_check_prompt_to_kv_decode_audio_head_graph(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    options.temperature = 0.0f;
    options.stop_on_eoc = false;
    std::vector<float> hidden_values = run_prompt_last_hidden(model_path, options.text, nullptr, options);
    if (hidden_values.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("prompt-to-KV decode hidden shape mismatch");
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden_values = run_kv_decode_graph_layer(model_path, layer, hidden_values);
    }
    const std::vector<int32_t> codes = run_audio_head_sample(model_path, hidden_values, options);
    if (codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("prompt-to-KV decode sampled code shape mismatch");
    }
    for (int32_t code : codes) {
        if (code < 0 || code >= k_codebook_vocab_size) {
            throw std::runtime_error("prompt-to-KV decode sampled code range mismatch");
        }
    }
}

void write_kv_cache_layer_from_hidden(const std::string & model_path, int layer, int64_t pos, const std::vector<float> & hidden_input, TextKvCache & cache) {
    const TextConfig cfg = inspect_text_config(model_path);
    if (hidden_input.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("KV cache write hidden input shape mismatch");
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    const std::vector<std::string> names{
        prefix + "attn_norm.weight",
        prefix + "attn_k.weight",
        prefix + "attn_v.weight",
        prefix + "attn_k_norm.weight",
    };
    if (!weights_buffer || !load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load KV cache write tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 128 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, attn_norm);
    ggml_tensor * k = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
    ggml_tensor * v = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
    k = ggml_reshape_3d(ctx.get(), k, cfg.key_length, cfg.head_count_kv, 1);
    v = ggml_reshape_3d(ctx.get(), v, cfg.key_length, cfg.head_count_kv, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), cfg.key_length, 1, 1);
    k = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k, cfg.rms_norm_eps), k_norm);
    k = ggml_rope_ext(ctx.get(), k, positions, nullptr, cfg.key_length, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, k);
    ggml_build_forward_expand(graph, v);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    const int32_t position = static_cast<int32_t>(pos);
    ggml_backend_tensor_set(hidden, hidden_input.data(), 0, hidden_input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, &position, 0, sizeof(position));
    if (!graph_buffer || ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("KV cache write graph compute failed");
    }
    std::vector<float> key(static_cast<size_t>(cfg.key_length * cfg.head_count_kv));
    std::vector<float> value(key.size());
    ggml_backend_tensor_get(k, key.data(), 0, key.size() * sizeof(float));
    ggml_backend_tensor_get(v, value.data(), 0, value.size() * sizeof(float));
    for (int64_t head = 0; head < cfg.head_count_kv; ++head) {
        for (int64_t dim = 0; dim < cfg.key_length; ++dim) {
            const size_t src = static_cast<size_t>(head * cfg.key_length + dim);
            const size_t dst = kv_cache_offset(cache, layer, pos, head, dim);
            cache.key[dst] = key[src];
            cache.value[dst] = value[src];
        }
    }
    cache.used = std::max(cache.used, pos + 1);
}

void write_backend_kv_cache_layer_from_hidden(const std::string & model_path, int layer, int64_t pos, const std::vector<float> & hidden_input, BackendKvCache & cache) {
    const TextConfig & cfg = cache.cfg;
    if (hidden_input.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("backend KV cache prefill hidden input shape mismatch");
    }
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    ggml_context * weights_ctx = cache.weights_ggml;
    const std::vector<std::string> names{
        prefix + "attn_norm.weight",
        prefix + "attn_k.weight",
        prefix + "attn_v.weight",
        prefix + "attn_k_norm.weight",
    };
    if (!ensure_backend_cache_weights(model_path, cache, names)) {
        throw std::runtime_error("failed to load backend KV cache prefill tensors from GGUF");
    }

    ggml_init_params params{};
    params.mem_size = 128 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, cfg.embedding_length, 1);
    ggml_tensor * positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
    ggml_tensor * normed = ggml_rms_norm(ctx.get(), hidden, cfg.rms_norm_eps);
    ggml_tensor * attn_norm = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_norm.weight").c_str()), cfg.embedding_length, 1);
    normed = ggml_mul(ctx.get(), normed, attn_norm);
    ggml_tensor * k = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k.weight").c_str()), normed);
    ggml_tensor * v = mul_mat_f32(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_v.weight").c_str()), normed);
    k = ggml_reshape_3d(ctx.get(), k, cfg.key_length, cfg.head_count_kv, 1);
    v = ggml_reshape_3d(ctx.get(), v, cfg.key_length, cfg.head_count_kv, 1);
    ggml_tensor * k_norm = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "attn_k_norm.weight").c_str()), cfg.key_length, 1, 1);
    k = ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), k, cfg.rms_norm_eps), k_norm);
    k = ggml_rope_ext(ctx.get(), k, positions, nullptr, cfg.key_length, 2, cfg.context_length, cfg.rope_freq_base, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    ggml_tensor * key_written = ggml_set_inplace(ctx.get(), cache.key, k, cache.key->nb[1], cache.key->nb[2], cache.key->nb[3], backend_kv_cache_offset_bytes(cache, cache.key, layer, pos, 0, 0));
    ggml_tensor * value_written = ggml_set_inplace(ctx.get(), cache.value, v, cache.value->nb[1], cache.value->nb[2], cache.value->nb[3], backend_kv_cache_offset_bytes(cache, cache.value, layer, pos, 0, 0));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, key_written);
    ggml_build_forward_expand(graph, value_written);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
    const int32_t position = static_cast<int32_t>(pos);
    ggml_backend_tensor_set(hidden, hidden_input.data(), 0, hidden_input.size() * sizeof(float));
    ggml_backend_tensor_set(positions, &position, 0, sizeof(position));
    if (!graph_buffer || ggml_backend_graph_compute(cache.backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("backend KV cache prefill graph compute failed");
    }
    cache.used = std::max(cache.used, pos + 1);
}

void self_check_prompt_kv_cache_write(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    TextKvCache cache = make_text_kv_cache(cfg, 4);
    const std::vector<float> hidden = run_prompt_last_hidden(model_path, options.text, nullptr, options);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        write_kv_cache_layer_from_hidden(model_path, layer, 0, hidden, cache);
    }
    if (cache.used != 1) {
        throw std::runtime_error("KV cache write used count mismatch");
    }
    float sum = 0.0f;
    for (float value : cache.key) {
        sum += std::abs(value);
    }
    for (float value : cache.value) {
        sum += std::abs(value);
    }
    if (!(sum > 0.0f) || !std::isfinite(sum)) {
        throw std::runtime_error("KV cache write produced invalid values");
    }
}

ggml_tensor * codec_snake_1d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * alpha, ggml_tensor * one, ggml_tensor * eps);
ggml_tensor * codec_conv_1d_bias(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & weight, const std::string & bias, int padding, int dilation, int stride = 1);
ggml_tensor * codec_conv_1d_bias_f32_im2col(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & weight, const std::string & bias, int padding, int dilation, int stride = 1);
ggml_tensor * codec_residual_unit_1d(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & prefix, ggml_tensor * one, ggml_tensor * eps, int dilation);
ggml_tensor * codec_encoder_residual_unit_1d(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & prefix, ggml_tensor * one, ggml_tensor * eps, int dilation, std::vector<std::pair<std::string, ggml_tensor *>> * stage_outputs = nullptr);

void self_check_text_logits_graph(const std::string & model_path) {
    GenerateOptions options;
    options.model_path = model_path;
    const std::vector<int32_t> codes = run_synthetic_text_codes(model_path, options);
    if (codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("text logits sampled code shape mismatch");
    }
}

void self_check_prompt_kv_cache_decode(const std::string & model_path) {
    const TextConfig cfg = inspect_text_config(model_path);
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    options.temperature = 0.0f;
    options.stop_on_eoc = false;
    TextKvCache cache = make_text_kv_cache(cfg, 4);
    std::vector<float> hidden = run_prompt_last_hidden(model_path, options.text, nullptr, options);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        write_kv_cache_layer_from_hidden(model_path, layer, 0, hidden, cache);
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden = run_kv_decode_graph_layer_with_cache(model_path, layer, hidden, cache);
    }
    cache.used = std::max<int64_t>(cache.used, 2);
    if (cache.used != 2 || hidden.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("prompt KV cache decode shape/used mismatch");
    }
    const std::vector<int32_t> codes = run_audio_head_sample(model_path, hidden, options);
    if (codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("prompt KV cache decode sampled code shape mismatch");
    }
    for (int32_t code : codes) {
        if (code < 0 || code >= k_codebook_vocab_size) {
            throw std::runtime_error("prompt KV cache decode sampled code range mismatch");
        }
    }
}

void self_check_backend_kv_cache_write(const std::string & model_path) {
    PipelineRuntime pipeline_runtime(model_path);
    BackendRuntime & runtime = pipeline_runtime.get(g_backend_kind);
    const TextConfig & cfg = *runtime.cfg;
    BackendKvCache & cache = get_backend_kv_cache(runtime, 4);

    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate backend KV write context");
    }
    ggml_tensor * k_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cache.head_dim, cache.kv_heads, 1);
    ggml_tensor * v_cur = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, cache.head_dim, cache.kv_heads, 1);
    const size_t pos1 = backend_kv_cache_offset_bytes(cache, cache.key, 0, 1, 0, 0);
    ggml_tensor * key_written = ggml_set_inplace(ctx.get(), cache.key, k_cur, cache.key->nb[1], cache.key->nb[2], cache.key->nb[3], pos1);
    ggml_tensor * value_written = ggml_set_inplace(ctx.get(), cache.value, v_cur, cache.value->nb[1], cache.value->nb[2], cache.value->nb[3], pos1);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, key_written);
    ggml_build_forward_expand(graph, value_written);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cache.backend));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate backend KV write graph tensors");
    }
    std::vector<float> key(static_cast<size_t>(cache.head_dim * cache.kv_heads));
    std::vector<float> value(key.size());
    for (size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<float>(i % 17) + 0.25f;
        value[i] = -static_cast<float>(i % 19) - 0.5f;
    }
    ggml_backend_tensor_set(k_cur, key.data(), 0, key.size() * sizeof(float));
    ggml_backend_tensor_set(v_cur, value.data(), 0, value.size() * sizeof(float));
    if (ggml_backend_graph_compute(cache.backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("backend KV write graph compute failed");
    }
    std::vector<float> got_key(key.size());
    std::vector<float> got_value(value.size());
    ggml_backend_tensor_get(cache.key, got_key.data(), pos1, got_key.size() * sizeof(float));
    ggml_backend_tensor_get(cache.value, got_value.data(), pos1, got_value.size() * sizeof(float));
    for (size_t i = 0; i < key.size(); ++i) {
        if (got_key[i] != key[i] || got_value[i] != value[i]) {
            throw std::runtime_error("backend KV write value mismatch");
        }
    }
    cache.used = 2;
}

void self_check_backend_kv_cache_decode(const std::string & model_path) {
    PipelineRuntime pipeline_runtime(model_path);
    BackendRuntime & runtime = pipeline_runtime.get(g_backend_kind);
    const TextConfig & cfg = *runtime.cfg;
    BackendKvCache & cache = get_backend_kv_cache(runtime, 4);
    std::vector<float> seed(static_cast<size_t>(cache.head_dim * cache.kv_heads));
    for (size_t i = 0; i < seed.size(); ++i) {
        seed[i] = static_cast<float>(static_cast<int>(i % 23) - 11) / 23.0f;
    }
    for (int64_t layer = 0; layer < cache.layers; ++layer) {
        const size_t offset = backend_kv_cache_offset_bytes(cache, cache.key, layer, 0, 0, 0);
        ggml_backend_tensor_set(cache.key, seed.data(), offset, seed.size() * sizeof(float));
        ggml_backend_tensor_set(cache.value, seed.data(), offset, seed.size() * sizeof(float));
    }
    cache.used = 1;

    std::vector<float> hidden(static_cast<size_t>(cfg.embedding_length));
    for (size_t i = 0; i < hidden.size(); ++i) {
        hidden[i] = static_cast<float>(static_cast<int>(i % 127) - 63) / 63.0f;
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden = run_kv_decode_graph_layer_with_backend_cache(model_path, layer, hidden, cache);
    }
    cache.used = 2;
    const size_t written_offset = backend_kv_cache_offset_bytes(cache, cache.key, 0, 1, 0, 0);
    std::vector<float> written(seed.size());
    ggml_backend_tensor_get(cache.key, written.data(), written_offset, written.size() * sizeof(float));
    float sum = 0.0f;
    for (float value : written) {
        sum += std::abs(value);
    }
    if (!(sum > 0.0f) || !std::isfinite(sum) || hidden.size() != static_cast<size_t>(cfg.embedding_length)) {
        throw std::runtime_error("backend KV cache decode produced invalid output");
    }
}

void self_check_backend_prompt_kv_cache_prefill(const std::string & model_path) {
    PipelineRuntime pipeline_runtime(model_path);
    BackendRuntime & runtime = pipeline_runtime.get(g_backend_kind);
    const TextConfig & cfg = *runtime.cfg;
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    options.temperature = 0.0f;
    options.stop_on_eoc = false;
    BackendKvCache & cache = get_backend_kv_cache(runtime, 4);
    std::vector<float> hidden = run_prompt_last_hidden(model_path, options.text, nullptr, options);
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        write_backend_kv_cache_layer_from_hidden(model_path, layer, 0, hidden, cache);
    }
    if (cache.used != 1) {
        throw std::runtime_error("backend prompt KV cache prefill used mismatch");
    }
    for (int layer = 0; layer < cfg.block_count; ++layer) {
        hidden = run_kv_decode_graph_layer_with_backend_cache(model_path, layer, hidden, cache);
    }
    cache.used = 2;
    const std::vector<int32_t> codes = run_audio_head_sample(model_path, hidden, options);
    if (codes.size() != static_cast<size_t>(k_num_codebooks)) {
        throw std::runtime_error("backend prompt KV cache prefill sampled code shape mismatch");
    }
    for (int32_t code : codes) {
        if (code < 0 || code >= k_codebook_vocab_size) {
            throw std::runtime_error("backend prompt KV cache prefill sampled code range mismatch");
        }
    }
}

void self_check_prompt_ar_graph(const std::string & model_path) {
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    options.steps = 2;
    options.temperature = 0.0f;
    const CodeMatrix codes = run_prompt_text_code_frames(model_path, options, options.steps);
    const int expected_frames = std::max(options.steps - k_num_codebooks, 0);
    if (codes.codebooks != k_num_codebooks || codes.frames != expected_frames) {
        throw std::runtime_error("prompt AR graph output shape mismatch");
    }
}

void self_check_prompt_ar_backend_kv(const std::string & model_path) {
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    options.steps = 2;
    options.temperature = 0.0f;
    options.stop_on_eoc = false;
    PipelineRuntime pipeline_runtime(model_path);
    BackendRuntime & runtime = pipeline_runtime.get(g_backend_kind);
    const CodeMatrix codes = run_prompt_text_code_frames_backend_kv(model_path, options, options.steps, runtime);
    const int expected_frames = std::max(options.steps - k_num_codebooks, 0);
    if (codes.codebooks != k_num_codebooks || codes.frames != expected_frames || codes.data.size() != static_cast<size_t>(expected_frames * k_num_codebooks)) {
        throw std::runtime_error("prompt AR backend KV output shape mismatch");
    }
    for (int32_t code : codes.data) {
        if (code < 0 || code >= k_codebook_vocab_size) {
            throw std::runtime_error("prompt AR backend KV code range mismatch");
        }
    }
}

void self_check_reference_ar_backend_kv(const std::string & model_path) {
    GenerateOptions options;
    options.model_path = model_path;
    options.text = "你好";
    options.ref_text = "参考音频";
    options.steps = 2;
    options.temperature = 0.0f;
    options.stop_on_eoc = false;
    CodeMatrix ref_codes;
    ref_codes.codebooks = k_num_codebooks;
    ref_codes.frames = 2;
    ref_codes.data.resize(static_cast<size_t>(ref_codes.frames * ref_codes.codebooks));
    for (int f = 0; f < ref_codes.frames; ++f) {
        for (int c = 0; c < ref_codes.codebooks; ++c) {
            ref_codes.at(c, f) = (f * 17 + c * 31) % k_codebook_vocab_size;
        }
    }
    PipelineRuntime pipeline_runtime(model_path);
    BackendRuntime & runtime = pipeline_runtime.get(g_backend_kind);
    const CodeMatrix codes = run_reference_text_code_frames_backend_kv(model_path, options, ref_codes, options.steps, runtime);
    const int expected_frames = std::max(options.steps - k_num_codebooks, 0);
    if (codes.codebooks != k_num_codebooks || codes.frames != expected_frames || codes.data.size() != static_cast<size_t>(expected_frames * k_num_codebooks)) {
        throw std::runtime_error("reference AR backend KV output shape mismatch");
    }
    for (int32_t code : codes.data) {
        if (code < 0 || code >= k_codebook_vocab_size) {
            throw std::runtime_error("reference AR backend KV code range mismatch");
        }
    }
}

void self_check_codec_encode_conv1_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, {"a.ae.conv1.weight", "a.ae.conv1.bias"})) {
        throw std::runtime_error("failed to load codec encode conv1 tensors from GGUF");
    }

    const int samples = 2400;
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec encode conv1 graph context");
    }
    ggml_tensor * waveform = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, samples, 1, 1);
    ggml_tensor * out = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ae.conv1.weight"), waveform, 1, 3, 1);
    ggml_tensor * bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ae.conv1.bias"), 1, 64, 1);
    out = ggml_add(ctx.get(), out, bias);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec encode conv1 graph tensors");
    }
    std::vector<float> waveform_values(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const float t = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(samples - 1);
        waveform_values[static_cast<size_t>(i)] = 0.2f * std::sin(17.0f * t) + 0.05f * t;
    }
    ggml_backend_tensor_set(waveform, waveform_values.data(), 0, waveform_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec encode conv1 graph compute failed");
    }
    if (out->ne[0] != samples || out->ne[1] != 64 || out->ne[2] != 1) {
        throw std::runtime_error("codec encode conv1 output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec encode conv1 output contains non-finite values");
        }
    }
}

void codec_add_encoder_block_names(std::vector<std::string> & names, int block) {
    const std::string prefix = "a.ae.block." + std::to_string(block);
    names.push_back(prefix + ".snake1.alpha");
    names.push_back(prefix + ".conv1.weight");
    names.push_back(prefix + ".conv1.bias");
    for (int unit = 1; unit <= 3; ++unit) {
        const std::string unit_prefix = prefix + ".res_unit" + std::to_string(unit);
        names.push_back(unit_prefix + ".snake1.alpha");
        names.push_back(unit_prefix + ".conv1.weight");
        names.push_back(unit_prefix + ".conv1.bias");
        names.push_back(unit_prefix + ".snake2.alpha");
        names.push_back(unit_prefix + ".conv2.weight");
        names.push_back(unit_prefix + ".conv2.bias");
    }
}

ggml_tensor * codec_encoder_block(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, int block, int stride, ggml_tensor * one, ggml_tensor * eps, std::vector<std::pair<std::string, ggml_tensor *>> * stage_outputs = nullptr) {
    const std::string prefix = "a.ae.block." + std::to_string(block);
    ggml_tensor * out = codec_encoder_residual_unit_1d(ctx, weights_ctx, inp, prefix + ".res_unit1", one, eps, 1, stage_outputs);
    if (stage_outputs != nullptr && block == 0) stage_outputs->push_back({"cpp_reference_acoustic_block0_res1", out});
    out = codec_encoder_residual_unit_1d(ctx, weights_ctx, out, prefix + ".res_unit2", one, eps, 3);
    if (stage_outputs != nullptr && block == 0) stage_outputs->push_back({"cpp_reference_acoustic_block0_res2", out});
    out = codec_encoder_residual_unit_1d(ctx, weights_ctx, out, prefix + ".res_unit3", one, eps, 9);
    if (stage_outputs != nullptr && block == 0) stage_outputs->push_back({"cpp_reference_acoustic_block0_res3", out});
    out = codec_snake_1d(ctx, out, ggml_get_tensor(weights_ctx, (prefix + ".snake1.alpha").c_str()), one, eps);
    if (stage_outputs != nullptr && block == 0) stage_outputs->push_back({"cpp_reference_acoustic_block0_snake", out});
    return codec_conv_1d_bias_f32_im2col(ctx, weights_ctx, out, prefix + ".conv1.weight", prefix + ".conv1.bias", (stride + 1) / 2, 1, stride);
}

std::vector<float> run_codec_encode_acoustic_graph(const std::string & model_path, const std::vector<float> & waveform_values, int blocks, bool final_project, int expected_channels, const char * label, int64_t * out_frames = nullptr) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{"a.ae.conv1.weight", "a.ae.conv1.bias"};
    for (int block = 0; block < blocks; ++block) {
        codec_add_encoder_block_names(names, block);
    }
    if (final_project) {
        names.push_back("a.ae.snake1.alpha");
        names.push_back("a.ae.conv2.weight");
        names.push_back("a.ae.conv2.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error(std::string("failed to load codec encode ") + label + " tensors from GGUF");
    }

    const int samples = static_cast<int>(waveform_values.size());
    if (samples <= 0) {
        throw std::runtime_error("codec encode acoustic input is empty");
    }
    ggml_init_params params{};
    params.mem_size = 1024ull * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error(std::string("failed to allocate codec encode ") + label + " graph context");
    }
    ggml_tensor * one = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * waveform = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, samples, 1, 1);
    ggml_tensor * out = codec_conv_1d_bias_f32_im2col(ctx.get(), weights_ctx, waveform, "a.ae.conv1.weight", "a.ae.conv1.bias", 3, 1);
    std::vector<std::pair<std::string, ggml_tensor *>> stage_outputs;
    if (!node_dump_dir().empty()) {
        stage_outputs.push_back({"cpp_reference_acoustic_conv1", out});
    }
    constexpr int strides[] = {8, 5, 4, 2, 3};
    for (int block = 0; block < blocks; ++block) {
        out = codec_encoder_block(ctx.get(), weights_ctx, out, block, strides[block], one, eps, node_dump_dir().empty() ? nullptr : &stage_outputs);
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_reference_acoustic_block" + std::to_string(block), out});
        }
    }
    if (final_project) {
        out = codec_snake_1d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.ae.snake1.alpha"), one, eps);
        out = codec_conv_1d_bias_f32_im2col(ctx.get(), weights_ctx, out, "a.ae.conv2.weight", "a.ae.conv2.bias", 1, 1);
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_reference_acoustic_project", out});
        }
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    for (const auto & stage : stage_outputs) {
        ggml_build_forward_expand(graph, stage.second);
    }
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error(std::string("failed to allocate codec encode ") + label + " graph tensors");
    }
    const float one_value = 1.0f;
    const float eps_value = 1e-9f;
    ggml_backend_tensor_set(one, &one_value, 0, sizeof(one_value));
    ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(eps_value));
    ggml_backend_tensor_set(waveform, waveform_values.data(), 0, waveform_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("codec encode ") + label + " graph compute failed");
    }
    for (const auto & stage : stage_outputs) {
        std::vector<float> stage_values(static_cast<size_t>(ggml_nelements(stage.second)));
        ggml_backend_tensor_get(stage.second, stage_values.data(), 0, stage_values.size() * sizeof(float));
        dump_node_values(stage.first, stage_values);
    }
    if (out->ne[0] <= 0 || out->ne[1] != expected_channels || out->ne[2] != 1) {
        throw std::runtime_error(std::string("codec encode ") + label + " output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("codec encode ") + label + " output contains non-finite values");
        }
    }
    if (out_frames != nullptr) {
        *out_frames = out->ne[0];
    }
    return values;
}

std::vector<float> synthetic_acoustic_waveform(int samples) {
    std::vector<float> waveform_values(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const float t = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(samples - 1);
        waveform_values[static_cast<size_t>(i)] = 0.2f * std::sin(17.0f * t) + 0.05f * t;
    }
    return waveform_values;
}

void self_check_codec_encode_blocks_graph(const std::string & model_path, int blocks, bool final_project, int expected_channels, const char * label) {
    (void) run_codec_encode_acoustic_graph(model_path, synthetic_acoustic_waveform(2400), blocks, final_project, expected_channels, label);
}

void self_check_codec_encode_block0_graph(const std::string & model_path) {
    self_check_codec_encode_blocks_graph(model_path, 1, false, 128, "block0");
}

void self_check_codec_encode_block1_graph(const std::string & model_path) {
    self_check_codec_encode_blocks_graph(model_path, 2, false, 256, "block1");
}

void self_check_codec_encode_block2_graph(const std::string & model_path) {
    self_check_codec_encode_blocks_graph(model_path, 3, false, 512, "block2");
}

void self_check_codec_encode_block3_graph(const std::string & model_path) {
    self_check_codec_encode_blocks_graph(model_path, 4, false, 1024, "block3");
}

void self_check_codec_encode_block4_graph(const std::string & model_path) {
    self_check_codec_encode_blocks_graph(model_path, 5, false, 2048, "block4");
}

void self_check_codec_encode_project_graph(const std::string & model_path) {
    self_check_codec_encode_blocks_graph(model_path, 5, true, 256, "project");
}

TensorBlock run_reference_acoustic_project(const std::string & model_path, const std::string & wav_path) {
    ReferenceAudioData ref;
    {
        ScopedProfileTimer timer("reference_acoustic_load_wav_resample");
        ref = load_reference_wav(wav_path);
    }
    std::vector<float> waveform = ref.audio_24k;
    if (waveform.size() < 24000) {
        waveform.resize(24000, 0.0f);
    }
    constexpr int pad = 480;
    std::vector<float> padded(static_cast<size_t>(pad) + waveform.size() + static_cast<size_t>(pad), 0.0f);
    std::copy(waveform.begin(), waveform.end(), padded.begin() + pad);
    dump_node_values("cpp_reference_acoustic_waveform", padded);
    int64_t frames = 0;
    TensorBlock block;
    {
        ScopedProfileTimer timer("reference_acoustic_graph");
        block.values = run_codec_encode_acoustic_graph(model_path, padded, 5, true, 256, "reference acoustic", &frames);
    }
    block.frames = frames;
    block.channels = 256;
    return block;
}

int64_t self_check_reference_acoustic(const std::string & model_path, const std::string & wav_path) {
    TensorBlock block = run_reference_acoustic_project(model_path, wav_path);
    const int64_t frames = block.frames;
    if (frames <= 0) {
        throw std::runtime_error("reference acoustic output is empty");
    }
    return frames;
}

void codec_add_semantic_block_names(std::vector<std::string> & names, int block) {
    const std::string prefix = "a.se.conv_blocks." + std::to_string(block);
    names.push_back(prefix + ".conv.weight");
    names.push_back(prefix + ".conv.bias");
    for (int unit = 0; unit < 2; ++unit) {
        const std::string unit_prefix = prefix + ".res_units." + std::to_string(unit);
        names.push_back(unit_prefix + ".conv1.weight");
        names.push_back(unit_prefix + ".conv2.weight");
    }
}

ggml_tensor * codec_semantic_residual_unit(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & prefix) {
    ggml_tensor * out = ggml_elu(ctx, inp);
    out = ggml_conv_1d(ctx, ggml_get_tensor(weights_ctx, (prefix + ".conv1.weight").c_str()), out, 1, 1, 1);
    out = ggml_elu(ctx, out);
    out = ggml_conv_1d(ctx, ggml_get_tensor(weights_ctx, (prefix + ".conv2.weight").c_str()), out, 1, 0, 1);
    return ggml_add(ctx, inp, out);
}

ggml_tensor * codec_semantic_block(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, int block) {
    const std::string prefix = "a.se.conv_blocks." + std::to_string(block);
    ggml_tensor * out = codec_semantic_residual_unit(ctx, weights_ctx, inp, prefix + ".res_units.0");
    out = codec_semantic_residual_unit(ctx, weights_ctx, out, prefix + ".res_units.1");
    return codec_conv_1d_bias(ctx, weights_ctx, out, prefix + ".conv.weight", prefix + ".conv.bias", 1, 1);
}

ggml_tensor * codec_layer_norm_2d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * out = ggml_norm(ctx, inp, 1e-5f);
    ggml_tensor * w = ggml_reshape_2d(ctx, weight, weight->ne[0], 1);
    ggml_tensor * b = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, ggml_mul(ctx, out, ggml_repeat(ctx, w, out)), ggml_repeat(ctx, b, out));
}

ggml_tensor * codec_linear_2d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * weight, ggml_tensor * bias) {
    ggml_tensor * out = ggml_mul_mat(ctx, weight, inp);
    ggml_tensor * b = ggml_reshape_2d(ctx, bias, bias->ne[0], 1);
    return ggml_add(ctx, out, b);
}

std::vector<float> tensor_to_f32(ggml_tensor * tensor) {
    const int64_t n = ggml_nelements(tensor);
    std::vector<float> values(static_cast<size_t>(n));
    if (tensor->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
        return values;
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(static_cast<size_t>(n));
        ggml_backend_tensor_get(tensor, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        ggml_fp16_to_fp32_row(half.data(), values.data(), n);
        return values;
    }
    throw std::runtime_error("unsupported tensor dtype for f32 readback");
}

std::vector<float> codec_codebook_norm(ggml_context * weights_ctx, const std::string & prefix) {
    const std::vector<float> embed = tensor_to_f32(ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()));
    std::vector<float> norm(1024);
    for (int code = 0; code < 1024; ++code) {
        float sum = 0.0f;
        for (int dim = 0; dim < 64; ++dim) {
            const float v = embed[static_cast<size_t>(dim + 64 * code)];
            sum += v * v;
        }
        norm[static_cast<size_t>(code)] = sum;
    }
    return norm;
}

ggml_tensor * codec_quantizer_encode_indices(ggml_context * ctx, ggml_context * weights_ctx, const std::string & prefix, ggml_tensor * residual, ggml_tensor * norm) {
    ggml_tensor * projected = ggml_mul_mat(ctx, ggml_get_tensor(weights_ctx, (prefix + "project_in.weight").c_str()), residual);
    projected = ggml_add(ctx, projected, ggml_reshape_2d(ctx, ggml_get_tensor(weights_ctx, (prefix + "project_in.bias").c_str()), 64, 1));
    ggml_tensor * score = ggml_scale(ctx, ggml_mul_mat(ctx, ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), projected), 2.0f);
    score = ggml_sub(ctx, score, ggml_repeat(ctx, norm, score));
    return ggml_argmax(ctx, score);
}

void self_check_codec_encode_semantic_blocks_graph(const std::string & model_path, int blocks, const char * label) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{"a.se.conv.weight"};
    for (int block = 0; block < blocks; ++block) {
        codec_add_semantic_block_names(names, block);
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error(std::string("failed to load codec semantic ") + label + " tensors from GGUF");
    }

    const int frames = 12;
    ggml_init_params params{};
    params.mem_size = 32 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error(std::string("failed to allocate codec semantic ") + label + " graph context");
    }
    ggml_tensor * semantic = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, frames, 768, 1);
    ggml_tensor * out = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.se.conv.weight"), semantic, 1, 1, 1);
    for (int block = 0; block < blocks; ++block) {
        out = codec_semantic_block(ctx.get(), weights_ctx, out, block);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error(std::string("failed to allocate codec semantic ") + label + " graph tensors");
    }
    std::vector<float> semantic_values(static_cast<size_t>(frames * 768));
    for (size_t i = 0; i < semantic_values.size(); ++i) {
        semantic_values[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 512.0f;
    }
    ggml_backend_tensor_set(semantic, semantic_values.data(), 0, semantic_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("codec semantic ") + label + " graph compute failed");
    }
    if (out->ne[0] != frames || out->ne[1] != 768 || out->ne[2] != 1) {
        throw std::runtime_error(std::string("codec semantic ") + label + " output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("codec semantic ") + label + " output contains non-finite values");
        }
    }
}

void self_check_codec_encode_semantic_conv_graph(const std::string & model_path) {
    self_check_codec_encode_semantic_blocks_graph(model_path, 0, "conv");
}

void self_check_codec_encode_semantic_block0_graph(const std::string & model_path) {
    self_check_codec_encode_semantic_blocks_graph(model_path, 1, "block0");
}

void self_check_codec_encode_semantic_block1_graph(const std::string & model_path) {
    self_check_codec_encode_semantic_blocks_graph(model_path, 2, "block1");
}

void self_check_codec_encode_hubert_fe_graph_impl(const std::string & model_path, int num_layers, bool conv_only, const char * label) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names;
    for (int layer = 0; layer < num_layers; ++layer) {
        names.push_back("a.sm.fe.conv_layers." + std::to_string(layer) + ".conv.weight");
    }
    if (!conv_only) {
        names.push_back("a.sm.fe.conv_layers.0.layer_norm.weight");
        names.push_back("a.sm.fe.conv_layers.0.layer_norm.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error(std::string("failed to load codec HuBERT feature ") + label + " tensors from GGUF");
    }

    const int samples = 1600;
    ggml_init_params params{};
    params.mem_size = 64 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error(std::string("failed to allocate codec HuBERT feature ") + label + " graph context");
    }
    ggml_tensor * waveform = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, samples, 1, 1);
    ggml_tensor * out = waveform;
    constexpr int strides[] = {5, 2, 2, 2, 2, 2, 2};
    for (int layer = 0; layer < num_layers; ++layer) {
        const std::string weight_name = "a.sm.fe.conv_layers." + std::to_string(layer) + ".conv.weight";
        out = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, weight_name.c_str()), out, strides[layer], 0, 1);
        if (layer == 0) {
            if (conv_only && num_layers == 1) {
                break;
            }
            out = ggml_reshape_3d(ctx.get(), out, out->ne[0], 1, out->ne[1]);
            out = ggml_group_norm(ctx.get(), out, 512, 1e-5f);
            out = ggml_reshape_2d(ctx.get(), out, out->ne[0], out->ne[2]);
            ggml_tensor * weight = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.sm.fe.conv_layers.0.layer_norm.weight"), 1, 512, 1);
            ggml_tensor * bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.sm.fe.conv_layers.0.layer_norm.bias"), 1, 512, 1);
            out = ggml_add(ctx.get(), ggml_mul(ctx.get(), out, ggml_repeat(ctx.get(), weight, out)), ggml_repeat(ctx.get(), bias, out));
        }
        out = ggml_gelu(ctx.get(), out);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error(std::string("failed to allocate codec HuBERT feature ") + label + " graph tensors");
    }
    std::vector<float> waveform_values(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const float t = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(samples - 1);
        waveform_values[static_cast<size_t>(i)] = 0.1f * std::sin(19.0f * t) + 0.02f * t;
    }
    ggml_backend_tensor_set(waveform, waveform_values.data(), 0, waveform_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("codec HuBERT feature ") + label + " graph compute failed");
    }
    if (out->ne[0] <= 0 || out->ne[1] != 512) {
        throw std::runtime_error(std::string("codec HuBERT feature ") + label + " output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * std::max<int64_t>(out->ne[2], 1)));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("codec HuBERT feature ") + label + " output contains non-finite values");
        }
    }
}

void self_check_codec_encode_hubert_fp_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{
        "a.sm.fe.conv_layers.0.layer_norm.weight",
        "a.sm.fe.conv_layers.0.layer_norm.bias",
        "a.sm.fp.layer_norm.weight",
        "a.sm.fp.layer_norm.bias",
        "a.sm.fp.projection.weight",
        "a.sm.fp.projection.bias",
    };
    for (int layer = 0; layer < 7; ++layer) {
        names.push_back("a.sm.fe.conv_layers." + std::to_string(layer) + ".conv.weight");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec HuBERT feature projection tensors from GGUF");
    }

    const int samples = 1600;
    ggml_init_params params{};
    params.mem_size = 96 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec HuBERT feature projection graph context");
    }
    ggml_tensor * waveform = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, samples, 1, 1);
    ggml_tensor * features = waveform;
    constexpr int strides[] = {5, 2, 2, 2, 2, 2, 2};
    for (int layer = 0; layer < 7; ++layer) {
        const std::string weight_name = "a.sm.fe.conv_layers." + std::to_string(layer) + ".conv.weight";
        features = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, weight_name.c_str()), features, strides[layer], 0, 1);
        if (layer == 0) {
            features = ggml_reshape_3d(ctx.get(), features, features->ne[0], 1, features->ne[1]);
            features = ggml_group_norm(ctx.get(), features, 512, 1e-5f);
            features = ggml_reshape_2d(ctx.get(), features, features->ne[0], features->ne[2]);
            ggml_tensor * weight = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.sm.fe.conv_layers.0.layer_norm.weight"), 1, 512, 1);
            ggml_tensor * bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.sm.fe.conv_layers.0.layer_norm.bias"), 1, 512, 1);
            features = ggml_add(ctx.get(), ggml_mul(ctx.get(), features, ggml_repeat(ctx.get(), weight, features)), ggml_repeat(ctx.get(), bias, features));
        }
        features = ggml_gelu(ctx.get(), features);
    }
    const int64_t n_frames = features->ne[0];
    ggml_tensor * out = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), features), 512, n_frames);
    out = codec_layer_norm_2d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.sm.fp.layer_norm.weight"), ggml_get_tensor(weights_ctx, "a.sm.fp.layer_norm.bias"));
    out = codec_linear_2d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.sm.fp.projection.weight"), ggml_get_tensor(weights_ctx, "a.sm.fp.projection.bias"));
    out = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), out), n_frames, 768);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec HuBERT feature projection graph tensors");
    }
    std::vector<float> waveform_values(static_cast<size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        const float t = -1.0f + 2.0f * static_cast<float>(i) / static_cast<float>(samples - 1);
        waveform_values[static_cast<size_t>(i)] = 0.1f * std::sin(19.0f * t) + 0.02f * t;
    }
    ggml_backend_tensor_set(waveform, waveform_values.data(), 0, waveform_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec HuBERT feature projection graph compute failed");
    }
    if (out->ne[0] <= 0 || out->ne[1] != 768) {
        throw std::runtime_error("codec HuBERT feature projection output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec HuBERT feature projection output contains non-finite values");
        }
    }
}

void codec_add_hubert_pce_names(std::vector<std::string> & names) {
    names.push_back("a.sm.encoder.pce.conv.pz.w0");
    names.push_back("a.sm.encoder.pce.conv.pz.w1");
    names.push_back("a.sm.encoder.pce.conv.bias");
}

ggml_tensor * codec_hubert_pce_from_hidden(
    ggml_context * ctx,
    ggml_context * weights_ctx,
    ggml_tensor * hidden,
    const std::vector<ggml_tensor *> & kernel_tensors) {
    ggml_tensor * pos = nullptr;
    ggml_tensor * bias = ggml_get_tensor(weights_ctx, "a.sm.encoder.pce.conv.bias");
    const int64_t frames = hidden->ne[0];
    for (int group = 0; group < 16; ++group) {
        ggml_tensor * hidden_group = ggml_view_2d(
            ctx,
            hidden,
            frames,
            48,
            hidden->nb[1],
            static_cast<size_t>(group) * 48ull * hidden->nb[1]);
        ggml_tensor * conv = ggml_conv_1d(ctx, kernel_tensors[static_cast<size_t>(group)], hidden_group, 1, 64, 1);
        conv = ggml_view_3d(ctx, conv, frames, 48, 1, conv->nb[1], conv->nb[2], 0);
        conv = ggml_cont_2d(ctx, conv, frames, 48);
        ggml_tensor * bias_group = ggml_view_1d(ctx, bias, 48, static_cast<size_t>(group) * 48ull * ggml_element_size(bias));
        conv = ggml_add(ctx, conv, ggml_reshape_2d(ctx, bias_group, 1, 48));
        pos = pos == nullptr ? conv : ggml_concat(ctx, pos, conv, 1);
    }
    return ggml_add(ctx, hidden, ggml_gelu(ctx, pos));
}

std::vector<ggml_fp16_t> codec_make_hubert_pce_kernels(ggml_context * weights_ctx) {
    const std::vector<float> w0 = tensor_to_f32(ggml_get_tensor(weights_ctx, "a.sm.encoder.pce.conv.pz.w0"));
    const std::vector<float> w1 = tensor_to_f32(ggml_get_tensor(weights_ctx, "a.sm.encoder.pce.conv.pz.w1"));
    std::vector<float> inv_norms(128);
    for (int k = 0; k < 128; ++k) {
        float norm_sq = 0.0f;
        for (int global_out = 0; global_out < 768; ++global_out) {
            for (int in_ch = 0; in_ch < 48; ++in_ch) {
                const size_t idx = static_cast<size_t>(k) + 128ull * (static_cast<size_t>(in_ch) + 48ull * static_cast<size_t>(global_out));
                norm_sq += w1[idx] * w1[idx];
            }
        }
        inv_norms[static_cast<size_t>(k)] = 1.0f / std::sqrt(norm_sq);
    }
    std::vector<float> kernels_f32(static_cast<size_t>(16 * 128 * 48 * 48));
    for (int group = 0; group < 16; ++group) {
        for (int out_ch = 0; out_ch < 48; ++out_ch) {
            const int global_out = group * 48 + out_ch;
            for (int k = 0; k < 128; ++k) {
                const float scale = w0[static_cast<size_t>(k)] * inv_norms[static_cast<size_t>(k)];
                for (int in_ch = 0; in_ch < 48; ++in_ch) {
                    const size_t src = static_cast<size_t>(k) + 128ull * (static_cast<size_t>(in_ch) + 48ull * static_cast<size_t>(global_out));
                    const size_t dst = static_cast<size_t>(group) * 128ull * 48ull * 48ull
                        + static_cast<size_t>(k)
                        + 128ull * (static_cast<size_t>(in_ch) + 48ull * static_cast<size_t>(out_ch));
                    kernels_f32[dst] = w1[src] * scale;
                }
            }
        }
    }
    std::vector<ggml_fp16_t> kernels(static_cast<size_t>(16 * 128 * 48 * 48));
    ggml_fp32_to_fp16_row(kernels_f32.data(), kernels.data(), static_cast<int64_t>(kernels_f32.size()));
    return kernels;
}

void self_check_codec_encode_hubert_pce_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names;
    codec_add_hubert_pce_names(names);
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec HuBERT positional conv tensors from GGUF");
    }

    const std::vector<ggml_fp16_t> kernels = codec_make_hubert_pce_kernels(weights_ctx);

    const int frames = 12;
    ggml_init_params params{};
    params.mem_size = 128 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec HuBERT positional conv graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, frames, 768);
    std::vector<ggml_tensor *> kernel_tensors;
    kernel_tensors.reserve(16);
    for (int group = 0; group < 16; ++group) {
        kernel_tensors.push_back(ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, 48, 48));
    }
    ggml_tensor * out = codec_hubert_pce_from_hidden(ctx.get(), weights_ctx, hidden, kernel_tensors);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec HuBERT positional conv graph tensors");
    }
    std::vector<float> hidden_values(static_cast<size_t>(frames * 768));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 1024.0f;
    }
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    for (int group = 0; group < 16; ++group) {
        const size_t offset = static_cast<size_t>(group) * 128ull * 48ull * 48ull;
        ggml_backend_tensor_set(kernel_tensors[static_cast<size_t>(group)], kernels.data() + offset, 0, 128ull * 48ull * 48ull * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec HuBERT positional conv graph compute failed");
    }
    if (out->ne[0] != frames || out->ne[1] != 768) {
        throw std::runtime_error("codec HuBERT positional conv output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec HuBERT positional conv output contains non-finite values");
        }
    }
}

void self_check_codec_encode_hubert_prelude_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{"a.sm.encoder.layer_norm.weight", "a.sm.encoder.layer_norm.bias"};
    codec_add_hubert_pce_names(names);
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec HuBERT prelude tensors from GGUF");
    }

    const std::vector<ggml_fp16_t> kernels = codec_make_hubert_pce_kernels(weights_ctx);
    const int frames = 12;
    ggml_init_params params{};
    params.mem_size = 128 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec HuBERT prelude graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, frames, 768);
    std::vector<ggml_tensor *> kernel_tensors;
    kernel_tensors.reserve(16);
    for (int group = 0; group < 16; ++group) {
        kernel_tensors.push_back(ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, 48, 48));
    }
    ggml_tensor * out = codec_hubert_pce_from_hidden(ctx.get(), weights_ctx, hidden, kernel_tensors);
    out = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), out), 768, frames);
    out = codec_layer_norm_2d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.sm.encoder.layer_norm.weight"), ggml_get_tensor(weights_ctx, "a.sm.encoder.layer_norm.bias"));
    out = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), out), frames, 768);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec HuBERT prelude graph tensors");
    }
    std::vector<float> hidden_values(static_cast<size_t>(frames * 768));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 1024.0f;
    }
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    for (int group = 0; group < 16; ++group) {
        const size_t offset = static_cast<size_t>(group) * 128ull * 48ull * 48ull;
        ggml_backend_tensor_set(kernel_tensors[static_cast<size_t>(group)], kernels.data() + offset, 0, 128ull * 48ull * 48ull * sizeof(ggml_fp16_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec HuBERT prelude graph compute failed");
    }
    if (out->ne[0] != frames || out->ne[1] != 768) {
        throw std::runtime_error("codec HuBERT prelude output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec HuBERT prelude output contains non-finite values");
        }
    }
}

void codec_add_hubert_layer_names(std::vector<std::string> & names, int layer) {
    const std::string prefix = "a.sm.encoder.layers." + std::to_string(layer);
    names.push_back(prefix + ".attention.q_proj.weight");
    names.push_back(prefix + ".attention.q_proj.bias");
    names.push_back(prefix + ".attention.k_proj.weight");
    names.push_back(prefix + ".attention.k_proj.bias");
    names.push_back(prefix + ".attention.v_proj.weight");
    names.push_back(prefix + ".attention.v_proj.bias");
    names.push_back(prefix + ".attention.out_proj.weight");
    names.push_back(prefix + ".attention.out_proj.bias");
    names.push_back(prefix + ".layer_norm.weight");
    names.push_back(prefix + ".layer_norm.bias");
    names.push_back(prefix + ".ff.int.weight");
    names.push_back(prefix + ".ff.int.bias");
    names.push_back(prefix + ".ff.out.weight");
    names.push_back(prefix + ".ff.out.bias");
    names.push_back(prefix + ".final_layer_norm.weight");
    names.push_back(prefix + ".final_layer_norm.bias");
}

ggml_tensor * codec_hubert_linear(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & weight, const std::string & bias) {
    return codec_linear_2d(ctx, inp, ggml_get_tensor(weights_ctx, weight.c_str()), ggml_get_tensor(weights_ctx, bias.c_str()));
}

ggml_tensor * codec_hubert_encoder_layer(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * hidden, int layer) {
    const std::string prefix = "a.sm.encoder.layers." + std::to_string(layer);
    const int64_t frames = hidden->ne[0];
    ggml_tensor * hidden_t = ggml_cont_2d(ctx, ggml_transpose(ctx, hidden), 768, frames);
    ggml_tensor * q = codec_hubert_linear(ctx, weights_ctx, hidden_t, prefix + ".attention.q_proj.weight", prefix + ".attention.q_proj.bias");
    ggml_tensor * k = codec_hubert_linear(ctx, weights_ctx, hidden_t, prefix + ".attention.k_proj.weight", prefix + ".attention.k_proj.bias");
    ggml_tensor * v = codec_hubert_linear(ctx, weights_ctx, hidden_t, prefix + ".attention.v_proj.weight", prefix + ".attention.v_proj.bias");
    q = ggml_reshape_3d(ctx, q, 64, 12, frames);
    k = ggml_reshape_3d(ctx, k, 64, 12, frames);
    v = ggml_reshape_3d(ctx, v, 64, 12, frames);
    ggml_tensor * q4 = ggml_permute(ctx, ggml_reshape_4d(ctx, q, 64, 12, frames, 1), 0, 2, 1, 3);
    ggml_tensor * k4 = ggml_permute(ctx, ggml_reshape_4d(ctx, k, 64, 12, frames, 1), 0, 2, 1, 3);
    ggml_tensor * attn = ggml_mul_mat(ctx, k4, q4);
    attn = ggml_scale(ctx, attn, 1.0f / 8.0f);
    attn = ggml_soft_max(ctx, attn);
    ggml_tensor * v_for_mm = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, v, 64, 12, frames, 1), 1, 2, 0, 3));
    ggml_tensor * out = ggml_mul_mat(ctx, v_for_mm, attn);
    out = ggml_permute(ctx, out, 0, 2, 1, 3);
    out = ggml_cont_2d(ctx, out, 768, frames);
    out = codec_hubert_linear(ctx, weights_ctx, out, prefix + ".attention.out_proj.weight", prefix + ".attention.out_proj.bias");
    out = ggml_add(ctx, hidden_t, out);
    out = codec_layer_norm_2d(ctx, out, ggml_get_tensor(weights_ctx, (prefix + ".layer_norm.weight").c_str()), ggml_get_tensor(weights_ctx, (prefix + ".layer_norm.bias").c_str()));
    ggml_tensor * ff = codec_hubert_linear(ctx, weights_ctx, out, prefix + ".ff.int.weight", prefix + ".ff.int.bias");
    ff = ggml_gelu(ctx, ff);
    ff = codec_hubert_linear(ctx, weights_ctx, ff, prefix + ".ff.out.weight", prefix + ".ff.out.bias");
    out = ggml_add(ctx, out, ff);
    out = codec_layer_norm_2d(ctx, out, ggml_get_tensor(weights_ctx, (prefix + ".final_layer_norm.weight").c_str()), ggml_get_tensor(weights_ctx, (prefix + ".final_layer_norm.bias").c_str()));
    return ggml_cont_2d(ctx, ggml_transpose(ctx, out), frames, 768);
}

void self_check_codec_encode_hubert_layers_graph_impl(const std::string & model_path, int num_layers, const char * label) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names;
    for (int layer = 0; layer < num_layers; ++layer) {
        codec_add_hubert_layer_names(names, layer);
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error(std::string("failed to load codec HuBERT ") + label + " tensors from GGUF");
    }

    const int frames = 8;
    ggml_init_params params{};
    params.mem_size = 1024ull * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error(std::string("failed to allocate codec HuBERT ") + label + " graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, frames, 768);
    ggml_tensor * out = hidden;
    for (int layer = 0; layer < num_layers; ++layer) {
        out = codec_hubert_encoder_layer(ctx.get(), weights_ctx, out, layer);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error(std::string("failed to allocate codec HuBERT ") + label + " graph tensors");
    }
    std::vector<float> hidden_values(static_cast<size_t>(frames * 768));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 1024.0f;
    }
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("codec HuBERT ") + label + " graph compute failed");
    }
    if (out->ne[0] != frames || out->ne[1] != 768) {
        throw std::runtime_error(std::string("codec HuBERT ") + label + " output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(std::string("codec HuBERT ") + label + " output contains non-finite values");
        }
    }
}

void self_check_codec_encode_hubert_layer0_graph(const std::string & model_path) {
    self_check_codec_encode_hubert_layers_graph_impl(model_path, 1, "layer0");
}

void self_check_codec_encode_hubert_layers_graph(const std::string & model_path) {
    self_check_codec_encode_hubert_layers_graph_impl(model_path, 12, "layers");
}

void self_check_codec_encode_hubert_mean_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names;
    for (int layer = 0; layer < 12; ++layer) {
        codec_add_hubert_layer_names(names, layer);
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec HuBERT mean tensors from GGUF");
    }

    const int frames = 8;
    const int out_frames = (frames + 1) / 2;
    ggml_init_params params{};
    params.mem_size = 1024ull * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec HuBERT mean graph context");
    }
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, frames, 768);
    ggml_tensor * out = hidden;
    ggml_tensor * hidden_sum = hidden;
    for (int layer = 0; layer < 12; ++layer) {
        out = codec_hubert_encoder_layer(ctx.get(), weights_ctx, out, layer);
        hidden_sum = ggml_add(ctx.get(), hidden_sum, out);
    }
    ggml_tensor * mean = ggml_scale(ctx.get(), hidden_sum, 1.0f / 13.0f);
    mean = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), mean), 768, frames);
    ggml_tensor * row_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, out_frames);
    ggml_tensor * downsampled = ggml_get_rows(ctx.get(), mean, row_ids);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, downsampled);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec HuBERT mean graph tensors");
    }
    std::vector<float> hidden_values(static_cast<size_t>(frames * 768));
    for (size_t i = 0; i < hidden_values.size(); ++i) {
        hidden_values[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 1024.0f;
    }
    std::vector<int32_t> ids(static_cast<size_t>(out_frames));
    for (int i = 0; i < out_frames; ++i) {
        ids[static_cast<size_t>(i)] = i * 2;
    }
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, hidden_values.size() * sizeof(float));
    ggml_backend_tensor_set(row_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec HuBERT mean graph compute failed");
    }
    if (downsampled->ne[0] != 768 || downsampled->ne[1] != out_frames) {
        throw std::runtime_error("codec HuBERT mean output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(downsampled->ne[0] * downsampled->ne[1]));
    ggml_backend_tensor_get(downsampled, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec HuBERT mean output contains non-finite values");
        }
    }
}

void self_check_codec_encode_hubert_fe_conv0_conv_graph(const std::string & model_path) {
    self_check_codec_encode_hubert_fe_graph_impl(model_path, 1, true, "conv0 conv");
}

void self_check_codec_encode_hubert_fe_conv0_graph(const std::string & model_path) {
    self_check_codec_encode_hubert_fe_graph_impl(model_path, 1, false, "conv0");
}

void self_check_codec_encode_hubert_fe_graph(const std::string & model_path) {
    self_check_codec_encode_hubert_fe_graph_impl(model_path, 7, false, "feature");
}

TensorBlock run_reference_semantic_project(const std::string & model_path, const std::string & wav_path) {
    ReferenceAudioData ref;
    {
        ScopedProfileTimer timer("reference_semantic_load_wav_resample");
        ref = load_reference_wav(wav_path);
    }
    constexpr int pad = 160;
    std::vector<float> waveform(static_cast<size_t>(pad) + ref.semantic_16k.size() + static_cast<size_t>(pad), 0.0f);
    std::copy(ref.semantic_16k.begin(), ref.semantic_16k.end(), waveform.begin() + pad);
    dump_node_values("cpp_reference_semantic_waveform", waveform);

    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{
        "a.sm.fe.conv_layers.0.layer_norm.weight",
        "a.sm.fe.conv_layers.0.layer_norm.bias",
        "a.sm.fp.layer_norm.weight",
        "a.sm.fp.layer_norm.bias",
        "a.sm.fp.projection.weight",
        "a.sm.fp.projection.bias",
        "a.sm.encoder.layer_norm.weight",
        "a.sm.encoder.layer_norm.bias",
        "a.se.conv.weight",
    };
    for (int layer = 0; layer < 7; ++layer) {
        names.push_back("a.sm.fe.conv_layers." + std::to_string(layer) + ".conv.weight");
    }
    codec_add_hubert_pce_names(names);
    for (int layer = 0; layer < 12; ++layer) {
        codec_add_hubert_layer_names(names, layer);
    }
    for (int block = 0; block < 2; ++block) {
        codec_add_semantic_block_names(names, block);
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load reference semantic tensors from GGUF");
    }

    const std::vector<ggml_fp16_t> kernels = codec_make_hubert_pce_kernels(weights_ctx);
    ggml_init_params params{};
    params.mem_size = 3072ull * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate reference semantic graph context");
    }
    ggml_tensor * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, static_cast<int64_t>(waveform.size()), 1, 1);
    ggml_tensor * features = input;
    std::vector<std::pair<std::string, ggml_tensor *>> stage_outputs;
    constexpr int strides[] = {5, 2, 2, 2, 2, 2, 2};
    for (int layer = 0; layer < 7; ++layer) {
        const std::string weight_name = "a.sm.fe.conv_layers." + std::to_string(layer) + ".conv.weight";
        features = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, weight_name.c_str()), features, strides[layer], 0, 1);
        if (layer == 0) {
            features = ggml_reshape_3d(ctx.get(), features, features->ne[0], 1, features->ne[1]);
            features = ggml_group_norm(ctx.get(), features, 512, 1e-5f);
            features = ggml_reshape_2d(ctx.get(), features, features->ne[0], features->ne[2]);
            ggml_tensor * weight = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.sm.fe.conv_layers.0.layer_norm.weight"), 1, 512, 1);
            ggml_tensor * bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.sm.fe.conv_layers.0.layer_norm.bias"), 1, 512, 1);
            features = ggml_add(ctx.get(), ggml_mul(ctx.get(), features, ggml_repeat(ctx.get(), weight, features)), ggml_repeat(ctx.get(), bias, features));
        }
        features = ggml_gelu(ctx.get(), features);
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_reference_semantic_fe" + std::to_string(layer), features});
        }
    }
    const int64_t frames = features->ne[0];
    ggml_tensor * hidden = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), features), 512, frames);
    hidden = codec_layer_norm_2d(ctx.get(), hidden, ggml_get_tensor(weights_ctx, "a.sm.fp.layer_norm.weight"), ggml_get_tensor(weights_ctx, "a.sm.fp.layer_norm.bias"));
    hidden = codec_linear_2d(ctx.get(), hidden, ggml_get_tensor(weights_ctx, "a.sm.fp.projection.weight"), ggml_get_tensor(weights_ctx, "a.sm.fp.projection.bias"));
    hidden = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), hidden), frames, 768);
    if (!node_dump_dir().empty()) {
        stage_outputs.push_back({"cpp_reference_semantic_feature_project", hidden});
    }
    std::vector<ggml_tensor *> kernel_tensors;
    for (int group = 0; group < 16; ++group) {
        kernel_tensors.push_back(ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 128, 48, 48));
    }
    hidden = codec_hubert_pce_from_hidden(ctx.get(), weights_ctx, hidden, kernel_tensors);
    hidden = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), hidden), 768, frames);
    hidden = codec_layer_norm_2d(ctx.get(), hidden, ggml_get_tensor(weights_ctx, "a.sm.encoder.layer_norm.weight"), ggml_get_tensor(weights_ctx, "a.sm.encoder.layer_norm.bias"));
    hidden = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), hidden), frames, 768);
    if (!node_dump_dir().empty()) {
        stage_outputs.push_back({"cpp_reference_semantic_encoder_prelude", hidden});
    }
    ggml_tensor * hidden_sum = hidden;
    ggml_tensor * out = hidden;
    for (int layer = 0; layer < 12; ++layer) {
        out = codec_hubert_encoder_layer(ctx.get(), weights_ctx, out, layer);
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_reference_semantic_encoder_layer" + std::to_string(layer), out});
        }
        hidden_sum = ggml_add(ctx.get(), hidden_sum, out);
    }
    out = ggml_scale(ctx.get(), hidden_sum, 1.0f / 13.0f);
    if (!node_dump_dir().empty()) {
        stage_outputs.push_back({"cpp_reference_semantic_hidden_mean", out});
    }
    const int64_t out_frames = (frames + 1) / 2;
    out = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), out), 768, frames);
    ggml_tensor * row_ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, out_frames);
    out = ggml_get_rows(ctx.get(), out, row_ids);
    out = ggml_cont_3d(ctx.get(), ggml_transpose(ctx.get(), out), out_frames, 768, 1);
    if (!node_dump_dir().empty()) {
        stage_outputs.push_back({"cpp_reference_semantic_downsampled", out});
    }
    out = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.se.conv.weight"), out, 1, 1, 1);
    if (!node_dump_dir().empty()) {
        stage_outputs.push_back({"cpp_reference_semantic_conv", out});
    }
    for (int block = 0; block < 2; ++block) {
        out = codec_semantic_block(ctx.get(), weights_ctx, out, block);
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_reference_semantic_block" + std::to_string(block), out});
        }
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    for (const auto & stage : stage_outputs) {
        ggml_build_forward_expand(graph, stage.second);
    }
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate reference semantic graph tensors");
    }
    ggml_backend_tensor_set(input, waveform.data(), 0, waveform.size() * sizeof(float));
    for (int group = 0; group < 16; ++group) {
        const size_t offset = static_cast<size_t>(group) * 128ull * 48ull * 48ull;
        ggml_backend_tensor_set(kernel_tensors[static_cast<size_t>(group)], kernels.data() + offset, 0, 128ull * 48ull * 48ull * sizeof(ggml_fp16_t));
    }
    std::vector<int32_t> ids(static_cast<size_t>(out_frames));
    for (int64_t i = 0; i < out_frames; ++i) {
        ids[static_cast<size_t>(i)] = static_cast<int32_t>(i * 2);
    }
    ggml_backend_tensor_set(row_ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    {
        ScopedProfileTimer timer("reference_semantic_compute_graph");
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("reference semantic graph compute failed");
        }
    }
    for (const auto & stage : stage_outputs) {
        std::vector<float> stage_values(static_cast<size_t>(ggml_nelements(stage.second)));
        ggml_backend_tensor_get(stage.second, stage_values.data(), 0, stage_values.size() * sizeof(float));
        dump_node_values(stage.first, stage_values);
    }
    if (out->ne[0] != out_frames || out->ne[1] != 768 || out->ne[2] != 1) {
        throw std::runtime_error("reference semantic output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("reference semantic output contains non-finite values");
        }
    }
    TensorBlock block;
    block.values = std::move(values);
    block.frames = out->ne[0];
    block.channels = 768;
    return block;
}

int64_t self_check_reference_semantic(const std::string & model_path, const std::string & wav_path) {
    TensorBlock block = run_reference_semantic_project(model_path, wav_path);
    return block.frames;
}

void self_check_codec_encode_fc_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, {"a.fc.weight", "a.fc.bias"})) {
        throw std::runtime_error("failed to load codec encode fc tensors from GGUF");
    }

    const int frames = 4;
    ggml_init_params params{};
    params.mem_size = 256 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec encode fc graph context");
    }
    ggml_tensor * acoustic = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, frames, 256, 1);
    ggml_tensor * semantic = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, frames, 768, 1);
    ggml_tensor * embeddings = ggml_concat(ctx.get(), acoustic, semantic, 1);
    embeddings = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), ggml_reshape_2d(ctx.get(), embeddings, frames, 1024)), 1024, frames);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc.weight"), embeddings);
    out = ggml_add(ctx.get(), out, ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc.bias"), 1024, 1));
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec encode fc graph tensors");
    }
    std::vector<float> acoustic_values(static_cast<size_t>(frames * 256));
    std::vector<float> semantic_values(static_cast<size_t>(frames * 768));
    for (size_t i = 0; i < acoustic_values.size(); ++i) {
        acoustic_values[i] = static_cast<float>(static_cast<int>(i % 97) - 48) / 512.0f;
    }
    for (size_t i = 0; i < semantic_values.size(); ++i) {
        semantic_values[i] = static_cast<float>(static_cast<int>(i % 131) - 65) / 512.0f;
    }
    ggml_backend_tensor_set(acoustic, acoustic_values.data(), 0, acoustic_values.size() * sizeof(float));
    ggml_backend_tensor_set(semantic, semantic_values.data(), 0, semantic_values.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec encode fc graph compute failed");
    }
    if (out->ne[0] != 1024 || out->ne[1] != frames) {
        throw std::runtime_error("codec encode fc output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec encode fc output contains non-finite values");
        }
    }
}

void self_check_codec_encode_quantizer0_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    const std::string prefix = "a.q.quantizers.0.";
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, {
            prefix + "codebook.embed",
            prefix + "project_in.weight",
            prefix + "project_in.bias",
            prefix + "project_out.weight",
            prefix + "project_out.bias",
        })) {
        throw std::runtime_error("failed to load codec encode quantizer0 tensors from GGUF");
    }
    const std::vector<float> embed_norm = codec_codebook_norm(weights_ctx, prefix);

    const int frames = 4;
    ggml_init_params params{};
    params.mem_size = 256 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec encode quantizer0 graph context");
    }
    ggml_tensor * residual = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1024, frames);
    ggml_tensor * norm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1024, 1);
    ggml_tensor * ids = codec_quantizer_encode_indices(ctx.get(), weights_ctx, prefix, residual, norm);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, ids);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec encode quantizer0 graph tensors");
    }
    std::vector<float> residual_values(static_cast<size_t>(frames * 1024));
    for (size_t i = 0; i < residual_values.size(); ++i) {
        residual_values[i] = static_cast<float>(static_cast<int>(i % 193) - 96) / 1024.0f;
    }
    ggml_backend_tensor_set(residual, residual_values.data(), 0, residual_values.size() * sizeof(float));
    ggml_backend_tensor_set(norm, embed_norm.data(), 0, embed_norm.size() * sizeof(float));
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec encode quantizer0 graph compute failed");
    }
    if (ids->ne[0] != frames) {
        throw std::runtime_error("codec encode quantizer0 output shape mismatch");
    }
    std::vector<int32_t> values(static_cast<size_t>(frames));
    ggml_backend_tensor_get(ids, values.data(), 0, values.size() * sizeof(int32_t));
    for (int32_t value : values) {
        if (value < 0 || value >= 1024) {
            throw std::runtime_error("codec encode quantizer0 output id out of range");
        }
    }
}

void self_check_codec_encode_quantizers_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_in.weight");
        names.push_back(prefix + "project_in.bias");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec encode quantizer tensors from GGUF");
    }
    std::vector<std::vector<float>> embed_norms;
    for (int c = 0; c < k_num_codebooks; ++c) {
        embed_norms.push_back(codec_codebook_norm(weights_ctx, "a.q.quantizers." + std::to_string(c) + "."));
    }

    const int frames = 4;
    ggml_init_params params{};
    params.mem_size = 1024 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec encode quantizers graph context");
    }
    ggml_tensor * residual_input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1024, frames);
    ggml_tensor * residual = residual_input;
    std::vector<ggml_tensor *> norms;
    std::vector<ggml_tensor *> outputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * norm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1024, 1);
        norms.push_back(norm);
        ggml_tensor * ids = codec_quantizer_encode_indices(ctx.get(), weights_ctx, prefix, residual, norm);
        outputs.push_back(ids);
        ggml_tensor * quantized = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        quantized = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), quantized);
        quantized = ggml_add(ctx.get(), quantized, ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1));
        residual = ggml_sub(ctx.get(), residual, quantized);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    for (ggml_tensor * out : outputs) {
        ggml_build_forward_expand(graph, out);
    }
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec encode quantizers graph tensors");
    }
    std::vector<float> residual_values(static_cast<size_t>(frames * 1024));
    for (size_t i = 0; i < residual_values.size(); ++i) {
        residual_values[i] = static_cast<float>(static_cast<int>(i % 193) - 96) / 1024.0f;
    }
    ggml_backend_tensor_set(residual_input, residual_values.data(), 0, residual_values.size() * sizeof(float));
    for (int c = 0; c < k_num_codebooks; ++c) {
        ggml_backend_tensor_set(norms[static_cast<size_t>(c)], embed_norms[static_cast<size_t>(c)].data(), 0, embed_norms[static_cast<size_t>(c)].size() * sizeof(float));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec encode quantizers graph compute failed");
    }
    for (ggml_tensor * out : outputs) {
        if (out->ne[0] != frames) {
            throw std::runtime_error("codec encode quantizers output shape mismatch");
        }
        std::vector<int32_t> values(static_cast<size_t>(frames));
        ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(int32_t));
        for (int32_t value : values) {
            if (value < 0 || value >= 1024) {
                throw std::runtime_error("codec encode quantizers output id out of range");
            }
        }
    }
}

CodeMatrix encode_reference_codes(const std::string & model_path, const std::string & wav_path) {
    TensorBlock acoustic;
    {
        ScopedProfileTimer timer("reference_encode_acoustic_total");
        acoustic = run_reference_acoustic_project(model_path, wav_path);
    }
    TensorBlock semantic;
    {
        ScopedProfileTimer timer("reference_encode_semantic_total");
        semantic = run_reference_semantic_project(model_path, wav_path);
    }
    dump_node_values("cpp_reference_acoustic", acoustic.values);
    dump_node_values("cpp_reference_semantic", semantic.values);
    const int frames = static_cast<int>(std::min(acoustic.frames, semantic.frames));
    if (frames <= 0) {
        throw std::runtime_error("reference codes have no overlapping frames");
    }

    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{"a.fc.weight", "a.fc.bias"};
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_in.weight");
        names.push_back(prefix + "project_in.bias");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load reference code tensors from GGUF");
    }
    std::vector<std::vector<float>> embed_norms;
    for (int c = 0; c < k_num_codebooks; ++c) {
        embed_norms.push_back(codec_codebook_norm(weights_ctx, "a.q.quantizers." + std::to_string(c) + "."));
    }

    ggml_init_params params{};
    params.mem_size = 1024ull * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate reference code graph context");
    }
    ggml_tensor * acoustic_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, frames, 256, 1);
    ggml_tensor * semantic_input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, frames, 768, 1);
    ggml_tensor * embeddings = ggml_concat(ctx.get(), acoustic_input, semantic_input, 1);
    embeddings = ggml_cont_2d(ctx.get(), ggml_transpose(ctx.get(), ggml_reshape_2d(ctx.get(), embeddings, frames, 1024)), 1024, frames);
    ggml_tensor * residual = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc.weight"), embeddings);
    residual = ggml_add(ctx.get(), residual, ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc.bias"), 1024, 1));
    std::vector<ggml_tensor *> norms;
    std::vector<ggml_tensor *> outputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * norm = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1024, 1);
        norms.push_back(norm);
        ggml_tensor * ids = codec_quantizer_encode_indices(ctx.get(), weights_ctx, prefix, residual, norm);
        outputs.push_back(ids);
        ggml_tensor * quantized = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        quantized = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), quantized);
        quantized = ggml_add(ctx.get(), quantized, ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1));
        residual = ggml_sub(ctx.get(), residual, quantized);
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    for (ggml_tensor * out : outputs) {
        ggml_build_forward_expand(graph, out);
    }
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate reference code graph tensors");
    }
    ggml_backend_tensor_set(acoustic_input, acoustic.values.data(), 0, static_cast<size_t>(frames * 256) * sizeof(float));
    ggml_backend_tensor_set(semantic_input, semantic.values.data(), 0, static_cast<size_t>(frames * 768) * sizeof(float));
    for (int c = 0; c < k_num_codebooks; ++c) {
        ggml_backend_tensor_set(norms[static_cast<size_t>(c)], embed_norms[static_cast<size_t>(c)].data(), 0, embed_norms[static_cast<size_t>(c)].size() * sizeof(float));
    }
    {
        ScopedProfileTimer timer("reference_encode_fc_quantizer_compute_graph");
        if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("reference code graph compute failed");
        }
    }
    CodeMatrix codes;
    codes.frames = frames;
    codes.data.assign(static_cast<size_t>(frames * k_num_codebooks), 0);
    for (int c = 0; c < k_num_codebooks; ++c) {
        if (outputs[static_cast<size_t>(c)]->ne[0] != frames) {
            throw std::runtime_error("reference code output shape mismatch");
        }
        std::vector<int32_t> ids(static_cast<size_t>(frames));
        ggml_backend_tensor_get(outputs[static_cast<size_t>(c)], ids.data(), 0, ids.size() * sizeof(int32_t));
        for (int f = 0; f < frames; ++f) {
            const int32_t id = ids[static_cast<size_t>(f)];
            if (id < 0 || id >= 1024) {
                throw std::runtime_error("reference code id out of range");
            }
            codes.at(c, f) = id;
        }
    }
    return codes;
}

CodeMatrix self_check_reference_codes(const std::string & model_path, const std::string & wav_path) {
    return encode_reference_codes(model_path, wav_path);
}

void self_check_codec_project_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{"a.fc2.weight", "a.fc2.bias"};
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec project tensors from GGUF");
    }

    const int frames = 2;
    ggml_init_params params{};
    params.mem_size = 128 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec project graph context");
    }
    ggml_tensor * quantized = nullptr;
    std::vector<ggml_tensor *> code_inputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, frames);
        code_inputs.push_back(ids);
        ggml_tensor * embed = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        ggml_tensor * projected = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), embed);
        ggml_tensor * bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1);
        projected = ggml_add(ctx.get(), projected, bias);
        quantized = quantized == nullptr ? projected : ggml_add(ctx.get(), quantized, projected);
    }
    ggml_tensor * out = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.weight"), quantized);
    ggml_tensor * fc2_bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.bias"), 256, 1);
    out = ggml_add(ctx.get(), out, fc2_bias);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec project graph tensors");
    }
    std::vector<int32_t> code_values(static_cast<size_t>(frames));
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int f = 0; f < frames; ++f) {
            code_values[static_cast<size_t>(f)] = (c * 17 + f * 31) % k_codebook_data_size;
        }
        ggml_backend_tensor_set(code_inputs[static_cast<size_t>(c)], code_values.data(), 0, code_values.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec project graph compute failed");
    }
    if (out->ne[0] != 256 || out->ne[1] != frames) {
        throw std::runtime_error("codec project output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec project output contains non-finite values");
        }
    }
}

void self_check_codec_conv1_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{"a.fc2.weight", "a.fc2.bias", "a.ad.conv1.weight", "a.ad.conv1.bias"};
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec conv1 tensors from GGUF");
    }

    const int frames = 2;
    ggml_init_params params{};
    params.mem_size = 160 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec conv1 graph context");
    }
    ggml_tensor * quantized = nullptr;
    std::vector<ggml_tensor *> code_inputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, frames);
        code_inputs.push_back(ids);
        ggml_tensor * embed = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        ggml_tensor * projected = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), embed);
        ggml_tensor * bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1);
        projected = ggml_add(ctx.get(), projected, bias);
        quantized = quantized == nullptr ? projected : ggml_add(ctx.get(), quantized, projected);
    }
    ggml_tensor * acoustic = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.weight"), quantized);
    ggml_tensor * fc2_bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.bias"), 256, 1);
    acoustic = ggml_add(ctx.get(), acoustic, fc2_bias);
    acoustic = ggml_cont_3d(ctx.get(), ggml_transpose(ctx.get(), acoustic), frames, 256, 1);
    ggml_tensor * out = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.conv1.weight"), acoustic, 1, 3, 1);
    ggml_tensor * conv_bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.conv1.bias"), 1, 1024, 1);
    out = ggml_add(ctx.get(), out, conv_bias);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec conv1 graph tensors");
    }
    std::vector<int32_t> code_values(static_cast<size_t>(frames));
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int f = 0; f < frames; ++f) {
            code_values[static_cast<size_t>(f)] = (c * 17 + f * 31) % k_codebook_data_size;
        }
        ggml_backend_tensor_set(code_inputs[static_cast<size_t>(c)], code_values.data(), 0, code_values.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec conv1 graph compute failed");
    }
    if (out->ne[0] != frames || out->ne[1] != 1024 || out->ne[2] != 1) {
        throw std::runtime_error("codec conv1 output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec conv1 output contains non-finite values");
        }
    }
}

ggml_tensor * codec_snake_1d(ggml_context * ctx, ggml_tensor * inp, ggml_tensor * alpha, ggml_tensor * one, ggml_tensor * eps) {
    alpha = ggml_cast(ctx, alpha, GGML_TYPE_F32);
    alpha = ggml_reshape_3d(ctx, alpha, 1, alpha->ne[1], 1);
    ggml_tensor * alpha_full = ggml_repeat(ctx, alpha, inp);
    ggml_tensor * one_full = ggml_repeat(ctx, one, inp);
    ggml_tensor * eps_full = ggml_repeat(ctx, eps, inp);
    ggml_tensor * inv_alpha = ggml_div(ctx, one_full, ggml_add(ctx, alpha_full, eps_full));
    ggml_tensor * sin_sq = ggml_sqr(ctx, ggml_sin(ctx, ggml_mul(ctx, alpha_full, inp)));
    return ggml_add(ctx, inp, ggml_mul(ctx, inv_alpha, sin_sq));
}

ggml_tensor * codec_slice_time_3d(ggml_context * ctx, ggml_tensor * tensor, int64_t start, int64_t length) {
    ggml_tensor * view = ggml_view_3d(
        ctx,
        tensor,
        length,
        tensor->ne[1],
        tensor->ne[2],
        tensor->nb[1],
        tensor->nb[2],
        static_cast<size_t>(start) * ggml_element_size(tensor));
    return ggml_cont_3d(ctx, view, length, view->ne[1], view->ne[2]);
}

ggml_tensor * codec_conv_1d_bias(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & weight, const std::string & bias, int padding, int dilation, int stride) {
    ggml_tensor * out = ggml_conv_1d(ctx, ggml_get_tensor(weights_ctx, weight.c_str()), inp, stride, padding, dilation);
    ggml_tensor * conv_bias = ggml_reshape_3d(ctx, ggml_get_tensor(weights_ctx, bias.c_str()), 1, out->ne[1], 1);
    return ggml_add(ctx, out, conv_bias);
}

ggml_tensor * codec_conv_1d_bias_f32_im2col(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & weight, const std::string & bias, int padding, int dilation, int stride) {
    ggml_tensor * kernel = ggml_get_tensor(weights_ctx, weight.c_str());
    ggml_tensor * im2col = ggml_im2col(ctx, kernel, inp, stride, 0, padding, 0, dilation, 0, false, GGML_TYPE_F32);
    ggml_tensor * cols = ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]);
    kernel = ggml_cast(ctx, kernel, GGML_TYPE_F32);
    ggml_tensor * weights = ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]);
    ggml_tensor * out = mul_mat_f32(ctx, cols, weights);
    out = ggml_reshape_3d(ctx, out, im2col->ne[1], kernel->ne[2], im2col->ne[2]);
    ggml_tensor * conv_bias = ggml_reshape_3d(ctx, ggml_get_tensor(weights_ctx, bias.c_str()), 1, out->ne[1], 1);
    return ggml_add(ctx, out, conv_bias);
}

ggml_tensor * codec_residual_unit_1d(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & prefix, ggml_tensor * one, ggml_tensor * eps, int dilation) {
    ggml_tensor * out = codec_snake_1d(ctx, inp, ggml_get_tensor(weights_ctx, (prefix + ".snake1.alpha").c_str()), one, eps);
    out = codec_conv_1d_bias(ctx, weights_ctx, out, prefix + ".conv1.weight", prefix + ".conv1.bias", ((7 - 1) * dilation) / 2, dilation);
    out = codec_snake_1d(ctx, out, ggml_get_tensor(weights_ctx, (prefix + ".snake2.alpha").c_str()), one, eps);
    out = codec_conv_1d_bias(ctx, weights_ctx, out, prefix + ".conv2.weight", prefix + ".conv2.bias", 0, 1);
    return ggml_add(ctx, inp, out);
}

ggml_tensor * codec_encoder_residual_unit_1d(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, const std::string & prefix, ggml_tensor * one, ggml_tensor * eps, int dilation, std::vector<std::pair<std::string, ggml_tensor *>> * stage_outputs) {
    ggml_tensor * out = codec_snake_1d(ctx, inp, ggml_get_tensor(weights_ctx, (prefix + ".snake1.alpha").c_str()), one, eps);
    if (stage_outputs != nullptr && prefix == "a.ae.block.0.res_unit1") stage_outputs->push_back({"cpp_reference_acoustic_block0_res1_snake1", out});
    out = codec_conv_1d_bias_f32_im2col(ctx, weights_ctx, out, prefix + ".conv1.weight", prefix + ".conv1.bias", ((7 - 1) * dilation) / 2, dilation);
    if (stage_outputs != nullptr && prefix == "a.ae.block.0.res_unit1") stage_outputs->push_back({"cpp_reference_acoustic_block0_res1_conv1", out});
    out = codec_snake_1d(ctx, out, ggml_get_tensor(weights_ctx, (prefix + ".snake2.alpha").c_str()), one, eps);
    if (stage_outputs != nullptr && prefix == "a.ae.block.0.res_unit1") stage_outputs->push_back({"cpp_reference_acoustic_block0_res1_snake2", out});
    out = codec_conv_1d_bias_f32_im2col(ctx, weights_ctx, out, prefix + ".conv2.weight", prefix + ".conv2.bias", 0, 1);
    if (stage_outputs != nullptr && prefix == "a.ae.block.0.res_unit1") stage_outputs->push_back({"cpp_reference_acoustic_block0_res1_conv2", out});
    return ggml_add(ctx, inp, out);
}

void codec_add_decoder_block_names(std::vector<std::string> & names, int block) {
    const std::string prefix = "a.ad.block." + std::to_string(block);
    names.push_back(prefix + ".snake1.alpha");
    names.push_back(prefix + ".conv_t1.weight");
    names.push_back(prefix + ".conv_t1.bias");
    for (int unit = 1; unit <= 3; ++unit) {
        const std::string unit_prefix = prefix + ".res_unit" + std::to_string(unit);
        names.push_back(unit_prefix + ".snake1.alpha");
        names.push_back(unit_prefix + ".conv1.weight");
        names.push_back(unit_prefix + ".conv1.bias");
        names.push_back(unit_prefix + ".snake2.alpha");
        names.push_back(unit_prefix + ".conv2.weight");
        names.push_back(unit_prefix + ".conv2.bias");
    }
}

ggml_tensor * codec_decoder_block(ggml_context * ctx, ggml_context * weights_ctx, ggml_tensor * inp, int block, int stride, ggml_tensor * one, ggml_tensor * eps, std::vector<std::pair<std::string, ggml_tensor *>> * profile_stages = nullptr) {
    const std::string prefix = "a.ad.block." + std::to_string(block);
    ggml_tensor * out = codec_snake_1d(ctx, inp, ggml_get_tensor(weights_ctx, (prefix + ".snake1.alpha").c_str()), one, eps);
    ggml_tensor * conv_t1_weight = ggml_cast(ctx, ggml_get_tensor(weights_ctx, (prefix + ".conv_t1.weight").c_str()), GGML_TYPE_F32);
    ggml_tensor * full = ggml_conv_transpose_1d(ctx, conv_t1_weight, out, stride, 0, 1);
    if (profile_stages != nullptr) {
        profile_stages->push_back({"codec_stage_block" + std::to_string(block) + "_conv_transpose", full});
    }
    const int padding = (stride + 1) / 2;
    const int output_padding = stride % 2;
    out = codec_slice_time_3d(ctx, full, padding, full->ne[0] - padding - (padding - output_padding));
    ggml_tensor * bias = ggml_reshape_3d(ctx, ggml_get_tensor(weights_ctx, (prefix + ".conv_t1.bias").c_str()), 1, out->ne[1], 1);
    out = ggml_add(ctx, out, bias);
    if (profile_stages != nullptr) {
        profile_stages->push_back({"codec_stage_block" + std::to_string(block) + "_upsample_bias", out});
    }
    out = codec_residual_unit_1d(ctx, weights_ctx, out, prefix + ".res_unit1", one, eps, 1);
    if (profile_stages != nullptr) {
        profile_stages->push_back({"codec_stage_block" + std::to_string(block) + "_res1", out});
    }
    out = codec_residual_unit_1d(ctx, weights_ctx, out, prefix + ".res_unit2", one, eps, 3);
    if (profile_stages != nullptr) {
        profile_stages->push_back({"codec_stage_block" + std::to_string(block) + "_res2", out});
    }
    out = codec_residual_unit_1d(ctx, weights_ctx, out, prefix + ".res_unit3", one, eps, 9);
    if (profile_stages != nullptr) {
        profile_stages->push_back({"codec_stage_block" + std::to_string(block) + "_res3", out});
    }
    return out;
}

void self_check_codec_block0_conv_t1_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{
        "a.fc2.weight", "a.fc2.bias",
        "a.ad.conv1.weight", "a.ad.conv1.bias",
        "a.ad.block.0.snake1.alpha",
        "a.ad.block.0.conv_t1.weight",
        "a.ad.block.0.conv_t1.bias",
    };
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec block0 conv_t1 tensors from GGUF");
    }

    const int frames = 2;
    ggml_init_params params{};
    params.mem_size = 256 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec block0 conv_t1 graph context");
    }
    ggml_tensor * quantized = nullptr;
    std::vector<ggml_tensor *> code_inputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, frames);
        code_inputs.push_back(ids);
        ggml_tensor * embed = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        ggml_tensor * projected = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), embed);
        ggml_tensor * bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1);
        projected = ggml_add(ctx.get(), projected, bias);
        quantized = quantized == nullptr ? projected : ggml_add(ctx.get(), quantized, projected);
    }
    ggml_tensor * acoustic = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.weight"), quantized);
    ggml_tensor * fc2_bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.bias"), 256, 1);
    acoustic = ggml_add(ctx.get(), acoustic, fc2_bias);
    acoustic = ggml_cont_3d(ctx.get(), ggml_transpose(ctx.get(), acoustic), frames, 256, 1);
    ggml_tensor * out = ggml_conv_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.conv1.weight"), acoustic, 1, 3, 1);
    ggml_tensor * conv_bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.conv1.bias"), 1, 1024, 1);
    out = ggml_add(ctx.get(), out, conv_bias);
    ggml_tensor * one = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    out = codec_snake_1d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.ad.block.0.snake1.alpha"), one, eps);
    ggml_tensor * full = ggml_conv_transpose_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.block.0.conv_t1.weight"), out, 8, 0, 1);
    const int padding = 4;
    const int output_padding = 0;
    const int64_t length = full->ne[0] - padding - (padding - output_padding);
    out = codec_slice_time_3d(ctx.get(), full, padding, length);
    ggml_tensor * t_bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.block.0.conv_t1.bias"), 1, 512, 1);
    out = ggml_add(ctx.get(), out, t_bias);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec block0 conv_t1 graph tensors");
    }
    const float one_value = 1.0f;
    const float eps_value = 1e-9f;
    ggml_backend_tensor_set(one, &one_value, 0, sizeof(one_value));
    ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(eps_value));
    std::vector<int32_t> code_values(static_cast<size_t>(frames));
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int f = 0; f < frames; ++f) {
            code_values[static_cast<size_t>(f)] = (c * 17 + f * 31) % k_codebook_data_size;
        }
        ggml_backend_tensor_set(code_inputs[static_cast<size_t>(c)], code_values.data(), 0, code_values.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec block0 conv_t1 graph compute failed");
    }
    if (out->ne[0] != frames * 8 || out->ne[1] != 512 || out->ne[2] != 1) {
        throw std::runtime_error("codec block0 conv_t1 output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec block0 conv_t1 output contains non-finite values");
        }
    }
}

void self_check_codec_block0_res_unit1_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{
        "a.fc2.weight", "a.fc2.bias",
        "a.ad.conv1.weight", "a.ad.conv1.bias",
        "a.ad.block.0.snake1.alpha",
        "a.ad.block.0.conv_t1.weight",
        "a.ad.block.0.conv_t1.bias",
        "a.ad.block.0.res_unit1.snake1.alpha",
        "a.ad.block.0.res_unit1.conv1.weight",
        "a.ad.block.0.res_unit1.conv1.bias",
        "a.ad.block.0.res_unit1.snake2.alpha",
        "a.ad.block.0.res_unit1.conv2.weight",
        "a.ad.block.0.res_unit1.conv2.bias",
    };
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec block0 res_unit1 tensors from GGUF");
    }

    const int frames = 2;
    ggml_init_params params{};
    params.mem_size = 320 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec block0 res_unit1 graph context");
    }
    ggml_tensor * quantized = nullptr;
    std::vector<ggml_tensor *> code_inputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, frames);
        code_inputs.push_back(ids);
        ggml_tensor * embed = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        ggml_tensor * projected = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), embed);
        ggml_tensor * bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1);
        projected = ggml_add(ctx.get(), projected, bias);
        quantized = quantized == nullptr ? projected : ggml_add(ctx.get(), quantized, projected);
    }
    ggml_tensor * acoustic = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.weight"), quantized);
    ggml_tensor * fc2_bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.bias"), 256, 1);
    acoustic = ggml_add(ctx.get(), acoustic, fc2_bias);
    acoustic = ggml_cont_3d(ctx.get(), ggml_transpose(ctx.get(), acoustic), frames, 256, 1);
    ggml_tensor * out = codec_conv_1d_bias(ctx.get(), weights_ctx, acoustic, "a.ad.conv1.weight", "a.ad.conv1.bias", 3, 1);
    ggml_tensor * one = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    out = codec_snake_1d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.ad.block.0.snake1.alpha"), one, eps);
    ggml_tensor * full = ggml_conv_transpose_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.block.0.conv_t1.weight"), out, 8, 0, 1);
    out = codec_slice_time_3d(ctx.get(), full, 4, full->ne[0] - 8);
    ggml_tensor * t_bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.block.0.conv_t1.bias"), 1, 512, 1);
    out = ggml_add(ctx.get(), out, t_bias);
    out = codec_residual_unit_1d(ctx.get(), weights_ctx, out, "a.ad.block.0.res_unit1", one, eps, 1);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec block0 res_unit1 graph tensors");
    }
    const float one_value = 1.0f;
    const float eps_value = 1e-9f;
    ggml_backend_tensor_set(one, &one_value, 0, sizeof(one_value));
    ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(eps_value));
    std::vector<int32_t> code_values(static_cast<size_t>(frames));
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int f = 0; f < frames; ++f) {
            code_values[static_cast<size_t>(f)] = (c * 17 + f * 31) % k_codebook_data_size;
        }
        ggml_backend_tensor_set(code_inputs[static_cast<size_t>(c)], code_values.data(), 0, code_values.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec block0 res_unit1 graph compute failed");
    }
    if (out->ne[0] != frames * 8 || out->ne[1] != 512 || out->ne[2] != 1) {
        throw std::runtime_error("codec block0 res_unit1 output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec block0 res_unit1 output contains non-finite values");
        }
    }
}

void self_check_codec_block0_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{
        "a.fc2.weight", "a.fc2.bias",
        "a.ad.conv1.weight", "a.ad.conv1.bias",
        "a.ad.block.0.snake1.alpha",
        "a.ad.block.0.conv_t1.weight",
        "a.ad.block.0.conv_t1.bias",
    };
    for (int unit = 1; unit <= 3; ++unit) {
        const std::string prefix = "a.ad.block.0.res_unit" + std::to_string(unit);
        names.push_back(prefix + ".snake1.alpha");
        names.push_back(prefix + ".conv1.weight");
        names.push_back(prefix + ".conv1.bias");
        names.push_back(prefix + ".snake2.alpha");
        names.push_back(prefix + ".conv2.weight");
        names.push_back(prefix + ".conv2.bias");
    }
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec block0 tensors from GGUF");
    }

    const int frames = 2;
    ggml_init_params params{};
    params.mem_size = 512 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec block0 graph context");
    }
    ggml_tensor * quantized = nullptr;
    std::vector<ggml_tensor *> code_inputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, frames);
        code_inputs.push_back(ids);
        ggml_tensor * embed = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        ggml_tensor * projected = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), embed);
        ggml_tensor * bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1);
        projected = ggml_add(ctx.get(), projected, bias);
        quantized = quantized == nullptr ? projected : ggml_add(ctx.get(), quantized, projected);
    }
    ggml_tensor * acoustic = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.weight"), quantized);
    ggml_tensor * fc2_bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.bias"), 256, 1);
    acoustic = ggml_add(ctx.get(), acoustic, fc2_bias);
    acoustic = ggml_cont_3d(ctx.get(), ggml_transpose(ctx.get(), acoustic), frames, 256, 1);
    ggml_tensor * out = codec_conv_1d_bias(ctx.get(), weights_ctx, acoustic, "a.ad.conv1.weight", "a.ad.conv1.bias", 3, 1);
    ggml_tensor * one = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    out = codec_snake_1d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.ad.block.0.snake1.alpha"), one, eps);
    ggml_tensor * full = ggml_conv_transpose_1d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.block.0.conv_t1.weight"), out, 8, 0, 1);
    out = codec_slice_time_3d(ctx.get(), full, 4, full->ne[0] - 8);
    ggml_tensor * t_bias = ggml_reshape_3d(ctx.get(), ggml_get_tensor(weights_ctx, "a.ad.block.0.conv_t1.bias"), 1, 512, 1);
    out = ggml_add(ctx.get(), out, t_bias);
    out = codec_residual_unit_1d(ctx.get(), weights_ctx, out, "a.ad.block.0.res_unit1", one, eps, 1);
    out = codec_residual_unit_1d(ctx.get(), weights_ctx, out, "a.ad.block.0.res_unit2", one, eps, 3);
    out = codec_residual_unit_1d(ctx.get(), weights_ctx, out, "a.ad.block.0.res_unit3", one, eps, 9);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec block0 graph tensors");
    }
    const float one_value = 1.0f;
    const float eps_value = 1e-9f;
    ggml_backend_tensor_set(one, &one_value, 0, sizeof(one_value));
    ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(eps_value));
    std::vector<int32_t> code_values(static_cast<size_t>(frames));
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int f = 0; f < frames; ++f) {
            code_values[static_cast<size_t>(f)] = (c * 17 + f * 31) % k_codebook_data_size;
        }
        ggml_backend_tensor_set(code_inputs[static_cast<size_t>(c)], code_values.data(), 0, code_values.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec block0 graph compute failed");
    }
    if (out->ne[0] != frames * 8 || out->ne[1] != 512 || out->ne[2] != 1) {
        throw std::runtime_error("codec block0 output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec block0 output contains non-finite values");
        }
    }
}

void self_check_codec_block1_graph(const std::string & model_path) {
    GgufWithTensors loaded = open_gguf_with_tensors(model_path);
    ggml_context * weights_ctx = loaded.ggml.get();
    const gguf_context * gguf_ctx = loaded.gguf.get();
    BackendPtr backend = make_backend();
    if (!backend) {
        throw std::runtime_error("failed to initialize GGML CPU backend");
    }
    BackendBufferPtr weights_buffer(ggml_backend_alloc_ctx_tensors(weights_ctx, backend.get()));
    if (!weights_buffer) {
        throw std::runtime_error("failed to allocate GGML weight tensors");
    }
    std::vector<std::string> names{
        "a.fc2.weight", "a.fc2.bias",
        "a.ad.conv1.weight", "a.ad.conv1.bias",
    };
    codec_add_decoder_block_names(names, 0);
    codec_add_decoder_block_names(names, 1);
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    if (!load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)) {
        throw std::runtime_error("failed to load codec block1 tensors from GGUF");
    }

    const int frames = 2;
    ggml_init_params params{};
    params.mem_size = 768 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec block1 graph context");
    }
    ggml_tensor * quantized = nullptr;
    std::vector<ggml_tensor *> code_inputs;
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, frames);
        code_inputs.push_back(ids);
        ggml_tensor * embed = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
        ggml_tensor * projected = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), embed);
        ggml_tensor * bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1);
        projected = ggml_add(ctx.get(), projected, bias);
        quantized = quantized == nullptr ? projected : ggml_add(ctx.get(), quantized, projected);
    }
    ggml_tensor * acoustic = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.weight"), quantized);
    ggml_tensor * fc2_bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.bias"), 256, 1);
    acoustic = ggml_add(ctx.get(), acoustic, fc2_bias);
    acoustic = ggml_cont_3d(ctx.get(), ggml_transpose(ctx.get(), acoustic), frames, 256, 1);
    ggml_tensor * out = codec_conv_1d_bias(ctx.get(), weights_ctx, acoustic, "a.ad.conv1.weight", "a.ad.conv1.bias", 3, 1);
    ggml_tensor * one = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    ggml_tensor * eps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
    out = codec_decoder_block(ctx.get(), weights_ctx, out, 0, 8, one, eps);
    out = codec_decoder_block(ctx.get(), weights_ctx, out, 1, 5, one, eps);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    BackendBufferPtr graph_buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec block1 graph tensors");
    }
    const float one_value = 1.0f;
    const float eps_value = 1e-9f;
    ggml_backend_tensor_set(one, &one_value, 0, sizeof(one_value));
    ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(eps_value));
    std::vector<int32_t> code_values(static_cast<size_t>(frames));
    for (int c = 0; c < k_num_codebooks; ++c) {
        for (int f = 0; f < frames; ++f) {
            code_values[static_cast<size_t>(f)] = (c * 17 + f * 31) % k_codebook_data_size;
        }
        ggml_backend_tensor_set(code_inputs[static_cast<size_t>(c)], code_values.data(), 0, code_values.size() * sizeof(int32_t));
    }
    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("codec block1 graph compute failed");
    }
    if (out->ne[0] != frames * 8 * 5 || out->ne[1] != 256 || out->ne[2] != 1) {
        throw std::runtime_error("codec block1 output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec block1 output contains non-finite values");
        }
    }
}

std::vector<float> run_codec_waveform_graph_impl(const std::string & model_path, const std::vector<int32_t> & codes_tn, int frames, BackendRuntime * runtime) {
    if (frames <= 0 || codes_tn.size() != static_cast<size_t>(frames * k_num_codebooks)) {
        throw std::invalid_argument("codec decode codes must have source-view shape [frames, codebooks]");
    }
    ScopedProfileTimer total_timer("codec_decode_total");
    GgufWithTensors loaded;
    BackendPtr backend;
    BackendBufferPtr weights_buffer;
    ggml_context * weights_ctx = nullptr;
    const gguf_context * gguf_ctx = nullptr;
    ggml_backend * backend_ptr = nullptr;
    if (runtime == nullptr) {
        loaded = open_gguf_with_tensors(model_path);
        weights_ctx = loaded.ggml.get();
        gguf_ctx = loaded.gguf.get();
        backend = make_backend();
        if (!backend) {
            throw std::runtime_error("failed to initialize GGML CPU backend");
        }
        backend_ptr = backend.get();
        weights_buffer.reset(ggml_backend_alloc_ctx_tensors(weights_ctx, backend_ptr));
        if (!weights_buffer) {
            throw std::runtime_error("failed to allocate GGML weight tensors");
        }
    } else {
        weights_ctx = runtime->weights_ggml.get();
        gguf_ctx = runtime->weights_gguf.get();
        backend_ptr = runtime->backend.get();
    }
    std::vector<std::string> names{
        "a.fc2.weight", "a.fc2.bias",
        "a.ad.conv1.weight", "a.ad.conv1.bias",
        "a.ad.snake1.alpha",
        "a.ad.conv2.weight", "a.ad.conv2.bias",
    };
    for (int block = 0; block < 5; ++block) {
        codec_add_decoder_block_names(names, block);
    }
    for (int c = 0; c < k_num_codebooks; ++c) {
        const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
        names.push_back(prefix + "codebook.embed");
        names.push_back(prefix + "project_out.weight");
        names.push_back(prefix + "project_out.bias");
    }
    {
        ScopedProfileTimer timer("codec_decode_load_weights");
        const bool loaded_ok = runtime == nullptr
            ? load_named_tensors_from_gguf(model_path, weights_ctx, gguf_ctx, names)
            : ensure_backend_runtime_weights(model_path, *runtime, names);
        if (!loaded_ok) {
            throw std::runtime_error("failed to load codec waveform tensors from GGUF");
        }
    }

    ggml_init_params params{};
    params.mem_size = 1024 * 1024 * 1024;
    params.no_alloc = true;
    GgmlPtr ctx(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to allocate codec waveform graph context");
    }
    ggml_tensor * out = nullptr;
    ggml_tensor * one = nullptr;
    ggml_tensor * eps = nullptr;
    std::vector<ggml_tensor *> code_inputs;
    std::vector<std::pair<std::string, ggml_tensor *>> stage_outputs;
    std::vector<std::pair<std::string, ggml_tensor *>> profile_stages;
    {
        ScopedProfileTimer timer("codec_decode_build_graph");
        ggml_tensor * quantized = nullptr;
        for (int c = 0; c < k_num_codebooks; ++c) {
            const std::string prefix = "a.q.quantizers." + std::to_string(c) + ".";
            ggml_tensor * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, frames);
            code_inputs.push_back(ids);
            ggml_tensor * embed = ggml_get_rows(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "codebook.embed").c_str()), ids);
            ggml_tensor * projected = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.weight").c_str()), embed);
            ggml_tensor * bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, (prefix + "project_out.bias").c_str()), 1024, 1);
            projected = ggml_add(ctx.get(), projected, bias);
            quantized = quantized == nullptr ? projected : ggml_add(ctx.get(), quantized, projected);
        }
        ggml_tensor * acoustic = ggml_mul_mat(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.weight"), quantized);
        ggml_tensor * fc2_bias = ggml_reshape_2d(ctx.get(), ggml_get_tensor(weights_ctx, "a.fc2.bias"), 256, 1);
        acoustic = ggml_add(ctx.get(), acoustic, fc2_bias);
        acoustic = ggml_cont_3d(ctx.get(), ggml_transpose(ctx.get(), acoustic), frames, 256, 1);
        if (codec_stage_profile_enabled()) {
            profile_stages.push_back({"codec_stage_project", acoustic});
        }
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_decode_project", acoustic});
        }
        out = codec_conv_1d_bias(ctx.get(), weights_ctx, acoustic, "a.ad.conv1.weight", "a.ad.conv1.bias", 3, 1);
        if (codec_stage_profile_enabled()) {
            profile_stages.push_back({"codec_stage_conv1", out});
        }
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_decode_conv1", out});
        }
        one = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        eps = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        const int strides[5] = {8, 5, 4, 2, 3};
        for (int block = 0; block < 5; ++block) {
            out = codec_decoder_block(ctx.get(), weights_ctx, out, block, strides[block], one, eps, codec_stage_profile_enabled() ? &profile_stages : nullptr);
            if (codec_stage_profile_enabled()) {
                profile_stages.push_back({"codec_stage_block" + std::to_string(block), out});
            }
            if (!node_dump_dir().empty()) {
                stage_outputs.push_back({"cpp_decode_block" + std::to_string(block), out});
            }
        }
        out = codec_snake_1d(ctx.get(), out, ggml_get_tensor(weights_ctx, "a.ad.snake1.alpha"), one, eps);
        out = codec_conv_1d_bias(ctx.get(), weights_ctx, out, "a.ad.conv2.weight", "a.ad.conv2.bias", 3, 1);
        if (codec_stage_profile_enabled()) {
            profile_stages.push_back({"codec_stage_final", out});
        }
        if (!node_dump_dir().empty()) {
            stage_outputs.push_back({"cpp_decode_waveform", out});
        }
    }
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    for (const auto & stage : stage_outputs) {
        ggml_build_forward_expand(graph, stage.second);
    }
    BackendBufferPtr graph_buffer(nullptr);
    {
        ScopedProfileTimer timer("codec_decode_alloc_graph");
        graph_buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend_ptr));
    }
    if (!graph_buffer) {
        throw std::runtime_error("failed to allocate codec waveform graph tensors");
    }
    const float one_value = 1.0f;
    const float eps_value = 1e-9f;
    {
        ScopedProfileTimer timer("codec_decode_set_inputs");
        ggml_backend_tensor_set(one, &one_value, 0, sizeof(one_value));
        ggml_backend_tensor_set(eps, &eps_value, 0, sizeof(eps_value));
        for (int c = 0; c < k_num_codebooks; ++c) {
            std::vector<int32_t> code_values(static_cast<size_t>(frames));
            for (int f = 0; f < frames; ++f) {
                code_values[static_cast<size_t>(f)] = codes_tn[static_cast<size_t>(f) * k_num_codebooks + static_cast<size_t>(c)];
            }
            ggml_backend_tensor_set(code_inputs[static_cast<size_t>(c)], code_values.data(), 0, code_values.size() * sizeof(int32_t));
        }
    }
    if (codec_stage_profile_enabled()) {
        for (const auto & stage : profile_stages) {
            ggml_cgraph * stage_graph = ggml_new_graph(ctx.get());
            ggml_build_forward_expand(stage_graph, stage.second);
            ScopedProfileTimer timer(stage.first.c_str());
            if (ggml_backend_graph_compute(backend_ptr, stage_graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("codec stage profile graph compute failed: " + stage.first);
            }
        }
    }
    {
        ScopedProfileTimer timer("codec_decode_compute_graph");
        if (ggml_backend_graph_compute(backend_ptr, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("codec waveform graph compute failed");
        }
    }
    for (const auto & stage : stage_outputs) {
        std::vector<float> stage_values(static_cast<size_t>(ggml_nelements(stage.second)));
        ggml_backend_tensor_get(stage.second, stage_values.data(), 0, stage_values.size() * sizeof(float));
        dump_node_values(stage.first, stage_values);
    }
    if (out->ne[0] != frames * 8 * 5 * 4 * 2 * 3 || out->ne[1] != 1 || out->ne[2] != 1) {
        throw std::runtime_error("codec waveform output shape mismatch");
    }
    std::vector<float> values(static_cast<size_t>(out->ne[0] * out->ne[1] * out->ne[2]));
    {
        ScopedProfileTimer timer("codec_decode_get_waveform");
        ggml_backend_tensor_get(out, values.data(), 0, values.size() * sizeof(float));
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("codec waveform output contains non-finite values");
        }
    }
    return values;
}

std::vector<float> run_codec_waveform_graph(const std::string & model_path, const std::vector<int32_t> & codes_tn, int frames) {
    return run_codec_waveform_graph_impl(model_path, codes_tn, frames, nullptr);
}

std::vector<float> run_codec_waveform_graph_with_runtime(const std::string & model_path, const std::vector<int32_t> & codes_tn, int frames, BackendRuntime & runtime) {
    return run_codec_waveform_graph_impl(model_path, codes_tn, frames, &runtime);
}

std::vector<float> run_synthetic_codec_waveform_graph(const std::string & model_path) {
    const int frames = 2;
    std::vector<int32_t> codes_tn(static_cast<size_t>(frames * k_num_codebooks));
    for (int f = 0; f < frames; ++f) {
        for (int c = 0; c < k_num_codebooks; ++c) {
            codes_tn[static_cast<size_t>(f) * k_num_codebooks + static_cast<size_t>(c)] = (c * 17 + f * 31) % k_codebook_data_size;
        }
    }
    return run_codec_waveform_graph(model_path, codes_tn, frames);
}

void self_check_codec_waveform_graph(const std::string & model_path) {
    (void) run_synthetic_codec_waveform_graph(model_path);
}

void self_check_codec_wav(const std::string & model_path, const std::string & out_path) {
    write_wav_mono_16(out_path, run_synthetic_codec_waveform_graph(model_path), 24000);
}

std::vector<int32_t> read_finalized_codes_tn(const std::string & trace_json_path, int & frames) {
    std::ifstream in(trace_json_path);
    if (!in) {
        throw std::runtime_error("failed to open trace json: " + trace_json_path);
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const std::string key = "\"finalized_codes\"";
    size_t pos = text.find(key);
    if (pos == std::string::npos) {
        throw std::runtime_error("trace json missing finalized_codes");
    }
    pos = text.find('[', pos + key.size());
    if (pos == std::string::npos) {
        throw std::runtime_error("trace json finalized_codes is not an array");
    }
    std::vector<std::vector<int32_t>> rows;
    int depth = 0;
    std::vector<int32_t> row;
    for (; pos < text.size(); ++pos) {
        const char ch = text[pos];
        if (ch == '[') {
            ++depth;
            if (depth == 2) {
                row.clear();
            }
        } else if (ch == ']') {
            if (depth == 2) {
                rows.push_back(row);
            }
            --depth;
            if (depth == 0) {
                break;
            }
        } else if (depth == 2 && (std::isdigit(static_cast<unsigned char>(ch)) || ch == '-')) {
            size_t end = pos;
            while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == '-')) {
                ++end;
            }
            row.push_back(std::stoi(text.substr(pos, end - pos)));
            pos = end - 1;
        }
    }
    if (rows.empty()) {
        throw std::runtime_error("trace json finalized_codes is empty");
    }
    if (rows.size() == static_cast<size_t>(k_num_codebooks) && rows[0].size() != static_cast<size_t>(k_num_codebooks)) {
        const int source_frames = static_cast<int>(rows[0].size());
        std::vector<std::vector<int32_t>> transposed(static_cast<size_t>(source_frames), std::vector<int32_t>(k_num_codebooks));
        for (int c = 0; c < k_num_codebooks; ++c) {
            if (rows[static_cast<size_t>(c)].size() != static_cast<size_t>(source_frames)) {
                throw std::runtime_error("finalized_codes codebook rows have inconsistent frame counts");
            }
            for (int f = 0; f < source_frames; ++f) {
                transposed[static_cast<size_t>(f)][static_cast<size_t>(c)] = rows[static_cast<size_t>(c)][static_cast<size_t>(f)];
            }
        }
        rows = std::move(transposed);
    }
    for (const auto & row_values : rows) {
        if (row_values.size() != static_cast<size_t>(k_num_codebooks)) {
            throw std::runtime_error("finalized_codes rows must have 8 codebooks");
        }
    }
    frames = static_cast<int>(rows.size());
    std::vector<int32_t> codes_tn;
    codes_tn.reserve(static_cast<size_t>(frames * k_num_codebooks));
    for (const auto & r : rows) {
        codes_tn.insert(codes_tn.end(), r.begin(), r.end());
    }
    return codes_tn;
}

void decode_trace_json_to_wav(const std::string & model_path, const std::string & trace_json_path, const std::string & out_path) {
    int frames = 0;
    const std::vector<int32_t> codes = read_finalized_codes_tn(trace_json_path, frames);
    write_wav_mono_16(out_path, run_codec_waveform_graph(model_path, codes, frames), 24000);
}

uint32_t read_u32_le(std::istream & in) {
    uint8_t b[4]{};
    in.read(reinterpret_cast<char *>(b), 4);
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading WAV");
    }
    return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) | (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
}

uint16_t read_u16_le(std::istream & in) {
    uint8_t b[2]{};
    in.read(reinterpret_cast<char *>(b), 2);
    if (!in) {
        throw std::runtime_error("unexpected EOF while reading WAV");
    }
    return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

int32_t read_i24_le(const uint8_t * p) {
    int32_t v = static_cast<int32_t>(p[0]) | (static_cast<int32_t>(p[1]) << 8) | (static_cast<int32_t>(p[2]) << 16);
    if (v & 0x800000) {
        v |= ~0xFFFFFF;
    }
    return v;
}

AudioData read_wav_mono(const std::string & path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("failed to open WAV input: " + path);
    }
    char tag[4]{};
    in.read(tag, 4);
    if (std::strncmp(tag, "RIFF", 4) != 0) {
        throw std::runtime_error("WAV input is not RIFF: " + path);
    }
    (void) read_u32_le(in);
    in.read(tag, 4);
    if (std::strncmp(tag, "WAVE", 4) != 0) {
        throw std::runtime_error("WAV input is not WAVE: " + path);
    }

    uint16_t format = 0;
    uint16_t channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits = 0;
    std::vector<uint8_t> data;
    while (in.read(tag, 4)) {
        const uint32_t size = read_u32_le(in);
        if (std::strncmp(tag, "fmt ", 4) == 0) {
            format = read_u16_le(in);
            channels = read_u16_le(in);
            sample_rate = read_u32_le(in);
            (void) read_u32_le(in);
            (void) read_u16_le(in);
            bits = read_u16_le(in);
            if (size > 16) {
                in.seekg(static_cast<std::streamoff>(size - 16), std::ios::cur);
            }
        } else if (std::strncmp(tag, "data", 4) == 0) {
            data.resize(size);
            in.read(reinterpret_cast<char *>(data.data()), size);
            if (!in) {
                throw std::runtime_error("truncated WAV data: " + path);
            }
        } else {
            in.seekg(static_cast<std::streamoff>(size), std::ios::cur);
        }
        if (size & 1u) {
            in.seekg(1, std::ios::cur);
        }
    }
    if (channels == 0 || sample_rate == 0 || data.empty()) {
        throw std::runtime_error("WAV input missing fmt/data chunk: " + path);
    }
    if (!((format == 1 && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) || (format == 3 && bits == 32))) {
        throw std::runtime_error("unsupported WAV format; expected PCM 8/16/24/32 or float32");
    }
    const int bytes = bits / 8;
    const size_t frames = data.size() / (static_cast<size_t>(bytes) * channels);
    AudioData out;
    out.sample_rate = static_cast<int>(sample_rate);
    out.samples.resize(frames);
    for (size_t f = 0; f < frames; ++f) {
        double sum = 0.0;
        for (uint16_t ch = 0; ch < channels; ++ch) {
            const uint8_t * p = data.data() + (f * channels + ch) * bytes;
            float s = 0.0f;
            if (format == 3) {
                std::memcpy(&s, p, sizeof(float));
            } else if (bits == 8) {
                s = (static_cast<int>(*p) - 128) / 128.0f;
            } else if (bits == 16) {
                int16_t v;
                std::memcpy(&v, p, sizeof(v));
                s = static_cast<float>(v) / 32768.0f;
            } else if (bits == 24) {
                s = static_cast<float>(read_i24_le(p)) / 8388608.0f;
            } else {
                int32_t v;
                std::memcpy(&v, p, sizeof(v));
                s = static_cast<float>(v) / 2147483648.0f;
            }
            sum += s;
        }
        out.samples[f] = static_cast<float>(sum / channels);
    }
    return out;
}

std::vector<float> resample_linear(const std::vector<float> & input, int in_rate, int out_rate) {
    if (input.empty() || in_rate <= 0 || out_rate <= 0) {
        return {};
    }
    if (in_rate == out_rate) {
        return input;
    }
    const int g = std::gcd(in_rate, out_rate);
    const int orig = in_rate / g;
    const int target = out_rate / g;
    constexpr int lowpass_width = 6;
    constexpr double rolloff = 0.99;
    const double base_freq = static_cast<double>(std::min(orig, target)) * rolloff;
    const int width = static_cast<int>(std::ceil(static_cast<double>(lowpass_width) * orig / base_freq));
    const size_t out_n = std::max<size_t>(1, static_cast<size_t>(std::ceil(static_cast<double>(target) * input.size() / orig)));
    std::vector<float> out(out_n);
    for (size_t i = 0; i < out_n; ++i) {
        const int phase = static_cast<int>(i % static_cast<size_t>(target));
        const int64_t center = static_cast<int64_t>(i / static_cast<size_t>(target)) * orig;
        double sum = 0.0;
        for (int k = 0; k < 2 * width + orig; ++k) {
            const int64_t padded_idx = center + k;
            const int64_t input_idx = padded_idx - width;
            if (input_idx < 0 || input_idx >= static_cast<int64_t>(input.size())) {
                continue;
            }
            double t = static_cast<double>(k - width) / static_cast<double>(orig) -
                       static_cast<double>(phase) / static_cast<double>(target);
            t *= base_freq;
            t = std::clamp(t, -static_cast<double>(lowpass_width), static_cast<double>(lowpass_width));
            const double window = std::pow(std::cos(t * M_PI / static_cast<double>(lowpass_width) / 2.0), 2.0);
            const double tp = t * M_PI;
            const double sinc = std::abs(tp) < 1e-12 ? 1.0 : std::sin(tp) / tp;
            const double w = sinc * window * base_freq / static_cast<double>(orig);
            sum += static_cast<double>(input[static_cast<size_t>(input_idx)]) * w;
        }
        out[i] = static_cast<float>(sum);
    }
    return out;
}

ReferenceAudioData load_reference_wav(const std::string & path) {
    ReferenceAudioData out;
    out.original = read_wav_mono(path);
    out.audio_24k = resample_linear(out.original.samples, out.original.sample_rate, 24000);
    out.semantic_16k = resample_linear(out.audio_24k, 24000, 16000);
    if (out.audio_24k.empty() || out.semantic_16k.empty()) {
        throw std::runtime_error("reference WAV produced no samples: " + path);
    }
    return out;
}

PromptFormat parse_prompt_format(const std::string & value) {
    if (value == "higgstts") {
        return PromptFormat::higgstts;
    }
    if (value == "chatml") {
        return PromptFormat::chatml;
    }
    if (value == "boson-chatml") {
        return PromptFormat::boson_chatml;
    }
    throw std::invalid_argument("prompt-format must be one of: higgstts, chatml, boson-chatml");
}

const char * prompt_format_name(PromptFormat value) {
    switch (value) {
        case PromptFormat::higgstts:
            return "higgstts";
        case PromptFormat::chatml:
            return "chatml";
        case PromptFormat::boson_chatml:
            return "boson-chatml";
    }
    return "higgstts";
}

BackendKind parse_backend_kind(const std::string & value) {
    if (value == "cpu") {
        return BackendKind::cpu;
    }
    if (value == "cuda") {
        return BackendKind::cuda;
    }
    throw std::invalid_argument("backend must be one of: cpu, cuda");
}

const char * backend_kind_name(BackendKind value) {
    switch (value) {
        case BackendKind::cpu:
            return "cpu";
        case BackendKind::cuda:
            return "cuda";
    }
    return "cpu";
}

void write_wav_mono_16(const std::string & path, const std::vector<float> & samples, int sample_rate) {
    ScopedProfileTimer timer("pipeline_wav_write");
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open wav output: " + path);
    }

    const uint16_t channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * channels * bits_per_sample / 8;
    const uint16_t block_align = channels * bits_per_sample / 8;
    const uint32_t data_size = static_cast<uint32_t>(samples.size() * sizeof(int16_t));
    const uint32_t riff_size = 36 + data_size;

    out.write("RIFF", 4);
    out.write(reinterpret_cast<const char *>(&riff_size), sizeof(riff_size));
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    const uint32_t fmt_size = 16;
    const uint16_t audio_format = 1;
    out.write(reinterpret_cast<const char *>(&fmt_size), sizeof(fmt_size));
    out.write(reinterpret_cast<const char *>(&audio_format), sizeof(audio_format));
    out.write(reinterpret_cast<const char *>(&channels), sizeof(channels));
    out.write(reinterpret_cast<const char *>(&sample_rate), sizeof(sample_rate));
    out.write(reinterpret_cast<const char *>(&byte_rate), sizeof(byte_rate));
    out.write(reinterpret_cast<const char *>(&block_align), sizeof(block_align));
    out.write(reinterpret_cast<const char *>(&bits_per_sample), sizeof(bits_per_sample));
    out.write("data", 4);
    out.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

    for (float sample : samples) {
        const float clipped = std::max(-1.0f, std::min(1.0f, sample));
        const auto pcm = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
        out.write(reinterpret_cast<const char *>(&pcm), sizeof(pcm));
    }
}

std::vector<uint8_t> encode_mp3_mono(const std::vector<float> & samples, int sample_rate, int bitrate_kbps) {
    ScopedProfileTimer timer("pipeline_mp3_encode");
    shine_config_t config{};
    config.wave.channels = PCM_MONO;
    config.wave.samplerate = sample_rate;
    shine_set_config_mpeg_defaults(&config.mpeg);
    config.mpeg.mode = MONO;
    config.mpeg.bitr = bitrate_kbps;
    if (shine_check_config(config.wave.samplerate, config.mpeg.bitr) < 0) {
        throw std::runtime_error("unsupported MP3 sample rate/bitrate for shine");
    }
    shine_t encoder = shine_initialise(&config);
    if (encoder == nullptr) {
        throw std::runtime_error("failed to initialize shine MP3 encoder");
    }
    const int samples_per_pass = shine_samples_per_pass(encoder);
    std::vector<int16_t> pcm(static_cast<size_t>(samples_per_pass), 0);
    std::vector<uint8_t> mp3;
    for (size_t offset = 0; offset < samples.size(); offset += static_cast<size_t>(samples_per_pass)) {
        std::fill(pcm.begin(), pcm.end(), 0);
        const size_t n = std::min(static_cast<size_t>(samples_per_pass), samples.size() - offset);
        for (size_t i = 0; i < n; ++i) {
            const float clipped = std::max(-1.0f, std::min(1.0f, samples[offset + i]));
            pcm[i] = static_cast<int16_t>(std::lrint(clipped * 32767.0f));
        }
        int written = 0;
        unsigned char * data = shine_encode_buffer_interleaved(encoder, pcm.data(), &written);
        if (data != nullptr && written > 0) {
            mp3.insert(mp3.end(), data, data + written);
        }
    }
    int written = 0;
    unsigned char * data = shine_flush(encoder, &written);
    if (data != nullptr && written > 0) {
        mp3.insert(mp3.end(), data, data + written);
    }
    shine_close(encoder);
    return mp3;
}

void write_mp3_mono(const std::string & path, const std::vector<float> & samples, int sample_rate, int bitrate_kbps) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("failed to open mp3 output: " + path);
    }
    const std::vector<uint8_t> mp3 = encode_mp3_mono(samples, sample_rate, bitrate_kbps);
    out.write(reinterpret_cast<const char *>(mp3.data()), static_cast<std::streamsize>(mp3.size()));
}

} // namespace higgs_audio
