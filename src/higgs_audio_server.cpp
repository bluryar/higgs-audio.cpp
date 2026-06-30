#include "higgs_audio.h"

#include <arpa/inet.h>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <csignal>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

volatile sig_atomic_t g_stop = 0;
volatile sig_atomic_t g_server_fd = -1;

void on_signal(int) {
    g_stop = 1;
    if (g_server_fd >= 0) {
        shutdown(static_cast<int>(g_server_fd), SHUT_RDWR);
    }
}

struct ServerConfig {
    std::string model_path = higgs_audio::k_default_model_path;
    higgs_audio::BackendKind backend = higgs_audio::BackendKind::cpu;
    int port = 8080;
    int workers = 1;
    int queue_size = 8;
    int request_timeout_sec = 300;
    bool ar_scheduler = false;
};

struct QueuedRequest {
    int fd = -1;
    std::chrono::steady_clock::time_point enqueued_at;
};

class RequestQueue {
public:
    explicit RequestQueue(size_t capacity) : capacity_(capacity) {}

    bool push(int fd) {
        std::lock_guard<std::mutex> lock(mu_);
        if (closed_ || queue_.size() >= capacity_) {
            return false;
        }
        queue_.push_back(QueuedRequest{fd, std::chrono::steady_clock::now()});
        cv_.notify_one();
        return true;
    }

    bool pop(QueuedRequest & request) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return closed_ || !queue_.empty(); });
        if (queue_.empty()) {
            return false;
        }
        request = queue_.front();
        queue_.pop_front();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        closed_ = true;
        cv_.notify_all();
    }

private:
    size_t capacity_;
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<QueuedRequest> queue_;
    bool closed_ = false;
};

std::string json_string(const std::string & body, const std::string & key, const std::string & fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = body.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = body.find('"', pos + 1);
    if (pos == std::string::npos) {
        return fallback;
    }
    std::string out;
    for (++pos; pos < body.size(); ++pos) {
        const char ch = body[pos];
        if (ch == '"') {
            return out;
        }
        if (ch == '\\' && pos + 1 < body.size()) {
            out.push_back(body[++pos]);
        } else {
            out.push_back(ch);
        }
    }
    return fallback;
}

int json_int(const std::string & body, const std::string & key, int fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = body.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return fallback;
    }
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) {
        ++pos;
    }
    size_t end = pos;
    while (end < body.size() && (std::isdigit(static_cast<unsigned char>(body[end])) || body[end] == '-')) {
        ++end;
    }
    if (end == pos) {
        return fallback;
    }
    return std::stoi(body.substr(pos, end - pos));
}

float json_float(const std::string & body, const std::string & key, float fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = body.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return fallback;
    }
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) {
        ++pos;
    }
    size_t end = pos;
    while (end < body.size() && (std::isdigit(static_cast<unsigned char>(body[end])) || body[end] == '-' || body[end] == '+' || body[end] == '.' || body[end] == 'e' || body[end] == 'E')) {
        ++end;
    }
    if (end == pos) {
        return fallback;
    }
    return std::stof(body.substr(pos, end - pos));
}

bool json_bool(const std::string & body, const std::string & key, bool fallback) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = body.find(needle);
    if (pos == std::string::npos) {
        return fallback;
    }
    pos = body.find(':', pos + needle.size());
    if (pos == std::string::npos) {
        return fallback;
    }
    ++pos;
    while (pos < body.size() && std::isspace(static_cast<unsigned char>(body[pos]))) {
        ++pos;
    }
    if (body.compare(pos, 4, "true") == 0) {
        return true;
    }
    if (body.compare(pos, 5, "false") == 0) {
        return false;
    }
    return fallback;
}

std::string json_escape(const std::string & value) {
    std::string out;
    out.reserve(value.size() + 8);
    for (char ch : value) {
        if (ch == '"' || ch == '\\') {
            out.push_back('\\');
        }
        out.push_back(ch);
    }
    return out;
}

bool ends_with(const std::string & value, const char * suffix) {
    const std::string s(suffix);
    return value.size() >= s.size() && value.compare(value.size() - s.size(), s.size(), s) == 0;
}

void write_codes_json_file(const std::string & path, const higgs_audio::CodeMatrix & delayed, const higgs_audio::CodeMatrix & finalized) {
    if (path.empty()) {
        return;
    }
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("failed to open codes json: " + path);
    }
    out << "{\"delayed_codes\":[";
    for (int f = 0; f < delayed.frames; ++f) {
        if (f) out << ",";
        out << "[";
        for (int c = 0; c < delayed.codebooks; ++c) {
            if (c) out << ",";
            out << delayed.at(c, f);
        }
        out << "]";
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

void send_all(int fd, const std::string & data) {
    const char * ptr = data.data();
    size_t left = data.size();
    while (left > 0) {
        const ssize_t n = send(fd, ptr, left, 0);
        if (n <= 0) {
            return;
        }
        ptr += n;
        left -= static_cast<size_t>(n);
    }
}

std::string read_request(int fd) {
    std::string req;
    char buf[4096];
    while (req.find("\r\n\r\n") == std::string::npos) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            return req;
        }
        req.append(buf, static_cast<size_t>(n));
        if (req.size() > 1024 * 1024) {
            throw std::runtime_error("request too large");
        }
    }
    const size_t header_end = req.find("\r\n\r\n") + 4;
    size_t content_length = 0;
    const size_t cl = req.find("Content-Length:");
    if (cl != std::string::npos) {
        size_t start = cl + std::strlen("Content-Length:");
        while (start < req.size() && std::isspace(static_cast<unsigned char>(req[start]))) {
            ++start;
        }
        size_t end = start;
        while (end < req.size() && std::isdigit(static_cast<unsigned char>(req[end]))) {
            ++end;
        }
        content_length = static_cast<size_t>(std::stoul(req.substr(start, end - start)));
    }
    while (req.size() < header_end + content_length) {
        const ssize_t n = recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) {
            break;
        }
        req.append(buf, static_cast<size_t>(n));
    }
    return req;
}

void respond_error(int fd, int status, const std::string & message) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " Error\r\n"
        << "Content-Type: text/plain\r\n"
        << "Content-Length: " << message.size() << "\r\n"
        << "Connection: close\r\n\r\n"
        << message;
    send_all(fd, oss.str());
}

void handle_client(int fd, long long queue_wait_ms, const ServerConfig & config, higgs_audio::HiggsPipeline & pipeline) {
    const std::string req = read_request(fd);
    if (req.rfind("POST /generate ", 0) != 0) {
        respond_error(fd, 404, "expected POST /generate\n");
        return;
    }
    const size_t body_pos = req.find("\r\n\r\n");
    const std::string body = body_pos == std::string::npos ? std::string() : req.substr(body_pos + 4);

    higgs_audio::GenerateOptions options;
    options.model_path = config.model_path;
    options.backend = config.backend;
    options.text = json_string(body, "text", "test");
    options.ref_wav = json_string(body, "ref_wav", options.ref_wav);
    options.ref_text = json_string(body, "ref_text", options.ref_text);
    options.system_prompt = json_string(body, "system_prompt", options.system_prompt);
    options.backend = higgs_audio::parse_backend_kind(json_string(body, "backend", higgs_audio::backend_kind_name(options.backend)));
    options.prompt_format = higgs_audio::parse_prompt_format(json_string(body, "prompt_format", higgs_audio::prompt_format_name(options.prompt_format)));
    options.steps = json_int(body, "steps", options.steps);
    options.temperature = json_float(body, "temperature", options.temperature);
    options.top_k = json_int(body, "top_k", options.top_k);
    options.top_p = json_float(body, "top_p", options.top_p);
    options.seed = static_cast<uint32_t>(json_int(body, "seed", static_cast<int>(options.seed)));
    options.stop_on_eoc = json_bool(body, "stop_on_eoc", options.stop_on_eoc);
    if (json_bool(body, "no_stop_on_eoc", false)) {
        options.stop_on_eoc = false;
    }
    options.cancel = &g_stop;
    options.deadline = std::chrono::steady_clock::now() + std::chrono::seconds(config.request_timeout_sec);
    const std::string out_wav = json_string(body, "out_wav", "");
    const std::string out_mp3 = json_string(body, "out_mp3", "");
    const std::string codes_json = json_string(body, "codes_json", "");
    const std::string response_format = json_string(body, "response_format", out_wav.empty() && out_mp3.empty() && codes_json.empty() ? "wav" : "json");
    const auto start = std::chrono::steady_clock::now();
    const higgs_audio::GenerateResult result = pipeline.generate(options);
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    const double audio_seconds = result.sample_rate > 0 ? static_cast<double>(result.waveform.size()) / static_cast<double>(result.sample_rate) : 0.0;
    const std::string scheduler_state = result.scheduler_mode.empty()
        ? (!config.ar_scheduler ? "off" : (result.backend_kv_slot >= 0 ? "kv_slots_round_robin" : "blocked_shared_kv_fallback"))
        : result.scheduler_mode;
    write_codes_json_file(codes_json, result.delayed_codes, result.finalized_codes);

    if (!out_wav.empty()) {
        if (ends_with(out_wav, ".mp3")) {
            higgs_audio::write_mp3_mono(out_wav, result.waveform, result.sample_rate);
        } else {
            higgs_audio::write_wav_mono_16(out_wav, result.waveform, result.sample_rate);
        }
    }
    if (!out_mp3.empty()) {
        higgs_audio::write_mp3_mono(out_mp3, result.waveform, result.sample_rate);
    }
    if (response_format == "json") {
        std::ostringstream payload;
        payload << "{\"ok\":true"
                << ",\"wav_path\":\"" << json_escape(out_wav) << "\""
                << ",\"mp3_path\":\"" << json_escape(out_mp3) << "\""
                << ",\"codes_json\":\"" << json_escape(codes_json) << "\""
                << ",\"sample_rate\":" << result.sample_rate
                << ",\"waveform_samples\":" << result.waveform.size()
                << ",\"audio_seconds\":" << audio_seconds
                << ",\"latency_ms\":" << elapsed
                << ",\"total_wall_ms\":" << result.total_wall_ms
                << ",\"decode_wall_ms\":" << result.reference_ar_wall_ms
                << ",\"reference_cache_wall_ms\":" << result.reference_cache_wall_ms
                << ",\"reference_ar_wall_ms\":" << result.reference_ar_wall_ms
                << ",\"codec_wall_ms\":" << result.codec_wall_ms
                << ",\"audio_head_wall_ms\":" << result.audio_head_wall_ms
                << ",\"audio_head_batch_calls\":" << result.audio_head_batch_calls
                << ",\"audio_head_fallback_calls\":" << result.audio_head_fallback_calls
                << ",\"audio_head_batch_size_avg\":" << result.audio_head_batch_size_avg
                << ",\"cuda_executor_wait_ms\":" << result.cuda_executor_wait_ms
                << ",\"cuda_executor_run_ms\":" << result.cuda_executor_run_ms
                << ",\"cuda_executor_queue_depth\":" << result.cuda_executor_queue_depth
                << ",\"queue_wait_ms\":" << queue_wait_ms
                << ",\"reference_cache\":\"" << json_escape(result.reference_cache_status.empty() ? "none" : result.reference_cache_status) << "\""
                << ",\"scheduler\":\"" << json_escape(scheduler_state) << "\""
                << ",\"scheduler_batch_size\":1"
                << ",\"scheduler_step_count\":" << options.steps
                << ",\"backend_kv_slot\":" << result.backend_kv_slot
                << ",\"delayed_frames\":" << result.delayed_codes.frames
                << ",\"finalized_frames\":" << result.finalized_codes.frames
                << "}\n";
        const std::string json = payload.str();
        std::ostringstream header;
        header << "HTTP/1.1 200 OK\r\n"
               << "Content-Type: application/json\r\n"
               << "Content-Length: " << json.size() << "\r\n"
               << "Connection: close\r\n\r\n";
        send_all(fd, header.str());
        send_all(fd, json);
        return;
    }
    if (response_format == "mp3") {
        const std::vector<uint8_t> mp3 = higgs_audio::encode_mp3_mono(result.waveform, result.sample_rate);
        std::ostringstream header;
        header << "HTTP/1.1 200 OK\r\n"
               << "Content-Type: audio/mpeg\r\n"
               << "X-Higgs-Latency-Ms: " << elapsed << "\r\n"
               << "X-Higgs-Audio-Seconds: " << audio_seconds << "\r\n"
               << "Content-Length: " << mp3.size() << "\r\n"
               << "Connection: close\r\n\r\n";
        send_all(fd, header.str());
        send_all(fd, std::string(reinterpret_cast<const char *>(mp3.data()), mp3.size()));
        return;
    }

    // ponytail: temp file until write_wav_mono_16 grows an ostream overload.
    std::ostringstream tmp;
    tmp << "/tmp/higgs-audio-server-response-" << getpid() << "-"
        << std::hash<std::thread::id>{}(std::this_thread::get_id()) << "-"
        << fd << ".wav";
    const std::string path = tmp.str();
    higgs_audio::write_wav_mono_16(path, result.waveform, result.sample_rate);
    std::ifstream in(path, std::ios::binary);
    const std::string wav((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::remove(path.c_str());

    std::ostringstream header;
    header << "HTTP/1.1 200 OK\r\n"
           << "Content-Type: audio/wav\r\n"
           << "X-Higgs-Latency-Ms: " << elapsed << "\r\n"
           << "X-Higgs-Total-Wall-Ms: " << result.total_wall_ms << "\r\n"
           << "X-Higgs-Decode-Wall-Ms: " << result.reference_ar_wall_ms << "\r\n"
           << "X-Higgs-Reference-Cache-Wall-Ms: " << result.reference_cache_wall_ms << "\r\n"
           << "X-Higgs-Reference-Ar-Wall-Ms: " << result.reference_ar_wall_ms << "\r\n"
           << "X-Higgs-Codec-Wall-Ms: " << result.codec_wall_ms << "\r\n"
           << "X-Higgs-Audio-Head-Wall-Ms: " << result.audio_head_wall_ms << "\r\n"
           << "X-Higgs-Audio-Head-Batch-Calls: " << result.audio_head_batch_calls << "\r\n"
           << "X-Higgs-Audio-Head-Fallback-Calls: " << result.audio_head_fallback_calls << "\r\n"
           << "X-Higgs-Audio-Head-Batch-Size-Avg: " << result.audio_head_batch_size_avg << "\r\n"
           << "X-Higgs-Cuda-Executor-Wait-Ms: " << result.cuda_executor_wait_ms << "\r\n"
           << "X-Higgs-Cuda-Executor-Run-Ms: " << result.cuda_executor_run_ms << "\r\n"
           << "X-Higgs-Cuda-Executor-Queue-Depth: " << result.cuda_executor_queue_depth << "\r\n"
           << "X-Higgs-Queue-Wait-Ms: " << queue_wait_ms << "\r\n"
           << "X-Higgs-Audio-Seconds: " << audio_seconds << "\r\n"
           << "X-Higgs-Reference-Cache: " << (result.reference_cache_status.empty() ? "none" : result.reference_cache_status) << "\r\n"
           << "X-Higgs-Scheduler: " << scheduler_state << "\r\n"
           << "X-Higgs-Backend-Kv-Slot: " << result.backend_kv_slot << "\r\n"
           << "Content-Length: " << wav.size() << "\r\n"
           << "Connection: close\r\n\r\n";
    send_all(fd, header.str());
    send_all(fd, wav);
}

void worker_loop(RequestQueue & queue, const ServerConfig & config, higgs_audio::HiggsPipeline & pipeline) {
    QueuedRequest request;
    while (queue.pop(request)) {
        const int client = request.fd;
        const long long queue_wait_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - request.enqueued_at).count();
        timeval timeout{};
        timeout.tv_sec = config.request_timeout_sec;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
        try {
            handle_client(client, queue_wait_ms, config, pipeline);
        } catch (const std::exception & err) {
            const std::string message = err.what();
            const int status = message == "request timed out" ? 504 : (message == "request cancelled" ? 503 : 500);
            respond_error(client, status, std::string("generate failed: ") + message + "\n");
        }
        close(client);
    }
}

} // namespace

int main(int argc, char ** argv) {
    ServerConfig config;
    config.ar_scheduler = std::getenv("HIGGS_SERVER_AR_SCHEDULER") != nullptr;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            config.model_path = argv[++i];
        } else if (arg == "--backend" && i + 1 < argc) {
            config.backend = higgs_audio::parse_backend_kind(argv[++i]);
        } else if (arg == "--port" && i + 1 < argc) {
            config.port = std::stoi(argv[++i]);
        } else if (arg == "--workers" && i + 1 < argc) {
            config.workers = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--queue-size" && i + 1 < argc) {
            config.queue_size = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--request-timeout-sec" && i + 1 < argc) {
            config.request_timeout_sec = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--ar-scheduler") {
            config.ar_scheduler = true;
        } else {
            std::cerr << "usage: higgs-audio-server [--model PATH] [--backend cpu|cuda] [--port PORT] [--workers N] [--queue-size N] [--request-timeout-sec N] [--ar-scheduler]\n";
            return 1;
        }
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    const int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        std::cerr << "failed to create socket\n";
        return 1;
    }
    g_server_fd = server_fd;
    const int yes = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons(static_cast<uint16_t>(config.port));
    if (bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 || listen(server_fd, 4) != 0) {
        std::cerr << "failed to listen on 127.0.0.1:" << config.port << "\n";
        close(server_fd);
        return 1;
    }
    std::cerr << "higgs-audio-server listening on 127.0.0.1:" << config.port
              << " backend=" << higgs_audio::backend_kind_name(config.backend)
              << " workers=" << config.workers
              << " queue_size=" << config.queue_size
              << " request_timeout_sec=" << config.request_timeout_sec
              << " ar_scheduler=" << (config.ar_scheduler ? "blocked_shared_kv_fallback" : "off") << "\n";
    higgs_audio::HiggsPipeline pipeline(config.model_path, config.backend);
    RequestQueue queue(static_cast<size_t>(config.queue_size));
    std::vector<std::thread> workers;
    for (int i = 0; i < config.workers; ++i) {
        workers.emplace_back(worker_loop, std::ref(queue), std::cref(config), std::ref(pipeline));
    }
    while (!g_stop) {
        const int client = accept(server_fd, nullptr, nullptr);
        if (client < 0) {
            if (g_stop) {
                break;
            }
            continue;
        }
        if (!queue.push(client)) {
            respond_error(client, 503, "server queue full\n");
            close(client);
        }
    }
    queue.close();
    for (std::thread & worker : workers) {
        worker.join();
    }
    g_server_fd = -1;
    close(server_fd);
    return 0;
}
