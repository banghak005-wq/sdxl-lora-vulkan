// sdxl_lora_server: CLI 학습기(sdxl_lora)를 자식 프로세스로 돌리는 가벼운 HTTP 작업 관리자
//
// 한 번에 작업 하나. 학습은 별도 프로세스라 끝나면 GPU 메모리가 확실히 반환되고, 서버 자체는 ggml에 의존하지 않는다.
// 요청에 들어오는 경로는 --root 안으로 제한한다. 인증이 없으므로 기본은 127.0.0.1에만 연다.
//
// 사용: sdxl_lora_server --root <작업 폴더> [--cli <sdxl_lora 경로>] [--host 127.0.0.1] [--port 8090]
//
// API
//   GET  /              상태 페이지 (5초마다 새로고침)
//   GET  /api/status    {"state": idle|running|done|failed|stopped, "args": [...], "step": n, "steps": n,
//                        "loss": f, "ema": f, "sec_per_step": f, "eta_sec": f, "exit_code": n, "log_tail": [...]}
//   POST /api/train     JSON {"data": "...", "out": "...", "steps": 200, "lr": 1e-4, "rank": 8, "alpha": 8,
//                        "seed": 42, "save_every": 0, "ckpt": true, "native": "all", "resume": "...",
//                        "model": "...", "fp16": false, "t_min": 0, "t_max": 999}
//   POST /api/stop      실행 중인 작업에 SIGTERM
//   GET  /api/log       전체 로그 (text/plain)
//   GET  /files/<경로>  --root 안의 파일 받기 (LoRA, 이미지 등)

#include "httplib.h"
#include "json.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

struct Job {
    std::string              state = "idle";
    std::vector<std::string> args;
    pid_t                    pid       = -1;
    int                      exit_code = 0;
    int                      steps     = 0;
    std::string              log_path;
    std::chrono::steady_clock::time_point started;
};

static std::mutex g_mu;
static Job        g_job;
static fs::path   g_root;
static std::string g_cli;

// --root 밖으로 나가는 경로 거부. 상대 경로는 root 기준.
static bool resolve_in_root(const std::string & p, fs::path & out) {
    if (p.empty()) return false;
    fs::path cand = fs::path(p).is_absolute() ? fs::path(p) : g_root / p;
    cand          = fs::weakly_canonical(cand);
    auto rel      = cand.lexically_relative(g_root);
    if (rel.empty() || *rel.begin() == "..") return false;
    out = cand;
    return true;
}

static std::vector<std::string> tail_lines(const std::string & path, size_t n) {
    std::ifstream            f(path);
    std::vector<std::string> all;
    std::string              line;
    while (std::getline(f, line)) all.push_back(line);
    if (all.size() > n) all.erase(all.begin(), all.end() - (long) n);
    return all;
}

static void reap_loop() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::lock_guard<std::mutex> lk(g_mu);
        if ((g_job.state != "running" && g_job.state != "stopping") || g_job.pid <= 0) continue;
        int         st = 0;
        const pid_t r  = waitpid(g_job.pid, &st, WNOHANG);
        if (r == g_job.pid) {
            if (WIFEXITED(st)) {
                g_job.exit_code = WEXITSTATUS(st);
                g_job.state     = g_job.state == "stopping" ? "stopped" : g_job.exit_code == 0 ? "done" : "failed";
            } else {
                g_job.exit_code = 128 + WTERMSIG(st);
                g_job.state     = g_job.state == "stopping" ? "stopped" : "failed";
            }
            g_job.pid = -1;
        }
    }
}

static json status_json() {
    std::lock_guard<std::mutex> lk(g_mu);
    json j = {{"state", g_job.state}, {"args", g_job.args}, {"exit_code", g_job.exit_code}, {"steps", g_job.steps}};
    if (g_job.log_path.empty()) return j;
    auto lines = tail_lines(g_job.log_path, 400);
    // 학습기 출력: "step   12  t=558  loss 0.04619  ema 0.04435  12.41s ..."
    static const std::regex re(R"(^step\s+(\d+)\s+t=\s*\d+\s+loss\s+([0-9.eE+-]+|nan|inf)\s+ema\s+([0-9.eE+-]+)\s+([0-9.]+)s)");
    for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
        std::smatch m;
        if (std::regex_search(*it, m, re)) {
            const int    step = std::stoi(m[1]);
            const double sps  = std::stod(m[4]);
            j["step"]         = step;
            j["loss"]         = m[2].str();
            j["ema"]          = std::stod(m[3]);
            j["sec_per_step"] = sps;
            if (g_job.steps > step) j["eta_sec"] = sps * (g_job.steps - step);
            break;
        }
    }
    std::vector<std::string> t(lines.size() > 20 ? lines.end() - 20 : lines.begin(), lines.end());
    j["log_tail"] = t;
    return j;
}

static std::string start_job(const json & req, int & http_code) {
    auto num = [&](const char * k, double def) { return req.contains(k) ? req[k].get<double>() : def; };
    fs::path data, out;
    if (!req.contains("data") || !resolve_in_root(req["data"].get<std::string>(), data) || !fs::is_directory(data)) {
        http_code = 400;
        return "data: --root 안의 캐시 폴더가 필요함";
    }
    if (!resolve_in_root(req.value("out", std::string("out/lora.safetensors")), out)) {
        http_code = 400;
        return "out: --root 안이어야 함";
    }
    const int steps = (int) num("steps", 200);
    std::vector<std::string> args = {g_cli, "--train", data.string(), "--out", out.string(),
                                     "--steps", std::to_string(steps)};
    char buf[64];
    auto add_num = [&](const char * flag, const char * key, double def, const char * fmt) {
        snprintf(buf, sizeof(buf), fmt, num(key, def));
        args.push_back(flag);
        args.push_back(buf);
    };
    add_num("--lr", "lr", 1e-4, "%g");
    add_num("--rank", "rank", 8, "%.0f");
    add_num("--alpha", "alpha", 8, "%g");
    add_num("--seed", "seed", 42, "%.0f");
    add_num("--save-every", "save_every", 0, "%.0f");
    add_num("--t-min", "t_min", 0, "%.0f");
    add_num("--t-max", "t_max", 999, "%.0f");
    if (req.value("ckpt", true)) args.push_back("--ckpt");
    const std::string native = req.value("native", std::string("all"));
    if (!native.empty()) {
        args.push_back("--native");
        args.push_back(native);
    }
    for (const char * k : {"resume", "model"}) {
        if (!req.contains(k)) continue;
        fs::path p;
        if (!resolve_in_root(req[k].get<std::string>(), p) || !fs::exists(p)) {
            http_code = 400;
            return std::string(k) + ": --root 안의 존재하는 파일이어야 함";
        }
        args.push_back(std::string("--") + k);
        args.push_back(p.string());
    }
    const bool fp16 = req.value("fp16", false);

    std::lock_guard<std::mutex> lk(g_mu);
    if (g_job.state == "running" || g_job.state == "stopping") {
        http_code = 409;
        return "이미 실행 중인 작업이 있음";
    }
    fs::create_directories(g_root / "logs");
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char       ts[32];
    strftime(ts, sizeof(ts), "%Y%m%d-%H%M%S", localtime(&now));
    const std::string log_path = (g_root / "logs" / (std::string("train-") + ts + ".log")).string();

    const pid_t pid = fork();
    if (pid < 0) {
        http_code = 500;
        return "fork 실패";
    }
    if (pid == 0) {
        const int fd = open(log_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        dup2(fd, 1);
        dup2(fd, 2);
        close(fd);
        if (chdir(g_root.c_str()) != 0) _exit(126);
        if (!fp16) setenv("GGML_VK_DISABLE_F16", "1", 1);  // 기본은 검증된 f32 경로
        std::vector<char *> av;
        for (auto & a : args) av.push_back(const_cast<char *>(a.c_str()));
        av.push_back(nullptr);
        execv(av[0], av.data());
        _exit(127);
    }
    g_job           = Job{};
    g_job.state     = "running";
    g_job.args      = args;
    g_job.pid       = pid;
    g_job.steps     = steps;
    g_job.log_path  = log_path;
    g_job.started   = std::chrono::steady_clock::now();
    http_code       = 200;
    return "";
}

static const char * PAGE = R"(<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>sdxl_lora_server</title><style>body{font-family:sans-serif;margin:16px;background:#111;color:#eee}
pre{background:#222;padding:8px;overflow:auto;font-size:.8em}</style>
<h1>SDXL LoRA 학습 (Vulkan)</h1><div id=s>...</div><pre id=l></pre>
<script>
async function tick(){const j=await (await fetch('/api/status')).json();
let t='상태: <b>'+j.state+'</b>';
if(j.step!==undefined)t+=` · step ${j.step}/${j.steps} · loss ${j.loss} · ema ${j.ema.toFixed(5)} · ${j.sec_per_step.toFixed(2)} s/step`;
if(j.eta_sec!==undefined)t+=` · 남은 시간 약 ${Math.round(j.eta_sec/60)}분`;
document.getElementById('s').innerHTML=t;document.getElementById('l').textContent=(j.log_tail||[]).join('\n');}
tick();setInterval(tick,5000);
</script>)";

int main(int argc, char ** argv) {
    std::string host = "127.0.0.1";
    int         port = 8090;
    std::string root;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "값 없음: %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--root") root = next();
        else if (a == "--cli") g_cli = next();
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else {
            fprintf(stderr, "사용: %s --root <작업 폴더> [--cli <sdxl_lora>] [--host 127.0.0.1] [--port 8090]\n", argv[0]);
            return 2;
        }
    }
    if (root.empty()) {
        fprintf(stderr, "--root가 필요함\n");
        return 2;
    }
    g_root = fs::weakly_canonical(fs::absolute(root));
    if (g_cli.empty()) g_cli = (fs::canonical("/proc/self/exe").parent_path() / "sdxl_lora").string();
    if (!fs::exists(g_cli)) {
        fprintf(stderr, "CLI 학습기를 찾을 수 없음: %s (--cli로 지정)\n", g_cli.c_str());
        return 2;
    }
    signal(SIGPIPE, SIG_IGN);
    std::thread(reap_loop).detach();

    httplib::Server svr;
    svr.Get("/", [](const httplib::Request &, httplib::Response & res) { res.set_content(PAGE, "text/html; charset=utf-8"); });
    svr.Get("/api/status", [](const httplib::Request &, httplib::Response & res) {
        res.set_content(status_json().dump(), "application/json");
    });
    svr.Get("/api/log", [](const httplib::Request &, httplib::Response & res) {
        std::string p;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            p = g_job.log_path;
        }
        std::ifstream     f(p);
        std::stringstream ss;
        ss << f.rdbuf();
        res.set_content(ss.str(), "text/plain; charset=utf-8");
    });
    svr.Post("/api/train", [](const httplib::Request & req, httplib::Response & res) {
        json body;
        try {
            body = json::parse(req.body);
        } catch (...) {
            res.status = 400;
            res.set_content(R"({"error":"JSON 파싱 실패"})", "application/json");
            return;
        }
        int         code = 500;
        std::string err;
        try {
            err = start_job(body, code);
        } catch (const std::exception & e) {
            code = 400;
            err  = e.what();
        }
        res.status = code;
        res.set_content(err.empty() ? status_json().dump() : json{{"error", err}}.dump(), "application/json");
    });
    svr.Post("/api/stop", [](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mu);
        if (g_job.state == "running" && g_job.pid > 0) {
            kill(g_job.pid, SIGTERM);
            g_job.state = "stopping";
            res.set_content(R"({"ok":true})", "application/json");
        } else {
            res.status = 409;
            res.set_content(R"({"error":"실행 중인 작업 없음"})", "application/json");
        }
    });
    svr.Get(R"(/files/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        fs::path p;
        if (!resolve_in_root(req.matches[1].str(), p) || !fs::is_regular_file(p)) {
            res.status = 404;
            return;
        }
        std::ifstream     f(p, std::ios::binary);
        std::stringstream ss;
        ss << f.rdbuf();
        const std::string ext = p.extension().string();
        const char * ct = ext == ".png" ? "image/png" : ext == ".jpg" || ext == ".jpeg" ? "image/jpeg"
                        : ext == ".json" ? "application/json" : ext == ".txt" || ext == ".log" ? "text/plain; charset=utf-8"
                        : "application/octet-stream";
        res.set_content(ss.str(), ct);
    });

    printf("sdxl_lora_server: http://%s:%d  (root %s, cli %s)\n", host.c_str(), port, g_root.c_str(), g_cli.c_str());
    if (host != "127.0.0.1" && host != "localhost") printf("주의: 인증 없음. 외부에 열면 누구나 학습을 시작/중지할 수 있음\n");
    fflush(stdout);
    return svr.listen(host, port) ? 0 : 1;
}
