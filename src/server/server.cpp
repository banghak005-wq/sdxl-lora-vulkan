// sdxl_lora_server (v0.4): GPU 작업 큐 웹서버
//
// 작업(캐시 만들기 / LoRA 학습 / 이미지 생성)을 여러 개 넣어 두면 한 번에 하나씩 차례로 실행한다.
// GPU 메모리 때문에 동시 실행은 하지 않는다. 각 작업은 별도 프로세스 그룹으로 실행되므로 끝나거나 중지되면
// GPU 메모리가 확실히 반환되고, 중지 시 자식(guard → 학습기/sd-cli)까지 한꺼번에 종료된다.
// 큐는 <root>/jobs/queue.json에 저장되어 서버를 다시 켜도 대기 작업이 남는다.
// 실행에 필요한 것: 이 실행 파일, sdxl_lora(학습), sd.cpp sd-cli(캐시·생성). 파이썬 등 다른 런타임은 필요 없다.
// (tools/guard.sh는 선택: AMD GPU 메모리 감시. bash, awk, free 사용)
//
// 사용: sdxl_lora_server --root <작업 폴더> [--cli <sdxl_lora>] [--sd-cli <sd-cli>] [--tools <tools 폴더>]
//                        [--model <ckpt>] [--vae <vae>] [--sd-args "<생성 공통 인자>"]
//                        [--guard auto|on|off] [--guard-limit-mib 11264] [--token <접속 토큰>|auto] [--threads N]
//                        [--ui builtin|off|<폴더>] [--cors <origin>]
//
// 웹 화면: --ui builtin(기본) = 내장 페이지, off = API만, <폴더> = 직접 만든 프론트(index.html 등)를 /에 서빙.
//          --cors는 프론트를 다른 주소에서 띄울 때 (예: --cors http://localhost:5173, 또는 "*")
//                        [--host 127.0.0.1] [--port 8090]
//
// 인증: --token auto면 켤 때마다 임시 토큰을 만들어 출력한다 (웹 첫 화면에서 입력해야 들어갈 수 있음).
// --token을 주면 모든 /api, /files 요청에 토큰이 필요하다 (헤더 "Authorization: Bearer <토큰>",
//       쿠키 sdxl_token=<토큰>, 또는 ?token=<토큰>). 외부(터널, 포트포워딩)에 열 때는 반드시 켤 것.
//
// API (JSON)
//   GET  /api/config                     서버 설정 요약 (sd-cli 유무, guard, 인증 여부)
//   GET  /api/jobs                       작업 목록 (진행률 포함)
//   POST /api/jobs                       작업 추가 {"type": "cache"|"train"|"generate", ...} → {"id": n}
//   GET  /api/jobs/<id>                  작업 상세 (로그 끝부분, 결과 파일)
//   GET  /api/jobs/<id>/log              전체 로그 (text/plain)
//   POST /api/jobs/<id>/cancel           대기 → 취소. 실행 중 → SIGTERM, 10초 안에 안 끝나면 SIGKILL (프로세스 그룹 전체)
//   POST /api/jobs/<id>/kill             실행 중 → 즉시 SIGKILL (프로세스 그룹 전체). 대기 → 취소
//   POST /api/jobs/<id>/move             {"dir": "up"|"down"|"top"} 대기 순서 바꾸기
//   POST /api/jobs/<id>/remove           끝난 작업을 목록에서 지우기. {"delete_files": true}면 그 작업이 만든 파일도 삭제
//   POST /api/jobs/remove-many           {"ids": [..], "delete_files": bool} 끝난 작업 여러 개 한꺼번에 (대기·실행 중은 건너뜀)
//   POST /api/queue/cancel-all           대기 작업 전부 취소. {"running": true}면 실행 중도 중지, {"force": true}면 즉시 SIGKILL
//   POST /api/queue/pause | /resume      새 작업 시작 멈춤 / 재개 (실행 중인 작업은 계속)
//   GET  /api/list                       고를 수 있는 것들: LoRA 파일, 캐시 폴더, 이미지 폴더 (root 안)
//   GET  /files/<경로>                   root 안의 파일 받기
//   v0.3 호환: POST /api/train (= 학습 작업 추가), GET /api/status, POST /api/stop, GET /api/log

#include "httplib.h"
#include "json.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using json   = nlohmann::json;

// ---------- 설정 ----------
struct Config {
    fs::path    root;
    std::string cli, sd_cli, tools;
    std::string model = "models/sd_xl_base_1.0.safetensors", vae = "models/sdxl_vae.safetensors";
    std::string sd_args = "--vae-tiling --flash-attn auto,vae=off --lora-apply-mode at_runtime";
    std::string guard   = "auto";
    bool        guard_on        = false;
    int         guard_limit_mib = 11264;
    int         threads         = 0;  // 캐시 작업 sd-cli -t (0 = CPU 수)
    std::string ui              = "builtin";  // builtin | off | <정적 파일 폴더>
    std::string cors;                          // 비우면 끔. "*" 또는 "http(s)://주소"
    std::string token;
};
static Config g_cfg;

// ---------- 작업 ----------
struct Job {
    int         id = 0;
    std::string type, state = "queued";  // queued running done failed cancelled stopped interrupted
    json        params = json::object();
    std::string error;
    double      created = 0, started = 0, finished = 0;  // unix 초 (밀리초까지)
    int         exit_code = 0;
    pid_t       pid       = -1;  // 실행 중일 때 = 프로세스 그룹 id
    std::string stop_mode;       // "" | "term" | "kill"
    std::chrono::steady_clock::time_point term_at{};
};

static std::mutex       g_mu;
static std::vector<Job> g_jobs;  // 순서 = 큐 순서 (대기 작업끼리의 순서가 실행 순서)
static int              g_next_id = 1;
static bool             g_paused  = false;

static double now_unix() {
    return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static fs::path jobs_dir() { return g_cfg.root / "jobs"; }
static fs::path job_dir(int id) { return jobs_dir() / std::to_string(id); }

// root 밖으로 나가는 경로 거부. 상대 경로는 root 기준. 존재하지 않아도 됨.
static bool resolve_in_root(const std::string & p, fs::path & out) {
    if (p.empty()) return false;
    fs::path cand = fs::path(p).is_absolute() ? fs::path(p) : g_cfg.root / p;
    cand          = fs::weakly_canonical(cand);
    auto rel      = cand.lexically_relative(g_cfg.root);
    if (rel.empty() || *rel.begin() == "..") return false;
    out = cand;
    return true;
}
static std::string rel_to_root(const fs::path & p) { return fs::weakly_canonical(p).lexically_relative(g_cfg.root).string(); }

static json job_to_json(const Job & j) {
    return {{"id", j.id},           {"type", j.type},         {"state", j.state},       {"params", j.params},
            {"error", j.error},     {"created", j.created},   {"started", j.started},   {"finished", j.finished},
            {"exit_code", j.exit_code}, {"pid", j.pid}};
}

static void save_locked() {
    json q = {{"next_id", g_next_id}, {"paused", g_paused}, {"jobs", json::array()}};
    for (auto & j : g_jobs) q["jobs"].push_back(job_to_json(j));
    fs::create_directories(jobs_dir());
    const fs::path tmp = jobs_dir() / "queue.json.tmp";
    {
        std::ofstream f(tmp);
        f << q.dump(1);
    }
    fs::rename(tmp, jobs_dir() / "queue.json");
}

static Job * find_locked(int id) {
    for (auto & j : g_jobs)
        if (j.id == id) return &j;
    return nullptr;
}

// ---------- 인자 만들기 (작업 시작 직전에. 앞 작업이 만들 파일을 뒤 작업이 쓸 수 있게) ----------
static double num(const json & p, const char * k, double def) {
    if (!p.contains(k)) return def;
    if (p[k].is_number()) return p[k].get<double>();
    if (p[k].is_string()) return std::stod(p[k].get<std::string>());
    throw std::runtime_error(std::string(k) + ": 숫자가 아님");
}
static std::string str(const json & p, const char * k, const std::string & def = "") {
    return p.contains(k) && p[k].is_string() ? p[k].get<std::string>() : def;
}
static std::string fmt(const char * f, double v) {
    char b[64];
    snprintf(b, sizeof(b), f, v);
    return b;
}
static std::string need_path(const json & p, const char * k, const std::string & def, bool must_exist) {
    fs::path out;
    const std::string v = str(p, k, def);
    if (!resolve_in_root(v, out)) throw std::runtime_error(std::string(k) + ": --root 안의 경로여야 함 (" + v + ")");
    if (must_exist && !fs::exists(out)) throw std::runtime_error(std::string(k) + ": 없음 (" + v + ")");
    return out.string();
}
static std::string root_path(const std::string & p) {  // 설정 경로 (상대면 root 기준)
    return fs::path(p).is_absolute() ? p : (g_cfg.root / p).string();
}
static void check_range(const char * k, double v, double lo, double hi) {
    if (!(v >= lo && v <= hi)) throw std::runtime_error(std::string(k) + ": " + fmt("%g", lo) + "~" + fmt("%g", hi) + " 범위여야 함");
}
static std::vector<std::string> split_ws(const std::string & s) {
    std::istringstream is(s);
    std::vector<std::string> v;
    for (std::string w; is >> w;) v.push_back(w);
    return v;
}

// 작업 종류별 기본 출력 이름
static std::string safe_name(const std::string & s) {
    std::string o;
    for (char c : s) o += (isalnum((unsigned char) c) || c == '-' || c == '_' || c == '.') ? c : '_';
    return o.empty() ? "job" : o;
}

// 학습 데이터 캐시용 sd.cpp DGL 스크립트 작성 (tools/make_cache.py를 C++로 옮긴 것. 같은 스크립트를 만든다)
// 이미지 → 가운데 정사각형 자르기 → size로 bicubic+antialias 축소 → VAE 잠재값, 캡션 → crossattn/vector (.npy)
static std::string write_cache_script(const std::string & src, const std::string & dst, int size, const std::string & caption,
                                      const std::string & prefix, bool vae_cpu) {
    auto q = [](const std::string & s) { return json(s).dump(-1, ' ', true); };  // JSON 문자열 리터럴 (비ASCII는 \uXXXX, 파이썬 json.dumps와 같음)
    static const std::set<std::string> exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"};
    std::vector<fs::path> files;
    for (auto & e : fs::directory_iterator(src)) {
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (e.is_regular_file() && exts.count(ext)) files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (files.empty()) throw std::runtime_error("이미지 없음: " + src);
    fs::create_directories(dst);
    struct Item { std::string path, cap, name; };
    std::vector<Item> items;
    std::set<std::string> used;
    for (auto & f : files) {
        std::string cap = caption;
        const fs::path txt = f.parent_path() / (f.stem().string() + ".txt");
        if (fs::exists(txt)) {
            std::ifstream t(txt);
            std::stringstream ss;
            ss << t.rdbuf();
            cap = ss.str();
            const auto b = cap.find_first_not_of(" \t\r\n"), e = cap.find_last_not_of(" \t\r\n");
            cap = b == std::string::npos ? "" : cap.substr(b, e - b + 1);
        }
        std::string name;
        for (char c : f.stem().string()) name += (isalnum((unsigned char) c) || c == '-' || c == '_') ? c : '_';
        std::string uniq = name;
        for (int k = 2; used.count(uniq); k++) uniq = name + "_" + std::to_string(k);  // 비ASCII 이름이 같은 _로 겹칠 때
        used.insert(uniq);
        items.push_back({f.string(), prefix + cap, uniq});
    }
    std::ostringstream o;
    o << "# version: 0.0.0\n# sdxl_lora_server가 생성 (손으로 고치지 말 것)\n\n"
      << "fn prep(path, size) {\n"
      << "    img = image::load(path)\n"
      << "    w = tensor::shape(img)[1]\n"
      << "    h = tensor::shape(img)[2]\n"
      << "    side = math::min(w, h)\n"
      << "    x0 = math::floor(math::div(math::sub(w, side), 2))\n"
      << "    y0 = math::floor(math::div(math::sub(h, side), 2))\n"
      << "    sq = tensor::crop(img, axes = [1, 2], start = [x0, y0], size = [side, side])\n"
      << "    return tensor::interpolate(sq, axes = [1, 2], size = [size, size], mode = \"bicubic\", antialias = true)\n"
      << "}\n\n"
      << "fn main() {\n"
      << "    text = load_text_encoder(" << q(root_path(g_cfg.model)) << ", family = \"sdxl\", backend = \"cpu\")\n"
      << "    vae = load_vae(" << q(root_path(g_cfg.vae)) << ", family = \"sdxl\", conv_scale = 0.03125, flash_attn_mode = \"off\""
      << (vae_cpu ? ", backend = \"cpu\"" : "") << ")\n"
      << "    rng = make_rng(\"cuda\", seed = 0)\n";
    for (size_t i = 0; i < items.size(); i++) {
        const std::string out = (fs::path(dst) / items[i].name).string();
        o << "    px" << i << " = prep(" << q(items[i].path) << ", " << size << ")\n"
          << "    image::save(" << q(out + ".check.png") << ", px" << i << ")\n"
          << "    lat" << i << " = vae_to_unet_latents(vae, vae_encode(vae, image::to_f32(px" << i
          << ", channels = 3), rng = rng, posterior = \"mode\"))\n"
          << "    tensor::save(" << q(out + ".latent.npy") << ", lat" << i << ")\n"
          << "    c" << i << " = encode_prompt(text, " << q(items[i].cap) << ", width = " << size << ", height = " << size << ")\n"
          << "    tensor::save(" << q(out + ".crossattn.npy") << ", c" << i << ".crossattn)\n"
          << "    tensor::save(" << q(out + ".vector.npy") << ", c" << i << ".vector)\n";
    }
    o << "    return;\n}\n";
    const std::string script = (fs::path(dst) / "cache.dgl").string();
    std::ofstream(script) << o.str();
    std::ofstream cf(fs::path(dst) / "captions.txt");
    for (auto & it : items) cf << it.name << "\t" << it.path << "\t" << it.cap << "\n";
    return script;
}

static std::vector<std::string> build_args(const Job & j, std::vector<std::pair<std::string, std::string>> & env) {
    const json & p = j.params;
    std::vector<std::string> a;
    if (j.type == "train") {
        const std::string data = need_path(p, "data", "", true);
        if (!fs::is_directory(data)) throw std::runtime_error("data: 캐시 폴더가 아님");
        const std::string out = need_path(p, "out", "out/" + safe_name(str(p, "name", fs::path(data).filename().string())) + ".safetensors", false);
        fs::create_directories(fs::path(out).parent_path());
        const double steps = num(p, "steps", 200);
        check_range("steps", steps, 1, 100000);
        a = {g_cfg.cli, "--train", data, "--out", out, "--steps", fmt("%.0f", steps)};
        a.insert(a.end(), {"--lr", fmt("%g", num(p, "lr", 1e-4)), "--rank", fmt("%.0f", num(p, "rank", 8)), "--alpha",
                           fmt("%g", num(p, "alpha", 8)), "--conv-rank", fmt("%.0f", num(p, "conv_rank", 0)), "--seed",
                           fmt("%.0f", num(p, "seed", 42)), "--save-every", fmt("%.0f", num(p, "save_every", 0)), "--t-min",
                           fmt("%.0f", num(p, "t_min", 0)), "--t-max", fmt("%.0f", num(p, "t_max", 999))});
        if (p.contains("conv_alpha")) a.insert(a.end(), {"--conv-alpha", fmt("%g", num(p, "conv_alpha", 0))});
        if (p.value("ckpt", true)) a.push_back("--ckpt");
        const std::string native = str(p, "native", "all");
        if (!native.empty()) a.insert(a.end(), {"--native", native});
        if (p.contains("resume") && !str(p, "resume").empty()) a.insert(a.end(), {"--resume", need_path(p, "resume", "", true)});
        a.insert(a.end(), {"--model", p.contains("model") ? need_path(p, "model", "", true) : root_path(g_cfg.model)});
        if (p.contains("loss_scale")) {
            const auto & v = p["loss_scale"];
            if (v.is_string() && v.get<std::string>() == "dynamic") a.insert(a.end(), {"--loss-scale", "dynamic"});
            else if (v.is_number() && v.get<double>() > 0) a.insert(a.end(), {"--loss-scale", fmt("%g", v.get<double>())});
            else throw std::runtime_error("loss_scale: \"dynamic\" 또는 양수");
        }
        if (!p.value("fp16", true)) env.push_back({"GGML_VK_DISABLE_F16", "1"});
    } else if (j.type == "generate") {
        if (g_cfg.sd_cli.empty()) throw std::runtime_error("서버에 --sd-cli가 설정되지 않아 생성을 할 수 없음");
        std::string prompt = str(p, "prompt");
        if (prompt.empty()) throw std::runtime_error("prompt가 필요함");
        if (prompt.find("<lora:") != std::string::npos) throw std::runtime_error("prompt에 <lora:...>를 직접 쓰지 말고 lora 항목을 쓸 것");
        const double w = num(p, "width", 1024), h = num(p, "height", 1024), st = num(p, "steps", 20), cfg = num(p, "cfg", 7);
        check_range("width", w, 256, 2048);
        check_range("height", h, 256, 2048);
        if (((int) w) % 64 || ((int) h) % 64) throw std::runtime_error("width/height는 64의 배수");
        check_range("steps", st, 1, 150);
        check_range("cfg", cfg, 0, 30);
        std::string lora_dir;
        if (!str(p, "lora").empty()) {
            const fs::path lf = need_path(p, "lora", "", true);
            const std::string stem = lf.stem().string();
            if (lf.extension() != ".safetensors" || stem.find_first_of(":<>") != std::string::npos)
                throw std::runtime_error("lora: .safetensors 파일이어야 함");
            const double wt = num(p, "lora_weight", 1.0);
            check_range("lora_weight", wt, -4, 4);
            prompt += "<lora:" + stem + ":" + fmt("%g", wt) + ">";
            lora_dir = lf.parent_path().string();
        }
        const std::string out = (job_dir(j.id) / "image.png").string();
        a = {g_cfg.sd_cli, "-m", root_path(g_cfg.model), "--vae", root_path(g_cfg.vae), "-p", prompt, "-W", fmt("%.0f", w), "-H",
             fmt("%.0f", h), "--steps", fmt("%.0f", st), "--cfg-scale", fmt("%g", cfg), "--seed", fmt("%.0f", num(p, "seed", 42)),
             "-o", out};
        if (!str(p, "negative").empty()) a.insert(a.end(), {"-n", str(p, "negative")});
        if (!lora_dir.empty()) a.insert(a.end(), {"--lora-model-dir", lora_dir});
        for (auto & x : split_ws(g_cfg.sd_args)) a.push_back(x);
    } else if (j.type == "cache") {
        if (g_cfg.sd_cli.empty()) throw std::runtime_error("서버에 --sd-cli가 설정되지 않아 캐시를 만들 수 없음");
        const std::string src = need_path(p, "src", "", true);
        if (!fs::is_directory(src)) throw std::runtime_error("src: 이미지 폴더가 아님");
        const double size = num(p, "size", 512);
        check_range("size", size, 256, 1536);
        if (((int) size) % 64) throw std::runtime_error("size는 64의 배수");
        const std::string dst = need_path(p, "dst", "cache/" + safe_name(fs::path(src).filename().string()) + "_" + fmt("%.0f", size), false);
        const std::string script = write_cache_script(src, dst, (int) size, str(p, "caption"), str(p, "prefix"), str(p, "vae_backend") == "cpu");
        a = {g_cfg.sd_cli, "-M", "dgl", "--script", script, "-t", std::to_string(g_cfg.threads > 0 ? (unsigned) g_cfg.threads : std::max(1u, std::thread::hardware_concurrency()))};
    } else {
        throw std::runtime_error("type: cache | train | generate");
    }
    if (g_cfg.guard_on) {
        a.insert(a.begin(), {"bash", (fs::path(g_cfg.tools) / "guard.sh").string(), (job_dir(j.id) / "mem.log").string()});
        env.push_back({"GUARD_LIMIT_MIB", std::to_string(g_cfg.guard_limit_mib)});
    }
    return a;
}

// 작업 추가 전 빠른 검사 (경로가 root 안인지, 숫자 범위). 존재 여부는 시작할 때 본다.
static void validate(const std::string & type, const json & p) {
    Job tmp;
    tmp.type   = type;
    tmp.params = p;
    tmp.id     = 0;
    // build_args는 존재하는 경로를 요구하므로 여기서는 형식만 확인
    auto chk = [&](const char * k) {
        fs::path o;
        if (p.contains(k) && !str(p, k).empty() && !resolve_in_root(str(p, k), o))
            throw std::runtime_error(std::string(k) + ": --root 안의 경로여야 함");
    };
    if (type == "train") {
        if (str(p, "data").empty()) throw std::runtime_error("data(캐시 폴더)가 필요함");
        for (auto k : {"data", "out", "resume", "model"}) chk(k);
    } else if (type == "generate") {
        if (str(p, "prompt").empty()) throw std::runtime_error("prompt가 필요함");
        if (g_cfg.sd_cli.empty()) throw std::runtime_error("서버에 --sd-cli가 설정되지 않음");
        chk("lora");
    } else if (type == "cache") {
        if (str(p, "src").empty()) throw std::runtime_error("src(이미지 폴더)가 필요함");
        if (g_cfg.sd_cli.empty()) throw std::runtime_error("서버에 --sd-cli가 설정되지 않음");
        for (auto k : {"src", "dst"}) chk(k);
    } else {
        throw std::runtime_error("type: cache | train | generate");
    }
}

// ---------- 실행 ----------
static void start_locked(Job & j) {
    fs::create_directories(job_dir(j.id));
    const std::string log = (job_dir(j.id) / "log.txt").string();
    std::vector<std::pair<std::string, std::string>> env;
    std::vector<std::string> args;
    try {
        args = build_args(j, env);
    } catch (const std::exception & e) {
        j.state    = "failed";
        j.error    = e.what();
        j.started  = j.finished = now_unix();
        std::ofstream(log) << "시작 실패: " << e.what() << "\n";
        return;
    }
    {
        std::ofstream f(log);
        f << "# 작업 " << j.id << " (" << j.type << ") 명령:";
        for (auto & x : args) f << " '" << x << "'";
        f << "\n";
    }
    const pid_t pid = fork();
    if (pid < 0) {
        j.state = "failed";
        j.error = "fork 실패";
        return;
    }
    if (pid == 0) {
        setpgid(0, 0);  // 작업 = 프로세스 그룹. 중지할 때 그룹 전체에 신호
        const int fd = open(log.c_str(), O_WRONLY | O_APPEND);
        dup2(fd, 1);
        dup2(fd, 2);
        for (int x = 3; x < 1024; x++) close(x);  // 서버 소켓 등을 자식에게 넘기지 않음
        if (chdir(g_cfg.root.c_str()) != 0) _exit(126);
        for (auto & [k, v] : env) setenv(k.c_str(), v.c_str(), 1);
        std::vector<char *> av;
        for (auto & x : args) av.push_back(const_cast<char *>(x.c_str()));
        av.push_back(nullptr);
        execvp(av[0], av.data());
        fprintf(stderr, "실행 실패: %s: %s\n", av[0], strerror(errno));
        _exit(127);
    }
    setpgid(pid, pid);  // 경쟁 상태 대비 (부모에서도)
    j.pid     = pid;
    j.state   = "running";
    j.started = now_unix();
}

static void signal_job_locked(Job & j, bool force) {
    if (j.state == "queued") {
        j.state    = "cancelled";
        j.finished = now_unix();
        return;
    }
    if (j.state != "running" || j.pid <= 0) return;
    if (force) {
        kill(-j.pid, SIGKILL);
        j.stop_mode = "kill";
    } else if (j.stop_mode.empty()) {
        kill(-j.pid, SIGTERM);
        j.stop_mode = "term";
        j.term_at   = std::chrono::steady_clock::now();
    }
}

static void worker_loop() {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        std::lock_guard<std::mutex> lk(g_mu);
        bool  changed = false;
        Job * run     = nullptr;
        for (auto & j : g_jobs)
            if (j.state == "running") run = &j;
        if (run) {
            int st = 0;
            const pid_t r = waitpid(run->pid, &st, WNOHANG);
            if (r == run->pid || (r < 0 && errno == ECHILD)) {
                kill(-run->pid, SIGKILL);  // 그룹에 남은 손자 프로세스 정리 (리더가 먼저 끝난 경우)
                run->exit_code = r == run->pid ? (WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st)) : -1;
                run->state     = !run->stop_mode.empty() ? "stopped" : run->exit_code == 0 ? "done" : "failed";
                if (run->exit_code != 0 && run->stop_mode.empty()) run->error = "종료 코드 " + std::to_string(run->exit_code);
                run->pid      = -1;
                run->finished = now_unix();
                changed       = true;
            } else if (run->stop_mode == "term" &&
                       std::chrono::steady_clock::now() - run->term_at > std::chrono::seconds(10)) {
                kill(-run->pid, SIGKILL);  // 10초 안에 안 끝나면 강제
                run->stop_mode = "kill";
            }
        } else if (!g_paused) {
            for (auto & j : g_jobs)
                if (j.state == "queued") {
                    start_locked(j);
                    changed = true;
                    break;
                }
        }
        if (changed) save_locked();
    }
}

// ---------- 진행률 / 결과 ----------
static std::string read_tail(const fs::path & p, size_t max_bytes) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return "";
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    const std::streamoff s = n > (std::streamoff) max_bytes ? n - (std::streamoff) max_bytes : 0;
    f.seekg(s);
    std::string b((size_t) (n - s), '\0');
    f.read(b.data(), (std::streamsize) b.size());
    return b;
}
static std::vector<std::string> log_lines(int id, size_t max_bytes) {
    std::string t = read_tail(job_dir(id) / "log.txt", max_bytes);
    std::vector<std::string> v;
    std::string cur;
    for (char c : t) {
        if (c == '\n' || c == '\r') {
            if (!cur.empty()) v.push_back(cur);
            cur.clear();
        } else if (c != '\x1b') cur += c;
    }
    if (!cur.empty()) v.push_back(cur);
    for (auto & s : v) {  // 터미널 제어 문자열 흔적([K) 제거
        size_t q;
        while ((q = s.find("[K")) != std::string::npos) s.erase(q, 2);
    }
    return v;
}

static void add_progress(const Job & j, json & o) {
    if (j.state != "running" && j.state != "done") return;
    auto lines = log_lines(j.id, 65536);
    if (j.type == "train") {
        static const std::regex re(R"(^step\s+(\d+)\s+t=\s*\d+\s+loss\s+([0-9.eE+-]+|nan|inf)\s+ema\s+([0-9.eE+-]+)\s+([0-9.]+)s)");
        const int steps = (int) num(j.params, "steps", 200);
        for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
            std::smatch m;
            if (std::regex_search(*it, m, re)) {
                const int step = std::stoi(m[1]);
                const double sps = std::stod(m[4]);
                o["progress"] = {{"step", step}, {"steps", steps}, {"loss", m[2].str()}, {"ema", std::stod(m[3])}, {"sec_per_step", sps}};
                if (steps > step && j.state == "running") o["progress"]["eta_sec"] = sps * (steps - step);
                break;
            }
        }
    } else if (j.type == "generate") {
        static const std::regex re(R"((\d+)/(\d+)\s+-\s+([0-9.]+)(s/it|it/s))");
        for (auto it = lines.rbegin(); it != lines.rend(); ++it) {
            std::smatch m;
            if (std::regex_search(*it, m, re)) {
                const int step = std::stoi(m[1]), steps = std::stoi(m[2]);
                double sps = std::stod(m[3]);
                if (m[4] == "it/s" && sps > 0) sps = 1.0 / sps;
                o["progress"] = {{"step", step}, {"steps", steps}, {"sec_per_step", sps}};
                if (steps > step && j.state == "running") o["progress"]["eta_sec"] = sps * (steps - step);
                break;
            }
        }
    }
}

static json outputs(const Job & j) {
    json o = json::array();
    auto add = [&](const fs::path & p, const char * kind) {
        if (fs::exists(p)) o.push_back({{"path", rel_to_root(p)}, {"kind", kind}});
    };
    try {
        if (j.type == "generate") add(job_dir(j.id) / "image.png", "image");
        else if (j.type == "train") {
            fs::path out;
            const std::string def = "out/" + safe_name(str(j.params, "name", fs::path(str(j.params, "data")).filename().string())) + ".safetensors";
            if (resolve_in_root(str(j.params, "out", def), out)) {
                add(out, "lora");
                if (fs::is_directory(out.parent_path()))
                    for (auto & e : fs::directory_iterator(out.parent_path())) {
                        const std::string n = e.path().filename().string(), stem = out.stem().string() + "_s";
                        if (n.rfind(stem, 0) == 0 && e.path().extension() == ".safetensors") add(e.path(), "lora");
                    }
            }
        } else if (j.type == "cache") {
            fs::path dst;
            const std::string src = str(j.params, "src");
            const std::string def = "cache/" + safe_name(fs::path(src).filename().string()) + "_" + fmt("%.0f", num(j.params, "size", 512));
            if (resolve_in_root(str(j.params, "dst", def), dst) && fs::is_directory(dst)) {
                o.push_back({{"path", rel_to_root(dst)}, {"kind", "cache"}});
                for (auto & e : fs::directory_iterator(dst))
                    if (e.path().string().size() > 10 && e.path().string().rfind(".check.png") == e.path().string().size() - 10)
                        add(e.path(), "image");
            }
        }
    } catch (...) {
    }
    return o;
}

// 작업이 만든 파일만 지운다 (root 안, 그 작업이 쓰는 이름의 파일만). 원본 사진이나 다른 파일은 건드리지 않음.
static int delete_job_files(const Job & j) {
    int n = 0;
    std::error_code ec;
    auto del = [&](const fs::path & p) {
        fs::path r;
        if (resolve_in_root(p.string(), r) && fs::is_regular_file(r, ec) && fs::remove(r, ec)) n++;
    };
    if (j.type == "train") {
        fs::path out;
        const std::string def = "out/" + safe_name(str(j.params, "name", fs::path(str(j.params, "data")).filename().string())) + ".safetensors";
        if (resolve_in_root(str(j.params, "out", def), out)) {
            del(out);
            if (fs::is_directory(out.parent_path(), ec)) {
                std::vector<fs::path> steps;
                const std::string stem = out.stem().string() + "_s";
                for (auto & e : fs::directory_iterator(out.parent_path(), ec)) {
                    const std::string nm = e.path().filename().string();
                    if (nm.rfind(stem, 0) == 0 && e.path().extension() == ".safetensors" &&
                        nm.find_first_not_of("0123456789", stem.size()) == nm.size() - 12)  // <stem>_s<숫자>.safetensors 만
                        steps.push_back(e.path());
                }
                for (auto & x : steps) del(x);
            }
        }
    } else if (j.type == "cache") {
        fs::path dst;
        const std::string def = "cache/" + safe_name(fs::path(str(j.params, "src")).filename().string()) + "_" + fmt("%.0f", num(j.params, "size", 512));
        if (resolve_in_root(str(j.params, "dst", def), dst) && fs::is_directory(dst, ec)) {
            static const std::vector<std::string> suf = {".latent.npy", ".crossattn.npy", ".vector.npy", ".check.png"};
            std::vector<fs::path> rm;
            for (auto & e : fs::directory_iterator(dst, ec)) {
                const std::string nm = e.path().filename().string();
                if (nm == "cache.dgl" || nm == "captions.txt") rm.push_back(e.path());
                for (auto & x : suf)
                    if (nm.size() > x.size() && nm.compare(nm.size() - x.size(), x.size(), x) == 0) rm.push_back(e.path());
            }
            for (auto & x : rm) del(x);
            if (fs::is_empty(dst, ec)) fs::remove(dst, ec);  // 비었을 때만
        }
    }
    const fs::path d = job_dir(j.id);  // 로그, 생성 그림, 메모리 기록
    if (fs::is_directory(d, ec)) n += (int) fs::remove_all(d, ec);
    return n;
}

static json job_view(const Job & j, bool detail) {
    json o = job_to_json(j);
    o.erase("pid");
    add_progress(j, o);
    if (detail || j.state == "done") o["outputs"] = outputs(j);
    if (detail) {
        auto lines = log_lines(j.id, 32768);
        if (lines.size() > 60) lines.erase(lines.begin(), lines.end() - 60);
        o["log_tail"] = lines;
    }
    return o;
}

// ---------- 목록 (폼에서 고를 것들) ----------
static json list_files() {
    json o = {{"loras", json::array()}, {"caches", json::array()}, {"image_dirs", json::array()}};
    static const std::set<std::string> skip = {"models", "jobs", ".cache", ".git", "build", "ggml"};
    std::error_code ec;
    for (auto it = fs::recursive_directory_iterator(g_cfg.root, fs::directory_options::skip_permission_denied, ec);
         it != fs::recursive_directory_iterator(); it.increment(ec)) {
        if (ec) break;
        const fs::path & p = it->path();
        if (it.depth() == 0 && it->is_directory() && skip.count(p.filename().string())) {
            it.disable_recursion_pending();
            continue;
        }
        if (it.depth() > 3) {
            it.disable_recursion_pending();
            continue;
        }
        if (it->is_regular_file() && p.extension() == ".safetensors" && o["loras"].size() < 500) o["loras"].push_back(rel_to_root(p));
        if (it->is_directory()) {
            bool has_latent = false, has_img = false;
            for (auto & e : fs::directory_iterator(p, ec)) {
                const std::string n = e.path().filename().string();
                std::string ext = e.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (n.size() > 11 && n.rfind(".latent.npy") == n.size() - 11) has_latent = true;
                if ((ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".webp" || ext == ".bmp") && n.find(".check.") == std::string::npos) has_img = true;
            }
            if (has_latent) o["caches"].push_back(rel_to_root(p));
            else if (has_img) o["image_dirs"].push_back(rel_to_root(p));
        }
    }
    return o;
}

// ---------- 시작 시 큐 복원 ----------
static void load_queue() {
    std::ifstream f(jobs_dir() / "queue.json");
    if (!f) return;
    json q;
    try {
        q = json::parse(f);
    } catch (...) {
        fprintf(stderr, "queue.json을 읽을 수 없음 - 빈 큐로 시작\n");
        return;
    }
    g_next_id = q.value("next_id", 1);
    g_paused  = q.value("paused", false);
    for (auto & x : q["jobs"]) {
        Job j;
        j.id        = x.value("id", 0);
        j.type      = x.value("type", "");
        j.state     = x.value("state", "failed");
        j.params    = x.value("params", json::object());
        j.error     = x.value("error", "");
        j.created   = x.value("created", 0.0);
        j.started   = x.value("started", 0.0);
        j.finished  = x.value("finished", 0.0);
        j.exit_code = x.value("exit_code", 0);
        const pid_t old = x.value("pid", -1);
        if (j.state == "running") {
            // 서버가 꺼진 사이 남은 작업 프로세스가 GPU를 쥐고 있으면 새 작업과 겹쳐 메모리가 넘친다 → 정리
            if (old > 0 && kill(-old, 0) == 0) {
                fprintf(stderr, "이전 실행의 작업 %d (프로세스 그룹 %d)이 남아 있어 종료함\n", j.id, old);
                kill(-old, SIGTERM);
                for (int i = 0; i < 30 && kill(-old, 0) == 0; i++)  // 최대 3초 기다린 뒤 강제 std::this_thread::sleep_for(std::chrono::milliseconds(100));
                kill(-old, SIGKILL);
            }
            j.state    = "interrupted";
            j.error    = "서버가 다시 시작되어 중단됨";
            j.finished = now_unix();
        }
        g_next_id = std::max(g_next_id, j.id + 1);
        g_jobs.push_back(j);
    }
}

// ---------- HTTP ----------
static void reply(httplib::Response & res, int code, const json & j) {
    res.status = code;
    res.set_content(j.dump(), "application/json");
}

static bool token_ok(const httplib::Request & req) {
    if (g_cfg.token.empty()) return true;
    auto eq = [](const std::string & a, const std::string & b) {  // 길이 노출 외에는 시간 일정
        if (a.size() != b.size()) return false;
        unsigned char d = 0;
        for (size_t i = 0; i < a.size(); i++) d |= (unsigned char) (a[i] ^ b[i]);
        return d == 0;
    };
    const std::string auth = req.get_header_value("Authorization");
    if (auth.rfind("Bearer ", 0) == 0 && eq(auth.substr(7), g_cfg.token)) return true;
    if (req.has_param("token") && eq(req.get_param_value("token"), g_cfg.token)) return true;
    const std::string ck = req.get_header_value("Cookie");
    size_t p = 0;
    while ((p = ck.find("sdxl_token=", p)) != std::string::npos) {
        if (p == 0 || ck[p - 1] == ' ' || ck[p - 1] == ';') {
            size_t e = ck.find(';', p);
            if (eq(ck.substr(p + 11, e == std::string::npos ? std::string::npos : e - p - 11), g_cfg.token)) return true;
        }
        p += 11;
    }
    return false;
}

static json parse_body(const httplib::Request & req) {
    if (req.body.empty()) return json::object();
    json b = json::parse(req.body);  // 실패하면 예외 → 400
    if (!b.is_object()) throw std::runtime_error("JSON 객체가 필요함");
    return b;
}

static int enqueue(const std::string & type, const json & params) {
    validate(type, params);
    std::lock_guard<std::mutex> lk(g_mu);
    Job j;
    j.id      = g_next_id++;
    j.type    = type;
    j.params  = params;
    j.created = now_unix();
    g_jobs.push_back(j);
    save_locked();
    return j.id;
}

#include "page.h"  // 웹 화면 (HTML 한 장)

int main(int argc, char ** argv) {
    std::string host = "127.0.0.1", root;
    int         port = 8090;
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
        else if (a == "--cli") g_cfg.cli = next();
        else if (a == "--sd-cli") g_cfg.sd_cli = next();
        else if (a == "--tools") g_cfg.tools = next();
        else if (a == "--model") g_cfg.model = next();
        else if (a == "--vae") g_cfg.vae = next();
        else if (a == "--sd-args") g_cfg.sd_args = next();
        else if (a == "--guard") g_cfg.guard = next();
        else if (a == "--guard-limit-mib") g_cfg.guard_limit_mib = std::stoi(next());
        else if (a == "--token") g_cfg.token = next();
        else if (a == "--ui") g_cfg.ui = next();
        else if (a == "--cors") g_cfg.cors = next();
        else if (a == "--threads") g_cfg.threads = std::stoi(next());
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::stoi(next());
        else {
            fprintf(stderr,
                    "사용: %s --root <작업 폴더> [--cli <sdxl_lora>] [--sd-cli <sd-cli>] [--tools <tools 폴더>] [--model <ckpt>]\n"
                    "         [--vae <vae>] [--sd-args \"...\"] [--guard auto|on|off] [--guard-limit-mib N]\n"
                    "         [--token <접속 토큰>|auto] [--threads N] [--ui builtin|off|<폴더>] [--cors <origin>]\n"
                    "         [--host 127.0.0.1] [--port 8090]\n",
                    argv[0]);
            return 2;
        }
    }
    if (root.empty()) {
        fprintf(stderr, "--root가 필요함\n");
        return 2;
    }
    if (!g_cfg.cors.empty() && g_cfg.cors != "*" && g_cfg.cors.rfind("http://", 0) != 0 && g_cfg.cors.rfind("https://", 0) != 0) {
        fprintf(stderr, "--cors: \"*\" 또는 http(s)://주소\n");
        return 2;
    }
    if (const char * t = getenv("SDXL_SERVER_TOKEN"); t && g_cfg.token.empty()) g_cfg.token = t;  // 명령줄에 안 남기고 싶을 때
    bool token_generated = false;
    if (g_cfg.token == "auto") {  // 켤 때마다 새 임시 토큰 (/dev/urandom 20바이트 → 32글자)
        unsigned char rnd[20];
        std::ifstream ur("/dev/urandom", std::ios::binary);
        if (!ur.read((char *) rnd, sizeof(rnd))) {
            fprintf(stderr, "/dev/urandom을 읽을 수 없음\n");
            return 2;
        }
        static const char * A = "abcdefghijkmnpqrstuvwxyz23456789";  // 헷갈리는 글자(l, o, 0, 1) 뺀 32자
        std::string tok;
        for (int i = 0; i < 20; i += 5) {  // 5바이트 = 40비트 → 8글자
            unsigned long long v = 0;
            for (int k = 0; k < 5; k++) v = (v << 8) | rnd[i + k];
            for (int k = 7; k >= 0; k--) tok += A[(v >> (5 * k)) & 31];
        }
        g_cfg.token     = tok;
        token_generated = true;
    }
    g_cfg.root = fs::weakly_canonical(fs::absolute(root));
    const fs::path exe_dir = fs::canonical("/proc/self/exe").parent_path();
    if (g_cfg.cli.empty()) g_cfg.cli = (exe_dir / "sdxl_lora").string();
    if (!fs::exists(g_cfg.cli)) {
        fprintf(stderr, "CLI 학습기를 찾을 수 없음: %s (--cli로 지정)\n", g_cfg.cli.c_str());
        return 2;
    }
    g_cfg.cli = fs::canonical(g_cfg.cli).string();
    if (g_cfg.tools.empty()) {
        for (auto c : {exe_dir / "tools", exe_dir / ".." / "tools", g_cfg.root / "tools"})  // 릴리스 묶음 / 저장소(bin/, build/) / 작업 폴더
            if (fs::exists(c / "guard.sh")) {
                g_cfg.tools = fs::canonical(c).string();
                break;
            }
    }
    if (!g_cfg.sd_cli.empty()) {
        if (!fs::exists(g_cfg.sd_cli)) {
            fprintf(stderr, "sd-cli를 찾을 수 없음: %s\n", g_cfg.sd_cli.c_str());
            return 2;
        }
        g_cfg.sd_cli = fs::canonical(g_cfg.sd_cli).string();
    }
    const bool gtt_readable = access("/sys/class/drm/card0/device/mem_info_gtt_used", R_OK) == 0;
    const bool guard_found  = !g_cfg.tools.empty() && fs::exists(fs::path(g_cfg.tools) / "guard.sh");
    g_cfg.guard_on          = g_cfg.guard == "on" || (g_cfg.guard == "auto" && gtt_readable && guard_found);
    if (g_cfg.guard_on && !guard_found) {
        fprintf(stderr, "--guard on인데 tools/guard.sh가 없음\n");
        return 2;
    }

    signal(SIGPIPE, SIG_IGN);
    load_queue();
    {
        std::lock_guard<std::mutex> lk(g_mu);
        save_locked();
    }
    std::thread(worker_loop).detach();

    httplib::Server svr;
    svr.set_pre_routing_handler([](const httplib::Request & req, httplib::Response & res) {
        if (!g_cfg.cors.empty()) {  // 다른 주소에서 띄운 프론트가 이 API를 부를 수 있게 (--cors)
            res.set_header("Access-Control-Allow-Origin", g_cfg.cors);
            res.set_header("Vary", "Origin");
            res.set_header("Access-Control-Allow-Headers", "Authorization, Content-Type");
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Max-Age", "600");
            if (req.method == "OPTIONS") {
                res.status = 204;
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        // 토큰은 API와 결과 파일에만. 웹 화면(내장이든 --ui 폴더든)의 정적 파일은 토큰 없이 열려야 토큰 입력 화면을 보여 줄 수 있다
        const bool needs_token = req.path == "/api" || req.path.rfind("/api/", 0) == 0 || req.path.rfind("/files/", 0) == 0;
        if (!needs_token || token_ok(req)) return httplib::Server::HandlerResponse::Unhandled;
        reply(res, 401, {{"error", "토큰이 필요함"}});
        return httplib::Server::HandlerResponse::Handled;
    });
    svr.set_exception_handler([](const httplib::Request &, httplib::Response & res, std::exception_ptr ep) {
        std::string m = "오류";
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception & e) {
            m = e.what();
        } catch (...) {
        }
        reply(res, 400, {{"error", m}});
    });

    // 웹 화면: builtin = 내장 페이지, off = API만 (/는 안내 JSON), 그 밖 = 그 폴더의 정적 파일(index.html 등)을 /에 서빙
    if (g_cfg.ui == "builtin") {
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) { res.set_content(PAGE, "text/html; charset=utf-8"); });
    } else if (g_cfg.ui == "off") {
        svr.Get("/", [](const httplib::Request &, httplib::Response & res) {
            reply(res, 200, {{"name", "sdxl_lora_server"}, {"version", "0.4"}, {"ui", "off"}, {"auth", !g_cfg.token.empty()},
                             {"api", {"GET /api/config", "GET /api/jobs", "POST /api/jobs", "GET /api/jobs/<id>", "GET /api/jobs/<id>/log",
                                      "POST /api/jobs/<id>/cancel|kill|move|remove", "POST /api/jobs/remove-many",
                                      "POST /api/queue/cancel-all|pause|resume", "GET /api/list", "GET /files/<path>"}}});
        });
    } else if (!svr.set_mount_point("/", g_cfg.ui)) {
        fprintf(stderr, "--ui: 폴더가 없음: %s (builtin | off | <폴더>)\n", g_cfg.ui.c_str());
        return 2;
    }
    svr.Get("/api/config", [](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mu);
        reply(res, 200, {{"sd_cli", !g_cfg.sd_cli.empty()}, {"cache", !g_cfg.sd_cli.empty()},
                         {"guard", g_cfg.guard_on}, {"guard_limit_mib", g_cfg.guard_limit_mib}, {"token", !g_cfg.token.empty()},
                         {"model", g_cfg.model}, {"vae", g_cfg.vae}, {"paused", g_paused}});
    });
    svr.Get("/api/jobs", [](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mu);
        json a = json::array();
        for (auto & j : g_jobs) a.push_back(job_view(j, false));
        reply(res, 200, {{"jobs", a}, {"paused", g_paused}});
    });
    svr.Post("/api/jobs", [](const httplib::Request & req, httplib::Response & res) {
        json b = parse_body(req);
        const std::string type = str(b, "type");
        b.erase("type");
        reply(res, 200, {{"id", enqueue(type, b)}});
    });
    svr.Get(R"(/api/jobs/(\d+))", [](const httplib::Request & req, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mu);
        Job * j = find_locked(std::stoi(req.matches[1]));
        if (!j) return reply(res, 404, {{"error", "작업 없음"}});
        reply(res, 200, job_view(*j, true));
    });
    svr.Get(R"(/api/jobs/(\d+)/log)", [](const httplib::Request & req, httplib::Response & res) {
        const std::string t = read_tail(job_dir(std::stoi(req.matches[1])) / "log.txt", 8u << 20);
        res.set_content(t, "text/plain; charset=utf-8");
    });
    svr.Post(R"(/api/jobs/(\d+)/(cancel|kill|move|remove))", [](const httplib::Request & req, httplib::Response & res) {
        const json b = parse_body(req);
        const std::string act = req.matches[2];
        std::lock_guard<std::mutex> lk(g_mu);
        Job * j = find_locked(std::stoi(req.matches[1]));
        if (!j) return reply(res, 404, {{"error", "작업 없음"}});
        if (act == "cancel" || act == "kill") {
            if (j->state != "queued" && j->state != "running") return reply(res, 409, {{"error", "대기·실행 중인 작업이 아님"}});
            signal_job_locked(*j, act == "kill");
        } else if (act == "move") {
            if (j->state != "queued") return reply(res, 409, {{"error", "대기 중인 작업만 옮길 수 있음"}});
            const std::string dir = str(b, "dir", "up");
            auto it = std::find_if(g_jobs.begin(), g_jobs.end(), [&](const Job & x) { return x.id == j->id; });
            Job moving = *it;
            const size_t pos = (size_t) (it - g_jobs.begin());
            std::vector<size_t> q;  // 대기 작업들의 위치
            for (size_t i = 0; i < g_jobs.size(); i++)
                if (g_jobs[i].state == "queued") q.push_back(i);
            const size_t k = (size_t) (std::find(q.begin(), q.end(), pos) - q.begin());
            size_t target = pos;
            if (dir == "up" && k > 0) target = q[k - 1];
            else if (dir == "down" && k + 1 < q.size()) target = q[k + 1];
            else if (dir == "top") target = q[0];
            if (target != pos) {
                g_jobs.erase(g_jobs.begin() + (long) pos);
                g_jobs.insert(g_jobs.begin() + (long) target, moving);
            }
        } else if (act == "remove") {
            if (j->state == "queued" || j->state == "running") return reply(res, 409, {{"error", "먼저 취소할 것"}});
            const int id = j->id;
            if (b.value("delete_files", false)) delete_job_files(*j);
            g_jobs.erase(std::remove_if(g_jobs.begin(), g_jobs.end(), [&](const Job & x) { return x.id == id; }), g_jobs.end());
        }
        save_locked();
        reply(res, 200, {{"ok", true}});
    });
    svr.Post("/api/jobs/remove-many", [](const httplib::Request & req, httplib::Response & res) {
        const json b = parse_body(req);
        if (!b.contains("ids") || !b["ids"].is_array()) return reply(res, 400, {{"error", "ids 배열이 필요함"}});
        const bool files = b.value("delete_files", false);
        std::lock_guard<std::mutex> lk(g_mu);
        int removed = 0, files_n = 0;
        json skipped = json::array();
        for (auto & x : b["ids"]) {
            const int id = x.get<int>();
            Job * j = find_locked(id);
            if (!j) continue;
            if (j->state == "queued" || j->state == "running") {
                skipped.push_back(id);
                continue;
            }
            if (files) files_n += delete_job_files(*j);
            g_jobs.erase(std::remove_if(g_jobs.begin(), g_jobs.end(), [&](const Job & y) { return y.id == id; }), g_jobs.end());
            removed++;
        }
        save_locked();
        reply(res, 200, {{"removed", removed}, {"files_deleted", files_n}, {"skipped_active", skipped}});
    });
    svr.Post(R"(/api/queue/(cancel-all|pause|resume))", [](const httplib::Request & req, httplib::Response & res) {
        const json b = parse_body(req);
        const std::string act = req.matches[1];
        std::lock_guard<std::mutex> lk(g_mu);
        int n = 0;
        if (act == "cancel-all") {
            for (auto & j : g_jobs) {
                if (j.state == "queued") {
                    signal_job_locked(j, false);
                    n++;
                } else if (j.state == "running" && b.value("running", false)) {
                    signal_job_locked(j, b.value("force", false));
                    n++;
                }
            }
        } else g_paused = act == "pause";
        save_locked();
        reply(res, 200, {{"ok", true}, {"count", n}, {"paused", g_paused}});
    });
    svr.Get("/api/list", [](const httplib::Request &, httplib::Response & res) { reply(res, 200, list_files()); });
    svr.Get(R"(/files/(.+))", [](const httplib::Request & req, httplib::Response & res) {
        fs::path p;
        if (!resolve_in_root(req.matches[1].str(), p) || !fs::is_regular_file(p)) {
            res.status = 404;
            return;
        }
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        const char * ct = ext == ".png" ? "image/png" : ext == ".jpg" || ext == ".jpeg" ? "image/jpeg" : ext == ".webp" ? "image/webp"
                        : ext == ".json" ? "application/json" : ext == ".txt" || ext == ".log" ? "text/plain; charset=utf-8"
                        : "application/octet-stream";
        const size_t sz = fs::file_size(p);
        res.set_content_provider(sz, ct, [p](size_t off, size_t len, httplib::DataSink & sink) {
            std::ifstream f(p, std::ios::binary);
            f.seekg((std::streamoff) off);
            std::vector<char> buf(std::min<size_t>(len, 1 << 20));
            f.read(buf.data(), (std::streamsize) buf.size());
            sink.write(buf.data(), (size_t) f.gcount());
            return true;
        });
        if (ext == ".safetensors") res.set_header("Content-Disposition", "attachment; filename=\"" + p.filename().string() + "\"");
    });

    // ---- v0.3 호환 ----
    svr.Post("/api/train", [](const httplib::Request & req, httplib::Response & res) {
        reply(res, 200, {{"id", enqueue("train", parse_body(req))}, {"queued", true}});
    });
    auto current = []() -> Job * {  // 실행 중 → 없으면 마지막으로 시작한 작업
        Job * last = nullptr;
        for (auto & j : g_jobs) {
            if (j.state == "running") return &j;
            if (j.started && (!last || j.started >= last->started)) last = &j;
        }
        return last;
    };
    svr.Get("/api/status", [current](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mu);
        Job * j = current();
        if (!j) return reply(res, 200, {{"state", "idle"}});
        json v = job_view(*j, true), o = {{"state", v["state"]}, {"id", j->id}, {"type", j->type}, {"exit_code", j->exit_code}};
        if (v.contains("progress")) {
            for (auto & [k, x] : v["progress"].items()) o[k] = x;
        }
        std::vector<std::string> t = v["log_tail"];
        if (t.size() > 20) t.erase(t.begin(), t.end() - 20);
        o["log_tail"] = t;
        int q = 0;
        for (auto & x : g_jobs) q += x.state == "queued";
        o["queued"] = q;
        reply(res, 200, o);
    });
    svr.Post("/api/stop", [](const httplib::Request &, httplib::Response & res) {
        std::lock_guard<std::mutex> lk(g_mu);
        for (auto & j : g_jobs)
            if (j.state == "running") {
                signal_job_locked(j, false);
                save_locked();
                return reply(res, 200, {{"ok", true}, {"id", j.id}});
            }
        reply(res, 409, {{"error", "실행 중인 작업 없음"}});
    });
    svr.Get("/api/log", [current](const httplib::Request &, httplib::Response & res) {
        int id = 0;
        {
            std::lock_guard<std::mutex> lk(g_mu);
            if (Job * j = current()) id = j->id;
        }
        res.set_content(id ? read_tail(job_dir(id) / "log.txt", 8u << 20) : "", "text/plain; charset=utf-8");
    });

    printf("sdxl_lora_server v0.4: http://%s:%d  (root %s)\n", host.c_str(), port, g_cfg.root.c_str());
    printf("  웹 화면 %s%s\n", g_cfg.ui == "builtin" ? "내장" : g_cfg.ui == "off" ? "끔 (API만)" : g_cfg.ui.c_str(),
           g_cfg.cors.empty() ? "" : ("  CORS 허용: " + g_cfg.cors).c_str());
    printf("  학습기 %s\n  sd-cli %s\n  tools %s\n  guard %s (합계 한도 %d MiB)  인증 %s\n", g_cfg.cli.c_str(),
           g_cfg.sd_cli.empty() ? "(없음: 캐시·생성 작업 불가)" : g_cfg.sd_cli.c_str(), g_cfg.tools.empty() ? "(없음)" : g_cfg.tools.c_str(),
           g_cfg.guard_on ? "켜짐" : "꺼짐", g_cfg.guard_limit_mib, g_cfg.token.empty() ? "없음" : "토큰");
    if (g_cfg.token.empty()) printf("주의: 인증 없음. 터널·포트포워딩으로 외부에 열 때는 --token auto(임시 토큰)나 --token <토큰>을 줄 것\n");
    if (token_generated) {
        printf("\n  접속 토큰 (이번 실행에만 유효): %s\n  바로 열기: http://%s:%d/?token=%s\n\n", g_cfg.token.c_str(),
               host == "0.0.0.0" ? "127.0.0.1" : host.c_str(), port, g_cfg.token.c_str());
    }
    fflush(stdout);
    return svr.listen(host, port) ? 0 : 1;
}
