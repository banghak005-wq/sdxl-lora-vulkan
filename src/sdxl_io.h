// 학습기 입출력 도우미: safetensors 헤더/텐서 로드, .npy 읽기/쓰기, SDXL 이산 노이즈 스케줄
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "json.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct StEntry {
    std::string          dtype;  // F16, F32, BF16 ...
    std::vector<int64_t> shape;  // safetensors(PyTorch) 순서
    uint64_t             begin = 0, end = 0;  // 데이터 영역 기준 오프셋
};

struct StFile {
    std::string                    path;
    uint64_t                       data_start = 0;
    std::map<std::string, StEntry> tensors;

    explicit StFile(const std::string & p) : path(p) {
        FILE * f = fopen(p.c_str(), "rb");
        if (!f) throw std::runtime_error("열기 실패: " + p);
        uint64_t n = 0;
        if (fread(&n, 8, 1, f) != 1 || n > (1ull << 30)) throw std::runtime_error("헤더 크기 이상: " + p);
        std::string h(n, '\0');
        if (fread(h.data(), 1, n, f) != n) throw std::runtime_error("헤더 읽기 실패: " + p);
        fclose(f);
        data_start = 8 + n;
        auto j     = nlohmann::json::parse(h);
        for (auto & [k, v] : j.items()) {
            if (k == "__metadata__") continue;
            StEntry e;
            e.dtype = v["dtype"].get<std::string>();
            e.shape = v["shape"].get<std::vector<int64_t>>();
            e.begin = v["data_offsets"][0].get<uint64_t>();
            e.end   = v["data_offsets"][1].get<uint64_t>();
            tensors[k] = e;
        }
    }

    static ggml_type to_ggml(const std::string & dt) {
        if (dt == "F16") return GGML_TYPE_F16;
        if (dt == "F32") return GGML_TYPE_F32;
        if (dt == "BF16") return GGML_TYPE_BF16;
        throw std::runtime_error("지원 안 하는 dtype: " + dt);
    }
};

// raw 데이터(src_type)를 dst 텐서 타입으로 바꿔 백엔드에 올린다 (f16→f32 변환만 지원)
static void st_upload(ggml_tensor * dst, ggml_type src_type, const std::vector<uint8_t> & raw) {
    if (dst->type == src_type) {
        ggml_backend_tensor_set(dst, raw.data(), 0, raw.size());
        return;
    }
    if (dst->type == GGML_TYPE_F32 && src_type == GGML_TYPE_F16) {
        const size_t       n = raw.size() / 2;
        std::vector<float> f(n);
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) raw.data(), f.data(), (int64_t) n);
        ggml_backend_tensor_set(dst, f.data(), 0, n * sizeof(float));
        return;
    }
    throw std::runtime_error(std::string("변환 미지원: ") + ggml_type_name(src_type) + " -> " + ggml_type_name(dst->type));
}

// ---------- .npy (little-endian f32, C 순서) ----------
struct Npy {
    std::vector<int64_t> shape;  // numpy 순서
    std::vector<float>   data;

    int64_t numel() const {
        int64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

static Npy npy_load(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) throw std::runtime_error("열기 실패: " + p);
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy 아님: " + p);
    uint32_t hl = 0;
    if (magic[6] == 1) {
        uint16_t h16;
        if (fread(&h16, 2, 1, f) != 1) throw std::runtime_error("npy 헤더: " + p);
        hl = h16;
    } else {
        if (fread(&hl, 4, 1, f) != 1) throw std::runtime_error("npy 헤더: " + p);
    }
    std::string h(hl, '\0');
    if (fread(h.data(), 1, hl, f) != hl) throw std::runtime_error("npy 헤더: " + p);
    if (h.find("'<f4'") == std::string::npos) throw std::runtime_error("f32(<f4)만 지원: " + p);
    if (h.find("'fortran_order': False") == std::string::npos) throw std::runtime_error("C 순서만 지원: " + p);
    Npy    r;
    size_t a = h.find("'shape': (") + 10, b = h.find(')', a);
    std::string s = h.substr(a, b - a);
    for (size_t pos = 0; pos < s.size();) {
        size_t e = s.find(',', pos);
        if (e == std::string::npos) e = s.size();
        std::string tok = s.substr(pos, e - pos);
        if (tok.find_first_of("0123456789") != std::string::npos) r.shape.push_back(std::stoll(tok));
        pos = e + 1;
    }
    r.data.resize(r.numel());
    if (fread(r.data.data(), 4, r.data.size(), f) != r.data.size()) throw std::runtime_error("npy 데이터: " + p);
    fclose(f);
    return r;
}

static void npy_save(const std::string & p, const std::vector<int64_t> & shape, const float * data) {
    std::string sh;
    int64_t     n = 1;
    for (auto s : shape) {
        sh += std::to_string(s) + ", ";
        n *= s;
    }
    if (shape.size() > 1) sh.resize(sh.size() - 1);
    std::string h = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + sh + "), }";
    while ((10 + h.size() + 1) % 64) h += ' ';
    h += '\n';
    FILE * f = fopen(p.c_str(), "wb");
    if (!f) throw std::runtime_error("쓰기 실패: " + p);
    fwrite("\x93NUMPY\x01\x00", 1, 8, f);
    uint16_t hl = (uint16_t) h.size();
    fwrite(&hl, 2, 1, f);
    fwrite(h.data(), 1, h.size(), f);
    fwrite(data, 4, n, f);
    fclose(f);
}

// ---------- SDXL 이산 스케줄 (scaled_linear, 1000스텝) ----------
struct DiscreteSchedule {
    std::vector<double> sigmas, log_sigmas, alphas_cumprod;

    DiscreteSchedule() {
        const int    T  = 1000;
        const double b0 = sqrt(0.00085), b1 = sqrt(0.012);
        double       ac = 1.0;
        for (int i = 0; i < T; i++) {
            const double b = b0 + (b1 - b0) * i / (T - 1);
            ac *= 1.0 - b * b;
            alphas_cumprod.push_back(ac);
            sigmas.push_back(sqrt((1.0 - ac) / ac));
            log_sigmas.push_back(log(sigmas.back()));
        }
    }

    // k-diffusion sigma_to_t (log sigma 선형 보간)
    double sigma_to_t(double sigma) const {
        const double ls  = log(sigma);
        int          low = 0;
        for (int i = 0; i < (int) log_sigmas.size() - 1; i++) {
            if (ls >= log_sigmas[i]) low = i;
        }
        const int    high = low + 1;
        const double w    = (log_sigmas[low] - ls) / (log_sigmas[low] - log_sigmas[high]);
        const double wc   = std::min(1.0, std::max(0.0, w));
        return (1.0 - wc) * low + wc * high;
    }
};
