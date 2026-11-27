// SDXL LoRA 학습기 (ggml, Vulkan/CPU)
//
// 모드:
//   --check-ref <dir>  : sd.cpp(DGL, trainer/dgl/ref_unet.dgl)가 만든 기준값으로 UNet 순전파 비교
//   --mem-plan <잠재값 한 변> : 학습 그래프 계산 버퍼 크기와 역전파용 활성값 내역 (GPU/할당 없이)
//   --train <dir>      : 캐시된 샘플(<이름>.latent.npy / .crossattn.npy / .vector.npy)로 LoRA 학습
//                        --steps N --lr F --rank R --alpha A --seed S --out <lora.safetensors> --save-every N
//                        --conv-rank R --conv-alpha A (LoCon: ResBlock/다운·업샘플 합성곱에도 LoRA, kohya conv_dim과 같음)
//                        --t-min/--t-max (timestep 범위, 기본 0..999)
//                        --ckpt (블록 단위 그래디언트 체크포인팅), --grad-check (ckpt 기울기 = 전체 그래프 기울기 확인)
//                        --resume <lora.safetensors> (이어서 학습. --lr 0이면 고정 seed 손실 평가용)
//                        --loss-scale <S|dynamic> (v0.3, fp16 행렬곱 학습용 손실 스케일링. 기본: fp16 행렬곱이면 dynamic, 아니면 1)
//                        --loss-scale-init S (dynamic 시작값, 기본 32768) --loss-scale-window N (연속 정상 N스텝마다 ×2, 기본 200)
//
// 공통 옵션: --model <sd_xl_base.safetensors> --backend vulkan|cpu --native <ops> --conv-f16 --threads N
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef SDXL_VULKAN
#include "ggml-vulkan.h"
#endif
#include "sdxl_io.h"
#include "unet_ggml.h"

#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <numeric>
#include <random>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

static const char * UNET_PREFIX = "model.diffusion_model.";
static void * const FAKE_DATA = reinterpret_cast<void *>(uintptr_t(0x1000));

struct Opts {
    std::string           model   = "models/sd_xl_base_1.0.safetensors";
    std::string           backend = "vulkan";
    std::string           check_ref;
    std::set<std::string> native;
    bool                  conv_f16 = false;
    int                   threads  = 8;
    // 학습
    std::string train_dir;
    int         mem_plan = 0;  // 잠재값 한 변 (64 = 512px)
    bool        ckpt       = false;  // 그래디언트 체크포인팅 (블록 단위 재계산)
    bool        grad_check = false;  // 체크포인팅 기울기 vs 전체 그래프 기울기 비교 (2번째 스텝에서 비교 후 종료)
    std::string lora_in;  // --check-ref에 걸 kohya LoRA
    std::string out        = "out/lora.safetensors";
    int         steps      = 100;
    float       lr         = 1e-4f;
    int         rank       = 8;
    float       alpha      = 8.0f;
    int         conv_rank  = 0;      // LoCon: k>1 합성곱 LoRA rank (0 = 끔, v0.1과 같음)
    float       conv_alpha = -1.0f;  // 기본 = conv_rank (scale 1)
    int         seed       = 42;
    int         save_every = 0;
    int         t_min = 0, t_max = 999;
    // v0.3 손실 스케일링: ls_mode "" = 자동, "dynamic", 또는 고정값
    std::string ls_mode;
    float       ls_init   = 32768.0f;
    int         ls_window = 200;
};

// 손실 스케일링 (fp16 행렬곱 학습용, v0.3)
// 손실 기울기 시작값을 1 대신 S로 → 역전파 중 fp16 변환에서 작은 기울기가 0이 되는(언더플로) 것을 막는다.
// LoRA 기울기는 f32 저장소로 복사된 뒤 AdamW 직전에 1/S를 곱해 원래 크기로 되돌린다 (2의 거듭제곱이라 f32에서 정확).
// dynamic: 기울기에 NaN/Inf(fp16 넘침)가 있으면 그 스텝의 갱신을 건너뛰고 S/2, window 스텝 연속 정상이면 S×2.
struct LossScaler {
    bool  dynamic = false;
    float S       = 1.0f;
    int   window  = 200, good = 0, n_skip = 0, n_skip_run = 0;
    static constexpr float S_MAX = 16777216.0f;  // 2^24
    void init(const Opts & O, bool fp16_mm) {
        std::string m = O.ls_mode.empty() ? (fp16_mm ? "dynamic" : "1") : O.ls_mode;
        window        = O.ls_window;
        if (m == "dynamic") {
            dynamic = true;
            S       = O.ls_init;
        } else {
            S = (float) atof(m.c_str());
            if (!(S > 0)) throw std::runtime_error("--loss-scale 값이 잘못됨: " + m);
        }
        printf("손실 스케일링: %s, S=%g (fp16 행렬곱 %s)\n", dynamic ? "dynamic" : "고정", S, fp16_mm ? "켜짐" : "꺼짐");
        if (fp16_mm && S == 1.0f) printf("경고: fp16 행렬곱인데 손실 스케일 1 → 기울기 언더플로로 LoRA가 거의 학습 안 됨 (v0.2 for_ai.md 1.8)\n");
    }
    // 기울기 검사 결과를 받아 갱신 여부를 돌려준다
    bool update(bool finite, int step) {
        if (finite) {
            n_skip_run = 0;
            if (dynamic && ++good >= window && S < S_MAX) {
                S *= 2;
                good = 0;
                printf("loss-scale: %d스텝 연속 정상 → S=%g\n", window, S);
            }
            return true;
        }
        if (!dynamic) throw std::runtime_error("기울기에 NaN/Inf (step " + std::to_string(step) + ", 고정 손실 스케일 S=" + std::to_string(S) + ") → --loss-scale을 줄이거나 dynamic");
        n_skip++;
        good = 0;
        S /= 2;
        printf("overflow step %d: 기울기 NaN/Inf → 갱신 건너뜀, S=%g\n", step, S);
        if (++n_skip_run >= 20 || S < 1.0f) throw std::runtime_error("기울기 NaN/Inf가 계속됨 (S=" + std::to_string(S) + ") → 손실 스케일링으로 해결 안 되는 수치 문제");
        return false;
    }
};

// fp16 행렬곱이 켜졌는지: ggml Vulkan은 fp16 지원 장치에서 GGML_VK_DISABLE_F16이 없으면 f32 행렬곱도 fp16으로 계산
static bool fp16_matmul_active(const Opts & O) {
    return O.backend == "vulkan" && getenv("GGML_VK_DISABLE_F16") == nullptr;
}

// LoRA 기울기 저장소 → 유한값 검사 그래프 (전체 합 하나; NaN/Inf는 합에 남는다) + 1/S 후 AdamW 그래프
static ggml_tensor * build_finite_check(ggml_context * ctx, ggml_cgraph * gf, const std::vector<ggml_tensor *> & gstore) {
    ggml_tensor * tot = nullptr;
    for (ggml_tensor * g : gstore) {
        ggml_tensor * s = ggml_sum(ctx, ggml_abs(ctx, g));  // abs: +Inf와 -Inf가 섞여도 NaN, 유한 합은 f32 범위 안
        tot             = tot ? ggml_add(ctx, tot, s) : s;
    }
    ggml_set_output(tot);
    ggml_build_forward_expand(gf, tot);
    return tot;
}


static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double rel_l2(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (double) (a[i] - b[i]) * (a[i] - b[i]);
        den += (double) b[i] * b[i];
    }
    return sqrt(num / den);
}

// UNet 가중치: 2D/4D는 파일 타입(f16) 그대로, 1D(bias, norm)는 f32로 바꿔 올린다
struct UNetWeights {
    ggml_context *                       ctx = nullptr;
    ggml_backend_buffer_t                buf = nullptr;
    std::map<std::string, ggml_tensor *> T;  // 접두사 뺀 이름

    void load(StFile & st, ggml_backend_t backend) {
        std::vector<std::pair<std::string, const StEntry *>> list;
        for (auto & [k, e] : st.tensors) {
            if (k.rfind(UNET_PREFIX, 0) == 0) list.push_back({k, &e});
        }
        ctx = ggml_init({(list.size() + 8) * ggml_tensor_overhead(), nullptr, true});
        for (auto & [k, e] : list) {
            int64_t ne[4] = {1, 1, 1, 1};
            const int nd = (int) e->shape.size();
            for (int i = 0; i < nd; i++) ne[i] = e->shape[nd - 1 - i];
            const ggml_type ft = StFile::to_ggml(e->dtype);
            ggml_tensor *   t  = ggml_new_tensor(ctx, nd == 1 ? GGML_TYPE_F32 : ft, nd, ne);
            const std::string name = k.substr(strlen(UNET_PREFIX));
            ggml_set_name(t, name.c_str());
            T[name] = t;
        }
        if (!backend) {  // --mem-plan: 모양만. 할당기가 "이미 할당됨"으로 보게 가짜 포인터
            for (auto & [n, t] : T) t->data = FAKE_DATA;
            return;
        }
        buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        printf("UNet 가중치: %zu개, %.1f MiB (%s)\n", list.size(), ggml_backend_buffer_get_size(buf) / 1048576.0,
               ggml_backend_buffer_name(buf));

        std::sort(list.begin(), list.end(), [](auto & a, auto & b) { return a.second->begin < b.second->begin; });
        FILE * f = fopen(st.path.c_str(), "rb");
        std::vector<uint8_t> raw;
        const double         t0 = now_s();
        for (auto & [k, e] : list) {
            raw.resize(e->end - e->begin);
            fseeko(f, (off_t) (st.data_start + e->begin), SEEK_SET);
            if (fread(raw.data(), 1, raw.size(), f) != raw.size()) throw std::runtime_error("읽기 실패: " + k);
            st_upload(T[k.substr(strlen(UNET_PREFIX))], StFile::to_ggml(e->dtype), raw);
        }
        // 가중치는 이미 GPU 버퍼(UMA host-visible)에 있으므로 파일 페이지 캐시(약 7GB)는 버리라고 알림
        posix_fadvise(fileno(f), 0, 0, POSIX_FADV_DONTNEED);
        fclose(f);
        printf("가중치 로드 %.1fs\n", now_s() - t0);
    }

    ggml_tensor * get(const std::string & n) const {
        auto it = T.find(n);
        if (it == T.end()) {
            fprintf(stderr, "텐서 없음: %s\n", n.c_str());
            exit(2);
        }
        return it->second;
    }
};

// ---------- LoRA ----------
struct LoraSet {
    struct Module {
        std::string   name;  // 예: input_blocks.4.1.transformer_blocks.0.attn1.to_q
        bool          conv  = false;  // 합성곱 (텐서 4D: down [KW,KH,IC,R], up [1,1,R,OC])
        int           rank  = 0;
        float         alpha = 0;
        ggml_tensor * down = nullptr, * up = nullptr;
    };
    std::vector<Module>                  mods;
    std::map<std::string, ggml_tensor *> T;  // "lora.<name>.lora_down|lora_up" -> 텐서
    std::vector<ggml_tensor *>           m, v;  // AdamW 모멘트 (params 순서)
    std::vector<ggml_tensor *>           params;
    ggml_tensor *                        adamw = nullptr;  // [7]
    ggml_context *                       ctx   = nullptr;
    ggml_backend_buffer_t                buf   = nullptr;
    int                                  rank = 0, conv_rank = 0;
    float                                alpha = 0, conv_alpha = 0;

    static bool ends_with(const std::string & n, const char * suf) {
        const size_t L = strlen(suf);
        return n.size() > L && n.compare(n.size() - L, L, suf) == 0;
    }

    // 대상 (kohya networks/lora.py 규칙):
    //  - 항상: 모든 attn1/attn2의 to_q/k/v/out → rank/alpha
    //  - conv_rank > 0 (LoCon): ResBlock/Downsample/Upsample 안의 k>1 합성곱(in_layers.2, out_layers.3, op, upsample conv)
    //    → conv_rank/conv_alpha, 같은 블록 안의 1x1 합성곱(skip_connection)과 Linear(emb_layers.1) → rank/alpha.
    //    conv_in(input_blocks.0.0), out.2는 대상 아님
    void create(const UNetWeights & w, int rank_, float alpha_, int conv_rank_, float conv_alpha_, ggml_backend_t backend,
                std::mt19937 & rng) {
        rank = rank_, alpha = alpha_, conv_rank = conv_rank_, conv_alpha = conv_alpha_;
        static const char * ATTN[] = {".to_q.weight", ".to_k.weight", ".to_v.weight", ".to_out.0.weight"};
        static const char * RES[]  = {".in_layers.2.weight", ".out_layers.3.weight", ".skip_connection.weight",
                                      ".emb_layers.1.weight", ".0.op.weight", ".conv.weight"};
        for (auto & [n, t] : w.T) {
            const std::string base = n.substr(0, n.size() - 7);  // ".weight" 제거
            if (n.find(".transformer_blocks.") != std::string::npos &&
                (n.find(".attn1.") != std::string::npos || n.find(".attn2.") != std::string::npos)) {
                for (const char * suf : ATTN) {
                    if (ends_with(n, suf)) mods.push_back({base, false, rank, alpha});
                }
                continue;
            }
            if (conv_rank <= 0) continue;
            if (n.rfind("input_blocks.", 0) != 0 && n.rfind("middle_block.", 0) != 0 && n.rfind("output_blocks.", 0) != 0) continue;
            if (n.find(".transformer_blocks.") != std::string::npos) continue;
            for (const char * suf : RES) {
                if (!ends_with(n, suf)) continue;
                const bool is_conv = ggml_n_dims(t) == 4;
                const bool k1      = !is_conv || (t->ne[0] == 1 && t->ne[1] == 1);
                mods.push_back({base, is_conv, k1 ? rank : conv_rank, k1 ? alpha : conv_alpha});
            }
        }
        ctx = ggml_init({(mods.size() * 6 + 8) * ggml_tensor_overhead(), nullptr, true});
        for (auto & md : mods) {
            ggml_tensor * W = w.get(md.name + ".weight");  // Linear ne = [in, out], 합성곱 ne = [KW, KH, IC, OC]
            if (md.conv) {
                md.down = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, W->ne[0], W->ne[1], W->ne[2], md.rank);
                md.up   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, 1, md.rank, W->ne[3]);
            } else {
                md.down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W->ne[0], md.rank);
                md.up   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, md.rank, W->ne[1]);
            }
            ggml_set_name(md.down, ("lora." + md.name + ".lora_down").c_str());
            ggml_set_name(md.up, ("lora." + md.name + ".lora_up").c_str());
            T["lora." + md.name + ".lora_down"] = md.down;
            T["lora." + md.name + ".lora_up"]   = md.up;
            for (ggml_tensor * p : {md.down, md.up}) {
                ggml_set_param(p);
                params.push_back(p);
                m.push_back(ggml_dup_tensor(ctx, p));
                v.push_back(ggml_dup_tensor(ctx, p));
            }
        }
        adamw = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 7);
        size_t np = 0, nconv = 0, nc = 0;
        for (auto & md : mods) {
            const size_t e = ggml_nelements(md.down) + ggml_nelements(md.up);
            np += e;
            if (md.conv || md.name.find(".emb_layers.") != std::string::npos) nconv += e, nc++;
        }
        printf("LoRA: 모듈 %zu개 (어텐션 %zu, ResBlock/샘플러 %zu), rank %d alpha %g, conv_rank %d conv_alpha %g, 파라미터 %.2fM (ResBlock 쪽 %.2fM)\n",
               mods.size(), mods.size() - nc, nc, rank, alpha, conv_rank, conv_alpha, np / 1e6, nconv / 1e6);
        if (!backend) {
            for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) t->data = FAKE_DATA;
            return;
        }
        buf   = ggml_backend_alloc_ctx_tensors(ctx, backend);
        ggml_backend_buffer_clear(buf, 0);
        // kohya(LoRAModule)와 같게: down = kaiming_uniform(a=sqrt(5)) = U(-1/sqrt(fan_in), 1/sqrt(fan_in)), up = 0
        // fan_in = Linear: in, 합성곱: IC*KH*KW
        // 어텐션 모듈은 rng로 먼저 (v0.1과 같은 순서·같은 값), ResBlock 쪽(LoCon)은 rng를 소비하지 않는 별도 난수열로.
        // → 같은 seed면 conv-rank와 상관없이 어텐션 초기값과 이후 t/노이즈/샘플 순서가 같다 (공정 비교, v0.1 재현)
        std::mt19937 rng_copy = rng;
        std::mt19937 crng((uint32_t) rng_copy() ^ 0x9e3779b9u);
        for (int pass = 0; pass < 2; pass++) {
            for (auto & md : mods) {
                const bool attn = md.name.find(".transformer_blocks.") != std::string::npos;
                if (attn != (pass == 0)) continue;
                const int64_t                         fan_in = ggml_nelements(md.down) / md.rank;
                const float                           b      = 1.0f / sqrtf((float) fan_in);
                std::uniform_real_distribution<float> U(-b, b);
                std::vector<float>                    d(ggml_nelements(md.down));
                for (float & x : d) x = U(pass == 0 ? rng : crng);
                ggml_backend_tensor_set(md.down, d.data(), 0, ggml_nbytes(md.down));
            }
        }
    }

    size_t n_params() const {
        size_t np = 0;
        for (ggml_tensor * p : params) np += ggml_nelements(p);
        return np;
    }

    // kohya 파일에서 rank/alpha 읽기 (create 전, --lora로 기존 파일을 걸 때). conv_rank = k>1 합성곱 down의 출력 채널
    static void probe(StFile & st, int & rank, int & conv_rank) {
        rank = 0, conv_rank = 0;
        for (auto & [k, e] : st.tensors) {
            if (k.find(".lora_down.weight") == std::string::npos) continue;
            const bool kconv = e.shape.size() == 4 && (e.shape[2] > 1 || e.shape[3] > 1);
            (kconv ? conv_rank : rank) = (int) e.shape[0];
        }
    }

    static std::string kohya_key(const std::string & name) {
        std::string k = "lora_unet_" + name;
        for (char & c : k) if (c == '.') c = '_';
        return k;
    }

    // kohya 파일에서 down/up 값과 alpha를 읽어 온다 (create 후). 파일의 alpha는 (어텐션/1x1/Linear), (k>1 합성곱)
    // 두 묶음 안에서 각각 같아야 함 → this->alpha, conv_alpha에 들어감. 파일에 없는 모듈이 있으면 오류
    void load_kohya(StFile & st) {
        FILE * f = fopen(st.path.c_str(), "rb");
        float  fa[2] = {-1, -1};
        auto   read  = [&](const std::string & key, std::vector<uint8_t> & raw) -> const StEntry & {
            auto it = st.tensors.find(key);
            if (it == st.tensors.end()) throw std::runtime_error("LoRA 키 없음: " + key);
            raw.resize(it->second.end - it->second.begin);
            fseeko(f, (off_t) (st.data_start + it->second.begin), SEEK_SET);
            if (fread(raw.data(), 1, raw.size(), f) != raw.size()) throw std::runtime_error("읽기 실패: " + key);
            return it->second;
        };
        std::vector<uint8_t> raw;
        for (auto & md : mods) {
            const std::string k = kohya_key(md.name);
            for (auto [t, suf] : {std::pair{md.down, ".lora_down.weight"}, std::pair{md.up, ".lora_up.weight"}}) {
                const StEntry & e = read(k + suf, raw);
                if (raw.size() != ggml_nelements(t) * ggml_type_size(StFile::to_ggml(e.dtype))) throw std::runtime_error("모양 다름: " + k + suf);
                st_upload(t, StFile::to_ggml(e.dtype), raw);
            }
            const StEntry & e = read(k + ".alpha", raw);
            float           a;
            if (e.dtype == "F32") memcpy(&a, raw.data(), 4);
            else a = ggml_fp16_to_fp32(*(const ggml_fp16_t *) raw.data());
            const int g = md.conv && md.down->ne[0] * md.down->ne[1] > 1;  // k>1 합성곱
            if (fa[g] >= 0 && a != fa[g]) throw std::runtime_error("같은 묶음 안에서 모듈마다 alpha가 다름 (미지원): " + k);
            fa[g]    = a;
            md.alpha = a;
        }
        fclose(f);
        if (fa[0] >= 0) alpha = fa[0];
        if (fa[1] >= 0) conv_alpha = fa[1];
    }

    float scale() const { return alpha / (float) rank; }
    float conv_scale() const { return conv_rank > 0 ? conv_alpha / (float) conv_rank : 1.0f; }

    // kohya 호환 safetensors (lora_unet_<이름의 . -> _>.lora_down.weight / .lora_up.weight / .alpha)
    void save(const std::string & path, int step) const {
        struct Item {
            std::string          key;
            std::vector<int64_t> shape;
            std::vector<float>   data;
        };
        std::vector<Item> items;
        for (auto & md : mods) {
            std::string k = "lora_unet_" + md.name;
            for (char & c : k) if (c == '.') c = '_';
            for (auto [t, suf] : {std::pair{md.down, ".lora_down.weight"}, std::pair{md.up, ".lora_up.weight"}}) {
                std::vector<int64_t> shape = md.conv ? std::vector<int64_t>{t->ne[3], t->ne[2], t->ne[1], t->ne[0]}  // [out, in, kh, kw]
                                                     : std::vector<int64_t>{t->ne[1], t->ne[0]};
                Item it{k + suf, shape, std::vector<float>(ggml_nelements(t))};
                ggml_backend_tensor_get(t, it.data.data(), 0, ggml_nbytes(t));
                items.push_back(std::move(it));
            }
            items.push_back({k + ".alpha", {}, {md.alpha}});
        }
        std::sort(items.begin(), items.end(), [](auto & a, auto & b) { return a.key < b.key; });
        nlohmann::json h;
        uint64_t       off = 0;
        for (auto & it : items) {
            const uint64_t n = it.data.size() * 4;
            h[it.key]        = {{"dtype", "F32"}, {"shape", it.shape}, {"data_offsets", {off, off + n}}};
            off += n;
        }
        h["__metadata__"] = {{"ss_network_module", "networks.lora"}, {"ss_network_dim", std::to_string(rank)},
                             {"ss_network_alpha", std::to_string(alpha)}, {"ss_base_model_version", "sdxl_base_v1-0"},
                             {"ss_steps", std::to_string(step)}, {"trainer", "sdxl-vulkan (ggml)"}};
        if (conv_rank > 0) {  // kohya와 같게 network_args에 conv_dim/conv_alpha (JSON 문자열)
            h["__metadata__"]["ss_network_args"] =
                nlohmann::json{{"conv_dim", std::to_string(conv_rank)}, {"conv_alpha", std::to_string(conv_alpha)}}.dump();
        }
        std::string hs = h.dump();
        while (hs.size() % 8) hs += ' ';
        std::filesystem::path pp(path);
        if (pp.has_parent_path()) std::filesystem::create_directories(pp.parent_path());
        const std::string tmp = path + ".tmp";
        FILE *            f   = fopen(tmp.c_str(), "wb");
        if (!f) throw std::runtime_error("쓰기 실패: " + tmp);
        const uint64_t hl = hs.size();
        fwrite(&hl, 8, 1, f);
        fwrite(hs.data(), 1, hs.size(), f);
        for (auto & it : items) fwrite(it.data.data(), 4, it.data.size(), f);
        fclose(f);
        std::filesystem::rename(tmp, path);
        printf("저장: %s (%zu 텐서, step %d)\n", path.c_str(), items.size(), step);
    }
};

// 중간 저장본: out/x.safetensors -> out/x_s200.safetensors (최종본은 원래 이름)
static std::string step_path(const std::string & out, int step) {
    const std::string suf = ".safetensors";
    std::string       base = out;
    if (base.size() > suf.size() && base.compare(base.size() - suf.size(), suf.size(), suf) == 0) base.resize(base.size() - suf.size());
    return base + "_s" + std::to_string(step) + suf;
}

struct Sample {
    std::string name;
    Npy         latent, crossattn, vector;
};

static std::vector<Sample> load_samples(const std::string & dir) {
    std::vector<Sample> out;
    std::vector<std::string> names;
    for (auto & e : std::filesystem::directory_iterator(dir)) {
        const std::string fn = e.path().filename().string();
        const std::string suf = ".latent.npy";
        if (fn.size() > suf.size() && fn.compare(fn.size() - suf.size(), suf.size(), suf) == 0) {
            names.push_back(fn.substr(0, fn.size() - suf.size()));
        }
    }
    std::sort(names.begin(), names.end());
    for (auto & n : names) {
        const std::string b = dir + "/" + n;
        out.push_back({n, npy_load(b + ".latent.npy"), npy_load(b + ".crossattn.npy"), npy_load(b + ".vector.npy")});
    }
    return out;
}

static long gtt_mib() {
    FILE * f = fopen("/sys/class/drm/card0/device/mem_info_gtt_used", "r");
    if (!f) return -1;
    long long v = 0;
    if (fscanf(f, "%lld", &v) != 1) v = 0;
    fclose(f);
    return (long) (v / 1048576);
}

static int check_ref(const Opts & O, ggml_backend_t backend) {
    const std::string d = O.check_ref + "/";
    Npy x = npy_load(d + "x.npy"), dref = npy_load(d + (O.lora_in.empty() ? "d.npy" : "d_lora.npy")), ca = npy_load(d + "crossattn.npy"),
        vec = npy_load(d + "vector.npy"), prm = npy_load(d + "params.npy");
    const float sigma = prm.data[0], c_in = prm.data[1], c_out = prm.data[2], c_skip = prm.data[3];
    DiscreteSchedule sch;
    const float      t = (float) sch.sigma_to_t(sigma);
    printf("sigma=%.5f c_in=%.5f c_out=%.5f c_skip=%.3f -> timestep %.3f\n", sigma, c_in, c_out, c_skip, t);

    StFile      st(O.model);
    UNetWeights w;
    w.load(st, backend);

    LoraSet lora;
    float   lora_scale = 1.0f, conv_lora_scale = 1.0f;
    if (!O.lora_in.empty()) {
        StFile lf(O.lora_in);
        int    rank = 0, conv_rank = 0;
        LoraSet::probe(lf, rank, conv_rank);
        std::mt19937 rng(0);
        lora.create(w, rank, (float) rank, conv_rank, (float) conv_rank, backend, rng);
        lora.load_kohya(lf);
        lora_scale      = lora.scale();
        conv_lora_scale = lora.conv_scale();
        printf("LoRA 로드: %s rank %d alpha %.3f scale %.4f, conv_rank %d conv_alpha %.3f scale %.4f\n", O.lora_in.c_str(),
               rank, lora.alpha, lora_scale, conv_rank, lora.conv_alpha, conv_lora_scale);
    }

    const int64_t H = x.shape[2], Wd = x.shape[3], M = ca.shape[1], CTX = ca.shape[2];
    ggml_context * C = ggml_init({ggml_tensor_overhead() * 65536 + ggml_graph_overhead_custom(65536, false), nullptr, true});
    ggml_tensor * in_x   = ggml_new_tensor_4d(C, GGML_TYPE_F32, Wd, H, 4, 1);
    ggml_tensor * in_t   = ggml_new_tensor_1d(C, GGML_TYPE_F32, 1);
    ggml_tensor * in_ctx = ggml_new_tensor_2d(C, GGML_TYPE_F32, CTX, M);
    ggml_tensor * in_y   = ggml_new_tensor_2d(C, GGML_TYPE_F32, vec.numel(), 1);
    for (ggml_tensor * i : {in_x, in_t, in_ctx, in_y}) ggml_set_input(i);

    UNetBuilder B;
    B.C        = C;
    B.cfg      = UNetConfig::sdxl();
    B.W        = [&](const std::string & n) {
        auto it = lora.T.find(n);
        return it != lora.T.end() ? it->second : w.get(n);
    };
    B.has        = [&](const std::string & n) { return lora.T.count(n) > 0 || w.T.count(n) > 0; };
    B.native     = O.native;
    B.conv_f16   = O.conv_f16;
    B.lora_scale = lora_scale;
    B.conv_lora_scale = conv_lora_scale;
    ggml_tensor * eps = B.unet(in_x, in_t, in_ctx, in_y);
    ggml_set_output(eps);

    ggml_cgraph * gf = ggml_new_graph_custom(C, 65536, false);
    ggml_build_forward_expand(gf, eps);
    int unsup = 0;
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) unsup++;
    }
    printf("nodes=%d unsupported=%d\n", ggml_graph_n_nodes(gf), unsup);
    if (unsup) return 2;

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) {
        fprintf(stderr, "그래프 메모리 할당 실패\n");
        return 2;
    }
    printf("compute buffer: %.1f MiB\n", ggml_gallocr_get_buffer_size(ga, 0) / 1048576.0);

    std::vector<float> xs(x.data);
    for (float & v : xs) v *= c_in;
    ggml_backend_tensor_set(in_x, xs.data(), 0, xs.size() * 4);
    ggml_backend_tensor_set(in_t, &t, 0, 4);
    ggml_backend_tensor_set(in_ctx, ca.data.data(), 0, ca.data.size() * 4);
    ggml_backend_tensor_set(in_y, vec.data.data(), 0, vec.data.size() * 4);
    for (ggml_tensor * z : B.zero_leafs) {
        std::vector<float> zeros(ggml_nelements(z), 0.0f);
        ggml_backend_tensor_set(z, zeros.data(), 0, ggml_nbytes(z));
    }

    const double t0 = now_s();
    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "계산 실패\n");
        return 2;
    }
    printf("순전파 %.2fs\n", now_s() - t0);

    std::vector<float> e(ggml_nelements(eps)), eref(e.size());
    ggml_backend_tensor_get(eps, e.data(), 0, e.size() * 4);
    for (size_t i = 0; i < e.size(); i++) eref[i] = (dref.data[i] - c_skip * x.data[i]) / c_out;
    const double err = rel_l2(e, eref);
    printf("eps rel_l2 vs sd.cpp = %.3e  (eps std ours %.4f)\n", err,
           sqrt(std::inner_product(e.begin(), e.end(), e.begin(), 0.0) / e.size()));
    npy_save(d + (O.lora_in.empty() ? "eps_ours.npy" : "eps_ours_lora.npy"), {1, 4, H, Wd}, e.data());

    ggml_gallocr_free(ga);
    ggml_free(C);
    return err < 1e-2 ? 0 : 1;
}

// NaN 디버그: 순전파+역전파 노드를 하나씩 (v0.3: 역전파까지) 계산해 처음 비유한 값이 나온 노드를 보고 (SDXL_NAN_DEBUG=1)
static float max_abs_finite(ggml_tensor * t, bool & bad) {
    bad = false;
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) return -1;
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    float m = 0;
    for (float x : v) {
        if (!std::isfinite(x)) bad = true;
        else m = std::max(m, fabsf(x));
    }
    return m;
}

static void nan_debug(ggml_backend_t backend, ggml_cgraph * gf, int n_fwd) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        ggml_tensor * n  = ggml_graph_node(gf, i);
        ggml_context * gc = ggml_init({ggml_graph_overhead_custom(1, false) + 1024, nullptr, true});
        ggml_cgraph *  gv = ggml_new_graph_custom(gc, 1, false);
        ggml_graph_add_node(gv, n);
        ggml_backend_graph_compute(backend, gv);
        ggml_free(gc);
        bool        bad;
        const float m = max_abs_finite(n, bad);
        if (bad) {
            printf("첫 비유한 노드 #%d (%s) %s (%s) %s [%lld,%lld,%lld,%lld]\n", i, i < n_fwd ? "순전파" : "역전파", ggml_op_desc(n), n->name,
                   ggml_type_name(n->type), (long long) n->ne[0],
                   (long long) n->ne[1], (long long) n->ne[2], (long long) n->ne[3]);
            for (int j = 0; j < GGML_MAX_SRC && n->src[j]; j++) {
                bool        sb;
                const float sm = max_abs_finite(n->src[j], sb);
                printf("  src%d %s (%s) %s max|x|=%g%s\n", j, ggml_op_desc(n->src[j]), n->src[j]->name,
                       ggml_type_name(n->src[j]->type), sm, sb ? " (비유한 포함)" : "");
            }
            return;
        }
        if (getenv("SDXL_NAN_TRACE") && m > 1e4f) printf("  큰 값 #%d %s (%s) max|x|=%g\n", i, ggml_op_desc(n), n->name, m);
    }
    printf("비유한 값 없음 (노드 %d개)\n", ggml_graph_n_nodes(gf));
}

// ---------- 메모리 계획 (할당 없이) ----------
static std::string op_key(const ggml_tensor * t) {
    std::string k = ggml_op_desc(t);
    // 어디서 온 텐서인지 대략: 이름에 블록 경로가 없으므로 연산 + 모양 계열로 묶는다
    return k;
}

static int mem_plan(const Opts & O) {
    const int64_t S = O.mem_plan;
    StFile        st(O.model);
    UNetWeights   w;
    w.load(st, nullptr);
    std::mt19937 rng(0);
    LoraSet      lora;
    lora.create(w, O.rank, O.alpha, O.conv_rank, O.conv_alpha, nullptr, rng);

    const size_t   GS = 131072;
    ggml_context * C  = ggml_init({ggml_tensor_overhead() * GS + ggml_graph_overhead_custom(GS, true), nullptr, true});
    ggml_tensor * in_x     = ggml_new_tensor_4d(C, GGML_TYPE_F32, S, S, 4, 1);
    ggml_tensor * in_t     = ggml_new_tensor_1d(C, GGML_TYPE_F32, 1);
    ggml_tensor * in_ctx   = ggml_new_tensor_2d(C, GGML_TYPE_F32, 2048, 77);
    ggml_tensor * in_y     = ggml_new_tensor_2d(C, GGML_TYPE_F32, 2816, 1);
    ggml_tensor * in_noise = ggml_new_tensor_4d(C, GGML_TYPE_F32, S, S, 4, 1);
    for (ggml_tensor * i : {in_x, in_t, in_ctx, in_y, in_noise}) ggml_set_input(i);
    UNetBuilder B;
    B.C          = C;
    B.cfg        = UNetConfig::sdxl();
    B.W          = [&](const std::string & n) {
        auto it = lora.T.find(n);
        return it != lora.T.end() ? it->second : w.get(n);
    };
    B.has        = [&](const std::string & n) { return lora.T.count(n) > 0 || w.T.count(n) > 0; };
    B.native     = O.native;
    B.conv_f16   = O.conv_f16;
    B.lora_scale = lora.scale();
    B.conv_lora_scale = lora.conv_scale();
    ggml_tensor * eps  = B.unet(in_x, in_t, in_ctx, in_y);
    ggml_tensor * loss = ggml_scale(C, ggml_sum(C, ggml_sqr(C, ggml_sub(C, eps, in_noise))), 1.0f / (float) ggml_nelements(eps));
    ggml_set_output(loss);
    ggml_set_loss(loss);
    ggml_cgraph * gf = ggml_new_graph_custom(C, GS, true);
    ggml_build_forward_expand(gf, loss);
    const int n_fwd = ggml_graph_n_nodes(gf);
    ggml_build_backward_expand(C, gf, nullptr);
    for (size_t i = 0; i < lora.params.size(); i++) {
        ggml_build_forward_expand(gf, ggml_opt_step_adamw(C, lora.params[i], ggml_graph_get_grad(gf, lora.params[i]),
                                                          lora.m[i], lora.v[i], lora.adamw));
    }
    const int n_all = ggml_graph_n_nodes(gf);

    // 순전파만 그래프의 버퍼 크기 (비교용)
    ggml_cgraph * gfwd = ggml_new_graph_custom(C, GS, false);
    ggml_build_forward_expand(gfwd, loss);

    ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
    size_t                     sz_all = 0, sz_fwd = 0;
    {
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        ggml_gallocr_reserve_n_size(ga, gfwd, nullptr, nullptr, &sz_fwd);
        ggml_gallocr_free(ga);
    }
    {
        ggml_gallocr_t ga = ggml_gallocr_new(buft);
        ggml_gallocr_reserve_n_size(ga, gf, nullptr, nullptr, &sz_all);
        ggml_gallocr_free(ga);
    }

    // 역전파 노드가 참조하는 순전파 텐서 = 역전파까지 살아 있어야 하는 활성값 (뷰는 원본으로)
    std::map<const ggml_tensor *, int> fwd_idx;
    for (int i = 0; i < n_fwd; i++) fwd_idx[ggml_graph_node(gf, i)] = i;
    std::set<const ggml_tensor *> saved;
    for (int i = n_fwd; i < n_all; i++) {
        const ggml_tensor * n = ggml_graph_node(gf, i);
        for (int j = 0; j < GGML_MAX_SRC && n->src[j]; j++) {
            const ggml_tensor * s = n->src[j];
            while (s->view_src) s = s->view_src;
            if (fwd_idx.count(s)) saved.insert(s);
        }
    }
    std::map<std::string, std::pair<int, double>> by_op;
    double                                         tot = 0;
    for (const ggml_tensor * t : saved) {
        auto & e = by_op[op_key(t)];
        e.first++;
        e.second += ggml_nbytes(t) / 1048576.0;
        tot += ggml_nbytes(t) / 1048576.0;
    }
    std::vector<std::pair<double, std::string>> rows;
    for (auto & [k, e] : by_op) {
        char line[200];
        snprintf(line, sizeof(line), "  %9.1f MiB  %5d  %s", e.second, e.first, k.c_str());
        rows.push_back({e.second, line});
    }
    std::sort(rows.rbegin(), rows.rend());

    printf("mem-plan: 잠재값 %lldx%lld (%lldpx), native=[", (long long) S, (long long) S, (long long) S * 8);
    for (auto & n : O.native) printf("%s ", n.c_str());
    printf("] conv_f16=%d rank=%d conv_rank=%d\n", O.conv_f16, O.rank, O.conv_rank);
    printf("nodes fwd=%d total=%d\n", n_fwd, n_all);
    printf("계산 버퍼: 순전파만 %.1f MiB, 학습(순전파+역전파+AdamW) %.1f MiB\n", sz_fwd / 1048576.0, sz_all / 1048576.0);
    printf("예상 GTT ≈ 가중치 4900 + LoRA/AdamW %.0f + 계산 버퍼 %.0f = %.0f MiB (+ 컨테이너 약 1200)\n",
           lora.params.size() ? 3.0 * 4 * 11.61e6 / 1048576.0 : 0.0, sz_all / 1048576.0,
           4900 + 3.0 * 4 * 11.61e6 / 1048576.0 + sz_all / 1048576.0);
    printf("역전파용으로 붙잡는 순전파 텐서: %zu개, 합 %.1f MiB (할당기 재사용 전 단순 합)\n", saved.size(), tot);
    for (size_t i = 0; i < rows.size() && i < 25; i++) printf("%s\n", rows[i].second.c_str());
    ggml_free(C);
    return 0;
}

static int train(const Opts & O, ggml_backend_t backend) {
    std::vector<Sample> data = load_samples(O.train_dir);
    if (data.empty()) throw std::runtime_error("샘플 없음: " + O.train_dir);
    const int64_t H = data[0].latent.shape[2], Wd = data[0].latent.shape[3];
    const int64_t M = data[0].crossattn.shape[1], CTX = data[0].crossattn.shape[2], ADM = data[0].vector.numel();
    for (auto & s : data) {
        if (s.latent.shape[2] != H || s.latent.shape[3] != Wd) throw std::runtime_error("샘플 크기가 다름: " + s.name);
    }
    printf("샘플 %zu개, 잠재값 %lldx%lld (%lldpx)\n", data.size(), (long long) Wd, (long long) H, (long long) Wd * 8);

    std::mt19937 rng(O.seed);
    StFile       st(O.model);
    UNetWeights  w;
    w.load(st, backend);
    LoraSet lora;
    lora.create(w, O.rank, O.alpha, O.conv_rank, O.conv_alpha, backend, rng);
    if (!O.lora_in.empty()) {
        StFile      lf(O.lora_in);
        lora.load_kohya(lf);
        printf("이어받기: %s (alpha %.3f, conv_alpha %.3f)\n", O.lora_in.c_str(), lora.alpha, lora.conv_alpha);
        if (lora.alpha != O.alpha || lora.conv_alpha != O.conv_alpha) throw std::runtime_error("--alpha/--conv-alpha가 파일과 다름");
    }

    const size_t   GS = 131072;
    ggml_context * C  = ggml_init({ggml_tensor_overhead() * GS + ggml_graph_overhead_custom(GS, true), nullptr, true});
    ggml_tensor * in_x     = ggml_new_tensor_4d(C, GGML_TYPE_F32, Wd, H, 4, 1);
    ggml_tensor * in_t     = ggml_new_tensor_1d(C, GGML_TYPE_F32, 1);
    ggml_tensor * in_ctx   = ggml_new_tensor_2d(C, GGML_TYPE_F32, CTX, M);
    ggml_tensor * in_y     = ggml_new_tensor_2d(C, GGML_TYPE_F32, ADM, 1);
    ggml_tensor * in_noise = ggml_new_tensor_4d(C, GGML_TYPE_F32, Wd, H, 4, 1);
    for (ggml_tensor * i : {in_x, in_t, in_ctx, in_y, in_noise}) ggml_set_input(i);

    UNetBuilder B;
    B.C          = C;
    B.cfg        = UNetConfig::sdxl();
    B.W          = [&](const std::string & n) {
        auto it = lora.T.find(n);
        return it != lora.T.end() ? it->second : w.get(n);
    };
    B.has        = [&](const std::string & n) { return lora.T.count(n) > 0 || w.T.count(n) > 0; };
    B.native     = O.native;
    B.conv_f16   = O.conv_f16;
    B.lora_scale = lora.scale();
    B.conv_lora_scale = lora.conv_scale();
    ggml_tensor * eps  = B.unet(in_x, in_t, in_ctx, in_y);
    ggml_tensor * diff = ggml_sub(C, eps, in_noise);
    ggml_tensor * loss = ggml_scale(C, ggml_sum(C, ggml_sqr(C, diff)), 1.0f / (float) ggml_nelements(eps));
    ggml_set_output(loss);
    ggml_set_loss(loss);

    ggml_cgraph * gf = ggml_new_graph_custom(C, GS, true);
    ggml_build_forward_expand(gf, loss);
    const int n_fwd = ggml_graph_n_nodes(gf);
    ggml_build_backward_expand(C, gf, nullptr);
    // v0.3: LoRA 기울기를 f32 저장소로 복사 → 유한값 검사 → (정상일 때만) 1/S 후 AdamW. 그래프 3개
    ggml_context * PC = ggml_init({ggml_tensor_overhead() * (lora.params.size() + 4), nullptr, true});
    std::vector<ggml_tensor *> gstore;
    for (ggml_tensor * p : lora.params) gstore.push_back(ggml_dup_tensor(PC, p));
    ggml_tensor *         inv_s = ggml_new_tensor_1d(PC, GGML_TYPE_F32, 1);
    ggml_backend_buffer_t pbuf  = ggml_backend_alloc_ctx_tensors(PC, backend);
    if (!pbuf) throw std::runtime_error("기울기 저장소 할당 실패");
    for (size_t i = 0; i < lora.params.size(); i++) {
        ggml_tensor * g = ggml_graph_get_grad(gf, lora.params[i]);
        if (!g) throw std::runtime_error(std::string("그래디언트 없음: ") + lora.params[i]->name);
        ggml_build_forward_expand(gf, ggml_cpy(C, g, gstore[i]));
    }
    ggml_context * OC = ggml_init({ggml_tensor_overhead() * GS + 2 * ggml_graph_overhead_custom(GS, false), nullptr, true});
    ggml_cgraph *  og = ggml_new_graph_custom(OC, GS, false);
    for (size_t i = 0; i < lora.params.size(); i++) {
        ggml_tensor * g = ggml_mul(OC, gstore[i], inv_s);
        ggml_build_forward_expand(og, ggml_opt_step_adamw(OC, lora.params[i], g, lora.m[i], lora.v[i], lora.adamw));
    }
    ggml_cgraph * cg   = ggml_new_graph_custom(OC, GS, false);
    ggml_tensor * gchk = build_finite_check(OC, cg, gstore);
    int unsup = 0;
    for (ggml_cgraph * g : {gf, og, cg}) {
        for (int i = 0; i < ggml_graph_n_nodes(g); i++) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(g, i))) unsup++;
        }
    }
    printf("nodes: fwd=%d total=%d unsupported=%d\n", n_fwd, ggml_graph_n_nodes(gf), unsup);
    if (unsup) return 2;

    ggml_gallocr_t ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga, gf)) throw std::runtime_error("그래프 메모리 할당 실패");
    ggml_gallocr_t ga_o = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_t ga_c = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(ga_o, og) || !ggml_gallocr_alloc_graph(ga_c, cg)) throw std::runtime_error("AdamW/검사 그래프 할당 실패");
    printf("compute buffer: %.1f MiB (+AdamW %.1f, 검사 %.1f), GTT %ld MiB\n", ggml_gallocr_get_buffer_size(ga, 0) / 1048576.0,
           ggml_gallocr_get_buffer_size(ga_o, 0) / 1048576.0, ggml_gallocr_get_buffer_size(ga_c, 0) / 1048576.0, gtt_mib());
    LossScaler LS;
    LS.init(O, fp16_matmul_active(O));
    int opt_step = 0;

    ggml_graph_reset(gf);  // 기울기 누적 0 (처음 한 번만)
    for (auto * vec : {&lora.m, &lora.v}) {  // AdamW 모멘트 = 0 (v0.3: AdamW가 별도 그래프라 graph_reset이 안 해 줌)
        for (ggml_tensor * t : *vec) {
            std::vector<uint8_t> z(ggml_nbytes(t), 0);
            ggml_backend_tensor_set(t, z.data(), 0, z.size());
        }
    }
    ggml_tensor * loss_grad = ggml_graph_get_grad_acc(gf, loss);

    DiscreteSchedule                    sch;
    std::uniform_int_distribution<int>  Ut(O.t_min, O.t_max);
    std::uniform_int_distribution<int>  Us(0, (int) data.size() - 1);
    std::normal_distribution<float>     N01(0.0f, 1.0f);
    const float                         beta1 = 0.9f, beta2 = 0.999f, aeps = 1e-8f, wd = 0.01f;
    std::vector<float>                  xt(Wd * H * 4), noise(xt.size());
    double                              ema = -1, t_sum = 0, l_sum = 0;

    for (int step = 1; step <= O.steps; step++) {
        const Sample & s  = data[Us(rng)];
        const int      ti = Ut(rng);
        const double   ac = sch.alphas_cumprod[ti];
        for (size_t i = 0; i < xt.size(); i++) {
            noise[i] = N01(rng);
            xt[i]    = (float) (sqrt(ac) * s.latent.data[i] + sqrt(1.0 - ac) * noise[i]);
        }
        const float tf = (float) ti;
        ggml_backend_tensor_set(in_x, xt.data(), 0, xt.size() * 4);
        ggml_backend_tensor_set(in_noise, noise.data(), 0, noise.size() * 4);
        ggml_backend_tensor_set(in_t, &tf, 0, 4);
        ggml_backend_tensor_set(in_ctx, s.crossattn.data.data(), 0, s.crossattn.data.size() * 4);
        ggml_backend_tensor_set(in_y, s.vector.data.data(), 0, s.vector.data.size() * 4);
        for (ggml_tensor * z : B.zero_leafs) {
            std::vector<float> zeros(ggml_nelements(z), 0.0f);
            ggml_backend_tensor_set(z, zeros.data(), 0, ggml_nbytes(z));
        }
        const float S_used = LS.S;
        ggml_backend_tensor_set(loss_grad, &S_used, 0, 4);

        const double t0 = now_s();
        if (getenv("SDXL_NAN_DEBUG") && step == O.steps) {
            nan_debug(backend, gf, n_fwd);
            return 3;
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("계산 실패");
        float l = 0;
        ggml_backend_tensor_get(loss, &l, 0, 4);
        if (!std::isfinite(l)) throw std::runtime_error("손실이 NaN/Inf (step " + std::to_string(step) + ", t=" + std::to_string(ti) + ", " + s.name + ")");
        if (ggml_backend_graph_compute(backend, cg) != GGML_STATUS_SUCCESS) throw std::runtime_error("계산 실패");
        float gsum = 0;
        ggml_backend_tensor_get(gchk, &gsum, 0, 4);
        if (LS.update(std::isfinite(gsum), step)) {
            opt_step++;
            const float is = 1.0f / S_used;
            ggml_backend_tensor_set(inv_s, &is, 0, 4);
            const float ap[7] = {O.lr, beta1, beta2, aeps, wd, 1.0f / (1.0f - powf(beta1, (float) opt_step)),
                                 1.0f / (1.0f - powf(beta2, (float) opt_step))};
            ggml_backend_tensor_set(lora.adamw, ap, 0, sizeof(ap));
            if (ggml_backend_graph_compute(backend, og) != GGML_STATUS_SUCCESS) throw std::runtime_error("계산 실패");
        }
        const double dt = now_s() - t0;
        if (step > 1) t_sum += dt;
        ema = ema < 0 ? l : 0.95 * ema + 0.05 * l;
        l_sum += l;
        printf("step %4d  t=%3d  loss %.5f  ema %.5f  %.2fs  GTT %ld MiB  [%s]  S=%g\n", step, ti, l, ema, dt, gtt_mib(),
               s.name.c_str(), S_used);
        fflush(stdout);
        if (O.save_every > 0 && step % O.save_every == 0 && step != O.steps) {
            lora.save(step_path(O.out, step), step);
        }
    }
    if (O.steps > 1) printf("평균 스텝 시간 (첫 스텝 제외): %.2fs\n", t_sum / (O.steps - 1));
    printf("평균 손실: %.5f\n", l_sum / O.steps);
    printf("손실 스케일: 마지막 S=%g, 건너뛴 스텝 %d (AdamW 갱신 %d번)\n", LS.S, LS.n_skip, opt_step);
    if (O.lr > 0) lora.save(O.out, O.steps);

    ggml_gallocr_free(ga);
    ggml_gallocr_free(ga_o);
    ggml_gallocr_free(ga_c);
    ggml_backend_buffer_free(pbuf);
    ggml_free(OC);
    ggml_free(PC);
    ggml_free(C);
    return 0;
}

// ---------- 그래디언트 체크포인팅 학습 ----------
// 순전파 그래프 F: 블록 출력 a[k]만 영구 버퍼에 복사 (중간값은 버림)
// 역전파: 블록을 뒤에서부터 하나씩 다시 계산. 블록 k 그래프의 손실 = sum(out_k * ga[k]) (head는 실제 MSE)
//   → 입력 a[k-1]과 skip 입력의 기울기를 ga[]에 더하고, 블록 안 LoRA 기울기는 g_store에 복사
// 마지막에 AdamW 그래프 한 번. 결과는 전체 그래프와 수학적으로 같다 (--grad-check로 확인)
struct CkptGraph {
    ggml_context * ctx  = nullptr;
    ggml_cgraph *  gf   = nullptr;
    ggml_tensor *  loss = nullptr;
    size_t         zl0 = 0, zl1 = 0;  // 이 그래프의 zero_leafs 범위
};

static void zero_tensor(ggml_tensor * t) {
    std::vector<uint8_t> z(ggml_nbytes(t), 0);
    ggml_backend_tensor_set(t, z.data(), 0, z.size());
}

static int train_ckpt(const Opts & O, ggml_backend_t backend_in) {
    const bool          meta = O.mem_plan > 0;  // --mem-plan --ckpt: 할당 없이 크기만
    ggml_backend_t      backend = meta ? nullptr : backend_in;
    std::vector<Sample> data;
    int64_t             H = O.mem_plan, Wd = O.mem_plan, M = 77, CTX = 2048, ADM = 2816;
    if (!meta) {
        data = load_samples(O.train_dir);
        if (data.empty()) throw std::runtime_error("샘플 없음: " + O.train_dir);
        H = data[0].latent.shape[2], Wd = data[0].latent.shape[3];
        M = data[0].crossattn.shape[1], CTX = data[0].crossattn.shape[2], ADM = data[0].vector.numel();
    }
    printf("[ckpt] 샘플 %zu개, 잠재값 %lldx%lld (%lldpx)\n", data.size(), (long long) Wd, (long long) H, (long long) Wd * 8);

    std::mt19937 rng(O.seed);
    StFile       st(O.model);
    UNetWeights  w;
    w.load(st, backend);
    LoraSet lora;
    lora.create(w, O.rank, O.alpha, O.conv_rank, O.conv_alpha, backend, rng);
    if (!meta && !O.lora_in.empty()) {
        StFile      lf(O.lora_in);
        lora.load_kohya(lf);
        printf("이어받기: %s (alpha %.3f, conv_alpha %.3f)\n", O.lora_in.c_str(), lora.alpha, lora.conv_alpha);
        if (lora.alpha != O.alpha || lora.conv_alpha != O.conv_alpha) throw std::runtime_error("--alpha/--conv-alpha가 파일과 다름");
    }

    UNetBuilder B;
    B.cfg        = UNetConfig::sdxl();
    B.W          = [&](const std::string & n) {
        auto it = lora.T.find(n);
        return it != lora.T.end() ? it->second : w.get(n);
    };
    B.has        = [&](const std::string & n) { return lora.T.count(n) > 0 || w.T.count(n) > 0; };
    B.native     = O.native;
    B.conv_f16   = O.conv_f16;
    B.lora_scale = lora.scale();
    B.conv_lora_scale = lora.conv_scale();
    const auto & blocks = B.plan();
    const int    K      = (int) blocks.size();  // 마지막 = head

    // 블록별 LoRA 유무, 그보다 앞(포함)에 LoRA가 있는지
    std::vector<bool> has_lora(K, false), lora_upto(K, false);
    for (int k = 0; k < K; k++) {
        for (auto & md : lora.mods) {
            if (md.name.rfind(blocks[k].prefix + ".", 0) == 0) has_lora[k] = true;
        }
        lora_upto[k] = has_lora[k] || (k > 0 && lora_upto[k - 1]);
    }

    // ---- 순전파 그래프 + 영구 텐서 ----
    const size_t GS = 131072;
    auto new_ctx = [&](bool grads) {
        return ggml_init({ggml_tensor_overhead() * GS + ggml_graph_overhead_custom(GS, grads), nullptr, true});
    };
    ggml_context * PC = ggml_init({ggml_tensor_overhead() * (4 * K + lora.params.size() + 16), nullptr, true});
    ggml_tensor * x_in   = ggml_new_tensor_4d(PC, GGML_TYPE_F32, Wd, H, 4, 1);
    ggml_tensor * noise  = ggml_new_tensor_4d(PC, GGML_TYPE_F32, Wd, H, 4, 1);
    ggml_tensor * t_in   = ggml_new_tensor_1d(PC, GGML_TYPE_F32, 1);
    ggml_tensor * y_in   = ggml_new_tensor_2d(PC, GGML_TYPE_F32, ADM, 1);
    ggml_tensor * ctx_in = ggml_new_tensor_2d(PC, GGML_TYPE_F32, CTX, M);
    ggml_tensor * emb_p  = ggml_new_tensor_2d(PC, GGML_TYPE_F32, B.cfg.model_channels * 4, 1);

    CkptGraph F;
    F.ctx = new_ctx(false);
    B.C   = F.ctx;
    F.zl0 = B.zero_leafs.size();
    std::vector<ggml_tensor *> outs;
    ggml_tensor *              emb = B.time_emb(t_in, y_in);
    {
        ggml_tensor * h = x_in;
        for (int k = 0; k < K - 1; k++) {
            h = B.block(k, h, blocks[k].skip >= 0 ? outs[blocks[k].skip] : nullptr, emb, ctx_in);
            outs.push_back(h);
        }
    }
    F.zl1 = B.zero_leafs.size();
    std::vector<ggml_tensor *> a(K - 1), ga(K - 1);
    for (int k = 0; k < K - 1; k++) {
        a[k]  = ggml_new_tensor(PC, GGML_TYPE_F32, 4, outs[k]->ne);
        ga[k] = ggml_new_tensor(PC, GGML_TYPE_F32, 4, outs[k]->ne);
    }
    std::vector<ggml_tensor *> gstore;
    for (ggml_tensor * p : lora.params) gstore.push_back(ggml_dup_tensor(PC, p));
    ggml_tensor * inv_s = ggml_new_tensor_1d(PC, GGML_TYPE_F32, 1);  // 1/손실 스케일 (v0.3)
    double act_mib = 0;
    for (int k = 0; k < K - 1; k++) act_mib += 2.0 * ggml_nbytes(a[k]) / 1048576.0;
    ggml_backend_buffer_t pbuf = nullptr;
    if (meta) {
        for (ggml_tensor * t = ggml_get_first_tensor(PC); t; t = ggml_get_next_tensor(PC, t)) t->data = FAKE_DATA;
    } else {
        pbuf = ggml_backend_alloc_ctx_tensors(PC, backend);
    }
    printf("[ckpt] 블록 %d개, 블록 출력+기울기 저장 %.1f MiB, 영구 버퍼 %.1f MiB\n", K, act_mib,
           pbuf ? ggml_backend_buffer_get_size(pbuf) / 1048576.0 : 0.0);

    F.gf = ggml_new_graph_custom(F.ctx, GS, false);
    ggml_build_forward_expand(F.gf, ggml_cpy(F.ctx, emb, emb_p));
    for (int k = 0; k < K - 1; k++) ggml_build_forward_expand(F.gf, ggml_cpy(F.ctx, outs[k], a[k]));

    // ---- 블록별 역전파 그래프 ----
    for (int k = 0; k < K - 1; k++) {
        if (lora_upto[k]) ggml_set_param(a[k]);  // 이 출력의 기울기가 필요한가
    }
    std::vector<int>       order;  // 실행 순서 (뒤에서부터)
    std::vector<CkptGraph> G(K);
    for (int k = K - 1; k >= 0; k--) {
        const auto & b       = blocks[k];
        const bool   need_in = k > 0 && lora_upto[k - 1];
        const bool   need_sk = b.skip >= 0 && lora_upto[b.skip];
        if (!has_lora[k] && !need_in && !need_sk) continue;
        CkptGraph & g = G[k];
        g.ctx = new_ctx(true);
        B.C   = g.ctx;
        g.zl0 = B.zero_leafs.size();
        ggml_tensor * in_h = k == 0 ? x_in : a[k - 1];
        ggml_tensor * sk   = b.skip >= 0 ? a[b.skip] : nullptr;
        ggml_tensor * out  = B.block(k, in_h, sk, emb_p, ctx_in);
        if (k == K - 1) {
            g.loss = ggml_scale(g.ctx, ggml_sum(g.ctx, ggml_sqr(g.ctx, ggml_sub(g.ctx, out, noise))), 1.0f / (float) ggml_nelements(out));
        } else {
            g.loss = ggml_sum(g.ctx, ggml_mul(g.ctx, out, ga[k]));
        }
        g.zl1 = B.zero_leafs.size();
        ggml_set_output(g.loss);
        ggml_set_loss(g.loss);
        g.gf = ggml_new_graph_custom(g.ctx, GS, true);
        ggml_build_forward_expand(g.gf, g.loss);
        ggml_build_backward_expand(g.ctx, g.gf, nullptr);
        if (need_in) ggml_build_forward_expand(g.gf, ggml_add_inplace(g.ctx, ga[k - 1], ggml_graph_get_grad(g.gf, in_h)));
        if (need_sk) ggml_build_forward_expand(g.gf, ggml_add_inplace(g.ctx, ga[b.skip], ggml_graph_get_grad(g.gf, sk)));
        int np = 0;
        for (size_t i = 0; i < lora.params.size(); i++) {
            ggml_tensor * gr = ggml_graph_get_grad(g.gf, lora.params[i]);
            if (gr) {
                ggml_build_forward_expand(g.gf, ggml_cpy(g.ctx, gr, gstore[i]));
                np++;
            }
        }
        if (has_lora[k] && np == 0) throw std::runtime_error("LoRA 기울기 없음: 블록 " + b.prefix);
        order.push_back(k);
    }

    // ---- AdamW 그래프 ----
    CkptGraph Og;
    Og.ctx = new_ctx(false);
    Og.gf  = ggml_new_graph_custom(Og.ctx, GS, false);
    for (size_t i = 0; i < lora.params.size(); i++) {
        ggml_tensor * g = ggml_mul(Og.ctx, gstore[i], inv_s);  // 손실 스케일 되돌리기
        ggml_build_forward_expand(Og.gf, ggml_opt_step_adamw(Og.ctx, lora.params[i], g, lora.m[i], lora.v[i], lora.adamw));
    }
    // ---- 기울기 유한값 검사 그래프 (v0.3) ----
    CkptGraph Cg;
    Cg.ctx = new_ctx(false);
    Cg.gf  = ggml_new_graph_custom(Cg.ctx, GS, false);
    ggml_tensor * gchk = build_finite_check(Cg.ctx, Cg.gf, gstore);

    if (meta) {
        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        auto size_of = [&](ggml_cgraph * gf) {
            size_t         sz = 0;
            ggml_gallocr_t g  = ggml_gallocr_new(buft);
            ggml_gallocr_reserve_n_size(g, gf, nullptr, nullptr, &sz);
            ggml_gallocr_free(g);
            return sz;
        };
        size_t mx = size_of(F.gf);
        printf("  순전파 F: %.1f MiB\n", mx / 1048576.0);
        int    kmax = -1;
        for (int k : order) {
            const size_t sz = size_of(G[k].gf);
            if (sz > mx) { mx = sz; kmax = k; }
        }
        const double lora_mib = 3.0 * 4 * (double) lora.n_params() / 1048576.0;
        printf("[ckpt mem-plan] 계산 버퍼 최대 %.1f MiB (블록 %s), 블록 저장 %.1f MiB\n", mx / 1048576.0,
               kmax >= 0 ? blocks[kmax].prefix.c_str() : "F", act_mib);
        printf("[ckpt mem-plan] 예상 GPU ≈ 가중치 4900 + LoRA/AdamW/기울기 %.0f + 저장 %.0f + 계산 %.0f = %.0f MiB\n",
               lora_mib * 4 / 3, act_mib, mx / 1048576.0, 4900 + lora_mib * 4 / 3 + act_mib + mx / 1048576.0);
        return 0;
    }

    int unsup = 0;
    auto count_unsup = [&](ggml_cgraph * gf) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) unsup++;
        }
    };
    count_unsup(F.gf);
    for (int k : order) count_unsup(G[k].gf);
    count_unsup(Og.gf);
    count_unsup(Cg.gf);
    printf("[ckpt] 역전파 블록 %zu개 (LoRA 없는 앞 블록은 생략), 지원 안 되는 노드 %d\n", order.size(), unsup);
    if (unsup) return 2;

    // grad-check: 같은 입력의 전체 그래프 (체크포인팅 없이)
    CkptGraph full;
    if (O.grad_check) {
        full.ctx = new_ctx(true);
        B.C      = full.ctx;
        full.zl0 = B.zero_leafs.size();
        ggml_tensor * eps = B.unet(x_in, t_in, ctx_in, y_in);
        full.loss = ggml_scale(full.ctx, ggml_sum(full.ctx, ggml_sqr(full.ctx, ggml_sub(full.ctx, eps, noise))), 1.0f / (float) ggml_nelements(eps));
        ggml_set_output(full.loss);
        ggml_set_loss(full.loss);
        full.zl1 = B.zero_leafs.size();
        full.gf  = ggml_new_graph_custom(full.ctx, GS, true);
        ggml_build_forward_expand(full.gf, full.loss);
        ggml_build_backward_expand(full.ctx, full.gf, nullptr);
        for (ggml_tensor * p : lora.params) ggml_set_output(ggml_graph_get_grad(full.gf, p));
    }

    // 모든 그래프가 할당기 하나(버퍼 하나)를 같이 쓴다. 그래프는 차례로 계산되므로 주소가 겹쳐도 된다.
    // 주의: 할당 중간에 버퍼가 커지면(재할당) 먼저 할당된 그래프의 텐서가 해제된 옛 버퍼를 가리키게 된다
    // (할당기는 data가 있는 텐서를 "이미 할당됨"으로 보고 다시 배정하지 않음 → 768px에서 assert, 512px에선 조용히 지나감).
    // 그래서 먼저 모든 그래프로 최대 크기를 예약한 뒤, 각 그래프를 딱 한 번씩 할당한다.
    std::vector<CkptGraph *> all = {&F};
    for (int k : order) all.push_back(&G[k]);
    all.push_back(&Og);
    all.push_back(&Cg);
    if (O.grad_check) all.push_back(&full);
    // 큰 버퍼는 할당기가 여러 조각(chunk)으로 잡고 그래프마다 조각 배치가 달라, 예약 후에도 재할당이 날 수 있다(768px+).
    // → 재할당이 한 번도 없는 할당 패스가 나올 때까지 반복. 매 패스 전에 그래프 소유 텐서의 주소를 지워 처음부터 배정.
    std::set<const ggml_tensor *> persistent;
    for (ggml_context * pc : {w.ctx, lora.ctx, PC}) {
        for (ggml_tensor * t = ggml_get_first_tensor(pc); t; t = ggml_get_next_tensor(pc, t)) persistent.insert(t);
    }
    // 그래프 컨텍스트 안의 텐서 = 그 그래프 소유 (영구 텐서는 w/lora/PC 컨텍스트에 있음)
    auto reset_graph = [&](ggml_context * gctx) {
        for (ggml_tensor * t = ggml_get_first_tensor(gctx); t; t = ggml_get_next_tensor(gctx, t)) {
            if (!persistent.count(t)) {
                t->data   = nullptr;
                t->buffer = nullptr;
            }
        }
    };
    ggml_gallocr_t ga_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    for (CkptGraph * g : all) {
        if (!ggml_gallocr_reserve(ga_, g->gf)) throw std::runtime_error("그래프 메모리 예약 실패");
    }
    size_t mx = 0;
    for (int pass = 1;; pass++) {
        if (pass > 5) throw std::runtime_error("그래프 할당이 안정되지 않음");
        for (CkptGraph * g : all) reset_graph(g->ctx);
        mx          = ggml_gallocr_get_buffer_size(ga_, 0);
        bool stable = true;
        for (CkptGraph * g : all) {
            if (!ggml_gallocr_alloc_graph(ga_, g->gf)) throw std::runtime_error("그래프 메모리 할당 실패");
            if (ggml_gallocr_get_buffer_size(ga_, 0) != mx) stable = false;
            mx = ggml_gallocr_get_buffer_size(ga_, 0);
        }
        if (stable) {
            if (pass > 1) printf("[ckpt] 할당 안정화: %d번째 패스\n", pass);
            break;
        }
    }
    printf("[ckpt] 계산 버퍼 %.1f MiB (그래프 %zu개 공유), GTT %ld MiB\n", mx / 1048576.0, all.size(), gtt_mib());
    LossScaler LS;
    LS.init(O, fp16_matmul_active(O));
    auto run = [&](CkptGraph & g, bool set_loss_grad) {
        for (size_t i = g.zl0; i < g.zl1; i++) zero_tensor(B.zero_leafs[i]);
        if (set_loss_grad) {
            // 손실 스케일은 실제 MSE 손실(head, grad-check 전체 그래프)에만. 중간 블록의 손실 sum(out*ga)은 위에서 받은
            // ga가 이미 S배이므로 기울기 시작값 1 (여기도 S면 블록마다 S가 곱해짐)
            const float s0 = (&g == &G[K - 1] || &g == &full) ? LS.S : 1.0f;
            ggml_backend_tensor_set(ggml_graph_get_grad_acc(g.gf, g.loss), &s0, 0, 4);
        }
        if (ggml_backend_graph_compute(backend, g.gf) != GGML_STATUS_SUCCESS) throw std::runtime_error("계산 실패");
    };
    for (ggml_tensor * m : lora.m) zero_tensor(m);
    for (ggml_tensor * v : lora.v) zero_tensor(v);

    DiscreteSchedule                   sch;
    std::uniform_int_distribution<int> Ut(O.t_min, O.t_max);
    std::uniform_int_distribution<int> Us(0, (int) data.size() - 1);
    std::normal_distribution<float>    N01(0.0f, 1.0f);
    const float                        beta1 = 0.9f, beta2 = 0.999f, aeps = 1e-8f, wd = 0.01f;
    std::vector<float>                 xt(Wd * H * 4), nz(xt.size());
    double                             ema = -1, t_sum = 0, l_sum = 0;
    int                                opt_step = 0;  // AdamW 스텝 (건너뛴 스텝은 안 셈)
    const int                          steps = O.grad_check ? 2 : O.steps;  // grad-check: 스텝 간 문제도 잡게 2번째 스텝에서 비교

    for (int step = 1; step <= steps; step++) {
        const Sample & s  = data[Us(rng)];
        const int      ti = Ut(rng);
        const double   ac = sch.alphas_cumprod[ti];
        for (size_t i = 0; i < xt.size(); i++) {
            nz[i] = N01(rng);
            xt[i] = (float) (sqrt(ac) * s.latent.data[i] + sqrt(1.0 - ac) * nz[i]);
        }
        const float tf = (float) ti;
        ggml_backend_tensor_set(x_in, xt.data(), 0, xt.size() * 4);
        ggml_backend_tensor_set(noise, nz.data(), 0, nz.size() * 4);
        ggml_backend_tensor_set(t_in, &tf, 0, 4);
        ggml_backend_tensor_set(ctx_in, s.crossattn.data.data(), 0, s.crossattn.data.size() * 4);
        ggml_backend_tensor_set(y_in, s.vector.data.data(), 0, s.vector.data.size() * 4);
        for (ggml_tensor * g : ga) zero_tensor(g);

        const double t0 = now_s();
        run(F, false);
        const double t1 = now_s();
        float        l  = 0;
        for (int k : order) {
            run(G[k], true);
            if (k == K - 1) ggml_backend_tensor_get(G[k].loss, &l, 0, 4);
        }
        const double t2 = now_s();
        if (!std::isfinite(l)) throw std::runtime_error("손실이 NaN/Inf (step " + std::to_string(step) + ", t=" + std::to_string(ti) + ", " + s.name + ")");

        if (O.grad_check && step == steps) {
            // 같은 입력으로 전체 그래프 기울기 계산 후 비교
            run(full, true);
            float lf = 0;
            ggml_backend_tensor_get(full.loss, &lf, 0, 4);
            double num = 0, den = 0, worst = 0;
            std::string wn;
            for (size_t i = 0; i < lora.params.size(); i++) {
                std::vector<float> g1(ggml_nelements(gstore[i])), g2(g1.size());
                ggml_backend_tensor_get(gstore[i], g1.data(), 0, g1.size() * 4);
                ggml_backend_tensor_get(ggml_graph_get_grad(full.gf, lora.params[i]), g2.data(), 0, g2.size() * 4);
                double n1 = 0, d1 = 0;
                for (size_t j = 0; j < g1.size(); j++) {
                    n1 += (double) (g1[j] - g2[j]) * (g1[j] - g2[j]);
                    d1 += (double) g2[j] * g2[j];
                }
                num += n1;
                den += d1;
                if (d1 > 0 && sqrt(n1 / d1) > worst) {
                    worst = sqrt(n1 / d1);
                    wn    = lora.params[i]->name;
                }
            }
            printf("[grad-check] step %d ", step);
            printf("t=%d loss ckpt %.6f / full %.6f, LoRA 기울기 rel_l2 전체 %.3e 최악 %.3e (%s), ckpt 계산 버퍼 최대 %.1f MiB\n",
                   ti, l, lf, sqrt(num / den), worst, wn.c_str(), mx / 1048576.0);
            ggml_gallocr_free(ga_);
            return sqrt(num / den) < 1e-4 ? 0 : 1;
        }

        const float S_used = LS.S;
        run(Cg, false);
        float gsum = 0;
        ggml_backend_tensor_get(gchk, &gsum, 0, 4);
        if (LS.update(std::isfinite(gsum), step)) {
            opt_step++;
            const float is = 1.0f / S_used;
            ggml_backend_tensor_set(inv_s, &is, 0, 4);
            const float ap[7] = {O.lr, beta1, beta2, aeps, wd, 1.0f / (1.0f - powf(beta1, (float) opt_step)),
                                 1.0f / (1.0f - powf(beta2, (float) opt_step))};
            ggml_backend_tensor_set(lora.adamw, ap, 0, sizeof(ap));
            run(Og, false);
        }
        const double dt = now_s() - t0;
        if (step > 1) t_sum += dt;
        ema = ema < 0 ? l : 0.95 * ema + 0.05 * l;
        l_sum += l;
        printf("step %4d  t=%3d  loss %.5f  ema %.5f  %.2fs (fwd %.2f bwd %.2f)  buf %.0f MiB  GTT %ld MiB  [%s]  S=%g\n", step, ti,
               l, ema, dt, t1 - t0, t2 - t1, mx / 1048576.0, gtt_mib(), s.name.c_str(), S_used);
        fflush(stdout);
        if (O.save_every > 0 && step % O.save_every == 0 && step != steps) lora.save(step_path(O.out, step), step);
    }
    if (steps > 1) printf("평균 스텝 시간 (첫 스텝 제외): %.2fs\n", t_sum / (steps - 1));
    printf("평균 손실: %.5f\n", l_sum / steps);
    printf("손실 스케일: 마지막 S=%g, 건너뛴 스텝 %d (AdamW 갱신 %d번)\n", LS.S, LS.n_skip, opt_step);
    if (O.lr > 0) lora.save(O.out, steps);
    ggml_gallocr_free(ga_);
    return 0;
}

int main(int argc, char ** argv) {
    Opts O;
    for (int i = 1; i < argc; i++) {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string {
            if (i + 1 >= argc) {
                fprintf(stderr, "값 없음: %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") O.model = next();
        else if (a == "--backend") O.backend = next();
        else if (a == "--check-ref") O.check_ref = next();
        else if (a == "--conv-f16") O.conv_f16 = true;
        else if (a == "--train") O.train_dir = next();
        else if (a == "--mem-plan") O.mem_plan = atoi(next().c_str());
        else if (a == "--ckpt") O.ckpt = true;
        else if (a == "--grad-check") { O.ckpt = true; O.grad_check = true; }
        else if (a == "--lora" || a == "--resume") O.lora_in = next();
        else if (a == "--out") O.out = next();
        else if (a == "--steps") O.steps = atoi(next().c_str());
        else if (a == "--lr") O.lr = (float) atof(next().c_str());
        else if (a == "--rank") O.rank = atoi(next().c_str());
        else if (a == "--alpha") O.alpha = (float) atof(next().c_str());
        else if (a == "--conv-rank" || a == "--conv-dim") O.conv_rank = atoi(next().c_str());
        else if (a == "--conv-alpha") O.conv_alpha = (float) atof(next().c_str());
        else if (a == "--seed") O.seed = atoi(next().c_str());
        else if (a == "--save-every") O.save_every = atoi(next().c_str());
        else if (a == "--t-min") O.t_min = atoi(next().c_str());
        else if (a == "--t-max") O.t_max = atoi(next().c_str());
        else if (a == "--threads") O.threads = atoi(next().c_str());
        else if (a == "--loss-scale") O.ls_mode = next();
        else if (a == "--loss-scale-init") O.ls_init = (float) atof(next().c_str());
        else if (a == "--loss-scale-window") O.ls_window = atoi(next().c_str());
        else if (a == "--native") {
            std::string s = next();
            for (size_t p = 0; p <= s.size();) {
                size_t e = s.find(',', p);
                if (e == std::string::npos) e = s.size();
                if (e > p) O.native.insert(s.substr(p, e - p));
                p = e + 1;
            }
        } else {
            fprintf(stderr, "알 수 없는 인자: %s\n", a.c_str());
            return 2;
        }
    }

    if (O.conv_rank < 0) O.conv_rank = 0;
    if (O.conv_alpha < 0) O.conv_alpha = (float) O.conv_rank;
    if (O.conv_rank == 0) O.conv_alpha = 0;

    // 계산 버퍼가 여러 조각(기본 1GiB)으로 나뉘면 체크포인팅의 그래프 여러 개가 할당기를 번갈아 쓸 때
    // 재할당이 끝없이 반복된다 (ggml-alloc은 조각별로 현재 그래프 크기에 맞춰 다시 잡음) → 한 조각에 들어가게 키운다.
    // 장치 한도(max_buffer_size)로 자동 제한됨. 직접 지정하면 그 값을 따름.
    setenv("GGML_VK_SUBALLOCATION_BLOCK_SIZE", "4294967296", 0);
    ggml_backend_t backend = nullptr;
    if (O.backend == "cpu") {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, O.threads);
    }
#ifdef SDXL_VULKAN
    else if (O.backend == "vulkan") {
        backend = ggml_backend_vk_init(0);
    }
#endif
    if (!backend) {
        fprintf(stderr, "백엔드 초기화 실패: %s\n", O.backend.c_str());
        return 2;
    }

    int rc = 2;
    try {
        if (O.mem_plan > 0) rc = O.ckpt ? train_ckpt(O, backend) : mem_plan(O);
        else if (!O.check_ref.empty()) rc = check_ref(O, backend);
        else if (!O.train_dir.empty()) rc = O.ckpt ? train_ckpt(O, backend) : train(O, backend);
        else fprintf(stderr, "모드를 지정하세요 (--check-ref <dir> | --train <dir>)\n");
    } catch (const std::exception & ex) {
        fprintf(stderr, "오류: %s\n", ex.what());
        rc = 2;
    }
    ggml_backend_free(backend);
    return rc;
}
