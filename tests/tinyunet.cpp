// 축소판 SDXL UNet을 ggml로 계산해 PyTorch 정답지(ref_tiny_unet.py가 만든 GGUF)와 비교한다.
//
// 사용: tinyunet <gguf> [--backend cpu|vulkan] [--native ops] [--fwd-only] [--conv-f16] [--threads N]
//   --native: 쉼표 목록 (norm,gn,geglu,concat,upscale,all). 지정한 연산은 ggml 원래 연산을 쓰고,
//             지정 안 한 연산은 역전파가 있는 기본 연산 조합으로 계산한다.
//             원래 연산은 아직 역전파가 없으므로 --native를 주면 보통 --fwd-only와 같이 쓴다.
//   --fwd-only: 순전파(출력, 손실)만 비교.
//   --conv-f16: im2col을 f16으로 (ggml_conv_2d/sd.cpp와 같은 조건). 기본 f32.
// 종료 코드: 0 = 허용 오차 안, 1 = 오차 초과, 2 = 실행 오류.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include "unet_ggml.h"
#ifdef SDXL_VULKAN
#include "ggml-vulkan.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>


struct Opts {
    std::string        gguf_path;
    std::string        backend = "cpu";
    std::set<std::string> native;
    bool               fwd_only = false;
    bool               conv_f16 = false;
    int                threads  = 8;
    bool               compare  = false;  // 노드별 백엔드 vs CPU 자체 오차 (입력은 매 노드 동기화)
};

static Opts O;
static std::map<std::string, ggml_tensor *> T;  // GGUF 텐서 (백엔드 버퍼에 올라감)
static ggml_context * C;                         // 그래프용 컨텍스트 (no_alloc)

// GGUF 텐서 이름 63자 제한 → ref_tiny_unet.py의 abbr()와 같은 규칙
static std::string abbr(std::string s) {
    static const std::pair<const char *, const char *> R[] = {
        {"input_blocks.", "ib."}, {"middle_block.", "mb."}, {"output_blocks.", "ob."},
        {".transformer_blocks.0.", ".tb0."}, {".lora_down", ".ld"}, {".lora_up", ".lu"}};
    for (auto & [a, b] : R) {
        for (size_t p = s.find(a); p != std::string::npos; p = s.find(a, p + strlen(b))) s.replace(p, strlen(a), b);
    }
    return s;
}

static ggml_tensor * W(const std::string & name) {
    auto it = T.find(abbr(name));
    if (it == T.end()) {
        fprintf(stderr, "텐서 없음: %s\n", name.c_str());
        exit(2);
    }
    return it->second;
}
static bool has(const std::string & name) { return T.count(abbr(name)) > 0; }

// ---------- 비교 ----------

static std::vector<float> get(ggml_tensor * t) {
    std::vector<float> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, v.size() * sizeof(float));
    return v;
}

static double rel_l2(const std::vector<float> & a, const std::vector<float> & ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < a.size(); i++) {
        num += (double) (a[i] - ref[i]) * (a[i] - ref[i]);
        den += (double) ref[i] * ref[i];
    }
    return sqrt(num / (den > 0 ? den : 1e-30));
}

int main(int argc, char ** argv) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
            O.backend = argv[++i];
        } else if (a == "--native" && i + 1 < argc) {
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos <= s.size()) {
                size_t e = s.find(',', pos);
                if (e == std::string::npos) e = s.size();
                if (e > pos) O.native.insert(s.substr(pos, e - pos));
                pos = e + 1;
            }
        } else if (a == "--conv-f16") {
            O.conv_f16 = true;
        } else if (a == "--compare") {
            O.compare = true;
        } else if (a == "--fwd-only") {
            O.fwd_only = true;
        } else if (a == "--threads" && i + 1 < argc) {
            O.threads = atoi(argv[++i]);
        } else if (O.gguf_path.empty()) {
            O.gguf_path = a;
        } else {
            fprintf(stderr, "알 수 없는 인자: %s\n", a.c_str());
            return 2;
        }
    }
    if (O.gguf_path.empty()) {
        fprintf(stderr, "사용: tinyunet <gguf> [--backend cpu|vulkan] [--native ops] [--fwd-only] [--conv-f16] [--threads N]\n");
        return 2;
    }

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

    // GGUF 읽기: 메타데이터만 읽고, 같은 모양의 텐서를 백엔드 버퍼에 만든 뒤 데이터를 복사
    ggml_context * meta = nullptr;
    gguf_context * g    = gguf_init_from_file(O.gguf_path.c_str(), {true, &meta});
    if (!g) {
        fprintf(stderr, "GGUF 열기 실패: %s\n", O.gguf_path.c_str());
        return 2;
    }
    const int64_t n_t = gguf_get_n_tensors(g);
    ggml_context * wctx = ggml_init({(size_t) (n_t + 8) * ggml_tensor_overhead(), nullptr, true});
    for (int64_t i = 0; i < n_t; i++) {
        const char *  name = gguf_get_tensor_name(g, i);
        ggml_tensor * dst  = ggml_dup_tensor(wctx, ggml_get_tensor(meta, name));
        ggml_set_name(dst, name);
        T[name] = dst;
    }
    ggml_backend_buffer_t wbuf = ggml_backend_alloc_ctx_tensors(wctx, backend);
    {
        FILE * f = fopen(O.gguf_path.c_str(), "rb");
        std::vector<char> buf;
        for (int64_t i = 0; i < n_t; i++) {
            ggml_tensor * dst = T[gguf_get_tensor_name(g, i)];
            buf.resize(ggml_nbytes(dst));
            fseek(f, (long) (gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i)), SEEK_SET);
            if (fread(buf.data(), 1, buf.size(), f) != buf.size()) {
                fprintf(stderr, "읽기 실패\n");
                return 2;
            }
            ggml_backend_tensor_set(dst, buf.data(), 0, buf.size());
        }
        fclose(f);
    }

    std::vector<ggml_tensor *> loras;
    for (auto & [name, t] : T) {
        if (name.rfind("lora.", 0) == 0) {
            loras.push_back(t);
            if (!O.fwd_only) ggml_set_param(t);
        }
    }

    // 그래프
    const size_t GSIZE = 65536;
    C = ggml_init({ggml_tensor_overhead() * GSIZE * 2 + ggml_graph_overhead_custom(GSIZE, true) * 2, nullptr, true});
    UNetBuilder B;
    B.C        = C;
    B.cfg      = UNetConfig::tiny();
    B.W        = [](const std::string & n) { return W(n); };
    B.has      = [](const std::string & n) { return has(n); };
    B.native   = O.native;
    B.conv_f16 = O.conv_f16;
    ggml_tensor * out = B.unet(W("in.x"), W("in.timestep"), W("in.context"), W("in.y"));
    ggml_set_output(out);
    ggml_tensor * diff = ggml_sub(C, out, W("in.target"));
    ggml_tensor * loss = ggml_scale(C, ggml_sum(C, ggml_sqr(C, diff)), 1.0f / (float) ggml_nelements(out));
    ggml_set_output(loss);
    if (!O.fwd_only) ggml_set_loss(loss);

    ggml_cgraph * gf = ggml_new_graph_custom(C, GSIZE, !O.fwd_only);
    ggml_build_forward_expand(gf, loss);
    const int n_fwd = ggml_graph_n_nodes(gf);
    if (!O.fwd_only) {
        ggml_build_backward_expand(C, gf, nullptr);
        for (ggml_tensor * p : loras) {
            ggml_tensor * gr = ggml_graph_get_grad(gf, p);
            if (!gr) {
                fprintf(stderr, "그래디언트 없음: %s\n", p->name);
                return 2;
            }
            ggml_set_output(gr);
        }
    }
    printf("backend=%s conv=%s native=[", O.backend.c_str(), O.conv_f16 ? "f16" : "f32");
    for (auto & s : O.native) printf("%s ", s.c_str());
    printf("] nodes: fwd=%d total=%d\n", n_fwd, ggml_graph_n_nodes(gf));

    // 백엔드가 지원 못 하는 노드 목록 (Vulkan 작업 목록 확인용)
    {
        std::map<std::string, int> unsup;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (!ggml_backend_supports_op(backend, n)) {
                char key[256];
                snprintf(key, sizeof(key), "%s %s <- %s%s %s%s", ggml_op_desc(n), ggml_type_name(n->type),
                         n->src[0] ? ggml_type_name(n->src[0]->type) : "-",
                         n->src[0] && !ggml_is_contiguous(n->src[0]) ? "(nc)" : "",
                         n->src[1] ? ggml_type_name(n->src[1]->type) : "-",
                         n->src[1] && !ggml_is_contiguous(n->src[1]) ? "(nc)" : "");
                unsup[key]++;
            }
        }
        if (!unsup.empty()) {
            printf("UNSUPPORTED on %s:\n", O.backend.c_str());
            for (auto & [k, v] : unsup) printf("  %5d  %s\n", v, k.c_str());
            return 2;
        }
    }

    if (getenv("TINY_DEBUG_NC")) {
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if ((n->op == GGML_OP_ADD || n->op == GGML_OP_SUB || n->op == GGML_OP_MUL || n->op == GGML_OP_DIV) &&
                n->src[0]->nb[0] != ggml_type_size(n->src[0]->type)) {
                printf("NC %d %s: src0=%s(op %s) src1=%s(op %s)\n", i, ggml_op_desc(n), n->src[0]->name,
                       ggml_op_desc(n->src[0]), n->src[1]->name, ggml_op_desc(n->src[1]));
            }
        }
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(galloc, gf)) {
        fprintf(stderr, "그래프 메모리 할당 실패\n");
        return 2;
    }
    printf("compute buffer: %.1f MiB\n", ggml_gallocr_get_buffer_size(galloc, 0) / 1048576.0);
    for (ggml_tensor * z : B.zero_leafs) ggml_set_zero(z);
    if (!O.fwd_only) ggml_graph_reset(gf);

    if (O.compare) {
        // 노드 하나씩 계산 → 오차 집계 → CPU 사본에 백엔드 결과를 덮어써서 다음 노드 입력을 같게 함.
        // 그래서 여기서 나오는 오차는 각 연산 자체의 오차다.
        struct CmpStat { int n = 0; double worst = 0; std::string worst_node; };
        static std::map<std::string, CmpStat> stats;
        ggml_backend_t cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, O.threads);
        auto cb = [](int idx, ggml_tensor * t1, ggml_tensor * t2, void *) -> bool {
            if (t1->type != GGML_TYPE_F32 || !ggml_is_contiguous(t1)) return true;
            const size_t n = ggml_nelements(t1);
            std::vector<float> a(n), b(n);
            ggml_backend_tensor_get(t1, a.data(), 0, n * sizeof(float));
            ggml_backend_tensor_get(t2, b.data(), 0, n * sizeof(float));
            double num = 0, den = 0;
            for (size_t i = 0; i < n; i++) {
                num += (double) (a[i] - b[i]) * (a[i] - b[i]);
                den += (double) b[i] * b[i];
            }
            const double e = den > 0 ? sqrt(num / den) : sqrt(num);
            char key[128];
            snprintf(key, sizeof(key), "%s %s <- %s %s", ggml_op_desc(t1), ggml_type_name(t1->type),
                     t1->src[0] ? ggml_type_name(t1->src[0]->type) : "-", t1->src[1] ? ggml_type_name(t1->src[1]->type) : "-");
            CmpStat & st = stats[key];
            st.n++;
            if (e > st.worst) {
                st.worst = e;
                char nm[160];
                snprintf(nm, sizeof(nm), "#%d %s [%lld,%lld,%lld,%lld]", idx, t1->name, (long long) t1->ne[0],
                         (long long) t1->ne[1], (long long) t1->ne[2], (long long) t1->ne[3]);
                st.worst_node = nm;
            }
            if (getenv("TINY_CMP_MM") && t1->op == GGML_OP_MUL_MAT && e > 1e-6) {
                const ggml_tensor * x = t1->src[0], * y = t1->src[1];
                printf("MM #%d e=%.2e  A[%lld,%lld,%lld,%lld]%s B[%lld,%lld,%lld,%lld]%s\n", idx, e,
                       (long long) x->ne[0], (long long) x->ne[1], (long long) x->ne[2], (long long) x->ne[3], ggml_is_contiguous(x) ? "" : "(nc)",
                       (long long) y->ne[0], (long long) y->ne[1], (long long) y->ne[2], (long long) y->ne[3], ggml_is_contiguous(y) ? "" : "(nc)");
            }
            ggml_backend_tensor_set(t2, a.data(), 0, n * sizeof(float));
            return true;
        };
        if (!ggml_backend_compare_graph_backend(backend, cpu, gf, cb, nullptr, nullptr, 0)) {
            fprintf(stderr, "compare 실패\n");
            return 2;
        }
        std::vector<std::pair<double, std::string>> rows;
        for (auto & [k, st] : stats) {
            char line[512];
            snprintf(line, sizeof(line), "  %9.3e  %5d  %-40s worst=%s", st.worst, st.n, k.c_str(), st.worst_node.c_str());
            rows.push_back({st.worst, line});
        }
        std::sort(rows.rbegin(), rows.rend());
        printf("per-op self error (%s vs CPU, worst rel_l2, count):\n", O.backend.c_str());
        for (auto & r : rows) printf("%s\n", r.second.c_str());
        ggml_backend_free(cpu);
    } else if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "계산 실패\n");
        return 2;
    }

    // 비교. 허용 오차: GELU는 ggml CPU가 fp16 표를 써서 1e-3 수준 오차가 생김
    // 허용 오차: f32 경로는 PyTorch f32 vs f64 차이(약 1e-6)의 수십 배까지.
    // fp16이 끼는 경로(--conv-f16, native geglu = CPU에서 fp16 GELU 표)는 느슨하게
    const bool   loose    = O.conv_f16 || B.nat("geglu");
    const double TOL_FWD  = loose ? 2e-3 : 1e-4;
    const double TOL_GRAD = loose ? 5e-3 : 2e-4;
    bool ok = true;
    const double e_out  = rel_l2(get(out), get(W("ref.out")));
    const float  l      = get(loss)[0], lref = get(W("ref.loss"))[0];
    const double e_loss = fabs(l - lref) / fabs(lref);
    printf("out  rel_l2=%.3e\nloss ggml=%.6f ref=%.6f rel=%.3e\n", e_out, l, lref, e_loss);
    ok = ok && e_out < TOL_FWD && e_loss < TOL_FWD;

    if (!O.fwd_only) {
        double worst = 0, num = 0, den = 0;
        std::string worst_name;
        for (ggml_tensor * p : loras) {
            std::vector<float> gg = get(ggml_graph_get_grad(gf, p));
            std::vector<float> gr = get(W(std::string("ref.grad.") + p->name));
            const double e = rel_l2(gg, gr);
            for (size_t i = 0; i < gg.size(); i++) {
                num += (double) (gg[i] - gr[i]) * (gg[i] - gr[i]);
                den += (double) gr[i] * gr[i];
            }
            if (e > worst) {
                worst      = e;
                worst_name = p->name;
            }
        }
        printf("lora grads: %zu tensors, overall rel_l2=%.3e, worst=%.3e (%s)\n", loras.size(), sqrt(num / den), worst,
               worst_name.c_str());
        ok = ok && sqrt(num / den) < TOL_GRAD && worst < TOL_GRAD * 4;
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    ggml_gallocr_free(galloc);
    ggml_backend_buffer_free(wbuf);
    ggml_free(C);
    ggml_free(wctx);
    ggml_free(meta);
    gguf_free(g);
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}
