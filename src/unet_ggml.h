// SDXL형 UNet을 ggml 그래프로 만드는 빌더 (tinyunet 검증용 축소판과 실제 SDXL 학습기가 같이 쓴다).
// 위상은 SGM/ldm UNetModel 규칙을 따른다:
//   input_blocks: 0 = conv_in, 레벨마다 ResBlock(+SpatialTransformer) x nrb, 마지막 레벨 빼고 downsample(.0.op)
//   middle_block: Res, ST, Res
//   output_blocks: 레벨 역순으로 (Res(+ST)) x (nrb+1), 레벨 0 빼고 각 레벨 마지막 블록에 upsample(.{1|2}.conv)
// 역전파가 없는 연산(NORM, GROUP_NORM, GEGLU, CONCAT, UPSCALE)은 기본값으로 역전파가 있는 기본 연산 조합을 쓴다.
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstring>
#include <functional>
#include <set>
#include <string>
#include <vector>

struct UNetConfig {
    int              model_channels = 320;
    std::vector<int> channel_mult   = {1, 2, 4};
    int              num_res_blocks = 2;
    std::vector<int> transformer_depth = {0, 2, 10};  // 레벨별
    int              head_dim = 64;
    int              groups   = 32;

    static UNetConfig sdxl() { return {}; }
    static UNetConfig tiny() { return {32, {1, 2}, 1, {0, 1}, 16, 8}; }
};

struct UNetBuilder {
    ggml_context *                                    C = nullptr;
    UNetConfig                                        cfg;
    std::function<ggml_tensor *(const std::string &)> W;    // 가중치 찾기 (없으면 종료)
    std::function<bool(const std::string &)>          has;  // 가중치 존재 여부
    std::set<std::string>                             native;  // norm, gn, geglu, concat, upscale, all
    bool                                              conv_f16   = false;  // im2col을 f16으로
    float                                             lora_scale = 1.0f;   // alpha / rank
    std::vector<ggml_tensor *>                        zero_leafs;          // 0으로 채워야 하는 상수 텐서

    bool nat(const char * op) const { return native.count(op) || native.count("all"); }

    // ---------- 역전파가 없는 연산의 조합 버전 ----------
    // SUB/DIV 역전파는 브로드캐스트를 처리하지 못하므로 통계값은 ggml_repeat로 먼저 펼친다.

    // x를 [n, rows]로 보고 행마다 정규화 (평균 0, 분산 1)
    ggml_tensor * norm_rows_composite(ggml_tensor * x2d, float eps) {
        const float n = (float) x2d->ne[0];
        ggml_tensor * mean = ggml_scale(C, ggml_sum_rows(C, x2d), 1.0f / n);
        ggml_tensor * xc   = ggml_sub(C, x2d, ggml_repeat(C, mean, x2d));
        ggml_tensor * var  = ggml_scale(C, ggml_sum_rows(C, ggml_sqr(C, xc)), 1.0f / n);
        ggml_tensor * sd   = ggml_sqrt(C, ggml_scale_bias(C, var, 1.0f, eps));
        return ggml_div(C, xc, ggml_repeat(C, sd, xc));
    }

    ggml_tensor * layer_norm(const std::string & p, ggml_tensor * x) {  // x: [C, N]
        ggml_tensor * y = nat("norm") ? ggml_norm(C, x, 1e-5f) : norm_rows_composite(x, 1e-5f);
        return ggml_add(C, ggml_mul(C, y, W(p + ".weight")), W(p + ".bias"));
    }

    ggml_tensor * group_norm(const std::string & p, ggml_tensor * x, float eps) {  // x: [W, H, C, 1]
        const int     G = cfg.groups;
        ggml_tensor * y;
        if (nat("gn")) {
            y = ggml_group_norm(C, x, G, eps);
        } else {
            ggml_tensor * x2 = ggml_reshape_2d(C, ggml_cont(C, x), x->ne[0] * x->ne[1] * (x->ne[2] / G), G);
            y = ggml_reshape_4d(C, norm_rows_composite(x2, eps), x->ne[0], x->ne[1], x->ne[2], x->ne[3]);
        }
        ggml_tensor * w = ggml_reshape_4d(C, W(p + ".weight"), 1, 1, x->ne[2], 1);
        ggml_tensor * b = ggml_reshape_4d(C, W(p + ".bias"), 1, 1, x->ne[2], 1);
        return ggml_add(C, ggml_mul(C, y, w), b);
    }

    // GEGLU: 앞 절반 * gelu_tanh(뒤 절반)  (sd.cpp는 ggml_geglu_swapped)
    ggml_tensor * geglu(ggml_tensor * x) {  // x: [2*inner, N]
        const int64_t inner = x->ne[0] / 2;
        ggml_tensor * hid   = ggml_view_2d(C, x, inner, x->ne[1], x->nb[1], 0);
        ggml_tensor * gate  = ggml_view_2d(C, x, inner, x->ne[1], x->nb[1], inner * ggml_element_size(x));
        if (nat("geglu")) {
            // split 형태 (gelu(gate) * hid): ggml.c의 GEGLU 역전파 규칙이 split만 지원
            return ggml_geglu_split(C, ggml_cont(C, gate), ggml_cont(C, hid));
        }
        // gelu_tanh(g) = g * sigmoid(2u),  u = sqrt(2/pi) * g * (1 + 0.044715 g^2)
        // sigmoid(z) = exp(-softplus(-z)): g/(1+exp(-2u)) 형태는 g < 약 -11에서 exp가 inf로 넘쳐
        // 순전파는 0이 나와도 역전파에서 0*inf = NaN이 된다 (SDXL 384px 학습에서 실제로 발생)
        const float   k   = sqrtf(2.0f / (float) M_PI);
        ggml_tensor * u   = ggml_scale(C, ggml_mul(C, gate, ggml_scale_bias(C, ggml_sqr(C, gate), 0.044715f, 1.0f)), k);
        ggml_tensor * sig = ggml_exp(C, ggml_neg(C, ggml_softplus(C, ggml_scale(C, u, -2.0f))));
        return ggml_mul(C, hid, ggml_mul(C, gate, sig));
    }

    ggml_tensor * concat_ch(ggml_tensor * a, ggml_tensor * b) {  // 채널(ne2) 방향, ne3 = 1
        if (nat("concat")) {
            return ggml_concat(C, a, b, 2);
        }
        ggml_tensor * z = ggml_new_tensor_4d(C, GGML_TYPE_F32, a->ne[0], a->ne[1], a->ne[2] + b->ne[2], 1);
        ggml_set_input(z);
        zero_leafs.push_back(z);
        z = ggml_acc(C, z, ggml_cont(C, a), z->nb[1], z->nb[2], z->nb[3], 0);
        z = ggml_acc(C, z, ggml_cont(C, b), z->nb[1], z->nb[2], z->nb[3], a->ne[2] * z->nb[2]);
        return z;
    }

    ggml_tensor * upscale2(ggml_tensor * x) {  // nearest 2배, x: [W, H, C, 1]
        if (nat("upscale")) {
            return ggml_upscale(C, x, 2, GGML_SCALE_MODE_NEAREST);
        }
        const int64_t w = x->ne[0], h = x->ne[1], c = x->ne[2];
        ggml_tensor * r  = ggml_reshape_4d(C, ggml_cont(C, x), 1, w, 1, h * c);
        ggml_tensor * tp = ggml_new_tensor_4d(C, GGML_TYPE_F32, 2, w, 2, h * c);
        return ggml_reshape_4d(C, ggml_repeat(C, r, tp), 2 * w, 2 * h, c, 1);
    }

    // ---------- 기본 블록 ----------

    ggml_tensor * linear(const std::string & p, ggml_tensor * x) {  // x: [in, N]
        ggml_tensor * y = ggml_mul_mat(C, W(p + ".weight"), x);
        if (has(p + ".bias")) {
            y = ggml_add(C, y, W(p + ".bias"));
        }
        return y;
    }

    ggml_tensor * lora_linear(const std::string & p, ggml_tensor * x) {
        ggml_tensor * y = linear(p, x);
        if (has("lora." + p + ".lora_down")) {
            ggml_tensor * d = ggml_mul_mat(C, W("lora." + p + ".lora_down"), x);
            ggml_tensor * u = ggml_mul_mat(C, W("lora." + p + ".lora_up"), d);
            if (lora_scale != 1.0f) {
                u = ggml_scale(C, u, lora_scale);
            }
            y = ggml_add(C, y, u);
        }
        return y;
    }

    // 커널을 src0로 둔 im2col + mul_mat. 커널이 f16이어도 역전파(OUT_PROD f16 src0)가 된다.
    // ggml_conv_2d는 im2col을 무조건 f16으로 만듦(sd.cpp도 이 경로) → 기본은 f32, conv_f16이면 sd.cpp와 같은 조건
    ggml_tensor * conv(const std::string & p, ggml_tensor * x, int stride = 1) {
        ggml_tensor * k   = W(p + ".weight");  // [KW, KH, IC, OC]
        const int     pad = (int) k->ne[0] / 2;
        ggml_tensor * col = ggml_im2col(C, k, x, stride, stride, pad, pad, 1, 1, true,
                                        conv_f16 ? GGML_TYPE_F16 : GGML_TYPE_F32);  // [N, OH, OW, IC*KH*KW]
        const int64_t ow = col->ne[1], oh = col->ne[2], oc = k->ne[3];
        ggml_tensor * y  = ggml_mul_mat(C, ggml_reshape_2d(C, k, k->ne[0] * k->ne[1] * k->ne[2], oc),
                                        ggml_reshape_2d(C, col, col->ne[0], ow * oh * col->ne[3]));  // [OC, OW*OH*N]
        y = ggml_reshape_4d(C, y, oc, ow, oh, col->ne[3]);
        y = ggml_cont(C, ggml_permute(C, y, 2, 0, 1, 3));  // [OW, OH, OC, N]
        return ggml_add(C, y, ggml_reshape_4d(C, W(p + ".bias"), 1, 1, oc, 1));
    }

    ggml_tensor * res(const std::string & p, ggml_tensor * x, ggml_tensor * emb) {
        ggml_tensor * h = conv(p + ".in_layers.2", ggml_silu(C, group_norm(p + ".in_layers.0", x, 1e-5f)));
        ggml_tensor * e = linear(p + ".emb_layers.1", ggml_silu(C, emb));  // [Cout, 1]
        h = ggml_add(C, h, ggml_reshape_4d(C, e, 1, 1, e->ne[0], 1));
        h = conv(p + ".out_layers.3", ggml_silu(C, group_norm(p + ".out_layers.0", h, 1e-5f)));
        ggml_tensor * skip = has(p + ".skip_connection.weight") ? conv(p + ".skip_connection", x) : x;
        return ggml_add(C, skip, h);
    }

    ggml_tensor * attention(const std::string & p, ggml_tensor * x, ggml_tensor * ctx) {
        // x: [C, N], ctx: [Cctx, M]
        const int     D  = cfg.head_dim;
        const int64_t ch = x->ne[0], n = x->ne[1], m = ctx->ne[1], nh = ch / D;
        ggml_tensor * q = lora_linear(p + ".to_q", x);
        ggml_tensor * k = lora_linear(p + ".to_k", ctx);
        ggml_tensor * v = lora_linear(p + ".to_v", ctx);
        q = ggml_permute(C, ggml_reshape_3d(C, q, D, nh, n), 0, 2, 1, 3);                   // [d, N, h]
        k = ggml_permute(C, ggml_reshape_3d(C, k, D, nh, m), 0, 2, 1, 3);                   // [d, M, h]
        v = ggml_cont(C, ggml_permute(C, ggml_reshape_3d(C, v, D, nh, m), 1, 2, 0, 3));     // [M, d, h]
        ggml_tensor * kq = ggml_mul_mat(C, k, q);                                           // [M, N, h]
        kq = ggml_soft_max(C, ggml_scale(C, kq, 1.0f / sqrtf((float) D)));
        ggml_tensor * o = ggml_mul_mat(C, v, kq);                                           // [d, N, h]
        o = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, o, 0, 2, 1, 3)), ch, n);         // [C, N]
        return lora_linear(p + ".to_out.0", o);
    }

    ggml_tensor * transformer_block(const std::string & tb, ggml_tensor * x, ggml_tensor * ctx) {
        ggml_tensor * n1 = layer_norm(tb + ".norm1", x);
        x = ggml_add(C, attention(tb + ".attn1", n1, n1), x);
        x = ggml_add(C, attention(tb + ".attn2", layer_norm(tb + ".norm2", x), ctx), x);
        ggml_tensor * ff = geglu(linear(tb + ".ff.net.0.proj", layer_norm(tb + ".norm3", x)));
        return ggml_add(C, linear(tb + ".ff.net.2", ff), x);
    }

    ggml_tensor * spatial_transformer(const std::string & p, ggml_tensor * x, ggml_tensor * ctx, int depth) {
        const int64_t w = x->ne[0], h = x->ne[1], ch = x->ne[2];
        ggml_tensor * x_in = x;
        x = group_norm(p + ".norm", x, 1e-6f);
        x = ggml_reshape_2d(C, ggml_cont(C, ggml_permute(C, x, 1, 2, 0, 3)), ch, w * h);  // [C, N]
        x = linear(p + ".proj_in", x);
        for (int d = 0; d < depth; d++) {
            x = transformer_block(p + ".transformer_blocks." + std::to_string(d), x, ctx);
        }
        x = linear(p + ".proj_out", x);
        x = ggml_cont(C, ggml_permute(C, ggml_reshape_4d(C, x, ch, w, h, 1), 2, 0, 1, 3));  // [W, H, C, 1]
        return ggml_add(C, x, x_in);
    }

    // ---------- 블록 단위 API (그래디언트 체크포인팅용) ----------
    // UNet을 블록 열로 본다: input_blocks 0..n_in-1, middle 1개, output_blocks 0..n_out-1, head(out) 1개.
    // input 블록 출력은 순서대로 skip 스택에 쌓이고 output 블록이 역순으로 꺼낸다.
    enum BlockKind { B_CONV_IN, B_IN_RES, B_DOWN, B_MID, B_OUT, B_HEAD };
    struct Block {
        BlockKind   kind;
        int         level = 0;
        std::string prefix;          // 예: input_blocks.4, output_blocks.2
        bool        upsample = false;
        int         skip     = -1;   // output 블록: 꺼내 쓰는 블록 번호
    };
    std::vector<Block> blocks;

    const std::vector<Block> & plan() {
        if (!blocks.empty()) return blocks;
        const int        L = (int) cfg.channel_mult.size();
        std::vector<int> stack;
        int              ib = 0;
        auto add = [&](Block b) {
            blocks.push_back(b);
            return (int) blocks.size() - 1;
        };
        stack.push_back(add({B_CONV_IN, 0, "input_blocks." + std::to_string(ib++)}));
        for (int l = 0; l < L; l++) {
            for (int r = 0; r < cfg.num_res_blocks; r++) {
                stack.push_back(add({B_IN_RES, l, "input_blocks." + std::to_string(ib++)}));
            }
            if (l != L - 1) {
                stack.push_back(add({B_DOWN, l, "input_blocks." + std::to_string(ib++)}));
            }
        }
        add({B_MID, L - 1, "middle_block"});
        int ob = 0;
        for (int l = L - 1; l >= 0; l--) {
            for (int r = 0; r <= cfg.num_res_blocks; r++) {
                Block b{B_OUT, l, "output_blocks." + std::to_string(ob++)};
                b.upsample = l > 0 && r == cfg.num_res_blocks;
                b.skip     = stack.back();
                stack.pop_back();
                add(b);
            }
        }
        add({B_HEAD, 0, "out"});
        return blocks;
    }

    // 시간+라벨 임베딩 (LoRA 없음 → 학습에선 상수)
    ggml_tensor * time_emb(ggml_tensor * t, ggml_tensor * y) {
        ggml_tensor * temb = ggml_timestep_embedding(C, t, cfg.model_channels, 10000);
        ggml_tensor * emb  = linear("time_embed.2", ggml_silu(C, linear("time_embed.0", temb)));
        return ggml_add(C, emb, linear("label_emb.0.2", ggml_silu(C, linear("label_emb.0.0", y))));
    }

    // 블록 k 하나. h: 이전 블록 출력 (conv_in이면 잠재값 x), skip: output 블록의 skip 입력
    ggml_tensor * block(int k, ggml_tensor * h, ggml_tensor * skip, ggml_tensor * emb, ggml_tensor * ctx) {
        const Block & b  = plan()[k];
        const int     dl = cfg.transformer_depth[b.level];
        switch (b.kind) {
            case B_CONV_IN: return conv(b.prefix + ".0", h);
            case B_DOWN:    return conv(b.prefix + ".0.op", h, 2);
            case B_IN_RES:
                h = res(b.prefix + ".0", h, emb);
                return dl > 0 ? spatial_transformer(b.prefix + ".1", h, ctx, dl) : h;
            case B_MID:
                h = res("middle_block.0", h, emb);
                h = spatial_transformer("middle_block.1", h, ctx, dl);
                return res("middle_block.2", h, emb);
            case B_OUT: {
                h       = res(b.prefix + ".0", concat_ch(h, skip), emb);
                int sub = 1;
                if (dl > 0) {
                    h   = spatial_transformer(b.prefix + ".1", h, ctx, dl);
                    sub = 2;
                }
                return b.upsample ? conv(b.prefix + "." + std::to_string(sub) + ".conv", upscale2(h)) : h;
            }
            case B_HEAD: return conv("out.2", ggml_silu(C, group_norm("out.0", h, 1e-5f)));
        }
        return nullptr;
    }

    // x: [W, H, 4, 1] 잠재값, t: [1] timestep, ctx: [Cctx, M], y: [adm, 1]
    ggml_tensor * unet(ggml_tensor * x, ggml_tensor * t, ggml_tensor * ctx, ggml_tensor * y) {
        ggml_tensor *              emb = time_emb(t, y);
        std::vector<ggml_tensor *> outs;
        ggml_tensor *              h = x;
        for (int k = 0; k < (int) plan().size(); k++) {
            const Block & b = plan()[k];
            h = block(k, h, b.skip >= 0 ? outs[b.skip] : nullptr, emb, ctx);
            outs.push_back(h);
        }
        return h;
    }
};
