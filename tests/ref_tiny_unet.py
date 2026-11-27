#!/usr/bin/env python3
"""축소판 SDXL UNet 정답지 생성 (PyTorch CPU, float64로 계산 후 float32로 저장).

구조는 sgm(SDXL) UNetModel과 같고 크기만 줄임:
  model_channels=32, channel_mult=(1,2), num_res_blocks=1,
  transformer_depth=(0,1), 중간 블록 depth=1, head_dim=16,
  GroupNorm 그룹 8 (SDXL은 32), context_dim=24, adm_in_channels=12.
sd.cpp와 똑같이 계산하도록 맞춘 점:
  - GEGLU의 GELU는 tanh 근사 (sd.cpp는 ggml_geglu_swapped = tanh 근사 사용)
  - SpatialTransformer의 GroupNorm eps=1e-6, ResBlock/출력 GroupNorm eps=1e-5, LayerNorm eps=1e-5
  - timestep embedding은 [cos, sin] 순서
LoRA: 모든 SpatialTransformer의 attn1/attn2 to_q/to_k/to_v/to_out.0, rank 4, scale 1.
--conv-lora (LoCon, kohya conv_dim 규칙): ResBlock/다운·업샘플의 3x3 합성곱 → rank CONV_RANK(3), scale CONV_SCALE(0.5),
  ResBlock의 1x1 skip_connection, emb_layers.1 → rank 4, scale 1. conv_in/out.2 제외.
  down = 원래와 같은 커널/stride/pad 합성곱 [r, cin, k, k], up = 1x1 [cout, r, 1, 1]
출력 GGUF:
  w.*       기본 가중치 (고정)
  lora.*    LoRA 가중치 (학습 대상)
  in.*      입력 (x, timestep, context, y, target)
  ref.out   UNet 출력,  ref.loss  MSE 손실,  ref.grad.lora.*  LoRA 그래디언트
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
import gguf

MC, CTX, ADM, HEAD_DIM, GROUPS, RANK = 32, 24, 12, 16, 8, 4
CONV_RANK, CONV_SCALE = 3, 0.5  # 어텐션 rank/scale과 일부러 다르게 (섞이면 드러나게)
CONV_LORA = False
TEMB = MC * 4

W = {}      # 이름 -> 텐서 (고정 가중치)
LORA = {}   # 이름 -> 텐서 (requires_grad)
gen = None


def rnd(*shape, std=1.0):
    return torch.randn(*shape, generator=gen, dtype=torch.float64) * std


def linear_w(name, din, dout, bias=True):
    W[name + ".weight"] = rnd(dout, din, std=1 / math.sqrt(din))
    if bias:
        W[name + ".bias"] = rnd(dout, std=0.1)


def conv_w(name, cin, cout, k):
    W[name + ".weight"] = rnd(cout, cin, k, k, std=1 / math.sqrt(cin * k * k))
    W[name + ".bias"] = rnd(cout, std=0.1)


def norm_w(name, c):
    W[name + ".weight"] = 1 + rnd(c, std=0.1)
    W[name + ".bias"] = rnd(c, std=0.1)


def lin(name, x):
    return F.linear(x, W[name + ".weight"], W.get(name + ".bias"))


def conv(name, x, stride=1):
    w = W[name + ".weight"]
    y = F.conv2d(x, w, W[name + ".bias"], stride=stride, padding=w.shape[-1] // 2)
    if name + ".lora_down" in LORA:
        d = F.conv2d(x, LORA[name + ".lora_down"], stride=stride, padding=w.shape[-1] // 2)
        sc = 1.0 if w.shape[-1] == 1 else CONV_SCALE
        y = y + sc * F.conv2d(d, LORA[name + ".lora_up"])
    return y


def make_conv_lora(name, cin, cout, k):
    if not CONV_LORA:
        return
    r = RANK if k == 1 else CONV_RANK
    LORA[name + ".lora_down"] = rnd(r, cin, k, k, std=0.5 / math.sqrt(cin * k * k)).requires_grad_()
    LORA[name + ".lora_up"] = rnd(cout, r, 1, 1, std=0.5 / math.sqrt(r)).requires_grad_()


def gn(name, x, eps):
    return F.group_norm(x, GROUPS, W[name + ".weight"], W[name + ".bias"], eps)


def ln(name, x):
    return F.layer_norm(x, x.shape[-1:], W[name + ".weight"], W[name + ".bias"], 1e-5)


# ---------- 블록 정의 (가중치 생성 + 순전파) ----------

def make_res(p, cin, cout):
    norm_w(p + ".in_layers.0", cin)
    conv_w(p + ".in_layers.2", cin, cout, 3)
    linear_w(p + ".emb_layers.1", TEMB, cout)
    norm_w(p + ".out_layers.0", cout)
    conv_w(p + ".out_layers.3", cout, cout, 3)
    if cin != cout:
        conv_w(p + ".skip_connection", cin, cout, 1)
    make_conv_lora(p + ".in_layers.2", cin, cout, 3)
    make_conv_lora(p + ".out_layers.3", cout, cout, 3)
    if cin != cout:
        make_conv_lora(p + ".skip_connection", cin, cout, 1)
    if CONV_LORA:
        make_lora(p + ".emb_layers.1", TEMB, cout)


def res(p, x, emb):
    h = conv(p + ".in_layers.2", F.silu(gn(p + ".in_layers.0", x, 1e-5)))
    h = h + lora_lin(p + ".emb_layers.1", F.silu(emb))[:, :, None, None]
    h = conv(p + ".out_layers.3", F.silu(gn(p + ".out_layers.0", h, 1e-5)))
    skip = conv(p + ".skip_connection", x) if (p + ".skip_connection.weight") in W else x
    return skip + h


def make_lora(name, din, dout):
    LORA[name + ".lora_down"] = rnd(RANK, din, std=0.5 / math.sqrt(din)).requires_grad_()
    LORA[name + ".lora_up"] = rnd(dout, RANK, std=0.5 / math.sqrt(RANK)).requires_grad_()


def lora_lin(name, x):
    y = lin(name, x)
    if name + ".lora_down" in LORA:
        y = y + F.linear(F.linear(x, LORA[name + ".lora_down"]), LORA[name + ".lora_up"])
    return y


def make_attn(p, dim, ctx_dim):
    for proj, din in (("to_q", dim), ("to_k", ctx_dim), ("to_v", ctx_dim)):
        linear_w(f"{p}.{proj}", din, dim, bias=False)
        make_lora(f"{p}.{proj}", din, dim)
    linear_w(f"{p}.to_out.0", dim, dim)
    make_lora(f"{p}.to_out.0", dim, dim)


def attn(p, x, ctx):
    b, n, c = x.shape
    h = c // HEAD_DIM
    q = lora_lin(p + ".to_q", x).view(b, n, h, HEAD_DIM).transpose(1, 2)
    k = lora_lin(p + ".to_k", ctx).view(b, -1, h, HEAD_DIM).transpose(1, 2)
    v = lora_lin(p + ".to_v", ctx).view(b, -1, h, HEAD_DIM).transpose(1, 2)
    a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(HEAD_DIM), dim=-1)
    o = (a @ v).transpose(1, 2).reshape(b, n, c)
    return lora_lin(p + ".to_out.0", o)


def make_st(p, c):
    norm_w(p + ".norm", c)
    linear_w(p + ".proj_in", c, c)
    tb = p + ".transformer_blocks.0"
    for i in (1, 2, 3):
        norm_w(f"{tb}.norm{i}", c)
    make_attn(tb + ".attn1", c, c)
    make_attn(tb + ".attn2", c, CTX)
    linear_w(tb + ".ff.net.0.proj", c, c * 4 * 2)
    linear_w(tb + ".ff.net.2", c * 4, c)
    linear_w(p + ".proj_out", c, c)


def st(p, x, ctx):
    b, c, hh, ww = x.shape
    x_in = x
    x = gn(p + ".norm", x, 1e-6)
    x = x.permute(0, 2, 3, 1).reshape(b, hh * ww, c)
    x = lin(p + ".proj_in", x)
    tb = p + ".transformer_blocks.0"
    x = attn(tb + ".attn1", ln(tb + ".norm1", x), ln(tb + ".norm1", x)) + x
    x = attn(tb + ".attn2", ln(tb + ".norm2", x), ctx) + x
    hcat = lin(tb + ".ff.net.0.proj", ln(tb + ".norm3", x))
    hid, gate = hcat.chunk(2, dim=-1)
    x = lin(tb + ".ff.net.2", hid * F.gelu(gate, approximate="tanh")) + x
    x = lin(p + ".proj_out", x)
    x = x.reshape(b, hh, ww, c).permute(0, 3, 1, 2)
    return x + x_in


# GGUF 텐서 이름은 63자 제한 → tinyunet.cpp의 abbr()와 같은 규칙으로 줄임
ABBR = [("input_blocks.", "ib."), ("middle_block.", "mb."), ("output_blocks.", "ob."),
        (".transformer_blocks.0.", ".tb0."), (".lora_down", ".ld"), (".lora_up", ".lu")]


def abbr(name):
    for a, b in ABBR:
        name = name.replace(a, b)
    return name


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float64) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def build():
    linear_w("time_embed.0", MC, TEMB)
    linear_w("time_embed.2", TEMB, TEMB)
    linear_w("label_emb.0.0", ADM, TEMB)
    linear_w("label_emb.0.2", TEMB, TEMB)
    conv_w("input_blocks.0.0", 4, MC, 3)
    make_res("input_blocks.1.0", MC, MC)
    conv_w("input_blocks.2.0.op", MC, MC, 3)
    make_conv_lora("input_blocks.2.0.op", MC, MC, 3)
    make_res("input_blocks.3.0", MC, 2 * MC)
    make_st("input_blocks.3.1", 2 * MC)
    make_res("middle_block.0", 2 * MC, 2 * MC)
    make_st("middle_block.1", 2 * MC)
    make_res("middle_block.2", 2 * MC, 2 * MC)
    make_res("output_blocks.0.0", 4 * MC, 2 * MC)
    make_st("output_blocks.0.1", 2 * MC)
    make_res("output_blocks.1.0", 3 * MC, 2 * MC)
    make_st("output_blocks.1.1", 2 * MC)
    conv_w("output_blocks.1.2.conv", 2 * MC, 2 * MC, 3)
    make_conv_lora("output_blocks.1.2.conv", 2 * MC, 2 * MC, 3)
    make_res("output_blocks.2.0", 3 * MC, MC)
    make_res("output_blocks.3.0", 2 * MC, MC)
    norm_w("out.0", MC)
    conv_w("out.2", MC, 4, 3)


def forward(x, t, ctx, y):
    emb = lin("time_embed.2", F.silu(lin("time_embed.0", timestep_embedding(t, MC))))
    emb = emb + lin("label_emb.0.2", F.silu(lin("label_emb.0.0", y)))
    hs = []
    h = conv("input_blocks.0.0", x); hs.append(h)
    h = res("input_blocks.1.0", h, emb); hs.append(h)
    h = conv("input_blocks.2.0.op", h, stride=2); hs.append(h)
    h = st("input_blocks.3.1", res("input_blocks.3.0", h, emb), ctx); hs.append(h)
    h = res("middle_block.0", h, emb)
    h = st("middle_block.1", h, ctx)
    h = res("middle_block.2", h, emb)
    h = st("output_blocks.0.1", res("output_blocks.0.0", torch.cat([h, hs.pop()], 1), emb), ctx)
    h = st("output_blocks.1.1", res("output_blocks.1.0", torch.cat([h, hs.pop()], 1), emb), ctx)
    h = conv("output_blocks.1.2.conv", F.interpolate(h, scale_factor=2, mode="nearest"))
    h = res("output_blocks.2.0", torch.cat([h, hs.pop()], 1), emb)
    h = res("output_blocks.3.0", torch.cat([h, hs.pop()], 1), emb)
    return conv("out.2", F.silu(gn("out.0", h, 1e-5)))


def main():
    global gen
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=8, help="잠재값 한 변 (짝수)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conv-lora", action="store_true", help="합성곱 LoRA(LoCon)도 넣기")
    a = ap.parse_args()
    global CONV_LORA
    CONV_LORA = a.conv_lora
    gen = torch.Generator().manual_seed(a.seed)

    build()
    s = a.size
    inputs = {
        "in.x": rnd(1, 4, s, s),
        "in.timestep": torch.tensor([500.0], dtype=torch.float64),
        "in.context": rnd(1, 5, CTX),
        "in.y": rnd(1, ADM),
        "in.target": rnd(1, 4, s, s),
    }
    out = forward(inputs["in.x"], inputs["in.timestep"], inputs["in.context"], inputs["in.y"])
    loss = F.mse_loss(out, inputs["in.target"])
    loss.backward()

    wr = gguf.GGUFWriter(a.out, "tinyunet")
    wr.add_uint32("tiny.size", s)
    wr.add_uint32("tiny.n_lora", len(LORA))
    wr.add_float32("tiny.conv_lora_scale", CONV_SCALE)

    def put(name, t):
        wr.add_tensor(abbr(name), np.ascontiguousarray(t.detach().numpy().astype(np.float32)))

    for k, v in {**W, **inputs}.items():
        put(k, v)
    for k, v in LORA.items():
        put("lora." + k, v)
        put("ref.grad.lora." + k, v.grad)
    put("ref.out", out)
    put("ref.loss", loss.reshape(1))
    wr.write_header_to_file(); wr.write_kv_data_to_file(); wr.write_tensors_to_file(); wr.close()

    gnorm = math.sqrt(sum(float((v.grad ** 2).sum()) for v in LORA.values()))
    print(f"weights={len(W)} lora={len(LORA)} params={sum(v.numel() for v in W.values())} "
          f"loss={loss.item():.6f} lora_grad_norm={gnorm:.6f}")


if __name__ == "__main__":
    main()
