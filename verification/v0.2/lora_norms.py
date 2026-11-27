# kohya LoRA 파일들의 가중치 크기 비교 (up은 0에서 시작하므로 학습량의 지표)
# 사용: python3 -I tools/lora_norms.py <a.safetensors> [b.safetensors ...]
import json
import struct
import sys

import numpy as np


def load(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        body = f.read()
    h.pop('__metadata__', None)
    out = {}
    for k, v in h.items():
        a = np.frombuffer(body[v['data_offsets'][0]:v['data_offsets'][1]], dtype='<f4')
        out[k] = a
    return out


ref = None
for p in sys.argv[1:]:
    t = load(p)
    up = np.concatenate([v for k, v in t.items() if k.endswith('lora_up.weight')])
    dn = np.concatenate([v for k, v in t.items() if k.endswith('lora_down.weight')])
    zero_mods = sum(1 for k, v in t.items() if k.endswith('lora_up.weight') and not np.any(v))
    n_mods = sum(1 for k in t if k.endswith('lora_up.weight'))
    print(f'{p.split("/")[-1]}: up |w| 평균 {np.abs(up).mean():.3e}, 최대 {np.abs(up).max():.3e}, '
          f'up이 전부 0인 모듈 {zero_mods}/{n_mods}, down |w| 평균 {np.abs(dn).mean():.3e}')
    # 모듈 종류별 (v0.2 LoCon: 어텐션 / 3x3 합성곱 / 1x1 skip / emb Linear)
    groups = {}
    for k, v in t.items():
        if k.endswith('lora_up.weight'):
            g = ('어텐션' if 'transformer_blocks' in k else '1x1 skip' if 'skip_connection' in k
                 else 'emb Linear' if 'emb_layers' in k else '3x3 합성곱')
            groups.setdefault(g, []).append(v)
    if len(groups) > 1:
        for g, L in groups.items():
            print(f'    {g}: {len(L)}모듈, up |w| 평균 {np.mean([np.abs(x).mean() for x in L]):.3e}, '
                  f'0인 모듈 {sum(1 for x in L if not np.any(x))}')
    if ref is None:
        ref = t
    else:
        # 모듈 집합이 달라도 (어텐션만 vs LoCon) 공통 up 키끼리 비교
        common = sorted(k for k in t if k.endswith('lora_up.weight') and k in ref)
        a_ = np.concatenate([ref[k] for k in common]); b_ = np.concatenate([t[k] for k in common])
        c = float(np.dot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_) + 1e-30))
        print(f'    첫 파일과 공통 모듈 {len(common)}개의 up 방향 유사도(cos) {c:.3f}, 크기 비 {np.linalg.norm(b_) / np.linalg.norm(a_):.3f}')
