# kohya LoRA 파일들의 가중치 크기 비교 (up은 0에서 시작하므로 학습량의 지표)
# 사용: python3 -I verification/lora_norms.py <a.safetensors> [b.safetensors ...]
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
    if ref is None:
        ref = up
    else:
        c = float(np.dot(ref, up) / (np.linalg.norm(ref) * np.linalg.norm(up) + 1e-30))
        print(f'    첫 파일과 up 방향 유사도(cos) {c:.3f}, 크기 비 {np.linalg.norm(up) / np.linalg.norm(ref):.3f}')
