#!/usr/bin/env python3
# kohya LoRA 파일에서 모듈 종류를 골라 새 파일을 만든다 (표준 라이브러리만)
# 사용: python3 -I tools/lora_filter.py <in.safetensors> <out.safetensors> <attn|nonattn|conv3x3|small>
#   attn    : transformer_blocks 안 (어텐션) 모듈만
#   nonattn : 어텐션 밖 (LoCon의 3x3 합성곱 + 1x1 skip + emb Linear)
#   conv3x3 : k>1 합성곱 모듈만
#   small   : 1x1 skip + emb Linear만
import json
import struct
import sys

src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
    body = f.read()
meta = h.pop('__metadata__', None)


def module(k):
    return k.rsplit('.', 2)[0] if k.endswith('.weight') else k.rsplit('.', 1)[0]


def kind(mod):
    if 'transformer_blocks' in mod:
        return 'attn'
    d = h[mod + '.lora_down.weight']['shape']
    return 'conv3x3' if len(d) == 4 and (d[2] > 1 or d[3] > 1) else 'small'


keep = {'attn': {'attn'}, 'nonattn': {'conv3x3', 'small'}, 'conv3x3': {'conv3x3'}, 'small': {'small'}}[mode]
out, off, blobs = {}, 0, []
for k in sorted(h):
    if kind(module(k)) not in keep:
        continue
    a, b = h[k]['data_offsets']
    blobs.append(body[a:b])
    out[k] = dict(h[k], data_offsets=[off, off + (b - a)])
    off += b - a
if meta:
    out['__metadata__'] = meta
hs = json.dumps(out).encode()
hs += b' ' * ((8 - len(hs) % 8) % 8)
with open(dst, 'wb') as f:
    f.write(struct.pack('<Q', len(hs)))
    f.write(hs)
    for b in blobs:
        f.write(b)
nm = len({module(k) for k in out if k != '__metadata__'})
print(f'{dst}: {mode}, 모듈 {nm}개, 텐서 {len(out) - (1 if meta else 0)}개, {off / 1048576:.1f} MiB')
