# safetensors 헤더만 읽어 텐서 이름/타입/모양 요약 (본문 데이터는 읽지 않음)
# 사용: python3 -I tools/st_header.py <file.safetensors> [정규식]
import collections
import json
import re
import struct
import sys

path = sys.argv[1]
pat = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
with open(path, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]
    h = json.loads(f.read(n))
meta = h.pop('__metadata__', None)
print(f'{len(h)} tensors, header {n} B, metadata keys: {list(meta) if meta else None}')
pre = collections.Counter('.'.join(k.split('.')[:2]) for k in h)
for p, c in pre.most_common():
    dt = collections.Counter(v['dtype'] for k, v in h.items() if k.startswith(p + '.'))
    print(f'  {p}: {c} {dict(dt)}')
if pat:
    for k in sorted(h):
        if pat.search(k):
            print(k, h[k]['dtype'], h[k]['shape'])
