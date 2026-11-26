# .npy 파일 모양/타입/통계 출력. 사용: python3 -I tools/npy_info.py <파일...>
import sys

import numpy as np

for p in sys.argv[1:]:
    a = np.load(p, allow_pickle=False)
    flat = a.reshape(-1).astype(np.float64)
    head = ', '.join(f'{v:.6g}' for v in flat[:6])
    print(f'{p}: shape={a.shape} dtype={a.dtype} min={flat.min():.4g} max={flat.max():.4g} '
          f'mean={flat.mean():.4g} std={flat.std():.4g} head=[{head}]')
