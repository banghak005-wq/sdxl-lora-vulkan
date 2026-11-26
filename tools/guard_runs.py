# guard 메모리 로그(.mem)에서 실행별 시작/끝 시각, 걸린 시간, 최대 합계를 뽑는다
# 사용: python3 -I tools/guard_runs.py <log.mem>
import sys
from datetime import datetime

runs, start, last, peak = [], None, None, 0
for ln in open(sys.argv[1]):
    p = ln.split()
    if not p:
        continue
    if p[0] == 'GUARD' and p[1] == 'done':
        if start:
            runs.append((start, last, peak, ln.strip()))
        start, last, peak = None, None, 0
        continue
    try:
        t = datetime.strptime(p[0], '%H:%M:%S')
    except ValueError:
        continue
    start = start or t
    last = t
    for kv in p[1:]:
        if kv.startswith('sum='):
            peak = max(peak, int(kv[4:]))
for i, (a, b, pk, end) in enumerate(runs, 1):
    d = (b - a).seconds
    print(f'{i}: {a:%H:%M:%S}~{b:%H:%M:%S}  {d // 60}분 {d % 60}초  최대 {pk} MiB  ({end})')
