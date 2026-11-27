# 학습 로그 요약: 스텝 수, 총 시간, 구간별 평균 s/스텝, 손실 평균, 최종 ema, NaN 개수
# 사용: python3 -I tools/train_stats.py <log...>
import re
import sys

R = re.compile(r'^step\s+(\d+)\s+t=\s*(\d+)\s+loss\s+(\S+)\s+ema\s+(\S+)\s+([0-9.]+)s')
for p in sys.argv[1:]:
    rows = [R.match(l) for l in open(p, encoding='utf-8', errors='replace')]
    rows = [m for m in rows if m]
    if not rows:
        print(f'{p}: step 줄 없음')
        continue
    t = [float(m[5]) for m in rows]
    loss = [float(m[3]) for m in rows]
    ema = float(rows[-1][4])
    nan = sum(1 for x in loss if x != x or x in (float('inf'), float('-inf')))
    head = t[1:20] or t
    tail = t[40:] or t
    print(f'{p}: steps {len(t)}, 총 {sum(t) / 3600:.2f}h, 2~20스텝 평균 {sum(head) / len(head):.1f}s, '
          f'41스텝~ 평균 {sum(tail) / len(tail):.1f}s, 손실 평균 {sum(loss) / len(loss):.4f}, 최종 ema {ema:.4f}, NaN {nan}')
