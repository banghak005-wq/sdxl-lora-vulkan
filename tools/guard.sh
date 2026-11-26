#!/bin/bash
# 사용: tools/guard.sh <메모리로그파일> <명령...>
# - 명령을 oom_score_adj=1000으로 실행 (호스트 OOM 시 1순위로 죽게)
# - 0.2초마다 GTT + 컨테이너 used(page cache 제외)를 기록하고 합계가 한도를 넘으면 명령을 죽임
# - 한도: GUARD_LIMIT_MIB (기본 11264 MiB). iGPU의 GTT는 시스템 RAM을 쓰므로 "GTT + 이 머신의 used"가 한도를 넘지 않게 지킴
# - AMD amdgpu 전제 (/sys/class/drm/card0/device/mem_info_gtt_used). 다른 카드 번호면 GUARD_DRM_DEV로 지정
LOG=$1; shift
LIM=${GUARD_LIMIT_MIB:-11264}
D=${GUARD_DRM_DEV:-/sys/class/drm/card0/device}
echo 1000 > /proc/self/oom_score_adj
"$@" &
PID=$!
peak=0
while kill -0 $PID 2>/dev/null; do
    g=$(( $(cat $D/mem_info_gtt_used) / 1048576 ))
    v=$(( $(cat $D/mem_info_vram_used) / 1048576 ))
    c=$(free -m | awk '/Mem/{print $3}')
    t=$(( g + c ))
    [ $t -gt $peak ] && peak=$t
    echo "$(date +%T) gtt=$g vram=$v ct=$c sum=$t" >> "$LOG"
    if [ $t -gt $LIM ]; then
        echo "GUARD KILL sum=${t}MiB > ${LIM}MiB" | tee -a "$LOG" >&2
        pkill -TERM -P $PID 2>/dev/null; kill -TERM $PID; sleep 3; kill -KILL $PID 2>/dev/null
    fi
    sleep 0.2
done
wait $PID; rc=$?
echo "GUARD done rc=$rc peak_sum=${peak}MiB" | tee -a "$LOG" >&2
exit $rc
