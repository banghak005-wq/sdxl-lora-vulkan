#!/bin/bash
# 사용: tools/guard.sh <메모리로그파일> <명령...>
# - 명령을 oom_score_adj=1000으로 실행 (호스트 OOM 시 1순위로 죽게)
# - 0.2초마다 GTT + 컨테이너 used(page cache 제외)를 기록하고 합계가 한도를 넘으면 명령을 죽임
# - 한도: GUARD_LIMIT_MIB (기본 11264 MiB). iGPU의 GTT는 시스템 RAM을 쓰므로 "GTT + 이 머신의 used"가 한도를 넘지 않게 지킴
# - used: LXC(lxcfs) 컨테이너 안이면 cgroup v2 memory.stat의 anon + kernel + shmem (page cache 제외, 커널 원본 장부),
#   그 밖(호스트에서 직접 실행 등)은 `free -m` (lxcfs가 없으면 free가 정확함, v0.1부터 검증된 방식).
#   LXC(lxcfs) 안의 `free -m`은 GTT로 모델을 올리는 순간 실제와 -0.9 ~ +5GiB 어긋나 거짓 종료를 낸다 (v0.2에서 확인).
#   로그의 fu= 는 비교용 `free -m` used
# - AMD amdgpu 전제 (/sys/class/drm/card0/device/mem_info_gtt_used). 다른 카드 번호면 GUARD_DRM_DEV로 지정
LOG=$1; shift
LIM=${GUARD_LIMIT_MIB:-11264}
D=${GUARD_DRM_DEV:-/sys/class/drm/card0/device}
# used 측정 방식: /proc/meminfo가 lxcfs로 덮인 컨테이너(LXC)에서만 cgroup 장부, 그 밖(호스트 직접 실행 등)은 free -m
if awk '$2=="/proc/meminfo" && $3 ~ /lxcfs/ {f=1} END{exit !f}' /proc/mounts && [ -r /sys/fs/cgroup/memory.stat ]; then MODE=cgroup; else MODE=free; fi
echo "GUARD mode=$MODE limit=${LIM}MiB" >> "$LOG"
echo 1000 > /proc/self/oom_score_adj
"$@" &
PID=$!
peak=0
while kill -0 $PID 2>/dev/null; do
    g=$(( $(cat $D/mem_info_gtt_used) / 1048576 ))
    v=$(( $(cat $D/mem_info_vram_used) / 1048576 ))
    fu=$(free -m | awk '/Mem/{print $3}')
    if [ $MODE = cgroup ]; then
        c=$(awk '/^(anon|kernel|shmem) /{s+=$2} END{print int(s/1048576)}' /sys/fs/cgroup/memory.stat)
    else
        c=$fu
    fi
    t=$(( g + c ))
    [ $t -gt $peak ] && peak=$t
    echo "$(date +%T) gtt=$g vram=$v ct=$c sum=$t fu=$fu" >> "$LOG"
    if [ $t -gt $LIM ]; then
        echo "GUARD KILL sum=${t}MiB > ${LIM}MiB" | tee -a "$LOG" >&2
        pkill -TERM -P $PID 2>/dev/null; kill -TERM $PID; sleep 3; kill -KILL $PID 2>/dev/null
    fi
    sleep 0.2
done
wait $PID; rc=$?
echo "GUARD done rc=$rc peak_sum=${peak}MiB" | tee -a "$LOG" >&2
exit $rc
