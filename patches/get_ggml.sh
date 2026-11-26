#!/bin/bash
# 패치된 ggml 소스 준비: stable-diffusion.cpp가 쓰는 ggml 커밋(eeda9ec4, leejet/ggml sd.cpp 브랜치)을 받아
# 이 프로젝트의 수정분(ggml-backward-fixes.patch)을 적용한다.
# 사용: patches/get_ggml.sh [대상 폴더=ggml]
set -e
DST=${1:-ggml}
HERE=$(cd "$(dirname "$0")" && pwd)
git clone --branch sd.cpp https://github.com/leejet/ggml.git "$DST"
git -C "$DST" checkout --detach eeda9ec48fc84a3df0bb9dcc19ae9053075a2539
git -C "$DST" apply "$HERE/ggml-backward-fixes.patch"
echo "준비됨: $DST   다음: cmake -B build -DGGML_DIR=$DST && cmake --build build -j4"
