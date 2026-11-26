#!/bin/bash
# 학습한 LoRA 효과 확인: 같은 프롬프트·seed로 LoRA 없음/있음 이미지를 sd.cpp(sd-cli, Vulkan)로 생성
#
# 사용: tools/eval_prompt.sh <이름표> "<프롬프트>" [lora 이름] [가중치=1] [크기=768] [seed=42]
#   - lora 이름: LORA_DIR 안의 <이름>.safetensors (확장자 빼고). 가중치 0이면 LoRA 없이 생성
#   - 결과: OUT_DIR/<이름표>_base_<크기>_s<seed>.png 또는 <이름표>_<lora>_w<가중치>_<크기>_s<seed>.png
# 환경변수 (기본값):
#   SD_CLI=sd-cli  MODEL=models/sd_xl_base_1.0.safetensors  VAE=models/sdxl_vae.safetensors
#   LORA_DIR=out  OUT_DIR=out/eval  STEPS=20  CFG=7  GUARD=tools/guard.sh (실행 파일이 없으면 guard 없이)
# 메모: sd.cpp의 LoRA 적용은 --lora-apply-mode at_runtime이 미리 합치는 방식보다 메모리 약 2GiB 적고 빠름
#       (분수 가중치를 합쳐서 적용하면 768px에서 RAM이 순간적으로 3GiB 더 듦).
#       --vae-tiling --flash-attn auto,vae=off 는 RADV(Mesa 25.0)의 VAE용 flash attention 셰이더 컴파일 크래시 우회.
TAGN=$1; P=$2; L=${3:-lora}; WT=${4:-1}; S=${5:-768}; SEED=${6:-42}
SD_CLI=${SD_CLI:-sd-cli}; MODEL=${MODEL:-models/sd_xl_base_1.0.safetensors}; VAE=${VAE:-models/sdxl_vae.safetensors}
LORA_DIR=${LORA_DIR:-out}; OUT_DIR=${OUT_DIR:-out/eval}; STEPS=${STEPS:-20}; CFG=${CFG:-7}; GUARD=${GUARD:-tools/guard.sh}
if [ -z "$TAGN" ] || [ -z "$P" ]; then sed -n '2,13p' "$0"; exit 2; fi
mkdir -p "$OUT_DIR"
if [ "$WT" = "0" ]; then
    OUT=$OUT_DIR/${TAGN}_base_${S}_s${SEED}.png; PP="$P"
else
    OUT=$OUT_DIR/${TAGN}_${L}_w${WT}_${S}_s${SEED}.png; PP="$P<lora:$L:$WT>"
fi
if [ -f "$OUT" ]; then echo "있음: $OUT"; exit 0; fi
RUN=(); [ -x "$GUARD" ] && RUN=("$GUARD" "$OUT_DIR/eval.mem")
echo "== $PP -> $OUT"
"${RUN[@]}" "$SD_CLI" -m "$MODEL" --vae "$VAE" --lora-model-dir "$LORA_DIR" \
    -p "$PP" -W "$S" -H "$S" --steps "$STEPS" --cfg-scale "$CFG" --seed "$SEED" \
    --vae-tiling --flash-attn auto,vae=off --lora-apply-mode at_runtime $SD_EXTRA -o "$OUT"
