# verification — 검증 결과물 (v0.1)

이 폴더는 v0.1 코드를 실제로 돌린 **증거**다. 검증 **코드**는 `../tests`, `../tools`, 학습기의 `--check-ref / --grad-check / --mem-plan`에 있고, 여기는 그 실행 결과(로그, 생성 이미지, 기준 데이터)를 담는다. 학습한 LoRA 파일은 넣지 않았다 (아래 "학습한 LoRA 재현").
환경: AMD Radeon 660M (RDNA2 iGPU, 6 CU, UMA), Mesa 25.0.7 RADV, Vulkan 1.4, Linux 7.0, glibc 2.41. 로그의 경로는 `<work>`(작업 폴더)로 치환했다.

## 한눈에

| 무엇을 | 결과 | 증거 |
|---|---|---|
| 축소판 SDXL형 UNet vs PyTorch(float64), CPU | 18/18 PASS, LoRA 기울기 rel L2 ~1e-6 | `logs/tinyunet_cpu_vs_pytorch.txt` |
| 축소판 vs PyTorch, Vulkan (f32) | 6/6 PASS, LoRA 기울기 rel L2 3.1e-6~3.5e-6. fp16 행렬곱 켜면 1e-3~3e-3 | `logs/tinyunet_vulkan_vs_pytorch.txt` |
| ggml Vulkan vs CPU 연산 단위 (test-backend-ops) | OUT_PROD 92/92, MUL_MAT 2592/2592 (f32·f16), IM2COL/IM2COL_BACK 통과 | `logs/stage5_test_backend_ops_*.txt` |
| 연산별 Vulkan 자체 오차로 정밀도 버그 찾기 | 행렬곱 수정 전 MUL_MAT만 3.2e-4 (어텐션 QKᵀ) | `logs/stage5_per_op_vulkan_vs_cpu_before_matmul_fix.txt`, `logs/stage5_tinyunet_vulkan_before_matmul_fix.txt` |
| SDXL UNet 출력 vs stable-diffusion.cpp | 2.3e-4 (무작위 LoRA 적용 시 2.36e-4) | `logs/stage6_sdcpp_reference_unet*.txt`, `sdxl_ref/` |
| 학습이 실제로 일어나는가 (같은 seed 손실 전후) | 고정 (t, 노이즈) 8개 모두 40~65% 감소 | `logs/stage6_selftest_loss_before_after.txt` |
| 속도 병목 (수정 전 프로파일) | OUT_PROD가 스텝의 89% → MUL_MAT 역전파로 교체 | `logs/stage6_profile_before_outprod_fix.txt` |
| 실제 학습 384px, 600스텝 | 6.34 s/스텝, NaN 없음 | `logs/stage6_train_dog_384px_600steps.txt`, `logs/memory_s6_dog.txt` |
| 생성 비교 (sd.cpp Vulkan, 768px, 같은 seed) | 학습 대상이 나오고 장면 지시도 따름 | `images/`, `images/index.html` |
| 체크포인팅 할당기 버그 (수정 전 증거) | 768px assert, 재할당 무한 반복 | `logs/stage7_ckpt_allocator_bug_*.txt` |
| 체크포인팅 해상도별 (512/768/1024) | 12.4 / 28.7 / 57.5 s/스텝 | `logs/memory_s7_bench*.txt` |
| **1024px 실제 학습 (f32, 200스텝)** | 3.76h, NaN 없음, 1024px 생성에서 학습 대상 재현 + 장면 지시 따름 | `logs/stage8_train_dog_1024px_f32_200steps.log`, `images/*1024*_dog1024_lora_*` |
| **fp16 행렬곱 학습 (같은 조건)** | **실패**: 1.4배 빠르고 NaN 없고 손실도 비슷하지만 up 가중치가 정확히 0인 모듈 486/560 → 생성 결과 ≈ LoRA 없음. 원인 기울기 언더플로 | `logs/stage8_train_dog_1024px_fp16_200steps.log`, `logs/stage8_lora_weight_norms_f32_vs_fp16.txt`, `images/*1024*_dog1024_fp16_lora_*` |
| 학습 해상도 공정 비교 (384px 학습본을 1024px로 생성) | 384px 600스텝본은 1024px 생성에서 오렌지 배경·붉은 색감이 강하게 따라옴, 1024px 200스텝본은 배경·색감이 자연스럽고 더 선명. 단 스텝 수(600 vs 200)도 달라 해상도만의 효과로 단정 못 함 | `images/*1024*_dog_lora_w1_1024_*` |
| 서버 형태로 학습 실행 | 경로 탈출 거부, 진행률/ETA, 중복 실행 409, 1024px f32·fp16 학습 모두 서버로 실행 | `logs/stage8_server_stdout.txt` |
| sd.cpp 추론 (2단계) | 512/1024px, VAE flash attention 크래시(RADV) 증거 | `logs/stage2_*.log` |
| SDXL 그래프 연산 목록 (3단계 조사 원본) | 노드별 연산/타입/백엔드 | `logs/stage3_opdump_*.txt` |

## 폴더
- `logs/` — 실행 로그. `stageN_`은 진행 단계, `memory_*.txt`는 guard 메모리 기록을 실행별(시작/끝/걸린 시간/최대 GTT+시스템 합)로 요약한 것
- `images/` — sd.cpp 생성 이미지. 768px: `base_*`(LoRA 없음), `dog_lora[_s200|_s400]_*`(384px LoRA, 스텝), `beach_*`/`snow_*`(배경 명시 프롬프트, `_w0.7` = LoRA 가중치). 1024px: `<장면>1024_base_*`, `<장면>1024_dog1024_lora_*`(1024px f32 학습), `<장면>1024_dog1024_fp16_lora_*`(fp16 학습), `<장면>1024_dog_lora_*`(384px 학습본을 1024px로). `index.html`은 같은 프롬프트를 원본 크기로 나란히 놓은 갤러리
- `lora_norms.py` — LoRA 가중치 크기 비교 (학습 여부 판단용, numpy 필요). `train_stats.py` — 학습 로그 요약
- `tiny_ref/` — 축소판 정답(PyTorch가 만든 GGUF). PyTorch 없이 `../build/tinyunet tiny_ref/tiny8.gguf [--backend vulkan]`로 재현
- `sdxl_ref/` — SDXL UNet 기준값 (sd.cpp DGL `../tools/ref_unet.dgl`, CPU, 256px, t=0.5). `GGML_VK_DISABLE_F16=1 ../build/sdxl_lora --check-ref sdxl_ref --native all` → eps rel_l2 ≈ 2.3e-4 (SDXL base 체크포인트 필요)

## 학습한 LoRA 재현
검증에 쓴 LoRA 7개(약 330MB)는 저장소에도 Releases에도 넣지 않았다. 필요하면 같은 조건으로 직접 학습한다.
공통 조건: 데이터 [diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example) 5장, 캡션 `a photo of sks dog`, rank 8, alpha 8, lr 1e-4, seed 1.
GPU 연산 순서 차이로 비트 단위까지 같지는 않지만 결과(손실 추세, 생성 이미지 경향)는 같아야 한다.

```sh
# 캐시 (해상도별 한 번)
python3 tools/make_cache.py dataset/dog cache/dog384  --size 384  --caption "a photo of sks dog" --sd-cli sd-cli
python3 tools/make_cache.py dataset/dog cache/dog1024 --size 1024 --caption "a photo of sks dog" --sd-cli sd-cli --vae-backend cpu

# dog384_s200/s400/s600 : 384px, 체크포인팅 없음, 600스텝 (약 1시간)
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog384 --native all --steps 600 --seed 1 \
    --save-every 200 --out out/dog384.safetensors

# dog1024_f32_s100/s200 : 1024px f32, 200스텝 (약 3.8시간)
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog1024 --ckpt --native all --steps 200 --seed 1 \
    --save-every 100 --out out/dog1024_f32.safetensors

# dog1024_fp16_s100/s200 : 같은 조건 fp16 행렬곱 = 학습 실패 사례 재현 (GGML_VK_DISABLE_F16 없이)
build/sdxl_lora --train cache/dog1024 --ckpt --native all --steps 200 --seed 1 \
    --save-every 100 --out out/dog1024_fp16.safetensors

# 학습 여부 확인 (fp16 쪽은 up 가중치가 0인 모듈이 대부분)
python3 -I verification/lora_norms.py out/dog1024_f32.safetensors out/dog1024_fp16.safetensors
```
저장소 루트에서 실행. 이미지 생성 비교는 루트 README의 "4. 결과 확인".
