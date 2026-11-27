[English](README.md) | **한국어**

# SDXL LoRA 학습 on Vulkan (ROCm/CUDA 없이) — v0.4

**SDXL LoRA training with plain Vulkan (no ROCm, no CUDA), tested on an AMD Radeon 660M iGPU (6 CU, shared memory).**

ggml 위에 SDXL UNet 학습(순전파 + 역전파 + AdamW)을 직접 구현해, ROCm을 지원하지 않는 내장 GPU에서 Vulkan만으로
SDXL LoRA를 학습한다. 결과 LoRA는 kohya 형식 safetensors라 stable-diffusion.cpp 등 일반 도구에 바로 걸 수 있다.

> 이 저장소의 코드와 문서는 Claude 에이전트(Anthropic)가 작성했고, 사람은 목표 설정·방향 결정·결과 판단을 맡았다. 설계 과정과 시행착오는 [for_ai.md](for_ai.md)에 그대로 남아 있다.

**v0.4 새 기능: 작업 큐 웹서버** — 캐시 만들기·학습·이미지 생성을 큐에 여러 개 넣어 두면 하나씩 차례로 실행한다. 넣어 두고 다른 일을 하다 돌아와 웹에서 결과를 보고 내려받으면 된다. 실행 중 작업 즉시 강제 종료, 재시작에도 남는 큐, 임시 접속 토큰(`--token auto`), 끝난 작업 골라 삭제. 서버는 C++ 실행 파일 하나라 파이썬이 필요 없다 (아래 "3. 작업 큐 웹서버").

v0.3 기능: **fp16 행렬곱 학습 (손실 스케일링)** — v0.2까지는 fp16 행렬곱으로 학습하면 기울기 언더플로로 LoRA가 학습되지 않아 f32 전용이었다.
v0.3은 손실 스케일링(dynamic)을 넣어 **fp16 행렬곱으로도 f32와 사실상 같은 LoRA**를 **1.33배 빠르게** 학습한다 (1024px 200스텝: LoRA 방향 유사도 0.999, 생성 이미지 거의 같음, 아래 "v0.3 결과").
`GGML_VK_DISABLE_F16=1` 없이 실행하면 자동으로 켜진다. f32 학습(`GGML_VK_DISABLE_F16=1`)은 v0.2와 비트 단위로 같은 결과.

v0.2 기능: **합성곱 LoRA (LoCon)** — `--conv-rank N`으로 ResBlock·다운/업샘플 합성곱에도 LoRA를 건다 (kohya `conv_dim`과 같은 규칙·같은 파일 형식).
옵션을 안 주면 v0.1과 똑같이 동작한다 (같은 seed에서 학습 손실이 비트 단위로 같음을 확인).
그 밖의 변경: `tools/guard.sh`(메모리 감시)가 LXC 안에서 멀쩡한 작업을 종료시키던 측정 오류 수정 (아래 5절).

## v0.4 결과: 작업 큐 서버로 처음부터 끝까지
강아지 사진 5장으로 **캐시 만들기 → 학습(512px, 500스텝) → 생성 3장**을 서버 큐에 한꺼번에 넣고 손대지 않았다. 전부 이 서버를 통해서만 실행했다.

| 프롬프트 | LoRA 없음 | 서버로 학습한 LoRA (512px 500스텝, fp16) |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket768_base.png) | ![](docs/images/bucket_v04_server_lora512.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach_base.png) | ![](docs/images/beach_v04_server_lora512.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow_base.png) | ![](docs/images/snow_v04_server_lora512.png) |

- 학습 63분(스텝당 7.5초, 넘침 0), 768px 생성 장당 약 5분, 메모리 합계 최대 8.0GiB. 학습한 코기(큰 귀, 얼굴 흰 줄)가 나오고 장면 지시도 따른다. 양동이 그림의 주황 배경은 학습 사진 배경을 따라간 것 (사진 5장·저해상도 학습의 한계, v0.1의 384px 학습과 같은 현상)
- **서버로 해도 결과가 같은지**: 서버 학습 LoRA = 명령줄 학습 LoRA **비트 일치**, 서버 생성 그림 = sd-cli 직접 실행 그림 **픽셀 일치**, 서버가 직접 만드는 캐시 = 기존 파이썬 스크립트 캐시 **비트 일치**
- **강제 종료**: 실제 학습 중 kill → 0.5초 안에 프로세스 그룹 전체 종료, GPU 메모리(GTT) 4.7GiB → 0 반환
- 서버 동작 시험 56항목 통과 (가짜 학습기로, GPU 없이 누구나 돌려 볼 수 있음). 상세: [`verification/`](verification/README.md)

## v0.3 결과: fp16 행렬곱 학습 (손실 스케일링)
v0.2에서 실패했던 실험(강아지 5장, 1024px, 200스텝, seed 1, rank 8)을 **같은 조건**으로 v0.3에서 다시 했다. 생성은 sd.cpp 1024px, seed 42.

| 프롬프트 | f32 학습 (v0.2) | fp16 손실 스케일링 없음 (v0.2) | **fp16 + 손실 스케일링 (v0.3)** |
|---|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket1024_lora_f32.png) | ![](docs/images/bucket1024_fp16_v02.png) | ![](docs/images/bucket1024_fp16_v03.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach1024_lora_f32.png) | ![](docs/images/beach1024_fp16_v02.png) | ![](docs/images/beach1024_fp16_v03.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow1024_lora_f32.png) | ![](docs/images/snow1024_fp16_v02.png) | ![](docs/images/snow1024_fp16_v03.png) |

- v0.2 fp16: 학습한 코기가 안 나옴 (LoRA 없는 것과 비슷). v0.3 fp16: **f32와 거의 같은 그림** (같은 개체·구도, 귀 각도·발 같은 작은 차이만)
- LoRA 가중치: v0.3 fp16은 f32와 up 방향 유사도 **0.999**, 크기 비 0.999, 0인 모듈 0/560 (v0.2 fp16: 유사도 0.000, 0인 모듈 486/560)
- 평균 손실 0.10518 (f32 0.1051), NaN·넘침 0번, 스텝당 50.7초 (f32 약 67.7초 → **1.33배 빠름**), 메모리는 f32와 같음 (합계 최대 9.2GiB)
- f32 학습은 v0.2와 비트 단위로 같은 결과 (손실 스케일을 켜도 2의 거듭제곱이라 f32에선 결과가 안 바뀜)
- 상세: `verification/v0.3/logs/v03_*.txt`, `for_ai.md` 1.5·1.8·2.13

## v0.2 결과: 어텐션 LoRA vs LoCon (같은 데이터·같은 seed·같은 학습 순서)
강아지 5장, rank 8, lr 1e-4. LoCon은 `--conv-rank 8 --conv-alpha 8`만 추가 (초기화·t·노이즈 순서가 어텐션 쪽과 똑같이 맞춰져 있어 차이는 합성곱 LoRA뿐).
생성: sd.cpp (Vulkan), 20스텝, cfg 7, seed 42, `--lora-apply-mode at_runtime`, 가중치 1. 학습 해상도 3가지 × 어텐션/LoCon.

**결론: 이 데이터(개체 하나 학습)에서는 LoCon이 확실히 낫다고 할 근거가 없다.** 두 방식 모두 학습 개체를 그리고 장면 지시를 따르며, 장면별 우열은 해상도마다 엇갈린다 (seed 하나 비교라 우연 범위로 봄). LoCon은 학습 시간이 6~14% 더 든다. 질감·화풍 위주 데이터에서는 다를 수 있으나 검증하지 않았다.

| 학습 해상도 (스텝) | LoCon이 나은 장면 | 어텐션이 나은 장면 | 비슷 | 학습 s/스텝 (어텐션 → LoCon) | 평균 손실 (어텐션 / LoCon) |
|---|---|---|---|---|---|
| 384px (600) | 4개 모두 조금씩 (털결·체형) | – | – | 6.34 → 6.70 (+6%) | 0.1101 / 0.1080 |
| 768px (300) | 눈숲 | sks dog, 양동이, 해변(근소) | – | 34.9 → 36.9 (+6%) | 0.0949 / 0.0946 |
| 1024px (200) | 해변 (색감) | 양동이 | 눈숲 | 67.7 → 77.3 (+14%) | 0.1051 / 0.1049 |

- 모든 해상도에서 합성곱 쪽 66모듈이 전부 학습됨 (up 크기 ≈ 어텐션, 0인 모듈 0/626). LoCon 안의 어텐션 LoRA는 어텐션만 학습한 것과 방향 유사도 0.91(384) / 0.97(768) / 0.98(1024)
- **합성곱 부분만 걸어 보면** (768px LoCon에서 어텐션 키를 빼고 합성곱 쪽 66모듈만, `verification/lora_filter.py nonattn`): 그림의 색·질감·장면은 바뀌지만 학습한 코기가 나오지 않는다 (`sks dog` → 흰 래브라도 ![](docs/images/dog0_768train_convonly.png), 양동이 → 다른 강아지 ![](docs/images/bucket_768train_convonly.png)). **개체의 정체성은 어텐션 LoRA가 배우고, 합성곱 LoRA는 질감·색감 쪽을 바꾼다** → 개체 학습에서 LoCon 이득이 작았던 이유
- **학습 해상도 효과가 LoCon 효과보다 훨씬 크다**: 384px 학습본은 주황 배경(학습 사진의 배경)이 강하게 따라오지만 768·1024px 학습본은 장면이 자연스럽다 (아래 표)
- 메모리 합계 최대: 384px(체크포인팅 없음) 9.2GiB, 768px 7.1GiB, 1024px 9.2GiB (`--ckpt`)

**768px 학습본** (768px 생성):

| 프롬프트 | 어텐션 | LoCon |
|---|---|---|
| `a photo of sks dog` | ![](docs/images/dog0_768train_attn.png) | ![](docs/images/dog0_768train_locon.png) |
| `... in a bucket` | ![](docs/images/bucket_768train_attn.png) | ![](docs/images/bucket_768train_locon.png) |
| `... on a sunny beach` | ![](docs/images/beach_768train_attn.png) | ![](docs/images/beach_768train_locon.png) |
| `... in a snowy forest` | ![](docs/images/snow_768train_attn.png) | ![](docs/images/snow_768train_locon.png) |

**1024px 학습본** (1024px 생성, 어텐션은 v0.1 학습본):

| 프롬프트 | 어텐션 | LoCon |
|---|---|---|
| `... in a bucket` | ![](docs/images/bucket_1024train_attn.png) | ![](docs/images/bucket_1024train_locon.png) |
| `... in a snowy forest` | ![](docs/images/snow_1024train_attn.png) | ![](docs/images/snow_1024train_locon.png) |
| `... on a sunny beach` | ![](docs/images/beach_1024train_attn.png) | ![](docs/images/beach_1024train_locon.png) |

**384px 학습본** (768px 생성, LoRA 없음 포함):

| 프롬프트 | LoRA 없음 | 어텐션 LoRA (v0.1) | LoCon (v0.2) |
|---|---|---|---|
| `a photo of sks dog` | ![](docs/images/dog_base.png) | ![](docs/images/dog_attn600.png) | ![](docs/images/dog_locon600.png) |
| `a photo of sks dog in a bucket` | ![](docs/images/bucket768_base.png) | ![](docs/images/bucket_attn600.png) | ![](docs/images/bucket_locon600.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach_base.png) | ![](docs/images/beach_attn600.png) | ![](docs/images/beach_locon600.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow_base.png) | ![](docs/images/snow_attn600.png) | ![](docs/images/snow_locon600.png) |

**384px 학습본을 1024px로 생성** (장당 약 11분, 메모리 합계 약 8.9GiB):

| 프롬프트 | 어텐션 LoRA (v0.1) | **LoCon (v0.2)** |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket1024_attn600.png) | ![](docs/images/bucket1024_locon600.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow1024_attn600.png) | ![](docs/images/snow1024_locon600.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach1024_attn600.png) | ![](docs/images/beach1024_locon600.png) |

- 눈숲에서 차이가 가장 큼: 어텐션판은 화면 전체가 붉게 물드는데(384px 학습본의 색 번짐, v0.1에서 확인) LoCon판은 눈·나무가 더 하얗고 선명. 양동이 손잡이처럼 형태도 LoCon이 더 온전. 단 주황 배경·붉은 색감 과적합은 둘 다 남음 (5장·384px 학습의 한계)
- 가중치 0.7 (768px 해변): ![](docs/images/beach_locon600_w07.png) — 어텐션판 0.7(아래 v0.1 결과)과 같은 구도, 가중치 1보다 과적합이 빠져 자연스러움

- 비용 (384px 기준): 파라미터 11.6M → 14.9M, 스텝 6.34 → 6.70초 (+6%), 메모리 합계 8.85 → 9.19GiB, 파일 47 → 60MB


## v0.1 결과 (어텐션 LoRA, v0.2에서도 그대로 재현됨)

학습: [diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example) 5장, 캡션 `a photo of sks dog`, 384px, rank 8.
생성: stable-diffusion.cpp (Vulkan), 768px, 20스텝, cfg 7, **같은 seed로 LoRA 없음 / 있음** 비교.

| 프롬프트 | LoRA 없음 | LoRA 있음 |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket_base.png) | ![](docs/images/bucket_lora400.png) 400스텝 |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach_base.png) | ![](docs/images/beach_lora600_w07.png) 600스텝, 가중치 0.7 |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow_base.png) | ![](docs/images/snow_lora600.png) 600스텝 |

학습한 강아지(큰 귀, 흰 줄무늬 얼굴)가 나오면서 프롬프트의 장면(양동이, 해변, 눈숲)도 따른다.

### 1024px (SDXL 표준 해상도) 학습
같은 데이터로 **1024px, 200스텝**(`--ckpt --native all`, f32) 학습 → sd.cpp로 1024px 생성, 같은 seed.

| 프롬프트 | LoRA 없음 | LoRA 있음 (1024px f32) |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket1024_base.png) | ![](docs/images/bucket1024_lora_f32.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow1024_base.png) | ![](docs/images/snow1024_lora_f32.png) |

학습 3.76시간 (스텝당 57~69초), GPU+시스템 메모리 합계 약 8.5GiB, NaN 없음.

### fp16 행렬곱 학습 (v0.2까지: 실패 → v0.3: 손실 스케일링으로 해결)
v0.2까지는 fp16 행렬곱(`GGML_VK_DISABLE_F16` 없이)으로 학습하면 NaN도 없고 손실도 f32와 비슷한데 **LoRA가 거의 학습되지 않았다** — up 가중치가 정확히 0으로 남은 모듈 486/560, 생성 결과는 LoRA 없는 것과 거의 같음.
원인은 **기울기 언더플로**(손실 기울기가 매우 작아 fp16 최소값 아래로 떨어져 0이 됨). v0.3의 손실 스케일링이 이를 해결한다 (아래 "v0.3 결과").

### 정확도
- 축소판 SDXL형 UNet: LoRA 기울기가 PyTorch(float64 정답)와 **CPU 1e-6 / Vulkan 3e-6** 상대오차로 일치
- 실제 SDXL UNet 출력: stable-diffusion.cpp와 **2.3e-4** (sd.cpp 쪽 f16 im2col 수준 차이), LoRA를 건 상태도 같음
- 그래디언트 체크포인팅 기울기 = 체크포인팅 없는 기울기 (**8e-7**)
- **v0.2 LoCon**: 축소판 기울기 PyTorch와 CPU 1.7e-6 / Vulkan 3.3e-6, 실제 SDXL 출력 sd.cpp와 **2.39e-4** (합성곱 LoRA를 빼면 1.1e-2로 48배 → 비교가 합성곱 기여까지 확인함), sd.cpp 1878/1878 키 인식, 체크포인팅 기울기 6.2e-7

### 속도 / 메모리 (Radeon 660M, Mesa 25.0.7 RADV, f32, rank 8, 배치 1)
| 해상도 | 모드 | s/스텝 | GPU 메모리 (가중치 4.9GiB 포함) |
|---|---|---|---|
| 384 | 체크포인팅 없음 | 6.3 | 약 8.0 GiB |
| 512 | `--ckpt` | 12.4 | 약 5.5 GiB |
| 768 | `--ckpt` | 28.7 | 약 6.6 GiB |
| **1024** | `--ckpt` | **57.5** (오래 돌리면 약 69) | **약 8.2 GiB** |
| **1024, fp16 + 손실 스케일링 (v0.3)** | `--ckpt` | **39.6** (오래 돌리면 약 51) | 약 8.2 GiB (같음) |

1024px 실제 학습에서는 20~40스텝쯤부터 스텝 시간이 약 15% 늘어 유지됐다 (384px 1시간 학습에선 없었음, 발열·클럭은 정상, 원인 미확정).

검증 결과물(로그, 생성 이미지 전체, 기준 데이터)은 [`verification/`](verification/README.md) (이전 판 것은 `verification/v0.1/`, `verification/v0.2/`, `verification/v0.3/`). 학습한 LoRA 파일은 배포하지 않는다.

## 요구 사항
- Linux, Vulkan 1.3 GPU 드라이버 (AMD RADV에서 검증). GPU가 쓸 수 있는 메모리 약 9GiB 이상 (iGPU면 GTT 크기 확인)
- CMake 3.19+, C++17 컴파일러, `glslc` (ggml Vulkan 셰이더 빌드)
- SDXL base 체크포인트 (`sd_xl_base_1.0.safetensors`), VAE (`sdxl-vae-fp16-fix` 권장)
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp)의 `sd-cli` (DGL 모드 지원 버전): 학습 데이터 캐시와 결과 확인에 사용
- Python 3 (캐시 스크립트는 표준 라이브러리만 씀). 검증 테스트는 PyTorch, NumPy, `gguf`

## 빌드
```sh
patches/get_ggml.sh ggml                 # ggml(eeda9ec4) 받기 + 이 프로젝트 수정분 적용
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_DIR=ggml
cmake --build build -j4                  # build/sdxl_lora, build/sdxl_lora_server, build/tinyunet
```
빌드된 x86-64 Linux 실행 파일(`sdxl_lora`, `sdxl_lora_server`)은 [Releases](../../releases)에 있다 (ggml 정적 링크, glibc 2.41에서 빌드). 받은 파일은 `bin/`에 두면 아래 예시 경로와 맞는다. v0.4 묶음에는 서버가 쓰는 메모리 감시 스크립트 `tools/guard.sh`도 들어 있다 (실행 파일 옆 `tools/`를 자동으로 찾음). 실행에 필요한 시스템 라이브러리: `libvulkan1`, `libgomp1`, `libstdc++6` + GPU의 Vulkan 드라이버. glibc가 더 오래된 배포판이면 소스에서 빌드.
빌드 의존(Debian/Ubuntu): `build-essential cmake git glslc libvulkan-dev spirv-headers`. 빌드·사용·문제 해결의 상세는 [for_ai.md](for_ai.md) 1.11~1.20.

## Docker로 실행 (v0.3)
Releases의 실행 파일(또는 직접 빌드한 `build/sdxl_lora`, `build/sdxl_lora_server`)을 `bin/`에 두고 쓴다. ggml까지 정적 링크라, **Debian 13 컨테이너에 Vulkan 드라이버만 깔고 이 폴더를 마운트하면 그대로 돈다.** ROCm/CUDA용 특수 이미지나 런타임은 필요 없고, iGPU는 `/dev/dri`만 넘기면 된다.
```sh
# 1) 환경 이미지: Debian 13 + Vulkan 드라이버만 (한 번만. Dockerfile 파일 없이 표준입력으로)
docker build -t sdxl-vulkan-env - <<'EOF'
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers libgomp1 procps \
 && rm -rf /var/lib/apt/lists/*
EOF

# 공통 옵션: GPU 넘기기 + 내 사용자로 실행(결과 파일이 root 소유가 되지 않게) + 폴더 연결
GPU="--device /dev/dri --group-add $(getent group render | cut -d: -f3) --user $(id -u):$(id -g) -e HOME=/work"
REPO=/path/to/sdxl-lora-vulkan   # 이 저장소 (bin/에 실행 파일)
WORK=/path/to/work               # models/, cache/, out/ 를 둘 작업 폴더 (내 사용자 소유)
VOL="-v $REPO:/app:ro -v $WORK:/work"

# 2) GPU가 보이는지 확인: "ggml_vulkan: 0 = AMD Radeon ... (RADV ...)" 줄이 나오면 됨
docker run --rm $GPU $VOL sdxl-vulkan-env /app/bin/sdxl_lora

# 3-a) 서버로 학습 (브라우저 http://127.0.0.1:8090/)
docker run -d --name sdxl-lora $GPU $VOL -p 127.0.0.1:8090:8090 sdxl-vulkan-env \
    /app/bin/sdxl_lora_server --root /work --host 0.0.0.0 --port 8090

# 3-b) 또는 CLI로 바로 학습
docker run --rm $GPU $VOL -w /work sdxl-vulkan-env \
    /app/bin/sdxl_lora --train cache/dog1024 --ckpt --native all --steps 200 --out out/dog_lora.safetensors
```
- `$WORK/models/sd_xl_base_1.0.safetensors`가 있어야 한다 (학습기의 기본 모델 경로가 작업 폴더 기준). 서버는 학습기를 `--root`(= `/work`) 안에서 실행한다
- 학습 데이터 캐시(아래 사용법 1절)는 stable-diffusion.cpp의 `sd-cli`가 필요해서 이 환경엔 없다 → 호스트에서 만들어 `$WORK/cache/`에 두기
- **v0.4 서버의 캐시·생성 작업**은 sd-cli가 필요하다: 호스트의 sd-cli를 같이 마운트하고 알려 준다 (예: `-v /path/to/sd-cli:/opt/sd-cli:ro` + 서버 인자 `--sd-cli /opt/sd-cli`). 없으면 학습 작업만 된다. 외부에 열 거면 서버 인자에 `--token auto`
- `-p 127.0.0.1:8090:8090`: 서버는 인증이 없으므로 호스트 자신에게만 연다. 컨테이너 안의 `--host 0.0.0.0`은 Docker 포트 연결용
- `HOME=/work`: Mesa가 GPU 셰이더 컴파일 결과를 `$WORK/.cache/`에 저장 → 다음 실행부터 첫 스텝이 빠름
- `render` 그룹이 없는 배포판이면 `--group-add`에 `/dev/dri/renderD128`의 그룹 번호를 쓴다 (`ls -ln /dev/dri`로 확인)
- 실행 파일은 glibc 2.38 이상이 필요하다 (Debian 13 = 2.41, Ubuntu 24.04 = 2.39도 됨)
- **iGPU 메모리 주의**: 내장 GPU가 쓰는 공유 메모리(GTT)는 `docker run --memory` 제한에 잡히지 않는다. 메모리가 빠듯하면 `tools/guard.sh`로 감시하며 실행 (`... sdxl-vulkan-env /app/tools/guard.sh /work/logs/mem.log /app/bin/sdxl_lora --train ...`). Docker 안의 guard는 `free`로 호스트 전체 사용량을 보므로 GTT가 겹쳐 세어질 수 있어 실제보다 일찍(안전한 쪽으로) 종료할 수 있다 → 필요하면 `-e GUARD_LIMIT_MIB=...`로 조정
- 컨테이너의 Mesa 버전은 그때의 Debian 13 패키지를 따른다 (검증은 Mesa 25.0.7). **이 Docker 절차는 문서로만 검토했고 Docker 위에서 실제로 돌려 보지는 않았다** (실행 파일과 필요한 라이브러리는 검증한 것과 같음)

### 다른 GPU에서 (Docker 포함, 미검증 — 검증한 GPU는 Radeon 660M 하나)
| GPU | 넘기는 방법 | 예상 |
|---|---|---|
| AMD (RDNA1~4, GCN, 다른 APU) | `--device /dev/dri` (위와 같음) | 같은 Mesa RADV 드라이버라 될 가능성 높음. RDNA3+는 ggml이 행렬 코어(coopmat) 경로를 쓸 수 있어 fp16 수치가 다를 수 있음 |
| Intel (Arc, Xe 내장) | `--device /dev/dri` | 같은 `mesa-vulkan-drivers`에 ANV 드라이버 포함 → 이론상 동작 |
| NVIDIA | `/dev/dri` 대신 NVIDIA Container Toolkit: `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` (graphics가 있어야 Vulkan 드라이버가 들어옴) | ggml Vulkan은 NVIDIA를 지원하므로 동작 예상. 단 NVIDIA라면 CUDA용 기존 도구가 더 빠름 |

- **메모리**: 외장 GPU는 VRAM에 다 들어가야 한다. 실측(rank 8, `--ckpt --native all`): 1024px 약 8.2GB → **12GB 이상 권장 (8GB 카드는 1024px 불가, 768px 이하로)**, 768px 약 6.6GB, 512px 약 5.5GB
- **처음 쓰는 GPU에선 확인부터**: 짧게(예: 512px 50스텝) 학습한 뒤 `python3 verification/lora_norms.py <결과>`로 `up이 전부 0인 모듈`이 0인지 본다. 이 프로젝트가 추가한 Vulkan 커널(OUT_PROD 확장, IM2COL_BACK)은 660M에서만 검증됐다. 이상하면 `GGML_VK_DISABLE_F16=1`(f32 학습)로 비교. 정밀 검증은 `tests/tinyunet`(PyTorch 대조)을 소스에서 빌드해 실행
- GPU가 여러 개면 학습기는 보이는 것 중 첫째를 쓴다 → `-e GGML_VK_VISIBLE_DEVICES=1` 등으로 고르기 (시작 로그의 `ggml_vulkan: N = ...` 목록 참고)

## 사용법

### 1. 데이터 캐시 (이미지 → 잠재값, 캡션 → 텍스트 조건)
학습 그래프에는 UNet만 들어간다. VAE/텍스트 인코더는 sd.cpp로 미리 한 번 돌려 `.npy`로 저장한다.
```sh
# 이미지 폴더: 그림 파일 + (선택) 같은 이름의 .txt 캡션. 가운데를 정사각형으로 잘라 --size로 줄인다
python3 tools/make_cache.py dataset/dog cache/dog1024 --size 1024 --caption "a photo of sks dog" \
    --sd-cli /path/to/sd-cli --model models/sd_xl_base_1.0.safetensors --vae models/sdxl_vae.safetensors \
    --vae-backend cpu    # 1024px VAE 인코딩은 GPU 메모리를 많이 써서 CPU 권장
```
결과: `<이름>.latent.npy`, `<이름>.crossattn.npy`, `<이름>.vector.npy`, 확인용 `<이름>.check.png`, `captions.txt`

### 2. 학습 (CLI)
```sh
# fp16 행렬곱 + 손실 스케일링 (v0.3, 1.33배 빠름, 결과는 f32와 사실상 같음)
build/sdxl_lora --train cache/dog1024 --ckpt --native all \
    --steps 200 --lr 1e-4 --rank 8 --alpha 8 --save-every 100 --out out/dog_lora.safetensors
# f32 (v0.2와 비트 단위로 같은 결과)
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog1024 --ckpt --native all --steps 200 --out out/dog_lora.safetensors
```
| 옵션 | 설명 |
|---|---|
| `--train <폴더>` | 캐시 폴더 (모든 샘플이 같은 해상도) |
| `--ckpt` | 그래디언트 체크포인팅 (블록 단위 재계산). 768px 이상은 필수, 약 1.3배 느림 |
| `--native all` | NORM/GROUP_NORM/GEGLU/CONCAT/UPSCALE를 ggml 원래 연산 + 새 역전파로 (메모리 절약, **권장**). 빼면 기본 연산 조합 버전 |
| `--steps --lr --rank --alpha --seed` | 학습 설정. LoRA 대상: 모든 attn1/attn2의 to_q/k/v/out (560 모듈) |
| `--conv-rank N` (`--conv-dim`) | **v0.2** LoCon. 0(기본) = 끔. 켜면 ResBlock의 3x3 합성곱 2개·다운샘플·업샘플(38 모듈)에 rank N, 같은 블록의 1x1 skip(11)·시간 임베딩 Linear(17)에 `--rank`로 LoRA 추가 = 626 모듈. `conv_in`/마지막 합성곱은 제외 (kohya와 같음) |
| `--conv-alpha A` | **v0.2** 3x3 쪽 alpha (scale = A/N). 기본 = conv-rank (scale 1) |
| `--loss-scale S\|dynamic` | **v0.3** 손실 스케일링. 기본: fp16 행렬곱이면(Vulkan + `GGML_VK_DISABLE_F16` 없음) `dynamic`, 아니면 1(끔). `dynamic`: 기울기에 NaN/Inf(fp16 넘침)가 생기면 그 스텝의 갱신을 건너뛰고 S를 절반으로, 연속 정상이면 2배로. 고정값이면 넘칠 때 오류로 종료 |
| `--loss-scale-init S` / `--loss-scale-window N` | **v0.3** dynamic 시작값 (기본 32768) / S를 2배로 올리기까지 연속 정상 스텝 수 (기본 200) |
| `--save-every N` | 중간 저장 `<out>_s<스텝>.safetensors` |
| `--resume <lora>` | 이어서 학습 (`--lr 0`이면 고정 seed 손실 평가) |
| `--t-min/--t-max` | timestep 범위 (기본 0..999) |
| `--model <ckpt>` | SDXL 체크포인트 (기본 `models/sd_xl_base_1.0.safetensors`) |
| `--backend vulkan\|cpu` | 기본 vulkan |

- `GGML_VK_DISABLE_F16=1`: 행렬곱을 f32로 (v0.2까지는 필수). **v0.3**: 없으면 fp16 행렬곱 + 손실 스케일링(자동)으로 학습한다. 로그 각 스텝 끝의 `S=`가 현재 손실 스케일, 넘침이 나면 `overflow step N: ...` 줄이 찍힌다. 끝에 `손실 스케일: 마지막 S=.., 건너뛴 스텝 ..` 요약
- fp16 학습 성공 여부는 손실 값으로 알 수 없다 (v0.2 실패 때도 손실은 f32와 비슷했음). 확인은 `python3 verification/lora_norms.py <f32 결과> <fp16 결과>`(up이 0인 모듈 수, 방향 유사도)나 생성 이미지로
- 출력 LoRA는 kohya 형식 (`lora_unet_<블록>_attn1_to_q.lora_down.weight` / `.lora_up.weight` / `.alpha`)
  - LoCon 모듈: `lora_unet_input_blocks_1_0_in_layers_2.lora_down.weight` `[rank, in, 3, 3]`, `.lora_up.weight` `[out, rank, 1, 1]`, 메타데이터 `ss_network_args = {"conv_dim", "conv_alpha"}`. stable-diffusion.cpp가 1878/1878 텐서 인식 (검증)
  - `--resume`은 파일과 `--rank/--alpha/--conv-rank/--conv-alpha`가 같아야 함

### 3. 작업 큐 웹서버 (v0.4)
캐시 만들기, 학습, 이미지 생성을 **큐에 여러 개 넣어 두면 한 번에 하나씩 차례로** 실행하는 웹서버. 넣어 두고 다른 일을 하다가 돌아와서 결과를 보면 된다.
- **C++ 실행 파일 하나** (`sdxl_lora_server`, 약 0.9MB). 파이썬 등 다른 런타임 없이 돈다. 필요한 것: `sdxl_lora`(학습), stable-diffusion.cpp의 `sd-cli`(캐시·생성)
- GPU 메모리 때문에 동시 실행은 하지 않는다. 작업마다 별도 프로세스라 끝나거나 멈추면 GPU 메모리가 바로 반환된다
- 큐는 `<root>/jobs/queue.json`에 저장 → 서버를 다시 켜도 대기 작업이 남고 이어서 실행된다 (돌던 작업은 "interrupted"로 표시, 남은 프로세스는 정리)
- 웹 화면: 작업 추가(캐시 / 학습 / 생성 / **한 번에**: 사진 폴더 → 캐시 → 학습 → 미리보기 생성), 큐와 진행률·남은 시간, 로그, 끝난 작업 보관함(LoRA·그림·로그 내려받기, 골라서 삭제), 생성 그림 갤러리, 한국어/English
- **멈추기**: 대기 작업 취소, 실행 중 작업 중지(SIGTERM → 10초 안에 안 끝나면 SIGKILL), **즉시 강제 종료**(SIGKILL), 대기 전부 취소, 큐 일시정지. 작업의 자식 프로세스(메모리 감시, sd-cli 등)까지 프로세스 그룹째 종료한다
- AMD GPU면 작업마다 메모리 감시(`tools/guard.sh`)를 자동으로 붙인다 (`--guard auto|on|off`, 한도 `--guard-limit-mib`)

```sh
build/sdxl_lora_server --root /path/to/work --sd-cli /path/to/sd-cli --token auto
#   접속 토큰 (이번 실행에만 유효): 9ryjewpxe64h6nbe7xapg8ftfxc24c8i
#   바로 열기: http://127.0.0.1:8090/?token=9ryjewpxe64h6nbe7xapg8ftfxc24c8i
```
- 작업 폴더(`--root`) 안에 `models/sd_xl_base_1.0.safetensors`, `models/sdxl_vae.safetensors`(다른 곳이면 `--model`, `--vae`), 학습 사진 폴더를 둔다. 요청의 경로는 모두 `--root` 안으로 제한된다
- 학습기는 같은 폴더의 `sdxl_lora`를 쓴다 (`--cli`로 지정 가능). `--sd-cli`가 없으면 학습 작업만 가능
- 생성 공통 인자는 `--sd-args` (기본 `--vae-tiling --flash-attn auto,vae=off --lora-apply-mode at_runtime`: RADV에서 VAE flash attention 셰이더가 깨지는 문제 회피 + LoRA 메모리 절약). 캐시 작업의 CPU 스레드는 `--threads`

**웹 화면 바꾸기 / 끄기** (프론트를 직접 만들 때)
| 옵션 | 동작 |
|---|---|
| `--ui builtin` (기본) | 내장 웹 화면 |
| `--ui off` | 화면 없이 **API만**. `/`는 API 안내 JSON |
| `--ui <폴더>` | 그 폴더의 정적 파일(`index.html`, js, css …)을 `/`에 서빙. 직접 만든 프론트를 서버 하나로 같이 띄울 때 |
| `--cors <origin>` | 프론트를 다른 주소에서 띄우고 이 API를 부를 때 허용할 출처 (예: `--cors http://localhost:5173`, 또는 `"*"`) |

- 토큰은 `/api/…`와 `/files/…`에만 걸린다. 프론트 파일은 토큰 없이 열리고, 프론트는 API를 부를 때 `Authorization: Bearer <토큰>` 헤더(또는 `sdxl_token` 쿠키)를 보내면 된다
- 내장 화면(`src/server/page.h`, HTML 한 장)이 API를 쓰는 예제이기도 하다

**접속과 보안**: 서버는 기본으로 **이 기계 안(127.0.0.1)에서만** 열린다. 밖(폰, 다른 PC)에서 보려면 아래 중 하나를 직접 고른다.
| 방법 | 예 | 비고 |
|---|---|---|
| Cloudflare 퀵 터널 | `cloudflared tunnel --url http://127.0.0.1:8090` → 출력되는 `https://….trycloudflare.com` | 계정 불필요. **주소를 아는 누구나 접속 가능 → 반드시 `--token`** |
| Tailscale | 같은 tailnet 기기에서 `http://<이 기계의 tailscale 이름>:8090` (서버는 `--host 0.0.0.0` 또는 tailscale IP로) | tailnet 밖에는 안 보임. 그래도 `--token` 권장 |
| 공유기 포트포워딩 | 공유기에서 외부 포트 → 이 기계 8090 (서버는 `--host 0.0.0.0`) | 인터넷 전체에 노출 → **`--token` 필수**, 가능하면 HTTPS 리버스 프록시 앞에 |

- `--token auto`: 켤 때마다 새 임시 토큰(32글자)을 만들어 콘솔에 출력. 웹 첫 화면에서 토큰을 넣어야 들어갈 수 있고, 토큰 없이 `/api`·`/files`에 오면 401. `?token=…`으로 들어오면 바로 통과(주소창에서는 지움)
- `--token <직접 정한 값>` 또는 환경변수 `SDXL_SERVER_TOKEN`(명령줄 기록에 안 남김). API에서는 `Authorization: Bearer <토큰>` 헤더
- 토큰은 평문 HTTP로 오간다. 인터넷에 직접 열 때는 HTTPS(Cloudflare 터널은 기본 HTTPS)로

**API** (JSON. 토큰을 켰으면 `-H "Authorization: Bearer <토큰>"`)
```sh
curl -X POST localhost:8090/api/jobs -d '{"type":"cache","src":"dataset/dog","size":512,"caption":"a photo of sks dog"}'
curl -X POST localhost:8090/api/jobs -d '{"type":"train","data":"cache/dog_512","steps":500,"out":"out/dog.safetensors"}'
curl -X POST localhost:8090/api/jobs -d '{"type":"generate","prompt":"a photo of sks dog on a sunny beach","lora":"out/dog.safetensors","width":768,"height":768}'
curl localhost:8090/api/jobs                       # 목록 + 진행률 (step, steps, loss, sec_per_step, eta_sec)
curl -X POST localhost:8090/api/jobs/3/cancel      # 대기 → 취소, 실행 중 → SIGTERM (10초 뒤 SIGKILL)
curl -X POST localhost:8090/api/jobs/3/kill        # 실행 중 → 즉시 SIGKILL
curl -X POST localhost:8090/api/jobs/remove-many -d '{"ids":[1,2],"delete_files":true}'   # 끝난 작업 + 그 결과 파일 삭제
```
| 엔드포인트 | 내용 |
|---|---|
| `POST /api/jobs` | `type`: `cache` (`src`, `dst`, `size` 512, `caption`, `prefix`, `vae_backend`:"cpu") / `train` (`data`, `out`, `name`, `steps` 200, `lr`, `rank`, `alpha`, `conv_rank`, `conv_alpha`, `seed`, `save_every`, `ckpt` true, `native` "all", `resume`, `model`, `fp16` true, `loss_scale`, `t_min`, `t_max`) / `generate` (`prompt`, `negative`, `lora`, `lora_weight` 1, `width`·`height` 1024 (64의 배수), `steps` 20, `cfg` 7, `seed` 42). 경로 존재는 **작업이 시작될 때** 검사 → 앞 작업이 만들 캐시·LoRA를 뒤 작업이 써도 된다 |
| `GET /api/jobs`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/log` | 목록, 상세(로그 끝 60줄, 결과 파일), 전체 로그 |
| `POST /api/jobs/<id>/cancel`, `/kill`, `/move` (`{"dir":"up"/"down"/"top"}`), `/remove` (`{"delete_files":bool}`) | 취소, 강제 종료, 대기 순서 바꾸기, 끝난 작업 지우기 |
| `POST /api/jobs/remove-many` | `{"ids":[…], "delete_files":bool}`. 대기·실행 중인 작업은 건너뜀. 파일 삭제는 **그 작업이 만든 파일만** (LoRA와 `_s<스텝>` 중간 저장본, 생성 그림·로그, 캐시의 `.npy`·`.check.png`·`cache.dgl`·`captions.txt`). 원본 사진은 지우지 않음 |
| `POST /api/queue/cancel-all` (`{"running":true,"force":true}` 선택), `/pause`, `/resume` | 대기 전부 취소, 큐 일시정지/재개 |
| `GET /api/config`, `GET /api/list` | 서버 설정, 폼에서 고를 LoRA·캐시·이미지 폴더 목록 |
| `GET /files/<경로>` | `--root` 안의 파일 받기 |
| v0.3 호환: `POST /api/train`, `GET /api/status`, `POST /api/stop`, `GET /api/log` | `/api/train`은 이제 거절(409) 대신 큐에 넣는다 |

### 4. 결과 확인 (sd.cpp로 생성)
```sh
SD_CLI=/path/to/sd-cli LORA_DIR=out tools/eval_prompt.sh beach "a photo of sks dog on a sunny beach" dog 0   # LoRA 없음
SD_CLI=/path/to/sd-cli LORA_DIR=out tools/eval_prompt.sh beach "a photo of sks dog on a sunny beach" dog 1   # LoRA 있음
python3 tools/gallery.py out/eval     # 같은 프롬프트를 나란히 놓은 index.html
```
sd-cli에 직접 걸 때는 `--lora-model-dir out -p "...<lora:dog:1>" --lora-apply-mode at_runtime` (미리 합치는 기본 방식보다 메모리 약 2GiB 적음).

### 5. 검증 / 진단 도구
| 명령 | 하는 일 |
|---|---|
| `build/tinyunet tests/tiny8.gguf [--backend vulkan] [--native all]` | 축소판 UNet의 순전파/LoRA 기울기를 PyTorch 정답과 비교 (정답은 `python3 tests/ref_tiny_unet.py --out tests/tiny8.gguf --size 8 [--conv-lora]`로 생성. `--conv-lora`: LoCon 포함). `--compare`: 연산별 Vulkan vs CPU 자체 오차 |
| `sdxl_lora --check-ref <폴더> [--lora x]` | 실제 SDXL UNet 출력을 sd.cpp 기준값과 비교 (기준값: `tools/ref_unet.dgl`을 `sd-cli -M dgl`로 실행) |
| `sdxl_lora --train <폴더> --grad-check` | 체크포인팅 기울기 = 전체 그래프 기울기 확인 (2번째 스텝에서 비교) |
| `sdxl_lora --mem-plan <잠재값 한 변> [--ckpt] [--native all]` | GPU 없이 학습 그래프 메모리 계산 (64 = 512px, 128 = 1024px) |
| `SDXL_NAN_DEBUG=1 sdxl_lora --train ... --steps N` | N번째 스텝에서 처음 비유한 값이 나오는 노드 찾기 (체크포인팅 없는 모드). **v0.3**: 역전파 노드까지 검사 |
| `python3 tests/test_server.py build/sdxl_lora_server` | **v0.4** 작업 큐 서버 시험 56항목 (가짜 학습기·sd-cli로, GPU 불필요): 차례 실행, 순서 바꾸기, 취소·강제 종료와 자식 프로세스 정리, 재시작 복원, 경로 제한, 선택 삭제와 원본 보존, 토큰, 화면 끄기·직접 만든 프론트·CORS |
| `tools/guard.sh <로그> <명령>` | iGPU용 메모리 감시: GTT + 시스템 used 합이 `GUARD_LIMIT_MIB`를 넘으면 명령 종료. **v0.2**: LXC 컨테이너 안에서는 used를 `free -m` 대신 cgroup `memory.stat`(anon+kernel+shmem)에서 읽음 (호스트 직접 실행은 그대로 `free`) — LXC(lxcfs)의 `free`는 GPU로 모델을 올리는 순간 수 GiB 거짓으로 튀어 멀쩡한 작업을 종료시켰음 (`verification/v0.2/logs/stage4_sdcpp_memory_spike_investigation.txt`) |

## 알려진 제한 (v0.4)
- 배치 1, 정사각형 한 해상도 (비율 버킷 없음), UNet LoRA만 (텍스트 인코더 LoRA 없음). 트랜스포머 안의 proj_in/out·ff Linear는 대상 아님 (kohya 기본값은 포함)
- 모듈별 rank/alpha는 두 묶음(어텐션·1x1·Linear / 3x3 합성곱)만. kohya `block_dims` 같은 블록별 rank 없음
- fp16 행렬곱 학습(v0.3)은 1024px 어텐션 LoRA 200스텝, 256px LoCon 40스텝으로 검증. 가중치·활성값 저장은 그대로(가중치 f16, 활성값 f32)라 메모리는 f32 학습과 같다 — 빨라질 뿐 메모리는 안 줄어듦
- 1024px 장시간 학습에서 스텝 시간이 약 15% 늘어나는 현상 (원인 미확정)
- 학습 데이터 캐시와 결과 확인에 stable-diffusion.cpp `sd-cli`가 필요 (서버의 캐시·생성 작업도 `--sd-cli`로 이것을 씀)
- 서버는 한 번에 작업 하나 (GPU 메모리). 배치 생성은 작업 여러 개로(웹의 "장 수"가 seed를 1씩 늘려 여러 작업을 넣음). 사용자 계정 구분 없음(토큰 하나)
- 검증한 GPU는 Radeon 660M (RADV) 하나

## 폴더
```
src/unet_ggml.h        SDXL형 UNet 그래프 빌더 (블록 단위 API, 검증용 축소판과 공유)
src/sdxl_io.h          safetensors/npy 입출력, SDXL 이산 노이즈 스케줄
src/cli/sdxl_lora.cpp  학습기 (학습, 체크포인팅, 검증 모드, kohya 저장)
src/server/server.cpp  HTTP 작업 관리자
tests/                 축소판 UNet PyTorch 대조 검증
tools/                 캐시, 평가, 갤러리, 메모리 감시, 헤더/npy 보기
patches/               ggml 수정분 (역전파 규칙, Vulkan 커널) + 받기 스크립트
for_ai.md              설계 과정과 근거 (AI/개발자용 상세 문서)
```

## 라이선스
[MIT](LICENSE) © 2026 [banghak005-wq](https://github.com/banghak005-wq). 저작권 표시만 유지하면 상업적 이용·수정·재배포 모두 자유.
`patches/`(ggml 수정분), `third_party/json.hpp`, `third_party/httplib.h`와 배포 실행 파일에 들어간 외부 코드는 각 원저작자의 라이선스를 따른다. 목록과 라이선스 전문은 [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
예시 그림과 기준 텐서는 SDXL base 1.0(CreativeML Open RAIL++-M)의 출력물이다. 모델 가중치, 학습 사진(diffusers/dog-example, Unsplash의 Alvan Nee 사진), 학습한 LoRA는 포함하지 않는다. 자세한 내용은 THIRD_PARTY_LICENSES.md의 "Models, data and generated content" 절.
