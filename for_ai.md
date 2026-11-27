> ⚠️ **이 문서는 AI(코딩 에이전트/LLM)가 읽고 이 저장소를 설명·수정하도록 쓴 것이다. 사람이 읽으라고 쓴 문서가 아니다.** 압축된 서술, 내부 용어, 파일·함수 이름 위주라 사람에게는 읽기 불편할 수 있다. 사람은 README.md(영어) / README.ko.md(한국어)를 볼 것.

# for_ai.md — SDXL LoRA 학습 on Vulkan v0.3

> v0.3 = v0.2 + **fp16 행렬곱 학습용 손실 스케일링**(`--loss-scale`, fp16이면 자동 dynamic) + `SDXL_NAN_DEBUG`가 역전파 노드까지 검사 + 서버 `loss_scale` 키. f32(`GGML_VK_DISABLE_F16=1`) 학습은 v0.2와 비트 단위로 같음. 바뀐 곳: 1.5, 1.7, 1.8, 1.9, 1.10, 1.13, 1.15, 1.18, 2.13
>
> v0.2 = v0.1 + **합성곱 LoRA(LoCon, `--conv-rank`)** + guard 측정 수정(cgroup 장부, 1.16/2.12). 옵션을 안 쓰면 v0.1과 비트 단위로 같은 학습. 바뀐 곳: 1.1, 1.4, 1.5, 1.8, 1.10, 1.13, 1.14, 1.15, 1.18, 1.19, 2.12

> 경로 안내 (v0.3 폴더 기준): v0.1·v0.2 시절 항목이 가리키는 `verification/logs/stage*`, `verification/images/`, `verification/sdxl_ref*` 등은 `verification/v0.1/…`, `verification/v0.2/…`로 옮겨 두었다. 학습한 LoRA(`loras/*.safetensors`)는 배포하지 않는다.

구성: **1부 최종 설계도**(현재 코드가 무엇을 어떻게 하는가, 1.11~1.20은 빌드·옵션·형식·API·문제 해결·확장까지 운영 레퍼런스) → **2부 시행착오**(그 설계에 이르기까지 겪은 문제와 판단 근거). 이 문서만으로 빌드·사용·검증·문제 해결·수정이 가능하도록 썼다.
수치는 AMD Radeon 660M(RDNA2, 6 CU, UMA, 전용 1GiB + GTT 최대 11GiB), Mesa 25.0.7 RADV, Vulkan 1.4 기준.

---

# 1부. 최종 설계도

## 1.1 목표와 범위
- ROCm/CUDA 없이 Vulkan만으로 SDXL UNet LoRA 학습. ggml(자동 미분 + Vulkan 백엔드) 위에 구현
- 범위(v0.2): UNet 어텐션 LoRA(모든 attn1/attn2의 to_q/k/v/out, 560 모듈) + 선택적 LoCon(`--conv-rank`, +66 모듈 = 626), 배치 1, 정사각형 단일 해상도, 512~1024px, f32 행렬곱으로 검증
- 결과 LoRA는 kohya 형식 → stable-diffusion.cpp 등에 그대로 사용

## 1.2 전체 흐름
```
이미지 + 캡션
  └─ tools/make_cache.py ── sd.cpp `sd-cli -M dgl` (VAE 인코드, CLIP-L/G 인코드)
       → <name>.latent.npy [1,4,H/8,W/8] (vae_to_unet_latents 적용됨)
         <name>.crossattn.npy [1,77,2048], <name>.vector.npy [2816]
  └─ src/cli/sdxl_lora (학습기, ggml + Vulkan)
       SDXL 체크포인트의 UNet 가중치만 로드 → LoRA 학습 → kohya safetensors
  └─ 확인: sd.cpp `sd-cli` + <lora:name:w> (--lora-apply-mode at_runtime)
서버: src/server/sdxl_lora_server = HTTP 작업 관리자, 학습기를 자식 프로세스로 실행
```
학습 그래프에는 UNet만 들어간다. 인코더는 학습되지 않으므로 미리 한 번 계산해 캐시한다.

## 1.3 소스 구성
| 파일 | 역할 |
|---|---|
| `src/unet_ggml.h` | `UNetBuilder`: SDXL형 UNet을 ggml 그래프로. `UNetConfig::sdxl()`(320/[1,2,4]/res 2/depth [0,2,10]/head_dim 64/GN 32) 과 `::tiny()`(검증용) 공유. 블록 API `plan()/block(k,h,skip,emb,ctx)/time_emb(t,y)`, `unet()`도 블록 API로 구현 |
| `src/sdxl_io.h` | safetensors 헤더 파싱(nlohmann/json), f16→f32 업로드, `.npy` 읽기/쓰기, SDXL 이산 스케줄(`DiscreteSchedule`: scaled_linear 1000스텝, k-diffusion `sigma_to_t`) |
| `src/cli/sdxl_lora.cpp` | 학습기. 모드: `--train`(전체 그래프), `--train --ckpt`(체크포인팅), `--grad-check`, `--check-ref`, `--mem-plan`. `UNetWeights`, `LoraSet`(생성/kohya 로드·저장), `train()`, `train_ckpt()`, `mem_plan()`, `nan_debug()` |
| `src/server/server.cpp` | cpp-httplib 서버. `/api/train`(fork+exec로 CLI 실행, 로그를 `<root>/logs/train-*.log`로), `/api/status`(로그의 `step` 줄을 정규식으로 파싱 → step/loss/ema/s/step/ETA), `/api/stop`(SIGTERM), `/api/log`, `/files/*`. 경로는 `--root` 안으로 제한(`weakly_canonical` + `lexically_relative`), 기본 127.0.0.1, 동시 1작업(409), 기본 `GGML_VK_DISABLE_F16=1` |
| `tests/tinyunet.cpp` + `tests/ref_tiny_unet.py` | 축소판 UNet을 PyTorch(float64)와 비교. `--native`, `--backend`, `--compare`(연산별 Vulkan vs CPU 자체 오차, 매 노드 입력 동기화) |
| `patches/ggml-backward-fixes.patch` | ggml(eeda9ec4, leejet/ggml sd.cpp 브랜치)에 대한 수정 (1.6절) |
| `tools/` | `make_cache.py`, `eval_prompt.sh`, `gallery.py`, `guard.sh`(GTT+시스템 used 합 감시, v0.2부터 used는 cgroup 장부), `ref_unet.dgl`(sd.cpp 기준값), `st_header.py`, `npy_info.py`, `guard_runs.py` |

## 1.4 UNet 그래프 세부
- 가중치: safetensors의 `model.diffusion_model.*`만 로드. 2D/4D는 f16 그대로, 1D(bias, norm)는 f32로 변환. UMA에서는 ggml Vulkan이 host-visible 버퍼에 memcpy하므로 사본 없음. 로드 후 `posix_fadvise(DONTNEED)`로 파일 페이지 캐시 해제
- 합성곱: `im2col`(f32 기본, `conv_f16`이면 f16) 후 `mul_mat(커널[KW*KH*IC, OC] f16, col)` → permute. f16 가중치가 src0이라 역전파가 f16 OUT_PROD/행렬곱 경로를 탐
- 어텐션: flash attention 없이 `mul_mat(k,q)` → `scale` → `soft_max` → `mul_mat(v,·)` (FA는 역전파 없음)
- `--native all`(권장): NORM/GROUP_NORM은 `ggml_norm`/`ggml_group_norm`, GEGLU는 `ggml_geglu_split(gate, hid)`(split 형태만 역전파 지원), CONCAT/UPSCALE는 `ggml_concat`/`ggml_upscale` — 1.6절의 역전파 규칙 사용. 끄면 기본 연산 조합 버전(검증용, 메모리 많이 씀)
- LoRA: `y = W x + b + scale · up(down(x))`, `scale = alpha/rank`. down `[in, rank]`(kaiming-uniform), up `[rank, out]`(0). kohya 저장 시 PyTorch 모양 `[rank,in]`/`[out,rank]` (메모리 배치 동일)
- **LoCon (v0.2, `conv()` 안)**: 원래 합성곱의 im2col 결과 `col2 [IC*KH*KW, P]`를 그대로 써서 `d = mul_mat(down[KW*KH*IC, R], col2)` → `u = mul_mat(up[R, OC], d)` → `y += scale·u` (permute 전 `[OC, P]`에서 더함). down = 원래와 같은 커널·stride·pad의 합성곱(ggml 4D `[KW,KH,IC,R]`), up = 1x1(`[1,1,R,OC]`). im2col을 공유하므로 추가 연산은 행렬곱 2개 + add, IM2COL_BACK도 하나로 합쳐짐 → **새 셰이더 없음**. `conv_f16`이면 LoRA 쪽만 col을 f32로 cast
  - 대상 = kohya `networks/lora.py` 규칙 (`UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ResnetBlock2D, Downsample2D, Upsample2D`, conv_dim 있을 때만): k>1 합성곱 `in_layers.2`, `out_layers.3`, `*.0.op`(다운), `output_blocks.*.conv`(업) = 38모듈 → `conv_rank/conv_alpha`. 같은 블록 안의 1x1 `skip_connection`(11)과 Linear `emb_layers.1`(17) → `rank/alpha` (kohya: Linear·1x1은 항상 network_dim). `input_blocks.0.0`(conv_in), `out.2` 제외. `UNetBuilder`의 scale은 `lora_scale`(Linear·1x1)과 `conv_lora_scale`(k>1) 두 개, 커널 크기로 고름
  - 초기화: kaiming fan_in = `ggml_nelements(down)/rank` (합성곱 = IC*KH*KW). **어텐션 모듈은 seed rng로 먼저(v0.1과 같은 값), ResBlock 쪽은 rng를 소비하지 않는 파생 난수열** → 같은 seed면 conv-rank와 상관없이 어텐션 초기값·t/노이즈/샘플 순서가 같다 (공정 비교, conv-rank 0 = v0.1 비트 일치)
- 손실: ε 예측 MSE. `x_t = √ᾱ_t x0 + √(1−ᾱ_t) ε`, t ~ U{t_min..t_max}. UNet 입력은 `x_t` 그대로 (c_in = √ᾱ와 같음)
- 옵티마이저: `ggml_opt_step_adamw` (β1 0.9, β2 0.999, eps 1e-8, wd 0.01, bias correction은 매 스텝 파라미터 텐서에 기록)

## 1.5 학습 루프 두 가지
**전체 그래프 (`train()`)**: 순전파 + `ggml_build_backward_expand(grads_acc=nullptr)` + 파라미터별 AdamW 노드를 한 그래프로, 할당 한 번, 매 스텝 재사용.
- `grads_acc` 없이 만들면 파라미터 기울기는 누적되지 않고 매 스텝 새로 계산된다
- `ggml_graph_reset`은 처음 한 번만. 매 스텝은 손실 기울기 누적 텐서(`ggml_graph_get_grad_acc(gf, loss)`)에 S(손실 스케일, 기본 1)를 직접 쓴다
- **v0.3**: AdamW를 학습 그래프에서 분리. 학습 그래프 끝에서 LoRA 기울기를 f32 저장소 `gstore`로 `ggml_cpy` → 검사 그래프(`build_finite_check`: Σ sum(|g|) 하나, NaN/Inf는 합에 남음) → 정상이면 AdamW 그래프(`g × inv_s` 후 `ggml_opt_step_adamw`). 그래프 3개는 gallocr를 각각 씀. AdamW가 빠졌으므로 `ggml_graph_reset`이 m/v를 안 지워 줌 → 직접 0으로

**체크포인팅 (`train_ckpt()`, `--ckpt`)**:
- UNet = 블록 20개 (conv_in, input 8, middle, output 9, head). `plan()`이 skip 연결(output 블록이 꺼낼 input 블록 번호)까지 계산
- 순전파 그래프 F: 블록 출력 a[k]만 영구 버퍼로 `ggml_cpy`, 시간 임베딩도 영구 버퍼로
- 역전파: k를 뒤에서부터. 블록 그래프 G[k]의 손실 = `sum(out_k ⊙ ga[k])` (head는 실제 MSE). 입력 a[k-1]과 skip 입력은 `ggml_set_param` → 기울기를 `ggml_add_inplace`로 ga[]에 누적, 블록 안 LoRA 기울기는 `ggml_cpy`로 g_store에
- 기울기가 필요 없는 블록은 건너뜀: 블록 k는 (자체 LoRA 있음) 또는 (앞 블록에 LoRA 있음)일 때만. SDXL은 input 0~3이 생략 → 역전파 그래프 16개 (LoCon을 켜면 input_blocks.1부터 LoRA가 있어 19개)
- AdamW 그래프 하나 (g_store 사용). **v0.3**: 앞에 `ggml_mul(g_store, inv_s)` (손실 스케일 되돌리기), 별도 검사 그래프 Cg (g_store 유한값 검사)도 같은 gallocr 공유 목록에 들어감
- **v0.3 손실 스케일 S는 head 그래프(실제 MSE)와 grad-check 전체 그래프의 손실 기울기에만**. 중간 블록 G[k]의 손실 `sum(out⊙ga[k])`은 기울기 시작값 1 (ga[]가 이미 S배). 여기도 S를 쓰면 블록마다 S가 곱해져 f32에서도 Inf (2.13에서 실제로 밟음)

**손실 스케일링 (v0.3, `LossScaler`)**: 손실 기울기 시작값을 S로 → 역전파 전체가 S배 → fp16 행렬곱 입력 변환에서 작은 기울기가 0이 되지 않음. LoRA 기울기는 f32 `gstore`에 S배로 모인 뒤 AdamW 직전에 1/S. S는 2의 거듭제곱이라 f32에서 곱/나눗셈이 정확 → f32 모드에선 S와 무관하게 결과가 비트 단위로 같음(검증).
- 1/S를 되돌리는 이유: AdamW 갱신 m̂/(√v̂+ε)은 고정 S에는 불변이지만, dynamic에서 S가 바뀌면 m(S배 기울기의 EMA)과 v(S²배)의 스케일이 섞임 → 되돌려서 항상 원래 스케일로 누적
- dynamic: 검사 합이 비유한 → 그 스텝은 AdamW 건너뜀(m/v/파라미터 그대로, AdamW 스텝 번호 `opt_step`도 안 늘림 → bias correction 유지), S/2, 로그 `overflow step N`. `window`(기본 200) 스텝 연속 정상 → S×2 (최대 2^24). 20번 연속 넘침 또는 S<1 → 오류 종료(스케일링으로 못 고치는 수치 문제). 고정 S에서 넘치면 즉시 오류
- 기본값: `O.backend == "vulkan" && GGML_VK_DISABLE_F16 없음` → dynamic(시작 32768), 아니면 1. 순전파 손실이 NaN이면 v0.2처럼 즉시 오류 (스케일링과 무관한 문제)
- 비용: 검사 그래프(파라미터 수만큼 abs+sum) + 4바이트 읽기, AdamW 앞 mul. 스텝 시간 대비 무시할 수준
- **할당**: 모든 그래프(F, G[*], AdamW, 선택적으로 grad-check용 전체 그래프)가 gallocr 하나를 공유하고 차례로 계산되므로 주소가 겹쳐도 된다. 단 2부 2.8의 함정 때문에: ① `GGML_VK_SUBALLOCATION_BLOCK_SIZE`를 4GiB로 기본 설정(main에서 `setenv(...,0)`, 장치 한도로 자동 제한) ② 모든 그래프 `ggml_gallocr_reserve` ③ 그래프 컨텍스트 소유 텐서의 `data/buffer`를 지우고 모든 그래프를 할당하는 패스를, 버퍼 크기가 한 번도 안 바뀔 때까지 반복(최대 5) ④ 이후 매 스텝은 할당 없이 계산만

## 1.6 ggml 수정분 (패치 내용)
| 위치 | 내용 |
|---|---|
| `ggml.c` 역전파 | PERMUTE/TRANSPOSE: 기울기를 `ggml_cont`. SUB/DIV: 브로드캐스트 src1에 `repeat_back` |
| `ggml.c` 역전파 | MUL_MAT: src0 grad = `mul_mat(cont(src1ᵀ), cont(gradᵀ))`, src1 grad = `mul_mat(cont(src0ᵀ), grad)` (OUT_PROD 대신) |
| `ggml.c` 역전파 (신규) | NORM: `ggml_norm_rows_back` = (g − mean(g) − y·mean(g·y))/σ, y·σ는 x에서 재계산. GROUP_NORM: `[W·H·C/G, G·N]`로 reshape 후 같은 식. GEGLU(split): da = g·b·gelu′(a), db = g·gelu(a), gelu/gelu′는 `ggml_sigmoid`로(`ggml_gelu_tanh_sigmoid`, `ggml_gelu_tanh_grad`). CONCAT: grad 뷰를 잘라 cont. UPSCALE(nearest, 2배만): `pool_2d(avg, 2×2) × 4` |
| `ggml-vulkan.cpp` | OUT_PROD: 비연속 입력 허용, f16 src0 파이프라인(`out_prod_f16_f32`). IM2COL_BACK: 새 셰이더 `im2col_back.comp` + 연결. MUL_MAT: fp16 미지원/비활성 장치에서 비연속 f32×f32를 f32로 스테이징(`keep_f32`) |
| `tests/test-backend-ops.cpp` | OUT_PROD 전치 입력 케이스, `test_im2col_back` 8개 |
새 역전파 규칙은 기존 순전파 연산의 조합이라 셰이더가 필요 없고, 순전파는 입력 하나만 보관한다.

## 1.7 성능 / 메모리 (f32, rank 8, 배치 1)
| 해상도 | 모드 | s/스텝 (순전파/역전파) | 계산 버퍼 | GTT |
|---|---|---|---|---|
| 384 | 전체 그래프, 조합 버전 | 6.34 | 2,964 MiB | 7,027 MiB |
| 512 | ckpt + native | 12.4 (3.6/8.8) | 502 MiB | 4,747 MiB |
| 768 | ckpt + native | 28.7 (8.1/20.6) | 1,227 MiB | 5,655 MiB |
| 1024 | ckpt + native | 57.5 (15.7/41.7) | 2,607 MiB | 7,257 MiB |
| 1024 실제 200스텝 학습 | ckpt + native, f32 | 2~20스텝 59.8 → 41스텝~ 68.9, 총 3.76h | 2,607 MiB | 7,257 MiB (합계 약 8.5GiB) |
| 1024 실제 200스텝 학습 | ckpt + native, **fp16** | 45.6 → 48.3, 총 2.67h (1.4배) | 2,607 MiB | 7,257 MiB |
| **v0.2** 384 LoCon conv-rank 8 | 전체 그래프, 조합 버전 | 6.70 (+6%) | 3,755 MiB | 7,877 MiB (합계 9.19GiB, v0.1 8.85) |
| **v0.2** 768 어텐션 / LoCon conv-rank 8, 300스텝 | ckpt + native | 34.9 / 36.9 (+6%) | 1,227 / 1,396 MiB | 합계 7.2 / 7.1GiB |
| **v0.3** 1024 실제 200스텝 학습 | ckpt + native, **fp16 + 손실 스케일링 dynamic** | 39.6 → 40스텝~ 51, 평균 50.7 (같은 조건 f32 약 67.7 → **1.33배**), 넘침 0 | 2,607 MiB | 7,257 MiB (합계 최대 9.2GiB) |
| **v0.3** 256 LoCon conv-rank 4, 40스텝 | ckpt + native, fp16 dynamic | 2.72 (f32 4.08 → 1.5배) | 162 MiB | – |
| **v0.2** 1024 LoCon conv-rank 8, 200스텝 | ckpt + native | 77.3 (v0.1 어텐션 67.7, +14%, 역전파 블록 16→19) | 3,015 MiB | 7,700 MiB (합계 9.24GiB) |
| **v0.2** 256 LoCon conv-rank 4 | ckpt + native | 4.06 (1.10/2.88) | 162 MiB | 4,341 MiB |
| **v0.2** 512 LoCon conv-rank 8 (`--mem-plan`) | ckpt + native | – | 578 MiB (v0.1 502) | 예상 GPU 5,832 MiB (v0.1 5,706) |
- GPU 메모리 = GTT + 전용 VRAM(약 1GiB). UNet 가중치 4,900 MiB, LoRA+AdamW 약 133 MiB (LoCon conv-rank 8: 14.93M 파라미터 → 약 171 MiB)
- 행렬곱 실효 약 540 GFLOPS (f32). 660M f32 이론 최대 ≈ 6 CU × 64 × 2 × 1.9GHz ≈ 1.46 TFLOPS → 약 37%. 이 장치에서 SDXL 1024px 학습은 스텝당 약 1분
- fp16 행렬곱은 1.4배 빠름 (2배가 아님 → 행렬곱 외 대역폭 위주 연산, 전치 복사 등의 몫이 큼). 단 학습에는 쓸 수 없음 (1.8)
- 1024px 장시간 학습에서 20~40스텝 무렵부터 스텝 시간이 f32 약 15%, fp16 약 6% 늘어 유지. 384px 1시간 학습에선 없음. GPU 60~63°C, sclk 1900MHz 정상, `power_dpm_force_performance_level=high`(fclk 800→1600MHz)로도 변화 없음 → 원인 미확정 (APU 전력 한도/메모리 쪽 의심)

## 1.8 검증 현황
| 검증 | 결과 |
|---|---|
| tinyunet vs PyTorch (LoRA 기울기 rel L2) | CPU 1.8e-6, Vulkan 3.4e-6 (`--native all` 포함) |
| SDXL UNet 출력 vs sd.cpp (`--check-ref`) | 2.3e-4 (sd.cpp는 f16 im2col), 무작위 LoRA 적용 시 2.36e-4 |
| 체크포인팅 vs 전체 그래프 (`--grad-check`, 2번째 스텝) | 5.5e-7 ~ 8.2e-7 |
| test-backend-ops (Vulkan vs CPU) | MUL_MAT 2592/2592 (fp16 켬·끔), OUT_PROD 92/92, IM2COL 113/113, IM2COL_BACK 8/8 |
| 실제 학습 (dog-example 5장, 384px) | sd.cpp 생성에서 학습 대상이 나오고 프롬프트 장면도 따름 (2.9절) |
| tinyunet vs PyTorch, Vulkan 6조합 (v0.1 빌드) | 6/6 PASS, 기울기 3.1e-6~3.5e-6 (`verification/logs/tinyunet_vulkan_vs_pytorch.txt`) |
| 1024px 실제 학습 (f32, 200스텝) | NaN 없음, sd.cpp 1024px 생성에서 학습 대상 재현 + 장면 지시 따름 |
| **v0.2** tinyunet `--conv-lora` vs PyTorch (31 LoCon 모듈 추가, conv rank 3, 3x3 scale 0.5) | 10조건 PASS: CPU 1.7e-6/최악 3.9e-6, 16×16 seed 1 1.3e-6, Vulkan 3.3e-6, conv-f16 6e-4, native all Vulkan 3.3e-6 (CPU 1.5e-4 = CPU GELU fp16 표), 미지원 노드 0 (`verification/logs/stage2_tiny_unet.txt`) |
| **v0.2** SDXL LoCon 출력 vs sd.cpp (rank 8/alpha 4 + conv 4/alpha 8, up~N(0,0.05)) | **2.388e-4**, sd.cpp 1878/1878 적용. 대조: 합성곱 쪽 up만 0 → 1.143e-2 (48배) = 비교가 합성곱 기여를 잡음. v0.1 LoRA 회귀 2.355e-4 (`verification/sdxl_ref_locon/`, `logs/stage3_sdxl.txt`) |
| **v0.2** `--grad-check --conv-rank 4` (256px) | 6.2e-7 / 최악 2.5e-6, 역전파 블록 19개 |
| **v0.2** conv-rank 0 = v0.1 | 강아지 384px 학습 첫 20스텝 손실·t·샘플 비트 일치 |
| **v0.2** LoCon 자가 시험 (여우 1장 256px 40스텝 lr 1e-3) | 고정 seed 평균 손실 0.2445 → 0.1086, ckpt/비ckpt `--resume` 손실 완전 일치, up 0인 모듈 0 (`logs/stage3_selftest_train.txt`) |
| **v0.2** 강아지 384px 600스텝 LoCon vs 어텐션 | NaN 없음, 평균 손실 0.10797 vs 0.11009, 합성곱 쪽 66모듈 모두 학습, sd.cpp 생성에서 둘 다 개체+장면, LoCon이 세부 약간 나음 (2.12) |
| **v0.2** 해상도별 어텐션 vs LoCon (384/768/1024px, 같은 seed·학습 순서) | 1024px는 v0.1 조건 복원(1스텝 t=558 손실 0.03884 일치). 평균 손실 거의 같음(0.1101/0.1080, 0.0949/0.0946, 0.1051/0.1049). 합성곱 66모듈 전부 학습, 어텐션 방향 0.91/0.97/0.98. 생성 우열은 장면마다 엇갈림 → 이 데이터에선 차이 없음으로 결론 (README v0.2 결과, `verification/logs/stage5_*`) |
| **fp16 행렬곱 학습** | **실패: LoRA가 거의 학습 안 됨.** 1024px 200스텝, NaN 없음, 손실 평균 0.1065(f32 0.1051), 그러나 up 가중치가 정확히 0인 모듈 486/560, up 평균 크기 f32의 1/50, f32 결과와 방향 유사도 0, 생성 이미지 ≈ LoRA 없음. 원인 = 기울기 언더플로 (아래). tinyunet 기울기 오차 1e-3~3e-3 (작은 모델은 활성값이 커서 덜 드러남) |

| **v0.3** 손실 스케일 정확성 (f32 모드) | `--grad-check` S=1/1024/65536 모두 7.473e-7 동일. 20스텝 학습 결과 LoRA가 S=1과 S=1024에서 **1878/1878 텐서 비트 일치**(ckpt, 전체 그래프 각각), v0.2 실행 파일 결과와도 비트 일치 (`verification/logs/v03_loss_scale_exact.txt`) |
| **v0.3** fp16 + 손실 스케일링, 256px LoCon 40스텝 lr 1e-3 | 평균 손실 f32 0.11765 / fp16 dynamic 0.11790 / fp16 S=1 0.12986. up 방향 f32 대비 cos **0.960**·크기 비 0.997 (S=1: 0.390) |
| **v0.3** fp16 + 손실 스케일링, **1024px 200스텝** (v0.2 실패 실험과 같은 조건: dog1024, seed 1, rank 8, ckpt+native) | 넘침 0(200스텝 연속 정상 → 마지막 S=65536), 평균 손실 0.10518 (f32 0.1051). **up 방향 f32 대비 cos 0.999, 크기 비 0.999, 0인 모듈 0/560** (v0.2 fp16: cos 0.000, 크기 0.098, 0인 모듈 486). 생성 이미지 f32와 거의 같음 (README) |

**fp16 실패 원인 (기울기 언더플로)**: 손실은 잠재값 원소(1024px에서 4×128×128 = 65,536개) 평균이라 손실 기울기의 시작값이 원소당 약 2·diff/65536 ≈ 1e-5 규모. 이 기울기가 UNet을 거슬러 내려가며 더 작아지고, fp16 행렬곱 입력으로 변환될 때 fp16 최소 정규수(6.1e-5)·최소 비정규수(6e-8) 아래로 떨어져 0이 된다. 출력 쪽 일부 모듈(74/560)만 조금 학습됨. **손실 값으로는 이 실패를 알 수 없다**(손실은 대부분 원래 모델이 결정) → 학습 여부는 LoRA 가중치 크기(`verification/lora_norms.py`)나 생성 이미지로 확인할 것

## 1.9 플랫폼 메모
- ggml Vulkan은 fp16 지원 장치에서 f32 행렬곱도 fp16으로 계산 → 정확성 검증과 기본 학습은 `GGML_VK_DISABLE_F16=1`
- RADV(Mesa 25.0.7)는 VAE(head dim 512)용 flash attention 파이프라인 컴파일 중 크래시 → sd.cpp 생성은 `--flash-attn auto,vae=off` (+ `--vae-tiling`)
- iGPU의 GTT는 시스템 RAM이고 컨테이너/cgroup 메모리 한도에 잡히지 않음 → `tools/guard.sh`로 "GTT + 시스템 used" 합을 감시. guard는 합계를 보므로, 학습 중 다른 큰 작업(빌드 등)을 같이 돌리면 학습이 종료될 수 있음
- sd.cpp의 LoRA 적용: 미리 합치는 기본 방식은 분수 가중치에서 RAM이 순간 3GiB 더 듦 → `--lora-apply-mode at_runtime` (메모리 약 2GiB 적고 더 빠름)

## 1.10 다음 할 일
- ~~fp16 학습용 손실 스케일링~~ → **v0.3에서 완료** (1.5, 2.13)
- fp16 학습의 추가 검증: LoCon + fp16을 768/1024px 장시간으로, 다른 데이터(스타일). 서버 기본값은 2026-11-27 사용자 결정으로 fp16(true)
- 전용 역전파 커널(NORM/GROUP_NORM/GEGLU), 체크포인팅 재계산 줄이기
- 비율 버킷, 배치 > 1, 텍스트 인코더 LoRA, 트랜스포머 proj_in/out·ff LoRA(kohya 기본 대상), 블록별 rank, 다른 GPU 검증
- 사용자 우선순위 메모: 2026-11-26엔 fp16을 "반쯤 양자화"로 보고 보류 → 2026-11-27 v0.3 범위로 결정

## 1.11 빌드 (의존성, 함정)
- 빌드 의존 (Debian/Ubuntu 패키지명): `build-essential cmake git glslc libvulkan-dev spirv-headers` (+ 드라이버 `mesa-vulkan-drivers`, 확인용 `vulkan-tools`). 검증한 버전: glslc 2025.2, libvulkan-dev 1.4.309, spirv-headers 1.4.309, Mesa 25.0.7, GCC(Debian 13), glibc 2.41, Linux 7.0
  - `spirv-headers`가 없으면 ggml-vulkan CMake가 `SPIRV-HeadersConfig.cmake`를 못 찾아 실패
  - `glslc`가 없으면 Vulkan 셰이더 생성 단계에서 실패
- 순서: `patches/get_ggml.sh ggml` (leejet/ggml `sd.cpp` 브랜치 clone → `eeda9ec48fc84a3df0bb9dcc19ae9053075a2539` 체크아웃 → 패치 적용) → `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_DIR=ggml` → `cmake --build build -j4`
  - 패치 적용 실패 = 다른 ggml 커밋. 반드시 위 커밋 (패치는 이 커밋 기준 `git diff`)
  - CMake 옵션: `SDXL_VULKAN`(ON), `SDXL_TESTS`(ON, tinyunet). 내부에서 `BUILD_SHARED_LIBS=OFF`, `GGML_NATIVE=OFF` 강제
- 빌드 메모리: ggml-vulkan.cpp 등 큰 번역 단위가 작업당 1.5~2GiB → `-j4`면 약 6GiB. 메모리가 적으면 `-j1`/`-j2`
- `bin/`의 실행 파일: x86-64 Linux, glibc 2.41에서 빌드, ggml 정적 링크. 런타임 의존은 `libvulkan.so.1 libgomp.so.1 libstdc++.so.6 libgcc_s libm libc`. glibc가 더 오래된 배포판이면 `GLIBC_2.xx not found` → 소스에서 빌드
- ggml 자체 테스트(백엔드 비교)를 돌리려면 패치된 ggml을 단독 빌드: `cmake -S ggml -B gb -DGGML_VULKAN=ON -DGGML_BUILD_TESTS=ON && cmake --build gb -j4 --target test-backend-ops` → `gb/bin/test-backend-ops -o OUT_PROD -b Vulkan0` (`-o MUL_MAT`, `-o IM2COL`, `-o IM2COL_BACK`). 정밀도 비교 땐 `GGML_VK_DISABLE_F16=1`

## 1.12 실행 환경 요구
- Vulkan 장치가 ggml에 잡혀야 함: 학습기 시작 시 `ggml_vulkan: 0 = <장치 이름> ... | uma: 1 | fp16: ...` 출력. 안 잡히면 `vulkaninfo --summary`, `/dev/dri/renderD*` 권한(`render` 그룹) 확인. 여러 장치면 `GGML_VK_VISIBLE_DEVICES=<번호>`. 학습기는 항상 장치 0(보이는 것 중 첫째)을 씀. llvmpipe(CPU Vulkan)는 ggml이 안 잡음
- GPU 메모리 필요량(가중치 4.9GiB 포함, f32, rank 8): 512px ckpt 약 5.5GiB, 768px ckpt 약 6.6GiB, 1024px ckpt 약 8.2GiB, 384px 전체 그래프 약 8GiB. 다른 해상도/설정은 `--mem-plan`으로 계산 (GPU 없이 수 초)
- **AMD iGPU(amdgpu)의 GTT 크기**: GPU가 쓸 수 있는 시스템 RAM 상한. 기본값이 작으면(커널/TTM 기본은 RAM의 약 절반) 1024px에서 할당 실패. 현재 값: `cat /sys/class/drm/card0/device/mem_info_gtt_total` (바이트). 늘리기: 커널 파라미터 `amdgpu.gttsize=<MiB> ttm.pages_limit=<MiB×256>` (예: 11GiB = `amdgpu.gttsize=11264 ttm.pages_limit=2883584`) 후 재부팅. 시스템이 쓸 RAM은 남길 것
- GTT는 cgroup/컨테이너 메모리 한도에 잡히지 않는다 → 메모리 감시는 `tools/guard.sh` (GTT + 시스템 used 합). 학습 중 빌드 같은 큰 작업을 같이 돌리면 합계 초과로 guard가 학습을 종료할 수 있음
- 전용 VRAM(BIOS UMA 크기)은 GTT와 별개로 쓰인다. UMA를 키우면 그만큼 RAM에서 빠지므로 이득 없음
- CPU 백엔드(`--backend cpu`)도 동작하지만 SDXL 학습은 매우 느림 (검증/디버그용). `--threads N`(기본 8)

- **Docker (v0.3, README "Docker로 실행")**: 빌드 없이 `bin/`을 마운트. 환경 이미지 = `debian:trixie-slim` + `libvulkan1 mesa-vulkan-drivers libgomp1 procps`(guard용). `--device /dev/dri --group-add <render GID> --user $(id -u):$(id -g) -e HOME=/work`(Mesa 셰이더 캐시가 볼륨에 남게), `-v v0.3:/app:ro -v work:/work`, 서버는 `--host 0.0.0.0` + `-p 127.0.0.1:8090:8090`. 실행 파일 의존성: libvulkan.so.1, libgomp.so.1, libstdc++, glibc ≥ 2.38 (`objdump -T`로 확인). `--user`면 컨테이너 안 apt가 안 되므로 환경 이미지를 먼저 만드는 2단계. GTT는 Docker `--memory`에도 안 잡힘. guard는 lxcfs가 없으니 `free` 모드 → 호스트 전체 used를 봐서 보수적. 다른 GPU: AMD/Intel은 `/dev/dri`(같은 Mesa 패키지에 RADV·ANV), NVIDIA는 Container Toolkit `--gpus all` + `NVIDIA_DRIVER_CAPABILITIES`에 graphics 필요. 외장 GPU는 VRAM에 가중치 4.9GB + 계산 버퍼(1024px ckpt 2.6GB)가 다 들어가야 함 → 12GB 권장. coopmat(RDNA3+/NVIDIA/Arc) 경로의 fp16 수치·이 프로젝트 커널은 미검증. **Docker 위 실행은 미검증** (2026-11-27 사용자: 메모리가 빠듯해 이 기계에서 돌리지 말 것, 문서 검토만)

- **라이선스**: 이 프로젝트 MIT (`LICENSE`). 서드파티(ggml + 내장 YaRN, nlohmann/json 3.11.2 + 내장 Abseil/Hedley/Grisu2/UTF-8 디코더, cpp-httplib, Vulkan-Hpp/Headers 1.4.309)는 `THIRD_PARTY_LICENSES.md`에 버전·위치·원문 전체. 외부 코드를 넣거나 바꾸면 같은 변경에서 THIRD_PARTY_LICENSES.md와 LICENSE 끝 요약을 함께 갱신 (`AGENTS.md`). 실행 파일은 Releases로 배포, 학습한 LoRA는 배포 안 함 (`.gitignore`)

## 1.13 CLI 레퍼런스 (`sdxl_lora`)
모드는 하나만: `--train <dir>` (+`--ckpt`, `--grad-check`), `--check-ref <dir>`, `--mem-plan <N>`. 모드가 없으면 종료 코드 2.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model <file>` | `models/sd_xl_base_1.0.safetensors` | SDXL 단일 파일 체크포인트. `model.diffusion_model.*` 키만 읽음 (F16/F32/BF16 헤더 지원, 검증은 F16만). diffusers 폴더 형식은 미지원 |
| `--backend vulkan\|cpu` | `vulkan` | |
| `--threads N` | 8 | CPU 백엔드 스레드 |
| `--native <목록>` | 없음 | 쉼표 목록 `norm,gn,geglu,concat,upscale` 또는 `all`. 지정한 연산은 ggml 원래 연산 + 패치의 역전파. 없으면 조합 버전. **학습엔 `all` 권장** (메모리 −30%, 수치 안정) |
| `--conv-f16` | off | im2col을 f16으로 (sd.cpp와 같은 조건, 정확도 비교용) |
| `--train <dir>` | | 캐시 폴더. `*.latent.npy` 기준으로 샘플 목록, 이름순. 모든 샘플 같은 해상도여야 함 |
| `--ckpt` | off | 그래디언트 체크포인팅 (768px 이상 필수) |
| `--grad-check` | off | `--ckpt` 포함. 2스텝 돌고 2번째 스텝에서 ckpt 기울기 vs 전체 그래프 기울기 비교 후 종료 (전체 그래프 메모리가 필요하므로 384px 이하에서) |
| `--steps N` | 100 | |
| `--lr F` | 1e-4 | `0`이면 업데이트 없음 (고정 seed 손실 평가용). lr>0일 때만 마지막에 저장 |
| `--rank R` / `--alpha A` | 8 / 8 | LoRA 스케일 = alpha/rank (어텐션, LoCon의 1x1·emb Linear) |
| `--conv-rank R` (`--conv-dim`) / `--conv-alpha A` | 0 / =conv-rank | v0.2 LoCon. 0이면 끔. k>1 합성곱 38모듈의 rank/alpha |
| `--seed S` | 42 | 샘플 선택·timestep·노이즈·LoRA 초기화 RNG (mt19937) |
| `--t-min/--t-max` | 0 / 999 | timestep 균등 샘플 범위 |
| `--loss-scale S\|dynamic` | 자동 | **v0.3** 손실 스케일링 (1.5). 자동 = fp16 행렬곱이면 `dynamic`, 아니면 `1`. 고정값에서 기울기 NaN/Inf → `기울기에 NaN/Inf (step N, 고정 손실 스케일 ...)` 오류. 시작 로그 `손실 스케일링: dynamic, S=32768 (fp16 행렬곱 켜짐)`, fp16인데 S=1이면 경고 |
| `--loss-scale-init S` / `--loss-scale-window N` | 32768 / 200 | **v0.3** dynamic 시작값 / S×2까지 연속 정상 스텝 수 |
| `--save-every N` | 0 | N스텝마다 `<out>_s<step>.safetensors`. 최종본은 `<out>` |
| `--out <file>` | `out/lora.safetensors` | 폴더는 자동 생성, 임시 파일에 쓰고 rename |
| `--resume <file>` / `--lora <file>` | | 같은 옵션. `--train`: 이어서 학습 (`--alpha`/`--conv-alpha`가 파일과 같아야 함, rank는 파일에서 읽지 않으므로 `--rank`/`--conv-rank`도 맞출 것. 모양이 다르면 `모양 다름`, 모듈이 없으면 `LoRA 키 없음`). `--check-ref`: 그 LoRA를 건 출력 비교 (rank/conv_rank는 down 모양에서, alpha는 모듈별 `.alpha`에서 읽음: `LoraSet::probe`) |
| `--check-ref <dir>` | | sd.cpp 기준값과 UNet 출력 비교 (1.14) |
| `--mem-plan N` | | 잠재값 한 변 N(=픽셀/8)으로 학습 그래프 메모리 계산. `--ckpt`, `--native` 반영. GPU·데이터 불필요 |

**종료 코드**: 0 정상 / 1 허용 오차 초과(`--check-ref`: eps rel_l2 ≥ 1e-2, `--grad-check`: ≥ 1e-4) / 2 실행 오류(인자, 파일, 지원 안 되는 노드, 할당 실패, NaN 등. 메시지는 stderr `오류: ...`) / 3 `SDXL_NAN_DEBUG` 보고 후 종료

**출력 로그**:
```
샘플 5개, 잠재값 128x128 (1024px)          # (--ckpt면 앞에 [ckpt])
UNet 가중치: 1680개, 4900.1 MiB (Vulkan0)
LoRA: 모듈 560개, rank 8, 파라미터 11.61M
[ckpt] 계산 버퍼 2607.1 MiB (그래프 18개 공유), GTT 7246 MiB
step    1  t=558  loss 0.03884  ema 0.03884  57.53s (fwd 15.73 bwd 41.73)  buf 2607 MiB  GTT 7257 MiB  [샘플 이름]
...
평균 스텝 시간 (첫 스텝 제외): 57.49s
평균 손실: 0.05189
저장: out/x.safetensors (1680 텐서, step 200)
```
- `loss`는 그 스텝 하나의 MSE라 timestep에 따라 0.002~0.4로 크게 흔들림(정상). 추세는 `ema`(0.95 지수평균)로 본다. 학습이 되는지는 `--lr 0 --resume`으로 같은 seed의 손실을 전후 비교하는 게 확실
- `GTT`는 `/sys/class/drm/card0/device/mem_info_gtt_used` (AMD만, 없으면 -1)

**환경변수**
| 변수 | 효과 |
|---|---|
| `GGML_VK_DISABLE_F16=1` | 행렬곱을 f32로 (v0.2와 비트 일치하는 정밀 경로, 정확성 검증용). **v0.3**: 없으면 fp16 행렬곱 + 손실 스케일링 자동 dynamic → f32와 사실상 같은 LoRA를 1.33배 빠르게 (1.8). 서버는 `fp16` 키가 false일 때만 설정 (v0.3 기본 true) |
| `GGML_VK_SUBALLOCATION_BLOCK_SIZE` | 학습기가 기본 4GiB로 설정(이미 있으면 존중). 계산 버퍼보다 작게 지정하면 `--ckpt`에서 "그래프 할당이 안정되지 않음" |
| `GGML_VK_VISIBLE_DEVICES` | 사용할 Vulkan 장치 |
| `GGML_VK_PERF_LOGGER=1` | 그래프 계산마다 연산별 시간 출력 (프로파일) |
| `SDXL_NAN_DEBUG=1` | `--steps N`의 N번째 스텝에서 순전파·역전파(**v0.3**) 노드를 하나씩 계산해 처음 비유한 값이 나온 노드와 입력 최대값 출력, 종료 코드 3. `--ckpt` 아닌 `train()`에만 있음 |
| `SDXL_NAN_TRACE=1` | (NAN_DEBUG와 함께) |값|>1e4인 노드도 출력 |

## 1.14 데이터/파일 형식
**캐시 (`tools/make_cache.py` 출력, 모두 little-endian f32, C 순서)**
| 파일 | numpy 모양 | ggml ne | 내용 |
|---|---|---|---|
| `<n>.latent.npy` | (1, 4, H/8, W/8) | [W/8, H/8, 4, 1] | VAE 인코드(posterior `mode`) 후 `vae_to_unet_latents` (SDXL 스케일 0.13025 적용됨) |
| `<n>.crossattn.npy` | (1, 77, 2048) | [2048, 77] | CLIP-L(768) + OpenCLIP-G(1280) 끝에서 두 번째 층 concat |
| `<n>.vector.npy` | (2816,) | [2816, 1] | pooled CLIP-G 투영(1280) + original_size·crop(0,0)·target_size 임베딩(각 256×2). sd.cpp `encode_prompt(width=size, height=size)` |
| `<n>.check.png` | | | 잘라서 줄인 학습 이미지 (확인용) |
| `captions.txt` | | | `이름\t원본 경로\t캡션` |
| `cache.dgl`, `cache.log` | | | 생성된 sd.cpp DGL 스크립트와 실행 로그 |
- 학습기 `.npy` 로더는 `'<f4'`, `fortran_order: False`만 지원
- 다른 도구로 캐시를 만들 땐 위 모양/스케일만 맞으면 됨 (예: diffusers로 만든 SDXL 잠재값 × 0.13025, `text_embeds`/`time_ids` → vector)

**LoRA 출력 (kohya / SGM 이름)**: 모듈 이름 `input_blocks.4.1.transformer_blocks.0.attn1.to_q` → 키 접두사 `lora_unet_input_blocks_4_1_transformer_blocks_0_attn1_to_q` (`.`→`_`). 모듈당 3키: `.lora_down.weight` F32 `[rank, in]`, `.lora_up.weight` F32 `[out, rank]`, `.alpha` F32 스칼라(모양 `[]`). 560모듈 × 3 = 1680키, rank 8이면 46.7MB. 메타데이터: `ss_network_module=networks.lora`, `ss_network_dim`, `ss_network_alpha`, `ss_base_model_version=sdxl_base_v1-0`, `ss_steps`, `trainer`. sd.cpp에서 1680/1680 적용 확인.
**LoCon 키 (v0.2)**: 같은 규칙, 예 `lora_unet_input_blocks_1_0_in_layers_2` → `.lora_down.weight` `[rank, in, kh, kw]`, `.lora_up.weight` `[out, rank, 1, 1]`, `.alpha` 모듈별(3x3은 conv_alpha, 1x1·emb는 alpha). emb_layers.1은 Linear라 2D. 626모듈 × 3 = 1878키. 메타데이터에 `ss_network_args` = JSON 문자열 `{"conv_alpha":"..","conv_dim":".."}` (kohya와 같음). sd.cpp에서 1878/1878 적용 확인. 다른 UI(ComfyUI, A1111 계열)는 kohya SGM 이름을 보통 지원하지만 이 프로젝트에서 직접 검증하지 않음

**`--check-ref <dir>` 입력** (`tools/ref_unet.dgl`을 `sd-cli -M dgl --script tools/ref_unet.dgl --tensor x x.npy --tensor d d.npy --tensor crossattn crossattn.npy --tensor vector vector.npy --tensor params params.npy`로 생성; DGL 안의 `model_path`, `width/height/latent`, `t`(0~1 정규화), `seed`, `backend` 수정 가능):
`x.npy`(입력 잠재값), `d.npy`(sd.cpp의 D(x,σ); `--lora`면 `d_lora.npy`), `crossattn.npy`, `vector.npy`, `params.npy` = [σ, c_in, c_out, c_skip, t]. 비교: ε_ref = (D − c_skip·x)/c_out vs 학습기 ε(입력 x·c_in, timestep = sigma_to_t(σ)). 결과 `eps_ours.npy` 저장

## 1.15 서버 API 레퍼런스 (`sdxl_lora_server`)
실행: `sdxl_lora_server --root <작업 폴더> [--cli <sdxl_lora>] [--host 127.0.0.1] [--port 8090]`. `--cli` 기본은 서버 실행 파일과 같은 폴더의 `sdxl_lora`. 자식 프로세스는 `--root`를 현재 폴더로 실행되므로, 학습기의 상대 경로 기본값(`models/...`)도 root 기준.

| 메서드/경로 | 요청 | 응답 |
|---|---|---|
| `GET /` | | 진행 상황 HTML (5초 새로고침) |
| `GET /api/status` | | JSON: `state`(`idle`/`running`/`stopping`/`done`/`failed`/`stopped`), `args`(실행 명령), `steps`, `exit_code`, 그리고 로그에 step 줄이 있으면 `step`, `loss`(문자열, nan 가능), `ema`, `sec_per_step`, `eta_sec`, `log_tail`(마지막 20줄) |
| `POST /api/train` | JSON (아래) | 200 + status / 400 `{"error"}` (경로, JSON) / 409 (실행 중) / 500 (fork 실패) |
| `POST /api/stop` | | 200 `{"ok":true}` (SIGTERM, 상태 `stopping`→`stopped`) / 409 |
| `GET /api/log` | | 현재(마지막) 작업 로그 전체 text |
| `GET /files/<경로>` | | root 안 파일 (png/jpg/json/txt/log는 해당 MIME, 나머지 octet-stream) / 404 |

`/api/train` JSON (괄호는 기본값): `data`(필수, root 안 폴더), `out`("out/lora.safetensors"), `steps`(200), `lr`(1e-4), `rank`(8), `alpha`(8), `conv_rank`(0, v0.2), `conv_alpha`(키가 있을 때만 전달, 없으면 학습기 기본 = conv_rank), `seed`(42), `save_every`(0), `t_min`(0), `t_max`(999), `ckpt`(true), `native`("all", 빈 문자열이면 옵션 생략), `resume`(root 안 존재 파일), `model`(root 안 존재 파일), `fp16`(**v0.3 기본 true** → 학습기가 자동으로 손실 스케일링 dynamic. false → 자식에 `GGML_VK_DISABLE_F16=1`, f32), `loss_scale`(**v0.3**, `"dynamic"` 또는 양수 → `--loss-scale`, 그 밖은 400. 없으면 학습기 자동). 진행 파싱 정규식은 `^step ...`이라 v0.3의 줄 끝 `S=`와 `overflow step` 줄의 영향 없음.
- 로그: `<root>/logs/train-YYYYmmdd-HHMMSS.log`. 상태는 메모리에만 있어 서버 재시작 시 사라짐 (자식은 계속 돌 수 있음 → `ps`로 확인)
- 보안: 인증 없음, 경로 검사는 `weakly_canonical` 후 root 기준 상대 경로가 `..`로 시작하면 거부. 외부 공개가 필요하면 리버스 프록시 + 인증을 앞에 둘 것

## 1.16 도구 레퍼런스
| 도구 | 사용 / 옵션 |
|---|---|
| `tools/make_cache.py <src> <dst>` | `--size 512`(64의 배수), `--caption ""`(같은 이름 .txt가 있으면 그게 우선), `--prefix ""`(캡션 앞에 붙임, 트리거 단어용), `--model`, `--vae`(sdxl-vae-fp16-fix 권장), `--vae-backend ""`(빈값=Vulkan, `cpu` 가능; 1024px는 cpu 권장), `--sd-cli sd-cli`, `--dry-run`(스크립트 생성+문법 검사만). 이미지 확장자 png/jpg/jpeg/bmp/webp. 가운데 정사각형 자르기 → bicubic+antialias 축소. 텍스트 인코더는 항상 CPU |
| `tools/eval_prompt.sh <tag> "<prompt>" [lora] [w=1] [size=768] [seed=42]` | 환경변수 `SD_CLI MODEL VAE LORA_DIR(out) OUT_DIR(out/eval) STEPS(20) CFG(7) GUARD SD_EXTRA`. w=0이면 LoRA 없음. 이미 있으면 건너뜀 |
| `tools/gallery.py [dir]` | 파일 이름 규칙으로 같은 프롬프트를 묶은 `index.html`. 정적 서빙 예: `python3 -m http.server 8000 --bind 127.0.0.1 --directory out/eval` |
| `tools/guard.sh <log> <cmd...>` | `GUARD_LIMIT_MIB`(11264), `GUARD_DRM_DEV`(/sys/class/drm/card0/device). 0.2초마다 `시각 gtt= vram= ct= sum= fu=` 기록 (v0.2: 첫 줄 `GUARD mode=cgroup|free`. `/proc/meminfo`가 lxcfs면 ct = cgroup v2 `memory.stat` anon+kernel+shmem, 아니면 `free -m`. fu = 비교용 `free -m` used — LXC에선 GTT 업로드 순간 수 GiB 거짓으로 튐, 2.12), 합 초과 시 SIGTERM(자식 포함) → 3초 후 SIGKILL, 명령은 `oom_score_adj=1000`. 마지막 줄 `GUARD done rc= peak_sum=` |
| `tools/guard_runs.py <log>` | guard 로그의 실행별 시작/끝/걸린 시간/최대 합계 |
| `tools/st_header.py <file> [regex]` | safetensors 헤더 요약 (데이터는 안 읽음) |
| `tools/npy_info.py <files>` | npy 모양/통계 (numpy 필요) |
| `tools/ref_unet.dgl` | `--check-ref` 기준값 생성용 sd.cpp DGL |
| `tests/ref_tiny_unet.py --out f.gguf [--size 8] [--seed 0]` | 축소판 정답 생성 (torch, numpy, gguf 필요, float64). 크기는 짝수 |
| `tests/tinyunet <gguf> [--backend cpu\|vulkan] [--native ...] [--fwd-only] [--conv-f16] [--threads N] [--compare]` | 허용 오차: f32 순전파 1e-4 / 기울기 전체 2e-4·최악 8e-4. fp16이 끼는 경로(`--conv-f16`, CPU의 native geglu는 GELU를 fp16 표로 계산)는 2e-3 / 5e-3. 종료 코드 0/1/2. `TINY_CMP_MM=1`(+`--compare`): 오차 큰 MUL_MAT 모양 출력 |

## 1.17 처음부터 재현 (검증 순서)
```sh
# 0) 빌드 (1.11). 모델: models/sd_xl_base_1.0.safetensors, models/sdxl_vae.safetensors (fp16-fix)
# 1) 축소판 vs PyTorch
python3 tests/ref_tiny_unet.py --out tests/tiny8.gguf --size 8
build/tinyunet tests/tiny8.gguf                                                  # CPU: PASS, 기울기 ~1.8e-6
GGML_VK_DISABLE_F16=1 build/tinyunet tests/tiny8.gguf --backend vulkan --native all   # PASS, ~3.4e-6
# 2) SDXL UNet vs sd.cpp
mkdir -p ref && sd-cli -M dgl --script tools/ref_unet.dgl --tensor x ref/x.npy --tensor d ref/d.npy \
   --tensor crossattn ref/crossattn.npy --tensor vector ref/vector.npy --tensor params ref/params.npy
GGML_VK_DISABLE_F16=1 build/sdxl_lora --check-ref ref --native all              # eps rel_l2 ≈ 2.3e-4
# 3) 데이터 캐시 + 체크포인팅 기울기 검사
python3 tools/make_cache.py dataset/dog cache/dog384 --size 384 --caption "a photo of sks dog" --sd-cli sd-cli
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog384 --grad-check --native all  # rel_l2 < 1e-6
# 4) 학습 효과 (같은 seed 손실 전후)
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog384 --native all --steps 8 --seed 123 --lr 0
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog384 --native all --steps 200 --out out/dog.safetensors
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog384 --native all --steps 8 --seed 123 --lr 0 --resume out/dog.safetensors  # 평균 손실 감소
# 5) 생성 비교
SD_CLI=sd-cli tools/eval_prompt.sh beach "a photo of sks dog on a sunny beach" dog 0
SD_CLI=sd-cli tools/eval_prompt.sh beach "a photo of sks dog on a sunny beach" dog 1
```
v0.2 LoCon 검증 추가분:
```sh
python3 tests/ref_tiny_unet.py --out tests/tiny8c.gguf --size 8 --conv-lora
build/tinyunet tests/tiny8c.gguf && GGML_VK_DISABLE_F16=1 build/tinyunet tests/tiny8c.gguf --backend vulkan --native all
# 실물: 1스텝 학습으로 LoCon 키 파일 → up 무작위 → sd.cpp 기준값 → 비교
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/x256 --ckpt --native all --steps 1 --lr 1e-6 --rank 8 --alpha 4 --conv-rank 4 --conv-alpha 8 --out seed.safetensors
python3 make_test_lora.py seed.safetensors ref_conv/test_lora_conv.safetensors 0.05 0   # (v0.1 tools: lora_up만 N(0,std)로)
sd-cli -M dgl --script verification/sdxl_ref_locon/ref_unet_locon.dgl --tensor x ref_conv/x.npy --tensor d ref_conv/d_lora.npy \
   --tensor crossattn ref_conv/crossattn.npy --tensor vector ref_conv/vector.npy --tensor params ref_conv/params.npy   # 로그: 1878 / 1878
GGML_VK_DISABLE_F16=1 build/sdxl_lora --check-ref ref_conv --lora ref_conv/test_lora_conv.safetensors --native all   # ≈ 2.39e-4
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/x256 --grad-check --native all --conv-rank 4                    # < 1e-6
```
(`ref_unet_locon.dgl`의 `lora_path`는 저장소 루트 기준 경로로 바꿔 쓸 것. 기준 데이터는 `verification/sdxl_ref_locon/`에 그대로 있음)

검증에 쓴 sd.cpp: stable-diffusion.cpp `master-1127-79dc7ca` (79dc7ca9074d79419a46380cdb95a4524db9b74c), `-DSD_VULKAN=ON`. DGL 문법/함수는 버전에 따라 바뀔 수 있음

## 1.18 문제 해결
| 증상 | 원인 | 해결 |
|---|---|---|
| `백엔드 초기화 실패: vulkan` | Vulkan 장치 없음/권한 | `vulkaninfo --summary`, `/dev/dri/renderD*` 권한, 드라이버 설치 |
| 노드 목록과 함께 `unsupported=N` 또는 `UNSUPPORTED on vulkan` | 패치 안 된 ggml로 빌드 | `patches/get_ggml.sh`로 받은 ggml로 다시 빌드 |
| `텐서 없음: <이름>` | SDXL base 단일 파일 체크포인트가 아님 (diffusers 폴더, SD1.5, 다른 구조 등) | `model.diffusion_model.input_blocks...` 키가 있는 SDXL 체크포인트 사용 (`tools/st_header.py`로 확인) |
| `그래프 메모리 할당 실패`, Vulkan 메모리 할당 오류 | GPU 메모리 부족 | `--ckpt`, `--native all`, 해상도 낮추기, `--mem-plan`으로 필요량 확인, iGPU면 GTT 늘리기(1.12) |
| `그래프 할당이 안정되지 않음` (`--ckpt`) | 계산 버퍼가 Vulkan 조각 크기보다 커 할당기 재할당 반복 | `GGML_VK_SUBALLOCATION_BLOCK_SIZE`를 지정했다면 제거하거나 계산 버퍼보다 크게 (장치 `max_buffer_size`가 계산 버퍼보다 작으면 해상도를 낮출 것) |
| `ggml-vulkan.cpp: GGML_ASSERT(d_D->size >= d_buf_offset + d_sz)` | 그래프 여러 개가 할당기를 공유하다 버퍼가 다시 잡힘 (v0.1 이전 학습기 버그) | v0.1 학습기 사용 |
| `손실이 NaN/Inf (step N, t=.., <샘플>)` | 수치 문제 | `--native all`(GELU overflow 제거) 사용, fp16 끄기(`GGML_VK_DISABLE_F16=1`), lr 낮추기. 원인 노드: `SDXL_NAN_DEBUG=1 ... --steps N` (같은 seed) |
| `기울기에 NaN/Inf (step N, 고정 손실 스케일 S=..)` | **v0.3** 고정 S에서 fp16 넘침 | `--loss-scale dynamic`(fp16 기본) 또는 S 줄이기 |
| `overflow step N: 기울기 NaN/Inf → 갱신 건너뜀, S=..` | **v0.3** dynamic 정상 동작 (그 스텝 갱신 생략) | 가끔이면 무시. 계속 나면 `--loss-scale-init`을 낮추기 |
| `기울기 NaN/Inf가 계속됨` | **v0.3** 20번 연속 넘침 또는 S<1 | 스케일링 문제가 아님 → f32(`GGML_VK_DISABLE_F16=1`)로 재현되는지, `SDXL_NAN_DEBUG` |
| fp16 학습인데 LoRA 효과 없음 | S=1(손실 스케일링 꺼짐)이면 기울기 언더플로 | 로그의 `S=` 확인, `verification/lora_norms.py`로 up이 0인 모듈 수 |
| `샘플 크기가 다름` | 캐시 해상도 혼합 | 같은 `--size`로 다시 캐시 |
| `--alpha/--conv-alpha가 파일과 다름` / `LoRA 키 없음` / `같은 묶음 안에서 모듈마다 alpha가 다름` | `--resume` 파일이 이 학습기 형식/설정과 다름 | 같은 `--alpha --rank`, 이 학습기가 만든 파일 사용 (외부 LoRA 이어받기는 모든 560(LoCon이면 626)모듈 키가 있어야 함) |
| 첫 스텝만 느림 | 셰이더 파이프라인 컴파일 | 정상. 평균은 첫 스텝 제외 |
| 표보다 훨씬 느림 | 다른 GPU/드라이버, 클럭 | `GGML_VK_PERF_LOGGER=1`로 연산별 시간 확인. 역전파가 순전파의 3배를 크게 넘으면 OUT_PROD가 쓰이는지 확인 (패치의 MUL_MAT 역전파 변경 누락) |
| make_cache: `unknown posterior`, DGL 문법/함수 오류 | sd-cli 버전 차이 | 1.17의 sd.cpp 버전 사용, 또는 생성된 `cache.dgl`을 그 버전 문법에 맞게 수정 |
| make_cache 1024px에서 GPU 메모리 부족/크래시 | VAE 인코드 메모리 | `--vae-backend cpu` |
| sd-cli 생성 중 `free(): invalid pointer` / SIGSEGV (VAE 단계) | RADV의 VAE flash attention 파이프라인 컴파일 크래시 | `--flash-attn auto,vae=off` (+ `--vae-tiling`) |
| sd-cli에서 LoRA 걸 때 RAM 급증/OOM | 분수 가중치 LoRA를 미리 합치는 경로 | `--lora-apply-mode at_runtime` |
| LoRA 효과 없음 (sd.cpp) | 로그에 `(1680 / 1680)`(LoCon이면 `1878 / 1878`) `LoRA tensors have been applied`가 없음 | `--lora-model-dir`와 `<lora:파일이름:가중치>`(확장자 빼고) 확인 |
| LoRA를 걸어도 효과가 거의 없음, `lora_norms.py`로 보면 up 가중치 대부분이 정확히 0 | fp16 행렬곱으로 학습 (`GGML_VK_DISABLE_F16` 빠짐) → 기울기 언더플로 | v0.2 이하에서 생김. v0.3은 손실 스케일링 자동 (로그의 `S=`가 1이면 `--loss-scale dynamic`), 또는 `GGML_VK_DISABLE_F16=1`로 f32 학습 (서버는 `fp16:false`) |
| 결과에 학습 사진 배경/색이 따라옴 | 캡션이 모두 같아 배경이 문장에 묶임 + 과학습 | 프롬프트에 배경 명시, 스텝 줄이기(이 데이터는 200~400), LoRA 가중치 0.7, 배경 다양한 데이터/배경 캡션 |
| 서버 `CLI 학습기를 찾을 수 없음` | `sdxl_lora`가 서버와 같은 폴더에 없음 | `--cli <경로>` |
| 서버 400 `--root 안의...` | 경로가 root 밖/없음 | root 기준 상대 경로 사용 |
| `GLIBC_2.xx not found` | `bin/` 실행 파일이 새 glibc로 빌드됨 | 소스에서 빌드 |

## 1.19 확장 가이드
- **LoRA 대상 추가** (예: proj_in/out, ff): `LoraSet::create`의 접미사 목록에 추가 + `UNetBuilder`에서 해당 층을 `linear()` 대신 `lora_linear()`로. 합성곱 LoRA는 v0.2에서 구현됨(1.4) — 다른 합성곱을 대상에 넣으려면 `LoraSet::create`의 `RES` 접미사 목록만 고치면 `conv()`가 알아서 건다. kohya 키 이름은 모듈 이름에서 자동 생성. 검증은 `ref_tiny_unet.py`의 `make_conv_lora` 호출 추가 → tinyunet
- **다른 UNet 계열**: `UNetConfig`(채널/배수/res 개수/트랜스포머 깊이/head_dim/그룹)로 표현되는 SGM/ldm UNet이면 빌더 재사용 가능. SDXL 전용 가정: `label_emb`(ADM 2816) 존재, 트랜스포머가 linear proj_in/out, context 2048. SD1.5/2.x는 label_emb 없음·conv proj·head 수 고정 등 수정 필요
- **배치 > 1**: 입력 텐서 ne[3]과 빌더의 `[W,H,C,1]` 가정(reshape/permute, concat의 ne3=1)을 바꿔야 함. 대안은 기울기 누적(여러 스텝의 g_store 합산 후 AdamW)
- **비율 버킷**: 해상도마다 그래프가 다름 → 해상도별 그래프/할당 묶음을 만들어 샘플마다 선택 (체크포인팅 모드의 단일 할당기 공유 규칙(1.5)을 지킬 것)
- **fp16**: `GGML_VK_DISABLE_F16`를 빼면 ggml이 fp16 행렬곱 사용. 손실 스케일링은 없음 (AdamW·LoRA는 f32). 검증 방법: 같은 데이터·seed로 f32와 손실 곡선, `--lr 0 --resume` 손실, 생성 이미지 비교
- **전용 역전파 커널**: ggml에 새 op(`GGML_OP_*_BACK`) + CPU 구현 + Vulkan 셰이더. Vulkan 연결 지점은 IM2COL_BACK 예와 같음: `vulkan-shaders-gen.cpp`(`string_to_spv`), 파이프라인 생성(`ggml_vk_create_pipeline`), `ggml_vk_op_get_pipeline`, `ggml_vk_op_f32`의 디스패치 크기 switch, `ggml_vk_build_graph`의 op switch, `ggml_backend_vk_device_supports_op`, 푸시 상수 구조체(`ggml-vulkan-push-constants.h`), 파이프라인 멤버(`ggml-vulkan-types.h`). 검증은 `test-backend-ops`에 케이스 추가 + tinyunet
- **ggml 버전 올리기**: 패치를 새 ggml에 다시 적용(충돌 수동 해결) → 1.17의 1~3단계 재검증 필수. 특히 MUL_MAT 비연속 스테이징(`keep_f32`)과 역전파 규칙 부분은 상류 변경과 겹치기 쉬움

## 1.20 성능/메모리 조절 손잡이
- 해상도: 시간은 픽셀 수에 대략 비례(+어텐션 제곱 항). 1024px는 512px의 약 4.6배
- `--ckpt`: 메모리 대폭 감소, 시간 약 +30%. 384px 이하·메모리 여유 있으면 끄는 게 빠름
- `--native all`: 메모리 감소 + 노드 수 감소(SDXL 순전파 8,251 → 5,100)
- rank: 메모리/시간 영향 작음 (LoRA 파라미터 rank 8 = 11.6M)
- fp16 행렬곱: 미검증이지만 예상 1.5~2배
- 학습 이미지 수와 무관하게 시간 = 스텝 수 × s/스텝. 대상 하나(DreamBooth류) 5~20장은 200~1000스텝, 그림체는 보통 30~100장 이상·1500스텝 이상 (경험칙)

---

# 2부. 시행착오 (시간 순)

## 2.1 연산 조사
sd.cpp 그래프를 덤프해 연산별로 역전파 규칙 유무와 Vulkan 지원을 표로 만들었다. 결론: 순전파는 전부 Vulkan, 역전파 규칙 없는 연산 5개(NORM, GROUP_NORM, GEGLU, CONCAT, UPSCALE), Vulkan 쪽 빈칸 2개(OUT_PROD 제한, IM2COL_BACK 없음). flash attention은 역전파가 없어 학습은 FA off 경로. LoRA라 가중치 기울기는 불필요.

## 2.2 검증 틀부터
SDXL 규모에서 틀린 기울기를 찾는 건 사실상 불가능하므로, 같은 구조의 축소판을 PyTorch와 먼저 맞췄다. 역전파 규칙이 없는 5개 연산은 처음엔 **기본 연산 조합**(sum_rows/repeat/sub/div, exp 등)으로 써서 백엔드 작업 없이 정확성부터 확보. 이 과정에서 ggml 버그 2개 발견: PERMUTE/TRANSPOSE 역전파의 비연속 기울기(→ MUL 역전파 assert), SUB/DIV 역전파의 브로드캐스트 누락.

## 2.3 Vulkan 역전파 채우기
- OUT_PROD: 셰이더가 이미 stride 인덱싱 → `supports_op`의 연속성 제한만 해제 + f16 변형. 기존 테스트에 전치 입력 케이스가 없어 추가
- IM2COL_BACK: 셰이더 작성. GLSL에서 음수 `%`는 정의되지 않으므로 음수 먼저 거름. 1D 호출에서 s1=0이 넘어와 `% 0` → 호스트에서 보정
- 여기서 처음으로 역전파 전체가 Vulkan에서 돌았지만 기울기 오차 4e-4로 기준 초과

## 2.4 Vulkan 정밀도 버그 찾기
`--compare` 도구(노드 하나씩 계산, 매 노드 후 CPU 사본에 Vulkan 결과를 덮어써 입력을 같게 → **연산 자체 오차**만 측정)로 범인을 8개 노드로 좁힘: 어텐션 QKᵀ(두 입력 모두 permute된 비연속 f32). `ggml_vk_mul_mat_q_f16`이 비연속 피연산자를 fp16 비활성 여부와 상관없이 f16으로 복사하고 있었음 → f16 꺼진 장치에서는 f32로 복사. 오차 3.2e-4 → 5.6e-7, 전체 기울기 3.4e-6.

## 2.5 학습기 구조 결정
sd.cpp 내부(DGL/runner)에 학습을 넣는 대신 독립 학습기. 축소판 UNet 코드를 설정 기반 빌더로 일반화해 학습기와 공유 → 축소판 검증이 그대로 학습기 검증. 인코더는 sd.cpp DGL로 캐시. 실제 SDXL은 sd.cpp 출력과 비교(DGL에서 `sample(x, [σ,0], euler)`가 D(x,σ) 한 번이라는 점 이용), LoRA 형식은 무작위 kohya LoRA를 양쪽에 걸어 비교. 구현 중 ggml 텐서 이름 63자 제한으로 LoRA 맵 키가 잘려 충돌 → 맵 키를 std::string으로.

## 2.6 속도: OUT_PROD
첫 학습 스텝이 256px 16.5초인데 순전파는 1초. `GGML_VK_PERF_LOGGER`로 보니 OUT_PROD가 89%(원소마다 루프 도는 단순 셰이더). MUL_MAT 역전파를 전치 복사 + MUL_MAT으로 바꿔(ggml에 원래 주석 처리돼 있던 방식) 2.94초. 같은 수학이라 손실이 소수점까지 동일.

## 2.7 NaN: GEGLU 조합 버전
384px 학습 9스텝째 NaN. lr=0에서도 재현 → 계산 문제. `SDXL_NAN_DEBUG`(순전파 노드를 하나씩 계산하며 비유한 값 탐지)로 보니 첫 비유한 값이 LoRA 파라미터 → 이전 스텝 기울기가 NaN이었고 AdamW가 0×NaN을 넣음. 원인은 `g/(1+exp(−2u))`의 exp overflow(g < 약 −11) → 역전파 0×inf. `sigmoid = exp(−softplus(−z))`로 수정, 이후 GELU 미분은 sigmoid로 직접(1.6절).

## 2.8 메모리 → 7-1, 7-2
- 512px가 예산 초과 → `--mem-plan`(할당 없이 크기 계산, 실측과 MiB 단위 일치)으로 분석: 보관 활성값의 약 3GiB가 조합 버전 연산의 중간값 → 원래 연산 + 새 역전파 규칙(1.6절)으로 5.5 → 3.9GiB
- 768/1024px는 softmax 출력이 토큰 수 제곱(1024px에서 11.7GiB) → 그래디언트 체크포인팅
- 체크포인팅에서 할당기 함정 두 개:
  1. gallocr는 `data`가 있는 텐서를 "할당됨"으로 보고 다시 배정하지 않음. 다른 그래프가 버퍼를 다시 잡으면 앞 그래프 텐서가 해제된 버퍼를 가리킴 → 768px에서 assert, 512px에선 조용히 통과. 1스텝짜리 `--grad-check`가 못 잡았으므로 2번째 스텝에서 비교하도록 변경 (512px 결과는 수정 후와 손실이 동일 → 오염 없었음)
  2. gallocr는 버퍼를 조각(Vulkan 기본 1GiB)으로 나누고, 그래프마다 조각별 크기를 그 그래프 기준으로 다시 잡음(`ggml-alloc.c`, "reallocate buffers if needed") → 조각 배치가 다른 그래프를 번갈아 쓰면 재할당 무한 반복. 조각을 4GiB로 키워 한 조각에 들어가게 + 안정될 때까지 할당 패스 반복

## 2.9 평가에서 배운 것
- 384px 600스텝 강아지 LoRA: 학습 대상이 선명히 나오지만 배경 미지정 프롬프트에서 학습 사진의 오렌지 배경이 따라옴 → 처음엔 과적합으로 판단
- 중간 저장본을 같은 파일에 덮어쓰던 버그를 발견해 `_s<스텝>` 저장으로 수정 → 200/400/600 비교 가능해짐
- "프롬프트 문제일 수 있다"는 가설로 두 실험: ① 스텝별: 다른 장면("in a bucket")에서 200/400은 중립 배경, 600은 오렌지 유입 → 600은 가벼운 과적합 ② 배경 명시("on a sunny beach", "in a snowy forest"): 600스텝본도 배경을 정확히 따름 → 오렌지의 주원인은 캡션이 전부 같아 배경이 문장에 묶인 것
- 결론: 적정 200~400스텝 또는 600스텝 + 가중치 0.7, 다음 데이터는 배경 다양화나 배경 캡션

## 2.10 1024px 학습과 fp16 실험
- 1024px f32 200스텝을 서버 경유로 학습 → 3.76h, NaN 없음, 1024px 생성에서 384px 학습본보다 털·눈·귀 디테일이 확연히 좋음 (단 처음 비교는 생성 해상도도 달랐음(768 vs 1024) → 384px 학습본도 1024px로 생성해 맞춤 (`verification/images/*1024*_dog_lora_*`): 384px 600스텝본은 1024px에서 오렌지 배경·붉은 색감이 강하게 따라오고 1024px 200스텝본은 자연스러움. 스텝 수도 달라 해상도만의 효과로 단정 못 함)
- fp16 실험 설계: 같은 데이터·seed·스텝, fp16만 다르게. 판단 기준을 미리 정함 (NaN / 손실 / 생성 효과), 실패 시 대안 3개(lr 절반, t_min 50, 512px)를 짧게 시험할 계획
- 결과: 손실 곡선은 f32와 거의 같아 처음엔 "좋은 신호"로 판단했으나 **오판**. 생성 이미지가 LoRA 없는 것과 같아 가중치를 열어 보니 up이 정확히 0인 모듈 486/560 → 기울기 언더플로 확정. 준비했던 대안 3개는 0 기울기를 못 고치므로 시도하지 않고 중단, 해결책(손실 스케일링)만 1.10에 기록
- 교훈: LoRA 학습 여부는 손실이 아니라 가중치 크기·생성 결과로 판단
- 학습 중 다른 큰 작업(배포 빌드)을 같이 돌렸다가 guard(합계 감시)가 빌드와 학습을 모두 종료시킨 일이 있었음 → 학습 중엔 큰 작업 금지

## 2.11 기타 결정
- 서버를 학습 코어와 같은 프로세스에 넣지 않고 CLI를 자식 프로세스로: 작업이 끝나면 프로세스 종료로 GPU 메모리가 확실히 반환되고, 서버는 ggml 의존이 없어 작고 단순함
- 배포 빌드는 정적 링크 + `GGML_NATIVE=OFF` (계산은 Vulkan이 하므로 CPU 최적화 손실은 무의미)
- fp16 행렬곱은 f32 경로의 정확성·장시간 안정성을 먼저 확인한 뒤로 미룸 (2.7의 NaN을 깨끗하게 가려낼 수 있었던 이유)

## 2.12 v0.2: 합성곱 LoRA (LoCon)
- 범위 결정 (사용자, 2026-11-26): v0.2 = LoCon만. 1.10에 1순위로 적었던 fp16 손실 스케일링은 사용자가 "반쯤 양자화"로 보고 보류
- 대상 규칙은 기억이 아니라 kohya `networks/lora.py` 원본을 읽고 맞춤. 특히 conv_dim을 켜면 ResBlock 안의 **1x1 skip과 time_emb_proj(Linear)도 network_dim으로 같이 들어간다**는 점 (흔히 3x3만 생각하기 쉬움). SDXL(SGM) 이름으로: `in_layers.2`/`out_layers.3`/`op`/업샘플 `conv` = conv_dim, `skip_connection`/`emb_layers.1` = dim
- 구현 선택: down을 별도 im2col로 만들지 않고 원래 합성곱의 im2col을 공유 → 연산 2개(mul_mat) 추가로 끝, Vulkan 셰이더·ggml 수정 없음. 역전파에서 col 기울기가 원래 경로와 LoRA 경로에서 합쳐져 IM2COL_BACK은 그대로 하나
- 검증 순서는 v0.1과 같은 사다리: 축소판 PyTorch(rank·scale을 어텐션과 일부러 다르게: conv rank 3, scale 0.5 → 섞이면 드러남) → 실물 SDXL vs sd.cpp(역시 rank 8/alpha 4 vs conv 4/alpha 8) → grad-check → 자가 학습 → 실제 비교
- **대조 실험을 붙인 이유**: 2.39e-4 일치만으로는 "합성곱 LoRA가 출력에 거의 기여하지 않아서 맞아 보이는" 경우를 못 거름. 합성곱 쪽 up만 0으로 만든 파일로 같은 비교 → 1.1e-2로 48배 → 기여가 크고 그 기여까지 sd.cpp와 같음
- **공정 비교**: 처음엔 LoCon 모듈 초기화가 같은 rng를 소비해 conv-rank를 켜면 어텐션 초기값과 t/노이즈/샘플 순서가 다 달라짐 → 어텐션은 seed rng로 먼저, ResBlock 쪽은 파생 난수열로 분리. 결과: conv-rank 0은 v0.1과 비트 일치(→ v0.1 600스텝본을 기준선으로 재학습 없이 재사용), conv-rank 8도 t/샘플 순서 동일, 1스텝 손실 동일
- 결과 해석 (정직하게): 384px만 봤을 땐 손실 −1.9%, 생성은 같은 구도에서 털결·체형·얼굴 세부가 약간 자연스러운 정도로 보였으나, **768/1024px까지 학습해 보니 장면별 우열이 엇갈림**(768: LoCon 1승 3패, 1024: 1승 1패 1무, 768 LoCon엔 입·양동이 구조 결함도) → seed 하나 비교의 우연 범위, 이 데이터(개체 학습)에서는 LoCon 이득 없음으로 결론. 384px만으로 'LoCon이 조금 낫다'고 README에 썼던 것을 정정. 학습 해상도 효과(주황 배경 과적합 사라짐)가 LoCon 효과보다 훨씬 큼. 개체 학습은 대부분 어텐션 LoRA가 함 (LoCon의 어텐션 LoRA 크기는 v0.1의 0.94배, 방향 0.91 → 일부를 합성곱이 나눠 가짐). 스타일/질감 위주 데이터에서는 차이가 더 클 수 있으나 검증 안 함
- 함정: (1) 평가 스크립트를 `tools/eval_prompt.sh`로 직접 실행 → 실행 권한 없음으로 즉시 실패 (`bash tools/eval_prompt.sh`로). (2) `tools/lora_norms.py`가 모듈 집합이 다른 파일 비교에서 죽음 → `verification/lora_norms.py`를 공통 키 비교 + 종류별 출력으로 고침. (3) sd.cpp에 LoCon을 가중치 0.7로 걸었더니 768px에서 guard가 합계 11.7GiB로 종료 → 처음엔 "가중치≠1이면 sd.cpp가 합성곱 보정을 펼친다"고 **추측했으나 틀림**. 소스(sd.cpp `lora.hpp`)에 가중치로 분기하는 코드가 없고, 프로세스별 측정(`tools/lora_mem_probe.sh`: sd-cli RssAnon) + cgroup 분해 샘플러(`tools/memsample.sh`)로 확인하니: ① 가중치·LoRA 종류와 무관하게 간헐적(어텐션 w1도 1회 KILL, LoCon 0.7은 2회 통과) ② 튄 순간에도 sd-cli 자신의 anon은 300MiB 미만 ③ guard의 컨테이너 used(`free -m`, lxcfs)가 GTT 업로드 순간 cgroup 실제(anon+kernel)보다 -950 ~ +428MiB 흔들리고 243MiB 같은 불가능한 값도 찍힘 → **lxcfs 측정 착시로 인한 거짓 KILL**. lxcfs used = memory.current − 파일 캐시를 따로 읽어 계산하는데, 모델(7GB) 읽기와 GTT 확보로 페이지 캐시가 급히 회수되는 동안 두 값이 엇갈리는 것으로 추정. 작업 폴더의 guard는 cgroup `memory.stat` anon+kernel+shmem으로 바꿔 같은 768px 0.7이 합계 8.4GiB로 통과 (v0.2 배포본 guard에도 반영). 교훈: 측정값 하나로 원인을 단정하지 말고 프로세스별·계정별로 분해할 것 (사용자가 "CPU가 그렇게 먹을 이유가 없다"고 의심해 조사 시작)

## 2.13 v0.3: fp16 손실 스케일링
- 범위 결정 (사용자, 2026-11-27): v0.2 다음으로 "f16을 스케일링으로". 사용자 지시는 "알아서 f16 되는지 이것저것 다 해 보고 알려 달라" → 설계·검증·평가를 끝까지 진행
- 설계 선택: 1.10의 원래 메모는 "AdamW는 스케일 불변이라 되돌릴 필요 없음"이었으나, dynamic에서 S가 바뀌는 순간 m/v의 스케일이 섞이므로 **f32 저장소에서 1/S로 되돌린 뒤 AdamW**로 바꿈. 넘침 검사는 AdamW 전에 해야 건너뛸 수 있음 → 전체 그래프 모드도 AdamW를 별도 그래프로 분리 (ckpt 모드는 원래 분리돼 있었음)
- **밟은 함정 (1.10 메모가 경고했던 바로 그것)**: ckpt의 `run()`이 모든 블록 그래프의 손실 기울기를 같은 값으로 설정하는 구조라, 1을 S로 바꾸자 중간 블록마다 S가 곱해짐 → f32 모드에서도 `--grad-check --loss-scale 1024`가 1스텝에 NaN/Inf. 진단: S=1은 통과 → 스케일 의존, 전체 그래프(`train()`) S=1024는 통과 + `SDXL_NAN_DEBUG`를 역전파까지 넓혀 노드 단위로 비유한 값 없음 확인 → ckpt 전용 문제로 좁혀 원인 발견. head/full만 S로 고침
- 평가: sd.cpp 1024px 3장면(양동이/해변/눈숲, seed 42) 모두 f32 결과와 거의 같은 그림 (같은 개체·구도·배경, 귀 각도·발 위치 정도 차이). v0.2 fp16은 3장면 모두 학습 개체가 안 나옴 (`docs/images/*_fp16_v02/v03.png`)
- 검증 사다리: ① f32 모드 정확성(S가 2의 거듭제곱이면 결과가 비트 단위로 같아야 함 → 확인, v0.2 실행 파일과도 비트 일치) ② 256px 짧은 학습에서 fp16 S=1 / dynamic / f32 3자 비교 ③ v0.2 실패 실험(1024px 200스텝)을 같은 조건으로 재현해 가중치·이미지 비교
- 관찰: 이 데이터에선 시작 S=32768에서 넘침이 한 번도 없었음 (200스텝 뒤 65536). 1024px는 손실 기울기 시작값이 원소당 1e-5 규모라 S≈3e4를 곱해도 fp16 최대값(65504)에 한참 못 미침. 넘침→감소 경로는 `--loss-scale-init 2^28 --loss-scale-window 10`(256px)으로 별도 확인: 1~4스텝 넘침마다 S/2, 2^24에서 한 번 더 넘쳐 2^23, 10스텝 정상 후 2^24로 복귀, 건너뜀 5·갱신 25 (`verification/logs/v03_overflow_backoff.txt`). 256px에선 2^24 근처가 넘침 경계 → 시작값 2^15는 넉넉히 안전
- 속도: 1024px 평균 50.7s (v0.2 fp16 S=1 48.1s와 비슷, 검사·mul 비용 작음). 40스텝 무렵부터 스텝 시간이 39.6 → 51s로 느는 v0.2의 미해결 현상이 그대로 나타남

