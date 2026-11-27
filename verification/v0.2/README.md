# verification — 검증 결과물 (v0.2)

이 폴더는 v0.2 코드(LoCon 추가)를 실제로 돌린 **증거**다. v0.1의 검증 결과(연산 단위 테스트, 1024px 학습, fp16 실험 등)는 v0.1 패키지의 `verification/`에 있고 v0.2에서 바뀌지 않았다 (conv-rank 0은 v0.1과 비트 단위로 같은 학습).
환경: AMD Radeon 660M (RDNA2 iGPU, 6 CU, UMA), Mesa 25.0.7 RADV, Vulkan 1.4, Linux 7.0, glibc 2.41. 로그의 경로는 `<work>`(작업 폴더)로 치환했다.

## 한눈에

| 무엇을 | 결과 | 증거 |
|---|---|---|
| 축소판 UNet + LoCon vs PyTorch(float64) | 10/10 PASS. LoRA 기울기 rel L2 CPU 1.7e-6, Vulkan 3.3e-6, 미지원 노드 0 | `logs/stage2_tiny_unet.txt`, 정답지 `v02_tiny8c.gguf` |
| SDXL UNet + LoCon 출력 vs stable-diffusion.cpp | **2.388e-4** (v0.1 LoRA 회귀 2.355e-4). 합성곱 쪽만 끈 대조 1.143e-2 | `logs/stage3_sdxl.txt`, `sdxl_ref_locon/` |
| sd.cpp가 LoCon 파일을 읽는가 | `(1878 / 1878) LoRA tensors have been applied` | `logs/stage3_sdxl.txt`, `logs/stage4_eval_sdcpp.txt` |
| 체크포인팅 기울기 = 전체 그래프 기울기 | 6.2e-7 / 최악 2.5e-6 | `logs/stage3_sdxl.txt` |
| 학습이 실제로 일어나는가 (여우 1장, 고정 seed) | 평균 손실 0.2445 → 0.1086, 저장→이어받기 손실 일치, 모든 모듈 학습 | `logs/stage3_selftest_train.txt` |
| 강아지 384px 600스텝 LoCon 학습 | 6.70 s/스텝, 합계 메모리 9.19GiB, NaN 없음, 평균 손실 0.10797 (어텐션만 0.11009) | `logs/stage4_train_dog_384px_locon_600steps.txt`, `loras/` |
| 가중치 크기 (어텐션만 vs LoCon 200/400/600) | 0인 모듈 0/626, 합성곱 쪽 up 크기 ≈ 어텐션, 공통 어텐션 모듈 방향 유사도 0.71→0.88→0.91 | `logs/stage4_lora_weight_norms.txt`, `lora_norms.py` |
| 1024px 생성 비교 (어텐션 vs LoCon, 둘 다 384px 600스텝 학습본) | 3장 모두 통과 (합계 최대 8.9GiB, 1878/1878), 눈숲에서 LoCon이 색 번짐 덜함·형태 더 온전 | `logs/stage4_eval_sdcpp_1024.txt`, `images/*1024*` |
| 해상도별 학습 비교 (768px 어텐션/LoCon 300스텝, 1024px LoCon 200스텝 vs v0.1 1024px) | 모두 NaN 없음, 손실 거의 같음, 합성곱 66모듈 전부 학습, 생성 우열 장면마다 엇갈림 → 이 데이터에선 LoCon 이득 없음 | `logs/stage5_train_*.txt`, `logs/stage5_lora_weight_norms.txt`, `logs/stage5_eval_sdcpp.txt`, `images/*dog768_*`, `images/*dog1024_locon*`, `loras/dog768_*`, `loras/dog1024_locon_s200` |
| 합성곱 LoRA만 걸었을 때 (768px LoCon의 합성곱 66모듈만) | 색·질감·장면은 바뀌나 학습 개체(코기)는 안 나옴 → 정체성은 어텐션 몫 | `images/*_dog768_convonly_*`, `lora_filter.py` |
| sd.cpp 생성 중 '메모리 급증' (LoCon 가중치 0.7에서 guard 종료) | sd.cpp·LoRA와 무관한 lxcfs `free` 측정 착시로 확인 (-950~+428MiB 흔들림, 불가능한 243MiB 값), cgroup 직접 측정 guard로 같은 작업 8.4GiB 통과 | `logs/stage4_sdcpp_memory_spike_investigation.txt`, `images/beach_dog_locon_w0.7_768_s42.png` |
| 생성 비교 (sd.cpp Vulkan 768px, seed 42, 프롬프트 4개) | 둘 다 학습 개체 + 장면, LoCon이 털결·체형 세부 약간 나음 | `images/`, `images/index.html` |

## 폴더
- `logs/` — 실행 로그 (`stageN_`은 v0.2 진행 단계: 2 축소판, 3 실물 SDXL, 4 실제 비교 학습)
- `images/` — `<프롬프트>_<base|dog_lora_w1|dog_locon_w1>_768_s42.png`. 프롬프트: `dog0` = `a photo of sks dog`, `bucket` = `... in a bucket`, `beach` = `... on a sunny beach`, `snow` = `... in a snowy forest`. `dog_lora` = v0.1 어텐션 LoRA 600스텝 (v0.1 `loras/dog384_s600`과 같은 파일), `dog_locon` = v0.2 LoCon 600스텝
- `loras/` — `dog384_locon_s{200,400,600}.safetensors` (rank 8/alpha 8, conv_rank 8/conv_alpha 8, kohya 형식). **GitHub 저장소에는 올리지 않는다** (`.gitignore`가 `*.safetensors` 제외, 학습한 LoRA는 배포 안 함). 같은 명령·seed로 다시 학습하면 재현됨 (`logs/stage4_*`, `logs/stage5_*`의 명령)
- `sdxl_ref_locon/` — sd.cpp 기준값 (`x/d_lora/crossattn/vector/params.npy`), 시험 LoRA(`test_lora_conv.safetensors`: rank 8 alpha 4, conv 4 alpha 8, up ~ N(0, 0.05)), DGL 스크립트
- `lora_norms.py` — LoRA 가중치 크기·모듈 종류별·공통 모듈 방향 비교 (v0.1 판은 모듈 집합이 다르면 실패), `train_stats.py` — 학습 로그 통계
