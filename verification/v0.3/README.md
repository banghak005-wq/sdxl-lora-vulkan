# verification — 검증 결과물 (v0.3)

이 폴더는 v0.3 코드(fp16 행렬곱 학습용 손실 스케일링)를 실제로 돌린 **증거**다. v0.1/v0.2의 검증 결과는 이 폴더의 [`../v0.1/`](../v0.1/README.md), [`../v0.2/`](../v0.2/README.md)에 그대로 옮겨 두었고(학습한 LoRA 파일은 제외) v0.3에서 바뀌지 않았다 (f32 학습은 v0.2와 비트 단위로 같음, 아래 첫 줄).
환경: AMD Radeon 660M (RDNA2 iGPU, 6 CU, UMA), Mesa 25.0.7 RADV, Vulkan 1.4, Linux 7.0, glibc 2.41. 로그의 경로는 `<work>`(작업 폴더)로 치환했다.

## 한눈에

| 무엇을 | 결과 | 증거 |
|---|---|---|
| 손실 스케일이 f32 결과를 바꾸지 않는가 | grad-check S=1/1024/65536 모두 7.473e-7로 같음. 20스텝 학습 LoRA가 S=1·S=1024·**v0.2 실행 파일** 사이 1878/1878 텐서 비트 일치 | `logs/v03_loss_scale_exact.txt` |
| fp16 학습 256px (LoCon, 40스텝) | 평균 손실 f32 0.11765 / v0.3 fp16 0.11790 / 스케일링 없는 fp16 0.12986. up 방향 f32 대비 cos 0.960 (없을 때 0.390), 1.5배 빠름 | `logs/v03_fp16_256px.txt` |
| **fp16 학습 1024px 200스텝** (v0.2 실패 실험과 같은 조건) | 넘침 0, 평균 손실 0.10518 (f32 0.1051), up 방향 f32 대비 **cos 0.999**, 0인 모듈 0/560 (v0.2 fp16: cos 0.000, 486/560), 50.7 s/스텝 (f32 약 67.7) | `logs/v03_train_dog1024_fp16.txt`, `logs/v03_eval1024.txt` |
| sd.cpp 1024px 생성 (3장면, seed 42) | 3장 모두 f32 학습본과 거의 같은 그림. v0.2 fp16은 학습 개체가 안 나옴 | `../../docs/images/*1024_{lora_f32,fp16_v02,fp16_v03,base}.png` |
| dynamic 넘침 처리 | 시작 S=2^28 → 넘칠 때마다 갱신 건너뛰고 S/2, 10스텝 정상 후 S×2. 오류 없이 끝까지 | `logs/v03_overflow_backoff.txt` |

## 폴더
- `v0.1/`, `v0.2/` — 이전 판의 검증 결과 (각 판의 README 그대로. 그 안의 `loras/` 언급은 배포하지 않는 파일)
- `logs/` — 실행 로그
- 학습한 LoRA 파일은 배포하지 않는다 (저장소 `.gitignore`가 `*.safetensors` 제외). 같은 LoRA를 직접 만드는 명령: `sdxl_lora --train <dog1024 캐시> --ckpt --native all --steps 200 --lr 1e-4 --rank 8 --alpha 8 --seed 1 --save-every 100 --out out/dog1024_fp16.safetensors` (fp16 기본, f32 비교본은 같은 명령에 `GGML_VK_DISABLE_F16=1`). 데이터: 강아지 5장(dog-example), 1024px 캐시
- `lora_norms.py` — LoRA 가중치 크기·0인 모듈 수·공통 모듈 방향 비교. **fp16 학습 성공 여부는 손실로 알 수 없으므로 이걸로 확인**
