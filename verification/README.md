# verification — 검증 결과물 (v0.4)

이 폴더는 v0.4 코드(작업 큐 웹서버)를 실제로 돌린 **증거**다. 학습기와 ggml은 v0.3과 같으므로 그 검증은 [`v0.3/`](v0.3/README.md)에, 그 이전 판은 [`v0.1/`](v0.1/README.md), [`v0.2/`](v0.2/README.md)에 그대로 있다 (학습한 LoRA 파일은 배포하지 않음).
환경: AMD Radeon 660M (RDNA2 iGPU, 6 CU, UMA), Mesa 25.0.7 RADV, Vulkan 1.4, Linux 7.0, glibc 2.41. 로그의 경로는 `<work>`(작업 폴더)로 치환했다.

## 한눈에

| 무엇을 | 결과 | 증거 |
|---|---|---|
| 서버 동작 (가짜 학습기·sd-cli, GPU 안 씀) | **56/56 PASS**: 차례 실행·겹침 없음, 순서 바꾸기·일시정지, 대기 취소, 실행 중 취소(SIGTERM 0.4초), SIGTERM 무시 → 10초 뒤 자동 SIGKILL, 즉시 kill, **손자 프로세스까지 종료**, 서버 비정상 종료 후 재시작 시 돌던 작업 interrupted + 남은 프로세스 정리 + 대기 작업 이어서 실행, root 밖 경로 거부, 선택 삭제(원본 사진 보존), v0.3 호환 API, 토큰(헤더·쿠키·/files 보호), 화면 끄기(`--ui off`)·직접 만든 프론트(`--ui <폴더>`, API는 계속 토큰 필요)·CORS | `logs/v04_server_tests.txt`, `../tests/test_server.py` |
| 캐시 만들기를 파이썬 → C++로 옮긴 것이 같은가 | 같은 사진 5장·512px: DGL 스크립트 본문 동일, 캡션 동일, `.npy` 15개 **비트 일치**, `.check.png` 5장 **픽셀 일치** | `logs/v04_cache_cpp_vs_python.txt` |
| 실제 학습 강제 종료 | kill 후 0.5초 안에 작업 프로세스 그룹(guard, 학습기) 전부 종료, **GTT 4773 → 13 MiB 반환** (GPU에 넘어간 작업이 있으면 5초 정도) | `logs/v04_real_gpu_queue.txt` |
| 서버 경로 = 명령줄 경로 | 서버 캐시 + 서버 학습 20스텝 LoRA = 파이썬 캐시 + `sdxl_lora` 직접 실행 LoRA, **1680/1680 텐서 비트 일치**. 생성: LoRA 없음 그림이 예전 sd-cli 그림과 픽셀 일치, LoCon 그림도 명령줄 재실행과 픽셀 일치(같은 요청 한 번은 sd-cli 자체의 실행 간 차이로 미세하게 다름) | `logs/v04_real_gpu_queue.txt` |
| 큐에 넣고 손대지 않기 | 12개 작업 전부 사람 손 없이 완료. 본 학습 512px 500스텝 63분(7.5 s/스텝, 넘침 0, S 32768 → 131072, 0인 모듈 0/560, 합계 최대 6.0GiB), 768px 생성 장당 약 5분(합계 최대 8.0GiB) | `logs/v04_real_gpu_queue.txt`, `../docs/images/*_v04_server_lora512.png` |
