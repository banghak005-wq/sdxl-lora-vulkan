**English** | [한국어](README.ko.md)

# SDXL LoRA training on Vulkan — no CUDA, no ROCm (v0.4)

**Train SDXL LoRAs on an AMD GPU without ROCm.** This project implements SDXL UNet training (forward, backward and AdamW) on top of [ggml](https://github.com/ggml-org/ggml) and runs it with **plain Vulkan**, so it works on GPUs that ROCm does not support, including integrated graphics. It was developed and tested on an **AMD Radeon 660M iGPU** (6 CUs, shared system memory).

- No CUDA, no ROCm, no PyTorch at training time: a single C++ binary using the ggml Vulkan backend
- Output is a standard **kohya-format LoRA** (`.safetensors`) that loads in stable-diffusion.cpp and other tools
- Trains up to **1024px** (SDXL native resolution) with gradient checkpointing in about 8–9 GiB of GPU-visible memory
- Gradients verified against PyTorch (float64) and UNet outputs verified against stable-diffusion.cpp

> The code and documentation in this repository were written by a Claude agent (Anthropic); the human author set the goals, made the decisions and judged the results. The full design log, including the mistakes, is in [for_ai.md](for_ai.md) (Korean).

**New in v0.4: job-queue web server.** Queue up cache building, training and image generation and they run one after another. Leave it running, come back, and view and download the results in the browser. Kill a running job instantly, the queue survives restarts, temporary access tokens (`--token auto`), select and delete finished jobs. The server is a single C++ binary, so no Python is needed (see "3. Job-queue web server").

v0.3 feature: **fp16 matmul training (loss scaling).** Up to v0.2, training with fp16 matrix multiplication silently failed (gradient underflow: the LoRA did not learn), so only f32 was usable.
v0.3 adds dynamic loss scaling, and **fp16 training now produces practically the same LoRA as f32, 1.33× faster** (1024px, 200 steps: LoRA direction similarity 0.999, near-identical images; see "v0.3 results").
It is enabled automatically when `GGML_VK_DISABLE_F16=1` is not set. f32 training (`GGML_VK_DISABLE_F16=1`) gives bit-identical results to v0.2.

v0.2 feature: **convolution LoRA (LoCon)**. `--conv-rank N` also puts LoRA on the ResBlock and down/upsampling convolutions (same rules and file format as kohya `conv_dim`).
Without the option it behaves exactly like v0.1 (verified: bit-identical training loss with the same seed).
Also in v0.2: fixed a measurement error that made `tools/guard.sh` (memory guard) kill healthy jobs inside LXC (see section 5 of Usage).

## v0.4 results: end to end through the job server
With 5 dog photos, **build cache → train (512px, 500 steps) → generate 3 images** were queued at once and left alone. Everything ran through this server only.

| Prompt | Without LoRA | LoRA trained through the server (512px, 500 steps, fp16) |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket768_base.png) | ![](docs/images/bucket_v04_server_lora512.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach_base.png) | ![](docs/images/beach_v04_server_lora512.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow_base.png) | ![](docs/images/snow_v04_server_lora512.png) |

- Training took 63 min (7.5 s/step, no overflows), 768px generation about 5 min per image, peak total memory 8.0 GiB. The trained corgi (big ears, white stripe on the face) appears and the scene in the prompt is followed. The orange background in the bucket image comes from the training photos (a limit of 5 photos at low resolution, the same as v0.1's 384px training)
- **Same results through the server**: server-trained LoRA = CLI-trained LoRA **bit-identical**; server-generated image = image from running sd-cli directly, **pixel-identical**; the cache the server builds itself = the cache from the old Python script, **bit-identical**
- **Kill**: killing a real training job ends the whole process group within 0.5 s and GPU memory (GTT) goes from 4.7 GiB back to 0
- 56 server tests pass (with a fake trainer, so anyone can run them without a GPU). Details: [`verification/`](verification/README.md)

## v0.3 results: fp16 matmul training (loss scaling)
The experiment that failed in v0.2 (5 dog images, 1024px, 200 steps, seed 1, rank 8) was repeated in v0.3 **under the same conditions**. Generation: sd.cpp, 1024px, seed 42.

| Prompt | f32 training (v0.2) | fp16 without loss scaling (v0.2) | **fp16 + loss scaling (v0.3)** |
|---|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket1024_lora_f32.png) | ![](docs/images/bucket1024_fp16_v02.png) | ![](docs/images/bucket1024_fp16_v03.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach1024_lora_f32.png) | ![](docs/images/beach1024_fp16_v02.png) | ![](docs/images/beach1024_fp16_v03.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow1024_lora_f32.png) | ![](docs/images/snow1024_fp16_v02.png) | ![](docs/images/snow1024_fp16_v03.png) |

- v0.2 fp16: the trained corgi does not appear (looks like no LoRA). v0.3 fp16: **almost the same image as f32** (same subject and composition; only small differences such as ear angle or paw position)
- LoRA weights: v0.3 fp16 vs f32 up-weight direction similarity **0.999**, norm ratio 0.999, zero modules 0/560 (v0.2 fp16: similarity 0.000, zero modules 486/560)
- Mean loss 0.10518 (f32 0.1051), no NaNs and no overflows, 50.7 s/step (f32 about 67.7 s → **1.33× faster**), same memory as f32 (peak total 9.2 GiB)
- f32 training is bit-identical to v0.2 (the scale is a power of two, so it does not change f32 results)
- Details: `verification/v0.3/logs/v03_*.txt`, `for_ai.md` 1.5, 1.8, 2.13

## v0.2 results: attention LoRA vs LoCon (same data, same seed, same training order)
5 dog images, rank 8, lr 1e-4. LoCon only adds `--conv-rank 8 --conv-alpha 8` (initialization, timesteps and noise order are matched with the attention-only run, so the only difference is the convolution LoRA).
Generation: sd.cpp (Vulkan), 20 steps, cfg 7, seed 42, `--lora-apply-mode at_runtime`, weight 1. Three training resolutions × attention / LoCon.

**Conclusion: on this data (learning a single subject) there is no evidence that LoCon is clearly better.** Both draw the trained subject and follow the scene in the prompt; which one wins differs per scene and per resolution (single-seed comparisons, treated as noise). LoCon costs 6–14% more training time. Texture- or style-heavy data may behave differently but was not tested.

| Training resolution (steps) | LoCon better | Attention better | Similar | Training s/step (attention → LoCon) | Mean loss (attention / LoCon) |
|---|---|---|---|---|---|
| 384px (600) | all 4, slightly (fur, body shape) | – | – | 6.34 → 6.70 (+6%) | 0.1101 / 0.1080 |
| 768px (300) | snowy forest | sks dog, bucket, beach (slightly) | – | 34.9 → 36.9 (+6%) | 0.0949 / 0.0946 |
| 1024px (200) | beach (colors) | bucket | snowy forest | 67.7 → 77.3 (+14%) | 0.1051 / 0.1049 |

- At every resolution all 66 convolution-side modules learned (up-weight norms ≈ attention, zero modules 0/626). The attention LoRA inside LoCon has direction similarity 0.91 (384) / 0.97 (768) / 0.98 (1024) with the attention-only LoRA
- **Applying only the convolution part** (the 768px LoCon with attention keys removed, 66 convolution modules only, `verification/lora_filter.py nonattn`): colors, texture and scene change, but the trained corgi does not appear (`sks dog` → a white labrador ![](docs/images/dog0_768train_convonly.png), bucket → a different dog ![](docs/images/bucket_768train_convonly.png)). **The subject's identity is learned by the attention LoRA; the convolution LoRA changes texture and color**, which is why LoCon helped little for subject training
- **Training resolution matters far more than LoCon**: the 384px LoRAs strongly carry over the orange background of the training photos, while the 768/1024px LoRAs produce natural scenes (tables below)
- Peak total memory: 384px (no checkpointing) 9.2 GiB, 768px 7.1 GiB, 1024px 9.2 GiB (`--ckpt`)

**768px LoRAs** (generated at 768px):

| Prompt | Attention | LoCon |
|---|---|---|
| `a photo of sks dog` | ![](docs/images/dog0_768train_attn.png) | ![](docs/images/dog0_768train_locon.png) |
| `... in a bucket` | ![](docs/images/bucket_768train_attn.png) | ![](docs/images/bucket_768train_locon.png) |
| `... on a sunny beach` | ![](docs/images/beach_768train_attn.png) | ![](docs/images/beach_768train_locon.png) |
| `... in a snowy forest` | ![](docs/images/snow_768train_attn.png) | ![](docs/images/snow_768train_locon.png) |

**1024px LoRAs** (generated at 1024px; attention = the v0.1 LoRA):

| Prompt | Attention | LoCon |
|---|---|---|
| `... in a bucket` | ![](docs/images/bucket_1024train_attn.png) | ![](docs/images/bucket_1024train_locon.png) |
| `... in a snowy forest` | ![](docs/images/snow_1024train_attn.png) | ![](docs/images/snow_1024train_locon.png) |
| `... on a sunny beach` | ![](docs/images/beach_1024train_attn.png) | ![](docs/images/beach_1024train_locon.png) |

**384px LoRAs** (generated at 768px, with the no-LoRA baseline):

| Prompt | Without LoRA | Attention LoRA (v0.1) | LoCon (v0.2) |
|---|---|---|---|
| `a photo of sks dog` | ![](docs/images/dog_base.png) | ![](docs/images/dog_attn600.png) | ![](docs/images/dog_locon600.png) |
| `a photo of sks dog in a bucket` | ![](docs/images/bucket768_base.png) | ![](docs/images/bucket_attn600.png) | ![](docs/images/bucket_locon600.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach_base.png) | ![](docs/images/beach_attn600.png) | ![](docs/images/beach_locon600.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow_base.png) | ![](docs/images/snow_attn600.png) | ![](docs/images/snow_locon600.png) |

**384px LoRAs generated at 1024px** (about 11 min per image, peak total memory about 8.9 GiB):

| Prompt | Attention LoRA (v0.1) | **LoCon (v0.2)** |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket1024_attn600.png) | ![](docs/images/bucket1024_locon600.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow1024_attn600.png) | ![](docs/images/snow1024_locon600.png) |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach1024_attn600.png) | ![](docs/images/beach1024_locon600.png) |

- The snowy forest shows the largest difference: the attention version turns the whole frame red (color bleeding of the 384px LoRA, seen in v0.1), while the LoCon version keeps the snow and trees whiter and sharper. Shapes such as the bucket handle are also more intact with LoCon. Both still overfit to the orange background and reddish tones (a limit of 5 images at 384px)
- Weight 0.7 (768px beach): ![](docs/images/beach_locon600_w07.png). Same composition as the attention version at 0.7 (v0.1 results below), less overfitting and more natural than weight 1

- Cost (at 384px): parameters 11.6M → 14.9M, step time 6.34 → 6.70 s (+6%), peak total memory 8.85 → 9.19 GiB, file size 47 → 60 MB


## v0.1 results (attention LoRA, reproduced unchanged in v0.2)

Training: [diffusers/dog-example](https://huggingface.co/datasets/diffusers/dog-example), 5 images, caption `a photo of sks dog`, 384px, rank 8.
Generation: stable-diffusion.cpp (Vulkan), 768px, 20 steps, cfg 7, **same seed, without / with the LoRA**.

| Prompt | Without LoRA | With LoRA |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket_base.png) | ![](docs/images/bucket_lora400.png) 400 steps |
| `a photo of sks dog on a sunny beach` | ![](docs/images/beach_base.png) | ![](docs/images/beach_lora600_w07.png) 600 steps, weight 0.7 |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow_base.png) | ![](docs/images/snow_lora600.png) 600 steps |

The trained dog (big ears, white stripe on the face) appears while the scene in the prompt (bucket, beach, snowy forest) is still followed.

### 1024px training (SDXL native resolution)
Same data, **1024px, 200 steps** (`--ckpt --native all`, f32), generated at 1024px with sd.cpp, same seed.

| Prompt | Without LoRA | With LoRA (1024px, f32) |
|---|---|---|
| `a photo of sks dog in a bucket` | ![](docs/images/bucket1024_base.png) | ![](docs/images/bucket1024_lora_f32.png) |
| `a photo of sks dog in a snowy forest` | ![](docs/images/snow1024_base.png) | ![](docs/images/snow1024_lora_f32.png) |

Training took 3.76 hours (57–69 s/step), about 8.5 GiB of GPU + system memory in total, no NaNs.

### fp16 matmul training (up to v0.2: failed → v0.3: fixed with loss scaling)
Up to v0.2, training with fp16 matrix multiplication (without `GGML_VK_DISABLE_F16`) had no NaNs and a loss similar to f32, but **the LoRA barely learned**: 486 of 560 modules kept their up-weights at exactly zero and the generated images looked like the base model.
The cause is **gradient underflow** (the loss gradient is so small that it falls below the fp16 range and becomes zero). The loss scaling added in v0.3 fixes this (see "v0.3 results").

### Accuracy
- Reduced SDXL-style UNet: LoRA gradients match PyTorch (float64 reference) with relative error **1e-6 on CPU / 3e-6 on Vulkan**
- Full SDXL UNet output matches stable-diffusion.cpp to **2.3e-4** (the level of sd.cpp's f16 im2col), also with a LoRA applied
- Gradient checkpointing gradients = full-graph gradients (**8e-7**)
- **v0.2 LoCon**: reduced-model gradients vs PyTorch 1.7e-6 on CPU / 3.3e-6 on Vulkan, full SDXL output vs sd.cpp **2.39e-4** (removing the convolution LoRA gives 1.1e-2, 48× larger, so the comparison really checks the convolution contribution), sd.cpp loads 1878/1878 keys, checkpointing gradients 6.2e-7

### Speed / memory (Radeon 660M, Mesa 25.0.7 RADV, f32, rank 8, batch 1)
| Resolution | Mode | s/step | GPU memory (incl. 4.9 GiB weights) |
|---|---|---|---|
| 384 | no checkpointing | 6.3 | ~8.0 GiB |
| 512 | `--ckpt` | 12.4 | ~5.5 GiB |
| 768 | `--ckpt` | 28.7 | ~6.6 GiB |
| **1024** | `--ckpt` | **57.5** (~69 in long runs) | **~8.2 GiB** |
| **1024, fp16 + loss scaling (v0.3)** | `--ckpt` | **39.6** (~51 in long runs) | ~8.2 GiB (same) |

In the real 1024px runs the step time rose by about 15% after 20–40 steps and stayed there (not seen in a 1-hour 384px run; temperature and clocks were normal; cause unknown).

Verification artifacts (logs, all generated images, reference data) are in [`verification/`](verification/README.md) (those of earlier versions in `verification/v0.1/`, `verification/v0.2/` and `verification/v0.3/`). The trained LoRA files are not distributed.

## Requirements
- Linux with a Vulkan 1.3 GPU driver (verified on AMD RADV). About 9 GiB or more of GPU-usable memory (for an iGPU, check the GTT size)
- CMake 3.19+, a C++17 compiler, `glslc` (for ggml's Vulkan shaders)
- SDXL base checkpoint (`sd_xl_base_1.0.safetensors`) and VAE (`sdxl-vae-fp16-fix` recommended)
- `sd-cli` from [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) (a version with DGL mode), used to build the training cache and to check results
- Python 3 (the cache script uses only the standard library). The verification tests need PyTorch, NumPy and `gguf`

## Build
```sh
patches/get_ggml.sh ggml                 # fetch ggml (eeda9ec4) and apply this project's patch
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_DIR=ggml
cmake --build build -j4                  # build/sdxl_lora, build/sdxl_lora_server, build/tinyunet
```
Prebuilt x86-64 Linux binaries (`sdxl_lora`, `sdxl_lora_server`) are in [Releases](../../releases) (ggml statically linked, built on glibc 2.41). Put them in `bin/` to match the example paths below. The v0.4 archive also contains `tools/guard.sh`, the memory guard the server uses (found automatically in `tools/` next to the binary). Runtime requirements: `libvulkan1`, `libgomp1`, `libstdc++6` and your GPU's Vulkan driver. On distributions with an older glibc, build from source.
Build dependencies (Debian/Ubuntu): `build-essential cmake git glslc libvulkan-dev spirv-headers`. More on building, usage and troubleshooting: [for_ai.md](for_ai.md) 1.11–1.20.

## Run with Docker (v0.3)
Put the binaries from Releases (or your own `build/sdxl_lora` and `build/sdxl_lora_server`) in `bin/`. Since ggml is statically linked, **a Debian 13 container with only the Vulkan driver installed can run them directly from a mounted folder.** No ROCm/CUDA image or runtime is needed; for an iGPU you only pass `/dev/dri`.
```sh
# 1) Environment image: Debian 13 + Vulkan driver only (once; no Dockerfile needed, read from stdin)
docker build -t sdxl-vulkan-env - <<'EOF'
FROM debian:trixie-slim
RUN apt-get update && apt-get install -y --no-install-recommends libvulkan1 mesa-vulkan-drivers libgomp1 procps \
 && rm -rf /var/lib/apt/lists/*
EOF

# Common options: pass the GPU + run as your own user (so output files are not owned by root) + mount folders
GPU="--device /dev/dri --group-add $(getent group render | cut -d: -f3) --user $(id -u):$(id -g) -e HOME=/work"
REPO=/path/to/sdxl-lora-vulkan   # this repository (binaries in bin/)
WORK=/path/to/work               # working folder for models/, cache/, out/ (owned by your user)
VOL="-v $REPO:/app:ro -v $WORK:/work"

# 2) Check that the GPU is visible: you should see a line like "ggml_vulkan: 0 = AMD Radeon ... (RADV ...)"
docker run --rm $GPU $VOL sdxl-vulkan-env /app/bin/sdxl_lora

# 3-a) Train with the server (browser: http://127.0.0.1:8090/)
docker run -d --name sdxl-lora $GPU $VOL -p 127.0.0.1:8090:8090 sdxl-vulkan-env \
    /app/bin/sdxl_lora_server --root /work --host 0.0.0.0 --port 8090

# 3-b) Or train directly with the CLI
docker run --rm $GPU $VOL -w /work sdxl-vulkan-env \
    /app/bin/sdxl_lora --train cache/dog1024 --ckpt --native all --steps 200 --out out/dog_lora.safetensors
```
- `$WORK/models/sd_xl_base_1.0.safetensors` must exist (the trainer's default model path is relative to the working folder). The server runs the trainer inside `--root` (= `/work`)
- The data cache (Usage, section 1) needs stable-diffusion.cpp's `sd-cli`, which is not in this environment. Build the cache on the host and put it in `$WORK/cache/`
- **The v0.4 server's cache and generate jobs** need sd-cli: mount the host's sd-cli and point to it (e.g. `-v /path/to/sd-cli:/opt/sd-cli:ro` plus the server argument `--sd-cli /opt/sd-cli`). Without it only training jobs work. Add `--token auto` if you expose it
- `-p 127.0.0.1:8090:8090`: the server has no authentication, so it is exposed to the host only. `--host 0.0.0.0` inside the container is just for Docker's port mapping
- `HOME=/work`: Mesa stores compiled GPU shaders in `$WORK/.cache/`, so later runs start faster
- On distributions without a `render` group, use the group id of `/dev/dri/renderD128` for `--group-add` (check with `ls -ln /dev/dri`)
- The binaries need glibc 2.38 or newer (Debian 13 = 2.41; Ubuntu 24.04 = 2.39 also works)
- **iGPU memory**: the shared memory an integrated GPU uses (GTT) is not limited by `docker run --memory`. If memory is tight, run under `tools/guard.sh` (`... sdxl-vulkan-env /app/tools/guard.sh /work/logs/mem.log /app/bin/sdxl_lora --train ...`). Inside Docker the guard reads host-wide usage with `free`, which may count GTT twice and stop the job earlier than necessary (the safe side); adjust with `-e GUARD_LIMIT_MIB=...` if needed
- The container's Mesa version is whatever Debian 13 ships at the time (verified with Mesa 25.0.7). **This Docker procedure was reviewed on paper only and has not been run under Docker** (the binaries and the libraries they need are the same as in the verified setup)

### On other GPUs (including Docker; untested — the only verified GPU is the Radeon 660M)
| GPU | How to pass it | Expectation |
|---|---|---|
| AMD (RDNA1–4, GCN, other APUs) | `--device /dev/dri` (as above) | Same Mesa RADV driver, so likely to work. On RDNA3+ ggml may use the matrix-core (coopmat) path, so fp16 numerics may differ |
| Intel (Arc, Xe integrated) | `--device /dev/dri` | The same `mesa-vulkan-drivers` package includes the ANV driver, so it should work in principle |
| NVIDIA | NVIDIA Container Toolkit instead of `/dev/dri`: `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` (`graphics` is required to get the Vulkan driver) | ggml's Vulkan backend supports NVIDIA, so it is expected to work. On NVIDIA, existing CUDA tools are faster though |

- **Memory**: on a discrete GPU everything must fit in VRAM. Measured (rank 8, `--ckpt --native all`): 1024px about 8.2 GB → **12 GB or more recommended (8 GB cards cannot do 1024px; use 768px or less)**, 768px about 6.6 GB, 512px about 5.5 GB
- **On a new GPU, check first**: train briefly (e.g. 512px, 50 steps), then run `python3 verification/lora_norms.py <result>` and confirm that the number of modules whose up-weights are all zero is 0. The Vulkan kernels this project adds (OUT_PROD extension, IM2COL_BACK) were only verified on the 660M. If something looks wrong, compare with f32 training (`GGML_VK_DISABLE_F16=1`). For a precise check, build `tests/tinyunet` (comparison against PyTorch) from source and run it
- With several GPUs the trainer uses the first visible one. Select another with e.g. `-e GGML_VK_VISIBLE_DEVICES=1` (see the `ggml_vulkan: N = ...` list at startup)

## Usage

### 1. Build the data cache (images → latents, captions → text conditioning)
Only the UNet is in the training graph. The VAE and text encoders are run once in advance with sd.cpp and saved as `.npy`.
```sh
# image folder: image files + optional .txt captions with the same name. Center-cropped to a square and resized to --size
python3 tools/make_cache.py dataset/dog cache/dog1024 --size 1024 --caption "a photo of sks dog" \
    --sd-cli /path/to/sd-cli --model models/sd_xl_base_1.0.safetensors --vae models/sdxl_vae.safetensors \
    --vae-backend cpu    # 1024px VAE encoding uses a lot of GPU memory, CPU recommended
```
Output: `<name>.latent.npy`, `<name>.crossattn.npy`, `<name>.vector.npy`, a preview `<name>.check.png`, and `captions.txt`

### 2. Train (CLI)
```sh
# fp16 matmul + loss scaling (v0.3, 1.33x faster, practically the same result as f32)
build/sdxl_lora --train cache/dog1024 --ckpt --native all \
    --steps 200 --lr 1e-4 --rank 8 --alpha 8 --save-every 100 --out out/dog_lora.safetensors
# f32 (bit-identical to v0.2)
GGML_VK_DISABLE_F16=1 build/sdxl_lora --train cache/dog1024 --ckpt --native all --steps 200 --out out/dog_lora.safetensors
```
| Option | Description |
|---|---|
| `--train <dir>` | cache folder (all samples must have the same resolution) |
| `--ckpt` | gradient checkpointing (per-block recomputation). Required at 768px and above, about 1.3× slower |
| `--native all` | use ggml's native NORM/GROUP_NORM/GEGLU/CONCAT/UPSCALE with the new backward rules (less memory, **recommended**). Without it, a composite-op version is used |
| `--steps --lr --rank --alpha --seed` | training settings. LoRA targets: to_q/k/v/out of every attn1/attn2 (560 modules) |
| `--conv-rank N` (`--conv-dim`) | **v0.2** LoCon. 0 (default) = off. When on, rank-N LoRA on the two 3x3 ResBlock convolutions and the down/upsampling convolutions (38 modules), plus `--rank` LoRA on the same blocks' 1x1 skip (11) and time-embedding Linear (17) = 626 modules. `conv_in` and the final convolution are excluded (same as kohya) |
| `--conv-alpha A` | **v0.2** alpha for the 3x3 side (scale = A/N). Default = conv-rank (scale 1) |
| `--loss-scale S\|dynamic` | **v0.3** loss scaling. Default: `dynamic` with fp16 matmul (Vulkan without `GGML_VK_DISABLE_F16`), otherwise 1 (off). `dynamic`: when the gradients contain NaN/Inf (fp16 overflow), that step's update is skipped and S is halved; after enough clean steps S is doubled. With a fixed value, an overflow stops training with an error |
| `--loss-scale-init S` / `--loss-scale-window N` | **v0.3** initial value for dynamic (default 32768) / number of consecutive clean steps before doubling S (default 200) |
| `--save-every N` | intermediate saves `<out>_s<step>.safetensors` |
| `--resume <lora>` | continue training (with `--lr 0`: fixed-seed loss evaluation) |
| `--t-min/--t-max` | timestep range (default 0..999) |
| `--model <ckpt>` | SDXL checkpoint (default `models/sd_xl_base_1.0.safetensors`) |
| `--backend vulkan\|cpu` | default vulkan |

- `GGML_VK_DISABLE_F16=1`: run matrix multiplication in f32 (required up to v0.2). **v0.3**: without it, training uses fp16 matmul with automatic loss scaling. `S=` at the end of each step line is the current loss scale; an overflow prints an `overflow step N: ...` line. A summary (last S, skipped steps) is printed at the end
- Whether fp16 training worked cannot be judged from the loss (when it failed in v0.2 the loss looked the same as f32). Check with `python3 verification/lora_norms.py <f32 result> <fp16 result>` (number of zero modules, direction similarity) or with generated images
- The output LoRA is kohya format (`lora_unet_<block>_attn1_to_q.lora_down.weight` / `.lora_up.weight` / `.alpha`)
  - LoCon modules: `lora_unet_input_blocks_1_0_in_layers_2.lora_down.weight` `[rank, in, 3, 3]`, `.lora_up.weight` `[out, rank, 1, 1]`, metadata `ss_network_args = {"conv_dim", "conv_alpha"}`. stable-diffusion.cpp loads 1878/1878 tensors (verified)
  - `--resume` needs the same `--rank/--alpha/--conv-rank/--conv-alpha` as the file

### 3. Job-queue web server (v0.4)
A web server that runs cache building, training and image generation **one at a time from a queue**. Queue up several jobs, go do something else, and come back to the results.
- **A single C++ binary** (`sdxl_lora_server`, about 0.9 MB). No Python or other runtime needed. It uses `sdxl_lora` (training) and `sd-cli` from stable-diffusion.cpp (cache and generation)
- Jobs never run concurrently (GPU memory). Each job is a separate process, so GPU memory is released as soon as it ends or is stopped
- The queue is saved to `<root>/jobs/queue.json`: queued jobs survive a server restart and continue (a job that was running is marked "interrupted" and its leftover processes are cleaned up)
- Web UI: add jobs (cache / train / generate / **all in one**: photo folder → cache → train → preview images), queue with progress and time left, logs, finished-jobs archive (download LoRAs, images, logs; select and delete), gallery of generated images, Korean/English
- **Stopping**: cancel queued jobs, stop the running job (SIGTERM, then SIGKILL if it has not exited within 10 s), **kill immediately** (SIGKILL), cancel all queued, pause the queue. The job's child processes (memory guard, sd-cli, …) are terminated together as a process group
- On AMD GPUs a memory guard (`tools/guard.sh`) is attached to every job automatically (`--guard auto|on|off`, limit `--guard-limit-mib`)

```sh
build/sdxl_lora_server --root /path/to/work --sd-cli /path/to/sd-cli --token auto
#   접속 토큰 (이번 실행에만 유효): 9ryjewpxe64h6nbe7xapg8ftfxc24c8i      <- access token, valid for this run only
#   바로 열기: http://127.0.0.1:8090/?token=9ryjewpxe64h6nbe7xapg8ftfxc24c8i
```
- Put `models/sd_xl_base_1.0.safetensors`, `models/sdxl_vae.safetensors` (or use `--model`, `--vae`) and your photo folders inside the working folder (`--root`). All paths in requests are restricted to `--root`
- The trainer is `sdxl_lora` next to the server binary (or `--cli`). Without `--sd-cli` only training jobs are available
- Common generation arguments: `--sd-args` (default `--vae-tiling --flash-attn auto,vae=off --lora-apply-mode at_runtime`: works around a RADV crash in the VAE flash-attention shader and saves LoRA memory). CPU threads for cache jobs: `--threads`
- The server's console messages are in Korean

**Replacing or turning off the web UI** (if you build your own frontend)
| Option | Behavior |
|---|---|
| `--ui builtin` (default) | built-in web UI |
| `--ui off` | **API only**, no UI. `/` returns a JSON description of the API |
| `--ui <folder>` | serves the static files in that folder (`index.html`, js, css, …) at `/`, so your own frontend runs from the same server |
| `--cors <origin>` | allows a frontend served from another origin to call this API (e.g. `--cors http://localhost:5173`, or `"*"`) |

- The token only protects `/api/…` and `/files/…`. Frontend files load without it; your frontend sends `Authorization: Bearer <token>` (or the `sdxl_token` cookie) when calling the API
- The built-in UI (`src/server/page.h`, a single HTML page) doubles as an example of using the API

**Access and security**: by default the server listens **on this machine only (127.0.0.1)**. To reach it from elsewhere (phone, another PC), pick one of these yourself.
| Method | Example | Notes |
|---|---|---|
| Cloudflare quick tunnel | `cloudflared tunnel --url http://127.0.0.1:8090` → the printed `https://….trycloudflare.com` | No account needed. **Anyone with the URL can connect → always use `--token`** |
| Tailscale | `http://<this machine's tailscale name>:8090` from a device in the same tailnet (server with `--host 0.0.0.0` or the tailscale IP) | Not visible outside the tailnet. `--token` still recommended |
| Router port forwarding | external port → this machine's 8090 (server with `--host 0.0.0.0`) | Exposed to the whole internet → **`--token` required**, ideally behind an HTTPS reverse proxy |

- `--token auto`: generates a new temporary token (32 characters) at every start and prints it. The web UI asks for it on the first screen, and `/api` and `/files` return 401 without it. Opening `?token=…` logs in directly (the token is removed from the address bar)
- `--token <your value>` or the `SDXL_SERVER_TOKEN` environment variable (keeps it out of the command line). For the API: `Authorization: Bearer <token>`
- The token travels over plain HTTP. When exposing the server to the internet, use HTTPS (Cloudflare tunnels are HTTPS by default)

**API** (JSON; with a token add `-H "Authorization: Bearer <token>"`)
```sh
curl -X POST localhost:8090/api/jobs -d '{"type":"cache","src":"dataset/dog","size":512,"caption":"a photo of sks dog"}'
curl -X POST localhost:8090/api/jobs -d '{"type":"train","data":"cache/dog_512","steps":500,"out":"out/dog.safetensors"}'
curl -X POST localhost:8090/api/jobs -d '{"type":"generate","prompt":"a photo of sks dog on a sunny beach","lora":"out/dog.safetensors","width":768,"height":768}'
curl localhost:8090/api/jobs                       # list + progress (step, steps, loss, sec_per_step, eta_sec)
curl -X POST localhost:8090/api/jobs/3/cancel      # queued → cancelled, running → SIGTERM (SIGKILL after 10 s)
curl -X POST localhost:8090/api/jobs/3/kill        # running → SIGKILL now
curl -X POST localhost:8090/api/jobs/remove-many -d '{"ids":[1,2],"delete_files":true}'   # delete finished jobs and their result files
```
| Endpoint | Description |
|---|---|
| `POST /api/jobs` | `type`: `cache` (`src`, `dst`, `size` 512, `caption`, `prefix`, `vae_backend`:"cpu") / `train` (`data`, `out`, `name`, `steps` 200, `lr`, `rank`, `alpha`, `conv_rank`, `conv_alpha`, `seed`, `save_every`, `ckpt` true, `native` "all", `resume`, `model`, `fp16` true, `loss_scale`, `t_min`, `t_max`) / `generate` (`prompt`, `negative`, `lora`, `lora_weight` 1, `width`/`height` 1024 (multiples of 64), `steps` 20, `cfg` 7, `seed` 42). Whether paths exist is checked **when the job starts**, so a job can use a cache or LoRA that an earlier queued job will create |
| `GET /api/jobs`, `GET /api/jobs/<id>`, `GET /api/jobs/<id>/log` | list, details (last 60 log lines, result files), full log |
| `POST /api/jobs/<id>/cancel`, `/kill`, `/move` (`{"dir":"up"/"down"/"top"}`), `/remove` (`{"delete_files":bool}`) | cancel, kill, reorder queued jobs, remove a finished job |
| `POST /api/jobs/remove-many` | `{"ids":[…], "delete_files":bool}`. Queued and running jobs are skipped. File deletion only removes **files that job created** (the LoRA and its `_s<step>` checkpoints, generated image and log, the cache's `.npy`, `.check.png`, `cache.dgl`, `captions.txt`). Source photos are never deleted |
| `POST /api/queue/cancel-all` (optionally `{"running":true,"force":true}`), `/pause`, `/resume` | cancel all queued jobs, pause/resume the queue |
| `GET /api/config`, `GET /api/list` | server settings; LoRAs, caches and image folders to pick from in the forms |
| `GET /files/<path>` | download a file inside `--root` |
| v0.3 compatible: `POST /api/train`, `GET /api/status`, `POST /api/stop`, `GET /api/log` | `/api/train` now queues the job instead of refusing (409) |

### 4. Check the result (generate with sd.cpp)
```sh
SD_CLI=/path/to/sd-cli LORA_DIR=out tools/eval_prompt.sh beach "a photo of sks dog on a sunny beach" dog 0   # without LoRA
SD_CLI=/path/to/sd-cli LORA_DIR=out tools/eval_prompt.sh beach "a photo of sks dog on a sunny beach" dog 1   # with LoRA
python3 tools/gallery.py out/eval     # index.html with the same prompts side by side
```
To use it directly with sd-cli: `--lora-model-dir out -p "...<lora:dog:1>" --lora-apply-mode at_runtime` (about 2 GiB less memory than the default merge mode).

### 5. Verification / diagnostic tools
| Command | What it does |
|---|---|
| `build/tinyunet tests/tiny8.gguf [--backend vulkan] [--native all]` | compares the forward pass and LoRA gradients of a reduced UNet with a PyTorch reference (generate it with `python3 tests/ref_tiny_unet.py --out tests/tiny8.gguf --size 8 [--conv-lora]`; `--conv-lora` includes LoCon). `--compare`: per-op Vulkan vs CPU error |
| `sdxl_lora --check-ref <dir> [--lora x]` | compares the real SDXL UNet output with an sd.cpp reference (made by running `tools/ref_unet.dgl` with `sd-cli -M dgl`) |
| `sdxl_lora --train <dir> --grad-check` | checks checkpointing gradients = full-graph gradients (compared at step 2) |
| `sdxl_lora --mem-plan <latent side> [--ckpt] [--native all]` | computes training-graph memory without a GPU (64 = 512px, 128 = 1024px) |
| `SDXL_NAN_DEBUG=1 sdxl_lora --train ... --steps N` | finds the first node that produces a non-finite value at step N (non-checkpointing mode). **v0.3**: also checks backward nodes |
| `python3 tests/test_server.py build/sdxl_lora_server` | **v0.4** job-server tests, 56 checks with fake trainer and sd-cli (no GPU): ordering, reordering, cancel/kill including child processes, restart recovery, path restrictions, selective deletion that keeps source photos, tokens, UI off / custom frontend / CORS |
| `tools/guard.sh <log> <cmd>` | memory guard for iGPUs: kills the command when GTT + system used exceeds `GUARD_LIMIT_MIB`. **v0.2**: inside an LXC container, "used" is read from cgroup `memory.stat` (anon+kernel+shmem) instead of `free -m` (unchanged when run directly on a host). In LXC (lxcfs), `free` jumped by several GiB while a model was being uploaded to the GPU and killed healthy jobs (`verification/v0.2/logs/stage4_sdcpp_memory_spike_investigation.txt`) |

## Known limitations (v0.4)
- Batch size 1, a single square resolution (no aspect-ratio buckets), UNet LoRA only (no text-encoder LoRA). The proj_in/out and ff Linear layers inside the transformers are not targeted (kohya's default includes them)
- Only two groups of per-module rank/alpha (attention, 1x1 and Linear / 3x3 convolution). No per-block rank like kohya's `block_dims`
- fp16 matmul training (v0.3) was verified with a 1024px attention LoRA for 200 steps and a 256px LoCon for 40 steps. Weights and activations are stored as before (f16 weights, f32 activations), so memory is the same as f32 training: it is faster, not smaller
- Step time increases by about 15% in long 1024px runs (cause unknown)
- stable-diffusion.cpp `sd-cli` is needed for the data cache and for checking results (the server's cache and generate jobs use it via `--sd-cli`)
- The server runs one job at a time (GPU memory). Batches are several jobs (the web UI's image count adds one job per seed). No user accounts (a single token)
- Only one GPU has been verified: Radeon 660M (RADV)

## Layout
```
src/unet_ggml.h        SDXL-style UNet graph builder (block API, shared with the reduced test model)
src/sdxl_io.h          safetensors/npy I/O, SDXL discrete noise schedule
src/cli/sdxl_lora.cpp  trainer (training, checkpointing, verification modes, kohya export)
src/server/server.cpp  HTTP job manager
tests/                 reduced-UNet check against PyTorch
tools/                 cache, evaluation, gallery, memory guard, header/npy viewers
patches/               ggml changes (backward rules, Vulkan kernels) + fetch script
for_ai.md              design log and rationale (detailed, Korean)
```

## License
[MIT](LICENSE) © 2026 [banghak005-wq](https://github.com/banghak005-wq). You may use, modify and redistribute it, including commercially, as long as the copyright notice is kept.
`patches/` (ggml changes), `third_party/json.hpp`, `third_party/httplib.h` and the third-party code in the released binaries remain under their original authors' licenses. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the list and full license texts.
The example images and reference tensors are outputs of SDXL base 1.0 (CreativeML Open RAIL++-M). No model weights, training photos (diffusers/dog-example, photos by Alvan Nee on Unsplash) or trained LoRAs are included; see the "Models, data and generated content" section of THIRD_PARTY_LICENSES.md.
