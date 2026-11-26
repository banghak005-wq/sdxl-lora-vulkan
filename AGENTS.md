# AGENTS.md

Read [for_ai.md](for_ai.md) first. It is the full reference for this repository: design, build, CLI/API, file formats, troubleshooting (1.18) and the history of decisions (part 2). It is written in Korean. [README.md](README.md) is the user-facing overview.

Rules that are easy to miss:

- Always train with `GGML_VK_DISABLE_F16=1`. fp16 matmul runs without errors but the LoRA does not learn (gradient underflow, for_ai.md 1.8). Judge training by LoRA weight norms (`verification/lora_norms.py`) or generated images, not by the loss.
- Build only against the pinned ggml: `patches/get_ggml.sh ggml` (commit `eeda9ec4` + `patches/ggml-backward-fixes.patch`). Other ggml commits will not apply the patch.
- After changing gradients or ggml code, re-run the checks in for_ai.md 1.17 (tinyunet vs PyTorch, `--check-ref`, `--grad-check`).
- Do not commit models, LoRAs or build output (`*.safetensors`, `build/`, `ggml/`, `models/`). Trained LoRAs are not distributed.
- Third-party code keeps its own license; update [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) when adding or updating any.
