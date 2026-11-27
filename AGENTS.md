# AGENTS.md

Read [for_ai.md](for_ai.md) first. It is the full reference for this repository: design, build, CLI/API, file formats, troubleshooting (1.18) and the history of decisions (part 2). It is written in Korean. [README.md](README.md) (English) and [README.ko.md](README.ko.md) (Korean) are the user-facing overview.

Rules that are easy to miss:

- fp16 matmul training (the default since v0.3) depends on loss scaling (`--loss-scale`, automatic `dynamic` when fp16 is active). Never remove or bypass it: without it the LoRA silently does not learn (gradient underflow, for_ai.md 1.8). With gradient checkpointing, the scale goes only on the head graph's loss gradient (for_ai.md 1.5, 2.13). `GGML_VK_DISABLE_F16=1` gives the f32 path, which must stay bit-identical to v0.2. Judge training by LoRA weight norms (`verification/lora_norms.py`) or generated images, not by the loss.
- Build only against the pinned ggml: `patches/get_ggml.sh ggml` (commit `eeda9ec4` + `patches/ggml-backward-fixes.patch`). Other ggml commits will not apply the patch.
- After changing gradients or ggml code, re-run the checks in for_ai.md 1.17 (tinyunet vs PyTorch, `--check-ref`, `--grad-check`).
- Do not commit models, LoRAs or build output (`*.safetensors`, `build/`, `ggml/`, `models/`). Trained LoRAs are not distributed.
- Third-party code keeps its own license. When adding, updating or vendoring any third-party code (headers, patches, statically linked libraries, copied snippets), update [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) (version, license, where it appears, full license text) and the summary at the end of [LICENSE](LICENSE) in the same change.
