# W8A8/FP8 Roadmap: Phase 0/1 (Kernel Repo)

This repository now exposes the Phase-1 activation-quant contract surface:

- `activation_quant::ActivationQuantMode`
- `activation_quant::ActivationQuantConfig`
- `iq2_matmul::iq2_matmul_with_activation_quant(...)`

Current behavior:

- No kernel math changes yet.
- `iq2_matmul_with_activation_quant` delegates to existing `iq2_matmul`.
- If `strict=true` and mode is not `Off`, it returns an explicit error.

This lets `candelora` integrate stable API/CLI knobs now, while keeping current
inference numerics unchanged until W8A8/W4A8 kernels land in later phases.
