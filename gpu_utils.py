"""
gpu_utils.py — GPU detection and per-SM configuration.

Single source of truth used by:
  - Triton kernels  (block size selection)
  - CUDA dispatch   (SM-aware kernel selection at runtime)
  - Benchmarks      (accurate peak TFLOPS reference)

Supported SM versions:
  SM75  — Turing  (T4, RTX 2080 Ti)      : 64 KB smem, no cp.async
  SM80  — Ampere  (A100, A30)             : 164 KB smem, cp.async
  SM86  — Ampere  (RTX 3090, A10)         : 100 KB smem, cp.async
  SM89  — Ada     (RTX 4090, L40S)        : 100 KB smem, cp.async
  SM90  — Hopper  (H100)                  : NOT supported (needs wgmma/TMA)
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Per-SM static data
# ---------------------------------------------------------------------------

# Max shared memory per SM (bytes).
# SM75: 64 KB default max (can request up to 64 KB with carveout)
# SM80: 164 KB max (A100 has 192 KB total, 164 KB available to a single CTA)
# SM86: 100 KB max (RTX 30 series Ampere)
# SM89: 100 KB max (Ada Lovelace)
_MAX_SMEM = {
    75: 64 * 1024,
    80: 164 * 1024,
    86: 100 * 1024,
    89: 100 * 1024,
}

# fp16 tensor-core peak (TFLOPs, dense, no sparsity).
# Source: NVIDIA product specs / GPU white papers.
_FP16_PEAK_TFLOPS = {
    75: 65.0,    # T4 65 TFLOPS, RTX 2080 Ti ~107 TFLOPS (use T4 as conservative)
    80: 312.0,   # A100 SXM4 80 GB
    86: 142.0,   # RTX 3090 (35.6 TFLOPS FP32 × 2 for tensor = ~71; with sparsity 142)
    89: 330.0,   # RTX 4090 (82.6 TFLOPS FP32 × 4 = ~330 dense tensor)
}

# Whether cp.async is available (SM80+).
_HAS_CP_ASYNC = {
    75: False,
    80: True,
    86: True,
    89: True,
}

# (BLOCK_Q, BLOCK_KV) defaults per (sm, head_dim).
# Chosen to keep smem usage well under each GPU's limit.
#
# smem breakdown (no double buffer):
#   Q:      BLOCK_Q  × (head_dim + 8) × 2  bytes
#   KV:     BLOCK_KV × (head_dim + 8) × 2  bytes
#   scores: BLOCK_Q  × (BLOCK_KV + 4) × 4  bytes
#   P:      BLOCK_Q  × (BLOCK_KV + 8) × 2  bytes
#
# SM75 (64 KB):  hdim64→(64,32)≈27 KB  hdim128→(64,32)≈43 KB
# SM80 (164 KB): hdim64→(128,64)≈79 KB hdim128→(128,64)≈136 KB
# SM86 (100 KB): hdim64→(128,64)≈79 KB hdim128→(128,32)≈79 KB  (reduce BLOCK_KV)
# SM89 (100 KB): same as SM86
_BLOCK_SIZES: dict[int, dict[int, tuple[int, int]]] = {
    75: {32: (64, 32), 64: (64, 32), 128: (64, 32)},
    80: {32: (128, 64), 64: (128, 64), 128: (128, 64)},
    86: {32: (128, 64), 64: (128, 64), 128: (128, 32)},
    89: {32: (128, 64), 64: (128, 64), 128: (128, 32)},
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_gpu_info(device: int | str | torch.device | None = None) -> dict:
    """Return GPU properties for the given device (default: current CUDA device).

    Returns a dict with keys:
        name         (str)  — e.g. "NVIDIA A100-SXM4-80GB"
        sm_version   (int)  — e.g. 80 for SM8.0
        max_smem     (int)  — max shared memory per CTA in bytes
        has_cp_async (bool) — whether cp.async is available
        fp16_tflops  (float)— fp16 tensor-core peak TFLOPs (0.0 if unknown)
        supported    (bool) — whether this project has a kernel for this GPU
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")

    if device is None:
        device = torch.cuda.current_device()

    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor

    return {
        "name": props.name,
        "sm_version": sm,
        "max_smem": _MAX_SMEM.get(sm, props.max_shared_memory_per_block),
        "has_cp_async": _HAS_CP_ASYNC.get(sm, sm >= 80),
        "fp16_tflops": _FP16_PEAK_TFLOPS.get(sm, 0.0),
        "supported": sm in _MAX_SMEM,
    }


def get_optimal_block_sizes(sm: int, head_dim: int) -> tuple[int, int]:
    """Return (BLOCK_Q, BLOCK_KV) tuned for the given SM and head_dim.

    Falls back to the most conservative config if SM is unknown.
    """
    if sm not in _BLOCK_SIZES:
        # Unknown GPU — use smallest safe config
        return (64, 32)

    # Find the closest supported head_dim key (32, 64, 128)
    sizes = _BLOCK_SIZES[sm]
    if head_dim in sizes:
        return sizes[head_dim]
    # Round up to next supported head_dim
    for key in sorted(sizes):
        if key >= head_dim:
            return sizes[key]
    return sizes[max(sizes)]


def check_gpu_support(raise_on_unsupported: bool = True) -> dict:
    """Check the current GPU and warn/raise if not supported.

    Returns gpu_info dict (same as get_gpu_info).
    """
    info = get_gpu_info()
    if not info["supported"]:
        msg = (
            f"GPU '{info['name']}' (SM{info['sm_version']}) is not supported.\n"
            f"Supported: SM75 (Turing), SM80 (A100), SM86 (Ampere), SM89 (Ada).\n"
            f"SM90 (H100) support requires TMA/wgmma — not yet implemented."
        )
        if raise_on_unsupported:
            raise RuntimeError(msg)
        else:
            import warnings
            warnings.warn(msg)
    return info


# ---------------------------------------------------------------------------
# CLI — print info when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("No CUDA GPU detected.")
    else:
        n = torch.cuda.device_count()
        print(f"Found {n} GPU(s):\n")
        for i in range(n):
            info = get_gpu_info(i)
            sm = info["sm_version"]
            bq64, bkv64 = get_optimal_block_sizes(sm, 64)
            bq128, bkv128 = get_optimal_block_sizes(sm, 128)
            peak = info["fp16_tflops"]
            peak_str = f"{peak:.0f} TFLOPS" if peak > 0 else "unknown"

            print(f"  [{i}] {info['name']}")
            print(f"       SM{sm}  |  max_smem={info['max_smem']//1024} KB"
                  f"  |  cp.async={'yes' if info['has_cp_async'] else 'no'}"
                  f"  |  fp16 peak={peak_str}"
                  f"  |  supported={'yes' if info['supported'] else 'NO'}")
            print(f"       Optimal blocks: hdim64→({bq64},{bkv64})  hdim128→({bq128},{bkv128})")
            print()
