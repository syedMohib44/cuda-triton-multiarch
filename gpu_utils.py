"""
gpu_utils.py — GPU detection and per-SM configuration.

Single source of truth used by:
  - Triton kernels  (block size selection)
  - CUDA dispatch   (SM-aware kernel selection at runtime)
  - Benchmarks      (accurate peak TFLOPS reference)

Supported SM versions:
  SM75  — Turing   (T4, RTX 2080 Ti)      : 64 KB smem, no cp.async
  SM80  — Ampere   (A100, A30)             : 164 KB smem, cp.async
  SM86  — Ampere   (RTX 3090, A10)         : 100 KB smem, cp.async
  SM89  — Ada      (RTX 4090, L40S)        : 100 KB smem, cp.async
  SM90  — Hopper   (H100)                  : 228 KB smem, cp.async
  SM120 — Blackwell (RTX 5070/5090)        : 228 KB smem, cp.async
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
    75:  64  * 1024,
    80:  164 * 1024,
    86:  100 * 1024,
    89:  100 * 1024,
    90:  228 * 1024,
    120: 228 * 1024,
}

# fp16 tensor-core peak (TFLOPs, dense, no sparsity).
# Source: NVIDIA product specs / GPU white papers.
_FP16_PEAK_TFLOPS = {
    75:  65.0,   # T4 65 TFLOPS, RTX 2080 Ti ~107 TFLOPS (use T4 as conservative)
    80:  312.0,  # A100 SXM4 80 GB
    86:  142.0,  # RTX 3090 (35.6 TFLOPS FP32 × 2 for tensor = ~71; with sparsity 142)
    89:  330.0,  # RTX 4090 (82.6 TFLOPS FP32 × 4 = ~330 dense tensor)
    90:  494.0,  # H100 SXM5 (dense FP16 tensor core)
    120: 145.0,  # RTX 5070 (estimated; varies by model)
}

# HBM memory bandwidth (GB/s).
# Source: NVIDIA product specs.
# Used to determine whether a workload is compute-bound or memory-bandwidth-bound,
# and to size async KV prefetch buffers (DualPath-style overlap of I/O + compute).
_HBM_BW_GBS = {
    75:  320.0,  # T4 (GDDR6, not HBM; RTX 2080 Ti ~616 GB/s)
    80:  2000.0, # A100 SXM4 80 GB (HBM2e)
    86:  936.0,  # RTX 3090 (GDDR6X)
    89:  1008.0, # RTX 4090 (GDDR6X)
    90:  3350.0, # H100 SXM5 (HBM3)
    120: 896.0,  # RTX 5070 (GDDR7, estimated)
}

# PCIe / NVLink host-to-device bandwidth (GB/s) for CPU-offload transfers.
# PCIe 4.0 x16 ≈ 32 GB/s, PCIe 5.0 x16 ≈ 64 GB/s.
# Pinned memory transfers saturate this; pageable memory is ~50% of these values.
# Used by KVCachePrefetcher to decide prefetch lead time.
_H2D_BW_GBS = {
    75:  16.0,  # PCIe 3.0 era
    80:  32.0,  # PCIe 4.0 (A100 PCIe) / NVLink for SXM
    86:  32.0,  # PCIe 4.0
    89:  32.0,  # PCIe 4.0
    90:  64.0,  # PCIe 5.0 / NVLink
    120: 64.0,  # PCIe 5.0 (RTX 5070)
}

# Whether cp.async is available (SM80+).
_HAS_CP_ASYNC = {
    75:  False,
    80:  True,
    86:  True,
    89:  True,
    90:  True,
    120: True,
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
    75:  {32: (64, 32),  64: (64, 32),  128: (64, 32)},
    80:  {32: (128, 64), 64: (128, 64), 128: (128, 64)},
    86:  {32: (128, 64), 64: (128, 64), 128: (128, 32)},
    89:  {32: (128, 64), 64: (128, 64), 128: (128, 32)},
    90:  {32: (128, 64), 64: (128, 64), 128: (128, 64)},
    120: {32: (128, 64), 64: (128, 64), 128: (128, 64)},
}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Public API (Dynamically Updated)
# ---------------------------------------------------------------------------

def get_gpu_info(device: int | str | torch.device | None = None) -> dict:
    """Return GPU properties for the given device (default: current CUDA device)."""
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available.")

    if device is None:
        device = torch.cuda.current_device()
        
    props = torch.cuda.get_device_properties(device)
    sm = props.major * 10 + props.minor

    # OS-AGNOSTIC SMEM: 
    # Try the newer 'optin' property first, fallback to the standard property, 
    # and if Windows PyTorch is missing both, fallback to the 64KB CUDA minimum.
    dynamic_smem = getattr(props, 'max_shared_memory_per_block_optin', 
                   getattr(props, 'max_shared_memory_per_block', 65536))

    return {
        "name": props.name,
        "sm_version": sm,
        "max_smem": _MAX_SMEM.get(sm, dynamic_smem),
        "has_cp_async": sm >= 80,
        "fp16_tflops": _FP16_PEAK_TFLOPS.get(sm, 0.0),
        "hbm_bw_gbs": _HBM_BW_GBS.get(sm, 0.0),   # HBM bandwidth — 0.0 = unknown
        "h2d_bw_gbs": _H2D_BW_GBS.get(sm, 32.0),   # PCIe H2D bandwidth — 32 GB/s default
        "supported": sm >= 75,
    }


def get_optimal_block_sizes(sm: int, head_dim: int) -> tuple[int, int]:
    """Return (BLOCK_Q, BLOCK_KV) tuned for the given SM and head_dim."""
    
    # DYNAMIC FALLBACK:
    # If the exact SM isn't in our dictionary, gracefully downgrade to the 
    # closest known architecture instead of crashing or defaulting to the slowest.
    if sm not in _BLOCK_SIZES:
        if sm >= 120:   closest_sm = 120  # Future Blackwell+
        elif sm >= 90:  closest_sm = 90   # Hopper+
        elif sm >= 80:  closest_sm = 80   # Ampere/Ada+
        else:           closest_sm = 75   # Turing
        sizes = _BLOCK_SIZES[closest_sm]
    else:
        sizes = _BLOCK_SIZES[sm]

    # Find the closest supported head_dim key (32, 64, 128)
    if head_dim in sizes:
        return sizes[head_dim]
    # Round up to next supported head_dim
    for key in sorted(sizes):
        if key >= head_dim:
            return sizes[key]
    return sizes[max(sizes)]



def get_all_gpu_info() -> list:
    """Return info dicts for every available CUDA GPU."""
    if not torch.cuda.is_available():
        return []
    return [get_gpu_info(i) for i in range(torch.cuda.device_count())]


def get_best_device() -> int:
    """Return the index of the GPU with the highest fp16 TFLOPS.
    Falls back to device 0 if no GPU has known peak data."""
    gpus = get_all_gpu_info()
    if not gpus:
        raise RuntimeError("No CUDA GPU available.")
    return max(range(len(gpus)), key=lambda i: gpus[i]["fp16_tflops"])


def get_multi_gpu_strategy(batch_size: int, num_heads: int) -> str:
    """Return the recommended parallelism strategy for the given workload.

    Returns one of:
      'single' — only one GPU available, or workload too small to split
      'batch'  — split batch dimension across GPUs (batch_size >= n_gpus)
      'head'   — split head dimension across GPUs (num_heads >= n_gpus)
    """
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return "single"
    if batch_size >= n_gpus:
        return "batch"
    if num_heads >= n_gpus:
        return "head"
    return "single"


def check_gpu_support(raise_on_unsupported: bool = True) -> dict:
    """Check the current GPU and warn/raise if not supported."""
    info = get_gpu_info()
    if not info["supported"]:
        msg = (
            f"GPU '{info['name']}' (SM{info['sm_version']}) is too old.\n"
            f"This project requires an NVIDIA GPU with SM75 (Turing) or higher."
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
        strategy_b = get_multi_gpu_strategy(batch_size=8, num_heads=8)
        print(f"Found {n} GPU(s)  |  multi-GPU strategy (batch=8, heads=8): {strategy_b}\n")
        for i in range(n):
            info = get_gpu_info(i)
            sm = info["sm_version"]
            bq64, bkv64 = get_optimal_block_sizes(sm, 64)
            bq128, bkv128 = get_optimal_block_sizes(sm, 128)
            peak = info["fp16_tflops"]
            peak_str = f"{peak:.0f} TFLOPS" if peak > 0 else "unknown"

            hbm  = info["hbm_bw_gbs"]
            h2d  = info["h2d_bw_gbs"]
            hbm_str = f"{hbm:.0f} GB/s" if hbm > 0 else "unknown"
            print(f"  [{i}] {info['name']}")
            print(f"       SM{sm}  |  max_smem={info['max_smem']//1024} KB"
                  f"  |  cp.async={'yes' if info['has_cp_async'] else 'no'}"
                  f"  |  fp16 peak={peak_str}"
                  f"  |  supported={'yes' if info['supported'] else 'NO'}")
            print(f"       HBM bw={hbm_str}  |  PCIe H2D={h2d:.0f} GB/s")
            print(f"       Optimal blocks: hdim64→({bq64},{bkv64})  hdim128→({bq128},{bkv128})")
            print()
