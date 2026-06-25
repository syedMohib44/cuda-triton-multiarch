"""
cuda-triton-kernels — Triton GPU kernels for transformer operations.

Public API:
    softmax_triton(x)                         row-wise numerically-stable softmax
    rmsnorm_triton(x, weight, eps)            RMSNorm
    swiglu_triton(x, gate)                    SwiGLU activation
    flash_attention_triton(Q, K, V)           FlashAttention-2 forward — (T, d) single-head
    flash_attention_bhsd(q, k, v, is_causal)  FlashAttention-2 forward — (B, H, T, d) multi-head
    flash_attention_backend()                 returns active backend name
    matmul_fp16_triton(x, W)                  tiled fp16 matmul
    matmul_int8_triton(x, W, scale, zp)       int8 dequant matmul
    matmul_int4_triton(x, W, scales, zeros)   int4 dequant matmul
    KVCachePrefetcher                                async KV offload + prefetch (DualPath-style)
    is_bandwidth_bound(seqlen, heads, hdim)          True if workload is HBM-bound on current GPU
    flash_attention_multi_gpu(q, k, v, is_causal)   data-parallel attention across all GPUs
    MultiGPUAttentionPool                            load-aware GPU dispatcher (DualPath Algorithm 1)
    num_gpus()                                       number of available CUDA GPUs
    all_gpu_names()                                  list of GPU names

All functions fall back gracefully to single-GPU or PyTorch if Triton/CUDA
extensions are unavailable or tensors are on CPU.
"""

from .softmax import softmax_triton, softmax_pytorch
from .rmsnorm import rmsnorm_triton, rmsnorm_pytorch
from .swiglu import swiglu_triton
from .flash_attention import flash_attention_triton
from .attention_api import flash_attention_bhsd, flash_attention_backend
from .quantized_matmul import (
    matmul_fp16_triton,
    matmul_int8_triton,
    matmul_int4_triton,
    quantize_int8,
    quantize_int4,
    dequantize_int8,
    dequantize_int4,
)
from .kv_prefetch import KVCachePrefetcher, is_bandwidth_bound
from .multi_gpu import (
    flash_attention_multi_gpu,
    MultiGPUAttentionPool,
    num_gpus,
    all_gpu_names,
)

__all__ = [
    # softmax
    "softmax_triton",
    "softmax_pytorch",
    # rmsnorm
    "rmsnorm_triton",
    "rmsnorm_pytorch",
    # swiglu
    "swiglu_triton",
    # flash attention — single GPU
    "flash_attention_triton",
    "flash_attention_bhsd",
    "flash_attention_backend",
    # flash attention — multi GPU
    "flash_attention_multi_gpu",
    "MultiGPUAttentionPool",
    "num_gpus",
    "all_gpu_names",
    # matmul
    "matmul_fp16_triton",
    "matmul_int8_triton",
    "matmul_int4_triton",
    # quantization utilities
    "quantize_int8",
    "quantize_int4",
    "dequantize_int8",
    "dequantize_int4",
    # async KV prefetch (DualPath-style)
    "KVCachePrefetcher",
    "is_bandwidth_bound",
]
