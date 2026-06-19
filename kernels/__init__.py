"""
cuda-triton-kernels — Triton GPU kernels for transformer operations.

Public API:
    softmax_triton(x)                         row-wise numerically-stable softmax
    rmsnorm_triton(x, weight, eps)            RMSNorm
    swiglu_triton(x, gate)                    SwiGLU activation
    flash_attention_triton(Q, K, V, causal)   FlashAttention-2 forward
    matmul_fp16_triton(x, W)                  tiled fp16 matmul
    matmul_int8_triton(x, W, scale, zp)       int8 dequant matmul
    matmul_int4_triton(x, W, scales, zeros)   int4 dequant matmul

All functions fall back gracefully: if Triton is unavailable or the tensor
is on CPU, callers should catch ImportError and fall back to PyTorch.
"""

from .softmax import softmax_triton, softmax_pytorch
from .rmsnorm import rmsnorm_triton, rmsnorm_pytorch
from .swiglu import swiglu_triton
from .flash_attention import flash_attention_triton
from .quantized_matmul import (
    matmul_fp16_triton,
    matmul_int8_triton,
    matmul_int4_triton,
    quantize_int8,
    quantize_int4,
    dequantize_int8,
    dequantize_int4,
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
    # flash attention
    "flash_attention_triton",
    # matmul
    "matmul_fp16_triton",
    "matmul_int8_triton",
    "matmul_int4_triton",
    # quantization utilities
    "quantize_int8",
    "quantize_int4",
    "dequantize_int8",
    "dequantize_int4",
]
