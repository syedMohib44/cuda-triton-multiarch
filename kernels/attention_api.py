"""
flash_attention_bhsd — Flash attention for (B, H, T, d) layout (KVQuant-native).

KVQuant stores K/V as (batch, heads, seqlen, head_dim).
cuda-triton's CUDA FA kernels expect (batch, seqlen, heads, head_dim).
This module bridges the two with a simple transpose + the best available backend.

Backend priority:
  1. flash_attn_cuda    (WMMA, SM75/80/86/89/120)  — fastest, hardware tensor cores
  2. flash_attn_cutlass (CuTe, SM80+)              — slightly slower on SM86/89/120
  3. flash_attention_triton per head (Triton)       — any GPU, O(T) memory
  4. F.scaled_dot_product_attention                 — CPU / bf16 / fp32 fallback
"""

import torch
import torch.nn.functional as F
from torch import Tensor

# WMMA CUDA backend (built via: powershell -File Makefile.windows.ps1 build-fac)
try:
    import flash_attn_cuda as _wmma
    _WMMA_AVAILABLE = True
except ImportError:
    _WMMA_AVAILABLE = False

# CuTe/CUTLASS backend (built via: powershell -File Makefile.windows.ps1 build-fac-cutlass)
try:
    import flash_attn_cutlass as _cutlass
    _CUTLASS_AVAILABLE = True
except ImportError:
    _CUTLASS_AVAILABLE = False

# Triton backend (pure Python, JIT-compiled on first call)
try:
    from .flash_attention import flash_attention_triton as _triton_fa
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def flash_attention_bhsd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
) -> Tensor:
    """
    Multi-head flash attention for (B, H, T, d) tensors.

    Args:
        q:         Query  — (batch, heads, seqlen_q, head_dim), fp16/bf16/fp32
        k:         Key    — (batch, heads, seqlen_k, head_dim)
        v:         Value  — (batch, heads, seqlen_k, head_dim)
        is_causal: If True, applies causal (lower-triangular) mask.

    Returns:
        Attention output — same shape and dtype as q.
    """
    orig_dtype = q.dtype

    # CUDA backends only support fp16 on GPU
    if q.is_cuda and q.dtype in (torch.float16, torch.float32, torch.bfloat16):
        q_half = q.half()
        k_half = k.half()
        v_half = v.half()

        # (B, H, T, d) → (B, T, H, d) for CUDA kernels
        qT = q_half.transpose(1, 2).contiguous()
        kT = k_half.transpose(1, 2).contiguous()
        vT = v_half.transpose(1, 2).contiguous()

        out = None
        if _WMMA_AVAILABLE:
            try:
                out, _ = _wmma.mha_fwd(qT, kT, vT, is_causal)
            except Exception:
                out = None

        if out is None and _CUTLASS_AVAILABLE:
            try:
                out, _ = _cutlass.mha_fwd(qT, kT, vT, is_causal)
            except Exception:
                out = None

        if out is not None:
            # (B, T, H, d) → (B, H, T, d)
            return out.transpose(1, 2).to(orig_dtype)

        if _TRITON_AVAILABLE:
            # Triton FA is 2D single-head; loop over batch and heads
            B, H, T, d = q_half.shape
            result = torch.empty_like(q_half)
            for b in range(B):
                for h in range(H):
                    result[b, h] = _triton_fa(q_half[b, h], k_half[b, h], v_half[b, h])
            return result.to(orig_dtype)

    # PyTorch SDPA: handles CPU / bf16 / fp32 / MPS
    # F.sdpa expects (B, H, T, d) — same as our input, no transpose needed
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


def flash_attention_backend() -> str:
    """Return the name of the backend that flash_attention_bhsd will use."""
    if _WMMA_AVAILABLE:
        return "flash_attn_cuda (WMMA)"
    if _CUTLASS_AVAILABLE:
        return "flash_attn_cutlass (CuTe)"
    if _TRITON_AVAILABLE:
        return "flash_attention_triton (Triton)"
    return "scaled_dot_product_attention (PyTorch)"


__all__ = ["flash_attention_bhsd", "flash_attention_backend"]
