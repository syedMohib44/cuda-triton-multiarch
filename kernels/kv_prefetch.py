"""
kv_prefetch.py — DualPath-inspired async KV cache prefetcher.

DualPath (Wu et al., 2025) shows that GPU underutilization in agentic LLM
inference is caused by sequential KV cache I/O and compute: the GPU sits idle
waiting for KV tensors to transfer from storage. Their fix uses dual storage
paths on separate NICs in a multi-node cluster.

On a single GPU the same principle applies via CUDA streams:
  - Compute stream  → runs attention / model forward
  - Transfer stream → overlaps H2D copy of the *next* KV chunk with current compute

This module provides:
  KVCachePrefetcher  — pins KV tensors in CPU RAM, prefetches to GPU ahead of use
  is_bandwidth_bound — tells you if a given (seqlen, hdim) is memory-bound on
                       the current GPU (compute-bound workloads don't benefit)

Usage:
    from kernels.kv_prefetch import KVCachePrefetcher

    prefetcher = KVCachePrefetcher()

    # Offload compressed KV to CPU after each turn
    cpu_k = prefetcher.offload(k_gpu)   # async D2H, returns pinned CPU tensor
    cpu_v = prefetcher.offload(v_gpu)

    # Before the next turn, start prefetch in background
    prefetcher.prefetch(cpu_k, cpu_v)   # non-blocking H2D on transfer stream

    # When you need the tensors — overlapped with your compute
    k_gpu, v_gpu = prefetcher.wait()    # sync only if transfer not done yet
"""

from __future__ import annotations

import torch
from torch import Tensor

try:
    from gpu_utils import get_gpu_info
    _GPU_INFO_AVAILABLE = True
except ImportError:
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from gpu_utils import get_gpu_info
        _GPU_INFO_AVAILABLE = True
    except ImportError:
        _GPU_INFO_AVAILABLE = False


def is_bandwidth_bound(seqlen: int, num_heads: int, head_dim: int,
                       device: int | str | torch.device | None = None) -> bool:
    """
    Return True if the attention workload at this size is memory-bandwidth-bound
    on the current GPU — meaning async prefetch will meaningfully reduce idle time.

    Arithmetic intensity of attention (simplified, decode phase):
      FLOPs  ≈ 4 * seqlen * num_heads * head_dim  (QK^T + softmax*V, rough)
      Bytes  ≈ 2 * seqlen * num_heads * head_dim * 2  (fp16 K + V)
      AI     = FLOPs / Bytes  =  1.0  (ops/byte, always ~1 for decode)

    Ridge point = fp16_tflops (TOPs) / hbm_bw (TB/s).
    If AI < ridge point → memory bound → prefetch helps.
    """
    if not torch.cuda.is_available() or not _GPU_INFO_AVAILABLE:
        return True  # conservative: always prefetch if we can't tell

    try:
        info = get_gpu_info(device)
    except RuntimeError:
        return True

    hbm_bw   = info["hbm_bw_gbs"]   # GB/s
    fp16_peak = info["fp16_tflops"]  # TFLOPs

    if hbm_bw <= 0 or fp16_peak <= 0:
        return True

    # Ridge point in ops/byte
    ridge = (fp16_peak * 1e12) / (hbm_bw * 1e9)

    # Attention arithmetic intensity: FLOPs / Bytes
    flops = 4 * seqlen * num_heads * head_dim
    bytes_ = 2 * seqlen * num_heads * head_dim * 2  # fp16 K + V
    ai = flops / bytes_  # = 1.0 always, kept explicit for clarity

    return ai < ridge


class KVCachePrefetcher:
    """
    Async KV cache offload + prefetch using a dedicated CUDA transfer stream.

    The compute stream runs attention uninterrupted. The transfer stream moves
    KV tensors between CPU pinned memory and GPU in the background — the same
    dual-path overlap that DualPath achieves across storage NICs.

    Args:
        device: GPU device to prefetch to (default: current CUDA device).
        pin_memory: Use pinned (page-locked) CPU memory for maximum H2D bandwidth.
    """

    def __init__(self, device: int | str | torch.device | None = None,
                 pin_memory: bool = True):
        if not torch.cuda.is_available():
            raise RuntimeError("KVCachePrefetcher requires a CUDA GPU.")

        self.device = torch.device(device if device is not None
                                   else torch.cuda.current_device())
        self.pin_memory = pin_memory

        # Dedicated non-blocking stream for H2D / D2H transfers.
        # Runs concurrently with the default compute stream.
        self.transfer_stream = torch.cuda.Stream(device=self.device)

        # Staging area: GPU tensors ready to be consumed by compute stream
        self._prefetched_k: Tensor | None = None
        self._prefetched_v: Tensor | None = None
        self._event = torch.cuda.Event()

    # ------------------------------------------------------------------
    # Offload: GPU → CPU pinned  (async D2H)
    # ------------------------------------------------------------------

    def offload(self, t: Tensor) -> Tensor:
        """
        Asynchronously copy a GPU tensor to pinned CPU memory.
        Returns the CPU tensor (transfer runs in background on transfer_stream).
        Call torch.cuda.synchronize() or .wait() before reading the CPU tensor.
        """
        cpu_buf = torch.empty_like(t, device="cpu",
                                   pin_memory=self.pin_memory and torch.cuda.is_available())
        with torch.cuda.stream(self.transfer_stream):
            cpu_buf.copy_(t, non_blocking=True)
        return cpu_buf

    def offload_kv(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        """Offload a K,V pair to CPU in one call. Returns (cpu_k, cpu_v)."""
        return self.offload(k), self.offload(v)

    # ------------------------------------------------------------------
    # Prefetch: CPU pinned → GPU  (async H2D)
    # ------------------------------------------------------------------

    def prefetch(self, cpu_k: Tensor, cpu_v: Tensor) -> None:
        """
        Start async H2D transfer of cpu_k, cpu_v to GPU on the transfer stream.
        Returns immediately — call wait() to get the GPU tensors.
        """
        with torch.cuda.stream(self.transfer_stream):
            self._prefetched_k = cpu_k.to(self.device, non_blocking=True)
            self._prefetched_v = cpu_v.to(self.device, non_blocking=True)
            self._event.record(self.transfer_stream)

    def wait(self) -> tuple[Tensor, Tensor]:
        """
        Return prefetched (k_gpu, v_gpu). If the transfer is still in flight,
        blocks only until it completes — overlapping with any compute already
        issued to the default stream.
        """
        if self._prefetched_k is None:
            raise RuntimeError("No prefetch in flight. Call prefetch() first.")

        # Make the current compute stream wait for the transfer stream event.
        # This is a GPU-side dependency — the CPU is not blocked.
        torch.cuda.current_stream().wait_event(self._event)

        k, v = self._prefetched_k, self._prefetched_v
        self._prefetched_k = None
        self._prefetched_v = None
        return k, v

    # ------------------------------------------------------------------
    # Convenience: offload → prefetch in one call (round-trip cache swap)
    # ------------------------------------------------------------------

    def swap(self, k_gpu: Tensor, v_gpu: Tensor,
             next_cpu_k: Tensor, next_cpu_v: Tensor) -> tuple[Tensor, Tensor]:
        """
        Simultaneously:
          1. Start offloading k_gpu / v_gpu to CPU (D2H)
          2. Start prefetching next_cpu_k / next_cpu_v to GPU (H2D)

        Returns (next_k_gpu, next_v_gpu) after both complete.
        Both transfers run concurrently on the transfer stream, overlapping
        with any pending compute on the default stream.
        """
        with torch.cuda.stream(self.transfer_stream):
            # D2H — archive current turn's KV
            cpu_k_out = torch.empty_like(k_gpu, device="cpu",
                                         pin_memory=self.pin_memory)
            cpu_v_out = torch.empty_like(v_gpu, device="cpu",
                                         pin_memory=self.pin_memory)
            cpu_k_out.copy_(k_gpu, non_blocking=True)
            cpu_v_out.copy_(v_gpu, non_blocking=True)

            # H2D — load next turn's KV
            next_k_gpu = next_cpu_k.to(self.device, non_blocking=True)
            next_v_gpu = next_cpu_v.to(self.device, non_blocking=True)
            self._event.record(self.transfer_stream)

        torch.cuda.current_stream().wait_event(self._event)
        return next_k_gpu, next_v_gpu

    def __repr__(self) -> str:
        info_str = ""
        if _GPU_INFO_AVAILABLE and torch.cuda.is_available():
            try:
                info = get_gpu_info(self.device)
                info_str = (f"  SM{info['sm_version']}  "
                            f"HBM={info['hbm_bw_gbs']:.0f} GB/s  "
                            f"PCIe H2D={info['h2d_bw_gbs']:.0f} GB/s")
            except Exception:
                pass
        return f"KVCachePrefetcher(device={self.device}, pin_memory={self.pin_memory}){info_str}"
