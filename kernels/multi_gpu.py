"""
multi_gpu.py — Data-parallel flash attention across any number of GPUs.

Single-machine analogue of DualPath Algorithm 1 (Wu et al., 2025):
  - Each GPU is a "Processing Engine" (PE)
  - Requests (batch items) are assigned to the least-loaded GPU
  - Work runs concurrently on all GPUs via separate CUDA streams
  - Results are gathered back to the source device

Parallelism strategies (chosen automatically):
  'batch' — split B dimension across N GPUs  (default when B >= N)
  'head'  — split H dimension across N GPUs  (when B < N but H >= N)
  'single'— single GPU, no overhead          (N=1 or workload too small)

Usage:
    from kernels.multi_gpu import flash_attention_multi_gpu, num_gpus

    # q, k, v: (B, H, T, d) — any dtype, any device
    out = flash_attention_multi_gpu(q, k, v, is_causal=False)
    print(f"Used {num_gpus()} GPU(s)")
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import torch
from torch import Tensor

try:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from gpu_utils import get_all_gpu_info, get_best_device, get_multi_gpu_strategy
    _GPU_UTILS = True
except ImportError:
    _GPU_UTILS = False

from .attention_api import flash_attention_bhsd


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def num_gpus() -> int:
    """Number of available CUDA GPUs (0 if CPU-only)."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def all_gpu_names() -> list:
    """Names of all available GPUs, e.g. ['RTX 5070', 'A100']."""
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


# ---------------------------------------------------------------------------
# Core: multi-GPU attention
# ---------------------------------------------------------------------------

def flash_attention_multi_gpu(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    strategy: str = "auto",
) -> Tensor:
    """
    Flash attention over (B, H, T, d) tensors, dispatched across all GPUs.

    Args:
        q, k, v:    (batch, heads, seqlen, head_dim) — fp16/bf16/fp32
        is_causal:  apply causal mask
        strategy:   'auto' | 'batch' | 'head' | 'single'
                    'auto' picks the best strategy for the workload size.

    Returns:
        Output tensor on the same device as q.

    Single-GPU fallback:
        If only one GPU is available, or the workload is too small to split,
        delegates directly to flash_attention_bhsd — zero overhead.
    """
    n = num_gpus()

    # ---- CPU or single-GPU path -------------------------------------------
    if n <= 1 or not q.is_cuda:
        return flash_attention_bhsd(q, k, v, is_causal=is_causal)

    B, H, T, d = q.shape
    src_device = q.device

    # ---- Choose strategy ---------------------------------------------------
    if strategy == "auto":
        strategy = get_multi_gpu_strategy(B, H) if _GPU_UTILS else (
            "batch" if B >= n else "head" if H >= n else "single"
        )

    if strategy == "single" or n == 1:
        return flash_attention_bhsd(q, k, v, is_causal=is_causal)

    # ---- Split tensors and dispatch ----------------------------------------
    if strategy == "batch":
        return _dispatch_batch(q, k, v, is_causal, n, src_device)
    elif strategy == "head":
        return _dispatch_head(q, k, v, is_causal, n, src_device)
    else:
        return flash_attention_bhsd(q, k, v, is_causal=is_causal)


# ---------------------------------------------------------------------------
# Batch-parallel: split B across GPUs
# ---------------------------------------------------------------------------

def _dispatch_batch(
    q: Tensor, k: Tensor, v: Tensor,
    is_causal: bool, n_gpus: int, src_device: torch.device,
) -> Tensor:
    """
    Split the batch dimension across n_gpus GPUs.
    Each GPU gets ceil(B/n_gpus) samples. Runs concurrently via CUDA streams.
    Results gathered back to src_device.
    """
    B = q.shape[0]
    # Chunk sizes — last chunk may be smaller
    chunk = (B + n_gpus - 1) // n_gpus
    q_chunks = q.split(chunk, dim=0)
    k_chunks = k.split(chunk, dim=0)
    v_chunks = v.split(chunk, dim=0)

    streams  = [torch.cuda.Stream(device=i) for i in range(len(q_chunks))]
    results  = [None] * len(q_chunks)

    # Dispatch each chunk to its GPU concurrently
    for i, (qi, ki, vi) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        dev = torch.device(f"cuda:{i}")
        with torch.cuda.stream(streams[i]):
            qi_d = qi.to(dev, non_blocking=True)
            ki_d = ki.to(dev, non_blocking=True)
            vi_d = vi.to(dev, non_blocking=True)
            results[i] = flash_attention_bhsd(qi_d, ki_d, vi_d, is_causal=is_causal)

    # Synchronize all streams and gather to src_device
    for i, stream in enumerate(streams):
        stream.synchronize()
        results[i] = results[i].to(src_device, non_blocking=False)

    return torch.cat(results, dim=0)


# ---------------------------------------------------------------------------
# Head-parallel: split H across GPUs
# ---------------------------------------------------------------------------

def _dispatch_head(
    q: Tensor, k: Tensor, v: Tensor,
    is_causal: bool, n_gpus: int, src_device: torch.device,
) -> Tensor:
    """
    Split the head dimension across n_gpus GPUs.
    Each GPU handles ceil(H/n_gpus) attention heads independently.
    Results gathered and concatenated along the head dimension.
    """
    H = q.shape[1]
    chunk = (H + n_gpus - 1) // n_gpus
    q_chunks = q.split(chunk, dim=1)
    k_chunks = k.split(chunk, dim=1)
    v_chunks = v.split(chunk, dim=1)

    streams = [torch.cuda.Stream(device=i) for i in range(len(q_chunks))]
    results = [None] * len(q_chunks)

    for i, (qi, ki, vi) in enumerate(zip(q_chunks, k_chunks, v_chunks)):
        dev = torch.device(f"cuda:{i}")
        with torch.cuda.stream(streams[i]):
            qi_d = qi.to(dev, non_blocking=True)
            ki_d = ki.to(dev, non_blocking=True)
            vi_d = vi.to(dev, non_blocking=True)
            results[i] = flash_attention_bhsd(qi_d, ki_d, vi_d, is_causal=is_causal)

    for i, stream in enumerate(streams):
        stream.synchronize()
        results[i] = results[i].to(src_device, non_blocking=False)

    return torch.cat(results, dim=1)


# ---------------------------------------------------------------------------
# DualPath-inspired load-aware dispatcher
# ---------------------------------------------------------------------------

class MultiGPUAttentionPool:
    """
    Full implementation of DualPath Algorithm 1 (Inter-PE Scheduling).

    Maps DualPath multi-node concepts to a single machine:
      Storage NIC → PCIe transfer stream
      RDMA read   → CUDA async H2D copy
      PE          → GPU device

    Per-GPU state (Algorithm 1 notation):
      tok_e    — tokens currently in flight (compute load)
      read_q_e — number of pending async KV transfers (I/O queue depth)
      seq_e    — total sequences processed (tie-break metric)

    Classification (Algorithm 1):
      C1 — overloaded:            tok_e > β
      C2 — ideal:     read_q_e ≤ α  AND  tok_e ≤ β
      C3 — io-bound:  read_q_e > α  AND  tok_e ≤ β

    Assignment preference: C2 (argmin tok) → C3 (argmin tok) → None (fallback)

    Usage:
        pool = MultiGPUAttentionPool(alpha=2, beta=65536)
        out  = pool.forward(q, k, v, is_causal=False)

        # Batch of requests (Algorithm 1 queue loop):
        requests = [(q0, k0, v0), (q1, k1, v1), ...]
        outputs  = pool.process_queue(requests, is_causal=False)

        print(pool.load_summary())
    """

    def __init__(self, alpha: int = 2, beta: int = 65536):
        """
        Args:
            alpha: I/O queue depth threshold — C2 if read_q ≤ α, else C3.
                   Algorithm 1 uses α to distinguish "transfer idle" from
                   "transfer busy". Default 2 (at most 2 in-flight copies).
            beta:  Token load threshold — C1 if tok > β.
                   Default 65536 (64K tokens ≈ one A100 attention block).
        """
        self.n    = num_gpus()
        self.alpha = alpha
        self.beta  = beta
        # Per-GPU counters (index = GPU id)
        self._tok    = [0] * max(self.n, 1)   # tokens in flight
        self._read_q = [0] * max(self.n, 1)   # pending async transfers
        self._seq    = [0] * max(self.n, 1)   # total sequences served
        self._lock   = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify(self) -> Tuple[List[int], List[int], List[int]]:
        """Return (C1, C2, C3) GPU id lists under the current lock state."""
        c1, c2, c3 = [], [], []
        for i in range(self.n):
            if self._tok[i] > self.beta:
                c1.append(i)
            elif self._read_q[i] <= self.alpha:
                c2.append(i)
            else:
                c3.append(i)
        return c1, c2, c3

    def _select_pe(self, c1: List[int], c2: List[int], c3: List[int]) -> Optional[int]:
        """
        Algorithm 1 DE selection:
          1. Prefer C2 (transfer-idle, compute-idle) — argmin tok_e
          2. Fall back to C3 (transfer-busy, compute-idle) — argmin tok_e
          3. If only C1 (all overloaded), pick argmin tok_e from C1 as last resort
          4. Return None only if n == 0 (no GPUs at all)
        """
        if c2:
            return min(c2, key=lambda i: self._tok[i])
        if c3:
            return min(c3, key=lambda i: self._tok[i])
        if c1:
            return min(c1, key=lambda i: self._tok[i])
        return None

    def _has_hbm(self, gpu_id: int, tokens: int, head_dim: int) -> bool:
        """
        Check whether gpu_id has enough free HBM for this request.
        Rough estimate: tokens × head_dim × 3 (Q+K+V) × 2 bytes (fp16).
        Returns True if free memory > required, or if CUDA is unavailable.
        """
        try:
            free, _ = torch.cuda.mem_get_info(gpu_id)
            needed  = tokens * head_dim * 3 * 2
            return free > needed
        except Exception:
            return True  # can't check → optimistic

    def _run_on_gpu(
        self,
        gpu_id: int,
        q: Tensor, k: Tensor, v: Tensor,
        is_causal: bool,
        src_device: torch.device,
        tokens: int,
    ) -> Tensor:
        """Move tensors to gpu_id, run attention, return result on src_device."""
        dev = torch.device(f"cuda:{gpu_id}")
        stream = torch.cuda.Stream(device=dev)

        with self._lock:
            self._tok[gpu_id]    += tokens
            self._read_q[gpu_id] += 1   # one async copy in flight

        with torch.cuda.stream(stream):
            q_d = q.to(dev, non_blocking=True)
            k_d = k.to(dev, non_blocking=True)
            v_d = v.to(dev, non_blocking=True)

        with self._lock:
            self._read_q[gpu_id] -= 1   # transfer complete

        with torch.cuda.stream(stream):
            out = flash_attention_bhsd(q_d, k_d, v_d, is_causal=is_causal)

        stream.synchronize()
        result = out.to(src_device, non_blocking=False)

        with self._lock:
            self._tok[gpu_id] -= tokens
            self._seq[gpu_id] += q.shape[0]  # B sequences done

        return result

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                is_causal: bool = False) -> Tensor:
        """
        Route a single attention call to the best available GPU.

        Single-GPU / CPU path: zero overhead, direct call.
        Multi-GPU path: Algorithm 1 classification + selection.
        """
        if self.n <= 1 or not q.is_cuda:
            return flash_attention_bhsd(q, k, v, is_causal=is_causal)

        tokens     = q.shape[0] * q.shape[2]  # B × T
        head_dim   = q.shape[3]
        src_device = q.device

        with self._lock:
            c1, c2, c3 = self._classify()
            gpu_id = self._select_pe(c1, c2, c3)

        if gpu_id is None:
            return flash_attention_bhsd(q, k, v, is_causal=is_causal)

        if not self._has_hbm(gpu_id, tokens, head_dim):
            # HBM check failed — try any GPU with free memory
            for i in range(self.n):
                if i != gpu_id and self._has_hbm(i, tokens, head_dim):
                    gpu_id = i
                    break
            else:
                return flash_attention_bhsd(q, k, v, is_causal=is_causal)

        return self._run_on_gpu(gpu_id, q, k, v, is_causal, src_device, tokens)

    def process_queue(
        self,
        requests: List[Tuple[Tensor, Tensor, Tensor]],
        is_causal: bool = False,
    ) -> List[Tensor]:
        """
        Algorithm 1 queue loop: dispatch a list of (q, k, v) requests
        across all GPUs concurrently using a ThreadPoolExecutor.

        Each request is assigned to the best available GPU at submission
        time, respecting the C2 → C3 → C1 preference order. Multiple
        requests run in parallel — one thread per request — so in-flight
        token counts (tok_e) and transfer queue depths (read_q_e) drive
        real-time load balancing across the pool.

        Args:
            requests:  list of (q, k, v) tuples; each (B, H, T, d)
            is_causal: apply causal mask to all requests

        Returns:
            list of output tensors in the same order as requests
        """
        if not requests:
            return []

        outputs = [None] * len(requests)

        def _process_one(idx: int) -> Tuple[int, Tensor]:
            q, k, v = requests[idx]
            result  = self.forward(q, k, v, is_causal=is_causal)
            return idx, result

        # Submit all requests; ThreadPoolExecutor handles concurrency
        n_workers = min(len(requests), max(self.n, 1))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_process_one, i): i
                       for i in range(len(requests))}
            for future in as_completed(futures):
                idx, result = future.result()
                outputs[idx] = result

        return outputs

    def load_summary(self) -> str:
        """Current per-GPU load: tokens in flight, transfer queue, sequences served."""
        with self._lock:
            parts = [
                f"GPU{i}: tok={self._tok[i]} rq={self._read_q[i]} seq={self._seq[i]}"
                for i in range(self.n)
            ]
        return " | ".join(parts)

    def __repr__(self) -> str:
        names  = all_gpu_names()
        gpu_str = ", ".join(
            f"[{i}] {names[i] if i < len(names) else 'unknown'}"
            for i in range(self.n)
        )
        return (
            f"MultiGPUAttentionPool("
            f"n={self.n}, α={self.alpha}, β={self.beta}, gpus={gpu_str})"
        )
