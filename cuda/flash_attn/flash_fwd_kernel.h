/*
 * flash_fwd_kernel.h — FlashAttention-2 forward kernel using WMMA.
 *
 * One CTA handles BLOCK_M rows of Q against ALL of K/V for one (batch, head).
 * Uses nvcuda::wmma for the two matmuls (Q@K^T and P@V).
 *
 * Algorithm (same as your Triton kernel):
 *   1. Load Q tile (BLOCK_M × head_dim) into shared memory
 *   2. For each KV tile of BLOCK_N rows:
 *      a. Load K tile (BLOCK_N × head_dim) into shared memory
 *      b. Compute S = Q @ K^T via WMMA (BLOCK_M × BLOCK_N, fp32 accum)
 *      c. Apply causal mask
 *      d. Online softmax: update m, l, rescale O
 *      e. Load V tile (BLOCK_N × head_dim) into shared memory
 *      f. Compute O += P @ V via WMMA (BLOCK_M × head_dim, fp32 accum)
 *   3. Normalize O = O / l
 *   4. Write O to global memory
 *
 * Implementation phases:
 *
 *   PHASE 1 — Basic correctness (synchronous loads, smem softmax):
 *     [ ] Shared memory: allocate Q, K, V, scores tiles
 *     [ ] Load Q into shared memory (cooperative, synchronous)
 *     [ ] KV loop: load K, WMMA Q@K^T, store scores to smem
 *     [ ] Softmax via shared memory (store scores, read rows, compute)
 *     [ ] Convert P to fp16 in smem, WMMA P@V, accumulate O
 *     [ ] Rescale O accumulator when max changes
 *     [ ] Final O/l normalization, write to global
 *
 *   PHASE 2 — Async memory pipeline:
 *     [ ] Replace synchronous loads with cp.async (cp_async_16B)
 *     [ ] cp_async_commit + cp_async_wait around loads
 *     [ ] Double buffer K and V tiles in shared memory
 *     [ ] Prefetch next K tile while computing current tile
 *
 *   PHASE 3 — Performance:
 *     [ ] Causal early exit (skip KV tiles above diagonal)
 *     [ ] Warp-shuffle softmax (avoid smem round-trip for scores)
 *     [ ] Bank conflict reduction via smem padding or swizzling
 *     [ ] Tune BLOCK_M/BLOCK_N for occupancy
 *
 * WMMA matmul tiling:
 *   To compute S[BLOCK_M × BLOCK_N] = Q[BLOCK_M × D] @ K^T[D × BLOCK_N]:
 *   - Outer loops over WMMA tiles: i ∈ [0, BLOCK_M/16), j ∈ [0, BLOCK_N/16)
 *   - Inner K-loop: k ∈ [0, D/16)
 *   - Each iteration: load 16×16 fragment of Q and K, mma_sync, accumulate
 *
 *   Same pattern for O[BLOCK_M × D] += P[BLOCK_M × BLOCK_N] @ V[BLOCK_N × D]:
 *   - Outer loops: i ∈ [0, BLOCK_M/16), j ∈ [0, D/16)
 *   - Inner K-loop: k ∈ [0, BLOCK_N/16)
 *
 * Grid launch:
 *   grid.x = ceil(seqlen_q / BLOCK_M)
 *   grid.y = batch_size * num_heads
 *   block  = kNThreads (128 for 4 warps)
 *
 * Shared memory (hdim128, BLOCK_M=128, BLOCK_N=64, no double buffer):
 *   Q:      128 × 128 × 2 = 32 KB
 *   K:       64 × 128 × 2 = 16 KB
 *   V:       64 × 128 × 2 = 16 KB
 *   Scores: 128 ×  64 × 4 = 32 KB (fp32, for softmax)
 *   Total: ~96 KB → need cudaFuncSetAttribute for extended smem
 *
 *   For hdim64 it's half the Q/K/V sizes → ~64 KB.
 *   Phase 1 can reuse K's smem for V (they're not needed simultaneously)
 *   to save space: Q + K_or_V + Scores = 16 + 8 + 32 = 56 KB for hdim64.
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "flash.h"
#include "kernel_traits.h"
#include "softmax.h"
#include "utils.h"

using namespace nvcuda;

template <typename Traits, bool Is_causal>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
  // Compile-time constants
  constexpr int BLOCK_M = Traits::kBlockM;
  constexpr int BLOCK_N = Traits::kBlockN;
  constexpr int HEAD_DIM = Traits::kHeadDim;
  constexpr int NTHREADS = Traits::kNThreads;

  constexpr int Q_STRIDE = HEAD_DIM + 8;    // halves, +8 (16 bytes)
  constexpr int KV_STRIDE = HEAD_DIM + 8;   // halves
  constexpr int SCORE_STRIDE = BLOCK_N + 4; // FLOATS, +4 (16 bytes)
  constexpr int P_STRIDE = BLOCK_N + 8;     // halves
  // O_STRIDE eliminated: O accumulator lives in WMMA registers, not smem

  // WMMA matmul constants
  constexpr int NWARPS = NTHREADS / 32;
  constexpr int P_TILES_PER_WARP = BLOCK_M * BLOCK_N / (16 * 16 * NWARPS);
  constexpr int P_TILES_PER_ROW = BLOCK_N / 16;
  constexpr int O_TILES_PER_WARP = BLOCK_M * HEAD_DIM / (16 * 16 * NWARPS);
  constexpr int O_TILES_PER_ROW = HEAD_DIM / 16;

  // v1 assumption: each thread owns exactly 1 row of Q/scores/P/O.
  // Many indexing simplifications below depend on this — break it loudly if
  // violated.
  static_assert(BLOCK_M == NTHREADS, "v1: 1 thread per row");

  // O is kept in WMMA fragment registers (not smem). Only scores need this buffer.
  constexpr int SCORES_FLOATS = BLOCK_M * SCORE_STRIDE;

  // Thread / block indices
  const int m_block = blockIdx.x; // which chunk of Q rows
  const int bh_idx = blockIdx.y;  // which (batch, head)
  const int batch_idx = bh_idx / params.num_heads;
  const int head_idx = bh_idx % params.num_heads;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;

  // For GQA/MQA: kv_head = head_idx / (num_heads / num_heads_k)
  // const int kv_head_idx = head_idx / (params.num_heads / params.num_heads_k);

  // ========================================================================
  // Step 1: Shared memory setup
  // ========================================================================
  extern __shared__ char smem_raw[];

  half *smem_q = reinterpret_cast<half *>(smem_raw);
  half *smem_kv = smem_q + BLOCK_M * Q_STRIDE;
  // On SM80+ we double-buffer KV (2 slots); on SM75 only 1 slot.
  // scores/O/P must come AFTER both KV slots to avoid aliasing smem_kv1.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  float *smem_scores = reinterpret_cast<float *>(smem_kv + 2 * BLOCK_N * KV_STRIDE);
#else
  float *smem_scores = reinterpret_cast<float *>(smem_kv + BLOCK_N * KV_STRIDE);
#endif
  // smem_o is eliminated — O lives in WMMA fragment registers.
  half *smem_p = reinterpret_cast<half *>(smem_scores + SCORES_FLOATS);
  //
  // Partition shared memory:
  //   half *smem_q      = (half *)smem_raw;                         //
  //   BLOCK_M × HEAD_DIM half *smem_k      = smem_q + BLOCK_M * HEAD_DIM;
  //   // BLOCK_N × HEAD_DIM half *smem_v      = smem_k + BLOCK_N *
  //   HEAD_DIM;              // BLOCK_N × HEAD_DIM float *smem_scores =
  //   (float *)(smem_v + BLOCK_N * HEAD_DIM);  // BLOCK_M × BLOCK_N
  //
  // Or reuse K/V space (they're not live simultaneously):
  //   half *smem_kv     = smem_q + BLOCK_M * HEAD_DIM;              //
  //   BLOCK_N × HEAD_DIM float *smem_scores = (float *)(smem_kv + BLOCK_N *
  //   HEAD_DIM);
  //   // BLOCK_M × BLOCK_N

  // Use log2e-prescaled scale so the hot softmax loop can use exp2f (one PTX
  // instruction) instead of expf (multi-instruction approximation).
  // params.scale_softmax_log2 = (1/sqrt(HEAD_DIM)) * log2(e)
  float scale_log2 = params.scale_softmax_log2;   // for exp2f paths

  const half *q_ptr = reinterpret_cast<const half *>(params.q_ptr) +
                      BLOCK_M * m_block * params.q_row_stride +
                      batch_idx * params.q_batch_stride +
                      head_idx * params.q_head_stride;
  const half *k_ptr = reinterpret_cast<const half *>(params.k_ptr) +
                      batch_idx * params.k_batch_stride +
                      head_idx * params.k_head_stride;
  const half *v_ptr = reinterpret_cast<const half *>(params.v_ptr) +
                      batch_idx * params.v_batch_stride +
                      head_idx * params.v_head_stride;
  half *o_ptr = reinterpret_cast<half *>(params.o_ptr) +
                BLOCK_M * m_block * params.o_row_stride +
                batch_idx * params.o_batch_stride +
                head_idx * params.o_head_stride;

  // ========================================================================
  // Step 3: Load Q tile into shared memory (one-time, reused across KV tiles)
  // ========================================================================

  int q_rows_valid = min(BLOCK_M, params.seqlen_q - m_block * BLOCK_M);
  load_tile_sync<BLOCK_M, HEAD_DIM, Q_STRIDE, NTHREADS>(
      smem_q, q_ptr, params.q_row_stride, q_rows_valid, tid);

  // On SM80+: set up the second KV buffer pointer for double-buffering.
  // smem_kv   = slot 0 (current K/V)
  // smem_kv1  = slot 1 (prefetched K)
  // Both slots are BLOCK_N × KV_STRIDE halves.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  half *smem_kv1 = smem_kv + BLOCK_N * KV_STRIDE;  // second KV buffer
#endif

  // ========================================================================
  // Step 4: Initialize O accumulator and softmax state
  // ========================================================================
  // O is kept as persistent WMMA accumulator fragments in registers.
  // This eliminates the per-tile smem_o round-trip:
  //   old: store_matrix_sync → smem_o → __syncthreads → read back into o_acc
  //   new: apply correction directly on o_frags[i].x[e] (warp shuffles for
  //        per-row correction), then mma_sync accumulates into o_frags.
  //
  // WMMA m16n16k16 accumulator layout (fp32, 8 elements per thread, lane t):
  //   element e: row = (t/4) + 8*(e/4),  col = (t%4)*2 + (e%2) + 8*((e/2)%2)
  //
  // softmax state: 1 float per row per thread (BLOCK_M == NTHREADS → 1 row).
  const int laneId = tid % 32;

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> o_frags[O_TILES_PER_WARP];
#pragma unroll
  for (int i = 0; i < O_TILES_PER_WARP; i++) wmma::fill_fragment(o_frags[i], 0.0f);

  float m_state = -INFINITY;
  float l_state = 0.0f;

  // ========================================================================
  // Step 5: KV tile loop
  // ========================================================================
  // Causal bound: only process KV tiles up to and including the diagonal.
  int kv_end = (Is_causal) ? min((m_block + 1) * BLOCK_M, params.seqlen_k)
                           : params.seqlen_k;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // ------------------------------------------------------------------
  // SM80+ path: cp.async pipeline with K double-buffering
  //
  // Timeline per KV tile n:
  //   [prologue]  issue async copy of K0 into slot 0
  //   [loop]
  //     wait K_n (slot n%2) ready
  //     issue async copy of V_n into slot 0       (reuses slot 0 after K)
  //     compute S = Q @ K_n
  //     wait V_n ready
  //     if not last: issue async copy of K_{n+1} into slot (n+1)%2
  //     softmax(S) → P
  //     compute O += P @ V_n
  //
  // This hides HBM latency of K_{n+1} behind the P@V matmul of tile n.
  // ------------------------------------------------------------------

  // Prologue: issue K0 async copy into slot 0.
  if (kv_end > 0) {
    int kv_rows_valid = min(BLOCK_N, params.seqlen_k);
    load_tile_async<BLOCK_N, HEAD_DIM, KV_STRIDE, NTHREADS>(
        smem_kv, k_ptr, params.k_row_stride, kv_rows_valid, tid);
    asm volatile("cp.async.commit_group;");
  }

  for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_N) {
    int kv_rows_valid = min(BLOCK_N, params.seqlen_k - kv_start);

    // Select which KV smem slot holds the current K tile.
    int cur_buf  = (kv_start / BLOCK_N) & 1;
    half *cur_kv = cur_buf ? smem_kv1 : smem_kv;

    // Wait for K_n to arrive (committed in prologue or previous iteration).
    // The WMMA prologue pre-issues only K[0]; subsequent K tiles are issued at
    // the end of the previous iteration (before P@V), so by the time we reach
    // here K[n] has had the entire P@V GEMM to complete — wait_group 0 is
    // safe and ensures the smem slot is fully written before the next GEMM.
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();

    // NOTE: Q@K WMMA happens below (reads cur_kv for K data).
    // We must NOT issue V into cur_kv until after Q@K is complete.

#else
  // ------------------------------------------------------------------
  // SM75 / fallback path: synchronous loads, no double buffering
  // ------------------------------------------------------------------
  for (int kv_start = 0; kv_start < kv_end; kv_start += BLOCK_N) {
    int kv_rows_valid = min(BLOCK_N, params.seqlen_k - kv_start);
    half *cur_kv = smem_kv;
    load_tile_sync<BLOCK_N, HEAD_DIM, KV_STRIDE, NTHREADS>(
        cur_kv, k_ptr + kv_start * params.k_row_stride, params.k_row_stride,
        kv_rows_valid, tid);
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // Issue V_n async load NOW, before the Q@K WMMA loop.
    // cur_kv holds K_n which WMMA is about to READ (not write), so issuing
    // an async store into cur_kv (which cp.async writes from the L2 side)
    // is safe only AFTER the WMMA reads finish.  We therefore use a
    // *separate* staging approach: commit V into cur_kv after the sync that
    // follows the WMMA loop — but we can at least overlap the DMA latency
    // with the WMMA compute by issuing the prefetch of K_{n+1} here instead
    // (moved up from after softmax), freeing the post-softmax window for V wait.
    //
    // Revised pipeline per tile n (SM80+):
    //   [top of loop]  wait K_n  (issued end-of-prev or prologue)
    //   ** issue K_{n+1} prefetch NOW (into alt slot) **
    //   Q @ K_n WMMA
    //   __syncthreads()
    //   issue V_n async (into cur_kv slot, K_n no longer needed)
    //   commit V_n group
    //   softmax (hides V_n latency)
    //   wait V_n (≤1 group left: V_n; K_{n+1} may still be in flight → wait_group 1)
    //   P @ V_n WMMA
    //
    // This way K_{n+1} overlaps with both Q@K WMMA *and* softmax.
    int next_kv_start_early = kv_start + BLOCK_N;
    if (next_kv_start_early < kv_end) {
      int next_buf_early  = 1 - cur_buf;
      half *next_kv_early = next_buf_early ? smem_kv1 : smem_kv;
      int next_rows_early = min(BLOCK_N, params.seqlen_k - next_kv_start_early);
      load_tile_async<BLOCK_N, HEAD_DIM, KV_STRIDE, NTHREADS>(
          next_kv_early,
          k_ptr + next_kv_start_early * params.k_row_stride,
          params.k_row_stride, next_rows_early, tid);
      asm volatile("cp.async.commit_group;");
      // Outstanding: 1 group (K_{n+1}).
    }
#endif

    // Q @ K^T using WMMA
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> q_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> k_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> s_frag;

    for (int i = 0; i < P_TILES_PER_WARP; i++) {
      int tile_idx = warp_id * P_TILES_PER_WARP + i;
      int wi = tile_idx / P_TILES_PER_ROW * 16;
      int wj = tile_idx % P_TILES_PER_ROW * 16;

      wmma::fill_fragment(s_frag, 0.0f);
      for (int wk = 0; wk < HEAD_DIM; wk += 16) {
        wmma::load_matrix_sync(q_frag, &smem_q[wi * Q_STRIDE + wk], Q_STRIDE);
        wmma::load_matrix_sync(k_frag, &cur_kv[wj * KV_STRIDE + wk], KV_STRIDE);
        wmma::mma_sync(s_frag, q_frag, k_frag, s_frag);
      }
      wmma::store_matrix_sync(&smem_scores[wi * SCORE_STRIDE + wj], s_frag,
                              SCORE_STRIDE, wmma::mem_row_major);
    }

    // Q@K complete; all warps wrote scores to smem. K_n is no longer needed.
    __syncthreads();

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // K_n is consumed — safe to overwrite cur_kv with V_n.
    load_tile_async<BLOCK_N, HEAD_DIM, KV_STRIDE, NTHREADS>(
        cur_kv, v_ptr + kv_start * params.v_row_stride, params.v_row_stride,
        kv_rows_valid, tid);
    asm volatile("cp.async.commit_group;");
    // Outstanding: K_{n+1} (if not last tile) + V_n = up to 2 groups.
#endif

    int score_base = tid * SCORE_STRIDE;
    int p_base = tid * P_STRIDE;

    // Apply causal mask: set scores for key positions > query position to -inf.
    // Each thread owns exactly one Q row (tid), so no extra sync needed here.
    if (Is_causal) {
      const int q_row = m_block * BLOCK_M + tid;
      for (int j = 0; j < BLOCK_N; j++) {
        if (kv_start + j > q_row) {
          smem_scores[score_base + j] = -INFINITY;
        }
      }
    }

    // Online softmax with exp2f (single PTX instruction vs multi-step expf).
    // All exponents are in log2 base: exp2(x * scale_log2 - adj_max).
    float max_old = m_state;
    float max_new = max_old;
#pragma unroll
    for (int j = 0; j < BLOCK_N; j++) {
      // Pre-scale once; max reduction in same pass.
      float s = smem_scores[score_base + j] * scale_log2;
      smem_scores[score_base + j] = s;
      max_new = fmaxf(max_new, s);
    }
    m_state = max_new;

    // Rescale O accumulator and running sum by exp2(max_old - max_new).
    float correction = exp2f(max_old - max_new);
    float l_sum = 0.f;
#pragma unroll
    for (int j = 0; j < BLOCK_N; j++) {
      float score = exp2f(smem_scores[score_base + j] - max_new);
      l_sum += score;
      smem_p[p_base + j] = __float2half(score);
    }
    l_state = l_state * correction + l_sum;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // At this point outstanding groups are:
    //   - K_{n+1} (if not last tile, issued before Q@K)
    //   - V_n     (issued after Q@K sync, just above)
    // We need V_n in smem before P@V; K_{n+1} can still be in flight.
    // If last tile: only V_n is outstanding → wait_group 0.
    // Otherwise:   V_n + K_{n+1} outstanding → wait_group 1 (wait all but 1).
    if (next_kv_start_early < kv_end) {
      asm volatile("cp.async.wait_group 1;");
    } else {
      asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();
#else
    // SM75 fallback: synchronous V load.
    __syncthreads();
    load_tile_sync<BLOCK_N, HEAD_DIM, KV_STRIDE, NTHREADS>(
        cur_kv, v_ptr + kv_start * params.v_row_stride, params.v_row_stride,
        kv_rows_valid, tid);
#endif

    // Apply correction to persistent o_frags before accumulating P@V.
    // WMMA m16n16k16 accumulator layout: thread laneId t, element e:
    //   row = (t/4) + 8*(e/4),   col = (t%4)*2 + (e%2) + 8*((e/2)%2)
    // Each tile (wi, wj) has:
    //   elements 0-3 → row wi + laneId/4     (owned by laneId = wi_local + laneId/4)
    //   elements 4-7 → row wi + laneId/4 + 8 (owned by laneId = wi_local + laneId/4 + 8)
    // wi_local = wi - warp_id*32 ∈ {0,16} for all configs (invariant for power-of-2 tiling).
    {
      const int warp_base = warp_id * 32;
#pragma unroll
      for (int i = 0; i < O_TILES_PER_WARP; i++) {
        int tidx = warp_id * O_TILES_PER_WARP + i;
        int wi_local = (tidx / O_TILES_PER_ROW) * 16 - warp_base;
        float corr0 = __shfl_sync(0xffffffff, correction, wi_local + laneId / 4);
        float corr8 = __shfl_sync(0xffffffff, correction, wi_local + laneId / 4 + 8);
#pragma unroll
        for (int e = 0; e < 4; e++) o_frags[i].x[e] *= corr0;
#pragma unroll
        for (int e = 4; e < 8; e++) o_frags[i].x[e] *= corr8;
      }
    }

    // P @ V — accumulate directly into o_frags (D = A*B + D, no smem_o needed).
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> p_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> v_frag;

#pragma unroll
    for (int i = 0; i < O_TILES_PER_WARP; i++) {
      int tile_idx = warp_id * O_TILES_PER_WARP + i;
      int wi = (tile_idx / O_TILES_PER_ROW) * 16;
      int wj = (tile_idx % O_TILES_PER_ROW) * 16;
      for (int wk = 0; wk < BLOCK_N; wk += 16) {
        wmma::load_matrix_sync(p_frag, &smem_p[wi * P_STRIDE + wk], P_STRIDE);
        wmma::load_matrix_sync(v_frag, &cur_kv[wk * KV_STRIDE + wj], KV_STRIDE);
        wmma::mma_sync(o_frags[i], p_frag, v_frag, o_frags[i]);
      }
    }

    // Ensure all warps finish reading smem_p and cur_kv (V) before next tile
    // writes to smem_p (softmax) and cur_kv (next K). No smem_o write needed.
    __syncthreads();
  }

  // ========================================================================
  // Epilogue: normalize o_frags by l_state, scatter to smem_q (fp16), write gmem.
  // ========================================================================
  // o_frags elements are written to smem_q[global_row * Q_STRIDE + global_col]
  // using the WMMA accumulator thread-element-to-(row,col) mapping.
  // smem_q is reused as output staging (Q is no longer needed).
  static_assert(HEAD_DIM % 8 == 0, "HEAD_DIM must be multiple of 8 for 128-bit stores");
  {
    const int warp_base = warp_id * 32;
    half *smem_out = smem_q;
#pragma unroll
    for (int i = 0; i < O_TILES_PER_WARP; i++) {
      int tidx    = warp_id * O_TILES_PER_WARP + i;
      int wi      = (tidx / O_TILES_PER_ROW) * 16;
      int wj      = (tidx % O_TILES_PER_ROW) * 16;
      int wi_local = wi - warp_base;
      // Fetch per-row inv_l via warp shuffle (l_state[lane] == l for row 'lane').
      float inv_l0 = 1.f / __shfl_sync(0xffffffff, l_state, wi_local + laneId / 4);
      float inv_l8 = 1.f / __shfl_sync(0xffffffff, l_state, wi_local + laneId / 4 + 8);
#pragma unroll
      for (int e = 0; e < 8; e++) {
        // WMMA accumulator layout: row = (laneId/4) + 8*(e/4),
        //                          col = (laneId%4)*2 + (e%2) + 8*((e/2)%2)
        int row = wi + (laneId / 4) + 8 * (e / 4);
        int col = wj + (laneId % 4) * 2 + (e % 2) + 8 * ((e / 2) % 2);
        float inv_l = (e < 4) ? inv_l0 : inv_l8;
        smem_out[row * Q_STRIDE + col] = __float2half(o_frags[i].x[e] * inv_l);
      }
    }
  }
  __syncthreads();

  // Write output tile to global memory using 128-bit (8 × half) stores.
  // smem_q is now the normalized fp16 output staging buffer.
  constexpr int HALVES_PER_STORE = 8;
  constexpr int N_STORES = BLOCK_M * HEAD_DIM / HALVES_PER_STORE;
  for (int idx = tid; idx < N_STORES; idx += NTHREADS) {
    int elem = idx * HALVES_PER_STORE;
    int row  = elem / HEAD_DIM;
    int col  = elem % HEAD_DIM;
    uint4 chunk = *reinterpret_cast<const uint4 *>(&smem_q[row * Q_STRIDE + col]);
    *reinterpret_cast<uint4 *>(&o_ptr[row * params.o_row_stride + col]) = chunk;
  }
}
