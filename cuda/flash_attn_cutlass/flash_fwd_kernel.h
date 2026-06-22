#pragma once

#include "flash.h"
#include "kernel_traits.cuh"
#include "softmax.cuh"
#include "utils.cuh"
#include <cstdio>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

namespace FLASH {
using namespace cute;

template <typename Traits, bool Is_causal>
__global__ void flash_fwd_kernel(Flash_fwd_params params) {
  const int tid = threadIdx.x;

  constexpr int kBlockM = Traits::kBlockM;
  constexpr int kBlockN = Traits::kBlockN;
  constexpr int kHeadDim = Traits::kHeadDim;
  constexpr int kNWarps = Traits::kNThreads / 32;

  const int m_block = blockIdx.x; // which chunk of Q rows
  const int bh_idx = blockIdx.y;  // which (batch, head)
  const int batch_idx = bh_idx / params.num_heads;
  const int head_idx = bh_idx % params.num_heads;

  // auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);
  extern __shared__ char smem[];

  // gmem Q + bh
  Tensor mQ = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cute::half_t *>(params.q_ptr) +
                    batch_idx * params.q_batch_stride +
                    head_idx * params.q_head_stride),
      make_shape(params.seqlen_q, params.head_dim),
      make_stride(params.q_row_stride, _1{}));
  Tensor gQ = local_tile(mQ, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                         make_coord(m_block, 0));
  Tensor mK = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cute::half_t *>(params.k_ptr) +
                    batch_idx * params.k_batch_stride +
                    head_idx * params.k_head_stride),
      make_shape(params.seqlen_k, params.head_dim),
      make_stride(params.k_row_stride, _1{}));
  // Allow K dim traversal
  Tensor gK = local_tile(mK, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
                         make_coord(_, 0));
  Tensor mV = make_tensor(
      make_gmem_ptr(reinterpret_cast<const cute::half_t *>(params.v_ptr) +
                    batch_idx * params.v_batch_stride +
                    head_idx * params.v_head_stride),
      make_shape(params.seqlen_k, params.head_dim),
      make_stride(params.v_row_stride, _1{}));
  Tensor gV = local_tile(mV, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}),
                         make_coord(_, 0));

  // smem defs
  // TODO: fancy cutlass version
  //   Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()),
  //                         sA_layout); // (BLK_M,BLK_K,PIPE)
  // Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()),
  //                         sB_layout); // (BLK_N,BLK_K,PIPE)
  // TODO: Q in register; QK temporal smem sharing, validate on block sizes
  Tensor sQ = make_tensor(reinterpret_cast<cute::half_t *>(smem),
                          typename Traits::SmemLayoutQ{});
  Tensor sK =
      make_tensor(sQ.data() + size(sQ), typename Traits::SmemLayoutKV{});
  Tensor sV =
      make_tensor(sK.data() + size(sK), typename Traits::SmemLayoutKV{});
  Tensor sVt = make_tensor(sV.data(), typename Traits::SmemLayoutVt{});
  Tensor sVtNoSwizzle =
      make_tensor(sV.data(), typename Traits::SmemLayoutVtNoSwizzle{});

  typename Traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
  auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tid);

  Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
  Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
  Tensor tKgK =
      gmem_thr_copy_QKV.partition_S(gK); // (KCPY, KCPY_N, KCPY_K, nblocksN)
  Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
  Tensor tVgV =
      gmem_thr_copy_QKV.partition_S(gV); // (VCPY, VCPY_N, VCPY_K, nblocksN)
  Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

  typename Traits::TiledMma tiled_mma;
  auto thr_mma = tiled_mma.get_thread_slice(tid);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
  Tensor tSrK = thr_mma.partition_fragment_B(sK);
  Tensor tOrV = thr_mma.partition_fragment_B(sVtNoSwizzle);

  Tensor acc_o = partition_fragment_C(
      tiled_mma, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}));

  // smem to R copy
  auto smem_tiled_copy_Q =
      make_tiled_copy_A(typename Traits::SmemCopyAtom{}, tiled_mma);
  auto smem_tiled_copy_K =
      make_tiled_copy_B(typename Traits::SmemCopyAtom{}, tiled_mma);
  // probably need to tranpose here
  auto smem_tiled_copy_V =
      make_tiled_copy_B(typename Traits::SmemCopyAtomTransposed{}, tiled_mma);

  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid);
  auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tid);

  // partition smem->register copy
  auto tSsQ = smem_thr_copy_Q.partition_S(sQ);
  auto tSsK = smem_thr_copy_K.partition_S(sK);
  auto tOsVt = smem_thr_copy_V.partition_S(sVt);

  // copy Q
  // TODO: check if this does load in one go
  cute::copy(gmem_tiled_copy_QKV, tQgQ, tQsQ);
  // issue first K copy tile "0"
  cute::copy(gmem_tiled_copy_QKV, tKgK(_, _, _, _0{}), tKsK);
  cute::cp_async_fence();

  clear(acc_o);
  // initialize softmax, acc_o: (MMA, MMA_M, MMA_HEAD_DIM)
  // rows is 2*MMA_M dim, 2 rows per thread for each MMA tile
  FLASH::Softmax<2 * size<1>(acc_o)> softmax;

  // For causal attention: CTA m_block only needs KV tiles where kv_start <
  // (m_block+1)*kBlockM. Tiles above the diagonal are all -inf → zero weight.
  const int nBlocksN = Is_causal
      ? cute::ceil_div(min((m_block + 1) * kBlockM, (int)params.seqlen_k), kBlockN)
      : cute::ceil_div(params.seqlen_k, kBlockN);

#pragma unroll
  for (int nblock = 0; nblock < nBlocksN; nblock++) {
    Tensor acc_s = partition_fragment_C(
        tiled_mma, make_shape(Int<kBlockM>{}, Int<kBlockN>{}));
    clear(acc_s);
    // wait on K
    cute::cp_async_wait<0>();
    __syncthreads();
    // issue V copy
    cute::copy(gmem_tiled_copy_QKV, tVgV(_, _, _, nblock), tVsV);
    cute::cp_async_fence();

    // 1. gemm S=Q@K.T
    FLASH::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q,
                smem_tiled_copy_K, smem_thr_copy_Q, smem_thr_copy_K);
    // wait for V
    cute::cp_async_wait<0>();
    __syncthreads();

    // next K block prefetch (only if there is one)
    if (nblock < nBlocksN - 1) {
      cute::copy(gmem_tiled_copy_QKV, tKgK(_, _, _, nblock + 1), tKsK);
      cute::cp_async_fence();
    }

    // 2. Causal mask: zero out scores for positions j > i.
    // The partial tile at the diagonal needs masking; all earlier tiles are fully below diagonal.
    if constexpr (Is_causal) {
      // q row indices for this CTA: [m_block*kBlockM, (m_block+1)*kBlockM)
      // kv col indices for this tile: [nblock*kBlockN, (nblock+1)*kBlockN)
      // Mask if q_row < kv_col, i.e. j > i.
      // Only needed for the tile straddling the diagonal.
      // A tile needs masking if it's not fully below the diagonal.
      // "Fully below" means max kv_col in tile < min q_row:
      //   (nblock+1)*kBlockN <= m_block*kBlockM
      // So mask is needed when (nblock+1)*kBlockN > m_block*kBlockM.
      // NOTE: the last-tile-only optimization is only safe when kBlockN >= kBlockM.
      // For kBlockM > kBlockN (e.g. 128 vs 64), m_block=0 has multiple diagonal tiles.
      const bool is_diagonal_tile = ((nblock + 1) * kBlockN > m_block * kBlockM);
      if (is_diagonal_tile) {
        // acc_s layout: (MMA, MMA_M, MMA_N) — apply mask element-wise.
        // Each thread owns a subset of (kBlockM × kBlockN) scores.
        // We use the MMA coordinate helpers to find (row, col) per element.
        auto cS = tiled_mma.get_slice(tid).partition_C(
            make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
        CUTE_UNROLL
        for (int i = 0; i < size(acc_s); i++) {
          auto coord = cS(i);  // (m_coord, n_coord)
          int q_row = m_block * kBlockM + get<0>(coord);
          int kv_col = nblock * kBlockN + get<1>(coord);
          if (kv_col > q_row) {
            acc_s(i) = -INFINITY;
          }
        }
      }
    }

    // 3. P=softmax(S)
    if (nblock == 0) {
      softmax.template softmax_rescale_o</*Is_first*/ true>(
          acc_s, acc_o, params.scale_softmax_log2);
    } else {
      softmax.template softmax_rescale_o</*Is_first*/ false>(
          acc_s, acc_o, params.scale_softmax_log2);
    }

    Tensor acc_s_fp16 = FLASH::convert_type<cute::half_t>(acc_s);
    // reshape to A fragment for next matmul
    Tensor tOrP =
        make_tensor(acc_s_fp16.data(),
                    FLASH::convert_c_frag_to_a_frag(acc_s_fp16.layout()));

    // o = P @ V
    FLASH::gemm_rs(acc_o, tOrP, tOrV, tOsVt, tiled_mma, smem_tiled_copy_V,
                   smem_thr_copy_V);
  }

  // final o scaling
  softmax.normalize_softmax(acc_o);

  // convert o from fp32 to fp16
  Tensor o_fp16 = FLASH::convert_type<cute::half_t>(acc_o);

  // stage O to smem, reuse Q
  Tensor sO = make_tensor(sQ.data(), typename Traits::SmemLayoutO{});
  auto smem_tiled_copy_O =
      make_tiled_copy_C(typename Traits::SmemCopyAtomO{}, tiled_mma);
  auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tid);
  auto trO = smem_thr_copy_O.retile_S(o_fp16);
  auto tsO = smem_thr_copy_O.partition_D(sO);
  cute::copy(smem_tiled_copy_O, trO, tsO);

  // gmem O, same as Q
  typename Traits::GmemTiledCopyO gmem_tiled_copy_O;
  Tensor mO =
      make_tensor(make_gmem_ptr(reinterpret_cast<cute::half_t *>(params.o_ptr) +
                                batch_idx * params.o_batch_stride +
                                head_idx * params.o_head_stride),
                  make_shape(params.seqlen_q, params.head_dim),
                  make_stride(params.q_row_stride, _1{}));
  Tensor gO = local_tile(mO, make_shape(Int<kBlockM>{}, Int<kHeadDim>{}),
                         make_coord(m_block, 0));

  auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tid);
  Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
  Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

  // register buffer
  Tensor tOrO = make_fragment_like(tOgO);

  // sync after r->smem copy
  // previous section no r/wb conflicts
  __syncthreads();

  // smem->registers
  cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
  // registers->gmem
  cute::copy(gmem_tiled_copy_O, tOrO, tOgO);
}

} // namespace FLASH
