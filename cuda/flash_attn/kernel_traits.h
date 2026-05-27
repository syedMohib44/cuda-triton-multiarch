/*
 * kernel_traits.h — Compile-time kernel configuration.
 *
 * With WMMA, this is much simpler than the CuTe version — just block sizes,
 * thread counts, and shared memory math. No layout algebra needed.
 *
 * Block size choices (from Tri Dao, A100/SM80):
 *   hdim64:  BLOCK_M=128, BLOCK_N=64,  4 warps → 128 threads
 *   hdim128: BLOCK_M=128, BLOCK_N=64,  4 warps → 128 threads
 *
 * WMMA fragment size is fixed at 16×16×16 (fp16 inputs, fp32 accumulator).
 * We tile the BLOCK_M×BLOCK_N matmul into (BLOCK_M/16) × (BLOCK_N/16) WMMA tiles,
 * each accumulating over head_dim in chunks of 16.
 */

#pragma once

template <int kHeadDim_, int kBlockM_, int kBlockN_, int kNWarps_>
struct Flash_fwd_kernel_traits {
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockM = kBlockM_;     // Q rows per CTA
    static constexpr int kBlockN = kBlockN_;     // KV columns per CTA (inner loop tile)
    static constexpr int kNWarps = kNWarps_;
    static constexpr int kNThreads = kNWarps * 32;

    // WMMA tile dimensions (fixed by the API)
    static constexpr int WMMA_M = 16;
    static constexpr int WMMA_N = 16;
    static constexpr int WMMA_K = 16;

    // Number of WMMA tiles needed to cover each block dimension
    static constexpr int kWmmaTilesM = kBlockM / WMMA_M;  // e.g. 128/16 = 8
    static constexpr int kWmmaTilesN = kBlockN / WMMA_N;  // e.g. 64/16 = 4
    static constexpr int kWmmaTilesK = kHeadDim / WMMA_K; // e.g. 64/16 = 4

    // Padded strides (must match flash_fwd_kernel.h) to break smem bank-conflict
    // alignment. fp16 buffers padded by 8 halves (16 bytes); fp32 buffers padded
    // by 4 floats (16 bytes). Both maintain the 16-byte alignment required by
    // ldmatrix / store_matrix_sync.
    static constexpr int kQStride     = kHeadDim + 8;   // halves
    static constexpr int kKVStride    = kHeadDim + 8;   // halves
    static constexpr int kScoreStride = kBlockN  + 4;   // floats
    static constexpr int kPStride     = kBlockN  + 8;   // halves
    static constexpr int kOStride     = kHeadDim + 4;   // floats

    // Shared memory sizes (in bytes), accounting for padded strides.
    // Q tile:    BLOCK_M × Q_STRIDE × sizeof(half)
    // K/V tile:  BLOCK_N × KV_STRIDE × sizeof(half) — reused (K then V, not alive together)
    // scores/O:  BLOCK_M × max(SCORE_STRIDE, O_STRIDE) × sizeof(float) — same buffer, two roles
    // P tile:    BLOCK_M × P_STRIDE × sizeof(half)
    static constexpr int kSmemQ      = kBlockM * kQStride     * sizeof(half);
    static constexpr int kSmemKV     = kBlockN * kKVStride    * sizeof(half);
    static constexpr int kSmemScores = kBlockM * kScoreStride * sizeof(float);
    static constexpr int kSmemO      = kBlockM * kOStride     * sizeof(float);
    static constexpr int kSmemP      = kBlockM * kPStride     * sizeof(half);

    // scores and O alias the same buffer, so we need max of the two
    static constexpr int kSmemScoresO = kSmemScores > kSmemO ? kSmemScores : kSmemO;

    // Without double buffering (SM75 / Phase 1 fallback):
    static constexpr int kSmemSizeNoPipeline = kSmemQ + kSmemKV + kSmemScoresO + kSmemP;

    // With double buffering (SM80+ cp.async pipeline):
    // Two KV slots: current tile being consumed + next tile being prefetched.
    static constexpr int kSmemSizePipeline = kSmemQ + 2 * kSmemKV + kSmemScoresO + kSmemP;

    // kSmemSize used by the launch template — pick the right one at compile time.
    // The kernel selects which path to execute via __CUDA_ARCH__.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static constexpr int kSmemSize = kSmemSizePipeline;
#else
    static constexpr int kSmemSize = kSmemSizeNoPipeline;
#endif

    // Register pressure check:
    // O accumulator: BLOCK_M × head_dim / kNThreads floats per thread
    // For hdim128, BLOCK_M=128, 128 threads: 128 × 128 / 128 = 128 fp32 regs
    // Plus WMMA fragments, softmax state, loop vars → ~180-200 total
    // A100 max: 65536 / 128 threads = 512 regs/thread — plenty of headroom

    // Smem totals with padding (for reference):
    //   hdim64:   18K + 9K  + 34K + 18K ≈  79 KB
    //   hdim128:  35K + 17K + 66K + 18K ≈ 136 KB  (still fits A100's 164 KB max)
};

// ---------------------------------------------------------------------------
// Concrete configs — per SM version
//
// smem budget per SM (no double buffer):
//   hdim64:  18K(Q) + 9K(KV) + 35K(scores) + 18K(P) ≈  80 KB
//   hdim128: 35K(Q) + 17K(KV)+ 66K(scores) + 18K(P) ≈ 136 KB
//
// SM75 (Turing, 64 KB max): use BLOCK_M=64, BLOCK_N=32
//   hdim64:  9K+5K+9K+5K  ≈ 28 KB  ✓
//   hdim128: 9K+9K+9K+5K  ≈ 32 KB  ✓  (BLOCK_M reduced to 64)
//
// SM80 (A100, 164 KB max): original sizing, fits easily
//   hdim64:  80 KB  ✓
//   hdim128: 136 KB ✓
//
// SM86/89 (RTX 30xx/40xx, 100 KB max):
//   hdim64:  BLOCK_M=128,BLOCK_N=64 → 80 KB  ✓ (same as SM80)
//   hdim128: BLOCK_M=128,BLOCK_N=32 → reduce BLOCK_N to cut scores/P by 2×
//            35K+9K+17K+9K ≈ 70 KB  ✓
// ---------------------------------------------------------------------------

// SM75 (Turing: T4, RTX 2080 Ti) — 64 KB smem, no cp.async
// WARPS=2 → NTHREADS=64 == BLOCK_M (required by static_assert in flash_fwd_kernel.h)
using Traits_hdim64_sm75  = Flash_fwd_kernel_traits<64,   64, 32, 2>;
using Traits_hdim128_sm75 = Flash_fwd_kernel_traits<128,  64, 32, 2>;

// SM80 (A100/A30) — 164 KB smem, cp.async available
using Traits_hdim64  = Flash_fwd_kernel_traits<64,  128, 64, 4>;
using Traits_hdim128 = Flash_fwd_kernel_traits<128, 128, 64, 4>;

// SM86 (RTX 3090, A10), SM89 (RTX 4090, L40S), SM120 (RTX 50xx) — 100 KB smem max
// hdim64:  BLOCK_M=128, BLOCK_N=64, NWARPS=4  → kSmemSizePipeline ≈ 80 KB ✓
// hdim128: BLOCK_M=64,  BLOCK_N=32, NWARPS=2  → kSmemSizePipeline ≈ 72 KB ✓
//   (BLOCK_M=128 with BLOCK_N=32 gives 130 KB — exceeds 100 KB limit!)
using Traits_hdim64_sm86  = Flash_fwd_kernel_traits<64,  128, 64, 4>;
using Traits_hdim128_sm86 = Flash_fwd_kernel_traits<128,  64, 32, 2>;
