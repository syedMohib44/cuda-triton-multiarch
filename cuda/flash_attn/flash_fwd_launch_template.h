/*
 * flash_fwd_launch_template.h — Kernel launch configuration and dispatch.
 *
 * Bridges runtime head_dim → compile-time template parameters.
 * Each .cu instantiation file includes this and calls run_flash_fwd<Traits>().
 */

#pragma once

#include <cuda_runtime.h>

#include "flash.h"
#include "flash_fwd_kernel.h"
#include "kernel_traits.h"

inline int current_sm_version() {
    int device;
    cudaGetDevice(&device);
    int major, minor;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    return major * 10 + minor;
}

template <typename Traits, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr int kBlockM = Traits::kBlockM;
    // kSmemSize is evaluated at host compile time (no __CUDA_ARCH__), so it
    // always resolves to kSmemSizeNoPipeline — wrong for SM80+ which uses the
    // double-buffered pipeline path and needs kSmemSizePipeline.
    // Query the actual SM at runtime and pick the right size.
    const int smem_size = (current_sm_version() >= 80)
                              ? Traits::kSmemSizePipeline
                              : Traits::kSmemSizeNoPipeline;

    const int num_m_blocks = (params.seqlen_q + kBlockM - 1) / kBlockM;
    dim3 grid(num_m_blocks, params.batch_size * params.num_heads);
    dim3 block(Traits::kNThreads);

    auto kernel = &flash_fwd_kernel<Traits, Is_causal>;

    // Default shared memory carveout is 48 KB. Request more if needed.
    if (smem_size > 48 * 1024) {
        cudaFuncSetAttribute(
            kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size
        );
    }

    kernel<<<grid, block, smem_size, stream>>>(params);
}

// ---------------------------------------------------------------------------
// SM-aware dispatch helpers
//
// We query the device's compute capability at runtime and route to the
// appropriate Traits instantiation:
//   SM75  (Turing):  smaller blocks to fit 64 KB smem
//   SM80  (A100):    original full-size blocks (164 KB smem)
//   SM86+ (Ampere/Ada): same block sizes as SM80 for hdim64; narrower for hdim128
// ---------------------------------------------------------------------------

inline void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    const int sm = current_sm_version();
    if (sm == 75) {
        params.is_causal
            ? run_flash_fwd<Traits_hdim64_sm75, true>(params, stream)
            : run_flash_fwd<Traits_hdim64_sm75, false>(params, stream);
    } else {
        // SM80, SM86, SM89: same block sizes for hdim64
        params.is_causal
            ? run_flash_fwd<Traits_hdim64, true>(params, stream)
            : run_flash_fwd<Traits_hdim64, false>(params, stream);
    }
}

inline void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    const int sm = current_sm_version();
    if (sm == 75) {
        params.is_causal
            ? run_flash_fwd<Traits_hdim128_sm75, true>(params, stream)
            : run_flash_fwd<Traits_hdim128_sm75, false>(params, stream);
    } else if (sm >= 86) {
        // SM86/89: reduced BLOCK_N to stay under 100 KB smem
        params.is_causal
            ? run_flash_fwd<Traits_hdim128_sm86, true>(params, stream)
            : run_flash_fwd<Traits_hdim128_sm86, false>(params, stream);
    } else {
        // SM80: full-size blocks
        params.is_causal
            ? run_flash_fwd<Traits_hdim128, true>(params, stream)
            : run_flash_fwd<Traits_hdim128, false>(params, stream);
    }
}
