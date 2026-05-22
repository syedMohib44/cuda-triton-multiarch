from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Multi-arch: SM75 (Turing/T4), SM80 (A100), SM86 (RTX 30xx/A10), SM89 (RTX 40xx/L40S)
# SM75 kernels fall back to the SM80 PTX (forward-compatible) for cp.async-free paths.
GENCODE_FLAGS = [
    "-gencode", "arch=compute_75,code=sm_75",
    "-gencode", "arch=compute_80,code=sm_80",
    "-gencode", "arch=compute_86,code=sm_86",
    "-gencode", "arch=compute_89,code=sm_89",
]

setup(
    name="flash_attn_cuda",
    ext_modules=[
        CUDAExtension(
            "flash_attn_cuda",
            [
                "flash_api.cu",
                "flash_fwd_hdim64_fp16_sm80.cu",
                "flash_fwd_hdim64_fp16_causal_sm80.cu",
                "flash_fwd_hdim128_fp16_sm80.cu",
                "flash_fwd_hdim128_fp16_causal_sm80.cu",
            ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    *GENCODE_FLAGS,
                    "-std=c++17",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
