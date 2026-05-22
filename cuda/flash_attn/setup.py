import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arch_utils import get_gencode_flags

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

GENCODE_FLAGS = get_gencode_flags("flash_attn_cuda")

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
