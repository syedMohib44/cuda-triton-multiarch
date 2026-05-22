import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_utils import get_gencode_flags

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

GENCODE_FLAGS = get_gencode_flags("cuda_kernels")

setup(
    name="cuda_kernels",
    ext_modules=[
        CUDAExtension(
            "cuda_kernels",
            [
                "bindings.cu",
                "softmax.cu",
                "softmax_triton.cu",
                "fused_rmsnorm_swiglu.cu",
                "wmma_matmul.cu",
            ],
            extra_compile_args={"nvcc": ["-O3", "--use_fast_math", *GENCODE_FLAGS, "-std=c++17"]},
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
