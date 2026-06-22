import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arch_utils import get_gencode_flags

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

GENCODE_FLAGS = get_gencode_flags("cuda_kernels")

extra_link_args = []
if sys.platform == "win32":
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_lib = os.path.join(os.path.dirname(os.path.dirname(nvcc)), "lib")
        if os.path.exists(cuda_lib):
            extra_link_args.append("/LIBPATH:" + cuda_lib)

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
            extra_compile_args={"nvcc": [
                                         *( ["-std=c++20"] if sys.platform == "win32" else ["-std=c++17"] ),
                                         "-O3", "--use_fast_math", *GENCODE_FLAGS]},
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
   