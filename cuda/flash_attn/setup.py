import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arch_utils import get_gencode_flags

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

GENCODE_FLAGS = get_gencode_flags("flash_attn_cuda")

# On Windows + conda, cudart.lib lives in Library\lib (not lib\x64).
# Find it automatically from wherever nvcc is installed.
extra_link_args = []
if sys.platform == "win32":
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_lib = os.path.join(os.path.dirname(os.path.dirname(nvcc)), "lib")
        if os.path.exists(cuda_lib):
            extra_link_args.append("/LIBPATH:" + cuda_lib)

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
                    *( ["-std=c++20"] if sys.platform == "win32" else ["-std=c++17"] ),
                ],
            },
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
