"""Build script for the CUTLASS-based FlashAttention extension.

Requires CUTLASS headers. Set CUTLASS_DIR env var to point at the cutlass repo
root (the directory containing `include/cutlass/cutlass.h`). Defaults to
`third_party/cutlass` relative to the repo root.

Build:
    make build-fac-cutlass
    # or directly:
    cd cuda/flash_attn_cutlass && CUTLASS_DIR=/path/to/cutlass python setup.py build_ext --inplace
"""

import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from arch_utils import get_gencode_flags

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


# Resolve CUTLASS include path
DEFAULT_CUTLASS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "cutlass"
)
CUTLASS_DIR = os.environ.get("CUTLASS_DIR", DEFAULT_CUTLASS_DIR)
CUTLASS_INCLUDE = os.path.join(CUTLASS_DIR, "include")
CUTLASS_TOOLS_INCLUDE = os.path.join(CUTLASS_DIR, "tools", "util", "include")

if not os.path.isdir(CUTLASS_INCLUDE):
    print(
        f"ERROR: CUTLASS not found at {CUTLASS_INCLUDE}\n"
        f"Either clone it:\n"
        f"  git clone https://github.com/NVIDIA/cutlass.git {DEFAULT_CUTLASS_DIR}\n"
        f"Or set CUTLASS_DIR to point at your existing CUTLASS checkout.",
        file=sys.stderr,
    )
    sys.exit(1)


extra_link_args = []
if sys.platform == "win32":
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_lib = os.path.join(os.path.dirname(os.path.dirname(nvcc)), "lib")
        if os.path.exists(cuda_lib):
            extra_link_args.append("/LIBPATH:" + cuda_lib)

setup(
    name="flash_attn_cutlass",
    ext_modules=[
        CUDAExtension(
            "flash_attn_cutlass",
            [
                "flash_api.cu",
                "flash_fwd_hdim32_fp16_sm80.cu",
                "flash_fwd_hdim32_fp16_causal_sm80.cu",
                "flash_fwd_hdim64_fp16_sm80.cu",
                "flash_fwd_hdim64_fp16_causal_sm80.cu",
                "flash_fwd_hdim128_fp16_sm80.cu",
                "flash_fwd_hdim128_fp16_causal_sm80.cu",
            ],
            include_dirs=[CUTLASS_INCLUDE, CUTLASS_TOOLS_INCLUDE],
            extra_compile_args={
                "nvcc": [
                    *( ["-std=c++20"] if sys.platform == "win32" else ["-std=c++17"] ),
                    "-O3",
                    "--use_fast_math",
                    *get_gencode_flags("flash_attn_cutlass"),
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-lineinfo",  # for ncu source attribution
                ],
            },
            extra_link_args=extra_link_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
