"""
setup.py — Top-level build script for cuda-triton-kernels.

When pip runs `pip install cuda-triton-kernels` (or `pip install -e .`):
  - If nvcc is found: WMMA + CUTLASS CUDA extensions are compiled and placed
    inside the kernels/ package directory so they are directly importable.
  - If nvcc is not found: package installs without CUDA extensions; Triton JIT
    kernels (softmax, flash attention, matmul) still work out of the box.

This is the same pattern used by flash-attn and xformers.

Environment variables:
  TORCH_CUDA_ARCH_LIST   — override target GPU architectures (e.g. "8.0 8.6")
  CUTLASS_DIR            — path to existing CUTLASS checkout (default: third_party/cutlass)
  SKIP_CUDA_BUILD        — set to "1" to force-skip CUDA extension compilation
"""

import os
import shutil
import subprocess
import sys

from setuptools import setup

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.join(HERE, "kernels")

# ---------------------------------------------------------------------------
# Decide whether to build CUDA extensions
# ---------------------------------------------------------------------------
SKIP_CUDA      = os.environ.get("SKIP_CUDA_BUILD",    "0").strip() == "1"
CUDA_REQUIRED  = os.environ.get("CUDA_BUILD_REQUIRED", "0").strip() == "1"
HAS_NVCC       = shutil.which("nvcc") is not None

# On Windows also need MSVC (cl.exe)
HAS_MSVC = True
if sys.platform == "win32":
    HAS_MSVC = shutil.which("cl") is not None
    if not HAS_MSVC:
        msg = (
            "[cuda-triton] cl.exe (MSVC) not found in PATH. "
            "Run from an 'x64 Native Tools Command Prompt' to enable CUDA extensions."
        )
        if CUDA_REQUIRED:
            print(msg, file=sys.stderr)
            sys.exit(1)
        print("WARNING: " + msg, file=sys.stderr)

COMPILE_CUDA = HAS_NVCC and HAS_MSVC and not SKIP_CUDA

if not HAS_NVCC and not SKIP_CUDA:
    msg = (
        "[cuda-triton] nvcc not found — CUDA extensions will not be compiled. "
        "Triton JIT kernels still work. Install CUDA Toolkit to enable WMMA/CUTLASS."
    )
    if CUDA_REQUIRED:
        print(msg, file=sys.stderr)
        sys.exit(1)
    print(msg, file=sys.stderr)


# ---------------------------------------------------------------------------
# CUTLASS header resolution (needed for CuTe backend)
# ---------------------------------------------------------------------------
DEFAULT_CUTLASS_DIR = os.path.join(HERE, "third_party", "cutlass")
CUTLASS_DIR = os.environ.get("CUTLASS_DIR", DEFAULT_CUTLASS_DIR)
CUTLASS_INCLUDE = os.path.join(CUTLASS_DIR, "include")


def _ensure_cutlass() -> bool:
    """Clone CUTLASS (headers only) if not present. Returns True if available."""
    if os.path.isdir(CUTLASS_INCLUDE):
        return True
    if shutil.which("git") is None:
        print("[cuda-triton] git not found — skipping CUTLASS build.", file=sys.stderr)
        return False
    print(f"[cuda-triton] Cloning CUTLASS headers into {DEFAULT_CUTLASS_DIR} …")
    os.makedirs(DEFAULT_CUTLASS_DIR, exist_ok=True)
    cmds = [
        ["git", "clone", "--depth", "1", "--filter=blob:none",
         "--no-checkout", "https://github.com/NVIDIA/cutlass.git", DEFAULT_CUTLASS_DIR],
        ["git", "-C", DEFAULT_CUTLASS_DIR, "sparse-checkout", "set",
         "include", "tools/util/include"],
        ["git", "-C", DEFAULT_CUTLASS_DIR, "checkout"],
    ]
    for cmd in cmds:
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            print(
                f"[cuda-triton] CUTLASS clone failed: {r.stderr.decode()}\n"
                "  Set CUTLASS_DIR or build WMMA only with SKIP_CUDA_BUILD=1.",
                file=sys.stderr,
            )
            return False
    return os.path.isdir(CUTLASS_INCLUDE)


# ---------------------------------------------------------------------------
# Build CUDA extensions inline using torch CUDAExtension machinery
# ---------------------------------------------------------------------------
ext_modules = []
cmdclass = {}

if COMPILE_CUDA:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        sys.path.insert(0, os.path.join(HERE, "cuda"))
        from arch_utils import get_gencode_flags

        extra_link_args = []
        if sys.platform == "win32":
            nvcc_path = shutil.which("nvcc")
            if nvcc_path:
                cuda_lib = os.path.join(os.path.dirname(os.path.dirname(nvcc_path)), "lib")
                if os.path.exists(cuda_lib):
                    extra_link_args.append("/LIBPATH:" + cuda_lib)

        cpp_std = ["-std=c++20"] if sys.platform == "win32" else ["-std=c++17"]

        # ---- WMMA extension ------------------------------------------------
        wmma_dir = os.path.join(HERE, "cuda", "flash_attn")
        ext_modules.append(
            CUDAExtension(
                "kernels.flash_attn_cuda",   # installs as kernels/flash_attn_cuda*.so
                sources=[
                    os.path.join(wmma_dir, "flash_api.cu"),
                    os.path.join(wmma_dir, "flash_fwd_hdim64_fp16_sm80.cu"),
                    os.path.join(wmma_dir, "flash_fwd_hdim64_fp16_causal_sm80.cu"),
                    os.path.join(wmma_dir, "flash_fwd_hdim128_fp16_sm80.cu"),
                    os.path.join(wmma_dir, "flash_fwd_hdim128_fp16_causal_sm80.cu"),
                ],
                extra_compile_args={
                    "nvcc": ["-O3", "--use_fast_math", *cpp_std,
                             *get_gencode_flags("flash_attn_cuda")],
                },
                extra_link_args=extra_link_args,
            )
        )
        print("[cuda-triton] WMMA extension queued for build.")

        # ---- CUTLASS extension ---------------------------------------------
        if _ensure_cutlass():
            cutlass_dir = os.path.join(HERE, "cuda", "flash_attn_cutlass")
            ext_modules.append(
                CUDAExtension(
                    "kernels.flash_attn_cutlass",
                    sources=[
                        os.path.join(cutlass_dir, "flash_api.cu"),
                        os.path.join(cutlass_dir, "flash_fwd_hdim32_fp16_sm80.cu"),
                        os.path.join(cutlass_dir, "flash_fwd_hdim32_fp16_causal_sm80.cu"),
                        os.path.join(cutlass_dir, "flash_fwd_hdim64_fp16_sm80.cu"),
                        os.path.join(cutlass_dir, "flash_fwd_hdim64_fp16_causal_sm80.cu"),
                        os.path.join(cutlass_dir, "flash_fwd_hdim128_fp16_sm80.cu"),
                        os.path.join(cutlass_dir, "flash_fwd_hdim128_fp16_causal_sm80.cu"),
                    ],
                    include_dirs=[CUTLASS_INCLUDE,
                                  os.path.join(CUTLASS_DIR, "tools", "util", "include")],
                    extra_compile_args={
                        "nvcc": ["-O3", "--use_fast_math", *cpp_std,
                                 *get_gencode_flags("flash_attn_cutlass"),
                                 "--expt-relaxed-constexpr",
                                 "--expt-extended-lambda",
                                 "-lineinfo"],
                    },
                    extra_link_args=extra_link_args,
                )
            )
            print("[cuda-triton] CUTLASS extension queued for build.")
        else:
            print("[cuda-triton] CUTLASS headers unavailable — skipping CuTe backend.")

        cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=False)

    except Exception as exc:
        print(f"[cuda-triton] WARNING: Could not set up CUDA extensions: {exc}",
              file=sys.stderr)
        ext_modules = []
        cmdclass = {}


# ---------------------------------------------------------------------------
# Hand off to setuptools (pyproject.toml provides the metadata)
# ---------------------------------------------------------------------------
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
