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

# Auto-derive CUDA_HOME from nvcc location if not already set.
# torch.utils.cpp_extension requires CUDA_HOME; conda environments often place
# nvcc in <env>/Library/bin/nvcc.exe without setting CUDA_HOME automatically.
if HAS_NVCC and not os.environ.get("CUDA_HOME") and not os.environ.get("CUDA_PATH"):
    _nvcc = shutil.which("nvcc")
    # nvcc lives at <cuda_root>/bin/nvcc.exe → go up two levels
    _cuda_root = os.path.dirname(os.path.dirname(_nvcc))
    os.environ["CUDA_HOME"] = _cuda_root
    os.environ["CUDA_PATH"] = _cuda_root
    print(f"[cuda-triton] Auto-set CUDA_HOME={_cuda_root}")

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
        import torch.utils.cpp_extension as _torch_ext

        # torch 2.x caches CUDA_HOME at import time; if it resolved to None
        # (e.g. conda env where CUDA_HOME points to Library/ not a standard install),
        # patch it directly so CUDAExtension / library_paths() can find the libs.
        if _torch_ext.CUDA_HOME is None:
            _nvcc = shutil.which("nvcc")
            if _nvcc:
                _torch_ext.CUDA_HOME = os.path.dirname(os.path.dirname(_nvcc))
                print(f"[cuda-triton] Patched torch CUDA_HOME={_torch_ext.CUDA_HOME}")

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
        # setuptools requires paths relative to the setup.py (repo root), never absolute
        ext_modules.append(
            CUDAExtension(
                "kernels.flash_attn_cuda",   # installs as kernels/flash_attn_cuda*.so
                sources=[
                    "cuda/flash_attn/flash_api.cu",
                    "cuda/flash_attn/flash_fwd_hdim64_fp16_sm80.cu",
                    "cuda/flash_attn/flash_fwd_hdim64_fp16_causal_sm80.cu",
                    "cuda/flash_attn/flash_fwd_hdim128_fp16_sm80.cu",
                    "cuda/flash_attn/flash_fwd_hdim128_fp16_causal_sm80.cu",
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
            ext_modules.append(
                CUDAExtension(
                    "kernels.flash_attn_cutlass",
                    sources=[
                        "cuda/flash_attn_cutlass/flash_api.cu",
                        "cuda/flash_attn_cutlass/flash_fwd_hdim32_fp16_sm80.cu",
                        "cuda/flash_attn_cutlass/flash_fwd_hdim32_fp16_causal_sm80.cu",
                        "cuda/flash_attn_cutlass/flash_fwd_hdim64_fp16_sm80.cu",
                        "cuda/flash_attn_cutlass/flash_fwd_hdim64_fp16_causal_sm80.cu",
                        "cuda/flash_attn_cutlass/flash_fwd_hdim128_fp16_sm80.cu",
                        "cuda/flash_attn_cutlass/flash_fwd_hdim128_fp16_causal_sm80.cu",
                    ],
                    include_dirs=[
                        os.path.join(CUTLASS_DIR, "include"),
                        os.path.join(CUTLASS_DIR, "tools", "util", "include"),
                    ],
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

        # torch 2.12+ merged c10_cuda and torch_cuda into c10/torch.
        # On Windows, CUDAExtension still injects the old lib names which no
        # longer have corresponding .lib files — strip any that are missing.
        if sys.platform == "win32":
            import torch as _torch
            _torch_lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
            for _ext in ext_modules:
                _before = list(_ext.libraries)
                _ext.libraries = [
                    lib for lib in _ext.libraries
                    if not lib.startswith(("c10_cuda", "torch_cuda"))
                    or os.path.exists(os.path.join(_torch_lib, lib + ".lib"))
                ]
                _removed = set(_before) - set(_ext.libraries)
                if _removed:
                    print(f"[cuda-triton] Stripped missing libs from {_ext.name}: {_removed}")

    except Exception as exc:
        import traceback
        traceback.print_exc()
        msg = f"[cuda-triton] Could not set up CUDA extensions: {exc}"
        if CUDA_REQUIRED:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
        print(f"WARNING: {msg}", file=sys.stderr)
        ext_modules = []
        cmdclass = {}


# ---------------------------------------------------------------------------
# Hand off to setuptools (pyproject.toml provides the metadata)
# ---------------------------------------------------------------------------
setup(
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)
