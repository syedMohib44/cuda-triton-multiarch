"""
build_extensions.py — Build WMMA and CUTLASS CUDA extensions and install them
into the kernels/ package directory so they are importable as:
    import flash_attn_cuda
    import flash_attn_cutlass

Usage (after pip install):
    build-cuda-kernels                  # builds WMMA + CUTLASS
    build-cuda-kernels --wmma-only      # builds WMMA only (no CUTLASS headers needed)
    build-cuda-kernels --cutlass-only

From source:
    python -m kernels.build_extensions

Prerequisites:
  - nvcc in PATH  (comes with CUDA Toolkit)
  - MSVC in PATH on Windows  (run from "x64 Native Tools Command Prompt" or vcvars64.bat)
  - CUTLASS:  auto-cloned to third_party/cutlass, OR set CUTLASS_DIR env var
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> str:
    """Root of the cuda-triton repo (one level above kernels/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _kernels_dir() -> str:
    """The kernels/ package directory — .pyd files are copied here."""
    return os.path.dirname(os.path.abspath(__file__))


def _check_nvcc() -> bool:
    return shutil.which("nvcc") is not None


def _check_msvc() -> bool:
    """On Windows, verify that cl.exe (MSVC compiler) is in PATH."""
    if sys.platform != "win32":
        return True
    return shutil.which("cl") is not None


def _build_extension(setup_py_dir: str, env: dict | None = None) -> bool:
    """Run `python setup.py build_ext --inplace` in the given directory."""
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    merged_env = {**os.environ, **(env or {})}
    print(f"\n[build-cuda-kernels] Building in {setup_py_dir}")
    print(f"  {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=setup_py_dir, env=merged_env)
    return result.returncode == 0


def _copy_built_files(src_dir: str, dest_dir: str, ext_name: str) -> list[str]:
    """
    Find built .pyd / .so files matching ext_name in src_dir (and its build/
    subdirs) and copy them to dest_dir.  Returns list of copied filenames.
    """
    # Match both inplace result and potential build/lib.* layout
    patterns = [
        os.path.join(src_dir, f"{ext_name}*.pyd"),
        os.path.join(src_dir, f"{ext_name}*.so"),
        os.path.join(src_dir, "build", "**", f"{ext_name}*.pyd"),
        os.path.join(src_dir, "build", "**", f"{ext_name}*.so"),
    ]
    found: list[str] = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))

    copied: list[str] = []
    for src in found:
        fname = os.path.basename(src)
        dst = os.path.join(dest_dir, fname)
        shutil.copy2(src, dst)
        print(f"  copied {src}\n      → {dst}")
        copied.append(fname)
    return copied


def _ensure_cutlass(repo_root: str) -> str | None:
    """
    Ensure CUTLASS headers are available.
    Returns the CUTLASS root dir (containing include/cutlass/cutlass.h),
    or None if unavailable (prints instructions).
    """
    # 1. Explicit env var
    cutlass_dir = os.environ.get("CUTLASS_DIR", "")
    if cutlass_dir and os.path.isdir(os.path.join(cutlass_dir, "include")):
        return cutlass_dir

    # 2. third_party/cutlass (already cloned)
    default = os.path.join(repo_root, "third_party", "cutlass")
    if os.path.isdir(os.path.join(default, "include")):
        return default

    # 3. Try to auto-clone (shallow, headers only — no blobs)
    print(
        "\n[build-cuda-kernels] CUTLASS not found. Attempting shallow clone into "
        f"{default} …"
    )
    if shutil.which("git") is None:
        print("  ERROR: git not found in PATH. Install git or set CUTLASS_DIR.")
        return None

    os.makedirs(default, exist_ok=True)
    cmd = [
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",          # sparse-checkout style, fast
        "--no-checkout",
        "https://github.com/NVIDIA/cutlass.git",
        default,
    ]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(
            "  Auto-clone failed. Run manually:\n"
            f"    git clone --depth 1 https://github.com/NVIDIA/cutlass.git {default}\n"
            "  Or set CUTLASS_DIR to an existing checkout."
        )
        return None

    # Sparse-checkout just the include/ and tools/util/include/ trees
    sparse_cmd = [
        "git", "-C", default, "sparse-checkout", "set",
        "include", "tools/util/include",
    ]
    subprocess.run(sparse_cmd)
    checkout_cmd = ["git", "-C", default, "checkout"]
    subprocess.run(checkout_cmd)

    if os.path.isdir(os.path.join(default, "include")):
        print("  CUTLASS cloned successfully.")
        return default

    print(
        "  Sparse checkout did not produce expected layout.\n"
        f"  Set CUTLASS_DIR to a full CUTLASS clone and retry."
    )
    return None


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def build_wmma(kernels_dir: str, repo_root: str) -> bool:
    """Build flash_attn_cuda (WMMA backend)."""
    setup_dir = os.path.join(repo_root, "cuda", "flash_attn")
    ok = _build_extension(setup_dir)
    if not ok:
        print("\n[build-cuda-kernels] WMMA build FAILED.")
        return False
    copied = _copy_built_files(setup_dir, kernels_dir, "flash_attn_cuda")
    if copied:
        print(f"\n[build-cuda-kernels] WMMA extension installed: {copied}")
        return True
    print("\n[build-cuda-kernels] WMMA build ran but no .pyd/.so found.")
    return False


def build_cutlass(kernels_dir: str, repo_root: str) -> bool:
    """Build flash_attn_cutlass (CuTe backend)."""
    cutlass_dir = _ensure_cutlass(repo_root)
    if cutlass_dir is None:
        print("[build-cuda-kernels] Skipping CUTLASS build (headers unavailable).")
        return False

    setup_dir = os.path.join(repo_root, "cuda", "flash_attn_cutlass")
    env = {"CUTLASS_DIR": cutlass_dir}
    ok = _build_extension(setup_dir, env=env)
    if not ok:
        print("\n[build-cuda-kernels] CUTLASS build FAILED.")
        return False
    copied = _copy_built_files(setup_dir, kernels_dir, "flash_attn_cutlass")
    if copied:
        print(f"\n[build-cuda-kernels] CUTLASS extension installed: {copied}")
        return True
    print("\n[build-cuda-kernels] CUTLASS build ran but no .pyd/.so found.")
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build CUDA flash-attention extensions and install into kernels/."
    )
    parser.add_argument("--wmma-only",    action="store_true", help="Build WMMA only")
    parser.add_argument("--cutlass-only", action="store_true", help="Build CUTLASS only")
    args = parser.parse_args(argv)

    # --- Pre-flight checks ---
    if not _check_nvcc():
        print(
            "ERROR: nvcc not found in PATH.\n"
            "Install the CUDA Toolkit and make sure nvcc is on your PATH.\n"
            "  Windows: add C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\vX.Y\\bin\n"
            "  Linux:   add /usr/local/cuda/bin"
        )
        return 1

    if not _check_msvc():
        print(
            "ERROR: cl.exe (MSVC) not found in PATH.\n"
            "Run this script from an 'x64 Native Tools Command Prompt for VS 2019/2022'\n"
            "or call vcvars64.bat first:\n"
            '  "C:\\Program Files\\Microsoft Visual Studio\\2022\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat"'
        )
        return 1

    repo_root   = _repo_root()
    kernels_dir = _kernels_dir()

    build_wmma_flag    = not args.cutlass_only
    build_cutlass_flag = not args.wmma_only

    results = {}
    if build_wmma_flag:
        results["wmma"] = build_wmma(kernels_dir, repo_root)
    if build_cutlass_flag:
        results["cutlass"] = build_cutlass(kernels_dir, repo_root)

    print("\n" + "=" * 60)
    print("Build summary:")
    for name, ok in results.items():
        print(f"  {name:12s}  {'OK' if ok else 'FAILED'}")

    any_ok = any(results.values())
    if any_ok:
        print(
            "\nTo verify, run:\n"
            "  python -c \"from kernels import flash_attention_bhsd, flash_attention_backend;"
            " print(flash_attention_backend())\""
        )
    return 0 if any_ok or not results else 1


if __name__ == "__main__":
    sys.exit(main())
