# cuda-triton Windows build script (PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File Makefile.windows.ps1 <target>
# Equivalent to the Linux Makefile targets.
#
# Prerequisites:
#   - conda environment active with torch + triton-windows installed
#   - Visual Studio Build Tools 2019+ (for CUDA extension compilation)
#   - CUDA Toolkit matching your PyTorch build (e.g. 12.8 for cu128)
#
# Quick install:
#   pip install torch --index-url https://download.pytorch.org/whl/cu128
#   pip install triton-windows ninja pytest numpy

param([string]$Target = "help")

function Build-Cuda {
    Write-Host "Building CUDA extension (arch auto-detected)..."
    Push-Location cuda
    python setup.py build_ext --inplace
    Pop-Location
    # Copy built .pyd/.dll to project root
    Get-ChildItem cuda -Filter "cuda_kernels*.pyd" | Copy-Item -Destination .
    Get-ChildItem cuda\build -Recurse -Filter "cuda_kernels*.pyd" -ErrorAction SilentlyContinue |
        Copy-Item -Destination .
    Write-Host "Built cuda_kernels extension"
}

function Build-Fac {
    Write-Host "Building WMMA FlashAttention CUDA extension..."
    Push-Location cuda\flash_attn
    python setup.py build_ext --inplace
    Pop-Location
    Write-Host "Built flash_attn_cuda extension"
}

function Build-FacCutlass {
    $CutlassDir = "third_party\cutlass"
    if (-not (Test-Path $CutlassDir)) {
        Write-Host "Cloning CUTLASS v3.6.0..."
        New-Item -ItemType Directory -Force third_party | Out-Null
        git clone --depth 1 --branch v3.6.0 https://github.com/NVIDIA/cutlass.git $CutlassDir
    }
    Push-Location cuda\flash_attn_cutlass
    $env:CUTLASS_DIR = (Resolve-Path "..\..\$CutlassDir").Path
    python setup.py build_ext --inplace
    Pop-Location
    Write-Host "Built flash_attn_cutlass extension"
}

function Run-Tests { python -m pytest tests/test_kernels.py -v }
function Run-TestsTriton { python -m pytest tests/test_kernels.py -v -k "not CUDA" }
function Run-TestsCuda { python -m pytest tests/test_kernels.py -v -k "CUDA" }
function Run-TestsFac { python -m pytest tests/test_kernels.py -v -k "CUDAFlashAttention" }
function Run-TestsFacCutlass { python -m pytest tests/test_kernels.py -v -k "CUTLASSFlashAttention" }

function Clean-Cuda {
    Remove-Item -Recurse -Force cuda\build, cuda\dist, "cuda\*.egg-info", "cuda\*.pyd", "cuda_kernels*.pyd" -ErrorAction SilentlyContinue
    Write-Host "Cleaned CUDA build artifacts"
}

function Detect-GPU { python gpu_utils.py }

function Show-Help {
    Write-Host @"
cuda-triton Windows build targets:

  build-cuda          Build CUDA softmax/matmul extension
  build-fac           Build WMMA FlashAttention extension
  build-fac-cutlass   Build CuTe FlashAttention extension (clones CUTLASS)
  test                Run all tests (Triton + CUDA)
  test-triton         Run Triton-only tests (no build needed)
  test-cuda           Run CUDA extension tests
  test-fac            Run WMMA FlashAttention tests
  test-fac-cutlass    Run CuTe FlashAttention tests
  clean-cuda          Remove CUDA build artifacts
  detect-gpu          Print GPU SM version and block size recommendations

Examples:
  powershell -File Makefile.windows.ps1 build-cuda
  powershell -File Makefile.windows.ps1 test-triton
  powershell -File Makefile.windows.ps1 detect-gpu
"@
}

switch ($Target) {
    "build-cuda"        { Build-Cuda }
    "build-fac"         { Build-Fac }
    "build-fac-cutlass" { Build-FacCutlass }
    "test"              { Run-Tests }
    "test-triton"       { Run-TestsTriton }
    "test-cuda"         { Run-TestsCuda }
    "test-fac"          { Run-TestsFac }
    "test-fac-cutlass"  { Run-TestsFacCutlass }
    "clean-cuda"        { Clean-Cuda }
    "detect-gpu"        { Detect-GPU }
    default             { Show-Help }
}
