"""
arch_utils.py — GPU architecture detection for setup.py build scripts.

Priority order:
  1. TORCH_CUDA_ARCH_LIST env var  — explicit user/CI override (e.g. "8.0 8.6")
  2. Auto-detect from installed GPUs via torch.cuda
  3. Fallback: compile for all supported architectures (SM75/80/86/89)

All setup.py files in this directory import get_gencode_flags() from here
so there is one place to update the supported-arch list.
"""

import os
import warnings

# All SM versions this project has kernels for.
SUPPORTED_SMS = [75, 80, 86, 89, 90, 120]

# Human-readable names for print output.
_SM_NAMES = {
    75:  "Turing (T4 / RTX 20xx)",
    80:  "Ampere A100",
    86:  "Ampere (RTX 30xx / A10)",
    89:  "Ada (RTX 40xx / L40S)",
    90:  "Hopper (H100)",
    120: "Blackwell (RTX 50xx)",
}


def _sm_to_gencode(sm: int) -> list[str]:
    """Return the nvcc -gencode pair for a single SM version."""
    return ["-gencode", f"arch=compute_{sm},code=sm_{sm}"]


def _all_gencode() -> list[str]:
    flags = []
    for sm in SUPPORTED_SMS:
        flags += _sm_to_gencode(sm)
    return flags


def _detect_gpu_sms() -> list[int]:
    """Return sorted list of unique SM versions for all installed GPUs."""
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        sms = set()
        for i in range(torch.cuda.device_count()):
            major, minor = torch.cuda.get_device_capability(i)
            sms.add(major * 10 + minor)
        return sorted(sms)
    except Exception:
        return []


def _parse_arch_list(arch_list: str) -> list[int]:
    """Parse TORCH_CUDA_ARCH_LIST string like '7.5 8.0 8.6+PTX' into SM ints."""
    sms = []
    for token in arch_list.replace(",", " ").split():
        token = token.replace("+PTX", "").strip()
        try:
            major, minor = token.split(".")
            sms.append(int(major) * 10 + int(minor))
        except ValueError:
            pass  # ignore malformed tokens
    return sorted(set(sms))


def get_gencode_flags(label: str = "") -> list[str]:
    """Return nvcc -gencode flags, with source printed to stdout.

    Args:
        label: short name of the extension being built (for print output).
    """
    prefix = f"[{label}] " if label else ""

    # 1. Explicit env override.
    arch_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_env:
        sms = _parse_arch_list(arch_env)
        unsupported = [s for s in sms if s not in SUPPORTED_SMS]
        if unsupported:
            warnings.warn(
                f"{prefix}TORCH_CUDA_ARCH_LIST includes SM{unsupported} which "
                f"this project has no kernel traits for. It will still compile "
                f"(PTX JIT), but runtime dispatch may not be optimal."
            )
        flags = []
        for sm in sms:
            flags += _sm_to_gencode(sm)
        names = [_SM_NAMES.get(sm, f"SM{sm}") for sm in sms]
        print(f"{prefix}Building for architectures from TORCH_CUDA_ARCH_LIST: {names}")
        return flags

    # 2. Auto-detect installed GPUs.
    detected = _detect_gpu_sms()
    if detected:
        # Only compile for detected SMs that we actually have kernels for.
        # For unknown SMs, include the nearest supported SM so PTX JIT can cover it.
        build_sms = []
        for sm in detected:
            if sm in SUPPORTED_SMS:
                build_sms.append(sm)
            else:
                # Find closest supported SM <= detected (PTX forward-compat).
                candidates = [s for s in SUPPORTED_SMS if s <= sm]
                if candidates:
                    fallback = max(candidates)
                    if fallback not in build_sms:
                        build_sms.append(fallback)
                    warnings.warn(
                        f"{prefix}Detected SM{sm} has no dedicated kernel traits. "
                        f"Compiling SM{fallback} PTX for JIT compatibility."
                    )
                else:
                    warnings.warn(
                        f"{prefix}Detected SM{sm} is below minimum supported SM75. "
                        f"Kernel may not load."
                    )

        build_sms = sorted(set(build_sms))
        flags = []
        for sm in build_sms:
            flags += _sm_to_gencode(sm)
        names = [_SM_NAMES.get(sm, f"SM{sm}") for sm in build_sms]
        gpu_names = []
        try:
            import torch
            for i in range(torch.cuda.device_count()):
                gpu_names.append(torch.cuda.get_device_name(i))
        except Exception:
            pass
        gpus_str = ", ".join(gpu_names) if gpu_names else "unknown"
        print(f"{prefix}Auto-detected GPU(s): {gpus_str}")
        print(f"{prefix}Building for: {names}")
        return flags

    # 3. No GPU found — build for all supported archs (CI / headless build).
    print(
        f"{prefix}No GPU detected. Building for all supported architectures: "
        f"{[_SM_NAMES.get(sm, f'SM{sm}') for sm in SUPPORTED_SMS]}"
    )
    return _all_gencode()
