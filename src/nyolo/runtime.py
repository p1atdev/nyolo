import os

import torch


def resolve_device(device: str | torch.device) -> torch.device:
    result = torch.device(device)
    if result.type == "cpu":
        return result
    if result.type != "mps" or result.index not in (None, 0):
        raise ValueError("This release supports explicit cpu or mps execution")
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError(
            "Disable PYTORCH_ENABLE_MPS_FALLBACK; implicit CPU execution is forbidden"
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    return torch.device("mps")
