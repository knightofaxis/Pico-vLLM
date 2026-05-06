import torch

from .base import OpsBackend


def get_ops_backend(device: torch.device | str, backend: str = "auto") -> OpsBackend:
    """Select an operator backend for the requested device.

    `auto` keeps the current behavior on CUDA and chooses the placeholder
    torch backend for CPU until CPU operators are implemented.
    """

    device = torch.device(device)
    backend = backend.lower()

    if backend == "auto":
        backend = "triton" if device.type == "cuda" else "torch"

    if backend == "triton":
        if device.type != "cuda":
            raise ValueError("Triton ops backend requires a CUDA device.")
        from .triton import TritonOps

        return TritonOps()

    if backend in ("torch", "cpu"):
        if device.type != "cpu":
            raise ValueError("Torch ops backend currently supports CPU devices only.")
        from .torch import TorchOps

        return TorchOps()

    raise ValueError(f"Unknown ops backend: {backend}")
