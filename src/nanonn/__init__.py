"""nanonn：基于 CuPy 手写的深度学习基础设施（不依赖 PyTorch）。"""

from . import functional, nn, optim
from .tensor import Function, Tensor, is_grad_enabled, no_grad

__all__ = [
    "Tensor",
    "Function",
    "no_grad",
    "is_grad_enabled",
    "functional",
    "nn",
    "optim",
]


def main() -> None:
    print("Hello from nanonn!")
