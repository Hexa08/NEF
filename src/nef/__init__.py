"""NEF public API."""

from .core import Tensor, add, matmul, mul, softmax, tensor

float32 = "float32"
float16 = "float16"
bfloat16 = "float16"  # compatibility placeholder

__all__ = [
    "Tensor",
    "add",
    "matmul",
    "mul",
    "softmax",
    "tensor",
    "float32",
    "float16",
    "bfloat16",
]
