"""NEF public API."""

from .core import Graph, GraphNode, Tensor, add, matmul, mul, softmax, tensor

float32 = "float32"
float16 = "float16"
bfloat16 = "float16"  # compatibility placeholder

__all__ = [
    "Graph",
    "GraphNode",
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
