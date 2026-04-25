"""NEF lazy graph runtime prototype (stdlib-only).

This module provides a minimal implementation of the NEF execution model:
- operations are represented as nodes in a lazy DAG
- execution is deferred until Tensor.execute() or value materialization
- device planning and optimization hooks exist, but CPU is the only backend implemented
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
from typing import Any


@dataclass(slots=True)
class GraphNode:
    op: str
    inputs: list["Tensor"] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)


class Tensor:
    """Lazy tensor view into the NEF graph."""

    def __init__(
        self,
        node: GraphNode,
        *,
        shape: tuple[int, ...],
        dtype: str,
        device: str | None = None,
        value: Any = None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.graph_node = node
        self._materialized = value is not None
        self._value = value

    def execute(self) -> "Tensor":
        if self._materialized:
            return self
        planner = DevicePlanner()
        planner.assign(self)
        runtime = ExecutionRuntime()
        self._value = runtime.evaluate(self)
        self._materialized = True
        return self

    def numpy(self) -> Any:
        """Compatibility method returning the materialized Python value."""
        self.execute()
        return self._value

    def __repr__(self) -> str:
        if self._materialized:
            return f"Tensor(shape={self.shape}, dtype={self.dtype}, value={self._value!r})"
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, lazy_op={self.graph_node.op!r})"


class DevicePlanner:
    def assign(self, output: Tensor) -> None:
        for t in _walk(output):
            if t.device is None:
                t.device = "cpu"


class ExecutionRuntime:
    def __init__(self) -> None:
        self._cache: dict[int, Any] = {}

    def evaluate(self, output: Tensor) -> Any:
        return self._eval_tensor(output)

    def _eval_tensor(self, tensor: Tensor) -> Any:
        key = id(tensor)
        if key in self._cache:
            return self._cache[key]

        node = tensor.graph_node
        if node.op == "const":
            result = node.attrs["value"]
        else:
            args = [self._eval_tensor(inp) for inp in node.inputs]
            result = KERNELS[node.op](*args, **node.attrs)

        self._cache[key] = result
        return result


def _shape_of(data: Any) -> tuple[int, ...]:
    if isinstance(data, list):
        if not data:
            return (0,)
        return (len(data), *_shape_of(data[0]))
    return ()


def _kernel_add(a: Any, b: Any) -> Any:
    if isinstance(a, list):
        return [_kernel_add(x, y) for x, y in zip(a, b)]
    return a + b


def _kernel_mul(a: Any, b: Any) -> Any:
    if isinstance(a, list):
        return [_kernel_mul(x, y) for x, y in zip(a, b)]
    return a * b


def _kernel_matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b or not b[0]:
        return []
    bt = list(zip(*b))
    return [[sum(x * y for x, y in zip(row, col)) for col in bt] for row in a]


def _softmax_row(row: list[float]) -> list[float]:
    max_v = max(row)
    exps = [exp(v - max_v) for v in row]
    total = sum(exps)
    return [v / total for v in exps]


def _kernel_softmax(x: Any, dim: int = -1) -> Any:
    if dim != -1:
        raise ValueError("Prototype softmax currently supports dim=-1 only")
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [_softmax_row(row) for row in x]
    if isinstance(x, list):
        return _softmax_row(x)
    return [1.0]


KERNELS = {
    "add": _kernel_add,
    "mul": _kernel_mul,
    "matmul": _kernel_matmul,
    "softmax": _kernel_softmax,
}


def _walk(output: Tensor) -> list[Tensor]:
    seen: set[int] = set()
    ordered: list[Tensor] = []

    def visit(tensor: Tensor) -> None:
        if id(tensor) in seen:
            return
        seen.add(id(tensor))
        for inp in tensor.graph_node.inputs:
            visit(inp)
        ordered.append(tensor)

    visit(output)
    return ordered


def tensor(data: Any, *, dtype: str = "float32", device: str | None = None) -> Tensor:
    node = GraphNode("const", attrs={"value": data})
    return Tensor(node, shape=_shape_of(data), dtype=dtype, device=device, value=data)


def _binary_tensor_op(op: str, a: Tensor, b: Tensor) -> Tensor:
    if not isinstance(a, Tensor) or not isinstance(b, Tensor):
        raise TypeError(f"{op} expects Tensor inputs")
    if a.shape != b.shape:
        raise ValueError(f"{op} expects matching shapes")
    node = GraphNode(op, inputs=[a, b])
    return Tensor(node, shape=a.shape, dtype=a.dtype)


def add(a: Tensor, b: Tensor) -> Tensor:
    return _binary_tensor_op("add", a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return _binary_tensor_op("mul", a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("Prototype matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul dimension mismatch")
    out_shape = (a.shape[0], b.shape[1])
    node = GraphNode("matmul", inputs=[a, b])
    return Tensor(node, shape=out_shape, dtype=a.dtype)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    node = GraphNode("softmax", inputs=[x], attrs={"dim": dim})
    return Tensor(node, shape=x.shape, dtype=x.dtype)
