"""NEF lazy graph runtime with build/compile support (stdlib-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GraphNode:
    id: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)
    device: str | None = None


class Graph:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.dtypes: dict[str, str] = {}
        self._next_id = 0

    def add_node(
        self,
        op: str,
        *,
        inputs: list[str] | None = None,
        attrs: dict[str, Any] | None = None,
        shape: tuple[int, ...] = (),
        dtype: str = "float32",
    ) -> GraphNode:
        node_id = f"node_{self._next_id}"
        self._next_id += 1
        node = GraphNode(node_id, op, inputs=inputs or [], attrs=attrs or {})
        self.nodes[node_id] = node
        self.shapes[node_id] = shape
        self.dtypes[node_id] = dtype
        return node

    def topo(self, output_id: str) -> list[GraphNode]:
        ordered: list[GraphNode] = []
        seen: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in seen:
                return
            seen.add(node_id)
            node = self.nodes[node_id]
            for dep in node.inputs:
                visit(dep)
            ordered.append(node)

        visit(output_id)
        return ordered


class Tensor:
    """Lazy tensor view into a computation graph."""

    def __init__(
        self,
        graph: Graph,
        node_id: str,
        *,
        shape: tuple[int, ...],
        dtype: str,
        device: str | None = None,
        value: Any = None,
    ) -> None:
        self.graph = graph
        self.node_id = node_id
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._materialized = value is not None
        self._value = value

    def execute(self) -> "Tensor":
        if self._materialized:
            return self
        optimizer = GraphOptimizer()
        optimizer.run(self.graph, self.node_id)
        planner = DevicePlanner()
        planner.assign(self.graph, self.node_id)
        compiler = KernelCompiler()
        compiler.compile(self.graph, self.node_id)
        runtime = ExecutionRuntime()
        self._value = runtime.evaluate(self.graph, self.node_id)
        self._materialized = True
        self.device = self.graph.nodes[self.node_id].device
        return self

    def numpy(self) -> Any:
        self.execute()
        return self._value

    def build(self, path: str | Path) -> Path:
        """Build/compile graph and serialize to .nef artifact."""
        output = Path(path)
        optimizer = GraphOptimizer()
        optimizer.run(self.graph, self.node_id)
        planner = DevicePlanner()
        planner.assign(self.graph, self.node_id)
        compiler = KernelCompiler()
        compiler.compile(self.graph, self.node_id)

        nodes = []
        for n in self.graph.topo(self.node_id):
            nodes.append(
                {
                    "id": n.id,
                    "op": n.op,
                    "inputs": n.inputs,
                    "attrs": n.attrs,
                    "shape": list(self.graph.shapes[n.id]),
                    "dtype": self.graph.dtypes[n.id],
                    "assigned_device": n.device,
                }
            )

        payload = {
            "version": "0.1",
            "nef_format": "graph-v1",
            "output": self.node_id,
            "nodes": nodes,
        }
        output.write_text(json.dumps(payload, indent=2))
        return output

    def __repr__(self) -> str:
        if self._materialized:
            return f"Tensor(shape={self.shape}, dtype={self.dtype}, value={self._value!r})"
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, lazy_node={self.node_id!r})"


class GraphOptimizer:
    """Simple optimizer with constant folding."""

    def run(self, graph: Graph, output_id: str) -> None:
        for node in graph.topo(output_id):
            if node.op == "const":
                continue
            if not node.inputs:
                continue
            inputs = [graph.nodes[i] for i in node.inputs]
            if all(inp.op == "const" for inp in inputs):
                values = [inp.attrs["value"] for inp in inputs]
                folded = KERNELS[node.op](*values, **node.attrs)
                node.op = "const"
                node.inputs = []
                node.attrs = {"value": folded}


class DevicePlanner:
    """Heuristic planner stub that assigns symbolic devices."""

    def assign(self, graph: Graph, output_id: str) -> None:
        for node in graph.topo(output_id):
            if node.op in {"const", "add", "mul", "softmax"}:
                node.device = "cpu"
                continue

            shape = graph.shapes[node.id]
            elements = 1
            for dim in shape:
                elements *= max(dim, 1)

            if node.op == "matmul" and elements >= 1_000_000:
                node.device = "gpu"
            else:
                node.device = "cpu"


class KernelCompiler:
    """Compiler/cache stub keyed by op/shape/dtype/device."""

    _cache: set[tuple[str, tuple[int, ...], str, str]] = set()

    def compile(self, graph: Graph, output_id: str) -> None:
        for node in graph.topo(output_id):
            if node.op == "const":
                continue
            key = (
                node.op,
                graph.shapes[node.id],
                graph.dtypes[node.id],
                node.device or "cpu",
            )
            self._cache.add(key)


class ExecutionRuntime:
    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def evaluate(self, graph: Graph, output_id: str) -> Any:
        return self._eval_node(graph, output_id)

    def _eval_node(self, graph: Graph, node_id: str) -> Any:
        if node_id in self._cache:
            return self._cache[node_id]

        node = graph.nodes[node_id]
        if node.op == "const":
            result = node.attrs["value"]
        else:
            args = [self._eval_node(graph, dep) for dep in node.inputs]
            result = KERNELS[node.op](*args, **node.attrs)

        self._cache[node_id] = result
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


def tensor(data: Any, *, dtype: str = "float32", device: str | None = None) -> Tensor:
    graph = Graph()
    node = graph.add_node("const", attrs={"value": data}, shape=_shape_of(data), dtype=dtype)
    return Tensor(graph, node.id, shape=graph.shapes[node.id], dtype=dtype, device=device, value=data)


def _merge_graphs(a: Tensor, b: Tensor) -> Graph:
    graph = Graph()
    id_map: dict[tuple[int, str], str] = {}

    def import_node(src_graph: Graph, node_id: str) -> str:
        key = (id(src_graph), node_id)
        if key in id_map:
            return id_map[key]
        src = src_graph.nodes[node_id]
        imported_inputs = [import_node(src_graph, i) for i in src.inputs]
        new = graph.add_node(
            src.op,
            inputs=imported_inputs,
            attrs=dict(src.attrs),
            shape=src_graph.shapes[node_id],
            dtype=src_graph.dtypes[node_id],
        )
        new.device = src.device
        id_map[key] = new.id
        return new.id

    import_node(a.graph, a.node_id)
    import_node(b.graph, b.node_id)
    return graph, id_map


def _binary_tensor_op(op: str, a: Tensor, b: Tensor) -> Tensor:
    if a.shape != b.shape:
        raise ValueError(f"{op} expects matching shapes")
    graph, id_map = _merge_graphs(a, b)
    left_id = id_map[(id(a.graph), a.node_id)]
    right_id = id_map[(id(b.graph), b.node_id)]
    node = graph.add_node(op, inputs=[left_id, right_id], shape=a.shape, dtype=a.dtype)
    return Tensor(graph, node.id, shape=a.shape, dtype=a.dtype)


def add(a: Tensor, b: Tensor) -> Tensor:
    return _binary_tensor_op("add", a, b)


def mul(a: Tensor, b: Tensor) -> Tensor:
    return _binary_tensor_op("mul", a, b)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    if len(a.shape) != 2 or len(b.shape) != 2:
        raise ValueError("matmul expects 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("matmul dimension mismatch")
    graph, id_map = _merge_graphs(a, b)
    left_id = id_map[(id(a.graph), a.node_id)]
    right_id = id_map[(id(b.graph), b.node_id)]
    out_shape = (a.shape[0], b.shape[1])
    node = graph.add_node("matmul", inputs=[left_id, right_id], shape=out_shape, dtype=a.dtype)
    return Tensor(graph, node.id, shape=out_shape, dtype=a.dtype)


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    graph, id_map = _merge_graphs(x, x)
    x_id = id_map[(id(x.graph), x.node_id)]
    node = graph.add_node("softmax", inputs=[x_id], attrs={"dim": dim}, shape=x.shape, dtype=x.dtype)
    return Tensor(graph, node.id, shape=x.shape, dtype=x.dtype)
