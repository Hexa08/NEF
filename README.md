<div align="center">

<br/>

```
███╗   ██╗███████╗███████╗
████╗  ██║██╔════╝██╔════╝
██╔██╗ ██║█████╗  █████╗  
██║╚██╗██║██╔══╝  ██╔══╝  
██║ ╚████║███████╗██║     
╚═╝  ╚═══╝╚══════╝╚═╝     
```

### **Neural Essence Format**
*The portable computation graph engine for AI workloads inside Hydra OS*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-00e5ff?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/Version-0.1.0--draft-f5c842?style=flat-square)](https://github.com/Hexa08/NEF)
[![Status](https://img.shields.io/badge/Status-Active%20Development-00ff88?style=flat-square)](https://github.com/Hexa08/NEF)
[![PRD](https://img.shields.io/badge/Spec-PRD%20v0.1-a855f7?style=flat-square)](https://github.com/Hexa08/NEF/blob/main/README.md)
[![Hydra OS](https://img.shields.io/badge/Runtime-Hydra%20OS-ff4455?style=flat-square)](https://github.com/Hexa08)

<br/>

> **Write once. Run anywhere Hydra runs. No device management.**

<br/>

</div>

---

## What is NEF?

NEF is **not** a model format, training framework, or GPU driver wrapper.

It is a **lazy computation graph system** — a complete pipeline from operator definition through device planning, kernel compilation, and hardware execution — targeting heterogeneous compute across CPU, GPU (NVIDIA / AMD / Intel), and NPU, all from a single unified API.

---

## The Problem It Solves

Running AI workloads on heterogeneous hardware today means writing this kind of code:

```python
# Without NEF — you manage everything manually
tensor = tensor.to("cuda:0")                         # device hell
if torch.cuda.is_available():
    kernel = cuda_kernel(tensor)                     # backend-specific paths
elif rocm_available():
    kernel = rocm_kernel(tensor)                     # more branching
memory_pool.pin(tensor)                              # manual memory
torch.cuda.synchronize()                             # explicit sync
```

**NEF eliminates all of it:**

```python
import nef

a = nef.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=nef.float32)
b = nef.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=nef.float32)

c = nef.matmul(a, b)   # ← no execution yet. graph node created.
c.execute()            # ← optimizer → planner → compiler → hardware. done.
```

No `tensor.to("cuda")`. No backend conditionals. No memory calls.

---

## Architecture

```
┌─────────────────────────────────────────┐
│          NEF API  (Python / Go)         │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────▼─────────┐
         │   Graph Builder   │  ←  Lazy DAG / IR
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │     Optimizer     │  ←  Fusion · Folding · Elimination
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │   Device Planner  │  ←  Op → Hardware assignment
         └─────────┬─────────┘
                   │
         ┌─────────▼─────────┐
         │  Kernel Compiler  │  ←  Backend-specific lowering
         └──┬──┬──┬──┬───────┘
            │  │  │  │
   ┌────────┘  │  │  └────────┐
   ▼           ▼  ▼           ▼
NVIDIA GPU   AMD  CPU SIMD   NPU
(CUDA/PTX) (ROCm)(AVX-512) (Vendor)
   │           │  │           │
   └───────────┴──┴───────────┘
                   │
         ┌─────────▼─────────┐
         │ Execution Runtime │  ←  Async · Parallel · Streamed
         └───────────────────┘
```

---

## Core Components

<details>
<summary><strong>① Graph Builder — Lazy IR Layer</strong></summary>

<br/>

Converts API calls into a **Directed Acyclic Graph (DAG)**. No hardware decisions happen here. No execution happens here. Every op call simply extends the graph.

- **Nodes** — individual ops (MatMul, Softmax, LayerNorm, RMSNorm …)  
- **Edges** — tensor dependencies between nodes  
- **Metadata** — shape, dtype, estimated FLOPs, device hint  

```python
a = nef.tensor([1, 2, 3])
b = nef.tensor([4, 5, 6])
c = nef.matmul(a, b)    # → DAG node added. Nothing ran.
d = nef.softmax(c)      # → DAG node added. Nothing ran.
                        # Graph: a,b → matmul → softmax → d
```

</details>

<details>
<summary><strong>② Optimizer — Graph Transformation Passes</strong></summary>

<br/>

Runs a deterministic sequence of passes before compilation. Does not alter numerical output beyond floating-point rounding equivalence.

| Pass | What it does |
|---|---|
| **Node Fusion** | Adjacent elementwise ops collapse into a single kernel |
| **Constant Folding** | Static subgraphs computed at compile time |
| **Dead Node Elimination** | Unreachable nodes removed from graph |
| **Memory Reuse** | Tensors that can share buffers are identified |
| **Op Simplification** | Expensive ops replaced with cheaper equivalents |

</details>

<details>
<summary><strong>③ Device Planner — Hardware Assignment</strong></summary>

<br/>

Maps each graph node to the best available hardware target. Heuristic-driven, with developer override support.

| Op Pattern | Default Target | Why |
|---|---|---|
| Large MatMul (≥ 1M params) | CUDA / ROCm GPU | Parallelism |
| Transformer Attention | GPU / NPU | Memory-bandwidth bound |
| Small elementwise ops | CPU SIMD | GPU launch overhead > cost |
| Quantized ops | NPU (if present) | Power efficiency |
| Everything else | CPU SIMD | Correctness fallback |

Inserts **memory transfer nodes** automatically at device boundaries.

</details>

<details>
<summary><strong>④ Kernel Compiler — Backend Lowering</strong></summary>

<br/>

Lowers abstract ops to backend-specific executable kernels. Results are cached by `(op_type, shape, dtype, backend)` — warm re-execution skips compilation entirely.

| Backend | Target | Compilation path |
|---|---|---|
| CUDA | NVIDIA GPUs | PTX / cuBLAS / cuDNN |
| ROCm | AMD GPUs | HIP / hipBLAS |
| Level Zero | Intel Arc GPUs | SPIR-V |
| CPU SIMD | x86-64 / ARM | AVX2 / AVX-512 / NEON |
| NPU Delegate | Qualcomm / Apple ANE | Vendor SDK |

</details>

<details>
<summary><strong>⑤ Execution Runtime — Async Graph Dispatch</strong></summary>

<br/>

- **Parallel scheduling** — independent branches execute concurrently  
- **Async dispatch** — non-blocking kernel launch with explicit sync barriers  
- **Stream management** — per-device CUDA/HIP streams; CPU thread pool  
- **Memory coordination** — host↔device transfers inserted at boundary nodes  
- **Materialization** — tensors pulled to CPU memory only when accessed  

</details>

---

## Lazy Execution Model

NEF follows **define-then-run**, identical in philosophy to JAX JIT and MLX lazy evaluation.

```
nef.matmul(a, b)    →  graph node added.  zero compute.
nef.softmax(c)      →  graph node added.  zero compute.
nef.layernorm(d)    →  graph node added.  zero compute.
result.execute()    →  full graph: optimized → compiled → dispatched.
```

Execution triggers:

1. An explicit `.execute()` or `.eval()` call
2. A Python operation requiring a concrete value (`print(t)`, `t.numpy()`)
3. Hydra scheduler forcing materialization for downstream consumers

---

## Supported Hardware

<div align="center">

| Platform | Backend | Status |
|:---:|:---:|:---:|
| 🟢 NVIDIA GPU | CUDA / PTX / cuBLAS | `active` |
| 🟢 AMD GPU | ROCm / HIP / hipBLAS | `active` |
| 🟡 Intel Arc GPU | Level Zero / SPIR-V | `planned` |
| 🟢 CPU (x86-64) | AVX2 / AVX-512 / VNNI | `active` |
| 🟢 CPU (ARM) | NEON / SVE | `active` |
| 🟡 NPU | Qualcomm QNN / Apple ANE | `planned` |
| ⚪ WASM | Browser / Edge | `future` |

</div>

---

## Performance Targets

<div align="center">

| Metric | Target |
|:---|:---|
| Graph planning overhead | `< 1ms` for graphs ≤ 10K nodes |
| GPU utilization (LLM inference) | `≥ 85%` sustained |
| Kernel cache hit rate (warm) | `≥ 95%` |
| CPU fallback penalty vs GPU | `≤ 2×` for ops ≤ 1M elements |
| Memory transfer overhead | Zero-copy where supported; otherwise `< 5%` of total |

</div>

---

## Go API

```go
import "github.com/Hexa08/NEF"

a := nef.Tensor([]float32{1, 2, 3, 4}, []int{2, 2})
b := nef.Tensor([]float32{5, 6, 7, 8}, []int{2, 2})

c := nef.MatMul(a, b)
c.Execute()

fmt.Println(c.Numpy())
```

---

## Serialized Graph Format

NEF graphs can be saved to `.nef` files for deployment via the Hydra registry.

```json
{
  "version": "1.0",
  "nef_format": "graph-v1",
  "graph": {
    "nodes": [
      {
        "id": "node_0",
        "op": "matmul",
        "inputs": ["tensor_a", "tensor_b"],
        "output": "tensor_c",
        "preferred_device": "gpu"
      }
    ],
    "edges": [{ "from": "node_0", "to": "node_1" }]
  },
  "tensors": {
    "tensor_a": { "shape": [1024, 4096], "dtype": "float16" }
  },
  "target": "auto",
  "compiler_cache": "embedded"
}
```

Deploy with:

```bash
hydra run model.nef
```

---

## Hydra OS Integration

NEF is a first-class subsystem of Hydra OS — not a plugin.

```
hydra run model.nef
        │
        ▼
    hydrad daemon
        ├── resource allocation  (GPU slots · memory budget)
        ├── NEF runtime init     (device detection · graph deserialization)
        ├── execution dispatch   (async graph scheduling)
        └── result → Hydra scheduler / output consumer
```

`hydrad` controls NEF lifecycle. NEF exposes a **gRPC control interface** consumed by the scheduler. Graphs can be **preempted, paused, and resumed** mid-execution.

---

## Failure Handling

| Failure | Response |
|---|---|
| GPU out of memory | Evict cache → retry on smaller batch → CPU fallback |
| Backend compile failure | Log → reroute to CPU → continue |
| NPU probe failure | Silently disable NPU; proceed with GPU/CPU |
| Graph cycle detected | `NEFGraphCycleError` raised at construction time |
| `hydrad` preemption | Serialize in-progress graph state; resume on restart |

No silent corruption. All failures surface through structured error types.

---

## Quick Start

```bash
git clone https://github.com/Hexa08/NEF
cd NEF
pip install -e .
```

Run the test suite:

```bash
PYTHONPATH=src python -m pytest
```

Build a `.nef` file:

```bash
PYTHONPATH=src python -c "
import nef
a = nef.tensor([[1.0]], dtype=nef.float32)
b = nef.tensor([[2.0]], dtype=nef.float32)
nef.matmul(a, b).build('model.nef')
"
```

---

## Roadmap

- [x] Lazy tensor graph construction
- [x] Deferred execution via `.execute()` / `.numpy()`
- [x] Graph optimizer + device planner stubs
- [x] Kernel compiler pipeline (CPU path)
- [ ] CUDA backend integration
- [ ] ROCm backend integration
- [ ] Distributed NEF (multi-node tensor/pipeline parallel)
- [ ] Streaming graphs (real-time token-by-token LLM execution)
- [ ] Quantization-aware execution (INT4 / INT8)
- [ ] Dynamic shape support (variable-length sequences)
- [ ] WASM backend (browser / edge inference)

---

## What NEF Is NOT

> Clarifying the scope to avoid confusion.

- ❌ Not a replacement for PyTorch / JAX in training workflows
- ❌ Not a low-level GPU driver or CUDA wrapper
- ❌ Not a model storage format (≠ GGUF, ONNX, SafeTensors)
- ❌ Not a distributed training coordinator

NEF is the **execution layer** — everything below the graph, everything above the hardware.

---

## Security

- No direct hardware access from user code — all execution routes through `hydrad`
- Execution sandboxed under Hydra's process isolation
- No arbitrary kernel injection — kernels compiled from whitelisted op templates only
- Graph validation runs before optimization; malformed graphs are rejected at construction

---

<div align="center">

<br/>

**NEF — Neural Essence Format**  
`v0.1.0-draft` · HydraLogOS  Internal · [github.com/Hexa08/NEF](https://github.com/Hexa08/NEF)

<br/>

*Built for Hydra OS. Designed to disappear into the hardware.*

<br/>

</div>
