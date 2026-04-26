# NEF — Hydra Neural Essence Format
### Product Requirements Document · v0.1.0-draft

---

## 1. Overview

**NEF** (Neural Essence Format) is the portable computation and execution system for AI workloads inside **Hydra OS**. It is not a model format, not a training framework, and not a GPU driver abstraction. It is a **lazy computation graph system** — a complete pipeline from operator definition through device planning, kernel compilation, and hardware execution — targeting heterogeneous compute: CPU, GPU (NVIDIA / AMD / Intel), and NPU.

NEF's core contract to the developer is simple:

> *Write once. Run anywhere Hydra runs. No device management.*

---

## 2. Problem Statement

Running AI workloads across heterogeneous hardware today requires:

- Manual device placement (`tensor.to("cuda")`)
- Backend-specific code paths (CUDA vs. ROCm vs. CPU)
- Hand-tuned kernel selection
- Explicit memory management across devices

This is incompatible with Hydra OS's goal of transparent, scheduler-driven compute. NEF solves this by absorbing all hardware complexity below a single unified API surface.

---

## 3. Goals

### 3.1 Primary Goals

| Goal | Description |
|---|---|
| **Hardware Unification** | One execution API for CPU, CUDA GPU, ROCm GPU, Intel Arc GPU, and NPU |
| **Lazy Execution** | Ops build a computation graph; execution defers until `.execute()` or result demand |
| **Zero Device Management** | Developer never writes device placement code |
| **Automatic Optimization** | Graph optimizer fuses, folds, and simplifies before compilation |
| **Hydra-Native Scheduling** | NEF graphs are first-class citizens in `hydrad`'s scheduler |

### 3.2 Secondary Goals

- Sub-millisecond graph planning overhead
- Zero-copy memory movement where hardware permits
- Portable serialized graph format for deployment via Hydra registry
- Graceful CPU fallback for all operations

### 3.3 Non-Goals

NEF is **not**:
- A replacement for PyTorch / JAX in training workflows
- A low-level GPU driver or CUDA wrapper
- A model storage format (≠ GGUF, ONNX, SafeTensors)
- A distributed training coordinator

---

## 4. Architecture

```
┌──────────────────────────────────────────────────┐
│             NEF API  (Python / Go / C++)          │
└─────────────────────┬────────────────────────────┘
                       │
              ┌────────▼────────┐
              │  Graph Builder  │  ← Lazy IR / DAG
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │    Optimizer    │  ← Fusion, folding, simplification
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │  Device Planner │  ← Op → hardware assignment
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │ Kernel Compiler │  ← Backend-specific lowering
              └────┬──┬──┬──┬──┘
                   │  │  │  │
          ┌────────┘  │  │  └────────┐
          ▼           ▼  ▼           ▼
       CUDA GPU    ROCm  CPU SIMD   NPU
       (NVIDIA)   (AMD) (AVX-512) (Delegate)
          │           │  │           │
          └───────────┴──┴───────────┘
                       │
              ┌────────▼────────┐
              │Execution Runtime│  ← Async, parallel, streamed
              └─────────────────┘
```

---

## 5. Core Components

### 5.1 NEF API Layer

The developer-facing interface. Responsible for tensor creation, operator registration, and graph construction. No execution happens at this layer.

**Python Example:**
```python
import nef

a = nef.tensor([1, 2, 3], dtype=nef.float32)
b = nef.tensor([4, 5, 6], dtype=nef.float32)

c = nef.matmul(a, b)   # No execution yet — graph node created
c.execute()            # Graph is optimized, compiled, and run
```

**Go Example:**
```go
a := nef.Tensor([]float32{1, 2, 3})
b := nef.Tensor([]float32{4, 5, 6})
c := nef.MatMul(a, b)
c.Execute()
```

---

### 5.2 Graph Builder (IR Layer)

Converts API calls into an internal computation graph.

- **Structure:** Directed Acyclic Graph (DAG)
- **Nodes:** Individual ops (MatMul, Softmax, LayerNorm, etc.)
- **Edges:** Tensor dependencies between ops
- **Properties:** Node metadata — shape, dtype, estimated FLOPs, preferred device hint

The IR is device-agnostic at construction time. No hardware decisions are made here.

---

### 5.3 Optimizer Layer

Transforms the raw computation graph before compilation. Runs a sequence of passes:

| Pass | Description |
|---|---|
| **Node Fusion** | Fuse adjacent elementwise ops into single kernel |
| **Constant Folding** | Pre-compute static subgraphs at compile time |
| **Dead Node Elimination** | Remove ops whose output is never consumed |
| **Memory Reuse** | Identify tensors that can share memory buffers |
| **Operator Simplification** | Replace expensive ops with mathematically equivalent cheaper forms |

Optimization is deterministic and does not alter numerical output beyond floating-point rounding equivalence.

---

### 5.4 Device Planner

Assigns each graph node to a hardware target. Rules are heuristic-based with override support:

| Op Pattern | Default Target | Rationale |
|---|---|---|
| Large MatMul (≥ 1M params) | CUDA / ROCm GPU | Parallelism advantage |
| Transformer Attention | GPU / NPU | Memory bandwidth bound |
| Small elementwise ops | CPU SIMD | GPU launch overhead > compute cost |
| Quantized inference ops | NPU (if present) | NPU power efficiency |
| All else (fallback) | CPU SIMD | Correctness guarantee |

The planner also handles **placement boundaries** — inserting memory transfer nodes between ops assigned to different devices.

---

### 5.5 Kernel Compiler

Lowers abstract ops to backend-specific executable kernels.

**Supported backends:**

| Backend | Target Hardware | Compilation Path |
|---|---|---|
| CUDA | NVIDIA GPUs | PTX / cuBLAS / cuDNN |
| ROCm | AMD GPUs | HIP / hipBLAS |
| Level Zero / oneAPI | Intel Arc GPUs | SPIR-V |
| CPU SIMD | x86-64 / ARM | AVX2 / AVX-512 / NEON |
| NPU Delegate | Qualcomm / Apple ANE / etc. | Vendor SDK |

The compiler caches compiled kernels keyed by `(op_type, shape, dtype, backend)`. On re-execution with the same graph signature, no recompilation occurs.

---

### 5.6 Execution Runtime

Executes the compiled graph:

- **Parallel scheduling:** Independent graph branches execute concurrently
- **Async execution:** Non-blocking kernel dispatch with explicit synchronization barriers
- **Stream management:** Per-device CUDA/HIP streams; CPU thread pool
- **Memory coordination:** Automatic host↔device transfers at boundary nodes
- **Result materialization:** Tensors are pulled to CPU memory only when explicitly accessed

---

## 6. Lazy Execution Model

NEF follows a **define-then-run** model, identical in philosophy to JAX's JIT and MLX's lazy evaluation.

```
nef.matmul(a, b)   →  DAG node added. No compute.
nef.softmax(c)     →  DAG node added. No compute.
d.execute()        →  Full graph optimized, compiled, executed.
```

Execution is triggered by:
1. An explicit `.execute()` / `.eval()` call
2. A Python operation that requires a concrete value (e.g., `print(tensor)`, `numpy()`)
3. Hydra scheduler forcing materialization for downstream consumers

---

## 7. Tensor System

```python
class Tensor:
    shape: Tuple[int, ...]       # e.g. (1024, 4096)
    dtype: DType                 # float32 | float16 | bfloat16 | int8 | ...
    device: Optional[Device]     # None = auto-assigned by Device Planner
    graph_node: GraphNode        # Reference into the computation DAG
    _materialized: bool          # False until .execute() completes
```

Tensors are **views into the graph** until materialized. Slicing, reshaping, and transposing produce new graph nodes, not new memory allocations.

---

## 8. Memory Model

NEF provides a unified memory abstraction across all devices:

- **Automatic migration:** Tensors move between CPU/GPU/NPU as the planner requires
- **Zero-copy:** When hardware supports unified memory (e.g., Apple Silicon, some NVIDIA configs), transfers are avoided entirely
- **Intermediate caching:** Frequently accessed intermediate tensors are optionally pinned to avoid recomputation
- **Eviction policy:** LRU-based cache eviction under memory pressure, configurable per-device

---

## 9. Device Detection

At runtime initialization, NEF performs hardware enumeration:

```
NEF Device Scan
├── GPU: nvidia-smi / rocm-smi / level-zero enumeration
│   ├── VRAM capacity
│   ├── Compute capability / arch
│   └── NVLINK / PCIe topology
├── CPU: CPUID
│   ├── AVX / AVX2 / AVX-512 / VNNI support
│   └── Core count / NUMA topology
└── NPU: Vendor SDK probes
    ├── Qualcomm QNN
    ├── Apple ANE (via CoreML)
    └── Intel NPU (via OpenVINO)
```

Detection results are cached for the process lifetime. Manual overrides available via environment variables or `nef.config()`.

---

## 10. Backend Interface Contract

All backends implement the following interface (Go canonical form):

```go
type Backend interface {
    // Core linear algebra
    MatMul(a, b Tensor) Tensor
    Add(a, b Tensor) Tensor
    Mul(a, b Tensor) Tensor

    // Attention primitives
    Attention(q, k, v Tensor, mask *Tensor, scale float32) Tensor
    FlashAttention(q, k, v Tensor) Tensor  // optional, if supported

    // Activation functions
    Softmax(x Tensor, dim int) Tensor
    GELU(x Tensor) Tensor
    SiLU(x Tensor) Tensor

    // Normalization
    LayerNorm(x, weight, bias Tensor, eps float32) Tensor
    RMSNorm(x, weight Tensor, eps float32) Tensor

    // Memory ops
    Allocate(shape []int, dtype DType) Tensor
    Free(t Tensor)
    Transfer(t Tensor, dst Device) Tensor

    // Backend info
    Name() string
    Device() Device
    Capabilities() BackendCaps
}
```

Backends that don't support an op return `ErrNotSupported`; the Device Planner reroutes that node to the next eligible backend.

---

## 11. Execution Pipeline (End-to-End)

```
1. User Code
   └─ nef.matmul(a, b) → graph node

2. Lazy Graph Construction
   └─ DAG grows with each op call

3. .execute() triggered
   │
   ├─ 4. Optimizer passes (fusion, folding, elimination)
   │
   ├─ 5. Device Planner (op → hardware assignment)
   │
   ├─ 6. Kernel Compiler (abstract op → backend kernel)
   │      └─ Cache hit? → skip recompilation
   │
   └─ 7. Execution Runtime
          ├─ Async dispatch to device streams
          ├─ Memory transfers at device boundaries
          └─ Synchronization → result materialization
```

---

## 12. Hydra OS Integration

NEF is a first-class subsystem of Hydra OS, not a plugin.

```
hydra run model.nef
        │
        ▼
    hydrad daemon
        │
        ├─ Resource allocation (GPU slots, memory budget)
        ├─ NEF runtime initialization
        │       │
        │       ├─ Device detection
        │       ├─ Graph deserialization (if .nef file)
        │       └─ Execution dispatch
        │
        └─ Result → Hydra scheduler / output consumer
```

**Integration surface:**
- `hydrad` controls NEF process lifecycle and resource limits
- Hydra model registry stores serialized `.nef` graph files
- Hydra scheduler can preempt, pause, and resume NEF execution graphs
- NEF exposes a gRPC control interface consumed by `hydrad`

---

## 13. Serialized Graph Format (Optional)

NEF can optionally serialize a compiled graph to disk for deployment. This is not required for in-process execution.

**File: `model.nef`**
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
    "edges": [
      { "from": "node_0", "to": "node_1" }
    ]
  },
  "tensors": {
    "tensor_a": { "shape": [1024, 4096], "dtype": "float16" }
  },
  "target": "auto",
  "compiler_cache": "embedded"
}
```

The serialized format is used by the Hydra registry for model distribution and by `hydrad` for precompiled deployment.

---

## 14. Failure Handling

| Failure Mode | NEF Response |
|---|---|
| GPU out of memory | Evict cache → retry on smaller batch → fallback to CPU |
| Backend compilation failure | Log error → reroute to CPU fallback → continue |
| NPU probe failure at init | Silently disable NPU; proceed with GPU/CPU |
| Graph cycle detected | Raise `NEFGraphCycleError` at construction time |
| Partial graph failure | Rollback completed nodes; recompute from last checkpoint |
| `hydrad` preemption signal | Serialize in-progress graph state; resume on restart |

All failures are surfaced through structured error types. No silent corruption.

---

## 15. Security Model

- **No direct hardware access from user code.** All execution goes through `hydrad`.
- **Execution sandboxed** under Hydra's process isolation.
- **No arbitrary kernel injection.** Backend kernels are compiled from whitelisted op templates only.
- **Backend isolation layer:** User-defined ops cannot bypass the Backend interface contract.
- **Graph validation** runs before optimization to reject malformed or adversarially crafted graphs.

---

## 16. Performance Targets

| Metric | Target |
|---|---|
| Graph planning overhead | < 1ms for graphs ≤ 10K nodes |
| GPU utilization (LLM inference) | ≥ 85% |
| CPU fallback penalty vs GPU | ≤ 2× for ops ≤ 1M elements |
| Kernel cache hit rate (warm) | ≥ 95% |
| Memory transfer overhead | Zero-copy where hardware allows; otherwise < 5% of total exec time |

---

## 17. Future Extensions

| Feature | Priority | Notes |
|---|---|---|
| Distributed NEF (multi-node) | High | Tensor parallel + pipeline parallel across nodes |
| Streaming graphs | High | Real-time LLM token-by-token execution |
| Quantization-aware execution | Medium | INT4/INT8 graph lowering with calibration support |
| Federated compute | Low | Split execution across untrusted nodes |
| Dynamic shape support | Medium | Variable-length sequences without recompilation |
| WASM backend | Low | Browser / edge inference |

---

## 18. Success Criteria

NEF v1.0 is considered successful when:

- [ ] Same Python/Go code runs on CPU, CUDA GPU, ROCm GPU, and NPU without modification
- [ ] No developer-written device placement code in any test workload
- [ ] Full LLM inference (7B parameter transformer) runs end-to-end via NEF on GPU
- [ ] `hydrad` can schedule, preempt, and resume NEF workloads natively
- [ ] Kernel cache achieves ≥ 95% hit rate on warm re-execution
- [ ] GPU utilization ≥ 85% sustained on transformer attention workloads

---

## 19. Glossary

| Term | Definition |
|---|---|
| **DAG** | Directed Acyclic Graph — the internal graph structure NEF uses to represent computation |
| **IR** | Intermediate Representation — the hardware-agnostic op graph before compilation |
| **Lazy Execution** | Computation is deferred; ops define graph nodes and do not run until execution is triggered |
| **hydrad** | The Hydra OS daemon responsible for resource management and scheduling |
| **Backend** | A hardware-specific execution implementation (CUDA, ROCm, CPU SIMD, NPU) |
| **Kernel** | A compiled, backend-specific function that executes a single op on hardware |
| **Materialization** | The act of executing a lazy tensor graph and producing a concrete value in memory |

---

*NEF — Hydra Neural Essence Format · PRD v0.1.0-draft*
*Hexa / Hydra OS Internal · Not for external distribution*

---

## 20. Repository Build Status (Implemented)

This repository now includes a runnable NEF build pipeline in Python under `src/nef`:

- Lazy tensor graph construction (`nef.tensor`, `nef.add`, `nef.mul`, `nef.matmul`, `nef.softmax`)
- Deferred execution via `Tensor.execute()`, `Tensor.eval()`, and `Tensor.value()` (`Tensor.numpy()` is a stub until a NumPy backend is implemented)
- Graph optimizer + device planner + kernel-compiler stubs executed during `Tensor.build()` / `Tensor.execute()`
- Deterministic heuristic device planner with CPU default and GPU selection for large matmul workloads

### One-command setup

```bash
./setup_nef.sh
```

The setup script creates `.venv`, installs NEF with dev dependencies, and runs the test suite.

### Local build artifact & test

```bash
PYTHONPATH=src python -m pytest
PYTHONPATH=src python -c "import nef; a=nef.tensor([[1.0]],dtype=nef.float32); b=nef.tensor([[2.0]],dtype=nef.float32); nef.matmul(a,b).build('model.nef')"
```
