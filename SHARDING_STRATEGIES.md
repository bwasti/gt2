# GT Sharding Strategies

This document explains how distributed matrix operations work in GT.

## Strategy 1: Row-Sharded Matmul (Current Implementation)

**Use case**: `A @ B` where A is row-sharded

### Sharding Pattern
```
A (128, 64) sharded along axis 0 (rows)
→ Worker 0: A₀ (32, 64)  [rows 0-31]
→ Worker 1: A₁ (32, 64)  [rows 32-63]
→ Worker 2: A₂ (32, 64)  [rows 64-95]
→ Worker 3: A₃ (32, 64)  [rows 96-127]

B (64, 32) broadcast to all workers
→ All workers: B (64, 32)
```

### Computation
Each worker computes independently:
```
Worker 0: C₀ = A₀ @ B = (32, 64) @ (64, 32) = (32, 32)
Worker 1: C₁ = A₁ @ B = (32, 64) @ (64, 32) = (32, 32)
Worker 2: C₂ = A₂ @ B = (32, 64) @ (64, 32) = (32, 32)
Worker 3: C₃ = A₃ @ B = (32, 64) @ (64, 32) = (32, 32)
```

### Communication
**Gather** (concatenate):
```
C = [C₀; C₁; C₂; C₃] = (128, 32)
```

**No all-reduce needed!** Each worker computed a different part of the output.

### Implementation
- Broadcast B to all workers (one-time cost)
- Each worker computes local matmul
- Dispatcher gathers results by concatenating along axis 0
- See: `_handle_distributed_matmul()` in `dispatcher.py`

---

## Strategy 2: Reduction-Dimension Sharding (Requires All-Reduce)

**Use case**: `sum(A)` or `mean(A)` where A is sharded

### Example: sum() on Sharded Tensor

```
A (128, 64) sharded along axis 0
→ Worker 0: A₀ (32, 64)
→ Worker 1: A₁ (32, 64)
→ Worker 2: A₂ (32, 64)
→ Worker 3: A₃ (32, 64)
```

### Computation
Each worker computes partial sum:
```
Worker 0: s₀ = sum(A₀) = scalar
Worker 1: s₁ = sum(A₁) = scalar
Worker 2: s₂ = sum(A₂) = scalar
Worker 3: s₃ = sum(A₃) = scalar
```

### Communication
**All-reduce (sum)**:
```
result = s₀ + s₁ + s₂ + s₃
```

All workers receive the final result.

### Why All-Reduce?
- Each worker has only **part** of the data
- The final result depends on **combining** all partial results
- Unlike matmul, we can't just concatenate - we need to SUM

---

## Strategy 3: Column-Sharded Matmul (Future - Also Needs All-Reduce)

**Use case**: `A @ B` where B is column-sharded along inner dimension

### Sharding Pattern
```
A (128, 64) sharded along axis 1 (columns)
→ Worker 0: A₀ (128, 16)  [cols 0-15]
→ Worker 1: A₁ (128, 16)  [cols 16-31]
→ Worker 2: A₂ (128, 16)  [cols 32-47]
→ Worker 3: A₃ (128, 16)  [cols 48-63]

B (64, 32) sharded along axis 0 (rows - same as A's columns!)
→ Worker 0: B₀ (16, 32)  [rows 0-15]
→ Worker 1: B₁ (16, 32)  [rows 16-31]
→ Worker 2: B₂ (16, 32)  [rows 32-47]
→ Worker 3: B₃ (16, 32)  [rows 48-63]
```

### Computation
Each worker computes partial matmul:
```
Worker 0: C₀ = A₀ @ B₀ = (128, 16) @ (16, 32) = (128, 32)
Worker 1: C₁ = A₁ @ B₁ = (128, 16) @ (16, 32) = (128, 32)
Worker 2: C₂ = A₂ @ B₂ = (128, 16) @ (16, 32) = (128, 32)
Worker 3: C₃ = A₃ @ B₃ = (128, 16) @ (16, 32) = (128, 32)
```

### Communication
**All-reduce (sum)**:
```
C = C₀ + C₁ + C₂ + C₃ = (128, 32)
```

This is the **reduction over the inner dimension (k)**!

---

## Implementation Status

✅ **Implemented**:
- Row-sharded matmul (Strategy 1)
- Auto-sharding for tensor creation (randn, zeros)
- Gather-based result collection

🚧 **In Progress**:
- Engine abstraction (numpy vs pytorch)
- Distributed sum/mean with all-reduce

⏳ **TODO**:
- Column-sharded matmul (Strategy 3)
- PyTorch distributed backend integration (NCCL)
- Automatic sharding strategy selection

---

## When to Use Each Strategy

| Operation | Input Sharding | Strategy | Communication |
|-----------|---------------|----------|---------------|
| `A @ B` | A row-sharded | 1 (current) | Broadcast B, Gather result |
| `sum(A)`, `mean(A)` | A sharded any axis | 2 (in progress) | All-reduce sum |
| `A @ B` | Both col-sharded on inner dim | 3 (future) | All-reduce sum |

---

## PyTorch Distributed Primitives

The `PyTorchEngine` provides distributed operations:

```python
# All-reduce sum (Strategy 2 & 3)
tensor = engine.all_reduce_sum(tensor, group=None)

# Future: other collectives
tensor = engine.all_gather(tensor, group=None)  # Gather to all workers
tensor = engine.reduce_scatter(tensor, group=None)  # Reduce and scatter
```

Only PyTorch engine supports distributed ops (`engine.supports_distributed() == True`).

NumPy engine will raise `NotImplementedError` for distributed operations.
