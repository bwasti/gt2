# Worker Backends

GT workers support multiple computational backends for executing operations.

## Supported Backends

### PyTorch (Default)

**Features:**
- GPU acceleration via CUDA
- CPU execution (fallback when GPU unavailable)
- Automatic mixed precision
- Native support for autograd operations
- JIT compilation with `torch.compile`

**When to use:**
- Production workloads
- GPU training and inference
- Maximum performance

**Setup:**

Workers automatically use PyTorch backend by default:

```bash
# PyTorch backend (default)
python -m gt.worker --host localhost -p 9000

# Explicit backend specification
python -m gt.worker --host localhost -p 9000 --backend pytorch
```

### NumPy

**Features:**
- CPU-only execution
- Pure Python + NumPy operations
- No GPU dependencies
- Reference implementation

**When to use:**
- Testing and development
- Systems without GPU
- CPU-only workloads
- Debugging numerical correctness

**Setup:**

```bash
# NumPy backend
python -m gt.worker --host localhost -p 9000 --backend numpy
```

## Backend Comparison

| Feature | PyTorch | NumPy |
|---------|---------|-------|
| GPU Support | ✅ Yes | ❌ No |
| CPU Support | ✅ Yes | ✅ Yes |
| Performance | High | Medium |
| Compilation | ✅ Yes | ❌ No |
| Dependencies | PyTorch | NumPy only |
| Maturity | Production | Reference |

## GPU Configuration

### Assigning Workers to GPUs

Use `CUDA_VISIBLE_DEVICES` to assign each worker to a specific GPU:

```bash
# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 9000 &

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 9000 &

# Worker 2 on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m gt.worker --host localhost -p 9000 &

# Worker 3 on GPU 3
CUDA_VISIBLE_DEVICES=3 python -m gt.worker --host localhost -p 9000 &
```

**Best practice:** One worker per GPU for optimal performance.

### Multi-GPU Systems

For systems with multiple GPUs:

```bash
# Check available GPUs
nvidia-smi

# Start workers (one per GPU)
for i in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$i python -m gt.worker --host localhost -p 9000 &
done
```

### CPU-Only Execution

PyTorch workers automatically fall back to CPU when CUDA is unavailable:

```bash
# Force CPU execution
CUDA_VISIBLE_DEVICES="" python -m gt.worker --host localhost -p 9000
```

Or use NumPy backend:

```bash
python -m gt.worker --host localhost -p 9000 --backend numpy
```

## Backend Implementation

### Creating Custom Backends

GT's worker architecture supports custom backends. A backend must implement:

1. **Tensor creation** - `randn`, `zeros`, `ones`, `from_numpy`
2. **Binary operations** - `add`, `sub`, `mul`, `div`, `matmul`
3. **Unary operations** - `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sum`, `mean`
4. **Shape operations** - `reshape`, `transpose`, `unsqueeze`, `squeeze`
5. **Data access** - `to_numpy`

**Example skeleton:**

```python
class CustomBackend:
    def __init__(self):
        self.tensors = {}

    def create_tensor(self, tensor_id, data, dtype, shape):
        # Create tensor in backend format
        self.tensors[tensor_id] = ...

    def binary_op(self, result_id, op, left_id, right_id):
        # Execute binary operation
        left = self.tensors[left_id]
        right = self.tensors[right_id]

        if op == "add":
            result = left + right
        elif op == "matmul":
            result = left @ right
        # ... more operations

        self.tensors[result_id] = result

    def unary_op(self, result_id, op, input_id):
        # Execute unary operation
        input_tensor = self.tensors[input_id]

        if op == "relu":
            result = max(0, input_tensor)
        # ... more operations

        self.tensors[result_id] = result

    def get_data(self, tensor_id):
        # Return NumPy array
        return self.tensors[tensor_id].numpy()
```

See `gt/worker/engine/` for full implementations.

## Performance Considerations

### PyTorch Backend

**GPU Memory:**
- Pre-allocates CUDA memory
- Reuses allocations when possible
- May require tuning for large models

**Compilation:**
- Use `GT_AUTO_COMPILE=1` for hot path optimization
- First iteration slower (compilation overhead)
- Subsequent iterations much faster

**Optimization:**
```bash
# Enable compilation
GT_AUTO_COMPILE=1 python train.py

# Set memory growth (if needed)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### NumPy Backend

**CPU-Bound:**
- Single-threaded by default
- Use NumPy with MKL/OpenBLAS for multi-core
- Much slower than GPU for large operations

**Optimization:**
```bash
# Use optimized BLAS
export MKL_NUM_THREADS=8
export OMP_NUM_THREADS=8

python -m gt.worker --backend numpy --host localhost -p 9000
```

## Debugging Backend Issues

### Verify GPU Access

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
```

```bash
# From command line
python -c "import torch; print(torch.cuda.is_available())"
```

### Debug Worker Backend

```bash
# Enable worker debug output
GT_DEBUG_WORKER=1 python -m gt.worker --host localhost -p 9000
```

### Test Backend Operations

```python
# test_backend.py
import gt

# This uses workers with their configured backend
a = gt.randn(100, 100)
b = gt.randn(100, 100)
c = a @ b

print(f"Result shape: {c.data.numpy().shape}")
print("Backend working correctly!")
```

## Troubleshooting

### "CUDA out of memory"

**Solutions:**
1. Reduce batch size
2. Use model parallelism
3. Free unused tensors
4. Enable memory growth

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### "No CUDA GPUs detected"

**Check:**
```bash
# Verify drivers
nvidia-smi

# Verify PyTorch sees GPUs
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA_VISIBLE_DEVICES
echo $CUDA_VISIBLE_DEVICES
```

### Worker Crashes

**Debug:**
```bash
# Run with debugging
GT_DEBUG_WORKER=1 GT_VERBOSE=1 python -m gt.worker --host localhost -p 9000

# Check for OOM or other errors in output
```

## Next Steps

- [Compilation Guide](compilation.md) - JIT compilation with PyTorch backend
- [Tuning Guide](../dispatcher/tuning.md) - Performance optimization
- [Contributing](../contributing.md) - Add new backends
