# GT Examples

Examples demonstrating GT features and usage patterns.

## Quick Start: `gt.gpu_workers()` API (`simple_gpu_workers.py`)

**The easiest way to use multiple GPUs!** Just one line before any tensor operations:

```python
import gt
gt.gpu_workers(4)  # Use 4 GPUs

a = gt.randn(128, 64)  # Auto-sharded across 4 workers
b = gt.tensor(b_data)  # Broadcast to all workers
c = a @ b              # Distributed computation
```

**Run it:**
```bash
python examples/simple_gpu_workers.py
```

**What it does:**
- Automatically starts local dispatcher
- Spawns 4 workers (one per GPU)
- Enables automatic tensor sharding
- No manual worker setup required!

**Note:** For production with remote workers, use manual setup (see below).

---

## Simple Auto-Shard (`simple_auto_shard.py`)

Demonstrates automatic tensor sharding across multiple workers for distributed matrix multiplication.

### Running with Auto-Connect (1 worker, no sharding)

```bash
python examples/simple_auto_shard.py
```

This uses auto-connect mode which starts a local server with **1 worker only**. No sharding occurs.

### Running with Multiple Workers (distributed sharding)

**Option 1: Server with auto-spawned workers (easiest)**

```bash
# Terminal 1 - Start server with 4 workers
python -m gt.server -p 12345 --spawn_gpu_workers 4

# Terminal 2 - Run example
python examples/simple_auto_shard.py --distributed
```

**Option 2: Manual worker setup (more control)**

**Terminal 1 - Start Server:**
```bash
python -m gt.server -p 12345
```

**Terminal 2-5 - Start 4 Workers (one per GPU):**
```bash
# Worker 0 on GPU 0
CUDA_VISIBLE_DEVICES=0 python -m gt.worker --host localhost -p 12345

# Worker 1 on GPU 1
CUDA_VISIBLE_DEVICES=1 python -m gt.worker --host localhost -p 12345

# Worker 2 on GPU 2
CUDA_VISIBLE_DEVICES=2 python -m gt.worker --host localhost -p 12345

# Worker 3 on GPU 3
CUDA_VISIBLE_DEVICES=3 python -m gt.worker --host localhost -p 12345
```

**Terminal 6 - Run Example:**
```bash
python examples/simple_auto_shard.py --distributed
```

### What Happens with Sharding

With 4 workers, tensors are automatically sharded:

```python
a = gt.randn(128, 64)  # Sharded as (32, 64) on each of 4 workers
b = gt.tensor(b_data)  # Stays on 1 worker (created from numpy)
c = a @ b              # Distributed: each worker computes A_shard @ B
```

**Result:**
- Each worker computes `(32, 64) @ (64, 32) = (32, 32)` in parallel
- Dispatcher automatically gathers: 4 × `(32, 32)` → `(128, 32)`

### Key Insights

**Auto-Connect Limitations:**
- Only starts 1 worker (no GPU detection)
- No automatic sharding
- Good for: local development, debugging, small experiments

**Manual Worker Setup (Recommended for Production):**
- Start 1 worker per GPU
- Full control over GPU assignment (`CUDA_VISIBLE_DEVICES`)
- Automatic sharding across all connected workers
- Good for: production, distributed training, large-scale experiments

**Sharding Heuristics:**
- Tensors with 2D shape where `dim[0] % num_workers == 0` are auto-sharded
- Sharding happens along axis 0 (rows)
- Tensors created from `from_numpy()` are **not** auto-sharded
- Created tensors (`randn`, `zeros`, etc.) **are** auto-sharded
