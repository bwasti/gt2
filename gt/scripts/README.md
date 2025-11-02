# GT Scripts

Utility scripts for GT framework analysis and visualization.

## Visualize - Instruction Tape Timeline Visualizer

Generate high-resolution timeline visualizations showing instruction flow from client through dispatcher to workers.

### Usage

```bash
# 1. Generate a tape log by setting GT_INSTRUCTION_LOG environment variable
GT_INSTRUCTION_LOG=/tmp/debug.log python your_script.py

# 2. Visualize the tape
python -m gt.scripts.visualize /tmp/debug.log --output timeline.png --dpi 200
```

### Options

- `<tape_log_path>` - Path to the instruction tape log file (required)
- `--output <path>` - Output image path (default: `tape_timeline.png`)
- `--dpi <dpi>` - Image resolution in DPI (default: 150)

### Example

```python
# example_workload.py
import gt

# Auto-start GT system
a = gt.randn(100, 100)
b = gt.randn(100, 100)

# Operations
c = a @ b
d = c.relu()
result = d.sum().item()

print(f"Result: {result}")
```

```bash
# Generate tape log
GT_INSTRUCTION_LOG=/tmp/workload.log python example_workload.py

# Visualize with high resolution
python -m gt.scripts.visualize /tmp/workload.log --output workload_timeline.png --dpi 300
```

### Visualization Features

The timeline shows:

**Y-axis (Lanes):**
- **Client** - Client-side requests
- **Dispatcher** - Instruction dispatch and coordination
- **Worker N** - Individual worker execution (one lane per worker)

**X-axis:** Time in seconds since dispatcher start

**Visual Elements:**
- **Colored markers** - Operation types (MatMul=red, BinaryOp=blue, ReLU=dark gray, etc.)
- **Marker shapes** - Event types (○=RECV, >=WORKER_SEND, <=WORKER_RECV, *=SIGNAL)
- **Marker size** - Data size (larger = more data transferred)
- **Arrows** - Communication flow between components
- **Instruction IDs** - Sequence numbers for key operations
- **Operation labels** - Short labels showing operation names

**Signal Overlays:**
- **Red vertical lines** - GT signals (when instrumented in code)
- Useful for correlating tape activity with application-level events

### Understanding the Timeline

#### Embarrassingly Parallel Pattern
```
Client:     ●●●                              ●●
Dispatcher: ●●● → → → →                      ●●
Worker 0:       ▶◀ ▶◀                           ▶◀
Worker 1:       ▶◀ ▶◀                           ▶◀
Worker 2:       ▶◀ ▶◀                           ▶◀
Worker 3:       ▶◀ ▶◀                           ▶◀
```
All workers process independently, no cross-worker communication.

#### All-Gather Pattern
```
Client:     ●●                               ●
Dispatcher: ●●  → → → →  (gather) → →        ●
Worker 0:      ▶◀                   ▶◀         ▶◀
Worker 1:      ▶◀         →→→→→    ▶◀         ▶◀
Worker 2:      ▶◀         ←←←←←    ▶◀         ▶◀
Worker 3:      ▶◀                   ▶◀         ▶◀
```
Workers communicate to gather distributed data before computation.

#### All-Reduce Pattern
```
Client:     ●●                          ●
Dispatcher: ●●  → → → →     (reduce)    ●
Worker 0:      ▶◀       →→→→→             ▶◀
Worker 1:      ▶◀       ←←←←←             ▶◀
Worker 2:      ▶◀       →→→→→             ▶◀
Worker 3:      ▶◀       ←←←←←             ▶◀
```
Workers reduce results through cross-worker communication.

### Color Legend

| Operation | Color |
|-----------|-------|
| MatMul | Red |
| BinaryOp | Blue |
| UnaryOp | Green |
| ReLU | Dark Gray |
| Sum | Purple |
| Mean | Teal |
| GetData | Dark Orange |
| AllGather | Dark Red |
| FreeTensor | Light Gray |

### Tips for Analysis

1. **Identify Bottlenecks:** Look for gaps where workers are idle
2. **Communication Overhead:** Large arrows indicate data transfer
3. **Worker Balance:** Check if work is evenly distributed across workers
4. **Instruction Density:** Dense clusters indicate compute-intensive periods
5. **Data Movement:** Track when tensors move between workers (Move operations)

### Troubleshooting

**No events in visualization:**
- Check that `GT_INSTRUCTION_LOG` was set when running your script
- Verify the log file exists and has content: `cat /tmp/debug.log`

**Only one worker shown:**
- Script may be using auto-server mode (single worker)
- For multi-worker visualization, manually start server + workers

**Workers not distinguished:**
- Ensure workers register with unique names
- Check log file has "WORKER worker_0", "WORKER worker_1", etc.

### Advanced: Instrumenting with Signals

Add markers to correlate application events with tape activity:

```python
import gt
from gt.signals import signal

# Your code
a = gt.randn(1000, 1000)

signal("START_FORWARD")  # Mark forward pass start
b = model.forward(a)
signal("END_FORWARD")

signal("START_BACKWARD")  # Mark backward pass start
loss = compute_loss(b)
loss.backward()
signal("END_BACKWARD")
```

Signals appear as red vertical lines in the visualization, making it easy to see what tape activity corresponds to each phase of your application.

### Output

The visualizer generates a high-resolution PNG with:
- **File size:** Typically 1-10 MB depending on complexity
- **Resolution:** Configurable via `--dpi` (150-300 recommended)
- **Dimensions:** Auto-scaled based on duration and number of workers
- **Format:** PNG with white background (suitable for papers/presentations)

Example stats output:
```
✓ Visualization saved to: timeline.png
  Resolution: 4000 x 2400 pixels
  Events: 156
  Workers: 4
  Duration: 2.341s
```
