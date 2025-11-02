# GT2 TODO

## Performance Optimizations

### High Priority

#### Client-Side Operation Batching
**Problem:** Training is 3-5x slower than PyTorch due to round-trip overhead.

**Analysis from profiling:**
- Training one epoch: 36,027 round trips over 4.6s
- Each operation (add, mul, matmul) is a separate round trip
- Average round-trip time: 0.15ms (ZMQ + pickle overhead)
- Bottleneck is round-trip count, not bandwidth (69MB in 4.6s = 15 MB/s)

**Current overhead breakdown:**
```
BinaryOp (matmul, add, mul): 12,786 ops × 0.15ms = 2.1s
FreeTensor (cleanup):        16,433 ops × 0.15ms = 2.5s
CreateTensor (scalars):       3,657 ops × 0.15ms = 0.5s
UnaryOp (zeros, etc):         2,450 ops × 0.15ms = 0.4s
GetData (.item() calls):        700 ops × 0.15ms = 0.1s
```

**Solution:** Implement client-side operation batching
- Accumulate operations in client before sending to dispatcher
- Flush batch on sync points (`.data`, `.backward()`, `.item()`)
- Similar to worker batching (GT_WORKER_BATCH_SIZE) but at client level
- Target: Reduce 12,786 BinaryOps to ~100 batches = 100x reduction in round trips

**Implementation notes:**
- Worker batching (`GT_WORKER_BATCH_SIZE`) exists but made things slower
- Need to investigate why worker batching didn't help
- May need lazy evaluation at client level to batch effectively
- Consider deferred execution model similar to TensorFlow 1.x

**Benefits:**
- Could reduce overhead from 4.6s to <0.5s per epoch
- Would make GT competitive with PyTorch for training

**Files to modify:**
- `gt/client/tensor.py` - Add batching accumulator
- `gt/client/client.py` - Add batch flush logic
- Consider adding `GT_CLIENT_BATCH_SIZE` environment variable

---

## Documentation

### Add Performance Tuning Guide
- Document instruction logging with message size tracking
- Explain bottlenecks (round trips vs bandwidth)
- Show how to profile with `GT_INSTRUCTION_LOG`
- Compare efficient vs inefficient patterns:
  - Efficient: `gt.randn(1000, 1000)` (137 bytes)
  - Inefficient: `gt.from_numpy(arr)` (3.81 MB)

---

## Future Work

### Shared Memory for Localhost
- Use shared memory for large tensor transfers on localhost
- Avoid serialization overhead for large tensors
- Could use `multiprocessing.shared_memory` or `/dev/shm`
- Would help with CreateTensor operations (currently 66MB in training)
