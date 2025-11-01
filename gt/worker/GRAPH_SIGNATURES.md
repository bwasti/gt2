# Graph-Based Instruction Signatures

## Problem

The original signature system only tracked operation types and input counts, not the actual dependency graph structure. This led to incorrect matching:

**WRONG:** These were treated as the same:
```
a = f(b, c)
e = g(a, d)  # e depends on a
```
vs
```
a = f(b, c)
e = g(b, c)  # e doesn't depend on a
```

## Solution

We now compute signatures based on the **dependency graph structure**, not just the operation sequence.

### Key Properties

1. **Dependency-aware**: Signatures capture which operations depend on which outputs
2. **Order-preserving**: Input order matters (e.g., `g(a,b)` ≠ `g(b,a)`)
3. **Tensor-agnostic**: Same structure with different tensor names matches (allows pattern reuse)

### How It Works

The `StreamSignature` class renormalizes tensor IDs to create a canonical representation:

```python
# Example: a = matmul(x, W); b = relu(a)

# Original IDs (arbitrary)
instructions = [
    ("result=tensor_123", "matmul", ["tensor_45", "tensor_67"]),
    ("result=tensor_456", "relu", ["tensor_123"]),
]

# Canonical form (structure-based)
canonical = [
    "t0=matmul(in0,in1)",  # First output, two external inputs
    "t1=relu(t0)",         # Second output, depends on first output
]

# Hash this canonical form → signature
```

### Canonicalization Algorithm

```python
for instruction in stream:
    # Renormalize inputs
    for input_id in instruction.inputs:
        if input_id not in mapping:
            mapping[input_id] = f"in{next_input}"  # External input
            next_input += 1
        canonical_inputs.append(mapping[input_id])

    # Renormalize output
    canonical_output = f"t{next_intermediate}"
    mapping[instruction.output] = canonical_output
    next_intermediate += 1

    # Create canonical op: "t0=op(in0,in1)"
    canonical_ops.append(f"{canonical_output}={op}({','.join(canonical_inputs)})")

# Hash the canonical representation
signature = hash("|".join(canonical_ops))
```

## Test Results

All critical tests pass:

✅ **Different dependencies** → Different signatures
```
a=f(b,c), e=g(a,d)  ≠  a=f(b,c), e=g(b,c)
```

✅ **Input order matters** → Different signatures
```
a=f(b), c=g(a,b)  ≠  a=f(b), c=g(b,a)
```

✅ **Same structure, different names** → Same signature
```
a=f(b,c), e=g(a,d)  ==  x=f(y,z), w=g(x,v)
```

✅ **Dependency chains** → Different from independent ops
```
a=f(x), b=g(a), c=h(b)  ≠  a=f(x), b=g(x), c=h(x)
```

## Future Work (TODO)

### Independent Subgraph Detection

Currently we detect hot paths in the instruction stream as-is. Future enhancement:

**Detect interleaved independent graphs:**
```python
a = f(b)      # Graph 1
x = g(y)      # Graph 2 (independent!)
c = h(a)      # Graph 1 continued
z = k(x)      # Graph 2 continued
```

**Optimization strategy:**
1. Detect the two graphs are independent (no shared tensors)
2. Analyze which graph(s) are hot
3. Re-sort stream to group operations: `[Graph 1 ops] [Graph 2 ops]`
4. Compile only hot graphs
5. Execute: `compiled_fn(b)` + `stream_execute(Graph 2)`

This maximizes compilation benefits without over-compiling cold paths.

### Implementation Steps

1. **Build dependency graph** for each sequence
2. **Partition into connected components** (independent subgraphs)
3. **Track hotness per subgraph** instead of per full sequence
4. **Re-sort instructions** to group hot subgraphs
5. **Generate compiled functions** for hot subgraphs only
6. **Interleave execution**: compiled functions + streaming for cold code

## Usage in Worker

```python
# Record instruction with tensor IDs
self.hotpath_detector.record_instruction(
    op_type='binary',
    op_name='matmul',
    result_id='tensor_123',
    input_ids=['tensor_45', 'tensor_67']  # ORDER MATTERS!
)

# Check if we should compile
is_hot, seq_length = self.hotpath_detector.record_instruction(...)
if is_hot:
    # This is a hot path, enable compilation
    enable_compile()
```

## References

- `gt/worker/hotpath_detector.py` - Implementation
- `gt/worker/worker.py` - Integration with worker
- Tests: `test_graph_signature.py` (removed after validation)
