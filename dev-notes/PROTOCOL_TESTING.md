# Protocol Testing Guide

## Quick Smoke Test (Recommended)

**Before making protocol changes:**
```bash
python test_protocol_smoke.py
```

**After making protocol changes:**
```bash
python test_protocol_smoke.py
```

**What it tests:**
- ✅ Chain of 15 operations (matmul, add, mul, div, sub, transpose, sum)
- ✅ Backward pass with 10 forward operations
- ✅ Gradient computation
- ✅ Verifies exact numeric correctness against numpy

**Time:** < 2 seconds

---

## Full Test Suite (When Needed)

For comprehensive testing:

```bash
# Basic operations (fast)
pytest tests/test_basic_ops.py -v

# Numeric correctness (fast)
pytest tests/test_numerics.py -v

# Autograd (medium)
pytest tests/test_autograd.py -v

# All tests (slow - 30+ seconds)
pytest tests/ -v
```

---

## What to Test When Changing Protocol

### 1. Message Format Changes
- Run smoke test to verify serialization/deserialization
- Check that tensor IDs translate correctly

### 2. New Operation Types
- Add test case to `test_protocol_smoke.py`
- Verify against numpy reference

### 3. Worker Communication Changes
- Run smoke test (tests full client→dispatcher→worker flow)
- Check `test_multi_client.py` for concurrent access

### 4. Optimization/Performance Changes
- Smoke test verifies correctness
- Run benchmarks separately for performance:
  ```bash
  python benchmarks/compilation_benchmark.py
  ```

---

## Protocol Change Checklist

Before changing protocol:
- [ ] Run `python test_protocol_smoke.py` to get baseline
- [ ] Make your changes
- [ ] Run smoke test again - should produce **identical output**
- [ ] If adding new operations, add test case
- [ ] Run full test suite: `pytest tests/test_basic_ops.py tests/test_numerics.py -v`

---

## Quick Debug Commands

If smoke test fails:

```bash
# Enable debug output
GT_DEBUG_DISPATCHER=1 python test_protocol_smoke.py

# See instruction log
GT_INSTRUCTION_LOG=/tmp/protocol.log python test_protocol_smoke.py
cat /tmp/protocol.log | grep -E "RECV|SEND|ERROR"

# See worker debug
GT_DEBUG_WORKER=1 python test_protocol_smoke.py
```

---

## Expected Output

```
============================================================
PROTOCOL SMOKE TEST
============================================================
Running operation chain test...
GT result:       6831.0
Expected result: 6831.0
Difference:      0.0
✅ PASS - Operation chain matches expected result

Running backward chain test...
Loss value: -0.08046232908964157
Gradient 'a' shape: (4, 4)
Gradient 'b' shape: (4, 4)
✅ PASS - Backward chain works correctly

============================================================
✅ ALL TESTS PASSED
============================================================
```

**Key point:** The exact values may vary (random initialization), but **before/after protocol changes should match** if you use the same random seed.

---

## Adding New Operations to Smoke Test

Example - adding a new operation:

```python
# In test_operation_chain():
x = x.your_new_op()  # Add GT operation

# And in numpy reference:
x_np = x_np.your_new_op()  # Add numpy equivalent
```

Both should produce identical results (within floating-point tolerance).
