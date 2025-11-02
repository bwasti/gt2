# Tensor API

GT provides a PyTorch-compatible tensor API for distributed operations.

## Creating Tensors

### Random Tensors

```python
import gt

# Random normal distribution
a = gt.randn(100, 100)

# Random uniform distribution [0, 1)
b = gt.rand(50, 50)

# Zeros
c = gt.zeros(10, 10)

# Ones
d = gt.ones(5, 5)
```

### From Existing Data

```python
import numpy as np

# From NumPy array
data = np.array([[1, 2], [3, 4]])
tensor = gt.from_numpy(data)

# Direct creation from Python list
tensor = gt.tensor([1, 2, 3, 4])
tensor = gt.tensor([[1, 2], [3, 4]])
```

### With Gradients

```python
# Enable gradient tracking
a = gt.randn(10, 10, requires_grad=True)
b = gt.from_numpy(data, requires_grad=True)
```

## Arithmetic Operations

### Element-wise Operations

```python
# Addition
c = a + b
c = a + 5  # Broadcasting

# Subtraction
c = a - b
c = a - 2

# Multiplication
c = a * b
c = a * 3

# Division
c = a / b
c = a / 2

# Power
c = a ** 2
```

### Matrix Operations

```python
# Matrix multiplication
c = a @ b          # Operator syntax
c = a.matmul(b)    # Method syntax

# Transpose
c = a.T
c = a.transpose()
```

## Activation Functions

```python
# ReLU (max(0, x))
y = x.relu()

# Sigmoid (1 / (1 + e^-x))
y = x.sigmoid()

# Tanh
y = x.tanh()
```

## Reduction Operations

```python
# Sum all elements
total = x.sum()

# Sum along axis
row_sums = x.sum(axis=0)    # Sum columns
col_sums = x.sum(axis=1)    # Sum rows

# Mean
avg = x.mean()
avg = x.mean(axis=0)

# Keep dimensions
y = x.sum(axis=0, keepdims=True)
```

## Math Functions

```python
# Exponential
y = x.exp()

# Natural logarithm
y = x.log()
```

## Shape Operations

### Reshaping

```python
# Reshape
x = gt.randn(100, 10)
y = x.reshape(10, 100)
y = x.reshape(1000)  # Flatten

# Add dimension
y = x.unsqueeze(0)   # Add batch dimension
y = x.unsqueeze(-1)  # Add trailing dimension

# Remove dimension
y = x.squeeze(0)     # Remove specific dimension
y = x.squeeze()      # Remove all size-1 dimensions
```

### Indexing and Slicing

```python
# Slicing
subset = x[0:10, :]     # First 10 rows
subset = x[:, 0:5]      # First 5 columns
subset = x[2:8, 1:4]    # Sub-matrix

# Single element
val = x[5, 3]
```

## In-Place Operations

```python
# In-place subtraction
a -= 0.01 * grad

# Zero tensor
a.zero_()
```

## Data Access

### Get NumPy Array

```python
# Fetch data from workers
numpy_array = tensor.data.numpy()

# Use in NumPy operations
result = np.sqrt(numpy_array)
```

### Printing

```python
# Automatically fetches and displays data
print(tensor)
```

## Comparison Operations

```python
# Element-wise comparison (returns boolean mask)
mask = x > 0
mask = x > threshold
```

## Properties

```python
# Check if requires gradient
if tensor.requires_grad:
    print("Gradients will be computed")

# Access gradient (after backward())
if tensor.grad is not None:
    grad_data = tensor.grad.data.numpy()
```

## PyTorch Compatibility

GT tensors follow PyTorch conventions:

| PyTorch | GT | Description |
|---------|-----|-------------|
| `torch.randn()` | `gt.randn()` | Random normal |
| `torch.zeros()` | `gt.zeros()` | Zero tensor |
| `torch.ones()` | `gt.ones()` | Ones tensor |
| `a @ b` | `a @ b` | Matrix multiply |
| `a.relu()` | `a.relu()` | ReLU activation |
| `a.sum()` | `a.sum()` | Reduction |
| `a.backward()` | `a.backward()` | Compute gradients |

## Complete Example

```python
import gt
import numpy as np

# Create tensors
a = gt.randn(1000, 1000)
b = gt.randn(1000, 1000)

# Operations
c = a @ b                    # Matrix multiply
d = c.relu()                 # Activation
e = d.sum()                  # Reduction

# Get result
result = e.data.numpy()
print(f"Result: {result}")

# With gradients
x = gt.randn(100, 10, requires_grad=True)
y = x.relu().sum()
y.backward()
print(f"Gradient shape: {x.grad.data.numpy().shape}")
```

## Next Steps

- [Autograd Guide](autograd.md) - Automatic differentiation
- [Neural Network Modules](../dispatcher/monitoring.md) - Building models
- [Sharding Configuration](../dispatcher/signaling.md) - Distributed tensors
