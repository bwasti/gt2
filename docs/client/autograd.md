# Autograd

GT implements tape-based automatic differentiation for computing gradients.

## Overview

GT's autograd system works like PyTorch:
- Operations are recorded on a computational graph (tape)
- Calling `.backward()` computes gradients via reverse-mode differentiation
- Gradients are stored in `.grad` attribute of tensors

## Enabling Gradients

Mark tensors that need gradients:

```python
import gt

# Create tensor with gradient tracking
a = gt.randn(10, 10, requires_grad=True)

# From NumPy with gradients
import numpy as np
data = np.random.randn(5, 5)
b = gt.from_numpy(data, requires_grad=True)
```

## Computing Gradients

### Basic Example

```python
# Forward pass
x = gt.randn(10, 10, requires_grad=True)
y = x * 2
loss = y.sum()

# Backward pass
loss.backward()

# Access gradients
print(x.grad.data.numpy())  # All values are 2.0
```

### Multi-Step Computation

```python
# More complex computation
a = gt.randn(100, 50, requires_grad=True)
b = gt.randn(50, 20, requires_grad=True)

# Forward
hidden = a @ b          # Matrix multiply
activated = hidden.relu()  # Activation
output = activated.sum()   # Reduction

# Backward
output.backward()

# Gradients available
a_grad = a.grad.data.numpy()
b_grad = b.grad.data.numpy()
```

## Training Neural Networks

### Complete Example

```python
from gt.nn import Module, Linear, SGD

class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x).relu()
        return self.fc2(x)

# Create model and optimizer
model = MLP()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    # Forward pass
    pred = model(X_train)
    loss = ((pred - y_train) ** 2).mean()

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data.numpy()}")
```

## Supported Operations

All standard operations support gradients:

### Arithmetic
- Addition: `a + b`
- Subtraction: `a - b`
- Multiplication: `a * b`
- Division: `a / b`
- Power: `a ** 2`

### Matrix Operations
- Matrix multiply: `a @ b`
- Transpose: `a.T`

### Activations
- ReLU: `x.relu()`
- Sigmoid: `x.sigmoid()`
- Tanh: `x.tanh()`

### Reductions
- Sum: `x.sum()`, `x.sum(axis=0)`
- Mean: `x.mean()`, `x.mean(axis=1)`

### Math Functions
- Exponential: `x.exp()`
- Logarithm: `x.log()`

## Gradient Flow

### Broadcasting

Gradients correctly handle broadcasting:

```python
x = gt.randn(100, 1, requires_grad=True)
y = gt.randn(1, 50, requires_grad=True)

# Broadcasting during forward
z = x + y  # Shape: (100, 50)

# Gradients sum over broadcast dimensions
z.sum().backward()

print(x.grad.data.numpy().shape)  # (100, 1)
print(y.grad.data.numpy().shape)  # (1, 50)
```

### Matrix Multiplication

```python
a = gt.randn(100, 50, requires_grad=True)
b = gt.randn(50, 20, requires_grad=True)

c = a @ b  # Shape: (100, 20)
loss = c.sum()

loss.backward()

# Gradients computed via chain rule
# da/dloss = dloss/dc @ b.T
# db/dloss = a.T @ dloss/dc
```

## Neural Network Modules

### Module Base Class

```python
from gt.nn import Module

class MyModule(Module):
    def __init__(self):
        super().__init__()
        # Initialize parameters

    def forward(self, x):
        # Define computation
        return output
```

### Linear Layer

```python
from gt.nn import Linear

# Create layer
layer = Linear(input_size=784, output_size=256)

# Forward pass
output = layer(input)  # Calls layer.forward(input)

# Access parameters
for param in layer.parameters():
    print(param.data.numpy().shape)
```

### Loss Functions

```python
from gt.nn import mse_loss, cross_entropy_loss, binary_cross_entropy

# Mean Squared Error
loss = mse_loss(predictions, targets)

# Cross Entropy (for classification)
loss = cross_entropy_loss(logits, targets)

# Binary Cross Entropy
loss = binary_cross_entropy(predictions, targets)
```

## Optimizer

### SGD (Stochastic Gradient Descent)

```python
from gt.nn import SGD

# Create optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training step
loss.backward()           # Compute gradients
optimizer.step()          # Update parameters
optimizer.zero_grad()     # Reset gradients
```

## Manual Parameter Updates

For simple cases, update parameters directly:

```python
# Manual SGD
learning_rate = 0.01
for param in model.parameters():
    param -= learning_rate * param.grad
    param.grad.zero_()  # Reset gradient
```

## Debugging Gradients

### Print Tape

View the computational graph:

```python
import gt

# Build computation
x = gt.randn(10, 10, requires_grad=True)
y = x.relu().sum()

# View tape before backward
gt.debug.print_tape()

# Shows recorded operations
```

### Compare with PyTorch

Verify gradient correctness:

```python
import torch
import numpy as np
import gt

# Same input data
data = np.random.randn(10, 10).astype(np.float32)

# PyTorch
x_pt = torch.from_numpy(data.copy())
x_pt.requires_grad = True
y_pt = (x_pt * 2).sum()
y_pt.backward()

# GT
x_gt = gt.from_numpy(data.copy(), requires_grad=True)
y_gt = (x_gt * 2).sum()
y_gt.backward()

# Compare
grad_match = np.allclose(
    x_pt.grad.numpy(),
    x_gt.grad.data.numpy()
)
print(f"Gradients match: {grad_match}")
```

## Implementation Details

### Tape-Based System

GT uses tape-based autograd (like PyTorch):
1. Forward pass records operations on a tape
2. Each operation stores its gradient function
3. Backward pass traverses tape in reverse
4. Gradient functions compute local gradients
5. Chain rule combines gradients

### Gradient Functions

Each operation defines how to compute gradients:

- **Add**: Gradients pass through unchanged
- **Multiply**: Scale by other input
- **MatMul**: Matrix multiply with transposed inputs
- **ReLU**: Zero gradient where input ≤ 0
- **Sum**: Broadcast gradient to input shape

### Memory Management

- Gradients accumulate in `.grad` attribute
- Call `.zero_()` or `optimizer.zero_grad()` to reset
- Tape is automatically built and cleared

## Next Steps

- [Tensor API](tensor-api.md) - Complete operation reference
- [Usage Guide](../usage.md) - Training examples
- [Contributing](../contributing.md) - Add new operations with gradients
