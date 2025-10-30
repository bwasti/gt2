"""
Neural network abstractions (PyTorch-like).

Keep this SIMPLE and READABLE.
"""

from typing import List
from gt.client.tensor import Tensor
import gt


class Module:
    """
    Base class for all neural network modules (like PyTorch nn.Module).

    Your models should subclass this and implement forward().
    """

    def __init__(self):
        self._parameters = []
        self._modules = {}  # Track child modules

    def __setattr__(self, name, value):
        """Override to track child modules."""
        if isinstance(value, Module):
            # Store child modules
            if not hasattr(self, '_modules'):
                # During __init__, _modules doesn't exist yet
                super().__setattr__('_modules', {})
            self._modules[name] = value
        super().__setattr__(name, value)

    def forward(self, *args, **kwargs):
        """
        Forward pass. Override this in your model.
        """
        raise NotImplementedError("Subclass must implement forward()")

    def __call__(self, *args, **kwargs):
        """
        Make the module callable.
        """
        return self.forward(*args, **kwargs)

    def parameters(self) -> List[Tensor]:
        """
        Return list of trainable parameters (recursively from all submodules).
        """
        params = list(self._parameters)
        # Recursively collect from child modules
        if hasattr(self, '_modules'):
            for module in self._modules.values():
                params.extend(module.parameters())
        return params

    def zero_grad(self):
        """
        Zero out gradients of all parameters.
        """
        for param in self._parameters:
            param._grad = None


class Linear(Module):
    """
    Linear layer: y = xW + b

    Args:
        in_features: Input size
        out_features: Output size
        bias: Whether to include bias (default: True)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights (small random values)
        import numpy as np
        w_data = np.random.randn(in_features, out_features).astype('float32') * 0.01
        self.weight = gt.tensor(w_data, requires_grad=True)
        self._parameters.append(self.weight)

        if bias:
            b_data = np.zeros(out_features, dtype='float32')
            self.bias = gt.tensor(b_data, requires_grad=True)
            self._parameters.append(self.bias)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass: x @ W + b
        """
        output = x @ self.weight
        if self.bias is not None:
            output = output + self.bias
        return output


# Loss functions

def mse_loss(predicted: Tensor, target: Tensor) -> Tensor:
    """
    Mean Squared Error loss.

    loss = mean((predicted - target)²)
    """
    diff = predicted - target
    squared = diff * diff
    return squared.mean()


def binary_cross_entropy(predicted: Tensor, target: Tensor) -> Tensor:
    """
    Binary Cross Entropy loss.

    loss = -mean(target * log(predicted) + (1-target) * log(1-predicted))
    """
    # Avoid log(0)
    import numpy as np
    eps = gt.tensor(np.array(1e-7, dtype='float32'))
    one = gt.tensor(np.array(1.0, dtype='float32'))

    term1 = target * (predicted + eps).log()
    term2 = (one - target) * (one - predicted + eps).log()
    return -(term1 + term2).mean()


def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Cross entropy loss (simplified version).

    For now, just implement for binary case.
    TODO: implement full multi-class version with softmax.
    """
    # Apply sigmoid to get probabilities
    probs = logits.sigmoid()
    return binary_cross_entropy(probs, targets)


# Activation functions (also available as methods on Tensor)

def relu(x: Tensor) -> Tensor:
    """ReLU activation: max(0, x)"""
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    """Tanh activation"""
    return x.tanh()
