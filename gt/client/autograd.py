"""
Tape-based autograd system.

This is PURELY ABSTRACT - no interaction with computation.
Just creates more tensors.

Keep this SIMPLE and READABLE.
"""

from typing import Callable, List, Optional
from gt.client.tensor import Tensor


class AutogradGraph:
    """
    Tape-based autograd system like PyTorch.

    Records operations and their gradients for backpropagation.
    """

    def __init__(self):
        self.tape = []  # List of (output, inputs, grad_fn)
        self.enabled = True

    def record(self, output: Tensor, inputs: List[Tensor], grad_fn: Callable):
        """
        Record an operation on the tape.

        Args:
            output: The result tensor
            inputs: Input tensors to the operation
            grad_fn: Function to compute gradients: (grad_output) -> List[grad_inputs]
        """
        if not self.enabled:
            return

        self.tape.append((output, inputs, grad_fn))

    def backward(self, loss: Tensor):
        """
        Compute gradients by walking backward through the tape.

        Args:
            loss: The loss tensor to backpropagate from
        """
        from gt.client.tensor import from_numpy
        import numpy as np

        # Initialize gradient for loss (scalar = 1.0)
        # For a scalar loss, gradient is just 1.0
        grads = {loss.id: from_numpy(np.array(1.0, dtype='float32'))}

        # Walk backward through tape
        for output, inputs, grad_fn in reversed(self.tape):
            if output.id not in grads:
                continue  # No gradient for this output

            grad_output = grads[output.id]

            # Compute gradients for inputs
            grad_inputs = grad_fn(grad_output)

            # Accumulate gradients
            for inp, grad_inp in zip(inputs, grad_inputs):
                if inp.id in grads:
                    # Accumulate gradient
                    grads[inp.id] = grads[inp.id] + grad_inp
                else:
                    grads[inp.id] = grad_inp

                # Set gradient on tensor
                inp._grad = grads[inp.id]

    def no_grad(self):
        """Context manager to disable gradient recording."""
        return _NoGrad(self)


class _NoGrad:
    """Context manager to disable gradient recording."""

    def __init__(self, graph: AutogradGraph):
        self.graph = graph
        self.prev = None

    def __enter__(self):
        self.prev = self.graph.enabled
        self.graph.enabled = False

    def __exit__(self, *args):
        self.graph.enabled = self.prev


# Global autograd graph (like PyTorch's global autograd)
_global_graph = AutogradGraph()


def get_graph() -> AutogradGraph:
    """Get the global autograd graph."""
    return _global_graph


def no_grad():
    """Context manager to disable gradient recording."""
    return _global_graph.no_grad()


# Gradient functions for common operations

def add_grad(grad_output):
    """Gradient for addition: both inputs get the same gradient."""
    return [grad_output, grad_output]


def mul_grad(left, right):
    """Gradient for multiplication."""
    def grad_fn(grad_output):
        return [grad_output * right, grad_output * left]
    return grad_fn


def matmul_grad(left, right):
    """Gradient for matrix multiplication."""
    def grad_fn(grad_output):
        # grad_left = grad_output @ right.T
        # grad_right = left.T @ grad_output
        # For now, just return placeholders
        # TODO: implement proper matmul gradients
        return [grad_output @ right, left @ grad_output]
    return grad_fn
