"""
Neural network components for GT.

Optimizers, loss functions, etc.
"""


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Like PyTorch's torch.optim.SGD but for GT tensors.
    """

    def __init__(self, params, lr=0.01):
        """
        Initialize SGD optimizer.

        Args:
            params: List of parameter tensors to optimize
            lr: Learning rate
        """
        self.params = list(params)
        self.lr = lr

    def step(self):
        """
        Update parameters using gradients.

        Performs: param = param - lr * grad
        """
        for param in self.params:
            if param.grad is not None:
                # Update: param -= lr * grad
                param -= self.lr * param.grad

    def zero_grad(self):
        """Zero out all parameter gradients."""
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()


__all__ = ['SGD']
