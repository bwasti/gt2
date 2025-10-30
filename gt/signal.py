"""
Signal API for configuration-based sharding.

Signals provide named scopes that can be configured via YAML to control
sharding strategies for tensors and compute.

Usage:
    # Context manager - shards tensors AND compute
    with gt.signal('layer1'):
        x = a + b  # Addition happens in sharded mode

    # Function call - only copy-shards the tensor
    x = gt.signal(x, name='layer1')

    # Backward pass handling
    with gt.signal('layer1', backward='layer1_bwd'):
        x = a + b  # Forward uses 'layer1' config, backward uses 'layer1_bwd'
"""

import threading
from typing import Optional, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from gt.client.tensor import Tensor


class SignalScope:
    """Represents a signal scope with optional backward signal."""

    def __init__(self, name: str, backward_name: Optional[str] = None):
        self.name = name
        self.backward_name = backward_name

    def __repr__(self):
        if self.backward_name:
            return f"SignalScope({self.name}, backward={self.backward_name})"
        return f"SignalScope({self.name})"


class SignalStack:
    """
    Thread-local stack of active signal scopes.

    Used to track which signal context we're currently in, so that
    tensor operations can inherit the signal.
    """

    def __init__(self):
        self._local = threading.local()

    def _get_stack(self):
        """Get the signal stack for this thread."""
        if not hasattr(self._local, 'stack'):
            self._local.stack = []
        return self._local.stack

    def push(self, scope: SignalScope):
        """Push a signal scope onto the stack."""
        self._get_stack().append(scope)

    def pop(self) -> Optional[SignalScope]:
        """Pop a signal scope from the stack."""
        stack = self._get_stack()
        if stack:
            return stack.pop()
        return None

    def current(self) -> Optional[SignalScope]:
        """Get the current signal scope (top of stack)."""
        stack = self._get_stack()
        if stack:
            return stack[-1]
        return None

    def current_name(self) -> Optional[str]:
        """Get the name of the current signal scope."""
        scope = self.current()
        return scope.name if scope else None

    def current_backward_name(self) -> Optional[str]:
        """Get the backward signal name for the current scope."""
        scope = self.current()
        return scope.backward_name if scope else None


# Global signal stack
_signal_stack = SignalStack()


def get_signal_stack() -> SignalStack:
    """Get the global signal stack."""
    return _signal_stack


def current_signal() -> Optional[str]:
    """Get the name of the current signal scope."""
    return _signal_stack.current_name()


def current_backward_signal() -> Optional[str]:
    """Get the backward signal name for the current scope."""
    return _signal_stack.current_backward_name()


@contextmanager
def context(name: str, backward: Optional[str] = None):
    """
    Signal context manager.

    All tensor operations within this context will be tagged with the signal name
    and sharded according to the configuration.

    Args:
        name: Signal name (must be defined in config)
        backward: Optional signal name for backward pass

    Example:
        with gt.signal.context('layer1'):
            x = a + b  # Addition happens in sharded mode

        with gt.signal.context('layer1', backward='layer1_bwd'):
            x = a + b  # Forward uses 'layer1', backward uses 'layer1_bwd'
    """
    scope = SignalScope(name, backward)
    _signal_stack.push(scope)
    try:
        yield
    finally:
        _signal_stack.pop()


# Alias for backward compatibility
signal = context


def tensor(tensor_arg: 'Tensor', name: str, backward_name: Optional[str] = None) -> 'Tensor':
    """
    Apply signal to a tensor (copy-shard only, not compute).

    This only shards the tensor itself, not the compute that created it.
    Use the context manager to shard the compute as well.

    Args:
        tensor_arg: Tensor to apply signal to
        name: Signal name
        backward_name: Optional signal name for gradients

    Returns:
        Tensor with signal metadata attached

    Example:
        x = gt.randn(100, 100)
        x_sharded = gt.signal.tensor(x, name='layer1')
    """
    # TODO: Implement tensor copy-sharding
    # For now, just attach signal metadata
    tensor_arg._signal_name = name
    tensor_arg._backward_signal_name = backward_name
    return tensor_arg


# Convenience functions for entering/exiting scopes (alternative API)
def enter(name: str, backward: Optional[str] = None):
    """
    Enter a signal scope.

    Alternative to context manager for cases where context manager is awkward.

    Example:
        gt.signal.enter('layer1')
        x = a + b
        gt.signal.exit('layer1')
    """
    scope = SignalScope(name, backward)
    _signal_stack.push(scope)


def exit(name: str):
    """
    Exit a signal scope.

    Args:
        name: Signal name to verify (must match current scope)

    Raises:
        RuntimeError: If name doesn't match current scope
    """
    current = _signal_stack.current()
    if current is None:
        raise RuntimeError(f"No signal scope to exit (expected '{name}')")
    if current.name != name:
        raise RuntimeError(f"Signal scope mismatch: expected '{name}', got '{current.name}'")
    _signal_stack.pop()
