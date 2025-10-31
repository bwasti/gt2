"""
Engine abstraction for different backends (NumPy, PyTorch).

Engines handle the low-level tensor operations and provide distributed primitives.
Some engines support instruction batching and compilation (PyTorch).
"""

from .base import Engine, Operation
from .numpy import NumpyEngine
from .pytorch import PyTorchEngine


def create_engine(backend: str = 'numpy', enable_compilation: bool = False) -> Engine:
    """
    Factory function to create an engine.

    Args:
        backend: 'numpy' or 'pytorch'
        enable_compilation: Enable torch.compile() for PyTorch (default: False)

    Returns:
        Engine instance
    """
    if backend == 'numpy':
        return NumpyEngine()
    elif backend == 'pytorch':
        return PyTorchEngine(enable_compilation=enable_compilation)
    else:
        raise ValueError(f"Unknown backend: {backend}")


__all__ = ['Engine', 'Operation', 'NumpyEngine', 'PyTorchEngine', 'create_engine']
