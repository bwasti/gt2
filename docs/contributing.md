# Contributing to GT

Thank you for your interest in contributing to GT! This is a research prototype focused on simplicity and readability.

## Getting Started

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/yourusername/gt.git
   cd gt
   ```

2. **Install in editable mode:**
   ```bash
   pip install -e .
   ```

3. **Run tests to verify setup:**
   ```bash
   pytest tests/ -v
   ```

## Development Workflow

### Making Changes

1. **Create a branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Keep code simple and readable
   - Follow existing code style
   - Add tests for new functionality

3. **Run tests:**
   ```bash
   pytest tests/ -v
   ```

4. **Commit changes:**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then open a pull request on GitHub.

## Running Tests

GT uses pytest for testing:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_numerics.py -v

# Run specific test
pytest tests/test_numerics.py::test_addition -v

# Run with output (see prints)
pytest tests/ -v -s

# Run tests matching pattern
pytest tests/ -k "matmul" -v
```

### Test Structure

- `tests/conftest.py` - Test fixtures (auto-starts GT system)
- `tests/test_numerics.py` - Basic operation correctness
- `tests/test_autograd.py` - Gradient computation tests
- `tests/test_operations.py` - Operation tests (relu, sigmoid, etc.)
- `tests/test_nn.py` - Neural network module tests
- `tests/test_sharding.py` - Distributed operation tests
- `tests/test_signal_sharding.py` - Signal-based sharding tests

### Writing Tests

Tests automatically start a GT system:

```python
def test_my_feature(client):
    """Test description."""
    # client fixture provides connected GT client
    x = gt.randn(10, 10)
    result = my_function(x)

    # Compare against NumPy/PyTorch
    expected = compute_expected(x.data.numpy())
    assert np.allclose(result.data.numpy(), expected)
```

## Code Style

### Simplicity First

GT prioritizes **readability over cleverness**. The target audience is researchers (not engineers).

**Good:**
```python
def add_tensors(a, b):
    """Add two tensors together."""
    result = a + b
    return result
```

**Bad:**
```python
def add_tensors(*args, **kwargs):
    """Tensor addition with advanced features."""
    return reduce(operator.add, map(lambda x: x if isinstance(x, Tensor) else gt.tensor(x), args))
```

### Key Principles

1. **Keep functions short** - Aim for <50 lines
2. **Explicit over implicit** - No magic behavior
3. **Minimal dependencies** - Avoid adding new packages
4. **Comments for "why" not "what"** - Code should be self-explanatory
5. **Use type hints** - Help with IDE support and clarity

### File Organization

- One abstraction per file when possible
- Clear separation between protocol and implementation
- Minimal dependencies between components

## AI-Assisted Development

GT is **designed to be understood and modified with AI coding assistants** (Claude, GPT-4, etc.).

### CLAUDE.md

The [`CLAUDE.md`](https://github.com/bwasti/gt/blob/main/CLAUDE.md) file provides detailed architectural context for AI assistants:

- Codebase structure and component descriptions
- Design decisions and implementation patterns
- Development commands and workflows
- Debugging strategies

**Before contributing,** have your AI assistant read `CLAUDE.md` for full context.

### AI-Friendly Design

GT follows patterns that make it easy for AI to understand:

1. **Tape-based autograd** - Simple to trace and explain
2. **YAML configuration** - Declarative, easy to parse
3. **Instruction logging** - Observable system behavior
4. **Comprehensive tests** - Executable specifications

### Working with AI Assistants

**Effective prompts for GT development:**

```
"Read CLAUDE.md for context. I want to add support for [feature].
Let's start by understanding where in the codebase this should go."
```

```
"Looking at gt/client/tensor.py, can you help me add a new operation
[op_name] that works like PyTorch's torch.[op_name]?"
```

```
"I'm getting a test failure in test_autograd.py. Can you help debug
by analyzing the tape output?"
```

## Debugging Guide

### Instruction Stream Logging

Most powerful debugging tool:

```bash
GT_INSTRUCTION_LOG=/tmp/debug.log python script.py
```

Shows all operations with timestamps. See [Tuning & Debugging](dispatcher/tuning.md) for details.

### Common Bug Patterns

**Sharding issues:**
1. Enable `GT_INSTRUCTION_LOG`
2. Check tensor placement in logs
3. Verify shard shapes at each step

**Gradient bugs:**
1. Test against PyTorch reference
2. Use `gt.debug.print_tape()` to view operations
3. Verify gradient shapes match forward pass

**Cross-worker operations:**
1. Check dispatcher logs for tensor locations
2. Verify workers are registered
3. Test with single worker first

See [CLAUDE.md - Debugging Guide](https://github.com/bwasti/gt/blob/main/CLAUDE.md#debugging-guide) for detailed strategies.

## Pull Request Guidelines

### PR Checklist

- [ ] Tests pass locally (`pytest tests/ -v`)
- [ ] New functionality has tests
- [ ] Code follows existing style
- [ ] Docstrings added for public APIs
- [ ] CLAUDE.md updated if architecture changed

### PR Description

Include in your PR description:

1. **What** - What does this PR do?
2. **Why** - Why is this change needed?
3. **How** - How does it work (brief technical overview)?
4. **Testing** - What tests were added/updated?

**Example:**
```markdown
## Add RMSNorm layer

### What
Implements RMSNorm (Root Mean Square Layer Normalization) in gt.nn

### Why
RMSNorm is used in modern architectures (LLaMA, etc.) and is simpler
than LayerNorm.

### How
- Added RMSNorm class to gt/client/nn.py
- Implements: normalized = x / sqrt(mean(x^2) + eps) * weight
- Supports gradients via autograd tape

### Testing
- Added test_rmsnorm in tests/test_nn.py
- Verified against PyTorch implementation
- All tests pass
```

## Architecture Overview

Understanding GT's architecture helps contributions fit the design:

### Three-Layer Architecture

```
gt/client/     - User-facing API (tensors, autograd, nn)
gt/dispatcher/ - Coordinates clients, schedules to workers
gt/worker/     - Executes operations (PyTorch/NumPy backend)
gt/transport/  - ZMQ communication protocols
```

### Key Abstractions

- **Tensor (client)** - Location-transparent remote data
- **TensorHandle (dispatcher)** - Maps client tensors to physical locations
- **Engine (worker)** - Backend for executing operations

### Data Flow

```
Client → Dispatcher → Worker → Dispatcher → Client
```

1. Client emits functional operations (ADD, MATMUL, etc.)
2. Dispatcher transforms and routes to workers
3. Workers execute using backend (PyTorch/NumPy)
4. Results flow back through dispatcher to client

See [CLAUDE.md](https://github.com/bwasti/gt/blob/main/CLAUDE.md) for detailed architecture.

## Areas for Contribution

### High-Impact Areas

1. **New operations** - Add missing tensor operations
2. **Performance** - Optimize hot paths, reduce overhead
3. **Testing** - Increase test coverage, add benchmarks
4. **Documentation** - Improve docs, add examples
5. **Backends** - Add JAX, TensorFlow, Metal backends

### Good First Issues

Look for issues labeled `good first issue` on GitHub. These are:
- Well-scoped
- Good introduction to the codebase
- Don't require deep architectural knowledge

### Advanced Contributions

- Cross-worker collective operations (AllReduce, AllGather)
- Automatic sharding strategies
- Gradient checkpointing
- Memory-efficient autograd

## Questions?

- **GitHub Issues** - For bug reports and feature requests
- **GitHub Discussions** - For questions and general discussion
- **Pull Requests** - Include questions in PR description

We're a friendly community and happy to help newcomers!

## License

By contributing to GT, you agree that your contributions will be licensed under the [MIT License](license.md).
