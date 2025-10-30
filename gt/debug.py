"""
Debug utilities for GT.

Provides visibility into:
- Autograd tape
- Worker compilation statistics
- Operation flow
"""

from typing import List, Dict, Any
from gt.client.autograd import get_graph


def get_tape() -> List[Dict[str, Any]]:
    """
    Get the current autograd tape.

    Returns a list of recorded operations with details about:
    - Output tensor ID and shape
    - Input tensor IDs
    - Operation type

    Example:
        import gt
        a = gt.randn(10, 10, requires_grad=True)
        b = gt.randn(10, 10, requires_grad=True)
        c = a + b
        d = c * 2.0

        tape = gt.debug.get_tape()
        for i, entry in enumerate(tape):
            print(f"{i}: {entry}")
    """
    graph = get_graph()
    tape_info = []

    for output, inputs, grad_fn in graph.tape:
        entry = {
            'output_id': output.id,
            'output_shape': output.shape,
            'input_ids': [inp.id for inp in inputs],
            'grad_fn': grad_fn.__name__ if hasattr(grad_fn, '__name__') else str(grad_fn),
            'requires_grad': output.requires_grad,
        }
        tape_info.append(entry)

    return tape_info


def print_tape():
    """
    Pretty-print the autograd tape.

    Example:
        import gt
        a = gt.randn(10, 10, requires_grad=True)
        b = gt.randn(10, 10, requires_grad=True)
        c = a + b
        d = c * 2.0

        gt.debug.print_tape()
        # Output:
        # Autograd Tape (2 operations):
        # 0: Tensor[2] = op(Tensor[0], Tensor[1]) shape=(10, 10)
        # 1: Tensor[3] = op(Tensor[2]) shape=(10, 10)
    """
    tape = get_tape()

    if not tape:
        print("Autograd Tape: Empty (no operations recorded)")
        return

    print(f"Autograd Tape ({len(tape)} operations):")
    print("-" * 60)

    for i, entry in enumerate(tape):
        input_ids = ', '.join(f"Tensor[{tid}]" for tid in entry['input_ids'])
        print(f"{i}: Tensor[{entry['output_id']}] = {entry['grad_fn']}({input_ids})")
        print(f"   shape={entry['output_shape']} requires_grad={entry['requires_grad']}")


def get_worker_stats() -> Dict[str, Any]:
    """
    Get compilation statistics from workers.

    Returns statistics including:
    - Cache hits/misses
    - Compilation times
    - Number of compiled graphs

    Example:
        import gt
        stats = gt.debug.get_worker_stats()
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Cache misses: {stats['cache_misses']}")
    """
    from gt.client.tensor import _client_connection, _connection_lock
    from gt.transport.protocol import GetWorkerStats

    if not _client_connection:
        return {'error': 'Not connected to dispatcher'}

    with _connection_lock:
        _client_connection.send(GetWorkerStats())
        response = _client_connection.recv()

        if not response.success:
            return {'error': response.error}

        return response.data


def print_worker_stats():
    """
    Pretty-print worker compilation statistics.

    Example:
        import gt
        gt.debug.print_worker_stats()
        # Output:
        # Worker Statistics:
        # Cache hits: 10
        # Cache misses: 2
        # Hit rate: 83.3%
    """
    stats = get_worker_stats()

    if 'error' in stats:
        print(f"Error: {stats['error']}")
        return

    print("Worker Statistics:")
    print("-" * 60)

    for worker_id, worker_stats in stats.items():
        print(f"\nWorker: {worker_id}")
        if 'compilation' in worker_stats:
            comp = worker_stats['compilation']
            total = comp.get('cache_hits', 0) + comp.get('cache_misses', 0)
            hit_rate = (comp.get('cache_hits', 0) / total * 100) if total > 0 else 0

            print(f"  Compilation:")
            print(f"    Cache hits:   {comp.get('cache_hits', 0)}")
            print(f"    Cache misses: {comp.get('cache_misses', 0)}")
            print(f"    Hit rate:     {hit_rate:.1f}%")
            print(f"    Cached graphs: {comp.get('num_cached_graphs', 0)}")

        if 'operations' in worker_stats:
            ops = worker_stats['operations']
            print(f"  Operations:")
            print(f"    Total:        {ops.get('total', 0)}")
            print(f"    Batched:      {ops.get('batched', 0)}")
