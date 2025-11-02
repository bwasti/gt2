"""
Common error messages for GT framework.

Consolidates repetitive error handling to reduce code duplication.
"""


def not_connected_error():
    """Error when client is not connected to dispatcher."""
    return "Not connected to dispatcher. Use gt.connect() or let GT auto-start."


def no_workers_error(num_workers: int = 0):
    """Error when dispatcher has no workers available."""
    return f"No workers available (registered: {num_workers}). Start workers with: python -m gt.worker --host <host> -p <port>"


def operation_failed_error(op: str, details: str):
    """Error when an operation fails on the dispatcher/worker."""
    return f"Operation '{op}' failed: {details}"
