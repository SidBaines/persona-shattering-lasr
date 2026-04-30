"""Custom exceptions for the internals module."""


class StopForward(Exception):
    """Exception to stop forward pass after target layer."""
    pass
