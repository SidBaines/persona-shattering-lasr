"""Minimal flash_attn compatibility layer for OpenRLHF startup.

This shim exists so OpenRLHF can import its ring-attention helpers in
environments where the compiled flash-attn package is unavailable.
"""
