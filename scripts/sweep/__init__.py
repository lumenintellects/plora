"""
Package entry point for thesis-scale sweep utilities.

Expose :func:`main` so ``python -m scripts.sweep`` continues to work and
other modules can import the CLI programmatically.
"""

from .main import main

__all__ = ["main"]


