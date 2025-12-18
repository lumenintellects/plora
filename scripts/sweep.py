"""Compatibility shim for the thesis sweep CLI.

The real implementation lives in :mod:`scripts.sweep.main`.  This wrapper is
kept so that historical entrypoints such as ``python scripts/sweep.py`` keep
working.
"""

from scripts.sweep import main as _sweep_main


def main() -> None:  # pragma: no cover
    _sweep_main()


if __name__ == "__main__":  # pragma: no cover
    main()
