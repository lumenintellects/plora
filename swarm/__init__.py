from importlib import metadata as _metadata

__all__ = [
    "messages",
    "metrics",
]

try:
    __version__: str = _metadata.version(__name__)
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
