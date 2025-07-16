"""DY SDR Encoder - Dynamic Sparse Distributed Representations with Hebbian Learning."""

from __future__ import annotations

from .encoder import Encoder
from .train_loop import train

__version__ = "0.1.0"
__all__ = ["Encoder", "train"]
