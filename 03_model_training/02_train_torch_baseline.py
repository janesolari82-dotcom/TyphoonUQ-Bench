"""Thin wrapper for typhoonuq-train-torch."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from typhoonuq.cli import train_torch_main


if __name__ == "__main__":
    train_torch_main()
