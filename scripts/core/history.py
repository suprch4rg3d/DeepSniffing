"""The record of a training run: the numbers of every epoch in ``history.csv``, and the console log."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pandas as pd


def save(folder: Path, rows: list[dict[str, float]]) -> None:
    """Write every epoch so far to ``<folder>/history.csv``. Called after each epoch, so a crash keeps them."""
    pd.DataFrame(rows).to_csv(folder / "history.csv", index=False)


def load(folder: Path) -> pd.DataFrame:
    path = folder / "history.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Train this stage first.")
    return pd.read_csv(path)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def logged(path: Path):
    """Everything printed inside the block also goes to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        original = sys.stdout
        sys.stdout = _Tee(original, log)
        try:
            yield
        finally:
            sys.stdout = original
