from __future__ import annotations

import pickle
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")


def load_or_build(cache_path: Path, builder: Callable[[], T]) -> T:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        with cache_path.open("rb") as fh:
            return pickle.load(fh)
    value = builder()
    with cache_path.open("wb") as fh:
        pickle.dump(value, fh)
    return value
