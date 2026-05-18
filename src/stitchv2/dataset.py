from __future__ import annotations

from pathlib import Path

import pyarrow.dataset as ds


def open_block_dataset(path: str | Path) -> ds.Dataset:
    return ds.dataset(str(path), format="parquet")
