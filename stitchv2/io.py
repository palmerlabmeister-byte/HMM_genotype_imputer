from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_SAMPLE_COLUMNS = {"generation"}


def validate_samples(samples: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SAMPLE_COLUMNS.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing required sample columns: {sorted(missing)}")
    samples = samples.copy()
    if "sample_id" not in samples.columns:
        samples.insert(0, "sample_id", [f"sample_{i}" for i in range(len(samples))])
    if "bam_path" not in samples.columns:
        samples["bam_path"] = ""
    samples["sample_id"] = samples["sample_id"].astype(str)
    samples["generation"] = samples["generation"].astype(np.float32)
    samples["bam_path"] = samples["bam_path"].fillna("").astype(str)
    if "plink_path" in samples.columns:
        samples["plink_path"] = samples["plink_path"].fillna("").astype(str)
    if "sex" in samples.columns:
        samples["sex"] = samples["sex"].astype(str)
    return samples


def load_positions(
    path: str | Path,
    chromosome: str,
    *,
    start: int | None = None,
    end: int | None = None,
) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        table = pq.read_table(path)
        df = table.to_pandas()
    else:
        df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [col.upper() for col in df.columns]
    required = {"CHR", "POS"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Position file must contain columns {sorted(required)}")
    if "REF" not in df.columns:
        df["REF"] = "N"
    if "ALT" not in df.columns:
        df["ALT"] = "N"
    chr_mask = df["CHR"].astype(str) == str(chromosome)
    out = df.loc[chr_mask, ["CHR", "POS", "REF", "ALT"]].sort_values("POS").reset_index(drop=True)
    out["POS"] = out["POS"].astype(np.int64)
    if start is not None:
        out = out.loc[out["POS"] >= int(start)].copy()
    if end is not None:
        out = out.loc[out["POS"] <= int(end)].copy()
    out = out.reset_index(drop=True)
    return out


@dataclass(slots=True)
class PositionBlock:
    block_id: int
    dataframe: pd.DataFrame

    @property
    def start(self) -> int:
        return int(self.dataframe["POS"].iloc[0])

    @property
    def stop(self) -> int:
        return int(self.dataframe["POS"].iloc[-1])


def iter_position_blocks(positions_df: pd.DataFrame, block_size: int) -> Iterator[PositionBlock]:
    for block_id, start in enumerate(range(0, len(positions_df), block_size)):
        stop = min(start + block_size, len(positions_df))
        yield PositionBlock(block_id=block_id, dataframe=positions_df.iloc[start:stop].reset_index(drop=True))


def write_parquet(
    table: pa.Table,
    path: str | Path,
    compression: str = "zstd",
    compression_level: int = 6,
    row_group_size: int | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=True,
        data_page_version="2.0",
        write_statistics=True,
        row_group_size=row_group_size,
    )
