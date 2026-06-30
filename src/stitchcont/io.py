from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_SAMPLE_COLUMNS = {"generation"}
SUPPORTED_VARIANT_TYPES = {"snp", "insertion", "deletion"}


def infer_variant_type(ref: str, alt: str) -> str:
    ref = str(ref).upper()
    alt = str(alt).upper()
    if not ref or not alt or ref == "." or alt == ".":
        raise ValueError("REF and ALT must be non-empty normalized alleles.")
    if "," in alt:
        raise ValueError("STITCHCONT requires split biallelic variants; ALT contains a comma.")
    if len(ref) == 1 and len(alt) == 1:
        return "snp"
    if len(alt) > len(ref) and alt.startswith(ref):
        return "insertion"
    if len(ref) > len(alt) and ref.startswith(alt):
        return "deletion"
    raise ValueError(
        f"Unsupported variant alleles REF={ref!r}, ALT={alt!r}. "
        "Use normalized biallelic SNPs or simple left-anchored insertions/deletions."
    )


def normalize_position_table(df: pd.DataFrame, *, require_unique_positions: bool = True) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).upper() for col in out.columns]
    required = {"CHR", "POS"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"Position file must contain columns {sorted(required)}")
    if "REF" not in out.columns:
        out["REF"] = "N"
    if "ALT" not in out.columns:
        out["ALT"] = "N"
    if "CM" not in out.columns and "GENETIC_CM" in out.columns:
        out["CM"] = out["GENETIC_CM"]

    out["CHR"] = out["CHR"].astype(str)
    out["POS"] = pd.to_numeric(out["POS"], errors="raise").astype(np.int64)
    out["REF"] = out["REF"].fillna("").astype(str).str.upper()
    out["ALT"] = out["ALT"].fillna("").astype(str).str.upper()

    inferred = [infer_variant_type(ref, alt) for ref, alt in zip(out["REF"], out["ALT"], strict=False)]
    if "VARIANT_TYPE" in out.columns:
        provided = out["VARIANT_TYPE"].fillna("").astype(str).str.lower()
        provided = provided.replace({"ins": "insertion", "del": "deletion"})
        normalized: list[str] = []
        for idx, (ptype, itype) in enumerate(zip(provided, inferred, strict=False)):
            value = itype if ptype == "" else ptype
            if value not in SUPPORTED_VARIANT_TYPES:
                raise ValueError(f"Unsupported VARIANT_TYPE {value!r} at row {idx}.")
            if value != itype:
                raise ValueError(
                    f"VARIANT_TYPE {value!r} disagrees with REF/ALT-inferred type {itype!r} at row {idx}."
                )
            normalized.append(value)
        out["VARIANT_TYPE"] = normalized
    else:
        out["VARIANT_TYPE"] = inferred

    if require_unique_positions and out.duplicated(["CHR", "POS"]).any():
        dup = out.loc[out.duplicated(["CHR", "POS"], keep=False), ["CHR", "POS", "REF", "ALT"]].head(5)
        raise ValueError(
            "STITCHCONT targeted read evidence currently requires one biallelic variant per CHR/POS. "
            f"Split or disambiguate duplicate positions first. Examples: {dup.to_dict('records')}"
        )
    return out


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
    df = normalize_position_table(df)
    chr_mask = df["CHR"].astype(str) == str(chromosome)
    columns = ["CHR", "POS", "REF", "ALT", "VARIANT_TYPE"]
    if "CM" in df.columns:
        columns.append("CM")
    out = df.loc[chr_mask, columns].sort_values("POS").reset_index(drop=True)
    if "CM" in out.columns:
        out["CM"] = pd.to_numeric(out["CM"], errors="coerce").astype(np.float32)
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
    row_start: int = 0
    row_stop: int = 0
    core_row_start: int | None = None
    core_row_stop: int | None = None

    @property
    def start(self) -> int:
        return int(self.dataframe["POS"].iloc[0])

    @property
    def stop(self) -> int:
        return int(self.dataframe["POS"].iloc[-1])

    @property
    def has_core_overlap(self) -> bool:
        core_start = self.row_start if self.core_row_start is None else int(self.core_row_start)
        core_stop = self.row_stop if self.core_row_stop is None else int(self.core_row_stop)
        return core_start != int(self.row_start) or core_stop != int(self.row_stop)

    @property
    def core_slice(self) -> tuple[int, int]:
        core_start = self.row_start if self.core_row_start is None else int(self.core_row_start)
        core_stop = self.row_stop if self.core_row_stop is None else int(self.core_row_stop)
        return max(core_start - int(self.row_start), 0), max(core_stop - int(self.row_start), 0)

    def core_block(self) -> "PositionBlock":
        local_start, local_stop = self.core_slice
        core_start = self.row_start if self.core_row_start is None else int(self.core_row_start)
        core_stop = self.row_stop if self.core_row_stop is None else int(self.core_row_stop)
        return PositionBlock(
            block_id=int(self.block_id),
            dataframe=self.dataframe.iloc[local_start:local_stop].reset_index(drop=True),
            row_start=core_start,
            row_stop=core_stop,
            core_row_start=core_start,
            core_row_stop=core_stop,
        )


def iter_position_blocks(positions_df: pd.DataFrame, block_size: int) -> Iterator[PositionBlock]:
    for block_id, start in enumerate(range(0, len(positions_df), block_size)):
        stop = min(start + block_size, len(positions_df))
        yield PositionBlock(
            block_id=block_id,
            dataframe=positions_df.iloc[start:stop].reset_index(drop=True),
            row_start=start,
            row_stop=stop,
            core_row_start=start,
            core_row_stop=stop,
        )


def iter_density_balanced_overlap_blocks(
    positions_df: pd.DataFrame,
    block_size: int,
    *,
    overlap_fraction: float = 0.05,
    min_overlap_snps: int = 50,
) -> Iterator[PositionBlock]:
    n_positions = int(len(positions_df))
    block_size = max(int(block_size), 1)
    overlap_fraction = float(np.clip(overlap_fraction, 0.0, 0.49))
    min_overlap = max(int(min_overlap_snps), 0)
    for block_id, core_start in enumerate(range(0, n_positions, block_size)):
        core_stop = min(core_start + block_size, n_positions)
        core_len = max(core_stop - core_start, 1)
        overlap = max(min_overlap, int(np.ceil(core_len * overlap_fraction))) if n_positions > core_len else 0
        start = max(core_start - overlap, 0)
        stop = min(core_stop + overlap, n_positions)
        yield PositionBlock(
            block_id=block_id,
            dataframe=positions_df.iloc[start:stop].reset_index(drop=True),
            row_start=start,
            row_stop=stop,
            core_row_start=core_start,
            core_row_stop=core_stop,
        )


def write_parquet(
    table: pa.Table,
    path: str | Path,
    compression: str = "zstd",
    compression_level: int = 6,
    row_group_size: int | None = None,
    use_dictionary: bool | list[str] = True,
    use_byte_stream_split: bool | list[str] = False,
    column_encoding: dict[str, str] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=use_dictionary,
        use_byte_stream_split=use_byte_stream_split,
        column_encoding=column_encoding,
        data_page_version="2.0",
        write_statistics=True,
        row_group_size=row_group_size,
    )
