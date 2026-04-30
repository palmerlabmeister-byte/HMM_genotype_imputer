from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


_PLINK_GENO_LOOKUP = np.asarray([0.0, np.nan, 1.0, 2.0], dtype=np.float32)
_BYTE = np.arange(256, dtype=np.uint8)
_PLINK_BYTE_LOOKUP = np.stack(
    [(_BYTE >> 0) & 3, (_BYTE >> 2) & 3, (_BYTE >> 4) & 3, (_BYTE >> 6) & 3],
    axis=1,
)
_PLINK_BYTE_LOOKUP = _PLINK_GENO_LOOKUP[_PLINK_BYTE_LOOKUP]


@dataclass(slots=True)
class MicroarrayHardCalls:
    sample_ids: np.ndarray
    positions: np.ndarray
    dosage: np.ndarray
    n_loaded_variants: int
    n_target_variants: int


@dataclass(slots=True)
class SamplePlinkHardCallMap:
    dosage: np.ndarray
    unique_plink_files: int
    matched_samples: int


def _normalize_chrom(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("chr"):
        text = text[3:]
    return text.upper()


def _read_plink_fam(prefix: Path) -> pd.DataFrame:
    fam_path = prefix.with_suffix(".fam")
    fam = pd.read_csv(
        fam_path,
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "father", "mother", "sex", "pheno"],
        dtype={"fid": str, "iid": str},
        engine="c",
    )
    fam["iid"] = fam["iid"].astype(str)
    return fam


def _read_plink_bim(prefix: Path) -> pd.DataFrame:
    bim_path = prefix.with_suffix(".bim")
    bim = pd.read_csv(
        bim_path,
        sep=r"\s+",
        header=None,
        names=["chrom", "snp", "cm", "pos", "a1", "a2"],
        dtype={"chrom": str, "snp": str, "pos": np.int64, "a1": str, "a2": str},
        engine="c",
    )
    bim["chrom_norm"] = bim["chrom"].map(_normalize_chrom)
    return bim


def _decode_variant_block(
    bed_mem: np.memmap,
    *,
    n_samples: int,
    variant_start: int,
    variant_count: int,
) -> np.ndarray:
    bytes_per_variant = (n_samples + 3) // 4
    offset = 3 + variant_start * bytes_per_variant
    n_bytes = variant_count * bytes_per_variant
    block_bytes = np.asarray(bed_mem[offset : offset + n_bytes], dtype=np.uint8).reshape(
        variant_count, bytes_per_variant
    )
    decoded = _PLINK_BYTE_LOOKUP[block_bytes].reshape(variant_count, bytes_per_variant * 4)
    decoded = decoded[:, :n_samples]
    return decoded.T.astype(np.float32, copy=False)


def load_microarray_hardcalls_from_plink(
    plink_prefix: str | Path,
    *,
    chromosome: str,
    positions_df: pd.DataFrame,
    chunk_variants: int = 1024,
) -> MicroarrayHardCalls:
    prefix = Path(plink_prefix)
    fam = _read_plink_fam(prefix)
    bim = _read_plink_bim(prefix)

    chrom_norm = _normalize_chrom(chromosome)
    bim_chr = bim.loc[bim["chrom_norm"] == chrom_norm].copy()
    bim_chr = bim_chr.sort_values("pos").reset_index(drop=False).rename(columns={"index": "variant_index"})

    target_positions = positions_df["POS"].to_numpy(dtype=np.int64, copy=False)
    pos_to_variant: dict[int, int] = {}
    for row in bim_chr.itertuples(index=False):
        pos = int(row.pos)
        if pos not in pos_to_variant:
            pos_to_variant[pos] = int(row.variant_index)

    variant_idx = np.asarray([pos_to_variant.get(int(pos), -1) for pos in target_positions], dtype=np.int64)
    keep_mask = variant_idx >= 0
    keep_variant_idx = variant_idx[keep_mask]
    n_targets = int(target_positions.size)

    dosage = np.full((len(fam), n_targets), np.nan, dtype=np.float32)
    if keep_variant_idx.size == 0:
        return MicroarrayHardCalls(
            sample_ids=fam["iid"].to_numpy(dtype=object),
            positions=target_positions,
            dosage=dosage,
            n_loaded_variants=0,
            n_target_variants=n_targets,
        )

    order = np.argsort(keep_variant_idx)
    sorted_idx = keep_variant_idx[order]
    sorted_target_col = np.flatnonzero(keep_mask)[order]
    bed_mem = np.memmap(prefix.with_suffix(".bed"), mode="r", dtype=np.uint8)
    if bed_mem.shape[0] < 3 or tuple(bed_mem[:3].tolist()) != (108, 27, 1):
        raise ValueError("PLINK .bed header is invalid or not SNP-major.")

    run_start = 0
    while run_start < sorted_idx.size:
        run_end = run_start + 1
        while run_end < sorted_idx.size and sorted_idx[run_end] == sorted_idx[run_end - 1] + 1:
            run_end += 1
        var_start = int(sorted_idx[run_start])
        var_count = int(run_end - run_start)
        block = _decode_variant_block(
            bed_mem,
            n_samples=len(fam),
            variant_start=var_start,
            variant_count=var_count,
        )
        cols = sorted_target_col[run_start:run_end]
        dosage[:, cols] = block
        run_start = run_end

    return MicroarrayHardCalls(
        sample_ids=fam["iid"].to_numpy(dtype=object),
        positions=target_positions,
        dosage=dosage,
        n_loaded_variants=int(keep_variant_idx.size),
        n_target_variants=n_targets,
    )


def align_microarray_to_samples(
    samples: pd.DataFrame,
    hardcalls: MicroarrayHardCalls,
    *,
    add_missing_samples: bool = True,
    generation_default: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    out_samples = samples.copy()
    out_samples["sample_id"] = out_samples["sample_id"].astype(str)
    sample_ids = out_samples["sample_id"].to_numpy(dtype=object)
    hard_ids = hardcalls.sample_ids.astype(str)

    if add_missing_samples:
        new_ids = [sid for sid in hard_ids.tolist() if sid not in set(sample_ids.tolist())]
        if new_ids:
            if generation_default is None:
                if len(out_samples) > 0:
                    generation_default = float(
                        np.nanmedian(out_samples["generation"].to_numpy(dtype=np.float32, copy=False))
                    )
                else:
                    generation_default = 0.0
            extra = pd.DataFrame(
                {
                    "sample_id": np.asarray(new_ids, dtype=object),
                    "bam_path": np.repeat("", len(new_ids)),
                    "generation": np.repeat(float(generation_default), len(new_ids)),
                }
            )
            out_samples = pd.concat([out_samples, extra], ignore_index=True)
            sample_ids = out_samples["sample_id"].astype(str).to_numpy(dtype=object)

    hard_index = pd.Index(hard_ids)
    out_index = pd.Index(sample_ids.astype(str))
    take = hard_index.get_indexer(out_index)
    out = np.full((len(out_samples), hardcalls.dosage.shape[1]), np.nan, dtype=np.float32)
    valid = take >= 0
    if np.any(valid):
        out[valid] = hardcalls.dosage[take[valid]]
    return out_samples, out


def load_microarray_hardcalls_from_sample_plink_paths(
    samples: pd.DataFrame,
    *,
    chromosome: str,
    positions_df: pd.DataFrame,
    plink_path_column: str = "plink_path",
) -> SamplePlinkHardCallMap:
    if plink_path_column not in samples.columns:
        return SamplePlinkHardCallMap(
            dosage=np.full((len(samples), len(positions_df)), np.nan, dtype=np.float32),
            unique_plink_files=0,
            matched_samples=0,
        )

    sample_ids = samples["sample_id"].astype(str).to_numpy(dtype=object)
    plink_paths = samples[plink_path_column].fillna("").astype(str).to_numpy(dtype=object)
    out = np.full((len(samples), len(positions_df)), np.nan, dtype=np.float32)

    cache: dict[str, MicroarrayHardCalls] = {}
    unique = sorted({p.strip() for p in plink_paths.tolist() if str(p).strip()})
    for path in unique:
        cache[path] = load_microarray_hardcalls_from_plink(
            path,
            chromosome=chromosome,
            positions_df=positions_df,
        )

    matched = 0
    for idx, (sample_id, path) in enumerate(zip(sample_ids.tolist(), plink_paths.tolist(), strict=False)):
        key = str(path).strip()
        if not key:
            continue
        hard = cache.get(key)
        if hard is None:
            continue
        ids = hard.sample_ids.astype(str)
        hits = np.flatnonzero(ids == str(sample_id))
        if hits.size == 0:
            continue
        out[idx] = hard.dosage[int(hits[0])]
        matched += 1

    return SamplePlinkHardCallMap(
        dosage=out,
        unique_plink_files=len(unique),
        matched_samples=matched,
    )
