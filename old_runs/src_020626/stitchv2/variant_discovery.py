from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - depends on optional native extension.
    from ._htslib_readstream import discover_snp_candidates as _discover_snp_candidates_htslib
except Exception:  # pragma: no cover - depends on local HTSlib build.
    _discover_snp_candidates_htslib = None


_BASE_LUT = np.full(256, "N", dtype=object)
for _base in ("A", "C", "G", "T"):
    _BASE_LUT[ord(_base)] = _base


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def _reference_length(reference_fasta: str | Path, chromosome: str) -> int:
    try:
        import pysam
    except Exception as exc:  # pragma: no cover - dependency is required by package.
        raise RuntimeError("pysam is required to determine chromosome length from FASTA.") from exc

    with pysam.FastaFile(str(reference_fasta)) as fasta:
        try:
            return int(fasta.get_reference_length(str(chromosome)))
        except ValueError as exc:
            raise ValueError(f"Chromosome {chromosome!r} was not found in {reference_fasta}.") from exc


def _nonempty_bam_paths(samples: pd.DataFrame, bam_path_col: str) -> list[str]:
    if bam_path_col not in samples.columns:
        raise ValueError(f"samples table is missing BAM path column {bam_path_col!r}.")
    paths = samples[bam_path_col].fillna("").astype(str).tolist()
    return [path for path in paths if path]


def _chunk_ranges(start_1based: int, end_1based: int, window_size: int) -> Iterable[tuple[int, int]]:
    if start_1based < 1:
        raise ValueError("start position must be >= 1.")
    if end_1based < start_1based:
        raise ValueError("end position must be >= start position.")
    if window_size <= 0:
        raise ValueError("window_size must be positive.")
    start0 = int(start_1based) - 1
    stop0 = int(end_1based)
    for chunk_start0 in range(start0, stop0, int(window_size)):
        yield chunk_start0, min(chunk_start0 + int(window_size), stop0)


def _native_result_to_frame(result: dict, chromosome: str) -> pd.DataFrame:
    pos = np.asarray(result["position"], dtype=np.int64)
    if pos.size == 0:
        return empty_discovery_frame()
    ref_code = np.asarray(result["ref_code"], dtype=np.uint8)
    alt_code = np.asarray(result["alt_code"], dtype=np.uint8)
    depth = np.asarray(result["depth"], dtype=np.uint32)
    alt_count = np.asarray(result["alt_count"], dtype=np.uint32)
    other_count = np.asarray(result["other_count"], dtype=np.uint32)
    alt_fraction = alt_count.astype(np.float64) / np.maximum(depth.astype(np.float64), 1.0)
    other_fraction = other_count.astype(np.float64) / np.maximum(depth.astype(np.float64), 1.0)
    return pd.DataFrame(
        {
            "CHR": np.repeat(str(chromosome), pos.shape[0]),
            "POS": pos,
            "REF": _BASE_LUT[ref_code],
            "ALT": _BASE_LUT[alt_code],
            "variant_type": np.repeat("snp", pos.shape[0]),
            "depth": depth,
            "ref_count": np.asarray(result["ref_count"], dtype=np.uint32),
            "alt_count": alt_count,
            "other_count": other_count,
            "alt_fraction": alt_fraction.astype(np.float32),
            "other_fraction": other_fraction.astype(np.float32),
            "sample_support": np.asarray(result["sample_support"], dtype=np.uint32),
            "a_count": np.asarray(result["a_count"], dtype=np.uint32),
            "c_count": np.asarray(result["c_count"], dtype=np.uint32),
            "g_count": np.asarray(result["g_count"], dtype=np.uint32),
            "t_count": np.asarray(result["t_count"], dtype=np.uint32),
            "alt_forward_count": np.asarray(result["alt_forward_count"], dtype=np.uint32),
            "alt_reverse_count": np.asarray(result["alt_reverse_count"], dtype=np.uint32),
        }
    )


def _normalize_variant_types(variant_types: Iterable[str] | str) -> set[str]:
    if isinstance(variant_types, str):
        raw = [value.strip() for value in variant_types.split(",")]
    else:
        raw = [str(value).strip() for value in variant_types]
    out: set[str] = set()
    for value in raw:
        lowered = value.lower()
        if lowered in {"", "none"}:
            continue
        if lowered == "ins":
            lowered = "insertion"
        elif lowered == "del":
            lowered = "deletion"
        if lowered not in {"snp", "insertion", "deletion"}:
            raise ValueError(f"Unsupported discovery variant type {value!r}; use snp, ins, del.")
        out.add(lowered)
    return out or {"snp", "insertion", "deletion"}


def _capped_quality(q: int, mapq: int, cap_by_mapq: bool) -> int:
    if q == 255:
        q = 30
    return min(int(q), int(mapq)) if cap_by_mapq else int(q)


def _add_indel_candidate(
    counts: dict[tuple[int, str, str, str], dict[str, int | set[int]]],
    seen: set[tuple[int, str, str, str]],
    *,
    sample_idx: int,
    pos: int,
    ref: str,
    alt: str,
    variant_type: str,
    reverse: bool,
) -> None:
    key = (int(pos), str(ref).upper(), str(alt).upper(), str(variant_type))
    row = counts.setdefault(
        key,
        {
            "alt_count": 0,
            "sample_support": set(),
            "alt_forward_count": 0,
            "alt_reverse_count": 0,
        },
    )
    row["alt_count"] = int(row["alt_count"]) + 1
    if reverse:
        row["alt_reverse_count"] = int(row["alt_reverse_count"]) + 1
    else:
        row["alt_forward_count"] = int(row["alt_forward_count"]) + 1
    if key not in seen:
        support = row["sample_support"]
        assert isinstance(support, set)
        support.add(int(sample_idx))
        seen.add(key)


def _discover_indel_candidates_pysam(
    bam_paths: list[str],
    *,
    reference_fasta: str | Path,
    chromosome: str,
    region_start0: int,
    region_stop0: int,
    include_insertions: bool,
    include_deletions: bool,
    max_indel_len: int,
    min_base_quality: int,
    min_mapping_quality: int,
    htslib_threads_per_file: int,
    max_insert_size: int,
    cap_base_quality_by_mapping_quality: bool,
    min_alt_count: int,
    min_alt_samples: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        import pysam
    except Exception as exc:  # pragma: no cover - package dependency.
        raise RuntimeError("pysam is required for insertion/deletion discovery.") from exc

    counts: dict[tuple[int, str, str, str], dict[str, int | set[int]]] = {}
    n_reads_seen = 0
    with pysam.FastaFile(str(reference_fasta)) as fasta:
        for sample_idx, bam_path in enumerate(bam_paths):
            seen: set[tuple[int, str, str, str]] = set()
            with pysam.AlignmentFile(str(bam_path), "rb", threads=max(int(htslib_threads_per_file), 1)) as bam:
                for read in bam.fetch(str(chromosome), int(region_start0), int(region_stop0)):
                    if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate:
                        continue
                    if int(read.mapping_quality) < int(min_mapping_quality):
                        continue
                    if int(max_insert_size) > 0 and abs(int(read.template_length)) > int(max_insert_size):
                        continue
                    if read.query_sequence is None or read.cigartuples is None:
                        continue
                    qualities = read.query_qualities
                    if qualities is None:
                        qualities = [30] * len(read.query_sequence)
                    n_reads_seen += 1
                    ref_pos = int(read.reference_start)
                    qpos = 0
                    reverse = bool(read.is_reverse)
                    for op, length in read.cigartuples:
                        length = int(length)
                        if op in {0, 7, 8}:  # M, =, X
                            ref_pos += length
                            qpos += length
                        elif op == 1:  # insertion after previous reference base.
                            if include_insertions and 0 < length <= int(max_indel_len) and qpos > 0:
                                anchor0 = ref_pos - 1
                                if int(region_start0) <= anchor0 < int(region_stop0):
                                    q_anchor = _capped_quality(qualities[qpos - 1], int(read.mapping_quality), cap_base_quality_by_mapping_quality)
                                    q_insert = min(
                                        _capped_quality(qualities[qpos + offset], int(read.mapping_quality), cap_base_quality_by_mapping_quality)
                                        for offset in range(length)
                                    )
                                    if min(q_anchor, q_insert) >= int(min_base_quality):
                                        ref = fasta.fetch(str(chromosome), anchor0, anchor0 + 1).upper()
                                        inserted = read.query_sequence[qpos : qpos + length].upper()
                                        if ref and inserted:
                                            _add_indel_candidate(
                                                counts,
                                                seen,
                                                sample_idx=sample_idx,
                                                pos=anchor0 + 1,
                                                ref=ref,
                                                alt=ref + inserted,
                                                variant_type="insertion",
                                                reverse=reverse,
                                            )
                            qpos += length
                        elif op == 2:  # deletion after previous reference base.
                            if include_deletions and 0 < length <= int(max_indel_len) and qpos > 0:
                                anchor0 = ref_pos - 1
                                if int(region_start0) <= anchor0 < int(region_stop0):
                                    q_left = _capped_quality(qualities[qpos - 1], int(read.mapping_quality), cap_base_quality_by_mapping_quality)
                                    q_right = q_left
                                    if qpos < len(qualities):
                                        q_right = _capped_quality(qualities[qpos], int(read.mapping_quality), cap_base_quality_by_mapping_quality)
                                    if min(q_left, q_right) >= int(min_base_quality):
                                        ref = fasta.fetch(str(chromosome), anchor0, ref_pos + length).upper()
                                        if len(ref) == length + 1:
                                            _add_indel_candidate(
                                                counts,
                                                seen,
                                                sample_idx=sample_idx,
                                                pos=anchor0 + 1,
                                                ref=ref,
                                                alt=ref[0],
                                                variant_type="deletion",
                                                reverse=reverse,
                                            )
                            ref_pos += length
                        elif op == 3:  # reference skip
                            ref_pos += length
                        elif op == 4:  # soft clip
                            qpos += length
                        elif op in {5, 6}:  # hard clip, pad
                            continue

    rows: list[dict[str, object]] = []
    for (pos, ref, alt, variant_type), row in counts.items():
        support = row["sample_support"]
        assert isinstance(support, set)
        alt_count = int(row["alt_count"])
        sample_support = len(support)
        if alt_count < int(min_alt_count) or sample_support < int(min_alt_samples):
            continue
        rows.append(
            {
                "CHR": str(chromosome),
                "POS": int(pos),
                "REF": ref,
                "ALT": alt,
                "variant_type": variant_type,
                "depth": np.uint32(alt_count),
                "ref_count": np.uint32(0),
                "alt_count": np.uint32(alt_count),
                "other_count": np.uint32(0),
                "alt_fraction": np.float32(1.0),
                "other_fraction": np.float32(0.0),
                "sample_support": np.uint32(sample_support),
                "a_count": np.uint32(0),
                "c_count": np.uint32(0),
                "g_count": np.uint32(0),
                "t_count": np.uint32(0),
                "alt_forward_count": np.uint32(int(row["alt_forward_count"])),
                "alt_reverse_count": np.uint32(int(row["alt_reverse_count"])),
            }
        )
    frame = pd.DataFrame(rows, columns=empty_discovery_frame().columns) if rows else empty_discovery_frame()
    return frame, {"n_indel_reads_seen": int(n_reads_seen), "n_indel_candidates": int(frame.shape[0])}


def empty_discovery_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CHR": pd.Series(dtype="object"),
            "POS": pd.Series(dtype="int64"),
            "REF": pd.Series(dtype="object"),
            "ALT": pd.Series(dtype="object"),
            "variant_type": pd.Series(dtype="object"),
            "depth": pd.Series(dtype="uint32"),
            "ref_count": pd.Series(dtype="uint32"),
            "alt_count": pd.Series(dtype="uint32"),
            "other_count": pd.Series(dtype="uint32"),
            "alt_fraction": pd.Series(dtype="float32"),
            "other_fraction": pd.Series(dtype="float32"),
            "sample_support": pd.Series(dtype="uint32"),
            "a_count": pd.Series(dtype="uint32"),
            "c_count": pd.Series(dtype="uint32"),
            "g_count": pd.Series(dtype="uint32"),
            "t_count": pd.Series(dtype="uint32"),
            "alt_forward_count": pd.Series(dtype="uint32"),
            "alt_reverse_count": pd.Series(dtype="uint32"),
        }
    )


def discover_snp_positions(
    samples: pd.DataFrame | str | Path,
    *,
    reference_fasta: str | Path,
    chromosome: str,
    chr_start: int | None = None,
    chr_end: int | None = None,
    bam_path_col: str = "bam_path",
    max_bams: int = 0,
    window_size: int = 1_000_000,
    min_base_quality: int = 13,
    min_mapping_quality: int = 20,
    htslib_threads_per_file: int = 1,
    max_insert_size: int = 0,
    variant_types: Iterable[str] | str = ("snp", "ins", "del"),
    max_indel_len: int = 50,
    cap_base_quality_by_mapping_quality: bool = False,
    min_depth: int = 3,
    min_alt_count: int = 2,
    min_alt_samples: int = 1,
    min_alt_fraction: float = 0.05,
    max_other_fraction: float = 0.20,
) -> tuple[pd.DataFrame, dict]:
    """Discover candidate biallelic SNP/indel positions from BAM/CRAM reads.

    This is a pre-imputation discovery pass. It scans reads with the native
    HTSlib extension for SNPs and a CIGAR-aware insertion/deletion pass, then
    returns reviewable candidate rows. It does not run the STITCHV2 HMM.
    """

    if _discover_snp_candidates_htslib is None:
        raise RuntimeError(
            "Native HTSlib discovery is unavailable. Build the optional extension with "
            "`python setup.py build_ext --inplace` in an environment with HTSlib."
        )

    sample_df = _read_table(samples) if isinstance(samples, (str, Path)) else samples.copy()
    bam_paths = _nonempty_bam_paths(sample_df, bam_path_col)
    if max_bams and int(max_bams) > 0:
        bam_paths = bam_paths[: int(max_bams)]
    if not bam_paths:
        raise ValueError("No non-empty BAM/CRAM paths were found in the samples table.")
    requested_types = _normalize_variant_types(variant_types)

    ref_len = _reference_length(reference_fasta, chromosome)
    start = int(chr_start) if chr_start is not None else 1
    end = int(chr_end) if chr_end is not None else ref_len
    end = min(end, ref_len)
    if start > end:
        raise ValueError(f"Requested interval {chromosome}:{start}-{end} is empty.")

    frames: list[pd.DataFrame] = []
    chunks: list[dict] = []
    for region_start0, region_stop0 in _chunk_ranges(start, end, int(window_size)):
        chunk_frames: list[pd.DataFrame] = []
        chunk_reads_seen = 0
        chunk_bases_seen = 0
        if "snp" in requested_types:
            result = _discover_snp_candidates_htslib(
                bam_paths,
                str(reference_fasta),
                str(chromosome),
                int(region_start0),
                int(region_stop0),
                int(min_base_quality),
                int(min_mapping_quality),
                int(htslib_threads_per_file),
                int(max_insert_size),
                bool(cap_base_quality_by_mapping_quality),
                int(min_depth),
                int(min_alt_count),
                int(min_alt_samples),
                float(min_alt_fraction),
                float(max_other_fraction),
            )
            chunk_frames.append(_native_result_to_frame(result, str(chromosome)))
            chunk_reads_seen += int(result.get("n_reads_seen", 0))
            chunk_bases_seen += int(result.get("n_bases_seen", 0))
        if {"insertion", "deletion"}.intersection(requested_types):
            indel_frame, indel_summary = _discover_indel_candidates_pysam(
                bam_paths,
                reference_fasta=reference_fasta,
                chromosome=str(chromosome),
                region_start0=int(region_start0),
                region_stop0=int(region_stop0),
                include_insertions=("insertion" in requested_types),
                include_deletions=("deletion" in requested_types),
                max_indel_len=int(max_indel_len),
                min_base_quality=int(min_base_quality),
                min_mapping_quality=int(min_mapping_quality),
                htslib_threads_per_file=int(htslib_threads_per_file),
                max_insert_size=int(max_insert_size),
                cap_base_quality_by_mapping_quality=bool(cap_base_quality_by_mapping_quality),
                min_alt_count=int(min_alt_count),
                min_alt_samples=int(min_alt_samples),
            )
            chunk_frames.append(indel_frame)
            chunk_reads_seen += int(indel_summary.get("n_indel_reads_seen", 0))
        frame = pd.concat([part for part in chunk_frames if not part.empty], ignore_index=True) if chunk_frames else empty_discovery_frame()
        if not frame.empty:
            frames.append(frame)
        chunks.append(
            {
                "region_start_1based": int(region_start0) + 1,
                "region_end_1based": int(region_stop0),
                "n_candidates": int(frame.shape[0]),
                "n_reads_seen": int(chunk_reads_seen),
                "n_bases_seen": int(chunk_bases_seen),
            }
        )

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values(["CHR", "POS", "REF", "ALT"]).drop_duplicates(["CHR", "POS", "REF", "ALT"]).reset_index(drop=True)
    else:
        out = empty_discovery_frame()

    summary = {
        "chromosome": str(chromosome),
        "chr_start": int(start),
        "chr_end": int(end),
        "reference_fasta": str(reference_fasta),
        "n_samples_in_table": int(sample_df.shape[0]),
        "n_bams_scanned": int(len(bam_paths)),
        "window_size": int(window_size),
        "n_chunks": int(len(chunks)),
        "n_candidates": int(out.shape[0]),
        "min_base_quality": int(min_base_quality),
        "min_mapping_quality": int(min_mapping_quality),
        "max_insert_size": int(max_insert_size),
        "variant_types": sorted(requested_types),
        "max_indel_len": int(max_indel_len),
        "cap_base_quality_by_mapping_quality": bool(cap_base_quality_by_mapping_quality),
        "min_depth": int(min_depth),
        "min_alt_count": int(min_alt_count),
        "min_alt_samples": int(min_alt_samples),
        "min_alt_fraction": float(min_alt_fraction),
        "max_other_fraction": float(max_other_fraction),
        "chunks": chunks,
    }
    return out, summary


def write_discovered_positions(
    frame: pd.DataFrame,
    output_file: str | Path,
    *,
    compression: str = "zstd",
) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffixes = {suffix.lower() for suffix in output_path.suffixes}
    if ".parquet" in suffixes:
        frame.to_parquet(output_path, index=False, compression=compression)
    elif ".tsv" in suffixes or ".txt" in suffixes:
        frame.to_csv(output_path, sep="\t", index=False)
    elif ".csv" in suffixes:
        frame.to_csv(output_path, index=False)
    else:
        raise ValueError("output_file must end with .parquet, .csv, .tsv, or .txt.")


def write_discovery_summary(summary: dict, output_file: str | Path) -> None:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


discover_variants = discover_snp_positions
