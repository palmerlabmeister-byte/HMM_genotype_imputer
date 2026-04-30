from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pysam
import shutil
import tempfile

from .io import PositionBlock

try:
    from ._htslib_readstream import extract_sample_read_stream as _extract_sample_read_stream_htslib
except Exception:  # pragma: no cover - optional compiled extension.
    _extract_sample_read_stream_htslib = None


DEFAULT_BASE_QUALITY = 10
OBS_REF = np.int8(0)
OBS_ALT = np.int8(1)
OBS_OTHER = np.int8(2)
ASCII_UPPER_MASK = np.uint8(0xDF)


@dataclass(slots=True)
class ReadEvidenceBlock:
    block_id: int
    chromosome: str
    positions: np.ndarray
    ref: np.ndarray
    alt: np.ndarray
    ref_count: np.ndarray
    alt_count: np.ndarray
    other_count: np.ndarray
    depth: np.ndarray
    ref_weight: np.ndarray
    alt_weight: np.ndarray
    other_weight: np.ndarray
    sample_ids: np.ndarray
    n_overlapping_reads: np.ndarray
    fragment_sample_offsets: np.ndarray
    fragment_center_idx: np.ndarray
    fragment_obs_offsets: np.ndarray
    fragment_obs_pos_idx: np.ndarray
    fragment_obs_code: np.ndarray
    fragment_obs_qual: np.ndarray
    memmap_dir: str | None = None

    @property
    def n_fragments(self) -> int:
        return int(self.fragment_center_idx.shape[0])

    def release(self) -> None:
        for arr in (
            self.ref_count,
            self.alt_count,
            self.other_count,
            self.depth,
            self.ref_weight,
            self.alt_weight,
            self.other_weight,
        ):
            mmap_obj = getattr(arr, "_mmap", None)
            if mmap_obj is not None:
                arr.flush()
                mmap_obj.close()
        if self.memmap_dir is not None:
            shutil.rmtree(self.memmap_dir, ignore_errors=True)

    def to_arrow(self) -> pa.Table:
        sample_idx, pos_idx = np.indices(self.ref_count.shape)
        return pa.table(
            {
                "sample_id": self.sample_ids[sample_idx.reshape(-1)],
                "chromosome": np.repeat(self.chromosome, sample_idx.size),
                "position": self.positions[pos_idx.reshape(-1)],
                "ref": self.ref[pos_idx.reshape(-1)],
                "alt": self.alt[pos_idx.reshape(-1)],
                "ref_count": self.ref_count.reshape(-1),
                "alt_count": self.alt_count.reshape(-1),
                "other_count": self.other_count.reshape(-1),
                "depth": self.depth.reshape(-1),
                "ref_weight": self.ref_weight.reshape(-1),
                "alt_weight": self.alt_weight.reshape(-1),
                "other_weight": self.other_weight.reshape(-1),
                "block_id": np.repeat(self.block_id, sample_idx.size),
            }
        )


def _base_weight(quality: int) -> float:
    error_rate = 10.0 ** (-(float(quality) / 10.0))
    return 1.0 - error_rate


def _compress_fragment(
    pos_values: np.ndarray,
    obs_values: np.ndarray,
    qual_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if pos_values.size == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int8),
            np.empty((0,), dtype=np.uint8),
        )
    pos = pos_values.astype(np.int32, copy=False)
    obs = obs_values.astype(np.int8, copy=False)
    qual = qual_values.astype(np.uint8, copy=False)
    if pos.shape[0] == 1:
        return pos, obs, qual

    order = np.argsort(pos, kind="stable")
    pos_sorted = pos[order]
    obs_sorted = obs[order]
    qual_sorted = qual[order]

    unique_starts = np.flatnonzero(np.r_[True, pos_sorted[1:] != pos_sorted[:-1]])
    if unique_starts.shape[0] == pos_sorted.shape[0]:
        return pos_sorted, obs_sorted, qual_sorted

    unique_ends = np.r_[unique_starts[1:], pos_sorted.shape[0]]
    best_idx = np.empty(unique_starts.shape[0], dtype=np.int32)
    for idx, (start, end) in enumerate(zip(unique_starts, unique_ends, strict=False)):
        best_idx[idx] = int(start + np.argmax(qual_sorted[start:end]))
    return (
        pos_sorted[best_idx].astype(np.int32, copy=False),
        obs_sorted[best_idx].astype(np.int8, copy=False),
        qual_sorted[best_idx].astype(np.uint8, copy=False),
    )


class PysamReadExtractor:
    def __init__(
        self,
        chromosome: str,
        *,
        mode: Literal["read_stream", "pileup"] = "read_stream",
        read_stream_backend: Literal["auto", "python", "htslib"] = "auto",
        min_base_quality: int = 13,
        min_mapping_quality: int = 20,
        merge_fragments_by_query: bool = True,
        read_batch_size: int = 1024,
        io_workers: int = 1,
        htslib_threads_per_file: int = 1,
        memory_map_read_matrices: bool = False,
        memory_map_dir: str | Path | None = None,
        subsample_seed: int = 0,
    ):
        self.chromosome = chromosome
        self.mode = mode
        self.read_stream_backend = read_stream_backend
        self.min_base_quality = min_base_quality
        self.min_mapping_quality = min_mapping_quality
        self.merge_fragments_by_query = merge_fragments_by_query
        self.read_batch_size = max(int(read_batch_size), 64)
        self.io_workers = max(int(io_workers), 1)
        self.htslib_threads_per_file = max(int(htslib_threads_per_file), 1)
        self.memory_map_read_matrices = bool(memory_map_read_matrices)
        self.memory_map_dir = None if memory_map_dir is None else Path(memory_map_dir)
        self.subsample_seed = int(subsample_seed)
        self._qual_weight_lut = np.array([_base_weight(q) for q in range(256)], dtype=np.float32)
        self._handles: list[pysam.AlignmentFile] | None = None
        self._bam_paths: list[str] | None = None
        self._sample_ids: np.ndarray | None = None
        self._read_keep_prob: np.ndarray | None = None
        self._subsample_rngs: list[np.random.Generator] | None = None

        if self.read_stream_backend == "htslib" and _extract_sample_read_stream_htslib is None:
            raise RuntimeError(
                "read_stream_backend='htslib' requested but compiled extension is unavailable. "
                "Build/install STITCHV2 with the HTSlib extension or use read_stream_backend='python'/'auto'."
            )
        self._use_compiled_read_stream = (
            self.mode == "read_stream"
            and self.read_stream_backend != "python"
            and _extract_sample_read_stream_htslib is not None
        )
        self._use_compiled_read_stream_runtime = self._use_compiled_read_stream

    def open(self, samples: pd.DataFrame) -> None:
        self.close()
        self._sample_ids = samples["sample_id"].astype(str).to_numpy()
        self._bam_paths = [str(Path(path)) for path in samples["bam_path"].astype(str)]
        if "read_subsample_prob" in samples.columns:
            keep_prob = samples["read_subsample_prob"].to_numpy(dtype=np.float32, copy=False)
            keep_prob = np.clip(np.nan_to_num(keep_prob, nan=1.0), 0.0, 1.0).astype(np.float32, copy=False)
        else:
            keep_prob = np.ones(len(samples), dtype=np.float32)
        self._read_keep_prob = keep_prob

        if "read_subsample_seed" in samples.columns:
            seed_series = pd.to_numeric(samples["read_subsample_seed"], errors="coerce")
            seed_fill = np.arange(len(samples), dtype=np.int64) + int(self.subsample_seed)
            seed_vec = np.where(
                np.isfinite(seed_series.to_numpy(dtype=np.float64, copy=False)),
                seed_series.to_numpy(dtype=np.float64, copy=False).astype(np.int64, copy=False),
                seed_fill,
            ).astype(np.int64, copy=False)
        else:
            seed_vec = np.arange(len(samples), dtype=np.int64) + int(self.subsample_seed)
        self._subsample_rngs = [np.random.default_rng(int(seed)) for seed in seed_vec.tolist()]

        self._use_compiled_read_stream_runtime = bool(self._use_compiled_read_stream)
        if self._use_compiled_read_stream_runtime and np.any(keep_prob < 0.999999):
            # Compiled extension currently assumes full-read extraction. Fallback to Python path for on-the-fly subsampling.
            self._use_compiled_read_stream_runtime = False

        if self.mode == "pileup" or not self._use_compiled_read_stream_runtime:
            self._handles = [
                pysam.AlignmentFile(
                    path,
                    "rb",
                    threads=self.htslib_threads_per_file,
                )
                for path in self._bam_paths
            ]
        else:
            self._handles = None

    def close(self) -> None:
        if self._handles is not None:
            for handle in self._handles:
                handle.close()
        self._handles = None
        self._bam_paths = None
        self._sample_ids = None
        self._read_keep_prob = None
        self._subsample_rngs = None

    def _create_block_memmap_dir(self, block_id: int) -> str | None:
        if not self.memory_map_read_matrices:
            return None
        root = self.memory_map_dir
        if root is None:
            root = Path(tempfile.gettempdir()) / "stitchv2_memmap"
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.mkdtemp(prefix=f"block_{block_id:06d}_", dir=str(root))

    @staticmethod
    def _zero_matrix(shape: tuple[int, int], dtype, memmap_dir: str | None, name: str) -> np.ndarray:
        if memmap_dir is None:
            return np.zeros(shape, dtype=dtype)
        path = Path(memmap_dir) / f"{name}.mmap"
        out = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        out.fill(0)
        return out

    def extract_block(self, samples: pd.DataFrame, block: PositionBlock) -> ReadEvidenceBlock:
        if ((not self._use_compiled_read_stream_runtime) and self._handles is None) or self._sample_ids is None:
            self.open(samples)
        assert self._sample_ids is not None

        positions = block.dataframe["POS"].to_numpy(dtype=np.int64)
        ref = block.dataframe["REF"].astype(str).str.upper().to_numpy()
        alt = block.dataframe["ALT"].astype(str).str.upper().to_numpy()
        n_samples = len(samples)
        n_positions = len(positions)
        shape = (n_samples, n_positions)
        memmap_dir = self._create_block_memmap_dir(block.block_id)
        ref_count = self._zero_matrix(shape, np.uint16, memmap_dir, "ref_count")
        alt_count = self._zero_matrix(shape, np.uint16, memmap_dir, "alt_count")
        other_count = self._zero_matrix(shape, np.uint16, memmap_dir, "other_count")
        depth = self._zero_matrix(shape, np.uint16, memmap_dir, "depth")
        ref_weight = self._zero_matrix(shape, np.float32, memmap_dir, "ref_weight")
        alt_weight = self._zero_matrix(shape, np.float32, memmap_dir, "alt_weight")
        other_weight = self._zero_matrix(shape, np.float32, memmap_dir, "other_weight")
        n_overlapping_reads = np.zeros(n_samples, dtype=np.int32)

        fragment_sample_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        fragment_center_idx = np.empty((0,), dtype=np.int32)
        fragment_obs_offsets = np.zeros(1, dtype=np.int64)
        fragment_obs_pos_idx = np.empty((0,), dtype=np.int32)
        fragment_obs_code = np.empty((0,), dtype=np.int8)
        fragment_obs_qual = np.empty((0,), dtype=np.uint8)

        if n_positions > 0:
            if self.mode == "pileup":
                self._extract_with_pileup(
                    positions=positions,
                    ref=ref,
                    alt=alt,
                    ref_count=ref_count,
                    alt_count=alt_count,
                    other_count=other_count,
                    depth=depth,
                    ref_weight=ref_weight,
                    alt_weight=alt_weight,
                    other_weight=other_weight,
                    n_overlapping_reads=n_overlapping_reads,
                )
            else:
                (
                    fragment_sample_offsets,
                    fragment_center_idx,
                    fragment_obs_offsets,
                    fragment_obs_pos_idx,
                    fragment_obs_code,
                    fragment_obs_qual,
                ) = self._extract_with_read_stream(
                    positions=positions,
                    ref=ref,
                    alt=alt,
                    ref_count=ref_count,
                    alt_count=alt_count,
                    other_count=other_count,
                    depth=depth,
                    ref_weight=ref_weight,
                    alt_weight=alt_weight,
                    other_weight=other_weight,
                    n_overlapping_reads=n_overlapping_reads,
                )

        return ReadEvidenceBlock(
            block_id=block.block_id,
            chromosome=self.chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            depth=depth,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
            sample_ids=self._sample_ids,
            n_overlapping_reads=n_overlapping_reads,
            fragment_sample_offsets=fragment_sample_offsets,
            fragment_center_idx=fragment_center_idx,
            fragment_obs_offsets=fragment_obs_offsets,
            fragment_obs_pos_idx=fragment_obs_pos_idx,
            fragment_obs_code=fragment_obs_code,
            fragment_obs_qual=fragment_obs_qual,
            memmap_dir=memmap_dir,
        )

    def _extract_with_pileup(
        self,
        *,
        positions: np.ndarray,
        ref: np.ndarray,
        alt: np.ndarray,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray,
        depth: np.ndarray,
        ref_weight: np.ndarray,
        alt_weight: np.ndarray,
        other_weight: np.ndarray,
        n_overlapping_reads: np.ndarray,
    ) -> None:
        pos_to_idx = {int(pos): idx for idx, pos in enumerate(positions)}
        region_start = int(positions[0]) - 1
        region_stop = int(positions[-1])
        assert self._handles is not None
        for sample_idx, bam in enumerate(self._handles):
            keep_prob = 1.0 if self._read_keep_prob is None else float(self._read_keep_prob[sample_idx])
            rng = None if self._subsample_rngs is None else self._subsample_rngs[sample_idx]
            for pileup_col in bam.pileup(
                self.chromosome,
                region_start,
                region_stop,
                truncate=True,
                min_base_quality=self.min_base_quality,
                min_mapping_quality=self.min_mapping_quality,
                stepper="samtools",
            ):
                pos = int(pileup_col.reference_pos) + 1
                target_idx = pos_to_idx.get(pos)
                if target_idx is None:
                    continue
                ref_base = ref[target_idx]
                alt_base = alt[target_idx]
                for pileup_read in pileup_col.pileups:
                    if keep_prob < 0.999999:
                        draw = (rng.random() if rng is not None else np.random.random())
                        if draw > keep_prob:
                            continue
                    if pileup_read.is_del or pileup_read.is_refskip:
                        continue
                    query_position = pileup_read.query_position
                    if query_position is None:
                        continue
                    qual = (
                        int(pileup_read.alignment.query_qualities[query_position])
                        if pileup_read.alignment.query_qualities
                        else DEFAULT_BASE_QUALITY
                    )
                    if qual < self.min_base_quality:
                        continue
                    base = pileup_read.alignment.query_sequence[query_position].upper()
                    weight = _base_weight(qual)
                    depth[sample_idx, target_idx] += 1
                    if base == ref_base:
                        ref_count[sample_idx, target_idx] += 1
                        ref_weight[sample_idx, target_idx] += weight
                    elif base == alt_base:
                        alt_count[sample_idx, target_idx] += 1
                        alt_weight[sample_idx, target_idx] += weight
                    else:
                        other_count[sample_idx, target_idx] += 1
                        other_weight[sample_idx, target_idx] += weight
            n_overlapping_reads[sample_idx] = int(np.sum(depth[sample_idx] > 0))

    def _extract_with_read_stream(
        self,
        *,
        positions: np.ndarray,
        ref: np.ndarray,
        alt: np.ndarray,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray,
        depth: np.ndarray,
        ref_weight: np.ndarray,
        alt_weight: np.ndarray,
        other_weight: np.ndarray,
        n_overlapping_reads: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        region_start = int(positions[0]) - 1
        region_stop = int(positions[-1])
        lookup = np.full(region_stop - region_start + 1, -1, dtype=np.int32)
        lookup[positions - (region_start + 1)] = np.arange(positions.shape[0], dtype=np.int32)
        ref_codes = np.frombuffer("".join(ref.tolist()).encode("ascii"), dtype=np.uint8)
        alt_codes = np.frombuffer("".join(alt.tolist()).encode("ascii"), dtype=np.uint8)

        if self._use_compiled_read_stream_runtime:
            assert self._bam_paths is not None
            n_samples = len(self._bam_paths)
        else:
            assert self._handles is not None
            n_samples = len(self._handles)
        sample_results: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = [None] * n_samples

        def process_sample(
            sample_idx: int,
        ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            if self._use_compiled_read_stream_runtime:
                assert self._bam_paths is not None
                keep_prob = 1.0 if self._read_keep_prob is None else float(self._read_keep_prob[sample_idx])
                return self._extract_sample_read_stream_compiled(
                    bam_path=self._bam_paths[sample_idx],
                    region_start=region_start,
                    region_stop=region_stop,
                    lookup=lookup,
                    ref_codes=ref_codes,
                    alt_codes=alt_codes,
                    depth_row=depth[sample_idx],
                    ref_row=ref_count[sample_idx],
                    alt_row=alt_count[sample_idx],
                    other_row=other_count[sample_idx],
                    ref_w_row=ref_weight[sample_idx],
                    alt_w_row=alt_weight[sample_idx],
                    other_w_row=other_weight[sample_idx],
                    n_positions=positions.shape[0],
                    keep_prob=keep_prob,
                )
            assert self._handles is not None
            keep_prob = 1.0 if self._read_keep_prob is None else float(self._read_keep_prob[sample_idx])
            rng = None if self._subsample_rngs is None else self._subsample_rngs[sample_idx]
            return self._extract_sample_read_stream(
                bam=self._handles[sample_idx],
                region_start=region_start,
                region_stop=region_stop,
                lookup=lookup,
                ref_codes=ref_codes,
                alt_codes=alt_codes,
                depth_row=depth[sample_idx],
                ref_row=ref_count[sample_idx],
                alt_row=alt_count[sample_idx],
                other_row=other_count[sample_idx],
                ref_w_row=ref_weight[sample_idx],
                alt_w_row=alt_weight[sample_idx],
                other_w_row=other_weight[sample_idx],
                n_positions=positions.shape[0],
                keep_prob=keep_prob,
                rng=rng,
            )

        if self.io_workers > 1 and n_samples > 1:
            n_workers = min(self.io_workers, n_samples)
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                for sample_idx, sample_result in enumerate(executor.map(process_sample, range(n_samples))):
                    sample_results[sample_idx] = sample_result
        else:
            for sample_idx in range(n_samples):
                sample_results[sample_idx] = process_sample(sample_idx)

        sample_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        all_centers_chunks: list[np.ndarray] = []
        all_obs_offsets: list[int] = [0]
        all_obs_pos_chunks: list[np.ndarray] = []
        all_obs_code_chunks: list[np.ndarray] = []
        all_obs_qual_chunks: list[np.ndarray] = []

        for sample_idx, result in enumerate(sample_results):
            assert result is not None
            n_reads, sample_centers, sample_obs_offsets, sample_obs_pos, sample_obs_code, sample_obs_qual = result
            n_overlapping_reads[sample_idx] = int(n_reads)
            sample_offsets[sample_idx + 1] = sample_offsets[sample_idx] + sample_centers.shape[0]
            if sample_centers.size:
                all_centers_chunks.append(sample_centers)
            base_offset = all_obs_offsets[-1]
            if sample_obs_offsets.shape[0] > 1:
                all_obs_offsets.extend((base_offset + sample_obs_offsets[1:]).tolist())
            if sample_obs_pos.size:
                all_obs_pos_chunks.append(sample_obs_pos)
                all_obs_code_chunks.append(sample_obs_code)
                all_obs_qual_chunks.append(sample_obs_qual)

        return (
            sample_offsets,
            np.concatenate(all_centers_chunks) if all_centers_chunks else np.empty((0,), dtype=np.int32),
            np.asarray(all_obs_offsets, dtype=np.int64),
            np.concatenate(all_obs_pos_chunks) if all_obs_pos_chunks else np.empty((0,), dtype=np.int32),
            np.concatenate(all_obs_code_chunks) if all_obs_code_chunks else np.empty((0,), dtype=np.int8),
            np.concatenate(all_obs_qual_chunks) if all_obs_qual_chunks else np.empty((0,), dtype=np.uint8),
        )

    def _extract_sample_read_stream(
        self,
        *,
        bam: pysam.AlignmentFile,
        region_start: int,
        region_stop: int,
        lookup: np.ndarray,
        ref_codes: np.ndarray,
        alt_codes: np.ndarray,
        depth_row: np.ndarray,
        ref_row: np.ndarray,
        alt_row: np.ndarray,
        other_row: np.ndarray,
        ref_w_row: np.ndarray,
        alt_w_row: np.ndarray,
        other_w_row: np.ndarray,
        n_positions: int,
        keep_prob: float = 1.0,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_reads = 0
        sample_centers: list[int] = []
        sample_obs_offsets: list[int] = [0]
        fragment_pos_chunks: list[np.ndarray] = []
        fragment_code_chunks: list[np.ndarray] = []
        fragment_qual_chunks: list[np.ndarray] = []
        pending: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        sample_targets: list[np.ndarray] = []
        sample_codes_raw: list[np.ndarray] = []
        sample_quals_raw: list[np.ndarray] = []

        def append_fragment(pos_arr: np.ndarray, code_arr: np.ndarray, qual_arr: np.ndarray) -> None:
            pos_compact, code_compact, qual_compact = _compress_fragment(pos_arr, code_arr, qual_arr)
            if pos_compact.size == 0:
                return
            sample_centers.append(int(pos_compact[pos_compact.shape[0] // 2]))
            fragment_pos_chunks.append(pos_compact)
            fragment_code_chunks.append(code_compact)
            fragment_qual_chunks.append(qual_compact)
            sample_obs_offsets.append(sample_obs_offsets[-1] + int(pos_compact.shape[0]))

        def flush_sample_observations() -> None:
            if not sample_targets:
                return
            target_idx = np.concatenate(sample_targets, axis=0)
            obs_code = np.concatenate(sample_codes_raw, axis=0)
            qual = np.concatenate(sample_quals_raw, axis=0)
            w = self._qual_weight_lut[qual]
            depth_row[...] += np.bincount(target_idx, minlength=n_positions).astype(depth_row.dtype, copy=False)

            ref_hits = obs_code == int(OBS_REF)
            if np.any(ref_hits):
                idx = target_idx[ref_hits]
                ref_row[...] += np.bincount(idx, minlength=n_positions).astype(ref_row.dtype, copy=False)
                ref_w_row[...] += np.bincount(idx, weights=w[ref_hits], minlength=n_positions).astype(np.float32, copy=False)

            alt_hits = obs_code == int(OBS_ALT)
            if np.any(alt_hits):
                idx = target_idx[alt_hits]
                alt_row[...] += np.bincount(idx, minlength=n_positions).astype(alt_row.dtype, copy=False)
                alt_w_row[...] += np.bincount(idx, weights=w[alt_hits], minlength=n_positions).astype(np.float32, copy=False)

            other_hits = obs_code == int(OBS_OTHER)
            if np.any(other_hits):
                idx = target_idx[other_hits]
                other_row[...] += np.bincount(idx, minlength=n_positions).astype(other_row.dtype, copy=False)
                other_w_row[...] += np.bincount(idx, weights=w[other_hits], minlength=n_positions).astype(np.float32, copy=False)

            sample_targets.clear()
            sample_codes_raw.clear()
            sample_quals_raw.clear()

        for read in bam.fetch(self.chromosome, region_start, region_stop):
            if read.is_unmapped or read.is_secondary or read.is_supplementary or read.is_duplicate:
                continue
            if read.mapping_quality < self.min_mapping_quality:
                continue
            if keep_prob < 0.999999:
                draw = (rng.random() if rng is not None else np.random.random())
                if draw > keep_prob:
                    continue
            sequence = read.query_sequence
            if sequence is None:
                continue
            aligned_pairs = read.get_aligned_pairs(matches_only=True)
            if not aligned_pairs:
                continue
            pair_arr = np.asarray(aligned_pairs, dtype=np.int32)
            qpos = pair_arr[:, 0]
            rel = pair_arr[:, 1] - region_start
            in_region = (rel >= 0) & (rel < lookup.shape[0])
            if not np.any(in_region):
                continue
            qpos = qpos[in_region]
            target_idx = lookup[rel[in_region]]
            on_target = target_idx >= 0
            if not np.any(on_target):
                continue
            qpos = qpos[on_target]
            target_idx = target_idx[on_target].astype(np.int32, copy=False)

            quals = read.query_qualities
            if quals is None:
                qual = np.full(target_idx.shape[0], DEFAULT_BASE_QUALITY, dtype=np.uint8)
            else:
                try:
                    qual = np.frombuffer(quals, dtype=np.uint8)[qpos]
                except TypeError:
                    qual = np.asarray(quals, dtype=np.uint8)[qpos]
            pass_q = qual >= self.min_base_quality
            if not np.any(pass_q):
                continue
            qpos = qpos[pass_q]
            target_idx = target_idx[pass_q]
            qual = qual[pass_q]

            seq_codes = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
            obs_base = np.bitwise_and(seq_codes[qpos], ASCII_UPPER_MASK)
            obs_ref = ref_codes[target_idx]
            obs_alt = alt_codes[target_idx]
            obs_code = np.full(target_idx.shape[0], int(OBS_OTHER), dtype=np.int8)
            ref_mask = obs_base == obs_ref
            alt_mask = (~ref_mask) & (obs_base == obs_alt)
            obs_code[ref_mask] = int(OBS_REF)
            obs_code[alt_mask] = int(OBS_ALT)

            n_reads += 1
            sample_targets.append(target_idx)
            sample_codes_raw.append(obs_code)
            sample_quals_raw.append(qual)
            if len(sample_targets) >= self.read_batch_size:
                flush_sample_observations()

            if self.merge_fragments_by_query and read.is_paired:
                key = read.query_name
                existing = pending.pop(key, None)
                if existing is None:
                    pending[key] = (target_idx, obs_code, qual)
                else:
                    append_fragment(
                        np.concatenate((existing[0], target_idx), axis=0),
                        np.concatenate((existing[1], obs_code), axis=0),
                        np.concatenate((existing[2], qual), axis=0),
                    )
            else:
                append_fragment(target_idx, obs_code, qual)

        if self.merge_fragments_by_query:
            for pos_arr, code_arr, qual_arr in pending.values():
                append_fragment(pos_arr, code_arr, qual_arr)
        flush_sample_observations()

        return (
            n_reads,
            np.asarray(sample_centers, dtype=np.int32),
            np.asarray(sample_obs_offsets, dtype=np.int64),
            np.concatenate(fragment_pos_chunks) if fragment_pos_chunks else np.empty((0,), dtype=np.int32),
            np.concatenate(fragment_code_chunks) if fragment_code_chunks else np.empty((0,), dtype=np.int8),
            np.concatenate(fragment_qual_chunks) if fragment_qual_chunks else np.empty((0,), dtype=np.uint8),
        )

    def _extract_sample_read_stream_compiled(
        self,
        *,
        bam_path: str,
        region_start: int,
        region_stop: int,
        lookup: np.ndarray,
        ref_codes: np.ndarray,
        alt_codes: np.ndarray,
        depth_row: np.ndarray,
        ref_row: np.ndarray,
        alt_row: np.ndarray,
        other_row: np.ndarray,
        ref_w_row: np.ndarray,
        alt_w_row: np.ndarray,
        other_w_row: np.ndarray,
        n_positions: int,
        keep_prob: float = 1.0,
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if _extract_sample_read_stream_htslib is None:
            raise RuntimeError("Compiled HTSlib read extractor is not available.")
        if keep_prob < 0.999999:
            raise RuntimeError(
                "Compiled HTSlib read extraction does not support per-sample read subsampling; "
                "fallback to Python read_stream backend."
            )
        (
            depth_local,
            ref_local,
            alt_local,
            other_local,
            ref_w_local,
            alt_w_local,
            other_w_local,
            n_reads,
            sample_centers,
            sample_obs_offsets,
            sample_obs_pos,
            sample_obs_code,
            sample_obs_qual,
        ) = _extract_sample_read_stream_htslib(
            bam_path,
            self.chromosome,
            int(region_start),
            int(region_stop),
            lookup.astype(np.int32, copy=False),
            ref_codes.astype(np.uint8, copy=False),
            alt_codes.astype(np.uint8, copy=False),
            int(self.min_base_quality),
            int(self.min_mapping_quality),
            bool(self.merge_fragments_by_query),
            int(self.htslib_threads_per_file),
        )
        if depth_local.shape[0] != n_positions:
            raise RuntimeError(
                f"Compiled extractor returned unexpected depth shape {depth_local.shape} for n_positions={n_positions}."
            )
        depth_row[...] = depth_local.astype(depth_row.dtype, copy=False)
        ref_row[...] = ref_local.astype(ref_row.dtype, copy=False)
        alt_row[...] = alt_local.astype(alt_row.dtype, copy=False)
        other_row[...] = other_local.astype(other_row.dtype, copy=False)
        ref_w_row[...] = ref_w_local.astype(np.float32, copy=False)
        alt_w_row[...] = alt_w_local.astype(np.float32, copy=False)
        other_w_row[...] = other_w_local.astype(np.float32, copy=False)
        return (
            int(n_reads),
            sample_centers.astype(np.int32, copy=False),
            sample_obs_offsets.astype(np.int64, copy=False),
            sample_obs_pos.astype(np.int32, copy=False),
            sample_obs_code.astype(np.int8, copy=False),
            sample_obs_qual.astype(np.uint8, copy=False),
        )
