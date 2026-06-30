from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pysam
import shutil
import tempfile
import json

from .io import PositionBlock, infer_variant_type

try:
    from ._htslib_readstream import extract_sample_read_stream as _extract_sample_read_stream_htslib
    from ._htslib_readstream import extract_samples_read_stream_batch as _extract_samples_read_stream_batch_htslib
    from ._htslib_readstream import materialize_dense_from_fragments as _materialize_dense_from_fragments_htslib
except Exception:  # pragma: no cover - optional compiled extension.
    _extract_sample_read_stream_htslib = None
    _extract_samples_read_stream_batch_htslib = None
    _materialize_dense_from_fragments_htslib = None


DEFAULT_BASE_QUALITY = 10
OBS_REF = np.int8(0)
OBS_ALT = np.int8(1)
OBS_OTHER = np.int8(2)
ASCII_UPPER_MASK = np.uint8(0xDF)
VARIANT_SNP = np.uint8(0)
VARIANT_INSERTION = np.uint8(1)
VARIANT_DELETION = np.uint8(2)
_VARIANT_TYPE_TO_CODE = {
    "snp": VARIANT_SNP,
    "insertion": VARIANT_INSERTION,
    "ins": VARIANT_INSERTION,
    "deletion": VARIANT_DELETION,
    "del": VARIANT_DELETION,
}


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

    @property
    def dense_nbytes(self) -> int:
        return int(
            self.ref_count.nbytes
            + self.alt_count.nbytes
            + self.other_count.nbytes
            + self.depth.nbytes
            + self.ref_weight.nbytes
            + self.alt_weight.nbytes
            + self.other_weight.nbytes
            + self.n_overlapping_reads.nbytes
        )

    @property
    def compact_nbytes(self) -> int:
        return int(
            self.fragment_sample_offsets.nbytes
            + self.fragment_center_idx.nbytes
            + self.fragment_obs_offsets.nbytes
            + self.fragment_obs_pos_idx.nbytes
            + self.fragment_obs_code.nbytes
            + self.fragment_obs_qual.nbytes
            + self.n_overlapping_reads.nbytes
        )

    def save_compact_cache(
        self,
        path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
        include_dense_counts: bool = False,
    ) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(metadata or {})
        meta.setdefault("chromosome", str(self.chromosome))
        meta.setdefault("block_id", int(self.block_id))
        meta.setdefault("n_samples", int(self.sample_ids.shape[0]))
        meta.setdefault("n_positions", int(self.positions.shape[0]))
        meta.setdefault("compact_nbytes", int(self.compact_nbytes))
        meta.setdefault("dense_nbytes", int(self.dense_nbytes))
        meta.setdefault("includes_dense_counts", bool(include_dense_counts))
        payload: dict[str, np.ndarray] = {
            "metadata_json": np.asarray(json.dumps(meta, sort_keys=True)),
            "chromosome": np.asarray(str(self.chromosome)),
            "positions": self.positions.astype(np.int64, copy=False),
            "ref": np.asarray(self.ref, dtype="U16"),
            "alt": np.asarray(self.alt, dtype="U16"),
            "sample_ids": np.asarray(self.sample_ids, dtype="U256"),
            "n_overlapping_reads": self.n_overlapping_reads.astype(np.int32, copy=False),
            "fragment_sample_offsets": self.fragment_sample_offsets.astype(np.int64, copy=False),
            "fragment_center_idx": self.fragment_center_idx.astype(np.int32, copy=False),
            "fragment_obs_offsets": self.fragment_obs_offsets.astype(np.int64, copy=False),
            "fragment_obs_pos_idx": self.fragment_obs_pos_idx.astype(np.int32, copy=False),
            "fragment_obs_code": self.fragment_obs_code.astype(np.int8, copy=False),
            "fragment_obs_qual": self.fragment_obs_qual.astype(np.uint8, copy=False),
        }
        if include_dense_counts:
            payload.update(
                {
                    "ref_count": self.ref_count.astype(np.uint16, copy=False),
                    "alt_count": self.alt_count.astype(np.uint16, copy=False),
                    "other_count": self.other_count.astype(np.uint16, copy=False),
                    "depth": self.depth.astype(np.uint16, copy=False),
                    "ref_weight": self.ref_weight.astype(np.float32, copy=False),
                    "alt_weight": self.alt_weight.astype(np.float32, copy=False),
                    "other_weight": self.other_weight.astype(np.float32, copy=False),
                }
            )
        np.savez(out, **payload)

    @staticmethod
    def compact_cache_metadata(path: str | Path) -> dict[str, Any]:
        with np.load(Path(path), allow_pickle=False) as data:
            if "metadata_json" not in data.files:
                return {}
            raw = np.asarray(data["metadata_json"]).item()
            try:
                return json.loads(str(raw))
            except Exception:
                return {}

    @staticmethod
    def load_compact_cache(
        path: str | Path,
        *,
        block_id: int,
        materialize_dense_counts: bool = True,
    ) -> "ReadEvidenceBlock":
        with np.load(Path(path), allow_pickle=False) as data:
            chromosome = str(np.asarray(data["chromosome"]).item())
            positions = np.asarray(data["positions"], dtype=np.int64)
            ref = np.asarray(data["ref"]).astype(object)
            alt = np.asarray(data["alt"]).astype(object)
            sample_ids = np.asarray(data["sample_ids"]).astype(object)
            n_overlapping_reads = np.asarray(data["n_overlapping_reads"], dtype=np.int32)
            fragment_sample_offsets = np.asarray(data["fragment_sample_offsets"], dtype=np.int64)
            fragment_center_idx = np.asarray(data["fragment_center_idx"], dtype=np.int32)
            fragment_obs_offsets = np.asarray(data["fragment_obs_offsets"], dtype=np.int64)
            fragment_obs_pos_idx = np.asarray(data["fragment_obs_pos_idx"], dtype=np.int32)
            fragment_obs_code = np.asarray(data["fragment_obs_code"], dtype=np.int8)
            fragment_obs_qual = np.asarray(data["fragment_obs_qual"], dtype=np.uint8)
            has_dense_payload = all(
                key in data.files
                for key in (
                    "ref_count",
                    "alt_count",
                    "other_count",
                    "depth",
                    "ref_weight",
                    "alt_weight",
                    "other_weight",
                )
            )
            if materialize_dense_counts and has_dense_payload:
                ref_count = np.asarray(data["ref_count"], dtype=np.uint16)
                alt_count = np.asarray(data["alt_count"], dtype=np.uint16)
                other_count = np.asarray(data["other_count"], dtype=np.uint16)
                depth = np.asarray(data["depth"], dtype=np.uint16)
                ref_weight = np.asarray(data["ref_weight"], dtype=np.float32)
                alt_weight = np.asarray(data["alt_weight"], dtype=np.float32)
                other_weight = np.asarray(data["other_weight"], dtype=np.float32)
            else:
                ref_count = None
                alt_count = None
                other_count = None
                depth = None
                ref_weight = None
                alt_weight = None
                other_weight = None

        dense_loaded = ref_count is not None
        shape = (int(sample_ids.shape[0]), int(positions.shape[0]))
        if ref_count is None:
            ref_count = np.zeros(shape, dtype=np.uint16)
            alt_count = np.zeros(shape, dtype=np.uint16)
            other_count = np.zeros(shape, dtype=np.uint16)
            depth = np.zeros(shape, dtype=np.uint16)
            ref_weight = np.zeros(shape, dtype=np.float32)
            alt_weight = np.zeros(shape, dtype=np.float32)
            other_weight = np.zeros(shape, dtype=np.float32)
        if fragment_obs_pos_idx.size and not dense_loaded:
            materialized_native = False
            if _materialize_dense_from_fragments_htslib is not None:
                try:
                    (
                        ref_count,
                        alt_count,
                        other_count,
                        depth,
                        ref_weight,
                        alt_weight,
                        other_weight,
                    ) = _materialize_dense_from_fragments_htslib(
                        int(shape[0]),
                        int(shape[1]),
                        fragment_sample_offsets,
                        fragment_center_idx,
                        fragment_obs_offsets,
                        fragment_obs_pos_idx,
                        fragment_obs_code,
                        fragment_obs_qual,
                        support_only=not bool(materialize_dense_counts),
                    )
                    materialized_native = True
                except Exception:
                    materialized_native = False
            if not materialized_native:
                frag_counts = np.diff(fragment_sample_offsets).astype(np.int64, copy=False)
                sample_for_fragment = np.repeat(np.arange(shape[0], dtype=np.int64), frag_counts)
            if not materialized_native and sample_for_fragment.shape[0] == fragment_center_idx.shape[0]:
                for sample_idx, frag_start, frag_stop in zip(
                    sample_for_fragment.tolist(),
                    fragment_obs_offsets[:-1].tolist(),
                    fragment_obs_offsets[1:].tolist(),
                    strict=False,
                ):
                    pos = fragment_obs_pos_idx[int(frag_start) : int(frag_stop)]
                    if not pos.size:
                        continue
                    sidx = int(sample_idx)
                    if not materialize_dense_counts:
                        # Compact-only HMM runs still need a support mask for
                        # downstream gates; store binary depth without
                        # reconstructing allele counts.
                        depth[sidx, pos] = 1
                        continue
                    code = fragment_obs_code[int(frag_start) : int(frag_stop)]
                    qual = fragment_obs_qual[int(frag_start) : int(frag_stop)]
                    for p, c, q in zip(pos.tolist(), code.tolist(), qual.tolist(), strict=False):
                        pidx = int(p)
                        depth[sidx, pidx] = min(int(depth[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
                        w = np.float32(_base_weight(int(q)))
                        if int(c) == int(OBS_REF):
                            ref_count[sidx, pidx] = min(int(ref_count[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
                            ref_weight[sidx, pidx] += w
                        elif int(c) == int(OBS_ALT):
                            alt_count[sidx, pidx] = min(int(alt_count[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
                            alt_weight[sidx, pidx] += w
                        else:
                            other_count[sidx, pidx] = min(int(other_count[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
                            other_weight[sidx, pidx] += w

        return ReadEvidenceBlock(
            block_id=int(block_id),
            chromosome=chromosome,
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
            sample_ids=sample_ids,
            n_overlapping_reads=n_overlapping_reads,
            fragment_sample_offsets=fragment_sample_offsets,
            fragment_center_idx=fragment_center_idx,
            fragment_obs_offsets=fragment_obs_offsets,
            fragment_obs_pos_idx=fragment_obs_pos_idx,
            fragment_obs_code=fragment_obs_code,
            fragment_obs_qual=fragment_obs_qual,
            memmap_dir=None,
        )

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

    def select_samples(self, sample_indices: np.ndarray | list[int]) -> "ReadEvidenceBlock":
        idx = np.asarray(sample_indices, dtype=np.int64)
        n_selected = int(idx.shape[0])
        sample_offsets = np.zeros(n_selected + 1, dtype=np.int64)
        centers: list[np.ndarray] = []
        obs_offsets: list[int] = [0]
        obs_pos_chunks: list[np.ndarray] = []
        obs_code_chunks: list[np.ndarray] = []
        obs_qual_chunks: list[np.ndarray] = []
        for out_idx, sample_idx_raw in enumerate(idx.tolist()):
            sample_idx = int(sample_idx_raw)
            frag_start = int(self.fragment_sample_offsets[sample_idx])
            frag_stop = int(self.fragment_sample_offsets[sample_idx + 1])
            n_frag = frag_stop - frag_start
            sample_offsets[out_idx + 1] = sample_offsets[out_idx] + n_frag
            if n_frag <= 0:
                continue
            centers.append(self.fragment_center_idx[frag_start:frag_stop].astype(np.int32, copy=False))
            for frag_idx in range(frag_start, frag_stop):
                obs_start = int(self.fragment_obs_offsets[frag_idx])
                obs_stop = int(self.fragment_obs_offsets[frag_idx + 1])
                n_obs = obs_stop - obs_start
                if n_obs > 0:
                    obs_pos_chunks.append(self.fragment_obs_pos_idx[obs_start:obs_stop].astype(np.int32, copy=False))
                    obs_code_chunks.append(self.fragment_obs_code[obs_start:obs_stop].astype(np.int8, copy=False))
                    obs_qual_chunks.append(self.fragment_obs_qual[obs_start:obs_stop].astype(np.uint8, copy=False))
                obs_offsets.append(obs_offsets[-1] + n_obs)

        return ReadEvidenceBlock(
            block_id=self.block_id,
            chromosome=self.chromosome,
            positions=self.positions,
            ref=self.ref,
            alt=self.alt,
            ref_count=self.ref_count[idx],
            alt_count=self.alt_count[idx],
            other_count=self.other_count[idx],
            depth=self.depth[idx],
            ref_weight=self.ref_weight[idx],
            alt_weight=self.alt_weight[idx],
            other_weight=self.other_weight[idx],
            sample_ids=self.sample_ids[idx],
            n_overlapping_reads=self.n_overlapping_reads[idx],
            fragment_sample_offsets=sample_offsets,
            fragment_center_idx=np.concatenate(centers).astype(np.int32, copy=False) if centers else np.empty((0,), dtype=np.int32),
            fragment_obs_offsets=np.asarray(obs_offsets, dtype=np.int64),
            fragment_obs_pos_idx=np.concatenate(obs_pos_chunks).astype(np.int32, copy=False) if obs_pos_chunks else np.empty((0,), dtype=np.int32),
            fragment_obs_code=np.concatenate(obs_code_chunks).astype(np.int8, copy=False) if obs_code_chunks else np.empty((0,), dtype=np.int8),
            fragment_obs_qual=np.concatenate(obs_qual_chunks).astype(np.uint8, copy=False) if obs_qual_chunks else np.empty((0,), dtype=np.uint8),
            memmap_dir=None,
        )

    def _from_fragment_mask(self, fragment_mask: np.ndarray) -> "ReadEvidenceBlock":
        mask = np.asarray(fragment_mask, dtype=bool)
        if mask.shape != self.fragment_center_idx.shape:
            raise ValueError(
                f"fragment_mask shape {mask.shape} does not match n_fragments {self.fragment_center_idx.shape}."
            )
        n_samples = int(self.sample_ids.shape[0])
        n_positions = int(self.positions.shape[0])
        ref_count = np.zeros((n_samples, n_positions), dtype=np.uint16)
        alt_count = np.zeros((n_samples, n_positions), dtype=np.uint16)
        other_count = np.zeros((n_samples, n_positions), dtype=np.uint16)
        depth = np.zeros((n_samples, n_positions), dtype=np.uint16)
        ref_weight = np.zeros((n_samples, n_positions), dtype=np.float32)
        alt_weight = np.zeros((n_samples, n_positions), dtype=np.float32)
        other_weight = np.zeros((n_samples, n_positions), dtype=np.float32)
        sample_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        n_overlapping_reads = np.zeros(n_samples, dtype=np.int32)
        centers: list[np.ndarray] = []
        obs_offsets: list[int] = [0]
        obs_pos_chunks: list[np.ndarray] = []
        obs_code_chunks: list[np.ndarray] = []
        obs_qual_chunks: list[np.ndarray] = []

        for sample_idx in range(n_samples):
            frag_start = int(self.fragment_sample_offsets[sample_idx])
            frag_stop = int(self.fragment_sample_offsets[sample_idx + 1])
            selected = np.flatnonzero(mask[frag_start:frag_stop]).astype(np.int64, copy=False) + frag_start
            n_overlapping_reads[sample_idx] = int(selected.shape[0])
            sample_offsets[sample_idx + 1] = sample_offsets[sample_idx] + int(selected.shape[0])
            if selected.size == 0:
                continue
            centers.append(self.fragment_center_idx[selected].astype(np.int32, copy=False))
            for frag_idx in selected.tolist():
                obs_start = int(self.fragment_obs_offsets[int(frag_idx)])
                obs_stop = int(self.fragment_obs_offsets[int(frag_idx) + 1])
                pos = self.fragment_obs_pos_idx[obs_start:obs_stop].astype(np.int32, copy=False)
                code = self.fragment_obs_code[obs_start:obs_stop].astype(np.int8, copy=False)
                qual = self.fragment_obs_qual[obs_start:obs_stop].astype(np.uint8, copy=False)
                n_obs = int(pos.shape[0])
                if n_obs > 0:
                    obs_pos_chunks.append(pos)
                    obs_code_chunks.append(code)
                    obs_qual_chunks.append(qual)
                    for p, c, q in zip(pos.tolist(), code.tolist(), qual.tolist(), strict=False):
                        pidx = int(p)
                        if pidx < 0 or pidx >= n_positions:
                            continue
                        depth[sample_idx, pidx] = min(int(depth[sample_idx, pidx]) + 1, np.iinfo(np.uint16).max)
                        w = np.float32(_base_weight(int(q)))
                        if int(c) == int(OBS_REF):
                            ref_count[sample_idx, pidx] = min(int(ref_count[sample_idx, pidx]) + 1, np.iinfo(np.uint16).max)
                            ref_weight[sample_idx, pidx] += w
                        elif int(c) == int(OBS_ALT):
                            alt_count[sample_idx, pidx] = min(int(alt_count[sample_idx, pidx]) + 1, np.iinfo(np.uint16).max)
                            alt_weight[sample_idx, pidx] += w
                        else:
                            other_count[sample_idx, pidx] = min(int(other_count[sample_idx, pidx]) + 1, np.iinfo(np.uint16).max)
                            other_weight[sample_idx, pidx] += w
                obs_offsets.append(obs_offsets[-1] + n_obs)

        return ReadEvidenceBlock(
            block_id=self.block_id,
            chromosome=self.chromosome,
            positions=self.positions,
            ref=self.ref,
            alt=self.alt,
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            depth=depth,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
            sample_ids=self.sample_ids,
            n_overlapping_reads=n_overlapping_reads,
            fragment_sample_offsets=sample_offsets,
            fragment_center_idx=np.concatenate(centers).astype(np.int32, copy=False) if centers else np.empty((0,), dtype=np.int32),
            fragment_obs_offsets=np.asarray(obs_offsets, dtype=np.int64),
            fragment_obs_pos_idx=np.concatenate(obs_pos_chunks).astype(np.int32, copy=False) if obs_pos_chunks else np.empty((0,), dtype=np.int32),
            fragment_obs_code=np.concatenate(obs_code_chunks).astype(np.int8, copy=False) if obs_code_chunks else np.empty((0,), dtype=np.int8),
            fragment_obs_qual=np.concatenate(obs_qual_chunks).astype(np.uint8, copy=False) if obs_qual_chunks else np.empty((0,), dtype=np.uint8),
            memmap_dir=None,
        )

    def split_fragments_for_calibration(
        self,
        *,
        holdout_fraction: float,
        seed: int,
    ) -> tuple["ReadEvidenceBlock", "ReadEvidenceBlock", dict[str, Any]]:
        """Split compact fragment evidence into HMM and held-out calibration labels.

        The split happens at fragment level so read-evidence pseudo-truth is not
        learned from exactly the same observations consumed by the HMM.
        """
        n_fragments = int(self.fragment_center_idx.shape[0])
        frac = float(np.clip(float(holdout_fraction), 0.0, 0.95))
        if n_fragments < 2 or frac <= 0.0:
            empty_mask = np.zeros(n_fragments, dtype=bool)
            return self, self._from_fragment_mask(empty_mask), {
                "status": "skipped",
                "reason": "too_few_fragments_or_zero_fraction",
                "holdout_fraction_requested": float(holdout_fraction),
                "holdout_fraction_effective": 0.0,
                "n_total_fragments": int(n_fragments),
                "n_hmm_fragments": int(n_fragments),
                "n_calibration_fragments": 0,
            }
        rng = np.random.default_rng(int(seed))
        holdout_mask = rng.random(n_fragments) < frac
        if not np.any(holdout_mask):
            holdout_mask[int(rng.integers(0, n_fragments))] = True
        if np.all(holdout_mask):
            holdout_mask[int(rng.integers(0, n_fragments))] = False
        hmm_mask = ~holdout_mask
        hmm_evidence = self._from_fragment_mask(hmm_mask)
        heldout_evidence = self._from_fragment_mask(holdout_mask)
        return hmm_evidence, heldout_evidence, {
            "status": "ok",
            "holdout_fraction_requested": float(holdout_fraction),
            "holdout_fraction_effective": float(np.mean(holdout_mask)),
            "n_total_fragments": int(n_fragments),
            "n_hmm_fragments": int(np.sum(hmm_mask)),
            "n_calibration_fragments": int(np.sum(holdout_mask)),
            "hmm_compact_nbytes": int(hmm_evidence.compact_nbytes),
            "calibration_compact_nbytes": int(heldout_evidence.compact_nbytes),
        }

    def downsample_fragments(
        self,
        *,
        max_depth: int = 0,
        fraction: float = 1.0,
        seed: int = 0,
    ) -> tuple["ReadEvidenceBlock", dict[str, Any]]:
        """Fragment-level read downsampling/depth capping.

        The BAM reader still decodes all informative reads. This pass reduces
        the compact fragment set before HMM/calibration, which is the useful
        production cap for high-depth regions and keeps all dense matrices
        consistent with the retained fragments.
        """
        n_fragments = int(self.fragment_center_idx.shape[0])
        max_depth_i = max(int(max_depth), 0)
        fraction_f = float(np.clip(float(fraction), 0.0, 1.0))
        meta: dict[str, Any] = {
            "read_downsample_status": "skipped",
            "read_downsample_max_depth": int(max_depth_i),
            "read_downsample_fraction": float(fraction_f),
            "read_downsample_seed": int(seed),
            "read_downsample_fragments_before": int(n_fragments),
            "read_downsample_fragments_after": int(n_fragments),
        }
        if n_fragments <= 0 or (max_depth_i <= 0 and fraction_f >= 0.999999):
            return self, meta

        rng = np.random.default_rng(int(seed))
        keep = np.ones(n_fragments, dtype=bool)
        if fraction_f < 0.999999:
            keep &= rng.random(n_fragments) < fraction_f

        if max_depth_i > 0:
            n_positions = int(self.positions.shape[0])
            for sample_idx in range(int(self.sample_ids.shape[0])):
                frag_start = int(self.fragment_sample_offsets[sample_idx])
                frag_stop = int(self.fragment_sample_offsets[sample_idx + 1])
                if frag_stop <= frag_start:
                    continue
                sample_fragments = np.arange(frag_start, frag_stop, dtype=np.int64)
                rng.shuffle(sample_fragments)
                sample_keep = np.zeros(frag_stop - frag_start, dtype=bool)
                coverage = np.zeros(n_positions, dtype=np.uint16)
                for frag_idx_raw in sample_fragments.tolist():
                    frag_idx = int(frag_idx_raw)
                    if not keep[frag_idx]:
                        continue
                    obs_start = int(self.fragment_obs_offsets[frag_idx])
                    obs_stop = int(self.fragment_obs_offsets[frag_idx + 1])
                    pos = self.fragment_obs_pos_idx[obs_start:obs_stop]
                    if pos.size == 0:
                        continue
                    unique_pos = np.unique(pos[(pos >= 0) & (pos < n_positions)])
                    if unique_pos.size == 0:
                        continue
                    if np.all(coverage[unique_pos] >= max_depth_i):
                        continue
                    sample_keep[frag_idx - frag_start] = True
                    coverage[unique_pos] = np.minimum(coverage[unique_pos] + 1, max_depth_i).astype(
                        np.uint16,
                        copy=False,
                    )
                keep[frag_start:frag_stop] &= sample_keep

        kept = int(np.count_nonzero(keep))
        meta.update(
            {
                "read_downsample_status": "ok",
                "read_downsample_fragments_after": int(kept),
                "read_downsample_fragment_retention": float(kept / max(n_fragments, 1)),
            }
        )
        if kept == n_fragments:
            return self, meta
        downsampled = self._from_fragment_mask(keep)
        meta["read_downsample_compact_nbytes"] = int(downsampled.compact_nbytes)
        meta["read_downsample_dense_nbytes"] = int(downsampled.dense_nbytes)
        return downsampled, meta

    @staticmethod
    def merge_sample_blocks(
        blocks: list["ReadEvidenceBlock"],
        sample_ids: np.ndarray,
        *,
        block_id: int,
    ) -> "ReadEvidenceBlock":
        valid_blocks = [block for block in blocks if int(block.sample_ids.shape[0]) > 0]
        if not valid_blocks:
            raise ValueError("Cannot merge zero read-evidence sample blocks.")
        first = valid_blocks[0]
        sample_ids_obj = np.asarray(sample_ids, dtype=object)
        source: dict[str, tuple[ReadEvidenceBlock, int]] = {}
        for block in valid_blocks:
            if not np.array_equal(block.positions, first.positions):
                raise ValueError("Cannot merge evidence blocks with different positions.")
            for idx, sample_id in enumerate(block.sample_ids.astype(str).tolist()):
                source[str(sample_id)] = (block, int(idx))

        rows_ref: list[np.ndarray] = []
        rows_alt: list[np.ndarray] = []
        rows_other: list[np.ndarray] = []
        rows_depth: list[np.ndarray] = []
        rows_ref_w: list[np.ndarray] = []
        rows_alt_w: list[np.ndarray] = []
        rows_other_w: list[np.ndarray] = []
        rows_n_reads: list[int] = []
        sample_offsets = np.zeros(int(sample_ids_obj.shape[0]) + 1, dtype=np.int64)
        centers: list[np.ndarray] = []
        obs_offsets: list[int] = [0]
        obs_pos_chunks: list[np.ndarray] = []
        obs_code_chunks: list[np.ndarray] = []
        obs_qual_chunks: list[np.ndarray] = []

        for out_idx, sample_id in enumerate(sample_ids_obj.astype(str).tolist()):
            block, idx = source[str(sample_id)]
            rows_ref.append(block.ref_count[idx])
            rows_alt.append(block.alt_count[idx])
            rows_other.append(block.other_count[idx])
            rows_depth.append(block.depth[idx])
            rows_ref_w.append(block.ref_weight[idx])
            rows_alt_w.append(block.alt_weight[idx])
            rows_other_w.append(block.other_weight[idx])
            rows_n_reads.append(int(block.n_overlapping_reads[idx]))
            frag_start = int(block.fragment_sample_offsets[idx])
            frag_stop = int(block.fragment_sample_offsets[idx + 1])
            n_frag = frag_stop - frag_start
            sample_offsets[out_idx + 1] = sample_offsets[out_idx] + n_frag
            if n_frag <= 0:
                continue
            centers.append(block.fragment_center_idx[frag_start:frag_stop].astype(np.int32, copy=False))
            for frag_idx in range(frag_start, frag_stop):
                obs_start = int(block.fragment_obs_offsets[frag_idx])
                obs_stop = int(block.fragment_obs_offsets[frag_idx + 1])
                n_obs = obs_stop - obs_start
                if n_obs > 0:
                    obs_pos_chunks.append(block.fragment_obs_pos_idx[obs_start:obs_stop].astype(np.int32, copy=False))
                    obs_code_chunks.append(block.fragment_obs_code[obs_start:obs_stop].astype(np.int8, copy=False))
                    obs_qual_chunks.append(block.fragment_obs_qual[obs_start:obs_stop].astype(np.uint8, copy=False))
                obs_offsets.append(obs_offsets[-1] + n_obs)

        return ReadEvidenceBlock(
            block_id=int(block_id),
            chromosome=first.chromosome,
            positions=first.positions,
            ref=first.ref,
            alt=first.alt,
            ref_count=np.stack(rows_ref, axis=0).astype(np.uint16, copy=False),
            alt_count=np.stack(rows_alt, axis=0).astype(np.uint16, copy=False),
            other_count=np.stack(rows_other, axis=0).astype(np.uint16, copy=False),
            depth=np.stack(rows_depth, axis=0).astype(np.uint16, copy=False),
            ref_weight=np.stack(rows_ref_w, axis=0).astype(np.float32, copy=False),
            alt_weight=np.stack(rows_alt_w, axis=0).astype(np.float32, copy=False),
            other_weight=np.stack(rows_other_w, axis=0).astype(np.float32, copy=False),
            sample_ids=sample_ids_obj,
            n_overlapping_reads=np.asarray(rows_n_reads, dtype=np.int32),
            fragment_sample_offsets=sample_offsets,
            fragment_center_idx=np.concatenate(centers).astype(np.int32, copy=False) if centers else np.empty((0,), dtype=np.int32),
            fragment_obs_offsets=np.asarray(obs_offsets, dtype=np.int64),
            fragment_obs_pos_idx=np.concatenate(obs_pos_chunks).astype(np.int32, copy=False) if obs_pos_chunks else np.empty((0,), dtype=np.int32),
            fragment_obs_code=np.concatenate(obs_code_chunks).astype(np.int8, copy=False) if obs_code_chunks else np.empty((0,), dtype=np.int8),
            fragment_obs_qual=np.concatenate(obs_qual_chunks).astype(np.uint8, copy=False) if obs_qual_chunks else np.empty((0,), dtype=np.uint8),
            memmap_dir=None,
        )

    def slice_by_position_rows(self, row_start: int, row_stop: int, *, block_id: int) -> "ReadEvidenceBlock":
        start = max(int(row_start), 0)
        stop = min(max(int(row_stop), start), int(self.positions.shape[0]))
        positions = self.positions[start:stop]
        ref = self.ref[start:stop]
        alt = self.alt[start:stop]
        ref_count = self.ref_count[:, start:stop]
        alt_count = self.alt_count[:, start:stop]
        other_count = self.other_count[:, start:stop]
        depth = self.depth[:, start:stop]
        ref_weight = self.ref_weight[:, start:stop]
        alt_weight = self.alt_weight[:, start:stop]
        other_weight = self.other_weight[:, start:stop]
        n_samples = int(self.sample_ids.shape[0])
        sample_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        centers: list[int] = []
        obs_offsets: list[int] = [0]
        obs_pos_chunks: list[np.ndarray] = []
        obs_code_chunks: list[np.ndarray] = []
        obs_qual_chunks: list[np.ndarray] = []
        physical_positions = positions.astype(np.int64, copy=False)

        for sample_idx in range(n_samples):
            frag_start = int(self.fragment_sample_offsets[sample_idx])
            frag_stop = int(self.fragment_sample_offsets[sample_idx + 1])
            for frag_idx in range(frag_start, frag_stop):
                obs_start = int(self.fragment_obs_offsets[frag_idx])
                obs_stop = int(self.fragment_obs_offsets[frag_idx + 1])
                pos_abs = self.fragment_obs_pos_idx[obs_start:obs_stop]
                keep = (pos_abs >= start) & (pos_abs < stop)
                if not np.any(keep):
                    continue
                pos_local = (pos_abs[keep] - start).astype(np.int32, copy=False)
                code_local = self.fragment_obs_code[obs_start:obs_stop][keep].astype(np.int8, copy=False)
                qual_local = self.fragment_obs_qual[obs_start:obs_stop][keep].astype(np.uint8, copy=False)
                centers.append(_stitch_center_idx(pos_local, physical_positions))
                obs_pos_chunks.append(pos_local)
                obs_code_chunks.append(code_local)
                obs_qual_chunks.append(qual_local)
                obs_offsets.append(obs_offsets[-1] + int(pos_local.shape[0]))
            sample_offsets[sample_idx + 1] = len(centers)

        return ReadEvidenceBlock(
            block_id=block_id,
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
            sample_ids=self.sample_ids,
            n_overlapping_reads=np.sum(depth > 0, axis=1).astype(np.int32, copy=False),
            fragment_sample_offsets=sample_offsets,
            fragment_center_idx=np.asarray(centers, dtype=np.int32),
            fragment_obs_offsets=np.asarray(obs_offsets, dtype=np.int64),
            fragment_obs_pos_idx=(
                np.concatenate(obs_pos_chunks).astype(np.int32, copy=False)
                if obs_pos_chunks
                else np.empty((0,), dtype=np.int32)
            ),
            fragment_obs_code=(
                np.concatenate(obs_code_chunks).astype(np.int8, copy=False)
                if obs_code_chunks
                else np.empty((0,), dtype=np.int8)
            ),
            fragment_obs_qual=(
                np.concatenate(obs_qual_chunks).astype(np.uint8, copy=False)
                if obs_qual_chunks
                else np.empty((0,), dtype=np.uint8)
            ),
            memmap_dir=None,
        )

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


def _stitch_center_idx(pos_idx: np.ndarray, physical_positions: np.ndarray) -> int:
    if pos_idx.size == 0:
        return -1
    pos_i = pos_idx.astype(np.int32, copy=False)
    if pos_i.size == 1:
        return int(pos_i[0])
    physical = physical_positions[pos_i].astype(np.float64, copy=False)
    mean_physical = float(np.mean(physical))
    return int(pos_i[int(np.argmin(np.abs(physical - mean_physical)))])


def _first_allele_codes(alleles: np.ndarray) -> np.ndarray:
    out = np.empty(alleles.shape[0], dtype=np.uint8)
    for idx, allele in enumerate(alleles.tolist()):
        text = str(allele).upper()
        out[idx] = ord(text[0] if text else "N")
    return out


def _variant_type_codes(frame: pd.DataFrame, ref: np.ndarray, alt: np.ndarray, *, max_indel_len: int) -> np.ndarray:
    if "VARIANT_TYPE" in frame.columns:
        raw_types = frame["VARIANT_TYPE"].fillna("").astype(str).str.lower().tolist()
    else:
        raw_types = ["" for _ in range(len(ref))]
    codes = np.empty(len(ref), dtype=np.uint8)
    for idx, (raw_type, ref_allele, alt_allele) in enumerate(zip(raw_types, ref, alt, strict=False)):
        inferred = infer_variant_type(str(ref_allele), str(alt_allele))
        variant_type = inferred if raw_type == "" else raw_type
        code = _VARIANT_TYPE_TO_CODE.get(variant_type)
        if code is None:
            raise ValueError(f"Unsupported VARIANT_TYPE {variant_type!r} at position index {idx}.")
        inferred_code = _VARIANT_TYPE_TO_CODE[inferred]
        if int(code) != int(inferred_code):
            raise ValueError(
                f"VARIANT_TYPE {variant_type!r} disagrees with REF/ALT-inferred type {inferred!r} "
                f"at position index {idx}."
            )
        if code != VARIANT_SNP:
            indel_len = abs(len(str(ref_allele)) - len(str(alt_allele)))
            if indel_len <= 0 or indel_len > int(max_indel_len):
                raise ValueError(
                    f"Indel length {indel_len} at position index {idx} exceeds --max-indel-len={int(max_indel_len)}."
                )
        codes[idx] = np.uint8(code)
    return codes


class PysamReadExtractor:
    def __init__(
        self,
        chromosome: str,
        *,
        mode: Literal["read_stream", "pileup"] = "read_stream",
        read_stream_backend: Literal[
            "auto",
            "python",
            "htslib",
            "snp_only_bamreader",
            "stitch_style_bamreader",
            "variant_aware_bamreader",
        ] = "auto",
        min_base_quality: int = 13,
        min_mapping_quality: int = 20,
        max_insert_size: int = 0,
        max_indel_len: int = 50,
        cap_base_quality_by_mapping_quality: bool = False,
        ref_alt_only: bool = False,
        merge_fragments_by_query: bool = True,
        merge_unpaired_fragments_by_query: bool = False,
        use_bx_tag: bool = False,
        bx_tag: str = "BX",
        bx_tag_upper_limit: int = 50000,
        read_batch_size: int = 1024,
        io_workers: int = 1,
        htslib_threads_per_file: int = 1,
        io_window_size: int = 0,
        memory_map_read_matrices: bool = False,
        memory_map_dir: str | Path | None = None,
        subsample_seed: int = 0,
    ):
        self.chromosome = chromosome
        self.mode = mode
        self.read_stream_backend = read_stream_backend
        self.min_base_quality = min_base_quality
        self.min_mapping_quality = min_mapping_quality
        self.max_insert_size = max(int(max_insert_size), 0)
        self.max_indel_len = max(int(max_indel_len), 0)
        self.cap_base_quality_by_mapping_quality = bool(cap_base_quality_by_mapping_quality)
        self.ref_alt_only = bool(ref_alt_only)
        self.merge_fragments_by_query = merge_fragments_by_query
        self.merge_unpaired_fragments_by_query = bool(merge_unpaired_fragments_by_query)
        self.use_bx_tag = bool(use_bx_tag)
        self.bx_tag = str(bx_tag or "BX")[:2]
        self.bx_tag_upper_limit = max(int(bx_tag_upper_limit), 1)
        self.read_batch_size = max(int(read_batch_size), 64)
        self.io_workers = max(int(io_workers), 1)
        self.htslib_threads_per_file = max(int(htslib_threads_per_file), 1)
        self.io_window_size = max(int(io_window_size), 0)
        self.memory_map_read_matrices = bool(memory_map_read_matrices)
        self.memory_map_dir = None if memory_map_dir is None else Path(memory_map_dir)
        self.subsample_seed = int(subsample_seed)
        self._qual_weight_lut = np.array([_base_weight(q) for q in range(256)], dtype=np.float32)
        self._handles: list[pysam.AlignmentFile] | None = None
        self._bam_paths: list[str] | None = None
        self._sample_ids: np.ndarray | None = None
        self._read_keep_prob: np.ndarray | None = None
        self._subsample_rngs: list[np.random.Generator] | None = None
        self._position_table: pd.DataFrame | None = None
        self._cached_window_key: tuple[tuple[str, ...], int, int] | None = None
        self._cached_window_evidence: ReadEvidenceBlock | None = None

        if (
            self.read_stream_backend in {"htslib", "snp_only_bamreader", "stitch_style_bamreader", "variant_aware_bamreader"}
            and _extract_sample_read_stream_htslib is None
        ):
            raise RuntimeError(
                f"read_stream_backend={self.read_stream_backend!r} requested but compiled extension is unavailable. "
                "Build/install STITCHCONT with the HTSlib extension or use read_stream_backend='python'/'auto'."
            )
        self._snp_only_bamreader = self.read_stream_backend == "snp_only_bamreader"
        self._stitch_style_bamreader = self.read_stream_backend == "stitch_style_bamreader"
        self._variant_aware_bamreader = self.read_stream_backend == "variant_aware_bamreader" or (
            self.read_stream_backend == "auto" and _extract_sample_read_stream_htslib is not None
        )
        self._use_compiled_read_stream = (
            self.mode == "read_stream"
            and self.read_stream_backend != "python"
            and _extract_sample_read_stream_htslib is not None
        )
        self._use_compiled_read_stream_runtime = self._use_compiled_read_stream

    def _read_merge_key(self, read: pysam.AlignedSegment) -> tuple[str | None, bool]:
        if not self.merge_fragments_by_query:
            return None, False
        if self.use_bx_tag and self.bx_tag:
            try:
                bx_value = read.get_tag(self.bx_tag)
            except KeyError:
                bx_value = None
            if bx_value is not None:
                bx_text = str(bx_value)
                if bx_text:
                    return f"BX:{bx_text}", True
        if read.is_paired:
            return f"Q:{read.query_name}", False
        if self.merge_unpaired_fragments_by_query:
            return f"Q:{read.query_name}", True
        return None, False

    def set_position_table(self, positions_df: pd.DataFrame | None) -> None:
        self._position_table = None if positions_df is None else positions_df.reset_index(drop=True)
        self._cached_window_key = None
        if self._cached_window_evidence is not None:
            self._cached_window_evidence.release()
        self._cached_window_evidence = None

    def set_io_window_size(self, io_window_size: int) -> None:
        self.io_window_size = max(int(io_window_size), 0)
        self._cached_window_key = None
        if self._cached_window_evidence is not None:
            self._cached_window_evidence.release()
        self._cached_window_evidence = None

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
        self._cached_window_key = None
        if self._cached_window_evidence is not None:
            self._cached_window_evidence.release()
        self._cached_window_evidence = None

    def _create_block_memmap_dir(self, block_id: int) -> str | None:
        if not self.memory_map_read_matrices:
            return None
        root = self.memory_map_dir
        if root is None:
            root = Path(tempfile.gettempdir()) / "stitchcont_memmap"
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
        if (
            self.mode == "read_stream"
            and self._position_table is not None
            and int(self.io_window_size) > max(int(block.row_stop - block.row_start), 0)
            and int(block.row_stop) > int(block.row_start)
        ):
            n_positions_total = int(self._position_table.shape[0])
            window_size = min(max(int(self.io_window_size), 1), max(n_positions_total, 1))
            window_start = (int(block.row_start) // window_size) * window_size
            window_stop = min(window_start + window_size, n_positions_total)
            sample_key = tuple(samples["sample_id"].astype(str).tolist())
            cache_key = (sample_key, window_start, window_stop)
            if self._cached_window_key != cache_key or self._cached_window_evidence is None:
                if self._cached_window_evidence is not None:
                    self._cached_window_evidence.release()
                window_block = PositionBlock(
                    block_id=window_start // window_size,
                    dataframe=self._position_table.iloc[window_start:window_stop].reset_index(drop=True),
                    row_start=window_start,
                    row_stop=window_stop,
                )
                self._cached_window_evidence = self._extract_block_uncached(samples, window_block)
                self._cached_window_key = cache_key
            return self._cached_window_evidence.slice_by_position_rows(
                int(block.row_start) - window_start,
                int(block.row_stop) - window_start,
                block_id=int(block.block_id),
            )
        return self._extract_block_uncached(samples, block)

    def _extract_block_uncached(self, samples: pd.DataFrame, block: PositionBlock) -> ReadEvidenceBlock:
        if ((not self._use_compiled_read_stream_runtime) and self._handles is None) or self._sample_ids is None:
            self.open(samples)
        assert self._sample_ids is not None

        positions = block.dataframe["POS"].to_numpy(dtype=np.int64)
        ref = block.dataframe["REF"].astype(str).str.upper().to_numpy()
        alt = block.dataframe["ALT"].astype(str).str.upper().to_numpy()
        variant_type_codes = _variant_type_codes(block.dataframe, ref, alt, max_indel_len=self.max_indel_len)
        n_samples = len(samples)
        n_positions = len(positions)

        if (
            n_positions > 0
            and self.mode == "read_stream"
            and self._use_compiled_read_stream_runtime
            and _extract_samples_read_stream_batch_htslib is not None
            and not self.memory_map_read_matrices
        ):
            return self._extract_block_compiled_direct(
                block=block,
                positions=positions,
                ref=ref,
                alt=alt,
                variant_type_codes=variant_type_codes,
            )

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
                if np.any(variant_type_codes != int(VARIANT_SNP)):
                    raise RuntimeError("read_mode='pileup' is SNP-only; use read_mode='read_stream' with variant_aware_bamreader for indels.")
                self._extract_with_pileup(
                    positions=positions,
                    ref=ref,
                    alt=alt,
                    variant_type_codes=variant_type_codes,
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
                    variant_type_codes=variant_type_codes,
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

    def _extract_block_compiled_direct(
        self,
        *,
        block: PositionBlock,
        positions: np.ndarray,
        ref: np.ndarray,
        alt: np.ndarray,
        variant_type_codes: np.ndarray,
    ) -> ReadEvidenceBlock:
        if np.any(variant_type_codes != int(VARIANT_SNP)) and not self._variant_aware_bamreader:
            raise RuntimeError(
                "Targeted insertion/deletion evidence requires read_stream_backend='variant_aware_bamreader' "
                "or read_stream_backend='auto' with the compiled extension available."
            )
        assert self._sample_ids is not None
        region_start = int(positions[0]) - 1
        region_stop = int(positions[-1])
        lookup = np.full(region_stop - region_start + 1, -1, dtype=np.int32)
        lookup[positions - (region_start + 1)] = np.arange(positions.shape[0], dtype=np.int32)
        (
            depth,
            ref_count,
            alt_count,
            other_count,
            ref_weight,
            alt_weight,
            other_weight,
            n_overlapping_reads,
            fragment_sample_offsets,
            fragment_center_idx,
            fragment_obs_offsets,
            fragment_obs_pos_idx,
            fragment_obs_code,
            fragment_obs_qual,
        ) = self._extract_read_stream_compiled_batch(
            positions=positions,
            ref=ref,
            alt=alt,
            variant_type_codes=variant_type_codes,
            region_start=region_start,
            region_stop=region_stop,
            lookup=lookup,
            ref_codes=_first_allele_codes(ref),
            alt_codes=_first_allele_codes(alt),
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
            memmap_dir=None,
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
        variant_type_codes: np.ndarray,
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
        ref_codes = _first_allele_codes(ref)
        alt_codes = _first_allele_codes(alt)
        if np.any(variant_type_codes != int(VARIANT_SNP)) and not self._variant_aware_bamreader:
            raise RuntimeError(
                "Targeted insertion/deletion evidence requires read_stream_backend='variant_aware_bamreader' "
                "or read_stream_backend='auto' with the compiled extension available."
            )

        if self._use_compiled_read_stream_runtime:
            assert self._bam_paths is not None
            n_samples = len(self._bam_paths)
            if _extract_samples_read_stream_batch_htslib is not None:
                (
                    depth_local,
                    ref_local,
                    alt_local,
                    other_local,
                    ref_w_local,
                    alt_w_local,
                    other_w_local,
                    n_reads_local,
                    sample_offsets,
                    fragment_center_idx,
                    fragment_obs_offsets,
                    fragment_obs_pos_idx,
                    fragment_obs_code,
                    fragment_obs_qual,
                ) = self._extract_read_stream_compiled_batch(
                    positions=positions,
                    ref=ref,
                    alt=alt,
                    variant_type_codes=variant_type_codes,
                    region_start=region_start,
                    region_stop=region_stop,
                    lookup=lookup,
                    ref_codes=ref_codes,
                    alt_codes=alt_codes,
                )
                depth[...] = depth_local.astype(depth.dtype, copy=False)
                ref_count[...] = ref_local.astype(ref_count.dtype, copy=False)
                alt_count[...] = alt_local.astype(alt_count.dtype, copy=False)
                other_count[...] = other_local.astype(other_count.dtype, copy=False)
                ref_weight[...] = ref_w_local.astype(np.float32, copy=False)
                alt_weight[...] = alt_w_local.astype(np.float32, copy=False)
                other_weight[...] = other_w_local.astype(np.float32, copy=False)
                n_overlapping_reads[...] = n_reads_local.astype(np.int32, copy=False)
                return (
                    sample_offsets,
                    fragment_center_idx,
                    fragment_obs_offsets,
                    fragment_obs_pos_idx,
                    fragment_obs_code,
                    fragment_obs_qual,
                )
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
                    positions=positions,
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
                positions=positions,
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

    def _extract_read_stream_compiled_batch(
        self,
        *,
        positions: np.ndarray,
        ref: np.ndarray,
        alt: np.ndarray,
        variant_type_codes: np.ndarray,
        region_start: int,
        region_stop: int,
        lookup: np.ndarray,
        ref_codes: np.ndarray,
        alt_codes: np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if _extract_samples_read_stream_batch_htslib is None:
            raise RuntimeError("Compiled HTSlib batch read extractor is not available.")
        assert self._bam_paths is not None
        has_indels = bool(np.any(variant_type_codes != int(VARIANT_SNP)))
        effective_variant_aware = bool(self._variant_aware_bamreader and has_indels)
        effective_stitch_style = bool(self._stitch_style_bamreader or (self._variant_aware_bamreader and not has_indels))
        ref_alleles_arg = ref.astype(str).tolist() if effective_variant_aware else None
        alt_alleles_arg = alt.astype(str).tolist() if effective_variant_aware else None
        variant_types_arg = variant_type_codes.astype(np.uint8, copy=False) if effective_variant_aware else None
        result = _extract_samples_read_stream_batch_htslib(
            self._bam_paths,
            self.chromosome,
            int(region_start),
            int(region_stop),
            lookup.astype(np.int32, copy=False),
            ref_codes.astype(np.uint8, copy=False),
            alt_codes.astype(np.uint8, copy=False),
            positions.astype(np.int64, copy=False),
            int(self.min_base_quality),
            int(self.min_mapping_quality),
            bool(self.merge_fragments_by_query),
            int(self.htslib_threads_per_file),
            bool(self._snp_only_bamreader),
            effective_stitch_style,
            int(self.max_insert_size),
            bool(self.cap_base_quality_by_mapping_quality),
            bool(self.ref_alt_only),
            bool(self.merge_unpaired_fragments_by_query),
            bool(self.use_bx_tag),
            str(self.bx_tag),
            int(self.bx_tag_upper_limit),
            ref_alleles_arg,
            alt_alleles_arg,
            variant_types_arg,
            effective_variant_aware,
            int(self.max_indel_len),
            int(self.io_workers),
        )
        return (
            result[0].astype(np.uint16, copy=False),
            result[1].astype(np.uint16, copy=False),
            result[2].astype(np.uint16, copy=False),
            result[3].astype(np.uint16, copy=False),
            result[4].astype(np.float32, copy=False),
            result[5].astype(np.float32, copy=False),
            result[6].astype(np.float32, copy=False),
            result[7].astype(np.int32, copy=False),
            result[8].astype(np.int64, copy=False),
            result[9].astype(np.int32, copy=False),
            result[10].astype(np.int64, copy=False),
            result[11].astype(np.int32, copy=False),
            result[12].astype(np.int8, copy=False),
            result[13].astype(np.uint8, copy=False),
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
        positions: np.ndarray,
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
            sample_centers.append(_stitch_center_idx(pos_compact, positions))
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
            if self.max_insert_size > 0 and abs(int(read.template_length)) > self.max_insert_size:
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
            if self.cap_base_quality_by_mapping_quality:
                qual = np.minimum(qual, np.uint8(max(int(read.mapping_quality), 0))).astype(np.uint8, copy=False)
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
            if self.ref_alt_only:
                keep_ref_alt = ref_mask | alt_mask
                if not np.any(keep_ref_alt):
                    continue
                target_idx = target_idx[keep_ref_alt]
                qual = qual[keep_ref_alt]
                obs_code = obs_code[keep_ref_alt]

            n_reads += 1
            sample_targets.append(target_idx)
            sample_codes_raw.append(obs_code)
            sample_quals_raw.append(qual)
            if len(sample_targets) >= self.read_batch_size:
                flush_sample_observations()

            key, accumulate_to_end = self._read_merge_key(read)
            if key is not None and accumulate_to_end:
                existing = pending.get(key)
                if existing is None:
                    pending[key] = (target_idx, obs_code, qual)
                else:
                    merged_pos = np.concatenate((existing[0], target_idx), axis=0)
                    merged_code = np.concatenate((existing[1], obs_code), axis=0)
                    merged_qual = np.concatenate((existing[2], qual), axis=0)
                    if merged_pos.shape[0] > self.bx_tag_upper_limit:
                        append_fragment(existing[0], existing[1], existing[2])
                        pending[key] = (target_idx, obs_code, qual)
                    else:
                        pending[key] = (merged_pos, merged_code, merged_qual)
            elif key is not None:
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
        positions: np.ndarray,
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
        try:
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
                positions.astype(np.int64, copy=False),
                int(self.min_base_quality),
                int(self.min_mapping_quality),
                bool(self.merge_fragments_by_query),
                int(self.htslib_threads_per_file),
                bool(self._snp_only_bamreader),
                bool(self._stitch_style_bamreader),
                int(self.max_insert_size),
                bool(self.cap_base_quality_by_mapping_quality),
                bool(self.ref_alt_only),
                bool(self.merge_unpaired_fragments_by_query),
                bool(self.use_bx_tag),
                str(self.bx_tag),
                int(self.bx_tag_upper_limit),
            )
        except TypeError:
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
            if sample_obs_offsets.shape[0] > 1:
                sample_centers = np.asarray(
                    [
                        _stitch_center_idx(
                            sample_obs_pos[int(sample_obs_offsets[i]) : int(sample_obs_offsets[i + 1])],
                            positions,
                        )
                        for i in range(int(sample_obs_offsets.shape[0]) - 1)
                    ],
                    dtype=np.int32,
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
