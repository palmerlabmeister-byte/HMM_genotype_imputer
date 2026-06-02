from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .io import PositionBlock
from .pileup import OBS_ALT, OBS_REF, ReadEvidenceBlock, _base_weight, _materialize_dense_from_fragments_htslib

try:  # pragma: no cover - unavailable on non-POSIX platforms.
    import fcntl
except Exception:  # pragma: no cover
    fcntl = None


def _positions_hash(positions: np.ndarray, ref: np.ndarray, alt: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(positions, dtype=np.int64).tobytes())
    h.update("\0".join(np.asarray(ref, dtype=str).tolist()).encode("utf-8"))
    h.update(b"\0")
    h.update("\0".join(np.asarray(alt, dtype=str).tolist()).encode("utf-8"))
    return h.hexdigest()


def _metadata_hash(metadata: dict[str, Any]) -> str:
    payload = json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _part_id() -> str:
    return f"part-{time.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:12]}"


def _bitpack_support(depth: np.ndarray) -> list[bytes]:
    support = np.asarray(depth > 0, dtype=np.uint8)
    return [np.packbits(row, bitorder="little").tobytes() for row in support]


def _unpack_support(rows: list[bytes], n_positions: int) -> np.ndarray:
    out = np.zeros((len(rows), int(n_positions)), dtype=np.uint16)
    for idx, packed in enumerate(rows):
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
        out[idx] = bits[:n_positions].astype(np.uint16, copy=False)
    return out


def _dense_zero(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    return (
        np.zeros(shape, dtype=np.uint16),
        np.zeros(shape, dtype=np.uint16),
        np.zeros(shape, dtype=np.uint16),
        np.zeros(shape, dtype=np.uint16),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=np.float32),
    )


def _materialize_dense_from_fragments(evidence: ReadEvidenceBlock, *, support_only: bool) -> None:
    if evidence.fragment_obs_pos_idx.size == 0:
        return
    if _materialize_dense_from_fragments_htslib is not None:
        try:
            (
                evidence.ref_count,
                evidence.alt_count,
                evidence.other_count,
                evidence.depth,
                evidence.ref_weight,
                evidence.alt_weight,
                evidence.other_weight,
            ) = _materialize_dense_from_fragments_htslib(
                int(evidence.sample_ids.shape[0]),
                int(evidence.positions.shape[0]),
                evidence.fragment_sample_offsets,
                evidence.fragment_center_idx,
                evidence.fragment_obs_offsets,
                evidence.fragment_obs_pos_idx,
                evidence.fragment_obs_code,
                evidence.fragment_obs_qual,
                support_only=bool(support_only),
            )
            return
        except Exception:
            pass
    frag_counts = np.diff(evidence.fragment_sample_offsets).astype(np.int64, copy=False)
    sample_for_fragment = np.repeat(np.arange(evidence.sample_ids.shape[0], dtype=np.int64), frag_counts)
    if sample_for_fragment.shape[0] != evidence.fragment_center_idx.shape[0]:
        return
    for sample_idx, obs_start, obs_stop in zip(
        sample_for_fragment.tolist(),
        evidence.fragment_obs_offsets[:-1].tolist(),
        evidence.fragment_obs_offsets[1:].tolist(),
        strict=False,
    ):
        pos = evidence.fragment_obs_pos_idx[int(obs_start) : int(obs_stop)]
        if pos.size == 0:
            continue
        sidx = int(sample_idx)
        if support_only:
            evidence.depth[sidx, pos] = 1
            continue
        code = evidence.fragment_obs_code[int(obs_start) : int(obs_stop)]
        qual = evidence.fragment_obs_qual[int(obs_start) : int(obs_stop)]
        for p, c, q in zip(pos.tolist(), code.tolist(), qual.tolist(), strict=False):
            pidx = int(p)
            evidence.depth[sidx, pidx] = min(int(evidence.depth[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
            weight = np.float32(_base_weight(int(q)))
            if int(c) == int(OBS_REF):
                evidence.ref_count[sidx, pidx] = min(int(evidence.ref_count[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
                evidence.ref_weight[sidx, pidx] += weight
            elif int(c) == int(OBS_ALT):
                evidence.alt_count[sidx, pidx] = min(int(evidence.alt_count[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
                evidence.alt_weight[sidx, pidx] += weight
            else:
                evidence.other_count[sidx, pidx] = min(int(evidence.other_count[sidx, pidx]) + 1, np.iinfo(np.uint16).max)
                evidence.other_weight[sidx, pidx] += weight


class PartitionedEvidenceCache:
    """Partitioned evidence cache with Parquet compact fragments and optional Zarr dense arrays."""

    version = 1

    def __init__(
        self,
        root: str | Path,
        *,
        chromosome: str,
        compression: str = "zstd",
        sample_batch_size: int = 256,
        include_dense_counts: bool = False,
        materialize_dense_counts: bool = True,
        evidence_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.chromosome = str(chromosome)
        self.compression = compression
        self.sample_batch_size = max(int(sample_batch_size), 1)
        self.include_dense_counts = bool(include_dense_counts)
        self.materialize_dense_counts = bool(materialize_dense_counts)
        self.evidence_metadata = dict(evidence_metadata or {})
        self.evidence_metadata_hash = _metadata_hash(self.evidence_metadata)

    def block_dir(self, block: PositionBlock) -> Path:
        return self.root / f"chrom={self.chromosome}" / f"block={int(block.block_id):06d}_rows={int(block.row_start)}-{int(block.row_stop)}"

    def manifest_path(self, block: PositionBlock) -> Path:
        return self.block_dir(block) / "manifest.json"

    def _manifest(self, block: PositionBlock) -> dict[str, Any] | None:
        path = self.manifest_path(block)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_manifest(self, block: PositionBlock, manifest: dict[str, Any]) -> None:
        path = self.manifest_path(block)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    @contextmanager
    def _write_lock(self, block: PositionBlock):
        block_dir = self.block_dir(block)
        block_dir.mkdir(parents=True, exist_ok=True)
        lock_path = block_dir / ".cache.lock"
        with lock_path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _new_manifest(self, block: PositionBlock, evidence: ReadEvidenceBlock) -> dict[str, Any]:
        return {
            "format": "stitchv2.partitioned_evidence_cache",
            "version": self.version,
            "chromosome": self.chromosome,
            "block_id": int(block.block_id),
            "row_start": int(block.row_start),
            "row_stop": int(block.row_stop),
            "n_positions": int(evidence.positions.shape[0]),
            "positions_hash": _positions_hash(evidence.positions, evidence.ref, evidence.alt),
            "sample_batch_size": int(self.sample_batch_size),
            "compression": self.compression,
            "evidence_metadata": self.evidence_metadata,
            "evidence_metadata_hash": self.evidence_metadata_hash,
            "samples": {},
            "parts": {},
        }

    def _validate_manifest(self, manifest: dict[str, Any], block: PositionBlock) -> bool:
        positions = block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False)
        ref = block.dataframe["REF"].astype(str).to_numpy()
        alt = block.dataframe["ALT"].astype(str).to_numpy()
        return (
            str(manifest.get("chromosome")) == self.chromosome
            and int(manifest.get("row_start", -1)) == int(block.row_start)
            and int(manifest.get("row_stop", -1)) == int(block.row_stop)
            and str(manifest.get("positions_hash")) == _positions_hash(positions, ref, alt)
            and str(manifest.get("evidence_metadata_hash", "")) == self.evidence_metadata_hash
        )

    def _manifest_metadata_matches(self, manifest: dict[str, Any]) -> bool:
        return (
            str(manifest.get("chromosome")) == self.chromosome
            and str(manifest.get("evidence_metadata_hash", "")) == self.evidence_metadata_hash
        )

    def _find_covering_manifest(self, block: PositionBlock) -> tuple[Path, dict[str, Any], int, int] | None:
        """Find a cached superset block that can be sliced to the requested block.

        Exact-streaming runs intentionally cache one large chromosome/region partition.
        Approximate modes may then request smaller row ranges.  This lookup keeps the
        cache keyed by validated variant content while allowing safe row-slice reuse.
        """
        chrom_dir = self.root / f"chrom={self.chromosome}"
        if not chrom_dir.exists():
            return None
        row_start = int(block.row_start)
        row_stop = int(block.row_stop)
        wanted_pos = block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False)
        wanted_ref = block.dataframe["REF"].astype(str).to_numpy()
        wanted_alt = block.dataframe["ALT"].astype(str).to_numpy()
        for manifest_path in sorted(chrom_dir.glob("block=*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not self._manifest_metadata_matches(manifest):
                continue
            cached_start = int(manifest.get("row_start", -1))
            cached_stop = int(manifest.get("row_stop", -1))
            if cached_start > row_start or cached_stop < row_stop:
                continue
            rel_start = row_start - cached_start
            rel_stop = rel_start + int(wanted_pos.shape[0])
            block_dir = manifest_path.parent
            positions_path = block_dir / "positions.parquet"
            if not positions_path.exists():
                continue
            try:
                positions_table = pq.read_table(positions_path).to_pandas()
            except Exception:
                continue
            if rel_start < 0 or rel_stop > int(positions_table.shape[0]):
                continue
            pos = positions_table["position"].to_numpy(dtype=np.int64, copy=False)[rel_start:rel_stop]
            ref = positions_table["ref"].astype(str).to_numpy()[rel_start:rel_stop]
            alt = positions_table["alt"].astype(str).to_numpy()[rel_start:rel_stop]
            if (
                np.array_equal(pos, wanted_pos)
                and np.array_equal(ref, wanted_ref)
                and np.array_equal(alt, wanted_alt)
            ):
                return block_dir, manifest, int(rel_start), int(rel_stop)
        return None

    def write(self, block: PositionBlock, evidence: ReadEvidenceBlock) -> dict[str, Any]:
        with self._write_lock(block):
            block_dir = self.block_dir(block)
            manifest = self._manifest(block)
            if manifest is None or not self._validate_manifest(manifest, block):
                manifest = self._new_manifest(block, evidence)
                pq.write_table(
                    pa.table(
                        {
                            "position": evidence.positions.astype(np.int64, copy=False),
                            "ref": np.asarray(evidence.ref, dtype=str),
                            "alt": np.asarray(evidence.alt, dtype=str),
                        }
                    ),
                    block_dir / "positions.parquet",
                    compression=self.compression,
                )

            replaced_parts = {
                str(manifest.get("samples", {}).get(sample_id))
                for sample_id in evidence.sample_ids.astype(str).tolist()
                if sample_id in manifest.get("samples", {})
            }
            replaced_parts.discard("None")
            samples = evidence.sample_ids.astype(str).tolist()
            parts_written = 0
            for start in range(0, len(samples), self.sample_batch_size):
                stop = min(start + self.sample_batch_size, len(samples))
                sub = evidence.select_samples(np.arange(start, stop, dtype=np.int64))
                pid = _part_id()
                self._write_part(block_dir, pid, sub)
                for sample_id in sub.sample_ids.astype(str).tolist():
                    manifest["samples"][sample_id] = pid
                manifest["parts"][pid] = {
                    "n_samples": int(sub.sample_ids.shape[0]),
                    "compact_nbytes": int(sub.compact_nbytes),
                    "dense_nbytes": int(sub.dense_nbytes),
                    "includes_dense_counts": bool(self.include_dense_counts),
                }
                parts_written += 1
            referenced_parts = set(str(part_id) for part_id in manifest.get("samples", {}).values())
            for old_part in sorted(part for part in replaced_parts if part not in referenced_parts):
                self._remove_part_files(block_dir, old_part)
                manifest.get("parts", {}).pop(old_part, None)
            self._write_manifest(block, manifest)
        return self.stats(block)

    def _remove_part_files(self, block_dir: Path, part_id: str) -> None:
        for rel in (
            Path("fragments") / f"{part_id}.parquet",
            Path("summary") / f"{part_id}.parquet",
            Path("support") / f"{part_id}.parquet",
        ):
            path = block_dir / rel
            if path.exists():
                path.unlink()
        dense = block_dir / "dense" / f"{part_id}.zarr"
        if dense.exists():
            for path in sorted(dense.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            dense.rmdir()

    def _write_part(self, block_dir: Path, part_id: str, evidence: ReadEvidenceBlock) -> None:
        frag_rows: dict[str, list[Any]] = {
            "sample_id": [],
            "sample_index": [],
            "fragment_index": [],
            "center_idx": [],
            "obs_order": [],
            "pos_idx": [],
            "obs_code": [],
            "obs_qual": [],
        }
        summary_rows: dict[str, list[Any]] = {
            "sample_id": [],
            "sample_index": [],
            "n_overlapping_reads": [],
            "n_fragments": [],
            "n_fragment_observations": [],
        }
        for sample_idx, sample_id in enumerate(evidence.sample_ids.astype(str).tolist()):
            frag_start = int(evidence.fragment_sample_offsets[sample_idx])
            frag_stop = int(evidence.fragment_sample_offsets[sample_idx + 1])
            obs_count = 0
            for local_frag, frag_idx in enumerate(range(frag_start, frag_stop)):
                obs_start = int(evidence.fragment_obs_offsets[frag_idx])
                obs_stop = int(evidence.fragment_obs_offsets[frag_idx + 1])
                obs_count += obs_stop - obs_start
                for obs_order, obs_idx in enumerate(range(obs_start, obs_stop)):
                    frag_rows["sample_id"].append(sample_id)
                    frag_rows["sample_index"].append(sample_idx)
                    frag_rows["fragment_index"].append(local_frag)
                    frag_rows["center_idx"].append(int(evidence.fragment_center_idx[frag_idx]))
                    frag_rows["obs_order"].append(obs_order)
                    frag_rows["pos_idx"].append(int(evidence.fragment_obs_pos_idx[obs_idx]))
                    frag_rows["obs_code"].append(int(evidence.fragment_obs_code[obs_idx]))
                    frag_rows["obs_qual"].append(int(evidence.fragment_obs_qual[obs_idx]))
            summary_rows["sample_id"].append(sample_id)
            summary_rows["sample_index"].append(sample_idx)
            summary_rows["n_overlapping_reads"].append(int(evidence.n_overlapping_reads[sample_idx]))
            summary_rows["n_fragments"].append(frag_stop - frag_start)
            summary_rows["n_fragment_observations"].append(obs_count)

        frag_dir = block_dir / "fragments"
        summary_dir = block_dir / "summary"
        support_dir = block_dir / "support"
        frag_dir.mkdir(parents=True, exist_ok=True)
        summary_dir.mkdir(parents=True, exist_ok=True)
        support_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table(frag_rows), frag_dir / f"{part_id}.parquet", compression=self.compression)
        pq.write_table(pa.table(summary_rows), summary_dir / f"{part_id}.parquet", compression=self.compression)
        support_table = pa.table(
            {
                "sample_id": evidence.sample_ids.astype(str),
                "sample_index": np.arange(evidence.sample_ids.shape[0], dtype=np.int32),
                "n_positions": np.full(evidence.sample_ids.shape[0], evidence.positions.shape[0], dtype=np.int32),
                "bitorder": np.asarray(["little"] * evidence.sample_ids.shape[0]),
                "support_packed": pa.array(_bitpack_support(evidence.depth), type=pa.binary()),
            }
        )
        pq.write_table(support_table, support_dir / f"{part_id}.parquet", compression=self.compression)
        if self.include_dense_counts:
            self._write_dense_zarr(block_dir / "dense" / f"{part_id}.zarr", evidence)

    def _write_dense_zarr(self, path: Path, evidence: ReadEvidenceBlock) -> None:
        if path.exists():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(parents=True, exist_ok=True)
        (path / ".zgroup").write_text(json.dumps({"zarr_format": 2}, sort_keys=True), encoding="utf-8")
        (path / ".zattrs").write_text("{}", encoding="utf-8")
        chunks = (
            min(max(int(evidence.sample_ids.shape[0]), 1), 128),
            min(max(int(evidence.positions.shape[0]), 1), 4096),
        )
        compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
        for name in ("ref_count", "alt_count", "other_count", "depth", "ref_weight", "alt_weight", "other_weight"):
            self._write_zarr_v2_array(path / name, getattr(evidence, name), chunks, compressor)

    def _write_zarr_v2_array(
        self,
        path: Path,
        array: np.ndarray,
        chunks: tuple[int, int],
        compressor: numcodecs.abc.Codec,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(array)
        shape = tuple(int(v) for v in arr.shape)
        chunk_shape = tuple(int(v) for v in chunks)
        meta = {
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(chunk_shape),
            "dtype": np.dtype(arr.dtype).str,
            "compressor": compressor.get_config(),
            "fill_value": 0,
            "order": "C",
            "filters": None,
        }
        (path / ".zarray").write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        (path / ".zattrs").write_text("{}", encoding="utf-8")
        n0 = int(math.ceil(shape[0] / chunk_shape[0])) if shape[0] else 0
        n1 = int(math.ceil(shape[1] / chunk_shape[1])) if shape[1] else 0
        for i in range(n0):
            r0 = i * chunk_shape[0]
            r1 = min(r0 + chunk_shape[0], shape[0])
            for j in range(n1):
                c0 = j * chunk_shape[1]
                c1 = min(c0 + chunk_shape[1], shape[1])
                chunk = np.zeros(chunk_shape, dtype=arr.dtype)
                chunk[: r1 - r0, : c1 - c0] = arr[r0:r1, c0:c1]
                encoded = compressor.encode(np.ascontiguousarray(chunk))
                (path / f"{i}.{j}").write_bytes(bytes(encoded))

    def load_available(
        self,
        block: PositionBlock,
        sample_ids: np.ndarray,
    ) -> tuple[ReadEvidenceBlock | None, np.ndarray, dict[str, Any]]:
        manifest = self._manifest(block)
        requested = np.asarray(sample_ids, dtype=object)
        block_dir = self.block_dir(block)
        slice_start: int | None = None
        slice_stop: int | None = None
        cache_stats = self.stats(block)
        if manifest is None or not self._validate_manifest(manifest, block):
            covering = self._find_covering_manifest(block)
            if covering is None:
                return None, np.ones(requested.shape[0], dtype=bool), cache_stats
            block_dir, manifest, slice_start, slice_stop = covering
            cache_stats = self.stats_from_manifest(block_dir, manifest)
            cache_stats.update(
                {
                    "partitioned_cache_sliced_from_superset": True,
                    "partitioned_cache_superset_path": str(block_dir),
                    "partitioned_cache_slice_start": int(slice_start),
                    "partitioned_cache_slice_stop": int(slice_stop),
                }
            )
        sample_map = manifest.get("samples", {})
        cached_mask = np.asarray([str(sample_id) in sample_map for sample_id in requested.astype(str).tolist()], dtype=bool)
        if not np.any(cached_mask):
            return None, ~cached_mask, cache_stats
        grouped: dict[str, list[str]] = {}
        for sample_id in requested[cached_mask].astype(str).tolist():
            grouped.setdefault(str(sample_map[sample_id]), []).append(sample_id)
        blocks: list[ReadEvidenceBlock] = []
        for part_id, part_samples in grouped.items():
            part = self._read_part(block_dir, part_id, manifest)
            idx_by_sample = {sid: idx for idx, sid in enumerate(part.sample_ids.astype(str).tolist())}
            idx = np.asarray([idx_by_sample[sid] for sid in part_samples], dtype=np.int64)
            selected = part.select_samples(idx)
            if slice_start is not None and slice_stop is not None:
                selected = selected.slice_by_position_rows(slice_start, slice_stop, block_id=int(block.block_id))
            blocks.append(selected)
        cached = ReadEvidenceBlock.merge_sample_blocks(
            blocks,
            requested[cached_mask],
            block_id=int(block.block_id),
        )
        cache_stats["partitioned_cache_cached_samples"] = int(np.count_nonzero(cached_mask))
        cache_stats["partitioned_cache_missing_samples"] = int(np.count_nonzero(~cached_mask))
        return cached, ~cached_mask, cache_stats

    def _read_part(self, block_dir: Path, part_id: str, manifest: dict[str, Any]) -> ReadEvidenceBlock:
        summary = pq.read_table(block_dir / "summary" / f"{part_id}.parquet").to_pandas()
        fragments = pq.read_table(block_dir / "fragments" / f"{part_id}.parquet").to_pandas()
        positions_table = pq.read_table(block_dir / "positions.parquet").to_pandas()
        sample_ids = summary.sort_values("sample_index")["sample_id"].astype(str).to_numpy(dtype=object)
        positions = positions_table["position"].to_numpy(dtype=np.int64, copy=False)
        ref = positions_table["ref"].astype(str).to_numpy(dtype=object)
        alt = positions_table["alt"].astype(str).to_numpy(dtype=object)
        shape = (int(sample_ids.shape[0]), int(positions.shape[0]))
        ref_count, alt_count, other_count, depth, ref_weight, alt_weight, other_weight = _dense_zero(shape)
        part_meta = manifest.get("parts", {}).get(part_id, {})
        if self.materialize_dense_counts and bool(part_meta.get("includes_dense_counts", False)):
            ref_count, alt_count, other_count, depth, ref_weight, alt_weight, other_weight = self._read_dense_zarr(block_dir / "dense" / f"{part_id}.zarr")
        elif not self.materialize_dense_counts:
            support_path = block_dir / "support" / f"{part_id}.parquet"
            if support_path.exists():
                support = pq.read_table(support_path).to_pandas().sort_values("sample_index")
                depth = _unpack_support(support["support_packed"].tolist(), int(positions.shape[0]))

        sample_offsets = np.zeros(sample_ids.shape[0] + 1, dtype=np.int64)
        centers: list[int] = []
        obs_offsets: list[int] = [0]
        obs_pos: list[int] = []
        obs_code: list[int] = []
        obs_qual: list[int] = []
        frag_groups: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
        if not fragments.empty:
            fragments = fragments.sort_values(["sample_index", "fragment_index", "obs_order"])
            for row in fragments.itertuples(index=False):
                key = (int(getattr(row, "sample_index")), int(getattr(row, "fragment_index")))
                frag_groups.setdefault(key, []).append(
                    (
                        int(getattr(row, "center_idx")),
                        int(getattr(row, "pos_idx")),
                        int(getattr(row, "obs_code")),
                        int(getattr(row, "obs_qual")),
                    )
                )
        summary_sorted = summary.sort_values("sample_index")
        for out_idx, row in enumerate(summary_sorted.itertuples(index=False)):
            sample_index = int(getattr(row, "sample_index"))
            n_fragments = int(getattr(row, "n_fragments"))
            sample_offsets[out_idx + 1] = sample_offsets[out_idx] + n_fragments
            if n_fragments <= 0:
                continue
            for frag_idx in range(n_fragments):
                frag_rows = frag_groups.get((sample_index, frag_idx), [])
                if not frag_rows:
                    centers.append(-1)
                    obs_offsets.append(obs_offsets[-1])
                    continue
                centers.append(int(frag_rows[0][0]))
                obs_pos.extend(int(row[1]) for row in frag_rows)
                obs_code.extend(int(row[2]) for row in frag_rows)
                obs_qual.extend(int(row[3]) for row in frag_rows)
                obs_offsets.append(len(obs_pos))
        n_overlapping_reads = summary_sorted["n_overlapping_reads"].to_numpy(dtype=np.int32, copy=False)
        evidence = ReadEvidenceBlock(
            block_id=int(manifest.get("block_id", 0)),
            chromosome=str(manifest.get("chromosome", self.chromosome)),
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
            fragment_sample_offsets=sample_offsets,
            fragment_center_idx=np.asarray(centers, dtype=np.int32),
            fragment_obs_offsets=np.asarray(obs_offsets, dtype=np.int64),
            fragment_obs_pos_idx=np.asarray(obs_pos, dtype=np.int32),
            fragment_obs_code=np.asarray(obs_code, dtype=np.int8),
            fragment_obs_qual=np.asarray(obs_qual, dtype=np.uint8),
            memmap_dir=None,
        )
        if self.materialize_dense_counts and not bool(part_meta.get("includes_dense_counts", False)):
            _materialize_dense_from_fragments(evidence, support_only=False)
        return evidence

    def _read_dense_zarr(self, path: Path) -> tuple[np.ndarray, ...]:
        return tuple(
            self._read_zarr_v2_array(path / name)
            for name in ("ref_count", "alt_count", "other_count", "depth", "ref_weight", "alt_weight", "other_weight")
        )

    def _read_zarr_v2_array(self, path: Path) -> np.ndarray:
        meta = json.loads((path / ".zarray").read_text(encoding="utf-8"))
        shape = tuple(int(v) for v in meta["shape"])
        chunk_shape = tuple(int(v) for v in meta["chunks"])
        dtype = np.dtype(str(meta["dtype"]))
        compressor = numcodecs.get_codec(meta["compressor"])
        out = np.zeros(shape, dtype=dtype)
        n0 = int(math.ceil(shape[0] / chunk_shape[0])) if shape[0] else 0
        n1 = int(math.ceil(shape[1] / chunk_shape[1])) if shape[1] else 0
        for i in range(n0):
            r0 = i * chunk_shape[0]
            r1 = min(r0 + chunk_shape[0], shape[0])
            for j in range(n1):
                chunk_path = path / f"{i}.{j}"
                if not chunk_path.exists():
                    continue
                c0 = j * chunk_shape[1]
                c1 = min(c0 + chunk_shape[1], shape[1])
                decoded = compressor.decode(chunk_path.read_bytes())
                chunk = np.frombuffer(decoded, dtype=dtype).reshape(chunk_shape)
                out[r0:r1, c0:c1] = chunk[: r1 - r0, : c1 - c0]
        return out

    def stats(self, block: PositionBlock) -> dict[str, Any]:
        block_dir = self.block_dir(block)
        manifest = self._manifest(block)
        return self.stats_from_manifest(block_dir, manifest)

    def stats_from_manifest(self, block_dir: Path, manifest: dict[str, Any] | None) -> dict[str, Any]:
        total = 0
        parquet = 0
        zarr = 0
        if block_dir.exists():
            for path in block_dir.rglob("*"):
                if path.is_file():
                    size = int(path.stat().st_size)
                    total += size
                    if any(part.endswith(".zarr") for part in path.parts):
                        zarr += size
                    elif path.suffix == ".parquet":
                        parquet += size
        return {
            "partitioned_cache_format": "parquet_zarr",
            "partitioned_cache_path": str(block_dir),
            "partitioned_cache_file_bytes": int(total),
            "partitioned_cache_file_mb": float(total / 1_000_000.0),
            "partitioned_cache_parquet_bytes": int(parquet),
            "partitioned_cache_zarr_bytes": int(zarr),
            "partitioned_cache_manifest_samples": int(len((manifest or {}).get("samples", {}))),
            "partitioned_cache_manifest_parts": int(len((manifest or {}).get("parts", {}))),
            "partitioned_cache_include_dense_counts": bool(self.include_dense_counts),
            "partitioned_cache_materialize_dense_counts": bool(self.materialize_dense_counts),
            "partitioned_cache_evidence_metadata_hash": self.evidence_metadata_hash,
        }
