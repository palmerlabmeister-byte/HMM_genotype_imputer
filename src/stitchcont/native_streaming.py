from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zlib

import numpy as np

from .native_cpu import native_k8_counts_reduce, native_k8_counts_fragments_reduce


def _write_zarr_array_metadata(array_dir: Path, shape: tuple[int, ...], chunks: tuple[int, ...], dtype: np.dtype, attrs: dict[str, Any] | None = None) -> None:
    array_dir.mkdir(parents=True, exist_ok=True)
    (array_dir / ".zarray").write_text(
        json.dumps(
            {
                "zarr_format": 2,
                "shape": [int(x) for x in shape],
                "chunks": [int(x) for x in chunks],
                "dtype": np.dtype(dtype).str,
                "compressor": {"id": "zlib", "level": 1},
                "fill_value": 0,
                "order": "C",
                "filters": None,
                "dimension_separator": ".",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    zattrs = dict(attrs or {})
    if len(shape) == 2:
        zattrs.setdefault("_ARRAY_DIMENSIONS", ["sample", "position"])
    elif len(shape) == 3:
        zattrs.setdefault("_ARRAY_DIMENSIONS", ["sample", "position", "genotype"])
    (array_dir / ".zattrs").write_text(json.dumps(zattrs, indent=2, sort_keys=True), encoding="utf-8")


def _write_zarr_chunk(array_dir: Path, chunk_index: tuple[int, ...], arr: np.ndarray, *, compress: bool = True) -> None:
    raw = np.ascontiguousarray(arr).tobytes(order="C")
    if compress:
        raw = zlib.compress(raw, level=1)
    (array_dir / ".".join(str(int(i)) for i in chunk_index)).write_bytes(raw)


def stream_k8_counts_to_zarr(
    *,
    output_root: str | Path,
    ref_obs: np.ndarray,
    alt_obs: np.ndarray,
    other_obs: np.ndarray | None,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    sample_chunk_size: int = 512,
    position_chunk_size: int | None = None,
    sequencing_error_rate: float = 0.01,
    min_emission_prob: float = 1e-5,
    gp_dtype: str = "uint16",
    dosage_dtype: str = "float16",
    probability_scale: int = 65535,
    n_threads: int = 0,
    fragment_arrays: dict[str, np.ndarray] | None = None,
    checkpoint_interval: int = 0,
) -> dict[str, Any]:
    """Run native K8 count/fragment reduction by sample chunks and write Zarr chunks.

    This is a memory-bounded native-to-Zarr streaming path: C++ computes one
    sample chunk at a time, and Python writes each chunk immediately. It avoids
    holding full dosage/GP matrices in memory. Zarr metadata/writing stays in
    Python for portability; the expensive emission/HMM work stays native.
    """
    ref = np.asarray(ref_obs, dtype=np.float32, order="C")
    alt = np.asarray(alt_obs, dtype=np.float32, order="C")
    oth = np.zeros_like(ref) if other_obs is None else np.asarray(other_obs, dtype=np.float32, order="C")
    sw = np.asarray(switch, dtype=np.float32, order="C")
    fa = np.asarray(founder_alt, dtype=np.float32, order="C")
    if ref.shape != alt.shape or ref.shape != oth.shape or ref.shape != sw.shape:
        raise ValueError("ref_obs, alt_obs, other_obs and switch must have shape [n_samples, n_positions]")
    if fa.shape != (8, ref.shape[1]):
        raise ValueError("founder_alt must have shape [8, n_positions]")
    n_samples, n_positions = ref.shape
    sample_chunk_size = max(1, int(sample_chunk_size))
    position_chunk_size = int(position_chunk_size or n_positions)
    position_chunk_size = max(1, min(position_chunk_size, n_positions))
    out = Path(output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / ".zgroup").write_text(json.dumps({"zarr_format": 2}, sort_keys=True), encoding="utf-8")
    (out / ".zattrs").write_text(json.dumps({"format": "stitchcont.native_streaming_zarr.v1"}, indent=2), encoding="utf-8")
    dosage_store_dtype = np.dtype(np.float16 if str(dosage_dtype).lower() == "float16" else np.float32)
    gp_store_dtype = np.dtype(np.uint16 if str(gp_dtype).lower() == "uint16" else (np.float16 if str(gp_dtype).lower() == "float16" else np.float32))
    dosage_dir = out / "dosage"
    gp_dir = out / "genotype_posterior"
    _write_zarr_array_metadata(dosage_dir, (n_samples, n_positions), (sample_chunk_size, position_chunk_size), dosage_store_dtype, {"description": "ALT dosage"})
    _write_zarr_array_metadata(gp_dir, (n_samples, n_positions, 3), (sample_chunk_size, position_chunk_size, 3), gp_store_dtype, {"description": "genotype posterior", "probability_scale": int(probability_scale) if gp_store_dtype == np.dtype(np.uint16) else None})
    chunks_written = 0
    for sample_start in range(0, n_samples, sample_chunk_size):
        sample_stop = min(sample_start + sample_chunk_size, n_samples)
        sl = slice(sample_start, sample_stop)
        if fragment_arrays:
            # The compact fragment arrays are sample-relative in normal pipeline
            # use. For this generic helper, require caller to pass already sliced
            # arrays for the current sample chunk through a callback-like dict only
            # when fragment arrays are empty/global-compatible.
            result = native_k8_counts_fragments_reduce(
                ref[sl], alt[sl], oth[sl], sw[sl], fa,
                fragment_sample_offsets=fragment_arrays.get("fragment_sample_offsets", np.zeros(sample_stop - sample_start + 1, dtype=np.int64)),
                fragment_center_idx=fragment_arrays.get("fragment_center_idx", np.zeros(0, dtype=np.int64)),
                fragment_obs_offsets=fragment_arrays.get("fragment_obs_offsets", np.zeros(1, dtype=np.int64)),
                fragment_obs_pos_idx=fragment_arrays.get("fragment_obs_pos_idx", np.zeros(0, dtype=np.int64)),
                fragment_obs_code=fragment_arrays.get("fragment_obs_code", np.zeros(0, dtype=np.int8)),
                fragment_obs_qual=fragment_arrays.get("fragment_obs_qual", np.zeros(0, dtype=np.float32)),
                sequencing_error_rate=sequencing_error_rate,
                min_emission_prob=min_emission_prob,
                mode=str(fragment_arrays.get("mode", "replace")),
                rescale=bool(fragment_arrays.get("rescale", True)),
                max_emission_matrix_difference=float(fragment_arrays.get("max_emission_matrix_difference", 1e10)),
                n_threads=n_threads,
            )
        else:
            result = native_k8_counts_reduce(ref[sl], alt[sl], oth[sl], sw[sl], fa, sequencing_error_rate=sequencing_error_rate, min_emission_prob=min_emission_prob, n_threads=n_threads, checkpoint_interval=int(checkpoint_interval))
        dosage = result["dosage"]
        gp = result["genotype_posterior"]
        if dosage_store_dtype == np.dtype(np.float16):
            dosage = dosage.astype(np.float16)
        else:
            dosage = dosage.astype(np.float32)
        if gp_store_dtype == np.dtype(np.uint16):
            gp_out = np.rint(np.clip(gp, 0.0, 1.0) * float(probability_scale)).astype(np.uint16)
        else:
            gp_out = gp.astype(gp_store_dtype)
        for pos_start in range(0, n_positions, position_chunk_size):
            pos_stop = min(pos_start + position_chunk_size, n_positions)
            cidx = (sample_start // sample_chunk_size, pos_start // position_chunk_size)
            _write_zarr_chunk(dosage_dir, cidx, dosage[:, pos_start:pos_stop])
            _write_zarr_chunk(gp_dir, cidx + (0,), gp_out[:, pos_start:pos_stop, :])
            chunks_written += 2
    manifest = {"n_samples": int(n_samples), "n_positions": int(n_positions), "sample_chunk_size": int(sample_chunk_size), "position_chunk_size": int(position_chunk_size), "chunks_written": int(chunks_written)}
    (out / "native_streaming_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
