from __future__ import annotations

import json
from pathlib import Path
import shlex
import shutil
import zlib
from typing import Iterable

import numpy as np


def shell_join(parts: Iterable[object]) -> str:
    return " ".join(shlex.quote(str(x)) for x in parts if str(x) != "")


def write_slurm_sample_shards(
    *,
    output_dir: str | Path,
    n_shards: int,
    samples: str,
    positions: str,
    chromosome: str,
    founder_plink: str,
    evidence_cache_dir: str,
    run_root: str,
    preset: str = "hs-rat-k8-production",
    n_founders: int = 8,
    job_name: str = "stitchcont",
    account: str = "",
    partition: str = "",
    time_limit: str = "12:00:00",
    mem: str = "100G",
    cpus_per_task: int = 8,
    extra_args: str = "",
) -> dict[str, object]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scripts: list[str] = []
    run_root_p = Path(run_root)
    for shard_idx in range(int(n_shards)):
        shard_dir = run_root_p / f"shard_{shard_idx:04d}"
        cmd = [
            "stitchcont", "impute-from-evidence",
            "--preset", preset,
            "--samples", samples,
            "--positions", positions,
            "--chromosome", chromosome,
            "--output-dir", str(shard_dir),
            "--founder-plink", founder_plink,
            "--n-founders", str(int(n_founders)),
            "--founder-immutable",
            "--compact-evidence-cache-dir", evidence_cache_dir,
            "--hmm-backend", "numba",
            "--snp-block-mode", "exact_chunked",
            "--output-store", "zarr",
            "--output-minimal",
            "--use-unordered-diploid-states",
            "--store-xi", "False",
            "--write-gamma", "off",
            "--sample-shard-index", str(shard_idx),
            "--sample-shard-count", str(int(n_shards)),
        ]
        if extra_args.strip():
            cmd.extend(shlex.split(extra_args))
        lines = [
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name={job_name}_{shard_idx:04d}",
            f"#SBATCH --output={out / f'shard_{shard_idx:04d}.%j.out'}",
            f"#SBATCH --error={out / f'shard_{shard_idx:04d}.%j.err'}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --mem={mem}",
            f"#SBATCH --cpus-per-task={int(cpus_per_task)}",
        ]
        if account:
            lines.append(f"#SBATCH --account={account}")
        if partition:
            lines.append(f"#SBATCH --partition={partition}")
        lines.extend([
            "set -euo pipefail",
            "echo \"[$(date)] starting shard ${SLURM_JOB_ID:-local}\"",
            shell_join(cmd),
            "echo \"[$(date)] finished shard ${SLURM_JOB_ID:-local}\"",
            "",
        ])
        path = out / f"shard_{shard_idx:04d}.sbatch"
        path.write_text("\n".join(lines), encoding="utf-8")
        scripts.append(str(path))
    submit = out / "submit_all.sh"
    submit.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(f"sbatch {shlex.quote(str(p))}" for p in scripts) + "\n", encoding="utf-8")
    submit.chmod(0o755)
    merge = out / "merge_shards.sh"
    merge.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + shell_join(["stitchcont", "merge-shards", "--input-root", str(run_root_p), "--output-dir", str(run_root_p / "merged")])
        + "\n",
        encoding="utf-8",
    )
    merge.chmod(0o755)
    manifest = {
        "n_shards": int(n_shards),
        "scripts": scripts,
        "submit_script": str(submit),
        "merge_script": str(merge),
        "run_root": str(run_root_p),
        "preset": preset,
    }
    (out / "slurm_shards_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _chunk_path(array_dir: Path, chunk_index: tuple[int, ...]) -> Path:
    return array_dir / ".".join(str(int(i)) for i in chunk_index)


def _edge_shape(shape: tuple[int, ...], chunks: tuple[int, ...], chunk_index: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for dim, chunk, idx in zip(shape, chunks, chunk_index, strict=False):
        start = int(idx) * int(chunk)
        out.append(max(0, min(int(chunk), int(dim) - start)))
    return tuple(out)


def _zarr_uses_zlib(zarray: dict) -> bool:
    comp = zarray.get("compressor")
    return isinstance(comp, dict) and str(comp.get("id", "")).lower() == "zlib"


def _read_array_chunk(array_dir: Path, zarray: dict, chunk_index: tuple[int, ...]) -> np.ndarray:
    shape = tuple(int(x) for x in zarray["shape"])
    chunks = tuple(int(x) for x in zarray["chunks"])
    dtype = np.dtype(zarray["dtype"])
    chunk_shape = _edge_shape(shape, chunks, chunk_index)
    path = _chunk_path(array_dir, chunk_index)
    if path.exists():
        raw = path.read_bytes()
        if _zarr_uses_zlib(zarray):
            raw = zlib.decompress(raw)
        return np.frombuffer(raw, dtype=dtype).copy().reshape(chunk_shape)
    fill = zarray.get("fill_value", 0)
    if fill == "NaN":
        fill = np.nan
    return np.full(chunk_shape, fill, dtype=dtype)


def _write_array_chunk(array_dir: Path, zarray: dict, chunk_index: tuple[int, ...], arr: np.ndarray) -> None:
    path = _chunk_path(array_dir, chunk_index)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.ascontiguousarray(arr).tobytes(order="C")
    if _zarr_uses_zlib(zarray):
        level = int(zarray.get("compressor", {}).get("level", 1))
        raw = zlib.compress(raw, level=max(0, min(level, 9)))
    path.write_bytes(raw)


def _ensure_output_array(root: Path, name: str, source_zarray: dict, source_zattrs: dict, total_samples: int) -> tuple[Path, dict]:
    arr_dir = root / name
    arr_dir.mkdir(parents=True, exist_ok=True)
    zarray = dict(source_zarray)
    shape = list(zarray["shape"])
    shape[0] = int(total_samples)
    zarray["shape"] = shape
    (arr_dir / ".zarray").write_text(json.dumps(zarray, indent=2, sort_keys=True), encoding="utf-8")
    (arr_dir / ".zattrs").write_text(json.dumps(source_zattrs, indent=2, sort_keys=True), encoding="utf-8")
    return arr_dir, zarray


def merge_sharded_matrix_zarr(*, input_root: str | Path, output_dir: str | Path) -> dict[str, object]:
    """Merge sample-sharded `matrices.zarr` stores along sample axis.

    The implementation streams chunk files and never constructs a full output
    matrix in memory. It assumes all shards used the same positions and array
    schemas except for the sample dimension.
    """
    input_root = Path(input_root)
    output_dir = Path(output_dir)
    shard_dirs = sorted(p for p in input_root.glob("shard_*") if (p / "matrices.zarr").exists())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_*/matrices.zarr stores found under {input_root}")
    out_root = output_dir / "matrices.zarr"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ".zgroup").write_text(json.dumps({"zarr_format": 2}, sort_keys=True), encoding="utf-8")
    sample_ids: list[str] = []
    shard_sample_offsets: list[int] = []
    arrays = sorted([p.name for p in (shard_dirs[0] / "matrices.zarr").iterdir() if (p / ".zarray").exists()])
    # Copy shared position axis from first shard.
    first_root = shard_dirs[0] / "matrices.zarr"
    if (first_root / "positions.npy").exists():
        shutil.copy2(first_root / "positions.npy", out_root / "positions.npy")
    for shard in shard_dirs:
        ids = json.loads((shard / "matrices.zarr" / "sample_ids.json").read_text(encoding="utf-8"))
        shard_sample_offsets.append(len(sample_ids))
        sample_ids.extend([str(x) for x in ids])
    (out_root / "sample_ids.json").write_text(json.dumps(sample_ids, indent=2), encoding="utf-8")
    root_attrs = {
        "format": "stitchcont.matrix_zarr.v1",
        "merged_from_sample_shards": [str(p) for p in shard_dirs],
        # Rows are concatenated in shard-directory-sorted order, NOT original input
        # order (mod-based sharding interleaves input rows). sample_ids.json lists
        # the row labels in matrix-row order; consumers MUST map rows by sample_id,
        # not by assuming the original sample table order.
        "sample_order": "shard_concatenation_sorted_by_shard_dir",
    }
    (out_root / ".zattrs").write_text(json.dumps(root_attrs, indent=2, sort_keys=True), encoding="utf-8")
    manifest: dict[str, object] = {"n_shards": len(shard_dirs), "n_samples": len(sample_ids), "arrays": {}}
    for name in arrays:
        source_arr = shard_dirs[0] / "matrices.zarr" / name
        source_zarray = _read_json(source_arr / ".zarray")
        source_zattrs = _read_json(source_arr / ".zattrs")
        out_arr, out_zarray = _ensure_output_array(out_root, name, source_zarray, source_zattrs, len(sample_ids))
        out_shape = tuple(int(x) for x in out_zarray["shape"])
        out_chunks = tuple(int(x) for x in out_zarray["chunks"])
        dtype = np.dtype(out_zarray["dtype"])
        fill = out_zarray.get("fill_value", 0)
        if fill == "NaN":
            fill = np.nan
        for shard, sample_offset in zip(shard_dirs, shard_sample_offsets, strict=True):
            shard_arr = shard / "matrices.zarr" / name
            z = _read_json(shard_arr / ".zarray")
            local_shape = tuple(int(x) for x in z["shape"])
            local_chunks = tuple(int(x) for x in z["chunks"])
            # Validate that this shard's array schema matches the reference shard on
            # every axis except the sample axis (0).  The merge places data using
            # the reference chunk grid and copies axes >=2 as whole chunks, so a
            # mismatch in dtype, non-sample shape, or non-sample chunking would
            # silently misplace data.  Fail loudly instead.
            ref_shape = tuple(int(x) for x in source_zarray["shape"])
            ref_chunks = tuple(int(x) for x in source_zarray["chunks"])
            if str(z["dtype"]) != str(source_zarray["dtype"]):
                raise ValueError(
                    f"Shard {shard.name} array {name!r} dtype {z['dtype']} != reference {source_zarray['dtype']}."
                )
            if len(local_shape) != len(ref_shape) or local_shape[1:] != ref_shape[1:]:
                raise ValueError(
                    f"Shard {shard.name} array {name!r} non-sample shape {local_shape[1:]} "
                    f"!= reference {ref_shape[1:]}; shards must share positions and trailing dims."
                )
            if local_chunks[1:] != ref_chunks[1:]:
                raise ValueError(
                    f"Shard {shard.name} array {name!r} non-sample chunking {local_chunks[1:]} "
                    f"!= reference {ref_chunks[1:]}; sharded merge requires identical "
                    "position/trailing-axis chunking across shards."
                )
            # Iterate source sample/position chunks. Extra axes are copied as whole chunks.
            extra_shape = tuple(int(np.ceil(local_shape[d] / local_chunks[d])) for d in range(2, len(local_shape)))
            extra_indices = list(np.ndindex(*extra_shape)) if extra_shape else [()]
            for si in range(int(np.ceil(local_shape[0] / local_chunks[0]))):
                for pi in range(int(np.ceil(local_shape[1] / local_chunks[1]))):
                    for extra_idx in extra_indices:
                        src_idx = (si, pi) + tuple(int(x) for x in extra_idx)
                        block = _read_array_chunk(shard_arr, z, src_idx)
                        global_sample_start = sample_offset + si * local_chunks[0]
                        global_sample_stop = global_sample_start + block.shape[0]
                        global_pos_start = pi * local_chunks[1]
                        # May cross output sample chunks if local/out chunks differ.
                        s0 = global_sample_start
                        while s0 < global_sample_stop:
                            out_si = s0 // out_chunks[0]
                            out_s_start = out_si * out_chunks[0]
                            take_s0 = s0 - global_sample_start
                            take_s1 = min(global_sample_stop, out_s_start + out_chunks[0]) - global_sample_start
                            p0 = global_pos_start
                            p_stop = global_pos_start + block.shape[1]
                            while p0 < p_stop:
                                out_pi = p0 // out_chunks[1]
                                out_p_start = out_pi * out_chunks[1]
                                take_p0 = p0 - global_pos_start
                                take_p1 = min(p_stop, out_p_start + out_chunks[1]) - global_pos_start
                                out_idx = (out_si, out_pi) + tuple(int(x) for x in extra_idx)
                                out_chunk_shape = _edge_shape(out_shape, out_chunks, out_idx)
                                out_path = _chunk_path(out_arr, out_idx)
                                if out_path.exists():
                                    out_chunk = np.frombuffer(out_path.read_bytes(), dtype=dtype).copy().reshape(out_chunk_shape)
                                else:
                                    out_chunk = np.full(out_chunk_shape, fill, dtype=dtype)
                                dst_s0 = s0 - out_s_start
                                dst_s1 = dst_s0 + (take_s1 - take_s0)
                                dst_p0 = p0 - out_p_start
                                dst_p1 = dst_p0 + (take_p1 - take_p0)
                                if block.ndim == 2:
                                    out_chunk[dst_s0:dst_s1, dst_p0:dst_p1] = block[take_s0:take_s1, take_p0:take_p1]
                                else:
                                    out_chunk[dst_s0:dst_s1, dst_p0:dst_p1, ...] = block[take_s0:take_s1, take_p0:take_p1, ...]
                                _write_array_chunk(out_arr, out_zarray, out_idx, out_chunk)
                                p0 = global_pos_start + take_p1
                            s0 = global_sample_start + take_s1
        manifest["arrays"][name] = {"shape": list(out_shape), "chunks": list(out_chunks), "dtype": str(dtype)}
    (output_dir / "merge_shards_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def write_evidence_sample_shard_layout(
    *,
    output_dir: str | Path,
    samples_path: str | Path,
    evidence_cache_dir: str | Path,
    n_shards: int,
    scheme: str = "row_index_mod_sample_shard_count",
) -> dict[str, object]:
    """Create lightweight sample-shard manifests for a shared evidence cache.

    This does not duplicate the compact evidence cache.  It writes per-shard
    sample tables and a manifest pointing all shards at the same immutable cache,
    which makes large CPU runs easier to schedule and avoids repeated ad hoc
    shard bookkeeping in SLURM scripts.
    """
    import pandas as pd

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(samples_path) if str(samples_path).endswith(".parquet") else pd.read_csv(samples_path, sep=None, engine="python")
    n = int(samples.shape[0])
    n_shards = max(int(n_shards), 1)
    shards: list[dict[str, object]] = []
    for shard_idx in range(n_shards):
        row_index = np.arange(n, dtype=np.int64)
        mask = (row_index % n_shards) == shard_idx
        shard_samples = samples.loc[mask].reset_index(drop=True)
        shard_dir = out / f"shard_{shard_idx:04d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_samples_path = shard_dir / "samples.parquet"
        shard_samples.to_parquet(shard_samples_path, index=False)
        sample_ids = shard_samples.get("sample_id", shard_samples.iloc[:, 0]).astype(str).tolist()
        meta = {
            "shard_index": int(shard_idx),
            "shard_count": int(n_shards),
            "n_samples": int(shard_samples.shape[0]),
            "samples_path": str(shard_samples_path),
            "sample_ids_preview": sample_ids[:10],
            "evidence_cache_dir": str(evidence_cache_dir),
            "scheme": scheme,
        }
        (shard_dir / "evidence_shard.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        shards.append(meta)
    manifest = {
        "format": "stitchcont.evidence_sample_shard_layout.v1",
        "samples_path": str(samples_path),
        "evidence_cache_dir": str(evidence_cache_dir),
        "n_samples": n,
        "n_shards": n_shards,
        "scheme": scheme,
        "shards": shards,
    }
    (out / "evidence_sample_shards_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
