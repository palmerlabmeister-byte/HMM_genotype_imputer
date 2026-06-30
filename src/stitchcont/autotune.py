from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_memory_budget(value: str | int | float | None) -> int:
    if value is None:
        return 64 * 1024**3
    if isinstance(value, (int, float)):
        v = float(value)
        return int(v if v > 1 else v * 64 * 1024**3)
    text = str(value).strip().lower()
    if text.endswith('%'):
        # Conservative fallback: percent of 64 GiB when system memory is unknown.
        return int(64 * 1024**3 * float(text[:-1]) / 100.0)
    mult = 1
    for suffix, m in [("gib", 1024**3), ("gb", 1000**3), ("g", 1024**3), ("mib", 1024**2), ("mb", 1000**2), ("m", 1024**2), ("kib", 1024), ("kb", 1000), ("k", 1024)]:
        if text.endswith(suffix):
            mult = m
            text = text[: -len(suffix)]
            break
    return int(float(text) * mult)


def _read_table(path: str | Path, *, max_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path).head(max_rows) if max_rows else pd.read_parquet(path)
        except Exception:
            import pyarrow.parquet as pq
            tbl = pq.read_table(path)
            df = tbl.to_pandas()
            return df.head(max_rows) if max_rows else df
    df = pd.read_csv(path, sep=None, engine="python")
    return df.head(max_rows) if max_rows else df


def estimate_autotune_plan(
    *,
    samples_path: str | Path,
    positions_path: str | Path,
    n_founders: int = 8,
    target_memory: str | int | float = "100G",
    target_memory_fraction: float = 0.75,
    write_gp: bool = True,
    use_unordered_diploid_states: bool = True,
    checkpoint_interval: int = 512,
    dosage_dtype: str = "float32",
    gp_dtype: str = "float32",
    min_chunk_size: int = 5_000,
    max_chunk_size: int = 100_000,
    desired_sample_shards: int = 0,
) -> dict[str, Any]:
    samples = _read_table(samples_path, max_rows=None)
    positions = _read_table(positions_path, max_rows=None)
    n_samples = int(samples.shape[0])
    n_positions = int(positions.shape[0])
    k = int(n_founders)
    state_count = k * (k + 1) // 2 if use_unordered_diploid_states else k * k
    mem_budget = int(parse_memory_budget(target_memory) * float(target_memory_fraction))
    dosage_bytes = 2 if str(dosage_dtype).lower() == "float16" else 4
    gp_bytes = 2 if str(gp_dtype).lower() in {"float16", "uint16"} else 4
    out_per_cell = dosage_bytes + (3 * gp_bytes if write_gp else 0)
    # State memory estimate assumes checkpointed alpha plus local recompute buffers.
    chk = max(int(checkpoint_interval), 1)
    state_bytes_per_sample_pos = state_count * 4.0
    state_memory_factor = min(2.5, 1.0 + 2.0 * min(chk, max_chunk_size) / float(max_chunk_size))
    fixed_overhead = 2 * 1024**3
    # Allow one chunk of outputs and HMM buffers; keep generous overhead for evidence overlays.
    usable = max(mem_budget - fixed_overhead, 256 * 1024**2)
    # Start with sample shards if requested; otherwise solve for chunk/shard jointly.
    best = None
    candidate_shards = [desired_sample_shards] if desired_sample_shards > 0 else [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128]
    for shards in candidate_shards:
        samples_per_shard = int(math.ceil(n_samples / max(shards, 1)))
        for chunk_size in [max_chunk_size, 75_000, 50_000, 25_000, 20_000, 10_000, min_chunk_size]:
            chunk_size = int(min(max(chunk_size, min_chunk_size), max_chunk_size, max(n_positions, 1)))
            state_mem = samples_per_shard * chunk_size * state_bytes_per_sample_pos * state_memory_factor
            out_mem = samples_per_shard * chunk_size * out_per_cell * 1.25
            evidence_mem = samples_per_shard * chunk_size * 2.0  # sparse support arrays + bookkeeping estimate
            total = int(state_mem + out_mem + evidence_mem + fixed_overhead)
            if total <= usable:
                score = samples_per_shard * chunk_size
                if best is None or score > best["score"]:
                    best = {"sample_shard_count": int(shards), "samples_per_shard": samples_per_shard, "hmm_chunk_size": chunk_size, "estimated_peak_bytes": total, "score": score}
                break
    if best is None:
        shards = max(candidate_shards)
        best = {"sample_shard_count": int(shards), "samples_per_shard": int(math.ceil(n_samples / shards)), "hmm_chunk_size": int(min_chunk_size), "estimated_peak_bytes": int(usable), "score": 0}
    zarr_chunk_samples = min(512, max(64, 2 ** int(math.floor(math.log2(max(1, min(best["samples_per_shard"], 512)))))))
    zarr_chunk_positions = min(8192, max(1024, 2 ** int(math.floor(math.log2(max(1, min(best["hmm_chunk_size"], 8192)))))))
    return {
        "n_samples": n_samples,
        "n_positions": n_positions,
        "n_founders": k,
        "state_count": int(state_count),
        "target_memory_bytes": int(parse_memory_budget(target_memory)),
        "usable_memory_bytes": int(mem_budget),
        "recommendation": {
            "hmm_backend": "numba",
            "use_unordered_diploid_states": bool(use_unordered_diploid_states),
            "sample_shard_count": best["sample_shard_count"],
            "samples_per_shard": best["samples_per_shard"],
            "hmm_chunk_size": best["hmm_chunk_size"],
            "hmm_checkpoint_interval": int(checkpoint_interval),
            "output_store": "zarr",
            "output_minimal": True,
            "zarr_chunk_samples": int(zarr_chunk_samples),
            "zarr_chunk_positions": int(zarr_chunk_positions),
            "zarr_dosage_dtype": str(dosage_dtype),
            "zarr_gp_dtype": str(gp_dtype),
            "store_xi": "False",
            "write_gamma": "off",
        },
        "estimated_peak_memory_bytes": int(best["estimated_peak_bytes"]),
        "estimated_peak_memory_gib": float(best["estimated_peak_bytes"] / 1024**3),
    }


def write_autotune_plan(output_dir: str | Path, plan: dict[str, Any]) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "autotune_plan.json"
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return path
