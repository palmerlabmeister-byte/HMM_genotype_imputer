from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .pedigree import coerce_pedigree_graph, smooth_dosage_with_pedigree


def _read_samples(path: str | Path):
    import pandas as pd
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, sep=None, engine="python")


def _open_zarr_array(root: Path, name: str, mode: str = "r"):
    try:
        import zarr
    except Exception as exc:  # pragma: no cover
        raise ImportError("pedigree-postprocess requires zarr. Install package dependencies with pip install -e .") from exc
    return zarr.open_array(str(root / name), mode=mode)


def run_pedigree_postprocess_zarr(
    *,
    matrices_zarr: str | Path,
    samples: str | Path,
    pedigree: str | Path | None,
    output_dir: str | Path,
    strength: float = 0.10,
    chunk_positions: int = 4096,
    offspring_col: str = "sample_id",
    parent1_col: str = "father_id",
    parent2_col: str = "mother_id",
) -> dict[str, Any]:
    """Apply pedigree dosage smoothing to a matrix Zarr store chunkwise.

    This is a post-HMM CPU shard/merged-output helper. It reads only dosage
    chunks, applies the same parent-average smoothing used inside the pipeline,
    and writes a new `dosage_pedigree_smoothed` array in a new matrices.zarr.
    """
    matrices_root = Path(matrices_zarr)
    if matrices_root.name != "matrices.zarr":
        matrices_root = matrices_root / "matrices.zarr"
    out = Path(output_dir)
    out_root = out / "matrices.zarr"
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / ".zgroup").write_text(json.dumps({"zarr_format": 2}, sort_keys=True), encoding="utf-8")

    samples_df = _read_samples(samples)
    if (matrices_root / "sample_ids.json").exists():
        sample_ids = np.asarray(json.loads((matrices_root / "sample_ids.json").read_text(encoding="utf-8")), dtype=object)
    elif "sample_id" in samples_df.columns:
        sample_ids = samples_df["sample_id"].astype(str).to_numpy(dtype=object)
    else:
        sample_ids = samples_df.iloc[:, 0].astype(str).to_numpy(dtype=object)
    ped_source = pedigree if pedigree is not None else samples_df
    graph = coerce_pedigree_graph(
        ped_source,
        sample_ids,
        offspring_col=offspring_col,
        parent1_col=parent1_col,
        parent2_col=parent2_col,
    )
    if graph is None:
        raise ValueError("No usable pedigree graph could be built from supplied samples/pedigree.")

    dosage = _open_zarr_array(matrices_root, "dosage", mode="r")
    try:
        import zarr
    except Exception as exc:  # pragma: no cover
        raise ImportError("pedigree-postprocess requires zarr") from exc
    out_arr = zarr.open_array(
        str(out_root / "dosage_pedigree_smoothed"),
        mode="w",
        shape=dosage.shape,
        chunks=dosage.chunks,
        dtype=dosage.dtype,
        fill_value=np.nan,
    )
    n_samples, n_positions = int(dosage.shape[0]), int(dosage.shape[1])
    if n_samples != int(sample_ids.shape[0]):
        raise ValueError(f"dosage sample axis {n_samples} != sample_ids {sample_ids.shape[0]}")
    chunk_positions = max(int(chunk_positions), 1)
    for start in range(0, n_positions, chunk_positions):
        stop = min(start + chunk_positions, n_positions)
        block = np.asarray(dosage[:, start:stop], dtype=np.float32)
        smoothed = smooth_dosage_with_pedigree(block, graph, float(strength)).astype(dosage.dtype, copy=False)
        out_arr[:, start:stop] = smoothed
    if (matrices_root / "positions.npy").exists():
        import shutil
        shutil.copy2(matrices_root / "positions.npy", out_root / "positions.npy")
    (out_root / "sample_ids.json").write_text(json.dumps([str(x) for x in sample_ids.tolist()], indent=2), encoding="utf-8")
    summary = {
        "status": "ok",
        "input_matrices_zarr": str(matrices_root),
        "output_matrices_zarr": str(out_root),
        "array": "dosage_pedigree_smoothed",
        "strength": float(strength),
        "n_samples": n_samples,
        "n_positions": n_positions,
        "pedigree": graph.summary(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "pedigree_postprocess_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
