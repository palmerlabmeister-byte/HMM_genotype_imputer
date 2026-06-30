from __future__ import annotations

import json
from pathlib import Path
import zlib

import numpy as np

from stitchcont.shards import merge_sharded_matrix_zarr


def _write_array(root: Path, name: str, data: np.ndarray, chunks: tuple[int, ...]) -> None:
    arr = root / name
    arr.mkdir(parents=True, exist_ok=True)
    zarray = {
        "zarr_format": 2,
        "shape": list(data.shape),
        "chunks": list(chunks),
        "dtype": data.dtype.str,
        "compressor": {"id": "zlib", "level": 1},
        "fill_value": 0,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
    }
    (arr / ".zarray").write_text(json.dumps(zarray), encoding="utf-8")
    (arr / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["sample", "position"]}), encoding="utf-8")
    # one chunk per shard for this test
    (arr / "0.0").write_bytes(zlib.compress(np.ascontiguousarray(data).tobytes(), level=1))


def test_merge_sharded_matrix_zarr_preserves_zlib_chunks(tmp_path: Path) -> None:
    for shard_idx, block in enumerate([np.array([[1, 2, 3]], dtype=np.float32), np.array([[4, 5, 6]], dtype=np.float32)]):
        root = tmp_path / f"shard_{shard_idx:04d}" / "matrices.zarr"
        root.mkdir(parents=True)
        (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}), encoding="utf-8")
        (root / ".zattrs").write_text(json.dumps({}), encoding="utf-8")
        (root / "sample_ids.json").write_text(json.dumps([f"s{shard_idx}"]), encoding="utf-8")
        np.save(root / "positions.npy", np.array([10, 20, 30], dtype=np.int64))
        _write_array(root, "dosage", block, chunks=(1, 3))
    summary = merge_sharded_matrix_zarr(input_root=tmp_path, output_dir=tmp_path / "merged")
    assert summary["n_samples"] == 2
    arr = tmp_path / "merged" / "matrices.zarr" / "dosage"
    z = json.loads((arr / ".zarray").read_text())
    raw0 = zlib.decompress((arr / "0.0").read_bytes())
    raw1 = zlib.decompress((arr / "1.0").read_bytes())
    out = np.vstack([np.frombuffer(raw0, dtype=np.float32).reshape(1, 3), np.frombuffer(raw1, dtype=np.float32).reshape(1, 3)])
    np.testing.assert_allclose(out, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))
    assert z["compressor"]["id"] == "zlib"
