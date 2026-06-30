from __future__ import annotations

import itertools
import json
from pathlib import Path
import shutil
from typing import Iterable
import warnings

import numpy as np
import numcodecs
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pysam

# Default Zarr chunk caps for the dense float matrices.  Without a cap a
# non-dask DataArray would be written as a single whole-array chunk, which at
# 50k samples x 100k variants is a ~20 TB uncompressed serialization in one
# allocation.  These match the pipeline's zarr_chunk_samples/zarr_chunk_positions
# defaults.
_DEFAULT_ZARR_CHUNK_SAMPLES = 256
_DEFAULT_ZARR_CHUNK_POSITIONS = 4096


DEFAULT_DATASETS = (
    "dosage",
    "support_mask",
    "recombination",
    "transitions",
    "founder_updates",
    "haplotype_probabilities",
    "genotype_posteriors",
    "genotype_calls",
)
def _gt_tuple_from_alt_count(alt_count: int, ploidy: int) -> tuple[int | None, ...]:
    if int(ploidy) <= 0 or int(alt_count) < 0:
        return tuple([None] * max(int(ploidy), 2))
    alt = int(np.clip(int(alt_count), 0, int(ploidy)))
    return tuple([0] * (int(ploidy) - alt) + [1] * alt)


def combine_parquet_chunks(
    input_dir: str | Path,
    output_file: str | Path,
    *,
    columns: list[str] | None = None,
    compression: str = "zstd",
    compression_level: int = 6,
    row_group_size: int = 262_144,
) -> dict[str, object]:
    input_dir = Path(input_dir)
    output_file = Path(output_file)
    dataset = ds.dataset(str(input_dir), format="parquet")
    scanner = dataset.scanner(columns=columns, use_threads=True, batch_size=row_group_size)
    schema = scanner.projected_schema
    output_file.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(
        str(output_file),
        schema=schema,
        compression=compression,
        compression_level=compression_level,
        use_dictionary=True,
        data_page_version="2.0",
        write_statistics=True,
    )
    rows = 0
    batches = 0
    try:
        for batch in scanner.to_batches():
            writer.write_batch(batch)
            rows += int(batch.num_rows)
            batches += 1
    finally:
        writer.close()

    return {
        "input_dir": str(input_dir),
        "output_file": str(output_file),
        "rows": rows,
        "batches": batches,
        "columns": list(schema.names),
    }


def _require_xarray_dask():
    try:
        import dask.array as da
        import xarray as xr
        from dask import delayed
    except Exception as exc:  # pragma: no cover - optional runtime deps.
        raise ImportError(
            "combine_pipeline_outputs now returns a lazy xarray Dataset and requires "
            "'xarray' and 'dask'. Install them in your conda env to use this function."
        ) from exc
    return xr, da, delayed


def _arrow_primitive_to_numpy_dtype(type_: pa.DataType) -> np.dtype:
    if pa.types.is_boolean(type_):
        return np.dtype(np.bool_)
    if pa.types.is_integer(type_) or pa.types.is_floating(type_):
        return np.dtype(type_.to_pandas_dtype())
    if pa.types.is_timestamp(type_):
        return np.dtype("datetime64[ns]")
    return np.dtype(object)


def _read_row_group_column(path: str, row_group_idx: int, column_name: str, dtype_str: str) -> np.ndarray:
    dtype = np.dtype(dtype_str)
    parquet = pq.ParquetFile(path, memory_map=True)
    table = parquet.read_row_group(row_group_idx, columns=[column_name], use_threads=False)
    arr = table[column_name].combine_chunks()
    if pa.types.is_fixed_size_list(arr.type):
        values = np.asarray(arr.values)
        list_size = int(arr.type.list_size)
        return values.reshape(arr.length(), list_size).astype(dtype, copy=False)
    if pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
        return np.asarray(arr.to_pandas(), dtype=object)
    return np.asarray(arr).astype(dtype, copy=False)


def _lazy_array_from_parquet_column(path: Path, column: pa.Field):
    xr, da, delayed = _require_xarray_dask()
    parquet = pq.ParquetFile(path)
    num_row_groups = parquet.num_row_groups
    chunks = []
    dtype = _arrow_primitive_to_numpy_dtype(column.type)
    if pa.types.is_fixed_size_list(column.type):
        value_dtype = _arrow_primitive_to_numpy_dtype(column.type.value_type)
        list_size = int(column.type.list_size)
        for i in range(num_row_groups):
            n_rows = int(parquet.metadata.row_group(i).num_rows)
            block = delayed(_read_row_group_column)(str(path), i, column.name, value_dtype.str)
            chunks.append(da.from_delayed(block, shape=(n_rows, list_size), dtype=value_dtype))
        return da.concatenate(chunks, axis=0), 2, value_dtype
    for i in range(num_row_groups):
        n_rows = int(parquet.metadata.row_group(i).num_rows)
        block = delayed(_read_row_group_column)(str(path), i, column.name, dtype.str)
        chunks.append(da.from_delayed(block, shape=(n_rows,), dtype=dtype))
    return da.concatenate(chunks, axis=0), 1, dtype


def combine_pipeline_outputs(
    run_output_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    datasets: Iterable[str] = DEFAULT_DATASETS,
    compression: str = "zstd",
    compression_level: int = 6,
    row_group_size: int = 262_144,
) -> object:
    xr, _, _ = _require_xarray_dask()
    run_output_dir = Path(run_output_dir)
    out_dir = run_output_dir / "combined" if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, object]] = {}
    data_vars = {}
    for dataset_name in datasets:
        source = run_output_dir / dataset_name
        if not source.exists():
            continue
        target = out_dir / f"{dataset_name}.parquet"
        summary[dataset_name] = combine_parquet_chunks(
            source,
            target,
            compression=compression,
            compression_level=compression_level,
            row_group_size=row_group_size,
        )
        parquet = pq.ParquetFile(target)
        row_dim = f"{dataset_name}_row"
        for field in parquet.schema_arrow:
            lazy_arr, ndim, dtype = _lazy_array_from_parquet_column(target, field)
            var_name = f"{dataset_name}__{field.name}"
            if ndim == 1:
                dims = (row_dim,)
            else:
                dims = (row_dim, f"{dataset_name}_{field.name}_field")
            data_vars[var_name] = xr.DataArray(
                lazy_arr,
                dims=dims,
                attrs={
                    "source_dataset": dataset_name,
                    "source_column": field.name,
                    "arrow_type": str(field.type),
                    "dtype": str(dtype),
                    "combined_file": str(target),
                },
            )
    out = xr.Dataset(data_vars=data_vars)
    out.attrs["combine_summary"] = summary
    out.attrs["run_output_dir"] = str(run_output_dir)
    out.attrs["combined_output_dir"] = str(out_dir)
    out.attrs["lazy_backend"] = "dask"
    return out


def write_xarray_float_zarr(
    xds: object,
    store: str | Path,
    *,
    mode: str = "w",
    consolidated: bool = False,
) -> dict[str, object]:
    """Persist floating STITCHCONT outputs as an xarray-compatible Zarr v2 store.

    Hard calls, IDs, positions, and masks are intentionally left in Parquet,
    where dictionary/RLE/bit-packing encodings are a better fit.
    """
    store = Path(store)
    if store.exists() and mode == "w":
        shutil.rmtree(store)
    store.mkdir(parents=True, exist_ok=True)
    float_vars = [
        name
        for name, data_array in xds.data_vars.items()
        if np.issubdtype(data_array.dtype, np.floating)
    ]
    if not float_vars:
        raise ValueError("No floating-point variables are available for Zarr output.")
    root_attrs = _json_safe_attrs(getattr(xds, "attrs", {}))
    root_attrs["zarr_content"] = "floating STITCHCONT outputs only; hard calls and labels remain in Parquet"
    (store / ".zgroup").write_text(json.dumps({"zarr_format": 2}, indent=2), encoding="utf-8")
    (store / ".zattrs").write_text(json.dumps(root_attrs, indent=2), encoding="utf-8")
    if consolidated:
        warnings.warn(
            "Manual STITCHCONT Zarr writer emits xarray-compatible Zarr v2 metadata without consolidated .zmetadata.",
            RuntimeWarning,
            stacklevel=2,
        )
    for name in float_vars:
        _write_float_dataarray_zarr(xds[name], store / name)
    return {
        "output_zarr": str(store),
        "mode": mode,
        "consolidated": False,
        "n_variables": int(len(float_vars)),
        "variables": sorted(float_vars),
    }


def _json_safe_attrs(attrs: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(dict(attrs), default=str))


def _default_chunks(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Bounded chunk shape for a dense float matrix.

    Caps axis 0 (samples) and axis 1 (positions); any further axes (genotype
    classes, founders) are kept whole because they are small.  This guarantees
    we never emit a single whole-array chunk for an N x M matrix.
    """
    caps = (_DEFAULT_ZARR_CHUNK_SAMPLES, _DEFAULT_ZARR_CHUNK_POSITIONS)
    chunks: list[int] = []
    for axis, size in enumerate(shape):
        size = max(1, int(size))
        cap = caps[axis] if axis < len(caps) else size
        chunks.append(max(1, min(cap, size)))
    return tuple(chunks)


def _chunks_from_data(data: object, shape: tuple[int, ...]) -> tuple[int, ...]:
    chunks_attr = getattr(data, "chunks", None)
    if not chunks_attr:
        # Non-dask array: never fall back to a whole-array chunk.
        return _default_chunks(shape)
    chunks: list[int] = []
    for axis_chunks, size in zip(chunks_attr, shape):
        if axis_chunks:
            chunks.append(max(1, int(axis_chunks[0])))
        else:
            chunks.append(max(1, int(size)))
    return tuple(chunks)


def _write_float_dataarray_zarr(data_array: object, array_dir: Path) -> None:
    array_dir.mkdir(parents=True, exist_ok=True)
    data = data_array.data
    shape = tuple(int(size) for size in data_array.shape)
    chunks = _chunks_from_data(data, shape)
    dtype = np.dtype(data_array.dtype)
    # Compress the dense float matrices.  An uncompressed dosage matrix at
    # 50k x 100k float32 is 20 TB on disk; zstd typically shrinks low-entropy
    # dosage/posterior data by an order of magnitude or more.
    compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
    metadata = {
        "zarr_format": 2,
        "shape": list(shape),
        "chunks": list(chunks),
        "dtype": dtype.str,
        "compressor": compressor.get_config(),
        "fill_value": None,
        "order": "C",
        "filters": None,
        "dimension_separator": ".",
    }
    (array_dir / ".zarray").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    attrs = _json_safe_attrs(getattr(data_array, "attrs", {}))
    attrs["_ARRAY_DIMENSIONS"] = list(data_array.dims)
    (array_dir / ".zattrs").write_text(json.dumps(attrs, indent=2), encoding="utf-8")
    starts_by_axis = [range(0, dim, chunk) for dim, chunk in zip(shape, chunks)]
    for starts in itertools.product(*starts_by_axis):
        slices = tuple(slice(start, min(start + chunk, dim)) for start, chunk, dim in zip(starts, chunks, shape))
        chunk_index = ".".join(str(start // chunk) for start, chunk in zip(starts, chunks))
        block = data[slices]
        if hasattr(block, "compute"):
            block = block.compute(scheduler="synchronous")
        block_array = np.asarray(block, dtype=dtype)
        # Zarr v2 reads edge chunks using the declared chunk shape, so partial
        # edge chunks must be padded to the full chunk shape before encoding.
        if block_array.shape != chunks:
            padded = np.zeros(chunks, dtype=dtype)
            slot = tuple(slice(0, s) for s in block_array.shape)
            padded[slot] = block_array
            block_array = padded
        raw = np.ascontiguousarray(block_array).tobytes(order="C")
        (array_dir / chunk_index).write_bytes(compressor.encode(raw))


def _fixed_size_list_to_matrix(column: pa.ChunkedArray) -> np.ndarray:
    arr = column.combine_chunks()
    if not pa.types.is_fixed_size_list(arr.type):
        raise TypeError(f"Expected fixed-size list array, got {arr.type}")
    list_size = int(arr.type.list_size)
    values = np.asarray(arr.values).astype(np.float32, copy=False)
    return values.reshape(len(arr), list_size)


def export_stitch_vcf_from_parquet(
    run_output_dir: str | Path,
    output_vcf: str | Path,
    *,
    chromosome: str,
    include_haplotype_dosage: bool = False,
    gp_precision: int = 4,
    ds_precision: int = 4,
    hd_precision: int = 4,
    bgzip: bool = True,
    tabix_index: bool = True,
    threads: int = 1,
) -> dict[str, object]:
    raise RuntimeError(
        "VCF export is intentionally disabled in STITCHCONT. "
        "Use native Parquet/Zarr outputs for primary results; use export-bcf only when an external tool explicitly requires BCF."
    )


def export_stitch_bcf_from_parquet(
    run_output_dir: str | Path,
    output_bcf: str | Path,
    *,
    chromosome: str,
    include_haplotype_dosage: bool = False,
    tabix_index: bool = True,
) -> dict[str, object]:
    warnings.warn(
        "BCF export is an interoperability path and will slow down I/O; keep primary STITCHCONT results in Parquet/Zarr.",
        RuntimeWarning,
        stacklevel=2,
    )
    run_output_dir = Path(run_output_dir)
    output_bcf = Path(output_bcf)
    samples = pd.read_parquet(run_output_dir / "samples.parquet")
    positions_df = pd.read_parquet(run_output_dir / "positions.parquet")
    positions_df = positions_df.loc[positions_df["CHR"].astype(str) == str(chromosome)].copy()
    positions_df = positions_df.sort_values("POS").reset_index(drop=True)
    if positions_df.empty:
        raise ValueError(f"No positions found for chromosome '{chromosome}'.")

    sample_ids = samples["sample_id"].astype(str).to_numpy()
    n_samples = int(sample_ids.size)
    if n_samples == 0:
        raise ValueError("No samples found in samples.parquet")
    if "ploidy" in samples.columns:
        sample_ploidy = samples["ploidy"].to_numpy(dtype=np.int16, copy=False)
    else:
        sample_ploidy = np.full(n_samples, 2, dtype=np.int16)
    max_ploidy = max(int(np.max(sample_ploidy)) if sample_ploidy.size else 2, 0)

    dosage_dir = run_output_dir / "dosage"
    dosage_blocks = sorted(dosage_dir.glob("block=*.parquet"))
    if not dosage_blocks:
        raise FileNotFoundError(f"No dosage block parquet files in {dosage_dir}")

    hap_dir = run_output_dir / "haplotype_probabilities"
    if include_haplotype_dosage and not hap_dir.exists():
        raise FileNotFoundError(
            "include_haplotype_dosage=True requested but haplotype_probabilities/ is missing."
        )
    gp_dir = run_output_dir / "genotype_posteriors"
    gt_dir = run_output_dir / "genotype_calls"
    use_gp = gp_dir.exists()
    use_gt = gt_dir.exists()

    output_bcf.parent.mkdir(parents=True, exist_ok=True)
    header = pysam.VariantHeader()
    header.add_meta("fileformat", value="VCFv4.2")
    header.contigs.add(str(chromosome), length=int(positions_df["POS"].max()))
    header.formats.add("GT", "1", "String", "Genotype call from posterior")
    header.formats.add("GP", "G", "Float", "Genotype posterior probabilities")
    header.formats.add("DS", "1", "Float", "Dosage")
    if include_haplotype_dosage:
        header.formats.add("HD", ".", "Float", "Ancestral haplotype dosages")
    for sample_id in sample_ids.tolist():
        header.add_sample(sample_id)

    line_count = 0
    pos_cursor = 0
    with pysam.VariantFile(str(output_bcf), mode="wb", header=header) as bcf:
        for dosage_block in dosage_blocks:
            dosage_table = pq.read_table(dosage_block, columns=["dosage"])
            dosage = np.asarray(dosage_table["dosage"].combine_chunks()).astype(np.float32, copy=False)
            n_rows = int(dosage.shape[0])
            if n_rows % n_samples != 0:
                raise ValueError(f"Rows in {dosage_block} are not divisible by n_samples={n_samples}")
            n_positions_block = n_rows // n_samples
            dosage_matrix = dosage.reshape(n_samples, n_positions_block)

            block_positions = positions_df.iloc[pos_cursor : pos_cursor + n_positions_block]
            if len(block_positions) != n_positions_block:
                raise ValueError("Position cursor exceeded available positions while exporting BCF.")
            block_pos = block_positions["POS"].to_numpy(dtype=np.int64)
            block_ref = block_positions["REF"].astype(str).to_numpy()
            block_alt = block_positions["ALT"].astype(str).to_numpy()
            pos_cursor += n_positions_block

            hap_matrix = None
            if include_haplotype_dosage:
                hap_block = hap_dir / dosage_block.name
                if not hap_block.exists():
                    raise FileNotFoundError(f"Missing haplotype block for {dosage_block.name}")
                hap_table = pq.read_table(hap_block, columns=["hap_dosage"])
                hap_rows = _fixed_size_list_to_matrix(hap_table["hap_dosage"])
                if hap_rows.shape[0] != n_rows:
                    raise ValueError(f"hap_dosage row mismatch for {hap_block}")
                hap_matrix = hap_rows.reshape(n_samples, n_positions_block, hap_rows.shape[1]).astype(np.float32, copy=False)

            gp_matrix = None
            if use_gp:
                gp_block = gp_dir / dosage_block.name
                if gp_block.exists():
                    gp_table = pq.read_table(gp_block, columns=["genotype_posterior"])
                    gp_rows = _fixed_size_list_to_matrix(gp_table["genotype_posterior"])
                    if gp_rows.shape[0] == n_rows:
                        gp_matrix = gp_rows.reshape(n_samples, n_positions_block, gp_rows.shape[1]).astype(np.float32, copy=False)

            gt_matrix = None
            if use_gt:
                gt_block = gt_dir / dosage_block.name
                if gt_block.exists():
                    gt_table = pq.read_table(gt_block, columns=["genotype_call"])
                    gt_rows = np.asarray(gt_table["genotype_call"].combine_chunks()).astype(np.int8, copy=False)
                    if gt_rows.shape[0] == n_rows:
                        gt_matrix = gt_rows.reshape(n_samples, n_positions_block)

            for pos_idx in range(n_positions_block):
                ds_vec = dosage_matrix[:, pos_idx]
                if gp_matrix is None:
                    geno_axis = np.arange(max_ploidy + 1, dtype=np.float32)
                    logits = -((ds_vec[:, None].astype(np.float32, copy=False) - geno_axis[None, :]) ** 2) / 0.35
                    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    gp_vec = (exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)).astype(np.float32, copy=False)
                else:
                    gp_vec = gp_matrix[:, pos_idx, :].astype(np.float32, copy=False)
                    gp_vec = np.where(np.isfinite(gp_vec), gp_vec, 0.0)
                    gp_vec = np.clip(gp_vec, 0.0, 1.0)
                    gp_vec /= np.clip(np.sum(gp_vec, axis=1, keepdims=True), 1e-12, None)
                if gt_matrix is None:
                    gt_idx = np.argmax(gp_vec, axis=1).astype(np.int8, copy=False)
                else:
                    gt_idx = gt_matrix[:, pos_idx].astype(np.int8, copy=False)
                hd_vec = None if hap_matrix is None else hap_matrix[:, pos_idx, :]

                record = bcf.new_record(
                    contig=str(chromosome),
                    start=int(block_pos[pos_idx]) - 1,
                    stop=int(block_pos[pos_idx]),
                    alleles=(str(block_ref[pos_idx]), str(block_alt[pos_idx])),
                )
                for sample_idx, sample_id in enumerate(sample_ids.tolist()):
                    sample = record.samples[sample_id]
                    gt_sample = int(gt_idx[sample_idx])
                    sample["GT"] = _gt_tuple_from_alt_count(gt_sample, int(sample_ploidy[sample_idx]))
                    ploidy_s = int(sample_ploidy[sample_idx])
                    n_gp = min(gp_vec.shape[1], max(ploidy_s, 0) + 1) if ploidy_s > 0 else gp_vec.shape[1]
                    sample["GP"] = tuple(float(value) for value in gp_vec[sample_idx, :n_gp])
                    sample["DS"] = float(ds_vec[sample_idx])
                    if hd_vec is not None:
                        sample["HD"] = tuple(float(v) for v in hd_vec[sample_idx])
                bcf.write(record)
                line_count += 1

    if tabix_index:
        pysam.index(str(output_bcf), force=True)

    return {
        "output_bcf": str(output_bcf),
        "n_samples": n_samples,
        "n_sites": line_count,
        "include_haplotype_dosage": bool(include_haplotype_dosage),
        "tabix_index": bool(tabix_index),
    }
