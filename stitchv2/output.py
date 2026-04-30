from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pysam


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
GT_LUT = np.asarray(["0/0", "0/1", "1/1"], dtype="<U3")
GT_TUPLES = ((0, 0), (0, 1), (1, 1))


def _gt_string_from_alt_count(alt_count: int, ploidy: int) -> str:
    if int(ploidy) <= 0 or int(alt_count) < 0:
        return "./."
    alt = int(np.clip(int(alt_count), 0, int(ploidy)))
    alleles = ["0"] * (int(ploidy) - alt) + ["1"] * alt
    return "/".join(alleles)


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


def _fixed_size_list_to_matrix(column: pa.ChunkedArray) -> np.ndarray:
    arr = column.combine_chunks()
    if not pa.types.is_fixed_size_list(arr.type):
        raise TypeError(f"Expected fixed-size list array, got {arr.type}")
    list_size = int(arr.type.list_size)
    values = np.asarray(arr.values).astype(np.float32, copy=False)
    return values.reshape(len(arr), list_size)


@contextmanager
def _vcf_writer(path: Path, bgzip: bool):
    if bgzip:
        sink = pysam.BGZFile(str(path), mode="wb")

        def write_line(text: str) -> None:
            sink.write(text.encode("utf-8"))

        try:
            yield write_line
        finally:
            sink.close()
        return

    sink = path.open("w", encoding="utf-8")
    try:
        yield sink.write
    finally:
        sink.close()


def _build_sample_fields(
    ds_vec: np.ndarray,
    *,
    gp_fmt: str,
    ds_fmt: str,
    hd_fmt: str,
    hd_vec: np.ndarray | None,
    gp_vec: np.ndarray | None = None,
    gt_idx: np.ndarray | None = None,
    ploidy_vec: np.ndarray | None = None,
) -> list[str]:
    if ploidy_vec is None:
        inferred = 2 if gp_vec is None else max(int(gp_vec.shape[1]) - 1, 1)
        ploidy_vec = np.full(ds_vec.shape[0], inferred, dtype=np.int16)
    ploidy_i = ploidy_vec.astype(np.int16, copy=False)
    max_ploidy = max(int(np.nanmax(ploidy_i)) if ploidy_i.size else 2, 0)
    if gp_vec is None:
        geno_axis = np.arange(max_ploidy + 1, dtype=np.float32)
        temp = 0.35
        logits = -((ds_vec[:, None].astype(np.float32, copy=False) - geno_axis[None, :]) ** 2) / temp
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        gp_stack = (exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-12, None)).astype(np.float32, copy=False)
    else:
        gp_stack = gp_vec.astype(np.float32, copy=False)
        gp_stack = np.where(np.isfinite(gp_stack), gp_stack, 0.0)
        gp_stack = np.clip(gp_stack, 0.0, 1.0)
        gp_stack /= np.clip(np.sum(gp_stack, axis=1, keepdims=True), 1e-12, None)
    if gt_idx is None:
        gt_idx = np.argmax(gp_stack, axis=1).astype(np.int8, copy=False)
    gt_idx_i8 = gt_idx.astype(np.int8, copy=False)
    ds_s = np.char.mod(ds_fmt, ds_vec)
    out: list[str] = []
    for sample_idx in range(ds_vec.shape[0]):
        ploidy_s = int(ploidy_i[sample_idx])
        gt = _gt_string_from_alt_count(int(gt_idx_i8[sample_idx]), ploidy_s)
        n_gp = min(gp_stack.shape[1], max(ploidy_s, 0) + 1) if ploidy_s > 0 else gp_stack.shape[1]
        gp_s = ",".join(gp_fmt % float(value) for value in gp_stack[sample_idx, :n_gp])
        field = f"{gt}:{gp_s}:{ds_s[sample_idx]}"
        if hd_vec is not None:
            hd_s = ",".join(hd_fmt % float(value) for value in hd_vec[sample_idx])
            field = f"{field}:{hd_s}"
        out.append(field)
    return out


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
    run_output_dir = Path(run_output_dir)
    output_vcf = Path(output_vcf)
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

    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    gp_fmt = f"%.{int(gp_precision)}f"
    ds_fmt = f"%.{int(ds_precision)}f"
    hd_fmt = f"%.{int(hd_precision)}f"
    format_key = "GT:GP:DS:HD" if include_haplotype_dosage else "GT:GP:DS"
    line_count = 0
    pos_cursor = 0

    with _vcf_writer(output_vcf, bgzip=bgzip) as write_line:
        write_line("##fileformat=VCFv4.2\n")
        write_line(f"##contig=<ID={chromosome},length={int(positions_df['POS'].max())}>\n")
        write_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype call from posterior">\n')
        write_line('##FORMAT=<ID=GP,Number=G,Type=Float,Description="Genotype posterior probabilities">\n')
        write_line('##FORMAT=<ID=DS,Number=1,Type=Float,Description="Dosage">\n')
        if include_haplotype_dosage:
            write_line('##FORMAT=<ID=HD,Number=.,Type=Float,Description="Ancestral haplotype dosages">\n')
        header = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sample_ids.tolist()) + "\n"
        write_line(header)

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
                raise ValueError("Position cursor exceeded available positions while exporting VCF.")
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

            def build_line(pos_idx: int) -> str:
                ds_vec = dosage_matrix[:, pos_idx]
                hd_vec = None if hap_matrix is None else hap_matrix[:, pos_idx, :]
                gp_vec = None if gp_matrix is None else gp_matrix[:, pos_idx, :]
                gt_vec = None if gt_matrix is None else gt_matrix[:, pos_idx]
                sample_fields = _build_sample_fields(
                    ds_vec,
                    gp_fmt=gp_fmt,
                    ds_fmt=ds_fmt,
                    hd_fmt=hd_fmt,
                    hd_vec=hd_vec,
                    gp_vec=gp_vec,
                    gt_idx=gt_vec,
                    ploidy_vec=sample_ploidy,
                )
                return (
                    f"{chromosome}\t{int(block_pos[pos_idx])}\t.\t{block_ref[pos_idx]}\t{block_alt[pos_idx]}"
                    f"\t.\tPASS\t.\t{format_key}\t"
                    + "\t".join(sample_fields)
                    + "\n"
                )

            if threads > 1 and n_positions_block > 1:
                buffer: list[str] = []
                with ThreadPoolExecutor(max_workers=min(int(threads), n_positions_block)) as executor:
                    for line in executor.map(build_line, range(n_positions_block)):
                        buffer.append(line)
                        if len(buffer) >= 1024:
                            write_line("".join(buffer))
                            buffer.clear()
                        line_count += 1
                if buffer:
                    write_line("".join(buffer))
            else:
                buffer = []
                for pos_idx in range(n_positions_block):
                    buffer.append(build_line(pos_idx))
                    if len(buffer) >= 1024:
                        write_line("".join(buffer))
                        buffer.clear()
                    line_count += 1
                if buffer:
                    write_line("".join(buffer))

    if tabix_index and bgzip:
        pysam.tabix_index(str(output_vcf), preset="vcf", force=True)

    return {
        "output_vcf": str(output_vcf),
        "n_samples": n_samples,
        "n_sites": line_count,
        "include_haplotype_dosage": bool(include_haplotype_dosage),
        "bgzip": bool(bgzip),
        "tabix_index": bool(tabix_index and bgzip),
    }


def export_stitch_bcf_from_parquet(
    run_output_dir: str | Path,
    output_bcf: str | Path,
    *,
    chromosome: str,
    include_haplotype_dosage: bool = False,
    tabix_index: bool = True,
) -> dict[str, object]:
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
