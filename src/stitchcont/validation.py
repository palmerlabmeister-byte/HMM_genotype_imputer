from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .microarray import _normalize_chrom, _read_plink_bim, load_microarray_hardcalls_from_plink


def _read_many_parquet(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.glob("*.parquet"))
    if not files:
        files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {path}")
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def _zarr_chunk_shape(shape: tuple[int, ...], chunks: tuple[int, ...], index: tuple[int, ...]) -> tuple[int, ...]:
    out = []
    for dim, chunk, idx in zip(shape, chunks, index, strict=False):
        start = int(idx) * int(chunk)
        out.append(max(0, min(int(chunk), int(dim) - start)))
    return tuple(out)


def _read_zarr_v2_array(array_dir: Path) -> np.ndarray:
    meta = json.loads((array_dir / ".zarray").read_text(encoding="utf-8"))
    shape = tuple(int(x) for x in meta["shape"])
    chunks = tuple(int(x) for x in meta["chunks"])
    dtype = np.dtype(meta["dtype"])
    fill_value = meta.get("fill_value", 0)
    if isinstance(fill_value, str) and fill_value.lower() == "nan":
        fill = np.nan
    elif fill_value is None:
        fill = 0
    else:
        fill = fill_value
    out = np.full(shape, fill, dtype=dtype)
    ranges = [range(0, dim, chunk) for dim, chunk in zip(shape, chunks, strict=False)]
    for starts in __import__("itertools").product(*ranges):
        idx = tuple(int(start // chunk) for start, chunk in zip(starts, chunks, strict=False))
        path = array_dir / ".".join(str(i) for i in idx)
        if not path.exists():
            continue
        cshape = _zarr_chunk_shape(shape, chunks, idx)
        block = np.frombuffer(path.read_bytes(), dtype=dtype).reshape(cshape)
        slices = tuple(slice(start, start + size) for start, size in zip(starts, cshape, strict=False))
        out[slices] = block
    return out


def _dosage_matrix_from_long(dosage_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    required = {"sample_id", "position", "dosage"}
    missing = required.difference(dosage_df.columns)
    if missing:
        raise ValueError(f"Dosage parquet is missing required columns: {sorted(missing)}")
    chrom = ""
    if "chromosome" in dosage_df.columns and len(dosage_df):
        chrom = str(dosage_df["chromosome"].dropna().astype(str).iloc[0])
    sample_ids = np.asarray(sorted(dosage_df["sample_id"].astype(str).unique().tolist()), dtype=object)
    positions = np.asarray(sorted(pd.to_numeric(dosage_df["position"], errors="coerce").dropna().astype(np.int64).unique().tolist()), dtype=np.int64)
    sample_index = pd.Index(sample_ids.astype(str))
    pos_index = pd.Index(positions.astype(np.int64))
    s_codes = sample_index.get_indexer(dosage_df["sample_id"].astype(str))
    p_codes = pos_index.get_indexer(pd.to_numeric(dosage_df["position"], errors="coerce").astype(np.int64))
    mat = np.full((sample_ids.shape[0], positions.shape[0]), np.nan, dtype=np.float32)
    vals = pd.to_numeric(dosage_df["dosage"], errors="coerce").to_numpy(dtype=np.float32, copy=False)
    valid = (s_codes >= 0) & (p_codes >= 0) & np.isfinite(vals)
    mat[s_codes[valid], p_codes[valid]] = vals[valid]
    return sample_ids, positions, mat, chrom


def _dosage_matrix_from_run(run_output_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    matrix_root = run_output_dir / "matrices.zarr"
    dosage_zarr = matrix_root / "dosage"
    if dosage_zarr.exists():
        dosage = _read_zarr_v2_array(dosage_zarr).astype(np.float32, copy=False)
        sample_ids_path = matrix_root / "sample_ids.json"
        positions_path = matrix_root / "positions.npy"
        if not sample_ids_path.exists() or not positions_path.exists():
            raise FileNotFoundError("matrices.zarr is missing sample_ids.json or positions.npy")
        sample_ids = np.asarray(json.loads(sample_ids_path.read_text(encoding="utf-8")), dtype=object)
        positions = np.load(positions_path).astype(np.int64, copy=False)
        chrom = ""
        attrs = matrix_root / ".zattrs"
        if attrs.exists():
            try:
                chrom = str(json.loads(attrs.read_text(encoding="utf-8")).get("chromosome", ""))
            except Exception:
                chrom = ""
        return sample_ids, positions, dosage, chrom
    dosage_df = _read_many_parquet(run_output_dir / "dosage")
    return _dosage_matrix_from_long(dosage_df)


def _r2(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return float("nan")
    x0 = x[mask].astype(np.float64, copy=False)
    y0 = y[mask].astype(np.float64, copy=False)
    vx = float(np.var(x0))
    vy = float(np.var(y0))
    if vx <= 0.0 or vy <= 0.0:
        return float("nan")
    corr = float(np.corrcoef(x0, y0)[0, 1])
    return corr * corr


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _allele_norm(x: object) -> str:
    return str(x).strip().upper()


def _allele_comp(x: object) -> str:
    return _allele_norm(x).translate(_COMP)


def _load_run_positions_with_alleles(run_output_dir: Path, chrom: str, positions: np.ndarray) -> pd.DataFrame:
    pos_file = run_output_dir / "positions.parquet"
    if pos_file.exists():
        df = pd.read_parquet(pos_file).copy()
        cols = {str(c).upper(): c for c in df.columns}
        if "CHR" in cols:
            df = df.loc[df[cols["CHR"]].astype(str).map(_normalize_chrom) == _normalize_chrom(chrom)].copy()
        if {"POS", "REF", "ALT"}.issubset(cols):
            df = df.rename(columns={cols["POS"]: "POS", cols["REF"]: "REF", cols["ALT"]: "ALT"})
            return df[["POS", "REF", "ALT"]].drop_duplicates("POS", keep="first")
    return pd.DataFrame({"POS": positions, "REF": np.repeat("", len(positions)), "ALT": np.repeat("", len(positions))})


def _orientation_audit_and_flip(
    *,
    run_output_dir: Path,
    truth_plink: str | Path,
    chromosome: str,
    positions: np.ndarray,
    truth_dosage: np.ndarray,
    fail_on_mismatch: bool,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    run_pos = _load_run_positions_with_alleles(run_output_dir, chromosome, positions)
    run_pos["POS"] = pd.to_numeric(run_pos["POS"], errors="coerce").astype("Int64")
    run_by_pos = run_pos.dropna(subset=["POS"]).drop_duplicates("POS", keep="first").set_index("POS")
    bim = _read_plink_bim(Path(truth_plink))
    bim_chr = bim.loc[bim["chrom_norm"] == _normalize_chrom(chromosome)].copy()
    bim_chr = bim_chr.drop_duplicates("pos", keep="first").set_index("pos")
    flip = np.zeros(int(positions.shape[0]), dtype=bool)
    rows = []
    counts: dict[str, int] = {}
    for j, pos in enumerate(positions.tolist()):
        status = "missing_truth_variant"
        action = "leave_missing"
        ref = alt = a1 = a2 = ""
        if int(pos) in run_by_pos.index and int(pos) in bim_chr.index:
            ref = _allele_norm(run_by_pos.loc[int(pos), "REF"])
            alt = _allele_norm(run_by_pos.loc[int(pos), "ALT"])
            a1 = _allele_norm(bim_chr.loc[int(pos), "a1"])
            a2 = _allele_norm(bim_chr.loc[int(pos), "a2"])
            if alt and a2 == alt:
                status = "same_alt_equals_plink_a2"
                action = "none"
            elif ref and a2 == ref:
                status = "flipped_alt_equals_plink_a1"
                action = "flip_truth_dosage"
                flip[j] = True
            elif alt and _allele_comp(a2) == alt:
                status = "same_after_strand_complement"
                action = "none"
            elif ref and _allele_comp(a2) == ref:
                status = "flipped_after_strand_complement"
                action = "flip_truth_dosage"
                flip[j] = True
            else:
                status = "allele_mismatch"
                action = "mask_variant"
        elif int(pos) in run_by_pos.index:
            status = "missing_truth_variant"
        else:
            status = "missing_run_alleles"
        counts[status] = counts.get(status, 0) + 1
        rows.append({"chromosome": chromosome, "position": int(pos), "run_ref": ref, "run_alt": alt, "plink_a1": a1, "plink_a2": a2, "status": status, "action": action})
    audited = np.asarray(truth_dosage, dtype=np.float32, copy=True)
    if np.any(flip):
        audited[:, flip] = np.where(np.isfinite(audited[:, flip]), 2.0 - audited[:, flip], np.nan)
    mismatch_status = {"allele_mismatch"}
    mismatch_count = int(sum(counts.get(k, 0) for k in mismatch_status))
    if mismatch_count:
        bad = np.asarray([r["status"] in mismatch_status for r in rows], dtype=bool)
        audited[:, bad] = np.nan
    audit_df = pd.DataFrame(rows)
    audit_path = output_dir / "allele_orientation_audit.parquet"
    audit_df.to_parquet(audit_path, index=False)
    summary = {
        "enabled": True,
        "plink_bed_dosage_assumption": "dosage counts PLINK .bim A2 allele; validation flips truth when run ALT equals PLINK A1/REF",
        "status_counts": {str(k): int(v) for k, v in counts.items()},
        "n_flipped_variants": int(np.count_nonzero(flip)),
        "n_allele_mismatch_variants_masked": int(mismatch_count),
        "audit_table": str(audit_path),
    }
    if fail_on_mismatch and mismatch_count:
        raise ValueError(f"Allele-orientation audit found {mismatch_count} incompatible variants; see {audit_path}")
    return audited, summary


def validate_run_against_plink(
    *,
    run_output_dir: str | Path,
    truth_plink: str | Path,
    output_dir: str | Path,
    chromosome: str | None = None,
    min_truth_nonmissing: int = 10,
    allele_orientation_audit: bool = True,
    fail_on_allele_mismatch: bool = False,
) -> dict[str, Any]:
    run_output_dir = Path(run_output_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_ids, positions, dosage, inferred_chrom = _dosage_matrix_from_run(run_output_dir)
    chrom = str(chromosome or inferred_chrom or "")
    if not chrom:
        raise ValueError("Could not infer chromosome from dosage output; pass --chromosome.")
    positions_df = pd.DataFrame({"CHR": np.repeat(chrom, positions.shape[0]), "POS": positions})
    truth = load_microarray_hardcalls_from_plink(truth_plink, chromosome=chrom, positions_df=positions_df)
    truth_index = pd.Index(truth.sample_ids.astype(str))
    take = truth_index.get_indexer(sample_ids.astype(str))
    keep_samples = take >= 0
    if not np.any(keep_samples):
        raise ValueError("No overlapping samples between run dosage and truth PLINK .fam IDs.")
    dosage_aligned = dosage[keep_samples]
    truth_aligned = truth.dosage[take[keep_samples]]
    audit_summary: dict[str, Any] = {"enabled": False}
    if allele_orientation_audit:
        truth_aligned, audit_summary = _orientation_audit_and_flip(
            run_output_dir=run_output_dir,
            truth_plink=truth_plink,
            chromosome=chrom,
            positions=positions,
            truth_dosage=truth_aligned,
            fail_on_mismatch=bool(fail_on_allele_mismatch),
            output_dir=output_dir,
        )
    n_samples = int(dosage_aligned.shape[0])
    per_variant_rows: list[dict[str, Any]] = []
    for j, pos in enumerate(positions.tolist()):
        t = truth_aligned[:, j]
        d = dosage_aligned[:, j]
        valid = np.isfinite(t) & np.isfinite(d)
        n = int(np.sum(valid))
        truth_maf = float("nan")
        if n > 0:
            af = float(np.nanmean(t[valid]) / 2.0)
            truth_maf = float(min(max(af, 0.0), 1.0 - max(af, 0.0))) if af <= 1.0 else float("nan")
        per_variant_rows.append(
            {
                "chromosome": chrom,
                "position": int(pos),
                "n": n,
                "truth_maf": truth_maf,
                "r2": _r2(d, t) if n >= int(min_truth_nonmissing) else float("nan"),
                "mean_abs_error": float(np.nanmean(np.abs(d[valid] - t[valid]))) if n else float("nan"),
                "dosage_mean": float(np.nanmean(d[valid])) if n else float("nan"),
                "truth_mean": float(np.nanmean(t[valid])) if n else float("nan"),
            }
        )
    per_variant = pd.DataFrame(per_variant_rows)
    per_variant_path = output_dir / "per_variant_validation.parquet"
    per_variant.to_parquet(per_variant_path, index=False)
    bins = [0.0, 0.01, 0.05, 0.10, 0.20, 0.50]
    labels = ["0-1%", "1-5%", "5-10%", "10-20%", "20-50%"]
    per_variant["maf_bin"] = pd.cut(per_variant["truth_maf"], bins=bins, labels=labels, include_lowest=True, right=False)
    by_bin = (
        per_variant.groupby("maf_bin", observed=False)
        .agg(n_variants=("position", "count"), mean_r2=("r2", "mean"), median_r2=("r2", "median"), mean_abs_error=("mean_abs_error", "mean"))
        .reset_index()
    )
    by_bin_path = output_dir / "validation_by_maf_bin.parquet"
    by_bin.to_parquet(by_bin_path, index=False)
    valid = np.isfinite(truth_aligned) & np.isfinite(dosage_aligned)
    overall_r2 = _r2(dosage_aligned.reshape(-1), truth_aligned.reshape(-1))
    summary: dict[str, Any] = {
        "run_output_dir": str(run_output_dir),
        "truth_plink": str(truth_plink),
        "chromosome": chrom,
        "n_run_samples": int(sample_ids.shape[0]),
        "n_truth_samples": int(truth.sample_ids.shape[0]),
        "n_overlap_samples": n_samples,
        "n_positions": int(positions.shape[0]),
        "truth_loaded_variants": int(truth.n_loaded_variants),
        "truth_target_variants": int(truth.n_target_variants),
        "n_valid_cells": int(np.sum(valid)),
        "cell_missing_rate": float(1.0 - np.mean(valid)) if valid.size else float("nan"),
        "overall_cellwise_r2": overall_r2,
        "mean_per_variant_r2": float(np.nanmean(per_variant["r2"].to_numpy(dtype=np.float64))) if len(per_variant) else float("nan"),
        "median_per_variant_r2": float(np.nanmedian(per_variant["r2"].to_numpy(dtype=np.float64))) if len(per_variant) else float("nan"),
        "allele_orientation_audit": audit_summary,
        "per_variant_validation": str(per_variant_path),
        "validation_by_maf_bin": str(by_bin_path),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
