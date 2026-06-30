from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .calibration import (
    compute_hwe_features_from_posterior,
    compute_info_score_per_variant,
    posterior_confidence_metrics,
)


DEFAULT_DIAGNOSTIC_THRESHOLDS: dict[str, float] = {
    "warn_het_rate": 0.98,
    "fail_het_rate": 0.995,
    "warn_hom_rate": 0.995,
    "fail_hom_rate": 0.999,
    "warn_missing_rate": 0.50,
    "fail_missing_rate": 0.90,
    "warn_low_info": 0.05,
    "fail_low_info": -0.25,
    "warn_low_max_gp": 0.60,
    "fail_low_max_gp": 0.52,
    "warn_low_founder_std": 0.005,
    "fail_low_founder_std": 1e-5,
}


def _as_float_matrix(value: np.ndarray | None, shape: tuple[int, int], fill: float = np.nan) -> np.ndarray:
    if value is None:
        return np.full(shape, fill, dtype=np.float32)
    return value.astype(np.float32, copy=False)


def _posterior_dosage(posterior: np.ndarray) -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    axis = np.arange(gp.shape[2], dtype=np.float32)
    return np.sum(gp * axis[None, None, :], axis=2).astype(np.float32, copy=False)


def _safe_nanmean(value: np.ndarray, axis: int = 0) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(value, axis=axis).astype(np.float32, copy=False)


def compute_variant_diagnostics(
    *,
    block_id: int,
    chromosome: str,
    positions: np.ndarray,
    dosage: np.ndarray,
    raw_posterior: np.ndarray | None = None,
    calibrated_posterior: np.ndarray | None = None,
    genotype_call: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    sample_ploidy: np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute per-variant QC/model diagnostics for one pipeline block."""

    ds_matrix = dosage.astype(np.float32, copy=False)
    n_samples, n_positions = ds_matrix.shape
    gp = calibrated_posterior if calibrated_posterior is not None else raw_posterior
    valid_sample = np.ones(n_samples, dtype=bool)
    if sample_ploidy is not None:
        valid_sample = sample_ploidy.astype(np.int16, copy=False) > 0

    if genotype_call is not None:
        called = (genotype_call.astype(np.int16, copy=False) >= 0) & valid_sample[:, None]
    else:
        called = np.isfinite(ds_matrix) & valid_sample[:, None]
    n_callable_samples = int(np.sum(valid_sample))
    callable_denominator = float(max(n_callable_samples, 1))
    calling_rate = (np.sum(called, axis=0) / callable_denominator).astype(np.float32, copy=False)
    missing_rate = (1.0 - calling_rate).astype(np.float32, copy=False)
    depth_matrix = _as_float_matrix(depth, ds_matrix.shape, fill=0.0)
    support = (
        support_mask.astype(bool, copy=False)
        if support_mask is not None
        else depth_matrix > 0.0
    )
    support = support & valid_sample[:, None]

    ploidy_by_sample = (
        np.full(n_samples, max((gp.shape[2] - 1) if gp is not None else 2, 1), dtype=np.float32)
        if sample_ploidy is None
        else np.maximum(sample_ploidy.astype(np.float32, copy=False), 1.0)
    )
    if genotype_call is not None:
        gt_i = genotype_call.astype(np.int16, copy=False)
        ploidy_i = np.maximum(np.rint(ploidy_by_sample).astype(np.int16, copy=False), 1)
        callable_ploidy = called & (ploidy_i[:, None] > 0)
        hard_het_mask = (gt_i > 0) & (gt_i < ploidy_i[:, None]) & callable_ploidy
        hard_ref_mask = (gt_i == 0) & callable_ploidy
        hard_alt_mask = (gt_i == ploidy_i[:, None]) & callable_ploidy
        hard_het_rate = (np.sum(hard_het_mask, axis=0) / callable_denominator).astype(np.float32, copy=False)
        hard_hom_ref_rate = (np.sum(hard_ref_mask, axis=0) / callable_denominator).astype(np.float32, copy=False)
        hard_hom_alt_rate = (np.sum(hard_alt_mask, axis=0) / callable_denominator).astype(np.float32, copy=False)
    else:
        hard_het_rate = np.full(n_positions, np.nan, dtype=np.float32)
        hard_hom_ref_rate = np.full(n_positions, np.nan, dtype=np.float32)
        hard_hom_alt_rate = np.full(n_positions, np.nan, dtype=np.float32)
    hard_hom_rate = np.maximum(hard_hom_ref_rate, hard_hom_alt_rate).astype(np.float32, copy=False)

    alt_fraction = np.divide(
        ds_matrix,
        ploidy_by_sample[:, None],
        out=np.full_like(ds_matrix, np.nan, dtype=np.float32),
        where=ploidy_by_sample[:, None] > 0,
    )
    alt_af = _safe_nanmean(np.where(valid_sample[:, None], alt_fraction, np.nan), axis=0)
    maf = np.minimum(alt_af, 1.0 - alt_af).astype(np.float32, copy=False)

    if gp is not None:
        gp_f = gp.astype(np.float32, copy=False)
        info = compute_info_score_per_variant(gp_f)
        conf, _, entropy = posterior_confidence_metrics(gp_f)
        max_gp = _safe_nanmean(np.where(valid_sample[:, None], conf, np.nan), axis=0)
        mean_entropy = _safe_nanmean(np.where(valid_sample[:, None], entropy, np.nan), axis=0)
        if gp_f.shape[2] == 3:
            het_rate = _safe_nanmean(np.where(valid_sample[:, None], gp_f[:, :, 1], np.nan), axis=0)
            hom_ref_rate = _safe_nanmean(np.where(valid_sample[:, None], gp_f[:, :, 0], np.nan), axis=0)
            hom_alt_rate = _safe_nanmean(np.where(valid_sample[:, None], gp_f[:, :, 2], np.nan), axis=0)
            hwe = compute_hwe_features_from_posterior(gp_f)
            hwe_deviation = hwe["hwe_deviation"].astype(np.float32, copy=False)
            hwe_chisq = hwe["hwe_chisq"].astype(np.float32, copy=False)
            hwe_pvalue = hwe["hwe_pvalue"].astype(np.float32, copy=False)
        else:
            het_rate = np.full(n_positions, np.nan, dtype=np.float32)
            hom_ref_rate = _safe_nanmean(np.where(valid_sample[:, None], gp_f[:, :, 0], np.nan), axis=0)
            hom_alt_rate = _safe_nanmean(np.where(valid_sample[:, None], gp_f[:, :, -1], np.nan), axis=0)
            hwe_deviation = np.full(n_positions, np.nan, dtype=np.float32)
            hwe_chisq = np.full(n_positions, np.nan, dtype=np.float32)
            hwe_pvalue = np.full(n_positions, np.nan, dtype=np.float32)
    else:
        info = np.full(n_positions, np.nan, dtype=np.float32)
        max_gp = np.full(n_positions, np.nan, dtype=np.float32)
        mean_entropy = np.full(n_positions, np.nan, dtype=np.float32)
        if genotype_call is not None:
            het_rate = hard_het_rate.astype(np.float32, copy=False)
            hom_ref_rate = hard_hom_ref_rate.astype(np.float32, copy=False)
            hom_alt_rate = hard_hom_alt_rate.astype(np.float32, copy=False)
        else:
            het_rate = np.full(n_positions, np.nan, dtype=np.float32)
            hom_ref_rate = np.full(n_positions, np.nan, dtype=np.float32)
            hom_alt_rate = np.full(n_positions, np.nan, dtype=np.float32)
        hwe_deviation = np.full(n_positions, np.nan, dtype=np.float32)
        hwe_chisq = np.full(n_positions, np.nan, dtype=np.float32)
        hwe_pvalue = np.full(n_positions, np.nan, dtype=np.float32)

    hom_rate = np.maximum(hom_ref_rate, hom_alt_rate).astype(np.float32, copy=False)
    rows: dict[str, Any] = {
        "block_id": np.full(n_positions, int(block_id), dtype=np.int32),
        "chromosome": np.repeat(str(chromosome), n_positions),
        "position": positions.astype(np.int64, copy=False),
        "n_samples": np.full(n_positions, n_samples, dtype=np.int32),
        "n_callable_samples": np.full(n_positions, n_callable_samples, dtype=np.int32),
        "calling_rate": calling_rate,
        "missing_rate": missing_rate.astype(np.float32, copy=False),
        "maf": np.clip(maf, 0.0, 0.5),
        "alt_af": np.clip(alt_af, 0.0, 1.0),
        "het_rate": het_rate.astype(np.float32, copy=False),
        "hom_rate": hom_rate.astype(np.float32, copy=False),
        "hard_het_rate": hard_het_rate.astype(np.float32, copy=False),
        "hard_hom_rate": hard_hom_rate.astype(np.float32, copy=False),
        "hwe_deviation": hwe_deviation,
        "hwe_chisq": hwe_chisq,
        "hwe_pvalue": hwe_pvalue,
        "info": info.astype(np.float32, copy=False),
        "mean_entropy": mean_entropy.astype(np.float32, copy=False),
        "mean_max_gp": max_gp.astype(np.float32, copy=False),
        "mean_depth": _safe_nanmean(depth_matrix, axis=0),
        "support_rate": (np.sum(support, axis=0) / callable_denominator).astype(np.float32, copy=False),
    }

    if raw_posterior is not None and calibrated_posterior is not None and raw_posterior.shape == calibrated_posterior.shape:
        raw_ds = _posterior_dosage(raw_posterior)
        cal_ds = _posterior_dosage(calibrated_posterior)
        rows["calibration_abs_dosage_shift"] = _safe_nanmean(np.abs(cal_ds - raw_ds), axis=0)
        raw_gp = raw_posterior.astype(np.float32, copy=False)
        cal_gp = calibrated_posterior.astype(np.float32, copy=False)
        if raw_gp.shape[2] == 3:
            rows["calibration_het_rate_shift"] = (
                _safe_nanmean(cal_gp[:, :, 1], axis=0) - _safe_nanmean(raw_gp[:, :, 1], axis=0)
            ).astype(np.float32, copy=False)
        raw_ploidy = max(raw_gp.shape[2] - 1, 1)
        cal_ploidy = max(cal_gp.shape[2] - 1, 1)
        raw_af = _safe_nanmean(raw_ds / float(raw_ploidy), axis=0)
        cal_af = _safe_nanmean(cal_ds / float(cal_ploidy), axis=0)
        rows["calibration_abs_maf_shift"] = np.abs(
            np.minimum(cal_af, 1.0 - cal_af) - np.minimum(raw_af, 1.0 - raw_af)
        ).astype(np.float32, copy=False)

    return pd.DataFrame(rows)


def diagnostic_thresholds(**overrides: float | None) -> dict[str, float]:
    out = dict(DEFAULT_DIAGNOSTIC_THRESHOLDS)
    for key, value in overrides.items():
        if value is not None:
            out[key] = float(value)
    return out


def summarize_diagnostics(
    diagnostics: pd.DataFrame,
    *,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = diagnostic_thresholds(**(thresholds or {}))
    if diagnostics.empty:
        return {"status": "empty", "thresholds": thresholds, "warnings": [], "failures": []}

    numeric = diagnostics.copy()

    def count_where(col: str, op: str, threshold: float) -> int:
        if col not in numeric:
            return 0
        values = pd.to_numeric(numeric[col], errors="coerce")
        if op == ">=":
            mask = values >= float(threshold)
        else:
            mask = values <= float(threshold)
        return int(mask.fillna(False).sum())

    checks = {
        "het_rate": (
            count_where("het_rate", ">=", thresholds["warn_het_rate"]),
            count_where("het_rate", ">=", thresholds["fail_het_rate"]),
        ),
        "hard_het_rate": (
            count_where("hard_het_rate", ">=", thresholds["warn_het_rate"]),
            count_where("hard_het_rate", ">=", thresholds["fail_het_rate"]),
        ),
        "hom_rate": (
            count_where("hom_rate", ">=", thresholds["warn_hom_rate"]),
            count_where("hom_rate", ">=", thresholds["fail_hom_rate"]),
        ),
        "hard_hom_rate": (
            count_where("hard_hom_rate", ">=", thresholds["warn_hom_rate"]),
            count_where("hard_hom_rate", ">=", thresholds["fail_hom_rate"]),
        ),
        "missing_rate": (
            count_where("missing_rate", ">=", thresholds["warn_missing_rate"]),
            count_where("missing_rate", ">=", thresholds["fail_missing_rate"]),
        ),
        "info": (
            count_where("info", "<=", thresholds["warn_low_info"]),
            count_where("info", "<=", thresholds["fail_low_info"]),
        ),
        "mean_max_gp": (
            count_where("mean_max_gp", "<=", thresholds["warn_low_max_gp"]),
            count_where("mean_max_gp", "<=", thresholds["fail_low_max_gp"]),
        ),
    }
    warnings = [name for name, (warn_count, _) in checks.items() if warn_count > 0]
    failures = [name for name, (_, fail_count) in checks.items() if fail_count > 0]
    aggregate: dict[str, float] = {}
    for col in (
        "maf",
        "het_rate",
        "hom_rate",
        "hard_het_rate",
        "hard_hom_rate",
        "missing_rate",
        "info",
        "mean_entropy",
        "mean_max_gp",
        "mean_depth",
        "support_rate",
        "calibration_abs_dosage_shift",
        "calibration_abs_maf_shift",
    ):
        if col in numeric:
            values = pd.to_numeric(numeric[col], errors="coerce")
            aggregate[f"{col}_mean"] = float(values.mean(skipna=True))
            aggregate[f"{col}_median"] = float(values.median(skipna=True))

    return {
        "status": "fail" if failures else ("warn" if warnings else "ok"),
        "n_variants": int(len(diagnostics)),
        "thresholds": thresholds,
        "warnings": warnings,
        "failures": failures,
        "check_counts": {
            name: {"warn": int(warn_count), "fail": int(fail_count)}
            for name, (warn_count, fail_count) in checks.items()
        },
        "aggregate": aggregate,
    }


def write_block_diagnostics(
    output_dir: str | Path,
    diagnostics: pd.DataFrame,
    *,
    block_id: int,
    compression: str = "zstd",
    compression_level: int = 6,
) -> Path:
    out_dir = Path(output_dir) / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"block={int(block_id):06d}.parquet"
    table = pa.Table.from_pandas(diagnostics, preserve_index=False)
    pq.write_table(table, path, compression=compression, compression_level=compression_level)
    return path


def load_diagnostics(output_dir: str | Path) -> pd.DataFrame:
    diag_dir = Path(output_dir) / "diagnostics"
    if not diag_dir.exists() or not any(diag_dir.glob("block=*.parquet")):
        return pd.DataFrame()
    return ds.dataset(str(diag_dir), format="parquet").to_table().to_pandas()


def _founder_update_diversity(output_dir: str | Path) -> dict[str, Any]:
    founder_dir = Path(output_dir) / "founder_updates"
    if not founder_dir.exists() or not any(founder_dir.glob("block=*.parquet")):
        return {}
    try:
        founder_df = ds.dataset(str(founder_dir), format="parquet").to_table(
            columns=["position", "founder", "alt_prob"]
        ).to_pandas()
    except Exception as exc:
        return {"status": "error", "error": str(exc)}
    if founder_df.empty:
        return {}
    grouped = founder_df.groupby("position", observed=True)["alt_prob"].agg(["count", "std", "min", "max"])
    grouped = grouped[grouped["count"] > 1].copy()
    if grouped.empty:
        return {}
    std = pd.to_numeric(grouped["std"], errors="coerce").fillna(0.0).astype(float)
    span = (
        pd.to_numeric(grouped["max"], errors="coerce").fillna(0.0).astype(float)
        - pd.to_numeric(grouped["min"], errors="coerce").fillna(0.0).astype(float)
    )
    return {
        "status": "ok",
        "n_positions": int(len(grouped)),
        "founder_alt_prob_std_mean": float(std.mean()),
        "founder_alt_prob_std_median": float(std.median()),
        "founder_alt_prob_std_min": float(std.min()),
        "founder_alt_prob_span_mean": float(span.mean()),
        "founder_alt_prob_span_median": float(span.median()),
    }


def write_diagnostics_summary(
    output_dir: str | Path,
    *,
    thresholds: dict[str, float] | None = None,
    fail_on_error: bool = False,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    diagnostics = load_diagnostics(out_dir)
    summary = summarize_diagnostics(diagnostics, thresholds=thresholds)
    founder_diversity = _founder_update_diversity(out_dir)
    if founder_diversity:
        summary["founder_update_diversity"] = founder_diversity
        if founder_diversity.get("status") == "ok":
            thresholds_resolved = diagnostic_thresholds(**(thresholds or {}))
            median_std = float(founder_diversity.get("founder_alt_prob_std_median", float("nan")))
            warnings = list(summary.get("warnings", []))
            failures = list(summary.get("failures", []))
            if np.isfinite(median_std) and median_std <= thresholds_resolved["fail_low_founder_std"]:
                if "founder_collapse" not in failures:
                    failures.append("founder_collapse")
            elif np.isfinite(median_std) and median_std <= thresholds_resolved["warn_low_founder_std"]:
                if "founder_collapse" not in warnings:
                    warnings.append("founder_collapse")
            summary["warnings"] = warnings
            summary["failures"] = failures
            summary["status"] = "fail" if failures else ("warn" if warnings else summary.get("status", "ok"))
    path = out_dir / "diagnostics_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    if fail_on_error and summary.get("status") == "fail":
        raise RuntimeError(f"STITCHCONT diagnostics failed thresholds; see {path}")
    return summary
