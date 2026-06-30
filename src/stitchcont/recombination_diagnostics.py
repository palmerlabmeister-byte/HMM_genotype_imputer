from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def _standardize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).upper() for c in out.columns]
    if "GENETIC_CM" in out.columns and "CM" not in out.columns:
        out["CM"] = out["GENETIC_CM"]
    return out


def compute_recombination_diagnostics(
    *,
    positions_path: str | Path,
    output_dir: str | Path,
    chromosome: str,
    genetic_map_path: str | Path | None = None,
    recombination_rate_cM_per_Mb: float = 0.5,
    generation_values: np.ndarray | None = None,
    generation_bin_size: float = 0.0,
) -> dict[str, Any]:
    positions = _standardize(_read_table(positions_path))
    if "CHR" in positions.columns:
        positions = positions.loc[positions["CHR"].astype(str).str.replace("chr", "", case=False) == str(chromosome).replace("chr", "")]
    if "POS" not in positions.columns:
        raise ValueError("positions table must contain POS")
    positions = positions.sort_values("POS").drop_duplicates("POS")
    pos = positions["POS"].to_numpy(dtype=np.float64)
    used_map = False
    cm = None
    map_summary = None
    if genetic_map_path:
        gm = _standardize(_read_table(genetic_map_path))
        if "CHR" in gm.columns:
            gm = gm.loc[gm["CHR"].astype(str).str.replace("chr", "", case=False) == str(chromosome).replace("chr", "")]
        if {"POS", "CM"}.issubset(gm.columns) and gm.shape[0] >= 2:
            gm = gm.loc[np.isfinite(pd.to_numeric(gm["POS"], errors="coerce")) & np.isfinite(pd.to_numeric(gm["CM"], errors="coerce"))].copy()
            gm["POS"] = pd.to_numeric(gm["POS"], errors="coerce")
            gm["CM"] = pd.to_numeric(gm["CM"], errors="coerce")
            gm = gm.sort_values("POS").drop_duplicates("POS")
            if gm.shape[0] >= 2 and float(gm["CM"].max() - gm["CM"].min()) > 0:
                cm = np.interp(pos, gm["POS"].to_numpy(dtype=np.float64), gm["CM"].to_numpy(dtype=np.float64))
                used_map = True
                map_summary = {"n_map_rows": int(gm.shape[0]), "map_cm_min": float(gm["CM"].min()), "map_cm_max": float(gm["CM"].max())}
    if cm is None:
        delta_bp = np.diff(pos, prepend=pos[0])
        cm = np.cumsum(delta_bp / 1_000_000.0 * float(recombination_rate_cM_per_Mb))
    delta_cm = np.maximum(np.diff(cm, prepend=cm[0]), 0.0)
    rates_morgans = np.clip(delta_cm / 100.0, 1e-10, 0.25)
    generations = np.asarray(generation_values if generation_values is not None else np.array([1.0]), dtype=np.float64)
    generations = generations[np.isfinite(generations)]
    if generations.size == 0:
        generations = np.array([1.0])
    if generation_bin_size and generation_bin_size > 0:
        generations = np.round(generations / float(generation_bin_size)) * float(generation_bin_size)
    g_summary = np.percentile(generations, [0, 25, 50, 75, 100]).tolist()
    med_g = float(np.median(generations))
    switch = 1.0 - np.exp(-med_g * rates_morgans)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_window = pd.DataFrame({"POS": pos.astype(np.int64), "CM": cm.astype(float), "delta_cm": delta_cm.astype(float), "rate_morgans": rates_morgans.astype(float), "switch_probability_median_generation": switch.astype(float)})
    try:
        per_window.to_parquet(out_dir / "recombination_diagnostics.parquet", index=False)
        table_file = "recombination_diagnostics.parquet"
    except Exception:
        per_window.to_csv(out_dir / "recombination_diagnostics.csv", index=False)
        table_file = "recombination_diagnostics.csv"
    summary = {
        "chromosome": str(chromosome),
        "n_positions": int(pos.size),
        "used_explicit_genetic_map": bool(used_map),
        "map_summary": map_summary,
        "genetic_length_cm": float(cm[-1] - cm[0]) if pos.size else 0.0,
        "delta_cm_mean": float(np.mean(delta_cm)) if delta_cm.size else 0.0,
        "delta_cm_median": float(np.median(delta_cm)) if delta_cm.size else 0.0,
        "rate_morgans_max": float(np.max(rates_morgans)) if rates_morgans.size else 0.0,
        "generation_quantiles": g_summary,
        "median_generation": med_g,
        "switch_probability_mean_at_median_generation": float(np.mean(switch)) if switch.size else 0.0,
        "switch_probability_p99_at_median_generation": float(np.percentile(switch, 99)) if switch.size else 0.0,
        "table_file": table_file,
        "warnings": [],
    }
    if summary["genetic_length_cm"] <= 0:
        summary["warnings"].append("zero_genetic_length")
    if summary["switch_probability_p99_at_median_generation"] > 0.1:
        summary["warnings"].append("high_per_interval_switch_probability")
    (out_dir / "recombination_diagnostics_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
