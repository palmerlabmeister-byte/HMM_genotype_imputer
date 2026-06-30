from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


NA_TEXT = {"", "na", "nan", "n/a", "none", "null", "0"}


@dataclass(slots=True)
class PedigreeQCResult:
    curated_pedigree: pd.DataFrame
    edge_qc: pd.DataFrame
    sample_qc: pd.DataFrame
    similarity: pd.DataFrame
    embedding: pd.DataFrame
    long_edges: pd.DataFrame
    summary: dict[str, object]


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _first_present(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    cols = list(columns)
    exact = set(cols)
    for col in candidates:
        if col in exact:
            return col
    lower = {str(c).lower(): c for c in cols}
    for col in candidates:
        hit = lower.get(col.lower())
        if hit is not None:
            return hit
    return None


def _clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in NA_TEXT else text


def _variant_id(chromosome: object, position: object) -> str:
    return f"{str(chromosome)}:{int(position)}"


def resolve_pedigree_columns(
    table: pd.DataFrame,
    *,
    offspring_col: str = "sample_id",
    parent1_col: str = "father_id",
    parent2_col: str = "mother_id",
    family_col: str = "",
) -> dict[str, str | None]:
    cols = table.columns
    child = _first_present(cols, (offspring_col, "sample_id", "offspring", "child", "rfid", "iid"))
    parent1 = _first_present(cols, (parent1_col, "father_id", "father", "sire", "parent1", "parent_1", "dad"))
    parent2 = _first_present(cols, (parent2_col, "mother_id", "mother", "dam", "parent2", "parent_2", "mom"))
    family = _first_present(cols, (family_col, "fid", "family_id", "family", "familyid", "fidh")) if family_col else _first_present(
        cols,
        ("fid", "family_id", "family", "familyid", "fidh"),
    )
    if child is None:
        raise ValueError("Pedigree/sample table needs a sample/offspring column.")
    return {"offspring": child, "parent1": parent1, "parent2": parent2, "family": family}


def prepare_pedigree_table(
    samples: pd.DataFrame,
    pedigree: pd.DataFrame | None = None,
    *,
    sample_id_col: str = "sample_id",
    offspring_col: str = "sample_id",
    parent1_col: str = "father_id",
    parent2_col: str = "mother_id",
    family_col: str = "",
) -> tuple[pd.DataFrame, dict[str, str | None]]:
    source = samples if pedigree is None else pedigree
    cols = resolve_pedigree_columns(
        source,
        offspring_col=offspring_col,
        parent1_col=parent1_col,
        parent2_col=parent2_col,
        family_col=family_col,
    )
    if cols["parent1"] is None and cols["parent2"] is None:
        raise ValueError("Pedigree QC needs at least one parent column.")
    out = source.copy()
    out["_pedigree_child_id"] = out[str(cols["offspring"])].map(_clean_id)
    out["_pedigree_parent1_id"] = out[str(cols["parent1"])].map(_clean_id) if cols["parent1"] else ""
    out["_pedigree_parent2_id"] = out[str(cols["parent2"])].map(_clean_id) if cols["parent2"] else ""
    if cols["family"]:
        out["_pedigree_family_id"] = out[str(cols["family"])].map(_clean_id)
    else:
        out["_pedigree_family_id"] = ""
    out = out.drop_duplicates("_pedigree_child_id", keep="first")

    sample_ids = samples[sample_id_col].map(_clean_id).to_numpy(dtype=object)
    sample_order = pd.DataFrame({"_pedigree_child_id": sample_ids, "_stitchcont_sample_order": np.arange(len(sample_ids))})
    out = sample_order.merge(out, on="_pedigree_child_id", how="left", sort=False)
    out["_pedigree_parent1_id"] = out["_pedigree_parent1_id"].fillna("").map(_clean_id)
    out["_pedigree_parent2_id"] = out["_pedigree_parent2_id"].fillna("").map(_clean_id)
    out["_pedigree_family_id"] = out["_pedigree_family_id"].fillna("").map(_clean_id)
    out = out.sort_values("_stitchcont_sample_order").reset_index(drop=True)
    return out, cols


def _select_positions_from_run(
    run_output_dir: Path,
    *,
    max_variants: int,
    random_seed: int,
) -> set[tuple[str, int]]:
    pos_path = run_output_dir / "positions.parquet"
    if not pos_path.exists():
        return set()
    positions = pd.read_parquet(pos_path)
    chrom_col = "chromosome" if "chromosome" in positions.columns else "CHR"
    pos_col = "position" if "position" in positions.columns else "POS"
    positions = positions[[chrom_col, pos_col]].dropna().copy()
    positions[pos_col] = positions[pos_col].astype(np.int64)
    positions = positions.drop_duplicates()
    if max_variants > 0 and len(positions) > max_variants:
        rng = np.random.default_rng(int(random_seed))
        take = np.sort(rng.choice(len(positions), size=int(max_variants), replace=False))
        positions = positions.iloc[take]
    return {(str(row[chrom_col]), int(row[pos_col])) for _, row in positions.iterrows()}


def _read_long_genotype_source(
    path: str | Path,
    *,
    value_column: str = "auto",
    selected_positions: set[tuple[str, int]] | None = None,
) -> pd.DataFrame:
    path = Path(path)
    dataset = ds.dataset(str(path), format="parquet")
    schema_cols = set(dataset.schema.names)
    if value_column == "auto":
        if "genotype_call" in schema_cols:
            value_column = "genotype_call"
        elif "dosage" in schema_cols:
            value_column = "dosage"
        else:
            raise ValueError(f"{path} needs genotype_call or dosage column.")
    chrom_col = "chromosome" if "chromosome" in schema_cols else ("CHR" if "CHR" in schema_cols else None)
    pos_col = "position" if "position" in schema_cols else ("POS" if "POS" in schema_cols else None)
    if "sample_id" not in schema_cols or pos_col is None:
        raise ValueError(f"{path} needs sample_id and position/POS columns.")
    columns = ["sample_id", pos_col, value_column]
    if chrom_col is not None:
        columns.append(chrom_col)
    filt = None
    if selected_positions:
        pos_values = sorted({pos for _, pos in selected_positions})
        filt = ds.field(pos_col).isin(pos_values)
    table = dataset.to_table(columns=columns, filter=filt)
    df = table.to_pandas()
    df = df.rename(columns={pos_col: "position", value_column: "genotype_value"})
    if chrom_col is None:
        df["chromosome"] = "unknown"
    else:
        df = df.rename(columns={chrom_col: "chromosome"})
    if selected_positions:
        keep = {_variant_id(chrom, pos) for chrom, pos in selected_positions}
        variant = [_variant_id(c, p) for c, p in zip(df["chromosome"], df["position"], strict=False)]
        df = df.loc[pd.Series(variant, index=df.index).isin(keep)].copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["chromosome"] = df["chromosome"].astype(str)
    df["position"] = df["position"].astype(np.int64)
    df["variant_id"] = [_variant_id(c, p) for c, p in zip(df["chromosome"], df["position"], strict=False)]
    values = pd.to_numeric(df["genotype_value"], errors="coerce").astype(np.float32)
    values = values.mask(values < 0)
    df["genotype_value"] = values
    return df[["sample_id", "variant_id", "genotype_value", "chromosome", "position"]]


def load_genotype_matrix(
    *,
    sample_ids: np.ndarray,
    run_output_dirs: list[str | Path] | None = None,
    genotype_tables: list[str | Path] | None = None,
    value_column: str = "auto",
    max_variants: int = 50_000,
    random_seed: int = 0,
    min_call_rate: float = 0.80,
    min_maf: float = 0.005,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sources: list[tuple[Path, set[tuple[str, int]] | None]] = []
    for run_dir_raw in run_output_dirs or []:
        run_dir = Path(run_dir_raw)
        data_dir = run_dir / "genotype_calls"
        if not data_dir.exists():
            data_dir = run_dir / "dosage"
        if not data_dir.exists():
            raise FileNotFoundError(f"No genotype_calls/ or dosage/ directory found in {run_dir}")
        selected = _select_positions_from_run(run_dir, max_variants=max_variants, random_seed=random_seed)
        sources.append((data_dir, selected if selected else None))
    sources.extend((Path(p), None) for p in genotype_tables or [])
    if not sources:
        raise ValueError("Pedigree QC needs at least one --run-output-dir or --genotype-table.")

    frames = [
        _read_long_genotype_source(path, value_column=value_column, selected_positions=selected)
        for path, selected in sources
    ]
    long_df = pd.concat(frames, ignore_index=True)
    if max_variants > 0:
        variants = pd.Index(long_df["variant_id"].drop_duplicates())
        if len(variants) > max_variants:
            rng = np.random.default_rng(int(random_seed))
            keep = set(rng.choice(variants.to_numpy(dtype=object), size=int(max_variants), replace=False).tolist())
            long_df = long_df.loc[long_df["variant_id"].isin(keep)].copy()

    matrix_df = long_df.pivot_table(
        index="sample_id",
        columns="variant_id",
        values="genotype_value",
        aggfunc="first",
        observed=True,
    )
    matrix_df = matrix_df.reindex(sample_ids.astype(str))
    matrix = matrix_df.to_numpy(dtype=np.float32, copy=True)
    if matrix.size == 0:
        raise ValueError("No genotype values were loaded for pedigree QC.")

    call_rate = np.mean(np.isfinite(matrix), axis=0)
    alt_af = np.nanmean(matrix / 2.0, axis=0)
    maf = np.minimum(alt_af, 1.0 - alt_af)
    keep = np.isfinite(maf) & (call_rate >= float(min_call_rate)) & (maf >= float(min_maf)) & (maf <= 0.5)
    if not np.any(keep):
        raise ValueError(
            "No variants passed pedigree QC filters. "
            f"Try lower --min-call-rate or --min-maf; loaded_variants={matrix.shape[1]}."
        )
    matrix = matrix[:, keep].astype(np.float32, copy=False)
    kept_variants = matrix_df.columns.to_numpy(dtype=object)[keep]
    metadata = (
        long_df[["variant_id", "chromosome", "position"]]
        .drop_duplicates("variant_id")
        .set_index("variant_id")
        .reindex(kept_variants)
        .reset_index()
    )
    metadata["call_rate"] = call_rate[keep].astype(np.float32)
    metadata["maf"] = maf[keep].astype(np.float32)
    return matrix, kept_variants, metadata


def _standardize_genotypes(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    g = matrix.astype(np.float32, copy=True)
    call = np.isfinite(g)
    mean = np.nanmean(g, axis=0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    g = np.where(call, g, mean[None, :]).astype(np.float32, copy=False)
    g -= mean[None, :]
    scale = np.sqrt(np.mean(g * g, axis=0)).astype(np.float32)
    keep = np.isfinite(scale) & (scale > 1e-6)
    if not np.any(keep):
        raise ValueError("No variable variants are available for pedigree relatedness.")
    x = g[:, keep] / scale[None, keep]
    summary = {
        "n_variants_standardized": int(keep.sum()),
        "mean_genotype_missing_rate": float(1.0 - np.mean(call)),
    }
    return x.astype(np.float32, copy=False), summary


def _sample_correlation_matrix_input(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    g = matrix.astype(np.float32, copy=True)
    call = np.isfinite(g)
    variant_mean = np.nanmean(g, axis=0).astype(np.float32)
    variant_mean = np.where(np.isfinite(variant_mean), variant_mean, 0.0).astype(np.float32)
    g = np.where(call, g, variant_mean[None, :]).astype(np.float32, copy=False)
    sample_mean = np.mean(g, axis=1, keepdims=True).astype(np.float32)
    g = (g - sample_mean).astype(np.float32, copy=False)
    norm = np.sqrt(np.sum(g * g, axis=1, keepdims=True)).astype(np.float32)
    valid = np.isfinite(norm[:, 0]) & (norm[:, 0] > 1e-6)
    if not np.any(valid):
        raise ValueError("No samples have variable genotypes for pedigree relatedness.")
    g = np.where(valid[:, None], g / np.clip(norm, 1e-6, None), 0.0).astype(np.float32, copy=False)
    return g, {
        "n_variants_for_r": int(g.shape[1]),
        "mean_genotype_missing_rate": float(1.0 - np.mean(call)),
        "n_samples_with_nonzero_genotype_variance": int(valid.sum()),
    }


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _relationship_call(r: float, *, unrelated_max_r: float, first_degree_min_r: float, same_min_r: float) -> str:
    if not np.isfinite(r):
        return "not_measured"
    if r >= float(same_min_r):
        return "same"
    if r >= float(first_degree_min_r):
        return "first_degree"
    if r >= float(unrelated_max_r):
        return "inconclusive"
    return "unrelated"


def compute_similarity(
    matrix: np.ndarray,
    sample_ids: np.ndarray,
    *,
    query_pairs: set[tuple[str, str]] | None = None,
    report_min_r: float = 0.59,
    unrelated_max_r: float = 0.59,
    first_degree_min_r: float = 0.64,
    same_min_r: float = 0.88,
    sample_block_size: int = 1024,
    max_full_matrix_samples: int = 5000,
) -> tuple[pd.DataFrame, dict[tuple[str, str], float], np.ndarray | None, dict[str, float]]:
    x, summary = _sample_correlation_matrix_input(matrix)
    sample_ids = sample_ids.astype(str)
    n_samples, n_variants = x.shape
    query_pairs = query_pairs or set()
    r_lookup: dict[tuple[str, str], float] = {}
    rows: list[dict[str, object]] = []
    full_r: np.ndarray | None = None

    if n_samples <= int(max_full_matrix_samples):
        full_r = (x @ x.T).astype(np.float32, copy=False)
        np.fill_diagonal(full_r, 1.0)
        tri_i, tri_j = np.triu_indices(n_samples, k=1)
        vals = full_r[tri_i, tri_j]
        keep = vals >= float(report_min_r)
        sample_set = set(sample_ids.tolist())
        seen: set[tuple[str, str]] = set()
        for i, j, r in zip(tri_i[keep], tri_j[keep], vals[keep], strict=False):
            s1 = str(sample_ids[int(i)])
            s2 = str(sample_ids[int(j)])
            key = _canonical_pair(s1, s2)
            seen.add(key)
            r_lookup[key] = float(r)
            rows.append({"sample_id1": key[0], "sample_id2": key[1], "r": float(r), "required_pair": False})
        for s1, s2 in query_pairs:
            key = _canonical_pair(str(s1), str(s2))
            if key[0] not in sample_set or key[1] not in sample_set:
                continue
            i = int(np.flatnonzero(sample_ids == key[0])[0])
            j = int(np.flatnonzero(sample_ids == key[1])[0])
            r_lookup[key] = float(full_r[i, j])
            if key not in seen:
                rows.append({"sample_id1": key[0], "sample_id2": key[1], "r": float(full_r[i, j]), "required_pair": True})
                seen.add(key)
    else:
        id_to_idx = {str(s): i for i, s in enumerate(sample_ids.tolist())}
        query_by_block: dict[tuple[int, int], list[tuple[str, str, int, int]]] = {}
        for s1, s2 in query_pairs:
            if s1 not in id_to_idx or s2 not in id_to_idx:
                continue
            i, j = id_to_idx[str(s1)], id_to_idx[str(s2)]
            if i == j:
                continue
            if i > j:
                i, j = j, i
                s1, s2 = s2, s1
            bi = i // int(sample_block_size)
            bj = j // int(sample_block_size)
            query_by_block.setdefault((bi, bj), []).append((str(s1), str(s2), i, j))
        seen: set[tuple[str, str]] = set()
        for i0 in range(0, n_samples, int(sample_block_size)):
            i1 = min(i0 + int(sample_block_size), n_samples)
            for j0 in range(i0, n_samples, int(sample_block_size)):
                j1 = min(j0 + int(sample_block_size), n_samples)
                block = (x[i0:i1] @ x[j0:j1].T).astype(np.float32, copy=False)
                if i0 == j0:
                    ii, jj = np.triu_indices(i1 - i0, k=1)
                else:
                    ii, jj = np.nonzero(np.ones(block.shape, dtype=bool))
                vals = block[ii, jj]
                keep = vals >= float(report_min_r)
                for li, lj, r in zip(ii[keep], jj[keep], vals[keep], strict=False):
                    s1 = str(sample_ids[i0 + int(li)])
                    s2 = str(sample_ids[j0 + int(lj)])
                    key = _canonical_pair(s1, s2)
                    seen.add(key)
                    r_lookup[key] = float(r)
                    rows.append({"sample_id1": key[0], "sample_id2": key[1], "r": float(r), "required_pair": False})
                for s1, s2, i, j in query_by_block.get((i0 // int(sample_block_size), j0 // int(sample_block_size)), []):
                    r = float(block[i - i0, j - j0])
                    key = _canonical_pair(s1, s2)
                    r_lookup[key] = r
                    if key not in seen:
                        rows.append({"sample_id1": key[0], "sample_id2": key[1], "r": r, "required_pair": True})
                        seen.add(key)

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["sample_id1", "sample_id2", "r", "required_pair"])
    out["r2"] = pd.to_numeric(out["r"], errors="coerce") ** 2
    out["relationship_call"] = [
        _relationship_call(r, unrelated_max_r=unrelated_max_r, first_degree_min_r=first_degree_min_r, same_min_r=same_min_r)
        for r in out["r"].to_numpy(dtype=np.float32, copy=False)
    ]
    out = out.drop_duplicates(["sample_id1", "sample_id2"], keep="last").sort_values(
        ["r", "sample_id1", "sample_id2"],
        ascending=[False, True, True],
    )
    summary.update(
        {
            "n_samples": int(n_samples),
            "n_reported_similarity_pairs": int(len(out)),
            "report_min_r": float(report_min_r),
            "max_full_matrix_samples": int(max_full_matrix_samples),
            "full_matrix_computed": bool(full_r is not None),
        }
    )
    return out.reset_index(drop=True), r_lookup, full_r, summary


def _pedigree_query_pairs(pedigree: pd.DataFrame) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in pedigree[["_pedigree_child_id", "_pedigree_parent1_id", "_pedigree_parent2_id"]].itertuples(index=False, name=None):
        child = _clean_id(row[0])
        for parent_raw in row[1:]:
            parent = _clean_id(parent_raw)
            if child and parent and child != parent:
                pairs.add(_canonical_pair(child, parent))
    return pairs


def _expected_relationship_for_pair(pair: tuple[str, str], pedigree: pd.DataFrame) -> str:
    s1, s2 = pair
    ped = pedigree.set_index("_pedigree_child_id", drop=False)
    p1 = ped["_pedigree_parent1_id"].to_dict()
    p2 = ped["_pedigree_parent2_id"].to_dict()
    fam = ped["_pedigree_family_id"].to_dict()
    parents1 = {p for p in (p1.get(s1, ""), p2.get(s1, "")) if p}
    parents2 = {p for p in (p1.get(s2, ""), p2.get(s2, "")) if p}
    if s1 in parents2 or s2 in parents1:
        return "parent_offspring"
    shared = parents1 & parents2
    if len(shared) >= 2:
        return "full_sib"
    if len(shared) == 1:
        return "half_sib"
    if fam.get(s1, "") and fam.get(s1, "") == fam.get(s2, ""):
        return "same_family"
    return "unrelated"


def curate_pedigree(
    pedigree: pd.DataFrame,
    samples: pd.DataFrame,
    *,
    r_lookup: dict[tuple[str, str], float],
    sample_id_col: str = "sample_id",
    unrelated_max_r: float = 0.59,
    first_degree_min_r: float = 0.64,
    same_min_r: float = 0.88,
    unlink_calls: tuple[str, ...] = ("unrelated", "same"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_set = set(samples[sample_id_col].astype(str).tolist())
    curated = pedigree.copy()
    curated["curated_parent1_id"] = curated["_pedigree_parent1_id"]
    curated["curated_parent2_id"] = curated["_pedigree_parent2_id"]
    edge_rows: list[dict[str, object]] = []
    remove_by_role: dict[tuple[int, str], str] = {}

    for idx, row in curated.iterrows():
        child = _clean_id(row["_pedigree_child_id"])
        for role, src_col, out_col in (
            ("parent1", "_pedigree_parent1_id", "curated_parent1_id"),
            ("parent2", "_pedigree_parent2_id", "curated_parent2_id"),
        ):
            parent = _clean_id(row[src_col])
            if not parent:
                continue
            key = _canonical_pair(child, parent)
            r = r_lookup.get(key, np.nan)
            call = _relationship_call(
                r,
                unrelated_max_r=unrelated_max_r,
                first_degree_min_r=first_degree_min_r,
                same_min_r=same_min_r,
            )
            if child not in sample_set or parent not in sample_set:
                status = "no_evidence"
                reason = "child_or_parent_not_genotyped"
            elif call in unlink_calls:
                status = "removed"
                reason = f"observed_{call}_for_declared_parent"
                remove_by_role[(int(idx), out_col)] = reason
            elif call == "first_degree":
                status = "retained"
                reason = "observed_first_degree_for_declared_parent"
            elif call == "inconclusive":
                status = "inconclusive"
                reason = "observed_inconclusive_for_declared_parent"
            else:
                status = "no_evidence"
                reason = "relationship_not_measured"
            edge_rows.append(
                {
                    "child_id": child,
                    "parent_id": parent,
                    "parent_role": role,
                    "r": r,
                    "r2": r * r if np.isfinite(r) else np.nan,
                    "relationship_call": call,
                    "edge_qc_status": status,
                    "reason": reason,
                }
            )

    for (idx, out_col), _reason in remove_by_role.items():
        curated.loc[idx, out_col] = ""
    curated["pedigree_qc_removed_parent_count"] = (
        (curated["_pedigree_parent1_id"].ne("") & curated["curated_parent1_id"].eq(""))
        + (curated["_pedigree_parent2_id"].ne("") & curated["curated_parent2_id"].eq(""))
    ).astype(np.int8)
    curated["pedigree_qc_status"] = np.where(
        curated["pedigree_qc_removed_parent_count"].gt(0),
        "corrected",
        "unchanged",
    )
    # Canonical columns make pedigree_curated.parquet directly consumable by
    # stitchcont run with the default pedigree column arguments.
    curated["sample_id"] = curated["_pedigree_child_id"]
    curated["father_id"] = curated["curated_parent1_id"]
    curated["mother_id"] = curated["curated_parent2_id"]
    curated["family_id"] = curated["_pedigree_family_id"]
    curated["original_parent1_id"] = curated["_pedigree_parent1_id"]
    curated["original_parent2_id"] = curated["_pedigree_parent2_id"]
    edge_qc = pd.DataFrame(edge_rows)
    if edge_qc.empty:
        edge_qc = pd.DataFrame(
            columns=["child_id", "parent_id", "parent_role", "r", "r2", "relationship_call", "edge_qc_status", "reason"]
        )

    sample_qc = samples[[sample_id_col]].copy()
    sample_qc = sample_qc.rename(columns={sample_id_col: "sample_id"})
    removed_counts = edge_qc.loc[edge_qc["edge_qc_status"].eq("removed"), "child_id"].value_counts()
    inconclusive_counts = edge_qc.loc[edge_qc["edge_qc_status"].eq("inconclusive"), "child_id"].value_counts()
    sample_qc["removed_parent_edges"] = sample_qc["sample_id"].map(removed_counts).fillna(0).astype(np.int16)
    sample_qc["inconclusive_parent_edges"] = sample_qc["sample_id"].map(inconclusive_counts).fillna(0).astype(np.int16)
    sample_qc["pedigree_qc_status"] = np.where(
        sample_qc["removed_parent_edges"].gt(0),
        "corrected",
        np.where(sample_qc["inconclusive_parent_edges"].gt(0), "review", "pass"),
    )
    return curated, edge_qc, sample_qc


def annotate_similarity(similarity: pd.DataFrame, pedigree: pd.DataFrame) -> pd.DataFrame:
    if similarity.empty:
        similarity = similarity.copy()
        similarity["expected_relationship"] = pd.Series(dtype=object)
        similarity["pedigree_consistency"] = pd.Series(dtype=object)
        return similarity
    out = similarity.copy()
    expected = [
        _expected_relationship_for_pair((str(row.sample_id1), str(row.sample_id2)), pedigree)
        for row in out.itertuples(index=False)
    ]
    out["expected_relationship"] = expected
    first_like = {"parent_offspring", "full_sib", "half_sib", "same_family"}
    consistency = []
    for exp, obs in zip(out["expected_relationship"].tolist(), out["relationship_call"].tolist(), strict=False):
        if exp == "unrelated" and obs in {"first_degree", "same"}:
            consistency.append("unexpected_related")
        elif exp in first_like and obs == "unrelated":
            consistency.append("unexpected_unrelated")
        elif obs == "inconclusive":
            consistency.append("review")
        else:
            consistency.append("ok")
    out["pedigree_consistency"] = consistency
    return out


def compute_embedding(
    matrix: np.ndarray,
    sample_ids: np.ndarray,
    samples: pd.DataFrame,
    sample_qc: pd.DataFrame,
    *,
    sample_id_col: str = "sample_id",
    random_seed: int = 0,
    n_neighbors: int = 50,
    max_variants: int = 10_000,
) -> tuple[pd.DataFrame, dict[str, object]]:
    x, _ = _standardize_genotypes(matrix)
    if max_variants > 0 and x.shape[1] > max_variants:
        rng = np.random.default_rng(int(random_seed))
        cols = np.sort(rng.choice(x.shape[1], size=int(max_variants), replace=False))
        x = x[:, cols]
    method = "umap"
    try:
        from umap import UMAP

        reducer = UMAP(
            n_components=2,
            n_neighbors=min(max(int(n_neighbors), 2), max(int(x.shape[0] - 1), 2)),
            metric="euclidean",
            random_state=int(random_seed),
            low_memory=True,
        )
        emb = reducer.fit_transform(x)
    except Exception as exc:  # pragma: no cover - exercised when optional dep is missing.
        from sklearn.decomposition import PCA

        method = "pca_fallback"
        emb = PCA(n_components=2, random_state=int(random_seed)).fit_transform(x)
        meta = {"embedding_method": method, "fallback_reason": str(exc)}
    else:
        meta = {"embedding_method": method}

    embedding = pd.DataFrame(
        {
            "sample_id": sample_ids.astype(str),
            "umap1": emb[:, 0].astype(np.float32),
            "umap2": emb[:, 1].astype(np.float32),
        }
    )
    metadata_cols = [
        col
        for col in samples.columns
        if col in {sample_id_col, "rfid", "fid", "family", "family_id", "sex", "tissue", "library_id", "project_name"}
    ]
    if metadata_cols:
        metadata = samples[metadata_cols].copy().rename(columns={sample_id_col: "sample_id"})
        metadata["sample_id"] = metadata["sample_id"].astype(str)
        embedding = embedding.merge(metadata, on="sample_id", how="left")
    embedding = embedding.merge(sample_qc[["sample_id", "pedigree_qc_status"]], on="sample_id", how="left")
    meta.update({"n_embedding_samples": int(len(embedding)), "n_embedding_variants": int(x.shape[1])})
    return embedding, meta


def _edge_segments(edge_qc: pd.DataFrame, embedding: pd.DataFrame, *, curated: bool) -> pd.DataFrame:
    if edge_qc.empty:
        return pd.DataFrame()
    coord = embedding.set_index("sample_id")[["umap1", "umap2"]]
    rows: list[dict[str, object]] = []
    for row in edge_qc.itertuples(index=False):
        if curated and str(row.edge_qc_status) == "removed":
            continue
        child = str(row.child_id)
        parent = str(row.parent_id)
        if child not in coord.index or parent not in coord.index:
            continue
        x1, y1 = coord.loc[child].to_numpy(dtype=np.float32)
        x2, y2 = coord.loc[parent].to_numpy(dtype=np.float32)
        rows.append(
            {
                "child_id": child,
                "parent_id": parent,
                "parent_role": str(row.parent_role),
                "edge_qc_status": str(row.edge_qc_status),
                "relationship_call": str(row.relationship_call),
                "r": float(row.r) if np.isfinite(row.r) else np.nan,
                "r2": float(row.r2) if np.isfinite(row.r2) else np.nan,
                "umap1": float(x1),
                "umap2": float(y1),
                "parent_umap1": float(x2),
                "parent_umap2": float(y2),
                "edge_length_umap": float(np.hypot(x1 - x2, y1 - y2)),
            }
        )
    return pd.DataFrame(rows)


def _long_edge_threshold(lengths: np.ndarray, multiplier: float = 5.0) -> float:
    finite = lengths[np.isfinite(lengths)]
    if finite.size == 0:
        return np.nan
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    if mad <= 1e-8:
        return float(np.quantile(finite, 0.95))
    return median + float(multiplier) * 1.4826 * mad


def write_umap_plot(
    embedding: pd.DataFrame,
    before_edges: pd.DataFrame,
    after_edges: pd.DataFrame,
    output_html: Path,
) -> dict[str, object]:
    try:
        import holoviews as hv
        import hvplot.pandas  # noqa: F401

        hv.extension("bokeh")
    except Exception as exc:  # pragma: no cover - optional plotting stack.
        return {"status": "skipped", "reason": str(exc), "output_html": str(output_html)}

    cmap = {
        "retained": "green",
        "removed": "red",
        "inconclusive": "gold",
        "no_evidence": "gray",
    }
    hover_cols = [
        col
        for col in ["sample_id", "rfid", "fid", "family", "family_id", "sex", "tissue", "library_id", "project_name", "pedigree_qc_status"]
        if col in embedding.columns
    ]

    def panel(edges: pd.DataFrame, title: str):
        points = embedding.hvplot.scatter(
            x="umap1",
            y="umap2",
            color="pedigree_qc_status" if "pedigree_qc_status" in embedding.columns else None,
            hover_cols=hover_cols,
            frame_width=650,
            frame_height=650,
            size=65,
            alpha=0.85,
            title=title,
        )
        if edges.empty:
            return points
        seg = hv.Segments(
            edges,
            kdims=["umap1", "umap2", "parent_umap1", "parent_umap2"],
            vdims=["edge_qc_status", "parent_role", "relationship_call", "r", "edge_length_umap"],
        ).opts(
            color="edge_qc_status",
            cmap=cmap,
            line_width=2,
            alpha=0.75,
            tools=["hover"],
        )
        return seg * points

    layout = (panel(before_edges, "Pedigree edges before curation") + panel(after_edges, "Pedigree edges after curation")).cols(2)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    hv.save(layout.opts(shared_axes=False), str(output_html), backend="bokeh")
    return {"status": "written", "output_html": str(output_html)}


def run_pedigree_qc(
    *,
    samples: pd.DataFrame,
    pedigree: pd.DataFrame | None,
    output_dir: str | Path,
    run_output_dirs: list[str | Path] | None = None,
    genotype_tables: list[str | Path] | None = None,
    sample_id_col: str = "sample_id",
    offspring_col: str = "sample_id",
    parent1_col: str = "father_id",
    parent2_col: str = "mother_id",
    family_col: str = "",
    value_column: str = "auto",
    max_variants: int = 50_000,
    min_call_rate: float = 0.80,
    min_maf: float = 0.005,
    report_min_r: float = 0.59,
    unrelated_max_r: float = 0.59,
    first_degree_min_r: float = 0.64,
    same_min_r: float = 0.88,
    sample_block_size: int = 1024,
    max_full_matrix_samples: int = 5000,
    random_seed: int = 0,
    write_umap: bool = True,
    umap_neighbors: int = 50,
    umap_max_variants: int = 10_000,
    unlink_calls: tuple[str, ...] = ("unrelated", "same"),
) -> PedigreeQCResult:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = samples.copy()
    samples[sample_id_col] = samples[sample_id_col].astype(str)
    sample_ids = samples[sample_id_col].to_numpy(dtype=object)
    ped, resolved_cols = prepare_pedigree_table(
        samples,
        pedigree,
        sample_id_col=sample_id_col,
        offspring_col=offspring_col,
        parent1_col=parent1_col,
        parent2_col=parent2_col,
        family_col=family_col,
    )
    query_pairs = _pedigree_query_pairs(ped)
    matrix, variant_ids, variant_metadata = load_genotype_matrix(
        sample_ids=sample_ids,
        run_output_dirs=run_output_dirs,
        genotype_tables=genotype_tables,
        value_column=value_column,
        max_variants=max_variants,
        random_seed=random_seed,
        min_call_rate=min_call_rate,
        min_maf=min_maf,
    )
    similarity, r_lookup, full_r, rel_summary = compute_similarity(
        matrix,
        sample_ids,
        query_pairs=query_pairs,
        report_min_r=report_min_r,
        unrelated_max_r=unrelated_max_r,
        first_degree_min_r=first_degree_min_r,
        same_min_r=same_min_r,
        sample_block_size=sample_block_size,
        max_full_matrix_samples=max_full_matrix_samples,
    )
    similarity = annotate_similarity(similarity, ped)
    curated, edge_qc, sample_qc = curate_pedigree(
        ped,
        samples,
        r_lookup=r_lookup,
        sample_id_col=sample_id_col,
        unrelated_max_r=unrelated_max_r,
        first_degree_min_r=first_degree_min_r,
        same_min_r=same_min_r,
        unlink_calls=unlink_calls,
    )
    embedding, embedding_meta = compute_embedding(
        matrix,
        sample_ids,
        samples,
        sample_qc,
        sample_id_col=sample_id_col,
        random_seed=random_seed,
        n_neighbors=umap_neighbors,
        max_variants=umap_max_variants,
    )
    before_edges = _edge_segments(edge_qc, embedding, curated=False)
    after_edges = _edge_segments(edge_qc, embedding, curated=True)
    threshold = _long_edge_threshold(before_edges["edge_length_umap"].to_numpy(dtype=np.float32)) if not before_edges.empty else np.nan
    long_edges = before_edges.loc[before_edges["edge_length_umap"].ge(threshold)].copy() if np.isfinite(threshold) else before_edges.iloc[0:0].copy()

    plot_meta = {"status": "disabled"}
    if write_umap:
        plot_meta = write_umap_plot(
            embedding,
            before_edges,
            after_edges,
            output_dir / "pedigree_umap_before_after.html",
        )

    summary = {
        "command": "pedigree-qc",
        "output_dir": str(output_dir.resolve()),
        "n_samples": int(len(samples)),
        "n_variants_loaded": int(len(variant_ids)),
        "n_declared_parent_edges": int(len(edge_qc)),
        "n_removed_parent_edges": int(edge_qc["edge_qc_status"].eq("removed").sum()) if not edge_qc.empty else 0,
        "n_inconclusive_parent_edges": int(edge_qc["edge_qc_status"].eq("inconclusive").sum()) if not edge_qc.empty else 0,
        "n_similarity_pairs_reported": int(len(similarity)),
        "n_unexpected_related_pairs": int(similarity["pedigree_consistency"].eq("unexpected_related").sum()) if not similarity.empty else 0,
        "n_unexpected_unrelated_pairs": int(similarity["pedigree_consistency"].eq("unexpected_unrelated").sum()) if not similarity.empty else 0,
        "n_long_umap_edges": int(len(long_edges)),
        "long_umap_edge_threshold": float(threshold) if np.isfinite(threshold) else None,
        "thresholds": {
            "report_min_r": float(report_min_r),
            "unrelated_max_r": float(unrelated_max_r),
            "first_degree_min_r": float(first_degree_min_r),
            "same_min_r": float(same_min_r),
            "unlink_calls": list(unlink_calls),
        },
        "resolved_pedigree_columns": resolved_cols,
        "relationship_summary": rel_summary,
        "embedding": embedding_meta,
        "plot": plot_meta,
        "full_r_matrix_shape": None if full_r is None else [int(full_r.shape[0]), int(full_r.shape[1])],
    }

    _write_table(curated, output_dir / "pedigree_curated.parquet")
    _write_table(edge_qc, output_dir / "pedigree_edge_qc.parquet")
    _write_table(sample_qc, output_dir / "pedigree_sample_qc.parquet")
    _write_table(similarity, output_dir / "sample_similarity.parquet")
    _write_table(embedding, output_dir / "pedigree_umap_embedding.parquet")
    _write_table(before_edges, output_dir / "pedigree_umap_edges_before.parquet")
    _write_table(after_edges, output_dir / "pedigree_umap_edges_after.parquet")
    _write_table(long_edges, output_dir / "pedigree_long_edges.parquet")
    _write_table(variant_metadata, output_dir / "pedigree_qc_variants.parquet")
    (output_dir / "pedigree_qc_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return PedigreeQCResult(
        curated_pedigree=curated,
        edge_qc=edge_qc,
        sample_qc=sample_qc,
        similarity=similarity,
        embedding=embedding,
        long_edges=long_edges,
        summary=summary,
    )
