from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


PedigreeMode = Literal["off", "smooth", "kinship", "transmission"]


@dataclass(slots=True)
class PedigreeGraph:
    """Child-to-parent pedigree graph aligned to pipeline sample order."""

    parent_matrix: sparse.csr_matrix
    sample_ids: np.ndarray
    ignored_parent_ids: tuple[str, ...] = ()
    source: str = "provided"

    @property
    def n_samples(self) -> int:
        return int(self.parent_matrix.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.parent_matrix.nnz)

    @property
    def parent_lists(self) -> list[np.ndarray]:
        mat = self.parent_matrix.tocsr()
        return [
            mat.indices[mat.indptr[i] : mat.indptr[i + 1]].astype(np.int64, copy=False)
            for i in range(mat.shape[0])
        ]

    def summary(self) -> dict[str, object]:
        undirected = (self.parent_matrix + self.parent_matrix.T).tocsr()
        n_components, labels = connected_components(undirected, directed=False)
        component_sizes = np.bincount(labels, minlength=n_components).astype(np.int64, copy=False)
        return {
            "source": self.source,
            "n_samples": self.n_samples,
            "n_edges": self.n_edges,
            "n_components": int(n_components),
            "max_component_size": int(component_sizes.max()) if component_sizes.size else 0,
            "ignored_parent_ids": list(self.ignored_parent_ids),
        }


@dataclass(slots=True)
class PedigreeAdjustmentResult:
    dosage: np.ndarray
    genotype_posterior: np.ndarray | None
    mode: str
    summary: dict[str, object] = field(default_factory=dict)


def smooth_dosage_with_pedigree(
    dosage: np.ndarray,
    pedigree: sparse.csr_matrix | PedigreeGraph | None,
    strength: float,
) -> np.ndarray:
    if pedigree is None or strength <= 0.0:
        return dosage
    parent_matrix = pedigree.parent_matrix if isinstance(pedigree, PedigreeGraph) else pedigree
    parent_matrix = parent_matrix.tocsr()
    ds = dosage.astype(np.float32, copy=False)
    parent_degree = np.asarray(parent_matrix.sum(axis=1)).reshape(-1, 1).astype(np.float32, copy=False)
    has_parent = parent_degree > 0.0
    safe_degree = np.where(has_parent, parent_degree, 1.0).astype(np.float32, copy=False)
    parent_mean = (parent_matrix @ ds) / safe_degree
    # Samples without observed parents must remain unchanged. The previous
    # implementation divided by 1 for zero-degree rows and blended toward zero,
    # which incorrectly shrank founder/parent dosages during smoothing and
    # post-processing.
    blend = np.where(has_parent, np.clip(float(strength), 0.0, 1.0), 0.0).astype(np.float32, copy=False)
    return ((1.0 - blend) * ds + blend * parent_mean).astype(np.float32, copy=False)


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def _first_present(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    lower = {c.lower(): c for c in columns}
    for col in candidates:
        hit = lower.get(col.lower())
        if hit is not None:
            return hit
    return None


def has_pedigree_columns(
    table: pd.DataFrame,
    *,
    offspring_col: str = "sample_id",
    parent1_col: str = "father_id",
    parent2_col: str = "mother_id",
) -> bool:
    cols = set(table.columns)
    child_col = _first_present(cols, (offspring_col, "sample_id", "offspring", "child", "rfid", "iid"))
    p1_col = _first_present(cols, (parent1_col, "father_id", "father", "sire", "parent1", "parent_1", "dad"))
    p2_col = _first_present(cols, (parent2_col, "mother_id", "mother", "dam", "parent2", "parent_2", "mom"))
    return child_col is not None and (p1_col is not None or p2_col is not None)


def pedigree_table_to_graph(
    pedigree: pd.DataFrame,
    sample_ids: np.ndarray,
    *,
    offspring_col: str = "sample_id",
    parent1_col: str = "father_id",
    parent2_col: str = "mother_id",
    na_ids: tuple[object, ...] = ("", "NA", "NaN", "nan", "NAN", "n/a", "N/A", 0),
) -> PedigreeGraph:
    sample_ids = np.asarray(sample_ids, dtype=object)
    id_to_idx = {str(sample_id): i for i, sample_id in enumerate(sample_ids.tolist())}
    cols = set(pedigree.columns)
    child_col = _first_present(cols, (offspring_col, "sample_id", "offspring", "child", "rfid", "iid"))
    p1_col = _first_present(cols, (parent1_col, "father_id", "father", "sire", "parent1", "parent_1", "dad"))
    p2_col = _first_present(cols, (parent2_col, "mother_id", "mother", "dam", "parent2", "parent_2", "mom"))
    if child_col is None:
        raise ValueError(
            "Pedigree table needs an offspring/sample column. "
            f"Tried {offspring_col!r}, sample_id, offspring, child, rfid, iid."
        )
    parent_cols = [col for col in (p1_col, p2_col) if col is not None]
    if not parent_cols:
        raise ValueError(
            "Pedigree table needs at least one parent column. "
            f"Tried {parent1_col!r}/{parent2_col!r}, father_id/mother_id, sire/dam."
        )
    na_text = {str(x).strip().lower() for x in na_ids}
    rows: list[int] = []
    cols_idx: list[int] = []
    data: list[float] = []
    ignored: set[str] = set()
    for rec in pedigree[[child_col, *parent_cols]].itertuples(index=False, name=None):
        child = str(rec[0]).strip()
        child_idx = id_to_idx.get(child)
        if child_idx is None:
            continue
        parents = []
        for parent in rec[1:]:
            parent_text = str(parent).strip()
            if parent_text.lower() in na_text:
                continue
            parent_idx = id_to_idx.get(parent_text)
            if parent_idx is None:
                ignored.add(parent_text)
                continue
            parents.append(parent_idx)
        if not parents:
            continue
        for parent_idx in parents[:2]:
            rows.append(child_idx)
            cols_idx.append(parent_idx)
            data.append(0.5)
    mat = sparse.coo_matrix(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int32), np.asarray(cols_idx, dtype=np.int32)),
        ),
        shape=(sample_ids.shape[0], sample_ids.shape[0]),
    ).tocsr()
    mat.eliminate_zeros()
    return PedigreeGraph(
        parent_matrix=mat,
        sample_ids=sample_ids,
        ignored_parent_ids=tuple(sorted(ignored)),
        source="table",
    )


def coerce_pedigree_graph(
    pedigree: PedigreeGraph | sparse.spmatrix | pd.DataFrame | str | Path | None,
    sample_ids: np.ndarray,
    *,
    offspring_col: str = "sample_id",
    parent1_col: str = "father_id",
    parent2_col: str = "mother_id",
) -> PedigreeGraph | None:
    if pedigree is None:
        return None
    sample_ids = np.asarray(sample_ids, dtype=object)
    if isinstance(pedigree, PedigreeGraph):
        return pedigree
    if sparse.issparse(pedigree):
        mat = pedigree.tocsr().astype(np.float32)
        if mat.shape != (sample_ids.shape[0], sample_ids.shape[0]):
            raise ValueError(
                f"Pedigree sparse matrix shape {mat.shape} does not match "
                f"n_samples={sample_ids.shape[0]}."
            )
        return PedigreeGraph(parent_matrix=mat, sample_ids=sample_ids, source="sparse")
    if isinstance(pedigree, (str, Path)):
        pedigree = _read_table(pedigree)
    if isinstance(pedigree, pd.DataFrame):
        return pedigree_table_to_graph(
            pedigree,
            sample_ids,
            offspring_col=offspring_col,
            parent1_col=parent1_col,
            parent2_col=parent2_col,
        )
    raise TypeError(f"Unsupported pedigree type: {type(pedigree)!r}")


def _topological_order(parent_matrix: sparse.csr_matrix) -> tuple[list[int], list[np.ndarray]]:
    mat = parent_matrix.tocsr()
    n = int(mat.shape[0])
    parents = [mat.indices[mat.indptr[i] : mat.indptr[i + 1]].astype(np.int64, copy=False) for i in range(n)]
    remaining = set(range(n))
    order: list[int] = []
    while remaining:
        progressed = False
        for i in list(remaining):
            if all(int(p) not in remaining for p in parents[i]):
                order.append(i)
                remaining.remove(i)
                progressed = True
        if not progressed:
            # Pedigrees should be DAGs. If a cycle appears, keep deterministic order
            # and let loopy message passing handle the approximate inference.
            order.extend(sorted(remaining))
            break
    return order, parents


def ancestry_matrix_from_parent_matrix(
    parent_matrix: sparse.spmatrix,
    *,
    min_weight: float = 1e-6,
) -> sparse.csr_matrix:
    mat = parent_matrix.tocsr().astype(np.float32)
    order, parents = _topological_order(mat)
    n = int(mat.shape[0])
    rows: list[dict[int, float]] = [dict() for _ in range(n)]
    for i in order:
        row = rows[i]
        row[i] = row.get(i, 0.0) + 1.0
        for parent in parents[i]:
            weight = float(mat[i, int(parent)])
            for ancestor, value in rows[int(parent)].items():
                out = row.get(ancestor, 0.0) + weight * float(value)
                if abs(out) >= float(min_weight):
                    row[ancestor] = out
    indptr = [0]
    indices: list[int] = []
    data: list[float] = []
    for row in rows:
        items = sorted((j, v) for j, v in row.items() if abs(v) >= float(min_weight))
        indices.extend([j for j, _ in items])
        data.extend([v for _, v in items])
        indptr.append(len(indices))
    return sparse.csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(n, n),
    )


def kinship_from_parent_matrix(
    parent_matrix: sparse.spmatrix,
    *,
    threshold: float = 0.01,
) -> sparse.csr_matrix:
    ancestry = ancestry_matrix_from_parent_matrix(parent_matrix, min_weight=max(float(threshold) * 0.25, 1e-8))
    kinship = (0.5 * (ancestry @ ancestry.T)).tocsr()
    kinship.setdiag(0.0)
    if float(threshold) > 0.0:
        kinship.data[np.abs(kinship.data) < float(threshold)] = 0.0
    kinship.eliminate_zeros()
    return kinship.astype(np.float32)


def _posterior_dosage(gp: np.ndarray, ploidy: int = 2) -> np.ndarray:
    axis = np.arange(int(ploidy) + 1, dtype=np.float32)
    return np.sum(gp.astype(np.float32, copy=False) * axis[None, None, :], axis=2).astype(np.float32, copy=False)


def _posterior_from_dosage(dosage: np.ndarray, *, temperature: float = 0.25) -> np.ndarray:
    ds = dosage.astype(np.float32, copy=False)
    axis = np.arange(3, dtype=np.float32)
    logits = -((ds[..., None] - axis[None, None, :]) ** 2) / max(float(temperature), 1e-4)
    logits -= np.nanmax(logits, axis=2, keepdims=True)
    exp_logits = np.exp(logits)
    exp_logits = np.where(np.isfinite(exp_logits), exp_logits, 0.0)
    return (exp_logits / np.clip(exp_logits.sum(axis=2, keepdims=True), 1e-12, None)).astype(np.float32)


def _normalize_gp(gp: np.ndarray) -> np.ndarray:
    out = np.clip(gp.astype(np.float32, copy=False), 1e-8, 1.0)
    out /= np.clip(out.sum(axis=2, keepdims=True), 1e-12, None)
    return out.astype(np.float32, copy=False)


def _evidence_weights(gp: np.ndarray, support_mask: np.ndarray | None) -> np.ndarray:
    conf = np.max(np.nan_to_num(gp, nan=0.0), axis=2).astype(np.float32, copy=False)
    weights = 0.05 + 0.95 * np.clip((conf - 1.0 / 3.0) / (2.0 / 3.0), 0.0, 1.0)
    if support_mask is not None:
        support = support_mask.astype(bool, copy=False)
        weights = np.where(support, np.maximum(weights, 0.75), np.minimum(weights, 0.15))
    return weights.astype(np.float32, copy=False)


def _smooth_values_by_recombination(
    values: np.ndarray,
    positions: np.ndarray,
    generations: np.ndarray | float | None,
) -> np.ndarray:
    arr = values.astype(np.float32, copy=True)
    if arr.shape[1] <= 1:
        return arr
    pos = positions.astype(np.float64, copy=False)
    dist = np.maximum(np.diff(pos), 0.0)
    if generations is None:
        gen = np.full(arr.shape[0], 10.0, dtype=np.float32)
    elif np.isscalar(generations):
        gen = np.full(arr.shape[0], float(generations), dtype=np.float32)
    else:
        gen = np.asarray(generations, dtype=np.float32)
        if gen.shape[0] != arr.shape[0]:
            gen = np.full(arr.shape[0], float(np.nanmean(gen)), dtype=np.float32)
    # The stay probability is deliberately conservative: close markers share
    # transmission state strongly, distant markers are allowed to move.
    stay = np.exp(-np.clip(gen[:, None], 0.0, None) * dist[None, :] * 1e-8).astype(np.float32)
    fwd = arr.copy()
    for j in range(1, arr.shape[1]):
        s = stay[:, j - 1]
        fwd[:, j] = s * fwd[:, j - 1] + (1.0 - s) * arr[:, j]
    bwd = arr.copy()
    for j in range(arr.shape[1] - 2, -1, -1):
        s = stay[:, j]
        bwd[:, j] = s * bwd[:, j + 1] + (1.0 - s) * arr[:, j]
    return np.clip(0.5 * (fwd + bwd), 0.0, 1.0).astype(np.float32, copy=False)


def _child_distribution(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = np.clip(q1.astype(np.float32, copy=False), 1e-5, 1.0 - 1e-5)
    q2 = np.clip(q2.astype(np.float32, copy=False), 1e-5, 1.0 - 1e-5)
    out = np.empty(q1.shape + (3,), dtype=np.float32)
    out[..., 0] = (1.0 - q1) * (1.0 - q2)
    out[..., 1] = q1 * (1.0 - q2) + (1.0 - q1) * q2
    out[..., 2] = q1 * q2
    out /= np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)
    return out


def _kinship_dosage_prior(
    dosage: np.ndarray,
    evidence_weight: np.ndarray,
    kinship: sparse.csr_matrix,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples, n_positions = dosage.shape
    prior = np.full_like(dosage, np.nan, dtype=np.float32)
    strength = np.zeros((n_samples, n_positions), dtype=np.float32)
    kin = kinship.tocsr()
    for i in range(n_samples):
        start, end = kin.indptr[i], kin.indptr[i + 1]
        rel = kin.indices[start:end]
        w = kin.data[start:end].astype(np.float32, copy=False)
        if rel.size == 0:
            continue
        rel_ds = dosage[rel]
        rel_weight = w[:, None] * evidence_weight[rel]
        valid = np.isfinite(rel_ds)
        rel_weight = np.where(valid, rel_weight, 0.0)
        denom = rel_weight.sum(axis=0)
        good = denom > 1e-8
        if np.any(good):
            prior[i, good] = np.sum(rel_weight[:, good] * rel_ds[:, good], axis=0) / denom[good]
            strength[i, good] = np.clip(denom[good], 0.0, 1.0)
    return prior, strength


def apply_kinship_fallback(
    genotype_posterior: np.ndarray,
    graph: PedigreeGraph,
    *,
    support_mask: np.ndarray | None = None,
    strength: float = 0.5,
    kinship_threshold: float = 0.01,
) -> PedigreeAdjustmentResult:
    gp = _normalize_gp(genotype_posterior)
    dosage = _posterior_dosage(gp)
    evidence = _evidence_weights(gp, support_mask)
    kinship = kinship_from_parent_matrix(graph.parent_matrix, threshold=kinship_threshold)
    prior_ds, prior_strength = _kinship_dosage_prior(dosage, evidence, kinship)
    blend = np.clip(float(strength) * prior_strength, 0.0, 1.0)
    adjusted_ds = np.where(np.isfinite(prior_ds), (1.0 - blend) * dosage + blend * prior_ds, dosage)
    prior_gp = _posterior_from_dosage(np.clip(adjusted_ds, 0.0, 2.0))
    adjusted_gp = _normalize_gp((1.0 - blend[..., None]) * gp + blend[..., None] * prior_gp)
    return PedigreeAdjustmentResult(
        dosage=_posterior_dosage(adjusted_gp),
        genotype_posterior=adjusted_gp,
        mode="kinship",
        summary={
            **graph.summary(),
            "kinship_nnz": int(kinship.nnz),
            "mean_blend": float(np.mean(blend)),
            "max_blend": float(np.max(blend)) if blend.size else 0.0,
        },
    )


def apply_transmission_message_passing(
    genotype_posterior: np.ndarray,
    graph: PedigreeGraph,
    *,
    positions: np.ndarray,
    generations: np.ndarray | float | None = None,
    support_mask: np.ndarray | None = None,
    strength: float = 0.7,
    iterations: int = 4,
    kinship_threshold: float = 0.01,
) -> PedigreeAdjustmentResult:
    gp0 = _normalize_gp(genotype_posterior)
    gp = gp0.copy()
    parent_lists = graph.parent_lists
    child_lists: list[list[int]] = [[] for _ in range(graph.n_samples)]
    for child, parents in enumerate(parent_lists):
        for parent in parents.tolist():
            child_lists[int(parent)].append(int(child))
    evidence = _evidence_weights(gp0, support_mask)
    pop_q = np.nanmean(_posterior_dosage(gp0) / 2.0, axis=0).astype(np.float32, copy=False)
    pop_q = np.where(np.isfinite(pop_q), pop_q, 0.5).astype(np.float32, copy=False)
    n_messages_total = 0

    for _ in range(max(int(iterations), 1)):
        q_marker = _posterior_dosage(gp) / 2.0
        q_smooth = _smooth_values_by_recombination(q_marker, positions, generations)
        # Recombination should regularize noisy marker-wise messages, not erase
        # local genotype evidence. Keep most of the marker-local posterior and
        # use the chromosome smoother as a gentle prior.
        q = np.clip(0.85 * q_marker + 0.15 * q_smooth, 0.0, 1.0).astype(np.float32, copy=False)
        incoming = np.ones_like(gp, dtype=np.float32)
        counts = np.zeros(gp.shape[:2], dtype=np.float32)

        for child, parents in enumerate(parent_lists):
            if parents.size == 0:
                continue
            q_parents = []
            for parent in parents.tolist()[:2]:
                q_parents.append(q[int(parent)])
            if len(q_parents) == 1:
                q_parents.append(pop_q)
            child_msg = _child_distribution(q_parents[0], q_parents[1])
            incoming[child] *= child_msg
            counts[child] += 1.0
            n_messages_total += 1

        # Reverse messages allow sequenced offspring/siblings to inform latent parents.
        for parent, children in enumerate(child_lists):
            if not children:
                continue
            for child in children:
                parents = parent_lists[child].tolist()
                other = [p for p in parents if int(p) != int(parent)]
                other_q = q[int(other[0])] if other else pop_q
                child_gp = gp[child]
                likelihood = np.empty_like(child_gp, dtype=np.float32)
                for g in range(3):
                    parent_q = np.full_like(other_q, float(g) / 2.0, dtype=np.float32)
                    dist = _child_distribution(parent_q, other_q)
                    likelihood[:, g] = np.sum(child_gp * dist, axis=1)
                likelihood = _normalize_gp(likelihood[None, :, :])[0]
                incoming[parent] *= likelihood
                counts[parent] += 1.0
                n_messages_total += 1

        msg = _normalize_gp(incoming)
        if support_mask is not None:
            sample_has_direct_evidence = np.any(support_mask.astype(bool, copy=False), axis=1)
        else:
            sample_has_direct_evidence = np.max(evidence, axis=1) > 0.25
        # Keep sequenced/genotyped samples anchored to their STITCHCONT posterior.
        # Pedigree messages are most useful for no-read animals and latent
        # intermediates; over-powering observed samples can smear true signal.
        direct_scale = np.where(sample_has_direct_evidence[:, None], 0.30, 1.0).astype(np.float32)
        count_scale = np.clip(counts / (counts + 2.0), 0.0, 1.0)
        blend = np.clip(float(strength) * direct_scale * count_scale * (1.0 - np.clip(evidence, 0.0, 1.0)), 0.0, 0.85)
        combined = _normalize_gp((1.0 - blend[..., None]) * gp + blend[..., None] * msg)
        no_msg = counts <= 0.0
        gp = np.where(no_msg[..., None], gp, combined).astype(np.float32, copy=False)

    kin_result = apply_kinship_fallback(
        gp,
        graph,
        support_mask=support_mask,
        strength=min(float(strength), 0.35),
        kinship_threshold=kinship_threshold,
    )
    kin_blend = 0.15 * (1.0 - np.clip(evidence, 0.0, 1.0))
    gp = _normalize_gp((1.0 - kin_blend[..., None]) * gp + kin_blend[..., None] * kin_result.genotype_posterior)
    return PedigreeAdjustmentResult(
        dosage=_posterior_dosage(gp),
        genotype_posterior=gp,
        mode="transmission",
        summary={
            **graph.summary(),
            "iterations": int(max(int(iterations), 1)),
            "messages": int(n_messages_total),
            "kinship_fallback": kin_result.summary,
            "mean_evidence_weight": float(np.mean(evidence)),
        },
    )


def apply_pedigree_adjustment(
    *,
    dosage: np.ndarray,
    genotype_posterior: np.ndarray | None,
    pedigree: PedigreeGraph | sparse.spmatrix | None,
    mode: PedigreeMode,
    strength: float,
    positions: np.ndarray,
    generations: np.ndarray | float | None = None,
    support_mask: np.ndarray | None = None,
    iterations: int = 4,
    kinship_threshold: float = 0.01,
) -> PedigreeAdjustmentResult:
    if pedigree is None or mode == "off" or float(strength) <= 0.0:
        return PedigreeAdjustmentResult(dosage=dosage, genotype_posterior=genotype_posterior, mode="off")
    graph = pedigree if isinstance(pedigree, PedigreeGraph) else PedigreeGraph(pedigree.tocsr(), np.arange(dosage.shape[0]))
    if mode == "smooth":
        return PedigreeAdjustmentResult(
            dosage=smooth_dosage_with_pedigree(dosage, graph, strength),
            genotype_posterior=genotype_posterior,
            mode="smooth",
            summary=graph.summary(),
        )
    if genotype_posterior is None or genotype_posterior.shape[2] != 3:
        return PedigreeAdjustmentResult(
            dosage=smooth_dosage_with_pedigree(dosage, graph, strength),
            genotype_posterior=genotype_posterior,
            mode="smooth_fallback",
            summary={**graph.summary(), "reason": "requires diploid genotype posterior"},
        )
    if mode == "kinship":
        return apply_kinship_fallback(
            genotype_posterior,
            graph,
            support_mask=support_mask,
            strength=strength,
            kinship_threshold=kinship_threshold,
        )
    if mode == "transmission":
        return apply_transmission_message_passing(
            genotype_posterior,
            graph,
            positions=positions,
            generations=generations,
            support_mask=support_mask,
            strength=strength,
            iterations=iterations,
            kinship_threshold=kinship_threshold,
        )
    raise ValueError(f"Unknown pedigree mode: {mode!r}")
