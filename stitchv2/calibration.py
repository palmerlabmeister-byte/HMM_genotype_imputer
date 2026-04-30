from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(np.sum(exp_x, axis=axis, keepdims=True), 1e-12, None)


def posterior_confidence_metrics(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gp = posterior.astype(np.float32, copy=False)
    max_prob = np.max(gp, axis=2)
    top2 = np.partition(gp, kth=1, axis=2)[:, :, -2:]
    margin = np.clip(top2[:, :, 1] - top2[:, :, 0], 0.0, 1.0).astype(np.float32, copy=False)
    entropy = -np.sum(gp * np.log(np.clip(gp, 1e-12, 1.0)), axis=2) / np.log(max(gp.shape[2], 2))
    return (
        max_prob.astype(np.float32, copy=False),
        margin,
        entropy.astype(np.float32, copy=False),
    )


def dosage_to_genotype_posterior(
    dosage: np.ndarray,
    *,
    depth: np.ndarray | None = None,
    temperature: float = 0.35,
    min_prob: float = 1e-6,
    ploidy: int = 2,
) -> np.ndarray:
    dosage_f = dosage.astype(np.float32, copy=False)
    ploidy_i = max(int(ploidy), 0)
    geno_axis = np.arange(ploidy_i + 1, dtype=np.float32)
    if depth is None:
        temp = max(float(temperature), 1e-4)
        logits = -((dosage_f[..., None] - geno_axis[None, None, :]) ** 2) / temp
    else:
        depth_f = depth.astype(np.float32, copy=False)
        temp = max(float(temperature), 1e-4) / np.sqrt(np.clip(depth_f, 0.0, None) + 1.0)
        logits = -((dosage_f[..., None] - geno_axis[None, None, :]) ** 2) / np.clip(temp[..., None], 1e-6, None)
    post = _softmax(logits, axis=2).astype(np.float32, copy=False)
    if min_prob > 0.0:
        post = np.clip(post, float(min_prob), 1.0)
        post /= np.clip(post.sum(axis=2, keepdims=True), 1e-12, None)
    return post.astype(np.float32, copy=False)


def calibrate_genotype_posterior(
    raw_posterior: np.ndarray | None,
    *,
    dosage: np.ndarray,
    depth: np.ndarray | None = None,
    temperature: float = 0.35,
    blend: float = 0.35,
    min_prob: float = 1e-6,
    ploidy: int | None = None,
) -> np.ndarray:
    ploidy_i = int(raw_posterior.shape[2] - 1) if raw_posterior is not None else (2 if ploidy is None else int(ploidy))
    dosage_posterior = dosage_to_genotype_posterior(
        dosage,
        depth=depth,
        temperature=temperature,
        min_prob=min_prob,
        ploidy=ploidy_i,
    )
    if raw_posterior is None:
        return dosage_posterior
    weight = float(np.clip(blend, 0.0, 1.0))
    raw = raw_posterior.astype(np.float32, copy=False)
    mixed = (1.0 - weight) * raw + weight * dosage_posterior
    mixed = np.clip(mixed, float(min_prob), 1.0)
    mixed /= np.clip(mixed.sum(axis=2, keepdims=True), 1e-12, None)
    return mixed.astype(np.float32, copy=False)


def genotype_call_from_posterior(
    posterior: np.ndarray,
    *,
    min_confidence: float = 0.0,
    min_margin: float = 0.0,
    stitch_gp_threshold: float | None = None,
    call_correct_probability: np.ndarray | None = None,
    call_correct_threshold: float = 0.0,
    no_call_value: int = -1,
) -> np.ndarray:
    post = posterior.astype(np.float32, copy=False)
    call = np.argmax(post, axis=2).astype(np.int8, copy=False)
    conf, margin, _ = posterior_confidence_metrics(post)
    keep = np.ones(call.shape, dtype=bool)
    if stitch_gp_threshold is not None:
        keep &= conf >= float(stitch_gp_threshold)
    elif min_confidence > 0.0:
        keep &= conf >= float(min_confidence)
    if min_margin > 0.0:
        keep &= margin >= float(min_margin)
    if call_correct_probability is not None and call_correct_threshold > 0.0:
        prob = call_correct_probability.astype(np.float32, copy=False)
        if prob.shape != keep.shape:
            raise ValueError(
                f"call_correct_probability shape {prob.shape} does not match posterior calls shape {keep.shape}"
            )
        keep &= prob >= float(call_correct_threshold)
    call = np.where(keep, call, int(no_call_value)).astype(np.int8, copy=False)
    return call


def read_log_likelihood_from_posterior(
    *,
    ref_count: np.ndarray,
    alt_count: np.ndarray,
    other_count: np.ndarray | None,
    posterior: np.ndarray,
    sequencing_error_rate: float = 0.01,
) -> tuple[np.ndarray, float]:
    gp = posterior.astype(np.float32, copy=False)
    eps = float(sequencing_error_rate)
    p_other = eps / 3.0
    ploidy = max(gp.shape[2] - 1, 1)
    geno_axis = np.arange(gp.shape[2], dtype=np.float32)
    p_alt = np.clip(np.sum(gp * geno_axis[None, None, :], axis=2) / float(ploidy), 1e-6, 1.0 - 1e-6)
    p_ref = 1.0 - p_alt
    p_emit_ref = np.clip(p_ref * (1.0 - eps) + p_alt * p_other, 1e-6, 1.0)
    p_emit_alt = np.clip(p_alt * (1.0 - eps) + p_ref * p_other, 1e-6, 1.0)
    p_emit_other = np.clip(np.full_like(p_emit_ref, p_other), 1e-6, 1.0)
    ref = ref_count.astype(np.float32, copy=False)
    alt = alt_count.astype(np.float32, copy=False)
    oth = np.zeros_like(ref) if other_count is None else other_count.astype(np.float32, copy=False)
    log_lik = ref * np.log(p_emit_ref) + alt * np.log(p_emit_alt) + oth * np.log(p_emit_other)
    return log_lik.astype(np.float32, copy=False), float(np.sum(log_lik))


def build_calibration_feature_matrix(
    *,
    dosage: np.ndarray,
    depth: np.ndarray | None = None,
    alt_fraction: np.ndarray | None = None,
    switch_probability: np.ndarray | None = None,
    recombination_rate: np.ndarray | None = None,
    generations: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    ds = dosage.astype(np.float32, copy=False)
    n_samples, n_positions = ds.shape
    cols: list[np.ndarray] = [ds.reshape(-1)]
    names = ["dosage"]
    if depth is not None:
        cols.append(depth.astype(np.float32, copy=False).reshape(-1))
        names.append("depth")
    if alt_fraction is not None:
        cols.append(alt_fraction.astype(np.float32, copy=False).reshape(-1))
        names.append("alt_fraction")
    if switch_probability is not None:
        cols.append(switch_probability.astype(np.float32, copy=False).reshape(-1))
        names.append("switch_probability")
    if recombination_rate is not None:
        rr = recombination_rate.astype(np.float32, copy=False)[None, :]
        cols.append(np.broadcast_to(rr, (n_samples, n_positions)).reshape(-1))
        names.append("recombination_rate")
    if generations is not None:
        g = generations.astype(np.float32, copy=False)[:, None]
        cols.append(np.broadcast_to(g, (n_samples, n_positions)).reshape(-1))
        names.append("generation")
    features = np.stack(cols, axis=1).astype(np.float32, copy=False)
    return features, names


def compute_variant_maf_from_posterior(posterior: np.ndarray) -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    ploidy = max(gp.shape[2] - 1, 1)
    geno_axis = np.arange(gp.shape[2], dtype=np.float32)
    ds = np.sum(gp * geno_axis[None, None, :], axis=2)
    af = np.mean(ds, axis=0) / float(ploidy)
    maf = np.minimum(af, 1.0 - af)
    return np.clip(maf, 0.0, 0.5).astype(np.float32, copy=False)


def compute_variant_strata(
    maf: np.ndarray,
    info: np.ndarray,
    *,
    maf_bins: Sequence[float] = (0.0, 0.01, 0.05, 0.2, 0.5),
    info_bins: Sequence[float] = (-5.0, 0.2, 0.5, 0.8, 1.1),
) -> tuple[np.ndarray, dict[str, object]]:
    maf_arr = np.asarray(maf, dtype=np.float32)
    info_arr = np.asarray(info, dtype=np.float32)
    maf_edges = np.asarray(maf_bins, dtype=np.float32)
    info_edges = np.asarray(info_bins, dtype=np.float32)
    if maf_edges.ndim != 1 or maf_edges.size < 2:
        raise ValueError("maf_bins must provide at least 2 ordered edges.")
    if info_edges.ndim != 1 or info_edges.size < 2:
        raise ValueError("info_bins must provide at least 2 ordered edges.")
    maf_idx = np.digitize(np.clip(maf_arr, maf_edges[0], maf_edges[-1]), maf_edges[1:-1], right=False).astype(np.int16)
    info_fill = np.where(np.isfinite(info_arr), info_arr, float(info_edges[0]))
    info_idx = np.digitize(np.clip(info_fill, info_edges[0], info_edges[-1]), info_edges[1:-1], right=False).astype(np.int16)
    n_info_bins = int(max(info_edges.size - 1, 1))
    strata = (maf_idx.astype(np.int32) * n_info_bins + info_idx.astype(np.int32)).astype(np.int32, copy=False)
    meta = {
        "maf_bins": maf_edges.tolist(),
        "info_bins": info_edges.tolist(),
        "n_strata": int((maf_edges.size - 1) * (info_edges.size - 1)),
    }
    return strata, meta


def _balanced_row_indices(y: np.ndarray, *, max_rows: int, seed: int) -> np.ndarray:
    y_arr = y.astype(np.int32, copy=False).reshape(-1)
    n_rows = int(y_arr.shape[0])
    if max_rows <= 0 or n_rows <= max_rows:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    classes = np.unique(y_arr)
    takes: list[np.ndarray] = []
    per_cls = max(int(max_rows // max(classes.size, 1)), 1)
    for cls in classes.tolist():
        idx = np.flatnonzero(y_arr == cls)
        if idx.size == 0:
            continue
        n_take = min(int(idx.size), per_cls)
        if n_take < idx.size:
            idx = rng.choice(idx, size=n_take, replace=False)
        takes.append(np.asarray(idx, dtype=np.int64))
    if not takes:
        return np.asarray(rng.choice(n_rows, size=int(max_rows), replace=False), dtype=np.int64)
    picked = np.concatenate(takes, axis=0)
    if picked.size < int(max_rows):
        missing = int(max_rows - picked.size)
        remain = np.setdiff1d(np.arange(n_rows, dtype=np.int64), picked, assume_unique=False)
        if remain.size > 0:
            extra_n = min(int(remain.size), missing)
            extra = rng.choice(remain, size=extra_n, replace=False)
            picked = np.concatenate([picked, extra.astype(np.int64, copy=False)], axis=0)
    if picked.size > int(max_rows):
        picked = rng.choice(picked, size=int(max_rows), replace=False)
    return np.asarray(picked, dtype=np.int64)


def _subsample_rows_balanced(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    take = _balanced_row_indices(y, max_rows=int(max_rows), seed=int(seed))
    return x[take], y[take]


def build_readaware_calibration_feature_matrix(
    *,
    dosage: np.ndarray,
    posterior: np.ndarray,
    depth: np.ndarray | None = None,
    ref_count: np.ndarray | None = None,
    alt_count: np.ndarray | None = None,
    other_count: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    switch_probability: np.ndarray | None = None,
    recombination_rate: np.ndarray | None = None,
    generations: np.ndarray | None = None,
    sample_metadata: np.ndarray | None = None,
    variant_maf: np.ndarray | None = None,
    variant_info: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    ds = dosage.astype(np.float32, copy=False)
    gp = posterior.astype(np.float32, copy=False)
    n_samples, n_positions = ds.shape
    if gp.shape[:2] != ds.shape or gp.shape[2] != 3:
        raise ValueError(f"posterior shape {gp.shape} must match dosage shape {(n_samples, n_positions, 3)}")

    depth_arr = np.zeros_like(ds, dtype=np.float32) if depth is None else depth.astype(np.float32, copy=False)
    ref_arr = np.zeros_like(ds, dtype=np.float32) if ref_count is None else ref_count.astype(np.float32, copy=False)
    alt_arr = np.zeros_like(ds, dtype=np.float32) if alt_count is None else alt_count.astype(np.float32, copy=False)
    oth_arr = np.zeros_like(ds, dtype=np.float32) if other_count is None else other_count.astype(np.float32, copy=False)
    total = ref_arr + alt_arr + oth_arr
    alt_fraction = (alt_arr + 0.5) / np.clip(total + 1.0, 1.0, None)
    support = (total > 0.0).astype(np.float32, copy=False) if support_mask is None else support_mask.astype(np.float32, copy=False)

    conf, margin, entropy = posterior_confidence_metrics(gp)
    post_var = np.clip((gp[:, :, 1] + 4.0 * gp[:, :, 2]) - (ds * ds), 0.0, None).astype(np.float32, copy=False)
    log_depth = np.log1p(np.clip(depth_arr, 0.0, None)).astype(np.float32, copy=False)

    cols: list[np.ndarray] = []
    names: list[str] = []
    def _append(name: str, arr: np.ndarray) -> None:
        cols.append(arr.reshape(-1).astype(np.float32, copy=False))
        names.append(name)

    _append("dosage", ds)
    _append("depth", depth_arr)
    _append("log_depth", log_depth)
    _append("alt_fraction", alt_fraction)
    _append("support_mask", support)
    _append("ref_count", ref_arr)
    _append("alt_count", alt_arr)
    _append("other_count", oth_arr)
    _append("total_count", total)
    _append("gp0", gp[:, :, 0])
    _append("gp1", gp[:, :, 1])
    _append("gp2", gp[:, :, 2])
    _append("max_prob", conf)
    _append("margin", margin)
    _append("entropy", entropy)
    _append("posterior_var", post_var)

    if switch_probability is not None:
        _append("switch_probability", switch_probability.astype(np.float32, copy=False))
    if recombination_rate is not None:
        rr = recombination_rate.astype(np.float32, copy=False)[None, :]
        _append("recombination_rate", np.broadcast_to(rr, ds.shape))
    if generations is not None:
        gen = generations.astype(np.float32, copy=False)[:, None]
        _append("generation", np.broadcast_to(gen, ds.shape))

    if variant_maf is None:
        variant_maf = compute_variant_maf_from_posterior(gp)
    maf_col = np.broadcast_to(variant_maf.astype(np.float32, copy=False)[None, :], ds.shape)
    _append("variant_maf", maf_col)

    if variant_info is None:
        variant_info = compute_info_score_per_variant(gp)
    info_col = np.broadcast_to(variant_info.astype(np.float32, copy=False)[None, :], ds.shape)
    _append("variant_info", info_col)

    if sample_metadata is not None and sample_metadata.size > 0:
        sm = sample_metadata.astype(np.float32, copy=False)
        for j in range(sm.shape[1]):
            col = np.broadcast_to(sm[:, j][:, None], ds.shape)
            _append(f"sample_meta_{j}", col)

    features = np.stack(cols, axis=1).astype(np.float32, copy=False)
    return features, names


def train_lightgbm_multiclass_calibrator(
    features: np.ndarray,
    target: np.ndarray,
    *,
    seed: int = 0,
    n_estimators: int = 180,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    num_leaves: int = 47,
    min_data_in_leaf: int = 48,
    class_weight_mode: str = "none",
) -> Any:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise ImportError(
            "LightGBM post-calibration requested but 'lightgbm' is not available in the active conda env."
        ) from exc

    y = target.astype(np.int32, copy=False).reshape(-1)
    x = features.astype(np.float32, copy=False)
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=int(seed),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        num_leaves=int(num_leaves),
        min_data_in_leaf=int(min_data_in_leaf),
        class_weight=("balanced" if str(class_weight_mode).lower() == "balanced" else None),
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def predict_lightgbm_posterior(model: Any, features: np.ndarray, *, min_prob: float = 1e-6) -> np.ndarray:
    x = features.astype(np.float32, copy=False)
    proba = model.predict_proba(x)
    if isinstance(proba, list):
        proba = np.stack(proba, axis=1)
    proba_arr = np.asarray(proba, dtype=np.float32)
    if proba_arr.ndim != 2 or proba_arr.shape[1] != 3:
        raise ValueError(f"Expected LightGBM predict_proba output of shape (n, 3), got {proba_arr.shape}")
    proba_arr = np.clip(proba_arr, float(min_prob), 1.0)
    proba_arr /= np.clip(np.sum(proba_arr, axis=1, keepdims=True), 1e-12, None)
    return proba_arr


def train_lightgbm_binary_correctness_model(
    features: np.ndarray,
    correct_target: np.ndarray,
    *,
    seed: int = 0,
    n_estimators: int = 140,
    learning_rate: float = 0.05,
    max_depth: int = -1,
    num_leaves: int = 47,
    min_data_in_leaf: int = 96,
) -> Any:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise ImportError(
            "LightGBM call-correctness calibration requested but 'lightgbm' is not available in the active conda env."
        ) from exc

    x = features.astype(np.float32, copy=False)
    y = correct_target.astype(np.int32, copy=False).reshape(-1)
    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=int(seed),
        n_estimators=int(n_estimators),
        learning_rate=float(learning_rate),
        max_depth=int(max_depth),
        num_leaves=int(num_leaves),
        min_data_in_leaf=int(min_data_in_leaf),
        class_weight="balanced",
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def predict_lightgbm_binary_probability(model: Any, features: np.ndarray) -> np.ndarray:
    x = features.astype(np.float32, copy=False)
    proba = model.predict_proba(x)
    if isinstance(proba, list):
        proba = np.stack(proba, axis=1)
    arr = np.asarray(proba, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected binary LightGBM predict_proba shape (n,2), got {arr.shape}")
    return np.clip(arr[:, 1], 0.0, 1.0).astype(np.float32, copy=False)


def train_stratified_lightgbm_multiclass_calibrator(
    *,
    features: np.ndarray,
    target: np.ndarray,
    position_index: np.ndarray,
    variant_strata: np.ndarray,
    seed: int = 0,
    max_rows: int = 750_000,
    min_rows_per_stratum: int = 2_000,
    class_weight_mode: str = "balanced",
) -> tuple[Any, dict[int, Any], dict[str, object]]:
    x = features.astype(np.float32, copy=False)
    y = target.astype(np.int32, copy=False).reshape(-1)
    pos_idx = position_index.astype(np.int64, copy=False).reshape(-1)
    valid = (y >= 0) & (y <= 2) & (pos_idx >= 0) & (pos_idx < variant_strata.shape[0])
    if not np.any(valid):
        raise ValueError("No valid rows to train stratified multiclass calibrator.")
    x = x[valid]
    y = y[valid]
    pos_idx = pos_idx[valid]
    take = _balanced_row_indices(y, max_rows=int(max_rows), seed=int(seed))
    x = x[take]
    y = y[take]
    pos_idx = pos_idx[take]
    if x.shape[0] <= 0:
        raise ValueError("No rows available after subsampling.")
    if np.unique(y).size < 2:
        raise ValueError("Need at least two classes for multiclass calibration.")

    global_model = train_lightgbm_multiclass_calibrator(
        x,
        y,
        seed=int(seed),
        class_weight_mode=class_weight_mode,
    )

    strata_models: dict[int, Any] = {}
    strata_for_rows = variant_strata[pos_idx]
    unique_strata = np.unique(strata_for_rows.astype(np.int32, copy=False))
    trained_strata = 0
    for s in unique_strata.tolist():
        m = strata_for_rows == int(s)
        if int(np.sum(m)) < int(min_rows_per_stratum):
            continue
        ys = y[m]
        if np.unique(ys).size < 2:
            continue
        xs = x[m]
        model_s = train_lightgbm_multiclass_calibrator(
            xs,
            ys,
            seed=int(seed) + int(s) + 17,
            class_weight_mode=class_weight_mode,
        )
        strata_models[int(s)] = model_s
        trained_strata += 1

    meta = {
        "n_train_rows": int(x.shape[0]),
        "n_unique_strata_seen": int(unique_strata.size),
        "n_trained_strata": int(trained_strata),
    }
    return global_model, strata_models, meta


def predict_stratified_lightgbm_posterior(
    *,
    features: np.ndarray,
    position_index: np.ndarray,
    variant_strata: np.ndarray,
    global_model: Any,
    strata_models: dict[int, Any] | None = None,
    min_prob: float = 1e-6,
) -> np.ndarray:
    x = features.astype(np.float32, copy=False)
    pos_idx = position_index.astype(np.int64, copy=False).reshape(-1)
    if x.shape[0] != pos_idx.shape[0]:
        raise ValueError("features and position_index must have matching number of rows.")
    base = predict_lightgbm_posterior(global_model, x, min_prob=float(min_prob))
    if not strata_models:
        return base
    strata = variant_strata[np.clip(pos_idx, 0, variant_strata.shape[0] - 1)]
    out = base.copy()
    for s, model in strata_models.items():
        m = strata == int(s)
        if not np.any(m):
            continue
        out[m] = predict_lightgbm_posterior(model, x[m], min_prob=float(min_prob))
    out = np.clip(out, float(min_prob), 1.0)
    out /= np.clip(np.sum(out, axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32, copy=False)


def fit_multiclass_isotonic_calibrator(
    probability: np.ndarray,
    target: np.ndarray,
) -> list[Any]:
    try:
        from sklearn.isotonic import IsotonicRegression
    except Exception:
        return [None, None, None]
    p = probability.astype(np.float32, copy=False)
    y = target.astype(np.int32, copy=False).reshape(-1)
    models: list[Any] = []
    for cls in (0, 1, 2):
        y_cls = (y == cls).astype(np.float32, copy=False)
        if np.unique(y_cls).size < 2 or np.unique(p[:, cls]).size < 2:
            models.append(None)
            continue
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(p[:, cls], y_cls)
        models.append(model)
    return models


def apply_multiclass_isotonic_calibrator(
    probability: np.ndarray,
    models: Sequence[Any],
    *,
    min_prob: float = 1e-6,
) -> np.ndarray:
    p = probability.astype(np.float32, copy=False)
    if len(models) != 3:
        return p
    out = p.copy()
    for cls in (0, 1, 2):
        model = models[cls]
        if model is None:
            continue
        out[:, cls] = np.asarray(model.transform(p[:, cls]), dtype=np.float32)
    out = np.clip(out, float(min_prob), 1.0)
    out /= np.clip(np.sum(out, axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32, copy=False)


def optimize_call_correctness_threshold(
    *,
    correctness_probability: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray | None = None,
    min_call_rate: float = 0.2,
) -> tuple[float, float, float]:
    prob = correctness_probability.astype(np.float32, copy=False).reshape(-1)
    yt = y_true.astype(np.int8, copy=False).reshape(-1)
    yp = y_pred.astype(np.int8, copy=False).reshape(-1)
    valid = (yt >= 0) & (yp >= 0)
    if not np.any(valid):
        return 0.0, float("nan"), 0.0
    prob = prob[valid]
    yt = yt[valid]
    yp = yp[valid]
    if thresholds is None:
        thresholds = np.linspace(0.0, 0.99, 100, dtype=np.float32)
    best_thr = 0.0
    best_score = -np.inf
    best_call_rate = 1.0
    for thr in thresholds.tolist():
        called = prob >= float(thr)
        call_rate = float(np.mean(called))
        if call_rate < float(min_call_rate):
            continue
        if not np.any(called):
            continue
        ytc = yt[called]
        ypc = yp[called]
        recalls = []
        for cls in (0, 1, 2):
            m = ytc == cls
            if not np.any(m):
                continue
            recalls.append(float(np.mean(ypc[m] == cls)))
        if not recalls:
            continue
        bacc = float(np.mean(recalls))
        # Prefer high balanced accuracy while preserving useful call-rate.
        score = bacc + 0.05 * call_rate
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_call_rate = call_rate
    if not np.isfinite(best_score):
        return 0.0, float("nan"), 1.0
    return best_thr, float(best_score), float(best_call_rate)


def compute_info_score_per_variant(
    posterior: np.ndarray,
    *,
    min_denominator: float = 1e-8,
) -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    ploidy = max(gp.shape[2] - 1, 1)
    geno_axis = np.arange(gp.shape[2], dtype=np.float32)
    ds = np.sum(gp * geno_axis[None, None, :], axis=2)
    e2 = np.sum(gp * (geno_axis[None, None, :] ** 2), axis=2)
    post_var = np.clip(e2 - ds * ds, 0.0, None)
    p = np.mean(ds, axis=0) / float(ploidy)
    denom = float(ploidy) * p * (1.0 - p)
    info = np.full(ds.shape[1], np.nan, dtype=np.float32)
    ok = denom > float(min_denominator)
    if np.any(ok):
        info[ok] = 1.0 - (np.mean(post_var[:, ok], axis=0) / np.clip(denom[ok], float(min_denominator), None))
    return np.clip(info, -5.0, 1.0).astype(np.float32, copy=False)


def build_sample_metadata_features(
    samples_df: pd.DataFrame | None,
    *,
    sample_order: np.ndarray | None = None,
    exclude_columns: tuple[str, ...] = ("sample_id", "bam_path"),
    max_categories: int = 32,
) -> tuple[np.ndarray | None, list[str]]:
    if samples_df is None:
        return None, []
    samples = samples_df.copy()
    if "sample_id" in samples.columns:
        samples["sample_id"] = samples["sample_id"].astype(str)
    if sample_order is not None and "sample_id" in samples.columns:
        samples = samples.set_index("sample_id")
        samples = samples.reindex(pd.Index(sample_order.astype(str)))
        samples = samples.reset_index(names="sample_id")

    features: list[np.ndarray] = []
    names: list[str] = []
    for col in samples.columns:
        if col in exclude_columns:
            continue
        series = samples[col]
        if pd.api.types.is_numeric_dtype(series):
            arr = series.to_numpy(dtype=np.float32, copy=False)
            if np.any(~np.isfinite(arr)):
                fill = float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else 0.0
                arr = np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)
            features.append(arr[:, None].astype(np.float32, copy=False))
            names.append(str(col))
            continue
        uniq = series.astype(str).nunique(dropna=True)
        if uniq == 0 or uniq > int(max_categories):
            continue
        dummies = pd.get_dummies(series.astype("category"), prefix=str(col), dtype=np.uint8)
        if dummies.shape[1] == 0:
            continue
        features.append(dummies.to_numpy(dtype=np.float32, copy=False))
        names.extend(dummies.columns.astype(str).tolist())
    if not features:
        return None, []
    return np.concatenate(features, axis=1).astype(np.float32, copy=False), names


def build_block_context_features(
    *,
    dosage: np.ndarray,
    posterior: np.ndarray | None,
    position_index: np.ndarray,
    window: int,
    sample_metadata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = dosage.astype(np.float32, copy=False)
    n_samples, n_positions = ds.shape
    pos_idx = position_index.astype(np.int64, copy=False)
    if posterior is None:
        gp = dosage_to_genotype_posterior(ds, temperature=0.35)
    else:
        gp = posterior.astype(np.float32, copy=False)
    win = max(int(window), 0)
    offsets = np.arange(-win, win + 1, dtype=np.int64)
    context_idx = np.clip(pos_idx[None, :] + offsets[:, None], 0, n_positions - 1)
    n_block = int(pos_idx.shape[0])
    n_window = int(context_idx.shape[0])

    ds_ctx = np.transpose(ds[:, context_idx], (0, 2, 1)).reshape(n_samples * n_block, n_window)
    gp1_ctx = np.transpose(gp[:, :, 1][:, context_idx], (0, 2, 1)).reshape(n_samples * n_block, n_window)
    gp2_ctx = np.transpose(gp[:, :, 2][:, context_idx], (0, 2, 1)).reshape(n_samples * n_block, n_window)

    center_ds = ds[:, pos_idx].reshape(-1, 1).astype(np.float32, copy=False)
    center_gp = gp[:, pos_idx, :].reshape(-1, 3).astype(np.float32, copy=False)
    local_mean = np.mean(ds_ctx, axis=1, keepdims=True).astype(np.float32, copy=False)
    local_std = np.std(ds_ctx, axis=1, keepdims=True).astype(np.float32, copy=False)

    x_parts = [center_ds, center_gp, local_mean, local_std, ds_ctx, gp1_ctx, gp2_ctx]
    if sample_metadata is not None and sample_metadata.size > 0:
        x_parts.append(np.repeat(sample_metadata.astype(np.float32, copy=False), n_block, axis=0))
    x = np.concatenate(x_parts, axis=1).astype(np.float32, copy=False)
    sample_idx = np.repeat(np.arange(n_samples, dtype=np.int64), n_block)
    pos_out = np.tile(pos_idx, n_samples)
    return x, sample_idx, pos_out


def tune_lightgbm_classifier_optuna(
    features: np.ndarray,
    target: np.ndarray,
    *,
    seed: int = 0,
    n_trials: int = 20,
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
        import optuna
        from sklearn.metrics import balanced_accuracy_score
        from sklearn.model_selection import train_test_split
    except Exception:
        return {}

    y = target.astype(np.int32, copy=False).reshape(-1)
    x = features.astype(np.float32, copy=False)
    if x.shape[0] < 2000 or np.unique(y).size < 2:
        return {}
    x_train, x_valid, y_train, y_valid = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=int(seed),
        stratify=y,
    )

    def objective(trial: Any) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 120, 500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 16, 128),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
        }
        model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            random_state=int(seed),
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_valid)
        return float(balanced_accuracy_score(y_valid, pred))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=int(seed)))
    study.optimize(objective, n_trials=max(int(n_trials), 1), show_progress_bar=False)
    return dict(study.best_params)


def calibrate_genotype_posterior_block_context(
    *,
    raw_posterior: np.ndarray | None,
    dosage: np.ndarray,
    truth_genotype: np.ndarray,
    samples_df: pd.DataFrame | None = None,
    train_position_index: np.ndarray | None = None,
    predict_position_index: np.ndarray | None = None,
    window: int = 25,
    block_size: int = 64,
    max_train_rows: int = 750_000,
    use_optuna: bool = False,
    optuna_trials: int = 20,
    seed: int = 0,
    min_prob: float = 1e-6,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        import lightgbm as lgb
    except Exception as exc:  # pragma: no cover - optional dependency.
        raise ImportError(
            "LightGBM block-context calibration requested but 'lightgbm' is not available in this environment."
        ) from exc

    ds = dosage.astype(np.float32, copy=False)
    n_samples, n_positions = ds.shape
    gp_base = (
        dosage_to_genotype_posterior(ds, temperature=0.35)
        if raw_posterior is None
        else raw_posterior.astype(np.float32, copy=False)
    )
    y_truth = truth_genotype.astype(np.int16, copy=False)

    if train_position_index is None:
        train_position_index = np.arange(n_positions, dtype=np.int64)
    else:
        train_position_index = train_position_index.astype(np.int64, copy=False)
    if predict_position_index is None:
        predict_position_index = np.arange(n_positions, dtype=np.int64)
    else:
        predict_position_index = predict_position_index.astype(np.int64, copy=False)

    meta, _ = build_sample_metadata_features(
        samples_df,
        sample_order=(samples_df["sample_id"].astype(str).to_numpy() if samples_df is not None and "sample_id" in samples_df.columns else None),
    )
    n_blocks_train = max((len(train_position_index) + max(int(block_size), 1) - 1) // max(int(block_size), 1), 1)
    per_block_cap = max(int(max_train_rows // n_blocks_train), 2000) if max_train_rows > 0 else 0
    rng = np.random.default_rng(int(seed))

    x_chunks: list[np.ndarray] = []
    y_chunks: list[np.ndarray] = []
    for start in range(0, len(train_position_index), max(int(block_size), 1)):
        pos_block = train_position_index[start : start + max(int(block_size), 1)]
        if pos_block.size == 0:
            continue
        x_blk, s_idx, p_idx = build_block_context_features(
            dosage=ds,
            posterior=gp_base,
            position_index=pos_block,
            window=int(window),
            sample_metadata=meta,
        )
        y_blk = y_truth[s_idx, p_idx]
        valid = (y_blk >= 0) & (y_blk <= 2)
        if not np.any(valid):
            continue
        x_blk = x_blk[valid]
        y_blk = y_blk[valid].astype(np.int32, copy=False)
        if per_block_cap > 0 and x_blk.shape[0] > per_block_cap:
            take = rng.choice(x_blk.shape[0], size=per_block_cap, replace=False)
            x_blk = x_blk[take]
            y_blk = y_blk[take]
        x_chunks.append(x_blk.astype(np.float32, copy=False))
        y_chunks.append(y_blk.astype(np.int32, copy=False))
    if not x_chunks:
        return gp_base.copy(), {"status": "no_training_rows"}

    x_train = np.concatenate(x_chunks, axis=0).astype(np.float32, copy=False)
    y_train = np.concatenate(y_chunks, axis=0).astype(np.int32, copy=False)
    if np.unique(y_train).size < 2:
        return gp_base.copy(), {"status": "single_class_training"}

    params: dict[str, Any] = {
        "n_estimators": 320,
        "learning_rate": 0.05,
        "max_depth": -1,
        "num_leaves": 63,
        "min_data_in_leaf": 48,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
    }
    if use_optuna:
        tuned = tune_lightgbm_classifier_optuna(
            x_train,
            y_train,
            seed=int(seed),
            n_trials=max(int(optuna_trials), 1),
        )
        if tuned:
            params.update(tuned)

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=int(seed),
        n_jobs=1,
        verbosity=-1,
        **params,
    )
    model.fit(x_train, y_train)

    gp_out = gp_base.copy()
    for start in range(0, len(predict_position_index), max(int(block_size), 1)):
        pos_block = predict_position_index[start : start + max(int(block_size), 1)]
        if pos_block.size == 0:
            continue
        x_blk, s_idx, p_idx = build_block_context_features(
            dosage=ds,
            posterior=gp_base,
            position_index=pos_block,
            window=int(window),
            sample_metadata=meta,
        )
        pred = predict_lightgbm_posterior(model, x_blk, min_prob=float(min_prob))
        gp_out[s_idx, p_idx, :] = pred
    gp_out = np.clip(gp_out, float(min_prob), 1.0)
    gp_out /= np.clip(np.sum(gp_out, axis=2, keepdims=True), 1e-12, None)
    summary = {
        "status": "ok",
        "n_train_rows": int(x_train.shape[0]),
        "window": int(window),
        "block_size": int(block_size),
        "params": params,
    }
    return gp_out.astype(np.float32, copy=False), summary


def build_hardcall_quality_feature_matrix(
    *,
    dosage: np.ndarray,
    posterior: np.ndarray,
    depth: np.ndarray | None = None,
    ref_count: np.ndarray | None = None,
    alt_count: np.ndarray | None = None,
    other_count: np.ndarray | None = None,
    generations: np.ndarray | None = None,
    variant_maf: np.ndarray | None = None,
    variant_info: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    ds = dosage.astype(np.float32, copy=False)
    gp = posterior.astype(np.float32, copy=False)
    n_samples, n_positions = ds.shape
    conf, margin, entropy = posterior_confidence_metrics(gp)
    depth_arr = np.zeros_like(ds, dtype=np.float32) if depth is None else depth.astype(np.float32, copy=False)
    ref_arr = np.zeros_like(ds, dtype=np.float32) if ref_count is None else ref_count.astype(np.float32, copy=False)
    alt_arr = np.zeros_like(ds, dtype=np.float32) if alt_count is None else alt_count.astype(np.float32, copy=False)
    oth_arr = np.zeros_like(ds, dtype=np.float32) if other_count is None else other_count.astype(np.float32, copy=False)
    total = ref_arr + alt_arr + oth_arr
    alt_fraction = (alt_arr + 0.5) / np.clip(total + 1.0, 1.0, None)
    post_var = np.clip((gp[:, :, 1] + 4.0 * gp[:, :, 2]) - (ds * ds), 0.0, None).astype(np.float32, copy=False)
    if variant_maf is None:
        variant_maf = compute_variant_maf_from_posterior(gp)
    if variant_info is None:
        variant_info = compute_info_score_per_variant(gp)
    maf_col = np.broadcast_to(variant_maf.astype(np.float32, copy=False)[None, :], (n_samples, n_positions))
    info_col = np.broadcast_to(variant_info.astype(np.float32, copy=False)[None, :], (n_samples, n_positions))

    cols = [
        ds.reshape(-1),
        depth_arr.reshape(-1),
        np.log1p(np.clip(depth_arr, 0.0, None)).reshape(-1),
        alt_fraction.reshape(-1),
        ref_arr.reshape(-1),
        alt_arr.reshape(-1),
        oth_arr.reshape(-1),
        total.reshape(-1),
        gp[:, :, 0].reshape(-1),
        gp[:, :, 1].reshape(-1),
        gp[:, :, 2].reshape(-1),
        conf.reshape(-1),
        margin.reshape(-1),
        entropy.reshape(-1),
        post_var.reshape(-1),
        maf_col.reshape(-1),
        info_col.reshape(-1),
    ]
    names = [
        "dosage",
        "depth",
        "log_depth",
        "alt_fraction",
        "ref_count",
        "alt_count",
        "other_count",
        "total_count",
        "gp0",
        "gp1",
        "gp2",
        "max_prob",
        "margin",
        "entropy",
        "posterior_var",
        "variant_maf",
        "variant_info",
    ]
    if generations is not None:
        gen = generations.astype(np.float32, copy=False)[:, None]
        cols.append(np.broadcast_to(gen, (n_samples, n_positions)).reshape(-1))
        names.append("generation")
    features = np.stack(cols, axis=1).astype(np.float32, copy=False)
    return features, names


def calibrate_genotype_posterior_full_stack(
    *,
    raw_posterior: np.ndarray | None,
    dosage: np.ndarray,
    truth_genotype: np.ndarray,
    depth: np.ndarray | None = None,
    ref_count: np.ndarray | None = None,
    alt_count: np.ndarray | None = None,
    other_count: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    generations: np.ndarray | None = None,
    samples_df: pd.DataFrame | None = None,
    train_position_index: np.ndarray | None = None,
    predict_position_index: np.ndarray | None = None,
    window: int = 25,
    block_size: int = 64,
    max_train_rows: int = 750_000,
    use_optuna: bool = False,
    optuna_trials: int = 20,
    use_block_context: bool = True,
    seed: int = 0,
    class_weight_mode: str = "balanced",
    apply_isotonic: bool = True,
    min_prob: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    ds = dosage.astype(np.float32, copy=False)
    y_truth = truth_genotype.astype(np.int16, copy=False)
    n_samples, n_positions = ds.shape
    if train_position_index is None:
        train_position_index = np.arange(n_positions, dtype=np.int64)
    else:
        train_position_index = train_position_index.astype(np.int64, copy=False)
    if predict_position_index is None:
        predict_position_index = np.arange(n_positions, dtype=np.int64)
    else:
        predict_position_index = predict_position_index.astype(np.int64, copy=False)

    gp_stage0 = calibrate_genotype_posterior(
        raw_posterior,
        dosage=ds,
        depth=depth,
        temperature=0.35,
        blend=0.35,
        min_prob=float(min_prob),
    )
    if use_block_context:
        gp_stage1, block_meta = calibrate_genotype_posterior_block_context(
            raw_posterior=gp_stage0,
            dosage=ds,
            truth_genotype=y_truth,
            samples_df=samples_df,
            train_position_index=train_position_index,
            predict_position_index=predict_position_index,
            window=int(window),
            block_size=int(block_size),
            max_train_rows=int(max_train_rows),
            use_optuna=bool(use_optuna),
            optuna_trials=int(optuna_trials),
            seed=int(seed),
            min_prob=float(min_prob),
        )
    else:
        gp_stage1 = gp_stage0
        block_meta = {"status": "skipped"}

    sample_meta, _ = build_sample_metadata_features(
        samples_df,
        sample_order=(samples_df["sample_id"].astype(str).to_numpy() if samples_df is not None and "sample_id" in samples_df.columns else None),
    )
    variant_maf = compute_variant_maf_from_posterior(gp_stage1)
    variant_info = compute_info_score_per_variant(gp_stage1)
    variant_strata, strata_meta = compute_variant_strata(variant_maf, variant_info)

    features, _ = build_readaware_calibration_feature_matrix(
        dosage=ds,
        posterior=gp_stage1,
        depth=depth,
        ref_count=ref_count,
        alt_count=alt_count,
        other_count=other_count,
        support_mask=support_mask,
        generations=generations,
        sample_metadata=sample_meta,
        variant_maf=variant_maf,
        variant_info=variant_info,
    )
    pos_flat = np.tile(np.arange(n_positions, dtype=np.int64), n_samples)
    y_flat = y_truth.reshape(-1)
    train_mask_flat = np.tile(np.isin(np.arange(n_positions, dtype=np.int64), train_position_index), n_samples)
    valid_train = (y_flat >= 0) & train_mask_flat
    if not np.any(valid_train):
        return gp_stage1, None, {"status": "no_training_rows", "block_context": block_meta}

    x_train = features[valid_train]
    y_train = y_flat[valid_train].astype(np.int32, copy=False)
    pos_train = pos_flat[valid_train]
    x_train, y_train = _subsample_rows_balanced(x_train, y_train, max_rows=int(max_train_rows), seed=int(seed))
    pos_take = _balanced_row_indices(y_flat[valid_train], max_rows=int(max_train_rows), seed=int(seed))
    pos_train = pos_train[pos_take]

    global_model, strata_models, stratified_meta = train_stratified_lightgbm_multiclass_calibrator(
        features=x_train,
        target=y_train,
        position_index=pos_train,
        variant_strata=variant_strata,
        seed=int(seed),
        max_rows=int(max_train_rows),
        class_weight_mode=class_weight_mode,
    )

    pred_flat = predict_stratified_lightgbm_posterior(
        features=features,
        position_index=pos_flat,
        variant_strata=variant_strata,
        global_model=global_model,
        strata_models=strata_models,
        min_prob=float(min_prob),
    )

    isotonic_models: list[Any] = [None, None, None]
    if apply_isotonic:
        pred_train = pred_flat[valid_train]
        isotonic_models = fit_multiclass_isotonic_calibrator(pred_train, y_flat[valid_train])
        pred_flat = apply_multiclass_isotonic_calibrator(pred_flat, isotonic_models, min_prob=float(min_prob))

    gp_out = pred_flat.reshape(n_samples, n_positions, 3).astype(np.float32, copy=False)
    gp_out = np.clip(gp_out, float(min_prob), 1.0)
    gp_out /= np.clip(np.sum(gp_out, axis=2, keepdims=True), 1e-12, None)

    gt_pred = np.argmax(gp_out, axis=2).astype(np.int8, copy=False)
    ds_pred = (gp_out[:, :, 1] + 2.0 * gp_out[:, :, 2]).astype(np.float32, copy=False)
    call_features, _ = build_hardcall_quality_feature_matrix(
        dosage=ds_pred,
        posterior=gp_out,
        depth=depth,
        ref_count=ref_count,
        alt_count=alt_count,
        other_count=other_count,
        generations=generations,
        variant_maf=variant_maf,
        variant_info=variant_info,
    )
    y_correct = (gt_pred.reshape(-1) == y_truth.reshape(-1)).astype(np.int8, copy=False)
    valid_call_train = (y_truth.reshape(-1) >= 0) & train_mask_flat
    call_prob = None
    call_meta: dict[str, object] = {"status": "skipped"}
    if np.any(valid_call_train):
        x_call = call_features[valid_call_train]
        y_call = y_correct[valid_call_train]
        if np.unique(y_call).size >= 2:
            x_call, y_call = _subsample_rows_balanced(
                x_call,
                y_call.astype(np.int32, copy=False),
                max_rows=int(max_train_rows),
                seed=int(seed) + 11,
            )
            call_model = train_lightgbm_binary_correctness_model(
                x_call,
                y_call,
                seed=int(seed) + 17,
            )
            call_prob_flat = predict_lightgbm_binary_probability(call_model, call_features)
            call_prob = call_prob_flat.reshape(n_samples, n_positions).astype(np.float32, copy=False)
            thr, score, call_rate = optimize_call_correctness_threshold(
                correctness_probability=call_prob_flat[valid_call_train],
                y_true=y_truth.reshape(-1)[valid_call_train],
                y_pred=gt_pred.reshape(-1)[valid_call_train],
            )
            call_meta = {
                "status": "ok",
                "threshold": float(thr),
                "score": float(score),
                "call_rate": float(call_rate),
                "n_rows": int(np.sum(valid_call_train)),
            }

    summary = {
        "status": "ok",
        "block_context": block_meta,
        "strata": strata_meta,
        "stratified_fit": stratified_meta,
        "isotonic_enabled": bool(apply_isotonic),
        "isotonic_models": int(sum(1 for m in isotonic_models if m is not None)),
        "call_correctness": call_meta,
    }
    return gp_out.astype(np.float32, copy=False), call_prob, summary
