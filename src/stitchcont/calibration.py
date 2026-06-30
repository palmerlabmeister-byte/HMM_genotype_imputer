from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _feature_frame(features: np.ndarray, feature_names: Sequence[str] | None = None) -> pd.DataFrame:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"feature matrix must be 2D, got shape {x.shape}")
    names = list(feature_names or [])
    if len(names) != x.shape[1]:
        names = [f"feature_{i}" for i in range(x.shape[1])]
    return pd.DataFrame(x, columns=names, copy=False)


def _predict_proba_feature_safe(model: Any, features: np.ndarray, feature_names: Sequence[str] | None = None) -> Any:
    """Predict with the same feature-name convention used at fit time.

    LightGBM/sklearn emits noisy warnings when an estimator was fitted with
    named columns and then receives a raw ndarray.  This helper preserves column
    names when available and thereby also protects against silent feature-order
    mistakes in future refactors.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None and hasattr(model, "named_steps"):
        names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return model.predict_proba(_feature_frame(features, list(map(str, names))))
    if feature_names is not None:
        return model.predict_proba(_feature_frame(features, feature_names))
    return model.predict_proba(np.asarray(features, dtype=np.float32))


def _softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    x = logits - np.max(logits, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(np.sum(exp_x, axis=axis, keepdims=True), 1e-12, None)


def posterior_confidence_metrics(posterior: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gp = posterior.astype(np.float32, copy=False)
    gp_finite = np.nan_to_num(gp, nan=0.0)
    max_prob = np.max(gp_finite, axis=2)
    if gp.shape[2] < 2:
        margin = np.zeros(gp.shape[:2], dtype=np.float32)
    else:
        gp_for_rank = np.nan_to_num(gp, nan=-np.inf)
        top2 = np.partition(gp_for_rank, kth=gp.shape[2] - 2, axis=2)[:, :, -2:]
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
    call_correct_threshold: float | np.ndarray = 0.0,
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
    threshold = np.asarray(call_correct_threshold, dtype=np.float32)
    has_call_correct_threshold = bool(np.any(np.isfinite(threshold) & (threshold > 0.0)))
    if call_correct_probability is not None and has_call_correct_threshold:
        prob = call_correct_probability.astype(np.float32, copy=False)
        if prob.shape != keep.shape:
            raise ValueError(
                f"call_correct_probability shape {prob.shape} does not match posterior calls shape {keep.shape}"
            )
        if threshold.ndim == 0:
            keep &= prob >= float(threshold)
        else:
            if threshold.shape == (keep.shape[1],):
                threshold = np.broadcast_to(threshold[None, :], keep.shape)
            elif threshold.shape != keep.shape:
                raise ValueError(
                    f"call_correct_threshold shape {threshold.shape} must be scalar, "
                    f"(n_variants,), or {keep.shape}"
                )
            keep &= prob >= threshold
    call = np.where(keep, call, int(no_call_value)).astype(np.int8, copy=False)
    return call


def stitch_vcf_genotype_call_from_posterior(
    posterior: np.ndarray,
    *,
    threshold: float = 0.9,
    no_call_value: int = -1,
) -> np.ndarray:
    """Mirror STITCH's VCF GT writer: argmax GP, then no-call if max GP < threshold."""
    return genotype_call_from_posterior(
        posterior,
        stitch_gp_threshold=float(threshold),
        no_call_value=int(no_call_value),
    )


def _normalise_edges(edges: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(edges), dtype=np.float32)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError("MAF bins must contain at least two edges.")
    arr = np.unique(np.clip(arr, 0.0, 0.5)).astype(np.float32, copy=False)
    if arr.size < 2:
        raise ValueError("MAF bins collapse to fewer than two unique edges.")
    if float(arr[0]) > 0.0:
        arr = np.concatenate([np.asarray([0.0], dtype=np.float32), arr])
    if float(arr[-1]) < 0.5:
        arr = np.concatenate([arr, np.asarray([0.5], dtype=np.float32)])
    return arr


def _posterior_nll(gp: np.ndarray, truth: np.ndarray, *, min_prob: float = 1e-6) -> float:
    y = truth.astype(np.int16, copy=False).reshape(-1)
    p = gp.reshape(-1, gp.shape[-1])
    valid = (y >= 0) & (y < p.shape[1])
    if not np.any(valid):
        return float("inf")
    return float(-np.mean(np.log(np.clip(p[valid, y[valid]], float(min_prob), 1.0))))


def _posterior_brier(gp: np.ndarray, truth: np.ndarray) -> float:
    y = truth.astype(np.int16, copy=False).reshape(-1)
    p = gp.reshape(-1, gp.shape[-1])
    valid = (y >= 0) & (y < p.shape[1])
    if not np.any(valid):
        return float("inf")
    one_hot = np.zeros((int(np.sum(valid)), p.shape[1]), dtype=np.float32)
    one_hot[np.arange(one_hot.shape[0]), y[valid]] = 1.0
    return float(np.mean((p[valid] - one_hot) ** 2))


def _posterior_dosage_mse(gp: np.ndarray, truth: np.ndarray) -> float:
    y = truth.astype(np.int16, copy=False)
    valid = y >= 0
    if not np.any(valid):
        return float("inf")
    axis = np.arange(gp.shape[2], dtype=np.float32)
    ds = np.sum(gp * axis[None, None, :], axis=2)
    return float(np.mean((ds[valid] - y[valid].astype(np.float32, copy=False)) ** 2))


def _posterior_objective_components_flat(
    gp: np.ndarray,
    truth: np.ndarray,
    *,
    min_prob: float = 1e-6,
) -> tuple[float, float, float]:
    """NLL, Brier, and dosage MSE for already-flattened training rows."""
    p = gp.astype(np.float32, copy=False).reshape(-1, gp.shape[-1])
    y = truth.astype(np.int16, copy=False).reshape(-1)
    valid = (y >= 0) & (y < p.shape[1])
    if not np.any(valid):
        return float("inf"), float("inf"), float("inf")
    pv = p[valid]
    yv = y[valid]
    rows = np.arange(yv.shape[0])
    nll = float(-np.mean(np.log(np.clip(pv[rows, yv], float(min_prob), 1.0))))
    truth_prob = np.zeros_like(pv, dtype=np.float32)
    truth_prob[rows, yv] = 1.0
    brier = float(np.mean((pv - truth_prob) ** 2))
    axis = np.arange(pv.shape[1], dtype=np.float32)
    ds = pv @ axis
    mse = float(np.mean((ds - yv.astype(np.float32, copy=False)) ** 2))
    return nll, brier, mse


def _hwe_soft_penalty(
    gp: np.ndarray,
    *,
    maf: np.ndarray,
    hwe_weight: float,
    hwe_min_maf: float,
) -> float:
    weight = float(hwe_weight)
    if weight <= 0.0 or gp.shape[2] != 3:
        return 0.0
    maf_arr = maf.astype(np.float32, copy=False)
    use = np.isfinite(maf_arr) & (maf_arr >= float(hwe_min_maf))
    if not np.any(use):
        return 0.0
    geno_dist = np.nanmean(gp[:, use, :], axis=0)
    p = np.clip((geno_dist[:, 1] + 2.0 * geno_dist[:, 2]) / 2.0, 1e-5, 1.0 - 1e-5)
    q = 1.0 - p
    expected = np.stack([q * q, 2.0 * p * q, p * p], axis=1).astype(np.float32, copy=False)
    return float(weight * np.mean((geno_dist - expected) ** 2))


def estimate_maf_from_truth_or_posterior(
    *,
    truth_genotype: np.ndarray | None,
    posterior: np.ndarray,
    train_mask: np.ndarray | None = None,
) -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    n_samples, n_positions = gp.shape[:2]
    maf = np.full(n_positions, np.nan, dtype=np.float32)
    if truth_genotype is not None:
        truth = truth_genotype.astype(np.int16, copy=False)
        if truth.shape != (n_samples, n_positions):
            raise ValueError(f"truth_genotype shape {truth.shape} does not match posterior shape {(n_samples, n_positions)}")
        valid = truth >= 0
        if train_mask is not None:
            tm = train_mask.astype(bool, copy=False)
            if tm.shape != valid.shape:
                raise ValueError(f"train_mask shape {tm.shape} does not match truth shape {valid.shape}")
            valid &= tm
        for j in range(n_positions):
            m = valid[:, j]
            if np.any(m):
                af = float(np.mean(truth[m, j].astype(np.float32, copy=False)) / 2.0)
                maf[j] = min(max(af, 0.0), 1.0 - max(af, 0.0))
    fallback = compute_variant_maf_from_posterior(gp)
    maf = np.where(np.isfinite(maf), maf, fallback)
    return np.clip(maf, 0.0, 0.5).astype(np.float32, copy=False)


def estimate_alt_af_from_truth_or_posterior(
    *,
    truth_genotype: np.ndarray | None,
    posterior: np.ndarray,
    train_mask: np.ndarray | None = None,
) -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    n_samples, n_positions = gp.shape[:2]
    af = np.full(n_positions, np.nan, dtype=np.float32)
    if truth_genotype is not None:
        truth = truth_genotype.astype(np.int16, copy=False)
        if truth.shape != (n_samples, n_positions):
            raise ValueError(f"truth_genotype shape {truth.shape} does not match posterior shape {(n_samples, n_positions)}")
        valid = truth >= 0
        if train_mask is not None:
            tm = train_mask.astype(bool, copy=False)
            if tm.shape != valid.shape:
                raise ValueError(f"train_mask shape {tm.shape} does not match truth shape {valid.shape}")
            valid &= tm
        for j in range(n_positions):
            m = valid[:, j]
            if np.any(m):
                af[j] = float(np.mean(truth[m, j].astype(np.float32, copy=False)) / 2.0)
    geno_axis = np.arange(gp.shape[2], dtype=np.float32)
    ploidy = max(gp.shape[2] - 1, 1)
    fallback = np.mean(np.sum(gp * geno_axis[None, None, :], axis=2), axis=0) / float(ploidy)
    af = np.where(np.isfinite(af), af, fallback)
    return np.clip(af, 0.0, 1.0).astype(np.float32, copy=False)


def hwe_prior_from_alt_af(alt_af: np.ndarray, *, min_prob: float = 1e-6) -> np.ndarray:
    p = np.clip(alt_af.astype(np.float32, copy=False), float(min_prob), 1.0 - float(min_prob))
    q = 1.0 - p
    prior = np.stack([q * q, 2.0 * p * q, p * p], axis=1).astype(np.float32, copy=False)
    prior = np.clip(prior, float(min_prob), 1.0)
    prior /= np.clip(np.sum(prior, axis=1, keepdims=True), 1e-12, None)
    return prior.astype(np.float32, copy=False)


def apply_hwe_prior_to_posterior(
    posterior: np.ndarray,
    *,
    alt_af: np.ndarray,
    prior_weight: float,
    min_prob: float = 1e-6,
) -> np.ndarray:
    weight = float(prior_weight)
    gp = posterior.astype(np.float32, copy=False)
    if weight <= 0.0:
        return gp
    prior = hwe_prior_from_alt_af(alt_af, min_prob=float(min_prob))
    log_gp = np.log(np.clip(gp, float(min_prob), 1.0))
    log_prior = np.log(np.clip(prior[None, :, :], float(min_prob), 1.0))
    out = np.exp(log_gp + weight * log_prior).astype(np.float32, copy=False)
    out = np.clip(out, float(min_prob), 1.0)
    out /= np.clip(np.sum(out, axis=2, keepdims=True), 1e-12, None)
    return out.astype(np.float32, copy=False)


def posterior_call_score(posterior: np.ndarray, *, mode: str = "log_max_gp") -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    conf, margin, entropy = posterior_confidence_metrics(gp)
    mode_l = str(mode).lower()
    if mode_l == "max_gp":
        return conf.astype(np.float32, copy=False)
    if mode_l == "log_max_gp":
        return np.log(np.clip(conf, 1e-12, 1.0)).astype(np.float32, copy=False)
    if mode_l == "margin":
        return margin.astype(np.float32, copy=False)
    if mode_l == "neg_entropy":
        return (-entropy).astype(np.float32, copy=False)
    if mode_l == "log_max_gp_minus_entropy":
        return (np.log(np.clip(conf, 1e-12, 1.0)) - entropy).astype(np.float32, copy=False)
    raise ValueError(f"Unknown call-score mode: {mode}")


def _macro_f1_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(np.int16, copy=False).reshape(-1)
    yp = y_pred.astype(np.int16, copy=False).reshape(-1)
    classes = np.union1d(np.unique(yt), np.unique(yp))
    vals = []
    for cls in classes.tolist():
        if cls < 0:
            continue
        tp = float(np.sum((yt == cls) & (yp == cls)))
        fp = float(np.sum((yt != cls) & (yp == cls)))
        fn = float(np.sum((yt == cls) & (yp != cls)))
        denom = 2.0 * tp + fp + fn
        vals.append(0.0 if denom <= 0.0 else (2.0 * tp / denom))
    return float(np.mean(vals)) if vals else float("nan")


def _balanced_accuracy_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = y_true.astype(np.int16, copy=False).reshape(-1)
    yp = y_pred.astype(np.int16, copy=False).reshape(-1)
    vals = []
    for cls in np.unique(yt).tolist():
        if cls < 0:
            continue
        m = yt == cls
        if np.any(m):
            vals.append(float(np.mean(yp[m] == cls)))
    return float(np.mean(vals)) if vals else float("nan")


def _variant_hwe_deviation_from_posterior(posterior: np.ndarray, alt_af: np.ndarray) -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    obs = np.nanmean(gp, axis=0)
    exp = hwe_prior_from_alt_af(alt_af)
    return np.sqrt(np.mean((obs - exp) ** 2, axis=1)).astype(np.float32, copy=False)


def compute_hwe_features_from_posterior(
    posterior: np.ndarray,
    *,
    alt_af: np.ndarray | None = None,
    min_expected: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Compute diploid biallelic HWE diagnostics from posterior genotype counts.

    The chi-square p-value uses the 1-df survival function, erfc(sqrt(x / 2)).
    This keeps the feature dependency-free and stable for calibration models.
    """
    gp = posterior.astype(np.float32, copy=False)
    n_positions = gp.shape[1]
    if gp.shape[2] != 3:
        nan = np.full(n_positions, np.nan, dtype=np.float32)
        return {
            "hwe_deviation": nan,
            "hwe_chisq": nan,
            "hwe_pvalue": nan,
            "hwe_neg_log10_pvalue": nan,
        }
    if alt_af is None:
        alt_af = estimate_alt_af_from_truth_or_posterior(truth_genotype=None, posterior=gp)
    af = np.clip(alt_af.astype(np.float32, copy=False), 1e-6, 1.0 - 1e-6)
    obs_counts = np.sum(gp, axis=0).astype(np.float32, copy=False)
    n_eff = np.clip(np.sum(obs_counts, axis=1), float(min_expected), None)
    expected = hwe_prior_from_alt_af(af) * n_eff[:, None]
    chisq = np.sum(((obs_counts - expected) ** 2) / np.clip(expected, float(min_expected), None), axis=1)
    pvalue = np.asarray([math.erfc(math.sqrt(max(float(x), 0.0) / 2.0)) for x in chisq], dtype=np.float32)
    deviation = np.sqrt(np.mean(((obs_counts / n_eff[:, None]) - (expected / n_eff[:, None])) ** 2, axis=1))
    neg_log10 = -np.log10(np.clip(pvalue, 1e-30, 1.0)).astype(np.float32, copy=False)
    return {
        "hwe_deviation": deviation.astype(np.float32, copy=False),
        "hwe_chisq": chisq.astype(np.float32, copy=False),
        "hwe_pvalue": pvalue.astype(np.float32, copy=False),
        "hwe_neg_log10_pvalue": neg_log10.astype(np.float32, copy=False),
    }


def calibration_shift_metrics(raw_posterior: np.ndarray, calibrated_posterior: np.ndarray) -> dict[str, float]:
    raw = raw_posterior.astype(np.float32, copy=False)
    cal = calibrated_posterior.astype(np.float32, copy=False)
    if raw.shape != cal.shape:
        raise ValueError(f"raw and calibrated posterior shapes differ: {raw.shape} vs {cal.shape}")
    axis = np.arange(raw.shape[2], dtype=np.float32)
    ploidy = max(raw.shape[2] - 1, 1)
    raw_ds = np.sum(raw * axis[None, None, :], axis=2)
    cal_ds = np.sum(cal * axis[None, None, :], axis=2)
    raw_af = np.nanmean(raw_ds, axis=0) / float(ploidy)
    cal_af = np.nanmean(cal_ds, axis=0) / float(ploidy)
    _, _, raw_entropy = posterior_confidence_metrics(raw)
    _, _, cal_entropy = posterior_confidence_metrics(cal)
    metrics = {
        "mean_abs_dosage_shift": float(np.nanmean(np.abs(cal_ds - raw_ds))),
        "mean_abs_maf_shift": float(np.nanmean(np.abs(cal_af - raw_af))),
        "mean_entropy_shift": float(np.nanmean(cal_entropy) - np.nanmean(raw_entropy)),
        "mean_abs_entropy_shift": float(np.nanmean(np.abs(cal_entropy - raw_entropy))),
    }
    if raw.shape[2] == 3:
        metrics["het_rate_shift"] = float(np.nanmean(cal[:, :, 1]) - np.nanmean(raw[:, :, 1]))
        metrics["abs_het_rate_shift"] = abs(metrics["het_rate_shift"])
        raw_hwe = compute_hwe_features_from_posterior(raw, alt_af=raw_af)["hwe_deviation"]
        cal_hwe = compute_hwe_features_from_posterior(cal, alt_af=cal_af)["hwe_deviation"]
        metrics["mean_abs_hwe_deviation_shift"] = float(np.nanmean(np.abs(cal_hwe - raw_hwe)))
    else:
        metrics["het_rate_shift"] = float("nan")
        metrics["abs_het_rate_shift"] = float("nan")
        metrics["mean_abs_hwe_deviation_shift"] = float("nan")
    return metrics


def apply_calibration_sanity_guard(
    raw_posterior: np.ndarray | None,
    calibrated_posterior: np.ndarray,
    *,
    max_mean_abs_dosage_shift: float = 0.35,
    max_mean_abs_maf_shift: float = 0.20,
    max_het_rate_shift: float = 0.25,
    max_mean_entropy_shift: float = 0.50,
) -> tuple[np.ndarray, dict[str, object]]:
    if raw_posterior is None:
        return calibrated_posterior, {"status": "skipped_no_raw_posterior"}
    metrics = calibration_shift_metrics(raw_posterior, calibrated_posterior)
    failures: list[str] = []
    if metrics["mean_abs_dosage_shift"] > float(max_mean_abs_dosage_shift):
        failures.append("mean_abs_dosage_shift")
    if metrics["mean_abs_maf_shift"] > float(max_mean_abs_maf_shift):
        failures.append("mean_abs_maf_shift")
    if np.isfinite(metrics["abs_het_rate_shift"]) and metrics["abs_het_rate_shift"] > float(max_het_rate_shift):
        failures.append("het_rate_shift")
    if metrics["mean_abs_entropy_shift"] > float(max_mean_entropy_shift):
        failures.append("mean_abs_entropy_shift")
    meta: dict[str, object] = {
        "status": "ok" if not failures else "fallback_to_raw",
        "metrics": metrics,
        "failed_checks": failures,
        "thresholds": {
            "max_mean_abs_dosage_shift": float(max_mean_abs_dosage_shift),
            "max_mean_abs_maf_shift": float(max_mean_abs_maf_shift),
            "max_het_rate_shift": float(max_het_rate_shift),
            "max_mean_entropy_shift": float(max_mean_entropy_shift),
        },
    }
    if failures:
        return raw_posterior.astype(np.float32, copy=False), meta
    return calibrated_posterior.astype(np.float32, copy=False), meta


def fit_no_call_thresholds_by_calibration_class(
    *,
    posterior: np.ndarray,
    truth_genotype: np.ndarray,
    train_mask: np.ndarray,
    maf: np.ndarray | None = None,
    alt_af: np.ndarray | None = None,
    maf_bins: Sequence[float] = (0.0, 0.01, 0.05, 0.5),
    entropy_bins: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.01),
    hwe_deviation_bins: Sequence[float] = (0.0, 0.02, 0.05, 0.10, np.inf),
    score_mode: str = "log_max_gp",
    thresholds: Sequence[float] | None = None,
    min_train_rows_per_class: int = 16,
    min_call_rate: float = 0.05,
    max_no_call_rate_per_variant: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    gp = posterior.astype(np.float32, copy=False)
    truth = truth_genotype.astype(np.int16, copy=False)
    train = (truth >= 0) & train_mask.astype(bool, copy=False)
    if truth.shape != gp.shape[:2] or train.shape != truth.shape:
        raise ValueError("posterior, truth_genotype, and train_mask shapes are incompatible.")
    if maf is None:
        maf = estimate_maf_from_truth_or_posterior(truth_genotype=truth, posterior=gp, train_mask=train)
    if alt_af is None:
        alt_af = estimate_alt_af_from_truth_or_posterior(truth_genotype=truth, posterior=gp, train_mask=train)
    maf_edges = _normalise_edges(maf_bins)
    ent_edges = np.asarray(list(entropy_bins), dtype=np.float32)
    hwe_edges = np.asarray(list(hwe_deviation_bins), dtype=np.float32)
    _, _, entropy = posterior_confidence_metrics(gp)
    mean_entropy = np.nanmean(entropy, axis=0).astype(np.float32, copy=False)
    hwe_dev = _variant_hwe_deviation_from_posterior(gp, alt_af.astype(np.float32, copy=False))
    maf_idx = np.digitize(np.clip(maf, 0.0, 0.5), maf_edges[1:-1], right=False)
    ent_idx = np.digitize(mean_entropy, ent_edges[1:-1], right=False)
    hwe_idx = np.digitize(hwe_dev, hwe_edges[1:-1], right=False)
    n_ent = max(len(ent_edges) - 1, 1)
    n_hwe = max(len(hwe_edges) - 1, 1)
    class_id = (maf_idx.astype(np.int32) * n_ent * n_hwe + ent_idx.astype(np.int32) * n_hwe + hwe_idx.astype(np.int32)).astype(np.int32)
    score = posterior_call_score(gp, mode=str(score_mode))
    argmax = np.argmax(gp, axis=2).astype(np.int8, copy=False)
    if thresholds is None:
        finite_score = score[train]
        if finite_score.size == 0:
            thresholds_arr = np.asarray([-np.inf], dtype=np.float32)
        else:
            q = np.linspace(0.0, 0.95, 40, dtype=np.float32)
            thresholds_arr = np.unique(np.quantile(finite_score, q).astype(np.float32))
            thresholds_arr = np.concatenate([np.asarray([-np.inf], dtype=np.float32), thresholds_arr])
    else:
        thresholds_arr = np.asarray(list(thresholds), dtype=np.float32)
    threshold_by_variant = np.full(gp.shape[1], -np.inf, dtype=np.float32)
    rows: list[dict[str, object]] = []
    global_thr = -np.inf
    global_score = -np.inf
    flat_train = train.reshape(-1)
    if np.any(flat_train):
        yt_all = truth.reshape(-1)[flat_train]
        yp_all = argmax.reshape(-1)[flat_train]
        sc_all = score.reshape(-1)[flat_train]
        for thr in thresholds_arr.tolist():
            keep = sc_all >= float(thr)
            call_rate = float(np.mean(keep)) if keep.size else 0.0
            if call_rate < float(min_call_rate) or not np.any(keep):
                continue
            f1 = _macro_f1_np(yt_all[keep], yp_all[keep])
            bacc = _balanced_accuracy_np(yt_all[keep], yp_all[keep])
            obj = 0.5 * f1 + 0.5 * bacc + 0.03 * call_rate
            if np.isfinite(obj) and obj > global_score:
                global_score = float(obj)
                global_thr = float(thr)
    for cls in np.unique(class_id).tolist():
        cols = class_id == int(cls)
        cls_train = train[:, cols]
        n_train = int(np.sum(cls_train))
        best_thr = float(global_thr)
        best_obj = float(global_score)
        best_call_rate = float("nan")
        best_f1 = float("nan")
        best_bacc = float("nan")
        fallback = True
        if n_train >= int(min_train_rows_per_class):
            yt = truth[:, cols][cls_train]
            yp = argmax[:, cols][cls_train]
            sc = score[:, cols][cls_train]
            sc_mat = score[:, cols]
            train_mat = train[:, cols]
            for thr in thresholds_arr.tolist():
                if float(max_no_call_rate_per_variant) < 1.0:
                    denom = np.sum(train_mat, axis=0)
                    no_call = np.sum(train_mat & (sc_mat < float(thr)), axis=0)
                    valid_denom = denom > 0
                    if np.any(valid_denom):
                        max_rate = float(np.max(no_call[valid_denom] / np.clip(denom[valid_denom], 1, None)))
                        if max_rate > float(max_no_call_rate_per_variant):
                            continue
                keep = sc >= float(thr)
                call_rate = float(np.mean(keep)) if keep.size else 0.0
                if call_rate < float(min_call_rate) or not np.any(keep):
                    continue
                f1 = _macro_f1_np(yt[keep], yp[keep])
                bacc = _balanced_accuracy_np(yt[keep], yp[keep])
                obj = 0.5 * f1 + 0.5 * bacc + 0.03 * call_rate
                if np.isfinite(obj) and obj > best_obj:
                    best_thr = float(thr)
                    best_obj = float(obj)
                    best_call_rate = call_rate
                    best_f1 = float(f1)
                    best_bacc = float(bacc)
                    fallback = False
        threshold_by_variant[cols] = float(best_thr)
        rows.append(
            {
                "class_id": int(cls),
                "n_variants": int(np.sum(cols)),
                "n_train_rows": int(n_train),
                "threshold": float(best_thr),
                "score_mode": str(score_mode),
                "objective": float(best_obj),
                "train_call_rate": float(best_call_rate),
                "train_f1": float(best_f1),
                "train_balanced_accuracy": float(best_bacc),
                "fallback_to_global": bool(fallback),
            }
        )
    meta = {
        "status": "ok",
        "score_mode": str(score_mode),
        "global_threshold": float(global_thr),
        "global_objective": float(global_score),
        "maf_bins": maf_edges.tolist(),
        "entropy_bins": ent_edges.tolist(),
        "hwe_deviation_bins": hwe_edges.tolist(),
        "max_no_call_rate_per_variant": float(max_no_call_rate_per_variant),
        "classes": rows,
    }
    return threshold_by_variant.astype(np.float32, copy=False), meta


def apply_no_call_thresholds_by_variant(
    posterior: np.ndarray,
    threshold_by_variant: np.ndarray,
    *,
    score_mode: str = "log_max_gp",
    no_call_value: int = -1,
    max_no_call_rate_per_variant: float = 1.0,
) -> np.ndarray:
    gp = posterior.astype(np.float32, copy=False)
    thr = threshold_by_variant.astype(np.float32, copy=False)
    if thr.shape[0] != gp.shape[1]:
        raise ValueError(f"threshold_by_variant length {thr.shape[0]} does not match n_variants {gp.shape[1]}")
    call = np.argmax(gp, axis=2).astype(np.int8, copy=False)
    score = posterior_call_score(gp, mode=str(score_mode))
    if float(max_no_call_rate_per_variant) < 1.0:
        cap = float(np.clip(max_no_call_rate_per_variant, 0.0, 1.0))
        capped = thr.copy()
        max_missing = int(np.floor(cap * float(gp.shape[0])))
        for j in range(gp.shape[1]):
            sorted_score = np.sort(score[:, j])
            if max_missing <= 0:
                cap_threshold = float(sorted_score[0])
            elif max_missing < sorted_score.shape[0]:
                cap_threshold = float(sorted_score[max_missing])
            else:
                cap_threshold = np.inf
            capped[j] = min(float(capped[j]), cap_threshold)
        thr = capped
    keep = score >= thr[None, :]
    return np.where(keep, call, int(no_call_value)).astype(np.int8, copy=False)


def masked_cv_calibrate_genotype_posterior(
    *,
    raw_posterior: np.ndarray | None,
    dosage: np.ndarray,
    truth_genotype: np.ndarray,
    train_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    maf_bins: Sequence[float] = (0.0, 0.01, 0.05, 0.5),
    temperatures: Sequence[float] = (0.15, 0.25, 0.35, 0.5, 0.75, 1.0),
    blends: Sequence[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
    dosage_scales: Sequence[float] = (0.75, 1.0, 1.25, 1.5, 2.0),
    dosage_offsets: Sequence[float] = (-0.25, 0.0, 0.25),
    hwe_prior_weights: Sequence[float] = (0.0, 0.25, 0.5, 1.0),
    optimize_dosage_scale: bool = True,
    hwe_weight: float = 0.0,
    hwe_min_maf: float = 0.05,
    brier_weight: float = 0.25,
    dosage_mse_weight: float = 0.10,
    min_train_rows_per_bin: int = 16,
    min_prob: float = 1e-6,
    ploidy: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Tune fixed posterior calibration parameters on masked truth labels, stratified by MAF.

    The function is intentionally dependency-free and conservative: it only uses labels marked by
    ``train_mask`` to select calibration parameters, then applies the selected bin-specific
    parameters to all samples for variants in that MAF bin.
    """
    ds = dosage.astype(np.float32, copy=False)
    truth = truth_genotype.astype(np.int16, copy=False)
    if truth.shape != ds.shape:
        raise ValueError(f"truth_genotype shape {truth.shape} does not match dosage shape {ds.shape}")
    ploidy_i = int(raw_posterior.shape[2] - 1) if raw_posterior is not None else (2 if ploidy is None else int(ploidy))
    if ploidy_i != 2:
        # The search objective currently assumes biallelic diploid hard labels 0/1/2.
        base = calibrate_genotype_posterior(
            raw_posterior,
            dosage=ds,
            depth=depth,
            temperature=0.35,
            blend=0.35,
            min_prob=float(min_prob),
            ploidy=ploidy_i,
        )
        return base, {"status": "skipped_non_diploid", "ploidy": int(ploidy_i)}

    raw = (
        dosage_to_genotype_posterior(ds, depth=depth, temperature=0.35, min_prob=float(min_prob), ploidy=2)
        if raw_posterior is None
        else raw_posterior.astype(np.float32, copy=False)
    )
    if raw.shape != (ds.shape[0], ds.shape[1], 3):
        raise ValueError(f"raw_posterior shape {raw.shape} does not match expected {(ds.shape[0], ds.shape[1], 3)}")
    train = (truth >= 0) if train_mask is None else (truth >= 0) & train_mask.astype(bool, copy=False)
    if not np.any(train):
        base = calibrate_genotype_posterior(raw, dosage=ds, depth=depth, temperature=0.35, blend=0.35)
        return base, {"status": "no_training_rows"}

    edges = _normalise_edges(maf_bins)
    maf = estimate_maf_from_truth_or_posterior(truth_genotype=truth, posterior=raw, train_mask=train)
    alt_af = estimate_alt_af_from_truth_or_posterior(truth_genotype=truth, posterior=raw, train_mask=train)
    bin_idx = np.digitize(np.clip(maf, 0.0, 0.5), edges[1:-1], right=False).astype(np.int16, copy=False)
    temps = [float(x) for x in temperatures if float(x) > 0.0]
    blend_vals = [float(np.clip(x, 0.0, 1.0)) for x in blends]
    scales = [float(x) for x in dosage_scales] if optimize_dosage_scale else [1.0]
    offsets = [float(x) for x in dosage_offsets] if optimize_dosage_scale else [0.0]
    prior_weights = [float(max(x, 0.0)) for x in hwe_prior_weights]
    if not temps or not blend_vals or not scales or not offsets or not prior_weights:
        raise ValueError("Calibration parameter grids must not be empty.")

    out = np.empty_like(raw, dtype=np.float32)
    bin_meta: list[dict[str, object]] = []
    global_best: dict[str, float] | None = None
    for b in range(edges.size - 1):
        cols = bin_idx == b
        if not np.any(cols):
            continue
        train_bin = train[:, cols]
        n_train = int(np.sum(train_bin))
        if n_train < int(min_train_rows_per_bin) and global_best is not None:
            best = dict(global_best)
            best["fallback"] = 1.0
        else:
            best_score = float("inf")
            best = {
                "temperature": 0.35,
                "blend": 0.35,
                "dosage_scale": 1.0,
                "dosage_offset": 0.0,
                "hwe_prior_weight": 0.0,
                "score": float("inf"),
                "fallback": 0.0,
            }
            raw_bin = raw[:, cols, :]
            ds_bin0 = ds[:, cols]
            truth_bin = truth[:, cols]
            maf_bin = maf[cols]
            alt_af_bin = alt_af[cols]
            for scale in scales:
                for offset in offsets:
                    ds_bin = np.clip(ds_bin0 * float(scale) + float(offset), 0.0, 2.0).astype(np.float32, copy=False)
                    for temp in temps:
                        gp_dosage = dosage_to_genotype_posterior(
                            ds_bin,
                            depth=(None if depth is None else depth[:, cols]),
                            temperature=float(temp),
                            min_prob=float(min_prob),
                            ploidy=2,
                        )
                        for blend in blend_vals:
                            gp = (1.0 - float(blend)) * raw_bin + float(blend) * gp_dosage
                            gp = np.clip(gp, float(min_prob), 1.0)
                            gp /= np.clip(np.sum(gp, axis=2, keepdims=True), 1e-12, None)
                            for prior_weight in prior_weights:
                                gp_prior = apply_hwe_prior_to_posterior(
                                    gp,
                                    alt_af=alt_af_bin,
                                    prior_weight=float(prior_weight),
                                    min_prob=float(min_prob),
                                )
                                gp_train = gp_prior[train_bin]
                                truth_train = truth_bin[train_bin]
                                nll, brier, mse = _posterior_objective_components_flat(
                                    gp_train,
                                    truth_train,
                                    min_prob=float(min_prob),
                                )
                                hwe = _hwe_soft_penalty(
                                    gp_prior,
                                    maf=maf_bin,
                                    hwe_weight=float(hwe_weight),
                                    hwe_min_maf=float(hwe_min_maf),
                                )
                                score = nll + float(brier_weight) * brier + float(dosage_mse_weight) * mse + hwe
                                if score < best_score:
                                    best_score = float(score)
                                    best = {
                                        "temperature": float(temp),
                                        "blend": float(blend),
                                        "dosage_scale": float(scale),
                                        "dosage_offset": float(offset),
                                        "hwe_prior_weight": float(prior_weight),
                                        "score": float(score),
                                        "nll": float(nll),
                                        "brier": float(brier),
                                        "dosage_mse": float(mse),
                                        "hwe_penalty": float(hwe),
                                        "fallback": 0.0,
                                    }
            if n_train >= int(min_train_rows_per_bin):
                global_best = dict(best)
        scale = float(best["dosage_scale"])
        offset = float(best["dosage_offset"])
        ds_sel = np.clip(ds[:, cols] * scale + offset, 0.0, 2.0).astype(np.float32, copy=False)
        gp_dosage = dosage_to_genotype_posterior(
            ds_sel,
            depth=(None if depth is None else depth[:, cols]),
            temperature=float(best["temperature"]),
            min_prob=float(min_prob),
            ploidy=2,
        )
        blend = float(best["blend"])
        out[:, cols, :] = (1.0 - blend) * raw[:, cols, :] + blend * gp_dosage
        out[:, cols, :] = np.clip(out[:, cols, :], float(min_prob), 1.0)
        out[:, cols, :] /= np.clip(np.sum(out[:, cols, :], axis=2, keepdims=True), 1e-12, None)
        out[:, cols, :] = apply_hwe_prior_to_posterior(
            out[:, cols, :],
            alt_af=alt_af[cols],
            prior_weight=float(best.get("hwe_prior_weight", 0.0)),
            min_prob=float(min_prob),
        )
        meta_row = {
            "bin": int(b),
            "maf_min": float(edges[b]),
            "maf_max": float(edges[b + 1]),
            "n_variants": int(np.sum(cols)),
            "n_train_rows": int(n_train),
            **best,
        }
        bin_meta.append(meta_row)

    # Any empty/unassigned variants keep the regular fixed calibration.
    assigned = np.zeros(ds.shape[1], dtype=bool)
    for row in bin_meta:
        b = int(row["bin"])
        assigned |= bin_idx == b
    if np.any(~assigned):
        out[:, ~assigned, :] = calibrate_genotype_posterior(
            raw[:, ~assigned, :],
            dosage=ds[:, ~assigned],
            depth=(None if depth is None else depth[:, ~assigned]),
            temperature=0.35,
            blend=0.35,
            min_prob=float(min_prob),
            ploidy=2,
        )
    out = np.clip(out, float(min_prob), 1.0)
    out /= np.clip(np.sum(out, axis=2, keepdims=True), 1e-12, None)
    meta = {
        "status": "ok",
        "mode": "masked_cv",
        "maf_bins": edges.tolist(),
        "n_train_rows": int(np.sum(train)),
        "hwe_weight": float(hwe_weight),
        "hwe_min_maf": float(hwe_min_maf),
        "hwe_prior_weights": [float(x) for x in prior_weights],
        "optimize_dosage_scale": bool(optimize_dosage_scale),
        "bins": bin_meta,
    }
    return out.astype(np.float32, copy=False), meta


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
    support_rate: np.ndarray | None = None,
    missingness: np.ndarray | None = None,
    hwe_deviation: np.ndarray | None = None,
    mean_depth: np.ndarray | None = None,
    maf_bins: Sequence[float] = (0.0, 0.01, 0.05, 0.2, 0.5),
    info_bins: Sequence[float] = (-5.0, 0.2, 0.5, 0.8, 1.1),
    support_bins: Sequence[float] = (0.0, 0.25, 0.75, 1.01),
    missingness_bins: Sequence[float] = (0.0, 0.05, 0.25, 0.60, 1.01),
    hwe_deviation_bins: Sequence[float] = (0.0, 0.02, 0.08, 0.20, np.inf),
    depth_bins: Sequence[float] = (0.0, 0.5, 2.0, 10.0, np.inf),
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
    components: list[tuple[str, np.ndarray, np.ndarray]] = [
        ("maf", maf_idx.astype(np.int32, copy=False), maf_edges),
        ("info", info_idx.astype(np.int32, copy=False), info_edges),
    ]

    def optional_component(name: str, values: np.ndarray | None, edges_in: Sequence[float], fill_value: float) -> None:
        if values is None:
            return
        arr = np.asarray(values, dtype=np.float32)
        if arr.shape[0] != maf_arr.shape[0]:
            raise ValueError(f"{name} length {arr.shape[0]} does not match maf length {maf_arr.shape[0]}.")
        edges = np.asarray(edges_in, dtype=np.float32)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError(f"{name}_bins must provide at least 2 ordered edges.")
        filled = np.where(np.isfinite(arr), arr, float(fill_value))
        idx = np.digitize(np.clip(filled, edges[0], edges[-1]), edges[1:-1], right=False).astype(np.int32)
        components.append((name, idx, edges))

    optional_component("support_rate", support_rate, support_bins, 0.0)
    optional_component("missingness", missingness, missingness_bins, 1.0)
    optional_component("hwe_deviation", hwe_deviation, hwe_deviation_bins, 0.0)
    optional_component("mean_depth", mean_depth, depth_bins, 0.0)

    strata = np.zeros(maf_arr.shape[0], dtype=np.int32)
    multiplier = 1
    component_meta: dict[str, dict[str, object]] = {}
    for name, idx, edges in reversed(components):
        n_bins = int(max(edges.size - 1, 1))
        strata += idx.astype(np.int32, copy=False) * int(multiplier)
        component_meta[name] = {
            "bins": edges.tolist(),
            "n_bins": n_bins,
            "multiplier": int(multiplier),
        }
        multiplier *= n_bins
    meta = {
        "maf_bins": maf_edges.tolist(),
        "info_bins": info_edges.tolist(),
        "components": component_meta,
        "n_strata": int(multiplier),
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

    hwe_features = compute_hwe_features_from_posterior(gp)
    for name, values in hwe_features.items():
        fill = np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=300.0, neginf=0.0)
        _append(f"variant_{name}", np.broadcast_to(fill[None, :], ds.shape))

    site_ref_count = np.sum(ref_arr, axis=0).astype(np.float32, copy=False)
    site_alt_count = np.sum(alt_arr, axis=0).astype(np.float32, copy=False)
    site_other_count = np.sum(oth_arr, axis=0).astype(np.float32, copy=False)
    site_total_count = np.sum(total, axis=0).astype(np.float32, copy=False)
    site_support_samples = np.sum(total > 0.0, axis=0).astype(np.float32, copy=False)
    site_support_rate = site_support_samples / float(max(n_samples, 1))
    site_mean_depth = site_total_count / float(max(n_samples, 1))
    for name, values in {
        "site_ref_count": site_ref_count,
        "site_alt_count": site_alt_count,
        "site_other_count": site_other_count,
        "site_total_count": site_total_count,
        "site_support_samples": site_support_samples,
        "site_support_rate": site_support_rate,
        "site_mean_depth": site_mean_depth,
    }.items():
        _append(name, np.broadcast_to(values[None, :], ds.shape))

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
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x, y)
    return model


def predict_lightgbm_posterior(model: Any, features: np.ndarray, *, min_prob: float = 1e-6) -> np.ndarray:
    x = features.astype(np.float32, copy=False)
    proba = _predict_proba_feature_safe(model, x)
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
    proba = _predict_proba_feature_safe(model, x)
    if isinstance(proba, list):
        proba = np.stack(proba, axis=1)
    arr = np.asarray(proba, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected binary LightGBM predict_proba shape (n,2), got {arr.shape}")
    return np.clip(arr[:, 1], 0.0, 1.0).astype(np.float32, copy=False)


def lightgbm_feature_importance(
    model: Any,
    feature_names: Sequence[str],
    *,
    top_n: int = 40,
) -> list[dict[str, object]]:
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return []
    vals = np.asarray(importance, dtype=np.float64).reshape(-1)
    names = list(feature_names)
    if vals.size != len(names):
        names = [f"feature_{i}" for i in range(vals.size)]
    order = np.argsort(vals)[::-1]
    rows: list[dict[str, object]] = []
    for idx in order[: max(int(top_n), 0)].tolist():
        rows.append({"feature": str(names[idx]), "importance": float(vals[idx])})
    return rows


def train_stratified_lightgbm_multiclass_calibrator(
    *,
    features: np.ndarray,
    target: np.ndarray,
    position_index: np.ndarray,
    variant_strata: np.ndarray,
    fallback_variant_strata: np.ndarray | None = None,
    seed: int = 0,
    max_rows: int = 750_000,
    min_rows_per_stratum: int = 2_000,
    class_weight_mode: str = "balanced",
) -> tuple[Any, dict[int, Any], dict[int, Any], dict[str, object]]:
    x = features.astype(np.float32, copy=False)
    y = target.astype(np.int32, copy=False).reshape(-1)
    pos_idx = position_index.astype(np.int64, copy=False).reshape(-1)
    valid = (y >= 0) & (y <= 2) & (pos_idx >= 0) & (pos_idx < variant_strata.shape[0])
    if fallback_variant_strata is not None:
        valid &= pos_idx < fallback_variant_strata.shape[0]
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

    fallback_strata_models: dict[int, Any] = {}
    trained_fallback_strata = 0
    unique_fallback_strata = np.asarray([], dtype=np.int32)
    if fallback_variant_strata is not None:
        fallback_for_rows = fallback_variant_strata[pos_idx]
        unique_fallback_strata = np.unique(fallback_for_rows.astype(np.int32, copy=False))
        for s in unique_fallback_strata.tolist():
            m = fallback_for_rows == int(s)
            if int(np.sum(m)) < int(min_rows_per_stratum):
                continue
            ys = y[m]
            if np.unique(ys).size < 2:
                continue
            xs = x[m]
            model_s = train_lightgbm_multiclass_calibrator(
                xs,
                ys,
                seed=int(seed) + int(s) + 1009,
                class_weight_mode=class_weight_mode,
            )
            fallback_strata_models[int(s)] = model_s
            trained_fallback_strata += 1

    meta = {
        "n_train_rows": int(x.shape[0]),
        "n_unique_strata_seen": int(unique_strata.size),
        "n_trained_strata": int(trained_strata),
        "n_unique_fallback_strata_seen": int(unique_fallback_strata.size),
        "n_trained_fallback_strata": int(trained_fallback_strata),
        "fallback_hierarchy": "stratum_model -> broader_stratum_model -> global_model -> raw_gp",
    }
    return global_model, strata_models, fallback_strata_models, meta


def predict_stratified_lightgbm_posterior(
    *,
    features: np.ndarray,
    position_index: np.ndarray,
    variant_strata: np.ndarray,
    global_model: Any,
    strata_models: dict[int, Any] | None = None,
    fallback_variant_strata: np.ndarray | None = None,
    fallback_strata_models: dict[int, Any] | None = None,
    min_prob: float = 1e-6,
) -> np.ndarray:
    x = features.astype(np.float32, copy=False)
    pos_idx = position_index.astype(np.int64, copy=False).reshape(-1)
    if x.shape[0] != pos_idx.shape[0]:
        raise ValueError("features and position_index must have matching number of rows.")
    base = predict_lightgbm_posterior(global_model, x, min_prob=float(min_prob))
    out = base.copy()
    if fallback_variant_strata is not None and fallback_strata_models:
        fallback_strata = fallback_variant_strata[np.clip(pos_idx, 0, fallback_variant_strata.shape[0] - 1)]
        for s, model in fallback_strata_models.items():
            m = fallback_strata == int(s)
            if not np.any(m):
                continue
            out[m] = predict_lightgbm_posterior(model, x[m], min_prob=float(min_prob))
    if not strata_models:
        out = np.clip(out, float(min_prob), 1.0)
        out /= np.clip(np.sum(out, axis=1, keepdims=True), 1e-12, None)
        return out.astype(np.float32, copy=False)
    strata = variant_strata[np.clip(pos_idx, 0, variant_strata.shape[0] - 1)]
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
    call_rate_probability: np.ndarray | None = None,
    thresholds: np.ndarray | None = None,
    min_call_rate: float = 0.2,
    max_call_rate: float | None = None,
    balanced_accuracy_weight: float = 0.5,
    macro_f1_weight: float = 0.5,
    call_rate_weight: float = 0.03,
) -> tuple[float, float, float]:
    prob = correctness_probability.astype(np.float32, copy=False).reshape(-1)
    rate_prob = (
        prob
        if call_rate_probability is None
        else call_rate_probability.astype(np.float32, copy=False).reshape(-1)
    )
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
        call_rate = float(np.mean(rate_prob >= float(thr))) if rate_prob.size else float(np.mean(called))
        if call_rate < float(min_call_rate):
            continue
        if max_call_rate is not None and call_rate > float(max_call_rate):
            continue
        if not np.any(called):
            continue
        ytc = yt[called]
        ypc = yp[called]
        bacc = _balanced_accuracy_np(ytc, ypc)
        f1 = _macro_f1_np(ytc, ypc)
        if not np.isfinite(bacc) or not np.isfinite(f1):
            continue
        score = (
            float(balanced_accuracy_weight) * float(bacc)
            + float(macro_f1_weight) * float(f1)
            + float(call_rate_weight) * call_rate
        )
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_call_rate = call_rate
    if not np.isfinite(best_score):
        return 0.0, float("nan"), 1.0
    return best_thr, float(best_score), float(best_call_rate)


def _callability_objective(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    called: np.ndarray,
    call_rate_weight: float,
) -> dict[str, float]:
    yt = y_true.astype(np.int8, copy=False).reshape(-1)
    yp = y_pred.astype(np.int8, copy=False).reshape(-1)
    keep = called.astype(bool, copy=False).reshape(-1) & (yt >= 0) & (yp >= 0)
    call_rate = float(np.mean(keep)) if keep.size else 0.0
    if not np.any(keep):
        return {
            "objective": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "call_rate": call_rate,
        }
    bacc = _balanced_accuracy_np(yt[keep], yp[keep])
    f1 = _macro_f1_np(yt[keep], yp[keep])
    objective = 0.5 * float(bacc) + 0.5 * float(f1) + float(call_rate_weight) * call_rate
    return {
        "objective": float(objective),
        "balanced_accuracy": float(bacc),
        "macro_f1": float(f1),
        "call_rate": float(call_rate),
    }


def _hardcall_population_stats(call: np.ndarray, *, ploidy: int = 2) -> dict[str, float]:
    calls = call.astype(np.int16, copy=False)
    valid = calls >= 0
    if not np.any(valid):
        return {
            "mean_maf": float("nan"),
            "mean_het_rate": float("nan"),
            "mean_call_rate": 0.0,
        }
    denom = np.sum(valid, axis=0)
    call_rate = denom / max(calls.shape[0], 1)
    alt_sum = np.sum(np.where(valid, calls, 0), axis=0).astype(np.float32, copy=False)
    af = np.divide(alt_sum, np.clip(denom.astype(np.float32) * float(max(int(ploidy), 1)), 1.0, None))
    maf = np.minimum(np.clip(af, 0.0, 1.0), 1.0 - np.clip(af, 0.0, 1.0))
    het = np.divide(np.sum((calls == 1) & valid, axis=0), np.clip(denom, 1, None))
    use = denom > 0
    return {
        "mean_maf": float(np.mean(maf[use])) if np.any(use) else float("nan"),
        "mean_het_rate": float(np.mean(het[use])) if np.any(use) else float("nan"),
        "mean_call_rate": float(np.mean(call_rate)) if call_rate.size else 0.0,
    }


def _hardcall_stat_shift(lhs: dict[str, float], rhs: dict[str, float], key: str) -> float:
    a = float(lhs.get(key, float("nan")))
    b = float(rhs.get(key, float("nan")))
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan")
    return float(abs(a - b))


def build_standard_callability_feature_matrix(
    *,
    dosage: np.ndarray,
    posterior: np.ndarray,
    depth: np.ndarray | None = None,
    ref_count: np.ndarray | None = None,
    alt_count: np.ndarray | None = None,
    other_count: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    variant_maf: np.ndarray | None = None,
    variant_info: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the lightweight default hard-call callability feature matrix.

    These are diagnostic features, not posterior-transform knobs: the default
    model learns whether the raw HMM argmax call should be emitted.
    """
    ds = dosage.astype(np.float32, copy=False)
    gp = posterior.astype(np.float32, copy=False)
    if gp.ndim != 3 or gp.shape[2] != 3:
        raise ValueError("standard callability currently requires diploid biallelic GP with shape (samples, variants, 3).")
    if gp.shape[:2] != ds.shape:
        raise ValueError(f"posterior shape {gp.shape} does not match dosage shape {ds.shape}.")
    n_samples, n_positions = ds.shape
    conf, margin, entropy = posterior_confidence_metrics(gp)
    depth_arr = np.zeros_like(ds, dtype=np.float32) if depth is None else depth.astype(np.float32, copy=False)
    ref_arr = np.zeros_like(ds, dtype=np.float32) if ref_count is None else ref_count.astype(np.float32, copy=False)
    alt_arr = np.zeros_like(ds, dtype=np.float32) if alt_count is None else alt_count.astype(np.float32, copy=False)
    oth_arr = np.zeros_like(ds, dtype=np.float32) if other_count is None else other_count.astype(np.float32, copy=False)
    total = ref_arr + alt_arr + oth_arr
    observed_alt_fraction = (alt_arr + 0.5) / np.clip(total + 1.0, 1.0, None)
    expected_alt_fraction = np.clip(ds / 2.0, 0.0, 1.0)
    ref_alt_balance = np.abs(observed_alt_fraction - expected_alt_fraction).astype(np.float32, copy=False)
    dosage_integer_distance = np.abs(ds - np.rint(ds)).astype(np.float32, copy=False)

    if variant_maf is None:
        variant_maf = compute_variant_maf_from_posterior(gp)
    if variant_info is None:
        variant_info = compute_info_score_per_variant(gp)
    hwe = compute_hwe_features_from_posterior(gp)
    hwe_dev = np.nan_to_num(hwe["hwe_deviation"].astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=0.0)
    if support_mask is not None:
        support = support_mask.astype(bool, copy=False)
    else:
        support = total > 0.0
    support_rate = np.mean(support.astype(np.float32, copy=False), axis=0).astype(np.float32, copy=False)
    missingness = (1.0 - support_rate).astype(np.float32, copy=False)
    mean_depth = np.mean(depth_arr, axis=0).astype(np.float32, copy=False)

    broadcast = lambda v: np.broadcast_to(v.astype(np.float32, copy=False)[None, :], (n_samples, n_positions))
    cols = [
        conf,
        margin,
        entropy,
        dosage_integer_distance,
        broadcast(np.nan_to_num(variant_maf.astype(np.float32, copy=False), nan=0.0)),
        broadcast(hwe_dev),
        broadcast(missingness),
        broadcast(support_rate),
        depth_arr,
        np.log1p(np.clip(depth_arr, 0.0, None)).astype(np.float32, copy=False),
        broadcast(np.nan_to_num(variant_info.astype(np.float32, copy=False), nan=0.0, posinf=1.0, neginf=-5.0)),
        observed_alt_fraction,
        ref_alt_balance,
        total,
        broadcast(mean_depth),
    ]
    names = [
        "max_gp",
        "gp_margin",
        "entropy",
        "dosage_integer_distance",
        "variant_maf",
        "variant_hwe_deviation",
        "variant_missingness",
        "site_support_rate",
        "depth",
        "log_depth",
        "variant_info",
        "observed_alt_fraction",
        "ref_alt_balance",
        "total_count",
        "site_mean_depth",
    ]
    features = np.stack([c.reshape(-1).astype(np.float32, copy=False) for c in cols], axis=1)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False), names


def _callability_group_decision(
    *,
    variant_mask: np.ndarray,
    truth: np.ndarray,
    argmax: np.ndarray,
    threshold_mask: np.ndarray,
    call_prob: np.ndarray,
    stitch_called_full: np.ndarray,
    threshold_grid: np.ndarray,
    min_eval_rows: int,
    min_call_rate: float,
    call_rate_weight: float,
    stitch_gp_threshold: float,
    bound_to_stitch: bool,
    min_call_rate_delta: float,
    max_call_rate_delta: float,
    min_objective_improvement: float,
    max_hardcall_maf_shift: float,
    max_hardcall_het_shift: float,
    level: str,
) -> dict[str, Any]:
    cols = np.asarray(variant_mask, dtype=bool)
    n_variants = int(np.sum(cols))
    if n_variants <= 0:
        return {"accepted": False, "reason": "empty_variant_set", "level": level, "n_variants": 0, "n_labeled": 0}
    label_mask = threshold_mask[:, cols].astype(bool, copy=False)
    n_labeled = int(np.sum(label_mask))
    if n_labeled < int(min_eval_rows):
        return {
            "accepted": False,
            "reason": "insufficient_labels",
            "level": level,
            "n_variants": n_variants,
            "n_labeled": n_labeled,
            "min_eval_rows": int(min_eval_rows),
        }

    y_true = truth[:, cols][label_mask].astype(np.int8, copy=False)
    y_pred = argmax[:, cols][label_mask].astype(np.int8, copy=False)
    prob_labeled = call_prob[:, cols][label_mask].astype(np.float32, copy=False)
    stitch_labeled = stitch_called_full[:, cols][label_mask].astype(bool, copy=False)
    stitch_stats = _callability_objective(
        y_true=y_true,
        y_pred=y_pred,
        called=stitch_labeled,
        call_rate_weight=float(call_rate_weight),
    )
    parity_call_rate = float(np.mean(stitch_called_full[:, cols])) if stitch_called_full[:, cols].size else float(
        stitch_stats.get("call_rate", 0.0)
    )
    if bool(bound_to_stitch):
        effective_min_call_rate = max(0.0, parity_call_rate + float(min_call_rate_delta))
        effective_max_call_rate = min(1.0, parity_call_rate + float(max_call_rate_delta))
        if effective_max_call_rate < effective_min_call_rate:
            effective_max_call_rate = effective_min_call_rate
    else:
        effective_min_call_rate = float(min_call_rate)
        effective_max_call_rate = None

    threshold, score, call_rate = optimize_call_correctness_threshold(
        correctness_probability=prob_labeled,
        y_true=y_true,
        y_pred=y_pred,
        call_rate_probability=call_prob[:, cols].reshape(-1),
        thresholds=threshold_grid,
        min_call_rate=float(effective_min_call_rate),
        max_call_rate=effective_max_call_rate,
        call_rate_weight=float(call_rate_weight),
    )
    calibrated_labeled = prob_labeled >= float(threshold)
    calibrated_stats = _callability_objective(
        y_true=y_true,
        y_pred=y_pred,
        called=calibrated_labeled,
        call_rate_weight=float(call_rate_weight),
    )
    reasons: list[str] = []
    stitch_objective = float(stitch_stats.get("objective", float("nan")))
    calibrated_objective = float(calibrated_stats.get("objective", float("nan")))
    if not np.isfinite(score):
        reasons.append("no_threshold")
    if bool(bound_to_stitch):
        if call_rate < float(effective_min_call_rate) - 1e-6:
            reasons.append("call_rate_below_stitch_bound")
        if effective_max_call_rate is not None and call_rate > float(effective_max_call_rate) + 1e-6:
            reasons.append("call_rate_above_stitch_bound")
        if not np.isfinite(calibrated_objective):
            reasons.append("nonfinite_calibrated_objective")
        elif np.isfinite(stitch_objective) and calibrated_objective < stitch_objective + float(min_objective_improvement):
            reasons.append("no_heldout_objective_improvement")

    raw_call = np.where(stitch_called_full[:, cols], argmax[:, cols], -1).astype(np.int8, copy=False)
    calibrated_call = np.where(call_prob[:, cols] >= float(threshold), argmax[:, cols], -1).astype(np.int8, copy=False)
    raw_pop_stats = _hardcall_population_stats(raw_call, ploidy=2)
    calibrated_pop_stats = _hardcall_population_stats(calibrated_call, ploidy=2)
    maf_shift = _hardcall_stat_shift(raw_pop_stats, calibrated_pop_stats, "mean_maf")
    het_shift = _hardcall_stat_shift(raw_pop_stats, calibrated_pop_stats, "mean_het_rate")
    if np.isfinite(maf_shift) and maf_shift > float(max_hardcall_maf_shift):
        reasons.append("hardcall_maf_shift_guard")
    if np.isfinite(het_shift) and het_shift > float(max_hardcall_het_shift):
        reasons.append("hardcall_het_shift_guard")

    return {
        "accepted": not reasons,
        "reason": "ok" if not reasons else ",".join(sorted(set(reasons))),
        "level": str(level),
        "n_variants": n_variants,
        "n_labeled": n_labeled,
        "threshold": float(threshold),
        "objective": float(score),
        "call_rate": float(call_rate),
        "effective_min_call_rate": float(effective_min_call_rate),
        "effective_max_call_rate": (None if effective_max_call_rate is None else float(effective_max_call_rate)),
        "stitch_gp_threshold": float(stitch_gp_threshold),
        "heldout_stitch_objective": stitch_stats,
        "heldout_calibrated_objective": calibrated_stats,
        "population_stitch_no_call_stats": raw_pop_stats,
        "population_calibrated_stats": calibrated_pop_stats,
        "population_mean_maf_shift": float(maf_shift),
        "population_mean_het_shift": float(het_shift),
    }


def _fit_per_variant_callability_decisions(
    *,
    truth: np.ndarray,
    argmax: np.ndarray,
    threshold_mask: np.ndarray,
    call_prob: np.ndarray,
    gp_conf: np.ndarray,
    posterior: np.ndarray,
    support_mask: np.ndarray | None,
    depth: np.ndarray | None,
    threshold_grid: np.ndarray,
    min_train_rows: int,
    min_call_rate: float,
    call_rate_weight: float,
    stitch_gp_threshold: float,
    bound_to_stitch: bool,
    min_call_rate_delta: float,
    max_call_rate_delta: float,
    min_objective_improvement: float,
    max_hardcall_maf_shift: float,
    max_hardcall_het_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    gp = posterior.astype(np.float32, copy=False)
    n_positions = int(gp.shape[1])
    stitch_called_full = gp_conf >= float(stitch_gp_threshold)
    fallback = np.ones(n_positions, dtype=bool)
    threshold_by_variant = np.full(n_positions, np.nan, dtype=np.float32)
    source = np.full(n_positions, "stitch_no_call", dtype=object)
    reason = np.full(n_positions, "not_evaluated", dtype=object)
    objective_delta = np.full(n_positions, np.nan, dtype=np.float32)
    n_labeled_by_variant = np.sum(threshold_mask.astype(bool, copy=False), axis=0).astype(np.int32, copy=False)

    min_local_rows = max(8, min(int(min_train_rows), 32))
    per_variant_min_rows = max(6, min(16, int(min_train_rows)))

    variant_maf = compute_variant_maf_from_posterior(gp)
    variant_info = compute_info_score_per_variant(gp)
    hwe_dev = compute_hwe_features_from_posterior(gp)["hwe_deviation"]
    if support_mask is not None:
        support_rate = np.mean(support_mask.astype(np.float32, copy=False), axis=0).astype(np.float32, copy=False)
    else:
        support_rate = None
    missingness = None if support_rate is None else (1.0 - support_rate).astype(np.float32, copy=False)
    if depth is not None:
        mean_depth = np.mean(depth.astype(np.float32, copy=False), axis=0).astype(np.float32, copy=False)
    else:
        mean_depth = None
    strata, strata_meta = compute_variant_strata(
        variant_maf,
        variant_info,
        support_rate=support_rate,
        missingness=missingness,
        hwe_deviation=hwe_dev,
        mean_depth=mean_depth,
    )

    decision_rows: list[dict[str, object]] = []

    def evaluate(mask: np.ndarray, level: str, min_rows: int) -> dict[str, Any]:
        return _callability_group_decision(
            variant_mask=mask,
            truth=truth,
            argmax=argmax,
            threshold_mask=threshold_mask,
            call_prob=call_prob,
            stitch_called_full=stitch_called_full,
            threshold_grid=threshold_grid,
            min_eval_rows=int(min_rows),
            min_call_rate=float(min_call_rate),
            call_rate_weight=float(call_rate_weight),
            stitch_gp_threshold=float(stitch_gp_threshold),
            bound_to_stitch=bool(bound_to_stitch),
            min_call_rate_delta=float(min_call_rate_delta),
            max_call_rate_delta=float(max_call_rate_delta),
            min_objective_improvement=float(min_objective_improvement),
            max_hardcall_maf_shift=float(max_hardcall_maf_shift),
            max_hardcall_het_shift=float(max_hardcall_het_shift),
            level=level,
        )

    def apply_decision(mask: np.ndarray, dec: dict[str, Any], *, overwrite: bool) -> None:
        nonlocal threshold_by_variant, fallback, source, reason, objective_delta
        cols = np.asarray(mask, dtype=bool)
        if not bool(dec.get("accepted", False)):
            update = cols & fallback
            reason[update] = str(dec.get("reason", "rejected"))
            return
        update = cols if overwrite else (cols & fallback)
        if not np.any(update):
            return
        threshold_by_variant[update] = float(dec.get("threshold", np.nan))
        fallback[update] = False
        source[update] = str(dec.get("level", "calibrated"))
        stitch_obj = float((dec.get("heldout_stitch_objective") or {}).get("objective", float("nan")))
        cal_obj = float((dec.get("heldout_calibrated_objective") or {}).get("objective", float("nan")))
        if np.isfinite(stitch_obj) and np.isfinite(cal_obj):
            objective_delta[update] = float(cal_obj - stitch_obj)
        reason[update] = "ok"

    global_decision = evaluate(np.ones(n_positions, dtype=bool), "global", int(min_train_rows))
    decision_rows.append(global_decision)
    apply_decision(np.ones(n_positions, dtype=bool), global_decision, overwrite=False)

    for stratum in np.unique(strata).tolist():
        mask = strata == int(stratum)
        dec = evaluate(mask, f"feature_stratum:{int(stratum)}", int(min_train_rows))
        decision_rows.append(dec)
        apply_decision(mask, dec, overwrite=True)

    local_window_variants = 101
    for start in range(0, n_positions, local_window_variants):
        stop = min(start + local_window_variants, n_positions)
        mask = np.zeros(n_positions, dtype=bool)
        mask[start:stop] = True
        dec = evaluate(mask, f"local_window:{start}-{stop}", int(min_local_rows))
        decision_rows.append(dec)
        apply_decision(mask, dec, overwrite=True)

    candidate_per_variant = np.flatnonzero(n_labeled_by_variant >= int(per_variant_min_rows))
    for j in candidate_per_variant.tolist():
        mask = np.zeros(n_positions, dtype=bool)
        mask[int(j)] = True
        dec = evaluate(mask, f"per_snp:{int(j)}", int(per_variant_min_rows))
        decision_rows.append(dec)
        apply_decision(mask, dec, overwrite=True)

    fallback_threshold = float(stitch_gp_threshold)
    effective_threshold = np.where(fallback, fallback_threshold, threshold_by_variant).astype(np.float32, copy=False)
    gate_probability = np.where(fallback[None, :], gp_conf, call_prob).astype(np.float32, copy=False)

    raw_call_full = np.where(stitch_called_full, argmax, -1).astype(np.int8, copy=False)
    gated_call_full = np.where(gate_probability >= effective_threshold[None, :], argmax, -1).astype(np.int8, copy=False)
    raw_pop = _hardcall_population_stats(raw_call_full, ploidy=2)
    gated_pop = _hardcall_population_stats(gated_call_full, ploidy=2)
    maf_shift_by_variant = np.full(n_positions, np.nan, dtype=np.float32)
    het_shift_by_variant = np.full(n_positions, np.nan, dtype=np.float32)
    call_rate_stitch = np.mean(stitch_called_full.astype(np.float32, copy=False), axis=0).astype(np.float32, copy=False)
    call_rate_calibrated = np.mean((gate_probability >= effective_threshold[None, :]).astype(np.float32, copy=False), axis=0)
    for j in range(n_positions):
        raw_stats = _hardcall_population_stats(raw_call_full[:, j : j + 1], ploidy=2)
        gated_stats = _hardcall_population_stats(gated_call_full[:, j : j + 1], ploidy=2)
        maf_shift_by_variant[j] = _hardcall_stat_shift(raw_stats, gated_stats, "mean_maf")
        het_shift_by_variant[j] = _hardcall_stat_shift(raw_stats, gated_stats, "mean_het_rate")

    per_snp_guard_reason = np.full(n_positions, "", dtype=object)
    for j in np.flatnonzero(~fallback).tolist():
        reasons: list[str] = []
        delta = float(call_rate_calibrated[int(j)] - call_rate_stitch[int(j)])
        if bool(bound_to_stitch):
            if delta < float(min_call_rate_delta) - 1e-6:
                reasons.append("per_snp_call_rate_below_bound")
            if delta > float(max_call_rate_delta) + 1e-6:
                reasons.append("per_snp_call_rate_above_bound")
        if np.isfinite(maf_shift_by_variant[int(j)]) and maf_shift_by_variant[int(j)] > float(max_hardcall_maf_shift):
            reasons.append("per_snp_maf_shift_guard")
        if np.isfinite(het_shift_by_variant[int(j)]) and het_shift_by_variant[int(j)] > float(max_hardcall_het_shift):
            reasons.append("per_snp_het_shift_guard")
        if reasons:
            fallback[int(j)] = True
            threshold_by_variant[int(j)] = np.nan
            source[int(j)] = "stitch_no_call"
            reason[int(j)] = ",".join(reasons)
            objective_delta[int(j)] = np.nan
            per_snp_guard_reason[int(j)] = ",".join(reasons)

    if np.any(per_snp_guard_reason != ""):
        effective_threshold = np.where(fallback, fallback_threshold, threshold_by_variant).astype(np.float32, copy=False)
        gate_probability = np.where(fallback[None, :], gp_conf, call_prob).astype(np.float32, copy=False)
        gated_call_full = np.where(gate_probability >= effective_threshold[None, :], argmax, -1).astype(np.int8, copy=False)
        gated_pop = _hardcall_population_stats(gated_call_full, ploidy=2)
        call_rate_calibrated = np.mean(
            (gate_probability >= effective_threshold[None, :]).astype(np.float32, copy=False),
            axis=0,
        )
        for j in range(n_positions):
            raw_stats = _hardcall_population_stats(raw_call_full[:, j : j + 1], ploidy=2)
            gated_stats = _hardcall_population_stats(gated_call_full[:, j : j + 1], ploidy=2)
            maf_shift_by_variant[j] = _hardcall_stat_shift(raw_stats, gated_stats, "mean_maf")
            het_shift_by_variant[j] = _hardcall_stat_shift(raw_stats, gated_stats, "mean_het_rate")

    decisions = pd.DataFrame(
        {
            "variant_index": np.arange(n_positions, dtype=np.int32),
            "n_labeled": n_labeled_by_variant,
            "calibration_used": ~fallback,
            "fallback_to_stitch": fallback,
            "decision_source": source,
            "fallback_reason": reason,
            "threshold": effective_threshold,
            "call_rate_stitch": call_rate_stitch,
            "call_rate_calibrated": call_rate_calibrated.astype(np.float32, copy=False),
            "call_rate_delta": (call_rate_calibrated - call_rate_stitch).astype(np.float32, copy=False),
            "objective_delta": objective_delta,
            "maf": variant_maf.astype(np.float32, copy=False),
            "info": variant_info.astype(np.float32, copy=False),
            "hwe_deviation": hwe_dev.astype(np.float32, copy=False),
            "support_rate": (
                np.full(n_positions, np.nan, dtype=np.float32)
                if support_rate is None
                else support_rate.astype(np.float32, copy=False)
            ),
            "missingness": (
                np.full(n_positions, np.nan, dtype=np.float32)
                if missingness is None
                else missingness.astype(np.float32, copy=False)
            ),
            "mean_depth": (
                np.full(n_positions, np.nan, dtype=np.float32)
                if mean_depth is None
                else mean_depth.astype(np.float32, copy=False)
            ),
            "maf_shift": maf_shift_by_variant,
            "het_shift": het_shift_by_variant,
        }
    )
    n_used = int(np.sum(~fallback))
    reason_counts = (
        decisions.loc[decisions["fallback_to_stitch"], "fallback_reason"].value_counts(dropna=False).to_dict()
    )
    level_counts = decisions.loc[decisions["calibration_used"], "decision_source"].value_counts(dropna=False).to_dict()
    summary = {
        "n_snps": int(n_positions),
        "n_snps_calibration_used": n_used,
        "n_snps_fallback_to_stitch": int(np.sum(fallback)),
        "calibration_used_fraction": float(n_used / max(n_positions, 1)),
        "fallback_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "decision_source_counts": {str(k): int(v) for k, v in level_counts.items()},
        "local_window_variants": int(local_window_variants),
        "per_variant_min_rows": int(per_variant_min_rows),
        "local_min_rows": int(min_local_rows),
        "strata": strata_meta,
        "population_stitch_no_call_stats": raw_pop,
        "population_calibrated_or_fallback_stats": gated_pop,
        "population_mean_maf_shift": _hardcall_stat_shift(raw_pop, gated_pop, "mean_maf"),
        "population_mean_het_shift": _hardcall_stat_shift(raw_pop, gated_pop, "mean_het_rate"),
        "group_decisions": decision_rows,
    }
    return gate_probability, effective_threshold, fallback, threshold_by_variant, decisions, summary


def train_standard_callability_model(
    *,
    posterior: np.ndarray,
    dosage: np.ndarray,
    truth_genotype: np.ndarray,
    train_mask: np.ndarray | None = None,
    depth: np.ndarray | None = None,
    ref_count: np.ndarray | None = None,
    alt_count: np.ndarray | None = None,
    other_count: np.ndarray | None = None,
    support_mask: np.ndarray | None = None,
    max_train_rows: int = 750_000,
    model_type: str = "lightgbm",
    min_train_rows: int = 64,
    min_call_rate: float = 0.80,
    call_rate_weight: float = 0.03,
    stitch_gp_threshold: float = 0.9,
    bound_to_stitch: bool = True,
    min_call_rate_delta: float = -0.02,
    max_call_rate_delta: float = 0.05,
    validation_site_fraction: float = 0.20,
    min_objective_improvement: float = 0.0,
    max_hardcall_maf_shift: float = 0.08,
    max_hardcall_het_shift: float = 0.12,
    decision_mode: str = "per_snp_hierarchical",
    seed: int = 0,
    learning_rate: float = 0.08,
    l2: float = 1e-3,
    iterations: int = 160,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Fit the default continuous callability model.

    The raw HMM GP is not modified. The returned matrix is P(argmax call is
    correct), intended for hard-call gating only. By default, the fitted model
    is evaluated per SNP with local/stratum/global fallback to STITCH no-call.
    The model boundary is intentionally sklearn-compatible because
    sklearn/LightGBM require NumPy inputs; HMM/JAX arrays should be converted
    once at this boundary, not moved back and forth during the HMM itself.
    """
    gp = posterior.astype(np.float32, copy=False)
    truth = truth_genotype.astype(np.int16, copy=False)
    if gp.ndim != 3 or gp.shape[2] != 3:
        return None, {"status": "skipped_non_diploid", "mode": "standard_callability"}
    if truth.shape != gp.shape[:2]:
        raise ValueError(f"truth_genotype shape {truth.shape} does not match posterior shape {gp.shape[:2]}.")
    train = truth >= 0
    if train_mask is not None:
        train &= train_mask.astype(bool, copy=False)
    if int(np.sum(train)) < int(min_train_rows):
        return None, {
            "status": "skipped_insufficient_truth",
            "mode": "standard_callability",
            "n_train_rows": int(np.sum(train)),
            "min_train_rows": int(min_train_rows),
        }

    features, feature_names = build_standard_callability_feature_matrix(
        dosage=dosage,
        posterior=gp,
        depth=depth,
        ref_count=ref_count,
        alt_count=alt_count,
        other_count=other_count,
        support_mask=support_mask,
    )
    argmax = np.argmax(gp, axis=2).astype(np.int8, copy=False)
    correct = (argmax == truth).astype(np.int8, copy=False)
    rng = np.random.default_rng(int(seed) + 104729)
    flat_train = train.reshape(-1)
    threshold_mask = train.copy()
    fit_mask = train.copy()
    validation_status = "in_sample"
    val_frac = float(np.clip(float(validation_site_fraction), 0.0, 0.8))
    train_positions = np.flatnonzero(np.any(train, axis=0))
    if val_frac > 0.0 and train_positions.size >= 4:
        n_val = int(round(float(train_positions.size) * val_frac))
        n_val = min(max(n_val, 1), int(train_positions.size) - 1)
        val_positions = np.sort(rng.choice(train_positions, size=n_val, replace=False))
        val_site_mask = np.zeros(train.shape[1], dtype=bool)
        val_site_mask[val_positions] = True
        candidate_threshold_mask = train & val_site_mask[None, :]
        candidate_fit_mask = train & ~val_site_mask[None, :]
        if int(np.sum(candidate_fit_mask)) >= int(min_train_rows) and int(np.sum(candidate_threshold_mask)) > 0:
            fit_mask = candidate_fit_mask
            threshold_mask = candidate_threshold_mask
            validation_status = "heldout_sites"
    flat_fit = fit_mask.reshape(-1)
    flat_threshold = threshold_mask.reshape(-1)
    y_all = correct.reshape(-1).astype(np.float32, copy=False)
    x_fit_all = features[flat_fit]
    y_fit_all = y_all[flat_fit]
    if np.unique(y_fit_all.astype(np.int8, copy=False)).size < 2:
        return None, {
            "status": "skipped_single_class_correctness",
            "mode": "standard_callability",
            "n_train_rows": int(np.sum(flat_train)),
            "n_fit_rows": int(y_fit_all.shape[0]),
            "n_threshold_rows": int(np.sum(flat_threshold)),
            "threshold_validation": validation_status,
            "correct_rate": float(np.mean(y_fit_all)) if y_fit_all.size else float("nan"),
        }

    take = _balanced_row_indices(y_fit_all.astype(np.int8, copy=False), max_rows=int(max_train_rows), seed=int(seed))
    x_fit = x_fit_all[take]
    y_fit = y_fit_all[take].astype(np.int32, copy=False)
    model_name = str(model_type or "sklearn_logistic").lower()
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - dependency failure path.
        raise ImportError(
            "standard_callability requires scikit-learn in the active environment."
        ) from exc

    if model_name == "lightgbm":
        try:
            import lightgbm as lgb
        except Exception as exc:  # pragma: no cover - dependency failure path.
            raise ImportError(
                "standard_callability model_type='lightgbm' requires lightgbm in the active environment."
            ) from exc
        estimator = lgb.LGBMClassifier(
            objective="binary",
            random_state=int(seed),
            n_estimators=max(int(iterations), 50),
            learning_rate=float(learning_rate),
            max_depth=-1,
            num_leaves=47,
            min_data_in_leaf=96,
            class_weight="balanced",
            n_jobs=1,
            verbosity=-1,
        )
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ]
        )
    elif model_name in {"sklearn_logistic", "logistic", "sklearn"}:
        estimator = LogisticRegression(
            class_weight="balanced",
            C=float(1.0 / max(float(l2), 1e-6)),
            max_iter=max(int(iterations), 100),
            random_state=int(seed),
            solver="lbfgs",
        )
        model = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
        model_name = "sklearn_logistic"
    else:
        raise ValueError(f"Unknown standard callability model_type {model_type!r}.")

    if model_name == "lightgbm":
        model.fit(_feature_frame(x_fit, feature_names), y_fit)
    else:
        model.fit(x_fit.astype(np.float32, copy=False), y_fit)
    prob_flat = np.empty(features.shape[0], dtype=np.float32)
    chunk_size = 1_000_000
    for start in range(0, features.shape[0], chunk_size):
        stop = min(start + chunk_size, features.shape[0])
        if model_name == "lightgbm":
            proba = _predict_proba_feature_safe(model, features[start:stop].astype(np.float32, copy=False), feature_names)
        else:
            proba = model.predict_proba(features[start:stop].astype(np.float32, copy=False))
        arr = np.asarray(proba, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected callability predict_proba shape (n,2), got {arr.shape}")
        prob_flat[start:stop] = np.clip(arr[:, 1], 0.0, 1.0)
    call_prob = prob_flat.reshape(gp.shape[:2]).astype(np.float32, copy=False)
    y_true_flat = truth.reshape(-1)[flat_threshold]
    y_pred_flat = argmax.reshape(-1)[flat_threshold]
    prob_threshold = call_prob.reshape(-1)[flat_threshold]
    gp_conf = np.max(np.nan_to_num(gp, nan=0.0), axis=2).astype(np.float32, copy=False)
    stitch_called_full = gp_conf >= float(stitch_gp_threshold)
    stitch_called_threshold = stitch_called_full.reshape(-1)[flat_threshold]
    stitch_stats_threshold = _callability_objective(
        y_true=y_true_flat,
        y_pred=y_pred_flat,
        called=stitch_called_threshold,
        call_rate_weight=float(call_rate_weight),
    )
    parity_call_rate_labeled = float(stitch_stats_threshold.get("call_rate", 0.0))
    parity_call_rate = float(np.mean(stitch_called_full)) if stitch_called_full.size else parity_call_rate_labeled
    if bool(bound_to_stitch):
        effective_min_call_rate = max(0.0, parity_call_rate + float(min_call_rate_delta))
        effective_max_call_rate = min(1.0, parity_call_rate + float(max_call_rate_delta))
        if effective_max_call_rate < effective_min_call_rate:
            effective_max_call_rate = effective_min_call_rate
    else:
        effective_min_call_rate = float(min_call_rate)
        effective_max_call_rate = None
    prob_all = call_prob.reshape(-1)
    finite_prob = prob_all[np.isfinite(prob_all)]
    threshold_grid = np.linspace(0.0, 0.999, 140, dtype=np.float32)
    if finite_prob.size:
        quantile_grid = np.unique(
            np.quantile(finite_prob, np.linspace(0.0, 0.999, 180, dtype=np.float32)).astype(np.float32)
        )
        threshold_grid = np.unique(np.concatenate([threshold_grid, quantile_grid]).astype(np.float32, copy=False))
    threshold, score, call_rate = optimize_call_correctness_threshold(
        correctness_probability=prob_threshold,
        y_true=y_true_flat,
        y_pred=y_pred_flat,
        call_rate_probability=call_prob.reshape(-1),
        thresholds=threshold_grid,
        min_call_rate=float(effective_min_call_rate),
        max_call_rate=effective_max_call_rate,
        call_rate_weight=float(call_rate_weight),
    )
    calibrated_called_threshold = prob_threshold >= float(threshold)
    calibrated_stats_threshold = _callability_objective(
        y_true=y_true_flat,
        y_pred=y_pred_flat,
        called=calibrated_called_threshold,
        call_rate_weight=float(call_rate_weight),
    )
    fallback_reasons: list[str] = []
    stitch_objective = float(stitch_stats_threshold.get("objective", float("nan")))
    calibrated_objective = float(calibrated_stats_threshold.get("objective", float("nan")))
    if bool(bound_to_stitch):
        if not np.isfinite(score):
            fallback_reasons.append("no_threshold_within_stitch_call_rate_bound")
        if call_rate < float(effective_min_call_rate) - 1e-6:
            fallback_reasons.append("call_rate_below_stitch_bound")
        if effective_max_call_rate is not None and call_rate > float(effective_max_call_rate) + 1e-6:
            fallback_reasons.append("call_rate_above_stitch_bound")
        if not np.isfinite(calibrated_objective):
            fallback_reasons.append("nonfinite_calibrated_objective")
        elif np.isfinite(stitch_objective) and calibrated_objective < stitch_objective + float(min_objective_improvement):
            fallback_reasons.append("no_heldout_objective_improvement")
    raw_call_full = np.where(stitch_called_full, argmax, -1).astype(np.int8, copy=False)
    calibrated_call_full = np.where(call_prob >= float(threshold), argmax, -1).astype(np.int8, copy=False)
    raw_pop_stats = _hardcall_population_stats(raw_call_full, ploidy=2)
    calibrated_pop_stats = _hardcall_population_stats(calibrated_call_full, ploidy=2)
    maf_shift = _hardcall_stat_shift(raw_pop_stats, calibrated_pop_stats, "mean_maf")
    het_shift = _hardcall_stat_shift(raw_pop_stats, calibrated_pop_stats, "mean_het_rate")
    if np.isfinite(maf_shift) and maf_shift > float(max_hardcall_maf_shift):
        fallback_reasons.append("hardcall_maf_shift_guard")
    if np.isfinite(het_shift) and het_shift > float(max_hardcall_het_shift):
        fallback_reasons.append("hardcall_het_shift_guard")
    mode_name = str(decision_mode or "per_snp_hierarchical")
    if mode_name not in {"per_snp_hierarchical", "global"}:
        raise ValueError(
            "decision_mode must be 'per_snp_hierarchical' or 'global', "
            f"got {decision_mode!r}."
        )
    per_variant_gate_probability: np.ndarray | None = None
    per_variant_threshold: np.ndarray | None = None
    per_variant_fallback: np.ndarray | None = None
    per_variant_raw_threshold: np.ndarray | None = None
    variant_decisions: pd.DataFrame | None = None
    per_variant_summary: dict[str, Any] = {}
    if mode_name == "per_snp_hierarchical":
        try:
            (
                per_variant_gate_probability,
                per_variant_threshold,
                per_variant_fallback,
                per_variant_raw_threshold,
                variant_decisions,
                per_variant_summary,
            ) = _fit_per_variant_callability_decisions(
                truth=truth,
                argmax=argmax,
                threshold_mask=threshold_mask,
                call_prob=call_prob,
                gp_conf=gp_conf,
                posterior=gp,
                support_mask=support_mask,
                depth=depth,
                threshold_grid=threshold_grid,
                min_train_rows=int(min_train_rows),
                min_call_rate=float(min_call_rate),
                call_rate_weight=float(call_rate_weight),
                stitch_gp_threshold=float(stitch_gp_threshold),
                bound_to_stitch=bool(bound_to_stitch),
                min_call_rate_delta=float(min_call_rate_delta),
                max_call_rate_delta=float(max_call_rate_delta),
                min_objective_improvement=float(min_objective_improvement),
                max_hardcall_maf_shift=float(max_hardcall_maf_shift),
                max_hardcall_het_shift=float(max_hardcall_het_shift),
            )
        except Exception as exc:
            per_variant_summary = {
                "status": "failed",
                "reason": str(exc),
                "n_snps": int(gp.shape[1]),
                "n_snps_calibration_used": 0,
                "n_snps_fallback_to_stitch": int(gp.shape[1]),
                "fallback_reason_counts": {"per_variant_decision_failed": int(gp.shape[1])},
                "decision_source_counts": {},
            }
        n_snps_used = int(per_variant_summary.get("n_snps_calibration_used", 0))
        n_snps_total = int(per_variant_summary.get("n_snps", gp.shape[1]))
        if n_snps_used <= 0:
            status = "fallback_to_stitch_no_call"
        elif n_snps_used < n_snps_total:
            status = "partial_per_snp"
        else:
            status = "ok"
    else:
        n_snps_total = int(gp.shape[1])
        if fallback_reasons:
            status = "fallback_to_stitch_no_call"
            per_variant_summary = {
                "n_snps": n_snps_total,
                "n_snps_calibration_used": 0,
                "n_snps_fallback_to_stitch": n_snps_total,
                "fallback_reason_counts": {",".join(fallback_reasons): n_snps_total},
                "decision_source_counts": {},
            }
        else:
            status = "ok"
            per_variant_summary = {
                "n_snps": n_snps_total,
                "n_snps_calibration_used": n_snps_total,
                "n_snps_fallback_to_stitch": 0,
                "fallback_reason_counts": {},
                "decision_source_counts": {"global": n_snps_total},
            }
    estimator_out = model.named_steps.get("model") if hasattr(model, "named_steps") else model
    feature_rows: list[dict[str, float | str]] = []
    intercept: float | None = None
    if hasattr(estimator_out, "coef_"):
        coef_arr = np.asarray(estimator_out.coef_, dtype=np.float32).reshape(-1)
        intercept_arr = np.asarray(getattr(estimator_out, "intercept_", []), dtype=np.float32).reshape(-1)
        intercept = float(intercept_arr[0]) if intercept_arr.size else None
        feature_rows = [
            {"feature": name, "coefficient": float(value)}
            for name, value in zip(feature_names, coef_arr.tolist(), strict=False)
        ]
    elif hasattr(estimator_out, "feature_importances_"):
        imp_arr = np.asarray(estimator_out.feature_importances_, dtype=np.float32).reshape(-1)
        feature_rows = [
            {"feature": name, "importance": float(value)}
            for name, value in zip(feature_names, imp_arr.tolist(), strict=False)
        ]
    meta = {
        "status": status,
        "mode": "standard_callability",
        "decision_mode": mode_name,
        "model": model_name,
        "model_library": "sklearn_pipeline",
        "n_train_rows": int(np.sum(flat_train)),
        "n_fit_candidate_rows": int(y_fit_all.shape[0]),
        "n_fit_rows": int(x_fit.shape[0]),
        "n_threshold_rows": int(np.sum(flat_threshold)),
        "threshold_validation": validation_status,
        "correct_rate": float(np.mean(y_fit_all)),
        "threshold": float(threshold),
        "objective": float(score),
        "train_call_rate": float(call_rate),
        "min_call_rate": float(min_call_rate),
        "effective_min_call_rate": float(effective_min_call_rate),
        "effective_max_call_rate": (None if effective_max_call_rate is None else float(effective_max_call_rate)),
        "stitch_gp_threshold": float(stitch_gp_threshold),
        "bound_to_stitch": bool(bound_to_stitch),
        "min_call_rate_delta": float(min_call_rate_delta),
        "max_call_rate_delta": float(max_call_rate_delta),
        "validation_site_fraction": float(validation_site_fraction),
        "min_objective_improvement": float(min_objective_improvement),
        "heldout_stitch_objective": stitch_stats_threshold,
        "heldout_calibrated_objective": calibrated_stats_threshold,
        "stitch_call_rate_full": float(parity_call_rate),
        "stitch_call_rate_labeled": float(parity_call_rate_labeled),
        "calibrated_call_rate_full": float(np.mean(call_prob >= float(threshold))) if call_prob.size else 0.0,
        "population_stitch_no_call_stats": raw_pop_stats,
        "population_calibrated_stats": calibrated_pop_stats,
        "population_mean_maf_shift": float(maf_shift),
        "population_mean_het_shift": float(het_shift),
        "max_hardcall_maf_shift": float(max_hardcall_maf_shift),
        "max_hardcall_het_shift": float(max_hardcall_het_shift),
        "global_fallback_reasons": fallback_reasons,
        "fallback_reasons": (
            []
            if status != "fallback_to_stitch_no_call"
            else list(fallback_reasons or ["no_safe_per_snp_calibration"])
        ),
        "per_snp_decision": per_variant_summary,
        "n_snps_calibration_used": int(per_variant_summary.get("n_snps_calibration_used", 0)),
        "n_snps_fallback_to_stitch": int(per_variant_summary.get("n_snps_fallback_to_stitch", n_snps_total)),
        "fallback_reason_counts": per_variant_summary.get("fallback_reason_counts", {}),
        "decision_source_counts": per_variant_summary.get("decision_source_counts", {}),
        "call_rate_weight": float(call_rate_weight),
        "feature_names": feature_names,
        "feature_coefficients": feature_rows,
        "intercept": intercept,
    }
    if per_variant_gate_probability is not None and per_variant_threshold is not None and per_variant_fallback is not None:
        meta["gate_probability_mode"] = "per_snp_hierarchical"
        meta["threshold_by_variant"] = per_variant_threshold.astype(np.float32, copy=False)
        meta["fallback_to_stitch_by_variant"] = per_variant_fallback.astype(bool, copy=False)
        if per_variant_raw_threshold is not None:
            meta["calibrated_threshold_by_variant"] = per_variant_raw_threshold.astype(np.float32, copy=False)
        if variant_decisions is not None:
            meta["_variant_decisions"] = variant_decisions
        return per_variant_gate_probability.astype(np.float32, copy=False), meta
    return call_prob, meta


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
    hwe_features = compute_hwe_features_from_posterior(gp)
    hwe_cols = {
        name: np.broadcast_to(
            np.nan_to_num(values.astype(np.float32, copy=False), nan=0.0, posinf=300.0, neginf=0.0)[None, :],
            (n_samples, n_positions),
        )
        for name, values in hwe_features.items()
    }
    site_total_count = np.sum(total, axis=0).astype(np.float32, copy=False)
    site_support_samples = np.sum(total > 0.0, axis=0).astype(np.float32, copy=False)
    site_support_rate = site_support_samples / float(max(n_samples, 1))
    site_mean_depth = site_total_count / float(max(n_samples, 1))
    site_cols = {
        "site_total_count": np.broadcast_to(site_total_count[None, :], (n_samples, n_positions)),
        "site_support_samples": np.broadcast_to(site_support_samples[None, :], (n_samples, n_positions)),
        "site_support_rate": np.broadcast_to(site_support_rate[None, :], (n_samples, n_positions)),
        "site_mean_depth": np.broadcast_to(site_mean_depth[None, :], (n_samples, n_positions)),
    }

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
        *(arr.reshape(-1) for arr in hwe_cols.values()),
        *(arr.reshape(-1) for arr in site_cols.values()),
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
        *(f"variant_{name}" for name in hwe_cols.keys()),
        *site_cols.keys(),
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
    use_fixed_stage0_calibration: bool = False,
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

    if raw_posterior is None:
        gp_stage0 = dosage_to_genotype_posterior(ds, depth=depth, temperature=0.35, min_prob=float(min_prob), ploidy=2)
        stage0_meta = {"status": "dosage_fallback", "temperature": 0.35}
    elif use_fixed_stage0_calibration:
        gp_stage0 = calibrate_genotype_posterior(
            raw_posterior,
            dosage=ds,
            depth=depth,
            temperature=0.35,
            blend=0.35,
            min_prob=float(min_prob),
        )
        stage0_meta = {"status": "fixed_temperature_blend", "temperature": 0.35, "blend": 0.35}
    else:
        gp_stage0 = raw_posterior.astype(np.float32, copy=False)
        stage0_meta = {"status": "raw_hmm_gp", "temperature_blend_applied": False}
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
    hwe_features = compute_hwe_features_from_posterior(gp_stage1)
    depth_arr = np.zeros_like(ds, dtype=np.float32) if depth is None else depth.astype(np.float32, copy=False)
    if support_mask is not None:
        support_arr = support_mask.astype(np.float32, copy=False)
    elif ref_count is not None or alt_count is not None or other_count is not None:
        ref_arr = np.zeros_like(ds, dtype=np.float32) if ref_count is None else ref_count.astype(np.float32, copy=False)
        alt_arr = np.zeros_like(ds, dtype=np.float32) if alt_count is None else alt_count.astype(np.float32, copy=False)
        oth_arr = np.zeros_like(ds, dtype=np.float32) if other_count is None else other_count.astype(np.float32, copy=False)
        support_arr = ((ref_arr + alt_arr + oth_arr) > 0.0).astype(np.float32, copy=False)
    else:
        support_arr = (depth_arr > 0.0).astype(np.float32, copy=False)
    support_rate = np.nanmean(support_arr, axis=0).astype(np.float32, copy=False)
    missingness = (1.0 - support_rate).astype(np.float32, copy=False)
    mean_depth = np.nanmean(depth_arr, axis=0).astype(np.float32, copy=False)
    variant_strata, strata_meta = compute_variant_strata(
        variant_maf,
        variant_info,
        support_rate=support_rate,
        missingness=missingness,
        hwe_deviation=hwe_features["hwe_deviation"],
        mean_depth=mean_depth,
    )
    broad_variant_strata, broad_strata_meta = compute_variant_strata(variant_maf, variant_info)

    features, feature_names = build_readaware_calibration_feature_matrix(
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
    pos_take = _balanced_row_indices(y_train, max_rows=int(max_train_rows), seed=int(seed))
    x_train = x_train[pos_take]
    y_train = y_train[pos_take]
    pos_train = pos_train[pos_take]

    try:
        global_model, strata_models, fallback_strata_models, stratified_meta = train_stratified_lightgbm_multiclass_calibrator(
            features=x_train,
            target=y_train,
            position_index=pos_train,
            variant_strata=variant_strata,
            fallback_variant_strata=broad_variant_strata,
            seed=int(seed),
            max_rows=int(max_train_rows),
            class_weight_mode=class_weight_mode,
        )
    except Exception as exc:
        return (
            gp_stage1.astype(np.float32, copy=False),
            None,
            {
                "status": "fallback_to_raw_gp",
                "reason": "lightgbm_training_failed",
                "error": str(exc),
                "block_context": block_meta,
                "stage0": stage0_meta,
                "strata": strata_meta,
                "broad_strata": broad_strata_meta,
            },
        )
    posterior_feature_importance = lightgbm_feature_importance(global_model, feature_names)

    pred_flat = predict_stratified_lightgbm_posterior(
        features=features,
        position_index=pos_flat,
        variant_strata=variant_strata,
        global_model=global_model,
        strata_models=strata_models,
        fallback_variant_strata=broad_variant_strata,
        fallback_strata_models=fallback_strata_models,
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
    call_features, call_feature_names = build_hardcall_quality_feature_matrix(
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
            call_feature_importance = lightgbm_feature_importance(call_model, call_feature_names)
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
                "feature_importance": call_feature_importance,
            }

    summary = {
        "status": "ok",
        "stage0": stage0_meta,
        "block_context": block_meta,
        "strata": strata_meta,
        "broad_strata": broad_strata_meta,
        "stratified_fit": stratified_meta,
        "posterior_feature_names": feature_names,
        "posterior_feature_importance": posterior_feature_importance,
        "isotonic_enabled": bool(apply_isotonic),
        "isotonic_models": int(sum(1 for m in isotonic_models if m is not None)),
        "call_correctness": call_meta,
    }
    return gp_out.astype(np.float32, copy=False), call_prob, summary
