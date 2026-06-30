from __future__ import annotations

"""Compiled fragment aggregation helpers.

These functions replace repeated np.add.at over fragment/read records with a
single linear pass. They aggregate already scaled per-fragment haplotype read
probabilities into per sample/position ordered diploid log-emission matrices.
"""

import numpy as np

try:  # pragma: no cover - depends on installed optional dependency.
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return deco


@njit(cache=True)
def _aggregate_diploid_replace(sample_for_fragment: np.ndarray, centers: np.ndarray, probs: np.ndarray, n_samples: int, n_positions: int, min_prob: float) -> tuple[np.ndarray, np.ndarray]:
    k = probs.shape[1]
    acc = np.zeros((n_samples, n_positions, k, k), dtype=np.float32)
    touched = np.zeros((n_samples, n_positions), dtype=np.uint8)
    for f in range(centers.shape[0]):
        s = sample_for_fragment[f]
        p = centers[f]
        if s < 0 or s >= n_samples or p < 0 or p >= n_positions:
            continue
        touched[s, p] = 1
        for i in range(k):
            pi = probs[f, i]
            for j in range(k):
                v = 0.5 * (pi + probs[f, j])
                if v < min_prob:
                    v = min_prob
                acc[s, p, i, j] += np.log(v)
    return acc, touched


@njit(cache=True)
def _aggregate_haploid_replace(sample_for_fragment: np.ndarray, centers: np.ndarray, probs: np.ndarray, n_samples: int, n_positions: int, min_prob: float) -> tuple[np.ndarray, np.ndarray]:
    k = probs.shape[1]
    acc = np.zeros((n_samples, n_positions, k), dtype=np.float32)
    touched = np.zeros((n_samples, n_positions), dtype=np.uint8)
    for f in range(centers.shape[0]):
        s = sample_for_fragment[f]
        p = centers[f]
        if s < 0 or s >= n_samples or p < 0 or p >= n_positions:
            continue
        touched[s, p] = 1
        for i in range(k):
            v = probs[f, i]
            if v < min_prob:
                v = min_prob
            acc[s, p, i] += np.log(v)
    return acc, touched


@njit(cache=True)
def _rescale_touched_diploid(log_emission: np.ndarray, touched: np.ndarray, max_diff: float) -> None:
    n_samples = log_emission.shape[0]
    n_positions = log_emission.shape[1]
    k = log_emission.shape[2]
    min_state_log = -np.log(max_diff)
    for s in range(n_samples):
        for p in range(n_positions):
            if touched[s, p] == 0:
                continue
            maxv = log_emission[s, p, 0, 0]
            for i in range(k):
                for j in range(k):
                    if log_emission[s, p, i, j] > maxv:
                        maxv = log_emission[s, p, i, j]
            for i in range(k):
                for j in range(k):
                    v = log_emission[s, p, i, j] - maxv
                    if v < min_state_log:
                        v = min_state_log
                    log_emission[s, p, i, j] = v


@njit(cache=True)
def _rescale_touched_haploid(log_emission: np.ndarray, touched: np.ndarray, max_diff: float) -> None:
    n_samples = log_emission.shape[0]
    n_positions = log_emission.shape[1]
    k = log_emission.shape[2]
    min_state_log = -np.log(max_diff)
    for s in range(n_samples):
        for p in range(n_positions):
            if touched[s, p] == 0:
                continue
            maxv = log_emission[s, p, 0]
            for i in range(k):
                if log_emission[s, p, i] > maxv:
                    maxv = log_emission[s, p, i]
            for i in range(k):
                v = log_emission[s, p, i] - maxv
                if v < min_state_log:
                    v = min_state_log
                log_emission[s, p, i] = v


def apply_compiled_fragment_replace(
    log_emission: np.ndarray,
    *,
    sample_for_fragment: np.ndarray,
    centers: np.ndarray,
    read_prob_hap: np.ndarray,
    ploidy_mode: str,
    mode: str,
    min_prob: float,
    rescale: bool,
    max_emission_matrix_difference: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Apply compiled fragment aggregation when the shape/mode is supported.

    Returns (log_emission, touched_mask) or None when caller should use the
    generic path.
    """
    if mode not in {"replace", "augment"}:
        return None
    n_samples = int(log_emission.shape[0])
    n_positions = int(log_emission.shape[1])
    centers = np.asarray(centers, dtype=np.int64, order="C")
    sample_for_fragment = np.asarray(sample_for_fragment, dtype=np.int64, order="C")
    read_prob_hap = np.asarray(read_prob_hap, dtype=np.float32, order="C")
    if centers.size == 0:
        return log_emission, np.zeros(log_emission.shape[:2], dtype=bool)
    if ploidy_mode == "pseudo_haploid":
        acc, touched_u8 = _aggregate_haploid_replace(sample_for_fragment, centers, read_prob_hap, n_samples, n_positions, float(min_prob))
        touched = touched_u8.astype(bool)
        if mode == "replace":
            log_emission[touched] = acc[touched]
        else:
            log_emission += acc
        if rescale and np.any(touched):
            _rescale_touched_haploid(log_emission, touched_u8, float(max(max_emission_matrix_difference, 1.000001)))
        return log_emission, touched
    if log_emission.ndim != 4:
        return None
    acc, touched_u8 = _aggregate_diploid_replace(sample_for_fragment, centers, read_prob_hap, n_samples, n_positions, float(min_prob))
    touched = touched_u8.astype(bool)
    if mode == "replace":
        log_emission[touched] = acc[touched]
    else:
        log_emission += acc
    if rescale and np.any(touched):
        _rescale_touched_diploid(log_emission, touched_u8, float(max(max_emission_matrix_difference, 1.000001)))
    return log_emission, touched
