from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Callable

import numpy as np


@dataclass(slots=True)
class UnorderedDiploidOutput:
    dosage: np.ndarray
    genotype_posterior: np.ndarray | None
    haplotype_posterior: np.ndarray | None
    seconds: float


@lru_cache(maxsize=32)
def unordered_diploid_pairs(k: int) -> tuple[tuple[int, int], ...]:
    k = int(k)
    return tuple((i, j) for i in range(k) for j in range(i, k))


@lru_cache(maxsize=32)
def unordered_diploid_copy_matrix(k: int) -> np.ndarray:
    pairs = unordered_diploid_pairs(k)
    copy = np.zeros((len(pairs), int(k)), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        copy[idx, i] += 1.0
        copy[idx, j] += 1.0
    return copy


@lru_cache(maxsize=128)
def unordered_transition_from_haploid(k: int, switch_rounded: float, offdiag_key: tuple[float, ...] | None = None) -> np.ndarray:
    """Aggregate ordered diploid transitions into unordered diploid states.

    The returned matrix maps unordered-source probability rows to unordered-dest
    probability rows.  Within a heterozygous unordered source state, the two
    ordered phase states are averaged; destination ordered states are summed.
    """
    k = int(k)
    sw = float(switch_rounded)
    if offdiag_key is None:
        off = sw / float(max(k - 1, 1))
        hap = np.full((k, k), off, dtype=np.float32)
        np.fill_diagonal(hap, 1.0 - sw)
    else:
        offdiag = np.asarray(offdiag_key, dtype=np.float32).reshape(k, k)
        hap = offdiag * np.float32(sw)
        np.fill_diagonal(hap, np.float32(1.0 - sw))
    pairs = unordered_diploid_pairs(k)
    n_states = len(pairs)
    out = np.zeros((n_states, n_states), dtype=np.float32)
    for src_idx, (a, b) in enumerate(pairs):
        ordered_src = ((a, b),) if a == b else ((a, b), (b, a))
        src_weight = 1.0 / float(len(ordered_src))
        for x, y in ordered_src:
            # Independent chromosome transitions from ordered (x,y) to (u,v).
            ordered = hap[x, :, None] * hap[y, None, :]
            for dst_idx, (c, d) in enumerate(pairs):
                if c == d:
                    out[src_idx, dst_idx] += src_weight * ordered[c, d]
                else:
                    out[src_idx, dst_idx] += src_weight * (ordered[c, d] + ordered[d, c])
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out.astype(np.float32, copy=False)


def ordered_log_emission_to_unordered(log_emission: np.ndarray) -> np.ndarray:
    n_samples, n_positions, k, _ = log_emission.shape
    pairs = unordered_diploid_pairs(k)
    out = np.empty((n_samples, n_positions, len(pairs)), dtype=np.float32)
    for idx, (i, j) in enumerate(pairs):
        if i == j:
            out[:, :, idx] = log_emission[:, :, i, j]
        else:
            # Numerical symmetry guard for fragment and count emissions.
            a = log_emission[:, :, i, j]
            b = log_emission[:, :, j, i]
            m = np.maximum(a, b)
            out[:, :, idx] = m + np.log(0.5 * np.exp(a - m) + 0.5 * np.exp(b - m))
    return out


def unordered_diploid_forward_backward(
    log_emission_u: np.ndarray,
    switch: np.ndarray,
    *,
    k: int,
    offdiag_matrix: np.ndarray | None = None,
    switch_round_decimals: int = 8,
) -> np.ndarray:
    n_samples, n_positions, n_states = log_emission_u.shape
    pairs = unordered_diploid_pairs(k)
    if n_states != len(pairs):
        raise ValueError(f"Expected {len(pairs)} unordered states for K={k}, got {n_states}")
    emission = np.exp(log_emission_u - np.max(log_emission_u, axis=2, keepdims=True)).astype(np.float32)
    gamma = np.zeros_like(emission, dtype=np.float32)
    multiplicity = np.asarray([1.0 if i == j else 2.0 for i, j in pairs], dtype=np.float32)
    prior = multiplicity / float(k * k)
    offdiag_key = None if offdiag_matrix is None else tuple(np.asarray(offdiag_matrix, dtype=np.float32).reshape(-1).tolist())
    for sample_idx in range(n_samples):
        alpha = np.empty((n_positions, n_states), dtype=np.float32)
        beta = np.empty((n_positions, n_states), dtype=np.float32)
        alpha0 = prior * emission[sample_idx, 0]
        alpha0 /= np.clip(alpha0.sum(), 1e-12, None)
        alpha[0] = alpha0
        for pos_idx in range(1, n_positions):
            sw = round(float(switch[sample_idx, pos_idx]), int(switch_round_decimals))
            trans = unordered_transition_from_haploid(int(k), sw, offdiag_key)
            a = (alpha[pos_idx - 1] @ trans) * emission[sample_idx, pos_idx]
            alpha[pos_idx] = a / np.clip(a.sum(), 1e-12, None)
        beta[-1] = np.full(n_states, 1.0 / float(n_states), dtype=np.float32)
        for pos_idx in range(n_positions - 2, -1, -1):
            sw = round(float(switch[sample_idx, pos_idx + 1]), int(switch_round_decimals))
            trans = unordered_transition_from_haploid(int(k), sw, offdiag_key)
            b = trans @ (beta[pos_idx + 1] * emission[sample_idx, pos_idx + 1])
            beta[pos_idx] = b / np.clip(b.sum(), 1e-12, None)
        g = alpha * beta
        g /= np.clip(g.sum(axis=1, keepdims=True), 1e-12, None)
        gamma[sample_idx] = g
    return gamma


def reduce_unordered_diploid(
    gamma_u: np.ndarray,
    founder_alt: np.ndarray,
    *,
    return_genotype_posterior: bool = True,
    return_haplotype_posterior: bool = False,
) -> UnorderedDiploidOutput:
    t0 = time.perf_counter()
    k, n_positions = founder_alt.shape
    copy = unordered_diploid_copy_matrix(k)
    hap = 0.5 * np.einsum("spm,mk->spk", gamma_u, copy, optimize=True).astype(np.float32)
    dosage = 2.0 * np.einsum("spk,kp->sp", hap, founder_alt, optimize=True).astype(np.float32)
    gp = None
    if return_genotype_posterior:
        pairs = unordered_diploid_pairs(k)
        gp0 = np.zeros(gamma_u.shape[:2], dtype=np.float32)
        gp2 = np.zeros_like(gp0)
        founder_alt_t = founder_alt.T.astype(np.float32, copy=False)
        one_minus = 1.0 - founder_alt_t
        for state_idx, (i, j) in enumerate(pairs):
            w = gamma_u[:, :, state_idx]
            gp0 += w * one_minus[:, i][None, :] * one_minus[:, j][None, :]
            gp2 += w * founder_alt_t[:, i][None, :] * founder_alt_t[:, j][None, :]
        gp1 = np.clip(1.0 - gp0 - gp2, 0.0, 1.0)
        gp = np.stack([gp0, gp1, gp2], axis=2).astype(np.float32, copy=False)
        gp /= np.clip(gp.sum(axis=2, keepdims=True), 1e-8, None)
    return UnorderedDiploidOutput(
        dosage=dosage,
        genotype_posterior=gp,
        haplotype_posterior=(hap if return_haplotype_posterior else None),
        seconds=float(time.perf_counter() - t0),
    )
