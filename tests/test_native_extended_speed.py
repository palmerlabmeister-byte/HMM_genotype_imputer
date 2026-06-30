from __future__ import annotations

import numpy as np
import pytest

from stitchcont.config import HMMConfig
from stitchcont.hmm import JAXStitchHMM
from stitchcont.native_cpu import (
    native_backend_status,
    native_diploid_low_rank_counts_reduce,
    native_diploid_low_rank_reduce,
    native_diploid_sparse_topk_counts_reduce,
    native_diploid_sparse_topk_reduce,
    native_k8_counts_fragments_reduce,
    native_k8_counts_reduce,
)


def _require_native_extended() -> None:
    status = native_backend_status()
    needed = [
        "pybind11_low_rank_counts_fused_available",
        "pybind11_sparse_topk_reduce_available",
        "pybind11_sparse_topk_counts_fused_available",
        "pybind11_k8_counts_fragments_fused_available",
    ]
    if not all(status.get(k, False) for k in needed):
        pytest.skip(f"native extended pybind11 backend unavailable: {status}")


def test_native_low_rank_counts_matches_precomputed_log_emission() -> None:
    _require_native_extended()
    rng = np.random.default_rng(123)
    n_samples, n_positions, k, rank = 4, 11, 5, 2
    ref = rng.poisson(0.15, (n_samples, n_positions)).astype(np.float32)
    alt = rng.poisson(0.15, (n_samples, n_positions)).astype(np.float32)
    other = np.zeros_like(ref)
    founder_alt = rng.random((k, n_positions)).astype(np.float32)
    switch = np.full((n_samples, n_positions), 0.025, dtype=np.float32)
    cfg = HMMConfig(n_founders=k)
    hmm = JAXStitchHMM(cfg)
    log_emission, _, _ = hmm.emissions(founder_alt, ref, alt, other, None, None, None)
    offdiag = np.full((k, k), 1.0 / (k - 1), dtype=np.float32)
    np.fill_diagonal(offdiag, 0.0)
    left, right, _ = hmm._low_rank_transition_components(offdiag, rank)
    precomputed = native_diploid_low_rank_reduce(log_emission, switch, founder_alt, left, right)
    fused = native_diploid_low_rank_counts_reduce(
        ref,
        alt,
        other,
        switch,
        founder_alt,
        left,
        right,
        sequencing_error_rate=float(cfg.sequencing_error_rate),
        min_emission_prob=float(cfg.min_emission_prob),
    )
    np.testing.assert_allclose(fused["dosage"], precomputed["dosage"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fused["genotype_posterior"], precomputed["genotype_posterior"], rtol=1e-5, atol=1e-5)


def test_native_sparse_topk_counts_matches_precomputed_log_emission() -> None:
    _require_native_extended()
    rng = np.random.default_rng(456)
    n_samples, n_positions, k = 3, 9, 6
    ref = rng.poisson(0.1, (n_samples, n_positions)).astype(np.float32)
    alt = rng.poisson(0.1, (n_samples, n_positions)).astype(np.float32)
    other = np.zeros_like(ref)
    founder_alt = rng.random((k, n_positions)).astype(np.float32)
    switch = np.full((n_samples, n_positions), 0.03, dtype=np.float32)
    cfg = HMMConfig(n_founders=k)
    hmm = JAXStitchHMM(cfg)
    log_emission, _, _ = hmm.emissions(founder_alt, ref, alt, other, None, None, None)
    offdiag = rng.random((k, k)).astype(np.float32)
    np.fill_diagonal(offdiag, 0.0)
    offdiag /= offdiag.sum(axis=1, keepdims=True)
    precomputed = native_diploid_sparse_topk_reduce(log_emission, switch, founder_alt, offdiag, top_k=3)
    fused = native_diploid_sparse_topk_counts_reduce(
        ref,
        alt,
        other,
        switch,
        founder_alt,
        offdiag,
        top_k=3,
        sequencing_error_rate=float(cfg.sequencing_error_rate),
        min_emission_prob=float(cfg.min_emission_prob),
    )
    np.testing.assert_allclose(fused["dosage"], precomputed["dosage"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(fused["genotype_posterior"], precomputed["genotype_posterior"], rtol=1e-5, atol=1e-5)


def test_native_k8_counts_fragments_empty_matches_counts_only() -> None:
    _require_native_extended()
    rng = np.random.default_rng(789)
    n_samples, n_positions, k = 5, 13, 8
    ref = rng.poisson(0.1, (n_samples, n_positions)).astype(np.float32)
    alt = rng.poisson(0.1, (n_samples, n_positions)).astype(np.float32)
    other = np.zeros_like(ref)
    founder_alt = rng.random((k, n_positions)).astype(np.float32)
    switch = np.full((n_samples, n_positions), 0.02, dtype=np.float32)
    count_only = native_k8_counts_reduce(
        ref,
        alt,
        other,
        switch,
        founder_alt,
        sequencing_error_rate=0.01,
        min_emission_prob=1e-5,
    )
    fused = native_k8_counts_fragments_reduce(
        ref,
        alt,
        other,
        switch,
        founder_alt,
        fragment_sample_offsets=np.zeros(n_samples + 1, dtype=np.int64),
        fragment_center_idx=np.zeros(0, dtype=np.int64),
        fragment_obs_offsets=np.zeros(1, dtype=np.int64),
        fragment_obs_pos_idx=np.zeros(0, dtype=np.int64),
        fragment_obs_code=np.zeros(0, dtype=np.int8),
        fragment_obs_qual=np.zeros(0, dtype=np.float32),
        sequencing_error_rate=0.01,
        min_emission_prob=1e-5,
        mode="replace",
        rescale=True,
        max_emission_matrix_difference=1e10,
    )
    np.testing.assert_allclose(fused["dosage"], count_only["dosage"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(fused["genotype_posterior"], count_only["genotype_posterior"], rtol=1e-6, atol=1e-6)
