from __future__ import annotations

import numpy as np

from stitchcont.config import HMMConfig
from stitchcont.fast_hmm import ordered_log_emission_to_unordered
from stitchcont.hmm import JAXStitchHMM, _diploid_pair_to_haplotype_posterior
from stitchcont.cpu_kernels import numba_unordered_diploid_reduce


def test_numba_unordered_reduce_matches_ordered_diploid_parity():
    rng = np.random.default_rng(123)
    k = 4
    n_samples = 3
    n_positions = 25
    founder = (rng.random((k, n_positions)) > 0.5).astype(np.float32)
    ref = rng.poisson(0.12, size=(n_samples, n_positions)).astype(np.float32)
    alt = rng.poisson(0.12, size=(n_samples, n_positions)).astype(np.float32)
    hmm = JAXStitchHMM(HMMConfig(n_founders=k, backend="numpy", exact_hmm_chunk_size=10))
    switch = np.full((n_samples, n_positions), 0.01, dtype=np.float32)
    switch[:, 0] = 1e-10
    log_emission, _, founder_alt = hmm.emissions(founder, ref, alt, None, None, None, None)
    gamma = hmm._forward_backward_numpy_diploid_chunked(log_emission, switch, k)
    hap = _diploid_pair_to_haplotype_posterior(gamma)
    dosage_ordered = 2.0 * np.einsum("spk,kp->sp", hap, founder_alt, optimize=True)

    log_u = ordered_log_emission_to_unordered(log_emission)
    reduced = numba_unordered_diploid_reduce(log_u, switch, founder_alt)
    assert np.max(np.abs(reduced.dosage - dosage_ordered)) < 1e-5
    assert reduced.genotype_posterior is not None
    assert reduced.genotype_posterior.shape == (n_samples, n_positions, 3)


def test_native_low_rank_reduce_matches_numpy_low_rank_when_available():
    try:
        from stitchcont.native_cpu import native_diploid_low_rank_reduce, native_backend_status
    except Exception:
        return
    if not native_backend_status().get("pybind11_low_rank_reduce_available"):
        return
    rng = np.random.default_rng(456)
    k = 4
    rank = 2
    n_samples = 2
    n_positions = 11
    log_emission = rng.normal(0, 0.25, size=(n_samples, n_positions, k, k)).astype(np.float32)
    switch = rng.uniform(1e-4, 0.04, size=(n_samples, n_positions)).astype(np.float32)
    switch[:, 0] = 1e-8
    founder_alt = rng.random((k, n_positions)).astype(np.float32)
    offdiag = np.full((k, k), 1.0 / (k - 1), dtype=np.float32)
    np.fill_diagonal(offdiag, 0.0)
    hmm = JAXStitchHMM(HMMConfig(n_founders=k, backend="numpy", transition_factor_rank=rank, exact_hmm_chunk_size=5))
    left, right, _ = hmm._low_rank_transition_components(offdiag, rank)
    gamma = hmm._forward_backward_numpy_diploid_low_rank_chunked(log_emission, switch, offdiag, k)
    hap = _diploid_pair_to_haplotype_posterior(gamma)
    dosage_np = 2.0 * np.einsum("spk,kp->sp", hap, founder_alt, optimize=True)
    gp0 = np.einsum("spij,ip,jp->sp", gamma, 1.0 - founder_alt, 1.0 - founder_alt, optimize=True)
    gp2 = np.einsum("spij,ip,jp->sp", gamma, founder_alt, founder_alt, optimize=True)
    gp_np = np.stack([gp0, 1.0 - gp0 - gp2, gp2], axis=2).astype(np.float32)
    native = native_diploid_low_rank_reduce(log_emission, switch, founder_alt, left, right)
    np.testing.assert_allclose(native["dosage"], dosage_np, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(native["genotype_posterior"], gp_np, rtol=2e-4, atol=2e-4)
