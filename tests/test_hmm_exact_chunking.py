from types import SimpleNamespace

import numpy as np
import pytest

from stitchcont.config import HMMConfig
from stitchcont.hmm import JAXStitchHMM, _diploid_pair_to_haplotype_posterior


def test_diploid_exact_chunking_matches_full_forward_backward():
    rng = np.random.default_rng(123)
    n_samples, n_pos, k = 3, 17, 4
    log_emission = rng.normal(size=(n_samples, n_pos, k, k)).astype(np.float32)
    switch = rng.uniform(1e-5, 0.08, size=(n_samples, n_pos)).astype(np.float32)
    switch[:, 0] = 1e-10
    hmm_full = JAXStitchHMM(HMMConfig(n_founders=k, exact_hmm_chunk_size=0, backend="numpy"))
    hmm_chunk = JAXStitchHMM(HMMConfig(n_founders=k, exact_hmm_chunk_size=5, backend="numpy"))
    full = hmm_full._forward_backward_numpy_diploid(log_emission, switch, k)
    chunk = hmm_chunk._forward_backward_numpy_diploid_chunked(log_emission, switch, k)
    np.testing.assert_allclose(chunk, full, rtol=2e-5, atol=2e-5)


def test_haploid_exact_chunking_matches_full_forward_backward():
    rng = np.random.default_rng(321)
    n_samples, n_pos, k = 4, 19, 5
    log_emission = rng.normal(size=(n_samples, n_pos, k)).astype(np.float32)
    switch = rng.uniform(1e-5, 0.08, size=(n_samples, n_pos)).astype(np.float32)
    switch[:, 0] = 1e-10
    hmm_full = JAXStitchHMM(HMMConfig(n_founders=k, ploidy_mode="pseudo_haploid", ploidy=1, exact_hmm_chunk_size=0, backend="numpy"))
    hmm_chunk = JAXStitchHMM(HMMConfig(n_founders=k, ploidy_mode="pseudo_haploid", ploidy=1, exact_hmm_chunk_size=4, backend="numpy"))
    full = hmm_full._forward_backward_numpy_haploid(log_emission, switch, k)
    chunk = hmm_chunk._forward_backward_numpy_haploid_chunked(log_emission, switch, k)
    np.testing.assert_allclose(chunk, full, rtol=2e-5, atol=2e-5)


def test_fragment_single_snp_replace_matches_dense_count_emission():
    founder_alt = np.asarray([[0.0], [1.0]], dtype=np.float32)
    ref_count = np.zeros((1, 1), dtype=np.int16)
    alt_count = np.ones((1, 1), dtype=np.int16)
    zero_count = np.zeros((1, 1), dtype=np.int16)
    hmm = JAXStitchHMM(
        HMMConfig(
            n_founders=2,
            backend="numpy",
            sequencing_error_rate=0.01,
            use_quality_weights=False,
            use_fragment_likelihood=True,
            fragment_likelihood_mode="replace",
            fragment_rescale_read_likelihood=False,
        )
    )
    dense, _, founder = hmm.emissions(founder_alt, ref_count, alt_count, zero_count, None, None, None)
    base, _, founder = hmm.emissions(founder_alt, zero_count, zero_count, zero_count, None, None, None)
    frag = hmm._apply_fragment_likelihoods(
        log_emission=base.copy(),
        founder_alt=founder,
        fragment_sample_offsets=np.asarray([0, 1], dtype=np.int64),
        fragment_center_idx=np.asarray([0], dtype=np.int64),
        fragment_obs_offsets=np.asarray([0, 1], dtype=np.int64),
        fragment_obs_pos_idx=np.asarray([0], dtype=np.int64),
        fragment_obs_code=np.asarray([1], dtype=np.int8),
        fragment_obs_qual=np.asarray([20], dtype=np.uint8),
    )
    np.testing.assert_allclose(frag, dense, rtol=2e-5, atol=2e-5)


def test_k8_immutable_founders_are_not_updated_and_dosage_is_finite():
    rng = np.random.default_rng(42)
    k, n_pos, n_samples = 8, 12, 5
    founder_alt = rng.integers(0, 2, size=(k, n_pos)).astype(np.float32)
    panel = SimpleNamespace(
        alt_prob=founder_alt.copy(),
        immutable_mask=np.ones(k, dtype=bool),
        positions=np.arange(1, n_pos + 1, dtype=np.int64) * 1000,
        genetic_cm=None,
        n_founders=k,
        n_positions=n_pos,
    )
    # Generate simple read counts from known founder pairs.
    pairs = rng.integers(0, k, size=(n_samples, 2))
    true_dosage = founder_alt[pairs[:, 0]] + founder_alt[pairs[:, 1]]
    depth = 8
    alt_count = np.rint(true_dosage / 2.0 * depth).astype(np.int16)
    ref_count = (depth - alt_count).astype(np.int16)
    zero = np.zeros_like(ref_count, dtype=np.int16)
    hmm = JAXStitchHMM(
        HMMConfig(
            n_founders=k,
            backend="numpy",
            em_iterations=1,
            exact_hmm_chunk_size=5,
            use_fragment_likelihood=False,
            recombination_rate_cM_per_Mb=0.01,
        )
    )
    artifacts = hmm.run(
        founder_panel=panel,
        ref_count=ref_count,
        alt_count=alt_count,
        other_count=zero,
        ref_weight=None,
        alt_weight=None,
        other_weight=None,
        generations=np.ones(n_samples, dtype=np.float32),
        return_full_transition=False,
        return_haplotype_posterior=False,
        return_genotype_posterior=True,
        fragment_sample_offsets=None,
        fragment_center_idx=None,
        fragment_obs_offsets=None,
        fragment_obs_pos_idx=None,
        fragment_obs_code=None,
        fragment_obs_qual=None,
    )
    np.testing.assert_allclose(artifacts.founder_alt_prob, np.clip(founder_alt, 1e-4, 1.0 - 1e-4), rtol=1e-6, atol=1e-6)
    assert np.isfinite(artifacts.dosage).all()
    assert artifacts.genotype_posterior is not None


def test_jax_diploid_matches_numpy_when_jax_available():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    rng = np.random.default_rng(7)
    n_samples, n_pos, k = 2, 8, 3
    log_emission = rng.normal(size=(n_samples, n_pos, k, k)).astype(np.float32)
    switch = rng.uniform(1e-5, 0.05, size=(n_samples, n_pos)).astype(np.float32)
    switch[:, 0] = 1e-10
    hmm = JAXStitchHMM(HMMConfig(n_founders=k, backend="numpy", jax_aot_compile=False))
    expected = hmm._forward_backward_numpy_diploid(log_emission, switch, k)
    fn = hmm._get_jax_diploid_callable(n_samples, n_pos, k)
    observed = np.asarray(fn(jnp.asarray(log_emission), jnp.asarray(switch)), dtype=np.float32)
    np.testing.assert_allclose(observed, expected, rtol=2e-5, atol=2e-5)
