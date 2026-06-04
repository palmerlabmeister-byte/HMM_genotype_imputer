"""End-to-end correctness guard for the on-device EM accumulation change (P1.1)
and the resident-block / GPU-aware batching changes (P1.2 / P3.1).

The HMM result must be invariant to the JAX *sample batch size*: samples are
conditionally independent given the founders, and the EM sufficient statistics
are a plain sum over samples, so splitting samples into batches must not change
dosage or genotype posteriors (up to float round-off). This test fails if the
device-accumulation rewrite dropped/double-counted a batch or mis-sliced the
resident block.

Requires JAX + a working HMM environment; skipped otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")

from stitchv2.config import HMMConfig
from stitchv2.founders import FounderPanel
from stitchv2.hmm import JAXStitchHMM


def _make_panel(n_founders: int, n_positions: int, *, immutable: bool, seed: int = 0) -> FounderPanel:
    rng = np.random.default_rng(seed)
    positions = (np.arange(n_positions, dtype=np.int64) + 1) * 1000
    ref = np.array(["A"] * n_positions)
    alt = np.array(["G"] * n_positions)
    alt_prob = rng.uniform(0.1, 0.9, size=(n_founders, n_positions)).astype(np.float32)
    mask = np.ones(n_founders, dtype=bool) if immutable else np.zeros(n_founders, dtype=bool)
    return FounderPanel(chromosome="1", positions=positions, ref=ref, alt=alt, alt_prob=alt_prob, immutable_mask=mask)


def _run(batch_size: int, *, immutable: bool, ploidy_mode: str, seed: int = 1):
    n_founders, n_positions, n_samples = 4, 24, 6
    panel = _make_panel(n_founders, n_positions, immutable=immutable)
    rng = np.random.default_rng(seed)
    depth = rng.integers(0, 5, size=(n_samples, n_positions))
    alt_count = rng.integers(0, depth + 1).astype(np.int32)
    ref_count = (depth - alt_count).astype(np.int32)
    other_count = np.zeros_like(ref_count)
    generations = np.full(n_samples, 4.0, dtype=np.float32)
    cfg = HMMConfig(
        n_founders=n_founders,
        ploidy_mode=ploidy_mode,
        backend="jax",
        em_iterations=3,
        adaptive_em=False,
        jax_sample_batch_size=batch_size,
        # fragment likelihood off so the dense-count path is exercised deterministically
        use_fragment_likelihood=False,
    )
    hmm = JAXStitchHMM(cfg)
    art = hmm.run(
        panel,
        ref_count,
        alt_count,
        generations,
        other_count=other_count,
        return_genotype_posterior=True,
        return_haplotype_posterior=True,
    )
    return art


@pytest.mark.parametrize("ploidy_mode", ["diploid", "pseudo_haploid"])
@pytest.mark.parametrize("immutable", [True, False])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_batch_size_invariance(ploidy_mode, immutable, batch_size):
    full = _run(0, immutable=immutable, ploidy_mode=ploidy_mode)        # single batch (all samples)
    batched = _run(batch_size, immutable=immutable, ploidy_mode=ploidy_mode)  # multiple batches
    np.testing.assert_allclose(full.dosage, batched.dosage, atol=1e-4, rtol=1e-3)
    if full.genotype_posterior is not None and batched.genotype_posterior is not None:
        np.testing.assert_allclose(
            full.genotype_posterior, batched.genotype_posterior, atol=1e-4, rtol=1e-3
        )
    if full.founder_alt_prob is not None and batched.founder_alt_prob is not None:
        np.testing.assert_allclose(
            full.founder_alt_prob, batched.founder_alt_prob, atol=1e-4, rtol=1e-3
        )
