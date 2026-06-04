"""Unit tests for the GPU- and CPU-thread sizing helpers added for the
GPU-usage / BAM-reading / memory-management improvements.

These do not require a GPU or JAX: ``_gpu_free_bytes`` is monkeypatched so the
auto-sizing maths can be exercised deterministically.
"""
from __future__ import annotations

import numpy as np  # noqa: F401  (hmm imports numpy; ensures env is sane)

from stitchv2 import hmm as hmm_mod
from stitchv2.config import HMMConfig, PipelineConfig
from stitchv2.hmm import JAXStitchHMM
from stitchv2.pipeline import _resolve_io_threads


def _hmm(**overrides) -> JAXStitchHMM:
    cfg = HMMConfig(n_founders=4, **overrides)
    return JAXStitchHMM(cfg)


def test_batch_size_no_gpu_returns_all_samples(monkeypatch):
    monkeypatch.setattr(hmm_mod, "_gpu_free_bytes", lambda: 0)
    h = _hmm(jax_sample_batch_size=0)
    assert h._resolve_jax_sample_batch_size(1000, n_positions=5000, k=4) == 1000


def test_explicit_batch_size_is_respected(monkeypatch):
    # Even with plenty of GPU memory an explicit batch size wins.
    monkeypatch.setattr(hmm_mod, "_gpu_free_bytes", lambda: 10 * 1024**3)
    h = _hmm(jax_sample_batch_size=128, jax_bucket_batch_shapes=False)
    assert h._resolve_jax_sample_batch_size(1000, n_positions=5000, k=4) == 128


def test_gpu_auto_shrinks_batch(monkeypatch):
    # Small VRAM => batch should shrink below n_samples.
    monkeypatch.setattr(hmm_mod, "_gpu_free_bytes", lambda: 64 * 1024**2)
    h = _hmm(jax_sample_batch_size=0, jax_bucket_batch_shapes=False)
    b = h._resolve_jax_sample_batch_size(10_000, n_positions=2000, k=8)
    assert 1 <= b < 10_000


def test_gpu_auto_clamps_to_one(monkeypatch):
    monkeypatch.setattr(hmm_mod, "_gpu_free_bytes", lambda: 1)
    h = _hmm(jax_sample_batch_size=0)
    assert h._resolve_jax_sample_batch_size(100, n_positions=100000, k=64) == 1


def test_bytes_per_sample_diploid_gt_haploid():
    h_dip = _hmm(ploidy_mode="diploid")
    h_hap = _hmm(ploidy_mode="pseudo_haploid")
    assert h_dip._estimate_gpu_bytes_per_sample(1000, 8) > h_hap._estimate_gpu_bytes_per_sample(1000, 8)


def _pcfg(**ov) -> PipelineConfig:
    return PipelineConfig(chromosome="1", positions_path="p", output_dir="o", n_founders=4, **ov)


def test_io_threads_zero_keeps_explicit():
    assert _resolve_io_threads(_pcfg(io_workers=3, htslib_threads_per_file=2, io_threads_total=0)) == (3, 2)


def test_io_threads_positive_splits_budget():
    # total=8, explicit htslib=2 => io_workers=4
    assert _resolve_io_threads(_pcfg(htslib_threads_per_file=2, io_threads_total=8)) == (4, 2)
    # total=8, htslib default 1 => io_workers=8
    assert _resolve_io_threads(_pcfg(io_threads_total=8)) == (8, 1)


def test_io_threads_auto_negative_is_bounded():
    workers, htslib = _resolve_io_threads(_pcfg(io_threads_total=-1, executor="serial"))
    assert workers >= 1 and htslib >= 1
