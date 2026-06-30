from __future__ import annotations

import json
import math
import os
import resource
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .config import HMMConfig
from .founders import FounderPanel
from .hmm import HMMArtifacts, JAXStitchHMM, jax


@dataclass(slots=True)
class DaskChunkPlan:
    block_size: int
    sample_batch_size: int
    estimated_task_memory_mb: float
    state_count: int
    target_task_memory_mb: float
    reason: str

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "block_size": int(self.block_size),
            "sample_batch_size": int(self.sample_batch_size),
            "estimated_task_memory_mb": float(self.estimated_task_memory_mb),
            "state_count": int(self.state_count),
            "target_task_memory_mb": float(self.target_task_memory_mb),
            "reason": self.reason,
        }


@dataclass(slots=True)
class DaskHMMTaskResult:
    block_id: int
    sample_indices: np.ndarray
    ploidy: int
    artifact_path: str
    artifacts: HMMArtifacts | None
    seconds_hmm: float
    rss_mb: float
    jax_cache_key: str
    precompiled_shape_count: int
    jax_fast_cache_count: int
    jax_polyploid_cache_count: int
    jax_fragment_cache_cleared_count: int = 0

    def diagnostics(self) -> dict[str, Any]:
        return {
            "block_id": int(self.block_id),
            "ploidy": int(self.ploidy),
            "n_samples": int(self.sample_indices.shape[0]),
            "artifact_path": self.artifact_path,
            "artifact_transport": "memory" if self.artifacts is not None else "npz",
            "seconds_hmm": float(self.seconds_hmm),
            "rss_mb": float(self.rss_mb),
            "jax_cache_key": self.jax_cache_key,
            "precompiled_shape_count": int(self.precompiled_shape_count),
            "jax_fast_cache_count": int(self.jax_fast_cache_count),
            "jax_polyploid_cache_count": int(self.jax_polyploid_cache_count),
            "jax_fragment_cache_cleared_count": int(self.jax_fragment_cache_cleared_count),
        }


_HMM_CACHE: dict[tuple[Any, ...], JAXStitchHMM] = {}


def _clear_fragment_compile_cache(hmm: JAXStitchHMM) -> int:
    removed = 0
    for key in list(hmm._jax_compiled.keys()):
        if isinstance(key, tuple) and key and isinstance(key[0], str) and "fragment_count" in key[0]:
            hmm._jax_compiled.pop(key, None)
            removed += 1
    if removed and jax is not None and hasattr(jax, "clear_caches"):
        jax.clear_caches()
    return removed


def _artifact_array(value: np.ndarray | None) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float32)
    return np.asarray(value)


def _write_hmm_artifacts_npz(path: str | Path, artifacts: HMMArtifacts) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        dosage=_artifact_array(artifacts.dosage),
        haplotype_posterior=_artifact_array(artifacts.haplotype_posterior),
        genotype_posterior=_artifact_array(artifacts.genotype_posterior),
        genotype_call=_artifact_array(artifacts.genotype_call),
        recombination_rate=_artifact_array(artifacts.recombination_rate),
        switch_probability=_artifact_array(artifacts.switch_probability),
        stay_probability=_artifact_array(artifacts.stay_probability),
        offdiag_probability=_artifact_array(artifacts.offdiag_probability),
        founder_alt_prob=_artifact_array(artifacts.founder_alt_prob),
        transition_probability=_artifact_array(artifacts.transition_probability),
        transition_factor_source=_artifact_array(artifacts.transition_factor_source),
        transition_factor_destination=_artifact_array(artifacts.transition_factor_destination),
        transition_factor_offdiag=_artifact_array(artifacts.transition_factor_offdiag),
        em_diagnostics_json=np.asarray(
            [json.dumps(artifacts.em_diagnostics or {}, default=str)],
            dtype=np.str_,
        ),
        has_haplotype_posterior=np.asarray([artifacts.haplotype_posterior is not None], dtype=np.bool_),
        has_genotype_posterior=np.asarray([artifacts.genotype_posterior is not None], dtype=np.bool_),
        has_genotype_call=np.asarray([artifacts.genotype_call is not None], dtype=np.bool_),
        has_transition_probability=np.asarray([artifacts.transition_probability is not None], dtype=np.bool_),
        has_transition_factor_source=np.asarray([artifacts.transition_factor_source is not None], dtype=np.bool_),
        has_transition_factor_destination=np.asarray([artifacts.transition_factor_destination is not None], dtype=np.bool_),
        has_transition_factor_offdiag=np.asarray([artifacts.transition_factor_offdiag is not None], dtype=np.bool_),
        has_em_diagnostics=np.asarray([artifacts.em_diagnostics is not None], dtype=np.bool_),
    )


def load_hmm_artifacts_npz(path: str | Path) -> HMMArtifacts:
    with np.load(Path(path), allow_pickle=False) as data:
        has_hap = bool(data["has_haplotype_posterior"][0])
        has_gp = bool(data["has_genotype_posterior"][0])
        has_gt = bool(data["has_genotype_call"][0])
        has_full_transition = bool(data["has_transition_probability"][0])
        has_factor_source = bool(data["has_transition_factor_source"][0]) if "has_transition_factor_source" in data else False
        has_factor_destination = bool(data["has_transition_factor_destination"][0]) if "has_transition_factor_destination" in data else False
        has_factor_offdiag = bool(data["has_transition_factor_offdiag"][0]) if "has_transition_factor_offdiag" in data else False
        has_em_diagnostics = bool(data["has_em_diagnostics"][0]) if "has_em_diagnostics" in data else False
        em_diagnostics = None
        if has_em_diagnostics and "em_diagnostics_json" in data:
            try:
                em_diagnostics = json.loads(str(data["em_diagnostics_json"][0]))
            except Exception:
                em_diagnostics = None
        return HMMArtifacts(
            dosage=np.asarray(data["dosage"], dtype=np.float32),
            haplotype_posterior=(
                np.asarray(data["haplotype_posterior"], dtype=np.float32) if has_hap else None
            ),
            genotype_posterior=(
                np.asarray(data["genotype_posterior"], dtype=np.float32) if has_gp else None
            ),
            genotype_call=np.asarray(data["genotype_call"], dtype=np.int8) if has_gt else None,
            recombination_rate=np.asarray(data["recombination_rate"], dtype=np.float32),
            switch_probability=np.asarray(data["switch_probability"], dtype=np.float32),
            stay_probability=np.asarray(data["stay_probability"], dtype=np.float32),
            offdiag_probability=np.asarray(data["offdiag_probability"], dtype=np.float32),
            founder_alt_prob=np.asarray(data["founder_alt_prob"], dtype=np.float32),
            transition_probability=(
                np.asarray(data["transition_probability"], dtype=np.float32) if has_full_transition else None
            ),
            transition_factor_source=(
                np.asarray(data["transition_factor_source"], dtype=np.float32) if has_factor_source else None
            ),
            transition_factor_destination=(
                np.asarray(data["transition_factor_destination"], dtype=np.float32) if has_factor_destination else None
            ),
            transition_factor_offdiag=(
                np.asarray(data["transition_factor_offdiag"], dtype=np.float32) if has_factor_offdiag else None
            ),
            em_diagnostics=em_diagnostics,
        )


def _current_rss_mb() -> float:
    statm_path = Path("/proc/self/statm")
    if statm_path.exists():
        try:
            fields = statm_path.read_text(encoding="utf-8").split()
            if len(fields) >= 2:
                page_size = os.sysconf("SC_PAGE_SIZE")
                return (int(fields[1]) * float(page_size)) / (1024.0 * 1024.0)
        except Exception:
            pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def polyploid_state_count(k: int, ploidy: int, *, force_generic: bool = False) -> int:
    k = max(int(k), 1)
    ploidy = int(ploidy)
    if ploidy <= 0:
        return 1
    if ploidy == 1:
        return k
    if ploidy == 2 and not force_generic:
        return k * k
    return int(math.comb(k + ploidy - 1, ploidy))


def estimate_hmm_task_memory_mb(
    *,
    n_samples: int,
    n_variants: int,
    n_founders: int,
    ploidy: int,
    return_genotype_posterior: bool = False,
    return_haplotype_posterior: bool = False,
    return_full_transition: bool = False,
    force_generic_ploidy_hmm: bool = False,
    observed_fragments: int = 0,
    observed_fragment_observations: int = 0,
    fragment_density_multiplier: float = 1.25,
    bytes_per_float: int = 4,
) -> float:
    n_samples = max(int(n_samples), 1)
    n_variants = max(int(n_variants), 1)
    n_states = polyploid_state_count(
        int(n_founders),
        int(ploidy),
        force_generic=bool(force_generic_ploidy_hmm),
    )
    state_cells = n_samples * n_variants * n_states
    output_cells = n_samples * n_variants
    if return_haplotype_posterior:
        output_cells += n_samples * n_variants * max(int(n_founders), 1)
    if return_genotype_posterior:
        output_cells += n_samples * n_variants * (max(int(ploidy), 0) + 1)
    if return_full_transition:
        output_cells += state_cells * max(n_states, 1)
    # Emission, forward, backward/posterior, transition scratch, plus outputs.
    float_cells = (4 * state_cells) + output_cells
    base_mb = (float_cells * int(bytes_per_float)) / (1024.0 * 1024.0)
    fragment_cells = max(int(observed_fragments), 0) * max(n_states, 1)
    fragment_cells += max(int(observed_fragment_observations), 0) * max(int(n_founders), 1)
    fragment_mb = (fragment_cells * int(bytes_per_float)) / (1024.0 * 1024.0)
    return float(base_mb + (max(float(fragment_density_multiplier), 0.0) * fragment_mb))


def plan_dask_chunks(
    *,
    n_samples: int,
    n_variants: int,
    n_founders: int,
    max_ploidy: int,
    configured_block_size: int,
    configured_sample_batch_size: int = 0,
    target_task_memory_mb: float = 0.0,
    min_block_size: int = 128,
    min_sample_batch_size: int = 8,
    return_genotype_posterior: bool = False,
    return_haplotype_posterior: bool = False,
    return_full_transition: bool = False,
    force_generic_ploidy_hmm: bool = False,
    observed_fragments: int = 0,
    observed_fragment_observations: int = 0,
    fragment_density_multiplier: float = 1.25,
) -> DaskChunkPlan:
    n_samples = max(int(n_samples), 1)
    n_variants = max(int(n_variants), 1)
    block_size = min(max(int(configured_block_size), 1), n_variants)
    configured_sample_batch_size = int(configured_sample_batch_size)
    sample_batch_size = n_samples if configured_sample_batch_size <= 0 else min(configured_sample_batch_size, n_samples)
    min_block_size = max(min(int(min_block_size), block_size), 1)
    min_sample_batch_size = max(min(int(min_sample_batch_size), sample_batch_size), 1)
    target = max(float(target_task_memory_mb), 0.0)

    def estimate(bs: int, ss: int) -> float:
        return estimate_hmm_task_memory_mb(
            n_samples=ss,
            n_variants=bs,
            n_founders=n_founders,
            ploidy=max_ploidy,
            return_genotype_posterior=return_genotype_posterior,
            return_haplotype_posterior=return_haplotype_posterior,
            return_full_transition=return_full_transition,
            force_generic_ploidy_hmm=force_generic_ploidy_hmm,
            observed_fragments=int(math.ceil(max(int(observed_fragments), 0) * (float(ss) / float(n_samples)) * (float(bs) / float(n_variants)))),
            observed_fragment_observations=int(math.ceil(max(int(observed_fragment_observations), 0) * (float(ss) / float(n_samples)) * (float(bs) / float(n_variants)))),
            fragment_density_multiplier=float(fragment_density_multiplier),
        )

    reason = "configured"
    estimated = estimate(block_size, sample_batch_size)
    if target > 0.0:
        reason = "target_memory"
        while estimated > target and (block_size > min_block_size or sample_batch_size > min_sample_batch_size):
            if block_size > min_block_size and (block_size >= sample_batch_size or sample_batch_size <= min_sample_batch_size):
                block_size = max(min_block_size, int(math.ceil(block_size / 2.0)))
            elif sample_batch_size > min_sample_batch_size:
                sample_batch_size = max(min_sample_batch_size, int(math.ceil(sample_batch_size / 2.0)))
            else:
                break
            estimated = estimate(block_size, sample_batch_size)
        if estimated > target:
            reason = "minimum_chunk_exceeds_target"
        elif observed_fragments > 0 or observed_fragment_observations > 0:
            reason = "target_memory_with_fragments"

    return DaskChunkPlan(
        block_size=int(block_size),
        sample_batch_size=int(sample_batch_size),
        estimated_task_memory_mb=float(estimated),
        state_count=polyploid_state_count(
            int(n_founders),
            int(max_ploidy),
            force_generic=bool(force_generic_ploidy_hmm),
        ),
        target_task_memory_mb=float(target),
        reason=reason,
    )


def write_task_stream_artifact(path: str | Path, task_stream_data: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = json.loads(json.dumps(task_stream_data, default=str))
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(safe_data, indent=2), encoding="utf-8")
        return
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Dask Task Stream</title>"
        "<style>body{font-family:sans-serif;margin:2rem;line-height:1.45}"
        "pre{white-space:pre-wrap;background:#f6f8fa;padding:1rem;border-radius:8px}</style>"
        "</head><body><h1>Dask Task Stream</h1>"
        "<p>This static artifact records the task stream events captured during the run. "
        "Use the live Dask dashboard URL from <code>dask_run_summary.json</code> for the interactive timeline.</p>"
        f"<p><strong>Captured tasks:</strong> {len(safe_data)}</p>"
        f"<pre>{json.dumps(safe_data[:200], indent=2)}</pre>"
        "</body></html>"
    )
    path.write_text(html, encoding="utf-8")


def _hmm_config_key(config: HMMConfig, *, ploidy: int, n_samples: int, n_positions: int, k: int) -> tuple[Any, ...]:
    return (
        threading.get_ident(),
        int(n_samples),
        int(n_positions),
        int(k),
        int(ploidy),
        config.backend,
        bool(config.jax_precompile),
        bool(config.jax_aot_compile),
        bool(config.jax_bucket_batch_shapes),
        bool(config.jax_count_emission_kernel),
        bool(config.jax_fragment_emission_kernel),
        str(config.jax_persistent_cache_dir or ""),
        bool(config.force_generic_ploidy_hmm),
        int(config.em_iterations),
        int(config.final_diploid_iterations),
        float(config.em_convergence_tol),
        int(config.em_convergence_min_iterations),
        int(config.em_convergence_patience),
        float(config.em_founder_update_damping),
        bool(config.adaptive_em),
        bool(config.adaptive_em_restore_best_founders),
        int(config.em_multistarts),
        bool(config.founder_update_hardening),
        float(config.min_emission_prob),
        float(config.sequencing_error_rate),
        config.transition_model,
        int(config.transition_factor_rank),
        float(config.transition_factor_regularization),
        float(config.transition_factor_max_deviation),
        bool(config.transition_factor_train),
        bool(config.use_quality_weights),
        bool(config.use_fragment_likelihood),
        config.fragment_likelihood_mode,
        config.fragment_coupling_model,
        float(config.fragment_max_difference_between_reads),
        float(config.fragment_max_emission_matrix_difference),
        bool(config.fragment_rescale_read_likelihood),
        int(config.random_seed),
        float(config.founder_init_jitter),
    )


def run_hmm_leaf_task(
    *,
    block_id: int,
    sample_indices: np.ndarray,
    ploidy: int,
    artifact_dir: str | Path,
    hmm_config: HMMConfig,
    founder_panel: FounderPanel,
    ref_count: np.ndarray,
    alt_count: np.ndarray,
    generations: np.ndarray,
    other_count: np.ndarray | None,
    ref_weight: np.ndarray | None,
    alt_weight: np.ndarray | None,
    other_weight: np.ndarray | None,
    return_full_transition: bool,
    return_haplotype_posterior: bool,
    return_genotype_posterior: bool,
    fragment_sample_offsets: np.ndarray | None,
    fragment_center_idx: np.ndarray | None,
    fragment_obs_offsets: np.ndarray | None,
    fragment_obs_pos_idx: np.ndarray | None,
    fragment_obs_code: np.ndarray | None,
    fragment_obs_qual: np.ndarray | None,
    return_artifacts: bool = False,
) -> DaskHMMTaskResult:
    local_config = replace(
        hmm_config,
        ploidy=int(ploidy),
        ploidy_mode=("pseudo_haploid" if int(ploidy) == 1 else "diploid"),
        jax_sample_batch_size=int(hmm_config.jax_sample_batch_size),
        backend_autotune=False,
    )
    n_samples = int(ref_count.shape[0])
    n_positions = int(ref_count.shape[1])
    k = int(founder_panel.n_founders)
    cache_key = _hmm_config_key(
        local_config,
        ploidy=int(ploidy),
        n_samples=n_samples,
        n_positions=n_positions,
        k=k,
    )
    hmm = _HMM_CACHE.get(cache_key)
    if hmm is None:
        hmm = JAXStitchHMM(local_config)
        _HMM_CACHE[cache_key] = hmm
    if hmm.backend == "jax" and bool(hmm.config.jax_precompile):
        hmm._maybe_precompile(n_samples, n_positions, k)

    t0 = time.perf_counter()
    artifacts = hmm.run(
        founder_panel=founder_panel,
        ref_count=ref_count,
        alt_count=alt_count,
        generations=generations,
        other_count=other_count,
        ref_weight=ref_weight,
        alt_weight=alt_weight,
        other_weight=other_weight,
        return_full_transition=return_full_transition,
        return_haplotype_posterior=return_haplotype_posterior,
        return_genotype_posterior=return_genotype_posterior,
        fragment_sample_offsets=fragment_sample_offsets,
        fragment_center_idx=fragment_center_idx,
        fragment_obs_offsets=fragment_obs_offsets,
        fragment_obs_pos_idx=fragment_obs_pos_idx,
        fragment_obs_code=fragment_obs_code,
        fragment_obs_qual=fragment_obs_qual,
    )
    seconds = time.perf_counter() - t0
    sample_indices_arr = np.asarray(sample_indices, dtype=np.int64)
    if return_artifacts:
        artifact_path = ""
        result_artifacts = artifacts
    else:
        if sample_indices_arr.size:
            sample_tag = f"{int(sample_indices_arr[0]):06d}-{int(sample_indices_arr[-1]):06d}"
        else:
            sample_tag = "empty"
        artifact_path = Path(artifact_dir) / (
            f"block={int(block_id):06d}"
            f".ploidy={int(ploidy)}"
            f".samples={sample_tag}"
            f".thread={threading.get_ident()}.npz"
        )
        _write_hmm_artifacts_npz(artifact_path, artifacts)
        result_artifacts = None
    fragment_cache_cleared = _clear_fragment_compile_cache(hmm) if hmm.backend == "jax" else 0
    return DaskHMMTaskResult(
        block_id=int(block_id),
        sample_indices=sample_indices_arr,
        ploidy=int(ploidy),
        artifact_path=str(artifact_path),
        artifacts=result_artifacts,
        seconds_hmm=float(seconds),
        rss_mb=float(_current_rss_mb()),
        jax_cache_key=str(cache_key),
        precompiled_shape_count=int(len(hmm._precompiled_shapes)),
        jax_fast_cache_count=int(len(hmm._jax_compiled)),
        jax_polyploid_cache_count=int(len(hmm._jax_polyploid_compiled)),
        jax_fragment_cache_cleared_count=int(fragment_cache_cleared),
    )
