from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import factorial
from pathlib import Path
import threading
from time import perf_counter

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception:  # pragma: no cover - JAX is optional at runtime.
    jax = None
    jnp = None

try:
    import torch
except Exception:  # pragma: no cover - Torch is optional at runtime.
    torch = None

from .config import HMMConfig
from .founders import FounderPanel


LOG_HALF = float(np.log(0.5))
_JAX_COMPILE_LOCK = threading.Lock()


@dataclass(slots=True)
class HMMArtifacts:
    dosage: np.ndarray
    haplotype_posterior: np.ndarray | None
    genotype_posterior: np.ndarray | None
    genotype_call: np.ndarray | None
    recombination_rate: np.ndarray
    switch_probability: np.ndarray
    stay_probability: np.ndarray
    offdiag_probability: np.ndarray
    founder_alt_prob: np.ndarray
    transition_probability: np.ndarray | None = None
    em_diagnostics: dict[str, object] | None = None


def _safe_log_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, 1.0))


def _gpu_available() -> bool:
    if jax is None:
        return False
    try:
        return any(device.platform in {"gpu", "tpu"} for device in jax.devices())
    except Exception:
        return False


def _torch_available() -> bool:
    return torch is not None


def _build_transition_matrix_from_switch(switch: float, k: int) -> np.ndarray:
    stay = 1.0 - switch
    off = switch / max(k - 1, 1)
    mat = np.full((k, k), off, dtype=np.float32)
    np.fill_diagonal(mat, stay)
    return mat


def _diploid_pair_to_haplotype_posterior(pair_posterior: np.ndarray) -> np.ndarray:
    return 0.5 * (pair_posterior.sum(axis=3) + pair_posterior.sum(axis=2))


@lru_cache(maxsize=64)
def _polyploid_state_counts(k: int, ploidy: int) -> np.ndarray:
    states: list[tuple[int, ...]] = []

    def rec(founder_idx: int, remaining: int, prefix: list[int]) -> None:
        if founder_idx == k - 1:
            states.append(tuple(prefix + [remaining]))
            return
        for count in range(remaining + 1):
            rec(founder_idx + 1, remaining - count, prefix + [count])

    rec(0, int(ploidy), [])
    return np.asarray(states, dtype=np.int16)


@lru_cache(maxsize=64)
def _polyploid_allocations(k: int, n: int) -> tuple[tuple[tuple[int, ...], float], ...]:
    if n == 0:
        return ((tuple([0] * k), 1.0),)
    out: list[tuple[tuple[int, ...], float]] = []

    def rec(idx: int, remaining: int, prefix: list[int]) -> None:
        if idx == k - 1:
            counts = tuple(prefix + [remaining])
            coeff = float(factorial(n))
            for value in counts:
                coeff /= float(factorial(int(value)))
            out.append((counts, coeff))
            return
        for count in range(remaining + 1):
            rec(idx + 1, remaining - count, prefix + [count])

    rec(0, int(n), [])
    return tuple(out)


@lru_cache(maxsize=32)
def _polyploid_transition_coefficients(k: int, ploidy: int) -> np.ndarray:
    states = _polyploid_state_counts(k, ploidy)
    state_index = {tuple(row.tolist()): idx for idx, row in enumerate(states)}
    n_states = int(states.shape[0])
    coeff = np.zeros((n_states, n_states, int(ploidy) + 1), dtype=np.float32)
    for source_idx, source in enumerate(states):
        dist: dict[tuple[tuple[int, ...], int], float] = {(tuple([0] * k), 0): 1.0}
        for founder_idx, n_copies in enumerate(source.tolist()):
            if int(n_copies) == 0:
                continue
            new_dist: dict[tuple[tuple[int, ...], int], float] = {}
            for (target_prefix, stay_count), value in dist.items():
                for alloc, mult in _polyploid_allocations(k, int(n_copies)):
                    target = tuple(int(a) + int(b) for a, b in zip(target_prefix, alloc, strict=False))
                    stays = stay_count + int(alloc[founder_idx])
                    key = (target, stays)
                    new_dist[key] = new_dist.get(key, 0.0) + float(value) * float(mult)
            dist = new_dist
        for (target, stays), value in dist.items():
            coeff[source_idx, state_index[target], int(stays)] += float(value)
    return coeff


@lru_cache(maxsize=64)
def _polyploid_state_prior(k: int, ploidy: int) -> np.ndarray:
    states = _polyploid_state_counts(k, ploidy)
    denom = float(max(k, 1) ** max(int(ploidy), 0))
    prior = np.empty(states.shape[0], dtype=np.float32)
    for idx, counts in enumerate(states):
        mult = float(factorial(int(ploidy)))
        for value in counts.tolist():
            mult /= float(factorial(int(value)))
        prior[idx] = mult / denom
    prior /= np.clip(np.sum(prior), 1e-12, None)
    return prior.astype(np.float32, copy=False)


class JAXStitchHMM:
    def __init__(self, config: HMMConfig):
        self.config = config
        self._configured_backend = config.backend
        if config.backend == "auto":
            self.backend = "numpy"
        else:
            self.backend = config.backend
        self._precompiled_shapes: set[tuple[int, int, int, str, str]] = set()
        self._jax_compiled: dict[tuple[int, int, int], object] = {}
        self._jax_polyploid_compiled: dict[tuple[int, int, int, int, int], object] = {}
        self._autotuned = False
        self._rng = np.random.default_rng(int(config.random_seed))
        self._configure_jax_persistent_cache()

    def _configure_jax_persistent_cache(self) -> None:
        if jax is None:
            return
        cache_dir = self.config.jax_persistent_cache_dir
        if cache_dir is None or str(cache_dir).strip() == "":
            return
        try:
            path = Path(cache_dir)
            path.mkdir(parents=True, exist_ok=True)
            jax.config.update("jax_compilation_cache_dir", str(path))
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        except Exception:
            return

    def _read_log_likelihood_total(
        self,
        *,
        dosage: np.ndarray,
        genotype_posterior: np.ndarray | None,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray | None,
        ploidy: int,
    ) -> float:
        try:
            from .calibration import dosage_to_genotype_posterior, read_log_likelihood_from_posterior

            gp = genotype_posterior
            if gp is None:
                gp = dosage_to_genotype_posterior(
                    np.asarray(dosage, dtype=np.float32),
                    temperature=0.35,
                    ploidy=int(ploidy),
                )
            _, total = read_log_likelihood_from_posterior(
                ref_count=ref_count.astype(np.float32, copy=False),
                alt_count=alt_count.astype(np.float32, copy=False),
                other_count=(None if other_count is None else other_count.astype(np.float32, copy=False)),
                posterior=np.asarray(gp, dtype=np.float32),
                sequencing_error_rate=float(self.config.sequencing_error_rate),
            )
            return float(total)
        except Exception:
            return float("nan")

    def _run_multistart(
        self,
        *,
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
    ) -> HMMArtifacts:
        n_starts = max(int(self.config.em_multistarts), 1)
        best: HMMArtifacts | None = None
        best_score = -np.inf
        rows: list[dict[str, object]] = []
        base_jitter = float(self.config.founder_init_jitter)
        for start_idx in range(n_starts):
            cfg = replace(
                self.config,
                em_multistarts=1,
                random_seed=int(self.config.random_seed) + int(start_idx),
                founder_init_jitter=(base_jitter if start_idx == 0 else max(base_jitter, 0.02)),
            )
            out = JAXStitchHMM(cfg).run(
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
            diag = dict(out.em_diagnostics or {})
            read_ll = float(diag.get("final_read_log_likelihood", float("nan")))
            if np.isfinite(read_ll):
                score = read_ll
                score_name = "final_read_log_likelihood"
            else:
                score = -float(diag.get("final_founder_max_delta", float("inf")))
                score_name = "negative_final_founder_max_delta"
            rows.append(
                {
                    "start": int(start_idx),
                    "seed": int(cfg.random_seed),
                    "founder_init_jitter": float(cfg.founder_init_jitter),
                    "score": float(score),
                    "score_name": score_name,
                    "final_read_log_likelihood": read_ll,
                    "final_founder_max_delta": diag.get("final_founder_max_delta"),
                    "em_updates": diag.get("em_updates"),
                }
            )
            if best is None or score > best_score:
                best = out
                best_score = float(score)
        if best is None:
            raise RuntimeError("EM multistart produced no HMM result.")
        best_diag = dict(best.em_diagnostics or {})
        best_diag["multistart"] = {
            "n_starts": int(n_starts),
            "selected_start": int(max(range(len(rows)), key=lambda i: float(rows[i]["score"]))),
            "selected_score": float(best_score),
            "starts": rows,
        }
        best.em_diagnostics = best_diag
        return best

    def recombination_from_positions(self, positions: np.ndarray) -> np.ndarray:
        delta = np.diff(positions, prepend=positions[0]).astype(np.float32)
        # Convert cM/Mb to Morgans per bp, matching original STITCH's expRate scaling.
        rate_per_bp = float(self.config.recombination_rate_cM_per_Mb) / 100.0 / 1_000_000.0
        rates = np.clip(delta * rate_per_bp, 1e-10, 0.25)
        rates[0] = 1e-10
        return rates

    def _switch_probabilities(self, rates: np.ndarray, generations: np.ndarray) -> np.ndarray:
        generations = generations.astype(np.float32)[:, None]
        switch = 1.0 - np.exp(-generations * rates[None, :])
        switch[:, 0] = 1e-10
        return np.clip(switch, 1e-10, 1.0 - 1e-10).astype(np.float32)

    def emissions(
        self,
        founder_alt_prob: np.ndarray,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray | None,
        ref_weight: np.ndarray | None,
        alt_weight: np.ndarray | None,
        other_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        founder_alt = founder_alt_prob.astype(np.float32, copy=False)
        if self.config.use_quality_weights and ref_weight is not None and alt_weight is not None:
            ref_obs = ref_weight.astype(np.float32, copy=False)
            alt_obs = alt_weight.astype(np.float32, copy=False)
            if other_weight is None:
                other_obs = np.zeros_like(ref_obs, dtype=np.float32)
            else:
                other_obs = other_weight.astype(np.float32, copy=False)
        else:
            ref_obs = ref_count.astype(np.float32, copy=False)
            alt_obs = alt_count.astype(np.float32, copy=False)
            if other_count is None:
                other_obs = np.zeros_like(ref_obs, dtype=np.float32)
            else:
                other_obs = other_count.astype(np.float32, copy=False)

        eps = float(self.config.sequencing_error_rate)
        p_err_other = eps / 3.0
        if self.config.ploidy_mode == "pseudo_haploid":
            p_alt = np.clip(founder_alt * (1.0 - eps) + (1.0 - founder_alt) * p_err_other, self.config.min_emission_prob, 1.0)
            p_ref = np.clip((1.0 - founder_alt) * (1.0 - eps) + founder_alt * p_err_other, self.config.min_emission_prob, 1.0)
            p_oth = np.clip(np.full_like(p_alt, p_err_other), self.config.min_emission_prob, 1.0)
            log_emission = (
                alt_obs[:, None, :] * _safe_log_np(p_alt[None, :, :])
                + ref_obs[:, None, :] * _safe_log_np(p_ref[None, :, :])
                + other_obs[:, None, :] * _safe_log_np(p_oth[None, :, :])
            )
            return np.swapaxes(log_emission, 1, 2), founder_alt, founder_alt

        dosage = founder_alt[:, None, :] + founder_alt[None, :, :]
        dosage = np.clip(dosage, 0.0, 2.0)
        hap_alt = dosage / 2.0
        p_alt = np.clip(hap_alt * (1.0 - eps) + (1.0 - hap_alt) * p_err_other, self.config.min_emission_prob, 1.0)
        p_ref = np.clip((1.0 - hap_alt) * (1.0 - eps) + hap_alt * p_err_other, self.config.min_emission_prob, 1.0)
        p_oth = np.clip(np.full_like(p_alt, p_err_other), self.config.min_emission_prob, 1.0)
        log_emission = (
            alt_obs[:, None, None, :] * _safe_log_np(p_alt[None, :, :, :])
            + ref_obs[:, None, None, :] * _safe_log_np(p_ref[None, :, :, :])
            + other_obs[:, None, None, :] * _safe_log_np(p_oth[None, :, :, :])
        )
        return np.transpose(log_emission, (0, 3, 1, 2)), dosage, founder_alt

    def _fragment_haplotype_log_likelihood(
        self,
        founder_alt: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_pos_idx: np.ndarray,
        fragment_obs_code: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ) -> np.ndarray:
        n_fragments = int(fragment_obs_offsets.shape[0] - 1)
        if n_fragments <= 0:
            return np.empty((0, founder_alt.shape[0]), dtype=np.float32)

        obs_pos = fragment_obs_pos_idx.astype(np.int64, copy=False)
        obs_code = fragment_obs_code.astype(np.int8, copy=False)
        founder_at_obs = founder_alt[:, obs_pos].T.astype(np.float32, copy=False)
        if fragment_obs_qual is None or fragment_obs_qual.size == 0:
            err = np.full(obs_pos.shape[0], float(self.config.sequencing_error_rate), dtype=np.float32)
        else:
            err = np.power(10.0, -(fragment_obs_qual.astype(np.float32, copy=False) / 10.0)).astype(np.float32, copy=False)
        p_err_other = err / 3.0
        p_alt = np.clip(founder_at_obs * (1.0 - err[:, None]) + (1.0 - founder_at_obs) * p_err_other[:, None], self.config.min_emission_prob, 1.0)
        p_ref = np.clip((1.0 - founder_at_obs) * (1.0 - err[:, None]) + founder_at_obs * p_err_other[:, None], self.config.min_emission_prob, 1.0)
        p_oth = np.clip(np.broadcast_to(p_err_other[:, None], p_alt.shape), self.config.min_emission_prob, 1.0)
        log_p = np.where(
            obs_code[:, None] == 1,
            np.log(p_alt),
            np.where(obs_code[:, None] == 0, np.log(p_ref), np.log(p_oth)),
        ).astype(np.float32, copy=False)
        return np.add.reduceat(log_p, fragment_obs_offsets[:-1], axis=0).astype(np.float32, copy=False)

    def _fragment_haplotype_stitch_scaled(
        self,
        founder_alt: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_pos_idx: np.ndarray,
        fragment_obs_code: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ) -> np.ndarray:
        n_fragments = int(fragment_obs_offsets.shape[0] - 1)
        if n_fragments <= 0:
            return np.empty((0, founder_alt.shape[0]), dtype=np.float32)

        obs_pos = fragment_obs_pos_idx.astype(np.int64, copy=False)
        obs_code = fragment_obs_code.astype(np.int8, copy=False)
        founder_at_obs = founder_alt[:, obs_pos].T.astype(np.float32, copy=False)
        if fragment_obs_qual is None or fragment_obs_qual.size == 0:
            err = np.full(obs_pos.shape[0], float(self.config.sequencing_error_rate), dtype=np.float32)
        else:
            err = np.power(
                10.0,
                -(fragment_obs_qual.astype(np.float32, copy=False) / 10.0),
            ).astype(np.float32, copy=False)
        p_err_other = err / 3.0
        p_alt = np.clip(
            founder_at_obs * (1.0 - err[:, None]) + (1.0 - founder_at_obs) * p_err_other[:, None],
            self.config.min_emission_prob,
            1.0,
        )
        p_ref = np.clip(
            (1.0 - founder_at_obs) * (1.0 - err[:, None]) + founder_at_obs * p_err_other[:, None],
            self.config.min_emission_prob,
            1.0,
        )
        p_oth = np.clip(
            np.broadcast_to(p_err_other[:, None], p_alt.shape),
            self.config.min_emission_prob,
            1.0,
        )
        log_p = np.where(
            obs_code[:, None] == 1,
            np.log(p_alt),
            np.where(obs_code[:, None] == 0, np.log(p_ref), np.log(p_oth)),
        ).astype(np.float32, copy=False)
        frag_log = np.add.reduceat(log_p, fragment_obs_offsets[:-1], axis=0).astype(np.float32, copy=False)
        if not self.config.fragment_rescale_read_likelihood:
            return np.exp(frag_log).astype(np.float32, copy=False)
        frag_log = frag_log - np.max(frag_log, axis=1, keepdims=True)
        min_log = -float(np.log(max(self.config.fragment_max_difference_between_reads, 1.000001)))
        frag_log = np.maximum(frag_log, min_log)
        return np.exp(frag_log).astype(np.float32, copy=False)

    def _apply_fragment_likelihoods_legacy(
        self,
        log_emission: np.ndarray,
        founder_alt: np.ndarray,
        fragment_sample_offsets: np.ndarray,
        fragment_center_idx: np.ndarray,
        frag_ll_hap: np.ndarray,
    ) -> np.ndarray:
        mode = self.config.fragment_likelihood_mode
        n_samples = log_emission.shape[0]
        n_positions = log_emission.shape[1]
        for sample_idx in range(n_samples):
            frag_start = int(fragment_sample_offsets[sample_idx])
            frag_stop = int(fragment_sample_offsets[sample_idx + 1])
            if frag_stop <= frag_start:
                continue
            centers = fragment_center_idx[frag_start:frag_stop].astype(np.int32, copy=False)
            ll = frag_ll_hap[frag_start:frag_stop]
            sort_idx = np.argsort(centers, kind="mergesort")
            centers = centers[sort_idx]
            ll = ll[sort_idx]
            unique_centers, starts = np.unique(centers, return_index=True)
            ll_center = np.add.reduceat(ll, starts, axis=0)
            valid = (unique_centers >= 0) & (unique_centers < n_positions)
            if not np.any(valid):
                continue
            unique_centers = unique_centers[valid]
            ll_center = ll_center[valid]
            if self.config.ploidy_mode == "pseudo_haploid":
                if mode == "replace":
                    log_emission[sample_idx, unique_centers, :] = ll_center
                else:
                    log_emission[sample_idx, unique_centers, :] += ll_center
                continue
            for idx_center in range(unique_centers.shape[0]):
                center = int(unique_centers[idx_center])
                ll_h = ll_center[idx_center]
                pair_ll = np.logaddexp(ll_h[:, None], ll_h[None, :]) + LOG_HALF
                if mode == "replace":
                    log_emission[sample_idx, center, :, :] = pair_ll
                else:
                    log_emission[sample_idx, center, :, :] += pair_ll
        return log_emission

    def _apply_fragment_likelihoods_stitch_parity(
        self,
        log_emission: np.ndarray,
        founder_alt: np.ndarray,
        fragment_sample_offsets: np.ndarray,
        fragment_center_idx: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_pos_idx: np.ndarray,
        fragment_obs_code: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ) -> np.ndarray:
        read_prob_hap = self._fragment_haplotype_stitch_scaled(
            founder_alt=founder_alt,
            fragment_obs_offsets=fragment_obs_offsets,
            fragment_obs_pos_idx=fragment_obs_pos_idx,
            fragment_obs_code=fragment_obs_code,
            fragment_obs_qual=fragment_obs_qual,
        )
        if read_prob_hap.size == 0:
            return log_emission

        mode = self.config.fragment_likelihood_mode
        n_samples = log_emission.shape[0]
        n_positions = log_emission.shape[1]
        frag_counts = np.diff(fragment_sample_offsets).astype(np.int64, copy=False)
        sample_for_fragment = np.repeat(np.arange(n_samples, dtype=np.int64), frag_counts)
        if sample_for_fragment.shape[0] != fragment_center_idx.shape[0]:
            return self._apply_fragment_likelihoods_legacy(
                log_emission,
                founder_alt,
                fragment_sample_offsets,
                fragment_center_idx,
                np.log(np.clip(read_prob_hap, self.config.min_emission_prob, 1.0)),
            )
        centers = fragment_center_idx.astype(np.int64, copy=False)
        valid = (centers >= 0) & (centers < n_positions)
        if not np.any(valid):
            return log_emission
        centers = centers[valid]
        sample_for_fragment = sample_for_fragment[valid]
        probs = read_prob_hap[valid]

        if self.config.ploidy_mode == "pseudo_haploid":
            frag_ll = np.log(np.clip(probs, self.config.min_emission_prob, 1.0)).astype(np.float32, copy=False)
            if mode == "replace":
                acc = np.zeros_like(log_emission, dtype=np.float32)
                np.add.at(acc, (sample_for_fragment, centers), frag_ll)
                touched = np.zeros(log_emission.shape[:2], dtype=bool)
                touched[sample_for_fragment, centers] = True
                log_emission[touched] = acc[touched]
            else:
                np.add.at(log_emission, (sample_for_fragment, centers), frag_ll)
                touched = np.zeros(log_emission.shape[:2], dtype=bool)
                touched[sample_for_fragment, centers] = True
        else:
            pair = 0.5 * (probs[:, :, None] + probs[:, None, :])
            frag_ll = np.log(np.clip(pair, self.config.min_emission_prob, 1.0)).astype(np.float32, copy=False)
            if mode == "replace":
                acc = np.zeros_like(log_emission, dtype=np.float32)
                np.add.at(acc, (sample_for_fragment, centers), frag_ll)
                touched = np.zeros(log_emission.shape[:2], dtype=bool)
                touched[sample_for_fragment, centers] = True
                log_emission[touched] = acc[touched]
            else:
                np.add.at(log_emission, (sample_for_fragment, centers), frag_ll)
                touched = np.zeros(log_emission.shape[:2], dtype=bool)
                touched[sample_for_fragment, centers] = True

        if self.config.fragment_rescale_read_likelihood and np.any(touched):
            min_state_log = -float(np.log(max(self.config.fragment_max_emission_matrix_difference, 1.000001)))
            sample_idx, pos_idx = np.nonzero(touched)
            cols = log_emission[sample_idx, pos_idx]
            reduce_axes = tuple(range(1, cols.ndim))
            cols = cols - np.max(cols, axis=reduce_axes, keepdims=True)
            cols = np.maximum(cols, min_state_log)
            log_emission[sample_idx, pos_idx] = cols
        return log_emission

    def _apply_fragment_likelihoods(
        self,
        log_emission: np.ndarray,
        founder_alt: np.ndarray,
        fragment_sample_offsets: np.ndarray | None,
        fragment_center_idx: np.ndarray | None,
        fragment_obs_offsets: np.ndarray | None,
        fragment_obs_pos_idx: np.ndarray | None,
        fragment_obs_code: np.ndarray | None,
        fragment_obs_qual: np.ndarray | None,
    ) -> np.ndarray:
        if not self.config.use_fragment_likelihood:
            return log_emission
        if (
            fragment_sample_offsets is None
            or fragment_center_idx is None
            or fragment_obs_offsets is None
            or fragment_obs_pos_idx is None
            or fragment_obs_code is None
        ):
            return log_emission
        if fragment_center_idx.size == 0:
            return log_emission

        return self._apply_fragment_likelihoods_stitch_parity(
            log_emission=log_emission,
            founder_alt=founder_alt,
            fragment_sample_offsets=fragment_sample_offsets,
            fragment_center_idx=fragment_center_idx,
            fragment_obs_offsets=fragment_obs_offsets,
            fragment_obs_pos_idx=fragment_obs_pos_idx,
            fragment_obs_code=fragment_obs_code,
            fragment_obs_qual=fragment_obs_qual,
        )

    def _apply_fragment_likelihoods_polyploid(
        self,
        log_emission: np.ndarray,
        founder_alt: np.ndarray,
        state_counts: np.ndarray,
        ploidy: int,
        fragment_sample_offsets: np.ndarray | None,
        fragment_center_idx: np.ndarray | None,
        fragment_obs_offsets: np.ndarray | None,
        fragment_obs_pos_idx: np.ndarray | None,
        fragment_obs_code: np.ndarray | None,
        fragment_obs_qual: np.ndarray | None,
    ) -> np.ndarray:
        if not self.config.use_fragment_likelihood:
            return log_emission
        if (
            fragment_sample_offsets is None
            or fragment_center_idx is None
            or fragment_obs_offsets is None
            or fragment_obs_pos_idx is None
            or fragment_obs_code is None
        ):
            return log_emission
        if fragment_center_idx.size == 0:
            return log_emission

        read_prob_hap = self._fragment_haplotype_stitch_scaled(
            founder_alt=founder_alt,
            fragment_obs_offsets=fragment_obs_offsets,
            fragment_obs_pos_idx=fragment_obs_pos_idx,
            fragment_obs_code=fragment_obs_code,
            fragment_obs_qual=fragment_obs_qual,
        )
        if read_prob_hap.size == 0:
            return log_emission

        state_fraction = state_counts.astype(np.float32, copy=False) / float(ploidy)
        mode = self.config.fragment_likelihood_mode
        n_samples = log_emission.shape[0]
        n_positions = log_emission.shape[1]
        frag_counts = np.diff(fragment_sample_offsets).astype(np.int64, copy=False)
        sample_for_fragment = np.repeat(np.arange(n_samples, dtype=np.int64), frag_counts)
        if sample_for_fragment.shape[0] != fragment_center_idx.shape[0]:
            return log_emission
        centers = fragment_center_idx.astype(np.int64, copy=False)
        valid = (centers >= 0) & (centers < n_positions)
        if not np.any(valid):
            return log_emission
        centers = centers[valid]
        sample_for_fragment = sample_for_fragment[valid]
        probs = read_prob_hap[valid]
        n_state_cells = int(probs.shape[0]) * int(state_fraction.shape[0])
        if n_state_cells > 50_000_000:
            # Avoid a very large temporary on high-ploidy/read-heavy blocks.
            min_state_log = -float(np.log(max(self.config.fragment_max_emission_matrix_difference, 1.000001)))
            for sample_idx in range(n_samples):
                frag_start = int(fragment_sample_offsets[sample_idx])
                frag_stop = int(fragment_sample_offsets[sample_idx + 1])
                if frag_stop <= frag_start:
                    continue
                sample_centers = fragment_center_idx[frag_start:frag_stop].astype(np.int32, copy=False)
                sample_probs = read_prob_hap[frag_start:frag_stop]
                sample_valid = (sample_centers >= 0) & (sample_centers < n_positions)
                if not np.any(sample_valid):
                    continue
                sample_centers = sample_centers[sample_valid]
                sample_probs = sample_probs[sample_valid]
                order = np.argsort(sample_centers, kind="mergesort")
                sample_centers = sample_centers[order]
                sample_probs = sample_probs[order]
                unique_centers, starts = np.unique(sample_centers, return_index=True)
                ends = np.r_[starts[1:], sample_centers.shape[0]]
                for idx_center, center in enumerate(unique_centers.tolist()):
                    chunk = sample_probs[int(starts[idx_center]) : int(ends[idx_center])]
                    state_read_prob = chunk @ state_fraction.T
                    ll = np.sum(np.log(np.clip(state_read_prob, self.config.min_emission_prob, 1.0)), axis=0)
                    if mode == "replace":
                        log_emission[sample_idx, int(center), :] = ll
                    else:
                        log_emission[sample_idx, int(center), :] += ll
                    if self.config.fragment_rescale_read_likelihood:
                        col = log_emission[sample_idx, int(center), :]
                        col = col - np.max(col)
                        col = np.maximum(col, min_state_log)
                        log_emission[sample_idx, int(center), :] = col
            return log_emission

        state_read_prob = probs @ state_fraction.T
        frag_ll = np.log(np.clip(state_read_prob, self.config.min_emission_prob, 1.0)).astype(np.float32, copy=False)
        if mode == "replace":
            acc = np.zeros_like(log_emission, dtype=np.float32)
            np.add.at(acc, (sample_for_fragment, centers), frag_ll)
            touched = np.zeros(log_emission.shape[:2], dtype=bool)
            touched[sample_for_fragment, centers] = True
            log_emission[touched] = acc[touched]
        else:
            np.add.at(log_emission, (sample_for_fragment, centers), frag_ll)
            touched = np.zeros(log_emission.shape[:2], dtype=bool)
            touched[sample_for_fragment, centers] = True
        if self.config.fragment_rescale_read_likelihood and np.any(touched):
            min_state_log = -float(np.log(max(self.config.fragment_max_emission_matrix_difference, 1.000001)))
            sample_idx, pos_idx = np.nonzero(touched)
            cols = log_emission[sample_idx, pos_idx]
            cols = cols - np.max(cols, axis=1, keepdims=True)
            cols = np.maximum(cols, min_state_log)
            log_emission[sample_idx, pos_idx] = cols
        return log_emission

    def _forward_backward_numpy_diploid(self, log_emission: np.ndarray, switch: np.ndarray, k: int) -> np.ndarray:
        n_samples, n_positions = log_emission.shape[:2]
        emission = np.exp(log_emission - np.max(log_emission, axis=(2, 3), keepdims=True)).astype(np.float32)
        gamma = np.zeros_like(emission, dtype=np.float32)

        for sample_idx in range(n_samples):
            alpha = np.zeros((n_positions, k, k), dtype=np.float32)
            beta = np.zeros((n_positions, k, k), dtype=np.float32)
            alpha0 = emission[sample_idx, 0].copy()
            alpha0_sum = float(alpha0.sum())
            if alpha0_sum <= 0.0:
                alpha0.fill(1.0 / float(k * k))
            else:
                alpha0 /= alpha0_sum
            alpha[0] = alpha0

            for pos_idx in range(1, n_positions):
                sw = float(switch[sample_idx, pos_idx])
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                prev = alpha[pos_idx - 1]
                row_sum = prev.sum(axis=1, keepdims=True)
                col_sum = prev.sum(axis=0, keepdims=True)
                total = float(prev.sum())
                pred = (b * b) * prev + (b * off) * (row_sum + col_sum) + (off * off) * total
                alpha_t = pred * emission[sample_idx, pos_idx]
                z = float(alpha_t.sum())
                if z <= 0.0:
                    alpha_t.fill(1.0 / float(k * k))
                else:
                    alpha_t /= z
                alpha[pos_idx] = alpha_t

            beta[-1].fill(1.0 / float(k * k))
            for pos_idx in range(n_positions - 2, -1, -1):
                sw = float(switch[sample_idx, pos_idx + 1])
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                next_term = beta[pos_idx + 1] * emission[sample_idx, pos_idx + 1]
                row_sum = next_term.sum(axis=1, keepdims=True)
                col_sum = next_term.sum(axis=0, keepdims=True)
                total = float(next_term.sum())
                beta_t = (b * b) * next_term + (b * off) * (row_sum + col_sum) + (off * off) * total
                z = float(beta_t.sum())
                if z <= 0.0:
                    beta_t.fill(1.0 / float(k * k))
                else:
                    beta_t /= z
                beta[pos_idx] = beta_t

            gamma_s = alpha * beta
            gamma_s /= np.clip(gamma_s.sum(axis=(1, 2), keepdims=True), 1e-12, None)
            gamma[sample_idx] = gamma_s

        return gamma

    def _forward_backward_numpy_haploid(self, log_emission: np.ndarray, switch: np.ndarray, k: int) -> np.ndarray:
        n_samples, n_positions = log_emission.shape[:2]
        emission = np.exp(log_emission - np.max(log_emission, axis=2, keepdims=True)).astype(np.float32)
        gamma = np.zeros_like(emission, dtype=np.float32)
        for sample_idx in range(n_samples):
            alpha = np.zeros((n_positions, k), dtype=np.float32)
            beta = np.zeros((n_positions, k), dtype=np.float32)
            alpha0 = emission[sample_idx, 0].copy()
            alpha0_sum = float(alpha0.sum())
            if alpha0_sum <= 0.0:
                alpha0.fill(1.0 / float(k))
            else:
                alpha0 /= alpha0_sum
            alpha[0] = alpha0
            for pos_idx in range(1, n_positions):
                sw = float(switch[sample_idx, pos_idx])
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                pred = b * alpha[pos_idx - 1] + off * float(alpha[pos_idx - 1].sum())
                alpha_t = pred * emission[sample_idx, pos_idx]
                z = float(alpha_t.sum())
                if z <= 0.0:
                    alpha_t.fill(1.0 / float(k))
                else:
                    alpha_t /= z
                alpha[pos_idx] = alpha_t
            beta[-1].fill(1.0 / float(k))
            for pos_idx in range(n_positions - 2, -1, -1):
                sw = float(switch[sample_idx, pos_idx + 1])
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                next_term = beta[pos_idx + 1] * emission[sample_idx, pos_idx + 1]
                beta_t = b * next_term + off * float(next_term.sum())
                z = float(beta_t.sum())
                if z <= 0.0:
                    beta_t.fill(1.0 / float(k))
                else:
                    beta_t /= z
                beta[pos_idx] = beta_t
            gamma_s = alpha * beta
            gamma_s /= np.clip(gamma_s.sum(axis=1, keepdims=True), 1e-12, None)
            gamma[sample_idx] = gamma_s
        return gamma

    def _polyploid_emissions(
        self,
        founder_alt_prob: np.ndarray,
        state_counts: np.ndarray,
        ploidy: int,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray | None,
        ref_weight: np.ndarray | None,
        alt_weight: np.ndarray | None,
        other_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        founder_alt = founder_alt_prob.astype(np.float32, copy=False)
        if self.config.use_quality_weights and ref_weight is not None and alt_weight is not None:
            ref_obs = ref_weight.astype(np.float32, copy=False)
            alt_obs = alt_weight.astype(np.float32, copy=False)
            other_obs = np.zeros_like(ref_obs, dtype=np.float32) if other_weight is None else other_weight.astype(np.float32, copy=False)
        else:
            ref_obs = ref_count.astype(np.float32, copy=False)
            alt_obs = alt_count.astype(np.float32, copy=False)
            other_obs = np.zeros_like(ref_obs, dtype=np.float32) if other_count is None else other_count.astype(np.float32, copy=False)

        state_alt_fraction = (state_counts.astype(np.float32, copy=False) @ founder_alt) / float(ploidy)
        eps = float(self.config.sequencing_error_rate)
        p_err_other = eps / 3.0
        p_alt = np.clip(
            state_alt_fraction * (1.0 - eps) + (1.0 - state_alt_fraction) * p_err_other,
            self.config.min_emission_prob,
            1.0,
        ).T
        p_ref = np.clip(
            (1.0 - state_alt_fraction) * (1.0 - eps) + state_alt_fraction * p_err_other,
            self.config.min_emission_prob,
            1.0,
        ).T
        p_oth = np.clip(np.full_like(p_alt, p_err_other), self.config.min_emission_prob, 1.0)
        log_emission = (
            alt_obs[:, :, None] * _safe_log_np(p_alt[None, :, :])
            + ref_obs[:, :, None] * _safe_log_np(p_ref[None, :, :])
            + other_obs[:, :, None] * _safe_log_np(p_oth[None, :, :])
        )
        return log_emission.astype(np.float32, copy=False), founder_alt

    def _polyploid_transition_matrix(
        self,
        switch_value: float,
        coeff: np.ndarray,
        *,
        k: int,
        ploidy: int,
    ) -> np.ndarray:
        if k <= 1:
            return np.ones(coeff.shape[:2], dtype=np.float32)
        stay = 1.0 - float(switch_value)
        off = float(switch_value) / float(k - 1)
        powers = np.asarray([stay**m * off ** (ploidy - m) for m in range(ploidy + 1)], dtype=np.float32)
        return np.tensordot(coeff, powers, axes=([2], [0])).astype(np.float32, copy=False)

    def _forward_backward_numpy_polyploid(
        self,
        log_emission: np.ndarray,
        switch: np.ndarray,
        transition_coeff: np.ndarray,
        *,
        k: int,
        ploidy: int,
        state_prior: np.ndarray | None = None,
    ) -> np.ndarray:
        n_samples, n_positions, n_states = log_emission.shape
        emission = np.exp(log_emission - np.max(log_emission, axis=2, keepdims=True)).astype(np.float32)
        prior = (
            _polyploid_state_prior(k, ploidy)
            if state_prior is None
            else state_prior.astype(np.float32, copy=False)
        )
        prior = prior / np.clip(np.sum(prior), 1e-12, None)
        gamma = np.zeros_like(emission, dtype=np.float32)
        for sample_idx in range(n_samples):
            alpha = np.zeros((n_positions, n_states), dtype=np.float32)
            beta = np.zeros((n_positions, n_states), dtype=np.float32)
            alpha0 = emission[sample_idx, 0].copy() * prior
            alpha0_sum = float(alpha0.sum())
            if alpha0_sum <= 0.0:
                alpha0.fill(1.0 / float(n_states))
            else:
                alpha0 /= alpha0_sum
            alpha[0] = alpha0
            for pos_idx in range(1, n_positions):
                trans = self._polyploid_transition_matrix(
                    float(switch[sample_idx, pos_idx]),
                    transition_coeff,
                    k=k,
                    ploidy=ploidy,
                )
                pred = alpha[pos_idx - 1] @ trans
                alpha_t = pred * emission[sample_idx, pos_idx]
                z = float(alpha_t.sum())
                if z <= 0.0:
                    alpha_t.fill(1.0 / float(n_states))
                else:
                    alpha_t /= z
                alpha[pos_idx] = alpha_t

            beta[-1].fill(1.0 / float(n_states))
            for pos_idx in range(n_positions - 2, -1, -1):
                trans = self._polyploid_transition_matrix(
                    float(switch[sample_idx, pos_idx + 1]),
                    transition_coeff,
                    k=k,
                    ploidy=ploidy,
                )
                next_term = beta[pos_idx + 1] * emission[sample_idx, pos_idx + 1]
                beta_t = trans @ next_term
                z = float(beta_t.sum())
                if z <= 0.0:
                    beta_t.fill(1.0 / float(n_states))
                else:
                    beta_t /= z
                beta[pos_idx] = beta_t

            gamma_s = alpha * beta
            gamma_s /= np.clip(gamma_s.sum(axis=1, keepdims=True), 1e-12, None)
            gamma[sample_idx] = gamma_s
        return gamma

    def _forward_backward_torch_diploid(self, log_emission: np.ndarray, switch: np.ndarray, k: int) -> np.ndarray:
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is not available.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_samples, n_positions = log_emission.shape[:2]
        log_emission_t = torch.as_tensor(log_emission, dtype=torch.float32, device=device)
        switch_t = torch.as_tensor(switch, dtype=torch.float32, device=device)
        emission = torch.exp(log_emission_t - torch.amax(log_emission_t, dim=(2, 3), keepdim=True))
        out = np.empty((n_samples, n_positions, k, k), dtype=np.float32)

        for sample_idx in range(n_samples):
            alpha = torch.zeros((n_positions, k, k), dtype=torch.float32, device=device)
            beta = torch.zeros((n_positions, k, k), dtype=torch.float32, device=device)
            alpha0 = emission[sample_idx, 0].clone()
            alpha0_sum = torch.sum(alpha0)
            if float(alpha0_sum) <= 0.0:
                alpha0.fill_(1.0 / float(k * k))
            else:
                alpha0 = alpha0 / alpha0_sum
            alpha[0] = alpha0

            for pos_idx in range(1, n_positions):
                sw = switch_t[sample_idx, pos_idx]
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                prev = alpha[pos_idx - 1]
                row_sum = torch.sum(prev, dim=1, keepdim=True)
                col_sum = torch.sum(prev, dim=0, keepdim=True)
                total = torch.sum(prev)
                pred = (b * b) * prev + (b * off) * (row_sum + col_sum) + (off * off) * total
                alpha_t = pred * emission[sample_idx, pos_idx]
                z = torch.sum(alpha_t)
                if float(z) <= 0.0:
                    alpha_t.fill_(1.0 / float(k * k))
                else:
                    alpha_t = alpha_t / z
                alpha[pos_idx] = alpha_t

            beta[-1].fill_(1.0 / float(k * k))
            for pos_idx in range(n_positions - 2, -1, -1):
                sw = switch_t[sample_idx, pos_idx + 1]
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                next_term = beta[pos_idx + 1] * emission[sample_idx, pos_idx + 1]
                row_sum = torch.sum(next_term, dim=1, keepdim=True)
                col_sum = torch.sum(next_term, dim=0, keepdim=True)
                total = torch.sum(next_term)
                beta_t = (b * b) * next_term + (b * off) * (row_sum + col_sum) + (off * off) * total
                z = torch.sum(beta_t)
                if float(z) <= 0.0:
                    beta_t.fill_(1.0 / float(k * k))
                else:
                    beta_t = beta_t / z
                beta[pos_idx] = beta_t

            gamma_s = alpha * beta
            gamma_s = gamma_s / torch.clamp(torch.sum(gamma_s, dim=(1, 2), keepdim=True), min=1e-12)
            out[sample_idx] = gamma_s.detach().cpu().numpy()
        return out

    def _forward_backward_torch_haploid(self, log_emission: np.ndarray, switch: np.ndarray, k: int) -> np.ndarray:
        if torch is None:
            raise RuntimeError("Torch backend requested but torch is not available.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_samples, n_positions = log_emission.shape[:2]
        log_emission_t = torch.as_tensor(log_emission, dtype=torch.float32, device=device)
        switch_t = torch.as_tensor(switch, dtype=torch.float32, device=device)
        emission = torch.exp(log_emission_t - torch.amax(log_emission_t, dim=2, keepdim=True))
        out = np.empty((n_samples, n_positions, k), dtype=np.float32)

        for sample_idx in range(n_samples):
            alpha = torch.zeros((n_positions, k), dtype=torch.float32, device=device)
            beta = torch.zeros((n_positions, k), dtype=torch.float32, device=device)
            alpha0 = emission[sample_idx, 0].clone()
            alpha0_sum = torch.sum(alpha0)
            if float(alpha0_sum) <= 0.0:
                alpha0.fill_(1.0 / float(k))
            else:
                alpha0 = alpha0 / alpha0_sum
            alpha[0] = alpha0
            for pos_idx in range(1, n_positions):
                sw = switch_t[sample_idx, pos_idx]
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                pred = b * alpha[pos_idx - 1] + off * torch.sum(alpha[pos_idx - 1])
                alpha_t = pred * emission[sample_idx, pos_idx]
                z = torch.sum(alpha_t)
                if float(z) <= 0.0:
                    alpha_t.fill_(1.0 / float(k))
                else:
                    alpha_t = alpha_t / z
                alpha[pos_idx] = alpha_t
            beta[-1].fill_(1.0 / float(k))
            for pos_idx in range(n_positions - 2, -1, -1):
                sw = switch_t[sample_idx, pos_idx + 1]
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                next_term = beta[pos_idx + 1] * emission[sample_idx, pos_idx + 1]
                beta_t = b * next_term + off * torch.sum(next_term)
                z = torch.sum(beta_t)
                if float(z) <= 0.0:
                    beta_t.fill_(1.0 / float(k))
                else:
                    beta_t = beta_t / z
                beta[pos_idx] = beta_t
            gamma_s = alpha * beta
            gamma_s = gamma_s / torch.clamp(torch.sum(gamma_s, dim=1, keepdim=True), min=1e-12)
            out[sample_idx] = gamma_s.detach().cpu().numpy()
        return out

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_model(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        off_denom = float(max(k - 1, 1))
        inv_kk = 1.0 / float(k * k)

        def run_one_sample(log_emission_s: jnp.ndarray, switch_s: jnp.ndarray) -> jnp.ndarray:
            emission = jnp.exp(log_emission_s - jnp.max(log_emission_s, axis=(1, 2), keepdims=True))
            alpha0 = emission[0] / (jnp.sum(emission[0]) + 1e-12)

            def fwd_step(carry: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
                sw, emission_t = inputs
                off = sw / off_denom
                b = 1.0 - sw - off
                row_sum = jnp.sum(carry, axis=1, keepdims=True)
                col_sum = jnp.sum(carry, axis=0, keepdims=True)
                total = jnp.sum(carry)
                pred = (b * b) * carry + (b * off) * (row_sum + col_sum) + (off * off) * total
                alpha_t = pred * emission_t
                alpha_t = alpha_t / (jnp.sum(alpha_t) + 1e-12)
                return alpha_t, alpha_t

            _, alpha_rest = jax.lax.scan(fwd_step, alpha0, (switch_s[1:], emission[1:]))
            alpha = jnp.concatenate([alpha0[None, :, :], alpha_rest], axis=0)

            betaT = jnp.full((k, k), inv_kk, dtype=jnp.float32)

            def bwd_step(carry: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
                sw_next, emission_next = inputs
                off = sw_next / off_denom
                b = 1.0 - sw_next - off
                next_term = carry * emission_next
                row_sum = jnp.sum(next_term, axis=1, keepdims=True)
                col_sum = jnp.sum(next_term, axis=0, keepdims=True)
                total = jnp.sum(next_term)
                beta_t = (b * b) * next_term + (b * off) * (row_sum + col_sum) + (off * off) * total
                beta_t = beta_t / (jnp.sum(beta_t) + 1e-12)
                return beta_t, beta_t

            _, beta_rev = jax.lax.scan(bwd_step, betaT, (switch_s[1:][::-1], emission[1:][::-1]))
            beta = jnp.concatenate([beta_rev[::-1], betaT[None, :, :]], axis=0)
            gamma = alpha * beta
            gamma = gamma / (jnp.sum(gamma, axis=(1, 2), keepdims=True) + 1e-12)
            return gamma

        return jax.vmap(run_one_sample)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        return jax.jit(JAXStitchHMM._jax_diploid_model(k))

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_stats_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(log_emission: jnp.ndarray, switch: jnp.ndarray, sample_alt_fraction: jnp.ndarray):
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            numerator = jnp.einsum("spk,sp->kp", hap_gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", hap_gamma)
            return numerator, denominator

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_final_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            log_emission: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
        ):
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            numerator = jnp.einsum("spk,sp->kp", hap_gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", hap_gamma)
            dosage = 2.0 * jnp.einsum("spk,kp->sp", hap_gamma, founder_alt)
            founder_alt_t = founder_alt.T
            one_minus = 1.0 - founder_alt_t
            gp0 = jnp.einsum("spij,pi,pj->sp", gamma, one_minus, one_minus)
            gp2 = jnp.einsum("spij,pi,pj->sp", gamma, founder_alt_t, founder_alt_t)
            gp1 = jnp.clip(1.0 - gp0 - gp2, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1, gp2], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return numerator, denominator, hap_gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_output_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            log_emission: jnp.ndarray,
            switch: jnp.ndarray,
            founder_alt: jnp.ndarray,
        ):
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            dosage = 2.0 * jnp.einsum("spk,kp->sp", hap_gamma, founder_alt)
            founder_alt_t = founder_alt.T
            one_minus = 1.0 - founder_alt_t
            gp0 = jnp.einsum("spij,pi,pj->sp", gamma, one_minus, one_minus)
            gp2 = jnp.einsum("spij,pi,pj->sp", gamma, founder_alt_t, founder_alt_t)
            gp1 = jnp.clip(1.0 - gp0 - gp2, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1, gp2], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return hap_gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    def _jax_diploid_count_log_emission(
        ref_obs: jnp.ndarray,
        alt_obs: jnp.ndarray,
        other_obs: jnp.ndarray,
        founder_alt: jnp.ndarray,
        sequencing_error_rate: jnp.ndarray,
        min_emission_prob: jnp.ndarray,
    ) -> jnp.ndarray:
        founder_alt_t = founder_alt.T
        dosage = founder_alt_t[:, :, None] + founder_alt_t[:, None, :]
        hap_alt = jnp.clip(0.5 * dosage, 0.0, 1.0)
        eps = sequencing_error_rate
        p_err_other = eps / 3.0
        p_alt = jnp.clip(hap_alt * (1.0 - eps) + (1.0 - hap_alt) * p_err_other, min_emission_prob, 1.0)
        p_ref = jnp.clip((1.0 - hap_alt) * (1.0 - eps) + hap_alt * p_err_other, min_emission_prob, 1.0)
        p_oth = jnp.clip(jnp.full_like(p_alt, p_err_other), min_emission_prob, 1.0)
        return (
            alt_obs[:, :, None, None] * jnp.log(p_alt[None, :, :, :])
            + ref_obs[:, :, None, None] * jnp.log(p_ref[None, :, :, :])
            + other_obs[:, :, None, None] * jnp.log(p_oth[None, :, :, :])
        )

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_count_stats_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_diploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            numerator = jnp.einsum("spk,sp->kp", hap_gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", hap_gamma)
            return numerator, denominator

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_count_final_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_diploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            numerator = jnp.einsum("spk,sp->kp", hap_gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", hap_gamma)
            dosage = 2.0 * jnp.einsum("spk,kp->sp", hap_gamma, founder_alt)
            founder_alt_t = founder_alt.T
            one_minus = 1.0 - founder_alt_t
            gp0 = jnp.einsum("spij,pi,pj->sp", gamma, one_minus, one_minus)
            gp2 = jnp.einsum("spij,pi,pj->sp", gamma, founder_alt_t, founder_alt_t)
            gp1 = jnp.clip(1.0 - gp0 - gp2, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1, gp2], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return numerator, denominator, hap_gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_count_output_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            founder_alt: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_diploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            dosage = 2.0 * jnp.einsum("spk,kp->sp", hap_gamma, founder_alt)
            founder_alt_t = founder_alt.T
            one_minus = 1.0 - founder_alt_t
            gp0 = jnp.einsum("spij,pi,pj->sp", gamma, one_minus, one_minus)
            gp2 = jnp.einsum("spij,pi,pj->sp", gamma, founder_alt_t, founder_alt_t)
            gp1 = jnp.clip(1.0 - gp0 - gp2, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1, gp2], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return hap_gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    def _jax_fragment_hap_probs(
        founder_alt: jnp.ndarray,
        fragment_obs_fragment_idx: jnp.ndarray,
        fragment_obs_pos_idx: jnp.ndarray,
        fragment_obs_code: jnp.ndarray,
        fragment_obs_qual: jnp.ndarray,
        n_fragments: int,
        sequencing_error_rate: jnp.ndarray,
        min_emission_prob: jnp.ndarray,
        fragment_max_difference_between_reads: jnp.ndarray,
        fragment_rescale_read_likelihood: jnp.ndarray,
    ) -> jnp.ndarray:
        founder_at_obs = founder_alt.T[fragment_obs_pos_idx]
        qual = fragment_obs_qual.astype(jnp.float32)
        err_from_qual = jnp.power(10.0, -(qual / 10.0))
        err = jnp.where(qual >= 0.0, err_from_qual, sequencing_error_rate)
        p_err_other = err / 3.0
        p_alt = jnp.clip(
            founder_at_obs * (1.0 - err[:, None]) + (1.0 - founder_at_obs) * p_err_other[:, None],
            min_emission_prob,
            1.0,
        )
        p_ref = jnp.clip(
            (1.0 - founder_at_obs) * (1.0 - err[:, None]) + founder_at_obs * p_err_other[:, None],
            min_emission_prob,
            1.0,
        )
        p_oth = jnp.clip(jnp.broadcast_to(p_err_other[:, None], p_alt.shape), min_emission_prob, 1.0)
        log_p = jnp.where(
            fragment_obs_code[:, None] == 1,
            jnp.log(p_alt),
            jnp.where(fragment_obs_code[:, None] == 0, jnp.log(p_ref), jnp.log(p_oth)),
        )
        frag_log = jax.ops.segment_sum(
            log_p,
            fragment_obs_fragment_idx.astype(jnp.int32),
            num_segments=int(n_fragments),
        )
        scaled_log = frag_log - jnp.max(frag_log, axis=1, keepdims=True)
        min_log = -jnp.log(jnp.maximum(fragment_max_difference_between_reads, 1.000001))
        scaled_log = jnp.maximum(scaled_log, min_log)
        read_prob_hap = jax.lax.cond(
            fragment_rescale_read_likelihood.astype(jnp.bool_),
            lambda _: jnp.exp(scaled_log),
            lambda _: jnp.exp(frag_log),
            operand=None,
        )
        return read_prob_hap.astype(jnp.float32)

    @staticmethod
    def _jax_apply_diploid_fragment_likelihoods(
        log_emission: jnp.ndarray,
        founder_alt: jnp.ndarray,
        fragment_sample_idx: jnp.ndarray,
        fragment_center_idx: jnp.ndarray,
        fragment_obs_fragment_idx: jnp.ndarray,
        fragment_obs_pos_idx: jnp.ndarray,
        fragment_obs_code: jnp.ndarray,
        fragment_obs_qual: jnp.ndarray,
        sequencing_error_rate: jnp.ndarray,
        min_emission_prob: jnp.ndarray,
        fragment_max_difference_between_reads: jnp.ndarray,
        fragment_max_emission_matrix_difference: jnp.ndarray,
        fragment_rescale_read_likelihood: jnp.ndarray,
        fragment_replace_mode: jnp.ndarray,
    ) -> jnp.ndarray:
        n_samples = int(log_emission.shape[0])
        n_positions = int(log_emission.shape[1])
        n_fragments = int(fragment_center_idx.shape[0])
        read_prob_hap = JAXStitchHMM._jax_fragment_hap_probs(
            founder_alt,
            fragment_obs_fragment_idx,
            fragment_obs_pos_idx,
            fragment_obs_code,
            fragment_obs_qual,
            n_fragments,
            sequencing_error_rate,
            min_emission_prob,
            fragment_max_difference_between_reads,
            fragment_rescale_read_likelihood,
        )
        pair_prob = 0.5 * (read_prob_hap[:, :, None] + read_prob_hap[:, None, :])
        frag_ll = jnp.log(jnp.clip(pair_prob, min_emission_prob, 1.0)).astype(jnp.float32)
        flat_key = fragment_sample_idx.astype(jnp.int32) * n_positions + fragment_center_idx.astype(jnp.int32)
        n_flat = int(n_samples * n_positions)
        flat_delta = jax.ops.segment_sum(frag_ll, flat_key, num_segments=n_flat)
        delta = flat_delta.reshape((n_samples, n_positions, int(log_emission.shape[2]), int(log_emission.shape[3])))
        touched = (
            jax.ops.segment_sum(
                jnp.ones((n_fragments,), dtype=jnp.int32),
                flat_key,
                num_segments=n_flat,
            ).reshape((n_samples, n_positions))
            > 0
        )
        replace = fragment_replace_mode.astype(jnp.bool_)
        out = jnp.where(touched[:, :, None, None] & replace, delta, log_emission + delta)
        maxed = out - jnp.max(out, axis=(2, 3), keepdims=True)
        min_state_log = -jnp.log(jnp.maximum(fragment_max_emission_matrix_difference, 1.000001))
        maxed = jnp.maximum(maxed, min_state_log)
        return jnp.where(
            touched[:, :, None, None] & fragment_rescale_read_likelihood.astype(jnp.bool_),
            maxed,
            out,
        ).astype(jnp.float32)

    @staticmethod
    def _jax_apply_haploid_fragment_likelihoods(
        log_emission: jnp.ndarray,
        founder_alt: jnp.ndarray,
        fragment_sample_idx: jnp.ndarray,
        fragment_center_idx: jnp.ndarray,
        fragment_obs_fragment_idx: jnp.ndarray,
        fragment_obs_pos_idx: jnp.ndarray,
        fragment_obs_code: jnp.ndarray,
        fragment_obs_qual: jnp.ndarray,
        sequencing_error_rate: jnp.ndarray,
        min_emission_prob: jnp.ndarray,
        fragment_max_difference_between_reads: jnp.ndarray,
        fragment_max_emission_matrix_difference: jnp.ndarray,
        fragment_rescale_read_likelihood: jnp.ndarray,
        fragment_replace_mode: jnp.ndarray,
    ) -> jnp.ndarray:
        n_samples = int(log_emission.shape[0])
        n_positions = int(log_emission.shape[1])
        n_fragments = int(fragment_center_idx.shape[0])
        read_prob_hap = JAXStitchHMM._jax_fragment_hap_probs(
            founder_alt,
            fragment_obs_fragment_idx,
            fragment_obs_pos_idx,
            fragment_obs_code,
            fragment_obs_qual,
            n_fragments,
            sequencing_error_rate,
            min_emission_prob,
            fragment_max_difference_between_reads,
            fragment_rescale_read_likelihood,
        )
        frag_ll = jnp.log(jnp.clip(read_prob_hap, min_emission_prob, 1.0)).astype(jnp.float32)
        flat_key = fragment_sample_idx.astype(jnp.int32) * n_positions + fragment_center_idx.astype(jnp.int32)
        n_flat = int(n_samples * n_positions)
        flat_delta = jax.ops.segment_sum(frag_ll, flat_key, num_segments=n_flat)
        delta = flat_delta.reshape((n_samples, n_positions, int(log_emission.shape[2])))
        touched = (
            jax.ops.segment_sum(
                jnp.ones((n_fragments,), dtype=jnp.int32),
                flat_key,
                num_segments=n_flat,
            ).reshape((n_samples, n_positions))
            > 0
        )
        replace = fragment_replace_mode.astype(jnp.bool_)
        out = jnp.where(touched[:, :, None] & replace, delta, log_emission + delta)
        maxed = out - jnp.max(out, axis=2, keepdims=True)
        min_state_log = -jnp.log(jnp.maximum(fragment_max_emission_matrix_difference, 1.000001))
        maxed = jnp.maximum(maxed, min_state_log)
        return jnp.where(
            touched[:, :, None] & fragment_rescale_read_likelihood.astype(jnp.bool_),
            maxed,
            out,
        ).astype(jnp.float32)

    @staticmethod
    def _jax_apply_polyploid_fragment_likelihoods(
        log_emission: jnp.ndarray,
        founder_alt: jnp.ndarray,
        state_counts: jnp.ndarray,
        ploidy: jnp.ndarray,
        fragment_sample_idx: jnp.ndarray,
        fragment_center_idx: jnp.ndarray,
        fragment_obs_fragment_idx: jnp.ndarray,
        fragment_obs_pos_idx: jnp.ndarray,
        fragment_obs_code: jnp.ndarray,
        fragment_obs_qual: jnp.ndarray,
        sequencing_error_rate: jnp.ndarray,
        min_emission_prob: jnp.ndarray,
        fragment_max_difference_between_reads: jnp.ndarray,
        fragment_max_emission_matrix_difference: jnp.ndarray,
        fragment_rescale_read_likelihood: jnp.ndarray,
        fragment_replace_mode: jnp.ndarray,
    ) -> jnp.ndarray:
        n_samples = int(log_emission.shape[0])
        n_positions = int(log_emission.shape[1])
        n_states = int(log_emission.shape[2])
        n_fragments = int(fragment_center_idx.shape[0])
        read_prob_hap = JAXStitchHMM._jax_fragment_hap_probs(
            founder_alt,
            fragment_obs_fragment_idx,
            fragment_obs_pos_idx,
            fragment_obs_code,
            fragment_obs_qual,
            n_fragments,
            sequencing_error_rate,
            min_emission_prob,
            fragment_max_difference_between_reads,
            fragment_rescale_read_likelihood,
        )
        state_fraction = state_counts.astype(jnp.float32) / jnp.maximum(ploidy.astype(jnp.float32), 1.0)
        state_read_prob = read_prob_hap @ state_fraction.T
        frag_ll = jnp.log(jnp.clip(state_read_prob, min_emission_prob, 1.0)).astype(jnp.float32)
        flat_key = fragment_sample_idx.astype(jnp.int32) * n_positions + fragment_center_idx.astype(jnp.int32)
        n_flat = int(n_samples * n_positions)
        flat_delta = jax.ops.segment_sum(frag_ll, flat_key, num_segments=n_flat)
        delta = flat_delta.reshape((n_samples, n_positions, n_states))
        touched = (
            jax.ops.segment_sum(
                jnp.ones((n_fragments,), dtype=jnp.int32),
                flat_key,
                num_segments=n_flat,
            ).reshape((n_samples, n_positions))
            > 0
        )
        replace = fragment_replace_mode.astype(jnp.bool_)
        out = jnp.where(touched[:, :, None] & replace, delta, log_emission + delta)
        maxed = out - jnp.max(out, axis=2, keepdims=True)
        min_state_log = -jnp.log(jnp.maximum(fragment_max_emission_matrix_difference, 1.000001))
        maxed = jnp.maximum(maxed, min_state_log)
        return jnp.where(
            touched[:, :, None] & fragment_rescale_read_likelihood.astype(jnp.bool_),
            maxed,
            out,
        ).astype(jnp.float32)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_fragment_count_stats_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            fragment_sample_idx: jnp.ndarray,
            fragment_center_idx: jnp.ndarray,
            fragment_obs_fragment_idx: jnp.ndarray,
            fragment_obs_pos_idx: jnp.ndarray,
            fragment_obs_code: jnp.ndarray,
            fragment_obs_qual: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
            fragment_max_difference_between_reads: jnp.ndarray,
            fragment_max_emission_matrix_difference: jnp.ndarray,
            fragment_rescale_read_likelihood: jnp.ndarray,
            fragment_replace_mode: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_diploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            log_emission = JAXStitchHMM._jax_apply_diploid_fragment_likelihoods(
                log_emission,
                founder_alt,
                fragment_sample_idx,
                fragment_center_idx,
                fragment_obs_fragment_idx,
                fragment_obs_pos_idx,
                fragment_obs_code,
                fragment_obs_qual,
                sequencing_error_rate,
                min_emission_prob,
                fragment_max_difference_between_reads,
                fragment_max_emission_matrix_difference,
                fragment_rescale_read_likelihood,
                fragment_replace_mode,
            )
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            numerator = jnp.einsum("spk,sp->kp", hap_gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", hap_gamma)
            return numerator, denominator

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_fragment_count_final_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            fragment_sample_idx: jnp.ndarray,
            fragment_center_idx: jnp.ndarray,
            fragment_obs_fragment_idx: jnp.ndarray,
            fragment_obs_pos_idx: jnp.ndarray,
            fragment_obs_code: jnp.ndarray,
            fragment_obs_qual: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
            fragment_max_difference_between_reads: jnp.ndarray,
            fragment_max_emission_matrix_difference: jnp.ndarray,
            fragment_rescale_read_likelihood: jnp.ndarray,
            fragment_replace_mode: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_diploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            log_emission = JAXStitchHMM._jax_apply_diploid_fragment_likelihoods(
                log_emission,
                founder_alt,
                fragment_sample_idx,
                fragment_center_idx,
                fragment_obs_fragment_idx,
                fragment_obs_pos_idx,
                fragment_obs_code,
                fragment_obs_qual,
                sequencing_error_rate,
                min_emission_prob,
                fragment_max_difference_between_reads,
                fragment_max_emission_matrix_difference,
                fragment_rescale_read_likelihood,
                fragment_replace_mode,
            )
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            numerator = jnp.einsum("spk,sp->kp", hap_gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", hap_gamma)
            dosage = 2.0 * jnp.einsum("spk,kp->sp", hap_gamma, founder_alt)
            founder_alt_t = founder_alt.T
            one_minus = 1.0 - founder_alt_t
            gp0 = jnp.einsum("spij,pi,pj->sp", gamma, one_minus, one_minus)
            gp2 = jnp.einsum("spij,pi,pj->sp", gamma, founder_alt_t, founder_alt_t)
            gp1 = jnp.clip(1.0 - gp0 - gp2, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1, gp2], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return numerator, denominator, hap_gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_diploid_fragment_count_output_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_diploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            founder_alt: jnp.ndarray,
            fragment_sample_idx: jnp.ndarray,
            fragment_center_idx: jnp.ndarray,
            fragment_obs_fragment_idx: jnp.ndarray,
            fragment_obs_pos_idx: jnp.ndarray,
            fragment_obs_code: jnp.ndarray,
            fragment_obs_qual: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
            fragment_max_difference_between_reads: jnp.ndarray,
            fragment_max_emission_matrix_difference: jnp.ndarray,
            fragment_rescale_read_likelihood: jnp.ndarray,
            fragment_replace_mode: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_diploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            log_emission = JAXStitchHMM._jax_apply_diploid_fragment_likelihoods(
                log_emission,
                founder_alt,
                fragment_sample_idx,
                fragment_center_idx,
                fragment_obs_fragment_idx,
                fragment_obs_pos_idx,
                fragment_obs_code,
                fragment_obs_qual,
                sequencing_error_rate,
                min_emission_prob,
                fragment_max_difference_between_reads,
                fragment_max_emission_matrix_difference,
                fragment_rescale_read_likelihood,
                fragment_replace_mode,
            )
            gamma = posterior_model(log_emission, switch)
            hap_gamma = 0.5 * (jnp.sum(gamma, axis=3) + jnp.sum(gamma, axis=2))
            dosage = 2.0 * jnp.einsum("spk,kp->sp", hap_gamma, founder_alt)
            founder_alt_t = founder_alt.T
            one_minus = 1.0 - founder_alt_t
            gp0 = jnp.einsum("spij,pi,pj->sp", gamma, one_minus, one_minus)
            gp2 = jnp.einsum("spij,pi,pj->sp", gamma, founder_alt_t, founder_alt_t)
            gp1 = jnp.clip(1.0 - gp0 - gp2, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1, gp2], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return hap_gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_haploid_model(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        off_denom = float(max(k - 1, 1))
        inv_k = 1.0 / float(k)

        def run_one_sample(log_emission_s: jnp.ndarray, switch_s: jnp.ndarray) -> jnp.ndarray:
            emission = jnp.exp(log_emission_s - jnp.max(log_emission_s, axis=1, keepdims=True))
            alpha0 = emission[0] / (jnp.sum(emission[0]) + 1e-12)

            def fwd_step(carry: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
                sw, emission_t = inputs
                off = sw / off_denom
                b = 1.0 - sw - off
                pred = b * carry + off * jnp.sum(carry)
                alpha_t = pred * emission_t
                alpha_t = alpha_t / (jnp.sum(alpha_t) + 1e-12)
                return alpha_t, alpha_t

            _, alpha_rest = jax.lax.scan(fwd_step, alpha0, (switch_s[1:], emission[1:]))
            alpha = jnp.concatenate([alpha0[None, :], alpha_rest], axis=0)
            betaT = jnp.full((k,), inv_k, dtype=jnp.float32)

            def bwd_step(carry: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
                sw_next, emission_next = inputs
                off = sw_next / off_denom
                b = 1.0 - sw_next - off
                next_term = carry * emission_next
                beta_t = b * next_term + off * jnp.sum(next_term)
                beta_t = beta_t / (jnp.sum(beta_t) + 1e-12)
                return beta_t, beta_t

            _, beta_rev = jax.lax.scan(bwd_step, betaT, (switch_s[1:][::-1], emission[1:][::-1]))
            beta = jnp.concatenate([beta_rev[::-1], betaT[None, :]], axis=0)
            gamma = alpha * beta
            gamma = gamma / (jnp.sum(gamma, axis=1, keepdims=True) + 1e-12)
            return gamma

        return jax.vmap(run_one_sample)

    @staticmethod
    def _jax_haploid_count_log_emission(
        ref_obs: jnp.ndarray,
        alt_obs: jnp.ndarray,
        other_obs: jnp.ndarray,
        founder_alt: jnp.ndarray,
        sequencing_error_rate: jnp.ndarray,
        min_emission_prob: jnp.ndarray,
    ) -> jnp.ndarray:
        founder_alt_t = founder_alt.T
        eps = sequencing_error_rate
        p_err_other = eps / 3.0
        p_alt = jnp.clip(founder_alt_t * (1.0 - eps) + (1.0 - founder_alt_t) * p_err_other, min_emission_prob, 1.0)
        p_ref = jnp.clip((1.0 - founder_alt_t) * (1.0 - eps) + founder_alt_t * p_err_other, min_emission_prob, 1.0)
        p_oth = jnp.clip(jnp.full_like(p_alt, p_err_other), min_emission_prob, 1.0)
        return (
            alt_obs[:, :, None] * jnp.log(p_alt[None, :, :])
            + ref_obs[:, :, None] * jnp.log(p_ref[None, :, :])
            + other_obs[:, :, None] * jnp.log(p_oth[None, :, :])
        )

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_haploid_count_stats_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_haploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_haploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            gamma = posterior_model(log_emission, switch)
            numerator = jnp.einsum("spk,sp->kp", gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", gamma)
            return numerator, denominator

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_haploid_count_final_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_haploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_haploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            gamma = posterior_model(log_emission, switch)
            numerator = jnp.einsum("spk,sp->kp", gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", gamma)
            dosage = jnp.einsum("spk,kp->sp", gamma, founder_alt)
            gp0 = 1.0 - jnp.clip(dosage, 0.0, 1.0)
            gp1 = jnp.clip(dosage, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return numerator, denominator, gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_haploid_count_output_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_haploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            founder_alt: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_haploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            gamma = posterior_model(log_emission, switch)
            dosage = jnp.einsum("spk,kp->sp", gamma, founder_alt)
            gp0 = 1.0 - jnp.clip(dosage, 0.0, 1.0)
            gp1 = jnp.clip(dosage, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_haploid_fragment_count_stats_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_haploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            fragment_sample_idx: jnp.ndarray,
            fragment_center_idx: jnp.ndarray,
            fragment_obs_fragment_idx: jnp.ndarray,
            fragment_obs_pos_idx: jnp.ndarray,
            fragment_obs_code: jnp.ndarray,
            fragment_obs_qual: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
            fragment_max_difference_between_reads: jnp.ndarray,
            fragment_max_emission_matrix_difference: jnp.ndarray,
            fragment_rescale_read_likelihood: jnp.ndarray,
            fragment_replace_mode: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_haploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            log_emission = JAXStitchHMM._jax_apply_haploid_fragment_likelihoods(
                log_emission,
                founder_alt,
                fragment_sample_idx,
                fragment_center_idx,
                fragment_obs_fragment_idx,
                fragment_obs_pos_idx,
                fragment_obs_code,
                fragment_obs_qual,
                sequencing_error_rate,
                min_emission_prob,
                fragment_max_difference_between_reads,
                fragment_max_emission_matrix_difference,
                fragment_rescale_read_likelihood,
                fragment_replace_mode,
            )
            gamma = posterior_model(log_emission, switch)
            numerator = jnp.einsum("spk,sp->kp", gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", gamma)
            return numerator, denominator

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_haploid_fragment_count_final_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_haploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
            fragment_sample_idx: jnp.ndarray,
            fragment_center_idx: jnp.ndarray,
            fragment_obs_fragment_idx: jnp.ndarray,
            fragment_obs_pos_idx: jnp.ndarray,
            fragment_obs_code: jnp.ndarray,
            fragment_obs_qual: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
            fragment_max_difference_between_reads: jnp.ndarray,
            fragment_max_emission_matrix_difference: jnp.ndarray,
            fragment_rescale_read_likelihood: jnp.ndarray,
            fragment_replace_mode: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_haploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            log_emission = JAXStitchHMM._jax_apply_haploid_fragment_likelihoods(
                log_emission,
                founder_alt,
                fragment_sample_idx,
                fragment_center_idx,
                fragment_obs_fragment_idx,
                fragment_obs_pos_idx,
                fragment_obs_code,
                fragment_obs_qual,
                sequencing_error_rate,
                min_emission_prob,
                fragment_max_difference_between_reads,
                fragment_max_emission_matrix_difference,
                fragment_rescale_read_likelihood,
                fragment_replace_mode,
            )
            gamma = posterior_model(log_emission, switch)
            numerator = jnp.einsum("spk,sp->kp", gamma, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", gamma)
            dosage = jnp.einsum("spk,kp->sp", gamma, founder_alt)
            gp0 = 1.0 - jnp.clip(dosage, 0.0, 1.0)
            gp1 = jnp.clip(dosage, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return numerator, denominator, gamma, dosage, gp

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_haploid_fragment_count_output_kernel(k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_haploid_model(k)

        def model(
            ref_obs: jnp.ndarray,
            alt_obs: jnp.ndarray,
            other_obs: jnp.ndarray,
            switch: jnp.ndarray,
            founder_alt: jnp.ndarray,
            fragment_sample_idx: jnp.ndarray,
            fragment_center_idx: jnp.ndarray,
            fragment_obs_fragment_idx: jnp.ndarray,
            fragment_obs_pos_idx: jnp.ndarray,
            fragment_obs_code: jnp.ndarray,
            fragment_obs_qual: jnp.ndarray,
            sequencing_error_rate: jnp.ndarray,
            min_emission_prob: jnp.ndarray,
            fragment_max_difference_between_reads: jnp.ndarray,
            fragment_max_emission_matrix_difference: jnp.ndarray,
            fragment_rescale_read_likelihood: jnp.ndarray,
            fragment_replace_mode: jnp.ndarray,
        ):
            log_emission = JAXStitchHMM._jax_haploid_count_log_emission(
                ref_obs,
                alt_obs,
                other_obs,
                founder_alt,
                sequencing_error_rate,
                min_emission_prob,
            )
            log_emission = JAXStitchHMM._jax_apply_haploid_fragment_likelihoods(
                log_emission,
                founder_alt,
                fragment_sample_idx,
                fragment_center_idx,
                fragment_obs_fragment_idx,
                fragment_obs_pos_idx,
                fragment_obs_code,
                fragment_obs_qual,
                sequencing_error_rate,
                min_emission_prob,
                fragment_max_difference_between_reads,
                fragment_max_emission_matrix_difference,
                fragment_rescale_read_likelihood,
                fragment_replace_mode,
            )
            gamma = posterior_model(log_emission, switch)
            dosage = jnp.einsum("spk,kp->sp", gamma, founder_alt)
            gp0 = 1.0 - jnp.clip(dosage, 0.0, 1.0)
            gp1 = jnp.clip(dosage, 0.0, 1.0)
            gp = jnp.stack([gp0, gp1], axis=2)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return gamma, dosage, gp

        return jax.jit(model)

    def _get_jax_diploid_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_diploid_kernel(k)
        key = (n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_diploid_model(k)
        dummy_emission = jnp.zeros((n_samples, n_positions, k, k), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        compiled = jax.jit(model).lower(dummy_emission, dummy_switch).compile()
        warm = compiled(dummy_emission, dummy_switch)
        warm.block_until_ready()
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_diploid_stats_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_diploid_stats_kernel(k)
        key = ("stats", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_diploid_stats_kernel(k)
        dummy_emission = jnp.zeros((n_samples, n_positions, k, k), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_alt_fraction = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        compiled = model.lower(dummy_emission, dummy_switch, dummy_alt_fraction).compile()
        warm = compiled(dummy_emission, dummy_switch, dummy_alt_fraction)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_diploid_final_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_diploid_final_kernel(k)
        key = ("final", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_diploid_final_kernel(k)
        dummy_emission = jnp.zeros((n_samples, n_positions, k, k), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_alt_fraction = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        compiled = model.lower(dummy_emission, dummy_switch, dummy_alt_fraction, dummy_founder).compile()
        warm = compiled(dummy_emission, dummy_switch, dummy_alt_fraction, dummy_founder)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_diploid_output_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_diploid_output_kernel(k)
        key = ("output", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_diploid_output_kernel(k)
        dummy_emission = jnp.zeros((n_samples, n_positions, k, k), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        compiled = model.lower(dummy_emission, dummy_switch, dummy_founder).compile()
        warm = compiled(dummy_emission, dummy_switch, dummy_founder)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_diploid_count_stats_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_diploid_count_stats_kernel(k)
        key = ("count_stats", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_diploid_count_stats_kernel(k)
        dummy_obs = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        dummy_scalar = jnp.asarray(0.01, dtype=jnp.float32)
        compiled = model.lower(
            dummy_obs,
            dummy_obs,
            dummy_obs,
            dummy_switch,
            dummy_obs,
            dummy_founder,
            dummy_scalar,
            dummy_scalar,
        ).compile()
        warm = compiled(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_obs, dummy_founder, dummy_scalar, dummy_scalar)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_diploid_count_final_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_diploid_count_final_kernel(k)
        key = ("count_final", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_diploid_count_final_kernel(k)
        dummy_obs = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        dummy_scalar = jnp.asarray(0.01, dtype=jnp.float32)
        compiled = model.lower(
            dummy_obs,
            dummy_obs,
            dummy_obs,
            dummy_switch,
            dummy_obs,
            dummy_founder,
            dummy_scalar,
            dummy_scalar,
        ).compile()
        warm = compiled(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_obs, dummy_founder, dummy_scalar, dummy_scalar)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_diploid_count_output_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_diploid_count_output_kernel(k)
        key = ("count_output", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_diploid_count_output_kernel(k)
        dummy_obs = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        dummy_scalar = jnp.asarray(0.01, dtype=jnp.float32)
        compiled = model.lower(
            dummy_obs,
            dummy_obs,
            dummy_obs,
            dummy_switch,
            dummy_founder,
            dummy_scalar,
            dummy_scalar,
        ).compile()
        warm = compiled(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_founder, dummy_scalar, dummy_scalar)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _compile_fragment_count_callable(
        self,
        *,
        kind: str,
        ploidy_kind: str = "diploid",
        n_samples: int,
        n_positions: int,
        k: int,
        n_fragments: int,
        n_observations: int,
    ):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if ploidy_kind == "haploid":
            if kind == "stats":
                model = self._jax_haploid_fragment_count_stats_kernel(k)
            elif kind == "final":
                model = self._jax_haploid_fragment_count_final_kernel(k)
            elif kind == "output":
                model = self._jax_haploid_fragment_count_output_kernel(k)
            else:
                raise ValueError(f"Unknown fragment count callable kind: {kind}")
        elif ploidy_kind == "diploid":
            if kind == "stats":
                model = self._jax_diploid_fragment_count_stats_kernel(k)
            elif kind == "final":
                model = self._jax_diploid_fragment_count_final_kernel(k)
            elif kind == "output":
                model = self._jax_diploid_fragment_count_output_kernel(k)
            else:
                raise ValueError(f"Unknown fragment count callable kind: {kind}")
        else:
            raise ValueError(f"Unknown fragment count callable ploidy_kind: {ploidy_kind}")
        if not self.config.jax_aot_compile:
            return model
        key = (f"{ploidy_kind}_fragment_count_{kind}", n_samples, n_positions, k, n_fragments, n_observations)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        with _JAX_COMPILE_LOCK:
            compiled = self._jax_compiled.get(key)
            if compiled is not None:
                return compiled
            dummy_obs = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
            dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
            dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
            dummy_frag_sample = jnp.zeros((n_fragments,), dtype=jnp.int32)
            dummy_frag_center = jnp.zeros((n_fragments,), dtype=jnp.int32)
            dummy_obs_frag = jnp.zeros((n_observations,), dtype=jnp.int32)
            dummy_obs_pos = jnp.zeros((n_observations,), dtype=jnp.int32)
            dummy_obs_code = jnp.zeros((n_observations,), dtype=jnp.int8)
            dummy_obs_qual = jnp.full((n_observations,), -1.0, dtype=jnp.float32)
            dummy_scalar = jnp.asarray(0.01, dtype=jnp.float32)
            dummy_min = jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32)
            dummy_max_reads = jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32)
            dummy_max_emit = jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32)
            dummy_bool = jnp.asarray(True, dtype=jnp.bool_)
            args = [
                dummy_obs,
                dummy_obs,
                dummy_obs,
                dummy_switch,
            ]
            if kind != "output":
                args.append(dummy_obs)
            args.extend(
                [
                    dummy_founder,
                    dummy_frag_sample,
                    dummy_frag_center,
                    dummy_obs_frag,
                    dummy_obs_pos,
                    dummy_obs_code,
                    dummy_obs_qual,
                    dummy_scalar,
                    dummy_min,
                    dummy_max_reads,
                    dummy_max_emit,
                    dummy_bool,
                    dummy_bool,
                ]
            )
            compiled = model.lower(*args).compile()
            warm = compiled(*args)
            jax.block_until_ready(warm)
            self._jax_compiled[key] = compiled
            return compiled

    def _get_jax_diploid_fragment_count_stats_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        n_fragments: int,
        n_observations: int,
    ):
        return self._compile_fragment_count_callable(
            kind="stats",
            n_samples=n_samples,
            n_positions=n_positions,
            k=k,
            n_fragments=n_fragments,
            n_observations=n_observations,
        )

    def _get_jax_diploid_fragment_count_final_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        n_fragments: int,
        n_observations: int,
    ):
        return self._compile_fragment_count_callable(
            kind="final",
            n_samples=n_samples,
            n_positions=n_positions,
            k=k,
            n_fragments=n_fragments,
            n_observations=n_observations,
        )

    def _get_jax_diploid_fragment_count_output_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        n_fragments: int,
        n_observations: int,
    ):
        return self._compile_fragment_count_callable(
            kind="output",
            n_samples=n_samples,
            n_positions=n_positions,
            k=k,
            n_fragments=n_fragments,
            n_observations=n_observations,
        )

    def _get_jax_haploid_fragment_count_stats_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        n_fragments: int,
        n_observations: int,
    ):
        return self._compile_fragment_count_callable(
            kind="stats",
            ploidy_kind="haploid",
            n_samples=n_samples,
            n_positions=n_positions,
            k=k,
            n_fragments=n_fragments,
            n_observations=n_observations,
        )

    def _get_jax_haploid_fragment_count_final_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        n_fragments: int,
        n_observations: int,
    ):
        return self._compile_fragment_count_callable(
            kind="final",
            ploidy_kind="haploid",
            n_samples=n_samples,
            n_positions=n_positions,
            k=k,
            n_fragments=n_fragments,
            n_observations=n_observations,
        )

    def _get_jax_haploid_fragment_count_output_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        n_fragments: int,
        n_observations: int,
    ):
        return self._compile_fragment_count_callable(
            kind="output",
            ploidy_kind="haploid",
            n_samples=n_samples,
            n_positions=n_positions,
            k=k,
            n_fragments=n_fragments,
            n_observations=n_observations,
        )

    def _get_jax_haploid_count_stats_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_haploid_count_stats_kernel(k)
        key = ("haploid_count_stats", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_haploid_count_stats_kernel(k)
        dummy_obs = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        dummy_scalar = jnp.asarray(0.01, dtype=jnp.float32)
        compiled = model.lower(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_obs, dummy_founder, dummy_scalar, dummy_scalar).compile()
        warm = compiled(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_obs, dummy_founder, dummy_scalar, dummy_scalar)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_haploid_count_final_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_haploid_count_final_kernel(k)
        key = ("haploid_count_final", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_haploid_count_final_kernel(k)
        dummy_obs = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        dummy_scalar = jnp.asarray(0.01, dtype=jnp.float32)
        compiled = model.lower(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_obs, dummy_founder, dummy_scalar, dummy_scalar).compile()
        warm = compiled(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_obs, dummy_founder, dummy_scalar, dummy_scalar)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    def _get_jax_haploid_count_output_callable(self, n_samples: int, n_positions: int, k: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_haploid_count_output_kernel(k)
        key = ("haploid_count_output", n_samples, n_positions, k)
        compiled = self._jax_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_haploid_count_output_kernel(k)
        dummy_obs = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        dummy_scalar = jnp.asarray(0.01, dtype=jnp.float32)
        compiled = model.lower(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_founder, dummy_scalar, dummy_scalar).compile()
        warm = compiled(dummy_obs, dummy_obs, dummy_obs, dummy_switch, dummy_founder, dummy_scalar, dummy_scalar)
        jax.block_until_ready(warm)
        self._jax_compiled[key] = compiled
        return compiled

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_polyploid_model(k: int, ploidy: int, n_states: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        inv_states = 1.0 / float(n_states)
        exponent = jnp.arange(ploidy + 1, dtype=jnp.float32)
        reverse_exponent = float(ploidy) - exponent
        off_denom = float(max(k - 1, 1))

        def transition_from_switch(sw: jnp.ndarray, coeff: jnp.ndarray) -> jnp.ndarray:
            if k <= 1:
                return jnp.ones((n_states, n_states), dtype=jnp.float32)
            stay = 1.0 - sw
            off = sw / off_denom
            powers = (stay**exponent) * (off**reverse_exponent)
            return jnp.tensordot(coeff, powers, axes=((2,), (0,)))

        def run_one_sample(
            log_emission_s: jnp.ndarray,
            switch_s: jnp.ndarray,
            coeff: jnp.ndarray,
            state_prior: jnp.ndarray,
        ) -> jnp.ndarray:
            emission = jnp.exp(log_emission_s - jnp.max(log_emission_s, axis=1, keepdims=True))
            alpha0 = emission[0] * state_prior
            alpha0 = alpha0 / (jnp.sum(alpha0) + 1e-12)

            def fwd_step(carry: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
                sw, emission_t = inputs
                trans = transition_from_switch(sw, coeff)
                alpha_t = (carry @ trans) * emission_t
                alpha_t = alpha_t / (jnp.sum(alpha_t) + 1e-12)
                return alpha_t, alpha_t

            _, alpha_rest = jax.lax.scan(fwd_step, alpha0, (switch_s[1:], emission[1:]))
            alpha = jnp.concatenate([alpha0[None, :], alpha_rest], axis=0)

            betaT = jnp.full((n_states,), inv_states, dtype=jnp.float32)

            def bwd_step(carry: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]):
                sw_next, emission_next = inputs
                trans = transition_from_switch(sw_next, coeff)
                next_term = carry * emission_next
                beta_t = trans @ next_term
                beta_t = beta_t / (jnp.sum(beta_t) + 1e-12)
                return beta_t, beta_t

            _, beta_rev = jax.lax.scan(bwd_step, betaT, (switch_s[1:][::-1], emission[1:][::-1]))
            beta = jnp.concatenate([beta_rev[::-1], betaT[None, :]], axis=0)
            gamma = alpha * beta
            gamma = gamma / (jnp.sum(gamma, axis=1, keepdims=True) + 1e-12)
            return gamma

        return jax.vmap(run_one_sample, in_axes=(0, 0, None, None))

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_polyploid_stats_kernel(k: int, ploidy: int, n_states: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_polyploid_model(k, ploidy, n_states)

        def model(
            log_emission: jnp.ndarray,
            switch: jnp.ndarray,
            coeff: jnp.ndarray,
            state_prior: jnp.ndarray,
            state_counts: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
        ):
            gamma = posterior_model(log_emission, switch, coeff, state_prior)
            copy_prop = jnp.einsum("spm,mk->spk", gamma, state_counts) / float(ploidy)
            numerator = jnp.einsum("spk,sp->kp", copy_prop, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", copy_prop)
            return numerator, denominator

        return jax.jit(model)

    @staticmethod
    @lru_cache(maxsize=8)
    def _jax_polyploid_final_kernel(k: int, ploidy: int, n_states: int):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        posterior_model = JAXStitchHMM._jax_polyploid_model(k, ploidy, n_states)

        def model(
            log_emission: jnp.ndarray,
            switch: jnp.ndarray,
            coeff: jnp.ndarray,
            state_prior: jnp.ndarray,
            state_counts: jnp.ndarray,
            sample_alt_fraction: jnp.ndarray,
            founder_alt: jnp.ndarray,
        ):
            gamma = posterior_model(log_emission, switch, coeff, state_prior)
            copy_prop = jnp.einsum("spm,mk->spk", gamma, state_counts) / float(ploidy)
            numerator = jnp.einsum("spk,sp->kp", copy_prop, sample_alt_fraction)
            denominator = jnp.einsum("spk->kp", copy_prop)
            state_dosage = state_counts @ founder_alt
            dosage = jnp.einsum("spm,mp->sp", gamma, state_dosage)
            dist = jnp.zeros((n_states, founder_alt.shape[1], ploidy + 1), dtype=jnp.float32)
            dist = dist.at[:, :, 0].set(1.0)
            for founder_idx in range(k):
                q = jnp.clip(founder_alt[founder_idx], 1e-6, 1.0 - 1e-6)
                copies = state_counts[:, founder_idx]
                for copy_idx in range(ploidy):
                    active = copies > float(copy_idx)
                    prev = dist
                    updated = prev * (1.0 - q[None, :, None])
                    updated = updated.at[:, :, 1:].add(prev[:, :, :-1] * q[None, :, None])
                    dist = jnp.where(active[:, None, None], updated, dist)
            gp = jnp.einsum("spm,mpc->spc", gamma, dist)
            gp = jnp.clip(gp, 1e-8, 1.0)
            gp = gp / jnp.clip(jnp.sum(gp, axis=2, keepdims=True), 1e-8, None)
            return numerator, denominator, copy_prop, dosage, gp

        return jax.jit(model)

    def _get_jax_polyploid_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        ploidy: int,
        n_states: int,
    ):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        model = self._jax_polyploid_model(k, ploidy, n_states)
        if not self.config.jax_aot_compile:
            return jax.jit(model)
        key = (n_samples, n_positions, k, ploidy, n_states)
        compiled = self._jax_polyploid_compiled.get(key)
        if compiled is not None:
            return compiled
        dummy_emission = jnp.zeros((n_samples, n_positions, n_states), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_coeff = jnp.asarray(_polyploid_transition_coefficients(k, ploidy), dtype=jnp.float32)
        dummy_prior = jnp.asarray(_polyploid_state_prior(k, ploidy), dtype=jnp.float32)
        compiled = jax.jit(model).lower(dummy_emission, dummy_switch, dummy_coeff, dummy_prior).compile()
        warm = compiled(dummy_emission, dummy_switch, dummy_coeff, dummy_prior)
        warm.block_until_ready()
        self._jax_polyploid_compiled[key] = compiled
        return compiled

    def _get_jax_polyploid_stats_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        ploidy: int,
        n_states: int,
    ):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_polyploid_stats_kernel(k, ploidy, n_states)
        key = ("stats", n_samples, n_positions, k, ploidy, n_states)
        compiled = self._jax_polyploid_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_polyploid_stats_kernel(k, ploidy, n_states)
        dummy_emission = jnp.zeros((n_samples, n_positions, n_states), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_coeff = jnp.asarray(_polyploid_transition_coefficients(k, ploidy), dtype=jnp.float32)
        dummy_prior = jnp.asarray(_polyploid_state_prior(k, ploidy), dtype=jnp.float32)
        dummy_counts = jnp.asarray(_polyploid_state_counts(k, ploidy), dtype=jnp.float32)
        dummy_alt_fraction = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        compiled = model.lower(
            dummy_emission,
            dummy_switch,
            dummy_coeff,
            dummy_prior,
            dummy_counts,
            dummy_alt_fraction,
        ).compile()
        warm = compiled(
            dummy_emission,
            dummy_switch,
            dummy_coeff,
            dummy_prior,
            dummy_counts,
            dummy_alt_fraction,
        )
        jax.block_until_ready(warm)
        self._jax_polyploid_compiled[key] = compiled
        return compiled

    def _get_jax_polyploid_final_callable(
        self,
        n_samples: int,
        n_positions: int,
        k: int,
        ploidy: int,
        n_states: int,
    ):
        if jax is None or jnp is None:
            raise RuntimeError("JAX backend requested but jax/jaxlib are not available.")
        if not self.config.jax_aot_compile:
            return self._jax_polyploid_final_kernel(k, ploidy, n_states)
        key = ("final", n_samples, n_positions, k, ploidy, n_states)
        compiled = self._jax_polyploid_compiled.get(key)
        if compiled is not None:
            return compiled
        model = self._jax_polyploid_final_kernel(k, ploidy, n_states)
        dummy_emission = jnp.zeros((n_samples, n_positions, n_states), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        dummy_coeff = jnp.asarray(_polyploid_transition_coefficients(k, ploidy), dtype=jnp.float32)
        dummy_prior = jnp.asarray(_polyploid_state_prior(k, ploidy), dtype=jnp.float32)
        dummy_counts = jnp.asarray(_polyploid_state_counts(k, ploidy), dtype=jnp.float32)
        dummy_alt_fraction = jnp.zeros((n_samples, n_positions), dtype=jnp.float32)
        dummy_founder = jnp.zeros((k, n_positions), dtype=jnp.float32)
        compiled = model.lower(
            dummy_emission,
            dummy_switch,
            dummy_coeff,
            dummy_prior,
            dummy_counts,
            dummy_alt_fraction,
            dummy_founder,
        ).compile()
        warm = compiled(
            dummy_emission,
            dummy_switch,
            dummy_coeff,
            dummy_prior,
            dummy_counts,
            dummy_alt_fraction,
            dummy_founder,
        )
        jax.block_until_ready(warm)
        self._jax_polyploid_compiled[key] = compiled
        return compiled

    def _maybe_precompile(self, n_samples: int, n_positions: int, k: int) -> None:
        if self.backend != "jax" or not self.config.jax_precompile:
            return
        if jax is None or jnp is None:
            self.backend = "numpy"
            return
        ploidy = 1 if self.config.ploidy_mode == "pseudo_haploid" else int(self.config.ploidy)
        use_generic = bool(self.config.force_generic_ploidy_hmm) or ploidy >= 3
        if use_generic:
            n_states = int(_polyploid_state_counts(k, ploidy).shape[0])
            shape_key = (n_samples, n_positions, k, f"polyploid{ploidy}", self.backend)
            if shape_key in self._precompiled_shapes:
                return
            fn = self._get_jax_polyploid_callable(n_samples, n_positions, k, ploidy, n_states)
            dummy_emission = jnp.zeros((n_samples, n_positions, n_states), dtype=jnp.float32)
            dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
            dummy_coeff = jnp.asarray(_polyploid_transition_coefficients(k, ploidy), dtype=jnp.float32)
            dummy_prior = jnp.asarray(_polyploid_state_prior(k, ploidy), dtype=jnp.float32)
            out = fn(dummy_emission, dummy_switch, dummy_coeff, dummy_prior)
            out.block_until_ready()
            self._precompiled_shapes.add(shape_key)
            return
        if self.config.ploidy_mode != "diploid":
            return
        shape_key = (n_samples, n_positions, k, self.config.ploidy_mode, self.backend)
        if shape_key in self._precompiled_shapes:
            return
        if bool(self.config.jax_count_emission_kernel):
            # The production diploid JAX path uses count/moments kernels that do not
            # return full gamma to Python. Avoid precompiling the full-posterior kernel here.
            self._precompiled_shapes.add(shape_key)
            return
        fn = self._get_jax_diploid_callable(n_samples, n_positions, k)
        dummy_emission = jnp.zeros((n_samples, n_positions, k, k), dtype=jnp.float32)
        dummy_switch = jnp.full((n_samples, n_positions), 1e-5, dtype=jnp.float32)
        out = fn(dummy_emission, dummy_switch)
        out.block_until_ready()
        self._precompiled_shapes.add(shape_key)

    def _timed_posterior_step(self, log_emission: np.ndarray, switch: np.ndarray, k: int, backend: str) -> float:
        n_samples = log_emission.shape[0]
        n_positions = log_emission.shape[1]
        ploidy = 1 if self.config.ploidy_mode == "pseudo_haploid" else int(self.config.ploidy)
        use_generic = bool(self.config.force_generic_ploidy_hmm) or (self.config.ploidy_mode == "diploid" and ploidy >= 3)
        if use_generic:
            transition_coeff = _polyploid_transition_coefficients(k, ploidy).astype(np.float32, copy=False)
            if backend == "jax":
                self._maybe_precompile(n_samples, n_positions, k)
                fn = self._get_jax_polyploid_callable(n_samples, n_positions, k, ploidy, log_emission.shape[2])
                log_emission_jax = jnp.asarray(log_emission, dtype=jnp.float32)
                switch_jax = jnp.asarray(switch, dtype=jnp.float32)
                coeff_jax = jnp.asarray(transition_coeff, dtype=jnp.float32)
                prior_jax = jnp.asarray(_polyploid_state_prior(k, ploidy), dtype=jnp.float32)
                warm = fn(log_emission_jax, switch_jax, coeff_jax, prior_jax)
                warm.block_until_ready()
                t0 = perf_counter()
                gamma = fn(log_emission_jax, switch_jax, coeff_jax, prior_jax)
                gamma.block_until_ready()
                return perf_counter() - t0
            t0 = perf_counter()
            _ = self._forward_backward_numpy_polyploid(
                log_emission,
                switch,
                transition_coeff,
                k=k,
                ploidy=ploidy,
            )
            return perf_counter() - t0
        if self.config.ploidy_mode == "diploid":
            if backend == "jax":
                self._maybe_precompile(n_samples, n_positions, k)
                fn = self._get_jax_diploid_callable(n_samples, n_positions, k)
                log_emission_jax = jnp.asarray(log_emission, dtype=jnp.float32)
                switch_jax = jnp.asarray(switch, dtype=jnp.float32)
                warm = fn(log_emission_jax, switch_jax)
                warm.block_until_ready()
                t0 = perf_counter()
                gamma = fn(log_emission_jax, switch_jax)
                gamma.block_until_ready()
                return perf_counter() - t0
            if backend == "torch":
                t0 = perf_counter()
                _ = self._forward_backward_torch_diploid(log_emission, switch, k)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                return perf_counter() - t0
            else:
                t0 = perf_counter()
                _ = self._forward_backward_numpy_diploid(log_emission, switch, k)
                return perf_counter() - t0
        else:
            if backend == "torch":
                t0 = perf_counter()
                _ = self._forward_backward_torch_haploid(log_emission, switch, k)
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.synchronize()
                return perf_counter() - t0
            t0 = perf_counter()
            _ = self._forward_backward_numpy_haploid(log_emission, switch, k)
            return perf_counter() - t0

    def autotune_backend(
        self,
        founder_panel: FounderPanel,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        generations: np.ndarray,
        *,
        other_count: np.ndarray | None = None,
        ref_weight: np.ndarray | None = None,
        alt_weight: np.ndarray | None = None,
        other_weight: np.ndarray | None = None,
        fragment_sample_offsets: np.ndarray | None = None,
        fragment_center_idx: np.ndarray | None = None,
        fragment_obs_offsets: np.ndarray | None = None,
        fragment_obs_pos_idx: np.ndarray | None = None,
        fragment_obs_code: np.ndarray | None = None,
        fragment_obs_qual: np.ndarray | None = None,
    ) -> None:
        if self._autotuned:
            return
        self._autotuned = True
        if self._configured_backend != "auto" or not self.config.backend_autotune:
            return

        n_samples_cfg = int(self.config.autotune_samples)
        n_positions_cfg = int(self.config.autotune_positions)
        n_samples = ref_count.shape[0] if n_samples_cfg <= 0 else min(ref_count.shape[0], n_samples_cfg)
        n_positions = ref_count.shape[1] if n_positions_cfg <= 0 else min(ref_count.shape[1], n_positions_cfg)
        founder_sub = founder_panel.slice(0, n_positions)
        rates = self.recombination_from_positions(founder_sub.positions)
        switch = self._switch_probabilities(rates, generations[:n_samples])
        ploidy = 1 if self.config.ploidy_mode == "pseudo_haploid" else int(self.config.ploidy)
        use_generic = bool(self.config.force_generic_ploidy_hmm) or (self.config.ploidy_mode == "diploid" and ploidy >= 3)
        if use_generic:
            state_counts = _polyploid_state_counts(founder_sub.n_founders, ploidy).astype(np.float32, copy=False)
            log_emission, founder_alt = self._polyploid_emissions(
                founder_sub.alt_prob,
                state_counts,
                ploidy,
                ref_count[:n_samples, :n_positions],
                alt_count[:n_samples, :n_positions],
                None if other_count is None else other_count[:n_samples, :n_positions],
                None if ref_weight is None else ref_weight[:n_samples, :n_positions],
                None if alt_weight is None else alt_weight[:n_samples, :n_positions],
                None if other_weight is None else other_weight[:n_samples, :n_positions],
            )
            log_emission = self._apply_fragment_likelihoods_polyploid(
                log_emission=log_emission,
                founder_alt=founder_alt.astype(np.float32, copy=False),
                state_counts=state_counts,
                ploidy=ploidy,
                fragment_sample_offsets=None,
                fragment_center_idx=None,
                fragment_obs_offsets=None,
                fragment_obs_pos_idx=None,
                fragment_obs_code=None,
                fragment_obs_qual=None,
            )
        else:
            log_emission, _, founder_alt = self.emissions(
                founder_sub.alt_prob,
                ref_count[:n_samples, :n_positions],
                alt_count[:n_samples, :n_positions],
                None if other_count is None else other_count[:n_samples, :n_positions],
                None if ref_weight is None else ref_weight[:n_samples, :n_positions],
                None if alt_weight is None else alt_weight[:n_samples, :n_positions],
                None if other_weight is None else other_weight[:n_samples, :n_positions],
            )
            log_emission = self._apply_fragment_likelihoods(
                log_emission,
                founder_alt=founder_alt.astype(np.float32, copy=False),
                fragment_sample_offsets=None,
                fragment_center_idx=None,
                fragment_obs_offsets=None,
                fragment_obs_pos_idx=None,
                fragment_obs_code=None,
                fragment_obs_qual=None,
            )

        candidates = ["numpy"]
        if jax is not None and jnp is not None and (self.config.ploidy_mode == "diploid" or bool(self.config.force_generic_ploidy_hmm)):
            candidates.append("jax")
        if _torch_available() and torch is not None and torch.cuda.is_available():
            candidates.append("torch")

        best_backend = candidates[0]
        best_time = float("inf")
        previous_backend = self.backend
        for backend in candidates:
            self.backend = backend
            trial = self._timed_posterior_step(log_emission=log_emission, switch=switch, k=founder_sub.n_founders, backend=backend)
            if trial < best_time:
                best_time = trial
                best_backend = backend
        self.backend = best_backend if best_time < float("inf") else previous_backend

    def _observed_counts(
        self,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray | None,
        ref_weight: np.ndarray | None,
        alt_weight: np.ndarray | None,
        other_weight: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.config.use_quality_weights and ref_weight is not None and alt_weight is not None:
            ref_obs = ref_weight.astype(np.float32, copy=False)
            alt_obs = alt_weight.astype(np.float32, copy=False)
            if other_weight is None:
                other_obs = np.zeros_like(ref_obs)
            else:
                other_obs = other_weight.astype(np.float32, copy=False)
        else:
            ref_obs = ref_count.astype(np.float32, copy=False)
            alt_obs = alt_count.astype(np.float32, copy=False)
            if other_count is None:
                other_obs = np.zeros_like(ref_obs)
            else:
                other_obs = other_count.astype(np.float32, copy=False)
        return ref_obs, alt_obs, other_obs

    def _sample_alt_fraction(
        self,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray | None,
        ref_weight: np.ndarray | None,
        alt_weight: np.ndarray | None,
        other_weight: np.ndarray | None,
    ) -> np.ndarray:
        ref_obs, alt_obs, other_obs = self._observed_counts(
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
        )
        return (alt_obs + 0.5) / (ref_obs + alt_obs + other_obs + 1.0)

    def _update_founders_from_stats(
        self,
        founder_alt_prob: np.ndarray,
        founder_immutable_mask: np.ndarray,
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        proposed = np.clip(numerator / (denominator + 1e-8), 1e-4, 1.0 - 1e-4)
        proposed = self._harden_mutable_founder_updates(
            proposed=proposed,
            current=founder_alt_prob,
            founder_immutable_mask=founder_immutable_mask,
        )
        damping = float(np.clip(self.config.em_founder_update_damping, 1e-6, 1.0))
        updated = founder_alt_prob + damping * (proposed - founder_alt_prob)
        immutable = founder_immutable_mask[:, None]
        return np.where(immutable, founder_alt_prob, np.clip(updated, 1e-4, 1.0 - 1e-4))

    def _harden_mutable_founder_updates(
        self,
        *,
        proposed: np.ndarray,
        current: np.ndarray,
        founder_immutable_mask: np.ndarray,
    ) -> np.ndarray:
        if not bool(self.config.founder_update_hardening):
            return proposed
        mutable_idx = np.flatnonzero(~founder_immutable_mask.astype(bool, copy=False))
        n_mutable = int(mutable_idx.size)
        if n_mutable < 2:
            return proposed
        out = proposed.astype(np.float32, copy=True)
        proposed_mut = proposed[mutable_idx].astype(np.float32, copy=False)
        current_mut = current[mutable_idx].astype(np.float32, copy=False)
        site_p = np.clip(np.mean(proposed_mut, axis=0), 0.0, 1.0)
        alt_counts = np.rint(site_p * float(n_mutable)).astype(np.int16, copy=False)
        polymorphic = (site_p > (0.5 / float(n_mutable))) & (site_p < (1.0 - 0.5 / float(n_mutable)))
        alt_counts[polymorphic] = np.clip(alt_counts[polymorphic], 1, n_mutable - 1)
        alt_counts = np.clip(alt_counts, 0, n_mutable)
        score = proposed_mut + 0.05 * current_mut
        low = np.float32(0.02)
        high = np.float32(0.98)
        for pos_idx, n_alt in enumerate(alt_counts.tolist()):
            hardened = np.full(n_mutable, low, dtype=np.float32)
            if int(n_alt) > 0:
                top = np.argsort(score[:, pos_idx], kind="stable")[-int(n_alt):]
                hardened[top] = high
            out[mutable_idx, pos_idx] = hardened
        return out

    @staticmethod
    def _founder_update_delta(
        before: np.ndarray,
        after: np.ndarray,
        founder_immutable_mask: np.ndarray,
    ) -> tuple[float, float]:
        mutable = ~founder_immutable_mask.astype(bool, copy=False)
        if not np.any(mutable):
            return 0.0, 0.0
        delta = np.abs(
            after[mutable].astype(np.float32, copy=False)
            - before[mutable].astype(np.float32, copy=False)
        )
        if delta.size == 0:
            return 0.0, 0.0
        return float(np.max(delta)), float(np.mean(delta))

    def _em_converged(self, *, iteration_count: int, max_delta: float) -> bool:
        tol = float(self.config.em_convergence_tol)
        if tol <= 0.0:
            return False
        if int(iteration_count) < max(int(self.config.em_convergence_min_iterations), 1):
            return False
        return bool(np.isfinite(max_delta) and max_delta <= tol)

    def _update_founders(
        self,
        founder_alt_prob: np.ndarray,
        founder_immutable_mask: np.ndarray,
        hap_posterior: np.ndarray,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray | None,
        ref_weight: np.ndarray | None,
        alt_weight: np.ndarray | None,
        other_weight: np.ndarray | None,
    ) -> np.ndarray:
        sample_alt_fraction = self._sample_alt_fraction(
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
        )
        numerator = np.einsum("spk,sp->kp", hap_posterior, sample_alt_fraction)
        denominator = np.einsum("spk->kp", hap_posterior)
        return self._update_founders_from_stats(
            founder_alt_prob=founder_alt_prob,
            founder_immutable_mask=founder_immutable_mask,
            numerator=numerator,
            denominator=denominator,
        )

    def _resolve_jax_sample_batch_size(self, n_samples: int) -> int:
        batch = int(self.config.jax_sample_batch_size)
        if batch <= 0 or batch >= n_samples:
            return n_samples
        if bool(self.config.jax_bucket_batch_shapes) and n_samples % batch != 0:
            min_bucket = max(1, int(np.floor(0.5 * float(batch))))
            for candidate in range(batch, min_bucket - 1, -1):
                if n_samples % candidate == 0:
                    return candidate
        return max(batch, 1)

    @staticmethod
    def _has_fragment_batch(fragment_center_idx: np.ndarray | None) -> bool:
        return fragment_center_idx is not None and int(fragment_center_idx.size) > 0

    @staticmethod
    def _slice_fragment_batch(
        sample_start: int,
        sample_stop: int,
        fragment_sample_offsets: np.ndarray | None,
        fragment_center_idx: np.ndarray | None,
        fragment_obs_offsets: np.ndarray | None,
        fragment_obs_pos_idx: np.ndarray | None,
        fragment_obs_code: np.ndarray | None,
        fragment_obs_qual: np.ndarray | None,
    ) -> tuple[
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]:
        if (
            fragment_sample_offsets is None
            or fragment_center_idx is None
            or fragment_obs_offsets is None
            or fragment_obs_pos_idx is None
            or fragment_obs_code is None
        ):
            return None, None, None, None, None, None
        if sample_stop <= sample_start:
            return None, None, None, None, None, None

        frag_start = int(fragment_sample_offsets[sample_start])
        frag_stop = int(fragment_sample_offsets[sample_stop])
        local_sample_offsets = fragment_sample_offsets[sample_start : sample_stop + 1] - frag_start
        if frag_stop <= frag_start:
            return (
                local_sample_offsets.astype(np.int64, copy=False),
                np.empty((0,), dtype=np.int32),
                np.zeros(1, dtype=np.int64),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int8),
                np.empty((0,), dtype=np.uint8) if fragment_obs_qual is not None else None,
            )

        obs_start = int(fragment_obs_offsets[frag_start])
        obs_stop = int(fragment_obs_offsets[frag_stop])
        local_obs_offsets = fragment_obs_offsets[frag_start : frag_stop + 1] - obs_start
        local_obs_qual = None
        if fragment_obs_qual is not None:
            local_obs_qual = fragment_obs_qual[obs_start:obs_stop]
        return (
            local_sample_offsets.astype(np.int64, copy=False),
            fragment_center_idx[frag_start:frag_stop].astype(np.int32, copy=False),
            local_obs_offsets.astype(np.int64, copy=False),
            fragment_obs_pos_idx[obs_start:obs_stop].astype(np.int32, copy=False),
            fragment_obs_code[obs_start:obs_stop].astype(np.int8, copy=False),
            local_obs_qual.astype(np.uint8, copy=False) if local_obs_qual is not None else None,
        )

    @staticmethod
    def _jax_fragment_index_arrays(
        fragment_sample_offsets: np.ndarray,
        fragment_center_idx: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frag_counts = np.diff(fragment_sample_offsets).astype(np.int64, copy=False)
        fragment_sample_idx = np.repeat(np.arange(frag_counts.shape[0], dtype=np.int32), frag_counts)
        obs_counts = np.diff(fragment_obs_offsets).astype(np.int64, copy=False)
        fragment_obs_fragment_idx = np.repeat(np.arange(obs_counts.shape[0], dtype=np.int32), obs_counts)
        if fragment_obs_qual is None:
            obs_qual = np.full(int(np.sum(obs_counts)), -1.0, dtype=np.float32)
        else:
            obs_qual = fragment_obs_qual.astype(np.float32, copy=False)
        if fragment_sample_idx.shape[0] != fragment_center_idx.shape[0]:
            raise ValueError("Fragment sample offsets do not match fragment centers.")
        return (
            fragment_sample_idx.astype(np.int32, copy=False),
            fragment_obs_fragment_idx.astype(np.int32, copy=False),
            obs_qual.astype(np.float32, copy=False),
        )

    def _build_full_transitions(self, switch: np.ndarray, k: int) -> np.ndarray:
        n_samples, n_positions = switch.shape
        out = np.empty((n_samples, n_positions, k * k, k * k), dtype=np.float32)
        for sample_idx in range(n_samples):
            for pos_idx in range(n_positions):
                hap = _build_transition_matrix_from_switch(float(switch[sample_idx, pos_idx]), k)
                out[sample_idx, pos_idx] = np.einsum("ab,cd->acbd", hap, hap).reshape(k * k, k * k)
        return out

    def _build_full_transitions_polyploid(
        self,
        switch: np.ndarray,
        transition_coeff: np.ndarray,
        *,
        k: int,
        ploidy: int,
    ) -> np.ndarray:
        n_samples, n_positions = switch.shape
        n_states = int(transition_coeff.shape[0])
        out = np.empty((n_samples, n_positions, n_states, n_states), dtype=np.float32)
        for sample_idx in range(n_samples):
            for pos_idx in range(n_positions):
                out[sample_idx, pos_idx] = self._polyploid_transition_matrix(
                    float(switch[sample_idx, pos_idx]),
                    transition_coeff,
                    k=k,
                    ploidy=ploidy,
                )
        return out

    def _initialize_founder_alt_prob(
        self,
        founder_alt_prob: np.ndarray,
        founder_immutable_mask: np.ndarray,
        site_alt_fraction: np.ndarray | None = None,
    ) -> np.ndarray:
        out = founder_alt_prob.astype(np.float32, copy=True)
        mutable_mask_1d = ~founder_immutable_mask.astype(bool, copy=False)
        if not np.any(mutable_mask_1d):
            return np.clip(out, 1e-4, 1.0 - 1e-4).astype(np.float32, copy=False)

        mutable_idx = np.flatnonzero(mutable_mask_1d)
        mutable_values = out[mutable_idx]
        if mutable_values.shape[0] >= 2:
            per_site_std = np.nanstd(mutable_values, axis=0)
            collapsed_sites = per_site_std <= 1e-7
        else:
            collapsed_sites = np.full(out.shape[1], True, dtype=bool)

        if site_alt_fraction is not None and np.any(collapsed_sites):
            site_p = np.asarray(site_alt_fraction, dtype=np.float32).reshape(-1)
            if site_p.shape[0] != out.shape[1]:
                raise ValueError(
                    "site_alt_fraction length must match founder_alt_prob positions "
                    f"({site_p.shape[0]} != {out.shape[1]})."
                )
            site_p = np.where(np.isfinite(site_p), site_p, 0.5).astype(np.float32, copy=False)
            site_p = np.clip(site_p, 1e-3, 1.0 - 1e-3)
            n_mutable = int(mutable_idx.size)
            random_matrix = self._rng.random((n_mutable, int(out.shape[1]))).astype(np.float32, copy=False)
            seeded = (random_matrix < site_p[None, :]).astype(np.float32, copy=False)
            seeded = 0.02 + 0.96 * seeded
            if n_mutable < 2:
                seeded = np.broadcast_to(site_p[None, :], seeded.shape).astype(np.float32, copy=True)
            out[np.ix_(mutable_idx, collapsed_sites)] = seeded[:, collapsed_sites]

        jitter = float(self.config.founder_init_jitter)
        if jitter <= 0.0:
            return np.clip(out, 1e-4, 1.0 - 1e-4).astype(np.float32, copy=False)
        mutable = mutable_mask_1d[:, None]
        noise = self._rng.normal(0.0, jitter, size=out.shape).astype(np.float32, copy=False)
        out = out + mutable.astype(np.float32, copy=False) * noise
        return np.clip(out, 1e-4, 1.0 - 1e-4).astype(np.float32, copy=False)

    def _genotype_posterior_from_pair_gamma(
        self,
        pair_gamma: np.ndarray,
        founder_alt: np.ndarray,
    ) -> np.ndarray:
        q = founder_alt.astype(np.float32, copy=False).T  # (P, K)
        one_minus_q = 1.0 - q
        gp0 = np.einsum("spij,pi,pj->sp", pair_gamma, one_minus_q, one_minus_q, optimize=True)
        gp2 = np.einsum("spij,pi,pj->sp", pair_gamma, q, q, optimize=True)
        gp1 = np.clip(1.0 - gp0 - gp2, 0.0, 1.0)
        gp = np.stack([gp0, gp1, gp2], axis=2).astype(np.float32, copy=False)
        gp /= np.clip(np.sum(gp, axis=2, keepdims=True), 1e-8, None)
        return gp

    def _genotype_posterior_from_haplotype_posterior(
        self,
        hap_posterior: np.ndarray,
        founder_alt: np.ndarray,
    ) -> np.ndarray:
        p_alt = np.einsum("spk,kp->sp", hap_posterior, founder_alt.astype(np.float32, copy=False), optimize=True)
        p_alt = np.clip(p_alt, 1e-6, 1.0 - 1e-6)
        gp = np.stack([1.0 - p_alt, p_alt], axis=2).astype(np.float32, copy=False)
        gp /= np.clip(np.sum(gp, axis=2, keepdims=True), 1e-8, None)
        return gp

    def _genotype_posterior_from_state_gamma(
        self,
        state_gamma: np.ndarray,
        state_counts: np.ndarray,
        founder_alt: np.ndarray,
        *,
        ploidy: int,
    ) -> np.ndarray:
        n_samples, n_positions, n_states = state_gamma.shape
        state_counts_i16 = state_counts.astype(np.int16, copy=False)
        founder_alt_f32 = founder_alt.astype(np.float32, copy=False)
        gp = np.empty((n_samples, n_positions, int(ploidy) + 1), dtype=np.float32)
        max_count_by_founder = np.max(state_counts_i16, axis=0)
        for pos_idx in range(n_positions):
            dist = np.zeros((n_states, int(ploidy) + 1), dtype=np.float32)
            dist[:, 0] = 1.0
            for founder_idx, max_count in enumerate(max_count_by_founder.tolist()):
                q = float(np.clip(founder_alt_f32[founder_idx, pos_idx], 1e-6, 1.0 - 1e-6))
                copies = state_counts_i16[:, founder_idx]
                for copy_idx in range(int(max_count)):
                    active = copies > copy_idx
                    if not np.any(active):
                        continue
                    prev = dist[active].copy()
                    updated = prev * (1.0 - q)
                    updated[:, 1:] += prev[:, :-1] * q
                    dist[active] = updated
            gp[:, pos_idx, :] = state_gamma[:, pos_idx, :] @ dist
        gp = np.clip(gp, 1e-8, 1.0)
        gp /= np.clip(np.sum(gp, axis=2, keepdims=True), 1e-8, None)
        return gp.astype(np.float32, copy=False)

    def _run_polyploid(
        self,
        *,
        ploidy: int,
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
    ) -> HMMArtifacts:
        if int(ploidy) < 1:
            raise ValueError("_run_polyploid requires ploidy >= 1.")
        if self.backend == "torch":
            self.backend = "numpy"
        if self.backend == "jax" and (jax is None or jnp is None):
            self.backend = "numpy"

        k = founder_panel.n_founders
        rates = self.recombination_from_positions(founder_panel.positions)
        switch = self._switch_probabilities(rates, generations)
        stay = 1.0 - switch
        offdiag = switch / max(k - 1, 1)
        state_counts = _polyploid_state_counts(k, int(ploidy)).astype(np.float32, copy=False)
        transition_coeff = _polyploid_transition_coefficients(k, int(ploidy)).astype(np.float32, copy=False)
        sample_alt_fraction = self._sample_alt_fraction(
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
        ).astype(np.float32, copy=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            site_alt_fraction = np.nanmean(sample_alt_fraction, axis=0).astype(np.float32, copy=False)
        working_alt_prob = founder_panel.alt_prob.astype(np.float32, copy=True)
        founder_immutable_mask = founder_panel.immutable_mask
        working_alt_prob = self._initialize_founder_alt_prob(
            founder_alt_prob=working_alt_prob,
            founder_immutable_mask=founder_immutable_mask,
            site_alt_fraction=site_alt_fraction,
        )
        n_samples, n_positions = ref_count.shape
        if self.backend == "jax":
            batch_size = self._resolve_jax_sample_batch_size(n_samples)
            self._maybe_precompile(batch_size, n_positions, k)
            remainder = n_samples % batch_size
            if remainder:
                self._maybe_precompile(remainder, n_positions, k)
        else:
            self._maybe_precompile(n_samples, n_positions, k)
        dosage: np.ndarray | None = None
        haplotype_posterior: np.ndarray | None = None
        genotype_posterior: np.ndarray | None = None

        max_em_iterations = max(int(self.config.em_iterations), 1)
        all_founders_immutable = bool(np.all(founder_immutable_mask))
        final_pass_pending = all_founders_immutable or max_em_iterations == 1
        converged_passes = 0
        em_iter = 0
        em_history: list[dict[str, object]] = []
        best_founder_alt_prob = working_alt_prob.copy()
        best_founder_max_delta = float("inf")
        best_iteration = -1
        restored_best_founders = False
        while True:
            if not final_pass_pending and em_iter >= max_em_iterations:
                if (
                    bool(self.config.adaptive_em_restore_best_founders)
                    and best_iteration >= 0
                    and best_founder_alt_prob is not None
                ):
                    working_alt_prob = best_founder_alt_prob.copy()
                    restored_best_founders = True
                final_pass_pending = True
            is_last_iteration = final_pass_pending
            numerator = np.zeros((k, n_positions), dtype=np.float32)
            denominator = np.zeros((k, n_positions), dtype=np.float32)

            if self.backend == "jax":
                batch_size = self._resolve_jax_sample_batch_size(n_samples)
                if is_last_iteration:
                    dosage = np.empty((n_samples, n_positions), dtype=np.float32)
                    if return_haplotype_posterior:
                        haplotype_posterior = np.empty((n_samples, n_positions, k), dtype=np.float32)
                    if return_genotype_posterior:
                        genotype_posterior = np.empty((n_samples, n_positions, int(ploidy) + 1), dtype=np.float32)
                for sample_start in range(0, n_samples, batch_size):
                    sample_stop = min(sample_start + batch_size, n_samples)
                    n_batch = sample_stop - sample_start
                    log_emission_batch, founder_alt = self._polyploid_emissions(
                        working_alt_prob,
                        state_counts,
                        int(ploidy),
                        ref_count[sample_start:sample_stop],
                        alt_count[sample_start:sample_stop],
                        None if other_count is None else other_count[sample_start:sample_stop],
                        None if ref_weight is None else ref_weight[sample_start:sample_stop],
                        None if alt_weight is None else alt_weight[sample_start:sample_stop],
                        None if other_weight is None else other_weight[sample_start:sample_stop],
                    )
                    (
                        fso,
                        fci,
                        foo,
                        fop,
                        foc,
                        foq,
                    ) = self._slice_fragment_batch(
                        sample_start=sample_start,
                        sample_stop=sample_stop,
                        fragment_sample_offsets=fragment_sample_offsets,
                        fragment_center_idx=fragment_center_idx,
                        fragment_obs_offsets=fragment_obs_offsets,
                        fragment_obs_pos_idx=fragment_obs_pos_idx,
                        fragment_obs_code=fragment_obs_code,
                        fragment_obs_qual=fragment_obs_qual,
                    )
                    use_jax_polyploid_fragment = (
                        bool(self.config.use_fragment_likelihood)
                        and bool(self.config.jax_fragment_emission_kernel)
                        and self._has_fragment_batch(fci)
                        and fso is not None
                        and foo is not None
                        and fop is not None
                        and foc is not None
                        and int(fci.shape[0]) * int(state_counts.shape[0]) <= 50_000_000
                    )
                    if use_jax_polyploid_fragment:
                        fragment_sample_idx, fragment_obs_fragment_idx, fragment_obs_qual_f32 = self._jax_fragment_index_arrays(
                            fso,
                            fci,
                            foo,
                            foq,
                        )
                        log_emission_jax = self._jax_apply_polyploid_fragment_likelihoods(
                            jnp.asarray(log_emission_batch, dtype=jnp.float32),
                            jnp.asarray(founder_alt, dtype=jnp.float32),
                            jnp.asarray(state_counts, dtype=jnp.float32),
                            jnp.asarray(float(ploidy), dtype=jnp.float32),
                            jnp.asarray(fragment_sample_idx, dtype=jnp.int32),
                            jnp.asarray(fci, dtype=jnp.int32),
                            jnp.asarray(fragment_obs_fragment_idx, dtype=jnp.int32),
                            jnp.asarray(fop, dtype=jnp.int32),
                            jnp.asarray(foc, dtype=jnp.int8),
                            jnp.asarray(fragment_obs_qual_f32, dtype=jnp.float32),
                            jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                            jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                            jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32),
                            jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32),
                            jnp.asarray(bool(self.config.fragment_rescale_read_likelihood), dtype=jnp.bool_),
                            jnp.asarray(self.config.fragment_likelihood_mode == "replace", dtype=jnp.bool_),
                        )
                        jax.block_until_ready(log_emission_jax)
                        log_emission_batch = np.asarray(log_emission_jax, dtype=np.float32)
                    else:
                        log_emission_batch = self._apply_fragment_likelihoods_polyploid(
                            log_emission=log_emission_batch,
                            founder_alt=founder_alt,
                            state_counts=state_counts,
                            ploidy=int(ploidy),
                            fragment_sample_offsets=fso,
                            fragment_center_idx=fci,
                            fragment_obs_offsets=foo,
                            fragment_obs_pos_idx=fop,
                            fragment_obs_code=foc,
                            fragment_obs_qual=foq,
                        )
                    if is_last_iteration and dosage is not None:
                        founder_alt_f32 = founder_alt.astype(np.float32, copy=False)
                        fn = self._get_jax_polyploid_final_callable(
                            n_batch,
                            n_positions,
                            k,
                            int(ploidy),
                            int(state_counts.shape[0]),
                        )
                        (
                            numerator_jax,
                            denominator_jax,
                            copy_prop_jax,
                            dosage_jax,
                            gp_jax,
                        ) = fn(
                            jnp.asarray(log_emission_batch, dtype=jnp.float32),
                            jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                            jnp.asarray(transition_coeff, dtype=jnp.float32),
                            jnp.asarray(_polyploid_state_prior(k, int(ploidy)), dtype=jnp.float32),
                            jnp.asarray(state_counts, dtype=jnp.float32),
                            jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                            jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                        )
                        jax.block_until_ready((numerator_jax, denominator_jax, dosage_jax))
                        numerator += np.asarray(numerator_jax, dtype=np.float32)
                        denominator += np.asarray(denominator_jax, dtype=np.float32)
                        dosage[sample_start:sample_stop] = np.asarray(dosage_jax, dtype=np.float32)
                        if return_haplotype_posterior and haplotype_posterior is not None:
                            haplotype_posterior[sample_start:sample_stop] = np.asarray(copy_prop_jax, dtype=np.float32)
                        if return_genotype_posterior and genotype_posterior is not None:
                            genotype_posterior[sample_start:sample_stop] = np.asarray(gp_jax, dtype=np.float32)
                    else:
                        fn = self._get_jax_polyploid_stats_callable(
                            n_batch,
                            n_positions,
                            k,
                            int(ploidy),
                            int(state_counts.shape[0]),
                        )
                        numerator_jax, denominator_jax = fn(
                            jnp.asarray(log_emission_batch, dtype=jnp.float32),
                            jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                            jnp.asarray(transition_coeff, dtype=jnp.float32),
                            jnp.asarray(_polyploid_state_prior(k, int(ploidy)), dtype=jnp.float32),
                            jnp.asarray(state_counts, dtype=jnp.float32),
                            jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                        )
                        jax.block_until_ready((numerator_jax, denominator_jax))
                        numerator += np.asarray(numerator_jax, dtype=np.float32)
                        denominator += np.asarray(denominator_jax, dtype=np.float32)
            else:
                log_emission, founder_alt = self._polyploid_emissions(
                    working_alt_prob,
                    state_counts,
                    int(ploidy),
                    ref_count,
                    alt_count,
                    other_count,
                    ref_weight,
                    alt_weight,
                    other_weight,
                )
                log_emission = self._apply_fragment_likelihoods_polyploid(
                    log_emission=log_emission,
                    founder_alt=founder_alt,
                    state_counts=state_counts,
                    ploidy=int(ploidy),
                    fragment_sample_offsets=fragment_sample_offsets,
                    fragment_center_idx=fragment_center_idx,
                    fragment_obs_offsets=fragment_obs_offsets,
                    fragment_obs_pos_idx=fragment_obs_pos_idx,
                    fragment_obs_code=fragment_obs_code,
                    fragment_obs_qual=fragment_obs_qual,
                )
                gamma = self._forward_backward_numpy_polyploid(
                    log_emission,
                    switch,
                    transition_coeff,
                    k=k,
                    ploidy=int(ploidy),
                )
                copy_prop = np.einsum("spm,mk->spk", gamma, state_counts, optimize=True) / float(ploidy)
                numerator = np.einsum("spk,sp->kp", copy_prop, sample_alt_fraction, optimize=True)
                denominator = np.einsum("spk->kp", copy_prop, optimize=True)
                if is_last_iteration:
                    founder_alt_f32 = founder_alt.astype(np.float32, copy=False)
                    state_dosage = state_counts @ founder_alt_f32
                    dosage = np.einsum("spm,mp->sp", gamma, state_dosage, optimize=True)
                    if return_haplotype_posterior:
                        haplotype_posterior = copy_prop.astype(np.float32, copy=False)
                    if return_genotype_posterior:
                        genotype_posterior = self._genotype_posterior_from_state_gamma(
                            gamma,
                            state_counts,
                            founder_alt_f32,
                            ploidy=int(ploidy),
                        )

            if not all_founders_immutable and not is_last_iteration:
                previous_alt_prob = working_alt_prob
                working_alt_prob = self._update_founders_from_stats(
                    founder_alt_prob=working_alt_prob,
                    founder_immutable_mask=founder_immutable_mask,
                    numerator=numerator,
                    denominator=denominator,
                )
                max_delta, mean_delta = self._founder_update_delta(previous_alt_prob, working_alt_prob, founder_immutable_mask)
                if bool(self.config.adaptive_em):
                    em_history.append(
                        {
                            "iteration": int(em_iter + 1),
                            "founder_max_delta": float(max_delta),
                            "founder_mean_delta": float(mean_delta),
                            "read_log_likelihood": None,
                            "masked_genotype_likelihood": None,
                        }
                    )
                    if float(max_delta) < best_founder_max_delta:
                        best_founder_max_delta = float(max_delta)
                        best_founder_alt_prob = working_alt_prob.copy()
                        best_iteration = int(em_iter + 1)
                if self._em_converged(iteration_count=em_iter + 1, max_delta=max_delta):
                    converged_passes += 1
                    if converged_passes >= max(int(self.config.em_convergence_patience), 1):
                        if (
                            bool(self.config.adaptive_em_restore_best_founders)
                            and best_iteration >= 0
                            and best_founder_alt_prob is not None
                        ):
                            working_alt_prob = best_founder_alt_prob.copy()
                            restored_best_founders = True
                        final_pass_pending = True
                else:
                    converged_passes = 0
            if is_last_iteration:
                break
            em_iter += 1

        if dosage is None:
            raise RuntimeError("Polyploid HMM did not produce dosage output.")
        if return_genotype_posterior and genotype_posterior is None:
            from .calibration import dosage_to_genotype_posterior

            genotype_posterior = dosage_to_genotype_posterior(
                np.asarray(dosage, dtype=np.float32),
                temperature=0.35,
                ploidy=int(ploidy),
            )
        transition_probability = (
            self._build_full_transitions_polyploid(
                switch,
                transition_coeff,
                k=k,
                ploidy=int(ploidy),
            )
            if return_full_transition
            else None
        )
        gp_out = np.asarray(genotype_posterior, dtype=np.float32) if return_genotype_posterior else None
        final_read_ll = self._read_log_likelihood_total(
            dosage=np.asarray(dosage, dtype=np.float32),
            genotype_posterior=gp_out,
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            ploidy=int(ploidy),
        )
        em_diagnostics = {
            "adaptive_em": bool(self.config.adaptive_em),
            "all_founders_immutable": bool(all_founders_immutable),
            "em_updates": int(len(em_history)),
            "em_iterations_requested": int(max_em_iterations),
            "history": em_history,
            "best_iteration": int(best_iteration),
            "best_founder_max_delta": float(best_founder_max_delta) if np.isfinite(best_founder_max_delta) else None,
            "final_founder_max_delta": (
                float(em_history[-1]["founder_max_delta"]) if em_history else 0.0
            ),
            "restored_best_founders": bool(restored_best_founders),
            "final_read_log_likelihood": float(final_read_ll),
            "masked_genotype_likelihood": None,
        }
        return HMMArtifacts(
            dosage=np.asarray(dosage, dtype=np.float32),
            haplotype_posterior=(np.asarray(haplotype_posterior, dtype=np.float32) if return_haplotype_posterior else None),
            genotype_posterior=gp_out,
            genotype_call=(np.argmax(gp_out, axis=2).astype(np.int8, copy=False) if gp_out is not None else None),
            recombination_rate=np.asarray(rates, dtype=np.float32),
            switch_probability=switch.astype(np.float32),
            stay_probability=stay.astype(np.float32),
            offdiag_probability=offdiag.astype(np.float32),
            founder_alt_prob=np.asarray(working_alt_prob, dtype=np.float32),
            transition_probability=transition_probability,
            em_diagnostics=em_diagnostics,
        )

    def run(
        self,
        founder_panel: FounderPanel,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        generations: np.ndarray,
        *,
        other_count: np.ndarray | None = None,
        ref_weight: np.ndarray | None = None,
        alt_weight: np.ndarray | None = None,
        other_weight: np.ndarray | None = None,
        return_full_transition: bool = False,
        return_haplotype_posterior: bool = False,
        return_genotype_posterior: bool = False,
        fragment_sample_offsets: np.ndarray | None = None,
        fragment_center_idx: np.ndarray | None = None,
        fragment_obs_offsets: np.ndarray | None = None,
        fragment_obs_pos_idx: np.ndarray | None = None,
        fragment_obs_code: np.ndarray | None = None,
        fragment_obs_qual: np.ndarray | None = None,
    ) -> HMMArtifacts:
        if self.backend == "jax" and (jax is None or jnp is None):
            self.backend = "numpy"
        if self.backend == "torch" and not _torch_available():
            self.backend = "numpy"

        k = founder_panel.n_founders
        if int(self.config.em_multistarts) > 1 and not bool(np.all(founder_panel.immutable_mask)):
            return self._run_multistart(
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
        ploidy = 1 if self.config.ploidy_mode == "pseudo_haploid" else int(self.config.ploidy)
        use_generic = bool(self.config.force_generic_ploidy_hmm) or (self.config.ploidy_mode == "diploid" and ploidy >= 3)
        if use_generic:
            return self._run_polyploid(
                ploidy=ploidy,
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
        rates = self.recombination_from_positions(founder_panel.positions)
        switch = self._switch_probabilities(rates, generations)
        stay = 1.0 - switch
        offdiag = switch / max(k - 1, 1)
        working_alt_prob = founder_panel.alt_prob.astype(np.float32, copy=True)
        founder_immutable_mask = founder_panel.immutable_mask
        haplotype_posterior: np.ndarray | None = None
        genotype_posterior: np.ndarray | None = None
        founder_alt: np.ndarray | None = None
        dosage: np.ndarray | None = None
        n_samples = ref_count.shape[0]
        n_positions = ref_count.shape[1]
        if self.backend == "jax" and self.config.ploidy_mode == "diploid":
            batch_size = self._resolve_jax_sample_batch_size(n_samples)
            self._maybe_precompile(batch_size, n_positions, k)
            remainder = n_samples % batch_size
            if remainder != 0:
                self._maybe_precompile(remainder, n_positions, k)
        else:
            self._maybe_precompile(n_samples, n_positions, k)

        ref_obs_all, alt_obs_all, other_obs_all = self._observed_counts(
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
        )
        ref_obs_all = ref_obs_all.astype(np.float32, copy=False)
        alt_obs_all = alt_obs_all.astype(np.float32, copy=False)
        other_obs_all = other_obs_all.astype(np.float32, copy=False)
        sample_alt_fraction = ((alt_obs_all + 0.5) / (ref_obs_all + alt_obs_all + other_obs_all + 1.0)).astype(
            np.float32,
            copy=False,
        )
        site_depth = np.sum(ref_obs_all + alt_obs_all + other_obs_all, axis=0)
        site_alt_fraction = np.divide(
            np.sum(alt_obs_all, axis=0) + 0.5,
            site_depth + 1.0,
            out=np.full(n_positions, 0.5, dtype=np.float32),
            where=site_depth > 0,
        ).astype(np.float32, copy=False)
        working_alt_prob = self._initialize_founder_alt_prob(
            founder_alt_prob=working_alt_prob,
            founder_immutable_mask=founder_immutable_mask,
            site_alt_fraction=site_alt_fraction,
        )

        max_em_iterations = max(int(self.config.em_iterations), 1)
        all_founders_immutable = bool(np.all(founder_immutable_mask))
        final_pass_pending = all_founders_immutable or max_em_iterations == 1
        converged_passes = 0
        em_iter = 0
        em_history: list[dict[str, object]] = []
        best_founder_alt_prob = working_alt_prob.copy()
        best_founder_max_delta = float("inf")
        best_iteration = -1
        restored_best_founders = False
        while True:
            if not final_pass_pending and em_iter >= max_em_iterations:
                if (
                    bool(self.config.adaptive_em_restore_best_founders)
                    and best_iteration >= 0
                    and best_founder_alt_prob is not None
                ):
                    working_alt_prob = best_founder_alt_prob.copy()
                    restored_best_founders = True
                final_pass_pending = True
            is_last_iteration = final_pass_pending
            if (
                self.config.ploidy_mode == "pseudo_haploid"
                and self.backend == "jax"
                and bool(self.config.jax_count_emission_kernel)
                and (
                    not self._has_fragment_batch(fragment_center_idx)
                    or (
                        bool(self.config.jax_fragment_emission_kernel)
                        and fragment_sample_offsets is not None
                        and fragment_obs_offsets is not None
                        and fragment_obs_pos_idx is not None
                        and fragment_obs_code is not None
                    )
                )
            ):
                batch_size = self._resolve_jax_sample_batch_size(n_samples)
                numerator = np.zeros((k, n_positions), dtype=np.float32)
                denominator = np.zeros((k, n_positions), dtype=np.float32)
                if is_last_iteration:
                    dosage = np.empty((n_samples, n_positions), dtype=np.float32)
                    if return_haplotype_posterior:
                        haplotype_posterior = np.empty((n_samples, n_positions, k), dtype=np.float32)
                    if return_genotype_posterior:
                        genotype_posterior = np.empty((n_samples, n_positions, 2), dtype=np.float32)
                for sample_start in range(0, n_samples, batch_size):
                    sample_stop = min(sample_start + batch_size, n_samples)
                    n_batch = sample_stop - sample_start
                    founder_alt_f32 = working_alt_prob.astype(np.float32, copy=False)
                    (
                        fragment_sample_offsets_batch,
                        fragment_center_idx_batch,
                        fragment_obs_offsets_batch,
                        fragment_obs_pos_idx_batch,
                        fragment_obs_code_batch,
                        fragment_obs_qual_batch,
                    ) = self._slice_fragment_batch(
                        sample_start=sample_start,
                        sample_stop=sample_stop,
                        fragment_sample_offsets=fragment_sample_offsets,
                        fragment_center_idx=fragment_center_idx,
                        fragment_obs_offsets=fragment_obs_offsets,
                        fragment_obs_pos_idx=fragment_obs_pos_idx,
                        fragment_obs_code=fragment_obs_code,
                        fragment_obs_qual=fragment_obs_qual,
                    )
                    has_fragment_batch = self._has_fragment_batch(fragment_center_idx_batch)
                    use_fragment_count_kernel = (
                        has_fragment_batch
                        and bool(self.config.jax_fragment_emission_kernel)
                        and fragment_sample_offsets_batch is not None
                        and fragment_obs_offsets_batch is not None
                        and fragment_obs_pos_idx_batch is not None
                        and fragment_obs_code_batch is not None
                    )
                    if is_last_iteration and dosage is not None:
                        if use_fragment_count_kernel:
                            (
                                fragment_sample_idx,
                                fragment_obs_fragment_idx,
                                fragment_obs_qual_f32,
                            ) = self._jax_fragment_index_arrays(
                                fragment_sample_offsets_batch,
                                fragment_center_idx_batch,
                                fragment_obs_offsets_batch,
                                fragment_obs_qual_batch,
                            )
                            if all_founders_immutable:
                                fn = self._get_jax_haploid_fragment_count_output_callable(
                                    n_batch,
                                    n_positions,
                                    k,
                                    int(fragment_center_idx_batch.shape[0]),
                                    int(fragment_obs_pos_idx_batch.shape[0]),
                                )
                                hap_gamma_jax, dosage_batch_jax, gp_batch_jax = fn(
                                    jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                    jnp.asarray(fragment_sample_idx, dtype=jnp.int32),
                                    jnp.asarray(fragment_center_idx_batch, dtype=jnp.int32),
                                    jnp.asarray(fragment_obs_fragment_idx, dtype=jnp.int32),
                                    jnp.asarray(fragment_obs_pos_idx_batch, dtype=jnp.int32),
                                    jnp.asarray(fragment_obs_code_batch, dtype=jnp.int8),
                                    jnp.asarray(fragment_obs_qual_f32, dtype=jnp.float32),
                                    jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                    jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                                    jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32),
                                    jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32),
                                    jnp.asarray(bool(self.config.fragment_rescale_read_likelihood), dtype=jnp.bool_),
                                    jnp.asarray(self.config.fragment_likelihood_mode == "replace", dtype=jnp.bool_),
                                )
                                jax.block_until_ready((dosage_batch_jax, gp_batch_jax))
                            else:
                                fn = self._get_jax_haploid_fragment_count_final_callable(
                                    n_batch,
                                    n_positions,
                                    k,
                                    int(fragment_center_idx_batch.shape[0]),
                                    int(fragment_obs_pos_idx_batch.shape[0]),
                                )
                                (
                                    numerator_jax,
                                    denominator_jax,
                                    hap_gamma_jax,
                                    dosage_batch_jax,
                                    gp_batch_jax,
                                ) = fn(
                                    jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                    jnp.asarray(fragment_sample_idx, dtype=jnp.int32),
                                    jnp.asarray(fragment_center_idx_batch, dtype=jnp.int32),
                                    jnp.asarray(fragment_obs_fragment_idx, dtype=jnp.int32),
                                    jnp.asarray(fragment_obs_pos_idx_batch, dtype=jnp.int32),
                                    jnp.asarray(fragment_obs_code_batch, dtype=jnp.int8),
                                    jnp.asarray(fragment_obs_qual_f32, dtype=jnp.float32),
                                    jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                    jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                                    jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32),
                                    jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32),
                                    jnp.asarray(bool(self.config.fragment_rescale_read_likelihood), dtype=jnp.bool_),
                                    jnp.asarray(self.config.fragment_likelihood_mode == "replace", dtype=jnp.bool_),
                                )
                                jax.block_until_ready((numerator_jax, denominator_jax, dosage_batch_jax, gp_batch_jax))
                                numerator += np.asarray(numerator_jax, dtype=np.float32)
                                denominator += np.asarray(denominator_jax, dtype=np.float32)
                        elif all_founders_immutable:
                            fn = self._get_jax_haploid_count_output_callable(n_batch, n_positions, k)
                            hap_gamma_jax, dosage_batch_jax, gp_batch_jax = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                            )
                            jax.block_until_ready((dosage_batch_jax, gp_batch_jax))
                        else:
                            fn = self._get_jax_haploid_count_final_callable(n_batch, n_positions, k)
                            (
                                numerator_jax,
                                denominator_jax,
                                hap_gamma_jax,
                                dosage_batch_jax,
                                gp_batch_jax,
                            ) = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                            )
                            jax.block_until_ready((numerator_jax, denominator_jax, dosage_batch_jax, gp_batch_jax))
                            numerator += np.asarray(numerator_jax, dtype=np.float32)
                            denominator += np.asarray(denominator_jax, dtype=np.float32)
                        dosage[sample_start:sample_stop] = np.asarray(dosage_batch_jax, dtype=np.float32)
                        if return_haplotype_posterior and haplotype_posterior is not None:
                            haplotype_posterior[sample_start:sample_stop] = np.asarray(hap_gamma_jax, dtype=np.float32)
                        if return_genotype_posterior and genotype_posterior is not None:
                            genotype_posterior[sample_start:sample_stop] = np.asarray(gp_batch_jax, dtype=np.float32)
                    else:
                        if use_fragment_count_kernel:
                            (
                                fragment_sample_idx,
                                fragment_obs_fragment_idx,
                                fragment_obs_qual_f32,
                            ) = self._jax_fragment_index_arrays(
                                fragment_sample_offsets_batch,
                                fragment_center_idx_batch,
                                fragment_obs_offsets_batch,
                                fragment_obs_qual_batch,
                            )
                            fn = self._get_jax_haploid_fragment_count_stats_callable(
                                n_batch,
                                n_positions,
                                k,
                                int(fragment_center_idx_batch.shape[0]),
                                int(fragment_obs_pos_idx_batch.shape[0]),
                            )
                            numerator_jax, denominator_jax = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(working_alt_prob.astype(np.float32, copy=False), dtype=jnp.float32),
                                jnp.asarray(fragment_sample_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_center_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_fragment_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_pos_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_code_batch, dtype=jnp.int8),
                                jnp.asarray(fragment_obs_qual_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32),
                                jnp.asarray(bool(self.config.fragment_rescale_read_likelihood), dtype=jnp.bool_),
                                jnp.asarray(self.config.fragment_likelihood_mode == "replace", dtype=jnp.bool_),
                            )
                        else:
                            fn = self._get_jax_haploid_count_stats_callable(n_batch, n_positions, k)
                            numerator_jax, denominator_jax = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(working_alt_prob.astype(np.float32, copy=False), dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                            )
                        jax.block_until_ready((numerator_jax, denominator_jax))
                        numerator += np.asarray(numerator_jax, dtype=np.float32)
                        denominator += np.asarray(denominator_jax, dtype=np.float32)
                if not all_founders_immutable and not is_last_iteration:
                    previous_alt_prob = working_alt_prob
                    working_alt_prob = self._update_founders_from_stats(
                        founder_alt_prob=working_alt_prob,
                        founder_immutable_mask=founder_immutable_mask,
                        numerator=numerator,
                        denominator=denominator,
                    )
                    max_delta, mean_delta = self._founder_update_delta(previous_alt_prob, working_alt_prob, founder_immutable_mask)
                    if bool(self.config.adaptive_em):
                        em_history.append(
                            {
                                "iteration": int(em_iter + 1),
                                "founder_max_delta": float(max_delta),
                                "founder_mean_delta": float(mean_delta),
                                "read_log_likelihood": None,
                                "masked_genotype_likelihood": None,
                            }
                        )
                        if float(max_delta) < best_founder_max_delta:
                            best_founder_max_delta = float(max_delta)
                            best_founder_alt_prob = working_alt_prob.copy()
                            best_iteration = int(em_iter + 1)
                    if self._em_converged(iteration_count=em_iter + 1, max_delta=max_delta):
                        converged_passes += 1
                        if converged_passes >= max(int(self.config.em_convergence_patience), 1):
                            if (
                                bool(self.config.adaptive_em_restore_best_founders)
                                and best_iteration >= 0
                                and best_founder_alt_prob is not None
                            ):
                                working_alt_prob = best_founder_alt_prob.copy()
                                restored_best_founders = True
                            final_pass_pending = True
                    else:
                        converged_passes = 0
                if is_last_iteration:
                    break
                em_iter += 1
                continue
            if self.config.ploidy_mode == "diploid" and self.backend == "jax":
                batch_size = self._resolve_jax_sample_batch_size(n_samples)
                numerator = np.zeros((k, n_positions), dtype=np.float32)
                denominator = np.zeros((k, n_positions), dtype=np.float32)
                if is_last_iteration:
                    dosage = np.empty((n_samples, n_positions), dtype=np.float32)
                    if return_haplotype_posterior:
                        haplotype_posterior = np.empty((n_samples, n_positions, k), dtype=np.float32)
                    if return_genotype_posterior:
                        genotype_posterior = np.empty((n_samples, n_positions, 3), dtype=np.float32)

                for sample_start in range(0, n_samples, batch_size):
                    sample_stop = min(sample_start + batch_size, n_samples)
                    n_batch = sample_stop - sample_start
                    ref_batch = ref_count[sample_start:sample_stop]
                    alt_batch = alt_count[sample_start:sample_stop]
                    other_batch = None if other_count is None else other_count[sample_start:sample_stop]
                    ref_weight_batch = None if ref_weight is None else ref_weight[sample_start:sample_stop]
                    alt_weight_batch = None if alt_weight is None else alt_weight[sample_start:sample_stop]
                    other_weight_batch = None if other_weight is None else other_weight[sample_start:sample_stop]
                    (
                        fragment_sample_offsets_batch,
                        fragment_center_idx_batch,
                        fragment_obs_offsets_batch,
                        fragment_obs_pos_idx_batch,
                        fragment_obs_code_batch,
                        fragment_obs_qual_batch,
                    ) = self._slice_fragment_batch(
                        sample_start=sample_start,
                        sample_stop=sample_stop,
                        fragment_sample_offsets=fragment_sample_offsets,
                        fragment_center_idx=fragment_center_idx,
                        fragment_obs_offsets=fragment_obs_offsets,
                        fragment_obs_pos_idx=fragment_obs_pos_idx,
                        fragment_obs_code=fragment_obs_code,
                        fragment_obs_qual=fragment_obs_qual,
                    )
                    use_count_kernel = (
                        bool(self.config.jax_count_emission_kernel)
                    )
                    has_fragment_batch = self._has_fragment_batch(fragment_center_idx_batch)
                    use_fragment_count_kernel = (
                        use_count_kernel
                        and has_fragment_batch
                        and bool(self.config.jax_fragment_emission_kernel)
                        and fragment_sample_offsets_batch is not None
                        and fragment_obs_offsets_batch is not None
                        and fragment_obs_pos_idx_batch is not None
                        and fragment_obs_code_batch is not None
                    )
                    use_plain_count_kernel = use_count_kernel and not has_fragment_batch

                    if is_last_iteration and dosage is not None:
                        founder_alt_f32 = working_alt_prob.astype(np.float32, copy=False)
                        if use_fragment_count_kernel and all_founders_immutable:
                            (
                                fragment_sample_idx,
                                fragment_obs_fragment_idx,
                                fragment_obs_qual_f32,
                            ) = self._jax_fragment_index_arrays(
                                fragment_sample_offsets_batch,
                                fragment_center_idx_batch,
                                fragment_obs_offsets_batch,
                                fragment_obs_qual_batch,
                            )
                            fn = self._get_jax_diploid_fragment_count_output_callable(
                                n_batch,
                                n_positions,
                                k,
                                int(fragment_center_idx_batch.shape[0]),
                                int(fragment_obs_pos_idx_batch.shape[0]),
                            )
                            hap_gamma_jax, dosage_batch_jax, gp_batch_jax = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                jnp.asarray(fragment_sample_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_center_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_fragment_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_pos_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_code_batch, dtype=jnp.int8),
                                jnp.asarray(fragment_obs_qual_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32),
                                jnp.asarray(bool(self.config.fragment_rescale_read_likelihood), dtype=jnp.bool_),
                                jnp.asarray(self.config.fragment_likelihood_mode == "replace", dtype=jnp.bool_),
                            )
                            jax.block_until_ready((dosage_batch_jax, gp_batch_jax))
                        elif use_fragment_count_kernel:
                            (
                                fragment_sample_idx,
                                fragment_obs_fragment_idx,
                                fragment_obs_qual_f32,
                            ) = self._jax_fragment_index_arrays(
                                fragment_sample_offsets_batch,
                                fragment_center_idx_batch,
                                fragment_obs_offsets_batch,
                                fragment_obs_qual_batch,
                            )
                            fn = self._get_jax_diploid_fragment_count_final_callable(
                                n_batch,
                                n_positions,
                                k,
                                int(fragment_center_idx_batch.shape[0]),
                                int(fragment_obs_pos_idx_batch.shape[0]),
                            )
                            (
                                numerator_jax,
                                denominator_jax,
                                hap_gamma_jax,
                                dosage_batch_jax,
                                gp_batch_jax,
                            ) = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                jnp.asarray(fragment_sample_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_center_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_fragment_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_pos_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_code_batch, dtype=jnp.int8),
                                jnp.asarray(fragment_obs_qual_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32),
                                jnp.asarray(bool(self.config.fragment_rescale_read_likelihood), dtype=jnp.bool_),
                                jnp.asarray(self.config.fragment_likelihood_mode == "replace", dtype=jnp.bool_),
                            )
                            jax.block_until_ready((numerator_jax, denominator_jax, dosage_batch_jax, gp_batch_jax))
                            numerator += np.asarray(numerator_jax, dtype=np.float32)
                            denominator += np.asarray(denominator_jax, dtype=np.float32)
                        elif use_plain_count_kernel and all_founders_immutable:
                            fn = self._get_jax_diploid_count_output_callable(n_batch, n_positions, k)
                            hap_gamma_jax, dosage_batch_jax, gp_batch_jax = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                            )
                            jax.block_until_ready((dosage_batch_jax, gp_batch_jax))
                        elif use_plain_count_kernel:
                            fn = self._get_jax_diploid_count_final_callable(n_batch, n_positions, k)
                            (
                                numerator_jax,
                                denominator_jax,
                                hap_gamma_jax,
                                dosage_batch_jax,
                                gp_batch_jax,
                            ) = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                            )
                            jax.block_until_ready((numerator_jax, denominator_jax, dosage_batch_jax, gp_batch_jax))
                            numerator += np.asarray(numerator_jax, dtype=np.float32)
                            denominator += np.asarray(denominator_jax, dtype=np.float32)
                        else:
                            log_emission_batch, _, founder_alt = self.emissions(
                                working_alt_prob,
                                ref_batch,
                                alt_batch,
                                other_batch,
                                ref_weight_batch,
                                alt_weight_batch,
                                other_weight_batch,
                            )
                            log_emission_batch = self._apply_fragment_likelihoods(
                                log_emission=log_emission_batch,
                                founder_alt=founder_alt.astype(np.float32, copy=False),
                                fragment_sample_offsets=fragment_sample_offsets_batch,
                                fragment_center_idx=fragment_center_idx_batch,
                                fragment_obs_offsets=fragment_obs_offsets_batch,
                                fragment_obs_pos_idx=fragment_obs_pos_idx_batch,
                                fragment_obs_code=fragment_obs_code_batch,
                                fragment_obs_qual=fragment_obs_qual_batch,
                            )
                            if all_founders_immutable:
                                fn = self._get_jax_diploid_output_callable(n_batch, n_positions, k)
                                hap_gamma_jax, dosage_batch_jax, gp_batch_jax = fn(
                                    jnp.asarray(log_emission_batch, dtype=jnp.float32),
                                    jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                )
                                jax.block_until_ready((dosage_batch_jax, gp_batch_jax))
                            else:
                                fn = self._get_jax_diploid_final_callable(n_batch, n_positions, k)
                                (
                                    numerator_jax,
                                    denominator_jax,
                                    hap_gamma_jax,
                                    dosage_batch_jax,
                                    gp_batch_jax,
                                ) = fn(
                                    jnp.asarray(log_emission_batch, dtype=jnp.float32),
                                    jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                    jnp.asarray(founder_alt_f32, dtype=jnp.float32),
                                )
                                jax.block_until_ready((numerator_jax, denominator_jax, dosage_batch_jax, gp_batch_jax))
                                numerator += np.asarray(numerator_jax, dtype=np.float32)
                                denominator += np.asarray(denominator_jax, dtype=np.float32)
                        dosage[sample_start:sample_stop] = np.asarray(dosage_batch_jax, dtype=np.float32)
                        if return_haplotype_posterior and haplotype_posterior is not None:
                            haplotype_posterior[sample_start:sample_stop] = np.asarray(hap_gamma_jax, dtype=np.float32)
                        if return_genotype_posterior and genotype_posterior is not None:
                            genotype_posterior[sample_start:sample_stop] = np.asarray(
                                gp_batch_jax,
                                dtype=np.float32,
                            )
                    else:
                        if use_fragment_count_kernel:
                            (
                                fragment_sample_idx,
                                fragment_obs_fragment_idx,
                                fragment_obs_qual_f32,
                            ) = self._jax_fragment_index_arrays(
                                fragment_sample_offsets_batch,
                                fragment_center_idx_batch,
                                fragment_obs_offsets_batch,
                                fragment_obs_qual_batch,
                            )
                            fn = self._get_jax_diploid_fragment_count_stats_callable(
                                n_batch,
                                n_positions,
                                k,
                                int(fragment_center_idx_batch.shape[0]),
                                int(fragment_obs_pos_idx_batch.shape[0]),
                            )
                            numerator_jax, denominator_jax = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(working_alt_prob.astype(np.float32, copy=False), dtype=jnp.float32),
                                jnp.asarray(fragment_sample_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_center_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_fragment_idx, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_pos_idx_batch, dtype=jnp.int32),
                                jnp.asarray(fragment_obs_code_batch, dtype=jnp.int8),
                                jnp.asarray(fragment_obs_qual_f32, dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_difference_between_reads), dtype=jnp.float32),
                                jnp.asarray(float(self.config.fragment_max_emission_matrix_difference), dtype=jnp.float32),
                                jnp.asarray(bool(self.config.fragment_rescale_read_likelihood), dtype=jnp.bool_),
                                jnp.asarray(self.config.fragment_likelihood_mode == "replace", dtype=jnp.bool_),
                            )
                        elif use_plain_count_kernel:
                            fn = self._get_jax_diploid_count_stats_callable(n_batch, n_positions, k)
                            numerator_jax, denominator_jax = fn(
                                jnp.asarray(ref_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(alt_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(other_obs_all[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(working_alt_prob.astype(np.float32, copy=False), dtype=jnp.float32),
                                jnp.asarray(float(self.config.sequencing_error_rate), dtype=jnp.float32),
                                jnp.asarray(float(self.config.min_emission_prob), dtype=jnp.float32),
                            )
                        else:
                            log_emission_batch, _, founder_alt = self.emissions(
                                working_alt_prob,
                                ref_batch,
                                alt_batch,
                                other_batch,
                                ref_weight_batch,
                                alt_weight_batch,
                                other_weight_batch,
                            )
                            log_emission_batch = self._apply_fragment_likelihoods(
                                log_emission=log_emission_batch,
                                founder_alt=founder_alt.astype(np.float32, copy=False),
                                fragment_sample_offsets=fragment_sample_offsets_batch,
                                fragment_center_idx=fragment_center_idx_batch,
                                fragment_obs_offsets=fragment_obs_offsets_batch,
                                fragment_obs_pos_idx=fragment_obs_pos_idx_batch,
                                fragment_obs_code=fragment_obs_code_batch,
                                fragment_obs_qual=fragment_obs_qual_batch,
                            )
                            fn = self._get_jax_diploid_stats_callable(n_batch, n_positions, k)
                            numerator_jax, denominator_jax = fn(
                                jnp.asarray(log_emission_batch, dtype=jnp.float32),
                                jnp.asarray(switch[sample_start:sample_stop], dtype=jnp.float32),
                                jnp.asarray(sample_alt_fraction[sample_start:sample_stop], dtype=jnp.float32),
                            )
                        jax.block_until_ready((numerator_jax, denominator_jax))
                        numerator += np.asarray(numerator_jax, dtype=np.float32)
                        denominator += np.asarray(denominator_jax, dtype=np.float32)

                if not all_founders_immutable and not is_last_iteration:
                    previous_alt_prob = working_alt_prob
                    working_alt_prob = self._update_founders_from_stats(
                        founder_alt_prob=working_alt_prob,
                        founder_immutable_mask=founder_immutable_mask,
                        numerator=numerator,
                        denominator=denominator,
                    )
                    max_delta, mean_delta = self._founder_update_delta(previous_alt_prob, working_alt_prob, founder_immutable_mask)
                    if bool(self.config.adaptive_em):
                        em_history.append(
                            {
                                "iteration": int(em_iter + 1),
                                "founder_max_delta": float(max_delta),
                                "founder_mean_delta": float(mean_delta),
                                "read_log_likelihood": None,
                                "masked_genotype_likelihood": None,
                            }
                        )
                        if float(max_delta) < best_founder_max_delta:
                            best_founder_max_delta = float(max_delta)
                            best_founder_alt_prob = working_alt_prob.copy()
                            best_iteration = int(em_iter + 1)
                    if self._em_converged(iteration_count=em_iter + 1, max_delta=max_delta):
                        converged_passes += 1
                        if converged_passes >= max(int(self.config.em_convergence_patience), 1):
                            if (
                                bool(self.config.adaptive_em_restore_best_founders)
                                and best_iteration >= 0
                                and best_founder_alt_prob is not None
                            ):
                                working_alt_prob = best_founder_alt_prob.copy()
                                restored_best_founders = True
                            final_pass_pending = True
                    else:
                        converged_passes = 0
                if is_last_iteration:
                    break
                em_iter += 1
                continue

            log_emission, _, founder_alt = self.emissions(
                working_alt_prob,
                ref_count,
                alt_count,
                other_count,
                ref_weight,
                alt_weight,
                other_weight,
            )
            log_emission = self._apply_fragment_likelihoods(
                log_emission=log_emission,
                founder_alt=founder_alt.astype(np.float32, copy=False),
                fragment_sample_offsets=fragment_sample_offsets,
                fragment_center_idx=fragment_center_idx,
                fragment_obs_offsets=fragment_obs_offsets,
                fragment_obs_pos_idx=fragment_obs_pos_idx,
                fragment_obs_code=fragment_obs_code,
                fragment_obs_qual=fragment_obs_qual,
            )
            if self.config.ploidy_mode == "diploid":
                if self.backend == "torch":
                    gamma = self._forward_backward_torch_diploid(log_emission, switch, k)
                    haplotype_posterior = _diploid_pair_to_haplotype_posterior(gamma)
                else:
                    gamma = self._forward_backward_numpy_diploid(log_emission, switch, k)
                    haplotype_posterior = _diploid_pair_to_haplotype_posterior(gamma)
            else:
                if self.backend == "torch":
                    haplotype_posterior = self._forward_backward_torch_haploid(log_emission, switch, k)
                else:
                    haplotype_posterior = self._forward_backward_numpy_haploid(log_emission, switch, k)

            if haplotype_posterior is None:
                raise RuntimeError("HMM posterior step returned no posterior.")

            if not all_founders_immutable and not is_last_iteration:
                numerator = np.einsum("spk,sp->kp", haplotype_posterior, sample_alt_fraction, optimize=True)
                denominator = np.einsum("spk->kp", haplotype_posterior, optimize=True)
                previous_alt_prob = working_alt_prob
                working_alt_prob = self._update_founders_from_stats(
                    founder_alt_prob=working_alt_prob,
                    founder_immutable_mask=founder_immutable_mask,
                    numerator=numerator,
                    denominator=denominator,
                )
                max_delta, mean_delta = self._founder_update_delta(previous_alt_prob, working_alt_prob, founder_immutable_mask)
                if bool(self.config.adaptive_em):
                    em_history.append(
                        {
                            "iteration": int(em_iter + 1),
                            "founder_max_delta": float(max_delta),
                            "founder_mean_delta": float(mean_delta),
                            "read_log_likelihood": None,
                            "masked_genotype_likelihood": None,
                        }
                    )
                    if float(max_delta) < best_founder_max_delta:
                        best_founder_max_delta = float(max_delta)
                        best_founder_alt_prob = working_alt_prob.copy()
                        best_iteration = int(em_iter + 1)
                if self._em_converged(iteration_count=em_iter + 1, max_delta=max_delta):
                    converged_passes += 1
                    if converged_passes >= max(int(self.config.em_convergence_patience), 1):
                        if (
                            bool(self.config.adaptive_em_restore_best_founders)
                            and best_iteration >= 0
                            and best_founder_alt_prob is not None
                        ):
                            working_alt_prob = best_founder_alt_prob.copy()
                            restored_best_founders = True
                        final_pass_pending = True
                else:
                    converged_passes = 0

            if is_last_iteration:
                founder_alt_f32 = founder_alt.astype(np.float32, copy=False)
                if self.config.ploidy_mode == "pseudo_haploid":
                    dosage = np.einsum("spk,kp->sp", haplotype_posterior, founder_alt_f32, optimize=True)
                    if return_genotype_posterior:
                        genotype_posterior = self._genotype_posterior_from_haplotype_posterior(
                            hap_posterior=haplotype_posterior,
                            founder_alt=founder_alt_f32,
                        )
                else:
                    dosage = 2.0 * np.einsum("spk,kp->sp", haplotype_posterior, founder_alt_f32, optimize=True)
                    if return_genotype_posterior:
                        genotype_posterior = self._genotype_posterior_from_pair_gamma(
                            pair_gamma=gamma,
                            founder_alt=founder_alt_f32,
                        )
                if not return_haplotype_posterior:
                    haplotype_posterior = None
                break
            em_iter += 1

        if dosage is None:
            raise RuntimeError("HMM did not produce dosage output.")

        if return_genotype_posterior and genotype_posterior is None:
            p_alt = np.clip(np.asarray(dosage, dtype=np.float32) * 0.5, 1e-6, 1.0 - 1e-6)
            gp0 = (1.0 - p_alt) ** 2
            gp1 = 2.0 * p_alt * (1.0 - p_alt)
            gp2 = p_alt**2
            genotype_posterior = np.stack([gp0, gp1, gp2], axis=2).astype(np.float32, copy=False)
            genotype_posterior /= np.clip(np.sum(genotype_posterior, axis=2, keepdims=True), 1e-8, None)

        transition_probability = self._build_full_transitions(switch, k) if return_full_transition else None
        hap_out = np.asarray(haplotype_posterior, dtype=np.float32) if return_haplotype_posterior else None
        gp_out = np.asarray(genotype_posterior, dtype=np.float32) if return_genotype_posterior else None
        gt_out = np.argmax(gp_out, axis=2).astype(np.int8, copy=False) if gp_out is not None else None
        final_read_ll = self._read_log_likelihood_total(
            dosage=np.asarray(dosage, dtype=np.float32),
            genotype_posterior=gp_out,
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            ploidy=ploidy,
        )
        em_diagnostics = {
            "adaptive_em": bool(self.config.adaptive_em),
            "all_founders_immutable": bool(all_founders_immutable),
            "em_updates": int(len(em_history)),
            "em_iterations_requested": int(max_em_iterations),
            "history": em_history,
            "best_iteration": int(best_iteration),
            "best_founder_max_delta": float(best_founder_max_delta) if np.isfinite(best_founder_max_delta) else None,
            "final_founder_max_delta": (
                float(em_history[-1]["founder_max_delta"]) if em_history else 0.0
            ),
            "restored_best_founders": bool(restored_best_founders),
            "final_read_log_likelihood": float(final_read_ll),
            "masked_genotype_likelihood": None,
        }
        return HMMArtifacts(
            dosage=np.asarray(dosage, dtype=np.float32),
            haplotype_posterior=hap_out,
            genotype_posterior=gp_out,
            genotype_call=gt_out,
            recombination_rate=np.asarray(rates, dtype=np.float32),
            switch_probability=switch.astype(np.float32),
            stay_probability=stay.astype(np.float32),
            offdiag_probability=offdiag.astype(np.float32),
            founder_alt_prob=np.asarray(working_alt_prob, dtype=np.float32),
            transition_probability=transition_probability,
            em_diagnostics=em_diagnostics,
        )
