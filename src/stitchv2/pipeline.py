from __future__ import annotations

import gc
import json
import math
import os
import resource
import socket
import time
from contextlib import ExitStack
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numcodecs
import pandas as pd
import pyarrow as pa

from .calibration import (
    apply_calibration_sanity_guard,
    calibrate_genotype_posterior,
    calibrate_genotype_posterior_block_context,
    calibrate_genotype_posterior_full_stack,
    dosage_to_genotype_posterior,
    genotype_call_from_posterior,
    masked_cv_calibrate_genotype_posterior,
    train_standard_callability_model,
)
from .config import PipelineConfig
from .diagnostics import (
    compute_variant_diagnostics,
    diagnostic_thresholds,
    write_block_diagnostics,
    write_diagnostics_summary,
)
from .dask_executor import (
    DaskHMMTaskResult,
    load_hmm_artifacts_npz,
    plan_dask_chunks,
    run_hmm_leaf_task,
    write_task_stream_artifact,
)
from .evidence_cache import PartitionedEvidenceCache
from .founders import FounderPanel, load_founders
from .hmm import (
    HMMArtifacts,
    JAXStitchHMM,
    _build_transition_matrix_from_switch,
    _polyploid_transition_coefficients,
    _transition_matrix_from_switch_and_offdiag,
)
from .io import iter_density_balanced_overlap_blocks, iter_position_blocks, load_positions, validate_samples, write_parquet
from .microarray import (
    align_microarray_to_samples,
    load_microarray_hardcalls_from_plink,
    load_microarray_hardcalls_from_sample_plink_paths,
)
from .pedigree import apply_pedigree_adjustment, coerce_pedigree_graph, has_pedigree_columns
from .pileup import PysamReadExtractor, ReadEvidenceBlock


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
    # Fallback to peak RSS when current RSS is unavailable.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _validate_local_dask_socket_support() -> None:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
        finally:
            sock.close()
    except OSError as exc:
        raise RuntimeError(
            "Local Dask requires local socket creation for its scheduler/workers, "
            "but this runtime cannot bind a loopback socket. Run outside a "
            "socket-restricted sandbox/container or use a supported external "
            "scheduler/jobqueue backend."
        ) from exc


def _jax_accelerator_count() -> int:
    try:
        import jax

        return int(sum(device.platform in {"gpu", "tpu"} for device in jax.devices()))
    except Exception:
        return 0


def _resolve_io_threads(config: PipelineConfig) -> tuple[int, int]:
    """Resolve (io_workers, htslib_threads_per_file) for BAM reading.

    ``io_threads_total`` is the user-facing knob:
      * 0  -> keep the explicit io_workers / htslib_threads_per_file (default).
      * >0 -> treat as the total read-thread budget for this process.
      * <0 -> auto-derive from os.cpu_count(), reserving the cores already
              claimed by the Dask compute pool so reads do not oversubscribe.

    The budget is split as io_workers * htslib_threads_per_file ~= total,
    honoring any explicit htslib_threads_per_file and filling the remainder with
    parallel-sample workers (which scale across cores in the compiled HTSlib
    backend; the pure-Python path is GIL-bound and will not scale as well).
    """
    io_workers = max(int(config.io_workers), 1)
    htslib = max(int(config.htslib_threads_per_file), 1)
    total = int(getattr(config, "io_threads_total", 0) or 0)
    if total == 0:
        return io_workers, htslib
    if total < 0:
        cpu = int(os.cpu_count() or 1)
        if str(config.executor) == "dask":
            workers = int(config.dask_n_workers) or max(1, min(cpu, 4))
            total = max(1, cpu // max(workers, 1))
        else:
            total = max(1, cpu)
    total = max(1, total)
    io_workers = max(1, total // htslib)
    return io_workers, htslib


def _system_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    return 8 * 1024**3


def _parse_memory_budget(value: str | int | float | None) -> int:
    total = _system_memory_bytes()
    if value is None:
        return int(total * 0.90)
    if isinstance(value, (int, float)):
        numeric = float(value)
        if 0.0 < numeric <= 1.0:
            return int(total * numeric)
        return int(numeric)
    text = str(value).strip().lower()
    if not text:
        return int(total * 0.90)
    if text.endswith("%"):
        return int(total * (float(text[:-1]) / 100.0))
    suffixes = {
        "kib": 1024,
        "kb": 1000,
        "k": 1024,
        "mib": 1024**2,
        "mb": 1000**2,
        "m": 1024**2,
        "gib": 1024**3,
        "gb": 1000**3,
        "g": 1024**3,
        "tib": 1024**4,
        "tb": 1000**4,
        "t": 1024**4,
    }
    for suffix, scale in suffixes.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * scale)
    return int(float(text))


def _normalise_store_xi(store_xi: object, write_xi: object | None = None) -> str:
    if write_xi is not None:
        legacy = str(write_xi).strip().lower()
        legacy_map = {
            "force": "full",
            "auto": "per-snp",
            "off": "False",
            "false": "False",
            "none": "False",
        }
        if legacy not in legacy_map:
            raise ValueError("write_xi must be one of: auto, off, force.")
        return legacy_map[legacy]
    if isinstance(store_xi, bool):
        return "per-snp" if store_xi else "False"
    value = str(store_xi).strip()
    lowered = value.lower()
    if lowered == "full":
        return "full"
    if lowered in {"per-snp", "per_snp", "persnp", "summary"}:
        return "per-snp"
    if lowered in {"false", "off", "none", "no", "0"}:
        return "False"
    raise ValueError("store_xi must be one of: full, per-snp, False.")


def _chromosome_match_mask(values: pd.Series, chromosome: str) -> pd.Series:
    target = str(chromosome).strip().lower()
    target_no_chr = target[3:] if target.startswith("chr") else target
    as_text = values.astype(str).str.strip().str.lower()
    as_no_chr = as_text.str.replace(r"^chr", "", regex=True)
    return (as_text == target) | (as_no_chr == target_no_chr)


@dataclass(slots=True)
class RuntimeMemoryPlan:
    max_mem_bytes: int
    planned_budget_bytes: int
    safety_fraction: float
    n_samples: int
    n_read_samples: int
    n_positions: int
    n_founders: int
    max_ploidy: int
    state_count: int
    bytes_per_sample_variant_hmm: int
    bytes_per_sample_variant_io: int
    effective_block_size: int
    effective_io_window_size: int
    effective_sample_batch_size: int
    output_flush_size: int
    calibration_chunk_size: int
    xi_policy: str
    xi_decision: str
    xi_estimated_bytes: int
    gamma_policy: str
    gamma_estimated_bytes: int
    snp_block_mode: str
    approximate: bool
    exact_streaming_requires_boundary_pass: bool
    scratch_dir: str | None
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "max_mem_bytes": int(self.max_mem_bytes),
            "planned_budget_bytes": int(self.planned_budget_bytes),
            "safety_fraction": float(self.safety_fraction),
            "n_samples": int(self.n_samples),
            "n_read_samples": int(self.n_read_samples),
            "n_positions": int(self.n_positions),
            "n_founders": int(self.n_founders),
            "max_ploidy": int(self.max_ploidy),
            "state_count": int(self.state_count),
            "bytes_per_sample_variant_hmm": int(self.bytes_per_sample_variant_hmm),
            "bytes_per_sample_variant_io": int(self.bytes_per_sample_variant_io),
            "effective_block_size": int(self.effective_block_size),
            "effective_io_window_size": int(self.effective_io_window_size),
            "effective_sample_batch_size": int(self.effective_sample_batch_size),
            "output_flush_size": int(self.output_flush_size),
            "calibration_chunk_size": int(self.calibration_chunk_size),
            "xi_policy": str(self.xi_policy),
            "xi_decision": str(self.xi_decision),
            "xi_estimated_bytes": int(self.xi_estimated_bytes),
            "gamma_policy": str(self.gamma_policy),
            "gamma_estimated_bytes": int(self.gamma_estimated_bytes),
            "snp_block_mode": str(self.snp_block_mode),
            "approximate": bool(self.approximate),
            "exact_streaming_requires_boundary_pass": bool(self.exact_streaming_requires_boundary_pass),
            "scratch_dir": self.scratch_dir,
            "notes": list(self.notes),
        }


class StitchPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        if (
            not bool(config.compact_evidence_materialize_dense_counts)
            and (not bool(config.use_fragment_likelihood) or config.fragment_likelihood_mode != "replace")
        ):
            raise ValueError(
                "compact_evidence_materialize_dense_counts=False is only valid with "
                "use_fragment_likelihood=True and fragment_likelihood_mode='replace'."
            )
        if (
            not bool(config.compact_evidence_materialize_dense_counts)
            and (
                bool(config.calibrate_genotype_posteriors)
                or bool(config.write_diagnostics)
                or str(config.genotype_call_mode) == "quality_gated"
            )
        ):
            raise ValueError(
                "compact_evidence_materialize_dense_counts=False creates support-only dense matrices. "
                "It is unsafe with calibration, diagnostics, or quality-gated calls; keep dense materialization "
                "enabled or disable those features for a compact-only debug run."
            )
        self.io_config = config.io()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if config.scratch_dir is not None and str(config.scratch_dir).strip():
            Path(config.scratch_dir).mkdir(parents=True, exist_ok=True)
        if str(config.snp_block_mode) not in {"exact_streaming", "independent_approx", "density_balanced_overlap"}:
            raise ValueError(
                "snp_block_mode must be one of: exact_streaming, independent_approx, density_balanced_overlap."
            )
        if not (0.05 <= float(config.memory_safety_fraction) <= 0.95):
            raise ValueError("memory_safety_fraction must be between 0.05 and 0.95.")
        self._store_xi = _normalise_store_xi(config.store_xi, config.write_xi)
        if str(config.write_gamma) not in {"summary", "off", "full"}:
            raise ValueError("write_gamma must be one of: summary, off, full.")
        self._microarray_dosage: np.ndarray | None = None
        self._sample_ploidy: np.ndarray | None = None
        self.hmm = JAXStitchHMM(config.hmm())
        self._hmm_by_ploidy: dict[int, JAXStitchHMM] = {}
        self.read_extractor = PysamReadExtractor(
            config.chromosome,
            mode=config.read_mode,
            read_stream_backend=config.read_stream_backend,
            min_base_quality=config.min_base_quality,
            min_mapping_quality=config.min_mapping_quality,
            max_insert_size=config.max_insert_size,
            max_indel_len=config.max_indel_len,
            cap_base_quality_by_mapping_quality=config.cap_base_quality_by_mapping_quality,
            ref_alt_only=config.ref_alt_only,
            merge_fragments_by_query=config.merge_fragments_by_query,
            merge_unpaired_fragments_by_query=config.merge_unpaired_fragments_by_query,
            use_bx_tag=config.use_bx_tag,
            bx_tag=config.bx_tag,
            bx_tag_upper_limit=config.bx_tag_upper_limit,
            read_batch_size=config.read_batch_size,
            io_workers=_resolve_io_threads(config)[0],
            htslib_threads_per_file=_resolve_io_threads(config)[1],
            io_window_size=config.io_window_size,
            memory_map_read_matrices=config.memory_map_read_matrices,
            memory_map_dir=config.memory_map_dir,
            subsample_seed=int(config.random_seed),
        )

    def _attach_explicit_genetic_map(self, positions_df: pd.DataFrame) -> pd.DataFrame:
        path_value = self.config.genetic_map_path
        if path_value is None or str(path_value).strip() == "":
            return positions_df
        path = Path(path_value)
        if path.suffix == ".parquet":
            genetic_map = pd.read_parquet(path)
        else:
            genetic_map = pd.read_csv(path, sep=None, engine="python")
        genetic_map = genetic_map.copy()
        genetic_map.columns = [str(col).upper() for col in genetic_map.columns]
        if "CM" not in genetic_map.columns and "GENETIC_CM" in genetic_map.columns:
            genetic_map["CM"] = genetic_map["GENETIC_CM"]
        missing = {"POS", "CM"}.difference(genetic_map.columns)
        if missing:
            raise ValueError(f"genetic map must contain POS and CM columns; missing {sorted(missing)}")
        if "CHR" in genetic_map.columns:
            genetic_map = genetic_map.loc[_chromosome_match_mask(genetic_map["CHR"], self.config.chromosome)].copy()
        genetic_map["POS"] = pd.to_numeric(genetic_map["POS"], errors="coerce")
        genetic_map["CM"] = pd.to_numeric(genetic_map["CM"], errors="coerce")
        genetic_map = genetic_map.loc[np.isfinite(genetic_map["POS"]) & np.isfinite(genetic_map["CM"])].copy()
        genetic_map = genetic_map.sort_values("POS").drop_duplicates("POS", keep="first")
        if genetic_map.shape[0] < 2 or float(genetic_map["CM"].max() - genetic_map["CM"].min()) <= 0.0:
            raise ValueError(f"genetic map {path} has no usable non-zero cM range for {self.config.chromosome}")
        out = positions_df.copy()
        out["CM"] = np.interp(
            out["POS"].to_numpy(dtype=np.float64),
            genetic_map["POS"].to_numpy(dtype=np.float64),
            genetic_map["CM"].to_numpy(dtype=np.float64),
        ).astype(np.float32)
        return out

    def _evidence_cache_metadata(self) -> dict[str, object]:
        return {
            "read_mode": str(self.config.read_mode),
            "read_stream_backend": str(self.config.read_stream_backend),
            "min_base_quality": int(self.config.min_base_quality),
            "min_mapping_quality": int(self.config.min_mapping_quality),
            "max_insert_size": int(self.config.max_insert_size),
            "max_indel_len": int(self.config.max_indel_len),
            "cap_base_quality_by_mapping_quality": bool(self.config.cap_base_quality_by_mapping_quality),
            "ref_alt_only": bool(self.config.ref_alt_only),
            "merge_fragments_by_query": bool(self.config.merge_fragments_by_query),
            "merge_unpaired_fragments_by_query": bool(self.config.merge_unpaired_fragments_by_query),
            "use_bx_tag": bool(self.config.use_bx_tag),
            "bx_tag": str(self.config.bx_tag),
            "bx_tag_upper_limit": int(self.config.bx_tag_upper_limit),
        }

    def _partitioned_cache(self) -> PartitionedEvidenceCache | None:
        if not self.config.compact_evidence_cache_dir:
            return None
        return PartitionedEvidenceCache(
            self.config.compact_evidence_cache_dir,
            chromosome=self.config.chromosome,
            compression=self.config.compression,
            sample_batch_size=int(self.config.compact_evidence_cache_sample_batch_size),
            include_dense_counts=bool(self.config.compact_evidence_cache_include_dense_counts),
            materialize_dense_counts=bool(self.config.compact_evidence_materialize_dense_counts),
            evidence_metadata=self._evidence_cache_metadata(),
        )

    def _maybe_downsample_read_evidence(
        self,
        evidence: ReadEvidenceBlock,
        *,
        block: PositionBlock,
    ) -> tuple[ReadEvidenceBlock, dict[str, object]]:
        downsampled, meta = evidence.downsample_fragments(
            max_depth=int(self.config.downsample_to_coverage),
            fraction=float(self.config.downsample_fraction),
            seed=int(self.config.random_seed) + int(block.block_id) * 1009,
        )
        if downsampled is not evidence:
            evidence.release()
        return downsampled, meta

    def _extract_missing_read_samples(self, read_samples: pd.DataFrame, block: PositionBlock) -> ReadEvidenceBlock | None:
        if int(read_samples.shape[0]) <= 0:
            return None
        extractor = PysamReadExtractor(
            self.config.chromosome,
            mode=self.config.read_mode,
            read_stream_backend=self.config.read_stream_backend,
            min_base_quality=self.config.min_base_quality,
            min_mapping_quality=self.config.min_mapping_quality,
            max_insert_size=self.config.max_insert_size,
            max_indel_len=self.config.max_indel_len,
            cap_base_quality_by_mapping_quality=self.config.cap_base_quality_by_mapping_quality,
            ref_alt_only=self.config.ref_alt_only,
            merge_fragments_by_query=self.config.merge_fragments_by_query,
            merge_unpaired_fragments_by_query=self.config.merge_unpaired_fragments_by_query,
            use_bx_tag=self.config.use_bx_tag,
            bx_tag=self.config.bx_tag,
            bx_tag_upper_limit=self.config.bx_tag_upper_limit,
            read_batch_size=self.config.read_batch_size,
            io_workers=_resolve_io_threads(self.config)[0],
            htslib_threads_per_file=_resolve_io_threads(self.config)[1],
            io_window_size=max(int(block.row_stop) - int(block.row_start), 1),
            memory_map_read_matrices=self.config.memory_map_read_matrices,
            memory_map_dir=self.config.memory_map_dir,
            subsample_seed=int(self.config.random_seed),
        )
        extractor.set_position_table(block.dataframe)
        extractor.open(read_samples)
        try:
            return extractor.extract_block(read_samples, block)
        finally:
            extractor.close()

    def _extract_or_load_read_evidence(
        self,
        *,
        read_samples: pd.DataFrame,
        block: PositionBlock,
    ) -> tuple[ReadEvidenceBlock | None, dict[str, object]]:
        if int(read_samples.shape[0]) <= 0:
            return None, {
                "compact_cache_mode": str(self.config.compact_evidence_cache_mode),
                "compact_cache_hit": False,
                "compact_evidence_bytes": 0,
                "dense_evidence_bytes": 0,
            }
        cache_mode = str(self.config.compact_evidence_cache_mode)
        if cache_mode != "off" and str(self.config.compact_evidence_cache_format) != "parquet_zarr":
            raise ValueError(
                "compact_evidence_cache_format must be 'parquet_zarr'. "
                "STITCHV2 production caching stores compact 2D fragment/support data as compressed Parquet "
                "and optional dense multidimensional arrays as Zarr; NPZ is not supported for benchmark or production runs."
            )
        if cache_mode != "off" and str(self.config.compact_evidence_cache_format) == "parquet_zarr":
            cache = self._partitioned_cache()
            if cache is None:
                extracted = self.read_extractor.extract_block(read_samples, block)
                return extracted, {
                    "compact_cache_mode": cache_mode,
                    "compact_cache_format": "parquet_zarr",
                    "compact_cache_hit": False,
                    "compact_evidence_bytes": int(extracted.compact_nbytes),
                    "dense_evidence_bytes": int(extracted.dense_nbytes),
                }
            requested_ids = read_samples["sample_id"].astype(str).to_numpy(dtype=object)
            cached_evidence = None
            missing_mask = np.ones(requested_ids.shape[0], dtype=bool)
            cache_stats = cache.stats(block)
            if cache_mode in {"read", "readwrite"}:
                cached_evidence, missing_mask, cache_stats = cache.load_available(block, requested_ids)
            missing_evidence = None
            if np.any(missing_mask):
                missing_samples = read_samples.loc[missing_mask].reset_index(drop=True)
                missing_evidence = self._extract_missing_read_samples(missing_samples, block)
                if missing_evidence is not None and cache_mode in {"write", "readwrite"}:
                    cache.write(block, missing_evidence)
                    cache_stats = cache.stats(block)
            if cached_evidence is not None and missing_evidence is not None:
                evidence = ReadEvidenceBlock.merge_sample_blocks(
                    [cached_evidence, missing_evidence],
                    requested_ids,
                    block_id=int(block.block_id),
                )
            elif cached_evidence is not None:
                evidence = cached_evidence
            elif missing_evidence is not None:
                evidence = missing_evidence
            else:
                evidence = self.read_extractor.extract_block(read_samples, block)
            evidence, downsample_stats = self._maybe_downsample_read_evidence(evidence, block=block)
            cache_stats.update(
                {
                    "compact_cache_mode": cache_mode,
                    "compact_cache_format": "parquet_zarr",
                    "compact_cache_hit": bool(cached_evidence is not None and not np.any(missing_mask)),
                    "compact_cache_partial_hit": bool(cached_evidence is not None and np.any(missing_mask)),
                    "compact_cache_missing_samples": int(np.count_nonzero(missing_mask)),
                    "compact_cache_cached_samples": int(requested_ids.shape[0] - np.count_nonzero(missing_mask)),
                    "compact_evidence_bytes": int(evidence.compact_nbytes),
                    "dense_evidence_bytes": int(evidence.dense_nbytes),
                }
            )
            cache_stats.update(downsample_stats)
            return evidence, cache_stats

        evidence = self.read_extractor.extract_block(read_samples, block)
        evidence, downsample_stats = self._maybe_downsample_read_evidence(evidence, block=block)
        stats = {
            "compact_cache_mode": cache_mode,
            "compact_cache_format": "none",
            "compact_cache_hit": False,
            "compact_cache_path": None,
            "compact_cache_file_bytes": 0,
            "compact_cache_includes_dense_counts": False,
            "compact_evidence_bytes": int(evidence.compact_nbytes),
            "dense_evidence_bytes": int(evidence.dense_nbytes),
        }
        stats.update(downsample_stats)
        return evidence, stats

    def _calibration_train_position_index(self, n_positions: int, *, seed: int) -> np.ndarray:
        idx = np.arange(max(int(n_positions), 0), dtype=np.int64)
        if idx.size == 0:
            return idx
        frac = float(np.clip(self.config.calibration_train_site_fraction, 0.0, 1.0))
        if frac >= 1.0:
            return idx
        n_take = max(1, int(np.ceil(frac * float(idx.size))))
        rng = np.random.default_rng(int(seed))
        return np.sort(rng.choice(idx, size=n_take, replace=False).astype(np.int64, copy=False))

    def _hmm_runner_for_ploidy(self, ploidy: int) -> JAXStitchHMM:
        ploidy_i = int(ploidy)
        runner = self._hmm_by_ploidy.get(ploidy_i)
        if runner is None:
            cfg = replace(
                self.config.hmm(),
                ploidy=ploidy_i,
                ploidy_mode=("pseudo_haploid" if ploidy_i == 1 else "diploid"),
                backend_autotune=False,
            )
            runner = JAXStitchHMM(cfg)
            self._hmm_by_ploidy[ploidy_i] = runner
        return runner

    def _estimate_hmm_bytes_per_sample_variant(self, founder_panel: FounderPanel, sample_ploidy: np.ndarray) -> int:
        n_founders = max(int(founder_panel.n_founders), 1)
        max_ploidy = int(np.max(sample_ploidy)) if sample_ploidy.size else max(int(self.config.ploidy), 1)
        max_ploidy = max(max_ploidy, 1)
        genotype_classes = max_ploidy + 1
        bytes_per_cell = 96
        bytes_per_cell += 4 * genotype_classes * 3
        bytes_per_cell += 4 * n_founders * max_ploidy
        bytes_per_cell += 4 * n_founders * n_founders
        if bool(self.config.write_haplotype_probabilities):
            bytes_per_cell += 4 * n_founders
        if bool(self.config.write_genotype_posteriors) or self.config.pedigree_mode in {"kinship", "transmission"}:
            bytes_per_cell += 4 * genotype_classes
        if bool(self.config.use_fragment_likelihood):
            bytes_per_cell += 96
        return max(int(bytes_per_cell), 256)

    def _estimate_io_bytes_per_sample_variant(self) -> int:
        # Dense count/weight evidence is 20 bytes per sample/SNP before fragment vectors;
        # the extra headroom covers sparse fragment arrays and temporary HTSlib buffers.
        return 48

    def _max_positive_ploidy(self, sample_ploidy: np.ndarray) -> int:
        if sample_ploidy.size:
            positive = sample_ploidy[sample_ploidy > 0]
            if positive.size:
                return max(int(np.max(positive)), 1)
        return max(int(self.config.ploidy), 1)

    def _state_count_for_memory(self, *, n_founders: int, max_ploidy: int) -> int:
        n_founders = max(int(n_founders), 1)
        max_ploidy = max(int(max_ploidy), 1)
        if max_ploidy == 1:
            return n_founders
        if max_ploidy == 2 and not bool(self.config.force_generic_ploidy_hmm):
            return n_founders * n_founders
        # Generic polyploid states are ordered founder tuples in the current HMM.
        return int(n_founders**max_ploidy)

    def _planned_memory_budget(self) -> tuple[int, int, float]:
        max_mem = max(_parse_memory_budget(self.config.max_mem), 1)
        safety = float(np.clip(float(self.config.memory_safety_fraction), 0.05, 0.95))
        return max_mem, max(1, int(max_mem * safety)), safety

    def _resolve_effective_block_size(
        self,
        *,
        n_samples: int,
        n_positions: int,
        founder_panel: FounderPanel,
        sample_ploidy: np.ndarray,
    ) -> int:
        if str(self.config.snp_block_mode) == "exact_streaming":
            return max(int(n_positions), 1)
        configured = int(self.config.block_size)
        if configured > 0:
            return min(configured, max(int(n_positions), 1))
        _, budget, _ = self._planned_memory_budget()
        bytes_per = self._estimate_hmm_bytes_per_sample_variant(founder_panel, sample_ploidy)
        denom = max(int(n_samples), 1) * max(bytes_per, 1)
        block_size = int(max(1, budget // denom))
        return min(block_size, max(int(n_positions), 1))

    def _resolve_effective_io_window_size(
        self,
        *,
        n_read_samples: int,
        n_positions: int,
        effective_block_size: int,
    ) -> int:
        configured = int(self.config.io_window_size)
        max_safe_window = max(1, min(int(effective_block_size), max(int(n_positions), 1)))
        if configured > 0:
            return min(max(int(configured), 1), max_safe_window)
        _, budget, _ = self._planned_memory_budget()
        denom = max(int(n_read_samples), 1) * self._estimate_io_bytes_per_sample_variant()
        io_window_size = int(max(1, budget // denom))
        return min(max(io_window_size, 1), max_safe_window)

    def _build_runtime_memory_plan(
        self,
        *,
        n_samples: int,
        n_read_samples: int,
        n_positions: int,
        founder_panel: FounderPanel,
        sample_ploidy: np.ndarray,
    ) -> RuntimeMemoryPlan:
        max_mem, planned_budget, safety = self._planned_memory_budget()
        bytes_hmm = self._estimate_hmm_bytes_per_sample_variant(founder_panel, sample_ploidy)
        bytes_io = self._estimate_io_bytes_per_sample_variant()
        effective_block_size = self._resolve_effective_block_size(
            n_samples=n_samples,
            n_positions=n_positions,
            founder_panel=founder_panel,
            sample_ploidy=sample_ploidy,
        )
        effective_io_window_size = self._resolve_effective_io_window_size(
            n_read_samples=n_read_samples,
            n_positions=n_positions,
            effective_block_size=effective_block_size,
        )
        configured_batch = int(self.config.jax_sample_batch_size)
        effective_sample_batch_size = (
            max(1, min(configured_batch, max(int(n_samples), 1)))
            if configured_batch > 0
            else max(int(n_samples), 1)
        )
        output_flush_size = max(1, min(effective_block_size, 8192))
        calibration_chunk_size = max(1, min(int(self.config.calibration_block_snps), effective_block_size))
        max_ploidy = self._max_positive_ploidy(sample_ploidy)
        state_count = self._state_count_for_memory(
            n_founders=int(founder_panel.n_founders),
            max_ploidy=max_ploidy,
        )
        n_intervals = max(int(n_positions) - 1, 0)
        xi_estimated = int(max(int(n_samples), 1) * n_intervals * state_count * state_count * 4)
        genotype_estimated = int(max(int(n_samples), 1) * max(int(n_positions), 1) * max(max_ploidy + 1, 1) * 4)
        xi_policy = str(self._store_xi)
        if xi_policy == "False":
            xi_decision = "off"
        elif xi_policy == "full":
            xi_decision = "write_full_zarr_forced"
        else:
            xi_decision = "write_per_snp_transition_summary"
        gamma_policy = str(self.config.write_gamma)
        gamma_estimated = int(max(int(n_samples), 1) * max(int(n_positions), 1) * state_count * 4)
        mode = str(self.config.snp_block_mode)
        notes: list[str] = []
        if effective_io_window_size < int(self.config.io_window_size or 0):
            notes.append("io_window_size_clamped_to_hmm_block_for_safe_cache_slicing")
        if mode == "exact_streaming":
            notes.append("exact_streaming_preserves_full_snp_span; use sample batching or Dask sample chunks for memory")
        if mode in {"independent_approx", "density_balanced_overlap"}:
            notes.append("snp_blocks_are_independent_approximation_not_whole_region_hmm")
        return RuntimeMemoryPlan(
            max_mem_bytes=int(max_mem),
            planned_budget_bytes=int(planned_budget),
            safety_fraction=float(safety),
            n_samples=int(n_samples),
            n_read_samples=int(n_read_samples),
            n_positions=int(n_positions),
            n_founders=int(founder_panel.n_founders),
            max_ploidy=int(max_ploidy),
            state_count=int(state_count),
            bytes_per_sample_variant_hmm=int(bytes_hmm),
            bytes_per_sample_variant_io=int(bytes_io),
            effective_block_size=int(effective_block_size),
            effective_io_window_size=int(effective_io_window_size),
            effective_sample_batch_size=int(effective_sample_batch_size),
            output_flush_size=int(output_flush_size),
            calibration_chunk_size=int(calibration_chunk_size),
            xi_policy=xi_policy,
            xi_decision=xi_decision,
            xi_estimated_bytes=int(xi_estimated),
            gamma_policy=gamma_policy,
            gamma_estimated_bytes=int(gamma_estimated),
            snp_block_mode=mode,
            approximate=mode in {"independent_approx", "density_balanced_overlap"},
            exact_streaming_requires_boundary_pass=False,
            scratch_dir=(None if self.config.scratch_dir is None else str(self.config.scratch_dir)),
            notes=notes,
        )

    def _write_runtime_memory_plan(self, plan: RuntimeMemoryPlan) -> None:
        (self.output_dir / "runtime_memory_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2),
            encoding="utf-8",
        )
        xi_manifest = {
            "format": "stitchv2.xi_output_manifest",
            "policy": str(plan.xi_policy),
            "decision": str(plan.xi_decision),
            "estimated_full_xi_bytes": int(plan.xi_estimated_bytes),
            "state_count": int(plan.state_count),
            "n_samples": int(plan.n_samples),
            "n_positions": int(plan.n_positions),
            "storage": "zarr" if plan.xi_decision == "write_full_zarr_forced" else "parquet",
            "path": (
                str(self.output_dir / "transition_summary")
                if plan.xi_decision == "write_per_snp_transition_summary"
                else None
            ),
            "materialized": False,
            "reason": (
                "xi will be materialized during HMM block processing"
                if plan.xi_decision == "write_full_zarr_forced"
                else "per-SNP transition summaries will be written during HMM output"
                if plan.xi_decision == "write_per_snp_transition_summary"
                else str(plan.xi_decision)
            ),
        }
        (self.output_dir / "xi_output_manifest.json").write_text(
            json.dumps(xi_manifest, indent=2),
            encoding="utf-8",
        )

    def _should_materialize_xi(self, plan: RuntimeMemoryPlan) -> bool:
        return str(plan.xi_decision) == "write_full_zarr_forced"

    @staticmethod
    def _directory_size_bytes(path: Path) -> int:
        total = 0
        if not path.exists():
            return total
        for child in path.rglob("*"):
            if child.is_file():
                total += int(child.stat().st_size)
        return int(total)

    @staticmethod
    def _diploid_alpha_beta_from_log_emission(
        log_emission: np.ndarray,
        switch: np.ndarray,
        k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_samples, n_positions = log_emission.shape[:2]
        emission = np.exp(log_emission - np.max(log_emission, axis=(2, 3), keepdims=True)).astype(np.float32)
        alpha = np.zeros_like(emission, dtype=np.float32)
        beta = np.zeros_like(emission, dtype=np.float32)
        if n_positions == 0:
            return emission, alpha, beta

        uniform = np.float32(1.0 / float(k * k))
        for sample_idx in range(n_samples):
            alpha0 = emission[sample_idx, 0].copy()
            alpha0_sum = float(alpha0.sum())
            if alpha0_sum <= 0.0:
                alpha0.fill(uniform)
            else:
                alpha0 /= alpha0_sum
            alpha[sample_idx, 0] = alpha0

            for pos_idx in range(1, n_positions):
                sw = float(switch[sample_idx, pos_idx])
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                prev = alpha[sample_idx, pos_idx - 1]
                row_sum = prev.sum(axis=1, keepdims=True)
                col_sum = prev.sum(axis=0, keepdims=True)
                total = float(prev.sum())
                pred = (b * b) * prev + (b * off) * (row_sum + col_sum) + (off * off) * total
                alpha_t = pred * emission[sample_idx, pos_idx]
                z = float(alpha_t.sum())
                if z <= 0.0:
                    alpha_t.fill(uniform)
                else:
                    alpha_t /= z
                alpha[sample_idx, pos_idx] = alpha_t

            beta[sample_idx, -1].fill(uniform)
            for pos_idx in range(n_positions - 2, -1, -1):
                sw = float(switch[sample_idx, pos_idx + 1])
                off = sw / max(k - 1, 1)
                b = 1.0 - sw - off
                next_term = beta[sample_idx, pos_idx + 1] * emission[sample_idx, pos_idx + 1]
                row_sum = next_term.sum(axis=1, keepdims=True)
                col_sum = next_term.sum(axis=0, keepdims=True)
                total = float(next_term.sum())
                beta_t = (b * b) * next_term + (b * off) * (row_sum + col_sum) + (off * off) * total
                z = float(beta_t.sum())
                if z <= 0.0:
                    beta_t.fill(uniform)
                else:
                    beta_t /= z
                beta[sample_idx, pos_idx] = beta_t

        return emission, alpha, beta

    @staticmethod
    def _diploid_xi_chunk(
        *,
        alpha_prev: np.ndarray,
        emission_next: np.ndarray,
        beta_next: np.ndarray,
        switch_next: np.ndarray,
        k: int,
    ) -> np.ndarray:
        state_count = int(k * k)
        off = (switch_next.astype(np.float32, copy=False) / float(max(k - 1, 1))).astype(np.float32, copy=False)
        stay = (1.0 - switch_next.astype(np.float32, copy=False)).astype(np.float32, copy=False)
        trans = np.broadcast_to(off[:, :, None, None], (*off.shape, k, k)).copy()
        diag = np.arange(k)
        trans[:, :, diag, diag] = stay[:, :, None]
        next_term = (emission_next.astype(np.float32, copy=False) * beta_next.astype(np.float32, copy=False)).astype(
            np.float32,
            copy=False,
        )
        xi = (
            alpha_prev.astype(np.float32, copy=False)[:, :, :, :, None, None]
            * trans[:, :, :, None, :, None]
            * trans[:, :, None, :, None, :]
            * next_term[:, :, None, None, :, :]
        ).reshape(alpha_prev.shape[0], alpha_prev.shape[1], state_count, state_count)
        denom = xi.sum(axis=(2, 3), keepdims=True)
        fallback = np.float32(1.0 / float(state_count * state_count))
        out = np.full_like(xi, fallback, dtype=np.float32)
        np.divide(xi, denom, out=out, where=denom > 0.0)
        return out.astype(np.float32, copy=False)

    def _write_xi_output_manifest(self, block_meta: dict[str, object]) -> None:
        manifest_path = self.output_dir / "xi_output_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except Exception:
            manifest = {}
        blocks = [
            dict(row)
            for row in manifest.get("blocks", [])
            if isinstance(row, dict) and int(row.get("block_id", -1)) != int(block_meta.get("block_id", -1))
        ]
        blocks.append(dict(block_meta))
        xi_root = self.output_dir / "xi.zarr"
        raw_total = sum(int(row.get("raw_bytes", 0)) for row in blocks if bool(row.get("materialized", False)))
        file_total = self._directory_size_bytes(xi_root)
        manifest.update(
            {
                "format": "stitchv2.xi_output_manifest",
                "policy": str(self._store_xi),
                "decision": "materialized_full_zarr",
                "storage": "zarr_v2_blosc_zstd",
                "path": str(xi_root),
                "materialized": any(bool(row.get("materialized", False)) for row in blocks),
                "blocks": blocks,
                "raw_bytes": int(raw_total),
                "file_bytes": int(file_total),
                "file_mb": float(file_total / 1_000_000.0),
                "compression_ratio_raw_to_file": (
                    float(raw_total / file_total) if file_total > 0 and raw_total > 0 else None
                ),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _write_per_snp_xi_manifest(self, block_meta: dict[str, object]) -> None:
        manifest_path = self.output_dir / "xi_output_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        except Exception:
            manifest = {}
        blocks = [
            dict(row)
            for row in manifest.get("blocks", [])
            if isinstance(row, dict) and int(row.get("block_id", -1)) != int(block_meta.get("block_id", -1))
        ]
        blocks.append(dict(block_meta))
        file_total = sum(int(row.get("file_bytes", 0)) for row in blocks if bool(row.get("materialized", False)))
        manifest.update(
            {
                "format": "stitchv2.xi_output_manifest",
                "policy": str(self._store_xi),
                "decision": "materialized_per_snp_transition_summary",
                "storage": "parquet",
                "path": str(self.output_dir / "transition_summary"),
                "materialized": any(bool(row.get("materialized", False)) for row in blocks),
                "blocks": blocks,
                "file_bytes": int(file_total),
                "file_mb": float(file_total / 1_000_000.0),
                "full_xi_estimated_bytes": int(manifest.get("estimated_full_xi_bytes", 0) or 0),
            }
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _write_xi_zarr_for_block(
        self,
        *,
        block,
        block_founders: FounderPanel,
        evidence: ReadEvidenceBlock,
        artifacts: HMMArtifacts,
        sample_ploidy: np.ndarray,
    ) -> dict[str, object]:
        t0 = time.perf_counter()
        if str(self.config.transition_model) != "stitch_parity":
            reason = "unsupported_transition_model"
            if self._store_xi == "full":
                raise NotImplementedError("Full xi Zarr output is currently implemented for transition_model=stitch_parity.")
            meta = {"block_id": int(block.block_id), "materialized": False, "reason": reason}
            self._write_xi_output_manifest(meta)
            return meta
        if self.config.ploidy_mode != "diploid" or not np.all(sample_ploidy == 2):
            reason = "unsupported_non_diploid_or_mixed_ploidy"
            if self._store_xi == "full":
                raise NotImplementedError("Full xi Zarr output is currently implemented for all-diploid blocks.")
            meta = {"block_id": int(block.block_id), "materialized": False, "reason": reason}
            self._write_xi_output_manifest(meta)
            return meta
        if bool(getattr(block, "has_core_overlap", False)):
            reason = "unsupported_overlap_block"
            if self._store_xi == "full":
                raise NotImplementedError("Full xi output for overlapped approximate blocks is not implemented.")
            meta = {"block_id": int(block.block_id), "materialized": False, "reason": reason}
            self._write_xi_output_manifest(meta)
            return meta
        if artifacts.switch_probability is None:
            reason = "missing_switch_probability"
            if self._store_xi == "full":
                raise RuntimeError("Cannot write xi because HMM artifacts do not include switch probabilities.")
            meta = {"block_id": int(block.block_id), "materialized": False, "reason": reason}
            self._write_xi_output_manifest(meta)
            return meta

        n_samples = int(evidence.sample_ids.shape[0])
        n_positions = int(evidence.positions.shape[0])
        n_intervals = max(n_positions - 1, 0)
        k = int(block_founders.n_founders)
        state_count = int(k * k)
        dtype = np.dtype("float32")
        configured_batch = int(self.config.jax_sample_batch_size)
        sample_chunk = min(max(configured_batch, 1), 8) if configured_batch > 0 else 8
        interval_chunk = min(max(n_intervals, 1), 64)
        chunk_shape = (int(sample_chunk), int(interval_chunk), state_count, state_count)
        array_shape = (n_samples, n_intervals, state_count, state_count)

        xi_root = self.output_dir / "xi.zarr"
        array_path = xi_root / f"block={int(block.block_id):06d}"
        if array_path.exists():
            raise FileExistsError(f"xi Zarr block already exists: {array_path}")
        xi_root.mkdir(parents=True, exist_ok=True)
        (xi_root / ".zgroup").write_text(json.dumps({"zarr_format": 2}, sort_keys=True), encoding="utf-8")
        (xi_root / ".zattrs").write_text(
            json.dumps(
                {
                    "format": "stitchv2.full_xi",
                    "description": "posterior transition probabilities P(state_t, state_t+1 | reads)",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        array_path.mkdir(parents=True, exist_ok=True)
        compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
        zarray = {
            "zarr_format": 2,
            "shape": list(array_shape),
            "chunks": list(chunk_shape),
            "dtype": dtype.str,
            "compressor": compressor.get_config(),
            "fill_value": 0.0,
            "order": "C",
            "filters": None,
        }
        (array_path / ".zarray").write_text(json.dumps(zarray, sort_keys=True), encoding="utf-8")
        (array_path / ".zattrs").write_text(
            json.dumps(
                {
                    "block_id": int(block.block_id),
                    "chromosome": str(self.config.chromosome),
                    "row_start": int(block.row_start),
                    "row_stop": int(block.row_stop),
                    "n_founders": int(k),
                    "state_count": int(state_count),
                    "state_encoding": "diploid_founder_pair_flat=i*n_founders+j",
                    "interval_encoding": "interval t stores transition from variant t to variant t+1",
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        labels_dir = self.output_dir / "xi_labels"
        write_parquet(
            pa.table(
                {
                    "sample_index": np.arange(n_samples, dtype=np.int32),
                    "sample_id": evidence.sample_ids.astype(str),
                }
            ),
            labels_dir / f"block={int(block.block_id):06d}.samples.parquet",
            self.config.compression,
            self.config.compression_level,
        )
        write_parquet(
            pa.table(
                {
                    "interval_index": np.arange(n_intervals, dtype=np.int32),
                    "from_variant_index": np.arange(n_intervals, dtype=np.int32),
                    "to_variant_index": np.arange(1, n_positions, dtype=np.int32),
                    "from_position": evidence.positions[:-1].astype(np.int64, copy=False),
                    "to_position": evidence.positions[1:].astype(np.int64, copy=False),
                }
            ),
            labels_dir / f"block={int(block.block_id):06d}.intervals.parquet",
            self.config.compression,
            self.config.compression_level,
        )

        hmm = self._hmm_runner_for_ploidy(2)
        founder_alt_prob = artifacts.founder_alt_prob.astype(np.float32, copy=False)
        chunk_count = 0
        for sample_start in range(0, n_samples, sample_chunk):
            sample_stop = min(sample_start + sample_chunk, n_samples)
            sample_indices = np.arange(sample_start, sample_stop, dtype=np.int64)
            (
                fso,
                fci,
                foo,
                fop,
                foc,
                foq,
            ) = self._subset_fragments_by_sample_indices(
                sample_indices,
                evidence.fragment_sample_offsets,
                evidence.fragment_center_idx,
                evidence.fragment_obs_offsets,
                evidence.fragment_obs_pos_idx,
                evidence.fragment_obs_code,
                evidence.fragment_obs_qual,
            )
            log_emission, _, founder_alt = hmm.emissions(
                founder_alt_prob,
                evidence.ref_count[sample_start:sample_stop],
                evidence.alt_count[sample_start:sample_stop],
                evidence.other_count[sample_start:sample_stop],
                evidence.ref_weight[sample_start:sample_stop],
                evidence.alt_weight[sample_start:sample_stop],
                evidence.other_weight[sample_start:sample_stop],
            )
            log_emission = hmm._apply_fragment_likelihoods(
                log_emission=log_emission,
                founder_alt=founder_alt.astype(np.float32, copy=False),
                fragment_sample_offsets=fso,
                fragment_center_idx=fci,
                fragment_obs_offsets=foo,
                fragment_obs_pos_idx=fop,
                fragment_obs_code=foc,
                fragment_obs_qual=foq,
            )
            emission, alpha, beta = self._diploid_alpha_beta_from_log_emission(
                log_emission,
                artifacts.switch_probability[sample_start:sample_stop].astype(np.float32, copy=False),
                k,
            )
            sample_chunk_index = sample_start // sample_chunk
            for interval_start in range(0, n_intervals, interval_chunk):
                interval_stop = min(interval_start + interval_chunk, n_intervals)
                xi = self._diploid_xi_chunk(
                    alpha_prev=alpha[:, interval_start:interval_stop],
                    emission_next=emission[:, interval_start + 1 : interval_stop + 1],
                    beta_next=beta[:, interval_start + 1 : interval_stop + 1],
                    switch_next=artifacts.switch_probability[
                        sample_start:sample_stop,
                        interval_start + 1 : interval_stop + 1,
                    ],
                    k=k,
                )
                chunk = np.zeros(chunk_shape, dtype=dtype)
                chunk[: sample_stop - sample_start, : interval_stop - interval_start] = xi.astype(dtype, copy=False)
                encoded = compressor.encode(np.ascontiguousarray(chunk))
                interval_chunk_index = interval_start // interval_chunk
                (array_path / f"{sample_chunk_index}.{interval_chunk_index}.0.0").write_bytes(bytes(encoded))
                chunk_count += 1
            del log_emission, emission, alpha, beta

        raw_bytes = int(math.prod(array_shape) * dtype.itemsize)
        elapsed = time.perf_counter() - t0
        block_file_bytes = self._directory_size_bytes(array_path)
        meta = {
            "block_id": int(block.block_id),
            "materialized": True,
            "path": str(array_path),
            "shape": list(array_shape),
            "chunks": list(chunk_shape),
            "dtype": dtype.name,
            "compressor": "blosc_zstd_bitshuffle_clevel5",
            "n_samples": int(n_samples),
            "n_positions": int(n_positions),
            "n_intervals": int(n_intervals),
            "n_founders": int(k),
            "state_count": int(state_count),
            "raw_bytes": int(raw_bytes),
            "file_bytes": int(block_file_bytes),
            "file_mb": float(block_file_bytes / 1_000_000.0),
            "compression_ratio_raw_to_file": (
                float(raw_bytes / block_file_bytes) if block_file_bytes > 0 and raw_bytes > 0 else None
            ),
            "write_seconds": float(elapsed),
            "n_chunks_written": int(chunk_count),
        }
        self._write_xi_output_manifest(meta)
        return meta

    def _iter_position_blocks_for_mode(self, positions_df: pd.DataFrame, block_size: int):
        if str(self.config.snp_block_mode) == "density_balanced_overlap":
            return iter_density_balanced_overlap_blocks(
                positions_df,
                block_size,
                overlap_fraction=float(self.config.approx_overlap_fraction),
                min_overlap_snps=int(self.config.approx_min_overlap_snps),
            )
        return iter_position_blocks(positions_df, block_size)

    def _slice_artifacts_by_position(self, artifacts: HMMArtifacts, start: int, stop: int) -> HMMArtifacts:
        transition_probability = artifacts.transition_probability
        if transition_probability is not None:
            transition_probability = transition_probability[:, start:stop]
        return HMMArtifacts(
            dosage=artifacts.dosage[:, start:stop],
            haplotype_posterior=(
                None if artifacts.haplotype_posterior is None else artifacts.haplotype_posterior[:, start:stop]
            ),
            genotype_posterior=(
                None if artifacts.genotype_posterior is None else artifacts.genotype_posterior[:, start:stop]
            ),
            genotype_call=(None if artifacts.genotype_call is None else artifacts.genotype_call[:, start:stop]),
            recombination_rate=artifacts.recombination_rate[start:stop],
            switch_probability=artifacts.switch_probability[:, start:stop],
            stay_probability=artifacts.stay_probability[:, start:stop],
            offdiag_probability=artifacts.offdiag_probability[:, start:stop],
            founder_alt_prob=artifacts.founder_alt_prob[:, start:stop],
            transition_probability=transition_probability,
            transition_factor_source=artifacts.transition_factor_source,
            transition_factor_destination=artifacts.transition_factor_destination,
            transition_factor_offdiag=artifacts.transition_factor_offdiag,
            em_diagnostics=artifacts.em_diagnostics,
        )

    def _slice_calibration_meta_by_position(
        self,
        calibration_meta: dict[str, object],
        *,
        start: int,
        stop: int,
    ) -> dict[str, object]:
        cmeta = calibration_meta.get("call_correctness")
        if not isinstance(cmeta, dict):
            return calibration_meta
        decisions = cmeta.get("_variant_decisions")
        if not isinstance(decisions, pd.DataFrame) or decisions.empty:
            return calibration_meta
        out = dict(calibration_meta)
        cmeta_out = dict(cmeta)
        cmeta_out["_variant_decisions"] = decisions.iloc[start:stop].reset_index(drop=True)
        out["call_correctness"] = cmeta_out
        return out

    def _enforce_snp_block_mode(self, plan: RuntimeMemoryPlan) -> None:
        if plan.snp_block_mode != "exact_streaming":
            return
        return

    def prepare_inputs(
        self,
        samples: pd.DataFrame,
        pedigree=None,
        founder_panel: FounderPanel | None = None,
    ) -> FounderPanel:
        samples = validate_samples(samples)
        pedigree_source = pedigree
        if pedigree_source is None and has_pedigree_columns(
            samples,
            offspring_col=self.config.pedigree_offspring_col,
            parent1_col=self.config.pedigree_parent1_col,
            parent2_col=self.config.pedigree_parent2_col,
        ):
            pedigree_source = samples
        pedigree_graph = coerce_pedigree_graph(
            pedigree_source,
            samples["sample_id"].astype(str).to_numpy(dtype=object),
            offspring_col=self.config.pedigree_offspring_col,
            parent1_col=self.config.pedigree_parent1_col,
            parent2_col=self.config.pedigree_parent2_col,
        )
        if pedigree_graph is not None:
            (self.output_dir / "pedigree_summary.json").write_text(
                json.dumps(pedigree_graph.summary(), indent=2),
                encoding="utf-8",
            )
        positions_df = load_positions(
            self.config.positions_path,
            self.config.chromosome,
            start=self.config.chromosome_start,
            end=self.config.chromosome_end,
        )
        if positions_df.empty:
            raise ValueError(
                f"No positions available for chromosome={self.config.chromosome} "
                f"with start={self.config.chromosome_start} end={self.config.chromosome_end}."
            )
        positions_df = self._attach_explicit_genetic_map(positions_df)
        self._microarray_dosage = None
        if self.config.microarray_plink_path:
            hardcalls = load_microarray_hardcalls_from_plink(
                self.config.microarray_plink_path,
                chromosome=self.config.chromosome,
                positions_df=positions_df,
            )
            samples, self._microarray_dosage = align_microarray_to_samples(
                samples,
                hardcalls,
                add_missing_samples=bool(self.config.microarray_add_samples),
                generation_default=self.config.microarray_generation_default,
            )
        if "plink_path" in samples.columns:
            mapped = load_microarray_hardcalls_from_sample_plink_paths(
                samples,
                chromosome=self.config.chromosome,
                positions_df=positions_df,
                plink_path_column="plink_path",
            )
            if self._microarray_dosage is None:
                self._microarray_dosage = mapped.dosage
            else:
                sample_micro = mapped.dosage
                valid = np.isfinite(sample_micro)
                if np.any(valid):
                    self._microarray_dosage[valid] = sample_micro[valid]
        self._sample_ploidy = self._resolve_sample_ploidy(samples)
        samples = samples.copy()
        samples["ploidy"] = self._sample_ploidy.astype(np.int16, copy=False)
        write_parquet(
            pa.Table.from_pandas(samples),
            self.output_dir / "samples.parquet",
            self.config.compression,
            self.config.compression_level,
        )
        write_parquet(
            pa.Table.from_pandas(positions_df),
            self.output_dir / "positions.parquet",
            self.config.compression,
            self.config.compression_level,
        )

        if founder_panel is None:
            if self.config.founder.source_path is None:
                raise ValueError("Founder source_path is required unless founder_panel is provided.")
            founder_panel = load_founders(
                source_format=self.config.founder.source_format,
                source_path=self.config.founder.source_path,
                chromosome=self.config.chromosome,
                positions_df=positions_df,
                immutable=self.config.founder.immutable,
            )
        elif founder_panel.genetic_cm is None and "CM" in positions_df.columns:
            founder_panel = FounderPanel(
                chromosome=founder_panel.chromosome,
                positions=founder_panel.positions,
                ref=founder_panel.ref,
                alt=founder_panel.alt,
                alt_prob=founder_panel.alt_prob,
                immutable_mask=founder_panel.immutable_mask,
                genetic_cm=positions_df["CM"].to_numpy(dtype=np.float32),
            )
        founder_panel = self._expand_founders_to_configured_count(founder_panel)
        founder_panel.to_parquet(self.output_dir / "founders.parquet", compression=self.config.compression)
        self._run_blocks(samples, positions_df, founder_panel, pedigree_graph)
        return founder_panel

    def _resolve_sample_ploidy(self, samples: pd.DataFrame) -> np.ndarray:
        default_ploidy = int(self.config.ploidy)
        if default_ploidy < 0:
            raise ValueError(f"ploidy must be >= 0, got {default_ploidy}")
        out = np.full(len(samples), default_ploidy, dtype=np.int16)
        male_ploidy = self.config.ploidy_males
        female_ploidy = self.config.ploidy_females
        if male_ploidy is None and female_ploidy is None:
            return out
        if male_ploidy is None or female_ploidy is None:
            raise ValueError("Both ploidy_males and ploidy_females must be provided together.")
        male_ploidy = int(male_ploidy)
        female_ploidy = int(female_ploidy)
        if male_ploidy < 0 or female_ploidy < 0:
            raise ValueError("ploidy_males and ploidy_females must be >= 0.")
        if "sex" not in samples.columns:
            raise ValueError("samples table must contain a 'sex' column when ploidy_males/ploidy_females are used.")
        sex = samples["sex"].astype(str).str.strip().str.lower()
        male_mask = sex.isin({"m", "male", "1", "xy"}).to_numpy(dtype=bool)
        female_mask = sex.isin({"f", "female", "2", "xx"}).to_numpy(dtype=bool)
        out[male_mask] = np.int16(male_ploidy)
        out[female_mask] = np.int16(female_ploidy)
        return out

    @staticmethod
    def _mask_genotype_posterior_by_ploidy(
        posterior: np.ndarray,
        sample_ploidy: np.ndarray,
    ) -> np.ndarray:
        gp = posterior.astype(np.float32, copy=True)
        for sample_idx, ploidy in enumerate(sample_ploidy.astype(np.int16, copy=False).tolist()):
            if int(ploidy) <= 0:
                gp[sample_idx] = np.nan
                continue
            n_valid = min(int(ploidy) + 1, gp.shape[2])
            if n_valid < gp.shape[2]:
                gp[sample_idx, :, n_valid:] = 0.0
        row_sum = np.nansum(gp, axis=2, keepdims=True)
        valid = np.isfinite(row_sum) & (row_sum > 0.0)
        gp = np.divide(gp, np.clip(row_sum, 1e-12, None), out=np.full_like(gp, np.nan), where=valid)
        return gp.astype(np.float32, copy=False)

    @staticmethod
    def _build_full_sample_fragment_offsets(
        read_fragment_offsets: np.ndarray,
        has_bam_mask: np.ndarray,
    ) -> np.ndarray:
        n_samples = int(has_bam_mask.shape[0])
        full_offsets = np.zeros(n_samples + 1, dtype=np.int64)
        read_idx = 0
        for sample_idx in range(n_samples):
            if bool(has_bam_mask[sample_idx]):
                n_frag = int(read_fragment_offsets[read_idx + 1] - read_fragment_offsets[read_idx])
                read_idx += 1
            else:
                n_frag = 0
            full_offsets[sample_idx + 1] = full_offsets[sample_idx] + n_frag
        return full_offsets

    @staticmethod
    def _inject_microarray_hard_calls(
        evidence: ReadEvidenceBlock,
        microarray_dosage_block: np.ndarray,
        *,
        hard_call_weight: int,
    ) -> None:
        if microarray_dosage_block.size == 0:
            return
        calls = np.rint(microarray_dosage_block).astype(np.float32, copy=False)
        valid = np.isfinite(calls)
        if not np.any(valid):
            return
        gt = np.clip(calls, 0.0, 2.0).astype(np.int8, copy=False)
        weight = max(int(hard_call_weight), 1)
        w_ref = weight // 2
        w_alt = weight - w_ref

        mask0 = valid & (gt == 0)
        mask1 = valid & (gt == 1)
        mask2 = valid & (gt == 2)

        if np.any(mask0):
            evidence.ref_count[mask0] = np.clip(
                evidence.ref_count[mask0].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.ref_weight[mask0] += float(weight)
            evidence.depth[mask0] = np.clip(
                evidence.depth[mask0].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        if np.any(mask1):
            evidence.ref_count[mask1] = np.clip(
                evidence.ref_count[mask1].astype(np.int32) + w_ref,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.alt_count[mask1] = np.clip(
                evidence.alt_count[mask1].astype(np.int32) + w_alt,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.ref_weight[mask1] += float(w_ref)
            evidence.alt_weight[mask1] += float(w_alt)
            evidence.depth[mask1] = np.clip(
                evidence.depth[mask1].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
        if np.any(mask2):
            evidence.alt_count[mask2] = np.clip(
                evidence.alt_count[mask2].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)
            evidence.alt_weight[mask2] += float(weight)
            evidence.depth[mask2] = np.clip(
                evidence.depth[mask2].astype(np.int32) + weight,
                0,
                np.iinfo(np.uint16).max,
            ).astype(np.uint16)

    def _build_full_sample_evidence(
        self,
        *,
        block_id: int,
        chromosome: str,
        positions: np.ndarray,
        ref: np.ndarray,
        alt: np.ndarray,
        sample_ids: np.ndarray,
        has_bam_mask: np.ndarray,
        read_evidence: ReadEvidenceBlock | None,
    ) -> ReadEvidenceBlock:
        n_samples = int(sample_ids.shape[0])
        n_positions = int(positions.shape[0])
        shape = (n_samples, n_positions)
        ref_count = np.zeros(shape, dtype=np.uint16)
        alt_count = np.zeros(shape, dtype=np.uint16)
        other_count = np.zeros(shape, dtype=np.uint16)
        depth = np.zeros(shape, dtype=np.uint16)
        ref_weight = np.zeros(shape, dtype=np.float32)
        alt_weight = np.zeros(shape, dtype=np.float32)
        other_weight = np.zeros(shape, dtype=np.float32)
        n_overlapping_reads = np.zeros(n_samples, dtype=np.int32)

        if read_evidence is not None:
            bam_idx = np.flatnonzero(has_bam_mask)
            ref_count[bam_idx] = read_evidence.ref_count
            alt_count[bam_idx] = read_evidence.alt_count
            other_count[bam_idx] = read_evidence.other_count
            depth[bam_idx] = read_evidence.depth
            ref_weight[bam_idx] = read_evidence.ref_weight
            alt_weight[bam_idx] = read_evidence.alt_weight
            other_weight[bam_idx] = read_evidence.other_weight
            n_overlapping_reads[bam_idx] = read_evidence.n_overlapping_reads
            fragment_sample_offsets = self._build_full_sample_fragment_offsets(
                read_evidence.fragment_sample_offsets,
                has_bam_mask,
            )
            fragment_center_idx = read_evidence.fragment_center_idx
            fragment_obs_offsets = read_evidence.fragment_obs_offsets
            fragment_obs_pos_idx = read_evidence.fragment_obs_pos_idx
            fragment_obs_code = read_evidence.fragment_obs_code
            fragment_obs_qual = read_evidence.fragment_obs_qual
        else:
            fragment_sample_offsets = np.zeros(n_samples + 1, dtype=np.int64)
            fragment_center_idx = np.empty((0,), dtype=np.int32)
            fragment_obs_offsets = np.zeros(1, dtype=np.int64)
            fragment_obs_pos_idx = np.empty((0,), dtype=np.int32)
            fragment_obs_code = np.empty((0,), dtype=np.int8)
            fragment_obs_qual = np.empty((0,), dtype=np.uint8)

        return ReadEvidenceBlock(
            block_id=block_id,
            chromosome=chromosome,
            positions=positions,
            ref=ref,
            alt=alt,
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            depth=depth,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
            sample_ids=sample_ids,
            n_overlapping_reads=n_overlapping_reads,
            fragment_sample_offsets=fragment_sample_offsets,
            fragment_center_idx=fragment_center_idx,
            fragment_obs_offsets=fragment_obs_offsets,
            fragment_obs_pos_idx=fragment_obs_pos_idx,
            fragment_obs_code=fragment_obs_code,
            fragment_obs_qual=fragment_obs_qual,
            memmap_dir=None,
        )

    @staticmethod
    def _subset_fragments_by_sample_indices(
        sample_indices: np.ndarray,
        fragment_sample_offsets: np.ndarray,
        fragment_center_idx: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_pos_idx: np.ndarray,
        fragment_obs_code: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        n_selected = int(sample_indices.size)
        local_sample_offsets = np.zeros(n_selected + 1, dtype=np.int64)
        if n_selected == 0:
            return (
                local_sample_offsets,
                np.empty((0,), dtype=np.int32),
                np.zeros(1, dtype=np.int64),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.int8),
                np.empty((0,), dtype=np.uint8) if fragment_obs_qual is not None else None,
            )

        center_chunks: list[np.ndarray] = []
        pos_chunks: list[np.ndarray] = []
        code_chunks: list[np.ndarray] = []
        qual_chunks: list[np.ndarray] = []
        obs_cursor = 0
        obs_offsets = [0]
        for out_idx, sample_idx in enumerate(sample_indices.tolist()):
            frag_start = int(fragment_sample_offsets[int(sample_idx)])
            frag_stop = int(fragment_sample_offsets[int(sample_idx) + 1])
            local_sample_offsets[out_idx + 1] = local_sample_offsets[out_idx] + (frag_stop - frag_start)
            if frag_stop <= frag_start:
                continue
            center_chunks.append(fragment_center_idx[frag_start:frag_stop].astype(np.int32, copy=False))
            for frag in range(frag_start, frag_stop):
                o0 = int(fragment_obs_offsets[frag])
                o1 = int(fragment_obs_offsets[frag + 1])
                n_obs = o1 - o0
                if n_obs > 0:
                    pos_chunks.append(fragment_obs_pos_idx[o0:o1].astype(np.int32, copy=False))
                    code_chunks.append(fragment_obs_code[o0:o1].astype(np.int8, copy=False))
                    if fragment_obs_qual is not None:
                        qual_chunks.append(fragment_obs_qual[o0:o1].astype(np.uint8, copy=False))
                obs_cursor += n_obs
                obs_offsets.append(obs_cursor)

        centers = np.concatenate(center_chunks, axis=0) if center_chunks else np.empty((0,), dtype=np.int32)
        obs_pos = np.concatenate(pos_chunks, axis=0) if pos_chunks else np.empty((0,), dtype=np.int32)
        obs_code = np.concatenate(code_chunks, axis=0) if code_chunks else np.empty((0,), dtype=np.int8)
        obs_qual = None
        if fragment_obs_qual is not None:
            obs_qual = np.concatenate(qual_chunks, axis=0) if qual_chunks else np.empty((0,), dtype=np.uint8)
        return local_sample_offsets, centers, np.asarray(obs_offsets, dtype=np.int64), obs_pos, obs_code, obs_qual

    def _run_hmm_for_subset(
        self,
        *,
        ploidy: int,
        founder_panel: FounderPanel,
        ref_count: np.ndarray,
        alt_count: np.ndarray,
        other_count: np.ndarray,
        ref_weight: np.ndarray,
        alt_weight: np.ndarray,
        other_weight: np.ndarray,
        generations: np.ndarray,
        return_full_transition: bool,
        return_haplotype_posterior: bool,
        return_genotype_posterior: bool,
        fragment_sample_offsets: np.ndarray,
        fragment_center_idx: np.ndarray,
        fragment_obs_offsets: np.ndarray,
        fragment_obs_pos_idx: np.ndarray,
        fragment_obs_code: np.ndarray,
        fragment_obs_qual: np.ndarray | None,
    ):
        return self._hmm_runner_for_ploidy(int(ploidy)).run(
            founder_panel=founder_panel,
            ref_count=ref_count,
            alt_count=alt_count,
            other_count=other_count,
            ref_weight=ref_weight,
            alt_weight=alt_weight,
            other_weight=other_weight,
            generations=generations,
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

    def _diagnostic_thresholds(self) -> dict[str, float]:
        return diagnostic_thresholds(
            warn_het_rate=self.config.diagnostics_warn_het_rate,
            fail_het_rate=self.config.diagnostics_fail_het_rate,
            warn_hom_rate=self.config.diagnostics_warn_hom_rate,
            fail_hom_rate=self.config.diagnostics_fail_hom_rate,
            warn_missing_rate=self.config.diagnostics_warn_missing_rate,
            fail_missing_rate=self.config.diagnostics_fail_missing_rate,
            warn_low_info=self.config.diagnostics_warn_low_info,
            fail_low_info=self.config.diagnostics_fail_low_info,
        )

    def _expand_founders_to_configured_count(self, founder_panel: FounderPanel) -> FounderPanel:
        target = int(self.config.n_founders)
        loaded = int(founder_panel.n_founders)
        if target <= loaded:
            summary = {
                "status": "not_expanded",
                "configured_n_founders": target,
                "loaded_n_founders": loaded,
                "final_n_founders": loaded,
                "n_extra_mutable_founders": 0,
                "n_immutable_founders": int(np.count_nonzero(founder_panel.immutable_mask)),
                "n_mutable_founders": int(np.count_nonzero(~founder_panel.immutable_mask.astype(bool, copy=False))),
            }
            (self.output_dir / "founder_expansion_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )
            return founder_panel

        expanded = founder_panel.with_extra_mutable_founders(target)
        summary = {
            "status": "expanded_with_extra_mutable_founders",
            "configured_n_founders": target,
            "loaded_n_founders": loaded,
            "final_n_founders": int(expanded.n_founders),
            "n_extra_mutable_founders": int(target - loaded),
            "extra_founder_initial_alt_prob": 0.5,
            "n_immutable_founders": int(np.count_nonzero(expanded.immutable_mask)),
            "n_mutable_founders": int(np.count_nonzero(~expanded.immutable_mask.astype(bool, copy=False))),
        }
        (self.output_dir / "founder_expansion_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return expanded

    def _microarray_truth_block(
        self,
        *,
        block_start: int,
        n_positions: int,
        gp_shape: tuple[int, int, int],
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        if self._microarray_dosage is None or gp_shape[2] != 3:
            return None, {"source": "microarray", "status": "unavailable", "n_labeled": 0}
        start = int(block_start)
        stop = min(start + int(n_positions), self._microarray_dosage.shape[1])
        micro_block = self._microarray_dosage[:, start:stop]
        truth = np.full(gp_shape[:2], -1, dtype=np.int8)
        if micro_block.shape != truth.shape:
            rows = min(micro_block.shape[0], truth.shape[0])
            cols = min(micro_block.shape[1], truth.shape[1])
            micro_use = micro_block[:rows, :cols]
            valid = np.isfinite(micro_use)
            if np.any(valid):
                truth[:rows, :cols][valid] = np.clip(np.rint(micro_use[valid]), 0, 2).astype(np.int8, copy=False)
        else:
            valid = np.isfinite(micro_block)
            if np.any(valid):
                truth[valid] = np.clip(np.rint(micro_block[valid]), 0, 2).astype(np.int8, copy=False)
        return truth, {
            "source": "microarray",
            "status": "ok" if int(np.sum(truth >= 0)) > 0 else "empty",
            "n_labeled": int(np.sum(truth >= 0)),
        }

    def _read_evidence_truth_block(
        self,
        *,
        evidence: ReadEvidenceBlock,
        sample_ploidy: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object]]:
        ref = evidence.ref_count.astype(np.float32, copy=False)
        alt = evidence.alt_count.astype(np.float32, copy=False)
        other = evidence.other_count.astype(np.float32, copy=False)
        ref_alt = ref + alt
        total = ref_alt + other
        truth = np.full(ref.shape, -1, dtype=np.int8)

        diploid = sample_ploidy.astype(np.int16, copy=False) == 2
        diploid_mask = diploid[:, None] if diploid.shape[0] == ref.shape[0] else np.ones_like(ref, dtype=bool)
        min_depth = max(int(self.config.calibration_read_truth_min_depth), 1)
        min_hom_depth = max(int(self.config.calibration_read_truth_min_hom_depth), 1)
        min_het_depth = max(int(self.config.calibration_read_truth_min_het_depth), 1)
        min_het_allele_depth = max(int(self.config.calibration_read_truth_min_het_allele_depth), 1)
        hom_major = float(np.clip(self.config.calibration_read_truth_hom_major_fraction, 0.5, 1.0))
        het_lo = float(np.clip(self.config.calibration_read_truth_het_balance_min, 0.0, 1.0))
        het_hi = float(np.clip(self.config.calibration_read_truth_het_balance_max, 0.0, 1.0))
        if het_lo > het_hi:
            het_lo, het_hi = het_hi, het_lo
        max_other = float(np.clip(self.config.calibration_read_truth_max_other_fraction, 0.0, 1.0))

        alt_frac = np.divide(alt, np.clip(ref_alt, 1.0, None), out=np.zeros_like(alt), where=ref_alt > 0.0)
        ref_frac = np.divide(ref, np.clip(ref_alt, 1.0, None), out=np.zeros_like(ref), where=ref_alt > 0.0)
        other_frac = np.divide(other, np.clip(total, 1.0, None), out=np.zeros_like(other), where=total > 0.0)
        informative = diploid_mask & (ref_alt >= float(min_depth)) & (other_frac <= max_other)

        hom_ref = informative & (ref >= float(min_hom_depth)) & (ref_frac >= hom_major)
        hom_alt = informative & (alt >= float(min_hom_depth)) & (alt_frac >= hom_major)
        het = (
            informative
            & (ref_alt >= float(min_het_depth))
            & (ref >= float(min_het_allele_depth))
            & (alt >= float(min_het_allele_depth))
            & (alt_frac >= het_lo)
            & (alt_frac <= het_hi)
        )
        truth[hom_ref] = 0
        truth[het] = 1
        truth[hom_alt] = 2
        return truth, {
            "source": "read_evidence",
            "status": "ok" if int(np.sum(truth >= 0)) > 0 else "empty",
            "n_labeled": int(np.sum(truth >= 0)),
            "n_hom_ref": int(np.sum(truth == 0)),
            "n_het": int(np.sum(truth == 1)),
            "n_hom_alt": int(np.sum(truth == 2)),
            "min_depth": int(min_depth),
            "min_hom_depth": int(min_hom_depth),
            "min_het_depth": int(min_het_depth),
            "min_het_allele_depth": int(min_het_allele_depth),
            "hom_major_fraction": float(hom_major),
            "het_balance_min": float(het_lo),
            "het_balance_max": float(het_hi),
            "max_other_fraction": float(max_other),
        }

    def _calibration_truth_block(
        self,
        *,
        evidence: ReadEvidenceBlock,
        block_start: int,
        n_positions: int,
        gp_shape: tuple[int, int, int],
        sample_ploidy: np.ndarray,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        source = str(self.config.calibration_truth_source)
        if source == "none":
            return None, {"source": "none", "status": "disabled", "n_labeled": 0}
        if gp_shape[2] != 3:
            return None, {"source": source, "status": "skipped_non_diploid", "n_labeled": 0}
        if source in {"read_evidence", "auto"}:
            truth, meta = self._read_evidence_truth_block(evidence=evidence, sample_ploidy=sample_ploidy)
            if source == "read_evidence" or int(meta.get("n_labeled", 0)) > 0:
                return truth, meta
        if source in {"microarray", "auto"}:
            return self._microarray_truth_block(
                block_start=int(block_start),
                n_positions=int(n_positions),
                gp_shape=gp_shape,
            )
        raise ValueError(f"Unsupported calibration_truth_source: {source!r}")

    def _maybe_split_read_evidence_for_calibration(
        self,
        *,
        evidence: ReadEvidenceBlock,
        block_id: int,
    ) -> tuple[ReadEvidenceBlock, ReadEvidenceBlock | None, dict[str, object]]:
        if not (
            bool(self.config.calibrate_genotype_posteriors)
            and self.config.calibration_mode == "standard_callability"
            and str(self.config.calibration_truth_source) in {"read_evidence", "auto"}
        ):
            return evidence, None, {"status": "disabled"}
        fraction = float(np.clip(float(self.config.calibration_read_truth_holdout_fraction), 0.0, 0.95))
        if fraction <= 0.0:
            return evidence, None, {"status": "disabled", "holdout_fraction_requested": 0.0}
        split_hmm_evidence, heldout_evidence, meta = evidence.split_fragments_for_calibration(
            holdout_fraction=fraction,
            seed=int(self.config.random_seed + int(block_id) * 1009 + 17),
        )
        meta = dict(meta)
        meta["source"] = "fragment_holdout"
        meta["hmm_uses_all_read_evidence"] = True
        meta["calibration_label_holdout_only"] = True
        if split_hmm_evidence is not evidence:
            split_hmm_evidence.release()
        return evidence, heldout_evidence, meta

    @staticmethod
    def _callability_fell_back_to_stitch(calibration_meta: dict[str, object]) -> bool:
        cmeta = calibration_meta.get("call_correctness")
        if isinstance(cmeta, dict) and str(cmeta.get("status", "")) == "fallback_to_stitch_no_call":
            return True
        return str(calibration_meta.get("status", "")) == "fallback_to_stitch_no_call"

    def _callability_threshold_from_meta(
        self,
        *,
        calibration_meta: dict[str, object],
        call_correct_probability: np.ndarray | None,
        genotype_posterior: np.ndarray,
    ) -> tuple[np.ndarray | None, float | np.ndarray]:
        cmeta = calibration_meta.get("call_correctness")
        if not isinstance(cmeta, dict) or call_correct_probability is None:
            return call_correct_probability, float(self.config.genotype_call_correctness_threshold)
        threshold_by_variant = cmeta.get("threshold_by_variant")
        fallback_by_variant = cmeta.get("fallback_to_stitch_by_variant")
        if threshold_by_variant is None or fallback_by_variant is None:
            threshold = float(self.config.genotype_call_correctness_threshold)
            if threshold <= 0.0 and cmeta.get("threshold") is not None:
                threshold = float(cmeta.get("threshold"))
            return call_correct_probability, threshold
        threshold_arr = np.asarray(threshold_by_variant, dtype=np.float32)
        fallback_arr = np.asarray(fallback_by_variant, dtype=bool)
        if threshold_arr.shape != (genotype_posterior.shape[1],) or fallback_arr.shape != (genotype_posterior.shape[1],):
            threshold = float(self.config.genotype_call_correctness_threshold)
            if threshold <= 0.0 and cmeta.get("threshold") is not None:
                threshold = float(cmeta.get("threshold"))
            return call_correct_probability, threshold
        gp_conf = np.max(np.nan_to_num(genotype_posterior.astype(np.float32, copy=False), nan=0.0), axis=2)
        gate_probability = np.where(fallback_arr[None, :], gp_conf, call_correct_probability).astype(
            np.float32,
            copy=False,
        )
        threshold = np.where(
            fallback_arr,
            float(self.config.genotype_call_stitch_threshold),
            threshold_arr,
        ).astype(np.float32, copy=False)
        return gate_probability, threshold

    def _write_calibration_decisions(self, *, block, calibration_meta: dict[str, object]) -> None:
        cmeta = calibration_meta.get("call_correctness")
        if not isinstance(cmeta, dict):
            return
        decisions = cmeta.get("_variant_decisions")
        if not isinstance(decisions, pd.DataFrame) or decisions.empty:
            return
        out = decisions.copy()
        positions = block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False)
        if len(out) == len(positions):
            out.insert(0, "position", positions)
            out.insert(0, "chromosome", str(self.config.chromosome))
            out["block_id"] = int(block.block_id)
        table = pa.Table.from_pandas(out, preserve_index=False)
        write_parquet(
            table,
            self.output_dir / "calibration_decisions" / f"block={block.block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
            row_group_size=262_144,
            use_dictionary=["chromosome", "block_id", "calibration_used", "fallback_to_stitch", "decision_source", "fallback_reason"],
            use_byte_stream_split=[
                "threshold",
                "call_rate_stitch",
                "call_rate_calibrated",
                "call_rate_delta",
                "objective_delta",
                "maf",
                "info",
                "hwe_deviation",
                "support_rate",
                "missingness",
                "mean_depth",
                "maf_shift",
                "het_shift",
            ],
        )

    def _write_block_diagnostics(
        self,
        *,
        block,
        dosage: np.ndarray,
        raw_posterior: np.ndarray | None,
        calibrated_posterior: np.ndarray | None,
        genotype_call: np.ndarray | None,
        evidence: ReadEvidenceBlock,
        support_mask: np.ndarray,
        sample_ploidy: np.ndarray,
    ) -> None:
        if not bool(self.config.write_diagnostics):
            return
        diagnostics = compute_variant_diagnostics(
            block_id=int(block.block_id),
            chromosome=str(self.config.chromosome),
            positions=block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False),
            dosage=dosage,
            raw_posterior=raw_posterior,
            calibrated_posterior=calibrated_posterior,
            genotype_call=genotype_call,
            depth=evidence.depth.astype(np.float32, copy=False),
            support_mask=support_mask,
            sample_ploidy=sample_ploidy,
        )
        write_block_diagnostics(
            self.output_dir,
            diagnostics,
            block_id=int(block.block_id),
            compression=self.config.compression,
            compression_level=self.config.compression_level,
        )

    def _finalize_diagnostics_summary(self) -> None:
        if not bool(self.config.write_diagnostics):
            return
        write_diagnostics_summary(
            self.output_dir,
            thresholds=self._diagnostic_thresholds(),
            fail_on_error=bool(self.config.diagnostics_fail_on_error),
        )

    def _run_blocks(
        self,
        samples: pd.DataFrame,
        positions_df: pd.DataFrame,
        founder_panel: FounderPanel,
        pedigree,
    ) -> None:
        if self.config.executor == "dask":
            self._run_blocks_dask(samples, positions_df, founder_panel, pedigree)
            return
        timings: list[dict[str, float | int]] = []
        generations = samples["generation"].to_numpy(dtype=np.float32, copy=False)
        sample_ids = samples["sample_id"].astype(str).to_numpy()
        sample_ploidy = (
            self._sample_ploidy.copy()
            if self._sample_ploidy is not None
            else np.full(len(samples), int(self.config.ploidy), dtype=np.int16)
        )
        has_bam_mask = samples["bam_path"].fillna("").astype(str).str.len().to_numpy(dtype=np.int32) > 0
        read_samples = samples.loc[has_bam_mask].reset_index(drop=True)
        memory_plan = self._build_runtime_memory_plan(
            n_samples=int(samples.shape[0]),
            n_read_samples=int(read_samples.shape[0]),
            n_positions=int(positions_df.shape[0]),
            founder_panel=founder_panel,
            sample_ploidy=sample_ploidy,
        )
        self._write_runtime_memory_plan(memory_plan)
        self._enforce_snp_block_mode(memory_plan)
        effective_block_size = int(memory_plan.effective_block_size)
        effective_io_window_size = int(memory_plan.effective_io_window_size)
        self.read_extractor.set_position_table(positions_df)
        self.read_extractor.set_io_window_size(effective_io_window_size)
        if int(read_samples.shape[0]) > 0:
            self.read_extractor.open(read_samples)
        try:
            backend_autotuned = False
            for block in self._iter_position_blocks_for_mode(positions_df, effective_block_size):
                block_t0 = time.perf_counter()
                rss_block_start = _current_rss_mb() if self.config.profile_memory else None
                read_evidence, compact_cache_stats = self._extract_or_load_read_evidence(
                    read_samples=read_samples,
                    block=block,
                )
                full_evidence = self._build_full_sample_evidence(
                    block_id=block.block_id,
                    chromosome=self.config.chromosome,
                    positions=block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False),
                    ref=block.dataframe["REF"].astype(str).to_numpy(),
                    alt=block.dataframe["ALT"].astype(str).to_numpy(),
                    sample_ids=sample_ids,
                    has_bam_mask=has_bam_mask,
                    read_evidence=read_evidence,
                )
                evidence = full_evidence
                calibration_truth_evidence: ReadEvidenceBlock | None = None
                calibration_split_meta: dict[str, object] = {"status": "disabled"}
                try:
                    evidence, calibration_truth_evidence, calibration_split_meta = self._maybe_split_read_evidence_for_calibration(
                        evidence=full_evidence,
                        block_id=int(block.block_id),
                    )
                    support_mask = evidence.depth > 0
                    if self._microarray_dosage is not None and bool(self.config.microarray_use_as_hmm_evidence):
                        start = int(block.row_start)
                        stop = min(int(block.row_stop), self._microarray_dosage.shape[1])
                        self._inject_microarray_hard_calls(
                            evidence,
                            self._microarray_dosage[:, start:stop],
                            hard_call_weight=int(self.config.microarray_hard_call_weight),
                        )
                    t_after_reads = time.perf_counter()
                    rss_after_reads = _current_rss_mb() if self.config.profile_memory else None
                    if self.io_config.write_pileup:
                        write_parquet(
                            evidence.to_arrow(),
                            self.output_dir / "pileup" / f"block={block.block_id:06d}.parquet",
                            self.config.compression,
                            self.config.compression_level,
                        )

                    block_founders = founder_panel.slice(
                        int(block.row_start),
                        min(int(block.row_stop), founder_panel.n_positions),
                    )
                    requested_genotype_outputs = (
                        self.io_config.write_genotype_posteriors
                        or self.io_config.write_genotype_calls
                    )
                    need_genotype_posterior = requested_genotype_outputs or self.config.pedigree_mode in {
                        "kinship",
                        "transmission",
                    }
                    positive_ploidies = np.unique(sample_ploidy[sample_ploidy > 0])
                    max_ploidy = int(np.max(sample_ploidy)) if sample_ploidy.size else int(self.config.ploidy)
                    if not backend_autotuned and np.any(sample_ploidy == 2):
                        dip_idx = np.flatnonzero(sample_ploidy == 2)
                        (
                            fso_dip,
                            fci_dip,
                            foo_dip,
                            fop_dip,
                            foc_dip,
                            foq_dip,
                        ) = self._subset_fragments_by_sample_indices(
                            dip_idx,
                            evidence.fragment_sample_offsets,
                            evidence.fragment_center_idx,
                            evidence.fragment_obs_offsets,
                            evidence.fragment_obs_pos_idx,
                            evidence.fragment_obs_code,
                            evidence.fragment_obs_qual,
                        )
                        self._hmm_runner_for_ploidy(2).autotune_backend(
                            founder_panel=block_founders,
                            ref_count=evidence.ref_count[dip_idx],
                            alt_count=evidence.alt_count[dip_idx],
                            generations=generations[dip_idx],
                            other_count=evidence.other_count[dip_idx],
                            ref_weight=evidence.ref_weight[dip_idx],
                            alt_weight=evidence.alt_weight[dip_idx],
                            other_weight=evidence.other_weight[dip_idx],
                            fragment_sample_offsets=fso_dip,
                            fragment_center_idx=fci_dip,
                            fragment_obs_offsets=foo_dip,
                            fragment_obs_pos_idx=fop_dip,
                            fragment_obs_code=foc_dip,
                            fragment_obs_qual=foq_dip,
                        )
                        backend_autotuned = True

                    return_full_transition = False
                    return_haplotype = self.io_config.write_haplotype_probabilities
                    if np.all(sample_ploidy <= 0):
                        n_samples, n_positions = evidence.ref_count.shape
                        base_switch = self.hmm._switch_probabilities(
                            self.hmm.recombination_from_positions(block_founders.positions, block_founders.genetic_cm),
                            generations,
                        )
                        artifacts = HMMArtifacts(
                            dosage=np.full((n_samples, n_positions), np.nan, dtype=np.float32),
                            haplotype_posterior=(
                                np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
                                if return_haplotype
                                else None
                            ),
                            genotype_posterior=None,
                            genotype_call=None,
                            recombination_rate=self.hmm.recombination_from_positions(
                                block_founders.positions,
                                block_founders.genetic_cm,
                            ).astype(np.float32),
                            switch_probability=base_switch.astype(np.float32),
                            stay_probability=(1.0 - base_switch).astype(np.float32),
                            offdiag_probability=(base_switch / max(block_founders.n_founders - 1, 1)).astype(np.float32),
                            founder_alt_prob=block_founders.alt_prob.astype(np.float32, copy=True),
                            transition_probability=(
                                self.hmm._build_full_transitions(base_switch, block_founders.n_founders)
                                if return_full_transition
                                else None
                            ),
                        )
                    elif positive_ploidies.size == 1 and np.all(sample_ploidy > 0):
                        ploidy_i = int(positive_ploidies[0])
                        artifacts = self._run_hmm_for_subset(
                            ploidy=ploidy_i,
                            founder_panel=block_founders,
                            ref_count=evidence.ref_count,
                            alt_count=evidence.alt_count,
                            other_count=evidence.other_count,
                            ref_weight=evidence.ref_weight,
                            alt_weight=evidence.alt_weight,
                            other_weight=evidence.other_weight,
                            generations=generations,
                            return_full_transition=return_full_transition,
                            return_haplotype_posterior=return_haplotype,
                            return_genotype_posterior=need_genotype_posterior,
                            fragment_sample_offsets=evidence.fragment_sample_offsets,
                            fragment_center_idx=evidence.fragment_center_idx,
                            fragment_obs_offsets=evidence.fragment_obs_offsets,
                            fragment_obs_pos_idx=evidence.fragment_obs_pos_idx,
                            fragment_obs_code=evidence.fragment_obs_code,
                            fragment_obs_qual=evidence.fragment_obs_qual,
                        )
                    else:
                        n_samples, n_positions = evidence.ref_count.shape
                        dosage_out = np.full((n_samples, n_positions), np.nan, dtype=np.float32)
                        hap_out = (
                            np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
                            if return_haplotype
                            else None
                        )
                        gp_classes = max(max_ploidy + 1, 1)
                        gp_out = np.zeros((n_samples, n_positions, gp_classes), dtype=np.float32) if need_genotype_posterior else None
                        switch_out = np.zeros((n_samples, n_positions), dtype=np.float32)
                        stay_out = np.zeros((n_samples, n_positions), dtype=np.float32)
                        offdiag_out = np.zeros((n_samples, n_positions), dtype=np.float32)
                        founder_acc = np.zeros_like(block_founders.alt_prob, dtype=np.float32)
                        founder_w = 0.0

                        for ploidy_i in positive_ploidies.tolist():
                            mask = sample_ploidy == int(ploidy_i)
                            idx = np.flatnonzero(mask)
                            if idx.size == 0:
                                continue
                            (
                                fso,
                                fci,
                                foo,
                                fop,
                                foc,
                                foq,
                            ) = self._subset_fragments_by_sample_indices(
                                idx,
                                evidence.fragment_sample_offsets,
                                evidence.fragment_center_idx,
                                evidence.fragment_obs_offsets,
                                evidence.fragment_obs_pos_idx,
                                evidence.fragment_obs_code,
                                evidence.fragment_obs_qual,
                            )
                            sub = self._run_hmm_for_subset(
                                ploidy=int(ploidy_i),
                                founder_panel=block_founders,
                                ref_count=evidence.ref_count[idx],
                                alt_count=evidence.alt_count[idx],
                                other_count=evidence.other_count[idx],
                                ref_weight=evidence.ref_weight[idx],
                                alt_weight=evidence.alt_weight[idx],
                                other_weight=evidence.other_weight[idx],
                                generations=generations[idx],
                                return_full_transition=return_full_transition,
                                return_haplotype_posterior=return_haplotype,
                                return_genotype_posterior=need_genotype_posterior,
                                fragment_sample_offsets=fso,
                                fragment_center_idx=fci,
                                fragment_obs_offsets=foo,
                                fragment_obs_pos_idx=fop,
                                fragment_obs_code=foc,
                                fragment_obs_qual=foq,
                            )
                            dosage_out[idx] = sub.dosage.astype(np.float32, copy=False)
                            if hap_out is not None and sub.haplotype_posterior is not None:
                                hap_out[idx] = sub.haplotype_posterior
                            if gp_out is not None and sub.genotype_posterior is not None:
                                n_cls = min(gp_out.shape[2], sub.genotype_posterior.shape[2])
                                gp_out[idx, :, :n_cls] = sub.genotype_posterior[:, :, :n_cls]
                            switch_out[idx] = sub.switch_probability
                            stay_out[idx] = sub.stay_probability
                            offdiag_out[idx] = sub.offdiag_probability
                            founder_acc += sub.founder_alt_prob.astype(np.float32, copy=False) * float(idx.size)
                            founder_w += float(idx.size)

                        if np.any(sample_ploidy == 0):
                            zero_idx = np.flatnonzero(sample_ploidy == 0)
                            dosage_out[zero_idx] = np.nan
                            if gp_out is not None:
                                gp_out[zero_idx] = np.nan

                        founder_final = (
                            founder_acc / max(founder_w, 1.0)
                            if founder_w > 0.0
                            else block_founders.alt_prob.astype(np.float32, copy=True)
                        )
                        gt_out = None
                        if gp_out is not None:
                            gt_out = np.argmax(np.nan_to_num(gp_out, nan=-1.0), axis=2).astype(np.int8, copy=False)
                            gt_out[~np.isfinite(gp_out).any(axis=2)] = -1
                        artifacts = HMMArtifacts(
                            dosage=dosage_out,
                            haplotype_posterior=hap_out,
                            genotype_posterior=gp_out,
                            genotype_call=gt_out,
                            recombination_rate=self.hmm.recombination_from_positions(
                                block_founders.positions,
                                block_founders.genetic_cm,
                            ).astype(np.float32),
                            switch_probability=switch_out,
                            stay_probability=stay_out,
                            offdiag_probability=offdiag_out,
                            founder_alt_prob=founder_final.astype(np.float32, copy=False),
                            transition_probability=None,
                        )
                    t_after_hmm = time.perf_counter()
                    rss_after_hmm = _current_rss_mb() if self.config.profile_memory else None
                    dosage = artifacts.dosage.astype(np.float32, copy=False)
                    if self.config.pedigree_mode == "smooth":
                        pedigree_result = apply_pedigree_adjustment(
                            dosage=dosage,
                            genotype_posterior=None,
                            pedigree=pedigree,
                            mode=self.config.pedigree_mode,
                            strength=self.config.pedigree_strength,
                            positions=block_founders.positions,
                            generations=generations,
                            support_mask=support_mask,
                            iterations=self.config.pedigree_iterations,
                            kinship_threshold=self.config.pedigree_kinship_threshold,
                        )
                        dosage = pedigree_result.dosage
                    else:
                        pedigree_result = None
                    calibrated_gp = None
                    calibrated_gt = None
                    call_correct_probability = None
                    calibration_meta: dict[str, object] = {"status": "disabled"}
                    if need_genotype_posterior:
                        if artifacts.genotype_posterior is not None:
                            calibrated_gp = artifacts.genotype_posterior.astype(np.float32, copy=False)
                            calibration_meta = {
                                "status": "raw_hmm_gp",
                                "mode": "raw_hmm_gp",
                                "raw_gp_primary": True,
                                "posterior_calibration": "none",
                            }
                        else:
                            calibrated_gp = calibrate_genotype_posterior(
                                None,
                                dosage=dosage,
                                depth=evidence.depth.astype(np.float32, copy=False),
                                temperature=self.config.genotype_posterior_temperature,
                                blend=0.0,
                                ploidy=max_ploidy,
                            )
                            calibration_meta = {
                                "status": "dosage_fallback",
                                "mode": "dosage_fallback",
                                "raw_gp_primary": False,
                                "posterior_calibration": "none",
                                "temperature": float(self.config.genotype_posterior_temperature),
                                "blend": 0.0,
                            }
                        if self.config.calibrate_genotype_posteriors and self.config.calibration_mode == "fixed":
                            calibrated_gp = calibrate_genotype_posterior(
                                artifacts.genotype_posterior,
                                dosage=dosage,
                                depth=evidence.depth.astype(np.float32, copy=False),
                                temperature=self.config.genotype_posterior_temperature,
                                blend=self.config.genotype_posterior_blend,
                                ploidy=max_ploidy,
                            )
                            calibration_meta = {
                                "status": "applied",
                                "mode": "fixed",
                                "raw_gp_primary": False,
                                "posterior_calibration": "temperature_blend",
                                "temperature": float(self.config.genotype_posterior_temperature),
                                "blend": float(self.config.genotype_posterior_blend),
                            }
                        calibration_label_evidence = (
                            calibration_truth_evidence if calibration_truth_evidence is not None else evidence
                        )
                        truth_gt_block, truth_meta = self._calibration_truth_block(
                            evidence=calibration_label_evidence,
                            block_start=int(block.row_start),
                            n_positions=int(calibrated_gp.shape[1]),
                            gp_shape=calibrated_gp.shape,
                            sample_ploidy=sample_ploidy,
                        )
                        if isinstance(truth_meta, dict) and calibration_truth_evidence is not None:
                            truth_meta = dict(truth_meta)
                            truth_meta["read_evidence_holdout"] = calibration_split_meta
                        if (
                            self.config.calibrate_genotype_posteriors
                            and self.config.calibration_mode == "standard_callability"
                            and truth_gt_block is not None
                            and calibrated_gp is not None
                            and calibrated_gp.shape[2] == 3
                        ):
                            train_positions = self._calibration_train_position_index(
                                calibrated_gp.shape[1],
                                seed=int(self.config.random_seed + block.block_id),
                            )
                            train_site_mask = np.isin(
                                np.arange(calibrated_gp.shape[1], dtype=np.int64),
                                train_positions,
                            )
                            train_mask = (truth_gt_block >= 0) & train_site_mask[None, :]
                            call_correct_probability, callability_meta = train_standard_callability_model(
                                posterior=calibrated_gp,
                                dosage=dosage,
                                truth_genotype=truth_gt_block,
                                train_mask=train_mask,
                                depth=evidence.depth.astype(np.float32, copy=False),
                                ref_count=evidence.ref_count.astype(np.float32, copy=False),
                                alt_count=evidence.alt_count.astype(np.float32, copy=False),
                                other_count=evidence.other_count.astype(np.float32, copy=False),
                                support_mask=support_mask.astype(bool, copy=False),
                                max_train_rows=int(self.config.calibration_max_train_rows),
                                model_type=str(self.config.calibration_callability_model),
                                min_train_rows=int(self.config.calibration_callability_min_train_rows),
                                min_call_rate=float(self.config.calibration_callability_min_call_rate),
                                call_rate_weight=float(self.config.calibration_callability_call_rate_weight),
                                stitch_gp_threshold=float(self.config.genotype_call_stitch_threshold),
                                bound_to_stitch=bool(self.config.calibration_callability_bound_to_stitch),
                                min_call_rate_delta=float(self.config.calibration_callability_min_call_rate_delta),
                                max_call_rate_delta=float(self.config.calibration_callability_max_call_rate_delta),
                                validation_site_fraction=float(self.config.calibration_callability_validation_site_fraction),
                                min_objective_improvement=float(self.config.calibration_callability_min_objective_improvement),
                                max_hardcall_maf_shift=float(self.config.calibration_callability_max_hardcall_maf_shift),
                                max_hardcall_het_shift=float(self.config.calibration_callability_max_hardcall_het_shift),
                                decision_mode=str(self.config.calibration_callability_decision_mode),
                                seed=int(self.config.random_seed + block.block_id),
                            )
                            calibration_meta = {
                                "status": str(callability_meta.get("status", "ok")),
                                "mode": "standard_callability",
                                "raw_gp_primary": True,
                                "posterior_calibration": "none",
                                "truth": truth_meta,
                                "call_correctness": callability_meta,
                            }
                        elif (
                            self.config.calibrate_genotype_posteriors
                            and self.config.calibration_mode == "standard_callability"
                        ):
                            calibration_meta = {
                                "status": "skipped_no_truth",
                                "mode": "standard_callability",
                                "raw_gp_primary": artifacts.genotype_posterior is not None,
                                "posterior_calibration": "none",
                                "truth": truth_meta,
                            }
                        if (
                            self.config.calibrate_genotype_posteriors
                            and self.config.calibration_mode == "masked_cv"
                            and truth_gt_block is not None
                            and int(np.sum(truth_gt_block >= 0)) >= 16
                        ):
                            calibrated_gp, calibration_meta = masked_cv_calibrate_genotype_posterior(
                                raw_posterior=artifacts.genotype_posterior,
                                dosage=dosage,
                                truth_genotype=truth_gt_block,
                                train_mask=truth_gt_block >= 0,
                                depth=evidence.depth.astype(np.float32, copy=False),
                                maf_bins=self.config.calibration_maf_bins,
                                temperatures=self.config.calibration_temperatures,
                                blends=self.config.calibration_blends,
                                dosage_scales=self.config.calibration_dosage_scales,
                                dosage_offsets=self.config.calibration_dosage_offsets,
                                hwe_prior_weights=self.config.calibration_hwe_prior_weights,
                                optimize_dosage_scale=bool(self.config.calibration_optimize_dosage_scale),
                                hwe_weight=float(self.config.calibration_hwe_weight),
                                hwe_min_maf=float(self.config.calibration_hwe_min_maf),
                                ploidy=max_ploidy,
                            )
                            calibration_meta["truth"] = truth_meta
                        if self.config.use_lightgbm_calibrator and calibrated_gp is not None and calibrated_gp.shape[2] == 3:
                            if truth_gt_block is not None and int(np.sum(truth_gt_block >= 0)) >= 128:
                                train_positions = self._calibration_train_position_index(
                                    calibrated_gp.shape[1],
                                    seed=int(self.config.random_seed + block.block_id),
                                )
                                lgbm_input_gp = (
                                    artifacts.genotype_posterior
                                    if artifacts.genotype_posterior is not None
                                    else calibrated_gp
                                )
                                calibrated_gp, call_correct_probability, calibration_meta = calibrate_genotype_posterior_full_stack(
                                    raw_posterior=lgbm_input_gp,
                                    dosage=dosage,
                                    truth_genotype=truth_gt_block,
                                    depth=evidence.depth.astype(np.float32, copy=False),
                                    ref_count=evidence.ref_count.astype(np.float32, copy=False),
                                    alt_count=evidence.alt_count.astype(np.float32, copy=False),
                                    other_count=evidence.other_count.astype(np.float32, copy=False),
                                    support_mask=support_mask.astype(np.float32, copy=False),
                                    generations=generations.astype(np.float32, copy=False),
                                    samples_df=samples,
                                    train_position_index=train_positions,
                                    predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                                    window=int(self.config.calibration_context_window),
                                    block_size=int(self.config.calibration_block_snps),
                                    max_train_rows=int(self.config.calibration_max_train_rows),
                                    use_optuna=bool(self.config.calibration_use_optuna),
                                    optuna_trials=int(self.config.calibration_optuna_trials),
                                    use_block_context=bool(self.config.calibration_lightgbm_use_block_context),
                                    use_fixed_stage0_calibration=bool(self.config.calibration_lightgbm_use_fixed_stage0),
                                    seed=int(self.config.random_seed + block.block_id),
                                    class_weight_mode="balanced",
                                    apply_isotonic=True,
                                )
                                calibration_meta["truth"] = truth_meta
                            elif truth_gt_block is not None:
                                train_positions = self._calibration_train_position_index(
                                    calibrated_gp.shape[1],
                                    seed=int(self.config.random_seed + block.block_id),
                                )
                                calibrated_gp, calibration_meta = calibrate_genotype_posterior_block_context(
                                    raw_posterior=artifacts.genotype_posterior,
                                    dosage=dosage,
                                    truth_genotype=truth_gt_block,
                                    samples_df=samples,
                                    train_position_index=train_positions,
                                    predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                                    window=int(self.config.calibration_context_window),
                                    block_size=int(self.config.calibration_block_snps),
                                    max_train_rows=int(self.config.calibration_max_train_rows),
                                    use_optuna=bool(self.config.calibration_use_optuna),
                                    optuna_trials=int(self.config.calibration_optuna_trials),
                                    seed=int(self.config.random_seed + block.block_id),
                                )
                                calibration_meta["truth"] = truth_meta
                        if bool(self.config.calibration_sanity_checks) and calibrated_gp is not None:
                            guarded_gp, sanity_meta = apply_calibration_sanity_guard(
                                artifacts.genotype_posterior,
                                calibrated_gp,
                                max_mean_abs_dosage_shift=float(self.config.calibration_max_mean_abs_dosage_shift),
                                max_mean_abs_maf_shift=float(self.config.calibration_max_mean_abs_maf_shift),
                                max_het_rate_shift=float(self.config.calibration_max_het_rate_shift),
                                max_mean_entropy_shift=float(self.config.calibration_max_mean_entropy_shift),
                            )
                            if isinstance(calibration_meta, dict):
                                calibration_meta["sanity"] = sanity_meta
                            else:
                                calibration_meta = {"status": "ok", "sanity": sanity_meta}
                            if sanity_meta.get("status") == "fallback_to_raw":
                                calibrated_gp = guarded_gp
                                call_correct_probability = None
                            else:
                                calibrated_gp = guarded_gp
                        if self.config.pedigree_mode in {"kinship", "transmission"}:
                            pedigree_result = apply_pedigree_adjustment(
                                dosage=dosage,
                                genotype_posterior=calibrated_gp,
                                pedigree=pedigree,
                                mode=self.config.pedigree_mode,
                                strength=self.config.pedigree_strength,
                                positions=block_founders.positions,
                                generations=generations,
                                support_mask=support_mask,
                                iterations=self.config.pedigree_iterations,
                                kinship_threshold=self.config.pedigree_kinship_threshold,
                            )
                            dosage = pedigree_result.dosage
                            calibrated_gp = pedigree_result.genotype_posterior
                        calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
                        stitch_threshold = None
                        min_confidence = 0.0
                        min_margin = 0.0
                        call_correct_threshold = 0.0
                        if self.config.genotype_call_mode == "stitch_no_call":
                            stitch_threshold = float(self.config.genotype_call_stitch_threshold)
                            min_margin = float(self.config.genotype_call_min_margin)
                        elif self.config.genotype_call_mode == "quality_gated":
                            min_confidence = float(self.config.genotype_call_min_confidence)
                            min_margin = float(self.config.genotype_call_min_margin)
                            call_correct_threshold = float(self.config.genotype_call_correctness_threshold)
                            if (
                                isinstance(calibration_meta, dict)
                                and self._callability_fell_back_to_stitch(calibration_meta)
                            ):
                                stitch_threshold = float(self.config.genotype_call_stitch_threshold)
                                call_correct_probability = None
                                call_correct_threshold = 0.0
                            elif isinstance(calibration_meta, dict):
                                call_correct_probability, call_correct_threshold = self._callability_threshold_from_meta(
                                    calibration_meta=calibration_meta,
                                    call_correct_probability=call_correct_probability,
                                    genotype_posterior=calibrated_gp,
                                )
                        calibrated_gt = genotype_call_from_posterior(
                            calibrated_gp,
                            min_confidence=min_confidence,
                            min_margin=min_margin,
                            stitch_gp_threshold=stitch_threshold,
                            call_correct_probability=call_correct_probability,
                            call_correct_threshold=call_correct_threshold,
                        )
                    elif requested_genotype_outputs:
                        calibrated_gp = dosage_to_genotype_posterior(
                            np.clip(dosage, 0.0, max(max_ploidy, 0)).astype(np.float32, copy=False),
                            temperature=self.config.genotype_posterior_temperature,
                            ploidy=max_ploidy,
                        )
                        calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
                        max_gt = np.broadcast_to(sample_ploidy[:, None], dosage.shape)
                        gt = np.full(dosage.shape, -1, dtype=np.int8)
                        valid = np.isfinite(dosage) & (max_gt > 0)
                        if np.any(valid):
                            rounded = np.rint(dosage).astype(np.float32, copy=False)
                            clipped = np.clip(rounded, 0.0, max_gt.astype(np.float32, copy=False))
                            gt[valid] = clipped[valid].astype(np.int8, copy=False)
                        calibrated_gt = gt
                    if calibrated_gt is not None:
                        valid_call = np.isfinite(dosage) & (sample_ploidy[:, None] > 0)
                        calibrated_gt = np.where(valid_call, calibrated_gt, -1).astype(np.int8, copy=False)
                    output_block = block
                    output_evidence = evidence
                    output_artifacts = artifacts
                    output_dosage = dosage
                    output_support_mask = support_mask
                    output_calibrated_gp = calibrated_gp
                    output_calibrated_gt = calibrated_gt
                    output_calibration_meta = calibration_meta
                    if bool(getattr(block, "has_core_overlap", False)):
                        core_start, core_stop = block.core_slice
                        output_block = block.core_block()
                        output_evidence = evidence.slice_by_position_rows(
                            core_start,
                            core_stop,
                            block_id=int(block.block_id),
                        )
                        output_artifacts = self._slice_artifacts_by_position(artifacts, core_start, core_stop)
                        output_dosage = dosage[:, core_start:core_stop]
                        output_support_mask = support_mask[:, core_start:core_stop]
                        output_calibrated_gp = (
                            None if calibrated_gp is None else calibrated_gp[:, core_start:core_stop]
                        )
                        output_calibrated_gt = (
                            None if calibrated_gt is None else calibrated_gt[:, core_start:core_stop]
                        )
                        if isinstance(calibration_meta, dict):
                            output_calibration_meta = self._slice_calibration_meta_by_position(
                                calibration_meta,
                                start=core_start,
                                stop=core_stop,
                            )
                    if isinstance(output_calibration_meta, dict):
                        self._write_calibration_decisions(block=output_block, calibration_meta=output_calibration_meta)
                    t_after_calibration = time.perf_counter()
                    rss_after_calibration = _current_rss_mb() if self.config.profile_memory else None
                    xi_write_meta: dict[str, object] | None = None
                    if self._should_materialize_xi(memory_plan):
                        xi_write_meta = self._write_xi_zarr_for_block(
                            block=block,
                            block_founders=block_founders,
                            evidence=evidence,
                            artifacts=artifacts,
                            sample_ploidy=sample_ploidy,
                        )
                    t_after_xi = time.perf_counter()
                    self._write_block_diagnostics(
                        block=output_block,
                        dosage=output_dosage,
                        raw_posterior=output_artifacts.genotype_posterior,
                        calibrated_posterior=output_calibrated_gp,
                        genotype_call=output_calibrated_gt,
                        evidence=output_evidence,
                        support_mask=output_support_mask,
                        sample_ploidy=sample_ploidy,
                    )
                    self._write_hmm_outputs(
                        sample_ids,
                        output_block,
                        output_dosage,
                        output_artifacts,
                        support_mask=output_support_mask,
                        genotype_posterior=output_calibrated_gp,
                        genotype_call=output_calibrated_gt,
                    )
                    if output_evidence is not evidence:
                        output_evidence.release()
                    t_after_write = time.perf_counter()
                    rss_after_write = _current_rss_mb() if self.config.profile_memory else None
                    block_timing = {
                        "block_id": block.block_id,
                        "seconds_read_extract": t_after_reads - block_t0,
                        "seconds_hmm": t_after_hmm - t_after_reads,
                        "seconds_calibration": t_after_calibration - t_after_hmm,
                        "seconds_xi_write": t_after_xi - t_after_calibration,
                        "seconds_write": t_after_write - t_after_xi,
                        "seconds_total": t_after_write - block_t0,
                        "mean_depth": float(np.mean(evidence.depth)),
                        "n_reads": int(np.sum(evidence.n_overlapping_reads)),
                        "configured_block_size": int(self.config.block_size),
                        "effective_block_size": int(effective_block_size),
                        "effective_io_window_size": int(effective_io_window_size),
                        "output_row_start": int(output_block.row_start),
                        "output_row_stop": int(output_block.row_stop),
                        "input_row_start": int(block.row_start),
                        "input_row_stop": int(block.row_stop),
                        "overlap_left_snps": int(output_block.row_start - block.row_start),
                        "overlap_right_snps": int(block.row_stop - output_block.row_stop),
                        "max_mem_bytes": int(memory_plan.max_mem_bytes),
                        "planned_memory_budget_bytes": int(memory_plan.planned_budget_bytes),
                        "memory_safety_fraction": float(memory_plan.safety_fraction),
                        "snp_block_mode": str(memory_plan.snp_block_mode),
                        "snp_block_mode_approximate": bool(memory_plan.approximate),
                        "xi_policy": str(memory_plan.xi_policy),
                        "xi_decision": str(memory_plan.xi_decision),
                        "xi_estimated_bytes": int(memory_plan.xi_estimated_bytes),
                        "gamma_policy": str(memory_plan.gamma_policy),
                        "gamma_estimated_bytes": int(memory_plan.gamma_estimated_bytes),
                    }
                    if xi_write_meta is not None:
                        block_timing["xi_materialized"] = bool(xi_write_meta.get("materialized", False))
                        if xi_write_meta.get("file_bytes") is not None:
                            block_timing["xi_file_bytes"] = int(xi_write_meta.get("file_bytes", 0))
                        if xi_write_meta.get("raw_bytes") is not None:
                            block_timing["xi_raw_bytes"] = int(xi_write_meta.get("raw_bytes", 0))
                        if xi_write_meta.get("compression_ratio_raw_to_file") is not None:
                            block_timing["xi_compression_ratio_raw_to_file"] = float(
                                xi_write_meta.get("compression_ratio_raw_to_file")
                            )
                    block_timing.update(compact_cache_stats)
                    if calibration_split_meta and calibration_split_meta.get("status") != "disabled":
                        block_timing["calibration_read_holdout_status"] = str(calibration_split_meta.get("status"))
                        block_timing["calibration_label_holdout_only"] = bool(
                            calibration_split_meta.get("calibration_label_holdout_only", False)
                        )
                        block_timing["calibration_hmm_uses_all_read_evidence"] = bool(
                            calibration_split_meta.get("hmm_uses_all_read_evidence", False)
                        )
                        if calibration_split_meta.get("holdout_fraction_effective") is not None:
                            block_timing["calibration_read_holdout_fraction"] = float(
                                calibration_split_meta.get("holdout_fraction_effective")
                            )
                        if calibration_split_meta.get("n_calibration_fragments") is not None:
                            block_timing["calibration_read_holdout_fragments"] = int(
                                calibration_split_meta.get("n_calibration_fragments")
                            )
                    if calibrated_gt is not None:
                        block_timing["call_rate"] = float(np.mean(calibrated_gt >= 0))
                        block_timing["no_call_rate"] = float(np.mean(calibrated_gt < 0))
                    if getattr(artifacts, "em_diagnostics", None):
                        em_diag = artifacts.em_diagnostics or {}
                        block_timing["em_updates"] = int(em_diag.get("em_updates", 0))
                        block_timing["em_best_iteration"] = int(em_diag.get("best_iteration", -1))
                        block_timing["em_restored_best_founders"] = bool(em_diag.get("restored_best_founders", False))
                        if em_diag.get("final_read_log_likelihood") is not None:
                            block_timing["em_final_read_log_likelihood"] = float(em_diag.get("final_read_log_likelihood"))
                    if isinstance(calibration_meta, dict):
                        status = calibration_meta.get("status")
                        mode = calibration_meta.get("mode")
                        if status is not None:
                            block_timing["calibration_status"] = str(status)
                        if mode is not None:
                            block_timing["calibration_mode"] = str(mode)
                        truth = calibration_meta.get("truth")
                        if isinstance(truth, dict):
                            if truth.get("source") is not None:
                                block_timing["calibration_truth_source"] = str(truth.get("source"))
                            if truth.get("n_labeled") is not None:
                                block_timing["calibration_truth_n_labeled"] = int(truth.get("n_labeled"))
                        cc = calibration_meta.get("call_correctness")
                        if isinstance(cc, dict):
                            if cc.get("decision_mode") is not None:
                                block_timing["calibration_decision_mode"] = str(cc.get("decision_mode"))
                            thr = cc.get("threshold")
                            if thr is not None:
                                block_timing["call_correctness_threshold"] = float(thr)
                            for key in (
                                "n_snps_calibration_used",
                                "n_snps_fallback_to_stitch",
                            ):
                                if cc.get(key) is not None:
                                    block_timing[key] = int(cc.get(key))
                            decision_counts = cc.get("decision_source_counts")
                            if isinstance(decision_counts, dict):
                                block_timing["calibration_decision_source_counts"] = {
                                    str(k): int(v) for k, v in decision_counts.items()
                                }
                            fallback_counts = cc.get("fallback_reason_counts")
                            if isinstance(fallback_counts, dict):
                                block_timing["calibration_fallback_reason_counts"] = {
                                    str(k): int(v) for k, v in fallback_counts.items()
                                }
                    if pedigree_result is not None:
                        block_timing["pedigree_mode"] = str(pedigree_result.mode)
                        summary = pedigree_result.summary
                        if summary:
                            block_timing["pedigree_edges"] = int(summary.get("n_edges", 0))
                            block_timing["pedigree_components"] = int(summary.get("n_components", 0))
                            if "messages" in summary:
                                block_timing["pedigree_messages"] = int(summary.get("messages", 0))
                    if self.config.profile_memory:
                        block_timing.update(
                            {
                                "rss_mb_block_start": float(rss_block_start or 0.0),
                                "rss_mb_after_reads": float(rss_after_reads or 0.0),
                                "rss_mb_after_hmm": float(rss_after_hmm or 0.0),
                                "rss_mb_after_calibration": float(rss_after_calibration or 0.0),
                                "rss_mb_after_write": float(rss_after_write or 0.0),
                            }
                        )
                    timings.append(block_timing)
                finally:
                    if calibration_truth_evidence is not None and calibration_truth_evidence is not evidence:
                        calibration_truth_evidence.release()
                    if evidence is not full_evidence:
                        evidence.release()
                    if read_evidence is not None:
                        read_evidence.release()
                    full_evidence.release()
                    if self.config.gc_collect_every_block:
                        gc.collect()
        finally:
            if int(read_samples.shape[0]) > 0:
                self.read_extractor.close()

        timings_path = self.output_dir / "stage_timings.json"
        timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        if self.config.profile_memory and timings:
            peak_rss = max(float(row.get("rss_mb_after_write", 0.0)) for row in timings)
            peak_hmm_rss = max(float(row.get("rss_mb_after_hmm", 0.0)) for row in timings)
            mem_summary = {
                "peak_rss_mb": peak_rss,
                "peak_hmm_rss_mb": peak_hmm_rss,
                "blocks_profiled": len(timings),
            }
            (self.output_dir / "memory_profile_summary.json").write_text(
                json.dumps(mem_summary, indent=2),
                encoding="utf-8",
            )
        self._finalize_diagnostics_summary()

    def _missing_artifacts(
        self,
        *,
        n_samples: int,
        n_positions: int,
        generations: np.ndarray,
        block_founders: FounderPanel,
        return_haplotype: bool,
        return_full_transition: bool,
    ) -> HMMArtifacts:
        base_switch = self.hmm._switch_probabilities(
            self.hmm.recombination_from_positions(block_founders.positions, block_founders.genetic_cm),
            generations,
        )
        return HMMArtifacts(
            dosage=np.full((n_samples, n_positions), np.nan, dtype=np.float32),
            haplotype_posterior=(
                np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
                if return_haplotype
                else None
            ),
            genotype_posterior=None,
            genotype_call=None,
            recombination_rate=self.hmm.recombination_from_positions(
                block_founders.positions,
                block_founders.genetic_cm,
            ).astype(np.float32),
            switch_probability=base_switch.astype(np.float32),
            stay_probability=(1.0 - base_switch).astype(np.float32),
            offdiag_probability=(base_switch / max(block_founders.n_founders - 1, 1)).astype(np.float32),
            founder_alt_prob=block_founders.alt_prob.astype(np.float32, copy=True),
            transition_probability=(
                self.hmm._build_full_transitions(base_switch, block_founders.n_founders)
                if return_full_transition
                else None
            ),
        )

    def _merge_dask_hmm_task_results(
        self,
        *,
        results: list[DaskHMMTaskResult],
        n_samples: int,
        n_positions: int,
        sample_ploidy: np.ndarray,
        max_ploidy: int,
        block_founders: FounderPanel,
        return_haplotype: bool,
        return_genotype_posterior: bool,
    ) -> HMMArtifacts:
        dosage_out = np.full((n_samples, n_positions), np.nan, dtype=np.float32)
        hap_out = (
            np.zeros((n_samples, n_positions, block_founders.n_founders), dtype=np.float32)
            if return_haplotype
            else None
        )
        gp_classes = max(max_ploidy + 1, 1)
        gp_out = np.zeros((n_samples, n_positions, gp_classes), dtype=np.float32) if return_genotype_posterior else None
        switch_out = np.zeros((n_samples, n_positions), dtype=np.float32)
        stay_out = np.zeros((n_samples, n_positions), dtype=np.float32)
        offdiag_out = np.zeros((n_samples, n_positions), dtype=np.float32)
        founder_acc = np.zeros_like(block_founders.alt_prob, dtype=np.float32)
        founder_w = 0.0
        transition_probability = None
        transition_factor_source = None
        transition_factor_destination = None
        transition_factor_offdiag = None
        task_em_diagnostics: list[dict[str, object]] = []

        for result in results:
            idx = result.sample_indices.astype(np.int64, copy=False)
            sub = self._load_dask_hmm_task_artifacts(result)
            if sub.em_diagnostics is not None:
                task_em_diagnostics.append(
                    {
                        "ploidy": int(result.ploidy),
                        "n_samples": int(idx.size),
                        **dict(sub.em_diagnostics),
                    }
                )
            dosage_out[idx] = sub.dosage.astype(np.float32, copy=False)
            if hap_out is not None and sub.haplotype_posterior is not None:
                hap_out[idx] = sub.haplotype_posterior
            if gp_out is not None and sub.genotype_posterior is not None:
                n_cls = min(gp_out.shape[2], sub.genotype_posterior.shape[2])
                gp_out[idx, :, :n_cls] = sub.genotype_posterior[:, :, :n_cls]
            switch_out[idx] = sub.switch_probability
            stay_out[idx] = sub.stay_probability
            offdiag_out[idx] = sub.offdiag_probability
            founder_acc += sub.founder_alt_prob.astype(np.float32, copy=False) * float(idx.size)
            founder_w += float(idx.size)
            if (
                len(results) == 1
                and idx.size == n_samples
                and np.array_equal(idx, np.arange(n_samples, dtype=np.int64))
            ):
                transition_probability = sub.transition_probability
                transition_factor_source = sub.transition_factor_source
                transition_factor_destination = sub.transition_factor_destination
                transition_factor_offdiag = sub.transition_factor_offdiag
            elif transition_factor_offdiag is None and sub.transition_factor_offdiag is not None:
                transition_factor_source = sub.transition_factor_source
                transition_factor_destination = sub.transition_factor_destination
                transition_factor_offdiag = sub.transition_factor_offdiag

        if np.any(sample_ploidy == 0):
            zero_idx = np.flatnonzero(sample_ploidy == 0)
            dosage_out[zero_idx] = np.nan
            if gp_out is not None:
                gp_out[zero_idx] = np.nan

        founder_final = (
            founder_acc / max(founder_w, 1.0)
            if founder_w > 0.0
            else block_founders.alt_prob.astype(np.float32, copy=True)
        )
        gt_out = None
        if gp_out is not None:
            gt_out = np.argmax(np.nan_to_num(gp_out, nan=-1.0), axis=2).astype(np.int8, copy=False)
            gt_out[~np.isfinite(gp_out).any(axis=2)] = -1
        em_diagnostics = None
        if task_em_diagnostics:
            read_ll: list[float] = []
            for row in task_em_diagnostics:
                try:
                    val = float(row.get("final_read_log_likelihood", np.nan))
                except (TypeError, ValueError):
                    val = float("nan")
                if np.isfinite(val):
                    read_ll.append(val)
            em_diagnostics = {
                "executor": "dask",
                "n_tasks": int(len(task_em_diagnostics)),
                "em_updates": int(sum(int(row.get("em_updates", 0)) for row in task_em_diagnostics)),
                "final_read_log_likelihood": float(np.sum(read_ll)) if read_ll else float("nan"),
                "task_diagnostics": task_em_diagnostics,
            }
        return HMMArtifacts(
            dosage=dosage_out,
            haplotype_posterior=hap_out,
            genotype_posterior=gp_out,
            genotype_call=gt_out,
            recombination_rate=self.hmm.recombination_from_positions(
                block_founders.positions,
                block_founders.genetic_cm,
            ).astype(np.float32),
            switch_probability=switch_out,
            stay_probability=stay_out,
            offdiag_probability=offdiag_out,
            founder_alt_prob=founder_final.astype(np.float32, copy=False),
            transition_probability=transition_probability,
            transition_factor_source=transition_factor_source,
            transition_factor_destination=transition_factor_destination,
            transition_factor_offdiag=transition_factor_offdiag,
            em_diagnostics=em_diagnostics,
        )

    @staticmethod
    def _load_dask_hmm_task_artifacts(result: DaskHMMTaskResult) -> HMMArtifacts:
        if result.artifacts is not None:
            return result.artifacts
        sub = load_hmm_artifacts_npz(result.artifact_path)
        Path(result.artifact_path).unlink(missing_ok=True)
        return sub

    def _finalize_and_write_block(
        self,
        *,
        samples: pd.DataFrame,
        sample_ids: np.ndarray,
        sample_ploidy: np.ndarray,
        generations: np.ndarray,
        pedigree,
        block,
        block_start_offset: int,
        evidence: ReadEvidenceBlock,
        support_mask: np.ndarray,
        artifacts: HMMArtifacts,
        need_genotype_posterior: bool,
        requested_genotype_outputs: bool,
        max_ploidy: int,
        block_t0: float,
        t_after_reads: float,
        t_after_hmm: float,
        rss_block_start: float | None,
        rss_after_reads: float | None,
        calibration_truth_evidence: ReadEvidenceBlock | None = None,
        calibration_split_meta: dict[str, object] | None = None,
        dask_extra: dict[str, float | int | str | bool] | None = None,
    ) -> dict[str, float | int | str | bool]:
        rss_after_hmm = _current_rss_mb() if self.config.profile_memory else None
        dosage = artifacts.dosage.astype(np.float32, copy=False)
        if self.config.pedigree_mode == "smooth":
            pedigree_result = apply_pedigree_adjustment(
                dosage=dosage,
                genotype_posterior=None,
                pedigree=pedigree,
                mode=self.config.pedigree_mode,
                strength=self.config.pedigree_strength,
                positions=block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False),
                generations=generations,
                support_mask=support_mask,
                iterations=self.config.pedigree_iterations,
                kinship_threshold=self.config.pedigree_kinship_threshold,
            )
            dosage = pedigree_result.dosage
        else:
            pedigree_result = None
        calibrated_gp = None
        calibrated_gt = None
        call_correct_probability = None
        calibration_meta: dict[str, object] = {"status": "disabled"}
        if need_genotype_posterior:
            if artifacts.genotype_posterior is not None:
                calibrated_gp = artifacts.genotype_posterior.astype(np.float32, copy=False)
                calibration_meta = {
                    "status": "raw_hmm_gp",
                    "mode": "raw_hmm_gp",
                    "raw_gp_primary": True,
                    "posterior_calibration": "none",
                }
            else:
                calibrated_gp = calibrate_genotype_posterior(
                    None,
                    dosage=dosage,
                    depth=evidence.depth.astype(np.float32, copy=False),
                    temperature=self.config.genotype_posterior_temperature,
                    blend=0.0,
                    ploidy=max_ploidy,
                )
                calibration_meta = {
                    "status": "dosage_fallback",
                    "mode": "dosage_fallback",
                    "raw_gp_primary": False,
                    "posterior_calibration": "none",
                    "temperature": float(self.config.genotype_posterior_temperature),
                    "blend": 0.0,
                }
            if self.config.calibrate_genotype_posteriors and self.config.calibration_mode == "fixed":
                calibrated_gp = calibrate_genotype_posterior(
                    artifacts.genotype_posterior,
                    dosage=dosage,
                    depth=evidence.depth.astype(np.float32, copy=False),
                    temperature=self.config.genotype_posterior_temperature,
                    blend=self.config.genotype_posterior_blend,
                    ploidy=max_ploidy,
                )
                calibration_meta = {
                    "status": "applied",
                    "mode": "fixed",
                    "raw_gp_primary": False,
                    "posterior_calibration": "temperature_blend",
                    "temperature": float(self.config.genotype_posterior_temperature),
                    "blend": float(self.config.genotype_posterior_blend),
                }
            calibration_label_evidence = calibration_truth_evidence if calibration_truth_evidence is not None else evidence
            truth_gt_block, truth_meta = self._calibration_truth_block(
                evidence=calibration_label_evidence,
                block_start=int(block_start_offset),
                n_positions=int(calibrated_gp.shape[1]),
                gp_shape=calibrated_gp.shape,
                sample_ploidy=sample_ploidy,
            )
            if isinstance(truth_meta, dict) and calibration_truth_evidence is not None:
                truth_meta = dict(truth_meta)
                truth_meta["read_evidence_holdout"] = dict(calibration_split_meta or {})
            if (
                self.config.calibrate_genotype_posteriors
                and self.config.calibration_mode == "standard_callability"
                and truth_gt_block is not None
                and calibrated_gp is not None
                and calibrated_gp.shape[2] == 3
            ):
                train_positions = self._calibration_train_position_index(
                    calibrated_gp.shape[1],
                    seed=int(self.config.random_seed + block.block_id),
                )
                train_site_mask = np.isin(
                    np.arange(calibrated_gp.shape[1], dtype=np.int64),
                    train_positions,
                )
                train_mask = (truth_gt_block >= 0) & train_site_mask[None, :]
                call_correct_probability, callability_meta = train_standard_callability_model(
                    posterior=calibrated_gp,
                    dosage=dosage,
                    truth_genotype=truth_gt_block,
                    train_mask=train_mask,
                    depth=evidence.depth.astype(np.float32, copy=False),
                    ref_count=evidence.ref_count.astype(np.float32, copy=False),
                    alt_count=evidence.alt_count.astype(np.float32, copy=False),
                    other_count=evidence.other_count.astype(np.float32, copy=False),
                    support_mask=support_mask.astype(bool, copy=False),
                    max_train_rows=int(self.config.calibration_max_train_rows),
                    model_type=str(self.config.calibration_callability_model),
                    min_train_rows=int(self.config.calibration_callability_min_train_rows),
                    min_call_rate=float(self.config.calibration_callability_min_call_rate),
                    call_rate_weight=float(self.config.calibration_callability_call_rate_weight),
                    stitch_gp_threshold=float(self.config.genotype_call_stitch_threshold),
                    bound_to_stitch=bool(self.config.calibration_callability_bound_to_stitch),
                    min_call_rate_delta=float(self.config.calibration_callability_min_call_rate_delta),
                    max_call_rate_delta=float(self.config.calibration_callability_max_call_rate_delta),
                    validation_site_fraction=float(self.config.calibration_callability_validation_site_fraction),
                    min_objective_improvement=float(self.config.calibration_callability_min_objective_improvement),
                    max_hardcall_maf_shift=float(self.config.calibration_callability_max_hardcall_maf_shift),
                    max_hardcall_het_shift=float(self.config.calibration_callability_max_hardcall_het_shift),
                    decision_mode=str(self.config.calibration_callability_decision_mode),
                    seed=int(self.config.random_seed + block.block_id),
                )
                calibration_meta = {
                    "status": str(callability_meta.get("status", "ok")),
                    "mode": "standard_callability",
                    "raw_gp_primary": True,
                    "posterior_calibration": "none",
                    "truth": truth_meta,
                    "call_correctness": callability_meta,
                }
            elif (
                self.config.calibrate_genotype_posteriors
                and self.config.calibration_mode == "standard_callability"
            ):
                calibration_meta = {
                    "status": "skipped_no_truth",
                    "mode": "standard_callability",
                    "raw_gp_primary": artifacts.genotype_posterior is not None,
                    "posterior_calibration": "none",
                    "truth": truth_meta,
                }
            if (
                self.config.calibrate_genotype_posteriors
                and self.config.calibration_mode == "masked_cv"
                and truth_gt_block is not None
                and int(np.sum(truth_gt_block >= 0)) >= 16
            ):
                calibrated_gp, calibration_meta = masked_cv_calibrate_genotype_posterior(
                    raw_posterior=artifacts.genotype_posterior,
                    dosage=dosage,
                    truth_genotype=truth_gt_block,
                    train_mask=truth_gt_block >= 0,
                    depth=evidence.depth.astype(np.float32, copy=False),
                    maf_bins=self.config.calibration_maf_bins,
                    temperatures=self.config.calibration_temperatures,
                    blends=self.config.calibration_blends,
                    dosage_scales=self.config.calibration_dosage_scales,
                    dosage_offsets=self.config.calibration_dosage_offsets,
                    hwe_prior_weights=self.config.calibration_hwe_prior_weights,
                    optimize_dosage_scale=bool(self.config.calibration_optimize_dosage_scale),
                    hwe_weight=float(self.config.calibration_hwe_weight),
                    hwe_min_maf=float(self.config.calibration_hwe_min_maf),
                    ploidy=max_ploidy,
                )
                calibration_meta["truth"] = truth_meta
            if self.config.use_lightgbm_calibrator and calibrated_gp is not None and calibrated_gp.shape[2] == 3:
                if truth_gt_block is not None and int(np.sum(truth_gt_block >= 0)) >= 128:
                    train_positions = self._calibration_train_position_index(
                        calibrated_gp.shape[1],
                        seed=int(self.config.random_seed + block.block_id),
                    )
                    lgbm_input_gp = (
                        artifacts.genotype_posterior
                        if artifacts.genotype_posterior is not None
                        else calibrated_gp
                    )
                    calibrated_gp, call_correct_probability, calibration_meta = calibrate_genotype_posterior_full_stack(
                        raw_posterior=lgbm_input_gp,
                        dosage=dosage,
                        truth_genotype=truth_gt_block,
                        depth=evidence.depth.astype(np.float32, copy=False),
                        ref_count=evidence.ref_count.astype(np.float32, copy=False),
                        alt_count=evidence.alt_count.astype(np.float32, copy=False),
                        other_count=evidence.other_count.astype(np.float32, copy=False),
                        support_mask=support_mask.astype(np.float32, copy=False),
                        generations=generations.astype(np.float32, copy=False),
                        samples_df=samples,
                        train_position_index=train_positions,
                        predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                        window=int(self.config.calibration_context_window),
                        block_size=int(self.config.calibration_block_snps),
                        max_train_rows=int(self.config.calibration_max_train_rows),
                        use_optuna=bool(self.config.calibration_use_optuna),
                        optuna_trials=int(self.config.calibration_optuna_trials),
                        use_block_context=bool(self.config.calibration_lightgbm_use_block_context),
                        use_fixed_stage0_calibration=bool(self.config.calibration_lightgbm_use_fixed_stage0),
                        seed=int(self.config.random_seed + block.block_id),
                        class_weight_mode="balanced",
                        apply_isotonic=True,
                    )
                    calibration_meta["truth"] = truth_meta
                elif truth_gt_block is not None:
                    train_positions = self._calibration_train_position_index(
                        calibrated_gp.shape[1],
                        seed=int(self.config.random_seed + block.block_id),
                    )
                    calibrated_gp, calibration_meta = calibrate_genotype_posterior_block_context(
                        raw_posterior=artifacts.genotype_posterior,
                        dosage=dosage,
                        truth_genotype=truth_gt_block,
                        samples_df=samples,
                        train_position_index=train_positions,
                        predict_position_index=np.arange(calibrated_gp.shape[1], dtype=np.int64),
                        window=int(self.config.calibration_context_window),
                        block_size=int(self.config.calibration_block_snps),
                        max_train_rows=int(self.config.calibration_max_train_rows),
                        use_optuna=bool(self.config.calibration_use_optuna),
                        optuna_trials=int(self.config.calibration_optuna_trials),
                        seed=int(self.config.random_seed + block.block_id),
                    )
                    calibration_meta["truth"] = truth_meta
            if bool(self.config.calibration_sanity_checks) and calibrated_gp is not None:
                guarded_gp, sanity_meta = apply_calibration_sanity_guard(
                    artifacts.genotype_posterior,
                    calibrated_gp,
                    max_mean_abs_dosage_shift=float(self.config.calibration_max_mean_abs_dosage_shift),
                    max_mean_abs_maf_shift=float(self.config.calibration_max_mean_abs_maf_shift),
                    max_het_rate_shift=float(self.config.calibration_max_het_rate_shift),
                    max_mean_entropy_shift=float(self.config.calibration_max_mean_entropy_shift),
                )
                if isinstance(calibration_meta, dict):
                    calibration_meta["sanity"] = sanity_meta
                else:
                    calibration_meta = {"status": "ok", "sanity": sanity_meta}
                if sanity_meta.get("status") == "fallback_to_raw":
                    calibrated_gp = guarded_gp
                    call_correct_probability = None
                else:
                    calibrated_gp = guarded_gp
            if self.config.pedigree_mode in {"kinship", "transmission"}:
                pedigree_result = apply_pedigree_adjustment(
                    dosage=dosage,
                    genotype_posterior=calibrated_gp,
                    pedigree=pedigree,
                    mode=self.config.pedigree_mode,
                    strength=self.config.pedigree_strength,
                    positions=block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False),
                    generations=generations,
                    support_mask=support_mask,
                    iterations=self.config.pedigree_iterations,
                    kinship_threshold=self.config.pedigree_kinship_threshold,
                )
                dosage = pedigree_result.dosage
                calibrated_gp = pedigree_result.genotype_posterior
            calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
            stitch_threshold = None
            min_confidence = 0.0
            min_margin = 0.0
            call_correct_threshold = 0.0
            if self.config.genotype_call_mode == "stitch_no_call":
                stitch_threshold = float(self.config.genotype_call_stitch_threshold)
                min_margin = float(self.config.genotype_call_min_margin)
            elif self.config.genotype_call_mode == "quality_gated":
                min_confidence = float(self.config.genotype_call_min_confidence)
                min_margin = float(self.config.genotype_call_min_margin)
                call_correct_threshold = float(self.config.genotype_call_correctness_threshold)
                if isinstance(calibration_meta, dict) and self._callability_fell_back_to_stitch(calibration_meta):
                    stitch_threshold = float(self.config.genotype_call_stitch_threshold)
                    call_correct_probability = None
                    call_correct_threshold = 0.0
                elif isinstance(calibration_meta, dict):
                    call_correct_probability, call_correct_threshold = self._callability_threshold_from_meta(
                        calibration_meta=calibration_meta,
                        call_correct_probability=call_correct_probability,
                        genotype_posterior=calibrated_gp,
                    )
            calibrated_gt = genotype_call_from_posterior(
                calibrated_gp,
                min_confidence=min_confidence,
                min_margin=min_margin,
                stitch_gp_threshold=stitch_threshold,
                call_correct_probability=call_correct_probability,
                call_correct_threshold=call_correct_threshold,
            )
        elif requested_genotype_outputs:
            calibrated_gp = dosage_to_genotype_posterior(
                np.clip(dosage, 0.0, max(max_ploidy, 0)).astype(np.float32, copy=False),
                temperature=self.config.genotype_posterior_temperature,
                ploidy=max_ploidy,
            )
            calibrated_gp = self._mask_genotype_posterior_by_ploidy(calibrated_gp, sample_ploidy)
            max_gt = np.broadcast_to(sample_ploidy[:, None], dosage.shape)
            gt = np.full(dosage.shape, -1, dtype=np.int8)
            valid = np.isfinite(dosage) & (max_gt > 0)
            if np.any(valid):
                rounded = np.rint(dosage).astype(np.float32, copy=False)
                clipped = np.clip(rounded, 0.0, max_gt.astype(np.float32, copy=False))
                gt[valid] = clipped[valid].astype(np.int8, copy=False)
            calibrated_gt = gt
        if calibrated_gt is not None:
            valid_call = np.isfinite(dosage) & (sample_ploidy[:, None] > 0)
            calibrated_gt = np.where(valid_call, calibrated_gt, -1).astype(np.int8, copy=False)
        if isinstance(calibration_meta, dict):
            self._write_calibration_decisions(block=block, calibration_meta=calibration_meta)
        t_after_calibration = time.perf_counter()
        rss_after_calibration = _current_rss_mb() if self.config.profile_memory else None
        self._write_block_diagnostics(
            block=block,
            dosage=dosage,
            raw_posterior=artifacts.genotype_posterior,
            calibrated_posterior=calibrated_gp,
            genotype_call=calibrated_gt,
            evidence=evidence,
            support_mask=support_mask,
            sample_ploidy=sample_ploidy,
        )
        self._write_hmm_outputs(
            sample_ids,
            block,
            dosage,
            artifacts,
            support_mask=support_mask,
            genotype_posterior=calibrated_gp,
            genotype_call=calibrated_gt,
        )
        t_after_write = time.perf_counter()
        rss_after_write = _current_rss_mb() if self.config.profile_memory else None
        block_timing: dict[str, float | int | str | bool] = {
            "block_id": block.block_id,
            "seconds_read_extract": t_after_reads - block_t0,
            "seconds_hmm": t_after_hmm - t_after_reads,
            "seconds_calibration": t_after_calibration - t_after_hmm,
            "seconds_write": t_after_write - t_after_calibration,
            "seconds_total": t_after_write - block_t0,
            "mean_depth": float(np.mean(evidence.depth)),
            "n_reads": int(np.sum(evidence.n_overlapping_reads)),
        }
        if calibrated_gt is not None:
            block_timing["call_rate"] = float(np.mean(calibrated_gt >= 0))
            block_timing["no_call_rate"] = float(np.mean(calibrated_gt < 0))
        if calibration_split_meta and calibration_split_meta.get("status") != "disabled":
            block_timing["calibration_read_holdout_status"] = str(calibration_split_meta.get("status"))
            block_timing["calibration_label_holdout_only"] = bool(
                calibration_split_meta.get("calibration_label_holdout_only", False)
            )
            block_timing["calibration_hmm_uses_all_read_evidence"] = bool(
                calibration_split_meta.get("hmm_uses_all_read_evidence", False)
            )
            if calibration_split_meta.get("holdout_fraction_effective") is not None:
                block_timing["calibration_read_holdout_fraction"] = float(
                    calibration_split_meta.get("holdout_fraction_effective")
                )
            if calibration_split_meta.get("n_calibration_fragments") is not None:
                block_timing["calibration_read_holdout_fragments"] = int(
                    calibration_split_meta.get("n_calibration_fragments")
                )
        if getattr(artifacts, "em_diagnostics", None):
            em_diag = artifacts.em_diagnostics or {}
            block_timing["em_updates"] = int(em_diag.get("em_updates", 0))
            block_timing["em_best_iteration"] = int(em_diag.get("best_iteration", -1))
            block_timing["em_restored_best_founders"] = bool(em_diag.get("restored_best_founders", False))
            if em_diag.get("final_read_log_likelihood") is not None:
                block_timing["em_final_read_log_likelihood"] = float(em_diag.get("final_read_log_likelihood"))
        if isinstance(calibration_meta, dict):
            status = calibration_meta.get("status")
            mode = calibration_meta.get("mode")
            if status is not None:
                block_timing["calibration_status"] = str(status)
            if mode is not None:
                block_timing["calibration_mode"] = str(mode)
            truth = calibration_meta.get("truth")
            if isinstance(truth, dict):
                if truth.get("source") is not None:
                    block_timing["calibration_truth_source"] = str(truth.get("source"))
                if truth.get("n_labeled") is not None:
                    block_timing["calibration_truth_n_labeled"] = int(truth.get("n_labeled"))
            cc = calibration_meta.get("call_correctness")
            if isinstance(cc, dict):
                if cc.get("decision_mode") is not None:
                    block_timing["calibration_decision_mode"] = str(cc.get("decision_mode"))
                thr = cc.get("threshold")
                if thr is not None:
                    block_timing["call_correctness_threshold"] = float(thr)
                for key in (
                    "n_snps_calibration_used",
                    "n_snps_fallback_to_stitch",
                ):
                    if cc.get(key) is not None:
                        block_timing[key] = int(cc.get(key))
                decision_counts = cc.get("decision_source_counts")
                if isinstance(decision_counts, dict):
                    block_timing["calibration_decision_source_counts"] = {
                        str(k): int(v) for k, v in decision_counts.items()
                    }
                fallback_counts = cc.get("fallback_reason_counts")
                if isinstance(fallback_counts, dict):
                    block_timing["calibration_fallback_reason_counts"] = {
                        str(k): int(v) for k, v in fallback_counts.items()
                    }
        if dask_extra:
            block_timing.update(dask_extra)
        if pedigree_result is not None:
            block_timing["pedigree_mode"] = str(pedigree_result.mode)
            summary = pedigree_result.summary
            if summary:
                block_timing["pedigree_edges"] = int(summary.get("n_edges", 0))
                block_timing["pedigree_components"] = int(summary.get("n_components", 0))
                if "messages" in summary:
                    block_timing["pedigree_messages"] = int(summary.get("messages", 0))
        if self.config.profile_memory:
            block_timing.update(
                {
                    "rss_mb_block_start": float(rss_block_start or 0.0),
                    "rss_mb_after_reads": float(rss_after_reads or 0.0),
                    "rss_mb_after_hmm": float(rss_after_hmm or 0.0),
                    "rss_mb_after_calibration": float(rss_after_calibration or 0.0),
                    "rss_mb_after_write": float(rss_after_write or 0.0),
                }
            )
        return block_timing

    def _run_blocks_dask(
        self,
        samples: pd.DataFrame,
        positions_df: pd.DataFrame,
        founder_panel: FounderPanel,
        pedigree,
    ) -> None:
        try:
            from dask import compute as dask_compute
            from dask import delayed
        except ImportError as exc:  # pragma: no cover - depends on optional runtime packaging.
            raise ImportError(
                "The Dask executor requires dask. Install stitchv2 with the Dask dependencies."
            ) from exc

        scheduler_kind = str(self.config.dask_scheduler)
        if scheduler_kind not in {"local", "threads", "synchronous", "jobqueue"}:
            raise ValueError("--dask-scheduler must be one of: local, threads, synchronous, jobqueue.")
        if scheduler_kind in {"local", "jobqueue"}:
            try:
                from dask.distributed import Client, LocalCluster, get_task_stream, performance_report
            except ImportError as exc:  # pragma: no cover - depends on optional runtime packaging.
                raise ImportError(
                    "The Dask distributed scheduler requires dask.distributed. Use --dask-scheduler threads to avoid distributed."
                ) from exc
            if scheduler_kind == "local":
                _validate_local_dask_socket_support()
        else:
            Client = LocalCluster = get_task_stream = performance_report = None

        timings: list[dict[str, float | int | str | bool]] = []
        generations = samples["generation"].to_numpy(dtype=np.float32, copy=False)
        sample_ids = samples["sample_id"].astype(str).to_numpy()
        sample_ploidy = (
            self._sample_ploidy.copy()
            if self._sample_ploidy is not None
            else np.full(len(samples), int(self.config.ploidy), dtype=np.int16)
        )
        max_ploidy_global = int(np.max(sample_ploidy)) if sample_ploidy.size else int(self.config.ploidy)
        requested_genotype_outputs = self.io_config.write_genotype_posteriors or self.io_config.write_genotype_calls
        need_genotype_posterior = requested_genotype_outputs or self.config.pedigree_mode in {
            "kinship",
            "transmission",
        }
        return_full_transition = False
        return_haplotype = self.io_config.write_haplotype_probabilities
        has_bam_mask = samples["bam_path"].fillna("").astype(str).str.len().to_numpy(dtype=np.int32) > 0
        read_samples = samples.loc[has_bam_mask].reset_index(drop=True)
        memory_plan = self._build_runtime_memory_plan(
            n_samples=int(samples.shape[0]),
            n_read_samples=int(read_samples.shape[0]),
            n_positions=int(positions_df.shape[0]),
            founder_panel=founder_panel,
            sample_ploidy=sample_ploidy,
        )
        self._write_runtime_memory_plan(memory_plan)
        configured_sample_batch = int(self.config.dask_sample_batch_size) or int(self.config.jax_sample_batch_size)
        dask_artifact_dir = self.output_dir / "_dask_artifacts"
        dask_artifact_dir.mkdir(parents=True, exist_ok=True)
        configured_block_size_for_dask = int(self.config.block_size)
        if str(self.config.snp_block_mode) == "exact_streaming":
            configured_block_size_for_dask = int(positions_df.shape[0])
        elif configured_block_size_for_dask <= 0:
            configured_block_size_for_dask = int(memory_plan.effective_block_size)
        min_dask_block_size = (
            int(positions_df.shape[0])
            if str(self.config.snp_block_mode) == "exact_streaming"
            else self.config.dask_min_block_size
        )
        chunk_plan = plan_dask_chunks(
            n_samples=len(samples),
            n_variants=len(positions_df),
            n_founders=founder_panel.n_founders,
            max_ploidy=max(max_ploidy_global, 1),
            configured_block_size=configured_block_size_for_dask,
            configured_sample_batch_size=configured_sample_batch,
            target_task_memory_mb=self.config.dask_target_task_memory_mb,
            min_block_size=min_dask_block_size,
            min_sample_batch_size=self.config.dask_min_sample_batch_size,
            return_genotype_posterior=need_genotype_posterior,
            return_haplotype_posterior=return_haplotype,
            return_full_transition=return_full_transition,
            force_generic_ploidy_hmm=self.config.force_generic_ploidy_hmm,
        )
        effective_block_size = int(chunk_plan.block_size)
        sample_batch_size = int(chunk_plan.sample_batch_size)
        if str(self.config.snp_block_mode) == "exact_streaming" and effective_block_size < int(positions_df.shape[0]):
            raise NotImplementedError(
                "The Dask executor currently schedules independent SNP-block tasks. "
                "Use --snp-block-mode independent_approx/density_balanced_overlap for approximate Dask runs, "
                "or run serial exact_streaming with a single planned block until the boundary streaming "
                "Dask path is available."
            )

        accelerator_count = _jax_accelerator_count() if self.config.hmm_backend in {"auto", "jax"} else 0
        n_workers = int(self.config.dask_n_workers)
        if n_workers <= 0:
            if accelerator_count > 0:
                n_workers = accelerator_count
            else:
                n_workers = max(1, min(os.cpu_count() or 1, 4))
        dashboard_address = self.config.dask_dashboard_address
        memory_limit = self.config.dask_memory_limit or "auto"
        client = None
        cluster = None
        dashboard_link = None
        if scheduler_kind == "local":
            cluster_kwargs = {
                "n_workers": n_workers,
                "threads_per_worker": max(int(self.config.dask_threads_per_worker), 1),
                "processes": bool(self.config.dask_processes),
                "memory_limit": memory_limit,
                "dashboard_address": dashboard_address,
            }
            if bool(self.config.dask_processes) and isinstance(dashboard_address, str) and dashboard_address.startswith("127.0.0.1:"):
                cluster_kwargs["host"] = "127.0.0.1"
            if not bool(self.config.dask_processes):
                cluster_kwargs["protocol"] = "inproc://"
            cluster = LocalCluster(**cluster_kwargs)
            client = Client(cluster)
            dashboard_link = str(client.dashboard_link) if client.dashboard_link else None
            if dashboard_link:
                print(f"Dask dashboard: {dashboard_link}", flush=True)
        elif scheduler_kind == "jobqueue":
            try:
                import dask_jobqueue
            except ImportError as exc:  # pragma: no cover - optional HPC dependency.
                raise ImportError(
                    "The jobqueue scheduler requires dask-jobqueue. Install stitchv2 with the jobqueue extras."
                ) from exc
            cluster_cls = getattr(dask_jobqueue, str(self.config.dask_jobqueue_class), None)
            if cluster_cls is None:
                raise ValueError(f"Unknown dask-jobqueue class: {self.config.dask_jobqueue_class}")
            job_kwargs: dict[str, object] = {
                "cores": max(int(self.config.dask_jobqueue_cores), 1),
                "processes": 1,
            }
            if self.config.dask_jobqueue_memory:
                job_kwargs["memory"] = str(self.config.dask_jobqueue_memory)
            elif memory_limit and memory_limit != "auto":
                job_kwargs["memory"] = str(memory_limit)
            if self.config.dask_jobqueue_queue:
                job_kwargs["queue"] = str(self.config.dask_jobqueue_queue)
            if self.config.dask_jobqueue_account:
                job_kwargs["account"] = str(self.config.dask_jobqueue_account)
            if self.config.dask_jobqueue_walltime:
                job_kwargs["walltime"] = str(self.config.dask_jobqueue_walltime)
            cluster = cluster_cls(**job_kwargs)
            cluster.scale(jobs=max(int(n_workers), 1))
            client = Client(cluster)
            dashboard_link = str(client.dashboard_link) if client.dashboard_link else None
            if dashboard_link:
                print(f"Dask jobqueue dashboard: {dashboard_link}", flush=True)
        in_memory_task_artifacts = client is None and not bool(self.config.dask_processes)

        effective_io_window_size = self._resolve_effective_io_window_size(
            n_read_samples=int(read_samples.shape[0]),
            n_positions=int(positions_df.shape[0]),
            effective_block_size=effective_block_size,
        )
        self.read_extractor.set_position_table(positions_df)
        self.read_extractor.set_io_window_size(effective_io_window_size)
        if int(read_samples.shape[0]) > 0:
            self.read_extractor.open(read_samples)

        performance_report_path = Path(self.config.dask_performance_report) if self.config.dask_performance_report else None
        task_stream_path = Path(self.config.dask_task_stream) if self.config.dask_task_stream else None
        if performance_report_path is not None and not performance_report_path.is_absolute():
            performance_report_path = self.output_dir / performance_report_path
        if task_stream_path is not None and not task_stream_path.is_absolute():
            task_stream_path = self.output_dir / task_stream_path
        task_stream = None
        total_tasks = 0
        task_diagnostics: list[dict[str, object]] = []
        scheduler_health: dict[str, object] = {}
        mutable_founder_forced_batches = False
        try:
            with ExitStack() as stack:
                if performance_report_path is not None:
                    if scheduler_kind in {"local", "jobqueue"}:
                        performance_report_path.parent.mkdir(parents=True, exist_ok=True)
                        stack.enter_context(performance_report(filename=str(performance_report_path)))
                    else:
                        print(
                            "Dask performance reports require a distributed scheduler; "
                            f"skipping {performance_report_path}.",
                            flush=True,
                        )
                if task_stream_path is not None:
                    if scheduler_kind in {"local", "jobqueue"}:
                        task_stream = stack.enter_context(get_task_stream(client=client, plot=False))
                    else:
                        print(
                            "Dask task stream capture requires a distributed scheduler; "
                            f"skipping {task_stream_path}.",
                            flush=True,
                        )

                for block in iter_position_blocks(positions_df, effective_block_size):
                    block_t0 = time.perf_counter()
                    block_start_offset = int(block.row_start)
                    rss_block_start = _current_rss_mb() if self.config.profile_memory else None
                    read_evidence, compact_cache_stats = self._extract_or_load_read_evidence(
                        read_samples=read_samples,
                        block=block,
                    )
                    full_evidence = self._build_full_sample_evidence(
                        block_id=block.block_id,
                        chromosome=self.config.chromosome,
                        positions=block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False),
                        ref=block.dataframe["REF"].astype(str).to_numpy(),
                        alt=block.dataframe["ALT"].astype(str).to_numpy(),
                        sample_ids=sample_ids,
                        has_bam_mask=has_bam_mask,
                        read_evidence=read_evidence,
                    )
                    evidence = full_evidence
                    calibration_truth_evidence: ReadEvidenceBlock | None = None
                    calibration_split_meta: dict[str, object] = {"status": "disabled"}
                    try:
                        evidence, calibration_truth_evidence, calibration_split_meta = self._maybe_split_read_evidence_for_calibration(
                            evidence=full_evidence,
                            block_id=int(block.block_id),
                        )
                        support_mask = evidence.depth > 0
                        if self._microarray_dosage is not None and bool(self.config.microarray_use_as_hmm_evidence):
                            start = block_start_offset
                            stop = min(start + int(evidence.ref_count.shape[1]), self._microarray_dosage.shape[1])
                            self._inject_microarray_hard_calls(
                                evidence,
                                self._microarray_dosage[:, start:stop],
                                hard_call_weight=int(self.config.microarray_hard_call_weight),
                            )
                        t_after_reads = time.perf_counter()
                        rss_after_reads = _current_rss_mb() if self.config.profile_memory else None
                        if self.io_config.write_pileup:
                            write_parquet(
                                evidence.to_arrow(),
                                self.output_dir / "pileup" / f"block={block.block_id:06d}.parquet",
                                self.config.compression,
                                self.config.compression_level,
                            )

                        block_founders = founder_panel.slice(
                            block_start_offset,
                            min(block_start_offset + int(evidence.ref_count.shape[1]), founder_panel.n_positions),
                        )
                        positive_ploidies = np.unique(sample_ploidy[sample_ploidy > 0])
                        max_ploidy = int(np.max(sample_ploidy)) if sample_ploidy.size else int(self.config.ploidy)
                        n_samples, n_positions = evidence.ref_count.shape
                        block_chunk_plan = plan_dask_chunks(
                            n_samples=n_samples,
                            n_variants=n_positions,
                            n_founders=block_founders.n_founders,
                            max_ploidy=max(max_ploidy, 1),
                            configured_block_size=n_positions,
                            configured_sample_batch_size=sample_batch_size,
                            target_task_memory_mb=self.config.dask_target_task_memory_mb,
                            min_block_size=min(int(self.config.dask_min_block_size), max(n_positions, 1)),
                            min_sample_batch_size=self.config.dask_min_sample_batch_size,
                            return_genotype_posterior=need_genotype_posterior,
                            return_haplotype_posterior=return_haplotype,
                            return_full_transition=return_full_transition,
                            force_generic_ploidy_hmm=self.config.force_generic_ploidy_hmm,
                            observed_fragments=int(evidence.fragment_center_idx.shape[0]),
                            observed_fragment_observations=int(evidence.fragment_obs_pos_idx.shape[0]),
                        )
                        block_sample_batch_size = int(block_chunk_plan.sample_batch_size)
                        task_results: list[DaskHMMTaskResult] = []

                        if np.all(sample_ploidy <= 0):
                            artifacts = self._missing_artifacts(
                                n_samples=n_samples,
                                n_positions=n_positions,
                                generations=generations,
                                block_founders=block_founders,
                                return_haplotype=return_haplotype,
                                return_full_transition=return_full_transition,
                            )
                        else:
                            tasks = []
                            force_full_group = not bool(np.all(block_founders.immutable_mask))
                            mutable_founder_forced_batches = mutable_founder_forced_batches or force_full_group
                            hmm_config = self.config.hmm()
                            hmm_config_payload = client.scatter(hmm_config, broadcast=True) if client is not None else hmm_config
                            founder_payload = client.scatter(block_founders, broadcast=True) if client is not None else block_founders
                            copy_task_arrays = bool(self.config.dask_processes)

                            def task_indexer(idx: np.ndarray) -> slice | np.ndarray:
                                if idx.size == 0:
                                    return idx
                                start_i = int(idx[0])
                                stop_i = int(idx[-1]) + 1
                                if stop_i - start_i == int(idx.size) and np.array_equal(idx, np.arange(start_i, stop_i)):
                                    return slice(start_i, stop_i)
                                return idx

                            def task_array(array: np.ndarray, idx: np.ndarray) -> np.ndarray:
                                sliced = array[task_indexer(idx)]
                                return sliced.copy() if copy_task_arrays else sliced

                            for ploidy_i in positive_ploidies.tolist():
                                idx_all = np.flatnonzero(sample_ploidy == int(ploidy_i))
                                if idx_all.size == 0:
                                    continue
                                if force_full_group:
                                    batches = [idx_all]
                                else:
                                    batches = [
                                        idx_all[start : start + block_sample_batch_size]
                                        for start in range(0, idx_all.size, block_sample_batch_size)
                                    ]
                                for idx in batches:
                                    (
                                        fso,
                                        fci,
                                        foo,
                                        fop,
                                        foc,
                                        foq,
                                    ) = self._subset_fragments_by_sample_indices(
                                        idx,
                                        evidence.fragment_sample_offsets,
                                        evidence.fragment_center_idx,
                                        evidence.fragment_obs_offsets,
                                        evidence.fragment_obs_pos_idx,
                                        evidence.fragment_obs_code,
                                        evidence.fragment_obs_qual,
                                    )
                                    tasks.append(
                                        delayed(run_hmm_leaf_task, pure=False)(
                                            block_id=block.block_id,
                                            sample_indices=idx.astype(np.int64, copy=True),
                                            ploidy=int(ploidy_i),
                                            artifact_dir=dask_artifact_dir,
                                            hmm_config=hmm_config_payload,
                                            founder_panel=founder_payload,
                                            ref_count=task_array(evidence.ref_count, idx),
                                            alt_count=task_array(evidence.alt_count, idx),
                                            generations=task_array(generations, idx),
                                            other_count=task_array(evidence.other_count, idx),
                                            ref_weight=task_array(evidence.ref_weight, idx),
                                            alt_weight=task_array(evidence.alt_weight, idx),
                                            other_weight=task_array(evidence.other_weight, idx),
                                            return_full_transition=return_full_transition and len(batches) == 1 and idx.size == n_samples,
                                            return_haplotype_posterior=return_haplotype,
                                            return_genotype_posterior=need_genotype_posterior,
                                            fragment_sample_offsets=fso,
                                            fragment_center_idx=fci,
                                            fragment_obs_offsets=foo,
                                            fragment_obs_pos_idx=fop,
                                            fragment_obs_code=foc,
                                            fragment_obs_qual=foq,
                                            return_artifacts=in_memory_task_artifacts,
                                        )
                                    )

                            if client is not None:
                                futures = client.compute(tasks)
                                task_results = list(client.gather(futures))
                            elif scheduler_kind == "threads":
                                task_results = list(
                                    dask_compute(
                                        *tasks,
                                        scheduler="threads",
                                        num_workers=max(1, int(n_workers) * max(int(self.config.dask_threads_per_worker), 1)),
                                    )
                                )
                            else:
                                task_results = list(dask_compute(*tasks, scheduler="synchronous"))
                            total_tasks += len(task_results)
                            task_diagnostics.extend(result.diagnostics() for result in task_results)
                            if (
                                len(task_results) == 1
                                and np.array_equal(task_results[0].sample_indices, np.arange(n_samples, dtype=np.int64))
                                and not np.any(sample_ploidy == 0)
                            ):
                                artifacts = self._load_dask_hmm_task_artifacts(task_results[0])
                            else:
                                artifacts = self._merge_dask_hmm_task_results(
                                    results=task_results,
                                    n_samples=n_samples,
                                    n_positions=n_positions,
                                    sample_ploidy=sample_ploidy,
                                    max_ploidy=max_ploidy,
                                    block_founders=block_founders,
                                    return_haplotype=return_haplotype,
                                    return_genotype_posterior=need_genotype_posterior,
                                )

                        t_after_hmm = time.perf_counter()
                        task_seconds = float(sum(result.seconds_hmm for result in task_results))
                        task_max_rss = float(max((result.rss_mb for result in task_results), default=0.0))
                        block_timing = self._finalize_and_write_block(
                            samples=samples,
                            sample_ids=sample_ids,
                            sample_ploidy=sample_ploidy,
                            generations=generations,
                            pedigree=pedigree,
                            block=block,
                            block_start_offset=block_start_offset,
                            evidence=evidence,
                            calibration_truth_evidence=calibration_truth_evidence,
                            calibration_split_meta=calibration_split_meta,
                            support_mask=support_mask,
                            artifacts=artifacts,
                            need_genotype_posterior=need_genotype_posterior,
                            requested_genotype_outputs=requested_genotype_outputs,
                            max_ploidy=max_ploidy,
                            block_t0=block_t0,
                            t_after_reads=t_after_reads,
                            t_after_hmm=t_after_hmm,
                            rss_block_start=rss_block_start,
                            rss_after_reads=rss_after_reads,
                            dask_extra={
                                "executor": "dask",
                                "dask_tasks": int(len(task_results)),
                                "dask_task_seconds_sum": task_seconds,
                                "dask_task_max_rss_mb": task_max_rss,
                                "dask_block_size": int(effective_block_size),
                                "dask_sample_batch_size": int(block_sample_batch_size),
                                "dask_estimated_task_memory_mb": float(block_chunk_plan.estimated_task_memory_mb),
                                "dask_chunk_plan_reason": str(block_chunk_plan.reason),
                                "dask_observed_fragments": int(evidence.fragment_center_idx.shape[0]),
                                "dask_observed_fragment_observations": int(evidence.fragment_obs_pos_idx.shape[0]),
                                "dask_in_memory_task_artifacts": bool(in_memory_task_artifacts),
                                **compact_cache_stats,
                            },
                        )
                        timings.append(block_timing)
                    finally:
                        if calibration_truth_evidence is not None and calibration_truth_evidence is not evidence:
                            calibration_truth_evidence.release()
                        if evidence is not full_evidence:
                            evidence.release()
                        if read_evidence is not None:
                            read_evidence.release()
                        full_evidence.release()
                        if self.config.gc_collect_every_block:
                            gc.collect()
        finally:
            if int(read_samples.shape[0]) > 0:
                self.read_extractor.close()
            hold_seconds = max(float(self.config.dask_dashboard_hold_seconds), 0.0)
            if hold_seconds > 0.0 and dashboard_link:
                print(
                    f"Holding Dask dashboard at {dashboard_link} for {hold_seconds:.1f} seconds before shutdown.",
                    flush=True,
                )
                time.sleep(hold_seconds)
            if client is not None:
                try:
                    info = client.scheduler_info()
                    workers = info.get("workers", {}) if isinstance(info, dict) else {}
                    scheduler_health = {
                        "status": "ok",
                        "scheduler_id": info.get("id") if isinstance(info, dict) else None,
                        "n_workers_reported": int(len(workers)),
                        "worker_addresses": sorted(str(addr) for addr in workers.keys()),
                        "total_threads_reported": int(
                            sum(int(w.get("nthreads", 0)) for w in workers.values() if isinstance(w, dict))
                        ),
                        "total_memory_limit_reported": int(
                            sum(int(w.get("memory_limit", 0)) for w in workers.values() if isinstance(w, dict))
                        ),
                    }
                except Exception as exc:
                    scheduler_health = {"status": "unavailable", "error": str(exc)}
            if client is not None:
                client.close()
            if cluster is not None:
                cluster.close()

        if task_stream_path is not None and task_stream is not None:
            write_task_stream_artifact(task_stream_path, list(getattr(task_stream, "data", [])))

        timings_path = self.output_dir / "stage_timings.json"
        timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
        if self.config.profile_memory and timings:
            peak_rss = max(float(row.get("rss_mb_after_write", 0.0)) for row in timings)
            peak_hmm_rss = max(float(row.get("rss_mb_after_hmm", 0.0)) for row in timings)
            mem_summary = {
                "peak_rss_mb": peak_rss,
                "peak_hmm_rss_mb": peak_hmm_rss,
                "blocks_profiled": len(timings),
            }
            (self.output_dir / "memory_profile_summary.json").write_text(
                json.dumps(mem_summary, indent=2),
                encoding="utf-8",
            )
        dask_summary = {
            "executor": "dask",
            "scheduler": self.config.dask_scheduler,
            "dashboard_url": dashboard_link,
            "n_workers": int(n_workers),
            "jax_accelerator_count": int(accelerator_count),
            "threads_per_worker": int(self.config.dask_threads_per_worker),
            "processes": bool(self.config.dask_processes),
            "memory_limit": memory_limit,
            "chunk_plan": chunk_plan.to_dict(),
            "runtime_memory_plan": memory_plan.to_dict(),
            "effective_block_size": int(effective_block_size),
            "sample_batch_size": int(sample_batch_size),
            "snp_block_mode": str(self.config.snp_block_mode),
            "snp_block_mode_approximate": str(self.config.snp_block_mode) != "exact_streaming",
            "in_memory_task_artifacts": bool(in_memory_task_artifacts),
            "scattered_block_config_and_founders": bool(client is not None),
            "performance_report": (str(performance_report_path) if performance_report_path is not None else None),
            "task_stream": (str(task_stream_path) if task_stream_path is not None else None),
            "dashboard_hold_seconds": float(self.config.dask_dashboard_hold_seconds),
            "scheduler_health": scheduler_health,
            "jobqueue": {
                "class": str(self.config.dask_jobqueue_class),
                "queue": self.config.dask_jobqueue_queue,
                "account": self.config.dask_jobqueue_account,
                "cores": int(self.config.dask_jobqueue_cores),
                "memory": self.config.dask_jobqueue_memory,
                "walltime": self.config.dask_jobqueue_walltime,
            } if scheduler_kind == "jobqueue" else None,
            "total_hmm_tasks": int(total_tasks),
            "mutable_founder_forced_full_ploidy_group_batches": bool(mutable_founder_forced_batches),
            "task_diagnostics": task_diagnostics,
        }
        (self.output_dir / "dask_run_summary.json").write_text(
            json.dumps(dask_summary, indent=2),
            encoding="utf-8",
        )
        self._finalize_diagnostics_summary()

    def _full_transition_vectors_from_switch(
        self,
        switch_values: np.ndarray,
        *,
        n_founders: int,
        ploidy: int,
        offdiag_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        switch_values = np.asarray(switch_values, dtype=np.float32)
        k = int(n_founders)
        ploidy_i = max(int(ploidy), 1)
        if self.config.ploidy_mode == "pseudo_haploid" or ploidy_i == 1:
            out = np.empty((switch_values.shape[0], k * k), dtype=np.float32)
            for idx, switch_value in enumerate(switch_values.tolist()):
                mat = (
                    _transition_matrix_from_switch_and_offdiag(float(switch_value), offdiag_matrix)
                    if offdiag_matrix is not None
                    else _build_transition_matrix_from_switch(float(switch_value), k)
                )
                out[idx] = mat.reshape(-1)
            return out
        if ploidy_i == 2:
            out = np.empty((switch_values.shape[0], k * k * k * k), dtype=np.float32)
            for idx, switch_value in enumerate(switch_values.tolist()):
                hap = (
                    _transition_matrix_from_switch_and_offdiag(float(switch_value), offdiag_matrix)
                    if offdiag_matrix is not None
                    else _build_transition_matrix_from_switch(float(switch_value), k)
                )
                out[idx] = np.einsum("ab,cd->acbd", hap, hap).reshape(-1)
            return out
        coeff = _polyploid_transition_coefficients(k, ploidy_i).astype(np.float32, copy=False)
        n_states = int(coeff.shape[0])
        out = np.empty((switch_values.shape[0], n_states * n_states), dtype=np.float32)
        for idx, switch_value in enumerate(switch_values.tolist()):
            out[idx] = self.hmm._polyploid_transition_matrix(
                float(switch_value),
                coeff,
                k=k,
                ploidy=ploidy_i,
            ).reshape(-1)
        return out

    def _write_full_transition_outputs(
        self,
        *,
        sample_col: np.ndarray,
        position_col: np.ndarray,
        block_id: int,
        artifacts: HMMArtifacts,
    ) -> None:
        switch_flat = artifacts.switch_probability.reshape(-1).astype(np.float32, copy=False)
        if switch_flat.size == 0:
            return
        ploidy = 1 if self.config.ploidy_mode == "pseudo_haploid" else int(self.config.ploidy)
        n_founders = int(artifacts.founder_alt_prob.shape[0])
        probe = self._full_transition_vectors_from_switch(
            switch_flat[:1],
            n_founders=n_founders,
            ploidy=ploidy,
            offdiag_matrix=artifacts.transition_factor_offdiag,
        )
        vector_len = int(probe.shape[1])
        budget = max(_parse_memory_budget(self.config.max_mem), 1)
        target_raw_bytes = min(max(int(budget * 0.025), 64 * 1024**2), 256 * 1024**2)
        rows_per_chunk = max(1, int(target_raw_bytes // max(vector_len * 4, 1)))
        rows_per_chunk = min(rows_per_chunk, int(switch_flat.size))
        for chunk_id, start in enumerate(range(0, int(switch_flat.size), rows_per_chunk)):
            stop = min(start + rows_per_chunk, int(switch_flat.size))
            vectors = self._full_transition_vectors_from_switch(
                switch_flat[start:stop],
                n_founders=n_founders,
                ploidy=ploidy,
                offdiag_matrix=artifacts.transition_factor_offdiag,
            )
            values = pa.array(vectors.reshape(-1), type=pa.float32())
            vector = pa.FixedSizeListArray.from_arrays(values, vector_len)
            full_table = pa.table(
                {
                    "sample_id": sample_col[start:stop],
                    "chromosome": np.repeat(self.config.chromosome, stop - start),
                    "position": position_col[start:stop],
                    "transition_probability": vector,
                    "block_id": np.repeat(block_id, stop - start),
                    "transition_chunk": np.repeat(chunk_id, stop - start),
                }
            )
            suffix = (
                f"block={block_id:06d}.parquet"
                if start == 0 and stop == int(switch_flat.size)
                else f"block={block_id:06d}_chunk={chunk_id:06d}.parquet"
            )
            write_parquet(
                full_table,
                self.output_dir / "transitions_full" / suffix,
                self.config.compression,
                self.config.compression_level,
                row_group_size=rows_per_chunk,
                use_dictionary=["sample_id", "chromosome", "block_id", "transition_chunk"],
            )

    def _write_transition_factor_outputs(self, *, block_id: int, artifacts: HMMArtifacts) -> None:
        if (
            artifacts.transition_factor_source is None
            or artifacts.transition_factor_destination is None
            or artifacts.transition_factor_offdiag is None
        ):
            return
        source = np.asarray(artifacts.transition_factor_source, dtype=np.float32)
        destination = np.asarray(artifacts.transition_factor_destination, dtype=np.float32)
        offdiag = np.asarray(artifacts.transition_factor_offdiag, dtype=np.float32)
        k = int(offdiag.shape[0])
        rank = int(source.shape[1]) if source.ndim == 2 else 0
        if rank <= 0:
            return
        source_values = pa.array(source.reshape(-1), type=pa.float32())
        dest_values = pa.array(destination.reshape(-1), type=pa.float32())
        offdiag_values = pa.array(offdiag.reshape(-1), type=pa.float32())
        table = pa.table(
            {
                "founder_index": np.arange(k, dtype=np.int32),
                "source_factor": pa.FixedSizeListArray.from_arrays(source_values, rank),
                "destination_factor": pa.FixedSizeListArray.from_arrays(dest_values, rank),
                "offdiag_distribution": pa.FixedSizeListArray.from_arrays(offdiag_values, k),
                "rank": np.repeat(rank, k).astype(np.int16),
                "block_id": np.repeat(block_id, k).astype(np.int32),
            }
        )
        write_parquet(
            table,
            self.output_dir / "transition_factors" / f"block={block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
            use_dictionary=["block_id"],
        )

    def _write_transition_summary(self, *, block, artifacts: HMMArtifacts) -> None:
        switch = np.asarray(artifacts.switch_probability, dtype=np.float32)
        if switch.ndim != 2 or switch.shape[1] <= 1:
            return
        stay = np.asarray(artifacts.stay_probability, dtype=np.float32)
        offdiag = np.asarray(artifacts.offdiag_probability, dtype=np.float32)
        positions = block.dataframe["POS"].to_numpy(dtype=np.int64, copy=False)
        if positions.shape[0] <= 1:
            return
        n_samples = int(switch.shape[0])
        n_intervals = int(min(switch.shape[1], positions.shape[0]) - 1)
        interval_switch = switch[:, 1 : n_intervals + 1]
        interval_stay = stay[:, 1 : n_intervals + 1]
        interval_offdiag = offdiag[:, 1 : n_intervals + 1]
        k = int(artifacts.founder_alt_prob.shape[0])

        if self._sample_ploidy is not None and self._sample_ploidy.shape[0] == n_samples:
            ploidy = np.clip(self._sample_ploidy.astype(np.float32, copy=False), 0.0, None)[:, None]
        else:
            ploidy = np.full((n_samples, 1), float(max(int(self.config.ploidy), 1)), dtype=np.float32)
        expected_copy_switches = interval_switch * ploidy
        expected_any_copy_switch = 1.0 - np.power(np.clip(1.0 - interval_switch, 0.0, 1.0), ploidy)
        hap_entropy = -(
            interval_stay * np.log(np.clip(interval_stay, 1e-12, 1.0))
            + float(max(k - 1, 0)) * interval_offdiag * np.log(np.clip(interval_offdiag, 1e-12, 1.0))
        )
        ploidy_scaled_entropy = hap_entropy * ploidy

        rows: dict[str, object] = {
            "chromosome": np.repeat(str(self.config.chromosome), n_intervals),
            "block_id": np.repeat(int(block.block_id), n_intervals).astype(np.int32),
            "interval_index": np.arange(n_intervals, dtype=np.int32),
            "from_position": positions[:n_intervals].astype(np.int64, copy=False),
            "to_position": positions[1 : n_intervals + 1].astype(np.int64, copy=False),
            "physical_distance_bp": np.diff(positions[: n_intervals + 1]).astype(np.int64, copy=False),
            "n_samples": np.repeat(n_samples, n_intervals).astype(np.int32),
            "n_founders": np.repeat(k, n_intervals).astype(np.int16),
            "transition_model": np.repeat(str(self.config.transition_model), n_intervals),
            "transition_output": np.repeat(str(self.config.transition_output), n_intervals),
            "recombination_rate": np.asarray(artifacts.recombination_rate[1 : n_intervals + 1], dtype=np.float32),
            "switch_mean": np.nanmean(interval_switch, axis=0).astype(np.float32),
            "switch_median": np.nanmedian(interval_switch, axis=0).astype(np.float32),
            "switch_min": np.nanmin(interval_switch, axis=0).astype(np.float32),
            "switch_max": np.nanmax(interval_switch, axis=0).astype(np.float32),
            "switch_p95": np.nanquantile(interval_switch, 0.95, axis=0).astype(np.float32),
            "stay_mean": np.nanmean(interval_stay, axis=0).astype(np.float32),
            "offdiag_mean": np.nanmean(interval_offdiag, axis=0).astype(np.float32),
            "expected_copy_switches_mean": np.nanmean(expected_copy_switches, axis=0).astype(np.float32),
            "expected_any_copy_switch_probability_mean": np.nanmean(expected_any_copy_switch, axis=0).astype(np.float32),
            "hap_transition_entropy_mean": np.nanmean(hap_entropy, axis=0).astype(np.float32),
            "ploidy_scaled_transition_entropy_mean": np.nanmean(ploidy_scaled_entropy, axis=0).astype(np.float32),
        }

        genetic_col = None
        for candidate in ("GENETIC_CM", "genetic_cm", "CM", "cM"):
            if candidate in block.dataframe.columns:
                genetic_col = candidate
                break
        if genetic_col is not None:
            genetic = pd.to_numeric(block.dataframe[genetic_col], errors="coerce").to_numpy(dtype=np.float32)
            rows["genetic_distance_cM"] = np.diff(genetic[: n_intervals + 1]).astype(np.float32, copy=False)

        if artifacts.transition_factor_offdiag is not None:
            factor_offdiag = artifacts.transition_factor_offdiag.astype(np.float32, copy=False)
            row_entropy = -np.sum(factor_offdiag * np.log(np.clip(factor_offdiag, 1e-12, 1.0)), axis=1)
            rows["factorized_offdiag_entropy_mean"] = np.repeat(float(np.mean(row_entropy)), n_intervals).astype(np.float32)
            rows["factorized_offdiag_min"] = np.repeat(float(np.min(factor_offdiag)), n_intervals).astype(np.float32)
            rows["factorized_offdiag_max"] = np.repeat(float(np.max(factor_offdiag)), n_intervals).astype(np.float32)

        float_cols = [
            col
            for col in (
                "recombination_rate",
                "switch_mean",
                "switch_median",
                "switch_min",
                "switch_max",
                "switch_p95",
                "stay_mean",
                "offdiag_mean",
                "expected_copy_switches_mean",
                "expected_any_copy_switch_probability_mean",
                "hap_transition_entropy_mean",
                "ploidy_scaled_transition_entropy_mean",
                "genetic_distance_cM",
                "factorized_offdiag_entropy_mean",
                "factorized_offdiag_min",
                "factorized_offdiag_max",
            )
            if col in rows
        ]
        out_path = self.output_dir / "transition_summary" / f"block={int(block.block_id):06d}.parquet"
        write_parquet(
            pa.table(rows),
            out_path,
            self.config.compression,
            self.config.compression_level,
            use_dictionary=["chromosome", "block_id", "transition_model", "transition_output"],
            use_byte_stream_split=float_cols,
        )
        if self._store_xi == "per-snp":
            self._write_per_snp_xi_manifest(
                {
                    "block_id": int(block.block_id),
                    "materialized": True,
                    "path": str(out_path),
                    "n_intervals": int(n_intervals),
                    "n_samples": int(n_samples),
                    "n_founders": int(k),
                    "file_bytes": int(out_path.stat().st_size),
                    "file_mb": float(out_path.stat().st_size / 1_000_000.0),
                    "columns": list(rows.keys()),
                }
            )

    def _write_hmm_outputs(
        self,
        sample_ids: np.ndarray,
        block,
        dosage: np.ndarray,
        artifacts,
        *,
        support_mask: np.ndarray | None = None,
        genotype_posterior: np.ndarray | None = None,
        genotype_call: np.ndarray | None = None,
    ) -> None:
        positions = block.dataframe["POS"].to_numpy(dtype=np.int64)
        n_samples, n_positions = dosage.shape
        n_rows = n_samples * n_positions
        sample_col = np.repeat(sample_ids, n_positions)
        position_col = np.tile(positions, n_samples)
        block_col = np.repeat(block.block_id, n_rows)
        dosage_table = pa.table(
            {
                "sample_id": sample_col,
                "chromosome": np.repeat(self.config.chromosome, n_rows),
                "position": position_col,
                "dosage": dosage.reshape(-1).astype(np.float32),
                "block_id": block_col,
            }
        )
        write_parquet(
            dosage_table,
            self.output_dir / "dosage" / f"block={block.block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
            row_group_size=262_144,
            use_dictionary=["sample_id", "chromosome", "block_id"],
            use_byte_stream_split=["dosage"],
        )

        if self.io_config.write_support_mask:
            if support_mask is None:
                raise RuntimeError("Supporting-read mask missing while write_support_mask=True")
            support_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "has_supporting_read": support_mask.reshape(-1).astype(np.bool_, copy=False),
                    "block_id": block_col,
                }
            )
            write_parquet(
                support_table,
                self.output_dir / "support_mask" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
                use_dictionary=["sample_id", "chromosome", "block_id"],
            )

        rate_table = pa.table(
            {
                "chromosome": np.repeat(self.config.chromosome, positions.shape[0]),
                "position": positions,
                "recombination_rate": artifacts.recombination_rate.astype(np.float32),
                "block_id": np.repeat(block.block_id, positions.shape[0]),
            }
        )
        write_parquet(
            rate_table,
            self.output_dir / "recombination" / f"block={block.block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
            use_dictionary=["chromosome", "block_id"],
            use_byte_stream_split=["recombination_rate"],
        )

        if self.io_config.write_transitions:
            transition_rows = artifacts.switch_probability.size
            transitions = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, transition_rows),
                    "position": position_col,
                    "switch_probability": artifacts.switch_probability.reshape(-1).astype(np.float32),
                    "stay_probability": artifacts.stay_probability.reshape(-1).astype(np.float32),
                    "offdiag_probability": artifacts.offdiag_probability.reshape(-1).astype(np.float32),
                    "block_id": np.repeat(block.block_id, transition_rows),
                }
            )
            write_parquet(
                transitions,
                self.output_dir / "transitions" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                use_dictionary=["sample_id", "chromosome", "block_id"],
                use_byte_stream_split=["switch_probability", "stay_probability", "offdiag_probability"],
            )

        if self.io_config.transition_output in {"factorized", "full"}:
            self._write_transition_factor_outputs(block_id=int(block.block_id), artifacts=artifacts)

        if self.io_config.transition_output == "full":
            self._write_full_transition_outputs(
                sample_col=sample_col,
                position_col=position_col,
                block_id=int(block.block_id),
                artifacts=artifacts,
            )

        if self._store_xi == "per-snp" and self.io_config.write_transition_summary:
            self._write_transition_summary(block=block, artifacts=artifacts)

        if self.io_config.write_haplotype_probabilities:
            k = int(artifacts.founder_alt_prob.shape[0])
            if artifacts.haplotype_posterior is None:
                raise RuntimeError("HMM haplotype posterior missing while write_haplotype_probabilities=True")
            hap_probability = artifacts.haplotype_posterior.astype(np.float32, copy=False)
            if self._sample_ploidy is not None and self._sample_ploidy.shape[0] == hap_probability.shape[0]:
                ploidy_scale = self._sample_ploidy.astype(np.float32, copy=False)[:, None, None]
                hap_dosage = (hap_probability * ploidy_scale).astype(np.float32, copy=False)
            elif self.config.ploidy_mode == "diploid":
                hap_dosage = (2.0 * hap_probability).astype(np.float32, copy=False)
            else:
                hap_dosage = hap_probability
            hap_dosage = hap_dosage.astype(np.float32, copy=False)
            hap_probability = hap_probability.astype(np.float32, copy=False)
            hap_dosage_values = pa.array(hap_dosage.reshape(-1), type=pa.float32())
            hap_probability_values = pa.array(hap_probability.reshape(-1), type=pa.float32())
            hap_dosage_vector = pa.FixedSizeListArray.from_arrays(hap_dosage_values, k)
            hap_probability_vector = pa.FixedSizeListArray.from_arrays(hap_probability_values, k)
            hap_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "hap_dosage": hap_dosage_vector,
                    "hap_probability": hap_probability_vector,
                    "block_id": block_col,
                }
            )
            write_parquet(
                hap_table,
                self.output_dir / "haplotype_probabilities" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
                use_dictionary=["sample_id", "chromosome", "block_id"],
            )

        if self.io_config.write_genotype_posteriors:
            if genotype_posterior is None:
                raise RuntimeError("Genotype posterior missing while write_genotype_posteriors=True")
            gp_values = pa.array(
                genotype_posterior.astype(np.float32, copy=False).reshape(-1),
                type=pa.float32(),
            )
            gp_vector = pa.FixedSizeListArray.from_arrays(gp_values, int(genotype_posterior.shape[2]))
            gp_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "genotype_posterior": gp_vector,
                    "block_id": block_col,
                }
            )
            write_parquet(
                gp_table,
                self.output_dir / "genotype_posteriors" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
                use_dictionary=["sample_id", "chromosome", "block_id"],
            )

        if self.io_config.write_genotype_calls:
            if genotype_call is None:
                raise RuntimeError("Genotype call missing while write_genotype_calls=True")
            gt_table = pa.table(
                {
                    "sample_id": sample_col,
                    "chromosome": np.repeat(self.config.chromosome, n_rows),
                    "position": position_col,
                    "genotype_call": genotype_call.reshape(-1).astype(np.int8, copy=False),
                    "block_id": block_col,
                }
            )
            write_parquet(
                gt_table,
                self.output_dir / "genotype_calls" / f"block={block.block_id:06d}.parquet",
                self.config.compression,
                self.config.compression_level,
                row_group_size=262_144,
                use_dictionary=["sample_id", "chromosome", "genotype_call", "block_id"],
            )

        founder_idx, pos_idx = np.indices(artifacts.founder_alt_prob.shape)
        founder_table = pa.table(
            {
                "chromosome": np.repeat(self.config.chromosome, artifacts.founder_alt_prob.size),
                "position": positions[pos_idx.reshape(-1)],
                "founder": founder_idx.reshape(-1),
                "alt_prob": artifacts.founder_alt_prob.reshape(-1).astype(np.float32),
                "block_id": np.repeat(block.block_id, artifacts.founder_alt_prob.size),
            }
        )
        write_parquet(
            founder_table,
            self.output_dir / "founder_updates" / f"block={block.block_id:06d}.parquet",
            self.config.compression,
            self.config.compression_level,
            use_dictionary=["chromosome", "founder", "block_id"],
            use_byte_stream_split=["alt_prob"],
        )
