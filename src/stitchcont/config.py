from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(slots=True)
class IOConfig:
    positions_path: str | Path
    output_dir: str | Path
    compression: str = "zstd"
    compression_level: int = 6
    write_statistics: bool = True
    write_pileup: bool = False
    write_transitions: bool = True
    write_haplotype_probabilities: bool = False
    write_genotype_posteriors: bool = False
    write_genotype_calls: bool = False
    write_support_mask: bool = False
    write_transition_summary: bool = True
    transition_output: Literal["compact", "factorized", "full"] = "compact"
    output_store: Literal["parquet", "zarr"] = "parquet"
    zarr_chunk_samples: int = 256
    zarr_chunk_positions: int = 4096
    zarr_dosage_dtype: str = "float32"
    zarr_gp_dtype: str = "float32"
    zarr_probability_scale: int = 65535
    zarr_compressor: str = "none"
    zarr_compression_level: int = 1
    zarr_bitpack_support_mask: bool = False


@dataclass(slots=True)
class FounderConfig:
    source_format: Literal["vcf", "plink", "bam"] = "vcf"
    source_path: str | Path | None = None
    immutable: bool = True
    sample_id_column: str = "sample_id"
    dosage_field: str = "DS"
    genotype_field: str = "GT"


@dataclass(slots=True)
class HMMConfig:
    n_founders: int
    ploidy_mode: Literal["diploid", "pseudo_haploid"] = "diploid"
    ploidy: int = 2
    em_iterations: int = 20
    final_diploid_iterations: int = 2
    em_convergence_tol: float = 1e-4
    em_convergence_min_iterations: int = 2
    em_convergence_patience: int = 1
    em_founder_update_damping: float = 1.0
    adaptive_em: bool = True
    adaptive_em_restore_best_founders: bool = True
    em_multistarts: int = 1
    founder_update_hardening: bool = True
    min_emission_prob: float = 1e-5
    sequencing_error_rate: float = 0.01
    recombination_rate_cM_per_Mb: float = 1.0
    trainable_global_recombination_rate: bool = False
    allow_recombination_hotspot_window: int = 0
    transition_model: Literal["stitch_parity", "factorized", "low_rank_linear", "sparse_factorized"] = "stitch_parity"
    transition_factor_rank: int = 1
    transition_factor_regularization: float = 100.0
    transition_factor_max_deviation: float = 0.25
    transition_factor_train: bool = True
    transition_factor_init_max_cells: int = 50_000_000
    transition_sparse_top_k: int = 4
    known_founder_fast_path: bool = True
    output_minimal: bool = False
    founder_informative_compression: bool = False
    pseudo_haploid_cpu: bool = False
    use_unordered_diploid_states: bool = False
    sparse_emission_scan: bool = True
    generation_bin_size: float = 0.0
    exact_hmm_chunk_size: int = 20_000
    hmm_checkpoint_interval: int = 0
    block_size: int = 5000
    max_jit_positions: int = 10000
    pedigree_strength: float = 0.0
    backend: Literal["auto", "numpy", "numba", "cpp", "jax", "torch"] = "auto"
    jax_precompile: bool = True
    jax_aot_compile: bool = True
    jax_sample_batch_size: int = 0
    jax_bucket_batch_shapes: bool = True
    jax_count_emission_kernel: bool = True
    jax_fragment_emission_kernel: bool = True
    jax_persistent_cache_dir: str | Path | None = None
    force_generic_ploidy_hmm: bool = False
    use_quality_weights: bool = True
    use_fragment_likelihood: bool = True
    fragment_likelihood_mode: Literal["replace", "augment"] = "replace"
    fragment_coupling_model: Literal["stitch_parity"] = "stitch_parity"
    fragment_max_difference_between_reads: float = 100.0
    fragment_max_emission_matrix_difference: float = 1000.0
    fragment_rescale_read_likelihood: bool = True
    random_seed: int = 0
    founder_init_jitter: float = 0.0
    backend_autotune: bool = True
    autotune_samples: int = 0
    autotune_positions: int = 0


@dataclass(slots=True)
class PipelineConfig:
    chromosome: str
    positions_path: str | Path
    output_dir: str | Path
    n_founders: int
    chromosome_start: int | None = None
    chromosome_end: int | None = None
    block_size: int = 0
    max_mem: str | int | float | None = "90%"
    io_window_size: int = 0
    snp_block_mode: Literal["exact_streaming", "exact_chunked", "independent_approx", "density_balanced_overlap"] = "exact_streaming"
    memory_safety_fraction: float = 0.60
    # When False (default), the pipeline refuses to start a block whose estimated
    # peak HMM working set exceeds the planned memory budget, instead of silently
    # OOMing.  Set True to override the guard and accept the OOM risk.
    allow_memory_overcommit: bool = False
    scratch_dir: str | Path | None = None
    approx_overlap_fraction: float = 0.05
    approx_min_overlap_snps: int = 50
    store_xi: Literal["full", "per-snp", "False"] = "per-snp"
    write_xi: Literal["auto", "off", "force"] | None = None
    write_gamma: Literal["summary", "off", "full"] = "summary"
    write_hmm_boundaries: bool = True
    em_iterations: int = 20
    final_diploid_iterations: int = 2
    em_convergence_tol: float = 1e-4
    em_convergence_min_iterations: int = 2
    em_convergence_patience: int = 1
    em_founder_update_damping: float = 1.0
    adaptive_em: bool = True
    adaptive_em_restore_best_founders: bool = True
    em_multistarts: int = 1
    founder_update_hardening: bool = True
    ploidy_mode: Literal["diploid", "pseudo_haploid"] = "diploid"
    ploidy: int = 2
    ploidy_males: int | None = None
    ploidy_females: int | None = None
    compression: str = "zstd"
    compression_level: int = 6
    sequencing_error_rate: float = 0.01
    recombination_rate_cM_per_Mb: float = 1.0
    genetic_map_path: str | Path | None = None
    trainable_global_recombination_rate: bool = False
    allow_recombination_hotspot_window: int = 0
    pedigree_strength: float = 0.0
    pedigree_mode: Literal["off", "smooth", "kinship", "transmission"] = "smooth"
    pedigree_offspring_col: str = "sample_id"
    pedigree_parent1_col: str = "father_id"
    pedigree_parent2_col: str = "mother_id"
    pedigree_iterations: int = 4
    pedigree_kinship_threshold: float = 0.01
    hmm_backend: Literal["auto", "numpy", "numba", "cpp", "jax", "torch"] = "auto"
    jax_precompile: bool = True
    jax_aot_compile: bool = True
    jax_sample_batch_size: int = 0
    jax_bucket_batch_shapes: bool = True
    jax_count_emission_kernel: bool = True
    jax_fragment_emission_kernel: bool = True
    jax_persistent_cache_dir: str | Path | None = None
    force_generic_ploidy_hmm: bool = False
    use_quality_weights: bool = True
    use_fragment_likelihood: bool = True
    fragment_likelihood_mode: Literal["replace", "augment"] = "replace"
    hmm_backend_autotune: bool = True
    hmm_autotune_samples: int = 0
    hmm_autotune_positions: int = 0
    read_mode: Literal["read_stream", "pileup"] = "read_stream"
    read_stream_backend: Literal[
        "auto",
        "python",
        "htslib",
        "snp_only_bamreader",
        "stitch_style_bamreader",
        "variant_aware_bamreader",
    ] = "auto"
    min_base_quality: int = 13
    min_mapping_quality: int = 20
    max_insert_size: int = 0
    max_indel_len: int = 50
    cap_base_quality_by_mapping_quality: bool = False
    ref_alt_only: bool = False
    merge_fragments_by_query: bool = True
    merge_unpaired_fragments_by_query: bool = False
    use_bx_tag: bool = False
    bx_tag: str = "BX"
    bx_tag_upper_limit: int = 50000
    downsample_to_coverage: int = 0
    downsample_fraction: float = 1.0
    read_batch_size: int = 1024
    io_workers: int = 1
    htslib_threads_per_file: int = 1
    memory_map_read_matrices: bool = False
    memory_map_dir: str | Path | None = None
    compact_evidence_cache_dir: str | Path | None = None
    compact_evidence_cache_mode: Literal["off", "read", "write", "readwrite"] = "off"
    compact_evidence_cache_format: Literal["parquet_zarr"] = "parquet_zarr"
    compact_evidence_cache_sample_batch_size: int = 256
    compact_evidence_materialize_dense_counts: bool = True
    compact_evidence_cache_include_dense_counts: bool = False
    gc_collect_every_block: bool = False
    profile_memory: bool = True
    write_pileup: bool = False
    write_transitions: bool = True
    write_haplotype_probabilities: bool = False
    write_genotype_posteriors: bool = False
    write_genotype_calls: bool = False
    write_support_mask: bool = False
    write_transition_summary: bool = True
    calibrate_genotype_posteriors: bool = True
    calibration_mode: Literal["standard_callability", "fixed", "masked_cv"] = "standard_callability"
    genotype_posterior_temperature: float = 0.35
    genotype_posterior_blend: float = 0.35
    genotype_call_mode: Literal["argmax", "stitch_no_call", "quality_gated"] = "quality_gated"
    genotype_call_min_confidence: float = 0.0
    genotype_call_min_margin: float = 0.0
    genotype_call_stitch_threshold: float = 0.9
    genotype_call_correctness_threshold: float = 0.0
    use_lightgbm_calibrator: bool = False
    calibration_context_window: int = 25
    calibration_block_snps: int = 64
    calibration_truth_source: Literal["read_evidence", "microarray", "auto", "none"] = "read_evidence"
    calibration_read_truth_min_depth: int = 3
    calibration_read_truth_min_hom_depth: int = 2
    calibration_read_truth_min_het_depth: int = 4
    calibration_read_truth_min_het_allele_depth: int = 1
    calibration_read_truth_hom_major_fraction: float = 0.95
    calibration_read_truth_het_balance_min: float = 0.25
    calibration_read_truth_het_balance_max: float = 0.75
    calibration_read_truth_max_other_fraction: float = 0.05
    calibration_read_truth_holdout_fraction: float = 0.30
    calibration_use_optuna: bool = False
    calibration_optuna_trials: int = 20
    calibration_max_train_rows: int = 750_000
    calibration_callability_model: Literal["sklearn_logistic", "lightgbm"] = "lightgbm"
    calibration_callability_min_train_rows: int = 64
    calibration_callability_min_call_rate: float = 0.80
    calibration_callability_call_rate_weight: float = 0.03
    calibration_callability_decision_mode: Literal["per_snp_hierarchical", "global"] = "per_snp_hierarchical"
    calibration_callability_bound_to_stitch: bool = True
    calibration_callability_min_call_rate_delta: float = -0.02
    calibration_callability_max_call_rate_delta: float = 0.05
    calibration_callability_validation_site_fraction: float = 0.30
    calibration_callability_min_objective_improvement: float = 0.0
    calibration_callability_max_hardcall_maf_shift: float = 0.08
    calibration_callability_max_hardcall_het_shift: float = 0.12
    calibration_train_site_fraction: float = 1.0
    calibration_lightgbm_use_block_context: bool = False
    calibration_lightgbm_use_fixed_stage0: bool = False
    calibration_maf_bins: tuple[float, ...] = (0.0, 0.01, 0.05, 0.5)
    calibration_temperatures: tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.75, 1.0)
    calibration_blends: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    calibration_dosage_scales: tuple[float, ...] = (0.75, 1.0, 1.25, 1.5, 2.0)
    calibration_dosage_offsets: tuple[float, ...] = (-0.25, 0.0, 0.25)
    calibration_hwe_prior_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    calibration_optimize_dosage_scale: bool = True
    calibration_hwe_weight: float = 0.0
    calibration_hwe_min_maf: float = 0.05
    calibration_sanity_checks: bool = True
    calibration_max_mean_abs_dosage_shift: float = 0.35
    calibration_max_mean_abs_maf_shift: float = 0.20
    calibration_max_het_rate_shift: float = 0.25
    calibration_max_mean_entropy_shift: float = 0.50
    write_diagnostics: bool = True
    diagnostics_fail_on_error: bool = False
    diagnostics_warn_het_rate: float = 0.98
    diagnostics_fail_het_rate: float = 0.995
    diagnostics_warn_hom_rate: float = 0.995
    diagnostics_fail_hom_rate: float = 0.999
    diagnostics_warn_missing_rate: float = 0.50
    diagnostics_fail_missing_rate: float = 0.90
    diagnostics_warn_low_info: float = 0.05
    diagnostics_fail_low_info: float = -0.25
    microarray_plink_path: str | Path | None = None
    microarray_add_samples: bool = True
    microarray_use_as_hmm_evidence: bool = True
    microarray_generation_default: float | None = None
    microarray_hard_call_weight: int = 80
    transition_output: Literal["compact", "factorized", "full"] = "compact"
    output_store: Literal["parquet", "zarr"] = "parquet"
    zarr_chunk_samples: int = 256
    zarr_chunk_positions: int = 4096
    zarr_dosage_dtype: str = "float32"
    zarr_gp_dtype: str = "float32"
    zarr_probability_scale: int = 65535
    zarr_compressor: str = "none"
    zarr_compression_level: int = 1
    zarr_bitpack_support_mask: bool = False
    fragment_coupling_model: Literal["stitch_parity"] = "stitch_parity"
    transition_model: Literal["stitch_parity", "factorized", "low_rank_linear", "sparse_factorized"] = "stitch_parity"
    transition_factor_rank: int = 1
    transition_factor_regularization: float = 100.0
    transition_factor_max_deviation: float = 0.25
    transition_factor_train: bool = True
    transition_factor_init_max_cells: int = 50_000_000
    transition_sparse_top_k: int = 4
    known_founder_fast_path: bool = True
    output_minimal: bool = False
    founder_informative_compression: bool = False
    pseudo_haploid_cpu: bool = False
    use_unordered_diploid_states: bool = False
    sparse_emission_scan: bool = True
    generation_bin_size: float = 0.0
    sample_shard_index: int = 0
    sample_shard_count: int = 1
    exact_hmm_chunk_size: int = 20_000
    hmm_checkpoint_interval: int = 0
    fragment_max_difference_between_reads: float = 100.0
    fragment_max_emission_matrix_difference: float = 1000.0
    fragment_rescale_read_likelihood: bool = True
    random_seed: int = 0
    founder_init_jitter: float = 0.0
    executor: Literal["serial", "dask"] = "serial"
    dask_scheduler: Literal["local", "threads", "synchronous", "jobqueue"] = "local"
    dask_n_workers: int = 0
    dask_threads_per_worker: int = 1
    dask_processes: bool = False
    dask_memory_limit: str | None = None
    dask_dashboard_address: str | None = ":8787"
    dask_performance_report: str | Path | None = None
    dask_task_stream: str | Path | None = None
    dask_dashboard_hold_seconds: float = 0.0
    dask_target_task_memory_mb: float = 0.0
    dask_min_block_size: int = 128
    dask_min_sample_batch_size: int = 8
    dask_sample_batch_size: int = 0
    dask_jobqueue_class: str = "SLURMCluster"
    dask_jobqueue_queue: str | None = None
    dask_jobqueue_account: str | None = None
    dask_jobqueue_cores: int = 1
    dask_jobqueue_memory: str | None = None
    dask_jobqueue_walltime: str | None = None
    founder: FounderConfig = field(default_factory=FounderConfig)

    def io(self) -> IOConfig:
        return IOConfig(
            positions_path=self.positions_path,
            output_dir=self.output_dir,
            compression=self.compression,
            compression_level=self.compression_level,
            write_pileup=self.write_pileup,
            write_transitions=self.write_transitions,
            write_haplotype_probabilities=self.write_haplotype_probabilities,
            write_genotype_posteriors=self.write_genotype_posteriors,
            write_genotype_calls=self.write_genotype_calls,
            write_support_mask=self.write_support_mask,
            write_transition_summary=self.write_transition_summary,
            transition_output=self.transition_output,
            output_store=self.output_store,
            zarr_chunk_samples=self.zarr_chunk_samples,
            zarr_chunk_positions=self.zarr_chunk_positions,
            zarr_dosage_dtype=self.zarr_dosage_dtype,
            zarr_gp_dtype=self.zarr_gp_dtype,
            zarr_probability_scale=self.zarr_probability_scale,
            zarr_compressor=self.zarr_compressor,
            zarr_compression_level=self.zarr_compression_level,
            zarr_bitpack_support_mask=self.zarr_bitpack_support_mask,
        )

    def hmm(self) -> HMMConfig:
        return HMMConfig(
            n_founders=self.n_founders,
            ploidy_mode=self.ploidy_mode,
            ploidy=self.ploidy,
            em_iterations=self.em_iterations,
            final_diploid_iterations=self.final_diploid_iterations,
            em_convergence_tol=self.em_convergence_tol,
            em_convergence_min_iterations=self.em_convergence_min_iterations,
            em_convergence_patience=self.em_convergence_patience,
            em_founder_update_damping=self.em_founder_update_damping,
            adaptive_em=self.adaptive_em,
            adaptive_em_restore_best_founders=self.adaptive_em_restore_best_founders,
            em_multistarts=self.em_multistarts,
            founder_update_hardening=self.founder_update_hardening,
            sequencing_error_rate=self.sequencing_error_rate,
            recombination_rate_cM_per_Mb=self.recombination_rate_cM_per_Mb,
            trainable_global_recombination_rate=self.trainable_global_recombination_rate,
            allow_recombination_hotspot_window=self.allow_recombination_hotspot_window,
            transition_model=self.transition_model,
            transition_factor_rank=self.transition_factor_rank,
            transition_factor_regularization=self.transition_factor_regularization,
            transition_factor_max_deviation=self.transition_factor_max_deviation,
            transition_factor_train=self.transition_factor_train,
            transition_factor_init_max_cells=self.transition_factor_init_max_cells,
            transition_sparse_top_k=self.transition_sparse_top_k,
            known_founder_fast_path=self.known_founder_fast_path,
            output_minimal=self.output_minimal,
            founder_informative_compression=self.founder_informative_compression,
            pseudo_haploid_cpu=self.pseudo_haploid_cpu,
            use_unordered_diploid_states=self.use_unordered_diploid_states,
            sparse_emission_scan=self.sparse_emission_scan,
            generation_bin_size=self.generation_bin_size,
            exact_hmm_chunk_size=self.exact_hmm_chunk_size,
            hmm_checkpoint_interval=self.hmm_checkpoint_interval,
            block_size=self.block_size,
            pedigree_strength=self.pedigree_strength,
            backend=self.hmm_backend,
            jax_precompile=self.jax_precompile,
            jax_aot_compile=self.jax_aot_compile,
            jax_sample_batch_size=self.jax_sample_batch_size,
            jax_bucket_batch_shapes=self.jax_bucket_batch_shapes,
            jax_count_emission_kernel=self.jax_count_emission_kernel,
            jax_fragment_emission_kernel=self.jax_fragment_emission_kernel,
            jax_persistent_cache_dir=self.jax_persistent_cache_dir,
            force_generic_ploidy_hmm=self.force_generic_ploidy_hmm,
            use_quality_weights=self.use_quality_weights,
            use_fragment_likelihood=self.use_fragment_likelihood,
            fragment_likelihood_mode=self.fragment_likelihood_mode,
            fragment_coupling_model=self.fragment_coupling_model,
            fragment_max_difference_between_reads=self.fragment_max_difference_between_reads,
            fragment_max_emission_matrix_difference=self.fragment_max_emission_matrix_difference,
            fragment_rescale_read_likelihood=self.fragment_rescale_read_likelihood,
            random_seed=self.random_seed,
            founder_init_jitter=self.founder_init_jitter,
            backend_autotune=self.hmm_backend_autotune,
            autotune_samples=self.hmm_autotune_samples,
            autotune_positions=self.hmm_autotune_positions,
        )
