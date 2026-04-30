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
    transition_output: Literal["compact", "full"] = "compact"
    output_store: Literal["parquet", "zarr"] = "parquet"


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
    min_emission_prob: float = 1e-5
    sequencing_error_rate: float = 0.01
    block_size: int = 5000
    max_jit_positions: int = 10000
    pedigree_strength: float = 0.0
    backend: Literal["auto", "numpy", "jax", "torch"] = "auto"
    jax_precompile: bool = True
    jax_aot_compile: bool = True
    jax_sample_batch_size: int = 0
    force_generic_ploidy_hmm: bool = False
    use_quality_weights: bool = True
    use_fragment_likelihood: bool = True
    fragment_likelihood_mode: Literal["replace", "augment"] = "augment"
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
    block_size: int = 5000
    em_iterations: int = 20
    final_diploid_iterations: int = 2
    ploidy_mode: Literal["diploid", "pseudo_haploid"] = "diploid"
    ploidy: int = 2
    ploidy_males: int | None = None
    ploidy_females: int | None = None
    compression: str = "zstd"
    compression_level: int = 6
    sequencing_error_rate: float = 0.01
    pedigree_strength: float = 0.0
    hmm_backend: Literal["auto", "numpy", "jax", "torch"] = "auto"
    jax_precompile: bool = True
    jax_aot_compile: bool = True
    jax_sample_batch_size: int = 0
    force_generic_ploidy_hmm: bool = False
    use_quality_weights: bool = True
    use_fragment_likelihood: bool = True
    fragment_likelihood_mode: Literal["replace", "augment"] = "augment"
    hmm_backend_autotune: bool = True
    hmm_autotune_samples: int = 0
    hmm_autotune_positions: int = 0
    read_mode: Literal["read_stream", "pileup"] = "read_stream"
    read_stream_backend: Literal["auto", "python", "htslib"] = "auto"
    merge_fragments_by_query: bool = True
    read_batch_size: int = 1024
    io_workers: int = 1
    htslib_threads_per_file: int = 1
    memory_map_read_matrices: bool = False
    memory_map_dir: str | Path | None = None
    gc_collect_every_block: bool = False
    profile_memory: bool = True
    write_pileup: bool = False
    write_transitions: bool = True
    write_haplotype_probabilities: bool = False
    write_genotype_posteriors: bool = False
    write_genotype_calls: bool = False
    write_support_mask: bool = False
    calibrate_genotype_posteriors: bool = True
    genotype_posterior_temperature: float = 0.35
    genotype_posterior_blend: float = 0.35
    genotype_call_mode: Literal["argmax", "stitch_no_call", "quality_gated"] = "argmax"
    genotype_call_min_confidence: float = 0.0
    genotype_call_min_margin: float = 0.0
    genotype_call_stitch_threshold: float = 0.9
    genotype_call_correctness_threshold: float = 0.0
    use_lightgbm_calibrator: bool = False
    calibration_context_window: int = 25
    calibration_block_snps: int = 64
    calibration_use_optuna: bool = False
    calibration_optuna_trials: int = 20
    calibration_max_train_rows: int = 750_000
    microarray_plink_path: str | Path | None = None
    microarray_add_samples: bool = True
    microarray_generation_default: float | None = None
    microarray_hard_call_weight: int = 80
    transition_output: Literal["compact", "full"] = "compact"
    output_store: Literal["parquet", "zarr"] = "parquet"
    fragment_coupling_model: Literal["stitch_parity"] = "stitch_parity"
    fragment_max_difference_between_reads: float = 100.0
    fragment_max_emission_matrix_difference: float = 1000.0
    fragment_rescale_read_likelihood: bool = True
    random_seed: int = 0
    founder_init_jitter: float = 0.0
    executor: Literal["serial", "dask"] = "serial"
    dask_scheduler: Literal["local"] = "local"
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
            transition_output=self.transition_output,
            output_store=self.output_store,
        )

    def hmm(self) -> HMMConfig:
        return HMMConfig(
            n_founders=self.n_founders,
            ploidy_mode=self.ploidy_mode,
            ploidy=self.ploidy,
            em_iterations=self.em_iterations,
            final_diploid_iterations=self.final_diploid_iterations,
            sequencing_error_rate=self.sequencing_error_rate,
            block_size=self.block_size,
            pedigree_strength=self.pedigree_strength,
            backend=self.hmm_backend,
            jax_precompile=self.jax_precompile,
            jax_aot_compile=self.jax_aot_compile,
            jax_sample_batch_size=self.jax_sample_batch_size,
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
