from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import FounderConfig, PipelineConfig
from .cv import run_realdata_cv_harness
from .filenames import reformat_stitch_filenames
from .founders import FounderPanel, load_founders
from .io import load_positions
from .output import (
    DEFAULT_DATASETS,
    combine_parquet_chunks,
    combine_pipeline_outputs,
    export_stitch_bcf_from_parquet,
    write_xarray_float_zarr,
)
from .pedigree_qc import run_pedigree_qc
from .pipeline import StitchPipeline
from .variant_discovery import discover_snp_positions, write_discovered_positions, write_discovery_summary
from .validation import validate_run_against_plink
from .benchmarks import run_synthetic_benchmark
from .autotune import estimate_autotune_plan, write_autotune_plan
from .recombination_diagnostics import compute_recombination_diagnostics
from .postprocess import run_pedigree_postprocess_zarr
from .shards import write_slurm_sample_shards, merge_sharded_matrix_zarr, write_evidence_sample_shard_layout


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _jax_device_report() -> dict[str, object]:
    try:
        import jax

        devices = jax.devices()
        platforms = [str(getattr(device, "platform", "unknown")) for device in devices]
        accelerators = [p for p in platforms if p in {"gpu", "tpu"}]
        return {
            "jax_available": True,
            "devices": platforms,
            "n_devices": int(len(platforms)),
            "n_accelerators": int(len(accelerators)),
            "has_accelerator": bool(accelerators),
        }
    except Exception as exc:
        return {
            "jax_available": False,
            "devices": [],
            "n_devices": 0,
            "n_accelerators": 0,
            "has_accelerator": False,
            "error": str(exc),
        }


def _enforce_jax_accelerator(args: argparse.Namespace) -> None:
    if str(getattr(args, "hmm_backend", "")) != "jax":
        return
    report = _jax_device_report()
    if bool(getattr(args, "require_jax_accelerator", False)) and not bool(report.get("has_accelerator", False)):
        raise RuntimeError(
            "--hmm-backend jax was requested with --require-jax-accelerator, but JAX reports no GPU/TPU. "
            f"JAX device report: {json.dumps(report, sort_keys=True)}. Install a CUDA-enabled jaxlib on the compute node "
            "or remove --require-jax-accelerator to allow CPU fallback."
        )


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=None, engine="python")


def _parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in str(text).split(",") if x.strip())


def _parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _parse_bp_window(value: str | int | float | None) -> int:
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if not text or text in {"0", "false", "none", "off"}:
        return 0
    suffixes = {
        "gib": 1024**3,
        "gb": 1000**3,
        "g": 1000**3,
        "mib": 1024**2,
        "mb": 1000**2,
        "m": 1000**2,
        "kib": 1024,
        "kb": 1000,
        "k": 1000,
        "bp": 1,
        "b": 1,
    }
    for suffix, scale in suffixes.items():
        if text.endswith(suffix):
            return int(float(text[: -len(suffix)]) * scale)
    return int(float(text))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _validate_ploidy_args(args: argparse.Namespace) -> None:
    if int(args.ploidy) < 0:
        raise ValueError(f"--ploidy must be >= 0, got {args.ploidy}")
    males = args.ploidy_males
    females = args.ploidy_females
    if (males is None) ^ (females is None):
        raise ValueError("--ploidy-males and --ploidy-females must be set together.")
    if males is not None and int(males) < 0:
        raise ValueError(f"--ploidy-males must be >= 0, got {males}")
    if females is not None and int(females) < 0:
        raise ValueError(f"--ploidy-females must be >= 0, got {females}")
    if args.chr_start is not None and args.chr_end is not None and int(args.chr_start) > int(args.chr_end):
        raise ValueError("--chr-start must be <= --chr-end")


def _set_if_present(args: argparse.Namespace, name: str, value: object) -> None:
    if hasattr(args, name):
        setattr(args, name, value)


def _apply_preset(args: argparse.Namespace) -> None:
    preset = str(getattr(args, "preset", "") or "").strip().lower().replace("_", "-")
    if not preset:
        return
    valid = {"hs-rat-k8-validate", "hs-rat-k8-production", "hs-rat-k9-experimental", "debug-small"}
    if preset not in valid:
        raise ValueError(f"Unknown --preset {preset!r}. Valid presets: {', '.join(sorted(valid))}")
    common_fast = {
        "transition_model": "stitch_parity",
        "transition_output": "compact",
        "fragment_likelihood_mode": "replace",
        "fragment_coupling_model": "stitch_parity",
        "calibrate_genotype_posteriors": False,
        "genotype_call_mode": "stitch_no_call",
        "genotype_call_stitch_threshold": 0.9,
        "write_gamma": "off",
    }
    for key, value in common_fast.items():
        _set_if_present(args, key, value)
    if preset in {"hs-rat-k8-validate", "hs-rat-k8-production"}:
        _set_if_present(args, "n_founders", 8)
        _set_if_present(args, "founder_immutable", True)
        _set_if_present(args, "em_iterations", 1)
        _set_if_present(args, "jax_sample_batch_size", max(int(getattr(args, "jax_sample_batch_size", 0) or 0), 32))
        _set_if_present(args, "compact_evidence_materialize_dense_counts", False)
        _set_if_present(args, "write_diagnostics", False)
        _set_if_present(args, "output_store", "zarr")
        _set_if_present(args, "output_minimal", True)
        _set_if_present(args, "store_xi", "False")
    elif preset == "hs-rat-k9-experimental":
        _set_if_present(args, "n_founders", 9)
        _set_if_present(args, "transition_model", "factorized")
        _set_if_present(args, "transition_output", "factorized")
        _set_if_present(args, "transition_factor_rank", 4)
        _set_if_present(args, "transition_factor_regularization", 100.0)
        _set_if_present(args, "transition_factor_max_deviation", 0.25)
        _set_if_present(args, "jax_sample_batch_size", max(int(getattr(args, "jax_sample_batch_size", 0) or 0), 32))
    elif preset == "debug-small":
        _set_if_present(args, "em_iterations", min(int(getattr(args, "em_iterations", 2)), 1))
        _set_if_present(args, "jax_sample_batch_size", min(max(int(getattr(args, "jax_sample_batch_size", 0) or 8), 1), 8))
        _set_if_present(args, "write_gamma", "off")
    _set_if_present(args, "snp_block_mode", "exact_chunked")
    _set_if_present(args, "hmm_chunk_size", max(int(getattr(args, "hmm_chunk_size", 0) or 0), 20000))


def _add_common_run_args(parser: argparse.ArgumentParser, *, require_n_founders: bool = True) -> None:
    parser.add_argument("--samples", required=True, help="Samples table (parquet/csv) with bam_path and generation.")
    parser.add_argument("--positions", required=True, help="Position table (parquet/csv) with CHR, POS, REF, ALT.")
    parser.add_argument("--chromosome", required=True)
    parser.add_argument("--chr-start", type=int, default=None, help="Optional start position (inclusive).")
    parser.add_argument("--chr-end", type=int, default=None, help="Optional end position (inclusive).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--preset",
        choices=["hs-rat-k8-validate", "hs-rat-k8-production", "hs-rat-k9-experimental", "debug-small"],
        default="",
        help="Apply a safe, opinionated parameter bundle before running. Explicit flags after expansion are printed in run_config.resolved.json.",
    )
    parser.add_argument(
        "--n-founders",
        type=int,
        required=require_n_founders,
        default=8,
        help=(
            "Total founders in the HMM. If a founder VCF/PLINK has fewer samples, "
            "STITCHCONT appends extra mutable founders initialized at 0.5."
        ),
    )
    parser.add_argument("--ploidy", type=int, default=2, help="Default ploidy for all samples (>=0).")
    parser.add_argument("--ploidy-males", type=int, default=None, help="Male ploidy override (requires --ploidy-females and samples.sex).")
    parser.add_argument("--ploidy-females", type=int, default=None, help="Female ploidy override (requires --ploidy-males and samples.sex).")
    parser.add_argument(
        "--block-size",
        type=int,
        default=0,
        help=(
            "HMM SNP block size for approximate modes (0=auto from --max-mem). "
            "In exact_streaming/exact_chunked modes, STITCHCONT preserves the full SNP span; exact_chunked uses --hmm-chunk-size internally."
        ),
    )
    parser.add_argument("--max-mem", default="90%", help="Maximum memory budget for automatic block/window planning, e.g. 90%%, 64G, 32000M.")
    parser.add_argument("--allow-memory-overcommit", action="store_true", help="Bypass the pre-run memory guard that refuses blocks whose estimated peak HMM working set exceeds --max-mem. Accepts the OOM risk.")
    parser.add_argument("--io-window-size", type=int, default=0, help="SNPs per BAM extraction window (0=auto from --max-mem).")
    parser.add_argument(
        "--snp-block-mode",
        choices=["exact_streaming", "exact_chunked", "independent_approx", "density_balanced_overlap"],
        default="exact_streaming",
        help=(
            "SNP blocking mode. exact_streaming runs a whole-region HMM in one tensor; exact_chunked preserves whole-region HMM math "
            "but stores/recomputes boundary messages to lower peak memory; independent_approx and density_balanced_overlap are explicitly approximate QC modes."
        ),
    )
    parser.add_argument(
        "--hmm-chunk-size",
        type=int,
        default=20000,
        help="SNPs per exact HMM boundary-message chunk for --snp-block-mode exact_chunked; 0 disables internal exact chunking.",
    )
    parser.add_argument(
        "--output-store",
        choices=["parquet", "zarr"],
        default="parquet",
        help="Primary matrix output store. 'zarr' writes dosage/GP/calls/support as dense sample x variant Zarr arrays instead of long-form Parquet.",
    )
    parser.add_argument("--zarr-chunk-samples", type=int, default=256, help="Sample chunk size for --output-store zarr matrix outputs.")
    parser.add_argument("--zarr-chunk-positions", type=int, default=4096, help="Position chunk size for --output-store zarr matrix outputs.")
    parser.add_argument("--zarr-dosage-dtype", choices=["float32", "float16"], default="float32", help="Stored dtype for Zarr dosage arrays. float16 halves output size for tuning/production when acceptable.")
    parser.add_argument("--zarr-gp-dtype", choices=["float32", "float16", "uint16"], default="float32", help="Stored dtype for Zarr genotype posterior arrays. uint16 stores probabilities scaled by --zarr-probability-scale.")
    parser.add_argument("--zarr-probability-scale", type=int, default=65535, help="Scale used for uint16 probability Zarr output.")
    parser.add_argument("--zarr-compressor", choices=["none", "zlib"], default="none", help="Manual Zarr v2 chunk compressor. zlib is widely supported by numcodecs/zarr readers; none is fastest.")
    parser.add_argument("--zarr-compression-level", type=int, default=1, help="Compression level for --zarr-compressor zlib.")
    parser.add_argument("--zarr-bitpack-support-mask", action="store_true", help="When writing Zarr matrix outputs, store support_mask as packed uint8 bits instead of a dense bool matrix.")
    parser.add_argument("--output-minimal", action="store_true", help="Speed mode: do not materialize gamma/xi and only reduce to requested dosage/GP/calls.")
    parser.add_argument("--founder-informative-compression", action="store_true", help="Use exact fast reductions at founder-uninformative sites and report founder-informativeness diagnostics.")
    parser.add_argument("--pseudo-haploid-cpu", action="store_true", help="Experimental CPU approximation: process two haploid chains rather than an exact diploid chain for fast screening.")
    parser.add_argument("--hmm-checkpoint-interval", type=int, default=0, help="Store alpha checkpoints every N SNPs and recompute within segments during backward/output. 0 uses full chunk alpha/beta arrays.")
    parser.add_argument("--no-known-founder-fast-path", dest="known_founder_fast_path", action="store_false", help="Disable automatic one-pass immutable-founder fast path.")
    parser.add_argument("--use-unordered-diploid-states", action="store_true", help="Use K*(K+1)/2 unordered diploid states in CPU known-founder runs instead of K*K ordered states.")
    parser.add_argument("--no-sparse-emission-scan", dest="sparse_emission_scan", action="store_false", help="Disable sparse/on-the-fly emission helpers where implemented.")
    parser.add_argument("--generation-bin-size", type=float, default=0.0, help="Bin generation values to this width before switch-probability calculation; 0 keeps exact generation values.")
    parser.add_argument("--hmm-mode", choices=["diploid", "pseudo_haploid"], default="diploid", help="HMM mode. pseudo_haploid is a fast approximation and should be validated against diploid.")
    parser.add_argument("--sample-shard-index", type=int, default=0, help="Zero-based immutable-founder sample shard index.")
    parser.add_argument("--sample-shard-count", type=int, default=1, help="Number of immutable-founder sample shards; samples are partitioned by row modulo shard count.")
    parser.add_argument(
        "--memory-safety-fraction",
        type=float,
        default=0.60,
        help="Fraction of --max-mem available to the in-memory planner after write/cache/JAX headroom.",
    )
    parser.add_argument("--scratch-dir", default="", help="Optional Zarr/Parquet/Arrow scratch directory for streamed state/output chunks.")
    parser.add_argument(
        "--approx-overlap-fraction",
        type=float,
        default=0.05,
        help="Overlap fraction for density_balanced_overlap approximate SNP blocks.",
    )
    parser.add_argument(
        "--approx-min-overlap-snps",
        type=int,
        default=50,
        help="Minimum SNP overlap per side for density_balanced_overlap approximate blocks.",
    )
    parser.add_argument(
        "--store-xi",
        choices=["full", "per-snp", "False"],
        default="per-snp",
        help="Xi output policy: per-snp writes compact transition/hotspot summaries, full writes posterior xi Zarr, False disables xi output.",
    )
    parser.add_argument("--write-xi", choices=["auto", "off", "force"], default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--write-gamma",
        choices=["summary", "off", "full"],
        default="summary",
        help="Gamma posterior output policy. summary writes compact diagnostics; full is opt-in.",
    )
    parser.add_argument(
        "--no-write-hmm-boundaries",
        dest="write_hmm_boundaries",
        action="store_false",
        help="Disable boundary alpha/beta metadata for exact streaming diagnostics.",
    )
    parser.add_argument(
        "--stitch-compat",
        action="store_true",
        help="Use original STITCH-like read/HMM parity settings: expRate=0.5, bqFilter/mapQ=17, BQ capped by MQ, insert size <=600, ref/alt-only evidence, fragment replace mode, and STITCH likelihood caps.",
    )
    parser.add_argument("--recombination-rate-cm-per-mb", type=float, default=1.0)
    parser.add_argument(
        "--genetic-map",
        default="",
        help="Optional parquet/csv genetic map with POS and CM (or GENETIC_CM); cM is interpolated onto --positions.",
    )
    parser.add_argument(
        "--trainable-global-recombination-rate",
        "--trainable_global_recombination_rate",
        dest="trainable_global_recombination_rate",
        action="store_true",
        help="Estimate a block-level global recombination-rate scale from an initial HMM posterior pass.",
    )
    parser.add_argument(
        "--allow-recombination-hotspot-window",
        "--allow_recombination_hotspot_window",
        dest="allow_recombination_hotspot_window",
        default="0",
        help="Enable local recombination-rate scaling in windows such as 5Mb; 0 disables.",
    )
    parser.add_argument(
        "--transition-model",
        choices=["stitch_parity", "factorized", "low_rank_linear", "sparse_factorized"],
        default="stitch_parity",
        help="HMM transition model. stitch_parity is the production default; factorized is a dense learned offdiag model; low_rank_linear uses algebraic low-rank transitions; sparse_factorized keeps top-M learned destinations.",
    )
    parser.add_argument("--transition-factor-rank", type=int, default=1, help="Rank for experimental factorized/low-rank transition logits.")
    parser.add_argument("--transition-sparse-top-k", type=int, default=4, help="Destination founders per source for --transition-model sparse_factorized; <=0 disables pruning.")
    parser.add_argument(
        "--transition-factor-regularization",
        type=float,
        default=100.0,
        help="Uniform-parity pseudo-count strength for factorized transition fitting.",
    )
    parser.add_argument(
        "--transition-factor-max-deviation",
        type=float,
        default=0.25,
        help="Maximum multiplicative off-diagonal deviation from parity before row renormalization.",
    )
    parser.add_argument(
        "--no-transition-factor-train",
        dest="transition_factor_train",
        action="store_false",
        help="Keep factorized transition factors at the exact STITCH parity initialization.",
    )
    parser.add_argument(
        "--transition-factor-init-max-cells",
        type=int,
        default=50_000_000,
        help=(
            "Diagnostic threshold for factorized-transition initialization size. "
            "Large runs are initialized by sample-batched accumulation instead of materializing sample*SNP*K tensors."
        ),
    )
    parser.add_argument("--em-iterations", type=int, default=2)
    parser.add_argument("--em-convergence-tol", type=float, default=1e-4, help="Stop mutable-founder EM early when max founder-probability delta is at or below this value (0 disables).")
    parser.add_argument("--em-convergence-min-iterations", type=int, default=2, help="Minimum mutable-founder EM statistic passes before convergence checks can stop the loop.")
    parser.add_argument("--em-convergence-patience", type=int, default=1, help="Consecutive converged mutable-founder EM passes required before the final posterior pass.")
    parser.add_argument("--em-founder-update-damping", type=float, default=1.0, help="Mutable-founder EM update damping in (0,1]; lower values stabilize difficult no-founder runs.")
    parser.add_argument("--no-adaptive-em", dest="adaptive_em", action="store_false", help="Disable adaptive EM best-founder bookkeeping.")
    parser.add_argument("--no-adaptive-em-restore-best-founders", dest="adaptive_em_restore_best_founders", action="store_false")
    parser.add_argument("--em-multistarts", type=int, default=1, help="Optional mutable-founder multi-start count; fixed-founder parity mode ignores this.")
    parser.add_argument("--no-founder-update-hardening", dest="founder_update_hardening", action="store_false", help="Disable discrete hardening of mutable founder haplotype updates.")
    parser.add_argument("--hmm-backend", choices=["auto", "numpy", "numba", "cpp", "jax", "torch"], default="auto")
    parser.add_argument("--jax-sample-batch-size", type=int, default=0, help="Samples per JAX FB call (0=all samples).")
    parser.add_argument("--no-jax-bucket-batch-shapes", dest="jax_bucket_batch_shapes", action="store_false", help="Disable JAX batch-size bucketing that avoids compiling a separate remainder batch shape.")
    parser.add_argument("--no-jax-count-emission-kernel", dest="jax_count_emission_kernel", action="store_false", help="Disable count-to-emission work inside JAX for no-fragment diploid batches.")
    parser.add_argument("--no-jax-fragment-emission-kernel", dest="jax_fragment_emission_kernel", action="store_false", help="Disable fragment-aware JAX emission kernels and fall back to NumPy fragment emission assembly.")
    parser.add_argument("--jax-persistent-cache-dir", default="", help="Optional persistent JAX compilation cache directory reused across runs.")
    parser.add_argument(
        "--require-jax-accelerator",
        action="store_true",
        help="Fail immediately if --hmm-backend jax resolves to CPU-only JAX. Useful on SLURM GPU jobs.",
    )
    parser.add_argument("--executor", choices=["serial", "dask"], default="serial", help="Execution orchestrator.")
    parser.add_argument(
        "--dask-scheduler",
        choices=["local", "threads", "synchronous", "jobqueue"],
        default="local",
        help="Dask scheduler type. local uses distributed.LocalCluster; threads/synchronous avoid the distributed scheduler; jobqueue uses dask-jobqueue.",
    )
    parser.add_argument("--dask-n-workers", type=int, default=0, help="Local Dask worker count (0=auto).")
    parser.add_argument("--dask-threads-per-worker", type=int, default=1)
    parser.add_argument("--dask-processes", action="store_true", help="Use Dask worker processes instead of worker threads.")
    parser.add_argument("--dask-memory-limit", default="", help="Per-worker memory limit, e.g. 16GB (empty=auto).")
    parser.add_argument("--dask-dashboard-address", default=":8787", help="Dashboard bind address; empty disables dashboard.")
    parser.add_argument("--dask-performance-report", default="", help="Write Dask performance report HTML to this path.")
    parser.add_argument("--dask-task-stream", default="", help="Write captured Dask task stream to .html or .json.")
    parser.add_argument("--dask-dashboard-hold-seconds", type=float, default=0.0, help="Keep the Dask dashboard alive after compute before closing.")
    parser.add_argument("--dask-target-task-memory-mb", type=float, default=0.0, help="Auto-shrink chunks to this rough per-task memory target.")
    parser.add_argument("--dask-min-block-size", type=int, default=128)
    parser.add_argument("--dask-min-sample-batch-size", type=int, default=8)
    parser.add_argument("--dask-sample-batch-size", type=int, default=0, help="Samples per Dask HMM task (0=all or planner result).")
    parser.add_argument("--dask-jobqueue-class", default="SLURMCluster", help="dask-jobqueue cluster class for --dask-scheduler jobqueue.")
    parser.add_argument("--dask-jobqueue-queue", default="")
    parser.add_argument("--dask-jobqueue-account", default="")
    parser.add_argument("--dask-jobqueue-cores", type=int, default=1)
    parser.add_argument("--dask-jobqueue-memory", default="")
    parser.add_argument("--dask-jobqueue-walltime", default="")
    parser.add_argument("--read-mode", choices=["read_stream", "pileup"], default="read_stream")
    parser.add_argument(
        "--read-stream-backend",
        choices=["auto", "python", "htslib", "snp_only_bamreader", "stitch_style_bamreader", "variant_aware_bamreader"],
        default="auto",
    )
    parser.add_argument("--min-base-quality", type=int, default=13)
    parser.add_argument("--min-mapping-quality", type=int, default=20)
    parser.add_argument("--max-insert-size", type=int, default=0, help="Skip reads with absolute template length above this value; 0 disables.")
    parser.add_argument("--max-indel-len", type=int, default=50, help="Maximum targeted insertion/deletion length for variant-aware BAM reading.")
    parser.add_argument("--cap-base-quality-by-mapping-quality", action="store_true")
    parser.add_argument("--ref-alt-only", action="store_true", help="Ignore bases that are neither REF nor ALT instead of treating them as OTHER evidence.")
    parser.add_argument("--no-merge-fragments-by-query", dest="merge_fragments_by_query", action="store_false")
    parser.add_argument(
        "--merge-unpaired-fragments-by-query",
        action="store_true",
        help="STITCH-style fragment grouping: merge all informative reads sharing a query name, even if the BAM paired flag is absent.",
    )
    parser.add_argument("--use-bx-tag", action="store_true", help="Merge reads with the same linked-read BX tag when present.")
    parser.add_argument("--bx-tag", default="BX", help="Two-character SAM auxiliary tag used for linked-read fragment merging.")
    parser.add_argument("--bx-tag-upper-limit", type=int, default=50000, help="Maximum observations accumulated in one BX/query fragment before flushing.")
    parser.add_argument("--downsample-to-coverage", type=int, default=0, help="Fragment-level per-sample/site coverage cap before HMM; 0 disables.")
    parser.add_argument("--downsample-fraction", type=float, default=1.0, help="Random fragment retention fraction before HMM.")
    parser.add_argument("--io-workers", type=int, default=1)
    parser.add_argument("--htslib-threads-per-file", type=int, default=1)
    parser.add_argument("--compact-evidence-cache-dir", default="", help="Optional directory for compact fragment-evidence cache files.")
    parser.add_argument(
        "--compact-evidence-cache-mode",
        choices=["off", "read", "write", "readwrite"],
        default="off",
        help="Read/write compact fragment evidence before HMM. Cached runs can skip BAM decoding.",
    )
    parser.add_argument(
        "--compact-evidence-cache-format",
        choices=["parquet_zarr"],
        default="parquet_zarr",
        help="Evidence cache storage format. Compact fragments/support are compressed Parquet; optional dense arrays are Zarr.",
    )
    parser.add_argument("--compact-evidence-cache-sample-batch-size", type=int, default=256)
    parser.add_argument(
        "--compact-evidence-no-dense-counts",
        dest="compact_evidence_materialize_dense_counts",
        action="store_false",
        help="When loading compact evidence, build only a support mask and let replace-mode fragment likelihoods drive the HMM.",
    )
    parser.add_argument(
        "--compact-evidence-cache-include-dense-counts",
        action="store_true",
        help="Store dense count/weight matrices in the compact evidence cache. Faster and exact for augment mode, but larger on disk.",
    )
    parser.add_argument("--fragment-likelihood-mode", choices=["replace", "augment"], default="replace")
    parser.add_argument(
        "--fragment-coupling-model",
        choices=["stitch_parity"],
        default="stitch_parity",
        help="Fragment/read coupling behavior.",
    )
    parser.add_argument("--fragment-max-diff-reads", type=float, default=100.0)
    parser.add_argument("--fragment-max-emission-diff", type=float, default=1000.0)
    parser.add_argument(
        "--no-fragment-rescale-read-likelihood",
        dest="fragment_rescale_read_likelihood",
        action="store_false",
    )
    parser.add_argument("--write-transitions", action="store_true")
    parser.add_argument(
        "--transition-output",
        choices=["compact", "factorized", "full"],
        default="compact",
        help="Transition output form. compact writes switch/stay/offdiag; factorized also writes low-rank factor metadata; full writes expanded state matrices.",
    )
    parser.add_argument("--write-haplotype-probabilities", action="store_true")
    parser.add_argument("--write-genotype-posteriors", action="store_true")
    parser.add_argument("--write-genotype-calls", action="store_true")
    parser.add_argument("--write-support-mask", action="store_true")
    parser.add_argument(
        "--no-write-transition-summary",
        dest="write_transition_summary",
        action="store_false",
        help="Disable compact per-interval transition/hotspot summary output.",
    )
    parser.add_argument(
        "--no-calibrate-genotype-posteriors",
        dest="calibrate_genotype_posteriors",
        action="store_false",
    )
    parser.add_argument(
        "--calibration-mode",
        choices=["standard_callability", "fixed", "masked_cv"],
        default="standard_callability",
        help="Calibration mode. standard_callability keeps raw HMM GP and learns a P(correct) hard-call gate; fixed/masked_cv alter genotype posteriors.",
    )
    parser.add_argument("--genotype-posterior-temperature", type=float, default=0.35)
    parser.add_argument("--genotype-posterior-blend", type=float, default=0.35)
    parser.add_argument(
        "--genotype-call-mode",
        choices=["argmax", "stitch_no_call", "quality_gated"],
        default="quality_gated",
        help="Hard-call mode: argmax, STITCH-style GP>=0.9 no-call, or quality-gated (confidence+margin+correctness model).",
    )
    parser.add_argument("--genotype-call-min-confidence", type=float, default=0.0)
    parser.add_argument("--genotype-call-min-margin", type=float, default=0.0)
    parser.add_argument("--genotype-call-stitch-threshold", type=float, default=0.9)
    parser.add_argument("--genotype-call-correctness-threshold", type=float, default=0.0)
    parser.add_argument("--use-lightgbm-calibrator", action="store_true")
    parser.add_argument("--calibration-context-window", type=int, default=25)
    parser.add_argument("--calibration-block-snps", type=int, default=64)
    parser.add_argument(
        "--calibration-truth-source",
        choices=["read_evidence", "microarray", "auto", "none"],
        default="read_evidence",
        help="Truth source for learned calibration. read_evidence uses high-confidence read-backed sample-sites and avoids gold-standard labels.",
    )
    parser.add_argument("--calibration-read-truth-min-depth", type=int, default=3)
    parser.add_argument("--calibration-read-truth-min-hom-depth", type=int, default=2)
    parser.add_argument("--calibration-read-truth-min-het-depth", type=int, default=4)
    parser.add_argument("--calibration-read-truth-min-het-allele-depth", type=int, default=1)
    parser.add_argument("--calibration-read-truth-hom-major-fraction", type=float, default=0.95)
    parser.add_argument("--calibration-read-truth-het-balance-min", type=float, default=0.25)
    parser.add_argument("--calibration-read-truth-het-balance-max", type=float, default=0.75)
    parser.add_argument("--calibration-read-truth-max-other-fraction", type=float, default=0.05)
    parser.add_argument(
        "--calibration-read-truth-holdout-fraction",
        type=float,
        default=0.30,
        help="Fraction of read fragments sampled only for read-evidence pseudo-truth. The final HMM still uses all read evidence.",
    )
    parser.add_argument("--calibration-use-optuna", action="store_true")
    parser.add_argument("--calibration-optuna-trials", type=int, default=20)
    parser.add_argument("--calibration-max-train-rows", type=int, default=750000)
    parser.add_argument(
        "--calibration-callability-model",
        choices=["sklearn_logistic", "lightgbm"],
        default="lightgbm",
        help="Model used by standard_callability. lightgbm is the default; sklearn_logistic is the lighter linear option.",
    )
    parser.add_argument("--calibration-callability-min-train-rows", type=int, default=64)
    parser.add_argument("--calibration-callability-min-call-rate", type=float, default=0.80)
    parser.add_argument("--calibration-callability-call-rate-weight", type=float, default=0.03)
    parser.add_argument(
        "--calibration-callability-decision-mode",
        choices=["per_snp_hierarchical", "global"],
        default="per_snp_hierarchical",
        help=(
            "How standard_callability decides whether to use learned callability. "
            "Default per_snp_hierarchical evaluates each SNP with local/stratum/global fallback to STITCH no-call."
        ),
    )
    parser.add_argument(
        "--no-calibration-callability-bound-to-stitch",
        dest="calibration_callability_bound_to_stitch",
        action="store_false",
        help="Allow the learned callability threshold to move without the STITCH GP no-call call-rate bound.",
    )
    parser.add_argument("--calibration-callability-min-call-rate-delta", type=float, default=-0.02)
    parser.add_argument("--calibration-callability-max-call-rate-delta", type=float, default=0.05)
    parser.add_argument("--calibration-callability-validation-site-fraction", type=float, default=0.30)
    parser.add_argument("--calibration-callability-min-objective-improvement", type=float, default=0.0)
    parser.add_argument("--calibration-callability-max-hardcall-maf-shift", type=float, default=0.08)
    parser.add_argument("--calibration-callability-max-hardcall-het-shift", type=float, default=0.12)
    parser.add_argument(
        "--calibration-train-site-fraction",
        type=float,
        default=1.0,
        help="Fraction of labeled SNPs used to fit learned calibration; the fitted calibrator is applied to all SNPs.",
    )
    parser.add_argument(
        "--calibration-lightgbm-use-block-context",
        action="store_true",
        help="Enable the older local-context LightGBM stage before read/site-aware calibration.",
    )
    parser.add_argument(
        "--calibration-lightgbm-use-fixed-stage0",
        action="store_true",
        help="Use fixed temperature/blend calibration before LightGBM; default keeps raw HMM GP as the learned-calibration input.",
    )
    parser.add_argument("--calibration-maf-bins", default="0,0.01,0.05,0.5")
    parser.add_argument("--calibration-temperatures", default="0.15,0.25,0.35,0.5,0.75,1.0")
    parser.add_argument("--calibration-blends", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--calibration-dosage-scales", default="0.75,1,1.25,1.5,2")
    parser.add_argument("--calibration-dosage-offsets", default="-0.25,0,0.25")
    parser.add_argument("--calibration-hwe-prior-weights", default="0,0.25,0.5,1")
    parser.add_argument("--no-calibration-optimize-dosage-scale", dest="calibration_optimize_dosage_scale", action="store_false")
    parser.add_argument("--calibration-hwe-weight", type=float, default=0.0)
    parser.add_argument("--calibration-hwe-min-maf", type=float, default=0.05)
    parser.add_argument("--no-calibration-sanity-checks", dest="calibration_sanity_checks", action="store_false")
    parser.add_argument("--calibration-max-mean-abs-dosage-shift", type=float, default=0.35)
    parser.add_argument("--calibration-max-mean-abs-maf-shift", type=float, default=0.20)
    parser.add_argument("--calibration-max-het-rate-shift", type=float, default=0.25)
    parser.add_argument("--calibration-max-mean-entropy-shift", type=float, default=0.50)
    parser.add_argument("--no-write-diagnostics", dest="write_diagnostics", action="store_false")
    parser.add_argument("--diagnostics-fail-on-error", action="store_true")
    parser.add_argument("--diagnostics-warn-het-rate", type=float, default=0.98)
    parser.add_argument("--diagnostics-fail-het-rate", type=float, default=0.995)
    parser.add_argument("--diagnostics-warn-hom-rate", type=float, default=0.995)
    parser.add_argument("--diagnostics-fail-hom-rate", type=float, default=0.999)
    parser.add_argument("--diagnostics-warn-missing-rate", type=float, default=0.50)
    parser.add_argument("--diagnostics-fail-missing-rate", type=float, default=0.90)
    parser.add_argument("--diagnostics-warn-low-info", type=float, default=0.05)
    parser.add_argument("--diagnostics-fail-low-info", type=float, default=-0.25)
    parser.add_argument("--microarray-plink", default="", help="Optional PLINK prefix for hard microarray genotype evidence.")
    parser.add_argument("--microarray-generation-default", type=float, default=np.nan)
    parser.add_argument("--microarray-hard-call-weight", type=int, default=80)
    parser.add_argument(
        "--microarray-calibration-only",
        dest="microarray_use_as_hmm_evidence",
        action="store_false",
        help="Use --microarray-plink only as calibration truth; do not inject hard calls into HMM read evidence.",
    )
    parser.add_argument("--no-microarray-add-samples", dest="microarray_add_samples", action="store_false")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--founder-init-jitter", type=float, default=0.0)
    parser.add_argument("--memory-map-read-matrices", action="store_true")
    parser.add_argument("--memory-map-dir", default="")
    parser.add_argument("--no-profile-memory", dest="profile_memory", action="store_false")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--compression-level", type=int, default=6)
    parser.add_argument("--pedigree", default="", help="Optional pedigree table (parquet/csv) with offspring and parent columns.")
    parser.add_argument(
        "--pedigree-mode",
        choices=["off", "smooth", "kinship", "transmission"],
        default="smooth",
        help="Post-HMM pedigree mode: off, dosage smoothing, kinship fallback, or transmission message passing.",
    )
    parser.add_argument("--pedigree-strength", type=float, default=0.0)
    parser.add_argument("--pedigree-offspring-col", default="sample_id")
    parser.add_argument("--pedigree-parent1-col", default="father_id")
    parser.add_argument("--pedigree-parent2-col", default="mother_id")
    parser.add_argument("--pedigree-iterations", type=int, default=4)
    parser.add_argument("--pedigree-kinship-threshold", type=float, default=0.01)
    parser.set_defaults(
        profile_memory=True,
        fragment_rescale_read_likelihood=True,
        calibrate_genotype_posteriors=True,
        calibration_optimize_dosage_scale=True,
        calibration_sanity_checks=True,
        calibration_callability_bound_to_stitch=True,
        write_diagnostics=True,
        adaptive_em=True,
        adaptive_em_restore_best_founders=True,
        microarray_add_samples=True,
        transition_factor_train=True,
        merge_fragments_by_query=True,
    )


def cmd_doctor(args: argparse.Namespace) -> None:
    positions_df = load_positions(
        args.positions,
        args.chromosome,
        start=getattr(args, "chr_start", None),
        end=getattr(args, "chr_end", None),
    )
    samples = _read_table(args.samples)
    problems: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []

    def add_problem(code: str, message: str, **extra: object) -> None:
        problems.append({"code": code, "message": message, **extra})

    def add_warning(code: str, message: str, **extra: object) -> None:
        warnings.append({"code": code, "message": message, **extra})

    pos = positions_df["POS"].to_numpy(dtype=np.int64) if "POS" in positions_df.columns else np.array([], dtype=np.int64)
    if pos.size == 0:
        add_problem("empty_positions", "No positions were loaded for the requested chromosome/window.")
    else:
        if not bool(np.all(np.diff(pos) >= 0)):
            add_problem("positions_not_sorted", "Positions are not sorted in nondecreasing order.")
        dup_count = int(pd.Series(pos).duplicated().sum())
        if dup_count:
            add_problem("duplicate_positions", f"Found {dup_count} duplicate variant positions.", duplicate_count=dup_count)
    for column in ["REF", "ALT"]:
        if column not in positions_df.columns:
            add_problem("missing_position_column", f"Positions table is missing required column {column!r}.", column=column)
        else:
            missing = int(positions_df[column].isna().sum())
            if missing:
                add_warning("missing_alleles", f"Column {column!r} has {missing} missing values.", column=column, missing=missing)

    for column in ["bam_path", "generation"]:
        if column not in samples.columns:
            add_problem("missing_sample_column", f"Samples table is missing expected column {column!r}.", column=column)
    if "generation" in samples.columns:
        gen = pd.to_numeric(samples["generation"], errors="coerce")
        finite = gen[np.isfinite(gen)]
        if finite.empty:
            add_problem("invalid_generation", "No finite generation values were found; transition scaling will be invalid.")
        else:
            if float(finite.min()) < 0:
                add_warning("negative_generation", "Some generation values are negative.", min_generation=float(finite.min()))
            if float(finite.max()) > 500:
                add_warning("large_generation", "Generation values are very large; verify HMM switch scaling.", max_generation=float(finite.max()))
    if "bam_path" in samples.columns:
        empty_bam = int(samples["bam_path"].fillna("").astype(str).str.len().eq(0).sum())
        if empty_bam:
            add_warning("empty_bam_paths", f"{empty_bam} samples have empty bam_path values.", count=empty_bam)

    founder_summary: dict[str, object] | None = None
    if getattr(args, "founder_vcf", "") or getattr(args, "founder_plink", ""):
        try:
            if getattr(args, "founder_vcf", ""):
                founder_panel = load_founders("vcf", args.founder_vcf, args.chromosome, positions_df, immutable=bool(args.founder_immutable))
            else:
                founder_panel = load_founders("plink", args.founder_plink, args.chromosome, positions_df, immutable=bool(args.founder_immutable))
            n_loaded = int(founder_panel.n_founders)
            n_target = int(getattr(args, "n_founders", n_loaded) or n_loaded)
            founder_summary = {
                "loaded_founders": n_loaded,
                "target_n_founders": n_target,
                "immutable_loaded_founders": int(np.sum(founder_panel.immutable_mask)),
                "position_overlap": int(founder_panel.n_positions),
                "mean_alt_prob": float(np.nanmean(founder_panel.alt_prob)),
                "missing_or_ambiguous_fraction": float(np.mean(np.isclose(founder_panel.alt_prob, 0.5))),
                "monomorphic_site_fraction": float(np.mean(np.nanstd(founder_panel.alt_prob, axis=0) < 1e-6)),
                "has_genetic_cm": bool(founder_panel.genetic_cm is not None),
            }
            if n_target > n_loaded:
                add_warning(
                    "extra_mutable_founders",
                    f"Loaded {n_loaded} founders but --n-founders={n_target}; STITCHCONT will append {n_target - n_loaded} mutable founder(s).",
                    loaded_founders=n_loaded,
                    target_n_founders=n_target,
                )
            elif n_target < n_loaded:
                add_problem(
                    "target_founders_less_than_loaded",
                    f"--n-founders={n_target} is smaller than the loaded founder count {n_loaded}.",
                    loaded_founders=n_loaded,
                    target_n_founders=n_target,
                )
        except Exception as exc:
            add_problem("founder_load_failed", f"Could not load founder panel: {exc}")
    else:
        add_warning("no_founder_panel", "No founder VCF/PLINK was provided; all founders will be initialized as mutable synthetic haplotypes.")

    if getattr(args, "compact_evidence_cache_dir", ""):
        cache_dir = Path(args.compact_evidence_cache_dir)
        if not cache_dir.exists():
            add_warning("missing_evidence_cache_dir", f"Evidence cache directory does not exist yet: {cache_dir}")
        else:
            meta_files = sorted(cache_dir.rglob("*.json"))[:20]
            founder_summary = founder_summary or {}
            founder_summary["evidence_cache_json_files_seen"] = len(meta_files)

    generation_summary = None
    if "generation" in samples.columns:
        gen = pd.to_numeric(samples["generation"], errors="coerce")
        generation_summary = {
            "min": None if gen.dropna().empty else float(gen.min()),
            "median": None if gen.dropna().empty else float(gen.median()),
            "max": None if gen.dropna().empty else float(gen.max()),
            "missing": int(gen.isna().sum()),
        }
    jax_report = _jax_device_report()
    if not bool(jax_report.get("has_accelerator", False)):
        add_warning(
            "jax_no_accelerator",
            "JAX is available only on CPU in this environment. A `--hmm-backend jax` run will be much slower than expected unless CUDA-enabled jaxlib is installed on the compute node.",
            jax=jax_report,
        )

    try:
        from .native_cpu import native_backend_status

        native_report = native_backend_status()
    except Exception as exc:  # pragma: no cover - defensive reporting only.
        native_report = {"pybind11_extension_available": False, "status_error": repr(exc)}
    if not bool(native_report.get("pybind11_extension_available", False)):
        add_warning(
            "native_cpp_backend_unavailable",
            "The install-time pybind11 C++ K8 backend is not importable. `--hmm-backend cpp` will fall back to the legacy ctypes build path if a compiler is available, or you can reinstall with `pip install -e .`.",
            native=native_report,
        )

    report = {
        "ok": not problems,
        "n_problems": len(problems),
        "n_warnings": len(warnings),
        "problems": problems,
        "warnings": warnings,
        "samples": {"n_samples": int(len(samples)), "columns": list(map(str, samples.columns))},
        "positions": {
            "n_positions": int(len(positions_df)),
            "chromosome": str(args.chromosome),
            "min_pos": None if pos.size == 0 else int(np.min(pos)),
            "max_pos": None if pos.size == 0 else int(np.max(pos)),
        },
        "generation": generation_summary,
        "founders": founder_summary,
        "jax": jax_report,
        "native_cpu": native_report,
    }
    if getattr(args, "output_json", ""):
        _write_json(Path(args.output_json), report)
    print(json.dumps(report, indent=2))


def cmd_run(args: argparse.Namespace) -> None:
    _apply_preset(args)
    _validate_ploidy_args(args)
    _enforce_jax_accelerator(args)
    samples = _read_table(args.samples)
    positions_df = load_positions(
        args.positions,
        args.chromosome,
        start=args.chr_start,
        end=args.chr_end,
    )
    founder_panel = None
    founder_cfg = FounderConfig()
    if args.founder_vcf:
        founder_cfg = FounderConfig(source_format="vcf", source_path=args.founder_vcf, immutable=args.founder_immutable)
    elif args.founder_plink:
        founder_cfg = FounderConfig(source_format="plink", source_path=args.founder_plink, immutable=args.founder_immutable)
    else:
        founder_panel = FounderPanel(
            chromosome=args.chromosome,
            positions=positions_df["POS"].to_numpy(dtype=np.int64),
            ref=positions_df["REF"].astype(str).to_numpy(),
            alt=positions_df["ALT"].astype(str).to_numpy(),
            alt_prob=np.full((args.n_founders, len(positions_df)), 0.5, dtype=np.float32),
            immutable_mask=np.zeros(args.n_founders, dtype=bool),
        )

    recombination_rate = float(args.recombination_rate_cm_per_mb)
    recombination_hotspot_window = _parse_bp_window(args.allow_recombination_hotspot_window)
    read_stream_backend = args.read_stream_backend
    min_base_quality = int(args.min_base_quality)
    min_mapping_quality = int(args.min_mapping_quality)
    max_insert_size = int(args.max_insert_size)
    cap_bq_by_mq = bool(args.cap_base_quality_by_mapping_quality)
    ref_alt_only = bool(args.ref_alt_only)
    merge_fragments_by_query = bool(args.merge_fragments_by_query)
    merge_unpaired_fragments_by_query = bool(args.merge_unpaired_fragments_by_query)
    use_bx_tag = bool(args.use_bx_tag)
    bx_tag = str(args.bx_tag)
    bx_tag_upper_limit = int(args.bx_tag_upper_limit)
    downsample_to_coverage = int(args.downsample_to_coverage)
    downsample_fraction = float(args.downsample_fraction)
    fragment_likelihood_mode = args.fragment_likelihood_mode
    fragment_max_diff_reads = float(args.fragment_max_diff_reads)
    fragment_max_emission_diff = float(args.fragment_max_emission_diff)
    store_xi = args.store_xi
    if args.write_xi is not None:
        store_xi = {"force": "full", "auto": "per-snp", "off": "False"}[args.write_xi]
    if args.stitch_compat:
        recombination_rate = 0.5
        min_base_quality = 17
        min_mapping_quality = 17
        max_insert_size = 600
        cap_bq_by_mq = True
        ref_alt_only = True
        merge_fragments_by_query = True
        merge_unpaired_fragments_by_query = True
        use_bx_tag = True
        bx_tag = "BX"
        bx_tag_upper_limit = 50000
        downsample_to_coverage = 50
        downsample_fraction = 1.0
        fragment_likelihood_mode = "replace"
        fragment_max_diff_reads = 1000.0
        fragment_max_emission_diff = 1e10

    config = PipelineConfig(
        chromosome=args.chromosome,
        positions_path=args.positions,
        output_dir=args.output_dir,
        n_founders=args.n_founders,
        chromosome_start=args.chr_start,
        chromosome_end=args.chr_end,
        ploidy_mode=("pseudo_haploid" if bool(args.pseudo_haploid_cpu) else str(args.hmm_mode)),
        ploidy=int(args.ploidy),
        ploidy_males=(None if args.ploidy_males is None else int(args.ploidy_males)),
        ploidy_females=(None if args.ploidy_females is None else int(args.ploidy_females)),
        block_size=args.block_size,
        max_mem=args.max_mem,
        io_window_size=args.io_window_size,
        snp_block_mode=args.snp_block_mode,
        exact_hmm_chunk_size=int(args.hmm_chunk_size),
        hmm_checkpoint_interval=int(args.hmm_checkpoint_interval),
        output_store=args.output_store,
        zarr_chunk_samples=int(args.zarr_chunk_samples),
        zarr_chunk_positions=int(args.zarr_chunk_positions),
        zarr_dosage_dtype=str(args.zarr_dosage_dtype),
        zarr_gp_dtype=str(args.zarr_gp_dtype),
        zarr_probability_scale=int(args.zarr_probability_scale),
        zarr_compressor=str(args.zarr_compressor),
        zarr_compression_level=int(args.zarr_compression_level),
        zarr_bitpack_support_mask=bool(args.zarr_bitpack_support_mask),
        memory_safety_fraction=args.memory_safety_fraction,
        allow_memory_overcommit=bool(getattr(args, "allow_memory_overcommit", False)),
        scratch_dir=(args.scratch_dir if args.scratch_dir else None),
        approx_overlap_fraction=args.approx_overlap_fraction,
        approx_min_overlap_snps=args.approx_min_overlap_snps,
        store_xi=store_xi,
        founder_informative_compression=bool(args.founder_informative_compression),
        pseudo_haploid_cpu=bool(args.pseudo_haploid_cpu),
        write_xi=args.write_xi,
        write_gamma=args.write_gamma,
        write_hmm_boundaries=args.write_hmm_boundaries,
        recombination_rate_cM_per_Mb=recombination_rate,
        genetic_map_path=(args.genetic_map if args.genetic_map else None),
        trainable_global_recombination_rate=bool(args.trainable_global_recombination_rate),
        allow_recombination_hotspot_window=int(recombination_hotspot_window),
        em_iterations=args.em_iterations,
        read_mode=args.read_mode,
        read_stream_backend=read_stream_backend,
        min_base_quality=min_base_quality,
        min_mapping_quality=min_mapping_quality,
        max_insert_size=max_insert_size,
        max_indel_len=int(args.max_indel_len),
        cap_base_quality_by_mapping_quality=cap_bq_by_mq,
        ref_alt_only=ref_alt_only,
        merge_fragments_by_query=merge_fragments_by_query,
        merge_unpaired_fragments_by_query=merge_unpaired_fragments_by_query,
        use_bx_tag=use_bx_tag,
        bx_tag=bx_tag,
        bx_tag_upper_limit=bx_tag_upper_limit,
        downsample_to_coverage=downsample_to_coverage,
        downsample_fraction=downsample_fraction,
        io_workers=args.io_workers,
        htslib_threads_per_file=args.htslib_threads_per_file,
        compact_evidence_cache_dir=(args.compact_evidence_cache_dir if args.compact_evidence_cache_dir else None),
        compact_evidence_cache_mode=args.compact_evidence_cache_mode,
        compact_evidence_cache_format=args.compact_evidence_cache_format,
        compact_evidence_cache_sample_batch_size=int(args.compact_evidence_cache_sample_batch_size),
        compact_evidence_materialize_dense_counts=bool(args.compact_evidence_materialize_dense_counts),
        compact_evidence_cache_include_dense_counts=bool(args.compact_evidence_cache_include_dense_counts),
        hmm_backend=args.hmm_backend,
        hmm_backend_autotune=(args.hmm_backend == "auto"),
        em_convergence_tol=args.em_convergence_tol,
        em_convergence_min_iterations=args.em_convergence_min_iterations,
        em_convergence_patience=args.em_convergence_patience,
        em_founder_update_damping=args.em_founder_update_damping,
        adaptive_em=args.adaptive_em,
        adaptive_em_restore_best_founders=args.adaptive_em_restore_best_founders,
        em_multistarts=args.em_multistarts,
        founder_update_hardening=args.founder_update_hardening,
        jax_sample_batch_size=args.jax_sample_batch_size,
        jax_bucket_batch_shapes=args.jax_bucket_batch_shapes,
        jax_count_emission_kernel=args.jax_count_emission_kernel,
        jax_fragment_emission_kernel=args.jax_fragment_emission_kernel,
        jax_persistent_cache_dir=(args.jax_persistent_cache_dir if args.jax_persistent_cache_dir else None),
        use_fragment_likelihood=True,
        fragment_likelihood_mode=fragment_likelihood_mode,
        fragment_coupling_model=args.fragment_coupling_model,
        fragment_max_difference_between_reads=fragment_max_diff_reads,
        fragment_max_emission_matrix_difference=fragment_max_emission_diff,
        fragment_rescale_read_likelihood=args.fragment_rescale_read_likelihood,
        write_transitions=args.write_transitions,
        transition_output=args.transition_output,
        transition_model=args.transition_model,
        transition_factor_rank=int(args.transition_factor_rank),
        transition_factor_regularization=float(args.transition_factor_regularization),
        transition_factor_max_deviation=float(args.transition_factor_max_deviation),
        transition_factor_train=bool(args.transition_factor_train),
        transition_factor_init_max_cells=int(args.transition_factor_init_max_cells),
        transition_sparse_top_k=int(args.transition_sparse_top_k),
        known_founder_fast_path=bool(args.known_founder_fast_path),
        output_minimal=bool(args.output_minimal),
        use_unordered_diploid_states=bool(args.use_unordered_diploid_states),
        sparse_emission_scan=bool(args.sparse_emission_scan),
        generation_bin_size=float(args.generation_bin_size),
        sample_shard_index=int(args.sample_shard_index),
        sample_shard_count=int(args.sample_shard_count),
        write_haplotype_probabilities=args.write_haplotype_probabilities,
        write_genotype_posteriors=args.write_genotype_posteriors,
        write_genotype_calls=args.write_genotype_calls,
        write_support_mask=args.write_support_mask,
        write_transition_summary=args.write_transition_summary,
        calibrate_genotype_posteriors=args.calibrate_genotype_posteriors,
        calibration_mode=args.calibration_mode,
        genotype_posterior_temperature=args.genotype_posterior_temperature,
        genotype_posterior_blend=args.genotype_posterior_blend,
        genotype_call_mode=args.genotype_call_mode,
        genotype_call_min_confidence=args.genotype_call_min_confidence,
        genotype_call_min_margin=args.genotype_call_min_margin,
        genotype_call_stitch_threshold=args.genotype_call_stitch_threshold,
        genotype_call_correctness_threshold=args.genotype_call_correctness_threshold,
        use_lightgbm_calibrator=args.use_lightgbm_calibrator,
        calibration_context_window=args.calibration_context_window,
        calibration_block_snps=args.calibration_block_snps,
        calibration_truth_source=args.calibration_truth_source,
        calibration_read_truth_min_depth=args.calibration_read_truth_min_depth,
        calibration_read_truth_min_hom_depth=args.calibration_read_truth_min_hom_depth,
        calibration_read_truth_min_het_depth=args.calibration_read_truth_min_het_depth,
        calibration_read_truth_min_het_allele_depth=args.calibration_read_truth_min_het_allele_depth,
        calibration_read_truth_hom_major_fraction=args.calibration_read_truth_hom_major_fraction,
        calibration_read_truth_het_balance_min=args.calibration_read_truth_het_balance_min,
        calibration_read_truth_het_balance_max=args.calibration_read_truth_het_balance_max,
        calibration_read_truth_max_other_fraction=args.calibration_read_truth_max_other_fraction,
        calibration_read_truth_holdout_fraction=args.calibration_read_truth_holdout_fraction,
        calibration_use_optuna=args.calibration_use_optuna,
        calibration_optuna_trials=args.calibration_optuna_trials,
        calibration_max_train_rows=args.calibration_max_train_rows,
        calibration_callability_model=args.calibration_callability_model,
        calibration_callability_min_train_rows=args.calibration_callability_min_train_rows,
        calibration_callability_min_call_rate=args.calibration_callability_min_call_rate,
        calibration_callability_call_rate_weight=args.calibration_callability_call_rate_weight,
        calibration_callability_decision_mode=args.calibration_callability_decision_mode,
        calibration_callability_bound_to_stitch=args.calibration_callability_bound_to_stitch,
        calibration_callability_min_call_rate_delta=args.calibration_callability_min_call_rate_delta,
        calibration_callability_max_call_rate_delta=args.calibration_callability_max_call_rate_delta,
        calibration_callability_validation_site_fraction=args.calibration_callability_validation_site_fraction,
        calibration_callability_min_objective_improvement=args.calibration_callability_min_objective_improvement,
        calibration_callability_max_hardcall_maf_shift=args.calibration_callability_max_hardcall_maf_shift,
        calibration_callability_max_hardcall_het_shift=args.calibration_callability_max_hardcall_het_shift,
        calibration_train_site_fraction=args.calibration_train_site_fraction,
        calibration_lightgbm_use_block_context=args.calibration_lightgbm_use_block_context,
        calibration_lightgbm_use_fixed_stage0=args.calibration_lightgbm_use_fixed_stage0,
        calibration_maf_bins=_parse_csv_floats(args.calibration_maf_bins),
        calibration_temperatures=_parse_csv_floats(args.calibration_temperatures),
        calibration_blends=_parse_csv_floats(args.calibration_blends),
        calibration_dosage_scales=_parse_csv_floats(args.calibration_dosage_scales),
        calibration_dosage_offsets=_parse_csv_floats(args.calibration_dosage_offsets),
        calibration_hwe_prior_weights=_parse_csv_floats(args.calibration_hwe_prior_weights),
        calibration_optimize_dosage_scale=args.calibration_optimize_dosage_scale,
        calibration_hwe_weight=args.calibration_hwe_weight,
        calibration_hwe_min_maf=args.calibration_hwe_min_maf,
        calibration_sanity_checks=args.calibration_sanity_checks,
        calibration_max_mean_abs_dosage_shift=args.calibration_max_mean_abs_dosage_shift,
        calibration_max_mean_abs_maf_shift=args.calibration_max_mean_abs_maf_shift,
        calibration_max_het_rate_shift=args.calibration_max_het_rate_shift,
        calibration_max_mean_entropy_shift=args.calibration_max_mean_entropy_shift,
        write_diagnostics=args.write_diagnostics,
        diagnostics_fail_on_error=args.diagnostics_fail_on_error,
        diagnostics_warn_het_rate=args.diagnostics_warn_het_rate,
        diagnostics_fail_het_rate=args.diagnostics_fail_het_rate,
        diagnostics_warn_hom_rate=args.diagnostics_warn_hom_rate,
        diagnostics_fail_hom_rate=args.diagnostics_fail_hom_rate,
        diagnostics_warn_missing_rate=args.diagnostics_warn_missing_rate,
        diagnostics_fail_missing_rate=args.diagnostics_fail_missing_rate,
        diagnostics_warn_low_info=args.diagnostics_warn_low_info,
        diagnostics_fail_low_info=args.diagnostics_fail_low_info,
        microarray_plink_path=(args.microarray_plink if args.microarray_plink else None),
        microarray_add_samples=args.microarray_add_samples,
        microarray_use_as_hmm_evidence=args.microarray_use_as_hmm_evidence,
        microarray_generation_default=(
            None if (args.microarray_generation_default is None or np.isnan(args.microarray_generation_default))
            else float(args.microarray_generation_default)
        ),
        microarray_hard_call_weight=args.microarray_hard_call_weight,
        random_seed=args.random_seed,
        founder_init_jitter=args.founder_init_jitter,
        executor=args.executor,
        dask_scheduler=args.dask_scheduler,
        dask_n_workers=args.dask_n_workers,
        dask_threads_per_worker=args.dask_threads_per_worker,
        dask_processes=args.dask_processes,
        dask_memory_limit=(args.dask_memory_limit if args.dask_memory_limit else None),
        dask_dashboard_address=(args.dask_dashboard_address if args.dask_dashboard_address else None),
        dask_performance_report=(args.dask_performance_report if args.dask_performance_report else None),
        dask_task_stream=(args.dask_task_stream if args.dask_task_stream else None),
        dask_dashboard_hold_seconds=args.dask_dashboard_hold_seconds,
        dask_target_task_memory_mb=args.dask_target_task_memory_mb,
        dask_min_block_size=args.dask_min_block_size,
        dask_min_sample_batch_size=args.dask_min_sample_batch_size,
        dask_sample_batch_size=args.dask_sample_batch_size,
        dask_jobqueue_class=args.dask_jobqueue_class,
        dask_jobqueue_queue=(args.dask_jobqueue_queue if args.dask_jobqueue_queue else None),
        dask_jobqueue_account=(args.dask_jobqueue_account if args.dask_jobqueue_account else None),
        dask_jobqueue_cores=args.dask_jobqueue_cores,
        dask_jobqueue_memory=(args.dask_jobqueue_memory if args.dask_jobqueue_memory else None),
        dask_jobqueue_walltime=(args.dask_jobqueue_walltime if args.dask_jobqueue_walltime else None),
        memory_map_read_matrices=args.memory_map_read_matrices,
        memory_map_dir=args.memory_map_dir if args.memory_map_dir else None,
        profile_memory=args.profile_memory,
        compression=args.compression,
        compression_level=args.compression_level,
        pedigree_mode=args.pedigree_mode,
        pedigree_strength=args.pedigree_strength,
        pedigree_offspring_col=args.pedigree_offspring_col,
        pedigree_parent1_col=args.pedigree_parent1_col,
        pedigree_parent2_col=args.pedigree_parent2_col,
        pedigree_iterations=args.pedigree_iterations,
        pedigree_kinship_threshold=args.pedigree_kinship_threshold,
        founder=founder_cfg,
    )
    pipeline = StitchPipeline(config)
    pedigree = _read_table(args.pedigree) if args.pedigree else None
    start = time.perf_counter()
    rss0 = _peak_rss_mb()
    pipeline.prepare_inputs(samples, pedigree=pedigree, founder_panel=founder_panel)
    elapsed = time.perf_counter() - start
    rss1 = _peak_rss_mb()
    summary = {
        "command": "run",
        "output_dir": str(Path(args.output_dir).resolve()),
        "elapsed_seconds": elapsed,
        "peak_rss_mb": max(rss0, rss1),
        "n_samples": int(len(samples)),
        "n_positions": int(len(positions_df)),
        "chr_start": (None if args.chr_start is None else int(args.chr_start)),
        "chr_end": (None if args.chr_end is None else int(args.chr_end)),
        "hmm_mode": str(args.hmm_mode),
        "ploidy": int(args.ploidy),
        "ploidy_males": (None if args.ploidy_males is None else int(args.ploidy_males)),
        "ploidy_females": (None if args.ploidy_females is None else int(args.ploidy_females)),
        "founder_source": ("uniform" if founder_panel is not None else founder_cfg.source_format),
        "jax": _jax_device_report(),
        "executor": args.executor,
        "snp_block_mode": args.snp_block_mode,
        "hmm_chunk_size": int(args.hmm_chunk_size),
        "hmm_checkpoint_interval": int(args.hmm_checkpoint_interval),
        "output_store": args.output_store,
        "zarr_chunk_samples": int(args.zarr_chunk_samples),
        "zarr_chunk_positions": int(args.zarr_chunk_positions),
        "zarr_dosage_dtype": str(args.zarr_dosage_dtype),
        "zarr_gp_dtype": str(args.zarr_gp_dtype),
        "zarr_probability_scale": int(args.zarr_probability_scale),
        "zarr_compressor": str(args.zarr_compressor),
        "zarr_compression_level": int(args.zarr_compression_level),
        "zarr_bitpack_support_mask": bool(args.zarr_bitpack_support_mask),
        "block_size": int(args.block_size),
        "max_mem": args.max_mem,
        "memory_safety_fraction": float(args.memory_safety_fraction),
        "io_window_size": int(args.io_window_size),
        "scratch_dir": (args.scratch_dir if args.scratch_dir else None),
        "approx_overlap_fraction": float(args.approx_overlap_fraction),
        "approx_min_overlap_snps": int(args.approx_min_overlap_snps),
        "store_xi": store_xi,
        "write_xi_legacy": args.write_xi,
        "write_gamma": args.write_gamma,
        "write_hmm_boundaries": bool(args.write_hmm_boundaries),
        "write_transition_summary": bool(args.write_transition_summary),
        "stitch_compat": bool(args.stitch_compat),
        "recombination_rate_cM_per_Mb": float(recombination_rate),
        "genetic_map": (args.genetic_map if args.genetic_map else None),
        "trainable_global_recombination_rate": bool(args.trainable_global_recombination_rate),
        "allow_recombination_hotspot_window": int(recombination_hotspot_window),
        "read_stream_backend": read_stream_backend,
        "compact_evidence_cache_dir": (args.compact_evidence_cache_dir if args.compact_evidence_cache_dir else None),
        "compact_evidence_cache_mode": args.compact_evidence_cache_mode,
        "compact_evidence_cache_format": args.compact_evidence_cache_format,
        "compact_evidence_cache_sample_batch_size": int(args.compact_evidence_cache_sample_batch_size),
        "compact_evidence_materialize_dense_counts": bool(args.compact_evidence_materialize_dense_counts),
        "compact_evidence_cache_include_dense_counts": bool(args.compact_evidence_cache_include_dense_counts),
        "min_base_quality": int(min_base_quality),
        "min_mapping_quality": int(min_mapping_quality),
        "max_insert_size": int(max_insert_size),
        "max_indel_len": int(args.max_indel_len),
        "cap_base_quality_by_mapping_quality": bool(cap_bq_by_mq),
        "ref_alt_only": bool(ref_alt_only),
        "merge_fragments_by_query": bool(merge_fragments_by_query),
        "merge_unpaired_fragments_by_query": bool(merge_unpaired_fragments_by_query),
        "use_bx_tag": bool(use_bx_tag),
        "bx_tag": str(bx_tag),
        "bx_tag_upper_limit": int(bx_tag_upper_limit),
        "downsample_to_coverage": int(downsample_to_coverage),
        "downsample_fraction": float(downsample_fraction),
        "fragment_likelihood_mode": fragment_likelihood_mode,
        "fragment_max_difference_between_reads": float(fragment_max_diff_reads),
        "fragment_max_emission_matrix_difference": float(fragment_max_emission_diff),
        "transition_output": args.transition_output,
        "transition_model": args.transition_model,
        "transition_factor_rank": int(args.transition_factor_rank),
        "transition_factor_regularization": float(args.transition_factor_regularization),
        "transition_factor_max_deviation": float(args.transition_factor_max_deviation),
        "transition_factor_train": bool(args.transition_factor_train),
        "transition_factor_init_max_cells": int(args.transition_factor_init_max_cells),
        "transition_sparse_top_k": int(args.transition_sparse_top_k),
        "known_founder_fast_path": bool(args.known_founder_fast_path),
        "output_minimal": bool(args.output_minimal),
        "use_unordered_diploid_states": bool(args.use_unordered_diploid_states),
        "sparse_emission_scan": bool(args.sparse_emission_scan),
        "generation_bin_size": float(args.generation_bin_size),
        "sample_shard_index": int(args.sample_shard_index),
        "sample_shard_count": int(args.sample_shard_count),
        "em_iterations_requested": int(args.em_iterations),
        "em_convergence_tol": float(args.em_convergence_tol),
        "em_convergence_min_iterations": int(args.em_convergence_min_iterations),
        "em_convergence_patience": int(args.em_convergence_patience),
        "em_founder_update_damping": float(args.em_founder_update_damping),
        "adaptive_em": bool(args.adaptive_em),
        "adaptive_em_restore_best_founders": bool(args.adaptive_em_restore_best_founders),
        "em_multistarts": int(args.em_multistarts),
        "jax_sample_batch_size": int(args.jax_sample_batch_size),
        "jax_bucket_batch_shapes": bool(args.jax_bucket_batch_shapes),
        "jax_count_emission_kernel": bool(args.jax_count_emission_kernel),
        "jax_fragment_emission_kernel": bool(args.jax_fragment_emission_kernel),
        "jax_persistent_cache_dir": (args.jax_persistent_cache_dir if args.jax_persistent_cache_dir else None),
        "calibration_sanity_checks": bool(args.calibration_sanity_checks),
        "calibration_sanity_thresholds": {
            "max_mean_abs_dosage_shift": float(args.calibration_max_mean_abs_dosage_shift),
            "max_mean_abs_maf_shift": float(args.calibration_max_mean_abs_maf_shift),
            "max_het_rate_shift": float(args.calibration_max_het_rate_shift),
            "max_mean_entropy_shift": float(args.calibration_max_mean_entropy_shift),
        },
        "diagnostics": {
            "write": bool(args.write_diagnostics),
            "fail_on_error": bool(args.diagnostics_fail_on_error),
            "warn_het_rate": float(args.diagnostics_warn_het_rate),
            "fail_het_rate": float(args.diagnostics_fail_het_rate),
            "warn_hom_rate": float(args.diagnostics_warn_hom_rate),
            "fail_hom_rate": float(args.diagnostics_fail_hom_rate),
            "warn_missing_rate": float(args.diagnostics_warn_missing_rate),
            "fail_missing_rate": float(args.diagnostics_fail_missing_rate),
            "warn_low_info": float(args.diagnostics_warn_low_info),
            "fail_low_info": float(args.diagnostics_fail_low_info),
        },
        "dask": {
            "scheduler": args.dask_scheduler,
            "n_workers": int(args.dask_n_workers),
            "threads_per_worker": int(args.dask_threads_per_worker),
            "processes": bool(args.dask_processes),
            "memory_limit": (args.dask_memory_limit if args.dask_memory_limit else None),
            "dashboard_address": (args.dask_dashboard_address if args.dask_dashboard_address else None),
            "performance_report": (args.dask_performance_report if args.dask_performance_report else None),
            "task_stream": (args.dask_task_stream if args.dask_task_stream else None),
            "dashboard_hold_seconds": float(args.dask_dashboard_hold_seconds),
            "target_task_memory_mb": float(args.dask_target_task_memory_mb),
            "sample_batch_size": int(args.dask_sample_batch_size),
            "jobqueue_class": args.dask_jobqueue_class,
            "jobqueue_queue": (args.dask_jobqueue_queue if args.dask_jobqueue_queue else None),
            "jobqueue_account": (args.dask_jobqueue_account if args.dask_jobqueue_account else None),
            "jobqueue_cores": int(args.dask_jobqueue_cores),
            "jobqueue_memory": (args.dask_jobqueue_memory if args.dask_jobqueue_memory else None),
            "jobqueue_walltime": (args.dask_jobqueue_walltime if args.dask_jobqueue_walltime else None),
        },
        "profile_memory": bool(args.profile_memory),
        "write_support_mask": bool(args.write_support_mask),
        "microarray_plink": (args.microarray_plink if args.microarray_plink else None),
        "pedigree": {
            "path": (args.pedigree if args.pedigree else None),
            "mode": args.pedigree_mode,
            "strength": float(args.pedigree_strength),
            "offspring_col": args.pedigree_offspring_col,
            "parent1_col": args.pedigree_parent1_col,
            "parent2_col": args.pedigree_parent2_col,
            "iterations": int(args.pedigree_iterations),
            "kinship_threshold": float(args.pedigree_kinship_threshold),
        },
    }
    pedigree_summary_path = Path(args.output_dir) / "pedigree_summary.json"
    if pedigree_summary_path.exists():
        summary["pedigree"]["summary"] = json.loads(pedigree_summary_path.read_text(encoding="utf-8"))
    dask_run_summary_path = Path(args.output_dir) / "dask_run_summary.json"
    if dask_run_summary_path.exists():
        summary["dask_runtime"] = json.loads(dask_run_summary_path.read_text(encoding="utf-8"))
    diagnostics_summary_path = Path(args.output_dir) / "diagnostics_summary.json"
    if diagnostics_summary_path.exists():
        summary["diagnostics_summary"] = json.loads(diagnostics_summary_path.read_text(encoding="utf-8"))
    _write_json(Path(args.output_dir) / "cli_run_summary.json", summary)
    _write_json(Path(args.output_dir) / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))



def _stitch_compat_read_settings(args: argparse.Namespace) -> dict[str, object]:
    settings = {
        "recombination_rate_cM_per_Mb": float(args.recombination_rate_cm_per_mb),
        "read_stream_backend": args.read_stream_backend,
        "min_base_quality": int(args.min_base_quality),
        "min_mapping_quality": int(args.min_mapping_quality),
        "max_insert_size": int(args.max_insert_size),
        "cap_base_quality_by_mapping_quality": bool(args.cap_base_quality_by_mapping_quality),
        "ref_alt_only": bool(args.ref_alt_only),
        "merge_fragments_by_query": bool(args.merge_fragments_by_query),
        "merge_unpaired_fragments_by_query": bool(args.merge_unpaired_fragments_by_query),
        "use_bx_tag": bool(args.use_bx_tag),
        "bx_tag": str(args.bx_tag),
        "bx_tag_upper_limit": int(args.bx_tag_upper_limit),
        "downsample_to_coverage": int(args.downsample_to_coverage),
        "downsample_fraction": float(args.downsample_fraction),
        "fragment_likelihood_mode": args.fragment_likelihood_mode,
        "fragment_max_difference_between_reads": float(args.fragment_max_diff_reads),
        "fragment_max_emission_matrix_difference": float(args.fragment_max_emission_diff),
    }
    if bool(args.stitch_compat):
        settings.update(
            {
                "recombination_rate_cM_per_Mb": 0.5,
                "min_base_quality": 17,
                "min_mapping_quality": 17,
                "max_insert_size": 600,
                "cap_base_quality_by_mapping_quality": True,
                "ref_alt_only": True,
                "merge_fragments_by_query": True,
                "merge_unpaired_fragments_by_query": True,
                "use_bx_tag": True,
                "bx_tag": "BX",
                "bx_tag_upper_limit": 50000,
                "downsample_to_coverage": 50,
                "downsample_fraction": 1.0,
                "fragment_likelihood_mode": "replace",
                "fragment_max_difference_between_reads": 1000.0,
                "fragment_max_emission_matrix_difference": 1e10,
            }
        )
    return settings


def cmd_build_evidence(args: argparse.Namespace) -> None:
    _apply_preset(args)
    _validate_ploidy_args(args)
    if not args.compact_evidence_cache_dir:
        raise ValueError("evidence-build requires --compact-evidence-cache-dir.")
    if args.compact_evidence_cache_mode == "off":
        args.compact_evidence_cache_mode = "write"
    settings = _stitch_compat_read_settings(args)
    samples = _read_table(args.samples)
    config = PipelineConfig(
        chromosome=args.chromosome,
        positions_path=args.positions,
        output_dir=args.output_dir,
        n_founders=int(getattr(args, "n_founders", 8) or 8),
        chromosome_start=args.chr_start,
        chromosome_end=args.chr_end,
        ploidy_mode=("pseudo_haploid" if bool(args.pseudo_haploid_cpu) else str(args.hmm_mode)),
        ploidy=int(args.ploidy),
        ploidy_males=(None if args.ploidy_males is None else int(args.ploidy_males)),
        ploidy_females=(None if args.ploidy_females is None else int(args.ploidy_females)),
        block_size=args.block_size,
        max_mem=args.max_mem,
        io_window_size=args.io_window_size,
        snp_block_mode=args.snp_block_mode,
        exact_hmm_chunk_size=int(args.hmm_chunk_size),
        hmm_checkpoint_interval=int(args.hmm_checkpoint_interval),
        output_store=args.output_store,
        zarr_chunk_samples=int(args.zarr_chunk_samples),
        zarr_chunk_positions=int(args.zarr_chunk_positions),
        zarr_dosage_dtype=str(args.zarr_dosage_dtype),
        zarr_gp_dtype=str(args.zarr_gp_dtype),
        zarr_probability_scale=int(args.zarr_probability_scale),
        zarr_compressor=str(args.zarr_compressor),
        zarr_compression_level=int(args.zarr_compression_level),
        zarr_bitpack_support_mask=bool(args.zarr_bitpack_support_mask),
        memory_safety_fraction=args.memory_safety_fraction,
        allow_memory_overcommit=bool(getattr(args, "allow_memory_overcommit", False)),
        scratch_dir=(args.scratch_dir if args.scratch_dir else None),
        approx_overlap_fraction=args.approx_overlap_fraction,
        approx_min_overlap_snps=args.approx_min_overlap_snps,
        genetic_map_path=(args.genetic_map if args.genetic_map else None),
        read_mode=args.read_mode,
        read_stream_backend=str(settings["read_stream_backend"]),
        min_base_quality=int(settings["min_base_quality"]),
        min_mapping_quality=int(settings["min_mapping_quality"]),
        max_insert_size=int(settings["max_insert_size"]),
        max_indel_len=int(args.max_indel_len),
        cap_base_quality_by_mapping_quality=bool(settings["cap_base_quality_by_mapping_quality"]),
        ref_alt_only=bool(settings["ref_alt_only"]),
        merge_fragments_by_query=bool(settings["merge_fragments_by_query"]),
        merge_unpaired_fragments_by_query=bool(settings["merge_unpaired_fragments_by_query"]),
        use_bx_tag=bool(settings["use_bx_tag"]),
        bx_tag=str(settings["bx_tag"]),
        bx_tag_upper_limit=int(settings["bx_tag_upper_limit"]),
        downsample_to_coverage=int(settings["downsample_to_coverage"]),
        downsample_fraction=float(settings["downsample_fraction"]),
        io_workers=args.io_workers,
        htslib_threads_per_file=args.htslib_threads_per_file,
        compact_evidence_cache_dir=args.compact_evidence_cache_dir,
        compact_evidence_cache_mode=args.compact_evidence_cache_mode,
        compact_evidence_cache_format=args.compact_evidence_cache_format,
        compact_evidence_cache_sample_batch_size=int(args.compact_evidence_cache_sample_batch_size),
        compact_evidence_materialize_dense_counts=bool(args.compact_evidence_materialize_dense_counts),
        compact_evidence_cache_include_dense_counts=bool(args.compact_evidence_cache_include_dense_counts),
        sample_shard_index=int(args.sample_shard_index),
        sample_shard_count=int(args.sample_shard_count),
        compression=args.compression,
        compression_level=args.compression_level,
        random_seed=args.random_seed,
        memory_map_read_matrices=args.memory_map_read_matrices,
        memory_map_dir=args.memory_map_dir if args.memory_map_dir else None,
        profile_memory=args.profile_memory,
        founder=FounderConfig(),
    )
    summary = StitchPipeline(config).build_evidence_cache(samples)
    print(json.dumps(summary, indent=2))


def cmd_impute_from_evidence(args: argparse.Namespace) -> None:
    if not args.compact_evidence_cache_dir:
        raise ValueError("impute-from-evidence requires --compact-evidence-cache-dir.")
    args.compact_evidence_cache_mode = "read"
    cmd_run(args)


def cmd_validate_run(args: argparse.Namespace) -> None:
    summary = validate_run_against_plink(
        run_output_dir=args.run_output_dir,
        truth_plink=args.truth_plink,
        output_dir=args.output_dir,
        chromosome=(args.chromosome if args.chromosome else None),
        min_truth_nonmissing=int(args.min_truth_nonmissing),
        allele_orientation_audit=not bool(args.no_allele_orientation_audit),
        fail_on_allele_mismatch=bool(args.fail_on_allele_mismatch),
    )
    print(json.dumps(summary, indent=2))


def cmd_tune_jax_memory(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = pd.read_parquet(data_dir / "samples.parquet")
    positions = pd.read_parquet(data_dir / "positions.parquet")
    positions = positions.loc[positions["CHR"].astype(str) == str(args.chromosome)].copy()
    positions = positions.sort_values("POS").reset_index(drop=True)
    if args.max_samples > 0 and len(samples) > args.max_samples:
        samples = samples.iloc[: args.max_samples].reset_index(drop=True)
    if args.max_positions > 0 and len(positions) > args.max_positions:
        positions = positions.iloc[: args.max_positions].reset_index(drop=True)
    positions_path = output_dir / "positions.parquet"
    positions.to_parquet(positions_path, index=False)
    founder_panel = FounderPanel(
        chromosome=args.chromosome,
        positions=positions["POS"].to_numpy(dtype=np.int64),
        ref=positions["REF"].astype(str).to_numpy(),
        alt=positions["ALT"].astype(str).to_numpy(),
        alt_prob=np.full((args.n_founders, len(positions)), 0.5, dtype=np.float32),
        immutable_mask=np.zeros(args.n_founders, dtype=bool),
    )

    runs: list[dict[str, object]] = []
    for block_size in _parse_csv_ints(args.block_sizes):
        for memmap_mode in _parse_csv_strings(args.memory_map_modes):
            for jax_batch_size in _parse_csv_ints(args.jax_sample_batch_sizes):
                use_memmap = memmap_mode.lower() in {"1", "true", "on", "yes"}
                trial_dir = output_dir / (
                    f"block={block_size}_memmap={'on' if use_memmap else 'off'}_jax_batch={jax_batch_size}"
                )
                if trial_dir.exists():
                    shutil.rmtree(trial_dir)
                config = PipelineConfig(
                    chromosome=args.chromosome,
                    positions_path=positions_path,
                    output_dir=trial_dir,
                    n_founders=args.n_founders,
                    block_size=block_size,
                    em_iterations=args.em_iterations,
                    read_mode="read_stream",
                    read_stream_backend=args.read_stream_backend,
                    io_workers=args.io_workers,
                    htslib_threads_per_file=args.htslib_threads_per_file,
                    hmm_backend="jax",
                    jax_sample_batch_size=jax_batch_size,
                    use_fragment_likelihood=True,
                    fragment_likelihood_mode=args.fragment_likelihood_mode,
                    write_transitions=False,
                    write_haplotype_probabilities=False,
                    memory_map_read_matrices=use_memmap,
                    memory_map_dir=(args.memory_map_dir if args.memory_map_dir else None),
                    profile_memory=True,
                )
                pipeline = StitchPipeline(config)
                start = time.perf_counter()
                rss0 = _peak_rss_mb()
                pipeline.prepare_inputs(samples, founder_panel=founder_panel)
                elapsed = time.perf_counter() - start
                rss1 = _peak_rss_mb()
                timings_path = trial_dir / "stage_timings.json"
                rows = json.loads(timings_path.read_text(encoding="utf-8")) if timings_path.exists() else []
                read_s = float(sum(float(r.get("seconds_read_extract", 0.0)) for r in rows))
                hmm_s = float(sum(float(r.get("seconds_hmm", 0.0)) for r in rows))
                write_s = float(sum(float(r.get("seconds_write", 0.0)) for r in rows))
                peak_stage_rss = (
                    max(float(r.get("rss_mb_after_write", 0.0)) for r in rows) if rows else max(rss0, rss1)
                )
                runs.append(
                    {
                        "block_size": block_size,
                        "memory_map_read_matrices": use_memmap,
                        "jax_sample_batch_size": jax_batch_size,
                        "elapsed_seconds": elapsed,
                        "peak_rss_mb": max(rss0, rss1),
                        "peak_stage_rss_mb": peak_stage_rss,
                        "seconds_read_extract": read_s,
                        "seconds_hmm": hmm_s,
                        "seconds_write": write_s,
                        "output_dir": str(trial_dir),
                    }
                )

    eligible = runs
    if args.max_peak_rss_mb > 0:
        eligible = [r for r in runs if float(r["peak_rss_mb"]) <= args.max_peak_rss_mb]
    best = min(eligible, key=lambda r: float(r["elapsed_seconds"])) if eligible else None
    summary = {
        "command": "tune-jax-memory",
        "config": {
            "chromosome": args.chromosome,
            "n_founders": args.n_founders,
            "em_iterations": args.em_iterations,
            "read_stream_backend": args.read_stream_backend,
            "n_samples": int(len(samples)),
            "n_positions": int(len(positions)),
            "max_peak_rss_mb": args.max_peak_rss_mb,
            "jax_sample_batch_sizes": _parse_csv_ints(args.jax_sample_batch_sizes),
        },
        "runs": runs,
        "best": best,
    }
    _write_json(output_dir / "tune_jax_memory_summary.json", summary)
    print(json.dumps(summary, indent=2))


def cmd_combine(args: argparse.Namespace) -> None:
    run_output_dir = Path(args.run_output_dir)
    if args.input_dir and args.output_file:
        summary = combine_parquet_chunks(
            args.input_dir,
            args.output_file,
            compression=args.compression,
            compression_level=args.compression_level,
            row_group_size=args.row_group_size,
        )
        print(json.dumps(summary, indent=2))
        return
    datasets = DEFAULT_DATASETS if args.datasets == "all" else _parse_csv_strings(args.datasets)
    xds = combine_pipeline_outputs(
        run_output_dir,
        output_dir=args.output_dir,
        datasets=datasets,
        compression=args.compression,
        compression_level=args.compression_level,
        row_group_size=args.row_group_size,
    )
    summary = xds.attrs.get("combine_summary", {})
    summary["_xarray"] = {
        "n_variables": int(len(xds.data_vars)),
        "variables": sorted(list(xds.data_vars)),
    }
    out_path = Path(args.output_dir) if args.output_dir else run_output_dir / "combined"
    if args.write_zarr or args.zarr_output:
        zarr_path = Path(args.zarr_output) if args.zarr_output else out_path / "float_outputs.zarr"
        summary["_zarr"] = write_xarray_float_zarr(
            xds,
            zarr_path,
            consolidated=bool(args.zarr_consolidated),
        )
    _write_json(out_path / "combine_summary.json", summary)
    print(json.dumps(summary, indent=2))


def cmd_export_bcf(args: argparse.Namespace) -> None:
    print(
        "WARNING: BCF export is an interoperability path and WILL slow down I/O. "
        "Keep primary STITCHCONT results in Parquet/Zarr unless a downstream tool explicitly requires BCF.",
        flush=True,
    )
    summary = export_stitch_bcf_from_parquet(
        args.run_output_dir,
        args.output_bcf,
        chromosome=args.chromosome,
        include_haplotype_dosage=args.include_haplotype_dosage,
        tabix_index=args.tabix_index,
    )
    summary_path = Path(args.run_output_dir) / "export_bcf_summary.json"
    _write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


def cmd_reformat_stitch_filenames(args: argparse.Namespace) -> None:
    inputs: list[str] = []
    if args.input_file:
        lines = Path(args.input_file).read_text(encoding="utf-8").splitlines()
        inputs.extend([line.strip() for line in lines if line.strip()])
    if args.filenames:
        inputs.extend([x.strip() for x in str(args.filenames).split(",") if x.strip()])
    if not inputs:
        raise ValueError("No filenames provided. Use --input-file and/or --filenames.")
    out = reformat_stitch_filenames(inputs, output_path=args.output_file)
    payload = {
        "n_inputs": len(inputs),
        "output_file": str(Path(args.output_file).resolve()),
        "renamed": [{"original": src, "stitchcont": dst} for src, dst in zip(inputs, out, strict=False)],
    }
    print(json.dumps(payload, indent=2))


def cmd_pedigree_qc(args: argparse.Namespace) -> None:
    samples = _read_table(args.samples)
    pedigree = _read_table(args.pedigree) if args.pedigree else None
    result = run_pedigree_qc(
        samples=samples,
        pedigree=pedigree,
        output_dir=args.output_dir,
        run_output_dirs=args.run_output_dir or [],
        genotype_tables=args.genotype_table or [],
        sample_id_col=args.sample_id_col,
        offspring_col=args.pedigree_offspring_col,
        parent1_col=args.pedigree_parent1_col,
        parent2_col=args.pedigree_parent2_col,
        family_col=args.family_col,
        value_column=args.value_column,
        max_variants=args.max_variants,
        min_call_rate=args.min_call_rate,
        min_maf=args.min_maf,
        report_min_r=args.report_min_r,
        unrelated_max_r=args.unrelated_max_r,
        first_degree_min_r=args.first_degree_min_r,
        same_min_r=args.same_min_r,
        sample_block_size=args.sample_block_size,
        max_full_matrix_samples=args.max_full_matrix_samples,
        random_seed=args.random_seed,
        write_umap=bool(args.write_umap),
        umap_neighbors=args.umap_neighbors,
        umap_max_variants=args.umap_max_variants,
        unlink_calls=tuple(_parse_csv_strings(args.unlink_calls)),
    )
    print(json.dumps(result.summary, indent=2))


def cmd_discover_positions(args: argparse.Namespace) -> None:
    samples = _read_table(args.samples)
    frame, summary = discover_snp_positions(
        samples,
        reference_fasta=args.reference_fasta,
        chromosome=args.chromosome,
        chr_start=args.chr_start,
        chr_end=args.chr_end,
        bam_path_col=args.bam_path_col,
        max_bams=args.max_bams,
        window_size=args.window_size,
        min_base_quality=args.min_base_quality,
        min_mapping_quality=args.min_mapping_quality,
        htslib_threads_per_file=args.htslib_threads_per_file,
        max_insert_size=args.max_insert_size,
        variant_types=tuple(_parse_csv_strings(args.variant_types)),
        max_indel_len=args.max_indel_len,
        cap_base_quality_by_mapping_quality=bool(args.cap_base_quality_by_mapping_quality),
        min_depth=args.min_depth,
        min_alt_count=args.min_alt_count,
        min_alt_samples=args.min_alt_samples,
        min_alt_fraction=args.min_alt_fraction,
        max_other_fraction=args.max_other_fraction,
    )
    write_discovered_positions(frame, args.output_file, compression=args.compression)
    summary_path = Path(args.summary_file) if args.summary_file else Path(args.output_file).with_suffix(
        Path(args.output_file).suffix + ".summary.json"
    )
    summary["output_file"] = str(Path(args.output_file).resolve())
    summary["summary_file"] = str(summary_path.resolve())
    write_discovery_summary(summary, summary_path)
    print(json.dumps(summary, indent=2))


def cmd_cv(args: argparse.Namespace) -> None:
    _apply_preset(args)
    _validate_ploidy_args(args)
    samples = _read_table(args.samples)
    positions_df = load_positions(
        args.positions,
        args.chromosome,
        start=args.chr_start,
        end=args.chr_end,
    )
    pseudo_truth = _read_table(args.pseudo_truth) if args.pseudo_truth else None
    founder_panel = None
    if args.founder_vcf:
        founder_panel = load_founders(
            source_format="vcf",
            source_path=args.founder_vcf,
            chromosome=args.chromosome,
            positions_df=positions_df,
            immutable=args.founder_immutable,
        )
    elif args.founder_plink:
        founder_panel = load_founders(
            source_format="plink",
            source_path=args.founder_plink,
            chromosome=args.chromosome,
            positions_df=positions_df,
            immutable=args.founder_immutable,
        )
    summary = run_realdata_cv_harness(
        samples=samples,
        positions_df=positions_df,
        chromosome=args.chromosome,
        output_dir=args.output_dir,
        base_founder_panel=founder_panel,
        pseudo_truth=pseudo_truth,
        k_values=_parse_csv_ints(args.k_values),
        ngen_values=[float(x) for x in _parse_csv_strings(args.ngen_values)],
        s_values=_parse_csv_ints(args.s_values),
        seeds=_parse_csv_ints(args.seeds),
        n_folds=args.folds,
        holdout_fraction=args.holdout_fraction,
        block_size=args.block_size,
        hmm_backend=args.hmm_backend,
        read_mode=args.read_mode,
        read_stream_backend=args.read_stream_backend,
        io_workers=args.io_workers,
        htslib_threads_per_file=args.htslib_threads_per_file,
        use_fragment_likelihood=True,
        fragment_likelihood_mode=args.fragment_likelihood_mode,
        fragment_coupling_model=args.fragment_coupling_model,
        fragment_max_difference_between_reads=args.fragment_max_diff_reads,
        fragment_max_emission_matrix_difference=args.fragment_max_emission_diff,
        fragment_rescale_read_likelihood=args.fragment_rescale_read_likelihood,
        genotype_posterior_temperature=args.genotype_posterior_temperature,
        genotype_posterior_blend=args.genotype_posterior_blend,
        founder_init_jitter=args.founder_init_jitter,
        memory_map_read_matrices=args.memory_map_read_matrices,
        memory_map_dir=args.memory_map_dir if args.memory_map_dir else None,
        lightgbm_post_calibrator=args.lightgbm_post_calibrator,
    )
    _write_json(Path(args.output_dir) / "cv_cli_summary.json", summary)
    print(json.dumps(summary, indent=2))




def cmd_recombination_diagnostics(args: argparse.Namespace) -> None:
    samples = None
    generation_values = None
    if args.samples:
        samples = _read_table(args.samples)
        if "generation" in samples.columns:
            generation_values = samples["generation"].to_numpy(dtype=float)
    summary = compute_recombination_diagnostics(
        positions_path=args.positions,
        output_dir=args.output_dir,
        chromosome=args.chromosome,
        genetic_map_path=(args.genetic_map if args.genetic_map else None),
        recombination_rate_cM_per_Mb=float(args.recombination_rate_cM_per_Mb),
        generation_values=generation_values,
        generation_bin_size=float(args.generation_bin_size),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))

def cmd_autotune(args: argparse.Namespace) -> None:
    plan = estimate_autotune_plan(
        samples_path=args.samples,
        positions_path=args.positions,
        n_founders=int(args.n_founders),
        target_memory=args.target_memory,
        target_memory_fraction=float(args.target_memory_fraction),
        write_gp=not bool(args.dosage_only),
        use_unordered_diploid_states=not bool(args.ordered_diploid_states),
        checkpoint_interval=int(args.hmm_checkpoint_interval),
        dosage_dtype=str(args.zarr_dosage_dtype),
        gp_dtype=str(args.zarr_gp_dtype),
        min_chunk_size=int(args.min_chunk_size),
        max_chunk_size=int(args.max_chunk_size),
        desired_sample_shards=int(args.sample_shard_count),
    )
    path = write_autotune_plan(args.output_dir, plan)
    print(json.dumps({"autotune_plan": str(path), **plan}, indent=2, sort_keys=True))

def cmd_make_slurm_shards(args: argparse.Namespace) -> None:
    result = write_slurm_sample_shards(
        output_dir=args.output_dir,
        n_shards=int(args.n_shards),
        samples=args.samples,
        positions=args.positions,
        chromosome=args.chromosome,
        founder_plink=args.founder_plink,
        evidence_cache_dir=args.compact_evidence_cache_dir,
        run_root=args.run_root,
        preset=args.preset,
        n_founders=int(args.n_founders),
        job_name=args.job_name,
        account=args.account,
        partition=args.partition,
        time_limit=args.time_limit,
        mem=args.mem,
        cpus_per_task=int(args.cpus_per_task),
        extra_args=args.extra_args,
    )
    print(json.dumps(result, indent=2), flush=True)


def cmd_merge_shards(args: argparse.Namespace) -> None:
    result = merge_sharded_matrix_zarr(input_root=args.input_root, output_dir=args.output_dir)
    print(json.dumps(result, indent=2), flush=True)



def cmd_evidence_shard_layout(args: argparse.Namespace) -> None:
    result = write_evidence_sample_shard_layout(
        output_dir=args.output_dir,
        samples_path=args.samples,
        evidence_cache_dir=args.compact_evidence_cache_dir,
        n_shards=int(args.n_shards),
    )
    print(json.dumps(result, indent=2), flush=True)


def cmd_pedigree_postprocess(args: argparse.Namespace) -> None:
    result = run_pedigree_postprocess_zarr(
        matrices_zarr=args.matrices_zarr,
        samples=args.samples,
        pedigree=(args.pedigree if args.pedigree else None),
        output_dir=args.output_dir,
        strength=float(args.strength),
        chunk_positions=int(args.chunk_positions),
        offspring_col=args.pedigree_offspring_col,
        parent1_col=args.pedigree_parent1_col,
        parent2_col=args.pedigree_parent2_col,
    )
    print(json.dumps(result, indent=2), flush=True)

def cmd_benchmark(args: argparse.Namespace) -> None:
    result = run_synthetic_benchmark(
        output_dir=args.output_dir,
        n_samples=int(args.n_samples),
        n_positions=int(args.n_positions),
        n_founders=int(args.n_founders),
        chunk_size=int(args.hmm_chunk_size),
        repeat=int(args.repeat),
        seed=int(args.random_seed),
        include_jax=not bool(args.no_jax),
    )
    print(json.dumps(result, indent=2), flush=True)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stitchcont", description="STITCHCONT command line tools.")
    sub = parser.add_subparsers(dest="command", required=True)

    doctor_p = sub.add_parser("doctor", help="Preflight-check sample, position, founder, and optional evidence-cache inputs.")
    _add_common_run_args(doctor_p, require_n_founders=False)
    doctor_p.add_argument("--founder-vcf", default="")
    doctor_p.add_argument("--founder-plink", default="")
    doctor_p.add_argument("--founder-immutable", action="store_true")
    doctor_p.add_argument("--output-json", default="")
    doctor_p.set_defaults(func=cmd_doctor)

    run_p = sub.add_parser("run", help="Run STITCHCONT pipeline.")
    _add_common_run_args(run_p, require_n_founders=True)
    run_p.add_argument("--founder-vcf", default="")
    run_p.add_argument("--founder-plink", default="")
    run_p.add_argument("--founder-immutable", action="store_true")
    run_p.set_defaults(func=cmd_run)

    ev_p = sub.add_parser("evidence-build", help="Build/readwrite the compact read-evidence cache only; do not run the HMM.")
    _add_common_run_args(ev_p, require_n_founders=False)
    ev_p.set_defaults(func=cmd_build_evidence)

    imp_p = sub.add_parser("impute-from-evidence", help="Run the HMM using an existing compact evidence cache in strict read-only mode.")
    _add_common_run_args(imp_p, require_n_founders=True)
    imp_p.add_argument("--founder-vcf", default="")
    imp_p.add_argument("--founder-plink", default="")
    imp_p.add_argument("--founder-immutable", action="store_true")
    imp_p.set_defaults(func=cmd_impute_from_evidence)

    val_p = sub.add_parser("validate", help="Validate dosage output against a PLINK truth panel.")
    val_p.add_argument("--run-output-dir", required=True)
    val_p.add_argument("--truth-plink", required=True, help="PLINK prefix for validation genotypes.")
    val_p.add_argument("--output-dir", required=True)
    val_p.add_argument("--chromosome", default="", help="Optional chromosome if it cannot be inferred from dosage output.")
    val_p.add_argument("--min-truth-nonmissing", type=int, default=10)
    val_p.add_argument("--no-allele-orientation-audit", action="store_true", help="Disable REF/ALT vs PLINK A1/A2 orientation audit before r².")
    val_p.add_argument("--fail-on-allele-mismatch", action="store_true", help="Fail validation if incompatible REF/ALT mismatches are found.")
    val_p.set_defaults(func=cmd_validate_run)

    cv_p = sub.add_parser(
        "cv",
        help="Run 5-fold (or custom) real-data CV with K/nGen/S tuning, seed averaging, and optional LightGBM calibration.",
    )
    _add_common_run_args(cv_p, require_n_founders=False)
    cv_p.add_argument("--pseudo-truth", default="", help="Pseudo-truth table (parquet/csv) with sample_id, position, genotype or dosage_truth.")
    cv_p.add_argument("--k-values", default="8")
    cv_p.add_argument("--ngen-values", default="1.0")
    cv_p.add_argument("--s-values", default="2")
    cv_p.add_argument("--seeds", default="0,1,2")
    cv_p.add_argument("--folds", type=int, default=5)
    cv_p.add_argument("--holdout-fraction", type=float, default=0.2)
    cv_p.add_argument("--lightgbm-post-calibrator", action="store_true")
    cv_p.add_argument("--founder-vcf", default="")
    cv_p.add_argument("--founder-plink", default="")
    cv_p.add_argument("--founder-immutable", action="store_true")
    cv_p.set_defaults(func=cmd_cv)

    tune_p = sub.add_parser("tune-jax-memory", help="Profile block-size/memmap options for JAX memory/runtime.")
    tune_p.add_argument("--data-dir", required=True)
    tune_p.add_argument("--output-dir", required=True)
    tune_p.add_argument("--chromosome", required=True)
    tune_p.add_argument("--n-founders", type=int, default=8)
    tune_p.add_argument("--em-iterations", type=int, default=2)
    tune_p.add_argument("--block-sizes", default="500,1000,2000")
    tune_p.add_argument("--memory-map-modes", default="off,on")
    tune_p.add_argument("--jax-sample-batch-sizes", default="0,64,128")
    tune_p.add_argument(
        "--read-stream-backend",
        choices=["auto", "python", "htslib", "snp_only_bamreader", "stitch_style_bamreader", "variant_aware_bamreader"],
        default="auto",
    )
    tune_p.add_argument("--io-workers", type=int, default=1)
    tune_p.add_argument("--htslib-threads-per-file", type=int, default=1)
    tune_p.add_argument("--fragment-likelihood-mode", choices=["replace", "augment"], default="replace")
    tune_p.add_argument("--memory-map-dir", default="")
    tune_p.add_argument("--max-samples", type=int, default=0)
    tune_p.add_argument("--max-positions", type=int, default=0)
    tune_p.add_argument("--max-peak-rss-mb", type=float, default=0.0)
    tune_p.set_defaults(func=cmd_tune_jax_memory)

    combine_p = sub.add_parser("combine", help="Combine partitioned parquet chunks.")
    combine_p.add_argument("--run-output-dir", required=True)
    combine_p.add_argument("--datasets", default="all", help="Comma-separated dataset names or 'all'.")
    combine_p.add_argument("--output-dir", default="")
    combine_p.add_argument("--input-dir", default="", help="Optional explicit input parquet dir for one-off combine.")
    combine_p.add_argument("--output-file", default="", help="Optional explicit output parquet file for one-off combine.")
    combine_p.add_argument("--compression", default="zstd")
    combine_p.add_argument("--compression-level", type=int, default=6)
    combine_p.add_argument("--row-group-size", type=int, default=262_144)
    combine_p.add_argument(
        "--write-zarr",
        action="store_true",
        help="Also write floating dosage/posterior outputs as an xarray Zarr store.",
    )
    combine_p.add_argument("--zarr-output", default="", help="Optional output path for --write-zarr.")
    combine_p.add_argument("--zarr-consolidated", action="store_true", help="Request consolidated Zarr metadata.")
    combine_p.set_defaults(func=cmd_combine)

    bcf_p = sub.add_parser(
        "export-bcf",
        help="Explicitly export run Parquet/Zarr-oriented outputs to BCF; slower than native Parquet/Zarr output.",
    )
    bcf_p.add_argument("--run-output-dir", required=True)
    bcf_p.add_argument("--output-bcf", required=True)
    bcf_p.add_argument("--chromosome", required=True)
    bcf_p.add_argument("--include-haplotype-dosage", action="store_true")
    bcf_p.add_argument("--no-tabix-index", dest="tabix_index", action="store_false")
    bcf_p.set_defaults(tabix_index=True)
    bcf_p.set_defaults(func=cmd_export_bcf)

    remap_p = sub.add_parser(
        "reformat-stitch-filenames",
        help="Reformat STITCH-oriented filenames into STITCHCONT naming and save mapping.",
    )
    remap_p.add_argument("--input-file", default="", help="Optional text file with one filename per line.")
    remap_p.add_argument("--filenames", default="", help="Optional comma-separated filename list.")
    remap_p.add_argument("--output-file", required=True, help="Path to write JSON mapping.")
    remap_p.set_defaults(func=cmd_reformat_stitch_filenames)

    discover_p = sub.add_parser(
        "discover-positions",
        help=(
            "Pre-pass that discovers candidate SNP/indel positions from BAM/CRAM reads "
            "using CIGAR alignment and a reference FASTA. This writes a reviewable positions table "
            "and does not run the STITCHCONT HMM."
        ),
    )
    discover_p.add_argument("--samples", required=True, help="Samples table with a BAM/CRAM path column.")
    discover_p.add_argument("--reference-fasta", required=True, help="Indexed reference FASTA. Run samtools faidx if needed.")
    discover_p.add_argument("--chromosome", required=True)
    discover_p.add_argument("--chr-start", type=int, default=None, help="1-based inclusive discovery start. Defaults to chromosome start.")
    discover_p.add_argument("--chr-end", type=int, default=None, help="1-based inclusive discovery end. Defaults to chromosome end.")
    discover_p.add_argument("--output-file", required=True, help="Output positions table: .parquet, .csv, .tsv, or .txt.")
    discover_p.add_argument("--summary-file", default="", help="Optional JSON summary path.")
    discover_p.add_argument("--bam-path-col", default="bam_path")
    discover_p.add_argument("--max-bams", type=int, default=0, help="Optional cap for smoke tests; 0 scans all BAMs.")
    discover_p.add_argument("--window-size", type=int, default=1_000_000, help="Reference bases per native discovery chunk.")
    discover_p.add_argument("--min-base-quality", type=int, default=13)
    discover_p.add_argument("--min-mapping-quality", type=int, default=20)
    discover_p.add_argument("--htslib-threads-per-file", type=int, default=1)
    discover_p.add_argument("--max-insert-size", type=int, default=0, help="Skip reads with absolute template length above this value; 0 disables.")
    discover_p.add_argument(
        "--variant-types",
        default="snp,ins,del",
        help="Comma-separated discovery types: snp,ins,del. Discovery output is review-only and is not trusted automatically by the HMM.",
    )
    discover_p.add_argument("--max-indel-len", type=int, default=50, help="Maximum insertion/deletion length emitted by discovery.")
    discover_p.add_argument("--cap-base-quality-by-mapping-quality", action="store_true")
    discover_p.add_argument("--min-depth", type=int, default=3)
    discover_p.add_argument("--min-alt-count", type=int, default=2)
    discover_p.add_argument("--min-alt-samples", type=int, default=1)
    discover_p.add_argument("--min-alt-fraction", type=float, default=0.05)
    discover_p.add_argument("--max-other-fraction", type=float, default=0.20)
    discover_p.add_argument("--compression", default="zstd", help="Parquet compression when --output-file is parquet.")
    discover_p.set_defaults(func=cmd_discover_positions)

    pedqc_p = sub.add_parser(
        "pedigree-qc",
        help=(
            "Curate a global pedigree from preliminary no-pedigree STITCHCONT calls or other hard calls. "
            "Uses R-based genotype similarity, writes corrected pedigree and UMAP before/after edge plots."
        ),
    )
    pedqc_p.add_argument("--samples", required=True, help="Samples table with sample_id and optional pedigree metadata.")
    pedqc_p.add_argument("--pedigree", default="", help="Optional pedigree table. If absent, pedigree columns are read from --samples.")
    pedqc_p.add_argument(
        "--run-output-dir",
        action="append",
        default=[],
        help="Preliminary STITCHCONT run directory containing genotype_calls/ or dosage/. Repeat for multiple chromosomes.",
    )
    pedqc_p.add_argument(
        "--genotype-table",
        action="append",
        default=[],
        help="Long parquet table/dir with sample_id, chromosome, position, and genotype_call or dosage. Repeatable.",
    )
    pedqc_p.add_argument("--output-dir", required=True)
    pedqc_p.add_argument("--sample-id-col", default="sample_id")
    pedqc_p.add_argument("--pedigree-offspring-col", default="sample_id")
    pedqc_p.add_argument("--pedigree-parent1-col", default="father_id")
    pedqc_p.add_argument("--pedigree-parent2-col", default="mother_id")
    pedqc_p.add_argument("--family-col", default="", help="Optional family ID column; auto-detected if omitted.")
    pedqc_p.add_argument("--value-column", choices=["auto", "genotype_call", "dosage"], default="auto")
    pedqc_p.add_argument("--max-variants", type=int, default=50_000, help="Maximum variants sampled for pedigree QC.")
    pedqc_p.add_argument("--min-call-rate", type=float, default=0.80)
    pedqc_p.add_argument("--min-maf", type=float, default=0.005)
    pedqc_p.add_argument("--report-min-r", type=float, default=0.59, help="Write sample-similarity pairs with R at/above this value.")
    pedqc_p.add_argument("--unrelated-max-r", type=float, default=0.59)
    pedqc_p.add_argument("--first-degree-min-r", type=float, default=0.64)
    pedqc_p.add_argument("--same-min-r", type=float, default=0.88)
    pedqc_p.add_argument(
        "--unlink-calls",
        default="unrelated,same",
        help="Observed relationship calls that should remove a declared parent edge in pedigree_curated.parquet.",
    )
    pedqc_p.add_argument("--sample-block-size", type=int, default=1024)
    pedqc_p.add_argument("--max-full-matrix-samples", type=int, default=5000)
    pedqc_p.add_argument("--random-seed", type=int, default=0)
    pedqc_p.add_argument("--no-write-umap", dest="write_umap", action="store_false")
    pedqc_p.add_argument("--umap-neighbors", type=int, default=50)
    pedqc_p.add_argument("--umap-max-variants", type=int, default=10_000)
    pedqc_p.set_defaults(write_umap=True)
    pedqc_p.set_defaults(func=cmd_pedigree_qc)




    recomb_p = sub.add_parser("recombination-diagnostics", help="Validate genetic map/recombination distances and generation-scaled switch probabilities.")
    recomb_p.add_argument("--positions", required=True)
    recomb_p.add_argument("--chromosome", required=True)
    recomb_p.add_argument("--output-dir", required=True)
    recomb_p.add_argument("--genetic-map", default="")
    recomb_p.add_argument("--samples", default="", help="Optional samples table with generation column.")
    recomb_p.add_argument("--recombination-rate-cM-per-Mb", type=float, default=0.5)
    recomb_p.add_argument("--generation-bin-size", type=float, default=0.0)
    recomb_p.set_defaults(func=cmd_recombination_diagnostics)

    autotune_p = sub.add_parser("autotune", help="Estimate CPU memory plan: sample shards, HMM chunk size, checkpoints, and Zarr chunks.")
    autotune_p.add_argument("--samples", required=True)
    autotune_p.add_argument("--positions", required=True)
    autotune_p.add_argument("--output-dir", required=True)
    autotune_p.add_argument("--n-founders", type=int, default=8)
    autotune_p.add_argument("--target-memory", default="100G")
    autotune_p.add_argument("--target-memory-fraction", type=float, default=0.75)
    autotune_p.add_argument("--sample-shard-count", type=int, default=0, help="Force shard count; 0 lets autotune choose.")
    autotune_p.add_argument("--hmm-checkpoint-interval", type=int, default=512)
    autotune_p.add_argument("--zarr-dosage-dtype", choices=["float32", "float16"], default="float16")
    autotune_p.add_argument("--zarr-gp-dtype", choices=["float32", "float16", "uint16"], default="uint16")
    autotune_p.add_argument("--dosage-only", action="store_true")
    autotune_p.add_argument("--ordered-diploid-states", action="store_true")
    autotune_p.add_argument("--min-chunk-size", type=int, default=5000)
    autotune_p.add_argument("--max-chunk-size", type=int, default=100000)
    autotune_p.set_defaults(func=cmd_autotune)

    shard_p = sub.add_parser("make-slurm-shards", help="Generate exact sample-sharded SLURM scripts for immutable-founder CPU production runs.")
    shard_p.add_argument("--output-dir", required=True, help="Directory for generated .sbatch scripts.")
    shard_p.add_argument("--n-shards", type=int, required=True)
    shard_p.add_argument("--samples", required=True)
    shard_p.add_argument("--positions", required=True)
    shard_p.add_argument("--chromosome", required=True)
    shard_p.add_argument("--founder-plink", required=True)
    shard_p.add_argument("--compact-evidence-cache-dir", required=True)
    shard_p.add_argument("--run-root", required=True, help="Root directory where shard_0000 etc. outputs will be written.")
    shard_p.add_argument("--preset", default="hs-rat-k8-production")
    shard_p.add_argument("--n-founders", type=int, default=8)
    shard_p.add_argument("--job-name", default="stitchcont")
    shard_p.add_argument("--account", default="")
    shard_p.add_argument("--partition", default="")
    shard_p.add_argument("--time-limit", default="12:00:00")
    shard_p.add_argument("--mem", default="100G")
    shard_p.add_argument("--cpus-per-task", type=int, default=8)
    shard_p.add_argument("--extra-args", default="", help="Additional stitchcont impute-from-evidence args appended to each shard command.")
    shard_p.set_defaults(func=cmd_make_slurm_shards)

    merge_shards_p = sub.add_parser("merge-shards", help="Merge sample-sharded matrices.zarr outputs along sample axis.")
    merge_shards_p.add_argument("--input-root", required=True, help="Directory containing shard_*/matrices.zarr.")
    merge_shards_p.add_argument("--output-dir", required=True, help="Output directory for merged matrices.zarr.")
    merge_shards_p.set_defaults(func=cmd_merge_shards)

    evidence_layout_p = sub.add_parser("evidence-shard-layout", help="Write per-sample-shard sample tables/manifests for a shared compact evidence cache without duplicating evidence.")
    evidence_layout_p.add_argument("--output-dir", required=True)
    evidence_layout_p.add_argument("--samples", required=True)
    evidence_layout_p.add_argument("--compact-evidence-cache-dir", required=True)
    evidence_layout_p.add_argument("--n-shards", type=int, required=True)
    evidence_layout_p.set_defaults(func=cmd_evidence_shard_layout)

    pedpost_p = sub.add_parser("pedigree-postprocess", help="Apply pedigree dosage smoothing chunkwise to a matrices.zarr dosage array after shard merge.")
    pedpost_p.add_argument("--matrices-zarr", required=True, help="Run directory or matrices.zarr path containing a dosage array.")
    pedpost_p.add_argument("--samples", required=True, help="Samples table in the same order as matrices.zarr/sample_ids.json.")
    pedpost_p.add_argument("--pedigree", default="", help="Optional pedigree table. If omitted, pedigree columns are read from --samples.")
    pedpost_p.add_argument("--output-dir", required=True)
    pedpost_p.add_argument("--strength", type=float, default=0.10)
    pedpost_p.add_argument("--chunk-positions", type=int, default=4096)
    pedpost_p.add_argument("--pedigree-offspring-col", default="sample_id")
    pedpost_p.add_argument("--pedigree-parent1-col", default="father_id")
    pedpost_p.add_argument("--pedigree-parent2-col", default="mother_id")
    pedpost_p.set_defaults(func=cmd_pedigree_postprocess)

    bench_p = sub.add_parser(
        "benchmark",
        help="Synthetic CPU/GPU HMM benchmark suite for ordered, chunked, unordered, sparse/factorized, and JAX minimal-output kernels.",
    )
    bench_p.add_argument("--output-dir", required=True)
    bench_p.add_argument("--n-samples", type=int, default=64)
    bench_p.add_argument("--n-positions", type=int, default=2000)
    bench_p.add_argument("--n-founders", type=int, default=8)
    bench_p.add_argument("--hmm-chunk-size", type=int, default=500)
    bench_p.add_argument("--repeat", type=int, default=1)
    bench_p.add_argument("--random-seed", type=int, default=0)
    bench_p.add_argument("--no-jax", action="store_true", help="Skip JAX benchmark kernels.")
    bench_p.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
