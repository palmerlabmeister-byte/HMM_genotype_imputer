# STITCHV2 CLI Reference

This file is the production-facing command-line reference for STITCHV2. It was checked against the installed `stitchv2` entrypoint on 2026-06-01.

Use this file to understand the stable command surface. Use `stitchv2 <command> --help` as the final source for exact parser defaults in a specific installed revision.

## Commands

| Command | Purpose |
| --- | --- |
| `stitchv2 run` | Main imputation pipeline. |
| `stitchv2 cv` | Real-data cross-validation and parameter tuning over K/nGen/S, seeds, and optional calibration. |
| `stitchv2 tune-jax-memory` | Profile JAX block sizes, sample batches, memory mapping, and reader settings. |
| `stitchv2 combine` | Combine partitioned Parquet chunks and optionally write Zarr. |
| `stitchv2 export-bcf` | Explicitly export Parquet/Zarr-oriented run outputs to BCF. BCF is slower and is for interoperability only. |
| `stitchv2 reformat-stitch-filenames` | Convert STITCH-style filenames into STITCHV2-oriented names and save a JSON mapping. |
| `stitchv2 discover-positions` | Discover reviewable candidate SNP/indel positions from BAM/CRAM plus reference FASTA. |
| `stitchv2 pedigree-qc` | Curate pedigree edges from preliminary genotype calls and write before/after UMAP plots. |

## `stitchv2 run`

`stitchv2 run` is the main command. Required arguments are `--samples`, `--positions`, `--chromosome`, `--output-dir`, and `--n-founders`.

### Required Inputs

| Flag | Meaning |
| --- | --- |
| `--samples` | Samples table, Parquet or CSV-like text. Must include `generation`; production runs should include stable `sample_id` values and BAM/CRAM paths in `bam_path` for sequenced samples. |
| `--positions` | Target position table, Parquet or CSV-like text. Must include `CHR` and `POS`; production read-backed runs should include `REF` and `ALT`. |
| `--chromosome` | Chromosome/contig to run. |
| `--output-dir` | Run output directory. |
| `--n-founders` | Total founder states in the HMM. If a founder file contains fewer founder haplotypes than this value, STITCHV2 appends extra mutable founders. |

### Region And Ploidy

| Flag | Meaning |
| --- | --- |
| `--chr-start` | Inclusive 1-based start coordinate. |
| `--chr-end` | Inclusive 1-based end coordinate. |
| `--ploidy` | Default ploidy for all samples. Use `0` for samples that should stay in output with missing genotypes and zero haplotype probability. |
| `--ploidy-males` | Male ploidy override when `samples.sex` is available. |
| `--ploidy-females` | Female ploidy override when `samples.sex` is available. |

### Founders

| Flag | Meaning |
| --- | --- |
| `--founder-vcf` | Founder VCF/VCF.GZ input. |
| `--founder-plink` | Founder PLINK prefix, without `.bed/.bim/.fam`. |
| `--founder-immutable` | Freeze loaded founders. Use this for STITCH parity and known-founder production runs. |
| `--founder-init-jitter` | Random jitter for initialized mutable founders. |

### Memory, Blocks, And Execution Mode

| Flag | Meaning |
| --- | --- |
| `--max-mem` | Maximum memory budget for automatic planning, for example `90%`, `64G`, or `32000M`. |
| `--block-size` | HMM SNP block size for approximate modes. `0` lets the planner choose. Exact streaming preserves whole-region HMM math while using sample batching and disk-backed scratch as needed. |
| `--io-window-size` | SNPs per BAM extraction window. `0` lets the planner choose. |
| `--snp-block-mode` | `exact_streaming` is the production default. `independent_approx` and `density_balanced_overlap` are approximate QC/exploration modes. |
| `--memory-safety-fraction` | Fraction of `--max-mem` available to the in-memory planner after safety headroom. |
| `--scratch-dir` | Optional Zarr/Parquet/Arrow scratch directory for streamed state and chunked outputs. |
| `--approx-overlap-fraction` | Overlap fraction for `density_balanced_overlap`. |
| `--approx-min-overlap-snps` | Minimum overlap SNPs per side for `density_balanced_overlap`. |

### HMM Trace And Transition Outputs

| Flag | Meaning |
| --- | --- |
| `--store-xi` | `per-snp` writes compact transition/hotspot summaries, `full` writes posterior xi Zarr, and `False` disables xi output. |
| `--write-gamma` | `summary` writes compact posterior summaries, `off` disables gamma output, and `full` writes full gamma traces. |
| `--no-write-hmm-boundaries` | Disable alpha/beta boundary metadata for exact-streaming diagnostics. |
| `--write-transitions` | Write transition outputs. |
| `--transition-output` | `compact`, `factorized`, or expanded `full`. Compact is sufficient for STITCH-parity transition summaries; full is expensive. |
| `--write-haplotype-probabilities` | Write haplotype posterior probabilities. |
| `--write-genotype-posteriors` | Write genotype posterior probabilities. |
| `--write-genotype-calls` | Write hard genotype calls. |
| `--write-support-mask` | Write read-support mask. |
| `--no-write-transition-summary` | Disable compact per-interval transition/hotspot summary output. |

### STITCH Parity, Recombination, And Transition Model

| Flag | Meaning |
| --- | --- |
| `--stitch-compat` | Apply original-STITCH-like parity settings: recombination rate 0.5 cM/Mb, base/map quality 17, BQ capped by MQ, insert size cap 600, ref/alt-only evidence, replace-mode fragment likelihoods, and STITCH-like likelihood caps. |
| `--recombination-rate-cm-per-mb` | Fixed recombination rate when no genetic map or trained scale overrides it. |
| `--genetic-map` | Optional Parquet/CSV genetic map with `POS` and `CM` or `GENETIC_CM`; cM values are interpolated onto target positions. |
| `--trainable-global-recombination-rate` | Estimate a block-level global recombination-rate scale from an initial posterior pass. Alias: `--trainable_global_recombination_rate`. |
| `--allow-recombination-hotspot-window` | Enable local recombination-rate scaling in windows such as `5Mb`; `0` disables. Alias: `--allow_recombination_hotspot_window`. |
| `--transition-model` | `stitch_parity` is the production default. `factorized` is an experimental regularized low-rank founder-switch model. |
| `--transition-factor-rank` | Rank for experimental factorized transition logits. |
| `--transition-factor-regularization` | Uniform-parity pseudo-count strength for factorized transition fitting. |
| `--transition-factor-max-deviation` | Maximum multiplicative off-diagonal deviation from parity before row renormalization. |
| `--no-transition-factor-train` | Keep factorized transition factors at exact STITCH-parity initialization. |

### Mutable Founder EM

| Flag | Meaning |
| --- | --- |
| `--em-iterations` | Number of mutable-founder EM statistic passes. Fixed immutable-founder parity runs usually use one/few passes; mutable-founder runs may use many more. |
| `--em-convergence-tol` | Stop mutable-founder EM early when founder-probability delta is below this threshold. `0` disables this criterion. |
| `--em-convergence-min-iterations` | Minimum EM passes before convergence checks can stop the loop. |
| `--em-convergence-patience` | Consecutive converged EM passes required before the final posterior pass. |
| `--em-founder-update-damping` | Damping for mutable-founder EM updates. Lower values can stabilize difficult runs. |
| `--no-adaptive-em` | Disable adaptive EM best-founder bookkeeping. |
| `--no-adaptive-em-restore-best-founders` | Do not restore the best founder panel found by adaptive EM. |
| `--em-multistarts` | Mutable-founder multi-start count. Fixed-founder parity mode ignores this. |
| `--no-founder-update-hardening` | Disable discrete hardening of mutable founder haplotype updates. |

### HMM Backend And JAX

| Flag | Meaning |
| --- | --- |
| `--hmm-backend` | `auto`, `numpy`, `jax`, or experimental `torch`. |
| `--jax-sample-batch-size` | Samples per JAX forward/backward call. `0` means all samples or planner-selected behavior. |
| `--no-jax-bucket-batch-shapes` | Disable batch-size bucketing that avoids compiling separate remainder shapes. |
| `--no-jax-count-emission-kernel` | Disable count-to-emission work inside JAX for no-fragment diploid batches. |
| `--no-jax-fragment-emission-kernel` | Disable fragment-aware JAX emission kernels and fall back to NumPy fragment emission assembly. |
| `--jax-persistent-cache-dir` | Optional persistent JAX compilation cache directory reused across runs. |

### Dask And Cluster Execution

| Flag | Meaning |
| --- | --- |
| `--executor` | `serial` or `dask`. |
| `--dask-scheduler` | `local`, `threads`, `synchronous`, or `jobqueue`. |
| `--dask-n-workers` | Local Dask worker count. `0` means auto. |
| `--dask-threads-per-worker` | Threads per Dask worker. |
| `--dask-processes` | Use Dask worker processes instead of threads. Local default is threads. |
| `--dask-memory-limit` | Per-worker memory limit, for example `16GB`; empty means auto. |
| `--dask-dashboard-address` | Dashboard bind address. Empty disables dashboard. |
| `--dask-performance-report` | Write Dask performance report HTML. |
| `--dask-task-stream` | Write Dask task stream to `.html` or `.json`. |
| `--dask-dashboard-hold-seconds` | Keep dashboard alive after compute before closing. |
| `--dask-target-task-memory-mb` | Auto-shrink chunks to a rough per-task memory target. |
| `--dask-min-block-size` | Minimum SNP block size considered by Dask planning. |
| `--dask-min-sample-batch-size` | Minimum sample batch size considered by Dask planning. |
| `--dask-sample-batch-size` | Samples per Dask HMM task. `0` means all or planner-selected behavior. |
| `--dask-jobqueue-class` | Dask-jobqueue cluster class for `--dask-scheduler jobqueue`, default `SLURMCluster`. |
| `--dask-jobqueue-queue` | Queue/partition for dask-jobqueue. |
| `--dask-jobqueue-account` | Account/project for dask-jobqueue. |
| `--dask-jobqueue-cores` | Cores per dask-jobqueue job. |
| `--dask-jobqueue-memory` | Memory per dask-jobqueue job. |
| `--dask-jobqueue-walltime` | Walltime per dask-jobqueue job. |

### BAM/CRAM Reading And Variant Evidence

| Flag | Meaning |
| --- | --- |
| `--read-mode` | `read_stream` is the standard path. `pileup` is retained for fallback/debugging. |
| `--read-stream-backend` | `auto`, `python`, `htslib`, `snp_only_bamreader`, `stitch_style_bamreader`, or `variant_aware_bamreader`. `auto` chooses the compiled variant-aware path when available. |
| `--min-base-quality` | Minimum base quality. |
| `--min-mapping-quality` | Minimum mapping quality. |
| `--max-insert-size` | Skip reads/fragments above this absolute template length; `0` disables. |
| `--max-indel-len` | Maximum targeted insertion/deletion length for variant-aware BAM reading. |
| `--cap-base-quality-by-mapping-quality` | Cap base quality by mapping quality. |
| `--ref-alt-only` | Ignore bases that are neither REF nor ALT instead of treating them as OTHER evidence. |
| `--no-merge-fragments-by-query` | Disable query-name fragment merging. |
| `--merge-unpaired-fragments-by-query` | Merge all informative reads sharing a query name, even if paired flags are absent. |
| `--use-bx-tag` | Merge reads with the same linked-read BX tag when present. |
| `--bx-tag` | Two-character SAM auxiliary tag used for linked-read fragment merging. |
| `--bx-tag-upper-limit` | Maximum observations accumulated in one BX/query fragment before flushing. |
| `--downsample-to-coverage` | Fragment-level per-sample/site coverage cap before HMM; `0` disables. |
| `--downsample-fraction` | Random fragment retention fraction before HMM. |
| `--io-workers` | Native reader sample-level worker count. |
| `--htslib-threads-per-file` | HTSlib decompression threads per BAM/CRAM file. |

### Evidence Cache

| Flag | Meaning |
| --- | --- |
| `--compact-evidence-cache-dir` | Directory for compact fragment-evidence cache files. |
| `--compact-evidence-cache-mode` | `off`, `read`, `write`, or `readwrite`. Cached runs can skip BAM decoding for matching samples/windows/settings. |
| `--compact-evidence-cache-format` | Current production format is `parquet_zarr`: compact evidence/support in compressed Parquet, optional dense arrays in Zarr. |
| `--compact-evidence-cache-sample-batch-size` | Sample batch size for cache partitions. |
| `--compact-evidence-no-dense-counts` | When loading compact evidence, build only a support mask and let replace-mode fragment likelihoods drive the HMM. |
| `--compact-evidence-cache-include-dense-counts` | Store dense count/weight matrices in the cache. This is larger but needed for exact augment-mode cache reuse and some diagnostics. |

### Fragment Likelihood

| Flag | Meaning |
| --- | --- |
| `--fragment-likelihood-mode` | `replace` uses fragment likelihoods as the read-evidence emission and is the default/STITCH-parity setting. `augment` adds fragment likelihoods on top of dense counts and can double count unless dense/cache semantics are carefully controlled. |
| `--fragment-coupling-model` | Fragment/read coupling behavior; production option is `stitch_parity`. |
| `--fragment-max-diff-reads` | Read likelihood cap used inside fragment handling. |
| `--fragment-max-emission-diff` | Emission likelihood cap used inside fragment handling. |
| `--no-fragment-rescale-read-likelihood` | Disable fragment read-likelihood rescaling. |

### Calibration And Hard Calling

| Flag | Meaning |
| --- | --- |
| `--no-calibrate-genotype-posteriors` | Disable genotype-posterior calibration. |
| `--calibration-mode` | `standard_callability`, `fixed`, or `masked_cv`. Standard callability keeps raw HMM GP and learns a hard-call gate. |
| `--genotype-posterior-temperature` | Temperature for fixed posterior calibration. Not the default standard-callability behavior. |
| `--genotype-posterior-blend` | Blend factor for fixed posterior calibration. Not the default standard-callability behavior. |
| `--genotype-call-mode` | `argmax`, `stitch_no_call`, or `quality_gated`. |
| `--genotype-call-min-confidence` | Minimum max-GP confidence for quality-gated hard calls. |
| `--genotype-call-min-margin` | Minimum top-vs-second GP margin for quality-gated hard calls. |
| `--genotype-call-stitch-threshold` | STITCH-style no-call threshold, usually GP >= 0.9. |
| `--genotype-call-correctness-threshold` | Minimum predicted P(correct) for standard-callability hard calls. |
| `--use-lightgbm-calibrator` | Enable LightGBM calibration when available. |
| `--calibration-context-window` | Local context window for calibration features. |
| `--calibration-block-snps` | SNP block size for calibration feature assembly. |
| `--calibration-truth-source` | `read_evidence`, `microarray`, `auto`, or `none`. `read_evidence` uses high-confidence read-backed sample-sites without leaking external gold-standard labels. |
| `--calibration-read-truth-min-depth` | Minimum depth for read-evidence pseudo-truth. |
| `--calibration-read-truth-min-hom-depth` | Minimum depth for homozygous read-evidence labels. |
| `--calibration-read-truth-min-het-depth` | Minimum depth for heterozygous read-evidence labels. |
| `--calibration-read-truth-min-het-allele-depth` | Minimum per-allele depth for heterozygous read-evidence labels. |
| `--calibration-read-truth-hom-major-fraction` | Major-allele fraction threshold for homozygous read-evidence labels. |
| `--calibration-read-truth-het-balance-min` | Minimum allele balance for heterozygous read-evidence labels. |
| `--calibration-read-truth-het-balance-max` | Maximum allele balance for heterozygous read-evidence labels. |
| `--calibration-read-truth-max-other-fraction` | Maximum OTHER evidence fraction for read-evidence labels. |
| `--calibration-read-truth-holdout-fraction` | Fraction of read fragments reserved only for read-evidence pseudo-truth. Final HMM still uses all read evidence. |
| `--calibration-use-optuna` | Use Optuna for calibration tuning where supported. |
| `--calibration-optuna-trials` | Number of Optuna trials. |
| `--calibration-max-train-rows` | Maximum rows used to train calibration models. |
| `--calibration-callability-model` | `lightgbm` or `sklearn_logistic`. LightGBM is the richer model; sklearn logistic is a lighter fallback. |
| `--calibration-callability-min-train-rows` | Minimum training rows needed before a learned callability model is used. |
| `--calibration-callability-min-call-rate` | Minimum accepted call rate for callability gating. |
| `--calibration-callability-call-rate-weight` | Small reward for call rate in callability threshold selection. |
| `--calibration-callability-decision-mode` | `per_snp_hierarchical` or `global`. Per-SNP hierarchical is the default production behavior. |
| `--no-calibration-callability-bound-to-stitch` | Allow learned callability thresholds to move without the STITCH no-call call-rate bound. |
| `--calibration-callability-min-call-rate-delta` | Minimum allowed call-rate change relative to STITCH no-call for per-SNP decisions. |
| `--calibration-callability-max-call-rate-delta` | Maximum allowed call-rate change relative to STITCH no-call for per-SNP decisions. |
| `--calibration-callability-validation-site-fraction` | Fraction of labeled sites used for validation of per-SNP/local/global fallback decisions. |
| `--calibration-callability-min-objective-improvement` | Minimum objective improvement required before using learned callability at a SNP/window. |
| `--calibration-callability-max-hardcall-maf-shift` | Maximum allowed hard-call MAF shift. |
| `--calibration-callability-max-hardcall-het-shift` | Maximum allowed heterozygosity shift. |
| `--calibration-train-site-fraction` | Fraction of labeled SNPs used to fit learned calibration. The calibrator is then applied to all SNPs. |
| `--calibration-lightgbm-use-block-context` | Enable older local-context LightGBM stage before read/site-aware calibration. |
| `--calibration-lightgbm-use-fixed-stage0` | Use fixed temperature/blend stage before LightGBM. Default standard-callability uses raw HMM GP. |
| `--calibration-maf-bins` | MAF bins for fixed/masked calibration modes. |
| `--calibration-temperatures` | Temperature grid for fixed/masked calibration modes. |
| `--calibration-blends` | Blend grid for fixed/masked calibration modes. |
| `--calibration-dosage-scales` | Dosage-scale grid for fixed/masked calibration modes. |
| `--calibration-dosage-offsets` | Dosage-offset grid for fixed/masked calibration modes. |
| `--calibration-hwe-prior-weights` | HWE prior grid for fixed/masked calibration modes. |
| `--no-calibration-optimize-dosage-scale` | Disable dosage-scale optimization in fixed/masked calibration. |
| `--calibration-hwe-weight` | HWE weight for calibration objectives. |
| `--calibration-hwe-min-maf` | Minimum MAF for HWE-weighted calibration diagnostics/objectives. |
| `--no-calibration-sanity-checks` | Disable calibration sanity checks. |
| `--calibration-max-mean-abs-dosage-shift` | Maximum allowed mean absolute dosage shift after calibration. |
| `--calibration-max-mean-abs-maf-shift` | Maximum allowed mean absolute MAF shift after calibration. |
| `--calibration-max-het-rate-shift` | Maximum allowed heterozygosity-rate shift after calibration. |
| `--calibration-max-mean-entropy-shift` | Maximum allowed mean entropy shift after calibration. |

### Diagnostics

| Flag | Meaning |
| --- | --- |
| `--no-write-diagnostics` | Disable diagnostics output. |
| `--diagnostics-fail-on-error` | Turn diagnostic failures into run errors. |
| `--diagnostics-warn-het-rate` | Warning threshold for excessive heterozygosity. |
| `--diagnostics-fail-het-rate` | Failure threshold for excessive heterozygosity. |
| `--diagnostics-warn-hom-rate` | Warning threshold for excessive homozygosity. |
| `--diagnostics-fail-hom-rate` | Failure threshold for excessive homozygosity. |
| `--diagnostics-warn-missing-rate` | Warning threshold for missingness. |
| `--diagnostics-fail-missing-rate` | Failure threshold for missingness. |
| `--diagnostics-warn-low-info` | Warning threshold for low INFO. |
| `--diagnostics-fail-low-info` | Failure threshold for low INFO. |

### Microarray/Hard-Call Evidence

| Flag | Meaning |
| --- | --- |
| `--microarray-plink` | Optional PLINK prefix for hard microarray genotype evidence. |
| `--microarray-generation-default` | Default generation value for array-only samples. |
| `--microarray-hard-call-weight` | Weight assigned to microarray hard-call evidence. |
| `--microarray-calibration-only` | Use microarray PLINK only as calibration truth; do not inject hard calls into HMM read evidence. |
| `--no-microarray-add-samples` | Do not add samples present only in the microarray PLINK data. |

### Pedigree-Aware Posteriors

| Flag | Meaning |
| --- | --- |
| `--pedigree` | Optional pedigree table; if omitted, pedigree columns can be read from the samples table. |
| `--pedigree-mode` | `off`, `smooth`, `kinship`, or `transmission`. |
| `--pedigree-strength` | Strength of pedigree posterior adjustment. |
| `--pedigree-offspring-col` | Offspring/sample ID column in the pedigree table. |
| `--pedigree-parent1-col` | First parent column. |
| `--pedigree-parent2-col` | Second parent column. |
| `--pedigree-iterations` | Number of pedigree message-passing iterations. |
| `--pedigree-kinship-threshold` | Similarity threshold used by kinship fallback mode. |

### Compression And Miscellaneous

| Flag | Meaning |
| --- | --- |
| `--random-seed` | Random seed used by stochastic components. |
| `--memory-map-read-matrices` | Memory-map intermediate read matrices. |
| `--memory-map-dir` | Directory for memory-mapped matrices. |
| `--no-profile-memory` | Disable memory profiling. |
| `--compression` | Parquet compression, default `zstd`. |
| `--compression-level` | Compression level for applicable outputs. |

## `stitchv2 cv`

`stitchv2 cv` accepts the same common run options as `stitchv2 run`, including input, region, ploidy, memory, reader, cache, HMM, Dask, calibration, diagnostics, microarray, pedigree, and founder flags. It adds the following cross-validation-specific flags:

| Flag | Meaning |
| --- | --- |
| `--pseudo-truth` | Pseudo-truth table with `sample_id`, `position`, and `genotype` or `dosage_truth`. |
| `--k-values` | Comma-separated K/founder values to test. |
| `--ngen-values` | Comma-separated generation/recombination-scale values to test. |
| `--s-values` | Comma-separated S values to test. |
| `--seeds` | Comma-separated seeds. |
| `--folds` | Number of folds. |
| `--holdout-fraction` | Fraction of truth sites/samples held out. |
| `--lightgbm-post-calibrator` | Add a LightGBM post-calibrator in the CV workflow. |

## `stitchv2 tune-jax-memory`

| Flag | Meaning |
| --- | --- |
| `--data-dir` | Directory containing prepared run inputs. |
| `--output-dir` | Output directory for tuning reports. |
| `--chromosome` | Chromosome to profile. |
| `--n-founders` | Number of founders. |
| `--em-iterations` | EM iterations used in profiling. |
| `--block-sizes` | Comma-separated block sizes to test. |
| `--memory-map-modes` | Comma-separated memory-map modes, usually `off,on`. |
| `--jax-sample-batch-sizes` | Comma-separated JAX sample batch sizes. |
| `--read-stream-backend` | Reader backend to profile: `auto`, `python`, `htslib`, `snp_only_bamreader`, `stitch_style_bamreader`, or `variant_aware_bamreader`. |
| `--io-workers` | Reader workers. |
| `--htslib-threads-per-file` | HTSlib decompression threads per file. |
| `--fragment-likelihood-mode` | `replace` or `augment`. |
| `--memory-map-dir` | Directory for memory-mapped intermediates. |
| `--max-samples` | Optional sample cap for profiling. |
| `--max-positions` | Optional position cap for profiling. |
| `--max-peak-rss-mb` | Optional maximum peak RSS threshold. |

## `stitchv2 combine`

| Flag | Meaning |
| --- | --- |
| `--run-output-dir` | Run directory containing partitioned outputs. |
| `--datasets` | Comma-separated dataset names or `all`. |
| `--output-dir` | Output directory for combined files. |
| `--input-dir` | Optional explicit input Parquet directory for one-off combine. |
| `--output-file` | Optional explicit output Parquet file for one-off combine. |
| `--compression` | Parquet compression. |
| `--compression-level` | Compression level. |
| `--row-group-size` | Parquet row group size. |
| `--write-zarr` | Also write floating dosage/posterior outputs as xarray Zarr. |
| `--zarr-output` | Optional output path for `--write-zarr`. |
| `--zarr-consolidated` | Request consolidated Zarr metadata. |

## `stitchv2 export-bcf`

BCF export is intentionally explicit because STITCHV2's primary outputs are Parquet and Zarr. Use this only when an external tool requires BCF.

| Flag | Meaning |
| --- | --- |
| `--run-output-dir` | Completed STITCHV2 run directory. |
| `--output-bcf` | BCF output path. |
| `--chromosome` | Chromosome label to write. |
| `--include-haplotype-dosage` | Include haplotype dosage fields. |
| `--no-tabix-index` | Do not create a tabix index. |

## `stitchv2 reformat-stitch-filenames`

| Flag | Meaning |
| --- | --- |
| `--input-file` | Text file with one filename per line. |
| `--filenames` | Comma-separated filename list. |
| `--output-file` | JSON mapping output path. |

## `stitchv2 discover-positions`

Discovery is a review step, not an automatic imputation run. The output table should be filtered, normalized, and paired with founder genotypes before use as `--positions`.

| Flag | Meaning |
| --- | --- |
| `--samples` | Samples table with a BAM/CRAM path column. |
| `--reference-fasta` | Indexed reference FASTA. Run `samtools faidx` first if needed. |
| `--chromosome` | Chromosome/contig to scan. |
| `--chr-start` | Inclusive 1-based discovery start; defaults to chromosome start. |
| `--chr-end` | Inclusive 1-based discovery end; defaults to chromosome end. |
| `--output-file` | Output positions table, `.parquet`, `.csv`, `.tsv`, or `.txt`. |
| `--summary-file` | Optional JSON summary path. |
| `--bam-path-col` | Samples-table BAM/CRAM path column. |
| `--max-bams` | Optional cap for smoke tests. `0` scans all BAMs. |
| `--window-size` | Reference bases per native discovery chunk. |
| `--min-base-quality` | Minimum base quality. |
| `--min-mapping-quality` | Minimum mapping quality. |
| `--htslib-threads-per-file` | HTSlib decompression threads per file. |
| `--max-insert-size` | Skip reads with absolute template length above this value; `0` disables. |
| `--variant-types` | Comma-separated discovery types: `snp`, `ins`, `del`. |
| `--max-indel-len` | Maximum insertion/deletion length emitted by discovery. |
| `--cap-base-quality-by-mapping-quality` | Cap base quality by mapping quality. |
| `--min-depth` | Minimum total depth for a candidate. |
| `--min-alt-count` | Minimum alternate count. |
| `--min-alt-samples` | Minimum number of samples supporting ALT. |
| `--min-alt-fraction` | Minimum alternate fraction. |
| `--max-other-fraction` | Maximum ambiguous/other evidence fraction. |
| `--compression` | Parquet compression when output is Parquet. |

## `stitchv2 pedigree-qc`

Use `pedigree-qc` after an initial no-pedigree run, preferably across enough chromosomes to make relationship inference stable. Then rerun `stitchv2 run` with the curated pedigree and an appropriate `--pedigree-mode`.

| Flag | Meaning |
| --- | --- |
| `--samples` | Samples table with `sample_id` and optional pedigree metadata. |
| `--pedigree` | Optional pedigree table. If absent, pedigree columns are read from `--samples`. |
| `--run-output-dir` | Preliminary STITCHV2 run directory containing `genotype_calls/` or `dosage/`. Repeat for multiple chromosomes. |
| `--genotype-table` | Long Parquet table/directory with `sample_id`, `chromosome`, `position`, and `genotype_call` or `dosage`. Repeatable. |
| `--output-dir` | Pedigree QC output directory. |
| `--sample-id-col` | Sample ID column. |
| `--pedigree-offspring-col` | Offspring/sample ID column. |
| `--pedigree-parent1-col` | First parent column. |
| `--pedigree-parent2-col` | Second parent column. |
| `--family-col` | Optional family ID column; auto-detected if omitted. |
| `--value-column` | `auto`, `genotype_call`, or `dosage`. |
| `--max-variants` | Maximum variants sampled for pedigree QC. |
| `--min-call-rate` | Minimum variant/sample call rate used for QC. |
| `--min-maf` | Minimum MAF for variants used in QC. |
| `--report-min-r` | Write sample-similarity pairs with R at or above this value. |
| `--unrelated-max-r` | Maximum R considered unrelated. |
| `--first-degree-min-r` | Minimum R considered first-degree related. |
| `--same-min-r` | Minimum R considered same-sample/duplicate-like. |
| `--unlink-calls` | Observed relationship calls that should remove a declared parent edge in `pedigree_curated.parquet`. |
| `--sample-block-size` | Sample block size used by similarity computation. |
| `--max-full-matrix-samples` | Maximum samples for writing full similarity matrices. |
| `--random-seed` | Random seed for variant sampling/UMAP. |
| `--no-write-umap` | Disable UMAP output. |
| `--umap-neighbors` | UMAP neighbor count. |
| `--umap-max-variants` | Maximum variants sampled for UMAP. |

