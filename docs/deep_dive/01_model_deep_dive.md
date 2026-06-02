# STITCHV2 Model Deep Dive

STITCHV2 is a read-aware founder mosaic imputation model. It keeps the core idea of original STITCH: each sample is modeled as copying from a small set of unknown or known founder haplotypes along the chromosome, and the observed low-coverage sequencing reads provide noisy evidence for the copied haplotypes. STITCHV2 extends that model with explicit Parquet/Zarr IO, cached compact read evidence, JAX HMM kernels, ploidy-aware execution, learned callability calibration, and optional pedigree-aware post-HMM adjustment.

This file describes the model. File formats and cache details are covered separately in:

- [02_inputs_outputs_deep_dive.md](02_inputs_outputs_deep_dive.md)
- [03_cached_files_deep_dive.md](03_cached_files_deep_dive.md)
- [04_pedigree_transmission_deep_dive.md](04_pedigree_transmission_deep_dive.md)

## High-Level Data Flow

A normal run has these stages:

1. Validate sample, position, founder, optional microarray, and optional pedigree inputs.
2. Read BAM/CRAM evidence for the requested chromosome/window, or load compact evidence from cache.
3. Split positions into HMM blocks and samples into memory-safe sample batches.
4. Run the HMM using fixed founders, mutable founders, or both.
5. Optionally run EM to update mutable founders.
6. Build dosage, haplotype probabilities, genotype posteriors, and hard calls.
7. Optionally apply calibration and pedigree adjustment.
8. Write block-level Parquet outputs, diagnostics, timing, and memory summaries.

The main pipeline implementation is in `src/stitchv2/pipeline.py`. The major configuration object is `PipelineConfig` in `src/stitchv2/config.py`.

## Core Objects

### Samples

Each sample is an individual to impute. At minimum STITCHV2 needs:

- `sample_id`: unique sample name. If absent, STITCHV2 creates `sample_0`, `sample_1`, etc.
- `generation`: generation or genetic distance scale used by the transition model.
- `bam_path`: path to BAM/CRAM reads. This can be empty for samples driven only by microarray evidence or downstream outputs, but normal imputation expects a valid BAM/CRAM path.

Optional sample columns include:

- `sex`: used for sex-chromosome ploidy assignment when sex-specific ploidy is enabled.
- `father_id`, `mother_id`, or user-defined pedigree parent columns.
- `family_id` or similar family metadata.
- `plink_path` for sample-specific auxiliary genotype metadata in workflows that use it.

### Positions

Positions define the exact target variants to impute. STITCHV2 does not discover variants de novo during a normal run. The positions table is the contract between read evidence, founders, outputs, and cache.

Required columns:

- `CHR`: chromosome label.
- `POS`: 1-based position.

Optional but strongly recommended columns:

- `REF`: reference allele.
- `ALT`: alternate allele.
- `VARIANT_TYPE`: optional explicit variant type: `snp`, `insertion`, or `deletion`.

If `REF` or `ALT` is absent, STITCHV2 fills it with `N`. For real benchmarking and BAM read interpretation, real alleles should be supplied.

For targeted insertion/deletion evidence, variants must be normalized and biallelic. Multiallelic records should be split before STITCHV2. The variant-aware reader supports small targeted insertions/deletions up to `--max-indel-len`, default `50`.

### Founder Panel

The founder panel is a matrix:

```text
founder_alt_probability[k, variant]
```

where each founder haplotype has an alternate-allele probability at each variant. Hard founder alleles are represented as probabilities near 0 or 1. Uncertain or mutable founders can use intermediate probabilities.

STITCHV2 can load founders from:

- VCF founder samples.
- PLINK founders.
- Synthetic or programmatic founder panels.

Loaded founders can be marked immutable. If `--n-founders` is larger than the number of loaded founders, STITCHV2 pads the panel with extra mutable founders initialized at 0.5. This supports a hybrid model such as 8 known immutable founders plus 1 or more mutable founders:

```text
K_final = K_loaded_immutable + K_extra_mutable
```

The extra mutable founders can absorb haplotypes not represented by the known founders, founder drift, sample mix-ups, or real haplotypes introduced later in a breeding population.

### Read Evidence

Reads are converted into allele evidence at the target variants. STITCHV2 stores both dense and compact representations internally:

- Dense count matrices: `ref_count`, `alt_count`, `other_count`, `depth`, and quality-weighted versions.
- Compact fragment evidence: per-fragment observations around target SNPs.

The compact representation is the preferred cached representation because it avoids storing a huge sample-by-variant dense matrix when most sample-site pairs have no reads.

The HMM can use read evidence in two main ways:

- `fragment_likelihood_mode=replace`: fragment likelihoods replace count-based emissions. This is the current default and the closest mode for STITCH parity.
- `fragment_likelihood_mode=augment`: count-based emissions are retained and fragment likelihoods add extra evidence. This can be useful experimentally but can double-count evidence if the same reads contribute to both the count emission and the fragment emission.

The default fragment coupling model is:

```text
fragment_coupling_model=stitch_parity
```

This is the only currently exposed coupling model and is the one to use for STITCH comparisons.

## HMM State Model

### Diploid Samples

For a diploid sample, the hidden state at variant `j` is a pair of founder haplotypes:

```text
Z_j = (h1_j, h2_j)
```

where `h1_j` and `h2_j` are founder indices from `0` to `K-1`.

The genotype dosage expectation at a variant is:

```text
E[G_j] = E[founder_alt_prob[h1_j, j] + founder_alt_prob[h2_j, j]]
```

For hard founders this is the expected count of alternate alleles in `{0, 1, 2}`. For probabilistic founders it is the expected dosage induced by the founder probabilities.

### Haploid Samples

For haploid or pseudo-haploid regions, the hidden state is a single founder:

```text
Z_j = h_j
```

The dosage is:

```text
E[G_j] = E[founder_alt_prob[h_j, j]]
```

For haploid X, Y, or MT runs, the genotype posterior dimension is smaller than diploid. Functions that require a diploid posterior, especially full pedigree transmission, will not run the diploid transmission update directly and must fall back to simpler behavior.

### Ploidy-Zero Samples

Ploidy-zero samples are excluded from the HMM for the relevant block or chromosome, but they are kept in the output. Their behavior is:

- No HMM state is allocated.
- Dosage is `NaN`.
- Genotype calls are `-1`.
- Genotype posterior values are masked or missing.
- Haplotype probabilities are zero or absent, depending on output mode.

This matters for sex chromosomes. For example, a chromosome may be absent in one sex or one sample class but the output still needs a rectangular sample-by-variant table.

## Transition Model

The transition model controls how often the copied founder haplotype changes along the chromosome. The transition probability depends on:

- physical distance between variants,
- recombination rate in cM/Mb,
- sample generation value,
- number of founders,
- ploidy.

At a high level, nearby variants have high probability of staying on the same founder path. Farther variants and later generations have higher switch probability.

The output `recombination/` contains the per-position recombination rate used by the run. The output `transitions/` contains compact transition probabilities:

- `switch_probability`
- `stay_probability`
- `offdiag_probability`

If `transition_output=full`, STITCHV2 can also write `transitions_full/`, which stores the full transition matrix per sample and position. This is much larger and is mainly useful for debugging.

The default transition model is:

```text
--transition-model stitch_parity
```

`--transition-model factorized` is experimental and changes the transition parameterization and output representation. It does not currently replace the exact founder-state HMM with a low-rank approximate inference algorithm; exact inference is still run over the founder state space.

Transition and posterior-trace outputs are controlled separately:

```text
--transition-output compact|factorized|full
--store-xi per-snp|full|False
--write-gamma summary|off|full
```

The production default is `--store-xi per-snp`, which writes compressed per-SNP transition summaries for hotspot/coldspot analysis. Full xi is opt-in and can be much larger than genotype outputs.

## Emission Model

The emission model scores how compatible the read evidence is with each hidden founder state.

### Count-Based Emission

The count emission uses ref/alt/other counts and base quality information to score possible genotypes. User-facing read filtering is controlled by:

Base and mapping quality filters:

```text
--min-base-quality
--min-mapping-quality
--cap-base-quality-by-mapping-quality
--ref-alt-only
--max-insert-size
```

For strict STITCH-like parity, typical settings are:

```text
--fragment-coupling-model stitch_parity
--fragment-likelihood-mode replace
--min-base-quality 17
--min-mapping-quality 17
--cap-base-quality-by-mapping-quality
--ref-alt-only
--max-insert-size 600
```

The exact values should match the original STITCH comparison being performed.

### Fragment Likelihood

The fragment likelihood groups observations from the same sequencing fragment. This prevents treating linked observations on the same read pair as independent evidence. The compact fragment representation records:

- sample,
- fragment index,
- central SNP index,
- observation order,
- target SNP index,
- observed allele code,
- observation quality.

`replace` mode uses this fragment evidence as the main HMM emission. `augment` mode keeps the ordinary count emission and adds the fragment likelihood.

`replace` is usually preferred for STITCH parity because the same read observations should not be counted twice. `augment` can be useful as an experimental model when count evidence and fragment evidence are intentionally treated as complementary, but it should be benchmarked carefully.

## Founder Updating and EM

STITCHV2 supports two founder regimes.

### Fixed Founder Mode

Fixed founder mode is used when founders are known and should be treated as truth. In this mode:

- loaded founders are immutable,
- the HMM infers sample copying paths against those founders,
- founder allele probabilities are not updated,
- one or a few inference passes are enough.

This is the mode to use for direct STITCH parity with known founder panels:

```text
--founder-plink <prefix>
--founder-immutable
--n-founders <number_loaded_founders>
--em-iterations 1
```

For STITCH-like fixed-founder parity, supply the founder panel with `--founder-vcf` or `--founder-plink`, set `--founder-immutable`, and keep `--n-founders` equal to the loaded founder count unless extra mutable founders are intentionally being tested.

### Mutable Founder Mode

Mutable founder mode estimates founder haplotypes from the samples. It is useful when no reliable founder panel is available or when the known founder panel is incomplete.

The model alternates:

1. HMM inference using the current founder panel.
2. Founder update from expected sample haplotype assignments.
3. Convergence checking.

Important controls:

```text
--em-iterations
--em-convergence-tol
--em-convergence-min-iterations
--em-convergence-patience
--em-founder-update-damping
--no-adaptive-em
--em-multistarts
```

When variable founders are used, more iterations may be needed. A value near 40 iterations is a practical convergence target for difficult variable-founder runs, although adaptive EM may stop earlier if the objective stabilizes.

### Hybrid Fixed Plus Mutable Founders

If a founder panel has `K_loaded` founders and the user requests `--n-founders K_total` with `K_total > K_loaded`, STITCHV2 adds:

```text
K_total - K_loaded
```

extra mutable founders.

This is useful when the original founders are mostly correct but the population may contain:

- an accidental unrecorded founder,
- later breeding introgression,
- mutations or private haplotypes,
- founder genotype errors,
- imperfect founder array/sequence data.

A common experimental setting is:

```text
8 immutable loaded founders + 1 mutable extra founder
--em-iterations 40
```

This is more expensive than fixed-founder parity, but BAM reading often dominates runtime for low-coverage whole-genome runs unless cached evidence is already available.

## Adaptive EM

Adaptive EM is designed to avoid using the last iteration blindly when an earlier iteration had a better objective. STITCHV2 tracks quantities such as:

- founder delta,
- read likelihood,
- masked genotype likelihood when available,
- best iteration.

With `adaptive_em_restore_best_founders=True`, the pipeline restores the best founder panel instead of automatically using the final iteration. This reduces the risk that late iterations overfit noisy read evidence or drift toward degenerate founders.

Fixed-founder parity mode should remain simple: immutable founders, little or no founder EM, and no unnecessary mutable-founder search.

## HMM Backends

The HMM backend is controlled by:

```text
--hmm-backend auto|numpy|jax|torch
```

The current intended high-performance backend is JAX. Relevant controls include:

```text
--jax-sample-batch-size
--no-jax-bucket-batch-shapes
--no-jax-count-emission-kernel
--no-jax-fragment-emission-kernel
--jax-persistent-cache-dir
```

The `--no-jax-*` options are debugging/fallback switches; leave them unset for the normal accelerated JAX path.

Important practical notes:

- JAX can accelerate HMM math and emission kernels.
- BAM/CRAM decoding through HTSlib is CPU-side and is not automatically moved to GPU.
- Avoiding repeated BAM reads via compact evidence cache is often more important than micro-optimizing the HMM for repeated benchmarks.
- JAX compilation has overhead, so very small blocks may look slower than expected unless bucketed shapes and the persistent compilation cache are reused.

## Memory Planning and Blocks

STITCHV2 does not use a fixed hard-coded SNP block size by default. `PipelineConfig.max_mem` defaults to:

```text
90%
```

The pipeline estimates a safe block size from:

- available OS memory,
- number of samples,
- number of founders,
- maximum ploidy,
- requested outputs,
- whether genotype posteriors are needed,
- fragment likelihood mode.

Users can still force smaller blocks with:

```text
--block-size
```

or limit the budget with:

```text
--max-mem 64GB
--max-mem 50000MB
--max-mem 70%
```

Input/output windows can be decoupled from HMM blocks. The intended efficient path is:

1. Read BAM/CRAM once for a large genomic IO window.
2. Store compact evidence.
3. Slice compact evidence for HMM blocks.
4. Reuse the cache for future calibration, benchmark, or parameter sweeps.

For production inference, the default SNP block mode is:

```text
--snp-block-mode exact_streaming
```

Exact streaming carries HMM boundary state across SNP blocks so chunking is a memory-management detail rather than an independent-block approximation. The approximate modes are explicit:

```text
--snp-block-mode independent_approx
--snp-block-mode density_balanced_overlap
```

Use approximate modes for smoke tests, stress tests, and exploratory runs. Label them approximate in benchmark reports.

## Dask Execution

STITCHV2 can run serially or with Dask:

```text
--executor serial|dask
--dask-scheduler local|threads|synchronous|jobqueue
--dask-n-workers
--dask-threads-per-worker
--dask-processes
--dask-dashboard-address :8787
--dask-performance-report report.html
--dask-task-stream task_stream.html
```

The local default is process-free Dask:

```text
dask_processes=False
```

This avoids many serialization costs for large Python/JAX objects and tends to be more stable on local workstations. `dask-jobqueue` is intended for future or cluster-oriented execution, where the scheduler submits jobs to SLURM or another queueing system.

For moderate batch counts, Dask is most useful when:

- work per task is large enough to amortize scheduling overhead,
- large objects are cached or memory mapped rather than serialized repeatedly,
- sample and SNP chunks are chosen to fit memory,
- JAX compiled functions can reuse shapes.

## Calibration Model

The default calibration mode is:

```text
calibration_mode=standard_callability
genotype_call_mode=quality_gated
calibration_callability_model=lightgbm
calibration_callability_decision_mode=per_snp_hierarchical
```

The key design is that raw HMM genotype posteriors remain the primary probability object. Standard callability calibration does not default to temperature scaling, posterior blending, or dosage scaling. Instead, it trains a lightweight continuous model that predicts:

```text
P(argmax genotype is correct)
```

Features include posterior and variant diagnostics such as:

- max genotype posterior,
- posterior margin,
- entropy,
- dosage distance to nearest integer,
- MAF,
- HWE deviation,
- missingness/support,
- read depth,
- INFO,
- ref/alt balance.

The model then chooses a hard-call threshold to optimize an objective based on:

- balanced accuracy,
- macro F1,
- a small call-rate reward.

The callability decision is local by SNP. For each SNP, STITCHV2 can compare calibrated gating against STITCH-style no-call on held-out or read-backed labels. It uses calibration where it appears beneficial and falls back to STITCH no-call where calibration is unsupported or unsafe.

The per-SNP decisions are written under:

```text
calibration_decisions/block=000000.parquet
```

Important columns include:

- `calibration_used`
- `fallback_to_stitch`
- `decision_source`
- `fallback_reason`
- `threshold`
- `call_rate_stitch`
- `call_rate_calibrated`
- `objective_delta`
- `maf`
- `info`
- `hwe_deviation`
- `support_rate`
- `missingness`
- `mean_depth`

## Read-Evidence Pseudo-Truth for Calibration

When no external truth is available, STITCHV2 can train callability from high-confidence read-backed sample-sites:

```text
--calibration-truth-source read_evidence
--calibration-read-truth-holdout-fraction 0.30
```

The careful design is:

1. Hold out a fraction of read fragments for calibration labels.
2. Run the HMM on the remaining evidence.
3. Derive high-confidence pseudo-truth from the held-out fragments only.
4. Train the callability model.
5. Apply the learned callability to a final HMM run using all read evidence.

This avoids contaminating calibration with an external benchmark truth set such as microarray calls while still giving the model a read-backed signal for whether hard calls are likely to be correct.

The pseudo-truth labels are intentionally conservative. Defaults include:

```text
--calibration-read-truth-min-depth 3
--calibration-read-truth-min-hom-depth 2
--calibration-read-truth-min-het-depth 4
--calibration-read-truth-min-het-allele-depth 1
--calibration-read-truth-hom-major-fraction 0.95
--calibration-read-truth-het-balance-min 0.25
--calibration-read-truth-het-balance-max 0.75
--calibration-read-truth-max-other-fraction 0.05
```

## Diagnostics and Degenerate Failure Detection

STITCHV2 computes per-variant diagnostics so catastrophic failures are visible in the output rather than hidden inside an aggregate metric.

The diagnostics include:

- MAF and alternate allele frequency,
- HWE deviation, chi-square, and p-value when diploid GP is available,
- heterozygosity rate,
- homozygosity rate,
- missingness and calling rate,
- INFO score,
- mean entropy,
- mean max GP,
- mean depth,
- support rate,
- calibration dosage/MAF/heterozygosity shifts when raw and calibrated GP are both available.

Diagnostics are written under:

```text
diagnostics/block=000000.parquet
diagnostics_summary.json
```

Thresholds catch failure modes such as:

- nearly every genotype called heterozygous,
- nearly every genotype called homozygous,
- extremely high missingness,
- low INFO,
- low max GP,
- calibration-induced dosage or MAF shifts.

The relevant configuration includes:

```text
--no-write-diagnostics
--diagnostics-fail-on-error
--diagnostics-warn-het-rate
--diagnostics-fail-het-rate
--diagnostics-warn-hom-rate
--diagnostics-fail-hom-rate
--diagnostics-warn-missing-rate
--diagnostics-fail-missing-rate
--diagnostics-warn-low-info
--diagnostics-fail-low-info
```

Diagnostics are enabled by default. Use `--no-write-diagnostics` only when the extra QC files are intentionally not wanted.

## What the Model Produces

The central model outputs are:

- dosage: expected alternate allele count per sample and variant,
- haplotype probabilities: posterior copying probability per founder haplotype,
- genotype posteriors: posterior probability for each genotype class,
- genotype calls: hard genotype calls after no-call or quality gating,
- founder updates: final founder alternate allele probabilities,
- recombination and transition probabilities,
- diagnostics and calibration decisions.

The outputs are block-level Parquet files by default. Floating xarray/Zarr consolidation is supported for downstream array-style analysis. BCF export exists only as an interoperability path and is explicitly slower than native Parquet/Zarr outputs.
