# STITCHV2 Improvements and Differences Compared With Original STITCH

STITCHV2 is designed to keep the most important idea from original STITCH while modernizing the implementation, IO, scalability, diagnostics, and extensibility.

Original STITCH is a mature low-coverage sequencing imputation method. Its core model learns or uses a small set of founder-like ancestral haplotypes and imputes each sample as a mosaic of those haplotypes along the genome. STITCHV2 keeps that founder-mosaic HMM framing, but exposes more of the pipeline as inspectable files and adds workflows that were difficult to do cleanly in the original implementation.

This document describes the main differences.

## What STITCHV2 Keeps From STITCH

STITCHV2 keeps these core concepts:

- low-coverage sequencing imputation without requiring a dense reference panel,
- a small number `K` of founder or ancestral haplotypes,
- HMM copying paths along the chromosome,
- recombination-driven transitions,
- read-aware emissions,
- EM-like founder updates when founders are unknown,
- fixed founder behavior when founders are supplied and should remain immutable,
- STITCH-style no-call behavior for parity comparisons,
- STITCH-like fragment coupling behavior through `fragment_coupling_model=stitch_parity`.

For direct comparisons, STITCHV2 should be run with:

```bash
--fragment-coupling-model stitch_parity
--fragment-likelihood-mode replace
--genotype-call-mode stitch_no_call
--genotype-call-stitch-threshold 0.9
--founder-vcf <founders.vcf.gz>
# or: --founder-plink <plink_prefix>
--founder-immutable
--em-iterations 1
```

and with read filters matching the STITCH run being compared.

For STITCH-like fixed-founder parity, load known founders, mark them immutable, and do not learn over them unless extra mutable founders are intentionally added by setting `--n-founders` larger than the loaded founder count.

## Native Parquet/Zarr Outputs

Original STITCH workflows often require conversions through text, VCF/BCF, or custom formats. STITCHV2 writes analysis-friendly native files:

- Parquet for long tables and hard calls,
- Parquet for compact fragment evidence,
- bitpacked Parquet for support masks,
- Zarr/xarray for dense floating arrays when needed.

Benefits:

- faster downstream reads than VCF for large tables,
- column pruning,
- compression,
- row-group statistics,
- easier integration with pandas, pyarrow, dask, xarray, and plotting tools,
- easier benchmark and QC generation.

BCF export exists, but it is explicitly an interoperability path:

```text
Use native Parquet/Zarr outputs for primary results; use export-bcf only when an external tool explicitly requires BCF.
```

## Compact Evidence Cache

One of the largest practical differences is the compact evidence cache. Original STITCH reads BAM evidence as part of the run. STITCHV2 can decode evidence once and reuse it across many runs.

This matters because BAM/CRAM decoding is CPU-bound and often dominates total runtime.

STITCHV2 cache advantages:

- sample-partitioned cache,
- partial cache hits for newly added samples,
- compact fragment evidence in Parquet,
- bitpacked support masks,
- optional dense Zarr counts,
- cache validation with positions hash,
- reusable for calibration and founder sweeps.

This enables workflows such as:

```text
read BAM once -> cache evidence -> run parity -> run calibration -> run k9 EM -> run pedigree modes
```

without decoding the same BAMs every time.

## Multiple BAM Readers

STITCHV2 has several read-stream backends:

```bash
--read-stream-backend auto
--read-stream-backend python
--read-stream-backend htslib
--read-stream-backend snp_only_bamreader
--read-stream-backend stitch_style_bamreader
--read-stream-backend variant_aware_bamreader
```

The important modern additions are:

- HTSlib-backed native reading,
- SNP-only CIGAR scanning for sparse target variants,
- STITCH-style moving SNP-range scanning,
- variant-aware SNP/insertion/deletion evidence extraction,
- compact evidence output/cache.

Use `auto` for normal production runs. Use `variant_aware_bamreader` explicitly when targeted indel evidence is needed. Use `stitch_style_bamreader` for strict SNP-only STITCH compatibility checks.

## JAX HMM and Emission Kernels

STITCHV2 can run HMM computation through JAX. Relevant controls include:

```bash
--hmm-backend jax
--jax-persistent-cache-dir
--no-jax-count-emission-kernel
--no-jax-fragment-emission-kernel
```

The `--no-jax-*` flags disable JAX emission kernels for debugging or parity isolation; leave them unset for the normal accelerated path.

Advantages:

- compiled kernels for repeated shapes,
- vectorized emission and HMM operations,
- optional persistent compilation cache,
- easier GPU execution for HMM math where available,
- fewer Python loops in the hot HMM path.

Caveat:

- HTSlib BAM/CRAM decoding remains CPU-side. GPUs help the HMM and array math, not raw BAM decompression or CIGAR parsing.

The largest speed gains often come from combining:

1. efficient BAM reading,
2. compact evidence cache,
3. JAX HMM,
4. memory-aware block sizing.

## Memory-Aware Block Planning

STITCHV2 can infer a block size from available OS memory:

```bash
--max-mem 90%
```

or from a user-provided budget:

```bash
--max-mem 64GB
--max-mem 50000MB
--max-mem 70%
```

The planner considers:

- number of samples,
- number of founders,
- ploidy,
- whether genotype posteriors are written,
- whether pedigree needs posteriors,
- fragment likelihood mode,
- output choices.

This avoids relying on a fixed SNP block size such as 500 for every machine and dataset.

## Better Separation of IO Windows and HMM Blocks

STITCHV2 is moving toward a clean separation:

- IO windows: large genomic spans read once from BAM/CRAM.
- HMM blocks: memory-safe chunks used by the inference kernel.

This matters because the best IO unit is not always the best HMM unit. BAM reading benefits from fewer reopens and larger sequential scans, while HMM inference needs blocks sized to memory and compilation shape.

The compact cache is the bridge:

```text
BAM/CRAM -> compact evidence store -> many HMM blocks/runs
```

## Ploidy and Sex-Chromosome Handling

Original STITCH-style workflows are mainly diploid autosomal workflows. STITCHV2 explicitly tracks ploidy and supports:

- diploid autosomes,
- haploid/pseudo-haploid regions,
- sex-specific ploidy,
- MT/Y/sex chromosome absent samples,
- ploidy-zero samples,
- generic polyploid experiments.

Ploidy-zero behavior is explicit:

- sample remains in output,
- HMM skips it,
- genotype call is missing,
- dosage is `NaN`,
- haplotype probabilities are zero/missing.

This is important for whole-genome pipelines where every chromosome is run separately but the final output needs stable sample order and sample inclusion.

## Pedigree QC

STITCHV2 adds a built-in pedigree QC workflow:

```bash
stitchv2 pedigree-qc
```

It can:

- load first-pass calls from one or more chromosomes,
- compute efficient genotype similarity,
- classify relationships by similarity thresholds,
- remove likely wrong parent edges,
- output a curated pedigree,
- write UMAP before/after pedigree edge plots,
- report long UMAP edges that may identify bad pedigree links.

This is not just a visualization convenience. It is a safety step before using pedigree information to alter genotype probabilities.

## Pedigree-Aware Imputation

STITCHV2 supports post-HMM pedigree modes:

```bash
--pedigree-mode off
--pedigree-mode smooth
--pedigree-mode kinship
--pedigree-mode transmission
```

Original STITCH does not provide this same integrated pedigree QC plus transmission adjustment workflow.

The strongest mode, `transmission`, performs parent-child message passing on diploid genotype posteriors. It uses:

- parent-to-child Mendelian messages,
- child-to-parent reverse messages,
- population fallback for missing parents,
- recombination smoothing,
- direct read evidence anchoring,
- bounded blending.

This can help low-coverage samples when the pedigree is curated and reliable.

## Standard Callability Calibration

STITCHV2 adds a modern callability calibration layer.

The default calibration philosophy is:

- keep raw HMM genotype posteriors as the probability object,
- learn a model for whether the argmax genotype is likely correct,
- use calibration to decide hard-call/no-call status,
- fall back to STITCH no-call locally when calibration is unsafe.

The standard model can use LightGBM:

```bash
--calibration-mode standard_callability
--calibration-callability-model lightgbm
--calibration-callability-decision-mode per_snp_hierarchical
```

Features include:

- max GP,
- posterior margin,
- entropy,
- dosage distance to integer,
- MAF,
- HWE deviation,
- missingness,
- support rate,
- depth,
- INFO,
- ref/alt balance.

The goal is to improve balanced accuracy and F1 without silently shifting MAF, HWE, or heterozygosity into a degenerate state.

## Per-SNP Calibration Fallback

Calibration is not treated as a global all-or-nothing decision. STITCHV2 can decide locally:

```text
per SNP -> local/group fallback -> block/global fallback -> STITCH no-call
```

The output `calibration_decisions/` explains:

- how many SNPs used calibration,
- how many fell back to STITCH no-call,
- why fallback happened,
- call-rate shifts,
- objective deltas,
- MAF/HWE/support/depth context.

This is a major interpretability improvement over a single global threshold.

## Diagnostics and Failure Detection

STITCHV2 writes diagnostics designed to detect catastrophic failures quickly:

- all heterozygotes,
- all homozygotes,
- very high missingness,
- very low INFO,
- low mean max GP,
- calibration-induced MAF shift,
- calibration-induced heterozygosity shift,
- HWE deviations.

Outputs:

```text
diagnostics/block=*.parquet
diagnostics_summary.json
```

This directly addresses failure modes that can otherwise be hidden behind an average R2 or accuracy.

## Dask and Cluster Readiness

STITCHV2 can use Dask:

```bash
--executor dask
--dask-scheduler local|threads|synchronous|jobqueue
--dask-processes
--dask-performance-report
--dask-task-stream
```

Current local default:

```text
dask_processes=False
```

This avoids unnecessary serialization of large Python/JAX objects on local workstations. Future cluster usage can use:

```bash
--dask-scheduler jobqueue
```

with jobqueue-specific options for SLURM or similar schedulers.

## Fixed Plus Mutable Founders

Original STITCH can learn founder haplotypes, and STITCH can use known founder-like panels depending on workflow. STITCHV2 makes hybrid founder behavior explicit:

```text
loaded immutable founders + extra mutable founders
```

If `--n-founders` exceeds the number loaded from VCF/PLINK, extra founders are added as mutable. This supports models such as:

```text
8 immutable founders + 1 mutable founder
```

This is useful when the known founders are mostly right but incomplete.

## Benchmarking Improvements

STITCHV2 writes timing and memory by stage:

- IO/read extraction,
- HMM,
- calibration,
- writing,
- total,
- RSS after stages.

This makes it possible to distinguish:

- BAM reader bottlenecks,
- HMM bottlenecks,
- calibration overhead,
- Parquet/Zarr write overhead,
- Dask scheduler overhead,
- memory leaks versus expected block memory.

Original STITCH can be timed externally, but STITCHV2 exposes stage timing as part of the run output.

## Caveats and Honest Differences

STITCHV2 is not automatically better in every setting.

Important caveats:

- Original STITCH is mature and has carefully tuned behavior.
- STITCHV2 parity requires matching read filters, founder behavior, no-call threshold, fragment behavior, and iteration count.
- Calibration can improve hard calling only if labels and features are informative. It should fall back when it is not useful.
- GPU acceleration helps array math, not raw HTSlib BAM decoding.
- Dense caches can become huge at whole-genome scale; compact evidence should remain the default cache representation.
- Non-diploid pedigree transmission needs careful interpretation.
- Hybrid mutable founders can improve incomplete-founder settings but can also overfit if iterations or damping are poorly chosen.

## Practical Recommendation for STITCH Comparisons

For fair fixed-founder comparisons:

1. Use the same founder panel.
2. Keep founders immutable.
3. Use one or few iterations.
4. Match read filters.
5. Use `fragment_likelihood_mode=replace`.
6. Use `fragment_coupling_model=stitch_parity`.
7. Use STITCH-style no-call for parity.
8. Write haplotype probabilities if comparing haplotype calls.
9. Report runtime by stage for STITCHV2 and total runtime for original STITCH.
10. Evaluate R2, accuracy, balanced accuracy, F1, call rate, INFO, MAF, HWE, missingness, and ROC/AUC where appropriate.

For best STITCHV2 production runs:

1. Build compact evidence cache.
2. Use fixed founders when trusted.
3. Add one or a few mutable founders only when founders may be incomplete.
4. Use enough EM iterations for mutable founders.
5. Use standard callability calibration with per-SNP fallback.
6. Run first-pass pedigree QC before transmission.
7. Use curated pedigree for second-pass kinship or transmission.
8. Review diagnostics before trusting aggregate metrics.
