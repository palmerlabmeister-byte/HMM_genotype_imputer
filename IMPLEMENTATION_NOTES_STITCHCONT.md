# STITCHCONT implementation notes

This package is the renamed `stitchcont` package.  After `pip install -e .`, the console entry point is:

```bash
stitchcont --help
```

The old `stitchv2` console script is intentionally not installed by this package name.

## Implemented speed and usability changes

- Package/project rename from `stitchv2` to `stitchcont`.
- JAX fragment arrays are bucket-padded to reduce XLA recompilation from variable fragment/observation counts.
- JAX fragment kernels mask invalid padded fragment centers and observations.
- Factorized-transition initialization is now sample-batched.  It accumulates adjacent haplotype-posterior products across sample batches instead of materializing a full sample × SNP × K posterior over all samples.
- Known-founder K8 presets now disable dense-count materialization and diagnostics together, avoiding unsafe support-only diagnostics.
- `stitchcont doctor` now reports JAX devices and warns when JAX is CPU-only.
- `--require-jax-accelerator` can make GPU jobs fail early if CUDA-enabled `jaxlib` is not active.
- `stitchcont evidence-build` extracts and caches compact read/fragment evidence without running the HMM.
- `stitchcont impute-from-evidence` runs HMM imputation in strict read-only evidence-cache mode.
- Read-only evidence-cache mode now fails if any requested sample is missing from the cache instead of silently extracting BAMs or imputing from empty evidence.
- `stitchcont validate` computes dosage validation against a PLINK truth panel, including per-variant and MAF-bin summaries.
- LightGBM/sklearn prediction now preserves feature names when available to avoid feature-name warnings and reduce the chance of feature-order mismatches.
- Source distribution was cleaned by removing Python caches and compiled shared objects.

## Important performance note from the user log

The previous run reported:

```text
An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
```

That means `--hmm-backend jax` ran on CPU.  This is expected to be much slower and explains why a 1,500 sample × 20,000 SNP run took minutes and consumed high RAM.  On TSCC GPU nodes, install a CUDA-enabled JAX build in the active environment and run with `--require-jax-accelerator`.

## Recommended validation sequence

```bash
stitchcont doctor \
  --samples /tscc/projects/ps-palmer/gwas/STITCHV2/samples_fullpath.parquet \
  --positions /tscc/projects/ps-palmer/gwas/STITCHV2/chr12positions.parquet \
  --chromosome chr12 \
  --output-dir /tmp/stitchcont_doctor \
  --founder-plink /tscc/projects/ps-palmer/gwas/STITCHV2/founders8 \
  --founder-immutable \
  --n-founders 8
```

Build evidence once:

```bash
stitchcont evidence-build \
  --preset hs-rat-k8-validate \
  --samples /tscc/projects/ps-palmer/gwas/STITCHV2/samples_fullpath.parquet \
  --positions /tscc/projects/ps-palmer/gwas/STITCHV2/chr12positions.parquet \
  --chromosome chr12 \
  --output-dir /tscc/projects/ps-palmer/gwas/STITCHV2/stitchcont_chr12_evidence \
  --compact-evidence-cache-dir /tscc/projects/ps-palmer/gwas/STITCHV2/stitchcont_chr12_evidence/compact_evidence \
  --compact-evidence-cache-mode write \
  --stitch-compat \
  --io-workers 5 \
  --htslib-threads-per-file 8 \
  --max-mem 72GB
```

Then run the clean K8 HMM from cached evidence:

```bash
stitchcont impute-from-evidence \
  --preset hs-rat-k8-validate \
  --samples /tscc/projects/ps-palmer/gwas/STITCHV2/samples_fullpath.parquet \
  --positions /tscc/projects/ps-palmer/gwas/STITCHV2/chr12positions.parquet \
  --chromosome chr12 \
  --output-dir /tscc/projects/ps-palmer/gwas/STITCHV2/run_chr12_stitchcont_k8 \
  --founder-plink /tscc/projects/ps-palmer/gwas/STITCHV2/founders8 \
  --founder-immutable \
  --compact-evidence-cache-dir /tscc/projects/ps-palmer/gwas/STITCHV2/stitchcont_chr12_evidence/compact_evidence \
  --hmm-backend jax \
  --require-jax-accelerator \
  --jax-sample-batch-size 64 \
  --jax-persistent-cache-dir /tscc/projects/ps-palmer/gwas/STITCHV2/run_chr12_stitchcont_k8/jax_cache \
  --stitch-compat \
  --max-mem 72GB \
  --executor serial
```

Validate against a truth PLINK panel:

```bash
stitchcont validate \
  --run-output-dir /tscc/projects/ps-palmer/gwas/STITCHV2/run_chr12_stitchcont_k8 \
  --truth-plink /path/to/validation_truth_prefix \
  --chromosome chr12 \
  --output-dir /tscc/projects/ps-palmer/gwas/STITCHV2/run_chr12_stitchcont_k8_validation
```

## 2026-06 exact chunking, Zarr, validation audit update

New production-oriented switches:

```bash
--snp-block-mode exact_chunked \
--hmm-chunk-size 20000 \
--output-store zarr \
--zarr-chunk-samples 256 \
--zarr-chunk-positions 4096 \
--require-jax-accelerator
```

`exact_chunked` keeps the full chromosome HMM exact, but stores boundary alpha/beta messages and recomputes alpha/beta inside SNP chunks for posterior emission. It is not the old independent-block approximation.

Matrix outputs are written directly to `matrices.zarr/` when `--output-store zarr` is used. This avoids long-form `sample_id, position, dosage` Parquet for dense dosage/GP/call/support matrices. Metadata tables such as recombination, transition summaries, founder updates, diagnostics, and timing remain Parquet/JSON.

Validation now audits allele orientation before r². PLINK BED decoding in STITCHCONT treats hardcall dosage as the count of the `.bim` A2 allele. If run ALT equals PLINK A1 / run REF equals PLINK A2, validation flips truth dosage as `2 - truth`; incompatible allele mismatches are masked and reported in `allele_orientation_audit.parquet`.

A TSCC helper for CUDA JAX is included at:

```bash
scripts/install_cuda_jax_tscc.sh
```

After installing, use `stitchcont doctor` and production runs with `--require-jax-accelerator` so CPU fallback fails early instead of silently running slowly.
