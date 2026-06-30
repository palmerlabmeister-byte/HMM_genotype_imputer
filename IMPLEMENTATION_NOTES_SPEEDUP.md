# STITCHCONT speed/usability implementation notes

Implemented in this patched package:

1. JAX fragment-bucket padding
   - Fragment and observation arrays are padded to power-of-two buckets when `jax_bucket_batch_shapes=True`.
   - Padded fragments use center `-1` and padded observations use code `-127`.
   - JAX fragment kernels mask padded records so replace-mode emissions are not altered.
   - This reduces JAX/XLA compile cardinality from almost one compile per fragment-shape to a small set of bucketed shapes.

2. Safer JAX fragment masking
   - Diploid, haploid, and polyploid JAX fragment apply kernels now reject invalid sample/position centers before segment aggregation.
   - Padded observations contribute zero log-likelihood.

3. Factorized transition memory guard
   - Added `--transition-factor-init-max-cells` / `transition_factor_init_max_cells`.
   - Default: 50,000,000 sample × SNP × K cells.
   - If exceeded, factorized transition initialization is skipped and the run falls back to STITCH-parity transitions with diagnostics instead of attempting a huge posterior pass.
   - Set to `0` to disable the guard.

4. Presets
   - Added `--preset hs-rat-k8-validate`.
   - Added `--preset hs-rat-k8-production`.
   - Added `--preset hs-rat-k9-experimental`.
   - Added `--preset debug-small`.
   - Presets set conservative HMM/calling defaults and increase JAX sample batch size to at least 32 for HS-rat production/validation modes.

5. `doctor` command
   - Added `stitchcont doctor` for preflight checks.
   - Checks position sorting/duplicates, sample columns, generation values, founder count mismatches, extra mutable founder warnings, and basic evidence-cache presence.

Recommended first validation command:

```bash
stitchcont run \
  --preset hs-rat-k8-validate \
  --samples /tscc/projects/ps-palmer/gwas/STITCHCONT/samples_fullpath.parquet \
  --positions /tscc/projects/ps-palmer/gwas/STITCHCONT/chr12positions.parquet \
  --chromosome chr12 \
  --output-dir /tscc/projects/ps-palmer/gwas/STITCHCONT/run_chr12_k8_validate \
  --founder-plink /tscc/projects/ps-palmer/gwas/STITCHCONT/founders8 \
  --compact-evidence-cache-dir /tscc/projects/ps-palmer/gwas/STITCHCONT/run_chr12_k8_validate/compact_evidence \
  --compact-evidence-cache-mode readwrite \
  --jax-persistent-cache-dir /tscc/projects/ps-palmer/gwas/STITCHCONT/run_chr12_k8_validate/jax_cache \
  --hmm-backend jax \
  --read-mode read_stream \
  --read-stream-backend auto \
  --io-workers 5 \
  --htslib-threads-per-file 8 \
  --max-mem 72GB \
  --executor serial
```

Recommended preflight:

```bash
stitchcont doctor \
  --samples /tscc/projects/ps-palmer/gwas/STITCHCONT/samples_fullpath.parquet \
  --positions /tscc/projects/ps-palmer/gwas/STITCHCONT/chr12positions.parquet \
  --chromosome chr12 \
  --output-dir /tmp/stitchcont_doctor \
  --founder-plink /tscc/projects/ps-palmer/gwas/STITCHCONT/founders8 \
  --n-founders 8 \
  --founder-immutable
```
