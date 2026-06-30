# stitchcont CPU/GPU speed implementation notes

This patch adds CPU-first speed paths and keeps the JAX/GPU path optional.

## New run controls

- `--output-minimal`: reduce posteriors to dosage/GP/calls and avoid gamma/xi-style intermediates where possible.
- `--no-known-founder-fast-path`: disable the known-founder optimization. By default, all-immutable founders force a single posterior pass and disable transition-factor training. With `--output-minimal`, xi/gamma-style outputs are also disabled.
- `--sample-shard-index` / `--sample-shard-count`: split immutable-founder runs by sample rows. This is exact because samples are independent conditional on fixed founders and model parameters.
- `--use-unordered-diploid-states`: CPU NumPy fast path using K(K+1)/2 unordered diploid states instead of K^2 ordered states. For K=8 this reduces the state count from 64 to 36.
- `--generation-bin-size`: round sample generations to bins before computing switch probabilities. Use only after validating dosage r2.
- `--transition-model low_rank_linear`: experimental algebraic low-rank transition update. Uses low-rank transition identities in the CPU HMM; benchmark against dense baselines.
- `--transition-model sparse_factorized` and `--transition-sparse-top-k`: prune each source founder to top-M destination founders after factor fitting. This is approximate and must be validated.

## Recommended CPU validation run

```bash
stitchcont impute-from-evidence \
  --preset hs-rat-k8-validate \
  --samples samples_fullpath.parquet \
  --positions chr12positions.parquet \
  --chromosome chr12 \
  --output-dir run_chr12_k8_cpu_fast \
  --founder-plink founders8 \
  --compact-evidence-cache-dir evidence_chr12/compact_evidence \
  --hmm-backend numpy \
  --snp-block-mode exact_chunked \
  --hmm-chunk-size 20000 \
  --output-store zarr \
  --output-minimal \
  --use-unordered-diploid-states \
  --em-iterations 1 \
  --transition-model stitch_parity \
  --store-xi off \
  --write-gamma off \
  --max-mem 72GB
```

## Sample sharding

Run one job per shard. Example for 8 shards:

```bash
for i in 0 1 2 3 4 5 6 7; do
  stitchcont impute-from-evidence ... \
    --output-dir run_chr12_k8_cpu_fast/shard_${i} \
    --sample-shard-index ${i} \
    --sample-shard-count 8 &
done
wait
```

Each shard writes `sample_shard.json` and preserves the original row indices for later concatenation.

## Benchmark suite

The benchmark command measures ordered, exact-chunked, unordered, sparse-factorized, low-rank algebraic, and optional JAX count-kernel paths.

```bash
stitchcont benchmark \
  --output-dir bench_chr12_like \
  --n-samples 256 \
  --n-positions 20000 \
  --n-founders 8 \
  --hmm-chunk-size 2000 \
  --repeat 3 \
  --no-jax
```

Use `--no-jax` for CPU-only benchmarking. Omit it to include available JAX devices.

## Validation requirements

The approximate modes (`low_rank_linear`, `sparse_factorized`, generation binning) must be benchmarked against `stitch_parity` using `stitchcont validate` with allele-orientation audit enabled. Do not use approximate transition modes for production until dosage r2, GP entropy, call rate, and MAF-binned r2 are stable.
