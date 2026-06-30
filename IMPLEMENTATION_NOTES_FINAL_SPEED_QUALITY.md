# STITCHCONT final speed/quality notes

## Implemented in final speed/quality patch

- Native K8 checkpointed forward/backward for the C++ unordered `stitch_parity` backend.
- Pipeline routing passes `--hmm-checkpoint-interval` into the native K8 count-fused and precomputed-log-emission reducers.
- Shard merger now correctly reads and writes Zarr v2 chunks with `zlib` compression.
- Shard merger now preserves 2-D chunk indices correctly instead of adding a spurious extra axis.
- Native backend status now reports checkpointed K8 availability.

## Best production baseline

Use this for HS-rat K8 fixed-founder production before testing approximate transition models:

```bash
--hmm-backend cpp \
--transition-model stitch_parity \
--use-unordered-diploid-states \
--output-store zarr \
--output-minimal \
--store-xi off \
--write-gamma off \
--hmm-checkpoint-interval 512
```

## Speed improvements already implemented across recent patches

1. Sample sharding for immutable-founder exact runs.
2. Exact SNP chunking and checkpointed CPU smoothing.
3. Numba unordered K8 CPU reducer.
4. pybind11 C++ K8 unordered reducer.
5. C++ fused count-evidence to emission to HMM path.
6. C++ K8 fragment application and count+fragment fused path.
7. C++ low-rank algebraic transition reducer.
8. C++ low-rank count-fused reducer.
9. C++ sparse top-k reducer and count-fused reducer.
10. Quantized/compressed Zarr matrix output.
11. Bitpacked support-mask output.
12. Autotuner for shard/chunk/checkpoint/output dtype planning.
13. Native streaming helper for sample-chunked K8 output to Zarr.
14. Generation binning and recombination diagnostics.
15. Correct compressed-Zarr shard merging.

## Highest-value quality improvements to use or add next

1. Always run `stitchcont doctor` and `stitchcont validate` with allele-orientation audit.
2. Prefer K8 immutable `stitch_parity` as the validation baseline.
3. Validate low-rank and sparse transitions against K8 `stitch_parity` before production.
4. Use a rat genetic map instead of constant cM/Mb when available.
5. Run `recombination-diagnostics` to check generation-scaled switch probabilities.
6. Validate by MAF, depth/support, founder-informativeness, and genomic window.
7. Apply `pedigree-postprocess` after merging sample shards, not inside independent shards, unless shard halos include relatives.
8. Add pedigree halo sharding if pedigree smoothing must run before merge.
9. Add founder/sample contamination diagnostics: founder posterior entropy, sample-specific dosage residuals, and mismatch rate versus truth genotypes.
10. Add per-site founder-informativeness and support-stratified filters before GWAS.

## Remaining non-obvious optimization opportunities

- Direct native reading from compact evidence cache would require stabilizing the on-disk evidence schema and moving Arrow/Zarr decoding into C++ or Rust. This is possible but substantially larger than the HMM-kernel work.
- True transition-power compression for long neutral intervals is possible but requires deriving exact closed-form powers for the parity transition or using a numerically stable matrix-power checkpoint scheme. Current code performs exact neutral-emission skipping but still scans positions.
- Hand-coded AVX for all 36-state K8 transition/reduction loops may add another 1.2-2.5x on AVX2/AVX-512 CPUs, but the current compiler-vectorized native path is the safer baseline.
- A native Zarr writer could reduce Python I/O overhead, but keeping Zarr metadata/chunk compression in Python is more portable and easier to debug.
