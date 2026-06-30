# Native fused CPU speed patch

This patch adds the next round of CPU-first speed work for STITCHCONT.

## Added native pybind11 functions

The install-time extension `stitchcont.native._hs_k8_unordered` now exposes:

- `run_k8_unordered_reduce(log_emission_u, switch, founder_alt, n_threads=0)`
- `run_k8_counts_reduce(ref_obs, alt_obs, other_obs, switch, founder_alt, sequencing_error_rate, min_emission_prob, n_threads=0)`
- `build_k8_logu_from_counts(ref_obs, alt_obs, other_obs, founder_alt, sequencing_error_rate, min_emission_prob, n_threads=0)`
- `apply_fragments_unordered_inplace(log_emission_u, fragment arrays..., founder_alt, ...)`

The `run_k8_counts_reduce` path fuses count-emission construction and the K8 unordered diploid HMM reduction in C++, avoiding the ordered KxK log-emission tensor for immutable K8 `stitch_parity` runs.

When fragment likelihoods are present, the package now builds unordered K8 emissions directly in C++, applies fragment likelihoods directly to the unordered 36-state emission tensor, and then calls the native HMM reducer.

## Added output/storage helpers

- `--zarr-bitpack-support-mask` writes per-block bitpacked support masks under `support_mask_bitpacked/` instead of dense bool support-mask Zarr arrays.
- `stitchcont evidence-shard-layout` writes sample-shard manifests and sample tables for a shared compact evidence cache.
- `stitchcont pedigree-postprocess` applies pedigree dosage smoothing chunkwise to a merged/sharded `matrices.zarr` dosage array.

## Benchmark additions

`stitchcont benchmark` now reports native C++ fused count-emission + HMM timings when the pybind11 extension is available:

- `cpp_fused_counts_unordered_k8`
- `cpp_logu_unordered_k8`

## Recommended CPU production mode

```bash
stitchcont impute-from-evidence \
  --preset hs-rat-k8-production \
  --hmm-backend cpp \
  --use-unordered-diploid-states \
  --transition-model stitch_parity \
  --output-store zarr \
  --output-minimal \
  --zarr-dosage-dtype float16 \
  --zarr-gp-dtype uint16 \
  --zarr-bitpack-support-mask
```

The native fused path is intentionally exact only for immutable K8 diploid `stitch_parity` runs. Unsupported combinations fall back to the existing generic/Numba code paths.
