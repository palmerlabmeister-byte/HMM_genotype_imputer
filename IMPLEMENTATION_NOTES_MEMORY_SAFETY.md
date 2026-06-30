# STITCHCONT memory-safety & correctness changes

Companion to `MEMORY_SAFETY_REVIEW.md` (the audit). This document records the
fixes applied to make `stitchcont` safe and correct at production scale:
**N = 50,000 samples, M = 20,000–100,000 variants, K = 8 founders, ~0.25× coverage.**

The package was imported into this repo as `src/stitchcont/` and made the named
project (`pyproject.toml` → `stitchcont`, with the pybind native build in
`setup.py`). The older `src/stitchv2/` package is left untouched.

---

## Summary of the problem

At the target scale the dominant cost is the per-cohort HMM tensors, not read
coverage. The default configuration (serial executor, `exact_streaming`,
`jax_sample_batch_size = 0`) processed **all 50,000 samples in one HMM call over
the whole chromosome**, materializing `(N, M, K, K)` emission and gamma tensors
(2.56 TB at M = 100k, in float64). The computed memory plan even estimated these
multi-TB allocations and then ran anyway. Sample batching existed only on the JAX
`stitch_parity` path; the field `effective_sample_batch_size` was dead code on the
serial path.

---

## Changes applied

### F1 — Fail-fast memory guard (`pipeline.py`, `config.py`, `cli.py`)
`_enforce_snp_block_mode` was a no-op. It now estimates the peak working set for a
block and **raises `MemoryError` with actionable guidance** when it exceeds the
planned budget, instead of OOMing silently. The peak is split into:

- **persistent** bytes/cell (full-N: dense read evidence + output dosage/GP/hap
  matrices) — only bounded by sample sharding;
- **transient** bytes/cell (batch-bounded: forward/backward `alpha/beta/gamma/emission`).

`peak = block × (n_samples × persistent + batch × transient)` is stored on the
plan (`estimated_peak_bytes`) so the guard and the auto-batch logic agree. The
message names the relevant knobs (`--jax-sample-batch-size`, `--sample-shard-count`,
`--snp-block-mode independent_approx --block-size`, `--use-unordered-diploid-states`,
`--output-minimal`, `--max-mem`). A new `--allow-memory-overcommit` flag downgrades
the error to a warning for users who accept the risk.

### F5 — Serial HMM sample-batching (`pipeline.py`, `hmm.py`)
`_run_hmm_for_subset` now consumes the resolved sample-batch size: when samples are
**conditionally independent** (immutable founders, `em_iterations == 1`, no
transition-factor training — `_samples_are_independent`), it runs `hmm.run()` over
sample batches and reassembles the result with `_concat_hmm_artifacts`
(per-sample arrays concatenated on axis 0; shared per-position arrays taken from the
first batch). This bounds the dominant emission/gamma tensor to the batch size.

The batch size is auto-derived from the budget for immutable-founder runs (or taken
from `--jax-sample-batch-size` when set), and is the same value the F1 guard uses.
For **mutable founders / EM**, samples are coupled, so batching is disabled and the
guard directs the user to sample sharding (the exact, sample-independent split).

`autotune_backend` previously probed with the *entire* cohort when
`autotune_samples/positions` were 0; it now defaults to a bounded probe
(256 samples × 4096 positions) so backend selection cannot itself OOM.

### F2 — float32 emission tensor (`hmm.py`)
`_safe_log_np` returned float64 (`np.log`), so the `(N, M, K, K)` emission tensor was
silently float64. It now casts to float32, halving the dominant allocation. The log
is only consumed inside a max-subtracted exp, so the cast is numerically harmless.

### F3 — Native int overflow (`native/hs_k8_unordered.{cpp,hpp}`, `native/pybind_hs_k8_unordered.cpp`)
- The rescale loop bound `n_samples * n_positions` was computed in 32-bit `int`
  (5e9 overflows at 50k × 100k, silently skipping the rescale). It now uses a 64-bit
  signed `n_cells` computed before the OpenMP pragma.
- The C API counts `n_fragments` / `n_observations` and the pybind casts were
  widened to `int64_t`, so >2.1e9 observations no longer wrap and bypass the
  `obs_stop > n_observations` bounds guard.

Verified with `g++ -std=c++17 -fopenmp -fsyntax-only`.

### F4 — Zarr float writer (`output.py`)
`_write_float_dataarray_zarr` previously fell back to a **single whole-array chunk**
with `compressor: None` for non-dask arrays — a ~20 TB uncompressed single chunk at
scale. Now:
- chunks are bounded (`_default_chunks`: caps samples at 256, positions at 4096);
- a Blosc/zstd compressor is always set;
- partial edge chunks are padded to the declared chunk shape before encoding
  (also fixes a latent edge-chunk read bug in the uncompressed path).

### F9 — Long-form Parquet guard (`pipeline.py`)
When `output_store != "zarr"`, the output writer builds a dense `N*M`-row long-form
table (with repeated `sample_id`/`position`) in one allocation. The F1 guard now also
refuses this early when the estimated row build exceeds the budget, directing the user
to `--output-store zarr` (chunked + compressed matrices).

### M5 — Shard merge validation & ordering contract (`shards.py`)
`merge_sharded_matrix_zarr` assumed all shards shared positions, dtype, and
non-sample chunking without checking. It now **validates** each shard's array dtype,
non-sample shape, and non-sample chunking against the reference shard and raises on
mismatch (the merge copies axes ≥2 as whole chunks, so a chunking mismatch would
silently misplace data). The merged store's attrs now document that rows are in
shard-concatenation order and that consumers must map rows by `sample_ids.json`, not
by assuming original input order (mod-based sharding interleaves input rows).

---

## Investigated and found NOT to be a bug

**M7 (negative transition mass).** The parity transition step
`pred = b²·prev + b·off·(row+col) + off²·total`, with `off = sw/(k-1)` and
`b = 1 - sw - off`, has `b < 0` once `sw > (k-1)/k = 0.875`. This is **not** a
correctness bug: the form is algebraically identical to `Σ T_hap[i→c]·T_hap[j→d]·prev[i,j]`
where `T_hap[a→c] = b·δ + off` has diagonal `b + off = 1 - sw ≥ 0` and off-diagonal
`off ≥ 0` — a valid non-negative stochastic matrix for any `sw ∈ [0,1]`. Verified
numerically: the factored result matches the explicit double sum to ~1e-16 and stays
non-negative even at `sw = 0.999999`. No change made; clamping `b` would have
introduced an approximation.

---

## Remaining work (documented, not yet applied)

These are large, higher-risk refactors deferred for review with test coverage. Until
applied, the F1 guard prevents them from OOMing silently, and sample sharding bounds
their footprint.

- **Calibration full-stack chunking** (`calibration.py`): `calibrate_genotype_posterior_full_stack`
  and `build_*_feature_matrix` materialize multiple `N·M × ~37` feature matrices
  (600–740 GB each) — training is subsampled but **prediction is run on the full
  matrix**. These need position-chunking on the feature-build and predict paths.
  Until then, run calibration on a subsampled/sharded panel.
- **Pedigree `kinship`/`transmission`** (`pedigree.py`): hold ~6 full `(N, M, 3)`
  posteriors plus O(N) Python loops. They need a position-chunked driver like
  `postprocess.run_pedigree_postprocess_zarr` (the one path that streams correctly).
- **Microarray dosage** (`pipeline.py`): a full `(N, M)` matrix kept for the whole
  run (~20 GB at scale) — slice per block instead of holding it whole.

---

## Recommended production invocation (50k × 20–100k, 0.25×)

```bash
stitchcont impute-from-evidence \
  --hmm-backend jax --transition-model stitch_parity --use-unordered-diploid-states \
  --jax-sample-batch-size 64 --founder-immutable \
  --output-store zarr --output-minimal --store-xi off --write-gamma off \
  --hmm-checkpoint-interval 512 \
  --sample-shard-count S --sample-shard-index I    # one run per shard for full-cohort M=100k
```

With these changes a misconfigured default run now fails fast with guidance rather
than OOMing; immutable-founder serial runs auto-batch the HMM; and the dense Zarr
matrix output is chunked and compressed.

---

## Validation performed

No project dependencies are installable in this environment (offline index), so the
full pytest suite could not be run. Validation done:

- `python -m py_compile` on every `src/stitchcont/*.py` (all pass).
- `g++ -std=c++17 -fopenmp -fsyntax-only` on the native kernel (passes).
- Standalone numeric check of the parity factorization (M7) confirming exactness and
  non-negativity.

The fixes are designed to be behavior-preserving for already-safe configurations
(JAX batched / sharded production runs); they change behavior only for the previously
unsafe default paths (fail-fast guard, serial auto-batching, float32 emission,
compressed/bounded Zarr).
