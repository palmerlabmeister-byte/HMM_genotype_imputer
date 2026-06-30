# STITCHCONT memory-safety & correctness review

**Scope:** review of the `stitchcont` (a.k.a. "stitchv2 / final speed+quality") package for
correctness and memory safety at production scale:

- **N = 50,000 samples**
- **M = 20,000–100,000 variants**
- **K = 8 founders** → ordered diploid states `K² = 64`, unordered `K(K+1)/2 = 36`
- **coverage ≈ 0.25×**

All line numbers refer to the uploaded `stitchcont_final_speed_quality_package`
(`src/stitchcont/...`). The repository branch currently contains the older
`src/stitchv2/` package; the equivalent fixes apply to whichever tree is canonical.

---

## TL;DR

The implementation is **algorithmically reasonable but not memory-safe under default
settings** at the target scale. The dominant cost is the per-cohort HMM tensors, **not**
read coverage — 0.25× sparsity gives no relief because the dense
`samples × variants × states` arrays are materialized in full.

The production recipe documented in `IMPLEMENTATION_NOTES_*` **is** essentially safe:

```
--hmm-backend jax --transition-model stitch_parity --use-unordered-diploid-states
--jax-sample-batch-size 64 --founder-immutable
--output-store zarr --output-minimal --store-xi off --write-gamma off
--hmm-checkpoint-interval 512    + Dask/sample-shard fan-out
```

The risk is that **the defaults do not match that recipe**, and there is **no guard** that
stops a silently-launched run from attempting a multi-hundred-GB to multi-TB allocation.
A run with default flags on 50k samples will OOM by orders of magnitude with no actionable
error.

Severity legend: 🔴 CRITICAL (OOM or wrong results at scale) · 🟠 MAJOR · 🟡 MINOR.

---

## 🔴 CRITICAL

### C1 — The serial executor never batches samples; `effective_sample_batch_size` is dead code
`pipeline.py:649,697` compute and serialize `effective_sample_batch_size` into
`runtime_memory_plan.json`, but **nothing ever reads it** to drive a loop. The common
single-ploidy path calls the HMM on the entire cohort at once:

```python
# pipeline.py:2248
artifacts = self._run_hmm_for_subset(
    ploidy=ploidy_i, founder_panel=block_founders,
    ref_count=evidence.ref_count,   # all N samples
    ...
)
```

`jax_sample_batch_size` only takes effect inside the JAX `stitch_parity` branch
(`hmm.py:_resolve_jax_sample_batch_size`, 4928). For every other backend the sample axis
is unbounded.

### C2 — Default `exact_streaming` mode uses the whole chromosome as one block
`pipeline.py:598-599`:

```python
if str(self.config.snp_block_mode) in {"exact_streaming", "exact_chunked"}:
    return max(int(n_positions), 1)          # full chromosome, ignores memory budget
```

Only the `independent_approx` / `density_balanced_overlap` modes derive a block size from
the memory budget (`pipeline.py:603-607`). The default mode (`config.py:117`) does not.

### C3 — Full-cohort evidence / emission / gamma tensors
- `_build_full_sample_evidence` allocates **seven dense `(N, M)` arrays**
  (`pipeline.py:1497-1504`): 4×uint16 + 3×float32 = 20 B/cell → **~100 GB** at 50k×100k,
  before any HMM math, and independent of coverage.
- `emissions()` builds `log_emission` of shape `(N, M, K, K)` (`hmm.py:592-597`). Because
  `_safe_log_np` uses `np.log` (float64) and is multiplied into the sum, **the returned
  tensor is float64**: **512 GB at M=20k, 2.56 TB at M=100k**. `gamma = np.zeros_like(emission)`
  (`hmm.py:1053` and ~10 sibling sites) **doubles** it.
- `return_full_transition=True` allocates `(N, M, K², K²)` (`hmm.py:5093`) = **16–82 TB**.
- The "exact chunking" only chunks the *position* axis *within a single sample*
  (`hmm.py:968-980`) and is a no-op for M ≤ 20,000. It never reduces the N-dimension and
  does not address the tensors above.

### C4 — The two mitigations are OFF by default
`config.py:75` `use_unordered_diploid_states = False`; `config.py:79`
`hmm_checkpoint_interval = 0`. So:
- The cohort runs the **float64 ordered-state** path (C3), not the leaner 36-state path.
- The "checkpointed forward/backward at interval 512" described in the notes is **dormant**
  by default.
- Even when `use_unordered_diploid_states=True`, `ordered_log_emission_to_unordered`
  (`fast_hmm.py:71-84`) still allocates a full-cohort `(N, M, 36)` = **144–720 GB**.

### C5 — The memory plan is computed but never enforced
`_enforce_snp_block_mode` is a no-op (`pipeline.py:1184-1187`). The plan even estimates
`xi ≈ 8×10¹⁸ bytes` (`pipeline.py:662`) and writes it to JSON as a "note", then proceeds.
A dangerous configuration produces a JSON warning, not a refusal.

### C6 — Dask path disables sample batching whenever founders are mutable
`pipeline.py:3805`:

```python
if force_full_group:          # force_full_group = not all(immutable_mask)
    batches = [idx_all]       # ALL N samples of this ploidy in ONE task
```

Founder training (the normal STITCH case) makes founders mutable, so the Dask executor —
the one path with memory-aware autotuning — collapses to a single all-samples task and
OOMs the worker.

### C7 — C++ `int32` overflow in the rescale loop bound
`native/hs_k8_unordered.cpp:713`:

```cpp
for (int idx = 0; idx < n_samples * n_positions; ++idx) {   // int * int
```

`n_samples` and `n_positions` are `int`; `50000 * 100000 = 5e9` overflows signed 32-bit
(max 2.147e9) → UB / wraps negative → incomplete or skipped rescale. The allocation one line
above (`:664`) correctly casts to `size_t`; only this **loop bound** is wrong. Bites only
when the kernel is called un-chunked (the streaming wrapper keeps `n_samples ≤ 512`), but
it is a latent landmine.

Related: `pybind_hs_k8_unordered.cpp:267-268,532-533` narrow `n_fragments` /
`n_observations` from `ssize_t` to `int`. At 0.25× over the full cohort the total
observation count can exceed 2.1e9, wrapping the bound and bypassing the
`obs_stop > n_observations` guard. `stitchcont_k8_counts_fragments_reduce`
(`hs_k8_unordered.cpp:1406`) also materializes the full `(N, M, 36)` `logu` tensor
(720 GB at scale), defeating the fused/checkpointed design.

### C8 — Calibration "full-stack" builds multiple `N·M × ~37` feature matrices
`calibration.py:1074-1180` (`build_readaware_calibration_feature_matrix`),
`2754-2849` (`build_hardcall_quality_feature_matrix`), driven by
`calibrate_genotype_posterior_full_stack` (`2956`, `3011-3065`):

- Each `np.stack` produces a `5e9 × ~37 × 4 B ≈ 740 GB` matrix, with the ~37 input columns
  *also* live → ~2× transiently.
- **Training rows are subsampled** (`max_train_rows`), but **prediction runs on the full
  matrix** (`3011`), and a *second* full feature matrix is built for hard-call quality
  (`3034`). Several columns are `np.broadcast_to(...).reshape(-1)`, where `.reshape(-1)`
  forces a full copy.

This path is only viable on small training panels.

---

## 🟠 MAJOR

| ID | Location | Issue |
|----|----------|-------|
| M1 | `pipeline.py:4467-4489` | Non-Zarr Parquet output builds the full `N*M`-row long-form table (with repeated `sample_id`/`position` columns) in RAM. Unbounded for whole-chromosome blocks. No guard switches to a matrix layout at scale. |
| M2 | `output.py:243,266` | Zarr float writer non-dask fallback sets the chunk to the **entire array** and `compressor: None` → tries to serialize a 20 TB uncompressed single chunk. |
| M3 | `pipeline.py:1294,2152` | Microarray dosage kept as a full `(N, M)` array on the pipeline instance for the whole run → ~20 GB resident at scale. |
| M4 | `pedigree.py:364-388,421-521` | `kinship` / `transmission` modes hold ~6 full `(N, M, 3)` posteriors (≈300 GB) plus `O(N)` Python loops. Only `postprocess.py`'s `smooth` mode streams in position chunks. |
| M5 | `shards.py:196,221` | Merged sample order is **shard-major, not input order** (mod-based sharding interleaves samples), and cross-shard chunk/position schema is assumed identical without validation. Downstream consumers expecting input order will mis-map samples. |
| M6 | `evidence_cache.py:485-498` | `load_available` accumulates all requested blocks and `merge_sample_blocks` stacks them; requesting all 50k samples for a wide block reconstructs the full `(N, M)` stack in RAM (same ~100 GB as C3). |
| M7 | `hmm.py:984-1018`, `cpu_kernels.py:160-192` | `b = 1 - sw - off` goes **negative** once `sw > (K-1)/K = 0.875` (K=8). `switch` is clipped to `1-1e-10`, so high-recombination intervals get negative transition cross-terms; normalization masks it by reverting to uniform, but intermediate posteriors can be silently wrong rather than merely underflowing. |

---

## 🟡 MINOR

- `pileup.py` `bincount` depth accumulation has no uint16 saturation clip (the cache path
  clips, extraction path wraps) — not triggered at 0.25×.
- `fast_hmm.py:35-36,103` `unordered_transition_from_haploid` is `lru_cache`d on the full
  flattened offdiag matrix; with continuous per-position switch values the cache thrashes →
  O(N·M) Python matrix rebuilds (perf cliff in the pure-numpy unordered path).
- `calibration.py:281-313,580-674` use `for j in range(n_positions)` Python loops
  (per-variant stats / no-call thresholds) → 100k iterations, minutes at M=100k (perf only).
- `calibration.py:2930-2954` per-variant population statistics are computed over **all**
  samples including held-out ones, then used as features — mild train/test leakage for
  honest held-out evaluation (not a crash).
- `calibration.py:20-35,1217` the "preserve feature names" guard is only half-wired: the
  multiclass calibrator path relies on positional column agreement (currently correct
  because the same builder is used at fit and predict, but unprotected).

---

## ✅ Verified correct / bounded

- Checkpointed numba FB buffer is genuinely `O(n_checkpoints · S)`, not `O(M · S)`
  (`hs_k8_unordered.cpp:363-364`).
- All large-buffer **index** arithmetic consistently pre-casts to `size_t`
  (`hs_k8_unordered.cpp:304,331-332,480,691,…`) — only the bare loop bound at :713 and the
  pybind `int` counts are wrong.
- Founder accumulators are correctly `(K, M)` (`founders.py`) — ~3 MB, never `N×M`.
- The xi Zarr writer is correctly sample/interval-batched (`pipeline.py:957,1038`).
- Serial reassembly via per-task `sample_indices` is positionally correct; founder-frequency
  merge is a correct sample-count-weighted average.
- No CV fold leakage (`cv.py:64-72` folds partition positions disjointly).
- The compressed-Zarr shard merge fix (2-D chunk index, no spurious extra axis) is correct
  for matched chunking (`shards.py:221-259`).
- `postprocess.run_pedigree_postprocess_zarr` streams dosage in `chunk_positions` chunks —
  the one place the pipeline does streaming correctly.

---

## Recommended fixes

### Tier 0 — surgical, low-risk, no algorithm change (apply now)

**F1. Make the memory plan fail fast instead of silently OOMing.** Turn
`_enforce_snp_block_mode` (`pipeline.py:1184`) into a real guard: when
`snp_block_mode ∈ {exact_streaming, exact_chunked}` *and* there is no active sample
batching (serial executor with `jax_sample_batch_size ≤ 0`, or Dask with mutable founders),
compare the estimated peak (`n_samples × n_positions × state_count × bytes × small_factor`)
against `planned_budget_bytes` and `raise` a clear, actionable error naming the knobs
(`--jax-sample-batch-size`, `--snp-block-mode independent_approx --block-size`,
`--use-unordered-diploid-states`, `--sample-shard-count`, `--executor dask`). This converts
a multi-TB silent OOM into a one-line diagnostic.

**F2. Emit the emission tensor in float32.** `hmm.py:50`:

```python
def _safe_log_np(x):
    return np.log(np.clip(x, 1e-30, None)).astype(np.float32, copy=False)
```

Halves the dominant tensor (2.56 TB → 1.28 TB at M=100k). Numerically harmless for an
emission log used inside a max-subtracted exp.

**F3. Fix the C++ overflow.** `hs_k8_unordered.cpp:713`:

```cpp
for (long long idx = 0; idx < (long long)n_samples * (long long)n_positions; ++idx) {
    if (!touched[(size_t)idx]) continue;
    float* row = logu + (size_t)idx * S;
    ...
}
```

and widen the pybind counts to `int64_t`:

```cpp
const int64_t n_fragments    = static_cast<int64_t>(fci_b.shape[0]);
const int64_t n_observations = static_cast<int64_t>(fop_b.shape[0]);
```
(propagate `int64_t n_observations` into the C signatures' bound checks).

**F4. Always set a Zarr compressor and a bounded chunk on the float path.**
`output.py:240-266`: never fall back to a whole-array chunk; default to e.g.
`(min(zarr_chunk_samples, shape0), min(zarr_chunk_positions, shape1), …)` and a Blosc/zstd
compressor (matching the integer path), so a non-dask `DataArray` cannot produce a single
multi-TB uncompressed chunk.

### Tier 1 — make the defaults safe (moderate, needs tests)

**F5. Consume `effective_sample_batch_size` in the serial `_run_blocks` path.** Wrap the
single-ploidy and multi-ploidy HMM calls in a
`for s0 in range(0, n_samples, effective_sample_batch_size)` loop, slicing
evidence/fragments per batch and writing each batch's dosage/GP/support straight to the Zarr
store (the Zarr writer's `arr.shape[0] == n_samples_total` assertion at `pipeline.py:4389`
must be relaxed to accept a `(row_start, row_stop)` sub-range). This is the change that makes
a default 50k-sample run safe without requiring the JAX/Dask recipe.

**F6. Pick a memory-aware default `effective_sample_batch_size`** from `planned_budget` and
the per-sample HMM footprint (`state_count × n_positions × bytes × ~4 for alpha/beta/gamma/emit`)
instead of defaulting to all N.

**F7. Cap `exact_streaming`/`exact_chunked` block by the IO/HMM budget** (or document that
these modes *require* sample batching/sharding and have the guard in F1 enforce it).

### Tier 2 — calibration & pedigree at scale (largest diff)

**F8.** Chunk `calibrate_genotype_posterior_full_stack` and the `build_*_feature_matrix`
helpers over positions for **both** feature-build and predict (training already subsamples).
**F9.** Add an `N·M` row-count guard to the long-form Parquet path (M1) that refuses or
switches to matrix layout above a threshold. **F10.** Drive `pedigree` `kinship`/`transmission`
through a position-chunked driver like `postprocess.run_pedigree_postprocess_zarr` instead of
holding full `(N, M, 3)` posteriors.

---

## Bottom line for the target run (50k × 20–100k, 0.25×)

Run it with the documented production recipe **and** add the Tier-0 guard (F1) so an
accidental default run fails loudly instead of OOMing. With JAX `stitch_parity`,
`--jax-sample-batch-size 64`, immutable founders, unordered diploid states,
`--output-store zarr --output-minimal`, and sample sharding / Dask fan-out, the per-worker
footprint is bounded to roughly `batch × M × 36 × 4 B × O(few)` ≈ a few GB per batch — safe.
Without those flags, the defaults are not safe at this scale.
