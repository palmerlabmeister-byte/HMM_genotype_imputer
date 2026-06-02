# STITCHV2 Cached Files Deep Dive

STITCHV2 can cache read evidence so repeated runs do not repeatedly decode the same BAM/CRAM files. This is one of the most important practical speed features for benchmarking and model development, because BAM reading often dominates runtime once the JAX HMM is compiled and block sizes are sensible.

The preferred cache format is:

```text
parquet_zarr
```

`parquet_zarr` is the production cache format. Compact fragment evidence is stored in compressed Parquet-style partitions, and dense arrays, when explicitly requested, are stored in chunked compressed Zarr. Do not use or document NPZ as a production cache format.

## Why Cache Evidence?

The same raw BAM evidence is reused across many tasks:

- STITCHV2 parity run.
- STITCHV2 calibrated run.
- fixed-founder vs variable-founder comparisons.
- extra mutable founder sweeps.
- no-call threshold sweeps.
- pedigree first-pass and second-pass runs.
- diagnostics and benchmark plotting.

Without caching, every run reopens BAMs, loads headers and indexes, walks CIGAR strings, and reconstructs the same allele evidence. With caching, the expensive BAM read stage is done once for a given sample set, chromosome/window, and position table.

## Cache Controls

Important CLI flags:

```bash
--compact-evidence-cache-dir <cache_dir>
--compact-evidence-cache-mode off|read|write|readwrite
--compact-evidence-cache-format parquet_zarr
--compact-evidence-cache-sample-batch-size 256
--compact-evidence-no-dense-counts
--compact-evidence-cache-include-dense-counts
```

Modes:

| Mode | Meaning |
| --- | --- |
| `off` | Do not use the compact evidence cache. |
| `read` | Use existing cached evidence; missing samples must be read some other way or cause fallback depending on pipeline path. |
| `write` | Build cache from BAM/CRAM evidence but do not read preexisting cache. |
| `readwrite` | Use cached evidence where present and read/write only missing samples. |

Recommended pattern:

```bash
# First run on a chromosome/window:
--compact-evidence-cache-mode readwrite

# Later parameter sweeps on the same target variants:
--compact-evidence-cache-mode read
```

## Preferred Cache Design

The current scalable cache stores:

- compact fragment evidence in Parquet,
- sample-level fragment summaries in Parquet,
- bitpacked support masks in Parquet,
- optional dense count matrices in Zarr.

This layout is intentionally different from dense VCF/BCF-like outputs. It is built for sparse low-coverage evidence and repeated model runs.

## Directory Layout

For one chromosome and block, the layout is:

```text
cache_root/
  chrom=<chromosome>/
    block=<block_id>_rows=<row_start>-<row_stop>/
      manifest.json
      positions.parquet
      fragments/
        part-<timestamp>-<hash>.parquet
      summary/
        part-<timestamp>-<hash>.parquet
      support/
        part-<timestamp>-<hash>.parquet
      dense/
        part-<timestamp>-<hash>.zarr/
```

Example:

```text
cache/
  chrom=chr12/
    block=000000_rows=0-1491/
      manifest.json
      positions.parquet
      fragments/
      summary/
      support/
      dense/
```

The block row range is based on the target position table, not raw physical base pairs. A cache block is valid only for the same ordered target variants.

## `manifest.json`

The manifest records the identity and contents of the cache block.

Important fields:

| Field | Meaning |
| --- | --- |
| `format` | Cache format string, currently `stitchv2.partitioned_evidence_cache`. |
| `version` | Cache format version. |
| `chromosome` | Chromosome label. |
| `block_id` | Block ID. |
| `row_start` | First position-table row included. |
| `row_stop` | One-past-last position-table row included. |
| `n_positions` | Number of target positions. |
| `positions_hash` | Hash of target `POS`, `REF`, and `ALT`. |
| `sample_batch_size` | Intended sample batch partition size. |
| `compression` | Compression codec used for Parquet. |
| `samples` | Cached sample IDs. |
| `parts` | Written cache parts. |

The positions hash is critical. If positions, reference alleles, alternate alleles, or ordering change, the cache should not be reused silently.

## `positions.parquet`

This stores the target positions for the cache block.

Columns:

| Column | Meaning |
| --- | --- |
| `CHR` | Chromosome. |
| `POS` | Position. |
| `REF` | Reference allele. |
| `ALT` | Alternate allele. |

When loading cache, STITCHV2 verifies this table against the current run positions using the positions hash.

## `fragments/*.parquet`

This is the core compact evidence table.

Columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Sample ID. |
| `sample_index` | int | Sample index within the run or cache part. |
| `fragment_index` | int | Fragment ID within the sample. |
| `center_idx` | int | Target SNP index used as the fragment center. |
| `obs_order` | int | Observation order within the fragment. |
| `pos_idx` | int | Target SNP index of this observation. |
| `obs_code` | int | Encoded allele observation. |
| `obs_qual` | float | Observation quality or quality-derived weight. |

This table is sparse. It stores only observed fragment-SNP intersections, not every sample-SNP pair.

### Observation Codes

The exact allele code interpretation is internal, but conceptually:

- reference observations,
- alternate observations,
- other/non-ref-non-alt observations,
- missing or filtered evidence are separated before the HMM.

When `--ref-alt-only` is enabled, non-REF/non-ALT bases are ignored rather than treated as OTHER evidence.

## `summary/*.parquet`

Per-sample summary table.

Columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Sample ID. |
| `sample_index` | Sample index. |
| `n_overlapping_reads` | Reads overlapping target variants or windows. |
| `n_fragments` | Number of retained fragments. |
| `n_fragment_observations` | Number of retained fragment observations. |

Use this table to debug sample-specific IO problems, unexpectedly low coverage, or samples that were present in metadata but produced no evidence.

## `support/*.parquet`

Bitpacked support masks.

Columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Sample ID. |
| `sample_index` | Sample index. |
| `n_positions` | Number of target positions represented. |
| `bitorder` | Bitpacking order, currently little-endian. |
| `support_packed` | Packed boolean vector where true means `depth > 0`. |

The support mask is used for:

- calibration labels,
- diagnostics,
- identifying directly read-backed calls,
- reconstructing whether a no-call happened at an imputed site or a supported site.

Bitpacking keeps this small even for many samples and variants.

## Optional `dense/*.zarr`

Dense arrays are optional because they can be much larger than compact evidence. When included, they live under:

```text
dense/<part_id>.zarr/
```

Arrays:

| Array | Meaning |
| --- | --- |
| `ref_count` | Dense REF count matrix. |
| `alt_count` | Dense ALT count matrix. |
| `other_count` | Dense OTHER count matrix. |
| `depth` | Dense total depth matrix. |
| `ref_weight` | Quality-weighted REF evidence. |
| `alt_weight` | Quality-weighted ALT evidence. |
| `other_weight` | Quality-weighted OTHER evidence. |

The Zarr store uses chunking by sample and SNP block. It is intended for cases where dense counts are needed repeatedly, such as diagnostics or `fragment_likelihood_mode=augment`.

Important distinction:

- Compact Parquet fragment evidence is best for `fragment_likelihood_mode=replace`.
- Dense Zarr arrays can speed count-based workflows and augment mode but increase disk usage.

## Cache Loading and Partial Hits

STITCHV2 can load cache for some samples and read BAMs for missing samples in `readwrite` mode.

The loader validates:

- cache format and version,
- chromosome,
- block ID and row range,
- positions hash,
- sample IDs,
- available cache parts.

It then reports statistics such as:

- cached samples,
- missing samples,
- partial hit status,
- cache file bytes,
- Parquet bytes,
- Zarr bytes,
- whether dense counts were included,
- whether dense counts were materialized.

Useful timing/stat fields can appear in run summaries:

| Field | Meaning |
| --- | --- |
| `compact_cache_mode` | Requested cache mode. |
| `compact_cache_format` | `parquet_zarr`. |
| `compact_cache_hit` | Whether all requested samples were cached. |
| `compact_cache_partial_hit` | Whether only some samples were cached. |
| `compact_cache_missing_samples` | Count or list of missing samples. |
| `compact_cache_cached_samples` | Count or list of cached samples. |
| `compact_evidence_bytes` | In-memory compact evidence size. |
| `dense_evidence_bytes` | In-memory dense evidence size, if materialized. |
| `partitioned_cache_file_mb` | Disk size of the partitioned cache. |

## Adding New Samples

The cache is sample-partitioned. If new samples are added later:

1. Keep the same positions table and chromosome/window.
2. Add new samples to `samples.parquet` with valid `sample_id` and `bam_path`.
3. Run with `--compact-evidence-cache-mode readwrite`.
4. STITCHV2 checks which sample IDs are already cached.
5. Cached samples are loaded from Parquet/Zarr.
6. Missing samples are decoded from BAM/CRAM and appended as new cache parts.

This avoids rebuilding evidence for all existing samples when only a few new BAMs are added.

Important caveat: if a sample's BAM changes but its `sample_id` stays the same, the cache may still consider that sample present. Cache invalidation is based primarily on target positions and sample IDs, not full BAM content hashing. If the BAM content changes, remove or rebuild that sample's cache part.

## Cache Invalidation Rules

Rebuild the cache when any of these change:

- target positions,
- REF/ALT alleles,
- target position order,
- chromosome label in a non-equivalent way,
- BAM content for cached samples,
- read filters that alter retained evidence,
- read backend behavior if comparing backend parity,
- fragment center or fragment merge logic,
- `ref_alt_only`, base-quality, mapping-quality, or insert-size filters.

It is usually safe to reuse cache when changing only:

- HMM founder parameters,
- number of EM iterations,
- calibration mode,
- genotype call thresholds,
- pedigree mode,
- output flags,
- Dask scheduling,
- diagnostics thresholds.

## Dense Counts and `replace` Mode

For `fragment_likelihood_mode=replace`, the HMM can consume compact fragment evidence directly. In that case:

```bash
--compact-evidence-no-dense-counts
```

can avoid materializing full dense count matrices from cache.

This is memory efficient, but it is valid only when dense count emissions are not needed. The pipeline guards this: disabling dense materialization is only valid when fragment likelihoods are enabled and the mode is `replace`.

For `augment`, dense counts are needed because augment mode combines count-based emissions with fragment likelihoods.

## Cache Size Expectations

Cache size depends on:

- number of samples,
- target SNP density,
- coverage,
- fragment length,
- read filters,
- whether dense arrays are included,
- compression codec and chunk size.

A compact fragment cache scales with observed evidence, not with the full dense matrix size:

```text
O(number_of_retained_fragment_observations)
```

A dense count cache scales with:

```text
O(number_of_samples * number_of_target_variants * number_of_dense_arrays)
```

For very large cohorts, this distinction is enormous. A dense matrix for 30,000 individuals and 7,000,000 SNPs has:

```text
210,000,000,000 sample-SNP cells
```

Even one `float32` matrix for that shape is about:

```text
210e9 * 4 bytes = 840 GB
```

Seven dense arrays of that shape would be several terabytes before compression. Because low-coverage read evidence is sparse, compact fragment Parquet should be far smaller than dense arrays when coverage is low and target SNPs are sparse.

The practical recommendation is:

- use compact fragment evidence as the primary cache,
- include dense Zarr arrays only when a workflow truly needs repeated dense counts,
- do not keep VCF/BCF as the primary cache or result store,
- use Parquet for hard calls, support masks, and compact evidence,
- use Zarr/xarray for dense floating arrays when needed.

## Cache and Benchmarking

When comparing STITCHV2 configurations, do not let BAM reading dominate every repeated configuration if the question is about HMM/calibration/pedigree behavior.

A good benchmark structure is:

1. Build or validate cache once and record IO time.
2. Run STITCHV2 parity from cache.
3. Run STITCHV2 calibrated from cache.
4. Run STITCHV2 variable/hybrid founder settings from cache.
5. Report cache size and cache hit/miss status.
6. Report stage timings separately:
   - IO/cache load,
   - HMM,
   - calibration,
   - pedigree,
   - writing.

The benchmark should still include an uncached run when measuring full end-to-end runtime from raw BAMs.

## Cache Safety Checklist

Before trusting cached runs, verify:

- `manifest.json` exists for every expected block.
- `positions_hash` matches the current positions.
- All expected `sample_id` values are present.
- `compact_cache_hit=True` for cache-only benchmark runs.
- Read filters and backend choices match the intended comparison.
- `stage_timings.json` shows read/cache load time consistent with cache usage.
- `diagnostics_summary.json` does not show degenerate output.
