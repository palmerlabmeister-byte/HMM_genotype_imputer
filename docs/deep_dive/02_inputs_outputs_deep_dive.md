# STITCHV2 Inputs and Outputs Deep Dive

This file describes the external files STITCHV2 consumes and produces. It is meant to be practical: if a run fails because an input column is missing, or if a downstream script needs to read a result, this document should identify the relevant table and columns.

The model details are in [01_model_deep_dive.md](01_model_deep_dive.md). Cache internals are in [03_cached_files_deep_dive.md](03_cached_files_deep_dive.md).

## CLI Entrypoint

After installing the package in editable mode:

```bash
pip install -e .
```

the main entrypoint is:

```bash
stitchv2
```

The main imputation command is:

```bash
stitchv2 run
```

The pedigree curation command is:

```bash
stitchv2 pedigree-qc
```

The BCF export command is:

```bash
stitchv2 export-bcf
```

Native Parquet/Zarr outputs are preferred. Use BCF export only when an external tool explicitly requires BCF.

## Required Run Inputs

### Samples Table

The samples table can be Parquet, CSV, or another format accepted by the table loader. Parquet is preferred for reproducibility and speed.

Minimum columns:

| Column | Type | Required | Meaning |
| --- | --- | --- | --- |
| `sample_id` | string | recommended | Unique sample identifier. If absent, STITCHV2 creates `sample_0`, `sample_1`, etc. |
| `generation` | float | yes | Generation or genetic distance scaling value used by the HMM transition model. |
| `bam_path` | string | normal BAM workflows | BAM/CRAM path for each sample. If absent, STITCHV2 fills an empty string, but read-based imputation requires valid paths. |

Common optional columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `sex` | string | Used for sex-specific ploidy assignment. |
| `father_id` | string | Declared father or parent 1 for pedigree mode. |
| `mother_id` | string | Declared mother or parent 2 for pedigree mode. |
| `family_id` | string | Family or pedigree grouping variable. |
| `rfid` | string | External sample identifier, useful when BAM sample names differ from array names. |
| `plink_path` | string | Optional sample-level PLINK metadata in workflows that use it. |

Validation behavior:

- `generation` must exist.
- `sample_id` is coerced to string.
- `generation` is coerced to `float32`.
- `bam_path` is filled with empty strings if absent.
- `sex` is kept as string when present.

Example:

```text
sample_id,generation,bam_path,sex,father_id,mother_id
Outbred_001,12,/data/bams/animal001.bam,F,FounderA,FounderB
Outbred_002,12,/data/bams/animal002.bam,M,FounderA,FounderC
```

### Positions Table

The positions table defines the target variants.

Required columns:

| Column | Type | Required | Meaning |
| --- | --- | --- | --- |
| `CHR` | string | yes | Chromosome label. |
| `POS` | int64 | yes | 1-based genomic position. |

Recommended columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `REF` | string | Reference allele. |
| `ALT` | string | Alternate allele. |
| `VARIANT_TYPE` | string | Optional explicit type: `snp`, `insertion`, or `deletion`. If absent, STITCHV2 infers the type from `REF` and `ALT`. |

STITCHV2 uppercases position-table column names internally. `chr`, `pos`, `ref`, `alt` therefore become `CHR`, `POS`, `REF`, `ALT`.

If `REF` or `ALT` is missing, STITCHV2 fills it with `N`. Real runs should provide true alleles because read evidence and founder genotypes depend on allele identity.

Targeted indel runs require normalized biallelic records. Split multiallelic variants before STITCHV2. Small insertions and deletions are supported through the variant-aware reader up to `--max-indel-len`, default `50`. Nested or complex variants should be normalized upstream and reviewed before imputation.

Example:

```text
CHR,POS,REF,ALT
chr12,9000123,A,G
chr12,9001045,C,T
```

Chromosome labels must match the requested `--chromosome`, except for loader-specific normalization in founder PLINK loading where `chr12` and `12` may be reconciled.

### Founders

Founder input is optional for fully learned founder runs but required for fixed-founder parity with STITCH.

Supported loaded-founder formats:

```text
--founder-vcf <path.vcf.gz>
--founder-plink <plink_prefix>
```

VCF founders:

- Founder samples are read from the VCF header samples.
- `GT` is converted to alternate allele probability by mean allele count.
- `DS` is used when available and divided by 2.
- Missing sites default to 0.5.

PLINK founders:

- PLINK genotype values are loaded with `npplink`.
- Variants are aligned by position.
- Genotypes are divided by 2 to produce alternate allele probabilities.
- Missing or unmatched target positions default to 0.5.

Founder output schema:

```text
founders.parquet
```

| Column | Type | Meaning |
| --- | --- | --- |
| `chromosome` | string | Chromosome label. |
| `position` | int64 | Variant position. |
| `founder` | int | Founder index. |
| `alt_prob` | float32 | Founder alternate allele probability. |
| `immutable` | int8 | Whether this founder is fixed during EM. |

### Minimal Run Example

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr12 \
  --output-dir runs/chr12 \
  --n-founders 8 \
  --founder-plink founders8 \
  --founder-immutable \
  --em-iterations 1 \
  --fragment-coupling-model stitch_parity \
  --fragment-likelihood-mode replace \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-haplotype-probabilities \
  --write-support-mask
```

For STITCH-like fixed-founder parity, load the founder panel with `--founder-vcf` or `--founder-plink` and freeze it with `--founder-immutable`.

## Optional Run Inputs

### Pedigree Table

A pedigree table can be supplied separately:

```bash
--pedigree pedigree_curated.parquet
```

or read from columns in `samples.parquet`.

Default pedigree columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Child/offspring ID. |
| `father_id` | Parent 1 ID. |
| `mother_id` | Parent 2 ID. |

Custom column names:

```bash
--pedigree-offspring-col animal_id
--pedigree-parent1-col sire
--pedigree-parent2-col dam
```

Pedigree mode:

```bash
--pedigree-mode off|smooth|kinship|transmission
--pedigree-strength 0.0
--pedigree-iterations 4
--pedigree-kinship-threshold 0.01
```

`pedigree_strength=0.0` effectively disables pedigree adjustment even if a pedigree is present.

### Microarray or External Truth

Microarray data can be used for:

- adding high-confidence evidence to the HMM,
- calibration truth,
- benchmark evaluation.

Controls include:

```bash
--microarray-plink <prefix>
--microarray-calibration-only
--microarray-generation-default <value>
--microarray-hard-call-weight 80
```

For benchmark evaluation, external truth should not be injected into the HMM unless the benchmark is explicitly testing that behavior. To avoid leakage, use it as calibration-only or evaluation-only depending on the intended design.

### Compact Evidence Cache

Compact evidence cache controls:

```bash
--compact-evidence-cache-dir cache_dir
--compact-evidence-cache-mode off|read|write|readwrite
--compact-evidence-cache-format parquet_zarr
--compact-evidence-cache-sample-batch-size 256
--compact-evidence-no-dense-counts
--compact-evidence-cache-include-dense-counts
```

For repeated benchmarking on the same BAMs and positions, the recommended pattern is:

1. First run: `--compact-evidence-cache-mode readwrite`
2. Later runs: `--compact-evidence-cache-mode read`

This avoids repeatedly decoding BAMs for calibration sweeps, founder sweeps, and STITCHV2 parameter comparisons.

## Main Output Directory

A run output directory typically contains:

```text
output_dir/
  samples.parquet
  positions.parquet
  founders.parquet
  dosage/
  recombination/
  transitions/
  founder_updates/
  diagnostics/
  stage_timings.json
  memory_profile_summary.json
  diagnostics_summary.json
```

Optional directories:

```text
  support_mask/
  genotype_posteriors/
  genotype_calls/
  haplotype_probabilities/
  transitions_full/
  transition_summary/
  transition_factors/
  calibration_decisions/
  pileup/
  pedigree_summary.json
  dask_run_summary.json
```

Every block-level output is named:

```text
block=000000.parquet
block=000001.parquet
...
```

## Output Table Schemas

### `samples.parquet`

This is the validated sample table used by the run. It preserves useful metadata while normalizing core columns.

Important columns:

- `sample_id`
- `generation`
- `bam_path`
- optional `sex`
- optional pedigree columns

Use this file, not the original input, when aligning outputs back to the exact run order.

### `positions.parquet`

This is the validated and filtered position table used by the run.

Columns:

- `CHR`
- `POS`
- `REF`
- `ALT`

Use this file, not the original input, when aligning outputs back to the exact run order.

### `dosage/block=*.parquet`

Always written.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Sample ID. |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `dosage` | float32 | Expected alternate allele count. `NaN` for ploidy-zero samples. |
| `block_id` | int | HMM block. |

For diploid samples, dosage is normally in `[0, 2]`. For haploid samples, dosage is normally in `[0, 1]`. For polyploid samples, dosage is in `[0, ploidy]`.

### `genotype_posteriors/block=*.parquet`

Written when `--write-genotype-posteriors` is enabled or needed internally for pedigree/calibration.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Sample ID. |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `genotype_posterior` | fixed-size list<float32> | Posterior probability for genotype classes. |
| `block_id` | int | HMM block. |

For diploid data, `genotype_posterior` has length 3:

```text
[P(G=0), P(G=1), P(G=2)]
```

For haploid data, it has length 2:

```text
[P(G=0), P(G=1)]
```

For ploidy `P`, it has length `P + 1`.

### `genotype_calls/block=*.parquet`

Written when `--write-genotype-calls` is enabled.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Sample ID. |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `genotype_call` | int8 | Hard call. `-1` means no-call/missing. |
| `block_id` | int | HMM block. |

Hard-call mode:

```bash
--genotype-call-mode argmax|stitch_no_call|quality_gated
```

Meanings:

- `argmax`: call the genotype with maximum posterior.
- `stitch_no_call`: call only if max GP passes the STITCH-style threshold, default 0.9.
- `quality_gated`: apply confidence, margin, and optional callability probability thresholds.

### `haplotype_probabilities/block=*.parquet`

Written when `--write-haplotype-probabilities` is enabled.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Sample ID. |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `hap_dosage` | fixed-size list<float32> | Founder haplotype dosage per founder. |
| `hap_probability` | fixed-size list<float32> | Founder copying probability per founder. |
| `block_id` | int | HMM block. |

If there are `K` founders, each list has length `K`.

For diploid samples, `hap_dosage` is generally:

```text
2 * hap_probability
```

unless sample-specific ploidy scaling is active. For ploidy-zero samples, haplotype output should be zero or masked depending on the output path.

### `support_mask/block=*.parquet`

Written when `--write-support-mask` is enabled.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Sample ID. |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `has_supporting_read` | bool | Whether any read evidence supported this sample-site. |
| `block_id` | int | HMM block. |

This is useful for calibration, diagnostics, and separating directly read-backed calls from imputed calls.

### `recombination/block=*.parquet`

Always written.

| Column | Type | Meaning |
| --- | --- | --- |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `recombination_rate` | float32 | Recombination rate used by the HMM at this position. |
| `block_id` | int | HMM block. |

### `transitions/block=*.parquet`

Written when transitions are enabled.

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Sample ID. |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `switch_probability` | float32 | Probability of switching founder state. |
| `stay_probability` | float32 | Probability of staying in state. |
| `offdiag_probability` | float32 | Probability assigned to a specific non-current founder state. |
| `block_id` | int | HMM block. |

Related controls:

```bash
--transition-output compact|factorized|full
--store-xi per-snp|full|False
--write-gamma summary|off|full
```

`--transition-output compact` is sufficient for STITCH-parity transition parameters. `--transition-output factorized` writes factorized transition parameters when `--transition-model factorized` is used. `--transition-output full` and `--store-xi full` are exact but expensive and should be requested only when the full state-pair output is needed.

`transition_summary/` is the default xi-style output for `--store-xi per-snp`; it is intended for aggregate transition hotspot/coldspot analysis. `transition_factors/` is written for factorized transition output.

### `founder_updates/block=*.parquet`

Always written.

| Column | Type | Meaning |
| --- | --- | --- |
| `chromosome` | string | Chromosome. |
| `position` | int64 | Variant position. |
| `founder` | int | Founder index. |
| `alt_prob` | float32 | Founder alternate allele probability after the block run. |
| `block_id` | int | HMM block. |

In fixed-founder runs, this should remain aligned with the loaded immutable founders. In mutable-founder runs, this is the learned founder panel for each block.

### `diagnostics/block=*.parquet`

Written when diagnostics are enabled.

Important columns:

| Column | Meaning |
| --- | --- |
| `block_id` | Block ID. |
| `chromosome` | Chromosome. |
| `position` | Variant position. |
| `n_samples` | Number of rows in the sample table. |
| `n_callable_samples` | Samples with ploidy greater than zero. |
| `calling_rate` | Non-missing hard-call rate among callable samples. |
| `missing_rate` | 1 - calling rate. |
| `maf` | Minor allele frequency from dosage. |
| `alt_af` | Alternate allele frequency from dosage. |
| `het_rate` | Posterior heterozygosity rate when diploid GP is available. |
| `hom_rate` | Posterior homozygote rate. |
| `hard_het_rate` | Hard-call heterozygosity rate. |
| `hard_hom_rate` | Hard-call homozygosity rate. |
| `hwe_deviation` | HWE deviation feature for diploid GP. |
| `hwe_chisq` | HWE chi-square feature. |
| `hwe_pvalue` | HWE p-value feature. |
| `info` | INFO score from GP. |
| `mean_entropy` | Mean posterior entropy. |
| `mean_max_gp` | Mean maximum genotype posterior. |
| `mean_depth` | Mean read depth. |
| `support_rate` | Fraction of callable samples with supporting reads. |
| `calibration_abs_dosage_shift` | Mean absolute dosage shift after calibration, when available. |
| `calibration_het_rate_shift` | Heterozygosity shift after calibration, when available. |
| `calibration_abs_maf_shift` | MAF shift after calibration, when available. |

### `calibration_decisions/block=*.parquet`

Written for standard callability calibration when per-SNP decisions are available.

Important columns:

| Column | Meaning |
| --- | --- |
| `chromosome` | Chromosome. |
| `position` | Variant position. |
| `block_id` | Block ID. |
| `calibration_used` | Whether calibrated gating was used for this SNP. |
| `fallback_to_stitch` | Whether STITCH no-call fallback was used. |
| `decision_source` | Decision level: per-SNP, local, group, or global. |
| `fallback_reason` | Why calibration was not used, if applicable. |
| `threshold` | Selected P(correct) threshold. |
| `call_rate_stitch` | Validation call rate with STITCH no-call. |
| `call_rate_calibrated` | Validation call rate with calibration. |
| `call_rate_delta` | Calibrated minus STITCH call rate. |
| `objective_delta` | Objective improvement relative to STITCH no-call. |
| `maf` | SNP MAF feature. |
| `info` | SNP INFO feature. |
| `hwe_deviation` | SNP HWE feature. |
| `support_rate` | SNP support feature. |
| `missingness` | SNP missingness feature. |
| `mean_depth` | SNP depth feature. |

This table is essential when explaining why calibration helped, did nothing, or fell back for a SNP.

### `stage_timings.json`

Contains per-block stage timings and summary timing information.

Important per-block fields include:

| Field | Meaning |
| --- | --- |
| `seconds_read_extract` | Time spent reading/loading evidence. |
| `seconds_hmm` | HMM time. |
| `seconds_calibration` | Calibration and hard-call time. |
| `seconds_write` | Output write time. |
| `seconds_total` | Total block time. |
| `mean_depth` | Mean evidence depth for the block. |
| `n_reads` | Number of overlapping reads. |
| `rss_after_*` | Resident memory after stage, when memory profiling is enabled. |

### `memory_profile_summary.json`

Contains run-level memory statistics. Use this with `stage_timings.json` to identify whether memory growth happens during IO, HMM, calibration, or writing.

### `diagnostics_summary.json`

Contains warnings and failures derived from `diagnostics/`. It is designed to catch degenerate outcomes such as:

- all or nearly all heterozygotes,
- all or nearly all homozygotes,
- extremely high missingness,
- very low INFO,
- calibration shifts that are too large.

## Combining Outputs

STITCHV2 includes output utilities that can combine block-level Parquet chunks:

```python
from stitchv2.output import combine_pipeline_outputs

dataset, summary = combine_pipeline_outputs("runs/chr12")
```

The combined dataset is lazy and xarray-oriented. Floating arrays can be written to Zarr with:

```python
from stitchv2.output import write_xarray_float_zarr

write_xarray_float_zarr(dataset, "runs/chr12/combined_float.zarr")
```

Hard calls, IDs, positions, and masks remain best represented as Parquet.

## BCF Export

BCF export exists for interoperability:

```bash
stitchv2 export-bcf \
  --run-output-dir runs/chr12 \
  --output-bcf runs/chr12/output.bcf \
  --chromosome chr12
```

The package intentionally warns that BCF export is slower. Keep primary STITCHV2 outputs in Parquet/Zarr unless an external tool requires BCF.
