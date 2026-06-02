# STITCHV2 Inputs and Outputs

Status: retained as a legacy IO note. The current canonical IO reference is [docs/deep_dive/02_inputs_outputs_deep_dive.md](../deep_dive/02_inputs_outputs_deep_dive.md).

This file describes how to interface with STITCHV2 and what files are produced.

---

## 1) Inputs

## A) Variant positions table (required)

Supported formats:
- Parquet
- CSV/TSV

Required columns:
- `CHR`
- `POS`
- `REF`
- `ALT`

Example:

| CHR | POS | REF | ALT |
|---|---:|---|---|
| chr1 | 100123 | A | G |

## B) Sample table (required)

Supported formats:
- Parquet
- CSV/TSV

Required columns:
- `sample_id`
- `bam_path`
- `generation`

Optional useful columns:
- `read_subsample_prob`
- `read_subsample_seed`
- `sex`
- `plink_path`
- any metadata columns (kept in table for downstream use)

Example:

| sample_id | bam_path | generation | farm | sex |
|---|---|---:|---|---|
| animal_001 | /data/bams/animal_001.bam | 12 | A | F |

## C) Founder panel (required unless generated in memory)

Supported sources:
- VCF (`FounderConfig(source_format="vcf")`)
- PLINK prefix (`FounderConfig(source_format="plink")`)
- BAM-derived mode (configurable path)

Founders can be:
- Immutable: known/fixed genotypes
- Mutable: probabilistic founder alt-allele states updated by EM

## D) Optional microarray genotypes

Optional argument:
- `microarray_plink_path` (PLINK prefix with hard calls)

Use case:
- add samples that only have microarray data
- inject hard genotype evidence into HMM input counts

## E) Optional pedigree

Optional input:
- parent columns inside `samples.parquet`, a separate pedigree table, or a CSR-like sparse matrix

Use case:
- dosage smoothing/regularization after block-level HMM inference
- kinship-weighted fallback imputation across long-distance relatives
- recombination-smoothed parent-offspring transmission message passing after the STITCHV2 population HMM

CLI example:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out_pedigree \
  --n-founders 8 \
  --pedigree pedigree.parquet \
  --pedigree-mode transmission \
  --pedigree-strength 0.7 \
  --pedigree-offspring-col rfid \
  --pedigree-parent1-col dam \
  --pedigree-parent2-col sire \
  --pedigree-iterations 4
```

If `--pedigree` is omitted and `samples.parquet` contains parent columns, STITCHV2 automatically builds the pedigree from the samples table:

```text
sample_id  bam_path       generation  father  mother
dad        dad.bam        10          NA      NA
mom        mom.bam        10          NA      NA
kid        kid.bam        10          dad     mom
```

Supported pedigree modes:

| Mode | Meaning |
|---|---|
| `off` | Ignore pedigree even if provided. |
| `smooth` | Existing dosage smoothing toward parent/relative means. |
| `kinship` | Sparse long-distance kinship fallback from the child-parent graph. |
| `transmission` | Iterative Mendelian parent-child messages in both directions, with recombination-aware smoothing along positions and kinship fallback. |

For table input, STITCHV2 recognizes common column aliases such as:

| Role | Default | Common aliases |
|---|---|---|
| offspring | `sample_id` | `offspring`, `child`, `rfid`, `iid` |
| parent 1 | `father_id` | `father`, `sire`, `parent1`, `parent_1`, `dad` |
| parent 2 | `mother_id` | `mother`, `dam`, `parent2`, `parent_2`, `mom` |

The sparse matrix convention matches the GWAS-pipeline kinship helper:

```text
P[child, parent] = 0.5
```

Pedigree-aware runs write the usual `dosage/`, `genotype_posteriors/`, and `genotype_calls/` outputs. They also write `pedigree_summary.json`.

For the model details, equations, and efficiency notes, see `docs/pedigree_tutorial.md`.

---

## 2) Running interfaces

## Python API

```python
import pandas as pd
from stitchv2 import PipelineConfig, StitchPipeline

samples = pd.read_parquet("samples.parquet")

cfg = PipelineConfig(
    chromosome="chr1",
    positions_path="positions.parquet",
    output_dir="out",
    n_founders=8,
    em_iterations=2,
    block_size=1000,
    hmm_backend="jax",
    read_mode="read_stream",
    read_stream_backend="auto",
    write_genotype_posteriors=True,
    write_genotype_calls=True,
    write_support_mask=True,
)

pipeline = StitchPipeline(cfg)
pipeline.prepare_inputs(samples)
```

## CLI

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out \
  --n-founders 8 \
  --hmm-backend jax \
  --read-mode read_stream \
  --read-stream-backend auto \
  --em-iterations 2 \
  --block-size 1000 \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-support-mask
```

---

## 3) Output layout

Outputs are block-chunked parquet datasets under the run directory (`output_dir`).

Typical layout:

```text
out/
  samples.parquet
  positions.parquet
  founders.parquet
  stage_timings.json
  memory_profile_summary.json
  dosage/block=000000.parquet
  genotype_posteriors/block=000000.parquet
  genotype_calls/block=000000.parquet
  support_mask/block=000000.parquet
  transitions/block=000000.parquet
  recombination/block=000000.parquet
  founder_updates/block=000000.parquet
  haplotype_probabilities/block=000000.parquet   # optional
```

---

## 4) Core output datasets and schema

## Dosage (`dosage/`)

Columns:
- `sample_id`
- `chromosome`
- `position`
- `dosage` (float32)
- `block_id`

## Genotype posteriors (`genotype_posteriors/`, optional)

Columns:
- `sample_id`
- `chromosome`
- `position`
- `genotype_posterior` (fixed-size list length 3; GP for 0/0, 0/1, 1/1)
- `block_id`

## Hard calls (`genotype_calls/`, optional)

Columns:
- `sample_id`
- `chromosome`
- `position`
- `genotype_call` (int8; typically 0/1/2, and -1 for no-call)
- `block_id`

## Supporting-read mask (`support_mask/`, optional)

Columns:
- `sample_id`
- `chromosome`
- `position`
- `has_supporting_read` (bool)
- `block_id`

Interpretation:
- `true`: this sample x variant had direct read support
- `false`: fully imputed from model context/founders without direct read at that site

## Recombination (`recombination/`)

Columns:
- `chromosome`
- `position`
- `recombination_rate`
- `block_id`

## Transitions (`transitions/`)

Compact mode columns:
- `sample_id`
- `chromosome`
- `position`
- `switch_probability`
- `stay_probability`
- `offdiag_probability`
- `block_id`

Full mode can be emitted separately (`transitions_full/`) if configured.

## Founder updates (`founder_updates/`)

Columns:
- `chromosome`
- `position`
- `founder`
- `alt_prob`
- `block_id`

## Haplotype probabilities (`haplotype_probabilities/`, optional)

Columns:
- `sample_id`
- `chromosome`
- `position`
- `hap_dosage` (fixed-size list length `K`)
- `hap_probability` (fixed-size list length `K`)
- `block_id`

---

## 5) Assembling chunked outputs

## Combine chunks to one parquet per dataset

CLI:

```bash
stitchv2 combine --run-output-dir out
```

Programmatic:
- `combine_parquet_chunks(...)`

## Lazy combined dataset (xarray+dask)

Programmatic:
- `combine_pipeline_outputs(...)`

This returns a lazy `xarray.Dataset` so data can stay on disk until actually computed.

## Native Parquet/Zarr Outputs and Optional BCF

CLI:

```bash
stitchv2 combine --run-output-dir out --write-zarr --zarr-output out/combined/float_outputs.zarr
stitchv2 export-bcf --run-output-dir out --output-bcf out/stitch.chr1.bcf --chromosome chr1
```

Programmatic:
- `write_xarray_float_zarr(...)`
- `export_stitch_bcf_from_parquet(...)`

STITCHV2 does not allow VCF export. BCF export is an explicit interoperability path and WILL slow down I/O; keep primary results in Parquet/Zarr. Hard calls and categorical columns stay in Parquet with dictionary/RLE/bit-packing-friendly encodings, while dosages/posteriors and other float outputs can be written as an xarray/Zarr dataset.

---

## 6) Notes for large datasets

- Keep `block_size` moderate and write every block.
- Prefer `read_stream` mode and benchmark `python` vs `htslib` backend.
- Enable memory mapping for read matrices when RAM is constrained.
- Use compressed parquet (`zstd`) and avoid loading whole outputs into memory when not needed.
