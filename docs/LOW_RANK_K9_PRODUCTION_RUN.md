# STITCHV2 Low-Rank K9 Production Run Guide

This guide describes the production-style STITCHV2 configuration for a low-rank transition run with:

- `K=9` total founders,
- 8 immutable founders loaded from a founder PLINK prefix,
- 1 extra mutable founder appended by STITCHV2,
- factorized low-rank transition output with rank 4,
- compact evidence caching in `parquet_zarr`,
- optional LightGBM hard-call calibration,
- optional two-pass pedigree QC and pedigree-aware imputation.

This is a runbook and file-format guide. It is not a benchmark report and should not contain benchmark results. Benchmark results should go into a dataset-specific report or `final_benchmark_results.md`.

## Current Low-Rank Status

The current low-rank transition mode is selected with:

```bash
--transition-model factorized
--transition-factor-rank 4
--transition-output factorized
```

Important interpretation:

- The factorized model reduces the transition parameter/output representation.
- It does not yet remove the diploid `K x K` hidden-state grid from the HMM.
- With `K=9`, diploid inference still has 81 ordered pair states per sample/SNP.
- Full diploid xi would be `(K^2 x K^2)` per sample/SNP interval and is not a normal output.
- The production default is `--store-xi per-snp`, which writes compressed per-SNP transition summaries rather than full xi.

For publication or production reporting, describe this mode as:

```text
STITCHV2 K9 factorized-rank-4 transition model with 8 immutable founders and 1 mutable founder.
```

Do not describe it as a fully low-rank diploid HMM state-space implementation yet.

## Recommended Environment

Use one conda environment for the full analysis. After activation, install STITCHV2 as an editable package and build the native reader:

```bash
conda activate stitchv2-cpu-py313
cd /path/to/STITCHV2
pip install -e ".[plink,ml,plot,jobqueue,dev]"
python setup.py build_ext --inplace
stitchv2 --help
stitchv2 run --help
stitchv2 pedigree-qc --help
```

For a whole-genome or publication run, save:

- `conda env export`,
- the STITCHV2 git revision or package revision,
- `stitchv2 --help`,
- `stitchv2 run --help`,
- `stitchv2 pedigree-qc --help`,
- every `cli_run_summary.json`.

## Required Inputs

### Samples Table

Pass with `--samples`.

Preferred format: Parquet.

Minimum required columns:

| column | type | meaning |
| --- | --- | --- |
| `sample_id` | string | Stable sample ID used in all outputs. |
| `generation` | numeric | Generation or transition-scale value used by the recombination model. |
| `bam_path` | string | Indexed BAM/CRAM path for sequenced samples. Can be blank for unsequenced pedigree members. |

Recommended columns:

| column | meaning |
| --- | --- |
| `sex` | Used by sex-specific ploidy options. |
| `father_id` | Declared first parent. |
| `mother_id` | Declared second parent. |
| `family_id` | Optional family/group column for pedigree reports. |
| `rfid` or external ID | Useful for array/truth/sample-sheet matching. |

Unsequenced pedigree members should still be present if they are to receive pedigree-based inference. Give them a valid `sample_id`, parent metadata if known, `generation`, and an empty or missing `bam_path`.

### Positions Table

Pass with `--positions`.

Preferred format: Parquet.

Required columns:

| column | type | meaning |
| --- | --- | --- |
| `CHR` | string | Chromosome/contig label. |
| `POS` | int | 1-based position. |
| `REF` | string | Reference allele. |
| `ALT` | string | Alternate allele. |

Optional columns:

| column | meaning |
| --- | --- |
| `CM` or `GENETIC_CM` | Genetic map cM coordinate. Used when non-zero and valid. |
| `VARIANT_TYPE` | `snp`, `insertion`, or `deletion` for variant-aware reading. |

The normal STITCHV2 run imputes only variants present in this table. Discovery is a separate pre-pass and should produce reviewed candidate positions before imputation.

### Founder PLINK Prefix

Pass with:

```bash
--founder-plink /path/to/founders8
--founder-immutable
--n-founders 9
```

The prefix means these files exist:

```text
/path/to/founders8.bed
/path/to/founders8.bim
/path/to/founders8.fam
```

If the PLINK file contains 8 founder samples and `--n-founders 9` is used, STITCHV2:

1. loads the 8 founder samples,
2. marks those 8 founders immutable,
3. appends 1 mutable founder initialized around ALT probability `0.5`,
4. updates only the mutable founder during EM.

This mode is useful when the known founder panel is mostly correct but the real population may contain an unrecorded founder, drift, private haplotypes, or variants not well explained by the original founders.

### Pedigree Table

Pass with `--pedigree` on the second pass, or keep parent columns in the samples table.

Default pedigree columns:

| STITCHV2 role | default column |
| --- | --- |
| offspring/sample | `sample_id` |
| parent 1 | `father_id` |
| parent 2 | `mother_id` |

Custom names can be supplied with:

```bash
--pedigree-offspring-col <child_column>
--pedigree-parent1-col <parent1_column>
--pedigree-parent2-col <parent2_column>
```

## Core K9 Rank 4 Options

Use these options for the low-rank K9 model:

```bash
--n-founders 9
--founder-plink "$FOUNDER_PLINK"
--founder-immutable
--em-iterations 40
--transition-model factorized
--transition-factor-rank 4
--transition-factor-regularization 100
--transition-factor-max-deviation 0.25
--transition-output factorized
--store-xi per-snp
```

Recommended read/HMM settings:

```bash
--read-mode read_stream
--read-stream-backend auto
--hmm-backend jax
--stitch-compat
--fragment-coupling-model stitch_parity
--fragment-likelihood-mode replace
--max-mem 90%
--block-size 0
--io-window-size 0
--jax-persistent-cache-dir "$JAX_CACHE"
```

`--read-stream-backend auto` should use the compiled native path when available. For strict debugging, specify a backend explicitly:

```bash
--read-stream-backend variant_aware_bamreader
```

or, for STITCH-style SNP-only parity checks:

```bash
--read-stream-backend stitch_style_bamreader
```

## SNP Block Mode

Production default:

```bash
--snp-block-mode exact_streaming
```

This mode is intended to preserve whole-region HMM math while streaming through SNP blocks.

Approximate mode:

```bash
--snp-block-mode density_balanced_overlap
```

This mode is useful for stress tests, QC, and very large factorized runs that exceed local memory. It is explicitly approximate and should be labeled as approximate in reports.

Avoid using `independent_approx` for production-quality claims unless the purpose is a quick QC or fault-isolation run.

## Evidence Cache

Use the compressed `parquet_zarr` cache. Do not use NPZ for production runs.

First run, cache build or partial fill:

```bash
--compact-evidence-cache-dir "$CACHE"
--compact-evidence-cache-mode readwrite
--compact-evidence-cache-format parquet_zarr
--compact-evidence-cache-sample-batch-size 256
--compact-evidence-cache-include-dense-counts
```

Reruns with the same sample set and positions:

```bash
--compact-evidence-cache-dir "$CACHE"
--compact-evidence-cache-mode read
--compact-evidence-cache-format parquet_zarr
```

Adding new samples:

```bash
--compact-evidence-cache-mode readwrite
```

STITCHV2 will load cached samples, read missing samples from BAM/CRAM, then update the cache.

Use `--compact-evidence-cache-include-dense-counts` when calibration, diagnostics, or exact cached feature reuse matters. Dense Zarr arrays are larger, but they prevent cached calibration runs from losing depth/count information.

### Cache Layout

The cache root is partitioned by chromosome and row block:

```text
cache_root/
  chrom=chr12/
    block=000000_rows=0-4096/
      manifest.json
      positions.parquet
      fragments/
        part-*.parquet
      summary/
        part-*.parquet
      support/
        part-*.parquet
      dense/
        part-*.zarr/
          ref_count/
          alt_count/
          other_count/
          depth/
          ref_weight/
          alt_weight/
          other_weight/
```

`manifest.json` records:

- chromosome,
- block ID,
- row start/stop,
- position hash,
- evidence metadata hash,
- sample-to-part mapping,
- whether each part includes dense counts.

Compact fragment Parquet columns:

| column | meaning |
| --- | --- |
| `sample_id` | Sample ID. |
| `sample_index` | Index inside the sample batch. |
| `fragment_index` | Fragment index within the sample. |
| `center_idx` | Central target variant for the fragment. |
| `obs_order` | Observation order within the fragment. |
| `pos_idx` | Target-position row index inside the block. |
| `obs_code` | Observation code: REF, ALT, or OTHER. |
| `obs_qual` | Observation quality after reader filters. |

Dense Zarr arrays are sample-by-position matrices. Count arrays are unsigned integer-like counts. Weight arrays are floating quality-weighted evidence.

## First Pass Without Pedigree

The first pass creates genotype calls for pedigree QC. It should not use pedigree adjustment. For the production low-rank workflow, it is reasonable to use the same K9/rank4 model without pedigree so the QC sees the same caller family as the final pass.

Set project variables once:

```bash
PROJECT=/path/to/STITCHV2
RUN_ROOT="$PROJECT/runs/k9_rank4_production"
SAMPLES="$PROJECT/test_data_own/samples.parquet"
POSITIONS="$PROJECT/test_data_own/positions.parquet"
FOUNDER_PLINK="$PROJECT/test_data_own/founders8"
PEDIGREE_RAW="$PROJECT/test_data_own/pedigree.parquet"
CACHE_ROOT="$RUN_ROOT/evidence_cache"
JAX_CACHE="$RUN_ROOT/jax_cache"

mkdir -p "$RUN_ROOT" "$CACHE_ROOT" "$JAX_CACHE"
cd "$PROJECT"
```

Run one chromosome:

```bash
CHR=chr12
OUT1="$RUN_ROOT/first_pass_no_pedigree/$CHR"
CACHE="$CACHE_ROOT/$CHR"

stitchv2 run \
  --samples "$SAMPLES" \
  --positions "$POSITIONS" \
  --chromosome "$CHR" \
  --output-dir "$OUT1" \
  --n-founders 9 \
  --founder-plink "$FOUNDER_PLINK" \
  --founder-immutable \
  --em-iterations 40 \
  --transition-model factorized \
  --transition-factor-rank 4 \
  --transition-factor-regularization 100 \
  --transition-factor-max-deviation 0.25 \
  --transition-output factorized \
  --store-xi per-snp \
  --snp-block-mode exact_streaming \
  --read-mode read_stream \
  --read-stream-backend auto \
  --hmm-backend jax \
  --jax-persistent-cache-dir "$JAX_CACHE" \
  --max-mem 90% \
  --block-size 0 \
  --io-window-size 0 \
  --stitch-compat \
  --fragment-coupling-model stitch_parity \
  --fragment-likelihood-mode replace \
  --compact-evidence-cache-dir "$CACHE" \
  --compact-evidence-cache-mode readwrite \
  --compact-evidence-cache-format parquet_zarr \
  --compact-evidence-cache-include-dense-counts \
  --pedigree-mode off \
  --no-calibrate-genotype-posteriors \
  --genotype-call-mode stitch_no_call \
  --genotype-call-stitch-threshold 0.9 \
  --write-genotype-calls \
  --write-genotype-posteriors \
  --write-haplotype-probabilities \
  --write-transitions \
  --write-support-mask \
  --diagnostics-fail-on-error
```

If this exceeds memory on a full chromosome, rerun with:

```bash
--snp-block-mode density_balanced_overlap
```

and label the output as approximate in the report.

## Whole-Genome First Pass

Run one model at a time per job when benchmarking. For a production whole-genome run, chromosome-level parallelism is reasonable if each job has its own memory budget and output directory.

```bash
CHROMS="chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20"

for CHR in $CHROMS; do
  OUT1="$RUN_ROOT/first_pass_no_pedigree/$CHR"
  CACHE="$CACHE_ROOT/$CHR"
  stitchv2 run \
    --samples "$SAMPLES" \
    --positions "$POSITIONS" \
    --chromosome "$CHR" \
    --output-dir "$OUT1" \
    --n-founders 9 \
    --founder-plink "$FOUNDER_PLINK" \
    --founder-immutable \
    --em-iterations 40 \
    --transition-model factorized \
    --transition-factor-rank 4 \
    --transition-output factorized \
    --store-xi per-snp \
    --read-mode read_stream \
    --read-stream-backend auto \
    --hmm-backend jax \
    --jax-persistent-cache-dir "$JAX_CACHE" \
    --max-mem 90% \
    --stitch-compat \
    --fragment-coupling-model stitch_parity \
    --fragment-likelihood-mode replace \
    --compact-evidence-cache-dir "$CACHE" \
    --compact-evidence-cache-mode readwrite \
    --compact-evidence-cache-format parquet_zarr \
    --compact-evidence-cache-include-dense-counts \
    --pedigree-mode off \
    --no-calibrate-genotype-posteriors \
    --genotype-call-mode stitch_no_call \
    --genotype-call-stitch-threshold 0.9 \
    --write-genotype-calls \
    --write-genotype-posteriors \
    --write-haplotype-probabilities \
    --write-transitions \
    --write-support-mask \
    --diagnostics-fail-on-error
done
```

For X/Y/MT, add chromosome-specific ploidy flags. Examples:

```bash
# X
--ploidy 2 --ploidy-males 1 --ploidy-females 2

# Y
--ploidy 0 --ploidy-males 1 --ploidy-females 0

# MT
--ploidy 1
```

Ploidy-zero samples are excluded from the HMM but remain in rectangular output tables with missing genotype calls and zero or absent haplotype probabilities.

## Pedigree QC

Run pedigree QC once globally after first-pass calls. Do not curate pedigree independently per chromosome unless this is only a smoke test.

```bash
PED_QC="$RUN_ROOT/pedigree_qc"

PED_RUN_ARGS=()
for CHR in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20; do
  PED_RUN_ARGS+=(--run-output-dir "$RUN_ROOT/first_pass_no_pedigree/$CHR")
done

stitchv2 pedigree-qc \
  --samples "$SAMPLES" \
  --pedigree "$PEDIGREE_RAW" \
  "${PED_RUN_ARGS[@]}" \
  --output-dir "$PED_QC" \
  --pedigree-offspring-col sample_id \
  --pedigree-parent1-col father_id \
  --pedigree-parent2-col mother_id \
  --max-variants 50000 \
  --min-call-rate 0.80 \
  --min-maf 0.005 \
  --report-min-r 0.59 \
  --unrelated-max-r 0.59 \
  --first-degree-min-r 0.64 \
  --same-min-r 0.88 \
  --unlink-calls unrelated,same \
  --sample-block-size 1024 \
  --max-full-matrix-samples 5000 \
  --umap-neighbors 50 \
  --umap-max-variants 10000
```

Main pedigree QC outputs:

| file | meaning |
| --- | --- |
| `pedigree_curated.parquet` | Corrected pedigree directly consumable by `stitchv2 run --pedigree`. |
| `pedigree_edge_qc.parquet` | Parent-child edge status before and after curation. |
| `pedigree_sample_qc.parquet` | Sample-level pedigree QC status. |
| `sample_similarity.parquet` | High-similarity and suspicious sample pairs. |
| `pedigree_umap_embedding.parquet` | UMAP coordinates. |
| `pedigree_umap_edges_before.parquet` | Original pedigree edges on the UMAP. |
| `pedigree_umap_edges_after.parquet` | Curated pedigree edges on the UMAP. |
| `pedigree_long_edges.parquet` | Long UMAP pedigree edges worth reviewing. |
| `pedigree_umap_before_after.html` | Interactive before/after UMAP edge plot. |
| `pedigree_qc_summary.json` | Thresholds, counts, and resolved column names. |

Review `pedigree_qc_summary.json`, `pedigree_edge_qc.parquet`, `sample_similarity.parquet`, and `pedigree_umap_before_after.html` before using the curated pedigree.

## Second Pass With Curated Pedigree

Use the curated pedigree and reuse the evidence cache. This is the production-style K9 rank4 run.

```bash
CHR=chr12
CURATED_PED="$RUN_ROOT/pedigree_qc/pedigree_curated.parquet"
OUT2="$RUN_ROOT/second_pass_k9_rank4_pedigree/$CHR"
CACHE="$CACHE_ROOT/$CHR"

stitchv2 run \
  --samples "$SAMPLES" \
  --positions "$POSITIONS" \
  --chromosome "$CHR" \
  --output-dir "$OUT2" \
  --n-founders 9 \
  --founder-plink "$FOUNDER_PLINK" \
  --founder-immutable \
  --em-iterations 40 \
  --transition-model factorized \
  --transition-factor-rank 4 \
  --transition-factor-regularization 100 \
  --transition-factor-max-deviation 0.25 \
  --transition-output factorized \
  --store-xi per-snp \
  --snp-block-mode exact_streaming \
  --read-mode read_stream \
  --read-stream-backend auto \
  --hmm-backend jax \
  --jax-persistent-cache-dir "$JAX_CACHE" \
  --max-mem 90% \
  --stitch-compat \
  --fragment-coupling-model stitch_parity \
  --fragment-likelihood-mode replace \
  --compact-evidence-cache-dir "$CACHE" \
  --compact-evidence-cache-mode read \
  --compact-evidence-cache-format parquet_zarr \
  --pedigree "$CURATED_PED" \
  --pedigree-mode transmission \
  --pedigree-strength 0.20 \
  --pedigree-iterations 4 \
  --pedigree-kinship-threshold 0.01 \
  --calibration-mode standard_callability \
  --calibration-callability-model lightgbm \
  --calibration-truth-source read_evidence \
  --calibration-read-truth-holdout-fraction 0.30 \
  --calibration-callability-decision-mode per_snp_hierarchical \
  --calibration-callability-min-call-rate-delta -0.02 \
  --calibration-callability-max-call-rate-delta 0.05 \
  --genotype-call-mode quality_gated \
  --genotype-call-stitch-threshold 0.9 \
  --write-genotype-calls \
  --write-genotype-posteriors \
  --write-haplotype-probabilities \
  --write-transitions \
  --write-support-mask \
  --diagnostics-fail-on-error
```

Use `--pedigree-mode transmission` when the pedigree is trusted enough and there are informative relatives, especially multiple genotyped offspring for missing parents. Use `--pedigree-mode kinship` when relatedness is useful but strict declared transmission edges are less trusted. Use `--pedigree-mode smooth` only as a lighter post-HMM smoothing mode.

## LightGBM Calibration

The recommended calibration path is hard-call callability:

```bash
--calibration-mode standard_callability
--calibration-callability-model lightgbm
--calibration-truth-source read_evidence
--calibration-read-truth-holdout-fraction 0.30
--calibration-callability-decision-mode per_snp_hierarchical
--genotype-call-mode quality_gated
```

The raw HMM genotype posterior remains the main probability object. LightGBM predicts `P(argmax genotype is correct)` from HMM/read/QC features and gates hard calls. With `per_snp_hierarchical`, STITCHV2 decides per SNP whether calibrated gating is safe. Unsupported SNPs fall back to STITCH-style no-call.

Do not use external microarray/GATK truth for calibration in production unless the study design explicitly allows it. Use external truth for evaluation.

Calibration outputs:

| path | meaning |
| --- | --- |
| `calibration_decisions/` | Per-SNP use/fallback decisions and QC shifts. |
| `diagnostics/` | Per-variant MAF/HWE/missingness/INFO/entropy/depth diagnostics. |
| `diagnostics_summary.json` | Aggregate warning/failure summary. |

## Main Run Outputs

Each `stitchv2 run` writes a run directory.

Top-level files:

| file | meaning |
| --- | --- |
| `samples.parquet` | Validated sample table with resolved ploidy. |
| `positions.parquet` | Filtered chromosome/window target variants. |
| `founders.parquet` | Founder ALT probabilities after immutable loading and mutable-founder expansion. |
| `founder_expansion_summary.json` | Loaded founders, requested founders, immutable/mutable counts. |
| `cli_run_summary.json` | CLI options and runtime summary. |
| `run_summary.json` | Same run summary for programmatic use. |
| `stage_timings.json` | Per-stage timings including I/O, HMM, calibration, pedigree, and writing. |
| `memory_profile_summary.json` | Peak and per-stage memory summaries. |
| `runtime_memory_plan.json` | Planner decisions for sample/SNP/cache/output chunks. |
| `xi_output_manifest.json` | Xi policy and per-SNP summary materialization details. |
| `diagnostics_summary.json` | QC summary and warnings/failures. |
| `pedigree_summary.json` | Pedigree graph summary when pedigree is supplied or embedded. |

Partitioned Parquet output directories:

| directory | columns |
| --- | --- |
| `dosage/` | `sample_id`, `chromosome`, `position`, `dosage`, `block_id` |
| `genotype_calls/` | `sample_id`, `chromosome`, `position`, `genotype_call`, `block_id` |
| `genotype_posteriors/` | `sample_id`, `chromosome`, `position`, `genotype_posterior`, `block_id` |
| `haplotype_probabilities/` | `sample_id`, `chromosome`, `position`, `hap_dosage`, `hap_probability`, `block_id` |
| `support_mask/` | `sample_id`, `chromosome`, `position`, `has_supporting_read`, `block_id` |
| `recombination/` | `chromosome`, `position`, `recombination_rate`, `block_id` |
| `transitions/` | `sample_id`, `chromosome`, `position`, `switch_probability`, `stay_probability`, `offdiag_probability`, `block_id` |
| `transition_factors/` | `founder_index`, `source_factor`, `destination_factor`, `offdiag_distribution`, `rank`, `block_id` |
| `transition_summary/` | Per-interval transition and hotspot summaries. |
| `founder_updates/` | `chromosome`, `position`, `founder`, `alt_prob`, `block_id` |
| `diagnostics/` | Per-variant QC metrics. |
| `calibration_decisions/` | Per-SNP calibration/fallback decisions when calibration is enabled. |

Vector columns:

- `genotype_posterior` is a fixed-size list with one value per genotype class.
- `hap_dosage` and `hap_probability` are fixed-size lists with one value per founder.
- `source_factor` and `destination_factor` are fixed-size lists of length `4` for a rank-4 run.
- `offdiag_distribution` is a fixed-size list of length `K`, here `9`.

## Transition Outputs In K9 Rank 4

`transitions/` contains compact per-sample/per-position probabilities:

- `switch_probability`,
- `stay_probability`,
- `offdiag_probability`.

`transition_factors/` contains the low-rank factorized transition representation for each block:

- `founder_index`: source founder row,
- `source_factor`: rank-4 source factor vector,
- `destination_factor`: rank-4 destination factor vector,
- `offdiag_distribution`: learned/regularized off-diagonal destination distribution for that founder,
- `rank`: `4`,
- `block_id`.

`transition_summary/` is the default xi-style output for `--store-xi per-snp`. It is small and meant for transition hotspot/coldspot summaries. It includes columns such as:

- `chromosome`,
- `block_id`,
- `interval_index`,
- `from_position`,
- `to_position`,
- `physical_distance_bp`,
- `n_samples`,
- `n_founders`,
- `transition_model`,
- `transition_output`,
- `recombination_rate`,
- `switch_mean`,
- `switch_median`,
- `switch_min`,
- `switch_max`,
- `switch_p95`,
- `stay_mean`,
- `offdiag_mean`,
- `expected_copy_switches_mean`,
- `expected_any_copy_switch_probability_mean`,
- `hap_transition_entropy_mean`,
- `ploidy_scaled_transition_entropy_mean`,
- optional `genetic_distance_cM`,
- factorized diagnostics such as `factorized_offdiag_entropy_mean`, `factorized_offdiag_min`, and `factorized_offdiag_max`.

Full xi is not written by default. It should only be requested for small debugging runs.

## Validation Checklist

After each first-pass or second-pass run, inspect:

```bash
cat "$OUT/run_summary.json"
cat "$OUT/stage_timings.json"
cat "$OUT/memory_profile_summary.json"
cat "$OUT/founder_expansion_summary.json"
cat "$OUT/xi_output_manifest.json"
cat "$OUT/diagnostics_summary.json"
```

For a K9 rank4 run, confirm:

- `configured_n_founders` is `9`,
- `loaded_n_founders` is `8`,
- `final_n_founders` is `9`,
- `n_extra_mutable_founders` is `1`,
- `n_immutable_founders` is `8`,
- `n_mutable_founders` is `1`,
- `transition_model` is `factorized`,
- `transition_factor_rank` is `4`,
- `store_xi` is `per-snp`,
- calibration uses `quality_gated` when calibration is expected to affect hard calls,
- cache hits occur on second-pass or rerun commands.

For pedigree QC, inspect:

```bash
cat "$PED_QC/pedigree_qc_summary.json"
```

and open:

```text
$PED_QC/pedigree_umap_before_after.html
```

## Reading Results

Use Parquet/Zarr as the primary result format. VCF export is intentionally disabled. BCF export is only for interoperability and will slow I/O.

Example Python reading pattern:

```python
from pathlib import Path
import pandas as pd

run = Path("/project/runs/k9_rank4_production/second_pass_k9_rank4_pedigree/chr12")

samples = pd.read_parquet(run / "samples.parquet")
positions = pd.read_parquet(run / "positions.parquet")
calls = pd.read_parquet(run / "genotype_calls")
dosage = pd.read_parquet(run / "dosage")
transition_summary = pd.read_parquet(run / "transition_summary")
```

Combine partitioned outputs when convenient:

```bash
stitchv2 combine --run-output-dir "$OUT2"
```

Write combined floating outputs to Zarr:

```bash
stitchv2 combine \
  --run-output-dir "$OUT2" \
  --write-zarr \
  --zarr-output "$OUT2/combined/float_outputs.zarr" \
  --zarr-consolidated
```

## Reporting This Model

A clear methods sentence:

```text
We ran STITCHV2 with 8 immutable PLINK founders and 1 extra mutable founder (`K=9`), using factorized rank-4 founder-switch transitions, STITCH-compatible read filters and fragment likelihoods, JAX HMM execution, compressed Parquet/Zarr evidence caching, per-SNP transition summaries, and 40 requested EM iterations with adaptive best-founder restoration.
```

If pedigree is used, add:

```text
Pedigree information was not used in the first pass. First-pass genotype calls across chromosomes were used for R/R2-based pedigree QC and UMAP edge review. The curated pedigree was then supplied to the second-pass STITCHV2 run with transmission-mode pedigree adjustment.
```

If calibration is used, add:

```text
Hard calls were gated using standard callability calibration with LightGBM trained from held-out read-backed pseudo-truth. Calibration decisions were made per SNP with hierarchical fallback to STITCH-style no-call when unsupported or unsafe.
```

If `density_balanced_overlap` is used, add:

```text
The SNP block mode was `density_balanced_overlap`, an explicit approximate independent-block mode with overlap. Exact-streaming mode should be used for production parity claims when memory permits.
```
