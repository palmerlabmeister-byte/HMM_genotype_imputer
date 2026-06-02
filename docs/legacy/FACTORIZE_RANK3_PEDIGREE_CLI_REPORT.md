# STITCHV2 Factorized Rank 3, Calibration, and Pedigree CLI Report

Status: historical factorized/pedigree CLI report. For current production instructions, use [docs/LOW_RANK_K9_PRODUCTION_RUN.md](../LOW_RANK_K9_PRODUCTION_RUN.md).

This report explains the current HS rats parity/calibration findings and gives a command-line workflow for:

1. a first no-pedigree pass for pedigree QC,
2. pedigree QC with before/after UMAP edge plots,
3. a second pedigree-aware STITCHV2 pass with factorized rank 3 transitions and LightGBM callability calibration.

The commands below are instructions only. They do not modify `final_benchmark.md` and they do not contain benchmark results that should be appended to that file.

## Current Diagnosis

The most recent 10 Mb HS rats comparison was:

`benchmark_HSrats/hsrats_chr12_10mb_factorized_rank3_20260522_comparison.md`

| method | elapsed_s | peak_rss_mb | dosage_r2 | accuracy | balanced_accuracy | f1_macro | call_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| STITCHV2 parity | 132.51 | 9988.07 | 0.96971 | 0.98853 | 0.98869 | 0.98868 | 0.97663 |
| Original STITCH | 116.51 | 369.96 | 0.97608 | 0.99097 | 0.99106 | 0.99109 | 0.98536 |
| STITCHV2 LightGBM + factorized rank 3 | 312.96 | 16268.12 | 0.97601 | 0.99103 | 0.99112 | 0.99114 | 0.98523 |

This comparison was useful, but it was not clean enough to answer all model questions because the factorized run reused a support-only evidence cache.

## Why STITCHV2 Parity Is Not Exactly STITCH

The current parity gap is small but real:

- STITCHV2 parity dosage R2: `0.96971`
- STITCH dosage R2: `0.97608`
- STITCHV2 parity call rate: `0.97663`
- STITCH call rate: `0.98536`

The gap is not explained by generation count. The HS rats `samples.parquet` has `generation=80` for all 1,933 samples, and the STITCH wrapper ran with scalar `nGen=80`.

The likely remaining causes are:

1. **Read evidence is still not guaranteed byte-for-byte identical.**
   The benchmark wrapper calls original STITCH with its R defaults. STITCHV2 `--stitch-compat` sets `expRate=0.5`, `min_base_quality=17`, `min_mapping_quality=17`, base quality capped by mapping quality, `max_insert_size=600`, ref/alt-only evidence, `fragment_likelihood_mode=replace`, and STITCH-like likelihood caps. If any installed STITCH internal read filter differs from those assumptions, low-coverage hard calls can move.

2. **No-call/hard-call decoding can amplify tiny dosage differences.**
   At about 0.1x coverage, a small change in fragment assignment or GP near the 0.9 no-call boundary can change F1, balanced accuracy, and call rate more than it changes AUC.

3. **Fragment construction is close to STITCH but still needs a direct evidence audit.**
   The STITCHV2 `stitch_style_bamreader` is intended to match STITCH's moving SNP-range scan and central-SNP behavior. The next parity audit should compare original STITCH `input/sample.*.input.chr12.RData` read observations against STITCHV2 compact fragment evidence for the same samples and positions.

4. **Founder input is probably not the main issue.**
   Both methods used the same hardened founder PLINK source in the benchmark harness. The remaining founder risk is only allele hardening/orientation at ambiguous heterozygous founder sites, not mutable-founder EM.

The immediate validation run for parity should use no calibration, no pedigree, fixed immutable founders, `--transition-model stitch_parity`, and either no cache or a dense cache. Do not use support-only cached evidence for parity-sensitive checks.

## Why LightGBM Calibration Is Not Helping Much

There are three separate issues.

### 1. The prior HS rats benchmark did not let standard callability affect hard calls

The prior `benchmark_HSrats.py` run used:

`stitchv2_genotype_call_mode = stitch_no_call`

In the STITCHV2 pipeline, `standard_callability` keeps raw HMM GP unchanged and learns `P(argmax genotype is correct)`. That learned probability only affects hard calls when the run uses:

`--genotype-call-mode quality_gated`

With `--genotype-call-mode stitch_no_call`, hard calls are still emitted by GP >= `--genotype-call-stitch-threshold` and the LightGBM callability model is mostly diagnostic. This is the first thing to fix in the next calibration benchmark.

### 2. Read-backed labels are sparse at 0.1x coverage

The factorized run had:

- read-backed calibration labels: `4,370`
- HMM cells: `256 samples x 33,878 positions`
- mean labels per SNP: `0.0378`
- SNPs using calibration: `15,910`
- SNPs falling back to STITCH no-call: `17,968`
- main fallback reason: `insufficient_labels` on `17,242` SNPs

So LightGBM is learning from a very small number of high-confidence read-backed labels. That is the right direction to avoid microarray leakage, but it limits how much calibration can improve a 10 Mb, 0.1x subset.

### 3. The factorized cached run loaded support-only evidence

The shared cache was written with:

`--compact-evidence-no-dense-counts`

and without:

`--compact-evidence-cache-include-dense-counts`

The fresh STITCHV2 parity run reported mean depth `0.26968`, while the cached factorized run reported mean depth `0.18798`. That ratio is almost exactly the 30 percent read holdout effect and shows the cached run was not seeing the same dense count features. The compact fragment rows were present, but dense depth/count features for calibration were support-like rather than exact.

For future calibration and factorized runs, build the cache with dense counts:

`--compact-evidence-cache-include-dense-counts`

and do not pass:

`--compact-evidence-no-dense-counts`

For the 10 Mb subset, the compact cache without dense arrays was only `13.30 MB`; the dense logical matrix estimate was `173.46 MB`. Dense cached Zarr should still be affordable for this subset and is the correct comparison mode.

## Is The Same Issue Present In Factorized Rank 3?

Yes, the same calibration application issue is present if factorized rank 3 is run with `--genotype-call-mode stitch_no_call`.

The factorized run also shared the support-only cache problem. It reported a lower mean depth than the fresh parity extraction, so it should be rerun from a dense cache before deciding whether rank 3 itself helps or hurts.

The good news is that the rank 3 run still matched original STITCH closely on the microarray metrics. The caution is that the run was not a clean test of LightGBM callability because callability was not used for hard-call gating.

## Best Next Calibration Settings

For the next real calibration benchmark, use:

- `--calibration-mode standard_callability`
- `--calibration-callability-model lightgbm`
- `--calibration-callability-decision-mode per_snp_hierarchical`
- `--calibration-truth-source read_evidence`
- `--calibration-read-truth-holdout-fraction 0.30`
- `--genotype-call-mode quality_gated`
- `--compact-evidence-cache-include-dense-counts`

Do not add `--use-lightgbm-calibrator` unless intentionally testing the older multiclass posterior recalibration stack. The standard default should be LightGBM callability, not posterior reshaping.

To make calibration better and faster:

- train/apply on larger windows or whole chromosomes so read-backed labels are less sparse,
- reuse the dense evidence cache for all calibration experiments,
- keep `per_snp_hierarchical` as the default but report fallback reasons per SNP,
- use `quality_gated` hard calls so LightGBM actually changes calls,
- keep the STITCH call-rate bounds initially, then test small relaxations only in a benchmark,
- write calibration decisions and diagnostics on every run,
- avoid microarray truth for training unless explicitly running a leakage-positive control.

Factorized rank 3 is currently slower because the factorized transition path cannot use the fastest JAX count/fragment emission shortcut. It still uses JAX for factorized forward-backward, but fragment emission is less fused than the `stitch_parity` path. A later speed improvement should add a factorized-compatible JAX fragment emission/count kernel.

## Input Files

Required inputs for the CLI workflow:

| variable | expected file | required columns or content |
| --- | --- | --- |
| `SAMPLES` | parquet/csv | `sample_id`, `bam_path`, `generation`; optional `sex`, pedigree metadata |
| `POSITIONS` | parquet/csv | `CHR`, `POS`, `REF`, `ALT` |
| `FOUNDER_PLINK` | PLINK prefix | `.bed`, `.bim`, `.fam`; founder chromosome/positions must match `POSITIONS` |
| `PEDIGREE_RAW` | parquet/csv | child and parent columns, default `sample_id`, `father_id`, `mother_id` |
| `CACHE` | directory | partitioned evidence cache written by STITCHV2 |

If the founder file has 8 founder samples and the command uses `--n-founders 9 --founder-immutable`, STITCHV2 freezes the 8 loaded founders and appends 1 mutable founder initialized at ALT probability 0.5.

## Shell Setup

Edit these paths once per project.

```bash
PROJECT=/home/bonnie/Documents/codex/STITCHV2
RUN_ROOT="$PROJECT/runs/factorized_rank3_pedigree"
SAMPLES="$PROJECT/test_data_own/samples.parquet"
POSITIONS="$PROJECT/positions.parquet"
FOUNDER_PLINK="$PROJECT/test_data_own/founders8"
PEDIGREE_RAW="$PROJECT/pedigree.parquet"
CACHE="$RUN_ROOT/evidence_cache_dense"
JAX_CACHE="$RUN_ROOT/jax_cache"

mkdir -p "$RUN_ROOT" "$CACHE" "$JAX_CACHE"
cd "$PROJECT"
stitchv2 --help
```

`POSITIONS` must point to the reviewed position table for the chromosome(s) being run. For HS rats, this should be the founder-position table with columns `CHR`, `POS`, `REF`, and `ALT`.

## First Pass: No Pedigree, Fixed Founders, Dense Evidence Cache

This pass is for stable pedigree QC. It should be conservative and close to STITCH, so use `stitch_parity`, no calibration, no pedigree, and immutable founders.

```bash
CHR=chr12
OUT1="$RUN_ROOT/first_pass/$CHR"

stitchv2 run \
  --samples "$SAMPLES" \
  --positions "$POSITIONS" \
  --chromosome "$CHR" \
  --output-dir "$OUT1" \
  --n-founders 8 \
  --founder-plink "$FOUNDER_PLINK" \
  --founder-immutable \
  --stitch-compat \
  --read-mode read_stream \
  --read-stream-backend stitch_style_bamreader \
  --hmm-backend jax \
  --jax-sample-batch-size 0 \
  --jax-persistent-cache-dir "$JAX_CACHE" \
  --max-mem 90% \
  --block-size 0 \
  --io-window-size 0 \
  --compact-evidence-cache-dir "$CACHE" \
  --compact-evidence-cache-mode readwrite \
  --compact-evidence-cache-format parquet_zarr \
  --compact-evidence-cache-include-dense-counts \
  --fragment-coupling-model stitch_parity \
  --fragment-likelihood-mode replace \
  --transition-model stitch_parity \
  --transition-output compact \
  --em-iterations 1 \
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

Do not add `--compact-evidence-no-dense-counts` to this workflow. Dense counts are needed for exact cache reuse and for calibration features.

For all chromosomes, run the same command once per chromosome and write one output directory per chromosome:

```bash
for CHR in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20; do
  OUT1="$RUN_ROOT/first_pass/$CHR"
  stitchv2 run \
    --samples "$SAMPLES" \
    --positions "$POSITIONS" \
    --chromosome "$CHR" \
    --output-dir "$OUT1" \
    --n-founders 8 \
    --founder-plink "$FOUNDER_PLINK" \
    --founder-immutable \
    --stitch-compat \
    --read-mode read_stream \
    --read-stream-backend stitch_style_bamreader \
    --hmm-backend jax \
    --jax-sample-batch-size 0 \
    --jax-persistent-cache-dir "$JAX_CACHE" \
    --max-mem 90% \
    --block-size 0 \
    --io-window-size 0 \
    --compact-evidence-cache-dir "$CACHE" \
    --compact-evidence-cache-mode readwrite \
    --compact-evidence-cache-format parquet_zarr \
    --compact-evidence-cache-include-dense-counts \
    --fragment-coupling-model stitch_parity \
    --fragment-likelihood-mode replace \
    --transition-model stitch_parity \
    --transition-output compact \
    --em-iterations 1 \
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

## Pedigree QC

Run pedigree QC after the first pass. Use multiple `--run-output-dir` arguments to aggregate evidence across chromosomes. This is better than making each chromosome curate the pedigree independently.

```bash
PED_QC="$RUN_ROOT/pedigree_qc"

PED_RUN_ARGS=()
for CHR in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20; do
  PED_RUN_ARGS+=(--run-output-dir "$RUN_ROOT/first_pass/$CHR")
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

Main outputs:

| output | meaning |
| --- | --- |
| `pedigree_curated.parquet` | directly usable by `stitchv2 run --pedigree` |
| `pedigree_edge_qc.parquet` | parent-child edge status before/after curation |
| `pedigree_sample_qc.parquet` | sample-level pedigree QC status |
| `sample_similarity.parquet` | reported high-similarity or suspicious sample pairs |
| `pedigree_umap_before_after.html` | UMAP plot with pedigree edges before and after curation |
| `pedigree_long_edges.parquet` | long UMAP pedigree edges that may indicate wrong parents or sample swaps |
| `pedigree_qc_summary.json` | thresholds, counts, and resolved column names |

Review at least:

```bash
cat "$PED_QC/pedigree_qc_summary.json"
```

and open:

`$PED_QC/pedigree_umap_before_after.html`

## Second Pass: Pedigree-Aware Factorized Rank 3 With LightGBM Callability

This is the production-style exploratory model:

- 8 immutable PLINK founders plus 1 mutable founder,
- up to 40 EM iterations with adaptive convergence,
- factorized rank 3 transitions,
- LightGBM standard callability,
- quality-gated hard calls so calibration actually affects emitted calls,
- curated pedigree in `transmission` mode.

```bash
CHR=chr12
OUT2="$RUN_ROOT/factorized_rank3_pedigree_calibrated/$CHR"
CURATED_PED="$PED_QC/pedigree_curated.parquet"

stitchv2 run \
  --samples "$SAMPLES" \
  --positions "$POSITIONS" \
  --chromosome "$CHR" \
  --output-dir "$OUT2" \
  --n-founders 9 \
  --founder-plink "$FOUNDER_PLINK" \
  --founder-immutable \
  --stitch-compat \
  --read-mode read_stream \
  --read-stream-backend stitch_style_bamreader \
  --hmm-backend jax \
  --jax-sample-batch-size 0 \
  --jax-persistent-cache-dir "$JAX_CACHE" \
  --max-mem 90% \
  --block-size 0 \
  --io-window-size 0 \
  --compact-evidence-cache-dir "$CACHE" \
  --compact-evidence-cache-mode readwrite \
  --compact-evidence-cache-format parquet_zarr \
  --compact-evidence-cache-include-dense-counts \
  --fragment-coupling-model stitch_parity \
  --fragment-likelihood-mode replace \
  --transition-model factorized \
  --transition-factor-rank 3 \
  --transition-factor-regularization 100 \
  --transition-factor-max-deviation 0.25 \
  --transition-output factorized \
  --write-transitions \
  --em-iterations 40 \
  --em-convergence-tol 1e-4 \
  --em-convergence-min-iterations 2 \
  --em-convergence-patience 1 \
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
  --write-support-mask \
  --diagnostics-fail-on-error
```

If strict STITCH comparison is the goal rather than production exploratory imputation, use `--n-founders 8`, `--em-iterations 1`, `--transition-model stitch_parity`, `--transition-output compact`, and `--pedigree-mode off`.

For all chromosomes:

```bash
CURATED_PED="$RUN_ROOT/pedigree_qc/pedigree_curated.parquet"

for CHR in chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20; do
  OUT2="$RUN_ROOT/factorized_rank3_pedigree_calibrated/$CHR"
  stitchv2 run \
    --samples "$SAMPLES" \
    --positions "$POSITIONS" \
    --chromosome "$CHR" \
    --output-dir "$OUT2" \
    --n-founders 9 \
    --founder-plink "$FOUNDER_PLINK" \
    --founder-immutable \
    --stitch-compat \
    --read-mode read_stream \
    --read-stream-backend stitch_style_bamreader \
    --hmm-backend jax \
    --jax-sample-batch-size 0 \
    --jax-persistent-cache-dir "$JAX_CACHE" \
    --max-mem 90% \
    --block-size 0 \
    --io-window-size 0 \
    --compact-evidence-cache-dir "$CACHE" \
    --compact-evidence-cache-mode readwrite \
    --compact-evidence-cache-format parquet_zarr \
    --compact-evidence-cache-include-dense-counts \
    --fragment-coupling-model stitch_parity \
    --fragment-likelihood-mode replace \
    --transition-model factorized \
    --transition-factor-rank 3 \
    --transition-factor-regularization 100 \
    --transition-factor-max-deviation 0.25 \
    --transition-output factorized \
    --write-transitions \
    --em-iterations 40 \
    --em-convergence-tol 1e-4 \
    --em-convergence-min-iterations 2 \
    --em-convergence-patience 1 \
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
    --write-support-mask \
    --diagnostics-fail-on-error
done
```

## Optional Dask Mode

For local runs, keep worker processes off unless explicitly testing process isolation:

```bash
--executor dask \
--dask-scheduler local \
--dask-n-workers 4 \
--dask-threads-per-worker 1 \
--dask-dashboard-address :8787 \
--dask-performance-report "$OUT2/dask_performance_report.html" \
--dask-task-stream "$OUT2/dask_task_stream.html"
```

Do not add `--dask-processes` for the default local workflow. Threads avoid a lot of Python serialization overhead for the current cache and HMM objects.

## Validation Checks After Each Run

Check the run summary:

```bash
cat "$OUT2/run_summary.json"
cat "$OUT2/stage_timings.json"
cat "$OUT2/diagnostics_summary.json"
cat "$OUT2/pedigree_summary.json"
```

Check calibration usage:

```bash
ls "$OUT2/calibration_decisions"
```

Expected calibration audit fields:

- `n_snps_calibration_used`
- `n_snps_fallback_to_stitch`
- `calibration_fallback_reason_counts`
- per-SNP `call_rate_stitch`
- per-SNP `call_rate_calibrated`
- per-SNP `maf_shift`
- per-SNP `het_shift`

Check cache reuse:

- first pass should report `compact_cache_hit=false` when building the cache,
- later passes should report `compact_cache_hit=true`,
- dense cache runs should not show a large drop in `mean_depth` between fresh and cached runs,
- cache manifest parts should say `includes_dense_counts=true`.

## Clean Parity Recheck Command

Use this command when checking whether STITCHV2 parity has drifted from original STITCH. It intentionally disables calibration and pedigree effects.

```bash
CHR=chr12
PARITY_OUT="$RUN_ROOT/parity_recheck/$CHR"

stitchv2 run \
  --samples "$SAMPLES" \
  --positions "$POSITIONS" \
  --chromosome "$CHR" \
  --output-dir "$PARITY_OUT" \
  --n-founders 8 \
  --founder-plink "$FOUNDER_PLINK" \
  --founder-immutable \
  --stitch-compat \
  --read-mode read_stream \
  --read-stream-backend stitch_style_bamreader \
  --hmm-backend jax \
  --jax-sample-batch-size 0 \
  --jax-persistent-cache-dir "$JAX_CACHE" \
  --max-mem 90% \
  --compact-evidence-cache-dir "$CACHE" \
  --compact-evidence-cache-mode readwrite \
  --compact-evidence-cache-format parquet_zarr \
  --compact-evidence-cache-include-dense-counts \
  --fragment-coupling-model stitch_parity \
  --fragment-likelihood-mode replace \
  --transition-model stitch_parity \
  --transition-output compact \
  --em-iterations 1 \
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

If this still differs from original STITCH, the next audit should compare per-sample STITCH RData read observations to STITCHV2 compact fragments before touching the HMM.
