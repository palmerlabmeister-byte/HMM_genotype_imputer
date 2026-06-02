# STITCHV2 Benchmarks

Status: legacy benchmark planning/reference document. The stable current benchmark protocol is [final_benchmark.md](../../final_benchmark.md), and observed benchmark results belong in [final_benchmark_results.md](../../final_benchmark_results.md) or a dataset-specific report.

This document describes how to benchmark STITCHV2 against original STITCH, on both synthetic data and real data styled after the STITCH paper workflow.

## Scope

Benchmark goals:
- Speed
- Memory
- Imputation quality
- Calibration quality

Required comparisons:
- STITCHV2 (JAX baseline)
- STITCH (R/C++ original)

As of April 29, 2026:
- STITCHV2 benchmarking is standardized on `fragment_coupling_model=stitch_parity`.
- `legacy_center` is no longer used for benchmark comparisons.

Recommended environment manager:
- `conda` (conda-forge channels)

---

## 1) Benchmark A: Synthetic data generated under the model

Use the built-in generator (`benchmarks/synthetic_dataset.py`) to create controlled truth.

Example (5 Mb, 2000 variants, 8 founders, 0.1x, recombination across N generations):

```bash
cd /home/bonnie/Documents/codex/STITCHV2

python benchmarks/synthetic_dataset.py \
  --output-dir benchmark_runs/synth_5mb_2k_0p1x \
  --chromosome chrSynthetic \
  --chromosome-length 5000000 \
  --n-variants 2000 \
  --n-founders 8 \
  --n-samples 48 \
  --coverage 0.1 \
  --generations 10 \
  --seed 7
```

Run STITCHV2 vs STITCH (probabilistic founders):

```bash
python benchmarks/benchmark_compare.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/compare_synth \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 2 \
  --block-size 1000 \
  --hmm-backend jax \
  --read-mode read_stream \
  --read-stream-backend auto \
  --founder-mode probabilistic
```

Run STITCHV2 vs STITCH (hard immutable founders, synthetic founder-truth parity mode):

```bash
python benchmarks/benchmark_compare.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/compare_synth_hard_founders \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --read-mode read_stream \
  --read-stream-backend auto \
  --founder-mode hard_immutable \
  --use-true-founders
```

Calibration-focused benchmark on synthetic:

```bash
python benchmarks/benchmark_calibration_strategies.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --run-dir benchmark_runs/compare_synth \
  --output-dir benchmark_runs/calibration_synth
```

---

## 1.1 Full synthetic comparison matrix (required)

For a complete STITCH vs STITCHV2 comparison on synthetic data, run all of:
- founder modes: `probabilistic`, `hard_immutable`
- calibration: `on`, `off`
- EM iterations: `1`, `5`, `10`
- fixed seed and fixed dataset

Single-command grid runner:

```bash
python benchmarks/run_triage_grid.py \
  --data-dir benchmark_runs/pilot_base_5mb_5k_200_0p01x \
  --output-dir benchmark_runs/triage_pilot_5k_200_0p01x_founder_modes \
  --chromosome chrSynthetic \
  --k 8 \
  --block-size 1000 \
  --hmm-backend jax \
  --read-mode read_stream \
  --read-stream-backend auto \
  --seed 7 \
  --em-iters 1,5,10 \
  --calibration on,off
```

Expected output:
- `triage_results.csv` with STITCHV2 metrics and `r_status`.
- `r_status` must be `ok` to count as a complete STITCH-vs-STITCHV2 comparison.

If `r_status=not_installed`, install STITCH in the benchmark environment and re-run.

---

## 1.2 Focused Founder-Mode Test (new)

This test directly answers:
- STITCHV2 with mutable founders (`probabilistic`)
- STITCHV2 with STITCH-like founder behavior (`hard_immutable` + `--use-true-founders`)
- STITCH side-by-side in the same run folders

```bash
python benchmarks/benchmark_synthetic_founder_modes.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/synth_founder_mode_test \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --read-mode read_stream \
  --read-stream-backend auto \
  --seed 7 \
  --rscript-path /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified/bin/Rscript
```

Expected output:
- `founder_mode_comparison.csv`
- `founder_mode_comparison.json`
- per-mode `benchmark_summary.json` under:
  - `founder_probabilistic/`
  - `founder_hard_immutable/`

---

## 1.3 Ploidy + Sex-Chromosome Benchmark (new)

This test verifies the new ploidy features run correctly and reports the core metrics in this repo:
- `dosage_r2`
- `f1`
- `info_mean`
- `missing_rate`

```bash
python benchmarks/benchmark_ploidy_modes.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/synth_ploidy_modes \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 2 \
  --block-size 1000 \
  --backend jax
```

Scenarios:
- `ploidy1`
- `ploidy2`
- `ploidy3`
- `ploidy4`
- `ploidy0_all_missing`
- `chrX_like` (`--ploidy-males 1 --ploidy-females 2`)
- `chrY_like` (`--ploidy-males 1 --ploidy-females 0`)

Outputs:
- `ploidy_benchmark_summary.csv`
- `ploidy_benchmark_summary.json`

---

## 1.4 JAX Generic Ploidy HMM Parity Benchmark

This test checks that the new generic count-state JAX HMM:
- AOT/warm-compiles polyploid kernels before heavy matrix work.
- Produces the same outputs as the optimized existing `P=1`/`P=2` paths when forced into `P=1`/`P=2`.
- Keeps production dispatch for `P=1`/`P=2` on the fast paths, so runtime is unchanged.
- Runs higher ploidy (`P=3`, `P=4`) through the generic JAX kernel.
- Reports the standard benchmark QC metrics: `R2`, `F1`, `Accuracy`, `Balanced accuracy`, `INFO mean/median`, and `missing_rate`.

```bash
python benchmarks/benchmark_jax_generic_ploidy_parity.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/jax_generic_ploidy_parity \
  --chromosome chrSynthetic \
  --k 8 \
  --n-samples 8 \
  --n-positions 48 \
  --iterations 1 \
  --depth 6 \
  --seed 11
```

Outputs:
- `jax_generic_ploidy_parity_summary.csv`
- `jax_generic_ploidy_parity_summary.json`

Interpretation:
- `p1_dispatch_parity` and `p2_dispatch_parity` compare old optimized dispatch against current production dispatch.
- `p1_forced_generic_correctness` and `p2_forced_generic_correctness` compare optimized outputs against the forced generic count-state kernel.
- `p3_dispatch` and `p4_dispatch` verify higher ploidy JAX execution and record generic AOT cache state.

Observed local smoke result on April 30, 2026:
- `P=1` dispatch parity: dosage/GP exact, warm runtime ratio `1.00076`.
- `P=2` dispatch parity: dosage/GP exact, warm runtime ratio `0.96475`.
- Forced generic correctness: `P=1` max dosage diff `1.79e-7`, max GP diff `1e-6`; `P=2` max dosage diff `5.96e-7`, max GP diff `2e-6`.
- Higher ploidy generic JAX: `P=3` state count `120`, warm runtime `0.039 s`; `P=4` state count `330`, warm runtime `0.144 s`.
- Generic JAX AOT evidence: forced `P=2`, `P=3`, and `P=4` runs each populated `jax_compiled_generic_cache=1` with `polyploid*` precompiled shapes.

---

## 1.4b Synthetic Pedigree Benchmark

This benchmark validates the optional pedigree post-HMM modes on a synthetic family dataset with true Mendelian transmission. The generated cohort includes:
- sequenced parents, siblings, and grandparents
- indexed empty BAMs for unsequenced targets
- unsequenced latent parents connecting grandparents to offspring
- a pedigree table with `sample_id`, `father_id`, and `mother_id`

It compares:
- `STITCHV2-off`
- `STITCHV2-smooth`
- `STITCHV2-kinship`
- `STITCHV2-transmission`
- original `STITCH`

```bash
python benchmarks/benchmark_pedigree_synthetic.py \
  --output-dir benchmark_runs/pedigree_synthetic_2026-05-01 \
  --n-direct-families 5 \
  --n-grandparent-families 4 \
  --n-variants 400 \
  --chromosome-length 1500000 \
  --coverage 0.15 \
  --iterations 5 \
  --block-size 200 \
  --jax-sample-batch-size 32 \
  --rscript-path /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified/bin/Rscript \
  --force
```

Outputs:
- `BENCHMARK_REPORT.md`
- `aggregate_metrics.csv` / `aggregate_metrics.parquet`
- `sample_group_metrics.csv` / `sample_group_metrics.parquet`
- `per_snp_metrics_long.csv` / `per_snp_metrics_long.parquet`
- `roc_points.csv` / `roc_points.parquet`
- `runtime_memory.csv`
- `benchmark_summary.json`
- plots under `plots/`

Observed local result on May 1, 2026:

| Method | R2 | F1 | Accuracy | Balanced accuracy | INFO mean | Missing rate | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| STITCHV2-off | 0.6723 | 0.7875 | 0.8082 | 0.7783 | 0.6389 | 0.0000 | 0.9405 |
| STITCHV2-smooth | 0.6376 | 0.8027 | 0.8223 | 0.7869 | 0.5659 | 0.0000 | 0.9412 |
| STITCHV2-kinship | 0.6781 | 0.7921 | 0.8128 | 0.7821 | 0.6575 | 0.0000 | 0.9459 |
| STITCHV2-transmission | 0.7376 | 0.8226 | 0.8418 | 0.8165 | 0.3803 | 0.0000 | 0.9592 |
| STITCH | -0.5538 | 0.3737 | 0.5274 | 0.4733 | NA | 0.3321 | 0.6547 |

Key pedigree validation result for unsequenced samples:
- `STITCHV2-off all_unsequenced R2`: `0.2473`
- `STITCHV2-transmission all_unsequenced R2`: `0.5001`
- `STITCH all_unsequenced R2`: `-0.2363`
- `STITCHV2-transmission` executed `792` pedigree messages over `44` parent-child edges.

Interpretation:
- The transmission layer materially improves unsequenced-target and latent-parent imputation.
- Original STITCH is included as a no-pedigree baseline; it does not consume the pedigree table and produces many missing calls for empty-BAM animals.
- STITCHV2 uses hard immutable true synthetic founders here to isolate pedigree behavior from founder-learning noise.

---

## 1.5 Dask-Orchestrated JAX Executor Benchmark

This test compares the existing serial JAX pipeline against the Dask executor, where Dask schedules coarse HMM leaf tasks by:
- planned variant block
- sample batch
- ploidy group

The forward-backward math remains inside the existing JAX kernels. Dask is used for orchestration, memory-bounded chunking, dashboard diagnostics, and performance-report capture.

```bash
python benchmarks/benchmark_dask_executor.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/dask_executor_synth \
  --chromosome chrSynthetic \
  --k 8 \
  --n-samples 12 \
  --n-positions 256 \
  --block-size 128 \
  --em-iterations 2 \
  --dask-n-workers 2 \
  --dask-threads-per-worker 1 \
  --dask-sample-batch-size 4 \
  --dask-dashboard-address :8787 \
  --force
```

Outputs:
- `dask_executor_summary.csv`
- `dask_executor_summary.json`
- `dask_jax/dask_run_summary.json`
- `dask_jax/dask_performance_report.html`
- `dask_jax/dask_task_stream.json`

The benchmark reports:
- `R2`
- `F1`
- `Accuracy`
- `Balanced accuracy`
- `INFO mean/median`
- `missing_rate`
- runtime and peak RSS
- serial-vs-Dask dosage equality (`max_abs_dosage_diff`, `mean_abs_dosage_diff`)

Observed local smoke result on April 30, 2026:
- Dataset: `benchmark_runs/synth_5mb_2k_0p1x`, `4` samples, `8` positions, `block_size=4`, `em_iterations=1`.
- Serial JAX and Dask-JAX produced identical QC metrics and `max_abs_dosage_diff=0.0`.
- Dask wrote a live dashboard URL plus `dask_performance_report.html` and `dask_task_stream.json`.

CLI equivalent:

```bash
stitchv2 run \
  --samples /path/to/samples.parquet \
  --positions /path/to/positions.parquet \
  --chromosome chrSynthetic \
  --output-dir benchmark_runs/example_dask_run \
  --n-founders 8 \
  --hmm-backend jax \
  --executor dask \
  --dask-n-workers 4 \
  --dask-threads-per-worker 1 \
  --dask-memory-limit 16GB \
  --dask-dashboard-address :8787 \
  --dask-performance-report dask_report.html \
  --dask-task-stream dask_task_stream.json
```

Notes:
- Keep chunks coarse; tiny chunks can be slower than serial JAX because Dask scheduling overhead dominates.
- For mutable founders, the Dask executor keeps full ploidy groups together to avoid changing EM/founder-update semantics.
- The implementation streams variant blocks through the pipeline for deterministic read extraction and writes, while Dask schedules the HMM leaf tasks for each planned block.

Three-way STITCH/STITCHV2/STITCHV2-Dask report:

```bash
python benchmarks/benchmark_stitch_stitchv2_dask_report.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/stitch_stitchv2_dask_2026-04-30 \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 5 \
  --block-size 1000 \
  --dask-dashboard-address 127.0.0.1:8786 \
  --force
```

Outputs:
- `BENCHMARK_REPORT.md`
- `aggregate_metrics.csv`
- `runtime_memory.csv`
- `per_snp_metrics_long.parquet`
- `roc_points.parquet`
- `plots/aggregate_metrics.png`
- `plots/runtime_memory.png`
- `plots/per_snp_metric_violins.png`
- `plots/roc_curves.png`
- `stitchv2_dask/dask_performance_report.html`

---

## 2) Benchmark B: Real data (paper-style) with subsampling + GATK pseudo-truth

Because real datasets do not have full ground truth, use the following protocol.

## Protocol

1. Build input table from STITCH example data
- use `bamlist.txt` and `pos.txt`
- keep a sample metadata table with `sample_id`, `bam_path`, `generation`

2. Subsample reads
- for ~50% of samples, randomly subsample to target coverage in `[0.01x, 1x]`
- keep per-sample subsampling metadata

3. Run imputers
- run STITCH and STITCHV2 on the subsampled reads

4. Build pseudo-truth from held-out reads
- take the non-used reads (the complement of subsampled reads)
- run GATK HaplotypeCaller on held-out reads
- keep only confidently called GATK variants

5. Compare
- intersect by sample + variant position
- compare STITCH/STITCHV2 predictions against GATK calls

Current strict helper harness for real subsampling:

```bash
python benchmarks/benchmark_real_subsample_gatk_compare.py \
  --data-dir /path/to/stitch_example_like_data \
  --output-dir benchmark_runs/real_subsample_gatk \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 2 \
  --holdout-sample-fraction 0.5 \
  --target-coverage-min 0.01 \
  --target-coverage-max 1.0 \
  --gatk-path gatk
```

Note:
- `benchmark_real_subsample_jax.py` remains available for fast proxy truth checks.
- For strict paper-style benchmarking and apples-to-apples comparison, use the GATK harness above.

---

## 3) What to measure

Measure all metrics for STITCH and STITCHV2, with STITCHV2 runtime/memory also broken by stage.

## Runtime (seconds)

Per run:
- total runtime

For STITCHV2 (stage breakdown):
- read IO (`seconds_read_extract`)
- calculation/HMM (`seconds_hmm`)
- calibration/QC (`seconds_calibration`)
- write IO (`seconds_write`)

Source:
- `python/stage_timings.json`
- benchmark summary JSON files

## Memory (MB)

Per run:
- peak RSS

For STITCHV2 (stage breakdown):
- after read IO
- after HMM
- after calibration/QC
- after write IO

Source:
- `python/memory_profile_summary.json`
- per-block RSS fields in `stage_timings.json`

## Quality metrics

Evaluate overall and per-variant distributions:
- `R2` (dosage vs truth)
- `F1` (macro-F1 on hard calls)
- `Accuracy`
- `Balanced accuracy`
- `INFO score`
- optional no-call/missing rate (if no-call mode used)

Comparison requirements:
- Report STITCHV2 for both founder modes (`probabilistic`, `hard_immutable`).
- Report calibrated and uncalibrated STITCHV2 runs separately.
- Report STITCH and STITCHV2 with identical sample/position intersections.
- Mark runs with missing STITCH outputs as incomplete.

Recommended plots:
- runtime stacked bar plot (stage breakdown)
- memory stacked bar plot (stage breakdown)
- violin plots per variant: `R2`, `F1`, `Accuracy`, `Balanced accuracy`
- ROC curves (at least STITCHV2 JAX vs STITCH)

---

## 4) Reporting

Use report generators included in `benchmarks/`:

```bash
python benchmarks/generate_calibration_benchmark_report.py \
  --benchmark-dir benchmark_runs/calibration_synth

python benchmarks/generate_real_subsample_benchmark_report.py \
  --benchmark-dir benchmark_runs/real_subsample_jax

python benchmarks/generate_real_subsample_gatk_compare_report.py \
  --benchmark-dir benchmark_runs/real_subsample_gatk
```

Common artifacts:
- `*_benchmark_summary.json`
- per-variant parquet metrics
- ROC parquet/json points
- markdown report with plot references

---

## 5) Reproducibility checklist

- Fix seeds (`--seed` and any model seed args).
- Record environment YAML used to run.
- Record STITCH commit/version and STITCHV2 commit hash.
- Report command lines verbatim in the markdown report.
- Keep the exact variant set and sample list fixed when comparing tools.

---

## 6) Capacity-Constrained 10k x 5k Run (Executed April 22, 2026)

This section documents a real run under explicit constraints:
- Max benchmark storage budget: `< 200 GB`
- Runtime budget target: `~2-3 hours` end-to-end
- STITCHV2 mode: `JAX` only, `argmax`, `no-calibration`
- Compare `with` and `without` haplotype-probability output

### 6.1 INFO score verification

Before the large run, INFO implementation was validated against expected edge cases and direct manual formula equivalence.

- Artifact: `benchmark_runs/info_score_validation.json`
- `perfect_info_case` gives INFO = `1.0`
- uniform posterior case matches expected `-1/3`
- manual formula difference: `0.0` max absolute difference

### 6.2 Pilot scaling (required pre-check)

Pilot runs at coverage `0.01x` were used to pre-check memory/runtime shape and storage budget.

- Artifact: `benchmark_runs/pilot_scaling/scaling_results.csv`

### 6.3 Full synthetic dataset used

Generated dataset:
- Path: `benchmark_runs/synth_5mb_5k_10k_0p01x_full`
- Parameters: 5 Mb, 5000 variants, 8 founders, 10,000 samples, 0.01x, 2 EM iterations
- STITCH-compatible files included (`pos.txt` with `CHR POS REF ALT`, `sample_names.txt`)

Generation/storage:
- Dataset folder size: `~285 MB` (`0.279 GB`)

### 6.4 Full benchmark results

STITCHV2 (JAX, argmax, no-cal), full 10k x 5k:

1) Without haplotype probabilities
- Output dir: `benchmark_runs/full_benchmark_2026-04-22/stitchv2_jax_5mb_5k_10k_0p01x_hap0`
- Elapsed: `330.19 s` (5.50 min)
- Peak RSS: `3288.88 MB`
- Stage breakdown read IO: `20.45 s`
- Stage breakdown HMM: `256.73 s`
- Stage breakdown calibration/QC: `8.44 s`
- Stage breakdown write IO: `44.14 s`

2) With haplotype probabilities
- Output dir: `benchmark_runs/full_benchmark_2026-04-22/stitchv2_jax_5mb_5k_10k_0p01x_hap1`
- Elapsed: `349.75 s` (5.83 min)
- Peak RSS: `4146.02 MB`
- Stage breakdown read IO: `11.04 s`
- Stage breakdown HMM: `251.30 s`
- Stage breakdown calibration/QC: `7.94 s`
- Stage breakdown write IO: `79.03 s`

STITCH (R) full 10k x 5k:

1) Without haplotype dosages
- Output dir: `benchmark_runs/full_benchmark_2026-04-22/stitch_r_5mb_5k_10k_0p01x_hap0`
- Elapsed: `7:02.51` (`422.51 s`)
- Peak RSS: `1002.48 MB`
- Status: completed

2) With haplotype dosages
- Output dir: `benchmark_runs/full_benchmark_2026-04-22/stitch_r_5mb_5k_10k_0p01x_hap1`
- Elapsed: `37:12.88` (`2232.88 s`)
- Peak RSS: `13478.57 MB`
- Status: completed

### 6.5 Storage cap check

Observed folder sizes (major artifacts):
- synthetic dataset: `~285 MB`
- STITCHV2 no-hap: `~24 MB`
- STITCHV2 hap: `~122 MB`
- STITCH no-hap: `~412 MB`
- STITCH hap: `~1.20 GB`

Total observed footprint for this benchmark set is `~2.04 GB`, far below the `200 GB` cap.

### 6.6 Commands used (key full runs)

STITCHV2 full run (no-hap, unified conda env):

```bash
conda run -p /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified bash -lc '
cd /home/bonnie/Documents/codex/STITCHV2 && \
python -m pip install -e . --no-build-isolation && \
stitchv2 run \
  --samples benchmark_runs/synth_5mb_5k_10k_0p01x_full/samples.parquet \
  --positions benchmark_runs/synth_5mb_5k_10k_0p01x_full/positions.parquet \
  --chromosome chrSynthetic \
  --output-dir benchmark_runs/full_benchmark_2026-04-22/stitchv2_jax_5mb_5k_10k_0p01x_hap0 \
  --n-founders 8 --block-size 1000 --em-iterations 2 \
  --hmm-backend jax --jax-sample-batch-size 128 \
  --read-mode read_stream --read-stream-backend auto \
  --io-workers 8 --htslib-threads-per-file 2 \
  --genotype-call-mode argmax --no-calibrate-genotype-posteriors \
  --write-genotype-posteriors --write-genotype-calls'
```

STITCHV2 full run (hap, unified conda env):

```bash
conda run -p /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified bash -lc '
cd /home/bonnie/Documents/codex/STITCHV2 && \
python -m pip install -e . --no-build-isolation && \
stitchv2 run \
  --samples benchmark_runs/synth_5mb_5k_10k_0p01x_full/samples.parquet \
  --positions benchmark_runs/synth_5mb_5k_10k_0p01x_full/positions.parquet \
  --chromosome chrSynthetic \
  --output-dir benchmark_runs/full_benchmark_2026-04-22/stitchv2_jax_5mb_5k_10k_0p01x_hap1 \
  --n-founders 8 --block-size 1000 --em-iterations 2 \
  --hmm-backend jax --jax-sample-batch-size 128 \
  --read-mode read_stream --read-stream-backend auto \
  --io-workers 8 --htslib-threads-per-file 2 \
  --genotype-call-mode argmax --no-calibrate-genotype-posteriors \
  --write-genotype-posteriors --write-genotype-calls \
  --write-haplotype-probabilities'
```

STITCH full run (no-hap, unified conda env):

```bash
conda run -p /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified /usr/bin/time -v \
Rscript benchmarks/run_r_stitch_benchmark.R \
  benchmark_runs/synth_5mb_5k_10k_0p01x_full \
  benchmark_runs/full_benchmark_2026-04-22/stitch_r_5mb_5k_10k_0p01x_hap0 \
  chrSynthetic 8 10 2 0
```

Full summary artifact:
- `benchmark_runs/full_benchmark_2026-04-22/capacity_and_benchmark_summary_10k5k_0p01x.json`

## 7) Real-Data-Style Re-evaluation with GATK (Executed April 22, 2026)

Re-evaluated benchmark directory:
- `benchmark_runs/real_subsample_jax_proxy_v5`

Benchmark command used:

```bash
conda run -p /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified \
python benchmarks/benchmark_real_subsample_gatk_compare.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/real_subsample_jax_proxy_v5 \
  --chromosome chrSynthetic \
  --k 8 --iterations 2 --block-size 1000 \
  --holdout-sample-fraction 0.5 \
  --target-coverage-min 0.01 --target-coverage-max 1.0 \
  --min-truth-depth 1 --min-truth-gq 0 \
  --io-workers 8 --htslib-threads-per-file 2 \
  --gatk-path gatk
```

Observed summary:
- Samples: `48` (holdout: `24`)
- Positions: `2000`
- GATK pseudo-truth calls after filtering: `10`
- STITCHV2 runtime: `2.31 s`, peak RSS: `553.18 MB`
- STITCH runtime: `1.47 s`, peak RSS: `117.42 MB`

Artifacts:
- `benchmark_runs/real_subsample_jax_proxy_v5/real_subsample_gatk_benchmark_summary.json`
- `benchmark_runs/real_subsample_jax_proxy_v5/real_subsample_gatk_variant_metrics.parquet`
- `benchmark_runs/real_subsample_jax_proxy_v5/real_subsample_gatk_roc_points.parquet`
- `benchmark_runs/real_subsample_jax_proxy_v5/report/real_subsample_gatk_benchmark_report.md`

## 8) Official STITCH Mouse Data Benchmarks (Executed April 22, 2026)

### 8.1 Data download and dataset characteristics

Official mouse panel from STITCH ancillary:
- Archive: `benchmark_runs/STITCH_example_2016_05_10.tgz`
- Extracted data dir: `benchmark_runs/STITCH_example_2016_05_10_data`
- Samples (`bamlist.txt`): `2073`
- Positions (`pos.txt`): `1516`
- Chromosome/window: `chr19:10,000,105-10,996,035` (the 1 Mb profiling panel used in STITCH benchmarking docs)

Reference for GATK:
- `benchmark_runs/mm10_2016_10_02.fa.gz` (official ancillary reference)
- unpacked to `benchmark_runs/mm10_2016_10_02.fa`
- indexed files: `mm10_2016_10_02.fa.fai`, `mm10_2016_10_02.dict`

### 8.2 Paper-profile runtime mirror on full 2,073 samples

Run:
- Output dir: `benchmark_runs/stitch_paper_mouse_profile_2073_k4_apr22`
- Summary: `benchmark_runs/stitch_paper_mouse_profile_2073_k4_apr22/benchmark_summary.json`
- Short report: `benchmark_runs/stitch_paper_mouse_profile_2073_k4_apr22/report.md`

Key numbers (`K=4`, `nGen=100`, `niterations=2`, all 2073 samples, all 1516 SNPs):
- STITCHV2 JAX: `29.44 s`, peak RSS `1582.57 MB`
- STITCH (R/C++): `1:42.85` (`102.85 s`), peak RSS `192.07 MB`
- Runtime ratio (STITCH / STITCHV2): `3.49x` in favor of STITCHV2 on this profile run

### 8.3 Real holdout benchmark on official panel + GATK pseudo-truth

Latest rerun:
- Output dir: `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01`
- Report: `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/report/real_subsample_gatk_benchmark_report.md`
- Command used `benchmark_real_subsample_gatk_compare.py` with:
  - `max_samples=600` (few-hundred scale subset from original 2073)
  - `holdout_sample_fraction=0.5` (`300` holdout samples)
  - full panel positions (`1516`)
  - target holdout coverage sampled in `[0.01x, 1.0x]`
  - reference: `mm10_2016_10_02.fa`
  - `K=4`, `nGen=100`, `niterations=2`

```bash
PATH=/home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified/bin:$PATH \
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
.conda-benchmark-unified/bin/python benchmarks/benchmark_real_subsample_gatk_compare.py \
  --data-dir benchmark_runs/STITCH_example_2016_05_10_data \
  --output-dir benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01 \
  --chromosome chr19 \
  --k 4 \
  --generations 100 \
  --iterations 2 \
  --block-size 500 \
  --max-samples 600 \
  --region-start 10000105 \
  --region-end 10996035 \
  --holdout-sample-fraction 0.5 \
  --target-coverage-min 0.01 \
  --target-coverage-max 1.0 \
  --min-truth-depth 1 \
  --min-truth-gq 0 \
  --io-workers 8 \
  --htslib-threads-per-file 2 \
  --read-stream-backend auto \
  --jax-sample-batch-size 128 \
  --reference-fasta benchmark_runs/mm10_2016_10_02.fa \
  --rscript-path /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified/bin/Rscript \
  --gatk-path /home/bonnie/Documents/codex/STITCHV2/.conda-benchmark-unified/bin/gatk \
  --gatk-java-xmx 2g
```

Observed summary (`real_subsample_gatk_benchmark_summary.json`):
- `n_samples`: `600`
- `n_holdout_samples`: `300`
- `n_positions`: `1516`
- `n_truth_calls`: `2002`
- Coverage targets (holdout): min `0.01033`, mean `0.22691`, max `0.98635`

Runtime / memory:
- STITCHV2 JAX: `11.11 s`, peak RSS `735.41 MB`
- STITCH: `7.59 s`, peak RSS `166.34 MB`
- STITCHV2 stage runtime breakdown:
  - IO read: `2.89 s`
  - HMM: `6.67 s`
  - Calibration: `0.24 s`
  - IO write: `1.28 s`

Quality versus GATK pseudo-truth:
- STITCHV2:
  - R2 `-257.73`, F1 `0.0010`, Accuracy `0.0015`, Balanced Accuracy `0.3333`, ROC AUC `0.9835`, INFO mean `0.2815`
- STITCH:
  - R2 `-29.81`, F1 `0.3239`, Accuracy `0.9206`, Balanced Accuracy `0.4184`, ROC AUC `0.9845`

Artifacts:
- `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/real_subsample_gatk_benchmark_summary.json`
- `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/real_subsample_gatk_variant_metrics.parquet`
- `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/real_subsample_gatk_roc_points.parquet`
- `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/report/real_subsample_gatk_benchmark_report.md`

Note:
- On this holdout configuration, STITCHV2 hard-call outputs collapse near dosage `~1` at most pseudo-truth loci (genotype call mostly `1`), producing poor hard-call metrics despite high ROC AUC. This indicates a model/parity gap on this real panel that requires further work (founder initialization, allele handling parity, and/or calibration strategy).

### 8.3.1 Calibration and no-call sweep for STITCHV2 JAX

Follow-up question tested on the same official STITCH mouse + GATK pseudo-truth benchmark:
- Is the hard-call failure caused by STITCHV2's default posterior calibration?
- Does STITCH-style no-call thresholding recover calls?
- What happens when we force every genotype to be called versus allowing no-calls at several posterior thresholds?
- Can a masked-CV calibrator tune temperature, blend, dosage scale, MAF-bin behavior, and an HWE soft penalty from held-out pseudo-truth?

Runner:
- `benchmarks/benchmark_real_calling_calibration_sweep.py`

Artifacts:
- Report: `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/calibration_calling_sweep/CALLING_CALIBRATION_SWEEP.md`
- Overall metrics: `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/calibration_calling_sweep/calling_calibration_sweep_overall.csv`
- Per-variant metrics: `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/calibration_calling_sweep/calling_calibration_sweep_variant_metrics.parquet`
- ROC points: `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/calibration_calling_sweep/calling_calibration_sweep_roc_points.parquet`
- Plots: `benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/calibration_calling_sweep/plots/`

Command:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
.conda-benchmark-unified/bin/python benchmarks/benchmark_real_calling_calibration_sweep.py \
  --benchmark-dir benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01 \
  --chromosome chr19 \
  --k 4 \
  --iterations 2 \
  --block-size 500 \
  --jax-sample-batch-size 128 \
  --io-workers 8 \
  --htslib-threads-per-file 2 \
  --read-stream-backend auto \
  --thresholds 0.5,0.7,0.8,0.9,0.95,0.99 \
  --masked-cv-hwe-weight 0.02
```

Latest masked-CV rerun uses a stratified 50/50 split of the 2,002 GATK pseudo-truth calls:
- Train pseudo-truth calls: `1,001`
- Held-out evaluation pseudo-truth calls: `1,001`
- Evaluation truth counts: `GT0=0`, `GT1=2`, `GT2=999`

Key held-out result:

| Method | R2 | F1 | Accuracy | Missing rate | ROC AUC | Mean predicted dosage on truth calls | Pred no-call | Pred GT0 | Pred GT1 | Pred GT2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STITCHV2 calibrated, always call | -451.8546 | 0.0013 | 0.0020 | 0.0000 | nan | 1.0489 | 0 | 0 | 1001 | 0 |
| STITCHV2 raw, always call | -451.8546 | 0.0013 | 0.0020 | 0.0000 | nan | 1.0489 | 0 | 0 | 1001 | 0 |
| STITCHV2 masked-CV, always call | -201.9104 | 0.2501 | 0.5944 | 0.0000 | nan | 1.5941 | 0 | 0 | 406 | 595 |
| STITCHV2 masked-CV, STITCH-style no-call 0.9 | -201.9104 | 0.2501 | 0.5944 | 0.0000 | nan | 1.5941 | 0 | 0 | 406 | 595 |
| STITCH | -51.9655 | 0.3278 | 0.9201 | 0.0000 | nan | 1.7488 | 0 | 2 | 78 | 921 |

Important interpretation:
- The GATK pseudo-truth calls are extremely imbalanced, and the held-out split contains no `GT0`, so ROC AUC is undefined in this split.
- STITCHV2 JAX calls every evaluated genotype as heterozygous (`GT1`) whether calibration is enabled or disabled.
- Masked-CV calibration selected stronger dosage scaling in the nearly fixed/rare MAF bin (`temperature=0.15`, `blend=1.0`, `dosage_scale=2.0`, `dosage_offset=0.25`) and partially recovered homozygous-alt calls.
- The default STITCHV2 calibration used here is fixed posterior reshaping (`temperature=0.35`, `blend=0.35`), not a truth-trained calibrator and not STITCH's internal posterior model.
- STITCH's VCF writer uses a hard-call posterior threshold of `0.9`; STITCHV2's equivalent is `genotype_call_mode="stitch_no_call"` with `genotype_call_stitch_threshold=0.9`.
- The exact STITCH VCF writer rule is exposed in code as `stitch_vcf_genotype_call_from_posterior(...)`.
- Fixed calibration plus STITCH-style no-call does not recover accuracy. Masked-CV helps, but STITCH still calls many more held-out sites as `GT2`.
- Current evidence points to both: a useful calibration gap and an upstream HMM/founder/dosage-scale parity problem.

### 8.4 Synthetic haplotype-output comparison artifact (STITCH HD on)

To ensure a completed STITCH HD-on comparison while large 10k x 5k HD output remains long-running:
- Run dir: `benchmark_runs/compare_synth_5mb_2k_0p1x_hap_on_apr22`
- Summary: `benchmark_runs/compare_synth_5mb_2k_0p1x_hap_on_apr22/benchmark_summary.json`
- Short report: `benchmark_runs/compare_synth_5mb_2k_0p1x_hap_on_apr22/report.md`

## 9) Masked-CV Calibration Benchmark (Executed May 4, 2026)

Implementation:
- Core function: `masked_cv_calibrate_genotype_posterior(...)`
- CLI/pipeline mode: `--calibration-mode masked_cv`
- Tuned grids:
  - `--calibration-maf-bins`
  - `--calibration-temperatures`
  - `--calibration-blends`
  - `--calibration-dosage-scales`
  - `--calibration-dosage-offsets`
  - `--calibration-hwe-weight`

Synthetic rerun:
- Data dir: `benchmark_runs/synth_5mb_2k_0p1x`
- Base run dir: `benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_base_stitchok`
- Output dir: `benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_masked_cv_2026-05-04`
- Report: `benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_masked_cv_2026-05-04/report/calibration_benchmark_report.md`

Command:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/mpl \
.conda-benchmark-unified/bin/python benchmarks/benchmark_calibration_strategies.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --run-dir benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_base_stitchok \
  --output-dir benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_masked_cv_2026-05-04 \
  --holdout-fraction 0.2 \
  --masked-cv-hwe-weight 0.02
```

Synthetic held-out result:

| Strategy | R2 | F1 | Accuracy | Balanced accuracy | Call rate | INFO mean | Runtime (s) | Peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure_argmax | 0.9555 | 0.9666 | 0.9716 | 0.9707 | 1.0000 | 0.8989 | 3.38 | 504.21 |
| stitch_no_call_balanced | 0.9548 | 0.9859 | 0.9884 | 0.9867 | 0.9167 | 0.9050 | 3.44 | 505.21 |
| masked_cv_calibrated | 0.9570 | 0.9690 | 0.9733 | 0.9714 | 1.0000 | 0.9316 | 9.98 | 513.51 |
| STITCH | -0.7961 | 0.3666 | 0.5043 | 0.4370 | 0.8255 | 0.7600 | 2.09 | 129.06 |

Interpretation:
- Masked-CV improves always-call posterior quality and INFO on synthetic data.
- The best F1 is still the tuned no-call strategy because it deliberately drops low-confidence calls.
- STITCHV2's calibration behavior is now tunable by MAF bin; on this synthetic panel all variants landed in the common MAF bin (`MAF > 0.05`).
