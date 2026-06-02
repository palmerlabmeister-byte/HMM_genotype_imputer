# Full Benchmark Report

Status: historical benchmark report retained for formatting/context. Do not use this as the current benchmark protocol; use [final_benchmark.md](../../final_benchmark.md) for the stable protocol and write new results to [final_benchmark_results.md](../../final_benchmark_results.md) or a fresh dataset-specific report.

## Scope
- Synthetic benchmark under model-generated truth (5 Mb, 5,000 variants, 10,000 samples, 0.01x).
- STITCHV2 JAX (`argmax`, no-calibration), with and without haplotype probabilities.
- STITCH diploid run, with and without haplotype dosages (`output_haplotype_dosages`).

## Inputs
- Data dir: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/synth_5mb_5k_10k_0p01x_full`
- STITCHV2 hap0: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/stitchv2_jax_10k5k_hap0`
- STITCHV2 hap1: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/stitchv2_jax_10k5k_hap1`
- STITCH hap0: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/stitch_r_10k5k_hap0`
- STITCH hap1: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/stitch_r_10k5k_hap1`

## Runtime and Memory

![Runtime breakdown](plots/runtime_breakdown.png)

![Memory breakdown](plots/memory_breakdown.png)

| Method | Runtime (s) | Runtime (min) | Peak RSS (MB) |
|---|---:|---:|---:|
| STITCHV2 hap0 | 731.36 | 12.19 | 3450.24 |
| STITCHV2 hap1 | 747.95 | 12.47 | 4379.39 |
| STITCH hap0 | 1032.31 | 17.21 | 1007.34 |
| STITCH hap1 | 2097.18 | 34.95 | 13665.26 |

STITCH hap1 status:
- `completed`
- Note: 

## QC Metrics

![Per-variant violins](plots/variant_metric_violins.png)

![ROC curves](plots/roc_curves.png)

| Method | R2 | F1 | Accuracy | Balanced accuracy | INFO mean | INFO median | Missing rate | ROC AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| STITCHV2 JAX hap0 | -0.3115 | 0.1811 | 0.3729 | 0.3333 | 0.2784 | 0.2784 | 0.0000 | 0.6735 |
| STITCHV2 JAX hap1 | -0.3115 | 0.1811 | 0.3729 | 0.3333 | 0.2784 | 0.2784 | 0.0000 | 0.6735 |
| STITCH hap0 | 0.1255 | 0.4505 | 0.7715 | 0.4580 | nan | nan | 0.9562 | 0.7009 |
| STITCH hap1 | 0.1284 | 0.4466 | 0.7739 | 0.4516 | nan | nan | 0.9561 | 0.7021 |

## Discussion
- STITCHV2 runtime scales closely to pilot projections and stays within practical runtime limits at 10k x 5k.
- Read IO is the dominant STITCHV2 stage; output writing increases noticeably when haplotype probabilities are enabled.
- STITCH haplotype-output mode is now directly included in runtime/memory and QC comparisons.
- STITCH no-hap is slightly slower than STITCHV2 hap0 in this setup; STITCH peak RSS is lower.
- INFO score is available directly for STITCHV2 from genotype posterior outputs. STITCH VCF output in this run did not include an equivalent posterior matrix for direct INFO parity.
- Real-data GATK holdout benchmark was not supplied to this report run.

## Ploidy Validation Addendum
- Added a dedicated synthetic ploidy benchmark runner:
  - `benchmarks/benchmark_ploidy_modes.py`
- It validates:
  - `--ploidy` including `0` (all-missing fast path)
  - sex-aware overrides via `--ploidy-males` and `--ploidy-females` using `samples.sex`
  - reported metrics: `dosage_r2`, `f1`, `info_mean`, `missing_rate`
- Output artifacts:
  - `ploidy_benchmark_summary.csv`
  - `ploidy_benchmark_summary.json`

## Real-Data Calling Calibration Addendum
- Added a dedicated official STITCH mouse-data calibration/calling sweep:
  - `benchmarks/benchmark_real_calling_calibration_sweep.py`
- Source benchmark:
  - `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01`
- Output report:
  - `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/calibration_calling_sweep/CALLING_CALIBRATION_SWEEP.md`
- Tested STITCHV2 JAX modes:
  - calibrated GP + always-call argmax
  - calibrated GP + STITCH-style no-call thresholds `0.5, 0.7, 0.8, 0.9, 0.95, 0.99`
  - raw GP + always-call argmax
  - raw GP + STITCH-style no-call thresholds `0.5, 0.7, 0.8, 0.9, 0.95, 0.99`
- Key finding:
  - STITCHV2 JAX calls all 2,002 truth-overlap genotypes as heterozygous in both calibrated and raw always-call modes.
  - STITCH-style threshold `0.9` makes all STITCHV2 calls missing, while STITCH itself keeps calls and calls 1,844/2,002 as homozygous alternate.
  - Calibration is therefore not the main failure source in this run; the failure is upstream in dosage/founder/HMM parity or allele-scale behavior.

## Masked-CV Calibration Addendum
- Added dependency-free masked-CV calibration:
  - `masked_cv_calibrate_genotype_posterior(...)`
  - CLI option: `--calibration-mode masked_cv`
- The optimizer tunes calibration by MAF bin:
  - temperature
  - raw/dosage posterior blend
  - dosage scale
  - dosage offset
  - optional HWE soft penalty
- Synthetic rerun:
  - Output: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_masked_cv_2026-05-04`
  - Report: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_masked_cv_2026-05-04/report/calibration_benchmark_report.md`
- Synthetic result:
  - pure argmax: R2 `0.9555`, F1 `0.9666`, accuracy `0.9716`, INFO `0.8989`
  - masked-CV: R2 `0.9570`, F1 `0.9690`, accuracy `0.9733`, INFO `0.9316`
  - tuned no-call remains best F1: `0.9859`, with call rate `0.9167`
- STITCH paper real-data held-out pseudo-truth result:
  - fixed STITCHV2 always-call: accuracy `0.0020`, F1 `0.0013`
  - masked-CV STITCHV2 always-call: accuracy `0.5944`, F1 `0.2501`
  - STITCH: accuracy `0.9201`, F1 `0.3278`
- Interpretation:
  - Masked-CV calibration partially fixes the heterozygote-collapse on the official mouse panel by learning stronger dosage scaling.
  - The remaining gap to STITCH indicates calibration alone is not enough; HMM/founder parity remains the next target.


## Artifacts
- Overall metrics: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/report/overall_metrics.parquet`
- Per-variant metrics: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/report/variant_metrics.parquet`
- ROC points: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/report/roc_points.parquet`
- Capacity summary reference: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/full_benchmark_2026-04-29_parity/capacity_and_benchmark_summary_10k5k_0p01x.json`
