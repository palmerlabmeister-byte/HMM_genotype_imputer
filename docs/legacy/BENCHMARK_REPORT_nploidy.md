# N-Ploidy JAX HMM Benchmark Report

Status: historical ploidy benchmark report retained for context. Do not use this as the current benchmark protocol; use [final_benchmark.md](../../final_benchmark.md) for the stable benchmark protocol.

## Scope
- Synthetic benchmark using the existing STITCHV2 model-generated truth dataset.
- Compares optimized JAX `P=1`/`P=2` fast paths with current production dispatch.
- Forces the new generic unordered count-state JAX HMM for `P=1` and `P=2` to verify numerical parity.
- Exercises higher ploidy through generic JAX dispatch for `P=3` and `P=4`.
- Reports QC metrics used elsewhere in the benchmark suite: R2, F1, accuracy, balanced accuracy, INFO, and missing rate.

## Inputs
- Benchmark dir: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/jax_generic_ploidy_parity`
- Synthetic data dir: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/synth_5mb_2k_0p1x`
- Samples: `8`
- Positions: `48`
- Founders K: `8`
- EM iterations: `1`
- Synthetic count depth: `6.0`

## Runtime

![Runtime](benchmark_runs/jax_generic_ploidy_parity/report/plots/nploidy_runtime.png)

| Method | Ploidy | Path | States | First run (s) | Warm run (s) |
|---|---:|---:|---:|---:|---:|
| p1_fast | 1 | fast | 8 | 0.0069 | 0.0068 |
| p1_dispatch | 1 | dispatch | 8 | 0.0069 | 0.0068 |
| p1_generic_forced | 1 | generic_forced | 8 | 0.2578 | 0.0102 |
| p2_fast | 2 | fast | 64 | 0.4839 | 0.0032 |
| p2_dispatch | 2 | dispatch | 64 | 0.0059 | 0.0031 |
| p2_generic_forced | 2 | generic_forced | 36 | 0.2652 | 0.0189 |
| p3_dispatch | 3 | dispatch | 120 | 0.4508 | 0.0391 |
| p4_dispatch | 4 | dispatch | 330 | 2.4181 | 0.1442 |

## QC Metrics

![QC metrics](benchmark_runs/jax_generic_ploidy_parity/report/plots/nploidy_qc_metrics.png)

| Method | R2 | Corr R2 | F1 | Accuracy | Balanced accuracy | INFO mean | INFO median | Missing rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p1_fast | 0.2909 | 0.5747 | 0.7480 | 0.8177 | 0.8933 | 0.8732 | 0.9656 | 0.0000 |
| p1_dispatch | 0.2909 | 0.5747 | 0.7480 | 0.8177 | 0.8933 | 0.8732 | 0.9656 | 0.0000 |
| p1_generic_forced | 0.2909 | 0.5747 | 0.7480 | 0.8177 | 0.8933 | 0.8732 | 0.9656 | 0.0000 |
| p2_fast | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| p2_dispatch | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| p2_generic_forced | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| p3_dispatch | 0.9282 | 0.9349 | 0.6805 | 0.8255 | 0.8550 | 0.9660 | 0.9633 | 0.0000 |
| p4_dispatch | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9989 | 0.9994 | 0.0000 |

## Parity Checks

![Parity](benchmark_runs/jax_generic_ploidy_parity/report/plots/nploidy_parity_diffs.png)

| Comparison | Ploidy | Max dosage diff | Max GP diff | Dosage close | GP close | Dispatch/fast runtime | Forced generic/fast runtime | Forced generic AOT cache |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| p1_dispatch_parity | 1 | 0.000000 | 0.000000 | True | True | 1.000760 | NA | NA |
| p1_forced_generic_correctness | 1 | 0.000000 | 0.000001 | True | True | NA | 1.509329 | 1.000000 |
| p2_dispatch_parity | 2 | 0.000000 | 0.000000 | True | True | 0.964750 | NA | NA |
| p2_forced_generic_correctness | 2 | 0.000001 | 0.000002 | True | True | NA | 5.879077 | 1.000000 |

## JAX AOT/Warm Compilation

![AOT cache](benchmark_runs/jax_generic_ploidy_parity/report/plots/nploidy_aot_cache.png)

| Method | AOT compile | Precompile | Fast cache | Generic cache | Precompiled shapes |
|---|---:|---:|---:|---:|---|
| p1_fast | True | True | 0 | 0 | `` |
| p1_dispatch | True | True | 0 | 0 | `` |
| p1_generic_forced | True | True | 0 | 1 | `(8, 48, 8, 'polyploid1', 'jax')` |
| p2_fast | True | True | 1 | 0 | `(8, 48, 8, 'diploid', 'jax')` |
| p2_dispatch | True | True | 1 | 0 | `(8, 48, 8, 'diploid', 'jax')` |
| p2_generic_forced | True | True | 0 | 1 | `(8, 48, 8, 'polyploid2', 'jax')` |
| p3_dispatch | True | True | 0 | 1 | `(8, 48, 8, 'polyploid3', 'jax')` |
| p4_dispatch | True | True | 0 | 1 | `(8, 48, 8, 'polyploid4', 'jax')` |

## Discussion
- Production dispatch preserves the optimized JAX paths for `P=1` and `P=2`; those outputs are exactly matched by the explicit `fast` comparison rows.
- Forced generic `P=1` and `P=2` are numerically equivalent to the old fast paths within floating-point tolerance, validating the unordered count-state implementation.
- Forced generic `P=2` is slower than the specialized diploid JAX kernel on this small benchmark, so keeping the fast path is the right production choice.
- `P=3` and `P=4` run through the generic JAX count-state kernel and populate generic AOT cache entries with `polyploid*` precompiled shapes.
- This report uses aggregate benchmark rows rather than per-variant distributions; the plots therefore use grouped bar comparisons instead of the full report's per-variant violin plots.

## Artifacts
- Summary CSV: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/jax_generic_ploidy_parity/jax_generic_ploidy_parity_summary.csv`
- Summary JSON: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/jax_generic_ploidy_parity/jax_generic_ploidy_parity_summary.json`
- Overall metrics parquet: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/jax_generic_ploidy_parity/report/nploidy_overall_metrics.parquet`
- Parity metrics parquet: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/jax_generic_ploidy_parity/report/nploidy_parity_metrics.parquet`
- Plot directory: `/home/bonnie/Documents/codex/STITCHV2/benchmark_runs/jax_generic_ploidy_parity/report/plots`
