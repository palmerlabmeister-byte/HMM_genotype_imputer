# STITCHV2 Synthetic Tutorial

This compact tutorial shows how to generate synthetic data and run STITCHV2 in mutable-founder and STITCH-parity fixed-founder modes. For the full documentation path, start with [README.md](../README.md), [docs/README.md](README.md), and the deep-dive files.

This tutorial explains:
- what input files STITCHv2 expects
- what key parameters do
- how to run STITCHv2 on synthetic data
- how to run two important modes:
  - mutable founders (`probabilistic` behavior)
  - STITCH-parity-like founders (`hard immutable` behavior)

All examples use the synthetic dataset path:

```text
benchmark_runs/synth_5mb_2k_0p1x
```

---

## 1) Synthetic dataset generation

Example:

```bash
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

What this creates:
- `samples.parquet`: sample metadata + BAM path + generation
- `positions.parquet`: `CHR`, `POS`, `REF`, `ALT`
- `truth_dosage.parquet`: synthetic truth dosage matrix
- `founders.truth.vcf`: founder haplotypes used by the simulator
- `bamlist.txt`, `pos.txt`, `sample_names.txt`: STITCH-compatible text inputs

---

## 2) Input files for `stitchv2 run`

Required:
- `--samples`: table with at least:
  - `sample_id` (string)
  - `bam_path` (string path to BAM/CRAM)
  - `generation` (float/int)
- `--positions`: table with:
  - `CHR`, `POS`, `REF`, `ALT`
- `--chromosome`: chromosome label to process (example: `chrSynthetic`)
- optional range slicing:
  - `--chr-start` (inclusive)
  - `--chr-end` (inclusive)
- `--output-dir`: run output directory
- `--n-founders`: number of founder states
- ploidy controls:
  - `--ploidy` (default ploidy, `>=0`; `0` means all outputs missing)
  - `--ploidy-males` and `--ploidy-females` (use `samples.sex` for per-sample ploidy, useful for chrX/chrY)
- optional per-sample microarray evidence:
  - `samples.parquet` can include `plink_path` per row
  - each path points to a PLINK prefix; multiple samples can share the same file

Optional founder inputs:
- `--founder-vcf`: initialize founders from VCF
- `--founder-plink`: initialize founders from PLINK
- `--founder-immutable`: freeze founder updates (hard founders)

---

## 3) Parameter guide (practical)

### Core model/inference
- `--n-founders`:
  - HMM founder-state count `K`.
  - Higher `K` can model more haplotype diversity but costs compute.
- `--em-iterations`:
  - Number of EM refinement loops.
  - Too low can underfit; too high increases runtime.
- `--block-size`:
  - Variants per processing block.
  - `0` lets STITCHV2 choose from `--max-mem`, which is the preferred production behavior.
- `--max-mem`:
  - Memory budget for automatic sample/SNP/IO planning.
  - Examples: `90%`, `64GB`, `50000MB`.

### Backend/performance
- `--hmm-backend {auto,numpy,jax,torch}`:
  - JAX is usually fastest in this repo.
- `--jax-sample-batch-size`:
  - Number of samples per JAX FB call.
  - `0` = all samples together.
  - Lower values reduce memory spikes at some speed cost.
- `--read-mode {read_stream,pileup}`:
  - `read_stream` is the primary high-throughput path.
- `--read-stream-backend {auto,python,htslib,snp_only_bamreader,stitch_style_bamreader,variant_aware_bamreader}`:
  - `auto` is recommended for production.
  - `stitch_style_bamreader` is useful for strict SNP-only STITCH compatibility checks.
  - `variant_aware_bamreader` is required for targeted insertion/deletion evidence.

### Read-awareness / fragment model
- `--fragment-likelihood-mode {replace,augment}`:
  - `replace` is the current default and parity path.
  - `augment` adds read-aware fragment likelihood to baseline emission and should be benchmarked separately because it can double-count read evidence.
- `--fragment-coupling-model {stitch_parity}`:
  - STITCH-aligned coupling behavior.
- `--fragment-max-diff-reads`, `--fragment-max-emission-diff`, `--no-fragment-rescale-read-likelihood`:
  - Controls read-likelihood clipping/rescaling stability.

### Calibration / calling
- `--no-calibrate-genotype-posteriors`:
  - disables posterior calibration
  - useful for strict parity-style runs
- `--genotype-call-mode {argmax,stitch_no_call,quality_gated}`:
  - `argmax`: always pick most likely genotype
  - `stitch_no_call`: STITCH-style thresholded no-call behavior
  - `quality_gated`: additional confidence/margin filters
- `--genotype-call-stitch-threshold`:
  - threshold used by `stitch_no_call`

### Founder behavior
- Mutable founders (default behavior):
  - no founder input + no `--founder-immutable`
  - founder probabilities update during EM
- STITCH-parity-like hard founders:
  - `--founder-vcf <indexed founder VCF>`
  - `--founder-immutable`
  - in synthetic data, this usually means `founders.truth.vcf.gz` + `.tbi`

### Parameter mapping to STITCH (with examples)

| STITCHv2 parameter | Example | Closest STITCH parameter | Notes |
|---|---|---|---|
| `--chromosome` | `chrSynthetic` | `chr` | Target chromosome/contig. |
| `--chr-start` | `100000` | no direct equivalent | Process a position window start (inclusive). |
| `--chr-end` | `200000` | no direct equivalent | Process a position window end (inclusive). |
| `--ploidy` | `2` | `method` (diploid/haploid closest concept) | Default sample ploidy (`0` => all missing). |
| `--ploidy-males` | `1` | no direct equivalent | Male ploidy override via `samples.sex`. |
| `--ploidy-females` | `2` | no direct equivalent | Female ploidy override via `samples.sex`. |
| `--n-founders` | `8` | `K` | Number of founder states. |
| `--em-iterations` | `5` | `niterations` | EM iteration count. |
| `--block-size` | `1000` | `outputBlockSize` (closest) | Same intent (chunking), different implementation details. |
| `--hmm-backend` | `jax` | no direct equivalent | STITCH backend is fixed C++/R implementation. |
| `--jax-sample-batch-size` | `128` | no direct equivalent | JAX-specific memory/perf control. |
| `--read-mode` | `read_stream` | `readAware=TRUE` (closest concept) | Both use read-aware evidence; mechanism differs. |
| `--read-stream-backend` | `auto` | no direct equivalent | STITCHV2 IO backend switch only. Use `stitch_style_bamreader` for strict SNP parity and `variant_aware_bamreader` for targeted indels. |
| `--io-workers` | `8` | `nCores` (closest) | Parallelism control differs by pipeline stage. |
| `--htslib-threads-per-file` | `2` | `nCores` (closest) | STITCH has no direct per-file HTS thread flag. |
| `--fragment-likelihood-mode` | `replace` | no direct equivalent | `replace` is the current parity/default mode; `augment` is experimental and should be benchmarked separately. |
| `--fragment-coupling-model` | `stitch_parity` | `readAware=TRUE` + STITCH internal read coupling | STITCH-aligned read coupling path. |
| `--fragment-max-diff-reads` | `100.0` | `maxDifferenceBetweenReads` | Same conceptual bound. |
| `--fragment-max-emission-diff` | `1000.0` | `maxEmissionMatrixDifference` | Same conceptual bound. |
| `--no-fragment-rescale-read-likelihood` | off (default rescaling on) | no direct equivalent | STITCHv2 stabilization toggle. |
| `--genotype-call-mode` | `stitch_no_call` | STITCH GP-thresholded no-call behavior | Closest hard-call behavior in STITCHv2. |
| `--genotype-call-stitch-threshold` | `0.9` | STITCH GP threshold (implicit behavior) | Typical threshold for no-call gating. |
| `--no-calibrate-genotype-posteriors` | enabled | no direct equivalent | Disable STITCHv2-only calibration stack for parity tests. |
| `--genotype-posterior-temperature` | `0.35` | no direct equivalent | STITCHv2 calibration parameter. |
| `--genotype-posterior-blend` | `0.35` | no direct equivalent | STITCHv2 calibration parameter. |
| `--founder-vcf` | `.../founders.truth.vcf.gz` | reference haplotype inputs (`reference_haplotype_file`, `reference_legend_file`, `reference_sample_file`) | Closest founder anchoring concept; wire format differs. |
| `--founder-immutable` | enabled | STITCH reference/haplotype-anchored behavior (closest) | Freezes founder updates in STITCHv2. |
| `samples.parquet: plink_path` | `/path/panel` | array evidence injected externally | Per-sample PLINK hard calls mixed with read evidence. |
| `--random-seed` | `7` | random seed controls in STITCH workflow (varies by script) | Use fixed seed for reproducibility. |
| `--write-genotype-posteriors` | enabled | VCF `GP` output behavior (closest) | Output format differs. |
| `--write-genotype-calls` | enabled | VCF `GT` output | Same outcome type. |
| `--write-haplotype-probabilities` | enabled | `output_haplotype_dosages` | Closest output-equivalent flag. |
| `--memory-map-read-matrices` | off | no direct equivalent | STITCHv2 memory-management feature. |
| `--memory-map-dir` | `/tmp/stitchv2_memmap` | no direct equivalent | STITCHv2 memory-management feature. |

---

## 4) Example runs on synthetic data

Install and validate the package from the repository root:

```bash
conda activate stitchv2-cpu-py313
pip install -e ".[plink,ml,plot,jobqueue,dev]"
python setup.py build_ext --inplace
stitchv2 --help
```

### A) Mutable founders (probabilistic)

```bash
stitchv2 run \
  --samples benchmark_runs/synth_5mb_2k_0p1x/samples.parquet \
  --positions benchmark_runs/synth_5mb_2k_0p1x/positions.parquet \
  --chromosome chrSynthetic \
  --output-dir benchmark_runs/tutorial_mutable_run \
  --n-founders 8 \
  --em-iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --jax-sample-batch-size 128 \
  --read-mode read_stream \
  --read-stream-backend auto \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### B) STITCH-parity-like hard immutable founders

First index founder VCF (once):

```bash
bgzip -f -c benchmark_runs/synth_5mb_2k_0p1x/founders.truth.vcf \
  > benchmark_runs/synth_5mb_2k_0p1x/founders.truth.vcf.gz
tabix -f -p vcf benchmark_runs/synth_5mb_2k_0p1x/founders.truth.vcf.gz
```

Run:

```bash
stitchv2 run \
  --samples benchmark_runs/synth_5mb_2k_0p1x/samples.parquet \
  --positions benchmark_runs/synth_5mb_2k_0p1x/positions.parquet \
  --chromosome chrSynthetic \
  --output-dir benchmark_runs/tutorial_hard_founder_run \
  --n-founders 8 \
  --em-iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --jax-sample-batch-size 128 \
  --read-mode read_stream \
  --read-stream-backend auto \
  --fragment-coupling-model stitch_parity \
  --founder-vcf benchmark_runs/synth_5mb_2k_0p1x/founders.truth.vcf.gz \
  --founder-immutable \
  --genotype-call-mode stitch_no_call \
  --no-calibrate-genotype-posteriors \
  --write-genotype-posteriors \
  --write-genotype-calls
```

---

## 5) Output files to inspect

Common outputs in `--output-dir`:
- `cli_run_summary.json`: top-level runtime + memory summary
- `samples.parquet`, `positions.parquet`, `founders.parquet`: normalized inputs
- `dosage/block=*.parquet`: dosage matrix blocks
- `genotype_posteriors/block=*.parquet`: posterior blocks (if enabled)
- `genotype_calls/block=*.parquet`: hard calls (if enabled)
- `stage_timings.json`: stage-level timing breakdown
- `memory_profile_summary.json`: memory profile summary

---

## 6) Recommended starting presets

- Fast dev sanity check:
  - `--em-iterations 1`
  - small sample subset
  - `--hmm-backend jax`
- Quality-focused synthetic parity test:
  - `--em-iterations 5`
  - `--fragment-coupling-model stitch_parity`
  - hard founder mode with founder truth VCF
  - `--genotype-call-mode stitch_no_call`
  - `--no-calibrate-genotype-posteriors`
