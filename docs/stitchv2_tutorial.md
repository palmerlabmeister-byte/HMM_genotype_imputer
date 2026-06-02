# STITCHV2 Tutorial

This is a practical, end-to-end guide to STITCHV2. It explains input formats, the main `stitchv2 run` parameters, original STITCH equivalents, BAM/PLINK/Parquet workflows, pedigrees, mixed ploidy, scaling, outputs, and conversion to PLINK-friendly formats.

The examples are written so you can run them from the repository root:

```bash
cd /path/to/STITCHV2
conda activate stitchv2-cpu-py313
pip install -e ".[plink,ml,plot,jobqueue,dev]"
python setup.py build_ext --inplace
stitchv2 --help
```

The code blocks are intended for interactive use from the repository root. Full benchmark commands are marked explicitly.

## 0. Mental Model: What STITCHV2 Does

STITCHV2 imputes genotypes from low-coverage sequencing reads, optional array genotypes, and founder haplotype states.

At a high level:

1. Read the sample table and position table.
2. Extract read evidence from BAM/CRAM files by variant block.
3. Optionally inject PLINK microarray hard calls as strong genotype evidence.
4. Run a read-aware HMM over founders.
5. Optionally update mutable founders by EM.
6. Calibrate genotype posteriors and emit dosage, genotype calls, posteriors, transitions, recombination rates, and founder updates.
7. Keep primary results in Parquet/Zarr; export BCF only when an external tool explicitly requires it.

Original STITCH does similar imputation, but its public workflow is centered on `bamlist.txt`, `posfile`, `K`, `nGen`, `niterations`, and VCF outputs. STITCHV2 uses explicit parquet tables and Python/JAX/Dask backends so large jobs can be inspected and chunked more directly.

```python
from pathlib import Path
import json
import numpy as np
import pandas as pd

REPO = Path.cwd()
DATA_DIR = REPO / 'benchmark_runs' / 'synth_5mb_2k_0p1x'
OUT_BASE = REPO / 'benchmark_runs' / 'tutorial_examples'
OUT_BASE.mkdir(parents=True, exist_ok=True)

print('Repository:', REPO)
print('Synthetic data exists:', DATA_DIR.exists())
```

## 1. Input Formats

STITCHV2 accepts tables in parquet or CSV/TSV-like text for samples and positions. BAM/CRAM paths, PLINK prefixes, founder VCF/PLINK files, and optional pedigree matrices are layered on top.

### 1.1 Sample Table: `samples.parquet` or CSV/TSV

Required logical columns:

| Column | Required? | Type | Meaning | STITCH equivalent |
|---|---:|---|---|---|
| `sample_id` | recommended | string | Individual ID. If missing, STITCHV2 creates `sample_0`, `sample_1`, ... | `sampleNames_file` |
| `bam_path` | filled if missing | string | Path to BAM/CRAM. Empty string means no reads for that sample. | `bamlist.txt` |
| `generation` | yes | float | Generations or recombination scaling parameter for transitions. | `nGen`, but STITCHV2 can vary by sample |
| `sex` | optional | string | Used with `--ploidy-males/--ploidy-females`; accepts values like `M`, `F`, `1`, `2`, `XY`, `XX`. | no direct equivalent |
| `plink_path` | optional | string | Per-sample PLINK prefix used to inject hard microarray calls. Multiple rows may point to the same prefix. | no direct direct equivalent; closest is external genotype/reference evidence |

A sample can have reads, PLINK array data, both, or neither. If both reads and array genotypes are present, STITCHV2 combines them by adding strong hard-call evidence from the array.

```python
# Example sample table with reads only
samples_reads = pd.DataFrame({
    'sample_id': ['animal_001', 'animal_002'],
    'bam_path': ['/data/bams/animal_001.bam', '/data/bams/animal_002.bam'],
    'generation': [10.0, 10.0],
})
samples_reads
```

```python
# Example mixed sample table: reads + per-sample PLINK microarray evidence + sex labels
samples_mixed = pd.DataFrame({
    'sample_id': ['animal_001', 'animal_002', 'array_only_003'],
    'bam_path': ['/data/bams/animal_001.bam', '/data/bams/animal_002.bam', ''],
    'generation': [10.0, 10.0, 10.0],
    'sex': ['M', 'F', 'F'],
    'plink_path': ['/data/arrays/batch_A', '/data/arrays/batch_A', '/data/arrays/batch_B'],
})
samples_mixed
```

### 1.2 Position Table: `positions.parquet` or CSV/TSV

Required columns:

| Column | Type | Meaning | STITCH equivalent |
|---|---|---|---|
| `CHR` | string | Chromosome/contig label. | `posfile` chromosome column or `chr` argument |
| `POS` | int | 1-based genomic coordinate. | `posfile` position |
| `REF` | string | Reference allele. | `posfile` ref allele |
| `ALT` | string | Alternate allele. | `posfile` alt allele |

Positions are filtered by `--chromosome`, optionally by `--chr-start` and `--chr-end`, then sorted by `POS`.

```python
positions_example = pd.DataFrame({
    'CHR': ['chr1', 'chr1', 'chr1'],
    'POS': [100123, 100456, 101000],
    'REF': ['A', 'C', 'G'],
    'ALT': ['G', 'T', 'A'],
})
positions_example
```

### 1.3 BAM/CRAM Inputs

BAM/CRAM files are referenced from `samples.bam_path`. They should be coordinate-sorted and indexed (`.bai` for BAM, `.crai` for CRAM). The contig names must match the `CHR` labels in `positions.parquet` and `--chromosome`.

Useful checks:

```bash
samtools quickcheck sample.bam
samtools index sample.bam
samtools idxstats sample.bam | head
```

STITCH equivalent: original STITCH uses a `bamlist.txt` file, one BAM path per line. STITCHV2 stores the same information in `samples.parquet` so sample metadata, sex, generation, and array paths travel together.

```python
# Inspect the synthetic data's sample table.
# This dataset was generated with tiny BAM files for tutorial/benchmark use.
if DATA_DIR.exists():
    display(pd.read_parquet(DATA_DIR / 'samples.parquet').head())
    display(pd.read_parquet(DATA_DIR / 'positions.parquet').head())
```

### 1.4 PLINK Inputs: Founder Panels and Microarray Evidence

STITCHV2 uses PLINK BED/BIM/FAM prefixes in two different ways:

1. **Founder initialization** with `--founder-plink /path/to/prefix`.
2. **Microarray hard-call evidence** with `--microarray-plink /path/to/prefix` or `samples.parquet: plink_path`.

PLINK prefix means these files exist:

```text
/path/to/prefix.bed
/path/to/prefix.bim
/path/to/prefix.fam
```

For per-sample `plink_path`, STITCHV2 loads each unique prefix once, extracts only the rows matching `sample_id`, and aligns variants by chromosome and position.

```python
# Example: inspect PLINK metadata lazily with STITCHV2's npplink helper.
# Change prefix to a real PLINK prefix before running.
RUN_PLINK_EXAMPLE = False
if RUN_PLINK_EXAMPLE:
    from stitchv2.npplink import read_fam_bim, load_plink_xarray
    prefix = Path('/data/arrays/batch_A')
    fam, bim = read_fam_bim(prefix)
    display(fam.head())
    display(bim.head())
    geno = load_plink_xarray(prefix, chunk_variants=10_000)
    print(geno)
```

### 1.5 Founder Inputs

STITCHV2 can use:

| Founder source | STITCHV2 input | STITCH equivalent | When to use |
|---|---|---|---|
| Uniform mutable founders | no founder file | STITCH estimates ancestral haplotypes internally | Exploratory runs, no reference founders |
| Founder VCF | `--founder-vcf founders.vcf.gz` | reference haplotype/legend/sample files, closest | Parity or known-founder synthetic experiments |
| Founder PLINK | `--founder-plink founders_prefix` | reference haplotypes, closest | When founder genotypes are already in BED/BIM/FAM |
| In-memory `FounderPanel` | Python API | no direct equivalent | Tests, custom pipelines, synthetic truth |

Use `--founder-immutable` to freeze founder states. This is the closest STITCH-parity mode when you have known hard haplotypes.

### 1.6 Optional Pedigree Input

STITCHV2 accepts pedigree information through the CLI (`--pedigree`) or Python API (`pipeline.prepare_inputs(samples, pedigree=...)`).

Supported forms:

- A pedigree table with offspring and parent columns.
- A `scipy.sparse.csr_matrix`.
- A `PedigreeGraph` from `stitchv2.pedigree`.

Sparse matrix convention:

```text
P[child, parent] = 0.5
```

For table input, common aliases are recognized. Examples include `sample_id`, `offspring`, `child`, `rfid`, `father_id`, `mother_id`, `sire`, and `dam`. You can override them with:

```bash
--pedigree-offspring-col rfid
--pedigree-parent1-col dam
--pedigree-parent2-col sire
```

Pedigree modes:

| Mode | Meaning |
|---|---|
| `smooth` | Existing dosage smoothing toward parent/relative means. |
| `kinship` | Long-distance kinship-weighted fallback from the sparse child-parent graph. |
| `transmission` | Iterative Mendelian parent-child message passing, with recombination-aware smoothing along the chromosome and kinship fallback. |

In `smooth` mode, STITCHV2 computes:

```text
smoothed_dosage = (1 - pedigree_strength) * HMM_dosage + pedigree_strength * parent_mean_dosage
```

In `transmission` mode, sequenced offspring can inform parents, parents can inform unsequenced offspring, and grandparents can propagate evidence through latent unsequenced parents. This is a post-STITCHV2 approximation using genotype posteriors; it is not a fully joint pedigree-STITCH factor graph.

STITCH equivalent: no direct command-line equivalent in standard STITCH. This is a STITCHV2 extension.

```python
# Example pedigree matrix: child sample 2 is smoothed toward samples 0 and 1.
from scipy import sparse

n_samples = 3
rows = [2, 2]       # child row
cols = [0, 1]       # parent columns
values = [1.0, 1.0]
pedigree = sparse.csr_matrix((values, (rows, cols)), shape=(n_samples, n_samples))
pedigree.toarray()
```

## 2. `stitchv2 run` Parameters, Classes, and Closest STITCH Equivalent

This section groups the most important `stitchv2 run` flags by the part of the pipeline they control. See [CLI_REFERENCE.md](CLI_REFERENCE.md), or run `stitchv2 run --help`, for the exhaustive CLI surface.

Use the `Production guidance` column as a practical triage:

- `Review every run`: this changes the biological question, the target data, or the main output contract.
- `Review for scale`: tune this for large chromosomes, many samples, Dask, cache reuse, or memory limits.
- `Parity-critical`: set deliberately when comparing to original STITCH.
- `Usually leave default`: the default is the recommended production setting unless you are debugging or benchmarking.
- `Advanced/debug`: use only when you are testing a specific behavior.

### 2.1 Inputs, Region, And Run Identity

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--samples` | Path to Parquet/CSV-like table | `samples.parquet` | `bamlist + sampleNames_file` | Sample metadata. Required logical field is `generation`; production tables should include stable `sample_id` and `bam_path`. | Review every run |
| `--positions` | Path to Parquet/CSV-like table | `positions.parquet` | `posfile` | Target variant table with `CHR`, `POS`, and preferably `REF`, `ALT`. STITCHV2 imputes only variants in this table. | Review every run |
| `--chromosome` | Contig/chromosome label | `chr12` | `chr` | Chromosome/contig to process. Must match the positions table and BAM/CRAM contig names. | Review every run |
| `--chr-start` | Integer coordinate | `9000000` | no direct equivalent | Inclusive coordinate window start. Use with `--chr-end` for smoke tests, shards, or benchmark chunks. | Review every run |
| `--chr-end` | Integer coordinate | `19000000` | no direct equivalent | Inclusive coordinate window end. If omitted, runs through the selected chromosome positions. | Review every run |
| `--output-dir` | Directory path | `runs/chr12` | `outputdir` | Run directory for Parquet/Zarr outputs, logs, diagnostics, and summaries. | Review every run |

### 2.2 Founders, Ploidy, And Biological Model

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--n-founders` | Positive integer | `8` | `K` | Total founder states. If founder input has fewer founders, extra mutable founders are appended. More founders can improve flexibility but increases HMM work. | Review every run |
| `--founder-vcf` | VCF/VCF.GZ path | `founders.truth.vcf.gz` | reference haplotype inputs | Initializes founder haplotypes from VCF. | Review every run |
| `--founder-plink` | PLINK prefix | `/data/founders/founders8` | reference haplotype inputs | Initializes founder haplotypes from PLINK BED/BIM/FAM. | Review every run |
| `--founder-immutable` | Boolean flag | `--founder-immutable` | fixed reference haplotypes | When present, loaded founders are frozen. Without it, founders may be updated by EM if mutable. | Parity-critical |
| `--founder-init-jitter` | Float | `0.01` | no direct equivalent | Adds small random perturbation to initialized mutable founders. | Advanced/debug |
| `--ploidy` | Integer `>=0` | `2` | `method`, closest concept | Default ploidy. `0` keeps samples in output with missing genotypes and zero haplotype probability; `1` is haploid; `2` is diploid; `>=3` uses the generic polyploid state path. | Review every run |
| `--ploidy-males` | Integer `>=0` | `1` | no direct equivalent | Male ploidy override from `samples.sex`. Use for chrX, chrY, or sex-limited chromosomes. | Review every run for sex chromosomes |
| `--ploidy-females` | Integer `>=0` | `2` | no direct equivalent | Female ploidy override from `samples.sex`. Female chrY is typically `0`; female chrX is typically `2` in diploids. | Review every run for sex chromosomes |
| `--microarray-plink` | PLINK prefix | `/data/arrays/all_samples` | genotype/reference input, closest | Adds hard genotype evidence from PLINK. Can add samples absent from the BAM table unless disabled. | Review every run if using arrays |
| `--microarray-generation-default` | Number | `10` | no direct equivalent | Generation value assigned to PLINK-only samples added from microarray data. | Review if using arrays |
| `--microarray-hard-call-weight` | Integer weight | `80` | no direct equivalent | Strength of injected microarray hard-call evidence. Higher means the array call dominates read evidence more strongly. | Review if using arrays |
| `--microarray-calibration-only` | Boolean flag | `--microarray-calibration-only` | no direct equivalent | Uses microarray data only for calibration truth, not as HMM evidence. | Review if using arrays |
| `--no-microarray-add-samples` | Boolean flag | `--no-microarray-add-samples` | no direct equivalent | When present, PLINK-only samples are not appended to the run. | Review if using arrays |

### 2.3 STITCH Parity, Recombination, And Transition Model

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--stitch-compat` | Boolean flag | `--stitch-compat` | many STITCH defaults | Applies STITCH-like settings: recombination rate 0.5 cM/Mb, BQ/MQ 17, BQ capped by MQ, insert size cap 600, ref/alt-only evidence, replace fragment mode, and STITCH-like likelihood caps. | Parity-critical |
| `--recombination-rate-cm-per-mb` | Float | `0.5` | `expRate`, closest concept | Fixed recombination rate when no genetic map or trained recombination scale is used. | Review every run |
| `--genetic-map` | Parquet/CSV path | `genetic_map.parquet` | genetic map/reference map, closest | Uses a table with `POS` and `CM` or `GENETIC_CM`; cM is interpolated onto target positions. | Review if available |
| `--trainable-global-recombination-rate` | Boolean flag | `--trainable-global-recombination-rate` | no direct equivalent | Estimates a global recombination-rate scale from an initial posterior pass. | Advanced/model benchmark |
| `--trainable_global_recombination_rate` | Boolean flag alias | `--trainable_global_recombination_rate` | no direct equivalent | Underscore alias for `--trainable-global-recombination-rate`. It controls the same global recombination-rate training behavior. | Prefer hyphenated spelling |
| `--allow-recombination-hotspot-window` | Size string or `0` | `5Mb` | no direct equivalent | Enables local hotspot/coldspot recombination scaling in windows. `0` disables local scaling. | Advanced/model benchmark |
| `--allow_recombination_hotspot_window` | Size string or `0` alias | `5Mb` | no direct equivalent | Underscore alias for `--allow-recombination-hotspot-window`. It controls the same local hotspot/coldspot recombination windows. | Prefer hyphenated spelling |
| `--transition-model` | `stitch_parity`, `factorized` | `stitch_parity` | transition model internals | `stitch_parity` uses the compact STITCH-like stay/switch model. `factorized` fits an experimental regularized low-rank founder-switch model while still running exact founder-state inference. | Usually leave default unless testing low-rank |
| `--transition-factor-rank` | Integer | `4` | no direct equivalent | Rank for `--transition-model factorized`. Ignored by `stitch_parity`. | Advanced/model benchmark |
| `--transition-factor-regularization` | Float | `10.0` | no direct equivalent | Pulls factorized transitions toward uniform STITCH parity. Larger values are more conservative. | Advanced/model benchmark |
| `--transition-factor-max-deviation` | Float | `3.0` | no direct equivalent | Caps how far factorized off-diagonal transition rates may move from parity before row normalization. | Advanced/model benchmark |
| `--no-transition-factor-train` | Boolean flag | `--no-transition-factor-train` | no direct equivalent | Keeps factorized transitions at the exact STITCH-parity initialization. | Advanced/debug |

### 2.4 HMM, EM, And JAX Compute

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--em-iterations` | Integer | `40` | `niterations` | Number of mutable-founder EM passes. Fixed immutable-founder parity can use one/few final inference passes; no-founder or extra-mutable-founder runs may need many more. | Review every run |
| `--em-convergence-tol` | Float | `1e-4` | no direct equivalent | Stops EM early when founder-probability delta is below this threshold. `0` disables. | Review for mutable founders |
| `--em-convergence-min-iterations` | Integer | `2` | no direct equivalent | Minimum EM passes before convergence checks can stop the loop. | Usually leave default |
| `--em-convergence-patience` | Integer | `1` | no direct equivalent | Number of consecutive converged EM checks before final posterior pass. | Usually leave default |
| `--em-founder-update-damping` | Float in `(0,1]` | `0.5` | no direct equivalent | Dampens mutable-founder updates. Lower values can stabilize difficult variable-founder runs. | Review for unstable EM |
| `--no-adaptive-em` | Boolean flag | `--no-adaptive-em` | no direct equivalent | Disables adaptive EM best-founder tracking. | Usually leave default |
| `--no-adaptive-em-restore-best-founders` | Boolean flag | `--no-adaptive-em-restore-best-founders` | no direct equivalent | Keeps the last founder panel instead of restoring the best tracked panel. | Usually leave default |
| `--em-multistarts` | Integer | `3` | repeated runs with seeds, closest | Runs multiple mutable-founder starts. Fixed-founder parity ignores this. | Advanced/model benchmark |
| `--no-founder-update-hardening` | Boolean flag | `--no-founder-update-hardening` | no direct equivalent | Leaves mutable founder updates probabilistic instead of hardening them discretely. | Advanced/debug |
| `--hmm-backend` | `auto`, `numpy`, `jax`, `torch` | `auto` | no direct equivalent | `auto` chooses the recommended backend; `jax` uses JAX kernels; `numpy` is a CPU fallback/debug path; `torch` is experimental if available. | Usually leave `auto` |
| `--jax-sample-batch-size` | Integer, `0` for all/planner | `128` | no direct equivalent | Samples per JAX forward/backward call. Lower values reduce peak memory and can increase overhead. | Review for scale |
| `--no-jax-bucket-batch-shapes` | Boolean flag | `--no-jax-bucket-batch-shapes` | no direct equivalent | Disables shape bucketing that reduces JAX recompilation for remainder batches. | Usually leave default |
| `--no-jax-count-emission-kernel` | Boolean flag | `--no-jax-count-emission-kernel` | no direct equivalent | Forces count-to-emission assembly out of the JAX fast path. | Advanced/debug |
| `--no-jax-fragment-emission-kernel` | Boolean flag | `--no-jax-fragment-emission-kernel` | no direct equivalent | Forces fragment emission assembly out of the JAX fast path. | Advanced/debug, usually do not set |
| `--jax-persistent-cache-dir` | Directory path | `/scratch/jax_cache` | no direct equivalent | Reuses JAX compilation cache across runs. Useful on clusters or repeated benchmarks. | Review for repeated large runs |

### 2.5 BAM/CRAM Reading And Fragment Evidence

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--read-mode` | `read_stream`, `pileup` | `read_stream` | `readAware=TRUE`, closest concept | `read_stream` is the production read-aware path. `pileup` is a simpler fallback/debug reader path. | Usually leave default |
| `--read-stream-backend` | `auto`, `python`, `htslib`, `snp_only_bamreader`, `stitch_style_bamreader`, `variant_aware_bamreader` | `auto` | STITCH C++/HTSlib reader, closest | `auto` selects the compiled production reader when available; `variant_aware_bamreader` supports targeted SNP/indel evidence; `stitch_style_bamreader` is useful for strict SNP parity; `snp_only_bamreader` is a benchmark/debug reader; `python` and `htslib` are fallbacks. | Review for parity or indels |
| `--min-base-quality` | Integer | `17` | `bqFilter` | Minimum base quality for read observations. | Parity-critical and quality-critical |
| `--min-mapping-quality` | Integer | `17` | mapping-quality filter | Minimum mapping quality for read observations. | Parity-critical and quality-critical |
| `--max-insert-size` | Integer, `0` disables | `600` | insert size cap | Skips fragments with absolute template length above this value. | Parity-critical |
| `--max-indel-len` | Integer | `50` | no direct equivalent | Maximum targeted insertion/deletion length accepted by the variant-aware reader. | Review if using indels |
| `--cap-base-quality-by-mapping-quality` | Boolean flag | `--cap-base-quality-by-mapping-quality` | STITCH quality behavior | When present, base quality is capped by mapping quality. | Parity-critical |
| `--ref-alt-only` | Boolean flag | `--ref-alt-only` | STITCH ref/alt filtering | When present, bases that are neither REF nor ALT are ignored instead of treated as OTHER evidence. | Parity-critical |
| `--no-merge-fragments-by-query` | Boolean flag | `--no-merge-fragments-by-query` | read-name merging behavior | Disables query-name fragment merging. Default keeps STITCH-style query grouping. | Usually leave default |
| `--merge-unpaired-fragments-by-query` | Boolean flag | `--merge-unpaired-fragments-by-query` | STITCH query-name grouping | Merges unpaired reads with the same query name. | Review for STITCH parity |
| `--use-bx-tag` | Boolean flag | `--use-bx-tag` | BX linked-read support | Enables linked-read fragment grouping by BX tag when present. | Review if BAMs have BX tags |
| `--bx-tag` | Two-character SAM tag | `BX` | BX tag | Selects the auxiliary tag used by `--use-bx-tag`. | Usually leave default |
| `--bx-tag-upper-limit` | Integer | `50000` | linked-read cap, closest | Maximum observations in one BX/query fragment before flushing. | Usually leave default |
| `--downsample-to-coverage` | Integer, `0` disables | `20` | `downsampleToCov` | Caps fragment-level per-sample/site coverage before HMM to avoid high-depth blowups. | Review for high-depth data |
| `--downsample-fraction` | Float in `[0,1]` | `0.5` | `downsampleFraction` | Randomly keeps this fraction of fragments before HMM. | Advanced/benchmark |
| `--io-workers` | Integer | `8` | `nCores`, closest | Native reader workers across samples. Too high can oversubscribe disks. | Review for scale |
| `--htslib-threads-per-file` | Integer | `2` | HTSlib threads, closest | Decompression threads per BAM/CRAM file. Combine carefully with `--io-workers`. | Review for scale |

### 2.6 Fragment Likelihood And STITCH Read Coupling

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--fragment-likelihood-mode` | `replace`, `augment` | `replace` | read-aware emission internals | `replace` uses fragment likelihood as the read-emission object and is the default/STITCH-parity path. `augment` adds fragment likelihood on top of dense counts and should be benchmarked carefully to avoid unintended double counting. | Usually leave `replace` |
| `--fragment-coupling-model` | `stitch_parity` | `stitch_parity` | `readAware=TRUE` internals | Controls how linked observations in a fragment are coupled. Current production option is STITCH-parity coupling. | Usually leave default |
| `--fragment-max-diff-reads` | Float | `100.0` | `maxDifferenceBetweenReads` | Clips extreme multi-read likelihood differences for numerical stability. | Usually leave default |
| `--fragment-max-emission-diff` | Float | `1000.0` | `maxEmissionMatrixDifference` | Clips extreme emission matrix differences. | Usually leave default |
| `--no-fragment-rescale-read-likelihood` | Boolean flag | `--no-fragment-rescale-read-likelihood` | no direct equivalent | Disables read-likelihood rescaling inside fragment handling. | Advanced/debug |

### 2.7 Memory, SNP Blocking, Cache, And Scratch

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--max-mem` | Percent or size string | `90%` | memory limits in scripts, closest | Hard planner budget for IO, HMM, calibration, diagnostics, cache, and writing. | Review every run |
| `--block-size` | Integer, `0` auto | `0` | `outputBlockSize`, closest | Approximate-mode SNP block size. In exact streaming, whole-region HMM math is preserved and memory is controlled by sample/chunk planning. | Review for scale |
| `--io-window-size` | Integer, `0` auto | `0` | no direct equivalent | SNPs per BAM extraction window. Auto keeps IO windows safe for cache/HMM chunking. | Review for scale |
| `--snp-block-mode` | `exact_streaming`, `independent_approx`, `density_balanced_overlap` | `exact_streaming` | no direct equivalent | `exact_streaming` is production and preserves whole-region HMM math. `independent_approx` treats blocks independently for quick QC. `density_balanced_overlap` is an approximate density-aware block mode with overlaps. | Usually leave `exact_streaming` for final runs |
| `--memory-safety-fraction` | Float | `0.6` | no direct equivalent | Fraction of `--max-mem` available to in-memory arrays after safety headroom. | Usually leave default |
| `--scratch-dir` | Directory path | `/scratch/stitchv2_chr12` | tmp directory, closest | Zarr/Parquet/Arrow scratch for streamed state and chunked outputs. | Review for large runs |
| `--approx-overlap-fraction` | Float | `0.05` | no direct equivalent | Overlap fraction for `density_balanced_overlap` approximate blocks. | Advanced/approx mode only |
| `--approx-min-overlap-snps` | Integer | `128` | no direct equivalent | Minimum SNP overlap per side for approximate density-balanced blocks. | Advanced/approx mode only |
| `--compact-evidence-cache-dir` | Directory path | `cache/chr12` | no direct equivalent | Cache directory for compact read evidence. Use this to avoid re-reading BAMs. | Review for repeated runs |
| `--compact-evidence-cache-mode` | `off`, `read`, `write`, `readwrite` | `readwrite` | no direct equivalent | `off` ignores cache; `write` creates cache; `read` uses existing cache; `readwrite` loads matching cached samples and reads/writes missing data. | Review for repeated runs |
| `--compact-evidence-cache-format` | `parquet_zarr` | `parquet_zarr` | no direct equivalent | Stores compact evidence/support in compressed Parquet and optional dense arrays in Zarr. | Usually leave default |
| `--compact-evidence-cache-sample-batch-size` | Integer | `256` | no direct equivalent | Sample batch size for cache partitions. | Review for scale |
| `--compact-evidence-no-dense-counts` | Boolean flag | `--compact-evidence-no-dense-counts` | no direct equivalent | Loads compact evidence without dense counts; replace-mode fragment likelihood can still drive the HMM. | Review for cache size |
| `--compact-evidence-cache-include-dense-counts` | Boolean flag | `--compact-evidence-cache-include-dense-counts` | no direct equivalent | Stores dense count/weight arrays too. Larger cache, useful for augment mode and some diagnostics. | Review before using `augment` |
| `--memory-map-read-matrices` | Boolean flag | `--memory-map-read-matrices` | no direct equivalent | Uses memory-mapped read matrices to reduce RAM pressure. | Advanced/debug now that cache is preferred |
| `--memory-map-dir` | Directory path | `/tmp/stitchv2_memmap` | tmp directory, closest | Directory for memory-mapped read matrices. | Advanced/debug |

### 2.8 Dask And Parallel Orchestration

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--executor` | `serial`, `dask` | `serial` | no direct equivalent | `serial` runs in one process; `dask` schedules coarse chunks and reads/writes scratch paths rather than passing large arrays through the scheduler. | Review for scale |
| `--dask-scheduler` | `local`, `threads`, `synchronous`, `jobqueue` | `local` | no direct equivalent | `local` uses distributed LocalCluster; `threads` avoids distributed scheduler; `synchronous` is debug; `jobqueue` uses dask-jobqueue. | Review for cluster runs |
| `--dask-n-workers` | Integer, `0` auto | `4` | `nCores`, closest | Number of local workers. More workers can oversubscribe RAM or disk IO. | Review for scale |
| `--dask-threads-per-worker` | Integer | `1` | `nCores`, closest | Threads per worker. For JAX/GPU or heavy native IO, one thread per worker is often easier to control. | Review for scale |
| `--dask-processes` | Boolean flag | `--dask-processes` | no direct equivalent | Uses worker processes instead of threads. This can isolate CPU work but increases serialization and memory pressure. | Usually leave unset locally |
| `--dask-memory-limit` | Size string | `16GB` | no direct equivalent | Per-worker memory limit. | Review for scale |
| `--dask-dashboard-address` | Address string or empty | `127.0.0.1:8787` | no direct equivalent | Dask dashboard bind address. Empty disables dashboard. | Review for monitoring |
| `--dask-performance-report` | HTML path | `dask_report.html` | no direct equivalent | Writes permanent Dask performance report. | Review for benchmarks |
| `--dask-task-stream` | HTML/JSON path | `task_stream.json` | no direct equivalent | Captures task stream diagnostics. | Review for benchmarks |
| `--dask-dashboard-hold-seconds` | Float | `60` | no direct equivalent | Keeps dashboard alive briefly after compute before shutdown. | Advanced/debug |
| `--dask-target-task-memory-mb` | Float, `0` disables | `4096` | no direct equivalent | Coarsely constrains task memory when planning chunks. | Review for scale |
| `--dask-min-block-size` | Integer | `128` | no direct equivalent | Lower bound for Dask-planned SNP blocks. | Usually leave default |
| `--dask-min-sample-batch-size` | Integer | `8` | no direct equivalent | Lower bound for Dask-planned sample batches. | Usually leave default |
| `--dask-sample-batch-size` | Integer, `0` auto | `256` | no direct equivalent | Samples per Dask HMM task. Distinct from JAX internal sample batching. | Review for scale |
| `--dask-jobqueue-class` | Scheduler class name | `SLURMCluster` | no direct equivalent | Dask-jobqueue backend class when `--dask-scheduler jobqueue` is used. | Review only on clusters |
| `--dask-jobqueue-queue` | Queue/partition name | `normal` | no direct equivalent | Queue or partition requested from the cluster scheduler. | Review only on clusters |
| `--dask-jobqueue-account` | Account/project name | `my_project` | no direct equivalent | Account string passed to the cluster scheduler. | Review only on clusters |
| `--dask-jobqueue-cores` | Integer | `4` | no direct equivalent | CPU cores requested per jobqueue worker. | Review only on clusters |
| `--dask-jobqueue-memory` | Size string | `32GB` | no direct equivalent | Memory requested per jobqueue worker. | Review only on clusters |
| `--dask-jobqueue-walltime` | Walltime string | `04:00:00` | no direct equivalent | Walltime requested per jobqueue worker. | Review only on clusters |

### 2.9 Output, Trace, And Diagnostics

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--write-genotype-calls` | Boolean flag | `--write-genotype-calls` | GT output | Writes hard genotype calls; `-1` means no-call. | Review every run |
| `--write-genotype-posteriors` | Boolean flag | `--write-genotype-posteriors` | GP output, closest | Writes genotype posterior probabilities. Larger than calls/dosage but useful for downstream uncertainty. | Review every run |
| `--write-haplotype-probabilities` | Boolean flag | `--write-haplotype-probabilities` | `output_haplotype_dosages` | Writes founder haplotype probabilities/dosages. Useful for haplotype analysis but can be large. | Review for storage |
| `--write-support-mask` | Boolean flag | `--write-support-mask` | no direct equivalent | Writes whether each sample/SNP had direct read or microarray support. | Review for QC |
| `--write-transitions` | Boolean flag | `--write-transitions` | no direct equivalent | Writes transition output according to `--transition-output`. | Review for transition analysis |
| `--transition-output` | `compact`, `factorized`, `full` | `compact` | no direct equivalent | `compact` writes stay/switch/offdiag summaries; `factorized` also writes low-rank factor metadata; `full` expands state matrices and is expensive. | Usually leave `compact` |
| `--store-xi` | `per-snp`, `full`, `False` | `per-snp` | no direct equivalent | `per-snp` writes compact transition/hotspot summaries; `full` writes posterior xi Zarr; `False` disables xi output. | Review for storage |
| `--write-gamma` | `summary`, `off`, `full` | `summary` | no direct equivalent | `summary` writes compact posterior diagnostics; `full` writes full gamma traces; `off` disables gamma output. | Usually leave `summary` |
| `--no-write-hmm-boundaries` | Boolean flag | `--no-write-hmm-boundaries` | no direct equivalent | Disables exact-streaming boundary alpha/beta metadata. | Usually leave default |
| `--no-write-transition-summary` | Boolean flag | `--no-write-transition-summary` | no direct equivalent | Disables compact per-interval transition/hotspot summary output. | Usually leave default |
| `--no-write-diagnostics` | Boolean flag | `--no-write-diagnostics` | no direct equivalent | Disables diagnostics files. | Usually leave default |
| `--diagnostics-fail-on-error` | Boolean flag | `--diagnostics-fail-on-error` | no direct equivalent | Converts diagnostic failures into run errors. | Review for production QC gates |
| `--diagnostics-warn-het-rate` | Float | `0.98` | no direct equivalent | Warning threshold for excessive heterozygosity. | Review for QC policy |
| `--diagnostics-fail-het-rate` | Float | `0.995` | no direct equivalent | Failure threshold for excessive heterozygosity. | Review for QC policy |
| `--diagnostics-warn-hom-rate` | Float | `0.995` | no direct equivalent | Warning threshold for excessive homozygosity. | Review for QC policy |
| `--diagnostics-fail-hom-rate` | Float | `0.999` | no direct equivalent | Failure threshold for excessive homozygosity. | Review for QC policy |
| `--diagnostics-warn-missing-rate` | Float | `0.50` | no direct equivalent | Warning threshold for missingness. | Review for QC policy |
| `--diagnostics-fail-missing-rate` | Float | `0.90` | no direct equivalent | Failure threshold for missingness. | Review for QC policy |
| `--diagnostics-warn-low-info` | Float | `0.05` | no direct equivalent | Warning threshold for low INFO. | Review for QC policy |
| `--diagnostics-fail-low-info` | Float | `-0.25` | no direct equivalent | Failure threshold for low INFO. | Review for QC policy |

### 2.10 Hard Calling And Calibration

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--no-calibrate-genotype-posteriors` | Boolean flag | `--no-calibrate-genotype-posteriors` | no direct equivalent | Disables calibration. Use for strict parity diagnostics or raw model comparisons. | Parity-critical |
| `--calibration-mode` | `standard_callability`, `fixed`, `masked_cv` | `standard_callability` | no direct equivalent | `standard_callability` keeps raw HMM GP and learns a hard-call P(correct) gate; `fixed` and `masked_cv` modify posteriors using older calibration strategies. | Usually leave default |
| `--genotype-call-mode` | `argmax`, `stitch_no_call`, `quality_gated` | `stitch_no_call` | STITCH GP thresholding, closest | `argmax` always calls; `stitch_no_call` requires max GP above the STITCH threshold; `quality_gated` also uses confidence, margin, and optional learned correctness. | Review every run |
| `--genotype-call-stitch-threshold` | Float | `0.9` | STITCH GP no-call threshold | Max-GP threshold used by `stitch_no_call`. | Parity-critical |
| `--genotype-call-min-confidence` | Float | `0.8` | no direct equivalent | Minimum top posterior probability for `quality_gated`. | Review for hard-call policy |
| `--genotype-call-min-margin` | Float | `0.2` | no direct equivalent | Minimum gap between top two genotype posterior probabilities for `quality_gated`. | Review for hard-call policy |
| `--genotype-call-correctness-threshold` | Float | `0.8` | no direct equivalent | Minimum predicted P(correct) for learned callability gates. | Review for calibrated hard calls |
| `--calibration-callability-model` | `lightgbm`, `sklearn_logistic` | `lightgbm` | no direct equivalent | `lightgbm` is the flexible default; `sklearn_logistic` is a lighter linear fallback. | Usually leave default |
| `--calibration-callability-decision-mode` | `per_snp_hierarchical`, `global` | `per_snp_hierarchical` | no direct equivalent | `per_snp_hierarchical` decides per SNP with local/global fallback; `global` applies one decision rule broadly. | Usually leave default |
| `--calibration-truth-source` | `read_evidence`, `microarray`, `auto`, `none` | `read_evidence` | no direct equivalent | `read_evidence` trains from high-confidence held-out read evidence; `microarray` uses external array labels; `auto` chooses an available source; `none` skips truth-trained calibration. | Review every calibration run |
| `--calibration-read-truth-holdout-fraction` | Float | `0.30` | no direct equivalent | Fraction of read fragments reserved for pseudo-truth labels; final HMM still uses all read evidence. | Review for calibration |
| `--calibration-read-truth-min-depth` | Integer | `3` | no direct equivalent | Minimum depth for read-backed pseudo-truth labels. | Review for calibration |
| `--calibration-read-truth-min-hom-depth` | Integer | `2` | no direct equivalent | Minimum depth for homozygous read-backed labels. | Review for calibration |
| `--calibration-read-truth-min-het-depth` | Integer | `4` | no direct equivalent | Minimum depth for heterozygous read-backed labels. | Review for calibration |
| `--calibration-read-truth-min-het-allele-depth` | Integer | `1` | no direct equivalent | Minimum per-allele depth for heterozygous read-backed labels. | Review for calibration |
| `--calibration-read-truth-hom-major-fraction` | Float | `0.95` | no direct equivalent | Major-allele fraction required for homozygous read-backed labels. | Review for calibration |
| `--calibration-read-truth-het-balance-min` | Float | `0.25` | no direct equivalent | Minimum allele balance for heterozygous read-backed labels. | Review for calibration |
| `--calibration-read-truth-het-balance-max` | Float | `0.75` | no direct equivalent | Maximum allele balance for heterozygous read-backed labels. | Review for calibration |
| `--calibration-read-truth-max-other-fraction` | Float | `0.05` | no direct equivalent | Maximum OTHER evidence fraction for read-backed labels. | Review for calibration |
| `--use-lightgbm-calibrator` | Boolean flag | `--use-lightgbm-calibrator` | no direct equivalent | Enables older block-context LightGBM posterior calibration path; default callability can already use LightGBM without this legacy stage. | Advanced/benchmark |
| `--calibration-context-window` | Integer | `25` | no direct equivalent | Variant context window for calibration features. | Usually leave default |
| `--calibration-block-snps` | Integer | `64` | no direct equivalent | SNP chunk size for calibration feature assembly. | Usually leave default |
| `--calibration-use-optuna` | Boolean flag | `--calibration-use-optuna` | no direct equivalent | Tunes calibration hyperparameters with Optuna. | Advanced/benchmark |
| `--calibration-optuna-trials` | Integer | `20` | no direct equivalent | Number of Optuna trials. | Advanced/benchmark |
| `--calibration-max-train-rows` | Integer | `750000` | no direct equivalent | Maximum rows used for calibration training. | Review for scale |
| `--genotype-posterior-temperature` | Float | `0.35` | no direct equivalent | Temperature for older/fixed posterior calibration. Smaller values sharpen GP. | Usually leave default |
| `--genotype-posterior-blend` | Float | `0.35` | no direct equivalent | Blend between raw and calibrated/fallback posterior in older calibration paths. | Usually leave default |

### 2.11 Advanced Calibration Guardrails And Fixed-Grid Calibration

These flags are included for completeness because they are part of `stitchv2 run`, but they are not the first knobs to tune. For production, inspect diagnostics first. Change these only when a calibration benchmark shows why the default guardrail is too strict or too permissive.

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--calibration-callability-min-train-rows` | Integer | `64` | no direct equivalent | Minimum training rows before a learned callability model is trusted. | Usually leave default |
| `--calibration-callability-min-call-rate` | Float | `0.80` | no direct equivalent | Minimum accepted call rate for callability thresholding. | Review only if calibration undercalls |
| `--calibration-callability-call-rate-weight` | Float | `0.03` | no direct equivalent | Small objective reward for keeping call rate high while maximizing balanced accuracy and macro F1. | Usually leave default |
| `--no-calibration-callability-bound-to-stitch` | Boolean flag | `--no-calibration-callability-bound-to-stitch` | no direct equivalent | Removes the STITCH no-call call-rate bound from learned callability. This can increase calls but can also destabilize hard calls. | Better untouched unless benchmarking |
| `--calibration-callability-min-call-rate-delta` | Float | `-0.02` | no direct equivalent | Minimum allowed call-rate shift relative to STITCH no-call for per-SNP calibration decisions. | Usually leave default |
| `--calibration-callability-max-call-rate-delta` | Float | `0.05` | no direct equivalent | Maximum allowed call-rate shift relative to STITCH no-call for per-SNP calibration decisions. | Usually leave default |
| `--calibration-callability-validation-site-fraction` | Float | `0.30` | no direct equivalent | Fraction of labeled sites reserved to validate per-SNP/local/global calibration fallback decisions. | Usually leave default |
| `--calibration-callability-min-objective-improvement` | Float | `0.0` | no direct equivalent | Minimum balanced-accuracy/F1/call-rate objective improvement before learned callability is used. | Review only if calibration is too conservative |
| `--calibration-callability-max-hardcall-maf-shift` | Float | `0.08` | no direct equivalent | Maximum allowed hard-call MAF shift after calibration. | Usually leave default |
| `--calibration-callability-max-hardcall-het-shift` | Float | `0.12` | no direct equivalent | Maximum allowed hard-call heterozygosity shift after calibration. | Usually leave default |
| `--calibration-train-site-fraction` | Float | `0.8` | no direct equivalent | Fraction of labeled SNPs used to fit learned calibration before applying to all SNPs. | Review for calibration scale |
| `--calibration-lightgbm-use-block-context` | Boolean flag | `--calibration-lightgbm-use-block-context` | no direct equivalent | Enables the older block-context LightGBM stage before read/site-aware calibration. | Advanced/benchmark |
| `--calibration-lightgbm-use-fixed-stage0` | Boolean flag | `--calibration-lightgbm-use-fixed-stage0` | no direct equivalent | Runs fixed temperature/blend calibration before LightGBM; default standard callability uses raw HMM GP. | Better untouched unless benchmarking |
| `--calibration-maf-bins` | Comma-separated numeric cutpoints | `0,0.01,0.05,0.5` | no direct equivalent | MAF bins for older fixed/masked calibration modes. | Advanced/fixed calibration only |
| `--calibration-temperatures` | Comma-separated floats | `0.15,0.35,1.0` | no direct equivalent | Temperature grid for older fixed/masked posterior calibration. Smaller values sharpen GP. | Advanced/fixed calibration only |
| `--calibration-blends` | Comma-separated floats | `0,0.5,1` | no direct equivalent | Blend grid for older fixed/masked calibration. `0` keeps raw posterior, `1` uses calibrated posterior fully. | Advanced/fixed calibration only |
| `--calibration-dosage-scales` | Comma-separated floats | `0.75,1,1.25` | no direct equivalent | Dosage-scale grid for older fixed/masked calibration. Values above `1` expand dosage away from the center. | Advanced/fixed calibration only |
| `--calibration-dosage-offsets` | Comma-separated floats | `-0.25,0,0.25` | no direct equivalent | Dosage-offset grid for older fixed/masked calibration. | Advanced/fixed calibration only |
| `--calibration-hwe-prior-weights` | Comma-separated floats | `0,0.25,1` | no direct equivalent | HWE prior weights for older fixed/masked calibration. | Advanced/fixed calibration only |
| `--no-calibration-optimize-dosage-scale` | Boolean flag | `--no-calibration-optimize-dosage-scale` | no direct equivalent | Disables dosage-scale search in older fixed/masked calibration. | Better untouched unless debugging |
| `--calibration-hwe-weight` | Float | `0.0` | no direct equivalent | Adds HWE weight to calibration objective. | Advanced/fixed calibration only |
| `--calibration-hwe-min-maf` | Float | `0.05` | no direct equivalent | Minimum MAF for HWE-weighted calibration diagnostics/objectives. | Advanced/fixed calibration only |
| `--no-calibration-sanity-checks` | Boolean flag | `--no-calibration-sanity-checks` | no direct equivalent | Disables calibration sanity checks. | Better untouched |
| `--calibration-max-mean-abs-dosage-shift` | Float | `0.35` | no direct equivalent | Maximum allowed mean absolute dosage shift after calibration. | Usually leave default |
| `--calibration-max-mean-abs-maf-shift` | Float | `0.20` | no direct equivalent | Maximum allowed mean absolute MAF shift after calibration. | Usually leave default |
| `--calibration-max-het-rate-shift` | Float | `0.25` | no direct equivalent | Maximum allowed heterozygosity-rate shift after calibration. | Usually leave default |
| `--calibration-max-mean-entropy-shift` | Float | `0.50` | no direct equivalent | Maximum allowed mean entropy shift after calibration. | Usually leave default |

### 2.12 Pedigree Adjustment

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--pedigree` | Parquet/CSV path | `pedigree_curated.parquet` | no direct equivalent | Pedigree table for post-HMM posterior adjustment. Prefer a first no-pedigree pass, pedigree QC, then a second pedigree-aware pass. | Review every pedigree run |
| `--pedigree-mode` | `off`, `smooth`, `kinship`, `transmission` | `transmission` | no direct equivalent | `off` disables adjustment; `smooth` blends relatives; `kinship` uses similarity fallback; `transmission` runs Mendelian-style parent-offspring message passing. | Review every pedigree run |
| `--pedigree-strength` | Float | `0.7` | no direct equivalent | Strength of pedigree adjustment. Too high can overrule the HMM. | Review every pedigree run |
| `--pedigree-offspring-col` | Column name | `sample_id` | no direct equivalent | Offspring/sample ID column in pedigree table. | Review every pedigree run |
| `--pedigree-parent1-col` | Column name | `father_id` | no direct equivalent | First parent column. | Review every pedigree run |
| `--pedigree-parent2-col` | Column name | `mother_id` | no direct equivalent | Second parent column. | Review every pedigree run |
| `--pedigree-iterations` | Integer | `4` | no direct equivalent | Number of pedigree message-passing iterations. | Usually leave default |
| `--pedigree-kinship-threshold` | Float | `0.01` | no direct equivalent | Sparse kinship fallback pruning threshold. | Usually leave default |

### 2.13 Compression And Miscellaneous

| STITCHV2 parameter | Allowed values/options | Example | Closest STITCH equivalent | What it controls | Production guidance |
|---|---|---|---|---|---|
| `--random-seed` | Integer | `7` | seed in scripts, closest | Controls stochastic subsampling, jitter, and randomized components where applicable. | Review every benchmark |
| `--compression` | Compression codec | `zstd` | no direct equivalent | Parquet compression codec. STITCHV2 primary outputs are Parquet/Zarr, not VCF. | Usually leave `zstd` |
| `--compression-level` | Integer | `6` | no direct equivalent | Compression level for applicable outputs. Higher may shrink files at extra CPU cost. | Usually leave default |
| `--no-profile-memory` | Boolean flag | `--no-profile-memory` | no direct equivalent | Disables per-stage memory profiling. | Leave profiling on for benchmarks |

## 3. Minimal Runs

The examples below are written as shell commands. In JupyterLab, prefix shell commands with `!` or use `%%bash` cells.

```python
%%bash
# Minimal read-aware diploid run on the synthetic data.
# Remove the leading "echo" when you are ready to execute.
echo stitchv2 run \
  --samples benchmark_runs/synth_5mb_2k_0p1x/samples.parquet \
  --positions benchmark_runs/synth_5mb_2k_0p1x/positions.parquet \
  --chromosome chrSynthetic \
  --output-dir benchmark_runs/tutorial_minimal \
  --n-founders 8 \
  --em-iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### 3.1 Python API Run

The Python API gives access to features that are awkward in a CLI, especially custom founder panels and pedigree matrices.

```python
from stitchv2 import PipelineConfig, StitchPipeline

RUN_PIPELINE_EXAMPLE = False
if RUN_PIPELINE_EXAMPLE and DATA_DIR.exists():
    samples = pd.read_parquet(DATA_DIR / 'samples.parquet').head(8).copy()
    cfg = PipelineConfig(
        chromosome='chrSynthetic',
        positions_path=DATA_DIR / 'positions.parquet',
        chromosome_start=None,
        chromosome_end=None,
        output_dir=OUT_BASE / 'python_api_run',
        n_founders=8,
        em_iterations=2,
        block_size=250,
        hmm_backend='jax',
        jax_sample_batch_size=4,
        read_mode='read_stream',
        read_stream_backend='auto',
        use_fragment_likelihood=True,
        fragment_likelihood_mode='replace',
        fragment_coupling_model='stitch_parity',
        write_genotype_posteriors=True,
        write_genotype_calls=True,
        write_support_mask=True,
        calibrate_genotype_posteriors=True,
    )
    pipeline = StitchPipeline(cfg)
    pipeline.prepare_inputs(samples)
```

## 4. Input Workflow Examples

### 4.1 STITCH-Compatible Text Inputs to STITCHV2 Tables

Synthetic datasets in this repo include STITCH-compatible files:

```text
bamlist.txt
sample_names.txt
pos.txt
```

STITCHV2 prefers `samples.parquet` and `positions.parquet`. This cell shows how to convert the STITCH text layout into STITCHV2 tables.

```python
def stitch_text_to_stitchv2_tables(data_dir: Path, generation: float = 10.0):
    bam_paths = pd.read_csv(data_dir / 'bamlist.txt', header=None, names=['bam_path'])
    sample_names_path = data_dir / 'sample_names.txt'
    if sample_names_path.exists():
        sample_ids = pd.read_csv(sample_names_path, header=None, names=['sample_id'])
    else:
        sample_ids = pd.DataFrame({'sample_id': [f'sample_{i}' for i in range(len(bam_paths))]})
    samples = pd.concat([sample_ids, bam_paths], axis=1)
    samples['generation'] = float(generation)
    positions = pd.read_csv(data_dir / 'pos.txt', sep=r'\s+', header=None, names=['CHR', 'POS', 'REF', 'ALT'])
    return samples, positions

if DATA_DIR.exists():
    samples_from_stitch, positions_from_stitch = stitch_text_to_stitchv2_tables(DATA_DIR)
    display(samples_from_stitch.head())
    display(positions_from_stitch.head())
```

### 4.2 Global Microarray PLINK Evidence

Use `--microarray-plink` when one PLINK BED/BIM/FAM prefix contains hard genotype calls for many samples. STITCHV2 aligns PLINK samples by IID to `sample_id`, aligns variants by chromosome/POS, and injects hard-call evidence.

If PLINK contains samples absent from `samples.parquet`, STITCHV2 adds them by default with empty `bam_path`; use `--no-microarray-add-samples` to disable this.

```python
%%bash
# Example only; replace /data/arrays/all_samples with a real PLINK prefix.
echo stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out_array_augmented \
  --n-founders 8 \
  --microarray-plink /data/arrays/all_samples \
  --microarray-generation-default 10 \
  --microarray-hard-call-weight 80 \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### 4.3 Per-Sample PLINK Evidence with `plink_path`

Use `samples.parquet: plink_path` when different samples come from different array batches, or several rows share the same PLINK file. STITCHV2 loads each unique prefix once and extracts only matching sample IDs.

```python
samples_per_sample_plink = pd.DataFrame({
    'sample_id': ['animal_001', 'animal_002', 'animal_003'],
    'bam_path': ['/data/bams/animal_001.bam', '', '/data/bams/animal_003.bam'],
    'generation': [10.0, 10.0, 10.0],
    'plink_path': ['/data/arrays/batch_A', '/data/arrays/batch_A', '/data/arrays/batch_B'],
})
samples_per_sample_plink
```

### 4.4 Founder VCF and Founder PLINK Examples

Founder VCF should be indexed if random access is needed by upstream tools. STITCHV2 founder loading uses the provided variant table for alignment.

```python
%%bash
# Founder VCF example.
echo stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out_founder_vcf \
  --n-founders 8 \
  --founder-vcf /data/founders/founders.vcf.gz \
  --founder-immutable \
  --hmm-backend jax

# Founder PLINK example.
echo stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out_founder_plink \
  --n-founders 8 \
  --founder-plink /data/founders/founder_panel \
  --founder-immutable \
  --hmm-backend jax
```

## 5. Ploidy, Sex Chromosomes, and Mixed Haploid/Diploid Runs

STITCHV2 supports any integer ploidy `>= 0`.

Important behavior:

- `--ploidy 0`: all genotypes become missing without HMM computation.
- `--ploidy 1`: haploid/pseudo-haploid path.
- `--ploidy 2`: diploid fast path.
- `--ploidy >= 3`: generic unordered founder-count state HMM.
- `--ploidy-males` and `--ploidy-females`: per-sample ploidy from `samples.sex`.

For mixed groups, STITCHV2 runs each positive ploidy group and merges outputs. Samples with ploidy 0 receive missing dosage/GP/GT.

```python
%%bash
# chrX-like: male haploid, female diploid.
echo stitchv2 run \
  --samples samples_with_sex.parquet \
  --positions chrX_positions.parquet \
  --chromosome chrX \
  --output-dir out_chrX \
  --n-founders 8 \
  --ploidy-males 1 \
  --ploidy-females 2 \
  --write-genotype-posteriors \
  --write-genotype-calls

# chrY-like: male haploid, female absent/missing.
echo stitchv2 run \
  --samples samples_with_sex.parquet \
  --positions chrY_positions.parquet \
  --chromosome chrY \
  --output-dir out_chrY \
  --n-founders 8 \
  --ploidy-males 1 \
  --ploidy-females 0 \
  --write-genotype-posteriors \
  --write-genotype-calls
```

## 6. Resource Scaling: Samples, SNPs, Founders, Ploidy, and Outputs

The main cost drivers are:

| Driver | Scaling intuition | Practical control |
|---|---|---|
| Samples `N` | Almost linear for evidence/output; HMM batches can reduce memory. | `--jax-sample-batch-size`, `--dask-sample-batch-size` |
| SNPs per block `M` | HMM memory grows roughly with block size. | `--block-size`, `--chr-start`, `--chr-end` |
| Founders `K` | Diploid states are roughly `K^2`; generic ploidy states are `comb(K + P - 1, P)`. | `--n-founders` |
| Ploidy `P` | P=1/2 use fast paths; P>=3 uses generic count states. | `--ploidy`, sex ploidy flags |
| Genotype posterior output | Adds `N * M * (P+1)` floats. | `--write-genotype-posteriors` |
| Haplotype output | Adds `N * M * K` floats. | `--write-haplotype-probabilities` |
| Full transitions | Very expensive for state x state matrices. | keep compact transitions unless needed |
| Reads/fragments | Depends on coverage/read length/fragment coupling. | `--read-mode`, memmap, IO workers |

Rule of thumb:

```text
emission/posterior memory ~ samples_per_task * variants_per_block * state_count * 4 bytes * scratch_multiplier
```

For diploid fast path, `state_count ≈ K*K`. For generic ploidy P, `state_count = comb(K + P - 1, P)`.

```python
from math import comb

def state_count(k: int, ploidy: int, fast_diploid: bool = True) -> int:
    if ploidy <= 0:
        return 1
    if ploidy == 1:
        return k
    if ploidy == 2 and fast_diploid:
        return k * k
    return comb(k + ploidy - 1, ploidy)

for p in [1, 2, 3, 4, 6]:
    print(f'K=8, P={p}: states={state_count(8, p, fast_diploid=(p == 2))}')
```

### 6.1 JAX vs Dask

Use serial JAX when one block/sample-batch fits memory and the machine is already saturated.

Use Dask when you want:

- live dashboard diagnostics,
- performance reports,
- coarse scheduling over blocks/sample batches/ploidy groups,
- memory-bounded chromosome-scale runs,
- multi-worker or future multi-GPU scaling.

Dask should schedule **coarse** tasks. Avoid tiny blocks; scheduler overhead can dominate.

```python
%%bash
# Dask-orchestrated JAX example with dashboard/report.
echo stitchv2 run \
  --samples benchmark_runs/synth_5mb_2k_0p1x/samples.parquet \
  --positions benchmark_runs/synth_5mb_2k_0p1x/positions.parquet \
  --chromosome chrSynthetic \
  --output-dir benchmark_runs/tutorial_dask \
  --n-founders 8 \
  --hmm-backend jax \
  --executor dask \
  --dask-n-workers 2 \
  --dask-threads-per-worker 1 \
  --dask-sample-batch-size 12 \
  --dask-dashboard-address 127.0.0.1:8786 \
  --dask-performance-report dask_report.html \
  --dask-task-stream dask_task_stream.json \
  --write-genotype-posteriors \
  --write-genotype-calls
```

## 7. Output Files and Schemas

STITCHV2 writes block-partitioned parquet datasets under `output_dir`. Each `block=000000.parquet` contains all samples for a contiguous variant block.

Typical layout:

```text
out/
  samples.parquet
  positions.parquet
  founders.parquet
  stage_timings.json
  memory_profile_summary.json
  run_summary.json
  dosage/block=000000.parquet
  genotype_posteriors/block=000000.parquet
  genotype_calls/block=000000.parquet
  support_mask/block=000000.parquet
  transitions/block=000000.parquet
  recombination/block=000000.parquet
  founder_updates/block=000000.parquet
  haplotype_probabilities/block=000000.parquet
```

### 7.1 Core Output Schema

| Dataset | Columns | Notes |
|---|---|---|
| `dosage/` | `sample_id`, `chromosome`, `position`, `dosage`, `block_id` | Dosage is alt allele count, 0..ploidy; NaN if missing. |
| `genotype_posteriors/` | `sample_id`, `chromosome`, `position`, `genotype_posterior`, `block_id` | Fixed-size list length `P+1` for alt allele count 0..P. Diploid length 3. |
| `genotype_calls/` | `sample_id`, `chromosome`, `position`, `genotype_call`, `block_id` | Integer alt allele count; `-1` means no-call. |
| `support_mask/` | `sample_id`, `chromosome`, `position`, `has_supporting_read`, `block_id` | Whether direct read/array evidence supports that sample-site. |
| `transitions/` | `sample_id`, `chromosome`, `position`, `switch_probability`, `stay_probability`, `offdiag_probability`, `block_id` | Compact transition summary. |
| `recombination/` | `chromosome`, `position`, `recombination_rate`, `block_id` | Recombination/rate proxy used for transitions. |
| `founder_updates/` | `chromosome`, `position`, `founder`, `alt_prob`, `block_id` | Updated founder alt probabilities. |
| `haplotype_probabilities/` | `sample_id`, `chromosome`, `position`, `hap_dosage`, `hap_probability`, `block_id` | Fixed-size list length K; optional. |
| `dask_run_summary.json` | JSON | Dask dashboard URL, chunk plan, task diagnostics, report paths. |
| `stage_timings.json` | JSON list | Per-block read/HMM/calibration/write timing and RSS fields. |

```python
# Inspect output schemas from an existing run if present.
example_run = REPO / 'benchmark_runs' / 'stitch_stitchv2_dask_2026-04-30' / 'stitchv2'
if example_run.exists():
    import pyarrow.parquet as pq
    for dataset in ['dosage', 'genotype_posteriors', 'genotype_calls', 'transitions', 'founder_updates']:
        files = sorted((example_run / dataset).glob('block=*.parquet'))
        if files:
            print('
', dataset, files[0].name)
            print(pq.read_schema(files[0]))
```

### 7.2 Combine Chunked Outputs

Use `stitchv2 combine` to create one parquet file per dataset and a lazy xarray view backed by Dask.

```python
%%bash
# Combine all standard output datasets.
echo stitchv2 combine \
  --run-output-dir benchmark_runs/tutorial_minimal \
  --output-dir benchmark_runs/tutorial_minimal/combined

# Combine one explicit parquet directory.
echo stitchv2 combine \
  --run-output-dir benchmark_runs/tutorial_minimal \
  --input-dir benchmark_runs/tutorial_minimal/dosage \
  --output-file benchmark_runs/tutorial_minimal/dosage.parquet
```

```python
# Python API for lazy combined outputs.
RUN_COMBINE_EXAMPLE = False
if RUN_COMBINE_EXAMPLE:
    from stitchv2.output import combine_pipeline_outputs
    xds = combine_pipeline_outputs(OUT_BASE / 'python_api_run')
    print(xds)
    print(xds.attrs['combine_summary'].keys())
```

## 8. Native Parquet/Zarr Output and Explicit BCF Interoperability

STITCHV2 native outputs are parquet because parquet is efficient for block-wise writing and analytics. Hard calls and categorical columns remain in Parquet with dictionary/RLE/bit-packing-friendly encodings. Floating outputs such as dosage, GP, transitions, recombination rates, and founder probabilities can also be stored in xarray/Zarr:

```bash
stitchv2 combine \
  --run-output-dir benchmark_runs/tutorial_minimal \
  --output-dir benchmark_runs/tutorial_minimal/combined \
  --write-zarr
```

VCF export is not allowed in STITCHV2. BCF export exists only for interoperability and WILL slow down I/O compared with Parquet/Zarr.

```python
%%bash
echo stitchv2 export-bcf \
  --run-output-dir benchmark_runs/tutorial_minimal \
  --output-bcf benchmark_runs/tutorial_minimal/stitchv2.chrSynthetic.bcf \
  --chromosome chrSynthetic
```

For BCF-only downstream tools that require VCF, convert outside STITCHV2 as a final interoperability step:

```bash
bcftools view benchmark_runs/tutorial_minimal/stitchv2.chrSynthetic.bcf -Oz   -o benchmark_runs/tutorial_minimal/stitchv2.chrSynthetic.from_bcf.vcf.gz
bcftools index -t benchmark_runs/tutorial_minimal/stitchv2.chrSynthetic.from_bcf.vcf.gz
```

### 8.1 Export a Simple Dosage Matrix for Non-PLINK Tools

Sometimes a GWAS pipeline wants a sample-by-SNP dosage matrix. You can pivot the parquet dosage output directly.

```python
RUN_DOSAGE_MATRIX_EXAMPLE = False
if RUN_DOSAGE_MATRIX_EXAMPLE:
    import pyarrow.dataset as ds
    run_dir = OUT_BASE / 'python_api_run'
    dosage_df = ds.dataset(str(run_dir / 'dosage'), format='parquet').to_table().to_pandas()
    dosage_matrix = dosage_df.pivot(index='sample_id', columns='position', values='dosage')
    dosage_matrix.to_parquet(run_dir / 'dosage_matrix_samples_by_position.parquet')
    dosage_matrix.iloc[:5, :5]
```

## 9. Cross-Validation and Tuning

`stitchv2 cv` runs fold-based tuning over K/nGen/S-style settings and optional LightGBM post-calibration.

Additional CV-only parameters:

| Parameter | Example | STITCH equivalent | Meaning |
|---|---|---|---|
| `--pseudo-truth` | `pseudo_truth.parquet` | external truth/evaluation file | Table with sample_id, position, genotype or dosage truth. |
| `--k-values` | `6,8,10` | K grid | Founder-state grid. |
| `--ngen-values` | `0.75,1.0,1.25` | nGen grid | Generation/recombination scaling grid. |
| `--s-values` | `2,3` | STITCH smoothing/internal setting, closest | Additional tuning dimension used by harness. |
| `--seeds` | `0,1,2` | seed averaging in scripts | Repeat grid across seeds. |
| `--folds` | `5` | no direct equivalent | Number of CV folds. |
| `--holdout-fraction` | `0.2` | no direct equivalent | Fraction held out per fold. |
| `--lightgbm-post-calibrator` | flag | no direct equivalent | Extra post-calibration model. |

```python
%%bash
# CV/tuning example.
echo stitchv2 cv \
  --samples samples.parquet \
  --positions positions.parquet \
  --pseudo-truth pseudo_truth.parquet \
  --chromosome chr1 \
  --output-dir out_cv \
  --k-values 6,8,10 \
  --ngen-values 0.75,1.0,1.25 \
  --s-values 2,3 \
  --seeds 0,1,2 \
  --folds 5 \
  --holdout-fraction 0.2 \
  --hmm-backend jax \
  --fragment-coupling-model stitch_parity \
  --lightgbm-post-calibrator
```

## 10. Benchmarking and Diagnostics

Useful benchmark/report scripts in this repo:

| Script | Use |
|---|---|
| `benchmarks/synthetic_dataset.py` | Generate synthetic reads, truth dosage, founders, STITCH-compatible text inputs. |
| `benchmarks/benchmark_compare.py` | Compare STITCHV2 to original STITCH. |
| `benchmarks/benchmark_stitch_stitchv2_dask_report.py` | Three-way STITCH/STITCHV2/STITCHV2-Dask report with plots. |
| `benchmarks/benchmark_ploidy_modes.py` | Ploidy and sex chromosome scenarios. |
| `benchmarks/benchmark_jax_generic_ploidy_parity.py` | Generic ploidy JAX HMM correctness/runtime. |
| `benchmarks/check_memory_leak.py` | Repeated run memory-growth check. |
| `stitchv2 tune-jax-memory` | Block-size, memmap, and JAX sample-batch tuning. |

```python
%%bash
# Three-way benchmark with plots and Dask performance report.
echo python benchmarks/benchmark_stitch_stitchv2_dask_report.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/stitch_stitchv2_dask_tutorial \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 5 \
  --block-size 1000 \
  --dask-dashboard-address 127.0.0.1:8786 \
  --force
```

## 11. Practical Recipes

### Recipe A: Fast diagnostic run on a chromosome slice

Use `--chr-start/--chr-end`, small `--block-size`, and a subset sample table.

### Recipe B: STITCH-parity diagnostic

Use hard immutable founders, `fragment_coupling_model=stitch_parity`, enough EM iterations, and consider `--no-calibrate-genotype-posteriors` if you want raw model comparison.

### Recipe C: Production low-coverage run

Use `hmm_backend=jax`, `read_stream_backend=auto` or `htslib`, calibrated posteriors, genotype calls, support mask, and tune `block_size`/`jax_sample_batch_size`.

### Recipe D: Chromosome-scale memory-controlled run

Use Dask executor with coarse chunks and performance report:

```bash
stitchv2 run ...   --executor dask   --dask-n-workers 4   --dask-threads-per-worker 1   --dask-target-task-memory-mb 4096   --dask-dashboard-address 127.0.0.1:8786   --dask-performance-report dask_report.html
```

## 12. Troubleshooting Checklist

| Symptom | Likely cause | What to check |
|---|---|---|
| No reads overlap variants | contig mismatch or wrong coordinates | `samtools idxstats`, position `CHR`, `--chromosome` |
| Dask slower than serial | chunks too small or dataset too small | increase block/sample batch; use Dask for larger runs |
| JAX OOM | too many samples/SNPs/states per task | lower `--block-size` or `--jax-sample-batch-size`; enable Dask planner |
| STITCHV2 differs from STITCH | founder representation, calibration, no-call policy, read coupling | use hard immutable founders, stitch parity coupling, comparable call mode |
| Sex chromosome wrong missingness | missing/unclear `samples.sex` | use M/F, male/female, 1/2, XY/XX labels |
| PLINK sample not injected | IID does not match `sample_id` | inspect `.fam` IID column and samples table |
| PLINK variant not injected | BIM chromosome/POS mismatch | inspect `.bim`; align `CHR`, `POS`, REF/ALT conventions |
| VCF/PLINK conversion loses dosage | PLINK1 BED is hard-call only | use PLINK2 PGEN with `dosage=DS` |

## 13. Quick Reference: Original STITCH vs STITCHV2 Command Shape

Original STITCH call shape in R:

```r
STITCH::STITCH(
  tempdir = 'tmp',
  chr = 'chr1',
  bamlist = 'bamlist.txt',
  sampleNames_file = 'sample_names.txt',
  posfile = 'pos.txt',
  outputdir = 'stitch_out/',
  K = 8,
  nGen = 10,
  niterations = 5,
  nCores = 4,
  method = 'diploid',
  output_haplotype_dosages = FALSE
)
```

Equivalent STITCHV2 shape:

```bash
stitchv2 run   --samples samples.parquet   --positions positions.parquet   --chromosome chr1   --output-dir stitchv2_out   --n-founders 8   --em-iterations 5   --hmm-backend jax   --io-workers 4   --ploidy 2   --fragment-coupling-model stitch_parity   --write-genotype-posteriors   --write-genotype-calls
```
