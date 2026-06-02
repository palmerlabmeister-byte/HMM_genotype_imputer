# STITCHV2 Tutorial

Status: retained as a legacy tutorial. The current publication-facing docs are [README.md](../../README.md), [docs/README.md](../README.md), and the files under [docs/deep_dive](../deep_dive/README.md). Keep this file for historical context until the tutorial set is fully consolidated.

This tutorial is source-backed from the current STITCHV2 package and benchmark code. It was written from:

- `src/stitchv2/cli.py`
- `src/stitchv2/config.py`
- `src/stitchv2/pipeline.py`
- `src/stitchv2/io.py`
- `src/stitchv2/founders.py`
- `src/stitchv2/pileup.py`
- `src/stitchv2/output.py`
- `benchmarks/synthetic_dataset.py`
- `benchmarks/benchmark_compare.py`
- `benchmarks/benchmark_stitch_stitchv2_dask_report.py`
- `benchmarks/benchmark_ploidy_modes.py`
- `benchmarks/benchmark_jax_generic_ploidy_parity.py`
- `benchmarks/benchmark_pedigree_synthetic.py`
- `benchmarks/benchmark_real_subsample_gatk_compare.py`
- `benchmarks/benchmark_real_calling_calibration_sweep.py`
- `benchmarks/generate_full_benchmark_report.py`

It explains how to run STITCHV2 from the CLI and interactively from Python, how to run it in STITCH-parity mode, what exact file formats it reads and writes, and which benchmark scripts validate the main behaviors. It is documentation only; do not record benchmark results here.

## 1. What STITCHV2 Does

`stitchv2 run` is the package's main imputation command. Internally it:

1. reads a samples table and a positions table
2. optionally aligns PLINK microarray hard calls
3. resolves sample ploidy, including sex-specific ploidy
4. loads or constructs a founder panel
5. extracts BAM/CRAM read evidence block by block
6. runs the HMM with `numpy`, `jax`, `torch`, or `auto` backend selection
7. optionally calibrates genotype posteriors and applies pedigree adjustment
8. writes parquet outputs plus timing and memory summaries

The CLI has one important behavior difference from the Python API:

- CLI runs without `--founder-vcf` or `--founder-plink` create uniform mutable founders with ALT probability `0.5`.
- Python API runs without an explicit `founder_panel` require `PipelineConfig.founder.source_path`.

## 2. Environment

The package declares Python `>=3.10` in `pyproject.toml`. Core dependencies are `numpy`, `pandas`, `pyarrow`, `pysam`, `jax`, `jaxlib`, `scipy`, `xarray`, `zarr`, `dask[array]`, and `distributed`.

Create the tutorial environment from the repo root:

```bash
conda env create -f environment.tutorial.yml
conda activate stitchv2-tutorial
```

If you are using an existing environment, install from the repo root:

```bash
python -m pip install -e .
```

Optional features:

```bash
python -m pip install -e '.[ml]'
python -m pip install -e '.[plink]'
python -m pip install -e '.[dev]'
```

Check that the CLI resolves:

```bash
stitchv2 run --help
stitchv2 cv --help
stitchv2 combine --help
stitchv2 export-bcf --help
```

## 3. Exact Input Formats

### Samples Table

`--samples` accepts parquet or a delimited text file readable by `pandas.read_csv(..., sep=None, engine="python")`.

Only `generation` is strictly required by `validate_samples()`. If absent, STITCHV2 creates:

- `sample_id`: `sample_0`, `sample_1`, ...
- `bam_path`: empty string

For real runs, provide all three:

| Column | Required | Type | Meaning |
| --- | --- | --- | --- |
| `generation` | yes | float | generations since founder mosaic start |
| `sample_id` | recommended | string | sample identifier used in all outputs |
| `bam_path` | recommended | string | BAM/CRAM path; empty string means no read evidence |
| `sex` | optional | string | used with sex-specific ploidy |
| `plink_path` | optional | string | per-sample PLINK prefix for hard-call evidence |
| `father_id`, `mother_id` | optional | string | default embedded pedigree parent columns |
| `read_subsample_prob` | optional | float | per-sample read keep probability used by the read extractor |
| `read_subsample_seed` | optional | int | per-sample read subsampling seed |

Accepted sex labels are:

| Sex | Accepted labels |
| --- | --- |
| male | `M`, `male`, `1`, `XY` |
| female | `F`, `female`, `2`, `XX` |

Example CSV:

```text
sample_id,bam_path,generation,sex,father_id,mother_id
s1,/data/bams/s1.bam,100,F,,
s2,/data/bams/s2.bam,100,M,,
child1,/data/bams/child1.bam,101,F,s2,s1
```

### Positions Table

`--positions` accepts parquet or delimited text. Column names are uppercased by the loader.

| Column | Required | Type | Meaning |
| --- | --- | --- | --- |
| `CHR` | yes | string | chromosome or contig |
| `POS` | yes | integer | 1-based position |
| `REF` | no | string | reference allele; defaults to `N` if absent |
| `ALT` | no | string | alternate allele; defaults to `N` if absent |

Rows are filtered to `--chromosome`, sorted by `POS`, then optionally filtered by `--chr-start` and `--chr-end`.

Example:

```text
CHR POS REF ALT
chr19 3000001 A G
chr19 3000104 C T
```

### Founder Inputs

STITCHV2 supports founder panels from VCF, PLINK, or an in-memory `FounderPanel`.

| Source | CLI | Python | Notes |
| --- | --- | --- | --- |
| uniform mutable founders | omit founder flags | pass a custom `FounderPanel` | CLI only; ALT probability `0.5`, mutable |
| founder VCF | `--founder-vcf founders.vcf.gz` | `FounderConfig(source_format="vcf", source_path=...)` | reads `DS` if present, otherwise `GT` |
| founder PLINK | `--founder-plink founder_prefix` | `FounderConfig(source_format="plink", source_path=...)` | requires PLINK support |
| in-memory | not exposed | `pipeline.prepare_inputs(..., founder_panel=panel)` | useful for synthetic truth |

`--founder-immutable` sets the loaded founder panel's immutable mask. For STITCH-parity benchmark runs with true founders, use supplied founders plus immutable behavior.

`--n-founders` is the total HMM founder count. If a VCF/PLINK founder panel has fewer samples than `--n-founders`, STITCHV2 keeps the loaded founders as-is and appends the difference as mutable founders initialized at ALT probability `0.5`. For example, `--founder-vcf founders.truth.vcf.gz --founder-immutable --n-founders 13` with 8 VCF samples runs 8 fixed founders plus 5 mutable founders.

### Pedigree Input

`--pedigree` accepts parquet or delimited text. Default columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | offspring/sample identifier |
| `father_id` | first parent |
| `mother_id` | second parent |

Override with `--pedigree-offspring-col`, `--pedigree-parent1-col`, and `--pedigree-parent2-col`.

Supported pedigree modes:

| Mode | Behavior |
| --- | --- |
| `off` | no pedigree adjustment |
| `smooth` | dosage smoothing toward parent/relative mean |
| `kinship` | kinship fallback from the sparse pedigree graph |
| `transmission` | recombination-smoothed parent-child message passing |

### Microarray / PLINK Evidence

Use `--microarray-plink PREFIX` for a global PLINK BED/BIM/FAM hard-call source. STITCHV2 aligns variants by chromosome and position and samples by IID/`sample_id`, then injects hard calls as extra read-like evidence.

If `samples` contains `plink_path`, STITCHV2 also loads per-sample PLINK hard calls from those prefixes.

### Classic STITCH Text Inputs

The synthetic benchmark writes classic STITCH inputs:

| File | Meaning |
| --- | --- |
| `bamlist.txt` | one BAM path per line |
| `sample_names.txt` | one sample name per line |
| `pos.txt` | four-column `CHR POS REF ALT`, no header |
| `gen.txt` | genetic positions used by the R STITCH benchmark |

Convert classic STITCH inputs to STITCHV2 tables:

```python
from pathlib import Path
import pandas as pd

def stitch_text_to_stitchv2_tables(data_dir: Path, generation: float = 10.0):
    bam_paths = pd.read_csv(data_dir / "bamlist.txt", header=None, names=["bam_path"])
    names_path = data_dir / "sample_names.txt"
    if names_path.exists():
        sample_ids = pd.read_csv(names_path, header=None, names=["sample_id"])
    else:
        sample_ids = pd.DataFrame({"sample_id": [f"sample_{i}" for i in range(len(bam_paths))]})
    samples = pd.concat([sample_ids, bam_paths], axis=1)
    samples["generation"] = float(generation)
    positions = pd.read_csv(
        data_dir / "pos.txt",
        sep=r"\s+",
        header=None,
        names=["CHR", "POS", "REF", "ALT"],
    )
    return samples, positions
```

## 4. CLI Runs

### Minimal CLI Run

This uses uniform mutable founders because no founder file is supplied.

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/chr1 \
  --n-founders 8 \
  --block-size 1000 \
  --em-iterations 5 \
  --hmm-backend jax \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### STITCH-Parity Run With True Founders

Use this shape when comparing to original STITCH under known-founder synthetic truth.

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chrSynthetic \
  --output-dir runs/stitchv2_parity_raw \
  --n-founders 8 \
  --founder-vcf founders.truth.vcf.gz \
  --founder-immutable \
  --block-size 1000 \
  --em-iterations 5 \
  --hmm-backend jax \
  --fragment-likelihood-mode replace \
  --fragment-coupling-model stitch_parity \
  --genotype-call-mode stitch_no_call \
  --genotype-call-stitch-threshold 0.9 \
  --no-calibrate-genotype-posteriors \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-haplotype-probabilities \
  --write-support-mask
```

There is no `--use-true-founders` flag in the STITCHV2 CLI. In this package, true-founder parity means supplying the true founder panel and freezing it with `--founder-immutable`. Some benchmark scripts have their own `--use-true-founders` flag; that flag belongs to the benchmark harness, not the STITCHV2 CLI.

### Same Run With Legacy Fixed Calibration

The default `standard_callability` mode keeps raw HMM GP unchanged and learns a hard-call gate when truth labels exist. The default gate is partial per SNP: STITCHV2 trains one continuous callability model, evaluates whether learned gating is safe for each SNP, and falls back to STITCH no-call for unsupported or unsafe SNPs. This command opts into the older fixed temperature/blend posterior calibration explicitly:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chrSynthetic \
  --output-dir runs/stitchv2_parity_calibrated \
  --n-founders 8 \
  --founder-vcf founders.truth.vcf.gz \
  --founder-immutable \
  --block-size 1000 \
  --em-iterations 5 \
  --hmm-backend jax \
  --fragment-coupling-model stitch_parity \
  --calibration-mode fixed \
  --genotype-call-mode stitch_no_call \
  --genotype-call-stitch-threshold 0.9 \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### Masked-CV Calibration With Microarray Evidence

`masked_cv` uses available microarray hard calls as held-out labels when enough labels exist in a block.

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/masked_cv \
  --n-founders 8 \
  --microarray-plink /data/arrays/all_samples \
  --calibration-mode masked_cv \
  --calibration-temperatures 0.15,0.25,0.35,0.5,0.75,1.0 \
  --calibration-blends 0,0.25,0.5,0.75,1 \
  --calibration-dosage-scales 0.75,1,1.25,1.5,2 \
  --calibration-dosage-offsets -0.25,0,0.25 \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### Sex Chromosome / chrX-Like Run

Requires a `sex` column in `samples`.

```bash
stitchv2 run \
  --samples samples_with_sex.parquet \
  --positions chrX_positions.parquet \
  --chromosome chrX \
  --output-dir runs/chrX \
  --n-founders 8 \
  --ploidy 2 \
  --ploidy-males 1 \
  --ploidy-females 2 \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### chrY-Like or Ploidy-Zero Run

Ploidy-zero samples are kept in output but skipped by the HMM. Their dosage is missing, genotype calls are `-1`, genotype posterior rows are `NaN` when written, and haplotype probabilities are zero vectors when written.

```bash
stitchv2 run \
  --samples samples_with_sex.parquet \
  --positions chrY_positions.parquet \
  --chromosome chrY \
  --output-dir runs/chrY \
  --n-founders 8 \
  --ploidy 2 \
  --ploidy-males 1 \
  --ploidy-females 0 \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-haplotype-probabilities
```

### Pedigree Transmission Run

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/pedigree_transmission \
  --n-founders 8 \
  --pedigree pedigree.parquet \
  --pedigree-mode transmission \
  --pedigree-strength 0.7 \
  --pedigree-iterations 6 \
  --pedigree-kinship-threshold 0.01 \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### Dask Run With Dashboard and Reports

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/dask \
  --n-founders 8 \
  --executor dask \
  --dask-n-workers 4 \
  --dask-threads-per-worker 1 \
  --dask-dashboard-address 127.0.0.1:8786 \
  --dask-performance-report dask_performance_report.html \
  --dask-task-stream dask_task_stream.json \
  --dask-target-task-memory-mb 4096 \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

For thread workers, STITCHV2 uses Dask's in-process protocol. For process workers and a `127.0.0.1:*` dashboard address, it also sets `host=127.0.0.1`.

## 5. Interactive Python API

### Uniform Founder Panel in Python

The Python API does not automatically create uniform founders. Create and pass a `FounderPanel`:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from stitchv2 import PipelineConfig, StitchPipeline
from stitchv2.founders import FounderPanel

samples = pd.read_parquet("samples.parquet")
positions = pd.read_parquet("positions.parquet")

founders = FounderPanel(
    chromosome="chr1",
    positions=positions["POS"].to_numpy(dtype=np.int64),
    ref=positions["REF"].astype(str).to_numpy(),
    alt=positions["ALT"].astype(str).to_numpy(),
    alt_prob=np.full((8, len(positions)), 0.5, dtype=np.float32),
    immutable_mask=np.zeros(8, dtype=bool),
)

cfg = PipelineConfig(
    chromosome="chr1",
    positions_path=Path("positions.parquet"),
    output_dir="runs/python_uniform",
    n_founders=8,
    block_size=1000,
    em_iterations=5,
    hmm_backend="jax",
    fragment_coupling_model="stitch_parity",
    write_genotype_posteriors=True,
    write_genotype_calls=True,
)

StitchPipeline(cfg).prepare_inputs(samples, founder_panel=founders)
```

### Founder VCF in Python

```python
import pandas as pd

from stitchv2 import FounderConfig, PipelineConfig, StitchPipeline

samples = pd.read_parquet("samples.parquet")

cfg = PipelineConfig(
    chromosome="chr1",
    positions_path="positions.parquet",
    output_dir="runs/python_founder_vcf",
    n_founders=8,
    block_size=1000,
    em_iterations=5,
    hmm_backend="jax",
    fragment_coupling_model="stitch_parity",
    calibrate_genotype_posteriors=False,
    genotype_call_mode="stitch_no_call",
    genotype_call_stitch_threshold=0.9,
    write_genotype_posteriors=True,
    write_genotype_calls=True,
    founder=FounderConfig(
        source_format="vcf",
        source_path="founders.truth.vcf.gz",
        immutable=True,
    ),
)

StitchPipeline(cfg).prepare_inputs(samples)
```

### Pedigree in Python

```python
import pandas as pd

from stitchv2 import FounderConfig, PipelineConfig, StitchPipeline

samples = pd.read_parquet("samples.parquet")
pedigree = pd.read_parquet("pedigree.parquet")

cfg = PipelineConfig(
    chromosome="chr1",
    positions_path="positions.parquet",
    output_dir="runs/python_pedigree",
    n_founders=8,
    hmm_backend="jax",
    pedigree_mode="transmission",
    pedigree_strength=0.7,
    pedigree_iterations=6,
    write_genotype_posteriors=True,
    write_genotype_calls=True,
    founder=FounderConfig(
        source_format="vcf",
        source_path="founders.truth.vcf.gz",
        immutable=True,
    ),
)

StitchPipeline(cfg).prepare_inputs(samples, pedigree=pedigree)
```

## 6. Output Files and Schemas

STITCHV2 writes chunked parquet files under `output_dir`.

| Path | Written when | Columns |
| --- | --- | --- |
| `samples.parquet` | always | input sample columns plus resolved `ploidy` |
| `positions.parquet` | always | `CHR`, `POS`, `REF`, `ALT` after filtering |
| `founders.parquet` | always | `chromosome`, `position`, `founder`, `alt_prob`, `immutable` |
| `dosage/block=*.parquet` | always | `sample_id`, `chromosome`, `position`, `dosage`, `block_id` |
| `support_mask/block=*.parquet` | `--write-support-mask` | `sample_id`, `chromosome`, `position`, `has_supporting_read`, `block_id` |
| `recombination/block=*.parquet` | always | `chromosome`, `position`, `recombination_rate`, `block_id` |
| `transitions/block=*.parquet` | `--write-transitions` in CLI | `sample_id`, `chromosome`, `position`, `switch_probability`, `stay_probability`, `offdiag_probability`, `block_id` |
| `transitions_full/block=*.parquet` | Python `transition_output="full"` and available full transitions | `sample_id`, `chromosome`, `position`, `transition_probability`, `block_id` |
| `haplotype_probabilities/block=*.parquet` | `--write-haplotype-probabilities` | `sample_id`, `chromosome`, `position`, `hap_dosage`, `hap_probability`, `block_id` |
| `genotype_posteriors/block=*.parquet` | `--write-genotype-posteriors` | `sample_id`, `chromosome`, `position`, `genotype_posterior`, `block_id` |
| `genotype_calls/block=*.parquet` | `--write-genotype-calls` | `sample_id`, `chromosome`, `position`, `genotype_call`, `block_id` |
| `founder_updates/block=*.parquet` | always | `chromosome`, `position`, `founder`, `alt_prob`, `block_id` |
| `pileup/block=*.parquet` | Python `write_pileup=True` | read count and weight columns |
| `stage_timings.json` | always | per-block timing and memory fields |
| `memory_profile_summary.json` | profiling enabled | `peak_rss_mb`, `peak_hmm_rss_mb`, `blocks_profiled` |
| `pedigree_summary.json` | pedigree graph present | pedigree graph summary |
| `dask_run_summary.json` | Dask executor | dashboard, chunk plan, task diagnostics |
| `cli_run_summary.json` | CLI runs | command summary |
| `run_summary.json` | CLI runs | stable copy of CLI run summary |

`stage_timings.json` contains per-block fields including:

- `seconds_read_extract`
- `seconds_hmm`
- `seconds_calibration`
- `seconds_write`
- `seconds_total`
- `mean_depth`
- `n_reads`
- `call_rate` and `no_call_rate` when calls are written
- `rss_mb_block_start`, `rss_mb_after_reads`, `rss_mb_after_hmm`, `rss_mb_after_calibration`, `rss_mb_after_write` when profiling is enabled
- pedigree and calibration metadata when applicable

## 7. Export and Utility Commands

### Combine Chunked Parquet

```bash
stitchv2 combine \
  --run-output-dir runs/chr1 \
  --datasets all \
  --output-dir runs/chr1/combined
```

For a one-off directory combine:

```bash
stitchv2 combine \
  --run-output-dir runs/chr1 \
  --input-dir runs/chr1/genotype_calls \
  --output-file runs/chr1/genotype_calls.parquet
```

`combine_pipeline_outputs()` returns a lazy xarray dataset backed by Dask arrays and writes combined parquet files. Add `--write-zarr` when floating dosage/posterior outputs should also be persisted as an xarray/Zarr store:

```bash
stitchv2 combine \
  --run-output-dir runs/chr1 \
  --datasets all \
  --output-dir runs/chr1/combined \
  --write-zarr
```

Hard calls and categorical columns remain in Parquet, where dictionary/RLE/bit-packing encodings are a better fit. Dosages, genotype posteriors, transitions, recombination rates, and founder probabilities are the values written to Zarr.

### Optional BCF Interoperability

STITCHV2 does not allow VCF export. Keep primary results in Parquet/Zarr. BCF export is available only when an external tool explicitly requires it, and it WILL slow down I/O compared with the native Parquet/Zarr outputs.

### Export BCF

```bash
stitchv2 export-bcf \
  --run-output-dir runs/chr1 \
  --output-bcf runs/chr1/stitchv2.bcf \
  --chromosome chr1
```

### Reformat STITCH Filenames

```bash
stitchv2 reformat-stitch-filenames \
  --input-file stitch_files.txt \
  --output-file stitch_filename_map.json
```

## 8. Complete CLI Parameter Surface

### `stitchv2 run`

| Parameter | Default | Meaning |
| --- | --- | --- |
| `--samples` | required | samples table |
| `--positions` | required | positions table |
| `--chromosome` | required | chromosome/contig |
| `--chr-start` | none | inclusive coordinate start |
| `--chr-end` | none | inclusive coordinate end |
| `--output-dir` | required | run output directory |
| `--n-founders` | required | total HMM founders; loaded VCF/PLINK panels are padded with mutable founders if fewer samples are present |
| `--ploidy` | `2` | default sample ploidy |
| `--ploidy-males` | none | male ploidy override |
| `--ploidy-females` | none | female ploidy override |
| `--block-size` | `1000` | variants per CLI block |
| `--em-iterations` | `2` | EM iterations per block in CLI |
| `--hmm-backend` | `auto` | `auto`, `numpy`, `jax`, or `torch` |
| `--jax-sample-batch-size` | `0` | samples per JAX forward-backward call |
| `--executor` | `serial` | `serial` or `dask` |
| `--dask-scheduler` | `local` | local Dask scheduler |
| `--dask-n-workers` | `0` | local worker count; `0` auto-selects |
| `--dask-threads-per-worker` | `1` | Dask threads per worker |
| `--dask-processes` | false | use Dask processes |
| `--dask-memory-limit` | empty | per-worker memory limit |
| `--dask-dashboard-address` | `:8787` | dashboard bind address; empty disables |
| `--dask-performance-report` | empty | Dask HTML report path |
| `--dask-task-stream` | empty | task stream `.json` or `.html` path |
| `--dask-dashboard-hold-seconds` | `0.0` | keep dashboard alive after compute |
| `--dask-target-task-memory-mb` | `0.0` | chunk planner memory target |
| `--dask-min-block-size` | `128` | minimum Dask block size |
| `--dask-min-sample-batch-size` | `8` | minimum Dask sample batch |
| `--dask-sample-batch-size` | `0` | samples per Dask HMM task |
| `--read-mode` | `read_stream` | `read_stream` or `pileup` |
| `--read-stream-backend` | `auto` | `auto`, `python`, `htslib`, `snp_only_bamreader`, `stitch_style_bamreader`, or `variant_aware_bamreader` |
| `--io-workers` | `1` | read extraction workers |
| `--htslib-threads-per-file` | `1` | htslib threads per BAM/CRAM |
| `--fragment-likelihood-mode` | `replace` | `replace` or `augment`; `replace` is the parity/default path |
| `--fragment-coupling-model` | `stitch_parity` | current fragment coupling model |
| `--fragment-max-diff-reads` | `100.0` | read-difference guardrail |
| `--fragment-max-emission-diff` | `1000.0` | emission-difference guardrail |
| `--no-fragment-rescale-read-likelihood` | false | disable read likelihood rescaling |
| `--write-transitions` | false | write compact transition output |
| `--write-haplotype-probabilities` | false | write haplotype output |
| `--write-genotype-posteriors` | false | write GP output |
| `--write-genotype-calls` | false | write hard calls |
| `--write-support-mask` | false | write support mask |
| `--no-calibrate-genotype-posteriors` | false | disable calibration/callability fitting |
| `--calibration-mode` | `standard_callability` | `standard_callability`, `fixed`, or `masked_cv` |
| `--genotype-posterior-temperature` | `0.35` | fixed/dosage-fallback calibration temperature |
| `--genotype-posterior-blend` | `0.35` | fixed calibration blend; not used by default standard callability |
| `--genotype-call-mode` | `quality_gated` | `argmax`, `stitch_no_call`, or `quality_gated` |
| `--genotype-call-min-confidence` | `0.0` | quality gate confidence |
| `--genotype-call-min-margin` | `0.0` | quality gate or STITCH no-call margin |
| `--genotype-call-stitch-threshold` | `0.9` | STITCH-style GP no-call threshold |
| `--genotype-call-correctness-threshold` | `0.0` | learned correctness threshold |
| `--use-lightgbm-calibrator` | false | advanced legacy posterior/block-context calibrator, not required for default LightGBM callability |
| `--calibration-context-window` | `25` | calibration context window |
| `--calibration-block-snps` | `64` | calibration SNP block size |
| `--calibration-use-optuna` | false | tune LightGBM with Optuna |
| `--calibration-optuna-trials` | `20` | Optuna trial count |
| `--calibration-max-train-rows` | `750000` | learned calibration row cap |
| `--calibration-callability-model` | `lightgbm` | model used by default callability: `lightgbm` or `sklearn_logistic` |
| `--calibration-callability-min-train-rows` | `64` | minimum labeled rows for default callability fitting |
| `--calibration-callability-min-call-rate` | `0.80` | minimum call rate considered while choosing the default callability threshold |
| `--calibration-callability-call-rate-weight` | `0.03` | small call-rate reward in the default threshold objective |
| `--calibration-callability-decision-mode` | `per_snp_hierarchical` | default partial per-SNP calibration with local/feature/global fallback to STITCH no-call; `global` is the legacy scalar gate |
| `--calibration-train-site-fraction` | `1.0` | labeled SNP fraction for training |
| `--calibration-lightgbm-use-block-context` | false | enable older local-context stage |
| `--calibration-lightgbm-use-fixed-stage0` | false | fixed calibration before LightGBM |
| `--calibration-maf-bins` | `0,0.01,0.05,0.5` | MAF strata for masked-CV |
| `--calibration-temperatures` | `0.15,0.25,0.35,0.5,0.75,1.0` | candidate temperatures |
| `--calibration-blends` | `0,0.25,0.5,0.75,1` | candidate blends |
| `--calibration-dosage-scales` | `0.75,1,1.25,1.5,2` | candidate dosage scales |
| `--calibration-dosage-offsets` | `-0.25,0,0.25` | candidate dosage offsets |
| `--calibration-hwe-prior-weights` | `0,0.25,0.5,1` | candidate HWE prior weights |
| `--no-calibration-optimize-dosage-scale` | false | disable dosage scale/offset tuning |
| `--calibration-hwe-weight` | `0.0` | HWE weight |
| `--calibration-hwe-min-maf` | `0.05` | minimum MAF for HWE terms |
| `--microarray-plink` | empty | global PLINK hard-call prefix |
| `--microarray-generation-default` | `NaN` | generation for added PLINK-only samples |
| `--microarray-hard-call-weight` | `80` | hard-call evidence weight |
| `--no-microarray-add-samples` | false | do not add PLINK-only samples |
| `--random-seed` | `0` | random seed |
| `--founder-init-jitter` | `0.0` | mutable founder jitter |
| `--memory-map-read-matrices` | false | memmap read evidence matrices |
| `--memory-map-dir` | empty | memmap directory |
| `--no-profile-memory` | false | disable memory profiling |
| `--compression` | `zstd` | parquet compression |
| `--compression-level` | `6` | parquet compression level |
| `--pedigree` | empty | pedigree table |
| `--pedigree-mode` | `smooth` | `off`, `smooth`, `kinship`, `transmission` |
| `--pedigree-strength` | `0.0` | pedigree adjustment strength |
| `--pedigree-offspring-col` | `sample_id` | offspring column |
| `--pedigree-parent1-col` | `father_id` | first parent column |
| `--pedigree-parent2-col` | `mother_id` | second parent column |
| `--pedigree-iterations` | `4` | pedigree iterations |
| `--pedigree-kinship-threshold` | `0.01` | kinship threshold |
| `--founder-vcf` | empty | founder VCF |
| `--founder-plink` | empty | founder PLINK prefix |
| `--founder-immutable` | false | freeze loaded founders; extra padded founders remain mutable |

### Other CLI Commands

| Command | Main extra parameters |
| --- | --- |
| `stitchv2 cv` | all common run args, optional `--n-founders`, plus `--pseudo-truth`, `--k-values`, `--ngen-values`, `--s-values`, `--seeds`, `--folds`, `--holdout-fraction`, `--lightgbm-post-calibrator` |
| `stitchv2 tune-jax-memory` | `--data-dir`, `--output-dir`, `--chromosome`, `--n-founders`, `--em-iterations`, `--block-sizes`, `--memory-map-modes`, `--jax-sample-batch-sizes`, `--max-samples`, `--max-positions`, `--max-peak-rss-mb` |
| `stitchv2 combine` | `--run-output-dir`, `--datasets`, `--output-dir`, `--input-dir`, `--output-file`, `--compression`, `--compression-level`, `--row-group-size`, `--write-zarr`, `--zarr-output`, `--zarr-consolidated` |
| `stitchv2 export-bcf` | explicit slow interoperability export: `--run-output-dir`, `--output-bcf`, `--chromosome`, `--include-haplotype-dosage`, `--no-tabix-index` |
| `stitchv2 reformat-stitch-filenames` | `--input-file`, `--filenames`, `--output-file` |

## 9. Benchmark Workflows in This Repo

This section documents how the benchmark scripts are intended to be run. Do not paste benchmark results into this tutorial.

### Generate Synthetic Data Under the Model

```bash
python benchmarks/synthetic_dataset.py \
  --output-dir benchmark_runs/synth_5mb_2k_0p1x \
  --chromosome chrSynthetic \
  --chromosome-length 5000000 \
  --n-variants 2000 \
  --n-founders 8 \
  --n-samples 48 \
  --coverage 0.1 \
  --read-length 150 \
  --generations 10 \
  --seed 7
```

Important generated files:

- `samples.parquet`, `samples.tsv`
- `positions.parquet`, `positions.tsv`
- `truth_dosage.parquet`
- `founder_truth.parquet`
- `founders.truth.vcf`
- `reference.fa`
- `bamlist.txt`, `sample_names.txt`, `pos.txt`, `gen.txt`

### Synthetic STITCH vs STITCHV2 Baseline

`benchmark_compare.py` compares the Python STITCHV2 pipeline against original R STITCH and can use true synthetic founders through the benchmark-level `--use-true-founders` flag.

```bash
python benchmarks/benchmark_compare.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/compare_synth \
  --chromosome chrSynthetic \
  --k 8 \
  --generations 10 \
  --iterations 5 \
  --block-size 1000 \
  --use-true-founders \
  --fragment-likelihood-mode replace \
  --fragment-coupling-model stitch_parity \
  --read-mode read_stream \
  --read-stream-backend auto
```

### STITCHV2 Serial vs Dask vs Original STITCH With Plots

`benchmark_stitch_stitchv2_dask_report.py` writes per-SNP metrics, ROC points, violin plots, ROC curves, runtime summaries, STITCHV2 stage timing summaries, and Dask reports.

```bash
python benchmarks/benchmark_stitch_stitchv2_dask_report.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/stitch_stitchv2_dask \
  --chromosome chrSynthetic \
  --k 8 \
  --generations 10 \
  --iterations 5 \
  --block-size 1000 \
  --jax-sample-batch-size 128 \
  --dask-n-workers 2 \
  --dask-threads-per-worker 1 \
  --dask-sample-batch-size 12 \
  --dask-dashboard-address 127.0.0.1:8786 \
  --force
```

Expected artifacts include:

- `per_snp_metrics_long.parquet`
- `roc_points.parquet`
- `plots/per_snp_metric_violins.png`
- `plots/roc_curves.png`
- `stitchv2/stage_timings.json`
- `stitchv2_dask/dask_run_summary.json`
- `stitchv2_dask/dask_performance_report.html`
- `stitchv2_dask/dask_task_stream.json`

### Ploidy, Sex Chromosomes, and Polyploidy

`benchmark_ploidy_modes.py` runs:

- all ploidy zero
- ploidy 1
- ploidy 2
- ploidy 3
- ploidy 4
- chrX-like male/female ploidy
- chrY-like male/female ploidy with female ploidy zero

```bash
python benchmarks/benchmark_ploidy_modes.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/ploidy_modes \
  --chromosome chrSynthetic \
  --k 8 \
  --iterations 2 \
  --block-size 500 \
  --backend jax
```

`benchmark_jax_generic_ploidy_parity.py` compares optimized haploid/diploid JAX paths to the generic count-state ploidy HMM and includes ploidy 1, 2, 3, and 4:

```bash
python benchmarks/benchmark_jax_generic_ploidy_parity.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir benchmark_runs/jax_generic_ploidy \
  --chromosome chrSynthetic \
  --k 8 \
  --n-samples 12 \
  --n-positions 96 \
  --iterations 1
```

### Pedigree Benchmark

`benchmark_pedigree_synthetic.py` generates a family-structured synthetic dataset and validates:

- STITCHV2 without pedigree
- STITCHV2 `smooth`
- STITCHV2 `kinship`
- STITCHV2 `transmission`
- original STITCH baseline unless skipped

```bash
python benchmarks/benchmark_pedigree_synthetic.py \
  --output-dir benchmark_runs/pedigree_synthetic \
  --chromosome chrSynthetic \
  --n-variants 400 \
  --n-founders 8 \
  --n-direct-families 5 \
  --n-grandparent-families 4 \
  --coverage 0.15 \
  --iterations 5 \
  --block-size 200 \
  --pedigree-strength 0.70 \
  --pedigree-iterations 6 \
  --kinship-threshold 0.01 \
  --force
```

Expected artifacts include aggregate metrics, sample-group metrics, per-SNP metrics, ROC points, violin plots, ROC curves, and pedigree stage summaries.

### Real-Data GATK Pseudo-Truth Benchmark

`benchmark_real_subsample_gatk_compare.py` prepares a real-style benchmark using `bamlist.txt`, `pos.txt`, and `reference.fa`, creates GATK pseudo-truth calls, and compares STITCHV2 against original STITCH.

```bash
python benchmarks/benchmark_real_subsample_gatk_compare.py \
  --data-dir benchmark_runs/original_stitch_mouse \
  --output-dir benchmark_runs/real_gatk_compare \
  --chromosome chr19 \
  --k 8 \
  --generations 100 \
  --iterations 2 \
  --block-size 1000
```

Then run the calibration/calling sweep:

```bash
python benchmarks/benchmark_real_calling_calibration_sweep.py \
  --benchmark-dir benchmark_runs/real_gatk_compare \
  --output-dir benchmark_runs/real_gatk_compare/calibration_calling_sweep \
  --k 4 \
  --iterations 2 \
  --block-size 500 \
  --thresholds 0.5,0.7,0.8,0.9,0.95,0.99
```

### Full Report From Existing Artifacts

`generate_full_benchmark_report.py` is a report generator. It expects existing benchmark artifacts and writes summary tables and plots. It should not be treated as a benchmark runner.

```bash
python benchmarks/generate_full_benchmark_report.py \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --stitchv2-hap0-dir benchmark_runs/hap0/stitchv2 \
  --stitchv2-hap1-dir benchmark_runs/hap1/stitchv2 \
  --stitch-r-hap0-dir benchmark_runs/hap0/stitch \
  --stitch-r-hap1-dir benchmark_runs/hap1/stitch \
  --capacity-summary-json benchmark_runs/capacity/summary.json \
  --real-gatk-benchmark-dir benchmark_runs/real_gatk_compare \
  --output-dir benchmark_runs/full_report
```

## 10. Original STITCH Command Shape

The benchmark harness calls original STITCH through `benchmarks/run_r_stitch_benchmark.R`. A typical R-side shape is:

```r
STITCH::STITCH(
  tempdir = "tmp",
  chr = "chr1",
  bamlist = "bamlist.txt",
  sampleNames_file = "sample_names.txt",
  posfile = "pos.txt",
  outputdir = "stitch_out/",
  K = 8,
  nGen = 10,
  niterations = 5,
  nCores = 4,
  method = "diploid"
)
```

The closest STITCHV2 command shape is:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir stitchv2_out \
  --n-founders 8 \
  --em-iterations 5 \
  --hmm-backend jax \
  --io-workers 4 \
  --ploidy 2 \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

## 11. Practical Checks Before Running

Before large runs, check:

- `samples.generation` is correct
- contig names match across BAM/CRAM, positions, reference, founder VCF/PLINK, and `--chromosome`
- founder behavior is explicit: uniform mutable, loaded mutable, or loaded immutable
- parity runs use `fragment_coupling_model=stitch_parity`
- calibration is intentionally on or off
- `--write-genotype-posteriors` and `--write-genotype-calls` are set when metrics or explicit BCF interoperability output need GP/GT
- Dask report paths are set for Dask runs that need scheduler diagnostics

Common symptoms:

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| no reads overlap variants | contig or coordinate mismatch | BAM header, `positions.CHR`, `--chromosome` |
| Python API raises founder `source_path` error | no `founder_panel` was passed | pass a `FounderPanel` or set `FounderConfig.source_path` |
| chrX/chrY ploidy wrong | missing or unrecognized `sex` values | normalize `sex` and set both sex-specific ploidy flags |
| Dask slower than serial | tasks too small | increase block size or sample batch size |
| JAX out of memory | too many samples/SNPs/founders per task | lower `--block-size` or `--jax-sample-batch-size`; use Dask chunk planning |
| STITCHV2 differs from STITCH | founder, calibration, call policy, or read-coupling mismatch | use supplied immutable founders, `stitch_parity`, and comparable no-call policy |
