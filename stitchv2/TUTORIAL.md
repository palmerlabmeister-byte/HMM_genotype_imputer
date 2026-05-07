# STITCHV2 Tutorial

This tutorial is the practical runbook for STITCHV2: what files it expects, how to run it from the CLI and from Python, how to set it up for STITCH-style parity runs, and what every exposed CLI parameter does.

It is intentionally instruction-only. It explains how to run STITCHV2 and how to choose settings; it does not store benchmark results.

## Mental Model

`stitchv2 run` does four main things:

1. reads sample metadata plus variant positions
2. extracts read evidence from BAM/CRAM, optionally mixes in founder and microarray evidence
3. runs the HMM block by block
4. writes chunked parquet outputs, with optional posterior calibration and exports

Original STITCH is centered on an R entrypoint and VCF-oriented outputs. STITCHV2 is centered on a Python package, parquet-first outputs, explicit batching controls, explicit mixed-ploidy handling, and optional post-HMM calibration.

## Quick Start

Minimal CLI run:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/chr1 \
  --n-founders 8 \
  --em-iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

Minimal interactive Python run:

```python
import pandas as pd

from stitchv2 import PipelineConfig, StitchPipeline

samples = pd.read_parquet("samples.parquet")

cfg = PipelineConfig(
    chromosome="chr1",
    positions_path="positions.parquet",
    output_dir="runs/chr1",
    n_founders=8,
    em_iterations=5,
    block_size=1000,
    hmm_backend="jax",
    fragment_coupling_model="stitch_parity",
    write_genotype_posteriors=True,
    write_genotype_calls=True,
)

StitchPipeline(cfg).prepare_inputs(samples)
```

## STITCH Parity

For a STITCH-style comparison, the main idea is to remove avoidable behavioral differences:

- keep `fragment_coupling_model=stitch_parity`
- use enough EM iterations for a serious run, usually `5-10`
- use comparable hard-call behavior if comparing calls
- decide explicitly whether you want raw HMM posteriors or calibrated posteriors
- if you have known founders, provide them and freeze them

In this repo, the STITCHV2 equivalent of "true founders + hard immutable" is:

- provide founders through `--founder-vcf`, `--founder-plink`, or an in-memory `FounderPanel`
- set `--founder-immutable`

There is no literal `--use-true-founders` CLI flag in STITCHV2 today. In practice, "use true founders" means supplying the true founder panel as input rather than letting STITCHV2 start from uniform founders.

### STITCH-Parity CLI Recipe

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/parity_raw \
  --n-founders 8 \
  --founder-vcf founders.truth.vcf.gz \
  --founder-immutable \
  --em-iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --fragment-likelihood-mode augment \
  --fragment-coupling-model stitch_parity \
  --genotype-call-mode stitch_no_call \
  --genotype-call-stitch-threshold 0.9 \
  --no-calibrate-genotype-posteriors \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-support-mask
```

### STITCH-Parity CLI Recipe With Calibration Left On

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/parity_calibrated \
  --n-founders 8 \
  --founder-vcf founders.truth.vcf.gz \
  --founder-immutable \
  --em-iterations 5 \
  --block-size 1000 \
  --hmm-backend jax \
  --fragment-likelihood-mode augment \
  --fragment-coupling-model stitch_parity \
  --genotype-call-mode stitch_no_call \
  --genotype-call-stitch-threshold 0.9 \
  --calibration-mode fixed \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-support-mask
```

### STITCH-Parity Interactive Python Recipe

```python
import pandas as pd

from stitchv2 import FounderConfig, PipelineConfig, StitchPipeline

cfg = PipelineConfig(
    chromosome="chr1",
    positions_path="positions.parquet",
    output_dir="runs/parity_python",
    n_founders=8,
    em_iterations=5,
    block_size=1000,
    hmm_backend="jax",
    fragment_likelihood_mode="augment",
    fragment_coupling_model="stitch_parity",
    calibrate_genotype_posteriors=False,
    genotype_call_mode="stitch_no_call",
    genotype_call_stitch_threshold=0.9,
    write_genotype_posteriors=True,
    write_genotype_calls=True,
    write_support_mask=True,
    founder=FounderConfig(
        source_format="vcf",
        source_path="founders.truth.vcf.gz",
        immutable=True,
    ),
)

samples = pd.read_parquet("samples.parquet")
StitchPipeline(cfg).prepare_inputs(samples)
```

### Original STITCH vs STITCHV2 Command Shape

Original STITCH call shape:

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
  method = "diploid",
  output_haplotype_dosages = FALSE
)
```

Closest STITCHV2 shape:

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

## Input Files

### Samples Table

`--samples` accepts parquet or a delimited text file readable by pandas.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | Unique sample name used everywhere downstream. |
| `bam_path` | string | BAM/CRAM path. Empty string is allowed for samples with no reads. |
| `generation` | float | Generations since the founder mosaic start; used in recombination modeling. |

Optional columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `sex` | string | Used by `--ploidy-males` and `--ploidy-females`. |
| `father_id`, `mother_id` | string | Useful when pedigree data lives with the sample table. |
| `plink_path` | string | Per-sample PLINK prefix for hard microarray evidence. |

Accepted sex labels include `M`, `male`, `1`, `XY`, `F`, `female`, `2`, and `XX`.

Example:

```text
sample_id,bam_path,generation,sex,father_id,mother_id
s1,/data/bams/s1.bam,100,F,,
s2,/data/bams/s2.bam,100,M,,
child1,/data/bams/child1.bam,101,F,s2,s1
```

### Positions Table

`--positions` accepts parquet or delimited text.

Required columns:

| Column | Type | Meaning |
| --- | --- | --- |
| `CHR` | string | Chromosome or contig name. |
| `POS` | integer | 1-based variant position. |
| `REF` | string | Reference allele. |
| `ALT` | string | Alternate allele. |

Example:

```text
CHR POS REF ALT
chr19 3000001 A G
chr19 3000104 C T
```

### BAM/CRAM Inputs

Each row in `samples` points to a BAM or CRAM. The contig names must agree with:

- `--chromosome`
- the `CHR` column in `--positions`
- founder inputs if founders are provided

For CRAM, the normal reference requirements apply through the underlying HTS stack.

### Founder Inputs

STITCHV2 can initialize founders in four ways:

| Source | How to provide it | When to use it |
| --- | --- | --- |
| Uniform mutable founders | no founder input | exploratory runs, no known founders |
| Founder VCF | `--founder-vcf founders.vcf.gz` | parity runs or known-founder simulations |
| Founder PLINK | `--founder-plink founders_prefix` | founder genotypes already in BED/BIM/FAM |
| In-memory founder panel | Python `FounderPanel` | tests, custom workflows, synthetic truth |

`--founder-immutable` freezes supplied founders. This is the closest parity mode when founders are known and should not be updated.

### Pedigree Input

`--pedigree` accepts parquet or delimited text.

Default columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | offspring/sample identifier |
| `father_id` | first parent |
| `mother_id` | second parent |

Override them with:

```bash
--pedigree-offspring-col rfid
--pedigree-parent1-col dam
--pedigree-parent2-col sire
```

Supported pedigree modes:

| Mode | Meaning |
| --- | --- |
| `off` | ignore pedigree data |
| `smooth` | smooth dosage toward parent means |
| `kinship` | kinship-weighted fallback from sparse pedigree structure |
| `transmission` | Mendelian parent-child message passing with chromosome smoothing |

### Microarray Evidence

Use `--microarray-plink PREFIX` when you have hard array genotypes in a global PLINK BED/BIM/FAM prefix. STITCHV2 aligns samples by IID to `sample_id`, aligns variants by chromosome and position, and injects the hard calls as extra evidence.

Per-sample PLINK prefixes can also be supplied in a `plink_path` column in the samples table.

### STITCH-Compatible Text Inputs

If you already have classic STITCH text inputs such as:

```text
bamlist.txt
sample_names.txt
pos.txt
```

convert them into STITCHV2-style tables like this:

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

## CLI Workflows

### Standard Production Run

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/production \
  --n-founders 8 \
  --block-size 1000 \
  --em-iterations 5 \
  --hmm-backend jax \
  --read-stream-backend auto \
  --io-workers 4 \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-support-mask
```

### Disable Calibration

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir runs/raw_gp \
  --n-founders 8 \
  --fragment-coupling-model stitch_parity \
  --no-calibrate-genotype-posteriors \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### Mixed-Sex chrX Example

```bash
stitchv2 run \
  --samples samples_with_sex.parquet \
  --positions positions_chrX.parquet \
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

### Ploidy-Zero Behavior

When a sample resolves to ploidy `0`, STITCHV2 keeps the sample in the outputs but skips it in the HMM. The output remains shape-stable:

- genotype calls are missing (`-1`)
- dosages are missing
- haplotype probabilities are all zeros when haplotype output is requested

This is the right pattern for samples that should remain represented in files but should not contribute HMM state.

### Pedigree Transmission Example

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
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### Dask-Orchestrated Run

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
  --dask-dashboard-address :8787 \
  --dask-performance-report dask_report.html \
  --dask-task-stream dask_task_stream.json \
  --dask-target-task-memory-mb 4096 \
  --fragment-coupling-model stitch_parity \
  --write-genotype-posteriors \
  --write-genotype-calls
```

## Interactive Python Workflows

### Basic Python API Run

```python
from pathlib import Path

import numpy as np
import pandas as pd

from stitchv2 import PipelineConfig, StitchPipeline
from stitchv2.founders import FounderPanel

samples = pd.DataFrame(
    {
        "sample_id": ["s1", "s2"],
        "bam_path": ["/data/s1.bam", "/data/s2.bam"],
        "generation": [100.0, 100.0],
    }
)
positions = pd.DataFrame(
    {
        "CHR": ["chr1", "chr1"],
        "POS": [3000001, 3000104],
        "REF": ["A", "C"],
        "ALT": ["G", "T"],
    }
)
positions_path = Path("positions.parquet")
positions.to_parquet(positions_path, index=False)

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
    positions_path=positions_path,
    output_dir="runs/interactive",
    n_founders=8,
    block_size=1000,
    em_iterations=2,
    hmm_backend="jax",
    read_mode="read_stream",
    read_stream_backend="auto",
    fragment_likelihood_mode="augment",
    fragment_coupling_model="stitch_parity",
    write_genotype_posteriors=True,
    write_genotype_calls=True,
)

StitchPipeline(cfg).prepare_inputs(samples, founder_panel=founders)
```

### Pedigree Through Python

```python
import pandas as pd

from stitchv2 import PipelineConfig, StitchPipeline

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
)

StitchPipeline(cfg).prepare_inputs(samples, pedigree=pedigree)
```

## Outputs

STITCHV2 writes parquet datasets under `--output-dir`.

| Path | When written | Meaning |
| --- | --- | --- |
| `samples.parquet` | always | normalized sample table used by the run |
| `positions.parquet` | always | filtered positions used by the run |
| `founders.parquet` | always | founder panel used after initialization |
| `dosage/block=*.parquet` | always | per-sample, per-site dosage |
| `recombination/block=*.parquet` | always | per-site recombination summaries |
| `founder_updates/block=*.parquet` | always | founder ALT probabilities by site |
| `transitions/block=*.parquet` | when transition output is enabled | compact transition summaries |
| `transitions_full/block=*.parquet` | full transition mode | full transition vectors |
| `haplotype_probabilities/block=*.parquet` | `--write-haplotype-probabilities` | haplotype dosage and probability vectors |
| `genotype_posteriors/block=*.parquet` | `--write-genotype-posteriors` | genotype posterior vectors |
| `genotype_calls/block=*.parquet` | `--write-genotype-calls` | hard genotype calls; `-1` means missing or no-call |
| `support_mask/block=*.parquet` | `--write-support-mask` | direct evidence indicator |
| `stage_timings.json` | always | per-block read/HMM/calibration/write timing data |
| `memory_profile_summary.json` | unless profiling is disabled | memory summary |
| `dask_run_summary.json` | Dask runs | Dask runtime, report, and chunk-plan metadata |
| `cli_run_summary.json` | CLI runs | summary of the exact CLI invocation state |
| `run_summary.json` | CLI runs | same run summary in a stable location |

After a run, export helpers can write VCF or BCF from the parquet outputs.

## Complete CLI Reference

### `stitchv2 run`

#### Required Inputs and Region

| Parameter | Meaning |
| --- | --- |
| `--samples` | sample table with `sample_id`, `bam_path`, and `generation` |
| `--positions` | position table with `CHR`, `POS`, `REF`, `ALT` |
| `--chromosome` | chromosome or contig to process |
| `--chr-start` | optional inclusive start coordinate |
| `--chr-end` | optional inclusive end coordinate |
| `--output-dir` | output directory |
| `--n-founders` | number of founders or latent founder states |

#### Ploidy

| Parameter | Meaning |
| --- | --- |
| `--ploidy` | default ploidy for all samples; must be `>= 0` |
| `--ploidy-males` | male ploidy override; requires `--ploidy-females` and a `sex` column |
| `--ploidy-females` | female ploidy override; requires `--ploidy-males` and a `sex` column |

#### HMM and Blocking

| Parameter | Meaning |
| --- | --- |
| `--block-size` | variants per HMM and output block |
| `--em-iterations` | EM iterations per block |
| `--hmm-backend` | `auto`, `numpy`, `jax`, or `torch` |
| `--jax-sample-batch-size` | samples per JAX forward-backward call; `0` means all samples |
| `--random-seed` | seed for stochastic pieces |
| `--founder-init-jitter` | small initialization perturbation for mutable founders |

#### Execution and Dask

| Parameter | Meaning |
| --- | --- |
| `--executor` | `serial` or `dask` |
| `--dask-scheduler` | Dask scheduler type; currently `local` |
| `--dask-n-workers` | number of local workers; `0` lets STITCHV2 choose |
| `--dask-threads-per-worker` | threads per Dask worker |
| `--dask-processes` | use processes instead of thread workers |
| `--dask-memory-limit` | per-worker memory limit such as `16GB` |
| `--dask-dashboard-address` | dashboard bind address; empty disables the dashboard |
| `--dask-performance-report` | write a permanent Dask HTML report |
| `--dask-task-stream` | write a captured task stream as `.json` or `.html` |
| `--dask-dashboard-hold-seconds` | keep the dashboard alive briefly after compute |
| `--dask-target-task-memory-mb` | planner target for rough per-task memory |
| `--dask-min-block-size` | minimum Dask-planned block size |
| `--dask-min-sample-batch-size` | minimum Dask-planned sample batch size |
| `--dask-sample-batch-size` | samples per Dask HMM task; `0` uses planner or JAX batching |

#### Read Extraction

| Parameter | Meaning |
| --- | --- |
| `--read-mode` | `read_stream` or `pileup` |
| `--read-stream-backend` | `auto`, `python`, or `htslib` |
| `--io-workers` | read-extraction workers |
| `--htslib-threads-per-file` | decompression threads per input file |
| `--memory-map-read-matrices` | store read matrices as memory maps |
| `--memory-map-dir` | directory for memmap files |

#### Fragment Likelihood

| Parameter | Meaning |
| --- | --- |
| `--fragment-likelihood-mode` | `replace` or `augment` |
| `--fragment-coupling-model` | currently `stitch_parity` |
| `--fragment-max-diff-reads` | numerical guardrail for read-likelihood differences |
| `--fragment-max-emission-diff` | numerical guardrail for emission-matrix differences |
| `--no-fragment-rescale-read-likelihood` | disable read-likelihood rescaling |

#### Outputs

| Parameter | Meaning |
| --- | --- |
| `--write-transitions` | write transition summaries |
| `--write-haplotype-probabilities` | write haplotype dosage and probability outputs |
| `--write-genotype-posteriors` | write genotype posterior outputs |
| `--write-genotype-calls` | write hard genotype calls |
| `--write-support-mask` | write direct-support mask |
| `--compression` | parquet compression codec |
| `--compression-level` | parquet compression level |
| `--no-profile-memory` | disable per-block memory profiling |

#### Calibration and Calling

| Parameter | Meaning |
| --- | --- |
| `--no-calibrate-genotype-posteriors` | skip STITCHV2 posterior calibration and keep raw HMM posteriors |
| `--calibration-mode` | `fixed` or `masked_cv` |
| `--genotype-posterior-temperature` | fixed calibration temperature |
| `--genotype-posterior-blend` | fixed calibration blend weight |
| `--genotype-call-mode` | `argmax`, `stitch_no_call`, or `quality_gated` |
| `--genotype-call-min-confidence` | minimum top posterior for `quality_gated` |
| `--genotype-call-min-margin` | minimum posterior gap for `quality_gated` |
| `--genotype-call-stitch-threshold` | no-call threshold for `stitch_no_call` |
| `--genotype-call-correctness-threshold` | learned correctness threshold for `quality_gated` |
| `--use-lightgbm-calibrator` | enable learned LightGBM calibrator |
| `--calibration-context-window` | local SNP window for calibration features |
| `--calibration-block-snps` | SNP block size used in calibration feature extraction |
| `--calibration-use-optuna` | tune learned calibration with Optuna |
| `--calibration-optuna-trials` | number of Optuna trials |
| `--calibration-max-train-rows` | maximum training rows for learned calibration |
| `--calibration-train-site-fraction` | fraction of labeled sites used for training |
| `--calibration-lightgbm-use-block-context` | include older local-context LightGBM stage |
| `--calibration-lightgbm-use-fixed-stage0` | apply fixed calibration before LightGBM |
| `--calibration-maf-bins` | comma-separated MAF bin edges |
| `--calibration-temperatures` | candidate temperatures for masked-CV |
| `--calibration-blends` | candidate blend weights for masked-CV |
| `--calibration-dosage-scales` | candidate dosage scales for masked-CV |
| `--calibration-dosage-offsets` | candidate dosage offsets for masked-CV |
| `--calibration-hwe-prior-weights` | candidate HWE prior weights |
| `--no-calibration-optimize-dosage-scale` | disable dosage scale and offset optimization |
| `--calibration-hwe-weight` | HWE prior or penalty weight |
| `--calibration-hwe-min-maf` | minimum MAF used by HWE calibration terms |

#### Microarray

| Parameter | Meaning |
| --- | --- |
| `--microarray-plink` | PLINK prefix used for hard microarray evidence |
| `--microarray-generation-default` | generation assigned to microarray-only added samples |
| `--microarray-hard-call-weight` | evidence weight used for hard calls |
| `--no-microarray-add-samples` | do not add PLINK-only samples missing from `samples` |

#### Pedigree

| Parameter | Meaning |
| --- | --- |
| `--pedigree` | pedigree table path |
| `--pedigree-mode` | `off`, `smooth`, `kinship`, or `transmission` |
| `--pedigree-strength` | pedigree smoothing or message strength |
| `--pedigree-offspring-col` | offspring or sample column name |
| `--pedigree-parent1-col` | first parent column name |
| `--pedigree-parent2-col` | second parent column name |
| `--pedigree-iterations` | message-passing or smoothing iterations |
| `--pedigree-kinship-threshold` | kinship pruning threshold |

#### Founders

| Parameter | Meaning |
| --- | --- |
| `--founder-vcf` | founder VCF path |
| `--founder-plink` | founder PLINK prefix |
| `--founder-immutable` | freeze founder states after loading |

### `stitchv2 cv`

`stitchv2 cv` inherits the main `run` parameter surface except that `--n-founders` is optional rather than required, and adds the following tuning parameters:

| Parameter | Meaning |
| --- | --- |
| `--pseudo-truth` | parquet or delimited pseudo-truth table with `sample_id`, `position`, and genotype or dosage truth |
| `--k-values` | comma-separated founder counts to test |
| `--ngen-values` | comma-separated generation values to test |
| `--s-values` | comma-separated extra tuning values used by the harness |
| `--seeds` | comma-separated random seeds |
| `--folds` | number of CV folds |
| `--holdout-fraction` | fraction held out per fold |
| `--lightgbm-post-calibrator` | add a post-calibration model in the harness |
| `--founder-vcf` | founder VCF for CV runs |
| `--founder-plink` | founder PLINK prefix for CV runs |
| `--founder-immutable` | freeze founders during CV runs |

Example:

```bash
stitchv2 cv \
  --samples samples.parquet \
  --positions positions.parquet \
  --pseudo-truth pseudo_truth.parquet \
  --chromosome chr1 \
  --output-dir runs/cv \
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

### `stitchv2 tune-jax-memory`

This command profiles block size, memmap mode, and JAX sample batching over a prepared dataset directory.

| Parameter | Meaning |
| --- | --- |
| `--data-dir` | directory containing `samples.parquet` and `positions.parquet` |
| `--output-dir` | directory for profiling runs and summary output |
| `--chromosome` | chromosome or contig to profile |
| `--n-founders` | founder count |
| `--em-iterations` | EM iterations |
| `--block-sizes` | comma-separated block sizes to test |
| `--memory-map-modes` | comma-separated `on` or `off` values |
| `--jax-sample-batch-sizes` | comma-separated JAX batch sizes |
| `--read-stream-backend` | `auto`, `python`, or `htslib` |
| `--io-workers` | read-extraction workers |
| `--htslib-threads-per-file` | HTS threads per input file |
| `--fragment-likelihood-mode` | `replace` or `augment` |
| `--memory-map-dir` | memmap directory |
| `--max-samples` | optional cap on samples used in the profiling set |
| `--max-positions` | optional cap on positions used in the profiling set |
| `--max-peak-rss-mb` | optional memory ceiling for choosing the best run |

Example:

```bash
stitchv2 tune-jax-memory \
  --data-dir benchmark_runs/synth_5mb_2k_0p1x \
  --output-dir runs/jax_tuning \
  --chromosome chrSynthetic \
  --n-founders 8 \
  --block-sizes 500,1000,2000 \
  --memory-map-modes off,on \
  --jax-sample-batch-sizes 0,64,128
```

### `stitchv2 combine`

Use this to combine partitioned parquet chunks after a run.

| Parameter | Meaning |
| --- | --- |
| `--run-output-dir` | run directory containing chunked outputs |
| `--datasets` | comma-separated dataset names or `all` |
| `--output-dir` | directory for combined outputs |
| `--input-dir` | explicit input parquet directory for one-off combining |
| `--output-file` | explicit output parquet file for one-off combining |
| `--compression` | parquet compression codec |
| `--compression-level` | parquet compression level |
| `--row-group-size` | parquet row group size |

Examples:

```bash
stitchv2 combine \
  --run-output-dir runs/production \
  --datasets all \
  --output-dir runs/production/combined
```

```bash
stitchv2 combine \
  --run-output-dir runs/production \
  --input-dir runs/production/genotype_calls \
  --output-file runs/production/genotype_calls_combined.parquet
```

### `stitchv2 export-vcf`

Exports STITCHV2 parquet outputs into a STITCH-like VCF representation.

| Parameter | Meaning |
| --- | --- |
| `--run-output-dir` | STITCHV2 run directory |
| `--output-vcf` | path to the output VCF or VCF.GZ |
| `--chromosome` | chromosome or contig label to write |
| `--include-haplotype-dosage` | include haplotype dosage fields |
| `--threads` | compression and export threads |
| `--gp-precision` | decimal precision for GP |
| `--ds-precision` | decimal precision for DS |
| `--hd-precision` | decimal precision for haplotype dosage |
| `--no-bgzip` | disable bgzip compression |
| `--no-tabix-index` | do not write a tabix index |

Example:

```bash
stitchv2 export-vcf \
  --run-output-dir runs/production \
  --output-vcf runs/production/production.vcf.gz \
  --chromosome chr1 \
  --threads 4
```

### `stitchv2 export-bcf`

Exports STITCHV2 parquet outputs to BCF.

| Parameter | Meaning |
| --- | --- |
| `--run-output-dir` | STITCHV2 run directory |
| `--output-bcf` | path to the output BCF |
| `--chromosome` | chromosome or contig label to write |
| `--include-haplotype-dosage` | include haplotype dosage fields |
| `--no-tabix-index` | do not write a tabix index |

Example:

```bash
stitchv2 export-bcf \
  --run-output-dir runs/production \
  --output-bcf runs/production/production.bcf \
  --chromosome chr1
```

### `stitchv2 reformat-stitch-filenames`

Reformats STITCH-oriented filenames into STITCHV2 naming and records the mapping.

| Parameter | Meaning |
| --- | --- |
| `--input-file` | text file with one filename per line |
| `--filenames` | comma-separated filename list |
| `--output-file` | JSON path for the output mapping |

Example:

```bash
stitchv2 reformat-stitch-filenames \
  --input-file stitch_files.txt \
  --output-file stitch_filename_map.json
```

## Choosing Parameters

The parameters that usually matter first are:

1. `--n-founders`: model flexibility versus state size
2. `--em-iterations`: convergence versus runtime
3. `--block-size`: throughput versus memory
4. `--jax-sample-batch-size`: device memory versus call overhead
5. `--fragment-likelihood-mode` and `--fragment-coupling-model`: keep `stitch_parity` for STITCH comparisons
6. calibration and call settings: decide whether the goal is raw parity, calibrated GP quality, or conservative hard calls

Rules of thumb:

- use `hmm_backend=jax` for large serious runs
- start with `block-size=1000` unless memory or tiny-region behavior suggests otherwise
- use `em-iterations=5` for parity and benchmark runs, `1-2` only for smoke tests
- leave calibration on by default unless the purpose is raw model comparison
- use `genotype-call-mode=stitch_no_call` when you want STITCH-like no-call behavior

## Practical Recipes

### Fast Smoke Test on a Region Slice

Use `--chr-start`, `--chr-end`, a small sample subset, and maybe `--block-size 250`.

### Strict Raw-Model Parity Check

Use:

- supplied founders
- `--founder-immutable`
- `--fragment-coupling-model stitch_parity`
- `--genotype-call-mode stitch_no_call`
- `--genotype-call-stitch-threshold 0.9`
- `--no-calibrate-genotype-posteriors`

### Production Low-Coverage Run

Use:

- `--hmm-backend jax`
- `--read-stream-backend auto` or `htslib`
- calibration on
- `--write-genotype-posteriors`
- `--write-genotype-calls`
- `--write-support-mask`

### Memory-Controlled Chromosome Run

Use:

- `--executor dask`
- coarse blocks
- `--dask-target-task-memory-mb`
- `--dask-performance-report`
- `stitchv2 tune-jax-memory` before full scale-up

## Troubleshooting

| Symptom | Likely cause | What to check |
| --- | --- | --- |
| No reads overlap variants | contig mismatch or wrong coordinates | compare BAM headers, `--chromosome`, and `positions.CHR` |
| Dask is slower than serial | tasks are too small or the job is too small | increase block size or sample batch size; reserve Dask for larger jobs |
| JAX runs out of memory | too many samples, SNPs, or founders per task | lower `--block-size` or `--jax-sample-batch-size`; consider Dask |
| STITCHV2 differs from STITCH | founder behavior, calibration, call policy, or read coupling differs | use supplied immutable founders, `stitch_parity`, and comparable call settings |
| Sex chromosome missingness looks wrong | `sex` values are missing or inconsistent | normalize `sex` labels and set male and female ploidies explicitly |
| PLINK samples are not injected | PLINK IID does not match `sample_id` | inspect `.fam` IID values |
| PLINK variants are not injected | chromosome or position mismatch | inspect `.bim` and align `CHR` and `POS` conventions |
| Exported PLINK loses dosage | PLINK1 BED is hard-call only | use VCF/BCF or PLINK2 PGEN workflows for dosage |

## One Last Sanity Check Before Large Runs

Before launching a full benchmark or production job, confirm:

- the `samples` table has the right `generation` values
- contig names match across BAMs, positions, and founders
- founders are either intentionally omitted or intentionally frozen
- parity runs use `fragment_coupling_model=stitch_parity`
- the calibration choice is explicit
- the chosen outputs include the metrics or exports you will need later
