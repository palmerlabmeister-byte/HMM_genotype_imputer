# STITCHV2 Model

Status: retained as a legacy model note. The current canonical model reference is [docs/deep_dive/01_model_deep_dive.md](../deep_dive/01_model_deep_dive.md).

## What STITCHV2 is

`stitchv2` is a Python implementation of a STITCH-style low-coverage imputation pipeline, with a JAX-first HMM core and block-wise IO for large cohorts. The goal is to keep the read-aware behavior of STITCH while improving scalability, GPU usability, and interoperability with modern tabular formats (Arrow/Parquet).

Reference paper:
- Davies et al. (STITCH): https://pmc.ncbi.nlm.nih.gov/articles/PMC4966640/

Reference implementation:
- STITCH GitHub: https://github.com/rwdavies/STITCH

---

## Original STITCH model (paper-level summary)

At a high level, STITCH models each sample as a mosaic of `K` ancestral/founder haplotypes along the chromosome.

1. Hidden state
- For diploid samples, the latent state per variant is an ancestral haplotype pair.
- Across neighboring variants, state changes follow recombination-driven transitions (Li-Stephens-style intuition).

2. Read-aware emissions
- Emissions are based on aligned sequencing reads, not only pileup dosages.
- Multi-SNP fragments/read pairs contribute joint evidence across nearby variants.

3. EM updates
- E-step: infer posterior over hidden states/haplotypes given reads and current founder parameters.
- M-step: update founder allele probabilities and transition-related parameters.
- Repeat for multiple EM iterations.

4. Outputs
- Dosages, genotype probabilities/calls, and optional haplotype-level quantities.

---

## How STITCHV2 implements this

## 1) Input and block orchestration

- Position table (`CHR`, `POS`, `REF`, `ALT`) is loaded and split into blocks.
- Sample table contains at least `sample_id`, `bam_path`, `generation`.
- Pipeline processes one block at a time and writes outputs immediately to Parquet chunks.

Main implementation path:
- `src/stitchv2/pipeline.py`
- `src/stitchv2/config.py`

## 2) Read extraction (read-aware path)

STITCHV2 supports:
- `read_mode="read_stream"` (default, read-aware)
- `read_mode="pileup"` (fallback/diagnostic)

Read stream backends:
- `read_stream_backend="python"` (pysam path)
- `read_stream_backend="htslib"` (compiled extension, if available)
- `read_stream_backend="auto"`

Per block, extraction returns:
- per-sample x per-variant counts/weights (`ref_count`, `alt_count`, `other_count`, `depth`, weighted versions)
- compact fragment arrays (`fragment_*`) for multi-SNP read coupling

Main implementation path:
- `src/stitchv2/pileup.py` (`PysamReadExtractor`, `ReadEvidenceBlock`)

## 3) Founder modeling

Founder haplotypes can be loaded from:
- VCF
- PLINK (through `npplink`)
- BAM-derived workflows (if configured)

Founders support:
- Immutable founders: fixed/known genotype states
- Mutable founders: probabilistic alt-allele representation (`alt_prob`) with EM updates

Main implementation path:
- `src/stitchv2/founders.py`
- `src/stitchv2/npplink.py`

## 4) HMM backends and read coupling

The HMM engine is in:
- `src/stitchv2/hmm.py` (`JAXStitchHMM`)

Backends:
- `numpy`
- `jax`
- `torch`
- `auto` (autotune path exists)

Read coupling modes:
- `fragment_coupling_model="legacy_center"`
- `fragment_coupling_model="stitch_parity"` for closer STITCH-like centered fragment/read behavior

JAX options include:
- precompile/AOT controls
- sample batching for memory control

## 5) Pedigree-aware smoothing (optional)

After block inference, dosage can be smoothed with pedigree priors/regularization:
- optional pedigree input (CSR-like relation structure)
- configurable strength

Main implementation path:
- `src/stitchv2/pedigree.py`
- called from `pipeline.py` via `smooth_dosage_with_pedigree(...)`

## 6) Calibration and hard calls

STITCHV2 separates:
- posterior calibration
- hard call policy (`argmax`, `stitch_no_call`, `quality_gated`)

Optional calibration stack includes block-context and (optional) LightGBM-based modules.

Main implementation path:
- `src/stitchv2/calibration.py`

---

## Why this implementation is intended to be better

Compared to classic monolithic runs, STITCHV2 is designed for:

1. Better scale behavior
- block-wise processing
- chunked Parquet writes
- optional memory-mapped read matrices

2. Better hardware usage
- JAX backend for CPU/GPU
- optional AOT/JIT-shape stabilization

3. Better data engineering integration
- Arrow/Parquet chunk outputs
- lazy combine into xarray+dask
- optional xarray/Zarr stores for floating outputs
- explicit BCF interoperability export when external tools require it

4. Better workflow flexibility
- mixed sample generations
- optional microarray hard-genotype integration
- optional pedigree-aware post-HMM smoothing
- explicit support-mask output (`has_supporting_read`) to distinguish inferred-only vs read-supported genotype locations

---

## Parity and caveats

- STITCHV2 aims for read-aware parity with STITCH behavior, but it is not a line-by-line port of upstream STITCH C++/R internals.
- In practice, parity depends on matching settings (`K`, generations, fragment coupling settings, calibration policy, and data preprocessing).
- Accuracy/runtime outcomes should always be validated by benchmark scripts and reports in this repository.
