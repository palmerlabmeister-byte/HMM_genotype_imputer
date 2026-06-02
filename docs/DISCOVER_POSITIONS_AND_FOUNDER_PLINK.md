# STITCHV2 Discover Positions and Founder PLINK Guide

This guide explains how to use `stitchv2 discover-positions`, how to turn the discovered candidate table into a reviewed STITCHV2 positions table, and what is required to create the founder `.bed/.bim/.fam` files used by `--founder-plink`.

The most important point:

```text
discover-positions creates candidate variant positions and read-support counts.
It does not create founder genotypes.
```

A founder PLINK prefix requires a genotype matrix for the founder samples. The discovered Parquet table can tell you which positions to genotype, but it cannot be converted directly into a valid founder `.bed/.bim/.fam` unless you also have founder genotypes or dosages at those positions.

## What `discover-positions` Does

`stitchv2 discover-positions` is a pre-imputation scan over BAM/CRAM reads. It:

1. reads indexed BAM/CRAM files from a samples table,
2. uses an indexed reference FASTA,
3. scans CIGAR alignments over a chromosome/window,
4. emits reviewable candidate biallelic variants,
5. writes a Parquet/CSV/TSV positions-like table plus a JSON summary.

It does not run the HMM, does not update founders, and does not automatically add variants to an imputation run.

Supported discovery types:

| type | status |
| --- | --- |
| SNP | Native HTSlib-backed discovery. |
| small insertion | CIGAR-aware discovery, default max length 50 bp. |
| small deletion | CIGAR-aware discovery, default max length 50 bp. |

Complex/nested records are intentionally out of scope for now:

- multiallelic sites,
- symbolic alleles,
- complex substitutions,
- SNP inside insertion,
- overlapping biallelic records at the same `CHR:POS`.

## Prerequisites

Install STITCHV2 and build the native extension:

```bash
cd /path/to/STITCHV2
pip install -e ".[plink,ml,plot,dev]"
python setup.py build_ext --inplace
stitchv2 discover-positions --help
```

Reference FASTA must be indexed:

```bash
samtools faidx /path/to/reference.fa
```

BAM/CRAM files must be indexed:

```bash
samtools index sample.bam
```

The samples table needs a BAM/CRAM path column, default `bam_path`:

| column | meaning |
| --- | --- |
| `sample_id` | Sample ID. |
| `bam_path` | Indexed BAM/CRAM path. |
| `generation` | Required by STITCHV2 runs; not used by discovery itself but should stay in the table. |

## Basic Discovery Command

Example for all candidate SNPs, insertions, and deletions over a 10 Mb window:

```bash
stitchv2 discover-positions \
  --samples samples.parquet \
  --reference-fasta rn8.fa \
  --chromosome chr12 \
  --chr-start 9000000 \
  --chr-end 19000000 \
  --output-file discovered_chr12_9_19Mb.parquet \
  --summary-file discovered_chr12_9_19Mb.summary.json \
  --bam-path-col bam_path \
  --variant-types snp,ins,del \
  --max-indel-len 50 \
  --window-size 1000000 \
  --min-base-quality 17 \
  --min-mapping-quality 17 \
  --cap-base-quality-by-mapping-quality \
  --max-insert-size 600 \
  --min-depth 3 \
  --min-alt-count 2 \
  --min-alt-samples 1 \
  --min-alt-fraction 0.05 \
  --max-other-fraction 0.20 \
  --compression zstd
```

For a smoke test:

```bash
stitchv2 discover-positions \
  --samples samples.parquet \
  --reference-fasta rn8.fa \
  --chromosome chr12 \
  --chr-start 9000000 \
  --chr-end 9100000 \
  --max-bams 10 \
  --output-file smoke_discovered.parquet
```

SNP-only discovery:

```bash
--variant-types snp
```

Indel-only discovery:

```bash
--variant-types ins,del
```

## Discovery Output Columns

The output table contains candidate variants and support summaries.

| column | meaning |
| --- | --- |
| `CHR` | Chromosome/contig. |
| `POS` | 1-based position. For insertions/deletions this is the left anchor. |
| `REF` | Reference allele. |
| `ALT` | Alternate allele. |
| `variant_type` | `snp`, `insertion`, or `deletion`. |
| `depth` | Aggregate depth/support count used by discovery. |
| `ref_count` | Aggregate REF observation count when available. |
| `alt_count` | Aggregate ALT observation count. |
| `other_count` | Aggregate non-REF/non-ALT observation count. |
| `alt_fraction` | `alt_count / depth`. |
| `other_fraction` | `other_count / depth`. |
| `sample_support` | Number of samples with ALT support. |
| `a_count`, `c_count`, `g_count`, `t_count` | Base counts for SNP discovery. |
| `alt_forward_count`, `alt_reverse_count` | Strand support for ALT observations. |

The summary JSON records:

- chromosome/window,
- reference FASTA,
- number of scanned BAM/CRAM files,
- discovery thresholds,
- requested variant types,
- max indel length,
- per-window candidate counts.

## Review and Filter Before Imputation

Do not feed raw discovery output directly into a production imputation run. First create a reviewed positions table.

Recommended filters depend on coverage and study design, but typical filters include:

- minimum `sample_support`,
- minimum `alt_count`,
- minimum `alt_fraction`,
- maximum `other_fraction`,
- strand-balance checks for candidates with enough reads,
- removal of low-complexity or problematic regions,
- removal of duplicate `CHR, POS` records,
- removal or splitting of multiallelic candidates.

STITCHV2 normal imputation currently requires one biallelic variant per `CHR, POS`.

Example filtering scaffold:

```python
import pandas as pd

raw = pd.read_parquet("discovered_chr12_9_19Mb.parquet")

positions = raw.loc[
    (raw["sample_support"] >= 2)
    & (raw["alt_count"] >= 3)
    & (raw["alt_fraction"] >= 0.05)
    & (raw["other_fraction"] <= 0.20)
].copy()

positions["VARIANT_TYPE"] = positions["variant_type"].replace(
    {"ins": "insertion", "del": "deletion"}
)

positions = positions.rename(columns={"variant_type": "variant_type_original"})
positions = positions[["CHR", "POS", "REF", "ALT", "VARIANT_TYPE"]]
positions = positions.sort_values(["CHR", "POS", "REF", "ALT"])
positions = positions.drop_duplicates(["CHR", "POS"], keep=False)

positions.to_parquet("positions_chr12_discovered_reviewed.parquet", index=False)
```

If you are using external normalization tools, keep normalized left-anchored biallelic records. For small insertions/deletions, STITCHV2 expects:

```text
insertion: REF=A,   ALT=ATG
deletion:  REF=AAA, ALT=A
```

## Running STITCHV2 On Reviewed SNP/Indel Positions

Use the reviewed table as `--positions`.

For SNP-only positions, any compiled read stream backend can work. For positions containing insertions/deletions, use the variant-aware reader:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions_chr12_discovered_reviewed.parquet \
  --chromosome chr12 \
  --output-dir runs/chr12_discovered_positions \
  --founder-plink founders_discovered \
  --founder-immutable \
  --n-founders 8 \
  --read-mode read_stream \
  --read-stream-backend variant_aware_bamreader \
  --max-indel-len 50 \
  --hmm-backend jax \
  --stitch-compat \
  --fragment-coupling-model stitch_parity \
  --fragment-likelihood-mode replace \
  --write-genotype-calls \
  --write-genotype-posteriors
```

Important:

- `read_mode='pileup'` is SNP-only for indels.
- `variant_aware_bamreader` is required for targeted insertion/deletion evidence.
- If the compiled extension is available, `--read-stream-backend auto` also selects the variant-aware path when needed.

## Why The Discovery Parquet Is Not A Founder PLINK File

A founder PLINK prefix contains:

```text
founders.bed  # packed genotype matrix
founders.bim  # variant metadata
founders.fam  # founder sample metadata
```

The discovery Parquet contains:

```text
CHR, POS, REF, ALT, variant_type, depth, alt_count, sample_support, ...
```

It has aggregate read support across sequenced samples, not one genotype per founder. Therefore:

```text
discovered.parquet -> founders.bed/.bim/.fam
```

is not a valid direct conversion.

You need one of the following genotype sources for the founders:

1. Existing founder PLINK/VCF genotypes that can be subset to the discovered positions.
2. Founder BAM/CRAM files that can be genotyped at the discovered positions.
3. A long founder genotype/dosage Parquet table with one row per founder and position.

## Best Founder Input Options

### Option A: Existing Founder VCF

If you already have founder genotypes in VCF, the simplest path is often to subset the VCF to the reviewed positions and use it directly:

```bash
bcftools view \
  -R positions_chr12_discovered_reviewed.tsv \
  -Oz \
  -o founders_discovered.vcf.gz \
  founders_all_sites.vcf.gz

bcftools index -t founders_discovered.vcf.gz
```

Then run:

```bash
--founder-vcf founders_discovered.vcf.gz
--founder-immutable
```

This avoids unnecessary PLINK conversion, and VCF is often a better carrier for indels than PLINK1 BED.

### Option B: Existing Founder PLINK

If you already have founder genotypes in PLINK format, subset them to discovered positions using PLINK/PLINK2.

Example position list for PLINK workflows:

```python
import pandas as pd

pos = pd.read_parquet("positions_chr12_discovered_reviewed.parquet")
pos["ID"] = (
    pos["CHR"].astype(str)
    + ":"
    + pos["POS"].astype(str)
    + ":"
    + pos["REF"].astype(str)
    + ":"
    + pos["ALT"].astype(str)
)
pos[["ID"]].to_csv("discovered_variant_ids.txt", index=False, header=False)
```

Then, if the existing PLINK variant IDs match:

```bash
plink2 \
  --bfile founders_all_sites \
  --extract discovered_variant_ids.txt \
  --make-bed \
  --out founders_discovered
```

If IDs do not match, create a coordinate-based extraction workflow with PLINK2 or convert through VCF. For STITCHV2 itself, founder PLINK matching is primarily by chromosome and position; however, keeping BIM IDs/alleles correct is safer for inspection and downstream tools.

Run with:

```bash
--founder-plink founders_discovered
--founder-immutable
```

### Option C: Genotype Founder BAMs At Discovered Sites

If you only discovered positions from population BAMs and the founders have BAM/CRAM files, genotype the founders at the reviewed sites with an external genotyper.

Typical choices:

- `bcftools mpileup` + `bcftools call`,
- GATK HaplotypeCaller/GenotypeGVCFs,
- DeepVariant,
- another validated targeted genotyper.

Then either:

- use the founder VCF directly with `--founder-vcf`, or
- convert the founder VCF to PLINK with PLINK2.

This is the correct path when discovered sites were not already present in the founder PLINK/VCF.

### Option D: Long Founder Dosage Parquet

If you already have founder hard calls or dosages in a long Parquet table, `npplink` can help write a PLINK1 BED/BIM/FAM prefix.

Required long table shape:

| column | meaning |
| --- | --- |
| founder/sample ID | Founder sample ID. |
| chromosome | Chromosome label. |
| position | Position. |
| dosage or genotype | Numeric 0/1/2 value. Missing allowed. |

Example:

```text
sample_id  chromosome  position  dosage
F0         chr12       1000      0
F1         chr12       1000      1
F2         chr12       1000      2
```

Use the helper:

```python
from stitchv2.npplink import parquet_dosage_to_plink

parquet_dosage_to_plink(
    dosage_parquet="founder_discovered_dosage_long.parquet",
    output_prefix="founders_discovered_tmp",
    sample_col="sample_id",
    chrom_col="chromosome",
    pos_col="position",
    dosage_col="dosage",
)
```

This writes:

```text
founders_discovered_tmp.bed
founders_discovered_tmp.bim
founders_discovered_tmp.fam
```

The helper rounds dosage to hard 0/1/2 PLINK1 genotypes. It is useful for hard founder panels, but it does not preserve dosage uncertainty.

Current caveat: `parquet_dosage_to_plink` writes default BIM alleles unless the writer is extended. STITCHV2 founder loading uses the run `positions.parquet` for `REF/ALT` and matches PLINK founder genotypes by chromosome/position, so the run can still work. For production portability, patch the `.bim` alleles from your reviewed positions table:

```python
import pandas as pd

positions = pd.read_parquet("positions_chr12_discovered_reviewed.parquet")
positions = positions[["CHR", "POS", "REF", "ALT"]].copy()
positions["chrom_for_bim"] = positions["CHR"].astype(str).str.replace("^chr", "", regex=True)

bim = pd.read_csv(
    "founders_discovered_tmp.bim",
    sep=r"\s+",
    header=None,
    names=["chrom", "snp", "cm", "pos", "a0", "a1"],
)
bim["chrom"] = bim["chrom"].astype(str)

merged = bim.merge(
    positions,
    left_on=["chrom", "pos"],
    right_on=["chrom_for_bim", "POS"],
    how="left",
    validate="one_to_one",
)

if merged["REF"].isna().any() or merged["ALT"].isna().any():
    missing = merged.loc[merged["REF"].isna() | merged["ALT"].isna(), ["chrom", "pos"]]
    raise ValueError(f"Missing REF/ALT for BIM rows: {missing.head().to_dict('records')}")

out = pd.DataFrame(
    {
        0: merged["chrom"],
        1: (
            merged["chrom"].astype(str)
            + ":"
            + merged["pos"].astype(str)
            + ":"
            + merged["REF"].astype(str)
            + ":"
            + merged["ALT"].astype(str)
        ),
        2: merged["cm"],
        3: merged["pos"],
        4: merged["REF"],
        5: merged["ALT"],
    }
)
out.to_csv("founders_discovered_tmp.bim", sep="\t", header=False, index=False)
```

Then use:

```bash
--founder-plink founders_discovered_tmp
--founder-immutable
```

## Can `npplink` Help?

Yes, but with the right expectation.

Useful `npplink` functions:

| function | use |
| --- | --- |
| `read_fam_bim(prefix)` | Inspect existing PLINK `.fam` and `.bim`. |
| `load_plink(prefix)` | Lazily load PLINK genotypes. |
| `load_plink_xarray(prefix)` | Load PLINK as an xarray object. |
| `parquet_dosage_to_plink(...)` | Write PLINK1 BED/BIM/FAM from a long dosage/genotype Parquet table. |

What `npplink` cannot do by itself:

```text
discovered positions + aggregate read counts -> founder genotypes
```

It needs founder-level genotype/dosage values. Once you have those, it can help write a PLINK prefix.

## Validating A New Founder PLINK Prefix

First inspect the PLINK metadata:

```python
from stitchv2.npplink import read_fam_bim

fam, bim = read_fam_bim("founders_discovered")
print(fam.head())
print(bim.head())
print(fam.shape, bim.shape)
```

Check:

- founder sample IDs are correct,
- chromosome labels match the intended run,
- positions overlap the reviewed STITCHV2 positions table,
- REF/ALT are sensible in `.bim`,
- no unexpected duplicate positions.

Then run a small STITCHV2 smoke test:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions_chr12_discovered_reviewed.parquet \
  --chromosome chr12 \
  --chr-start 9000000 \
  --chr-end 9100000 \
  --output-dir runs/smoke_discovered_founders \
  --founder-plink founders_discovered \
  --founder-immutable \
  --n-founders 8 \
  --read-mode read_stream \
  --read-stream-backend variant_aware_bamreader \
  --max-indel-len 50 \
  --hmm-backend jax \
  --stitch-compat \
  --write-genotype-calls \
  --write-genotype-posteriors \
  --diagnostics-fail-on-error
```

Review:

```bash
cat runs/smoke_discovered_founders/founder_expansion_summary.json
cat runs/smoke_discovered_founders/diagnostics_summary.json
```

Also inspect `founders.parquet` from the run. If many known founder positions are still `0.5`, the PLINK founder file probably did not match those positions.

## Recommended Production Workflow

The safest workflow is:

1. Run `stitchv2 discover-positions`.
2. Review/filter candidate variants into `positions_discovered_reviewed.parquet`.
3. Genotype or subset founders at those reviewed positions.
4. Prefer `--founder-vcf` for mixed SNP/indel founder panels when possible.
5. Use `--founder-plink` only when you have a validated BED/BIM/FAM founder genotype panel.
6. Smoke-test one small interval.
7. Run the full STITCHV2 chromosome with `variant_aware_bamreader` for indels.

For immutable founder runs, every production target site should have a valid founder genotype. If a discovered site is absent from the immutable founder panel, STITCHV2 initializes that founder ALT probability as uncertain, which is not strict fixed-founder behavior. For those sites, either provide founder genotypes, use mutable founders, or exclude the site from the fixed-founder production table.
