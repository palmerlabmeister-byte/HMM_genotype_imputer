# STITCHV2 Whole-Genome Run With Pedigree QC

This is a practical runbook for running STITCHV2 across a whole genome, using compact evidence caches efficiently, curating the pedigree from first-pass genotypes, and rerunning imputation with the curated pedigree.

The intended workflow is:

1. Run every chromosome once without pedigree.
2. Reuse/write compact read-evidence caches during that first pass.
3. Run global pedigree QC across all autosomes or all high-quality chromosomes.
4. Review the pedigree UMAP and edge QC outputs.
5. Rerun every chromosome with `pedigree_curated.parquet`.

Do not run pedigree QC independently per chromosome unless you are doing a smoke test. The robust workflow is chromosome-level imputation first, global pedigree QC second, then chromosome-level imputation again with the curated pedigree.

## 1. Environment

Use one conda environment for the full run. The command examples assume `stitchv2` is installed with `pip install -e .` inside that environment.

```bash
conda activate stitchv2-benchmark
which stitchv2
stitchv2 --help
stitchv2 run --help
stitchv2 pedigree-qc --help
python - <<'PY'
import jax, lightgbm, pandas, pyarrow, dask
print("jax", jax.__version__)
print("lightgbm", lightgbm.__version__)
print("pandas", pandas.__version__)
print("pyarrow", pyarrow.__version__)
print("dask", dask.__version__)
PY
```

Recommended local defaults:

| setting | recommended value | reason |
| --- | --- | --- |
| `--hmm-backend` | `jax` | compiled HMM backend |
| `--executor` | `dask` for large chromosomes, `serial` for smoke tests | Dask schedules chromosome/block work |
| `--dask-processes` | omit it | local threads avoid serialization overhead |
| `--dask-dashboard-address` | unique port per simultaneous job | avoids dashboard collisions |
| `--max-mem` | `90%` or a fixed value like `64G` | controls automatic block/window sizing |
| `--compact-evidence-cache-format` | `parquet_zarr` | compressed, chunked evidence cache |
| `--compact-evidence-cache-mode` | first pass `readwrite`, reruns `read` | fill once, reuse later |

Read backend guidance:

- Use `--read-stream-backend auto` for normal production runs.
- Use `--read-stream-backend variant_aware_bamreader` explicitly when the target table includes insertions or deletions.
- Use `--read-stream-backend stitch_style_bamreader` for strict SNP-only STITCH compatibility checks.

## 2. Required Inputs

### 2.1 `samples.parquet`

Required columns:

| column | type/value | meaning |
| --- | --- | --- |
| `sample_id` | string, unique | STITCHV2 sample key used in all outputs |
| `bam_path` | string path | BAM/CRAM path for the sample |
| `generation` | numeric | generation or `nGen`-like recombination scale input |

Strongly recommended columns:

| column | type/value | meaning |
| --- | --- | --- |
| `sex` | `M`, `F`, `male`, `female`, `XY`, `XX`, etc. | sex-specific ploidy for X/Y/MT |
| `rfid` or external ID | string | useful for matching array truth or colony records |
| `father_id` | sample ID or blank | declared first parent |
| `mother_id` | sample ID or blank | declared second parent |
| `family_id` | string | optional family/group label for QC plots |

Missing parents can be blank, `NA`, `NaN`, `nan`, `n/a`, `N/A`, or `0`.

Example:

```text
sample_id                         bam_path                         generation  sex  father_id  mother_id  family_id
Riptide299_Outbred1290_GACATCTG   /data/bams/sample1290.bam        80          M    sire42     dam17      famA
Riptide299_Outbred1300_CAGTTGAC   /data/bams/sample1300.bam        80          M                          famB
```

### 2.2 `positions.parquet`

Required columns:

| column | type/value | meaning |
| --- | --- | --- |
| `CHR` | string | chromosome/contig label as used by STITCHV2 |
| `POS` | int | 1-based genomic coordinate |
| `REF` | string | reference allele |
| `ALT` | string | alternate allele |

For founder PLINK files that use numeric chromosomes while BAMs use `chr12`, keep the run command chromosome as the BAM label, for example `--chromosome chr12`, and make sure the founder loader can match the PLINK chromosome label. The HSRats benchmark uses founder chromosome `12` and BAM chromosome `chr12`.

### 2.3 Founder Panel

Preferred for fixed-founder parity:

```text
founders8.bed
founders8.bim
founders8.fam
```

Run with:

```bash
--founder-plink /path/to/founders8 \
--founder-immutable
```

If `--n-founders` is larger than the number of founder samples in the founder file, STITCHV2 keeps the supplied founders immutable and adds extra mutable founders. Example: 8 fixed founders plus 1 mutable founder:

```bash
--founder-plink /path/to/founders8 \
--founder-immutable \
--n-founders 9 \
--em-iterations 40
```

Use 40 EM iterations when mutable founders are enabled unless a smaller validated setting has converged. Adaptive EM can stop earlier and restores the best founder panel.

### 2.4 Pedigree Table

You can either keep parent columns in `samples.parquet` or provide a separate pedigree file:

```text
sample_id  father_id  mother_id  family_id
child1     sire1      dam1       famA
child2     sire2      dam2       famA
```

The curated output from `stitchv2 pedigree-qc` is directly consumable by `stitchv2 run` because it writes canonical columns:

| column | meaning |
| --- | --- |
| `sample_id` | child/sample ID |
| `father_id` | curated first parent, blank if removed |
| `mother_id` | curated second parent, blank if removed |
| `family_id` | family/group ID |
| `original_parent1_id` | declared first parent before QC |
| `original_parent2_id` | declared second parent before QC |
| `pedigree_qc_removed_parent_count` | number of removed parent edges |
| `pedigree_qc_status` | `unchanged` or `corrected` |

## 3. Chromosome Plan

Make one run directory per chromosome. Use chromosome-specific ploidy:

| chromosome class | recommended ploidy flags |
| --- | --- |
| autosome | `--ploidy 2` |
| X in mixed-sex population | `--ploidy 2 --ploidy-males 1 --ploidy-females 2` |
| Y in mixed-sex population | `--ploidy 0 --ploidy-males 1 --ploidy-females 0` |
| MT | choose the validated haploid or pseudo-diploid model for the dataset, commonly `--ploidy 1` |
| samples/chromosomes that should be absent | ploidy `0` |

Ploidy-zero samples are excluded from HMM computation. They remain in output files with missing genotypes and zero probability haplotypes.

## 4. Compact Evidence Cache

The compact evidence cache decouples BAM reading from HMM blocks.

Recommended first pass:

```bash
--compact-evidence-cache-dir /project/stitchv2_cache/chr12 \
--compact-evidence-cache-mode readwrite \
--compact-evidence-cache-format parquet_zarr \
--compact-evidence-cache-sample-batch-size 256
```

Recommended reruns:

```bash
--compact-evidence-cache-dir /project/stitchv2_cache/chr12 \
--compact-evidence-cache-mode read \
--compact-evidence-cache-format parquet_zarr
```

When adding new samples, use `readwrite`. STITCHV2 checks the cache by sample/chromosome/block. Cached samples are reused, missing samples are read from BAM and added. The stage timings report `compact_cache_cached_samples`, `compact_cache_missing_samples`, `compact_cache_hit`, and `compact_cache_partial_hit`.

Current cache layout:

```text
cache_root/
  chrom=chr12/
    block=000000_rows=0-1491/
      manifest.json
      positions.parquet
      fragments/
        part-*.parquet
      summary/
        part-*.parquet
      support/
        part-*.parquet
      dense/
        part-*.zarr/
```

Important cache fields reported in `stage_timings.json`:

| field | meaning |
| --- | --- |
| `partitioned_cache_format` | cache format, normally `parquet_zarr` |
| `partitioned_cache_path` | exact cache partition used by the block |
| `partitioned_cache_file_bytes` | total compressed cache bytes on disk for that partition |
| `partitioned_cache_parquet_bytes` | compressed parquet bytes |
| `partitioned_cache_zarr_bytes` | zarr bytes, if dense arrays were cached |
| `partitioned_cache_manifest_samples` | samples listed in the cache manifest |
| `partitioned_cache_manifest_parts` | number of sample-batch parts |
| `partitioned_cache_cached_samples` | samples loaded from cache |
| `partitioned_cache_missing_samples` | samples not found in cache |
| `compact_evidence_bytes` | compact evidence memory footprint after load |
| `dense_evidence_bytes` | dense matrix memory footprint if materialized |

Keep `--compact-evidence-cache-include-dense-counts` off for strict parity or cache-size-minimal runs. Enable it when repeated calibration, dense diagnostics, or augment-mode comparisons need the same dense count features many times. Dense counts are stored as Zarr and can be much larger than compact fragment evidence.

## 5. First Pass: Whole Genome Without Pedigree

Use this pass to create preliminary genotype calls for pedigree QC. This pass should be conservative and comparable across chromosomes.

Example autosome command:

```bash
stitchv2 run \
  --samples /project/inputs/samples.parquet \
  --positions /project/inputs/positions_chr12.parquet \
  --chromosome chr12 \
  --output-dir /project/runs/pass1_no_pedigree/chr12 \
  --founder-plink /project/inputs/founders8 \
  --founder-immutable \
  --n-founders 9 \
  --em-iterations 40 \
  --max-mem 90% \
  --hmm-backend jax \
  --jax-persistent-cache-dir /project/runs/jax_cache \
  --read-mode read_stream \
  --read-stream-backend auto \
  --snp-block-mode exact_streaming \
  --store-xi per-snp \
  --stitch-compat \
  --fragment-likelihood-mode replace \
  --compact-evidence-cache-dir /project/cache/chr12 \
  --compact-evidence-cache-mode readwrite \
  --compact-evidence-cache-format parquet_zarr \
  --genotype-call-mode stitch_no_call \
  --genotype-call-stitch-threshold 0.9 \
  --no-calibrate-genotype-posteriors \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-haplotype-probabilities \
  --write-support-mask \
  --executor dask \
  --dask-scheduler local \
  --dask-threads-per-worker 1 \
  --dask-dashboard-address :8787 \
  --dask-performance-report /project/runs/pass1_no_pedigree/chr12/dask_performance_report.html \
  --dask-task-stream /project/runs/pass1_no_pedigree/chr12/dask_task_stream.html
```

For a rerun of the same chromosome and samples, switch only:

```bash
--compact-evidence-cache-mode read
```

For X:

```bash
--ploidy 2 --ploidy-males 1 --ploidy-females 2
```

For Y:

```bash
--ploidy 0 --ploidy-males 1 --ploidy-females 0
```

## 6. First-Pass Outputs

Each chromosome run writes:

```text
chr12/
  samples.parquet
  positions.parquet
  stage_timings.json
  memory_profile_summary.json
  diagnostics_summary.json
  dosage/block=000000.parquet
  genotype_calls/block=000000.parquet
  genotype_posteriors/block=000000.parquet
  haplotype_probabilities/block=000000.parquet
  support_mask/block=000000.parquet
  diagnostics/block=000000.parquet
  recombination/block=000000.parquet
  transitions/block=000000.parquet
  founders.parquet
  founder_updates/block=000000.parquet
```

Important output columns:

| output | key columns | value columns |
| --- | --- | --- |
| `dosage` | `sample_id`, `chromosome`, `position` | `dosage` |
| `genotype_calls` | `sample_id`, `chromosome`, `position` | `genotype_call`; `-1` means no-call |
| `genotype_posteriors` | `sample_id`, `chromosome`, `position` | `genotype_posterior`, fixed-size GP vector |
| `haplotype_probabilities` | `sample_id`, `chromosome`, `position` | `hap_dosage`, fixed-size founder/haplotype vector |
| `support_mask` | `sample_id`, `chromosome`, `position` | direct evidence/support booleans |
| `diagnostics` | `chromosome`, `position`, `block_id` | MAF, HWE, missingness, INFO, entropy, depth, support |
| `stage_timings.json` | one row per block | IO, HMM, calibration, write time, cache and memory fields |

Diagnostics columns include:

```text
n_samples, n_callable_samples, calling_rate, missing_rate,
maf, alt_af, het_rate, hom_rate, hard_het_rate, hard_hom_rate,
hwe_deviation, hwe_chisq, hwe_pvalue, info,
mean_entropy, mean_max_gp, mean_depth, support_rate,
calibration_abs_dosage_shift, calibration_het_rate_shift,
calibration_abs_maf_shift
```

## 7. Global Pedigree QC

Run pedigree QC after first-pass genotype calls exist for all selected chromosomes.

Use all autosomes when possible. Include X/Y/MT only if their ploidy and missingness are well controlled; autosomes are usually enough to identify pedigree/sample swaps.

Example:

```bash
stitchv2 pedigree-qc \
  --samples /project/inputs/samples.parquet \
  --pedigree /project/inputs/pedigree.parquet \
  --run-output-dir /project/runs/pass1_no_pedigree/chr1 \
  --run-output-dir /project/runs/pass1_no_pedigree/chr2 \
  --run-output-dir /project/runs/pass1_no_pedigree/chr3 \
  --run-output-dir /project/runs/pass1_no_pedigree/chr4 \
  --run-output-dir /project/runs/pass1_no_pedigree/chr5 \
  --output-dir /project/runs/pedigree_qc_global \
  --pedigree-offspring-col sample_id \
  --pedigree-parent1-col father_id \
  --pedigree-parent2-col mother_id \
  --family-col family_id \
  --value-column genotype_call \
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
  --umap-max-variants 10000 \
  --random-seed 1
```

For all chromosomes, generate the repeated `--run-output-dir` arguments programmatically:

```bash
RUN_ARGS=""
for chr in $(seq 1 20); do
  RUN_ARGS="$RUN_ARGS --run-output-dir /project/runs/pass1_no_pedigree/chr${chr}"
done

stitchv2 pedigree-qc \
  --samples /project/inputs/samples.parquet \
  --pedigree /project/inputs/pedigree.parquet \
  $RUN_ARGS \
  --output-dir /project/runs/pedigree_qc_global
```

## 8. Pedigree QC Outputs

`stitchv2 pedigree-qc` writes:

| file | meaning |
| --- | --- |
| `pedigree_qc_summary.json` | summary counts, thresholds, resolved columns, UMAP metadata |
| `pedigree_curated.parquet` | corrected pedigree for rerun |
| `pedigree_edge_qc.parquet` | one row per declared parent edge |
| `pedigree_sample_qc.parquet` | one row per sample |
| `sample_similarity.parquet` | related/similar sample pairs above `--report-min-r` |
| `pedigree_umap_embedding.parquet` | UMAP coordinates and sample annotations |
| `pedigree_umap_edges_before.parquet` | declared pedigree edges before curation |
| `pedigree_umap_edges_after.parquet` | pedigree edges after curation |
| `pedigree_long_edges.parquet` | long UMAP edges, useful for visually suspicious pedigrees |
| `pedigree_qc_variants.parquet` | variants selected for the QC similarity matrix |
| `pedigree_umap_before_after.html` | interactive UMAP plot with before/after pedigree edges |

`pedigree_edge_qc.parquet` columns:

| column | values |
| --- | --- |
| `child_id` | child/sample ID |
| `parent_id` | declared parent ID |
| `parent_role` | `parent1` or `parent2` |
| `r` | genotype similarity R |
| `r2` | squared similarity |
| `relationship_call` | `unrelated`, `inconclusive`, `first_degree`, or `same` |
| `edge_qc_status` | `retained`, `removed`, `inconclusive`, or `no_evidence` |
| `reason` | machine-readable reason |

`pedigree_sample_qc.parquet` columns:

| column | values |
| --- | --- |
| `sample_id` | sample ID |
| `removed_parent_edges` | number of parent edges removed |
| `inconclusive_parent_edges` | number of parent edges needing review |
| `pedigree_qc_status` | `pass`, `review`, or `corrected` |

`sample_similarity.parquet` columns include:

```text
sample_id1, sample_id2, r, r2, relationship_call,
expected_relationship, pedigree_consistency
```

Use `sample_similarity.parquet` to find probable duplicate/same animal pairs, parent-offspring pairs, and sibling-like clusters. The `pedigree_consistency` field highlights `unexpected_related` and `unexpected_unrelated` pairs.

## 9. Review Before Rerun

Check these first:

```bash
python - <<'PY'
import json, pandas as pd
root = "/project/runs/pedigree_qc_global"
print(json.dumps(json.load(open(f"{root}/pedigree_qc_summary.json")), indent=2)[:4000])
print(pd.read_parquet(f"{root}/pedigree_edge_qc.parquet")["edge_qc_status"].value_counts(dropna=False))
print(pd.read_parquet(f"{root}/pedigree_sample_qc.parquet")["pedigree_qc_status"].value_counts(dropna=False))
print(pd.read_parquet(f"{root}/sample_similarity.parquet")["pedigree_consistency"].value_counts(dropna=False))
PY
```

Open:

```text
/project/runs/pedigree_qc_global/pedigree_umap_before_after.html
```

Look for:

- long pedigree edges before curation that disappear after curation
- samples marked `corrected`
- samples with many unexpected close relatives
- same-animal/duplicate pairs
- clusters separated by sex/chromosome artifacts, which usually means ploidy or missingness needs review

## 10. Second Pass: Rerun With Curated Pedigree

Use `pedigree_curated.parquet` from the global QC pass.

Recommended transmission rerun:

```bash
stitchv2 run \
  --samples /project/inputs/samples.parquet \
  --positions /project/inputs/positions_chr12.parquet \
  --chromosome chr12 \
  --output-dir /project/runs/pass2_pedigree_transmission/chr12 \
  --founder-plink /project/inputs/founders8 \
  --founder-immutable \
  --n-founders 9 \
  --em-iterations 40 \
  --max-mem 90% \
  --hmm-backend jax \
  --jax-persistent-cache-dir /project/runs/jax_cache \
  --read-mode read_stream \
  --read-stream-backend auto \
  --snp-block-mode exact_streaming \
  --store-xi per-snp \
  --stitch-compat \
  --fragment-likelihood-mode replace \
  --compact-evidence-cache-dir /project/cache/chr12 \
  --compact-evidence-cache-mode read \
  --compact-evidence-cache-format parquet_zarr \
  --pedigree /project/runs/pedigree_qc_global/pedigree_curated.parquet \
  --pedigree-mode transmission \
  --pedigree-strength 0.6 \
  --pedigree-iterations 6 \
  --pedigree-offspring-col sample_id \
  --pedigree-parent1-col father_id \
  --pedigree-parent2-col mother_id \
  --genotype-call-mode quality_gated \
  --calibration-mode standard_callability \
  --calibration-callability-decision-mode per_snp_hierarchical \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-haplotype-probabilities \
  --write-support-mask
```

Alternative modes:

| mode | when to use |
| --- | --- |
| `smooth` | cheap conservative dosage smoothing; use for smoke tests or weak pedigrees |
| `kinship` | use broader relatedness as fallback when parent graph is incomplete |
| `transmission` | best for curated parent-child pedigrees; most explicit family model |

Typical strengths:

| mode | starting `--pedigree-strength` |
| --- | --- |
| `smooth` | `0.1` to `0.3` |
| `kinship` | `0.3` to `0.6` |
| `transmission` | `0.5` to `0.8` |

Treat these as tuning values. Validate against held-out array or high-confidence truth before using a high strength genome-wide.

## 11. Efficient Whole-Genome Scheduling

Recommended cluster pattern:

- submit one job per chromosome
- keep `--dask-processes` off inside each job
- give each job a unique Dask dashboard port or disable the dashboard
- keep one cache directory per chromosome
- use `readwrite` only for the first pass or when new samples are added
- use `read` for calibration/pedigree reruns
- keep `--jax-persistent-cache-dir` shared across jobs if the filesystem supports it

Example job loop:

```bash
for chr in $(seq 1 20) X Y MT; do
  sbatch run_stitchv2_chr.sh "$chr"
done
```

Inside `run_stitchv2_chr.sh`, derive:

```bash
CHR="$1"
PORT=$((8700 + SLURM_ARRAY_TASK_ID))
CACHE="/project/cache/chr${CHR}"
OUT="/project/runs/pass1_no_pedigree/chr${CHR}"
```

Then call `stitchv2 run` with `--dask-dashboard-address :${PORT}`.

## 12. Validation Checklist

After pass 1:

- `stage_timings.json` shows `compact_cache_hit` or `compact_cache_partial_hit` as expected.
- `diagnostics_summary.json` has no catastrophic heterozygote/homozygote/missingness failures.
- `genotype_calls/` exists for chromosomes used in pedigree QC.
- X/Y/MT missingness matches expected sex/ploidy.

After pedigree QC:

- Review `pedigree_qc_summary.json`.
- Review `pedigree_umap_before_after.html`.
- Confirm removed edges are biologically plausible.
- Inspect `sample_similarity.parquet` for unexpected duplicates or sample swaps.

After pass 2:

- Compare pass 1 vs pass 2 call rate, MAF, HWE, missingness, INFO.
- Compare against held-out array truth if available.
- Check `stage_timings.json` for `pedigree_mode`, `pedigree_edges`, and `pedigree_messages`.
- Confirm the cache remains in `read` mode and no BAM rereading dominates runtime.

## 13. Result Locations To Archive

Archive these for reproducibility:

```text
inputs/
  samples.parquet
  positions_*.parquet
  pedigree_original.parquet
  founders8.bed/.bim/.fam

runs/pass1_no_pedigree/
runs/pedigree_qc_global/
runs/pass2_pedigree_transmission/
cache/
environment.yml
command_log.txt
```

Do not edit the cache by hand. If input samples, positions, reference alleles, read filters, or chromosome windows change, create a new cache root or use a clearly versioned subdirectory.
