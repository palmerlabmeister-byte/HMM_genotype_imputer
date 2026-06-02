# STITCHV2 Pedigree QC and Transmission Deep Dive

STITCHV2 has two separate pedigree concepts:

1. Pedigree QC: detect likely wrong parent links or suspicious sample relationships from genotype similarity.
2. Pedigree adjustment: use a curated pedigree to improve or stabilize imputation results after the HMM.

These should normally be run in two passes:

1. Run STITCHV2 without pedigree adjustment.
2. Run `stitchv2 pedigree-qc` across one or more chromosomes.
3. Review the curated pedigree and UMAP edge plot.
4. Re-run STITCHV2 with the curated pedigree and `pedigree_mode=kinship` or `pedigree_mode=transmission`.

This avoids letting an uncurated pedigree distort the HMM output.

## Why Pedigree QC Comes Before Transmission

A declared parent-child edge is powerful information. If it is wrong, transmission adjustment can pull the child toward the wrong haplotype or genotype. In real breeding data, pedigree errors can come from:

- sample swaps,
- animal ID mix-ups,
- missing parents,
- wrong sire or dam,
- duplicate animals,
- mislabeled sequencing files,
- unrecorded breeding events.

STITCHV2 therefore supports a first pass without pedigree. The first-pass genotypes are used only to assess relationship consistency. The second pass uses the curated pedigree.

## Pedigree QC Command

Example:

```bash
stitchv2 pedigree-qc \
  --samples samples.parquet \
  --pedigree pedigree.parquet \
  --run-output-dir runs/chr1_first_pass \
  --run-output-dir runs/chr2_first_pass \
  --run-output-dir runs/chr3_first_pass \
  --output-dir pedigree_qc \
  --max-variants 50000 \
  --min-call-rate 0.80 \
  --min-maf 0.005 \
  --sample-block-size 1024
```

Inputs can come from one or more first-pass STITCHV2 output directories:

```bash
--run-output-dir runs/chr12
```

or from one or more long genotype tables:

```bash
--genotype-table calls_chr12.parquet
```

A whole-genome workflow should normally pass multiple chromosome outputs to one global `pedigree-qc` command. This produces a single curated pedigree for the second pass.

## Pedigree QC Input Columns

Default pedigree columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Child/offspring ID. |
| `father_id` | Parent 1. |
| `mother_id` | Parent 2. |

Custom columns:

```bash
--pedigree-offspring-col animal_id
--pedigree-parent1-col sire
--pedigree-parent2-col dam
--family-col family
```

The pedigree table can be supplied with:

```bash
--pedigree pedigree.parquet
```

If absent, the pedigree columns are read from `--samples`.

## Genotype Matrix Used for QC

Pedigree QC loads hard calls or dosages from first-pass run outputs.

It filters variants by:

```bash
--max-variants 50000
--min-call-rate 0.80
--min-maf 0.005
```

It then computes pairwise genotype similarity using an R-like correlation metric. The implementation is designed to be block-wise so it does not require materializing an enormous all-samples-by-all-variants object in memory.

Important controls:

```bash
--sample-block-size 1024
--max-full-matrix-samples 5000
--report-min-r 0.59
```

The `npplink.R2` convention that motivated this implementation returns an R2 matrix over columns, not rows. STITCHV2 follows the same general idea: samples must be represented in the correct dimension before similarity is computed.

## Relationship Calls

Pedigree QC classifies observed similarity by thresholds.

Defaults:

```bash
--unrelated-max-r 0.59
--first-degree-min-r 0.64
--same-min-r 0.88
```

Conceptual classes:

| Observed relationship | Meaning |
| --- | --- |
| `unrelated` | Pair is too dissimilar for a declared close relation. |
| `first_degree` | Consistent with parent-child or sibling-like relatedness. |
| `same` | Very similar, possible duplicate/sample swap/same animal. |
| intermediate classes | Ambiguous or not confidently classified. |

Declared parent edges can be removed when the observed relationship call is unsafe.

Default unlink behavior:

```bash
--unlink-calls unrelated,same
```

This means:

- remove a declared parent edge if parent and child look unrelated,
- remove a declared parent edge if parent and child look like the same animal,
- keep or mark inconclusive edges when evidence is insufficient.

## Pedigree QC Outputs

The output directory contains:

```text
pedigree_qc/
  pedigree_qc_summary.json
  pedigree_curated.parquet
  pedigree_edge_qc.parquet
  pedigree_sample_qc.parquet
  sample_similarity.parquet
  pedigree_umap_embedding.parquet
  pedigree_umap_edges_before.parquet
  pedigree_umap_edges_after.parquet
  pedigree_long_edges.parquet
  pedigree_qc_variants.parquet
  pedigree_umap_before_after.html
```

### `pedigree_curated.parquet`

This is the table to pass into the second STITCHV2 run.

Canonical columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Child/offspring ID. |
| `father_id` | Curated parent 1 ID. Empty if removed. |
| `mother_id` | Curated parent 2 ID. Empty if removed. |
| `family_id` | Family ID when available. |
| `original_parent1_id` | Original parent 1 before curation. |
| `original_parent2_id` | Original parent 2 before curation. |
| `pedigree_qc_removed_parent_count` | Number of removed parent links for this child. |
| `pedigree_qc_status` | Status such as pass or parent_removed. |

The file is directly consumable by:

```bash
stitchv2 run --pedigree pedigree_qc/pedigree_curated.parquet
```

### `pedigree_edge_qc.parquet`

One row per declared parent-child edge.

Columns:

| Column | Meaning |
| --- | --- |
| `child_id` | Child sample. |
| `parent_id` | Declared parent sample. |
| `parent_role` | Parent role, such as parent1 or parent2. |
| `r` | Observed genotype similarity. |
| `r2` | Squared similarity. |
| `relationship_call` | Observed relationship class. |
| `edge_qc_status` | kept, removed, or inconclusive. |
| `reason` | Reason for the decision. |

This is the most important table for reviewing pedigree corrections.

### `pedigree_sample_qc.parquet`

One row per sample.

Columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Sample ID. |
| `removed_parent_edges` | Count of removed declared parents. |
| `inconclusive_parent_edges` | Count of parent edges without enough evidence. |
| `pedigree_qc_status` | Sample-level status. |

### `sample_similarity.parquet`

Pairs with high enough similarity to report.

Use cases:

- detect likely duplicate animals,
- identify parent-child-like pairs not declared in the pedigree,
- find sibling-like or close-relative clusters,
- identify sequencing/sample swaps.

### UMAP Outputs

Pedigree QC writes UMAP coordinates and edge files:

```text
pedigree_umap_embedding.parquet
pedigree_umap_edges_before.parquet
pedigree_umap_edges_after.parquet
pedigree_long_edges.parquet
pedigree_umap_before_after.html
```

The HTML plot is designed to show:

- genotype similarity structure,
- declared pedigree edges before curation,
- pedigree edges after curation,
- long UMAP edges that may indicate wrong pedigree links.

The before/after edge view is useful because bad pedigree edges often appear as long links crossing the embedding.

## Using Curated Pedigree in a Second Run

Example second pass:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr12 \
  --output-dir runs/chr12_second_pass_transmission \
  --n-founders 8 \
  --founder-plink founders8 \
  --founder-immutable \
  --em-iterations 1 \
  --pedigree pedigree_qc/pedigree_curated.parquet \
  --pedigree-mode transmission \
  --pedigree-strength 0.25 \
  --pedigree-iterations 4 \
  --write-genotype-posteriors \
  --write-genotype-calls \
  --write-haplotype-probabilities
```

Use `kinship` when you want a gentler relationship-aware fallback. Use `transmission` when parent-child Mendelian information is reliable enough to apply message passing.

## Pedigree Graph Representation

Internally, the pedigree becomes a directed parent-child graph:

```text
parent -> child
```

For diploid parentage, each known parent contributes half of the expected inherited alleles. Conceptually, the parent matrix has entries:

```text
P[child, parent] = 0.5
```

for each declared parent.

The graph stores:

- sample order,
- parent lists per child,
- child lists per parent,
- summary counts,
- parent edge counts.

If no valid pedigree is present, or if `pedigree_strength <= 0`, STITCHV2 returns the HMM outputs unchanged.

## Pedigree Modes

### `off`

No pedigree adjustment.

Use this for:

- first-pass pedigree QC,
- STITCH parity benchmarks,
- cases where pedigree metadata is incomplete or untrusted.

### `smooth`

Dosage-level smoothing. This mode blends a sample dosage toward parental mean dosage where parents are available.

It is intentionally simple:

- works without genotype posteriors,
- does not model Mendelian transmission fully,
- mostly acts as a gentle post-HMM smoother.

This is useful for smoke tests or weak pedigree regularization, but it is less principled than `transmission`.

### `kinship`

Kinship fallback uses the pedigree graph to compute a relationship-informed dosage prior. It then blends genotype posteriors toward that prior, with read support and posterior confidence anchoring directly observed genotypes.

Important properties:

- requires genotype posteriors for full effect,
- can help samples with weak read evidence,
- is less strict than full transmission,
- uses `pedigree_kinship_threshold` to control which relationships influence adjustment.

### `transmission`

Transmission mode performs iterative message passing over parent-child relationships.

It is the most explicit pedigree mode and requires diploid genotype posteriors:

```text
genotype_posterior.shape = (samples, positions, 3)
```

If this shape is not available, STITCHV2 falls back to smoother behavior rather than pretending a non-diploid posterior supports diploid Mendelian transmission.

## Transmission Message Passing

For a diploid genotype posterior:

```text
GP[sample, variant] = [P(G=0), P(G=1), P(G=2)]
```

the allele probability is:

```text
q = E[G] / 2
  = (0 * GP0 + 1 * GP1 + 2 * GP2) / 2
```

For two parents with allele probabilities `q1` and `q2`, the expected child genotype distribution is:

```text
P(child G=0) = (1 - q1) * (1 - q2)
P(child G=1) = q1 * (1 - q2) + (1 - q1) * q2
P(child G=2) = q1 * q2
```

If one parent is missing, STITCHV2 uses a population allele probability fallback for the missing parent.

### Forward Parent-to-Child Messages

For each child:

1. Get the current allele probability for known parent 1.
2. Get the current allele probability for known parent 2, or population fallback.
3. Compute the expected child genotype distribution.
4. Multiply this into the child's incoming message.

This encourages the child posterior to be compatible with its parents.

### Reverse Child-to-Parent Messages

Children also inform parents. For each parent:

1. Look at each child of that parent.
2. Use the other parent if known, or population fallback if missing.
3. Evaluate how likely the child's current posterior would be under each candidate parent genotype.
4. Multiply this as an incoming message to the parent.

This lets multiple children stabilize uncertain parent genotypes.

### Recombination Smoothing

Transmission mode smooths marker-level allele probabilities along positions with a gentle recombination-aware smoother. The smoother is not a replacement for the HMM. It is a prior used inside pedigree message passing to avoid extremely jagged per-site pedigree messages.

The code blends:

```text
q = 0.85 * q_marker + 0.15 * q_smooth
```

with clipping to `[0, 1]`.

### Direct Evidence Anchoring

Pedigree messages should not overwrite strong direct read evidence. STITCHV2 computes evidence weights from:

- support mask,
- posterior confidence,
- read-backed evidence.

Samples with direct evidence receive a smaller pedigree blend. Samples with weak or missing evidence can receive a stronger pedigree update.

The blend is bounded so pedigree messages cannot completely replace the HMM posterior:

```text
blend <= 0.85
```

### Iterations

Transmission runs for:

```bash
--pedigree-iterations 4
```

by default. More iterations can propagate information farther through the pedigree, but too many iterations may over-smooth if the pedigree is imperfect.

## Kinship Fallback After Transmission

After transmission message passing, STITCHV2 applies a small kinship fallback blend. This helps cases where:

- only one parent is known,
- sibling/relative information is present but direct parentage is incomplete,
- transmission messages are sparse.

The fallback is intentionally limited so it does not dominate direct HMM evidence.

## Sex Chromosomes, MT, Y, and Ploidy

Full Mendelian transmission is diploid. This matters for:

- male X,
- Y,
- MT,
- sex-specific absent chromosomes,
- polyploid samples.

Recommended workflow:

1. Run each chromosome with correct ploidy settings.
2. Produce first-pass calls per chromosome.
3. Run global `pedigree-qc` across all usable autosomes and other informative chromosomes.
4. Use the curated pedigree for second-pass runs.
5. For non-diploid chromosomes, rely on modes that are valid for that ploidy, or accept fallback behavior when full transmission does not apply.

Pedigree QC can be global even if each chromosome was imputed independently. Parentage errors are usually detectable from a subset of informative variants, especially autosomes. MT and Y can provide useful sex-lineage information, but they should not be the only basis for global parent-child curation.

## Pedigree Summary Output

When a run uses a pedigree, STITCHV2 writes:

```text
pedigree_summary.json
```

This contains graph summary information such as:

- number of samples,
- number of parent edges,
- number of children with parents,
- parent-column mapping,
- mode-specific summary when available.

For transmission and kinship runs, stage timing and diagnostics should also be checked because pedigree adjustment can change:

- dosage,
- genotype posterior,
- genotype call rate,
- heterozygosity,
- HWE deviation,
- calibration shifts.

## Reliability Recommendations

For production pedigree use:

1. Do not run full transmission on an uncurated pedigree.
2. Run a no-pedigree first pass.
3. Curate globally across multiple chromosomes where possible.
4. Review `pedigree_edge_qc.parquet`.
5. Review `pedigree_umap_before_after.html`.
6. Remove or flag samples with strong evidence of swaps or duplicates.
7. Use `pedigree_curated.parquet` in the second pass.
8. Compare `off`, `kinship`, and `transmission` on truth-backed subsets before using transmission as a default.

For benchmark reporting, include:

- number of declared edges,
- number of removed edges,
- number of inconclusive edges,
- relationship threshold settings,
- UMAP before/after plot,
- second-pass quality metrics vs first-pass no-pedigree calls.
