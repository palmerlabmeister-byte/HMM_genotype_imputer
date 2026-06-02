# STITCHV2 Pedigree Tutorial

This tutorial explains how to use pedigree information in STITCHV2, what the input files look like, and what the three pedigree modes do mathematically.

The short version:

- Use `--pedigree-mode smooth` for a very cheap conservative parent-mean regularizer.
- Use `--pedigree-mode kinship` when you have sparse or disconnected pedigrees and want long-distance relatives as a fallback.
- Use `--pedigree-mode transmission` when you want parent-child Mendelian messages, including information flowing through unsequenced parents.
- Original STITCH does not have an equivalent pedigree input in the standard CLI. These modes are STITCHV2 additions.

## 1. Where Pedigree Fits In The Model

STITCHV2 first runs the usual STITCH-like population HMM:

```text
reads / array calls -> STITCHV2 HMM -> genotype posterior GP[i, v, g]
```

where:

- `i` is sample
- `v` is variant
- `g` is genotype dosage class, for diploids `g in {0, 1, 2}`

The pedigree layer is applied after each HMM block:

```text
GP from population HMM
  -> optional calibration
  -> pedigree adjustment
  -> dosage, GP, GT outputs
```

For `smooth`, the adjustment can work from dosage alone. For `kinship` and `transmission`, STITCHV2 needs diploid genotype posteriors, so the pipeline makes sure `genotype_posteriors/` are available internally.

Current important limitation:

- Pedigree transmission is block-local because STITCHV2 writes blocks independently.
- Recombination smoothing works along the variants in each block.
- For pedigree-heavy runs, use a larger `--block-size` when memory allows so relatives can inform each other across longer chromosomal spans.

## 2. Input Formats

You can provide pedigree information in two ways.

## 2.1 Parent Columns Inside `samples.parquet`

Yes: parents can be added directly as columns in `samples.parquet`.

Minimal example:

```text
sample_id  bam_path              generation  father  mother
dad        /data/dad.bam         10          NA      NA
mom        /data/mom.bam         10          NA      NA
kid        /data/kid.bam         10          dad     mom
```

Parquet/CSV equivalent in Python:

```python
import pandas as pd

samples = pd.DataFrame(
    {
        "sample_id": ["dad", "mom", "kid"],
        "bam_path": ["/data/dad.bam", "/data/mom.bam", "/data/kid.bam"],
        "generation": [10.0, 10.0, 10.0],
        "father": ["", "", "dad"],
        "mother": ["", "", "mom"],
    }
)
samples.to_parquet("samples.parquet", index=False)
```

Supported parent column aliases:

| Meaning | Preferred | Also recognized |
|---|---|---|
| offspring/sample id | `sample_id` | `offspring`, `child`, `rfid`, `iid` |
| first parent | `father_id` | `father`, `sire`, `parent1`, `parent_1`, `dad` |
| second parent | `mother_id` | `mother`, `dam`, `parent2`, `parent_2`, `mom` |

Missing parents can be written as:

```text
NA, NaN, nan, n/a, N/A, "", 0
```

If `--pedigree` is not supplied and STITCHV2 sees parent columns in `samples.parquet`, it automatically builds the pedigree graph from the sample table.

Example CLI:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir run_pedigree_from_samples \
  --n-founders 8 \
  --pedigree-mode transmission \
  --pedigree-strength 0.7 \
  --pedigree-iterations 6 \
  --write-genotype-posteriors \
  --write-genotype-calls
```

## 2.2 Separate Pedigree Table

You can also keep the pedigree separate:

```text
sample_id  father_id  mother_id
kid        dad        mom
parent2    sire2      dam2
```

CLI:

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir run_pedigree_table \
  --n-founders 8 \
  --pedigree pedigree.parquet \
  --pedigree-mode kinship \
  --pedigree-strength 0.5 \
  --pedigree-offspring-col sample_id \
  --pedigree-parent1-col father_id \
  --pedigree-parent2-col mother_id
```

If the same parent columns exist in both places, the explicit `--pedigree` table wins.

## 2.3 Sparse Matrix From Python

The Python API can pass a sparse matrix directly:

```python
from scipy import sparse
from stitchv2 import PipelineConfig, StitchPipeline

# P[child, parent] = 0.5 for each listed parent.
P = sparse.csr_matrix(
    (
        [0.5, 0.5],
        ([2, 2], [0, 1]),
    ),
    shape=(3, 3),
)

config = PipelineConfig(
    chromosome="chr1",
    positions_path="positions.parquet",
    output_dir="run_sparse_pedigree",
    n_founders=8,
    pedigree_mode="transmission",
    pedigree_strength=0.7,
)

StitchPipeline(config).prepare_inputs(samples, pedigree=P, founder_panel=founder_panel)
```

Matrix convention:

```text
P[child, parent] = 0.5
```

If a child has one known parent, that row has one `0.5` edge. If it has two known parents, it has two `0.5` edges.

## 3. Parameters

| Parameter | Meaning | Typical value |
|---|---|---:|
| `--pedigree` | Separate pedigree table path. Optional if parent columns are in `samples.parquet`. | `pedigree.parquet` |
| `--pedigree-mode` | One of `off`, `smooth`, `kinship`, `transmission`. | `transmission` |
| `--pedigree-strength` | Blend weight for pedigree information. Higher values trust the pedigree more. | `0.5` to `0.8` |
| `--pedigree-iterations` | Message-passing iterations for `transmission`. | `4` to `8` |
| `--pedigree-kinship-threshold` | Drop tiny kinship edges for speed/sparsity. | `0.01` |
| `--pedigree-offspring-col` | Offspring column if your table uses a custom name. | `sample_id` |
| `--pedigree-parent1-col` | First parent column if custom. | `father_id` |
| `--pedigree-parent2-col` | Second parent column if custom. | `mother_id` |

STITCH equivalent:

```text
No direct original STITCH CLI equivalent.
```

The closest conceptual match in original STITCH is that relatives can help indirectly through shared population founders. STITCHV2 pedigree modes add explicit family graph information after the population HMM.

## 4. Mode 1: `smooth`

`smooth` is a dosage-level regularizer.

Let:

- `D[i, v]` be the HMM dosage for sample `i` at variant `v`.
- `P[i, j]` be the pedigree parent matrix, with `P[child, parent] = 0.5`.
- `lambda` be `--pedigree-strength`.

For sample `i`, compute the mean parent dosage:

```text
parent_mean[i, v] = sum_j P[i, j] D[j, v] / sum_j P[i, j]
```

Then blend:

```text
D_new[i, v] = (1 - lambda) D[i, v] + lambda parent_mean[i, v]
```

If a sample has no listed parents, its dosage is unchanged.

What it is good for:

- Very fast sanity check.
- Mild denoising when parents are sequenced.
- Simple parent-offspring consistency.

What it does not do:

- It does not use siblings except indirectly through parent rows.
- It does not model recombination.
- It does not propagate through unsequenced intermediates well.
- It does not update GP in the current simple smoothing path.

Efficiency:

```text
Time:   O(E * V)
Memory: O(N * V)
```

where `E` is number of parent edges, `N` is samples, and `V` is variants in the current block.

## 5. Mode 2: `kinship`

`kinship` turns the parent graph into an ancestry/kinship graph and uses relatives as a fallback prior.

First, STITCHV2 builds an ancestry matrix `A`.

For each sample:

```text
A[i, i] = 1
A[child, ancestor] += P[child, parent] * A[parent, ancestor]
```

So a grandparent contributes through the unsequenced parent:

```text
grandparent -> parent -> child

weight child to grandparent ~= 0.5 * 0.5 = 0.25
```

Then approximate kinship is:

```text
Phi = 0.5 * A * A.T
```

The diagonal is removed, and values below `--pedigree-kinship-threshold` are dropped.

For each target sample, STITCHV2 computes a relative dosage prior:

```text
prior[i, v] =
  sum_j Phi[i, j] evidence_weight[j, v] D[j, v]
  / sum_j Phi[i, j] evidence_weight[j, v]
```

The evidence weight comes from posterior confidence and direct support:

```text
high-confidence / read-supported calls -> high weight
uncertain / no-read calls              -> low weight
```

Then dosage and GP are blended toward the kinship prior.

What it is good for:

- Sparse pedigrees.
- Grandparents, cousins, siblings, and disconnected subgraphs.
- Datasets where the pedigree is large but only some animals are sequenced.
- A robust fallback when exact parent-child transmission is too aggressive.

What it does not do:

- It does not know which homolog was transmitted.
- It treats relatives as weighted predictors, not as exact inheritance paths.
- It is not a full pedigree HMM.

Efficiency:

```text
Build ancestry: roughly O(total retained ancestor links)
Apply prior:    O(nnz(Phi) * V)
Memory:         O(N * V + nnz(Phi))
```

The threshold matters. Lower thresholds keep more distant relatives and cost more memory/time.

## 6. Mode 3: `transmission`

`transmission` is the most pedigree-aware mode.

It performs iterative parent-child message passing on genotype posteriors. It is not yet a full phase-aware Lander-Green pedigree HMM, but it does explicitly model Mendelian parent-offspring dosage probabilities and allows information to flow through latent, unsequenced individuals.

The starting point is the HMM genotype posterior:

```text
GP0[i, v, g] = P(G[i, v] = g | reads, population HMM)
```

For diploids:

```text
g in {0, 1, 2}
q[i, v] = E[G[i, v]] / 2
```

where `q[i, v]` is the expected alt-allele fraction.

### 6.1 Recombination Smoothing

For each sample, the marker-level allele fraction is gently smoothed along the chromosome.

Between neighboring variants:

```text
stay = exp(-generation * distance_bp * 1e-8)
```

The code uses this as a conservative smoother, then blends:

```text
q_used = 0.85 * q_marker + 0.15 * q_smooth
```

This avoids erasing local evidence while still letting nearby markers support each other.

### 6.2 Parent To Child Message

If a child has two parents with allele fractions `q1` and `q2`, then the child genotype distribution is:

```text
P(G_child = 0) = (1 - q1) * (1 - q2)
P(G_child = 1) = q1 * (1 - q2) + (1 - q1) * q2
P(G_child = 2) = q1 * q2
```

If one parent is missing, STITCHV2 uses the population allele fraction as the other parent.

### 6.3 Child To Parent Message

Information also flows backward.

For a parent `p` and child `c`, STITCHV2 asks:

```text
How likely is the child's posterior if parent p had genotype g?
```

For each candidate parent genotype `g`, it computes:

```text
parent_q = g / 2
likelihood[g] = sum_child_g GP_child[child_g] *
                P(child_g | parent_q, other_parent_q)
```

This lets a sequenced child or sibling inform an unsequenced parent.

### 6.4 Combining Messages With HMM Evidence

Pedigree messages are blended with the original HMM posterior.

The implementation intentionally anchors samples with direct read/array support more strongly:

```text
observed samples      -> smaller pedigree blend
unsequenced samples   -> larger pedigree blend
```

This prevents relatives from overwriting good read evidence, while allowing strong updates for no-read animals.

After transmission iterations, STITCHV2 adds a small kinship fallback blend. This helps when a pedigree component has useful relatives but incomplete parent-child paths.

What it is good for:

- Imputing completely unsequenced animals.
- Latent unsequenced parents between sequenced grandparents and offspring.
- Sibling/offspring evidence predicting parents.
- Pedigrees where information needs to move in both directions.

What it does not yet do:

- It is not a fully phase-aware homolog transmission HMM.
- It does not explicitly enumerate inherited haplotypes through the whole pedigree.
- It works per STITCHV2 variant block, so very long-range propagation is strongest when `--block-size` is large enough.

Efficiency:

```text
Time:   O(iterations * E * V)
Memory: O(N * V * 3 + E)
```

plus the small kinship fallback:

```text
O(nnz(Phi) * V)
```

In practice, this is still cheap compared with BAM reading and the HMM for moderate pedigrees.

## 7. Worked CLI Examples

### 7.1 Pedigree In `samples.parquet`

```bash
stitchv2 run \
  --samples samples_with_parents.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out_transmission \
  --n-founders 8 \
  --hmm-backend jax \
  --em-iterations 5 \
  --block-size 2000 \
  --pedigree-mode transmission \
  --pedigree-strength 0.7 \
  --pedigree-iterations 6 \
  --write-genotype-posteriors \
  --write-genotype-calls
```

### 7.2 Separate Pedigree Table With Dog/Cattle-Style Names

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out_dam_sire \
  --n-founders 8 \
  --pedigree pedigree.parquet \
  --pedigree-mode transmission \
  --pedigree-strength 0.7 \
  --pedigree-offspring-col rfid \
  --pedigree-parent1-col sire \
  --pedigree-parent2-col dam
```

### 7.3 Conservative Kinship Fallback

```bash
stitchv2 run \
  --samples samples.parquet \
  --positions positions.parquet \
  --chromosome chr1 \
  --output-dir out_kinship \
  --n-founders 8 \
  --pedigree pedigree.parquet \
  --pedigree-mode kinship \
  --pedigree-strength 0.5 \
  --pedigree-kinship-threshold 0.02
```

Use a higher threshold for faster, more local relatives. Use a lower threshold if grandparents, great-grandparents, or remote connected relatives are important.

## 8. Output Files

Pedigree runs write the usual STITCHV2 outputs:

```text
dosage/
genotype_posteriors/
genotype_calls/
samples.parquet
positions.parquet
founders.parquet
stage_timings.json
memory_profile_summary.json
```

They also write:

```text
pedigree_summary.json
```

Example:

```json
{
  "source": "table",
  "n_samples": 44,
  "n_edges": 44,
  "n_components": 9,
  "max_component_size": 6,
  "ignored_parent_ids": []
}
```

`stage_timings.json` also records:

```text
pedigree_mode
pedigree_edges
pedigree_components
pedigree_messages
```

for each block when applicable.

## 9. Validation Benchmark

The synthetic pedigree benchmark generates nuclear families and grandparent-latent-parent-offspring structures with known Mendelian truth.

Run:

```bash
python benchmarks/benchmark_pedigree_synthetic.py \
  --output-dir benchmark_runs/pedigree_synthetic_validation \
  --n-direct-families 5 \
  --n-grandparent-families 4 \
  --n-variants 400 \
  --chromosome-length 1500000 \
  --coverage 0.15 \
  --iterations 5 \
  --block-size 200 \
  --jax-sample-batch-size 32 \
  --rscript-path "$(which Rscript)" \
  --force
```

The benchmark should write its own metrics and plots under the requested output directory. Do not copy historical local result values into a production validation report; rerun the benchmark in the target environment and keep the generated report with that run.

## 10. Choosing A Mode

Use this as a starting rule:

| Scenario | Recommended mode |
|---|---|
| You only want a very fast parent mean sanity check | `smooth` |
| Large sparse pedigree, uncertain exact paths, many disconnected pieces | `kinship` |
| Parents/siblings/grandparents should impute unsequenced animals | `transmission` |
| You want strict STITCH-style population-only behavior | `off` |

For production, start with:

```text
--pedigree-mode transmission
--pedigree-strength 0.5 to 0.7
--pedigree-iterations 4 to 6
--block-size as large as memory allows
```

Then check:

- dosage R2 or concordance on known truth
- missingness
- per-SNP F1 / balanced accuracy
- INFO calibration
- whether observed high-depth animals changed too much
