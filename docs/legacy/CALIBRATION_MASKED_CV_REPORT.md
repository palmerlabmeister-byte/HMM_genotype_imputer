# STITCHV2 Calibration Design And Reimplementation Report

Status: historical masked-CV calibration report. The current default hard-call calibration path is `--calibration-mode standard_callability` with `--calibration-callability-model lightgbm` and per-SNP hierarchical fallback. Do not treat this report as the current default calibration tutorial.

## Purpose

This report describes the current STITCHV2 genotype-posterior calibration system in enough detail for a new implementer to reproduce it. It also explains why calibration improved the official STITCH paper mouse benchmark but did not yet match STITCH, why no-call thresholding did not improve the QC metrics in that run, and how calibration can be made faster and more powerful in future versions.

The current implementation lives primarily in:

- `src/stitchv2/calibration.py`
- `src/stitchv2/pipeline.py`
- `benchmarks/benchmark_calibration_strategies.py`
- `benchmarks/benchmark_real_calling_calibration_sweep.py`

## What Calibration Receives

After the HMM finishes for a block of variants, calibration receives arrays with these conceptual shapes:

| Name | Shape | Meaning |
|---|---:|---|
| `raw_posterior` | `n_samples x n_variants x 3` | Raw diploid genotype posterior probabilities for `GT=0,1,2`. |
| `dosage` | `n_samples x n_variants` | Expected alternate allele count, usually `GP1 + 2 * GP2` or an HMM dosage output. |
| `depth` | `n_samples x n_variants` | Read depth or evidence depth used to sharpen or soften dosage-derived posterior probabilities. |
| `truth_genotype` | `n_samples x n_variants` | Optional masked training labels, encoded `0,1,2`, with `-1` for unknown. |
| `train_mask` | `n_samples x n_variants` | Boolean mask defining which truth labels are allowed to train the calibrator. |
| `ref_count`, `alt_count`, `other_count` | `n_samples x n_variants` | Per-sample read support used by the learned LightGBM calibration/call-correctness models. |
| `support_mask` | `n_samples x n_variants` | Whether an individual has any read support at a site. |

The calibration layer does not alter read extraction, fragment likelihoods, founder updates, or HMM transitions. It transforms posterior genotype probabilities and, optionally, hard-call behavior.

## Fixed Calibration

The simplest calibration path is `calibrate_genotype_posterior(...)`.

It computes a dosage-derived posterior and blends it with the raw HMM posterior.

For diploid genotypes, define the genotype axis:

```text
g = [0, 1, 2]
```

Given dosage `d`, the dosage-only posterior is:

```text
logit_g = -((d - g)^2) / temperature
P_dosage(g) = softmax(logit_g)
```

If depth is supplied, the effective temperature is reduced with depth:

```text
temperature_eff = temperature / sqrt(depth + 1)
logit_g = -((d - g)^2) / temperature_eff
```

Then raw and dosage-derived posteriors are blended:

```text
P_calibrated = (1 - blend) * P_raw + blend * P_dosage
P_calibrated = clip(P_calibrated, min_prob, 1)
P_calibrated = P_calibrated / sum(P_calibrated)
```

Default values before masked-CV tuning were:

```text
temperature = 0.35
blend = 0.35
```

These are fixed heuristics. They are not learned from truth. They are useful for smoothing posteriors but cannot fully repair a severe dosage-scale mismatch.

## Masked-CV Calibration

The new masked-CV calibrator is `masked_cv_calibrate_genotype_posterior(...)`.

Its goal is to learn calibration parameters from held-out genotype labels without using those labels for evaluation. In synthetic data, the labels come from known truth. In real-data benchmarks, they can come from pseudo-truth such as GATK calls, microarray genotypes, or intentionally masked high-confidence genotypes.

### Inputs

The function takes:

```python
masked_cv_calibrate_genotype_posterior(
    raw_posterior=raw_gp,
    dosage=dosage,
    truth_genotype=truth_gt,
    train_mask=train_mask,
    depth=depth,
    maf_bins=(0.0, 0.01, 0.05, 0.5),
    temperatures=(0.15, 0.25, 0.35, 0.5, 0.75, 1.0),
    blends=(0.0, 0.25, 0.5, 0.75, 1.0),
    dosage_scales=(0.75, 1.0, 1.25, 1.5, 2.0),
    dosage_offsets=(-0.25, 0.0, 0.25),
    hwe_prior_weights=(0.0, 0.25, 0.5, 1.0),
    hwe_weight=0.0,
    hwe_min_maf=0.05,
)
```

### Step 1: Select Training Labels

The training set is:

```text
train = (truth_genotype >= 0) AND train_mask
```

Only these entries may influence parameter selection. All other entries are ignored during optimization.

This is the crucial anti-leakage rule. The calibration may be fitted using data from the same run, but only on labels deliberately assigned to the fitting mask. Evaluation must use a disjoint held-out mask.

### Step 2: Estimate MAF

For every variant, estimate MAF from training labels when available:

```text
AF_j = mean(truth_genotype[:, j]) / 2
MAF_j = min(AF_j, 1 - AF_j)
```

Calibration also keeps the oriented alternate allele frequency:

```text
ALT_AF_j = mean(truth_genotype[:, j]) / 2
```

This matters because HWE priors need allele orientation. MAF alone cannot distinguish a site that is almost all reference from a site that is almost all alternate. Both have low MAF, but their expected genotype distributions are opposite.

If no training labels are available for a variant, fall back to posterior-derived MAF:

```text
dosage_ij = sum_g GP_ijg * g
AF_j = mean_i(dosage_ij) / 2
MAF_j = min(AF_j, 1 - AF_j)
```

The default MAF bins are:

```text
[0.00, 0.01) rare / nearly fixed
[0.01, 0.05) low-frequency
[0.05, 0.50] common
```

The purpose is to avoid forcing rare and common variants to share one calibration surface. Rare variants need stronger false-positive control. Common variants can support more aggressive dosage calibration.

### Step 3: Search Parameter Grid Per MAF Bin

For each MAF bin, the calibrator searches combinations of:

```text
temperature
a blend weight
a dosage scale
a dosage offset
an HWE prior weight
```

The transformed dosage is:

```text
d_scaled = clip(dosage_scale * dosage + dosage_offset, 0, 2)
```

Then the dosage-derived posterior is recomputed:

```text
P_dosage(g) = softmax(-((d_scaled - g)^2) / temperature_eff)
```

Then the candidate posterior is:

```text
P_candidate = (1 - blend) * P_raw + blend * P_dosage
P_candidate = normalize(clip(P_candidate, min_prob, 1))
```

The HWE prior starts the posterior from the expected genotype distribution implied by the oriented alternate allele frequency:

```text
P_HWE = [(1 - ALT_AF)^2, 2 * ALT_AF * (1 - ALT_AF), ALT_AF^2]
log P_candidate = log P_candidate + hwe_prior_weight * log P_HWE
P_candidate = normalize(exp(log P_candidate))
```

This is the answer to the "temperature assumes too many hets" concern. Temperature by itself is symmetric around the dosage value and has no knowledge of population genotype expectations. If dosage is close to 1, a temperature-only posterior will naturally favor heterozygotes. The HWE prior prevents that from being the only starting point: for a common allele it favors the HWE mixture, for a nearly fixed reference site it favors `GT0`, and for a nearly fixed alternate site it favors `GT2`.

### Step 4: Score Each Candidate

For the training entries in that MAF bin, the objective is:

```text
score = NLL
      + brier_weight * Brier
      + dosage_mse_weight * DosageMSE
      + HWEPenalty
```

The negative log likelihood is:

```text
NLL = -mean(log(P_candidate(true_genotype)))
```

The Brier term is:

```text
Brier = mean(sum_g (P_candidate(g) - I[g == truth])^2)
```

The dosage MSE term is:

```text
DosageMSE = mean((sum_g P_candidate(g) * g - truth_genotype)^2)
```

The default weights are:

```text
brier_weight = 0.25
dosage_mse_weight = 0.10
```

### Step 5: Optional HWE Soft Penalty

For common variants, a Hardy-Weinberg equilibrium penalty can be added. This is currently a soft penalty, not a hard constraint.

For each variant, estimate genotype distribution from posterior probabilities:

```text
obs = mean_i(P_candidate[i, j, :])
```

Estimate allele frequency:

```text
p = (obs_GT1 + 2 * obs_GT2) / 2
q = 1 - p
```

The HWE expected genotype distribution is:

```text
exp = [q^2, 2pq, p^2]
```

The penalty is:

```text
HWEPenalty = hwe_weight * mean((obs - exp)^2)
```

The penalty is only applied to variants with:

```text
MAF >= hwe_min_maf
```

Default benchmark setting used:

```text
hwe_weight = 0.02
hwe_min_maf = 0.05
```

HWE is deliberately soft because many real datasets have legitimate deviations from HWE:

- population structure
- inbreeding
- selected loci
- sex chromosomes
- related animals
- breed structure
- small sample sizes

### Step 6: Apply Best Parameters To All Variants In The Bin

After choosing the best parameter tuple for a MAF bin, STITCHV2 applies it to every sample and every variant in that bin, including evaluation entries.

This produces a calibrated posterior matrix:

```text
P_masked_cv[i, j, g]
```

Hard calls are then made with the regular call function, usually argmax unless no-call mode is requested.

### Step 7: Metadata Output

The calibrator returns:

```python
gp_calibrated, metadata
```

The metadata includes:

```text
status
mode
maf_bins
n_train_rows
hwe_weight
hwe_min_maf
optimize_dosage_scale
per-bin selected parameters
```

Example from the STITCH paper mouse benchmark:

```text
rare/nearly fixed bin:
temperature = 0.15
blend = 1.0
dosage_scale = 2.0
dosage_offset = 0.25
```

This explains why calibration improved the real-data calls: STITCHV2 raw dosage was near `1.05` at loci where pseudo-truth was mostly `GT2`, so the optimizer learned to scale dosage upward.

## Hard Calls And No-Calls

Hard calls are produced by `genotype_call_from_posterior(...)`.

The default argmax call is:

```text
GT = argmax_g P(g)
```

The STITCH-style no-call rule is implemented by `stitch_vcf_genotype_call_from_posterior(...)`:

```text
GT = argmax_g P(g)
if max_g P(g) < threshold:
    GT = missing
```

The threshold comparison is inclusive:

```text
max GP >= threshold means keep the call
max GP < threshold means no-call
```

This mirrors the STITCH VCF writer behavior in `external/STITCH_source/STITCH/src/writers.cpp`, where a genotype is emitted only when the winning GP is at least `0.90`; otherwise `./.` is written.

### Log-GP And Class-Aware No-Call Scores

STITCHV2 now also supports fitted no-call thresholds through:

```text
fit_no_call_thresholds_by_calibration_class(...)
apply_no_call_thresholds_by_variant(...)
```

The no-call score does not have to be raw `max(GP)`. The supported scores are:

```text
max_gp
log_max_gp
margin
neg_entropy
log_max_gp_minus_entropy
```

`log_max_gp` is useful when GP values are ill-conditioned or packed near 1.0. In probability space, `0.998` and `0.999` look nearly identical; in log space, the optimizer sees additive evidence differences and is less sensitive to floating-point saturation. `log_max_gp_minus_entropy` is usually safer because it rewards a confident winner and penalizes diffuse posterior mass.

The fitted threshold model currently builds calibration classes from:

```text
MAF bin
mean posterior entropy bin
HWE-deviation bin
```

It optimizes a threshold for each class using the masked training labels. The objective is balanced between macro-F1 and balanced accuracy, with a small reward for preserving call rate:

```text
objective = 0.5 * macro_F1 + 0.5 * balanced_accuracy + 0.03 * call_rate
```

The apply step enforces a rank-based per-variant no-call cap. For example, with `max_no_call_rate_per_variant = 0.10` and 48 samples, at most `floor(0.10 * 48) = 4` samples can be no-called at a SNP. This avoids the quantile rounding issue where a nominal 10% cap can become 5/48 calls in small panels.

## Why No-Call Thresholding Did Not Help The Paper Benchmark

The no-call implementation is correct. It did not help because the GP confidence distributions had cliff-like behavior.

On the held-out STITCH paper pseudo-truth split:

| Mode | Mean max GP | Threshold behavior |
|---|---:|---|
| raw STITCHV2 | ~0.499 | Almost everything becomes no-call at threshold `0.5` or above. |
| fixed calibrated STITCHV2 | ~0.640 | Everything is called at `0.5`, everything is missing at `0.7+`. |
| masked-CV STITCHV2 | ~0.998 | Everything remains called even at `0.99`. |

So no-call thresholding had no smooth operating region. It could not selectively remove wrong calls because confidence was not well-ranked among correct and incorrect hard calls.

This means no-call thresholding itself is not broken. The posterior confidence distribution is not calibrated enough for a global STITCH-style max-GP threshold to be useful in this run.

## Benchmark Findings

### Synthetic Data

Synthetic report:

```text
benchmark_runs/tutorial_deep_fresh_2026-05-01/calibration_masked_cv_2026-05-04/report/calibration_benchmark_report.md
```

Summary:

| Strategy | R2 | F1 | Accuracy | INFO | Call rate |
|---|---:|---:|---:|---:|---:|
| pure argmax | 0.9555 | 0.9666 | 0.9716 | 0.8989 | 1.0000 |
| masked-CV | 0.9584 | 0.9704 | 0.9742 | 0.9372 | 1.0000 |
| masked-CV fitted no-call | 0.9584 | 0.9884 | 0.9902 | 0.9372 | 0.9276 |
| LightGBM one-pass | 0.9443 | 0.9874 | 0.9902 | 0.9080 | 0.9167 |
| tuned no-call | 0.9548 | 0.9859 | 0.9884 | 0.9050 | 0.9167 |
| STITCH | -0.7961 | 0.3666 | 0.5043 | 0.7600 | 0.8255 |

Interpretation:

- Masked-CV improves always-call posterior quality and INFO.
- Fitted no-call improves F1 while enforcing the per-SNP missingness cap; on the 48-sample synthetic panel, the maximum realized no-call rate is 8.33% with a 10% cap.
- LightGBM is now installed in the benchmark environment and the one-pass row is a real learned calibration. The strongest synthetic feature importances are posterior margin, entropy, max confidence, the center-context call, no-call status, INFO, MAF, founder support, and HWE diagnostics.
- STITCH performs poorly on this synthetic data because the dataset is generated by the STITCHV2 synthetic model and run configuration, not necessarily by the same generative assumptions STITCH expects. In particular, STITCH may no-call many sites and may not share the synthetic founder/haplotype structure used for STITCHV2 parity diagnostics.

### STITCH Paper Mouse Data

Real-data calibration report:

```text
benchmark_runs/stitch_paper_mouse_gatk_600_2026-05-01/calibration_calling_sweep/CALLING_CALIBRATION_SWEEP.md
```

Held-out pseudo-truth result:

| Method | Accuracy | F1 | Mean dosage on truth | Hard-call pattern |
|---|---:|---:|---:|---|
| fixed STITCHV2 | 0.0020 | 0.0013 | 1.0489 | all `GT1` |
| masked-CV STITCHV2 | 0.5944 | 0.2501 | 1.5941 | mixed `GT1` and `GT2` |
| STITCH | 0.9201 | 0.3278 | 1.7488 | mostly `GT2` |

Interpretation:

- Calibration helps substantially.
- Calibration does not fully close the gap to STITCH.
- The remaining gap likely comes from upstream HMM/founder/read-emission parity, because calibration can only transform the posterior and dosage it receives.

## Learned LightGBM Calibration

The bin-based masked-CV calibration is intentionally transparent, but it is not the final shape I would want for production. A LightGBM calibration layer can learn a smooth calibration surface across SNPs and individuals without manually binning every class of SNP.

STITCHV2 already has a learned calibration path in:

```text
calibrate_genotype_posterior_full_stack(...)
```

It works in three stages:

1. Keep the raw HMM GP immutable as an input feature.
2. Fit a multiclass LightGBM model that predicts `P(GT=0), P(GT=1), P(GT=2)` from HMM GP, read, sample, and variant features.
3. Fit a binary LightGBM call-correctness model that predicts `P(hard call is correct)`, then use that probability for no-calling.

The learned model can be trained on a few hundred representative SNPs and a subset of samples, then applied block-by-block to the full dataset. Use `--calibration-train-site-fraction` to fit on a fraction of labeled SNPs. The important anti-leakage rule is the same as masked-CV: fitting labels must be disjoint from evaluation labels.

By default, the learned LightGBM path does not apply temperature/blend calibration before fitting. The original HMM GP remains visible to the model as `gp0`, `gp1`, `gp2`, `max_prob`, `margin`, `entropy`, and `posterior_var`. If a debugging run needs the old behavior, use:

```bash
--calibration-lightgbm-use-fixed-stage0
```

The older local-context LightGBM stage is also off by default and can be enabled only when wanted:

```bash
--calibration-lightgbm-use-block-context
```

### Current Learned Features

The learned calibration feature matrix now includes:

```text
dosage
depth
log_depth
alt_fraction
support_mask
ref_count
alt_count
other_count
total_count
GP0, GP1, GP2
max_GP
GP margin
posterior entropy
posterior variance
MAF
INFO
HWE deviation
HWE chi-square statistic
HWE p-value
-log10(HWE p-value)
site ref/alt/other read counts
site total read count
number of samples with site read support
fraction of samples with site read support
site mean depth
optional generation/sample metadata
```

This directly addresses the missing predictors from the earlier calibration: MAF, HWE p-value, site-level read support, and individual-level read support are now exposed to the LightGBM posterior and call-correctness models.

The calibration metadata also records LightGBM feature importance:

```text
posterior_feature_importance
call_correctness.feature_importance
```

These are the first diagnostic plots/tables to inspect when asking whether site coverage, individual reads, HWE p-value, MAF, or raw GP dominates the calibration.

### Why LightGBM Is Better Than Manual Bins

Manual bins say:

```text
if MAF in [0.01, 0.05) and entropy in [0.25, 0.50), use threshold T
```

LightGBM can instead learn interactions such as:

```text
rare variant + high HWE p-value + many site reads + low individual depth
common variant + low entropy + balanced allelic reads
near-fixed alternate site + high support + GT2-favoring HWE prior
```

This is a better match to the biology and the failure modes. It can learn a continuous call-correctness score instead of forcing all SNPs in a coarse class to share one threshold.

### Is Temperature The Right Parameter?

Temperature is still useful, but only as a simple parametric fallback. It controls posterior sharpness around dosage:

```text
low temperature -> sharper posterior
high temperature -> softer posterior
```

Temperature alone is not a complete calibration model because it does not know MAF, HWE, read support, mapping quality, founder uncertainty, or sex/ploidy context. In practice, temperature should be either:

```text
a small grid-searched fallback parameter
```

or:

```text
a learned function of MAF, HWE, INFO, depth, entropy, and read balance
```

The LightGBM posterior model effectively learns that function without explicitly naming it "temperature". It can sharpen, soften, or re-orient genotype probabilities depending on the full feature vector. If we later want an interpretable parametric version, the natural model would be:

```text
temperature_j = f(MAF_j, HWE_pvalue_j, INFO_j, entropy_j, site_depth_j)
blend_j = g(MAF_j, HWE_pvalue_j, INFO_j, entropy_j, site_depth_j)
no_call_threshold_ij = h(max_GP_ij, entropy_ij, depth_ij, site_support_j, MAF_j, HWE_pvalue_j)
```

LightGBM is the practical near-term implementation of `f`, `g`, and `h`.

## Current Limitations

### One global grid per MAF bin is crude

The current masked-CV optimizer chooses one parameter tuple per MAF bin. That is useful and easy to debug, but it does not distinguish other important variant classes.

### Rare bins need more careful training

If a rare MAF bin has very few labeled examples, the selected parameters can be unstable. The implementation falls back when bins lack training rows, but rare variants deserve a richer borrowing strategy.

### HWE is used in two ways

The masked-CV grid uses an HWE prior and a simple squared HWE penalty. The learned LightGBM path receives HWE deviation, chi-square statistic, p-value, and `-log10(p-value)` as features. A future version could replace the chi-square p-value with an exact test, inbreeding coefficients, or pedigree-aware expectations.

### Confidence is not yet well calibrated for no-calling

The paper benchmark shows max GP is not a useful no-call score in all cases. For fixed calibration it creates all-or-none behavior; for masked-CV it is nearly always above `0.99`.

## Future Improvements

### Class-Based Batch Calibration

A stronger and more efficient calibration scheme would group SNPs into calibration classes, fit once per class, and apply to all SNPs in that class.

Potential class keys:

```text
MAF bin
HWE deviation bin
INFO bin
mean depth bin
missingness/support bin
n_founders_seen bin
founder entropy bin
read balance bin
recombination-rate bin
chromosome/ploidy class
```

This is now the fallback path rather than the preferred production path. The preferred production path is the LightGBM call-correctness model because it can learn these interactions continuously instead of discretizing them by hand.

For example:

```text
class_id = (
    maf_bin,
    hwe_bin,
    n_founders_seen_bin,
    depth_bin,
    info_bin,
)
```

Then the calibrator could fit:

```text
parameters[class_id] = argmin validation loss for class_id
```

and apply:

```text
P_calibrated[:, variants_in_class, :] = calibrate(P_raw, dosage, parameters[class_id])
```

This would be more stable than per-SNP fitting and more specific than a single global MAF-bin fit.

### n_founders_seen

A useful feature would be `n_founders_seen`, the effective number of founders contributing to a SNP. It could be computed from posterior founder usage or founder dosages.

Possible definitions:

```text
founder_usage[k, j] = mean_i posterior copies assigned to founder k at SNP j
n_founders_seen[j] = count_k founder_usage[k, j] > threshold
```

or an entropy-based effective count:

```text
p_kj = founder_usage[k, j] / sum_k founder_usage[k, j]
n_effective_founders[j] = exp(-sum_k p_kj log(p_kj))
```

Variants with low founder diversity may need different calibration from variants with many plausible founder haplotypes.

### Runtime-Efficient Block Calibration

The current grid search loops over all parameter combinations and all variants in each MAF bin. That is fine for diagnostic benchmarks but can be optimized.

Recommended scalable design:

1. Process variants in coarse blocks.
2. Compute class features per SNP.
3. Sample a bounded number of training cells per class.
4. Fit calibration parameters per class.
5. Cache fitted parameters by class key.
6. Apply cached parameters to all SNPs in that class.

This avoids recalibrating every SNP independently.

### Streaming Or Dask Calibration

For chromosome-scale data, calibration should be done in two passes:

First pass:

```text
collect class summaries and masked training rows
```

Second pass:

```text
apply fitted class calibrations block-by-block
```

Dask can orchestrate this naturally:

```text
task 1: summarize block/class training rows
task 2: reduce summaries and fit class calibrators
task 3: apply class calibrators to block outputs
```

JAX should remain the HMM leaf engine. Calibration can remain NumPy/CPU unless profiling shows it dominates runtime.

### Better No-Call Scoring

Instead of using max GP alone, we should train or derive a call-correctness score:

```text
P(call is correct | max_GP, margin, entropy, depth, MAF, INFO, HWE, read balance)
```

Then no-calling uses:

```text
call if P(correct) >= threshold
```

This is likely to be better than STITCH-style max-GP thresholding for STITCHV2, because max GP currently saturates or collapses.

### Calibrating Rare Variants

Rare variants should use asymmetric loss: false alternate calls are costly, but missing true rare alternates also matters.

Possible rare-variant objectives:

```text
loss = NLL
     + lambda_false_alt * false_alt_penalty
     + lambda_sensitivity * missed_alt_penalty
     + lambda_maf * allele_frequency_consistency
```

The weights should be tuned separately for:

```text
MAF < 0.01
0.01 <= MAF < 0.05
MAF >= 0.05
```

### Sex Chromosome And Ploidy-Aware Calibration

HWE should be disabled or replaced for:

```text
male chrX haploid regions
chrY
MT
mixed-ploidy variant groups
```

For non-diploid ploidy, genotype posterior length is `ploidy + 1`, and the current diploid masked-CV search should be generalized.

## Reimplementation Checklist

A new implementation should reproduce these steps:

1. Load raw GP, dosage, optional depth, and masked truth labels.
2. Construct a boolean training mask disjoint from evaluation labels.
3. Estimate per-variant MAF from training truth, with posterior fallback.
4. Assign each variant to a MAF bin.
5. For each bin, grid-search temperature, blend, dosage scale, and dosage offset.
6. For each candidate, compute dosage-derived GP.
7. Blend raw GP with dosage-derived GP.
8. Score candidates using NLL, Brier loss, dosage MSE, and optional HWE penalty.
9. Select the lowest-scoring candidate per bin.
10. Apply selected parameters to all variants in the bin.
11. Normalize posterior probabilities.
12. Make hard calls with argmax or STITCH-style no-call threshold.
13. Report per-bin selected parameters and held-out QC metrics.
14. Verify no-call monotonicity: increasing threshold must never increase call count.
15. Evaluate calibration on held-out labels, never on the same labels used to fit.

## Practical Recommendation

For now, use masked-CV calibration as a diagnostic and optional advanced mode:

```bash
stitchv2 run \
  ... \
  --calibration-mode masked_cv \
  --calibration-maf-bins 0,0.01,0.05,0.5 \
  --calibration-hwe-prior-weights 0,0.25,0.5,1 \
  --calibration-hwe-weight 0.02
```

When enough masked truth or microarray labels are available, enable the learned calibrator:

```bash
stitchv2 run \
  ... \
  --calibration-mode masked_cv \
  --use-lightgbm-calibrator \
  --calibration-train-site-fraction 0.20 \
  --calibration-max-train-rows 750000 \
  --calibration-context-window 25 \
  --calibration-block-snps 64
```

That path learns posterior probabilities and no-call/call-correctness from MAF, HWE p-value, read support, entropy, INFO, sample metadata, and local context. It is the better long-term answer to "temperature should depend on MAF and HWE": yes, it should, and the learned model can represent that dependency directly.

But do not expect calibration alone to make STITCHV2 exactly match STITCH on the paper mouse panel. The real-data results show calibration repairs part of the dosage-scale error, while the remaining gap points back to HMM/founder/read-emission parity.
