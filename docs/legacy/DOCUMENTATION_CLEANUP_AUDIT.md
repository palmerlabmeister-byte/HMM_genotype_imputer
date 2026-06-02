# STITCHV2 Documentation Cleanup Audit

Audit date: 2026-06-01

This file is a non-destructive documentation cleanup plan. No documentation files should be removed based on this audit alone. The goal is to separate current production documentation from historical reports, identify stale claims, and define the edits needed before publication.

Implementation status: a first cleanup pass was applied on 2026-06-01. It added `docs/README.md`, corrected the main README and canonical deep-dive Markdown files, moved legacy or historical Markdown files into `docs/legacy/`, and refreshed the most hazardous stale wording around cache format, calibration, read backends, discovery, JAX flags, exact streaming, and factorized transitions. Notebook files still need regeneration from the corrected Markdown and current CLI help.

## Current Implementation Facts To Preserve

These points were checked against the current package and should be treated as the documentation source of truth unless the code changes:

- Main CLI entrypoint is `stitchv2`.
- Main run command is `stitchv2 run`.
- Default SNP block mode is `--snp-block-mode exact_streaming`.
- Approximate block modes are explicit: `independent_approx` and `density_balanced_overlap`.
- Default compact evidence cache format is only `parquet_zarr`.
- `npz` should not be documented as a supported production cache format.
- Default fragment likelihood mode is `--fragment-likelihood-mode replace`.
- `augment` is optional and should be described as an experimental or model-extension path, not parity default.
- Default transition model is `--transition-model stitch_parity`.
- `--transition-model factorized` is opt-in and experimental.
- Factorized transition mode reduces transition parameter/output representation; it does not currently remove the exact founder-state HMM state space.
- Default transition output is compact.
- Default xi-style output is `--store-xi per-snp`.
- Full xi is opt-in via `--store-xi full` and can be very large.
- Gamma summaries are default via `--write-gamma summary`; full gamma is opt-in.
- Standard calibration is `--calibration-mode standard_callability`.
- The standard callability model defaults to LightGBM through `--calibration-callability-model lightgbm`.
- `--use-lightgbm-calibrator` is not the primary standard-callability switch. It is an older optional posterior/block-context calibration path and should not be used in primary tutorials unless clearly explained.
- Current read-stream backends are `auto`, `python`, `htslib`, `snp_only_bamreader`, `stitch_style_bamreader`, and `variant_aware_bamreader`.
- `variant_aware_bamreader` is the intended compiled production backend when available, especially for targeted SNP plus indel evidence.
- `stitch_style_bamreader` is the strict SNP compatibility/debug backend.
- Standard imputation is target-variant based. Discovery is a separate pre-pass.
- `stitchv2 discover-positions` can discover SNP, insertion, and deletion candidates through `--variant-types snp,ins,del`.
- Targeted indel evidence requires normalized biallelic `CHR`, `POS`, `REF`, `ALT` records and the variant-aware reader.
- BCF export exists for compatibility. Native Parquet/Zarr outputs are preferred.
- VCF should not be presented as a default or preferred output path.
- Pedigree QC is a first-pass/curation/second-pass workflow. Whole-genome pedigree curation should aggregate evidence across informative chromosomes before the final pedigree-aware run.

## Documentation Status

| File or group | Status | Keep? | Cleanup action |
| --- | --- | --- | --- |
| `README.md` | Important public entrypoint, partly stale | Yes | Update discovery, calibration, cache, reader, and parameter-table sections. |
| `docs/deep_dive/README.md` | Important technical index | Yes | Add links to the low-rank production guide, discover-positions guide, whole-genome pedigree guide, and this audit or future docs index. |
| `docs/deep_dive/01_model_deep_dive.md` | Important model reference, partly stale | Yes | Remove stale CLI flags and add current exact-streaming, xi/gamma, factorized, calibration, and reader semantics. |
| `docs/deep_dive/02_inputs_outputs_deep_dive.md` | Important IO reference, mostly current | Yes | Add stronger variant-aware indel schema notes, current cache-output paths, xi/gamma output behavior, and BCF-only caveat. |
| `docs/deep_dive/03_cached_files_deep_dive.md` | Important cache reference, partly stale | Yes | Remove `npz` compatibility language; document only compressed Parquet/Zarr cache, dense-count Zarr, sample partial-hit behavior, and cache metadata. |
| `docs/deep_dive/04_pedigree_transmission_deep_dive.md` | Important pedigree reference | Yes | Keep. Cross-link to the whole-genome pedigree tutorial and clarify global multi-chromosome QC before second pass. |
| `docs/deep_dive/05_improvements_over_stitch.md` | Important comparison/reference, partly stale | Yes | Update read backend list, remove stale JAX precompile flags, add variant-aware indel support, exact streaming, and cache correctness caveats. |
| `docs/LOW_RANK_K9_PRODUCTION_RUN.md` | Current production-style guide | Yes | Keep. Add one warning that factorized transitions are not a low-rank approximate inference mode. |
| `docs/DISCOVER_POSITIONS_AND_FOUNDER_PLINK.md` | Current production-style guide | Yes | Keep. Link from README and inputs/outputs docs. |
| `docs/WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md` | Current production-style guide, minor drift | Yes | Update backend advice to prefer `auto`/`variant_aware_bamreader` for production and `stitch_style_bamreader` for strict SNP parity. Add exact-streaming and cache dense-count guidance. |
| `docs/pedigree_tutorial.md` | Useful conceptual tutorial, overlaps with deep dive | Yes, but consolidate later | Either merge into the whole-genome pedigree guide or keep as a shorter conceptual primer with links to the canonical workflow. |
| `docs/legacy/CALIBRATION_MASKED_CV_REPORT.md` | Historical report, partly superseded | Keep as historical | Moved to legacy folder with a banner saying it is not the current default calibration guide. |
| `docs/stitchv2_tutorial.md` | Useful but stale tutorial | Rewrite/regenerate | Regenerate from current CLI and make it the main hands-on tutorial, or replace with pointers to the new production guides. |
| `docs/legacy/stitchv2_tutorial.ipynb` | Notebook version of stale tutorial | Regenerate | Rebuild after the markdown tutorial is corrected. |
| `docs/stitchv2_tutorial_synthetic.md` | Useful synthetic tutorial, stale flags | Update | Update read backend list, default fragment mode, calibration wording, cache format, and block modes. |
| `docs/legacy/STITCHV2_MODEL_DEEP_DIVE.ipynb` | Notebook deep dive, likely stale | Regenerate or keep historical | Sync with `docs/deep_dive/01_model_deep_dive.md` after that file is updated. |
| `docs/legacy/tutorial_deep/*.ipynb` | Rich notebook series, likely stale parameter surfaces | Regenerate or keep historical | Regenerate the CLI crosswalk from current `stitchv2 run --help`; update examples and figures. |
| `docs/legacy/tutorial_deep/stitchv2_run_parameter_crosswalk.csv` | Stale generated parameter table | Regenerate | Rebuild from current CLI after deciding canonical parameter grouping. |
| `docs/legacy/whole_genome_stitchv2_pedigree_qc.ipynb` | Useful whole-genome notebook | Regenerate after markdown update | Keep paired with `docs/WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md`. |
| `docs/legacy/TUTORIAL.md` | Root-level legacy tutorial, stale flags | Keep as legacy | Moved to legacy folder. Replace with canonical tutorial after rewrite if needed. |
| `docs/legacy/MODEL.md` | Root-level legacy model doc | Keep as legacy | Moved to legacy folder. Canonical model doc is `docs/deep_dive/01_model_deep_dive.md`. |
| `docs/legacy/INPUTS_OUTPUTS.md` | Root-level legacy IO doc | Keep as legacy | Moved to legacy folder. Canonical IO doc is `docs/deep_dive/02_inputs_outputs_deep_dive.md`. |
| `docs/legacy/FACTORIZE_RANK3_PEDIGREE_CLI_REPORT.md` | Historical factorized/pedigree report | Keep as historical | Moved to legacy folder with pointer to `docs/LOW_RANK_K9_PRODUCTION_RUN.md`. |
| `docs/legacy/BENCHMARKS.md`, `docs/legacy/FULL_BENCHMARK_REPORT.md`, `final_benchmark.md`, `final_benchmark_results.md` | Benchmark specs/reports, not general docs | Keep separate | Historical benchmark docs moved to legacy. Stable protocol/results remain at root. |
| `benchmark_HSrats/*.md`, `benchmark_runs/*.md` | Dataset-specific reports | Keep as results | Do not use as canonical user instructions except when explicitly cited as a benchmark record. |

## Global Corrections Needed

### 1. Calibration Language

Several docs imply that `--use-lightgbm-calibrator` is the main way to enable LightGBM calibration. This is no longer the right public explanation.

Correct wording:

```text
The default calibration mode is standard callability calibration.
It keeps raw HMM genotype probabilities as the probability object, trains a lightweight callability model, and chooses local hard-call gating/fallback behavior.
The default callability model family is LightGBM through --calibration-callability-model lightgbm.
```

Use `--use-lightgbm-calibrator` only in advanced documentation, and explain that it is the older optional posterior/block-context calibrator.

Files to update:

- `README.md`
- `docs/stitchv2_tutorial.md`
- `docs/legacy/TUTORIAL.md`
- `docs/legacy/CALIBRATION_MASKED_CV_REPORT.md`
- `docs/legacy/tutorial_deep/*.ipynb`

### 2. Cache Format

Docs must stop presenting NPZ as an acceptable or current cache format. The code currently exposes:

```bash
--compact-evidence-cache-format parquet_zarr
```

Correct wording:

```text
Compact fragment evidence is stored in compressed Parquet/Arrow-style partitions.
Dense count matrices, when requested, are stored in chunked compressed Zarr.
NPZ is not a production cache format and should not be recommended.
```

Files to update:

- `README.md`
- `docs/deep_dive/03_cached_files_deep_dive.md`
- any tutorial notebooks that mention NPZ

### 3. Discovery Is SNP Plus Indel Candidate Discovery

The README still describes `discover-positions` as a candidate SNP-table pre-pass. The current discovery path can emit SNP, insertion, and deletion candidates.

Correct wording:

```text
discover-positions creates reviewable candidate variant tables for SNPs, insertions, and deletions.
It does not create founder genotypes and does not automatically trust discovered variants for imputation.
```

Files to update:

- `README.md`
- `docs/deep_dive/02_inputs_outputs_deep_dive.md`
- `docs/stitchv2_tutorial.md`

The detailed guide to link is:

- `docs/DISCOVER_POSITIONS_AND_FOUNDER_PLINK.md`

### 4. Targeted Indel Support

Docs should consistently state that first production indel support is targeted, normalized, and biallelic.

Required position-table columns for indel-aware runs:

```text
CHR, POS, REF, ALT
```

Optional:

```text
VARIANT_TYPE
```

Rules to document:

- Multiallelic records should be split before STITCHV2.
- `variant_aware_bamreader` is required for targeted insertion/deletion evidence.
- `--max-indel-len` controls small indel length, default `50`.
- Nested or complex variants are out of scope for the current production path.

### 5. JAX Flags

Docs mention non-existent CLI flags:

```bash
--jax-precompile
--jax-aot-compile
```

These should be removed from user-facing CLI examples. Current user-facing JAX controls include:

```bash
--hmm-backend auto|numpy|jax|torch
--jax-sample-batch-size
--jax-persistent-cache-dir
--no-jax-bucket-batch-shapes
--no-jax-count-emission-kernel
--no-jax-fragment-emission-kernel
```

Files to update:

- `docs/deep_dive/01_model_deep_dive.md`
- `docs/deep_dive/05_improvements_over_stitch.md`

### 6. Sequencing Error Rate Flag

`docs/deep_dive/01_model_deep_dive.md` mentions:

```bash
--sequencing-error-rate
```

The current CLI does not expose this flag. Either remove it from CLI documentation or move it to an internal-configuration note only if the code still uses it internally.

### 7. Fragment Likelihood Default

Some tutorials use:

```bash
--fragment-likelihood-mode augment
```

The production default is:

```bash
--fragment-likelihood-mode replace
```

Docs should say:

- `replace` is the parity/default path.
- `augment` is an optional extension that may add information but should be benchmarked separately.

Files to update:

- `docs/legacy/TUTORIAL.md`
- `docs/stitchv2_tutorial.md`
- `docs/stitchv2_tutorial_synthetic.md`
- any notebook examples that still use `augment` as the routine default

### 8. Read Backend Guidance

The recommended wording should be:

```text
Use --read-stream-backend auto for normal production runs.
When the compiled extension is available, auto should select the native path.
Use variant_aware_bamreader explicitly for targeted SNP plus indel evidence.
Use stitch_style_bamreader for strict SNP-only STITCH compatibility checks.
Use python/htslib/snp_only_bamreader mostly for debugging and backend benchmarks.
```

Files to update:

- `README.md`
- `docs/deep_dive/05_improvements_over_stitch.md`
- `docs/WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md`
- `docs/stitchv2_tutorial_synthetic.md`

### 9. Exact Streaming And Approximate Blocks

Docs should consistently separate exact production mode from approximate independent-block modes.

Correct wording:

```text
exact_streaming is the default production mode. It carries HMM boundary state across SNP blocks.
independent_approx and density_balanced_overlap are approximate independent-block modes for QC, stress tests, and exploratory runs.
Production parity claims should use exact_streaming unless explicitly labeled otherwise.
```

Files to update:

- `README.md`
- `docs/deep_dive/01_model_deep_dive.md`
- `docs/LOW_RANK_K9_PRODUCTION_RUN.md`
- tutorials and benchmark runbooks

### 10. Xi, Gamma, And Transition Output

Docs should distinguish three separate concepts:

- `--store-xi per-snp`: compressed per-SNP transition summaries; production default.
- `--store-xi full`: full pair posterior output; opt-in and potentially very large.
- `--write-gamma summary|off|full`: genotype/founder posterior summaries or full gamma trace.

Also clarify:

- `--transition-output compact` is enough for STITCH-parity transition parameters.
- `--transition-output factorized` writes factorized transition parameters when using `--transition-model factorized`.
- `--transition-output full` is exact but expensive and should not be a normal default.

Files to update:

- `README.md`
- `docs/deep_dive/01_model_deep_dive.md`
- `docs/deep_dive/02_inputs_outputs_deep_dive.md`
- `docs/LOW_RANK_K9_PRODUCTION_RUN.md`

### 11. Low-Rank Factorized Transition Scope

Current docs should avoid implying that factorized mode is a full approximate low-rank inference mode.

Correct wording:

```text
The factorized transition model changes the transition parameterization and output representation.
The current HMM still performs exact inference over the founder-state space.
Future inference modes such as low_rank_pair or sparse_pair would require a larger inference rewrite.
```

Files to update:

- `docs/LOW_RANK_K9_PRODUCTION_RUN.md`
- `docs/deep_dive/01_model_deep_dive.md`
- `docs/deep_dive/05_improvements_over_stitch.md`

### 12. BCF/VCF Guidance

Docs should be firm:

- Parquet/Zarr are preferred outputs.
- BCF export is for downstream tools that require it.
- VCF should not be recommended as a normal output path.

Files to update:

- `README.md`
- `docs/deep_dive/02_inputs_outputs_deep_dive.md`
- `docs/legacy/TUTORIAL.md`
- any old benchmark/tutorial examples

### 13. Pedigree Workflow

Pedigree docs are mostly aligned, but the workflow should be made more explicit everywhere:

1. Run STITCHV2 first pass without enforcing the declared pedigree, or with pedigree mode off/weak.
2. Run `stitchv2 pedigree-qc` across multiple informative chromosomes where possible.
3. Review UMAP edge plots and long-edge/discrepant-parent flags.
4. Use the curated pedigree for the second pass.
5. Use `pedigree_mode=transmission` when the goal is parent/offspring-aware imputation, including missing parents supported by many genotyped offspring.

Files to update:

- `README.md`
- `docs/WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md`
- `docs/pedigree_tutorial.md`
- `docs/deep_dive/04_pedigree_transmission_deep_dive.md`

### 14. Root-Level Duplicate Docs

Root-level docs are useful but now duplicate the newer `docs/` structure:

- `docs/legacy/TUTORIAL.md`
- `docs/legacy/MODEL.md`
- `docs/legacy/INPUTS_OUTPUTS.md`
- `docs/legacy/FACTORIZE_RANK3_PEDIGREE_CLI_REPORT.md`

Do not delete them yet. Recommended non-destructive step:

```text
Legacy files now live in `docs/legacy/`. Keep a short banner in each one:
"This file is retained for historical context. The current canonical documentation is ..."
```

After review, either keep them in `docs/legacy/` or replace them with short pointer files if the publication package should expose only canonical docs.

### 15. Notebook Regeneration

Notebook docs should not be hand-edited as the first cleanup step. Update the markdown source first, then regenerate notebooks.

Regenerate:

- `docs/legacy/stitchv2_tutorial.ipynb`
- `docs/legacy/STITCHV2_MODEL_DEEP_DIVE.ipynb`
- `docs/legacy/whole_genome_stitchv2_pedigree_qc.ipynb`
- `docs/legacy/tutorial_deep/*.ipynb`

The parameter crosswalk should be generated from the current CLI instead of maintained manually.

## Suggested Cleanup Order

### Phase 1: Non-Destructive Labeling

1. Add a top-level `docs/README.md` that explains which docs are canonical.
2. Keep historical/legacy banners in reports and moved duplicate docs under `docs/legacy/`.
3. Keep the historical banner in `docs/legacy/CALIBRATION_MASKED_CV_REPORT.md`.
4. Keep benchmark reports clearly separate from stable user docs.

### Phase 2: Correct Canonical Markdown

1. Update `README.md`.
2. Update `docs/deep_dive/01_model_deep_dive.md`.
3. Update `docs/deep_dive/02_inputs_outputs_deep_dive.md`.
4. Update `docs/deep_dive/03_cached_files_deep_dive.md`.
5. Update `docs/deep_dive/05_improvements_over_stitch.md`.
6. Update `docs/WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md`.
7. Refresh `docs/stitchv2_tutorial.md` as the main complete hands-on tutorial.

### Phase 3: Regenerate Derived Docs

1. Regenerate notebook tutorials from corrected markdown or notebook source.
2. Regenerate the CLI parameter crosswalk from `stitchv2 run --help`.
3. Verify every command example starts with `stitchv2`, not `python -c`.
4. Verify every cache example uses `parquet_zarr`.
5. Verify every production benchmark example uses one model at a time.

### Phase 4: Publication Polish

1. Decide whether files in `docs/legacy/` should stay there for historical context or become short pointer files before publication.
2. Add a compact "Start Here" path:
   - README
   - quick install/env
   - whole-genome tutorial
   - deep-dive technical reference
   - benchmark specification
3. Add a "Current Limitations" section that is honest but production-ready:
   - target-variant imputation, not de novo calling during HMM;
   - discovery candidates must be reviewed;
   - biallelic normalized indels only in first production indel path;
   - factorized transitions are experimental;
   - approximate block modes are approximate.

## Minimal Publication-Ready Doc Set

The final publication-facing docs should probably be:

- `README.md`: public overview and quick start.
- `docs/README.md`: documentation index.
- `docs/stitchv2_tutorial.md`: complete CLI and interactive tutorial.
- `docs/WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md`: whole-genome plus pedigree workflow.
- `docs/DISCOVER_POSITIONS_AND_FOUNDER_PLINK.md`: discovery and founder PLINK guide.
- `docs/LOW_RANK_K9_PRODUCTION_RUN.md`: K=9 low-rank/factorized production workflow.
- `docs/deep_dive/01_model_deep_dive.md`: model internals.
- `docs/deep_dive/02_inputs_outputs_deep_dive.md`: all input/output schemas.
- `docs/deep_dive/03_cached_files_deep_dive.md`: cache internals and reuse.
- `docs/deep_dive/04_pedigree_transmission_deep_dive.md`: pedigree mechanics.
- `docs/deep_dive/05_improvements_over_stitch.md`: comparison against original STITCH.
- `final_benchmark.md`: stable benchmark protocol with no results.
- `final_benchmark_results.md`: benchmark results only.

Everything else should be either regenerated, clearly labeled historical under `docs/legacy/`, or intentionally excluded from the publication-facing documentation set after review.
