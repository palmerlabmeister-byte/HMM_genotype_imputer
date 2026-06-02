# STITCHV2 Documentation Index

This directory contains the current STITCHV2 user and technical documentation. Benchmark reports and historical design notes are kept for traceability, but the files below should be treated as the canonical starting points.

## Start Here

- [../README.md](../README.md): public overview, installation, quick start, outputs, and troubleshooting.
- [CLI_REFERENCE.md](CLI_REFERENCE.md): complete command and flag reference for the installed `stitchv2` entrypoint.
- [stitchv2_tutorial.md](stitchv2_tutorial.md): hands-on CLI tutorial.
- [WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md](WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md): whole-genome runbook with first-pass imputation, pedigree QC, and second-pass pedigree-aware imputation.
- [DISCOVER_POSITIONS_AND_FOUNDER_PLINK.md](DISCOVER_POSITIONS_AND_FOUNDER_PLINK.md): SNP/indel discovery pre-pass and how discovered positions relate to founder PLINK files.
- [LOW_RANK_K9_PRODUCTION_RUN.md](LOW_RANK_K9_PRODUCTION_RUN.md): K=9, 8 immutable founders plus 1 mutable founder, factorized-transition production workflow.

For a live parser check in your active environment, run:

```bash
stitchv2 run --help
```

## Technical Deep Dives

- [deep_dive/01_model_deep_dive.md](deep_dive/01_model_deep_dive.md): HMM, founder behavior, ploidy, calibration, transitions, and execution modes.
- [deep_dive/02_inputs_outputs_deep_dive.md](deep_dive/02_inputs_outputs_deep_dive.md): input schemas, output schemas, and export behavior.
- [deep_dive/03_cached_files_deep_dive.md](deep_dive/03_cached_files_deep_dive.md): compact evidence cache layout and reuse.
- [deep_dive/04_pedigree_transmission_deep_dive.md](deep_dive/04_pedigree_transmission_deep_dive.md): pedigree QC and transmission adjustment internals.
- [deep_dive/05_improvements_over_stitch.md](deep_dive/05_improvements_over_stitch.md): STITCHV2 behavior compared with original STITCH.

## Historical Or Generated Docs

The following files are archived for traceability and are not the canonical source for current usage:

- [legacy/](legacy/): legacy root docs and historical reports moved out of the publication-facing docs path.
- [legacy/CALIBRATION_MASKED_CV_REPORT.md](legacy/CALIBRATION_MASKED_CV_REPORT.md): historical masked-CV calibration report.
- [legacy/STITCHV2_MODEL_DEEP_DIVE.ipynb](legacy/STITCHV2_MODEL_DEEP_DIVE.ipynb): notebook version of the model deep dive.
- [legacy/stitchv2_tutorial.ipynb](legacy/stitchv2_tutorial.ipynb): notebook version of the tutorial.
- [legacy/whole_genome_stitchv2_pedigree_qc.ipynb](legacy/whole_genome_stitchv2_pedigree_qc.ipynb): notebook version of the whole-genome pedigree workflow.
- [legacy/tutorial_deep/README.md](legacy/tutorial_deep/README.md): generated notebook series.

Stable benchmark instructions live in [../final_benchmark.md](../final_benchmark.md). Benchmark results belong in [../final_benchmark_results.md](../final_benchmark_results.md) or explicitly named dataset-specific report files, not in stable tutorial docs.
