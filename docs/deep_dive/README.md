# STITCHV2 Deep Dive Documentation

This directory is a descriptive reference for how STITCHV2 works internally and how to run it in a way that is reproducible, inspectable, and comparable to original STITCH.

The files are intentionally separated by topic:

- [01_model_deep_dive.md](01_model_deep_dive.md): the probabilistic model, HMM, founder behavior, fragment likelihoods, calibration, ploidy handling, batching, JAX, and Dask.
- [02_inputs_outputs_deep_dive.md](02_inputs_outputs_deep_dive.md): required inputs, optional inputs, output directories, output schemas, and how to read the outputs.
- [03_cached_files_deep_dive.md](03_cached_files_deep_dive.md): compact evidence cache layout, Parquet/Zarr files, sample-level cache reuse, validation rules, and memory/speed tradeoffs.
- [04_pedigree_transmission_deep_dive.md](04_pedigree_transmission_deep_dive.md): pedigree QC, relationship R2, UMAP edge plots, smooth/kinship/transmission modes, and how curated pedigrees feed back into imputation.
- [05_improvements_over_stitch.md](05_improvements_over_stitch.md): what STITCHV2 keeps from original STITCH, what it changes, and where the current advantages and caveats are.

Related production guides:

- [../CLI_REFERENCE.md](../CLI_REFERENCE.md): complete CLI command and flag reference.
- [../WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md](../WHOLE_GENOME_PEDIGREE_QC_TUTORIAL.md): whole-genome first pass, global pedigree QC, and second pass.
- [../DISCOVER_POSITIONS_AND_FOUNDER_PLINK.md](../DISCOVER_POSITIONS_AND_FOUNDER_PLINK.md): SNP/indel discovery and founder PLINK preparation.
- [../LOW_RANK_K9_PRODUCTION_RUN.md](../LOW_RANK_K9_PRODUCTION_RUN.md): K=9, 8 immutable founders plus 1 mutable founder, factorized-transition workflow.

These documents describe the intended and current implementation. They are not benchmark reports. Benchmark outputs should stay in benchmark-specific result files, such as `final_benchmark_results.md` or dataset-specific reports under `benchmark_HSrats/`.
