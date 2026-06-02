
# STITCHV2 Deep Tutorial Notebook Series

Run from the repository root with:

```bash
cd /home/bonnie/Documents/codex/STITCHV2
export PYTHONPATH=$PWD/src
export MPLCONFIGDIR=/tmp/mpl
jupyter lab docs/legacy/tutorial_deep
```

If JupyterLab is not installed in the current env, open these `.ipynb` files from any JupyterLab server that can use the repository Python environment. The paired validation run has already executed their Python cells with the project conda Python.

Fresh benchmark artifacts used by notebooks 05 and 06 are under:

```text
benchmark_runs/tutorial_deep_fresh_2026-05-01/
```

Notebook order:

1. `01_model.ipynb`: model equations and read-aware HMM intuition.
2. `02_parameters_stitch_crosswalk.ipynb`: detailed STITCHV2 parameter table and closest STITCH equivalent.
3. `03_inputs_cli_interactive.ipynb`: required columns, BAM/PLINK/parquet examples, CLI shape, and working Python API run.
4. `04_pedigree_ploidy_sex_chromosomes.ipynb`: pedigree smoothing, chrX/chrY/MT ploidy, and runnable mixed-ploidy examples.
5. `05_calibration_metrics.ipynb`: calibration explanation plus aggregate/per-SNP violin plots for missingness, accuracy, R2, F1, and INFO.
6. `06_scalability_dask_memory.ipynb`: O(N)/chunking intuition, Dask diagnostics, memory planning, and 10,000-sample benchmark visualization.

Additional focused tutorial:

- `../pedigree_tutorial.md`: detailed pedigree input formats, embedded `father`/`mother` columns in `samples.parquet`, `smooth`/`kinship`/`transmission` equations, efficiency, and validation benchmark results.
