from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import HMMConfig
from .fast_hmm import ordered_log_emission_to_unordered, reduce_unordered_diploid, unordered_diploid_forward_backward
from .cpu_kernels import numba_unordered_diploid_reduce, topk_offdiag_arrays, NUMBA_AVAILABLE
from .hmm import JAXStitchHMM, _uniform_offdiag_matrix


def _timeit(fn, *, repeat: int = 1):
    best = float("inf")
    last = None
    for _ in range(max(int(repeat), 1)):
        t0 = time.perf_counter()
        last = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, last




def _write_simple_zarr_matrix(root: Path, name: str, arr: np.ndarray, *, chunk_samples: int = 256, chunk_positions: int = 4096) -> float:
    t0 = time.perf_counter()
    array_dir = root / name
    array_dir.mkdir(parents=True, exist_ok=True)
    shape = tuple(int(x) for x in arr.shape)
    chunks = (min(int(chunk_samples), shape[0]), min(int(chunk_positions), shape[1])) + tuple(shape[2:])
    (array_dir / ".zarray").write_text(
        json.dumps({
            "zarr_format": 2,
            "shape": list(shape),
            "chunks": list(chunks),
            "dtype": np.dtype(arr.dtype).str,
            "compressor": None,
            "fill_value": "NaN" if np.issubdtype(arr.dtype, np.floating) else 0,
            "order": "C",
            "filters": None,
            "dimension_separator": ".",
        }, sort_keys=True),
        encoding="utf-8",
    )
    (array_dir / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["sample", "position"] + [f"axis_{i}" for i in range(2, arr.ndim)]}, sort_keys=True), encoding="utf-8")
    ranges = [range(0, dim, chunk) for dim, chunk in zip(shape, chunks, strict=False)]
    import itertools

    for starts in itertools.product(*ranges):
        slices = tuple(slice(st, min(st + ch, dim)) for st, ch, dim in zip(starts, chunks, shape, strict=False))
        idx = tuple(st // ch for st, ch in zip(starts, chunks, strict=False))
        (array_dir / ".".join(str(i) for i in idx)).write_bytes(np.ascontiguousarray(arr[slices]).tobytes(order="C"))
    return time.perf_counter() - t0

def run_synthetic_benchmark(
    *,
    output_dir: str | Path,
    n_samples: int = 64,
    n_positions: int = 2000,
    n_founders: int = 8,
    chunk_size: int = 500,
    repeat: int = 1,
    seed: int = 0,
    include_jax: bool = True,
) -> dict[str, object]:
    rng = np.random.default_rng(int(seed))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    k = int(n_founders)
    founder_alt = rng.beta(0.2, 0.2, size=(k, n_positions)).astype(np.float32)
    # Harden many sites to mimic fixed founder haplotypes.
    founder_alt = (founder_alt > 0.5).astype(np.float32)
    ref = rng.poisson(0.14, size=(n_samples, n_positions)).astype(np.float32)
    alt = rng.poisson(0.14, size=(n_samples, n_positions)).astype(np.float32)
    other = np.zeros_like(ref, dtype=np.float32)
    generations = rng.integers(40, 120, size=n_samples).astype(np.float32)
    positions = np.arange(n_positions, dtype=np.int64) * 1000 + 1
    cfg = HMMConfig(n_founders=k, ploidy_mode="diploid", ploidy=2, backend="numpy", exact_hmm_chunk_size=int(chunk_size))
    hmm = JAXStitchHMM(cfg)
    rates = hmm.recombination_from_positions(positions)
    switch = hmm._switch_probabilities(rates, generations)
    t_evidence0 = time.perf_counter()
    # Synthetic evidence-read phase: copy arrays to mimic materializing one cached evidence block.
    ref_e = np.asarray(ref, dtype=np.float32, order="C").copy()
    alt_e = np.asarray(alt, dtype=np.float32, order="C").copy()
    other_e = np.asarray(other, dtype=np.float32, order="C").copy()
    evidence_seconds = time.perf_counter() - t_evidence0
    t_emission0 = time.perf_counter()
    log_emission, _, founder_alt2 = hmm.emissions(founder_alt, ref_e, alt_e, other_e, None, None, None)
    emission_seconds = time.perf_counter() - t_emission0
    rows: list[dict[str, object]] = []

    def add(name: str, seconds: float, extra: dict[str, object] | None = None):
        row = {
            "benchmark": name,
            "seconds_best": float(seconds),
            "n_samples": int(n_samples),
            "n_positions": int(n_positions),
            "n_founders": int(k),
            "chunk_size": int(chunk_size),
        }
        if extra:
            row.update(extra)
        rows.append(row)

    add("synthetic_evidence_materialize", evidence_seconds, {"phase": "evidence_read"})
    add("count_emission_build", emission_seconds, {"phase": "emission"})

    dt, gamma_ordered = _timeit(lambda: hmm._forward_backward_numpy_diploid(log_emission, switch, k), repeat=repeat)
    add("numpy_ordered_full_gamma", dt, {"state_count": int(k * k), "phase": "hmm"})

    dt, gamma_chunk = _timeit(lambda: hmm._forward_backward_numpy_diploid_chunked(log_emission, switch, k), repeat=repeat)
    add("numpy_ordered_exact_chunked", dt, {"state_count": int(k * k), "phase": "hmm", "max_abs_delta_vs_full": float(np.max(np.abs(gamma_ordered - gamma_chunk)))})

    log_u = ordered_log_emission_to_unordered(log_emission)
    dt, gamma_u = _timeit(lambda: unordered_diploid_forward_backward(log_u, switch, k=k), repeat=repeat)
    red = reduce_unordered_diploid(gamma_u, founder_alt2, return_genotype_posterior=True)
    add("numpy_unordered_exact", dt, {"state_count": int(k * (k + 1) // 2), "dosage_mean": float(np.mean(red.dosage)), "phase": "hmm"})

    dt, red_numba = _timeit(lambda: numba_unordered_diploid_reduce(log_u, switch, founder_alt2), repeat=repeat)
    add("numba_unordered_k8_reduce", dt, {"state_count": int(k * (k + 1) // 2), "numba_available": bool(NUMBA_AVAILABLE), "phase": "hmm", "dosage_mean": float(np.mean(red_numba.dosage))})

    if k == 8:
        try:
            from .native_cpu import native_backend_status, native_k8_counts_reduce, native_k8_logu_from_counts, native_k8_apply_fragments_unordered, native_k8_unordered_reduce, native_diploid_low_rank_counts_reduce, native_diploid_sparse_topk_counts_reduce, native_diploid_sparse_topk_reduce, native_k8_counts_fragments_reduce
            status = native_backend_status()
            if bool(status.get("pybind11_counts_fused_available", False)):
                dt, red_cpp_fused = _timeit(
                    lambda: native_k8_counts_reduce(ref_e, alt_e, other_e, switch, founder_alt2, sequencing_error_rate=0.01, min_emission_prob=1e-5),
                    repeat=repeat,
                )
                add(
                    "cpp_fused_counts_unordered_k8",
                    dt,
                    {
                        "state_count": int(k * (k + 1) // 2),
                        "phase": "emission+hmm",
                        "dosage_mean": float(np.mean(red_cpp_fused["dosage"])),
                        "native_status": status,
                    },
                )
            if bool(status.get("pybind11_extension_available", False)):
                dt, red_cpp_logu = _timeit(lambda: native_k8_unordered_reduce(log_u, switch, founder_alt2), repeat=repeat)
                add(
                    "cpp_logu_unordered_k8",
                    dt,
                    {
                        "state_count": int(k * (k + 1) // 2),
                        "phase": "hmm",
                        "dosage_mean": float(np.mean(red_cpp_logu["dosage"])),
                        "native_status": status,
                    },
                )
            if bool(status.get("pybind11_k8_counts_fragments_fused_available", False)):
                empty_frag_offsets = np.zeros(int(n_samples) + 1, dtype=np.int64)
                dt, red_cpp_frag = _timeit(
                    lambda: native_k8_counts_fragments_reduce(
                        ref_e, alt_e, other_e, switch, founder_alt2,
                        fragment_sample_offsets=empty_frag_offsets,
                        fragment_center_idx=np.zeros(0, dtype=np.int64),
                        fragment_obs_offsets=np.zeros(1, dtype=np.int64),
                        fragment_obs_pos_idx=np.zeros(0, dtype=np.int64),
                        fragment_obs_code=np.zeros(0, dtype=np.int8),
                        fragment_obs_qual=np.zeros(0, dtype=np.float32),
                        sequencing_error_rate=0.01,
                        min_emission_prob=1e-5,
                        mode="replace",
                        rescale=True,
                        max_emission_matrix_difference=1e10,
                    ),
                    repeat=repeat,
                )
                add("cpp_fused_counts_fragments_unordered_k8", dt, {"state_count": int(k * (k + 1) // 2), "phase": "emission+fragment+hmm", "dosage_mean": float(np.mean(red_cpp_frag["dosage"])), "native_status": status})
        except Exception as exc:
            add("cpp_native_k8", float("nan"), {"phase": "emission+hmm", "error": str(exc)})

    uniform = _uniform_offdiag_matrix(k)
    sparse = uniform.copy()
    # Make an artificial top-3 sparse transition for measurement.
    top = min(3, max(k - 1, 1))
    for i in range(k):
        keep = [(i + j + 1) % k for j in range(top)]
        sparse[i, :] = 0.0
        sparse[i, keep] = 1.0 / float(top)
    dt, gamma_sparse = _timeit(lambda: hmm._forward_backward_numpy_diploid_factorized_chunked(log_emission, switch, sparse, k), repeat=repeat)
    add("numpy_sparse_factorized_dense_kernel", dt, {"state_count": int(k * k), "top_k": int(top), "phase": "hmm"})

    dt, red_sparse_numba = _timeit(lambda: numba_unordered_diploid_reduce(log_u, switch, founder_alt2, offdiag_matrix=sparse, sparse_top_k=top), repeat=repeat)
    add("numba_sparse_topk_unordered", dt, {"state_count": int(k * (k + 1) // 2), "top_k": int(top), "numba_available": bool(NUMBA_AVAILABLE), "phase": "hmm", "dosage_mean": float(np.mean(red_sparse_numba.dosage))})

    cfg_lr = HMMConfig(
        n_founders=k,
        ploidy_mode="diploid",
        ploidy=2,
        backend="numpy",
        exact_hmm_chunk_size=int(chunk_size),
        transition_model="low_rank_linear",
        transition_factor_rank=min(4, k),
    )
    hmm_lr = JAXStitchHMM(cfg_lr)
    dt, gamma_lr = _timeit(lambda: hmm_lr._forward_backward_numpy_diploid_low_rank_chunked(log_emission, switch, uniform, k), repeat=repeat)
    add(
        "numpy_low_rank_algebraic",
        dt,
        {
            "state_count": int(k * k),
            "rank": int(min(4, k)),
            "mean_abs_delta_vs_full": float(np.mean(np.abs(gamma_ordered - gamma_lr))),
            "phase": "hmm",
        },
    )

    try:
        from .native_cpu import native_backend_status, native_diploid_low_rank_counts_reduce, native_diploid_sparse_topk_counts_reduce
        status2 = native_backend_status()
        if bool(status2.get("pybind11_low_rank_counts_fused_available", False)):
            left, right, _ = hmm_lr._low_rank_transition_components(uniform, min(4, k))
            dt, red_lr_counts = _timeit(
                lambda: native_diploid_low_rank_counts_reduce(ref_e, alt_e, other_e, switch, founder_alt2, left, right, sequencing_error_rate=0.01, min_emission_prob=1e-5),
                repeat=repeat,
            )
            add("cpp_low_rank_counts_fused", dt, {"state_count": int(k * k), "rank": int(min(4, k)), "phase": "emission+hmm", "dosage_mean": float(np.mean(red_lr_counts["dosage"])), "native_status": status2})
        if bool(status2.get("pybind11_sparse_topk_counts_fused_available", False)):
            dt, red_sp_counts = _timeit(
                lambda: native_diploid_sparse_topk_counts_reduce(ref_e, alt_e, other_e, switch, founder_alt2, sparse, top_k=top, sequencing_error_rate=0.01, min_emission_prob=1e-5),
                repeat=repeat,
            )
            add("cpp_sparse_topk_counts_fused", dt, {"state_count": int(k * k), "top_k": int(top), "phase": "emission+hmm", "dosage_mean": float(np.mean(red_sp_counts["dosage"])), "native_status": status2})
    except Exception as exc:
        add("cpp_native_factorized_extended", float("nan"), {"phase": "emission+hmm", "error": str(exc)})

    if include_jax:
        try:
            import jax
            import jax.numpy as jnp

            cfg_j = HMMConfig(n_founders=k, ploidy_mode="diploid", ploidy=2, backend="jax", exact_hmm_chunk_size=int(chunk_size))
            hmm_j = JAXStitchHMM(cfg_j)
            fn = hmm_j._jax_diploid_count_chunked_minimal_output_kernel(k, int(chunk_size))
            dt, out = _timeit(
                lambda: fn(
                    jnp.asarray(ref),
                    jnp.asarray(alt),
                    jnp.asarray(other),
                    jnp.asarray(switch),
                    jnp.asarray(founder_alt2),
                    jnp.asarray(0.01, dtype=jnp.float32),
                    jnp.asarray(1e-5, dtype=jnp.float32),
                ),
                repeat=repeat,
            )
            jax.block_until_ready(out)
            add("jax_count_exact_chunked_minimal", dt, {"devices": [str(d.platform) for d in jax.devices()]})
        except Exception as exc:
            add("jax_count_exact_chunked_minimal", float("nan"), {"error": str(exc)})

    zarr_root = output_dir / "synthetic_matrices.zarr"
    if zarr_root.exists():
        import shutil
        shutil.rmtree(zarr_root)
    zarr_root.mkdir(parents=True, exist_ok=True)
    (zarr_root / ".zgroup").write_text(json.dumps({"zarr_format": 2}, sort_keys=True), encoding="utf-8")
    zarr_seconds = _write_simple_zarr_matrix(zarr_root, "dosage", red_numba.dosage.astype(np.float32, copy=False))
    zarr_seconds += _write_simple_zarr_matrix(zarr_root, "genotype_posterior", red_numba.genotype_posterior.astype(np.float32, copy=False))
    add("zarr_matrix_write", zarr_seconds, {"phase": "zarr_write", "output": str(zarr_root)})

    result = {
        "status": "ok",
        "parameters": {
            "n_samples": int(n_samples),
            "n_positions": int(n_positions),
            "n_founders": int(k),
            "chunk_size": int(chunk_size),
            "repeat": int(repeat),
            "seed": int(seed),
        },
        "benchmarks": rows,
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
        pq.write_table(table, output_dir / "benchmark_results.parquet")
    except Exception:
        pass
    return result
