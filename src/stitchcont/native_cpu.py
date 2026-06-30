from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import subprocess
import sysconfig
import tempfile
import time
from typing import Any

import numpy as np


try:  # Preferred install-time pybind11 extension.
    from stitchcont.native._hs_k8_unordered import (
        run_k8_unordered_reduce as _pybind_k8_reduce,
        run_k8_counts_reduce as _pybind_k8_counts_reduce,
        build_k8_logu_from_counts as _pybind_k8_logu_from_counts,
        apply_fragments_unordered_inplace as _pybind_k8_apply_fragments_unordered,
        run_diploid_low_rank_reduce as _pybind_diploid_low_rank_reduce,
        run_diploid_low_rank_counts_reduce as _pybind_diploid_low_rank_counts_reduce,
        run_diploid_sparse_topk_reduce as _pybind_diploid_sparse_topk_reduce,
        run_diploid_sparse_topk_counts_reduce as _pybind_diploid_sparse_topk_counts_reduce,
        run_k8_counts_fragments_reduce as _pybind_k8_counts_fragments_reduce,
    )
    _PYBIND_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on build environment.
    _pybind_k8_reduce = None
    _pybind_k8_counts_reduce = None
    _pybind_k8_logu_from_counts = None
    _pybind_k8_apply_fragments_unordered = None
    _pybind_diploid_low_rank_reduce = None
    _pybind_diploid_low_rank_counts_reduce = None
    _pybind_diploid_sparse_topk_reduce = None
    _pybind_diploid_sparse_topk_counts_reduce = None
    _pybind_k8_counts_fragments_reduce = None
    _PYBIND_IMPORT_ERROR = exc


def _source_path() -> Path:
    return Path(__file__).parent / "native" / "hs_k8_unordered.cpp"


def _library_path() -> Path:
    src = _source_path().read_bytes()
    digest = hashlib.sha1(src).hexdigest()[:12]
    cache = Path(os.environ.get("STITCHCONT_NATIVE_CACHE", Path.home() / ".cache" / "stitchcont" / "native"))
    cache.mkdir(parents=True, exist_ok=True)
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return cache / f"hs_k8_unordered_{digest}{suffix}"


def build_native_library(force: bool = False) -> Path:
    """Build the legacy ctypes shared library fallback.

    New installs should use the pybind11 extension built during
    `pip install -e .`.  This function remains useful for development
    environments where the package was installed with STITCHCONT_SKIP_NATIVE=1.
    """
    out = _library_path()
    if out.exists() and not force:
        return out
    src = _source_path()
    compiler = os.environ.get("CXX", "c++")
    cmd = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC", "-fopenmp", str(src), "-o", str(out)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except Exception:
        # Retry without OpenMP for systems whose default compiler lacks -fopenmp.
        cmd = [compiler, "-O3", "-std=c++17", "-shared", "-fPIC", str(src), "-o", str(out)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


def native_backend_status() -> dict[str, Any]:
    """Return import/build status for native CPU backends."""
    status: dict[str, Any] = {
        "pybind11_extension_available": _pybind_k8_reduce is not None,
        "pybind11_counts_fused_available": _pybind_k8_counts_reduce is not None,
        "pybind11_k8_checkpointed_available": _pybind_k8_counts_reduce is not None and _pybind_k8_reduce is not None,
        "pybind11_fragment_unordered_available": _pybind_k8_apply_fragments_unordered is not None,
        "pybind11_low_rank_reduce_available": _pybind_diploid_low_rank_reduce is not None,
        "pybind11_low_rank_counts_fused_available": _pybind_diploid_low_rank_counts_reduce is not None,
        "pybind11_sparse_topk_reduce_available": _pybind_diploid_sparse_topk_reduce is not None,
        "pybind11_sparse_topk_counts_fused_available": _pybind_diploid_sparse_topk_counts_reduce is not None,
        "pybind11_k8_counts_fragments_fused_available": _pybind_k8_counts_fragments_reduce is not None,
        "pybind11_import_error": None if _PYBIND_IMPORT_ERROR is None else repr(_PYBIND_IMPORT_ERROR),
        "ctypes_fallback_source": str(_source_path()),
        "ctypes_fallback_cache_path": str(_library_path()),
        "ctypes_fallback_cache_exists": bool(_library_path().exists()),
    }
    return status


def _validate_inputs(log_emission_u: np.ndarray, switch: np.ndarray, founder_alt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logu = np.asarray(log_emission_u, dtype=np.float32, order="C")
    sw = np.asarray(switch, dtype=np.float32, order="C")
    fa = np.asarray(founder_alt, dtype=np.float32, order="C")
    if fa.ndim != 2 or fa.shape[0] != 8:
        raise ValueError("native K8 unordered backend requires founder_alt with shape [8, n_positions]")
    if logu.ndim != 3 or logu.shape[2] != 36:
        raise ValueError("native K8 unordered backend requires log_emission_u with shape [n_samples, n_positions, 36]")
    if sw.shape != logu.shape[:2]:
        raise ValueError("native K8 unordered backend requires switch with shape [n_samples, n_positions]")
    if fa.shape[1] != logu.shape[1]:
        raise ValueError("founder_alt.shape[1] must equal n_positions")
    return logu, sw, fa


def native_k8_unordered_reduce(log_emission_u: np.ndarray, switch: np.ndarray, founder_alt: np.ndarray, *, n_threads: int = 0, checkpoint_interval: int = 0):
    t0 = time.perf_counter()
    logu, sw, fa = _validate_inputs(log_emission_u, switch, founder_alt)

    use_pybind = _pybind_k8_reduce is not None and os.environ.get("STITCHCONT_FORCE_CTYPES_NATIVE", "0") not in {"1", "true", "True"}
    if use_pybind:
        result = _pybind_k8_reduce(logu, sw, fa, int(n_threads), int(checkpoint_interval))
        # pybind11 returns a dict with NumPy arrays.  Normalize metadata and timing.
        return {
            "dosage": np.asarray(result["dosage"], dtype=np.float32),
            "genotype_posterior": np.asarray(result["genotype_posterior"], dtype=np.float32),
            "seconds": float(time.perf_counter() - t0),
            "backend": "pybind11_cpp_openmp_k8_unordered",
        }

    n_samples, n_positions, _ = logu.shape
    dosage = np.empty((n_samples, n_positions), dtype=np.float32)
    gp = np.empty((n_samples, n_positions, 3), dtype=np.float32)
    lib = ctypes.CDLL(str(build_native_library()))
    fn = lib.stitchcont_k8_unordered_reduce
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    fn.restype = ctypes.c_int
    rc = fn(
        logu.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sw.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        fa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(n_samples), int(n_positions),
        dosage.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        gp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        int(n_threads),
    )
    if rc != 0:
        raise RuntimeError(f"native K8 unordered kernel returned {rc}")
    return {"dosage": dosage, "genotype_posterior": gp, "seconds": float(time.perf_counter() - t0), "backend": "ctypes_cpp_openmp_k8_unordered"}


def _observations_to_float32(ref_obs: np.ndarray, alt_obs: np.ndarray, other_obs: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref = np.asarray(ref_obs, dtype=np.float32, order="C")
    alt = np.asarray(alt_obs, dtype=np.float32, order="C")
    if other_obs is None:
        oth = np.zeros_like(ref, dtype=np.float32)
    else:
        oth = np.asarray(other_obs, dtype=np.float32, order="C")
    if ref.shape != alt.shape or ref.shape != oth.shape:
        raise ValueError("ref_obs, alt_obs and other_obs must have the same shape")
    return ref, alt, oth


def native_k8_counts_reduce(
    ref_obs: np.ndarray,
    alt_obs: np.ndarray,
    other_obs: np.ndarray | None,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    *,
    sequencing_error_rate: float,
    min_emission_prob: float,
    n_threads: int = 0,
    checkpoint_interval: int = 0,
):
    """Fused native count-emission + unordered K8 HMM reduction.

    This avoids constructing the ordered KxK log-emission tensor and is the
    preferred native CPU path when fragment likelihoods are disabled or absent.
    """
    if _pybind_k8_counts_reduce is None:
        raise RuntimeError(
            "native fused count backend is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'"
        )
    t0 = time.perf_counter()
    ref, alt, oth = _observations_to_float32(ref_obs, alt_obs, other_obs)
    sw = np.asarray(switch, dtype=np.float32, order="C")
    fa = np.asarray(founder_alt, dtype=np.float32, order="C")
    result = _pybind_k8_counts_reduce(
        ref,
        alt,
        oth,
        sw,
        fa,
        float(sequencing_error_rate),
        float(min_emission_prob),
        int(n_threads),
        int(checkpoint_interval),
    )
    return {
        "dosage": np.asarray(result["dosage"], dtype=np.float32),
        "genotype_posterior": np.asarray(result["genotype_posterior"], dtype=np.float32),
        "seconds": float(time.perf_counter() - t0),
        "backend": str(result.get("backend", "pybind11_cpp_openmp_k8_counts_fused")),
    }


def native_k8_logu_from_counts(
    ref_obs: np.ndarray,
    alt_obs: np.ndarray,
    other_obs: np.ndarray | None,
    founder_alt: np.ndarray,
    *,
    sequencing_error_rate: float,
    min_emission_prob: float,
    n_threads: int = 0,
) -> np.ndarray:
    """Build unordered K8 log emissions in native code from dense observations."""
    if _pybind_k8_logu_from_counts is None:
        raise RuntimeError(
            "native unordered emission builder is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'"
        )
    ref, alt, oth = _observations_to_float32(ref_obs, alt_obs, other_obs)
    fa = np.asarray(founder_alt, dtype=np.float32, order="C")
    return np.asarray(
        _pybind_k8_logu_from_counts(
            ref,
            alt,
            oth,
            fa,
            float(sequencing_error_rate),
            float(min_emission_prob),
            int(n_threads),
        ),
        dtype=np.float32,
        order="C",
    )


def native_k8_apply_fragments_unordered(
    log_emission_u: np.ndarray,
    *,
    fragment_sample_offsets: np.ndarray,
    fragment_center_idx: np.ndarray,
    fragment_obs_offsets: np.ndarray,
    fragment_obs_pos_idx: np.ndarray,
    fragment_obs_code: np.ndarray,
    fragment_obs_qual: np.ndarray | None,
    founder_alt: np.ndarray,
    sequencing_error_rate: float,
    min_emission_prob: float,
    mode: str,
    rescale: bool,
    max_emission_matrix_difference: float,
    n_threads: int = 0,
) -> np.ndarray:
    """Apply STITCH-style fragment likelihoods directly to unordered K8 emissions."""
    if _pybind_k8_apply_fragments_unordered is None:
        raise RuntimeError(
            "native unordered fragment backend is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'"
        )
    logu = np.asarray(log_emission_u, dtype=np.float32, order="C")
    foq = (
        np.asarray(fragment_obs_qual, dtype=np.float32, order="C")
        if fragment_obs_qual is not None and np.asarray(fragment_obs_qual).size
        else np.full(np.asarray(fragment_obs_pos_idx).shape, float(0.0), dtype=np.float32)
    )
    result = _pybind_k8_apply_fragments_unordered(
        logu,
        np.asarray(fragment_sample_offsets, dtype=np.int64, order="C"),
        np.asarray(fragment_center_idx, dtype=np.int64, order="C"),
        np.asarray(fragment_obs_offsets, dtype=np.int64, order="C"),
        np.asarray(fragment_obs_pos_idx, dtype=np.int64, order="C"),
        np.asarray(fragment_obs_code, dtype=np.int8, order="C"),
        foq,
        np.asarray(founder_alt, dtype=np.float32, order="C"),
        float(sequencing_error_rate),
        float(min_emission_prob),
        str(mode) == "replace",
        bool(rescale),
        float(max_emission_matrix_difference),
        int(n_threads),
    )
    return np.asarray(result.get("log_emission_u", logu), dtype=np.float32, order="C")


def native_diploid_low_rank_reduce(
    log_emission: np.ndarray,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    left_factor: np.ndarray,
    right_factor: np.ndarray,
    *,
    n_threads: int = 0,
):
    """Native ordered diploid HMM reduction using algebraic low-rank transitions.

    This is the C++ implementation of the approximate low_rank_linear model:
    W ~= left_factor @ right_factor.T and T = (1-switch)I + switch*W.
    It reduces posterior states directly to dosage and genotype posteriors.
    """
    if _pybind_diploid_low_rank_reduce is None:
        raise RuntimeError(
            "native low-rank backend is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'"
        )
    t0 = time.perf_counter()
    loge = np.asarray(log_emission, dtype=np.float32, order="C")
    sw = np.asarray(switch, dtype=np.float32, order="C")
    fa = np.asarray(founder_alt, dtype=np.float32, order="C")
    left = np.asarray(left_factor, dtype=np.float32, order="C")
    right = np.asarray(right_factor, dtype=np.float32, order="C")
    if loge.ndim != 4 or loge.shape[2] != loge.shape[3]:
        raise ValueError("log_emission must have shape [n_samples, n_positions, K, K]")
    if sw.shape != loge.shape[:2]:
        raise ValueError("switch must have shape [n_samples, n_positions]")
    if fa.shape != (loge.shape[2], loge.shape[1]):
        raise ValueError("founder_alt must have shape [K, n_positions]")
    if left.shape[0] != loge.shape[2] or right.shape != left.shape:
        raise ValueError("left_factor and right_factor must both have shape [K, rank]")
    result = _pybind_diploid_low_rank_reduce(loge, sw, fa, left, right, int(n_threads))
    return {
        "dosage": np.asarray(result["dosage"], dtype=np.float32),
        "genotype_posterior": np.asarray(result["genotype_posterior"], dtype=np.float32),
        "seconds": float(time.perf_counter() - t0),
        "backend": str(result.get("backend", "pybind11_cpp_openmp_diploid_low_rank")),
    }



def native_diploid_low_rank_counts_reduce(
    ref_obs: np.ndarray,
    alt_obs: np.ndarray,
    other_obs: np.ndarray | None,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    left_factor: np.ndarray,
    right_factor: np.ndarray,
    *,
    sequencing_error_rate: float,
    min_emission_prob: float,
    n_threads: int = 0,
):
    """Native fused count-emission + ordered diploid low-rank HMM reduction."""
    if _pybind_diploid_low_rank_counts_reduce is None:
        raise RuntimeError("native low-rank fused count backend is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'")
    t0 = time.perf_counter()
    ref, alt, oth = _observations_to_float32(ref_obs, alt_obs, other_obs)
    result = _pybind_diploid_low_rank_counts_reduce(
        ref,
        alt,
        oth,
        np.asarray(switch, dtype=np.float32, order="C"),
        np.asarray(founder_alt, dtype=np.float32, order="C"),
        np.asarray(left_factor, dtype=np.float32, order="C"),
        np.asarray(right_factor, dtype=np.float32, order="C"),
        float(sequencing_error_rate),
        float(min_emission_prob),
        int(n_threads),
    )
    return {
        "dosage": np.asarray(result["dosage"], dtype=np.float32),
        "genotype_posterior": np.asarray(result["genotype_posterior"], dtype=np.float32),
        "seconds": float(time.perf_counter() - t0),
        "backend": str(result.get("backend", "pybind11_cpp_openmp_diploid_low_rank_counts_fused")),
    }


def native_diploid_sparse_topk_reduce(
    log_emission: np.ndarray,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    offdiag: np.ndarray,
    *,
    top_k: int,
    n_threads: int = 0,
):
    """Native ordered diploid sparse top-k transition HMM reduction."""
    if _pybind_diploid_sparse_topk_reduce is None:
        raise RuntimeError("native sparse top-k backend is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'")
    t0 = time.perf_counter()
    result = _pybind_diploid_sparse_topk_reduce(
        np.asarray(log_emission, dtype=np.float32, order="C"),
        np.asarray(switch, dtype=np.float32, order="C"),
        np.asarray(founder_alt, dtype=np.float32, order="C"),
        np.asarray(offdiag, dtype=np.float32, order="C"),
        int(top_k),
        int(n_threads),
    )
    return {
        "dosage": np.asarray(result["dosage"], dtype=np.float32),
        "genotype_posterior": np.asarray(result["genotype_posterior"], dtype=np.float32),
        "seconds": float(time.perf_counter() - t0),
        "backend": str(result.get("backend", "pybind11_cpp_openmp_diploid_sparse_topk")),
    }


def native_diploid_sparse_topk_counts_reduce(
    ref_obs: np.ndarray,
    alt_obs: np.ndarray,
    other_obs: np.ndarray | None,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    offdiag: np.ndarray,
    *,
    top_k: int,
    sequencing_error_rate: float,
    min_emission_prob: float,
    n_threads: int = 0,
):
    """Native fused count-emission + ordered diploid sparse top-k HMM reduction."""
    if _pybind_diploid_sparse_topk_counts_reduce is None:
        raise RuntimeError("native sparse top-k fused count backend is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'")
    t0 = time.perf_counter()
    ref, alt, oth = _observations_to_float32(ref_obs, alt_obs, other_obs)
    result = _pybind_diploid_sparse_topk_counts_reduce(
        ref,
        alt,
        oth,
        np.asarray(switch, dtype=np.float32, order="C"),
        np.asarray(founder_alt, dtype=np.float32, order="C"),
        np.asarray(offdiag, dtype=np.float32, order="C"),
        int(top_k),
        float(sequencing_error_rate),
        float(min_emission_prob),
        int(n_threads),
    )
    return {
        "dosage": np.asarray(result["dosage"], dtype=np.float32),
        "genotype_posterior": np.asarray(result["genotype_posterior"], dtype=np.float32),
        "seconds": float(time.perf_counter() - t0),
        "backend": str(result.get("backend", "pybind11_cpp_openmp_diploid_sparse_topk_counts_fused")),
    }


def native_k8_counts_fragments_reduce(
    ref_obs: np.ndarray,
    alt_obs: np.ndarray,
    other_obs: np.ndarray | None,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    *,
    fragment_sample_offsets: np.ndarray,
    fragment_center_idx: np.ndarray,
    fragment_obs_offsets: np.ndarray,
    fragment_obs_pos_idx: np.ndarray,
    fragment_obs_code: np.ndarray,
    fragment_obs_qual: np.ndarray | None,
    sequencing_error_rate: float,
    min_emission_prob: float,
    mode: str,
    rescale: bool,
    max_emission_matrix_difference: float,
    n_threads: int = 0,
):
    """Native fused K8 count evidence + fragment likelihood + unordered HMM reduction."""
    if _pybind_k8_counts_fragments_reduce is None:
        raise RuntimeError("native K8 fused count+fragment backend is unavailable. Reinstall with pybind11 support: pip install -e '.[cpu]'")
    t0 = time.perf_counter()
    ref, alt, oth = _observations_to_float32(ref_obs, alt_obs, other_obs)
    foq = (
        np.asarray(fragment_obs_qual, dtype=np.float32, order="C")
        if fragment_obs_qual is not None and np.asarray(fragment_obs_qual).size
        else np.full(np.asarray(fragment_obs_pos_idx).shape, float(0.0), dtype=np.float32)
    )
    result = _pybind_k8_counts_fragments_reduce(
        ref,
        alt,
        oth,
        np.asarray(switch, dtype=np.float32, order="C"),
        np.asarray(founder_alt, dtype=np.float32, order="C"),
        np.asarray(fragment_sample_offsets, dtype=np.int64, order="C"),
        np.asarray(fragment_center_idx, dtype=np.int64, order="C"),
        np.asarray(fragment_obs_offsets, dtype=np.int64, order="C"),
        np.asarray(fragment_obs_pos_idx, dtype=np.int64, order="C"),
        np.asarray(fragment_obs_code, dtype=np.int8, order="C"),
        foq,
        float(sequencing_error_rate),
        float(min_emission_prob),
        str(mode) == "replace",
        bool(rescale),
        float(max_emission_matrix_difference),
        int(n_threads),
    )
    return {
        "dosage": np.asarray(result["dosage"], dtype=np.float32),
        "genotype_posterior": np.asarray(result["genotype_posterior"], dtype=np.float32),
        "seconds": float(time.perf_counter() - t0),
        "backend": str(result.get("backend", "pybind11_cpp_openmp_k8_counts_fragments_fused")),
    }
