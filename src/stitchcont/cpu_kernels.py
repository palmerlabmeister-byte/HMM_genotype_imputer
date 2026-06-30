from __future__ import annotations

"""CPU-oriented kernels for STITCHCONT.

The functions in this module are intentionally dependency-tolerant: when numba
is installed, the inner loops are JIT compiled; otherwise the same public API
falls back to pure Python/NumPy.  The production target is fixed-founder K8
HS-rat imputation on CPU, where avoiding giant gamma/xi tensors matters more
than generic dense tensor throughput.
"""

from dataclasses import dataclass
import time

import numpy as np

try:  # pragma: no cover - availability depends on user env.
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore
        def deco(fn):
            return fn
        if args and callable(args[0]) and not kwargs:
            return args[0]
        return deco


@dataclass(slots=True)
class NumbaKernelResult:
    dosage: np.ndarray
    genotype_posterior: np.ndarray | None
    seconds: float
    backend: str
    diagnostics: dict[str, object]


def unordered_pairs_arrays(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    pair_index = np.empty((int(k), int(k)), dtype=np.int32)
    idx = 0
    for i in range(int(k)):
        for j in range(i, int(k)):
            pairs_i.append(i)
            pairs_j.append(j)
            pair_index[i, j] = idx
            pair_index[j, i] = idx
            idx += 1
    return np.asarray(pairs_i, dtype=np.int32), np.asarray(pairs_j, dtype=np.int32), pair_index


def topk_offdiag_arrays(offdiag: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    off = np.asarray(offdiag, dtype=np.float32)
    k = int(off.shape[0])
    m = max(0, min(int(top_k), max(k - 1, 0)))
    dest = np.full((k, m), -1, dtype=np.int32)
    prob = np.zeros((k, m), dtype=np.float32)
    if m == 0:
        return dest, prob
    for i in range(k):
        row = off[i].copy()
        if i < row.shape[0]:
            row[i] = -np.inf
        idx = np.argsort(row)[::-1][:m]
        vals = np.maximum(off[i, idx], 0.0).astype(np.float32, copy=False)
        denom = float(vals.sum())
        if denom <= 0.0 or not np.isfinite(denom):
            # Uniform fallback over non-self destinations.
            candidates = [j for j in range(k) if j != i]
            idx = np.asarray(candidates[:m], dtype=np.int32)
            vals = np.full(idx.shape[0], 1.0 / float(max(idx.shape[0], 1)), dtype=np.float32)
        else:
            vals = vals / denom
        dest[i, : idx.shape[0]] = idx.astype(np.int32, copy=False)
        prob[i, : idx.shape[0]] = vals.astype(np.float32, copy=False)
    return dest, prob


@njit(cache=True)
def _u_to_ordered(u: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.float32)
    n_states = pairs_i.shape[0]
    for s in range(n_states):
        i = pairs_i[s]
        j = pairs_j[s]
        if i == j:
            m[i, j] = u[s]
        else:
            v = 0.5 * u[s]
            m[i, j] = v
            m[j, i] = v
    return m


@njit(cache=True)
def _ordered_to_u(m: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray) -> np.ndarray:
    n_states = pairs_i.shape[0]
    out = np.empty(n_states, dtype=np.float32)
    for s in range(n_states):
        i = pairs_i[s]
        j = pairs_j[s]
        if i == j:
            out[s] = m[i, j]
        else:
            out[s] = m[i, j] + m[j, i]
    return out

@njit(cache=True)
def _u_to_ordered_broadcast(u: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.float32)
    n_states = pairs_i.shape[0]
    for s in range(n_states):
        i = pairs_i[s]
        j = pairs_j[s]
        if i == j:
            m[i, j] = u[s]
        else:
            m[i, j] = u[s]
            m[j, i] = u[s]
    return m

@njit(cache=True)
def _ordered_to_u_average(m: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray) -> np.ndarray:
    n_states = pairs_i.shape[0]
    out = np.empty(n_states, dtype=np.float32)
    for s in range(n_states):
        i = pairs_i[s]
        j = pairs_j[s]
        if i == j:
            out[s] = m[i, j]
        else:
            out[s] = 0.5 * (m[i, j] + m[j, i])
    return out


@njit(cache=True)
def _normalise_vec(x: np.ndarray) -> np.ndarray:
    s = 0.0
    for i in range(x.shape[0]):
        v = x[i]
        if np.isfinite(v) and v > 0.0:
            s += v
    if s <= 0.0 or not np.isfinite(s):
        fill = 1.0 / float(x.shape[0])
        for i in range(x.shape[0]):
            x[i] = fill
        return x
    inv = 1.0 / s
    for i in range(x.shape[0]):
        x[i] = x[i] * inv
    return x


@njit(cache=True)
def _parity_forward_u(prev_u: np.ndarray, emission_u: np.ndarray, sw: float, pairs_i: np.ndarray, pairs_j: np.ndarray, k: int) -> np.ndarray:
    prev = _u_to_ordered(prev_u, pairs_i, pairs_j, k)
    off = sw / float(max(k - 1, 1))
    b = 1.0 - sw - off
    row_sum = np.empty((k, 1), dtype=np.float32)
    col_sum = np.empty((1, k), dtype=np.float32)
    total = 0.0
    for i in range(k):
        r = 0.0
        for j in range(k):
            r += prev[i, j]
        row_sum[i, 0] = r
        total += r
    for j in range(k):
        c = 0.0
        for i in range(k):
            c += prev[i, j]
        col_sum[0, j] = c
    pred = np.empty((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            pred[i, j] = (b * b) * prev[i, j] + (b * off) * (row_sum[i, 0] + col_sum[0, j]) + (off * off) * total
    out_u = _ordered_to_u(pred, pairs_i, pairs_j)
    for s in range(out_u.shape[0]):
        out_u[s] *= emission_u[s]
    return _normalise_vec(out_u)


@njit(cache=True)
def _parity_backward_u(beta_next_u: np.ndarray, emission_next_u: np.ndarray, sw: float, pairs_i: np.ndarray, pairs_j: np.ndarray, k: int) -> np.ndarray:
    tmp_u = np.empty(beta_next_u.shape[0], dtype=np.float32)
    for s in range(tmp_u.shape[0]):
        tmp_u[s] = beta_next_u[s] * emission_next_u[s]
    tmp = _u_to_ordered_broadcast(tmp_u, pairs_i, pairs_j, k)
    off = sw / float(max(k - 1, 1))
    b = 1.0 - sw - off
    row_sum = np.empty((k, 1), dtype=np.float32)
    col_sum = np.empty((1, k), dtype=np.float32)
    total = 0.0
    for i in range(k):
        r = 0.0
        for j in range(k):
            r += tmp[i, j]
        row_sum[i, 0] = r
        total += r
    for j in range(k):
        c = 0.0
        for i in range(k):
            c += tmp[i, j]
        col_sum[0, j] = c
    out = np.empty((k, k), dtype=np.float32)
    for i in range(k):
        for j in range(k):
            out[i, j] = (b * b) * tmp[i, j] + (b * off) * (row_sum[i, 0] + col_sum[0, j]) + (off * off) * total
    out_u = _ordered_to_u_average(out, pairs_i, pairs_j)
    return _normalise_vec(out_u)


@njit(cache=True)
def _sparse_forward_u(prev_u: np.ndarray, emission_u: np.ndarray, sw: float, pairs_i: np.ndarray, pairs_j: np.ndarray, pair_index: np.ndarray, dest_idx: np.ndarray, dest_prob: np.ndarray, k: int) -> np.ndarray:
    n_states = pairs_i.shape[0]
    out = np.zeros(n_states, dtype=np.float32)
    m = dest_idx.shape[1]
    for s in range(n_states):
        a = pairs_i[s]
        b = pairs_j[s]
        src_prob = prev_u[s]
        if src_prob <= 0.0:
            continue
        n_phases = 1
        if a != b:
            n_phases = 2
        for phase in range(n_phases):
            if phase == 0:
                x = a
                y = b
            else:
                x = b
                y = a
            phase_prob = src_prob / float(n_phases)
            # First chromosome: stay plus sparse destinations.
            for ix in range(m + 1):
                if ix == 0:
                    tx = x
                    px = 1.0 - sw
                else:
                    tx = dest_idx[x, ix - 1]
                    if tx < 0:
                        continue
                    px = sw * dest_prob[x, ix - 1]
                if px <= 0.0:
                    continue
                for iy in range(m + 1):
                    if iy == 0:
                        ty = y
                        py = 1.0 - sw
                    else:
                        ty = dest_idx[y, iy - 1]
                        if ty < 0:
                            continue
                        py = sw * dest_prob[y, iy - 1]
                    if py <= 0.0:
                        continue
                    ds = pair_index[tx, ty]
                    out[ds] += phase_prob * px * py
    for s in range(n_states):
        out[s] *= emission_u[s]
    return _normalise_vec(out)


@njit(cache=True)
def _sparse_backward_u(beta_next_u: np.ndarray, emission_next_u: np.ndarray, sw: float, pairs_i: np.ndarray, pairs_j: np.ndarray, pair_index: np.ndarray, dest_idx: np.ndarray, dest_prob: np.ndarray, k: int) -> np.ndarray:
    n_states = pairs_i.shape[0]
    out = np.zeros(n_states, dtype=np.float32)
    m = dest_idx.shape[1]
    next_u = np.empty(n_states, dtype=np.float32)
    for s in range(n_states):
        next_u[s] = beta_next_u[s] * emission_next_u[s]
    for s in range(n_states):
        a = pairs_i[s]
        b = pairs_j[s]
        acc = 0.0
        n_phases = 1
        if a != b:
            n_phases = 2
        for phase in range(n_phases):
            if phase == 0:
                x = a
                y = b
            else:
                x = b
                y = a
            phase_acc = 0.0
            for ix in range(m + 1):
                if ix == 0:
                    tx = x
                    px = 1.0 - sw
                else:
                    tx = dest_idx[x, ix - 1]
                    if tx < 0:
                        continue
                    px = sw * dest_prob[x, ix - 1]
                if px <= 0.0:
                    continue
                for iy in range(m + 1):
                    if iy == 0:
                        ty = y
                        py = 1.0 - sw
                    else:
                        ty = dest_idx[y, iy - 1]
                        if ty < 0:
                            continue
                        py = sw * dest_prob[y, iy - 1]
                    if py <= 0.0:
                        continue
                    ds = pair_index[tx, ty]
                    phase_acc += px * py * next_u[ds]
            acc += phase_acc / float(n_phases)
        out[s] = acc
    return _normalise_vec(out)


@njit(cache=True)
def _reduce_state_to_outputs(state_u: np.ndarray, founder_alt_col: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray, gp_out: np.ndarray, sample_idx: int, pos_idx: int) -> float:
    min_a = founder_alt_col[0]
    max_a = founder_alt_col[0]
    for k0 in range(1, founder_alt_col.shape[0]):
        v0 = founder_alt_col[k0]
        if v0 < min_a:
            min_a = v0
        if v0 > max_a:
            max_a = v0
    if max_a - min_a < 1e-7:
        a = 0.5 * (min_a + max_a)
        dosage_const = 2.0 * a
        gp0c = (1.0 - a) * (1.0 - a)
        gp2c = a * a
        gp1c = 1.0 - gp0c - gp2c
        if gp1c < 0.0:
            gp1c = 0.0
        sgpc = gp0c + gp1c + gp2c
        gp_out[sample_idx, pos_idx, 0] = gp0c / sgpc
        gp_out[sample_idx, pos_idx, 1] = gp1c / sgpc
        gp_out[sample_idx, pos_idx, 2] = gp2c / sgpc
        return dosage_const
    dosage = 0.0
    gp0 = 0.0
    gp2 = 0.0
    for s in range(state_u.shape[0]):
        w = state_u[s]
        i = pairs_i[s]
        j = pairs_j[s]
        ai = founder_alt_col[i]
        aj = founder_alt_col[j]
        dosage += w * (ai + aj)
        gp0 += w * (1.0 - ai) * (1.0 - aj)
        gp2 += w * ai * aj
    gp1 = 1.0 - gp0 - gp2
    if gp1 < 0.0:
        gp1 = 0.0
    sgp = gp0 + gp1 + gp2
    if sgp <= 0.0:
        gp0 = 1.0 / 3.0
        gp1 = 1.0 / 3.0
        gp2 = 1.0 / 3.0
    else:
        gp0 /= sgp
        gp1 /= sgp
        gp2 /= sgp
    gp_out[sample_idx, pos_idx, 0] = gp0
    gp_out[sample_idx, pos_idx, 1] = gp1
    gp_out[sample_idx, pos_idx, 2] = gp2
    return dosage


@njit(cache=True)
def _numba_unordered_fb_reduce_parity(log_emission_u: np.ndarray, switch: np.ndarray, founder_alt: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_samples = log_emission_u.shape[0]
    n_positions = log_emission_u.shape[1]
    n_states = log_emission_u.shape[2]
    k = founder_alt.shape[0]
    dosage = np.empty((n_samples, n_positions), dtype=np.float32)
    gp = np.empty((n_samples, n_positions, 3), dtype=np.float32)
    for sample_idx in range(n_samples):
        emission = np.empty((n_positions, n_states), dtype=np.float32)
        for p in range(n_positions):
            maxv = log_emission_u[sample_idx, p, 0]
            minv = log_emission_u[sample_idx, p, 0]
            for s in range(1, n_states):
                v = log_emission_u[sample_idx, p, s]
                if v > maxv:
                    maxv = v
                if v < minv:
                    minv = v
            if maxv - minv < 1e-7:
                for s in range(n_states):
                    emission[p, s] = 1.0
            else:
                for s in range(n_states):
                    emission[p, s] = np.exp(log_emission_u[sample_idx, p, s] - maxv)
        alpha = np.empty((n_positions, n_states), dtype=np.float32)
        beta = np.empty((n_positions, n_states), dtype=np.float32)
        # unordered prior from ordered uniform K*K: diag 1/K^2, het 2/K^2
        for s in range(n_states):
            mult = 1.0
            if pairs_i[s] != pairs_j[s]:
                mult = 2.0
            alpha[0, s] = mult / float(k * k) * emission[0, s]
        _normalise_vec(alpha[0])
        for p in range(1, n_positions):
            alpha[p] = _parity_forward_u(alpha[p - 1], emission[p], switch[sample_idx, p], pairs_i, pairs_j, k)
        fill = 1.0 / float(n_states)
        for s in range(n_states):
            beta[n_positions - 1, s] = fill
        for p in range(n_positions - 2, -1, -1):
            beta[p] = _parity_backward_u(beta[p + 1], emission[p + 1], switch[sample_idx, p + 1], pairs_i, pairs_j, k)
        for p in range(n_positions):
            state = np.empty(n_states, dtype=np.float32)
            for s in range(n_states):
                state[s] = alpha[p, s] * beta[p, s]
            _normalise_vec(state)
            dosage[sample_idx, p] = _reduce_state_to_outputs(state, founder_alt[:, p], pairs_i, pairs_j, gp, sample_idx, p)
    return dosage, gp


@njit(cache=True)
def _numba_unordered_fb_reduce_parity_checkpointed(log_emission_u: np.ndarray, switch: np.ndarray, founder_alt: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray, checkpoint_interval: int) -> tuple[np.ndarray, np.ndarray]:
    n_samples = log_emission_u.shape[0]
    n_positions = log_emission_u.shape[1]
    n_states = log_emission_u.shape[2]
    k = founder_alt.shape[0]
    dosage = np.empty((n_samples, n_positions), dtype=np.float32)
    gp = np.empty((n_samples, n_positions, 3), dtype=np.float32)
    chk = max(1, int(checkpoint_interval))
    n_segments = (n_positions + chk - 1) // chk
    for sample_idx in range(n_samples):
        emission = np.empty((n_positions, n_states), dtype=np.float32)
        for p in range(n_positions):
            maxv = log_emission_u[sample_idx, p, 0]
            minv = log_emission_u[sample_idx, p, 0]
            for s in range(1, n_states):
                v = log_emission_u[sample_idx, p, s]
                if v > maxv:
                    maxv = v
                if v < minv:
                    minv = v
            if maxv - minv < 1e-7:
                for s in range(n_states):
                    emission[p, s] = 1.0
            else:
                for s in range(n_states):
                    emission[p, s] = np.exp(log_emission_u[sample_idx, p, s] - maxv)
        alpha_ckpt = np.empty((n_segments, n_states), dtype=np.float32)
        cur = np.empty(n_states, dtype=np.float32)
        for s in range(n_states):
            mult = 1.0
            if pairs_i[s] != pairs_j[s]:
                mult = 2.0
            cur[s] = mult / float(k * k) * emission[0, s]
        _normalise_vec(cur)
        alpha_ckpt[0] = cur
        for p in range(1, n_positions):
            cur = _parity_forward_u(cur, emission[p], switch[sample_idx, p], pairs_i, pairs_j, k)
            if p % chk == 0:
                alpha_ckpt[p // chk] = cur
        beta_end = np.empty(n_states, dtype=np.float32)
        fill = 1.0 / float(n_states)
        for s in range(n_states):
            beta_end[s] = fill
        for seg in range(n_segments - 1, -1, -1):
            start = seg * chk
            end = min(n_positions - 1, start + chk - 1)
            seg_len = end - start + 1
            alpha_local = np.empty((seg_len, n_states), dtype=np.float32)
            beta_local = np.empty((seg_len, n_states), dtype=np.float32)
            for s in range(n_states):
                alpha_local[0, s] = alpha_ckpt[seg, s]
            for p0 in range(1, seg_len):
                p = start + p0
                alpha_local[p0] = _parity_forward_u(alpha_local[p0 - 1], emission[p], switch[sample_idx, p], pairs_i, pairs_j, k)
            for s in range(n_states):
                beta_local[seg_len - 1, s] = beta_end[s]
            for p0 in range(seg_len - 2, -1, -1):
                p = start + p0
                beta_local[p0] = _parity_backward_u(beta_local[p0 + 1], emission[p + 1], switch[sample_idx, p + 1], pairs_i, pairs_j, k)
            for p0 in range(seg_len):
                p = start + p0
                state = np.empty(n_states, dtype=np.float32)
                for s in range(n_states):
                    state[s] = alpha_local[p0, s] * beta_local[p0, s]
                _normalise_vec(state)
                dosage[sample_idx, p] = _reduce_state_to_outputs(state, founder_alt[:, p], pairs_i, pairs_j, gp, sample_idx, p)
            if start > 0:
                beta_end = _parity_backward_u(beta_local[0], emission[start], switch[sample_idx, start], pairs_i, pairs_j, k)
    return dosage, gp


@njit(cache=True)
def _numba_unordered_fb_reduce_sparse(log_emission_u: np.ndarray, switch: np.ndarray, founder_alt: np.ndarray, pairs_i: np.ndarray, pairs_j: np.ndarray, pair_index: np.ndarray, dest_idx: np.ndarray, dest_prob: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_samples = log_emission_u.shape[0]
    n_positions = log_emission_u.shape[1]
    n_states = log_emission_u.shape[2]
    k = founder_alt.shape[0]
    dosage = np.empty((n_samples, n_positions), dtype=np.float32)
    gp = np.empty((n_samples, n_positions, 3), dtype=np.float32)
    for sample_idx in range(n_samples):
        emission = np.empty((n_positions, n_states), dtype=np.float32)
        for p in range(n_positions):
            maxv = log_emission_u[sample_idx, p, 0]
            minv = log_emission_u[sample_idx, p, 0]
            for s in range(1, n_states):
                v = log_emission_u[sample_idx, p, s]
                if v > maxv:
                    maxv = v
                if v < minv:
                    minv = v
            if maxv - minv < 1e-7:
                for s in range(n_states):
                    emission[p, s] = 1.0
            else:
                for s in range(n_states):
                    emission[p, s] = np.exp(log_emission_u[sample_idx, p, s] - maxv)
        alpha = np.empty((n_positions, n_states), dtype=np.float32)
        beta = np.empty((n_positions, n_states), dtype=np.float32)
        for s in range(n_states):
            mult = 1.0
            if pairs_i[s] != pairs_j[s]:
                mult = 2.0
            alpha[0, s] = mult / float(k * k) * emission[0, s]
        _normalise_vec(alpha[0])
        for p in range(1, n_positions):
            alpha[p] = _sparse_forward_u(alpha[p - 1], emission[p], switch[sample_idx, p], pairs_i, pairs_j, pair_index, dest_idx, dest_prob, k)
        fill = 1.0 / float(n_states)
        for s in range(n_states):
            beta[n_positions - 1, s] = fill
        for p in range(n_positions - 2, -1, -1):
            beta[p] = _sparse_backward_u(beta[p + 1], emission[p + 1], switch[sample_idx, p + 1], pairs_i, pairs_j, pair_index, dest_idx, dest_prob, k)
        for p in range(n_positions):
            state = np.empty(n_states, dtype=np.float32)
            for s in range(n_states):
                state[s] = alpha[p, s] * beta[p, s]
            _normalise_vec(state)
            dosage[sample_idx, p] = _reduce_state_to_outputs(state, founder_alt[:, p], pairs_i, pairs_j, gp, sample_idx, p)
    return dosage, gp


def numba_unordered_diploid_reduce(
    log_emission_u: np.ndarray,
    switch: np.ndarray,
    founder_alt: np.ndarray,
    *,
    offdiag_matrix: np.ndarray | None = None,
    sparse_top_k: int = 0,
    checkpoint_interval: int = 0,
) -> NumbaKernelResult:
    t0 = time.perf_counter()
    k = int(founder_alt.shape[0])
    pairs_i, pairs_j, pair_index = unordered_pairs_arrays(k)
    log_emission_u = np.asarray(log_emission_u, dtype=np.float32, order="C")
    switch = np.asarray(switch, dtype=np.float32, order="C")
    founder_alt = np.asarray(founder_alt, dtype=np.float32, order="C")
    use_sparse = offdiag_matrix is not None and int(sparse_top_k) > 0
    if use_sparse:
        dest_idx, dest_prob = topk_offdiag_arrays(np.asarray(offdiag_matrix, dtype=np.float32), int(sparse_top_k))
        dosage, gp = _numba_unordered_fb_reduce_sparse(log_emission_u, switch, founder_alt, pairs_i, pairs_j, pair_index, dest_idx, dest_prob)
        mode = "numba_unordered_sparse_topk" if NUMBA_AVAILABLE else "python_unordered_sparse_topk"
        extra = {"sparse_top_k": int(sparse_top_k)}
    else:
        if int(checkpoint_interval) > 0:
            dosage, gp = _numba_unordered_fb_reduce_parity_checkpointed(log_emission_u, switch, founder_alt, pairs_i, pairs_j, int(checkpoint_interval))
            mode = "numba_unordered_parity_checkpointed" if NUMBA_AVAILABLE else "python_unordered_parity_checkpointed"
            extra = {"checkpoint_interval": int(checkpoint_interval)}
        else:
            dosage, gp = _numba_unordered_fb_reduce_parity(log_emission_u, switch, founder_alt, pairs_i, pairs_j)
            mode = "numba_unordered_parity" if NUMBA_AVAILABLE else "python_unordered_parity"
            extra = {}
    return NumbaKernelResult(
        dosage=np.asarray(dosage, dtype=np.float32),
        genotype_posterior=np.asarray(gp, dtype=np.float32),
        seconds=float(time.perf_counter() - t0),
        backend=mode,
        diagnostics={"numba_available": bool(NUMBA_AVAILABLE), "state_count": int(k * (k + 1) // 2), **extra},
    )
