#include "hs_k8_unordered.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int K = 8;
constexpr int S = 36;

struct StateMap {
    int pi[S];
    int pj[S];
    StateMap() {
        int idx = 0;
        for (int i = 0; i < K; ++i) {
            for (int j = i; j < K; ++j) {
                pi[idx] = i;
                pj[idx] = j;
                ++idx;
            }
        }
    }
};

inline void set_threads(int n_threads) {
#ifdef _OPENMP
    if (n_threads > 0) omp_set_num_threads(n_threads);
#else
    (void)n_threads;
#endif
}

inline float clamp_prob(float x, float lo) {
    if (!std::isfinite(x)) return lo;
    if (x < lo) return lo;
    if (x > 1.0f) return 1.0f;
    return x;
}

inline float base_error_from_qual(float q, float fallback_eps) {
    if (!std::isfinite(q) || q <= 0.0f) return fallback_eps;
    return std::pow(10.0f, -q / 10.0f);
}

inline float obs_log_prob_for_founder(float founder_alt, int code, float err, float min_prob) {
    const float p_err_other = err / 3.0f;
    float p_alt = founder_alt * (1.0f - err) + (1.0f - founder_alt) * p_err_other;
    float p_ref = (1.0f - founder_alt) * (1.0f - err) + founder_alt * p_err_other;
    float p_oth = p_err_other;
    float p = (code == 1 ? p_alt : (code == 0 ? p_ref : p_oth));
    return std::log(clamp_prob(p, min_prob));
}

inline void compute_logu_row_from_counts(
    const float ref_obs,
    const float alt_obs,
    const float other_obs,
    const float* founder_alt,
    const int n_positions,
    const int p,
    const float eps,
    const float min_prob,
    const StateMap& sm,
    float* out36
) {
    const float p_err_other = eps / 3.0f;
    for (int s = 0; s < S; ++s) {
        const int i = sm.pi[s];
        const int j = sm.pj[s];
        const float ai = founder_alt[(size_t)i * n_positions + p];
        const float aj = founder_alt[(size_t)j * n_positions + p];
        float hap_alt = 0.5f * (ai + aj);
        if (hap_alt < 0.0f) hap_alt = 0.0f;
        if (hap_alt > 1.0f) hap_alt = 1.0f;
        const float p_alt = clamp_prob(hap_alt * (1.0f - eps) + (1.0f - hap_alt) * p_err_other, min_prob);
        const float p_ref = clamp_prob((1.0f - hap_alt) * (1.0f - eps) + hap_alt * p_err_other, min_prob);
        const float p_oth = clamp_prob(p_err_other, min_prob);
        out36[s] = alt_obs * std::log(p_alt) + ref_obs * std::log(p_ref) + other_obs * std::log(p_oth);
    }
}

inline void logu_to_emit_row(const float* row, float* erow) {
    float maxv = row[0];
    float minv = row[0];
    for (int s = 1; s < S; ++s) {
        if (row[s] > maxv) maxv = row[s];
        if (row[s] < minv) minv = row[s];
    }
    if (maxv - minv < 1e-7f) {
        for (int s = 0; s < S; ++s) erow[s] = 1.0f;
    } else {
        for (int s = 0; s < S; ++s) erow[s] = std::exp(row[s] - maxv);
    }
}

inline void transition_forward_parity(
    const float* prev,
    const float sw,
    const StateMap& sm,
    float* cur
) {
    float prevM[K][K];
    for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) prevM[i][j] = 0.0f;
    for (int s = 0; s < S; ++s) {
        const int i = sm.pi[s], j = sm.pj[s];
        if (i == j) prevM[i][j] = prev[s];
        else { prevM[i][j] = prev[s] * 0.5f; prevM[j][i] = prev[s] * 0.5f; }
    }
    float rs[K], cs[K], total = 0.0f;
    for (int i = 0; i < K; ++i) { rs[i] = 0.0f; cs[i] = 0.0f; }
    for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) { rs[i] += prevM[i][j]; cs[j] += prevM[i][j]; total += prevM[i][j]; }
    const float off = sw / 7.0f;
    const float b = 1.0f - sw - off;
    for (int s = 0; s < S; ++s) {
        const int i = sm.pi[s], j = sm.pj[s];
        float v = (b * b) * prevM[i][j] + (b * off) * (rs[i] + cs[j]) + (off * off) * total;
        if (i != j) {
            v += (b * b) * prevM[j][i] + (b * off) * (rs[j] + cs[i]) + (off * off) * total;
        }
        cur[s] = v;
    }
}

inline void transition_backward_parity(
    const float* nextb,
    const float* enext,
    const float sw,
    const StateMap& sm,
    float* cur
) {
    float tmpM[K][K];
    for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) tmpM[i][j] = 0.0f;
    for (int s = 0; s < S; ++s) {
        const int i = sm.pi[s], j = sm.pj[s];
        const float val = nextb[s] * enext[s];
        if (i == j) tmpM[i][j] = val;
        else { tmpM[i][j] = val; tmpM[j][i] = val; }
    }
    float rs[K], cs[K], total = 0.0f;
    for (int i = 0; i < K; ++i) { rs[i] = 0.0f; cs[i] = 0.0f; }
    for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) { rs[i] += tmpM[i][j]; cs[j] += tmpM[i][j]; total += tmpM[i][j]; }
    const float off = sw / 7.0f;
    const float b = 1.0f - sw - off;
    for (int s = 0; s < S; ++s) {
        const int i = sm.pi[s], j = sm.pj[s];
        float v = (b * b) * tmpM[i][j] + (b * off) * (rs[i] + cs[j]) + (off * off) * total;
        if (i != j) {
            v = 0.5f * (v + (b * b) * tmpM[j][i] + (b * off) * (rs[j] + cs[i]) + (off * off) * total);
        }
        cur[s] = v;
    }
}

inline float sum_float_vector(const float* v, int n) {
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 15 < n; i += 16) {
        acc = _mm512_add_ps(acc, _mm512_loadu_ps(v + i));
    }
    float tmp[16];
    _mm512_storeu_ps(tmp, acc);
    float sum = 0.0f;
    for (int j = 0; j < 16; ++j) sum += tmp[j];
    for (; i < n; ++i) sum += v[i];
    return sum;
#elif defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(v + i));
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, acc);
    float sum = 0.0f;
    for (int j = 0; j < 8; ++j) sum += tmp[j];
    for (; i < n; ++i) sum += v[i];
    return sum;
#else
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += v[i];
    return sum;
#endif
}

inline void scale_float_vector(float* v, int n, float c) {
#if defined(__AVX512F__)
    __m512 cc = _mm512_set1_ps(c);
    int i = 0;
    for (; i + 15 < n; i += 16) _mm512_storeu_ps(v + i, _mm512_mul_ps(_mm512_loadu_ps(v + i), cc));
    for (; i < n; ++i) v[i] *= c;
#elif defined(__AVX2__)
    __m256 cc = _mm256_set1_ps(c);
    int i = 0;
    for (; i + 7 < n; i += 8) _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_loadu_ps(v + i), cc));
    for (; i < n; ++i) v[i] *= c;
#else
    for (int i = 0; i < n; ++i) v[i] *= c;
#endif
}

inline void normalize(float* v, int n) {
    float sum = sum_float_vector(v, n);
    if (!std::isfinite(sum) || sum <= 0.0f) {
        const float u = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) v[i] = u;
    } else {
        scale_float_vector(v, n, 1.0f / sum);
    }
}

inline void reduce_state_to_outputs(
    const float* st,
    const float* founder_alt,
    int n_positions,
    int p,
    const StateMap& sm,
    float* dosage_out,
    float* gp_out
) {
    float min_a = founder_alt[p];
    float max_a = founder_alt[p];
    for (int k = 1; k < K; ++k) {
        const float a = founder_alt[(size_t)k * n_positions + p];
        if (a < min_a) min_a = a;
        if (a > max_a) max_a = a;
    }
    float ds = 0.0f, gp0 = 0.0f, gp2 = 0.0f;
    if (max_a - min_a < 1e-7f) {
        const float a = 0.5f * (min_a + max_a);
        ds = 2.0f * a;
        gp0 = (1.0f - a) * (1.0f - a);
        gp2 = a * a;
    } else {
        for (int s = 0; s < S; ++s) {
            const int i = sm.pi[s], j = sm.pj[s];
            const float ai = founder_alt[(size_t)i * n_positions + p];
            const float aj = founder_alt[(size_t)j * n_positions + p];
            ds += st[s] * (ai + aj);
            gp0 += st[s] * (1.0f - ai) * (1.0f - aj);
            gp2 += st[s] * ai * aj;
        }
    }
    float gp1 = 1.0f - gp0 - gp2;
    if (gp1 < 0.0f) gp1 = 0.0f;
    float gsum = gp0 + gp1 + gp2;
    if (!std::isfinite(gsum) || gsum <= 0.0f) {
        gp0 = gp1 = gp2 = 1.0f / 3.0f;
        gsum = 1.0f;
    }
    *dosage_out = ds;
    gp_out[0] = gp0 / gsum;
    gp_out[1] = gp1 / gsum;
    gp_out[2] = gp2 / gsum;
}

template <typename EmitBuilder>
int reduce_with_emitter(
    int n_samples,
    int n_positions,
    const float* sw,
    const float* founder_alt,
    float* dosage,
    float* gp,
    int n_threads,
    EmitBuilder emit_builder
) {
    if (n_samples <= 0 || n_positions <= 0) return 0;
    const StateMap sm;
    set_threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        std::vector<float> emit((size_t)n_positions * S);
        float logrow[S];
        for (int p = 0; p < n_positions; ++p) {
            emit_builder(sample, p, logrow);
            logu_to_emit_row(logrow, emit.data() + (size_t)p * S);
        }
        std::vector<float> alpha((size_t)n_positions * S), beta((size_t)n_positions * S);
        float sum = 0.0f;
        for (int s = 0; s < S; ++s) {
            const float mult = (sm.pi[s] == sm.pj[s]) ? 1.0f : 2.0f;
            alpha[s] = mult / 64.0f * emit[s];
            sum += alpha[s];
        }
        if (!std::isfinite(sum) || sum <= 0.0f) sum = 1.0f;
        for (int s = 0; s < S; ++s) alpha[s] /= sum;
        for (int p = 1; p < n_positions; ++p) {
            const float* prev = alpha.data() + (size_t)(p - 1) * S;
            float* cur = alpha.data() + (size_t)p * S;
            transition_forward_parity(prev, sw[(size_t)sample * n_positions + p], sm, cur);
            const float* erow = emit.data() + (size_t)p * S;
            for (int s = 0; s < S; ++s) cur[s] *= erow[s];
            normalize(cur, S);
        }
        for (int s = 0; s < S; ++s) beta[(size_t)(n_positions - 1) * S + s] = 1.0f / S;
        for (int p = n_positions - 2; p >= 0; --p) {
            float* cur = beta.data() + (size_t)p * S;
            const float* nextb = beta.data() + (size_t)(p + 1) * S;
            const float* enext = emit.data() + (size_t)(p + 1) * S;
            transition_backward_parity(nextb, enext, sw[(size_t)sample * n_positions + p + 1], sm, cur);
            normalize(cur, S);
        }
        for (int p = 0; p < n_positions; ++p) {
            float st[S];
            sum = 0.0f;
            const float* ap = alpha.data() + (size_t)p * S;
            const float* bp = beta.data() + (size_t)p * S;
            for (int s = 0; s < S; ++s) { st[s] = ap[s] * bp[s]; sum += st[s]; }
            if (!std::isfinite(sum) || sum <= 0.0f) sum = 1.0f;
            for (int s = 0; s < S; ++s) st[s] /= sum;
            reduce_state_to_outputs(
                st,
                founder_alt,
                n_positions,
                p,
                sm,
                dosage + (size_t)sample * n_positions + p,
                gp + ((size_t)sample * n_positions + p) * 3
            );
        }
    }
    return 0;
}


template <typename EmitBuilder>
int reduce_with_emitter_checkpointed(
    int n_samples,
    int n_positions,
    const float* sw,
    const float* founder_alt,
    float* dosage,
    float* gp,
    int n_threads,
    int checkpoint_interval,
    EmitBuilder emit_builder
) {
    if (n_samples <= 0 || n_positions <= 0) return 0;
    if (checkpoint_interval <= 0 || checkpoint_interval >= n_positions) {
        return reduce_with_emitter(n_samples, n_positions, sw, founder_alt, dosage, gp, n_threads, emit_builder);
    }
    checkpoint_interval = std::max(1, checkpoint_interval);
    const StateMap sm;
    set_threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        const int n_cp = (n_positions + checkpoint_interval - 1) / checkpoint_interval;
        std::vector<float> alpha_cp((size_t)n_cp * S);
        std::vector<float> prev(S), cur(S), erow(S);
        float logrow[S];

        // Forward pass storing only alpha at segment starts.
        emit_builder(sample, 0, logrow);
        logu_to_emit_row(logrow, erow.data());
        float z = 0.0f;
        for (int s = 0; s < S; ++s) {
            const float mult = (sm.pi[s] == sm.pj[s]) ? 1.0f : 2.0f;
            prev[s] = mult / 64.0f * erow[s];
            z += prev[s];
        }
        if (!std::isfinite(z) || z <= 0.0f) z = 1.0f;
        for (int s = 0; s < S; ++s) prev[s] /= z;
        std::memcpy(alpha_cp.data(), prev.data(), sizeof(float) * S);
        for (int p = 1; p < n_positions; ++p) {
            transition_forward_parity(prev.data(), sw[(size_t)sample * n_positions + p], sm, cur.data());
            emit_builder(sample, p, logrow);
            logu_to_emit_row(logrow, erow.data());
            for (int s = 0; s < S; ++s) cur[s] *= erow[s];
            normalize(cur.data(), S);
            if ((p % checkpoint_interval) == 0) {
                const int cpi = p / checkpoint_interval;
                std::memcpy(alpha_cp.data() + (size_t)cpi * S, cur.data(), sizeof(float) * S);
            }
            prev.swap(cur);
        }

        // Backward/output pass by segments. Recompute alpha locally from nearest checkpoint.
        std::vector<float> beta_start_next(S, 1.0f / S);  // beta at segment end+1 for the already-processed right segment.
        std::vector<float> alpha_seg, beta_seg, emit_seg, enext(S), st(S);
        for (int seg_start = ((n_positions - 1) / checkpoint_interval) * checkpoint_interval; seg_start >= 0; seg_start -= checkpoint_interval) {
            const int seg_end = std::min(n_positions - 1, seg_start + checkpoint_interval - 1);
            const int len = seg_end - seg_start + 1;
            alpha_seg.assign((size_t)len * S, 0.0f);
            beta_seg.assign((size_t)len * S, 0.0f);
            emit_seg.assign((size_t)len * S, 0.0f);

            const int cpi = seg_start / checkpoint_interval;
            std::memcpy(alpha_seg.data(), alpha_cp.data() + (size_t)cpi * S, sizeof(float) * S);
            emit_builder(sample, seg_start, logrow);
            logu_to_emit_row(logrow, emit_seg.data());
            for (int off = 1; off < len; ++off) {
                const int p = seg_start + off;
                transition_forward_parity(alpha_seg.data() + (size_t)(off - 1) * S, sw[(size_t)sample * n_positions + p], sm, alpha_seg.data() + (size_t)off * S);
                emit_builder(sample, p, logrow);
                logu_to_emit_row(logrow, emit_seg.data() + (size_t)off * S);
                float* arow = alpha_seg.data() + (size_t)off * S;
                const float* e = emit_seg.data() + (size_t)off * S;
                for (int s = 0; s < S; ++s) arow[s] *= e[s];
                normalize(arow, S);
            }

            float* beta_last = beta_seg.data() + (size_t)(len - 1) * S;
            if (seg_end == n_positions - 1) {
                for (int s = 0; s < S; ++s) beta_last[s] = 1.0f / S;
            } else {
                emit_builder(sample, seg_end + 1, logrow);
                logu_to_emit_row(logrow, enext.data());
                transition_backward_parity(beta_start_next.data(), enext.data(), sw[(size_t)sample * n_positions + seg_end + 1], sm, beta_last);
                normalize(beta_last, S);
            }
            for (int off = len - 2; off >= 0; --off) {
                transition_backward_parity(
                    beta_seg.data() + (size_t)(off + 1) * S,
                    emit_seg.data() + (size_t)(off + 1) * S,
                    sw[(size_t)sample * n_positions + seg_start + off + 1],
                    sm,
                    beta_seg.data() + (size_t)off * S
                );
                normalize(beta_seg.data() + (size_t)off * S, S);
            }

            for (int off = 0; off < len; ++off) {
                const int p = seg_start + off;
                const float* ap = alpha_seg.data() + (size_t)off * S;
                const float* bp = beta_seg.data() + (size_t)off * S;
                float sum = 0.0f;
                for (int s = 0; s < S; ++s) { st[s] = ap[s] * bp[s]; sum += st[s]; }
                if (!std::isfinite(sum) || sum <= 0.0f) sum = 1.0f;
                for (int s = 0; s < S; ++s) st[s] /= sum;
                reduce_state_to_outputs(
                    st.data(), founder_alt, n_positions, p, sm,
                    dosage + (size_t)sample * n_positions + p,
                    gp + ((size_t)sample * n_positions + p) * 3
                );
            }
            std::memcpy(beta_start_next.data(), beta_seg.data(), sizeof(float) * S);
        }
    }
    return 0;
}

}  // namespace

extern "C" int stitchcont_k8_counts_to_logu(
    const float* ref_obs,
    const float* alt_obs,
    const float* other_obs,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    float sequencing_error_rate,
    float min_emission_prob,
    float* logu,
    int n_threads
) {
    if (!ref_obs || !alt_obs || !founder_alt || !logu || n_samples <= 0 || n_positions <= 0) return -1;
    const StateMap sm;
    set_threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        for (int p = 0; p < n_positions; ++p) {
            const size_t idx = (size_t)sample * n_positions + p;
            compute_logu_row_from_counts(
                ref_obs[idx],
                alt_obs[idx],
                other_obs ? other_obs[idx] : 0.0f,
                founder_alt,
                n_positions,
                p,
                sequencing_error_rate,
                min_emission_prob,
                sm,
                logu + idx * S
            );
        }
    }
    return 0;
}

extern "C" int stitchcont_k8_unordered_reduce(
    const float* logu,
    const float* sw,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!logu || !sw || !founder_alt || !dosage || !gp) return -1;
    return reduce_with_emitter(
        n_samples,
        n_positions,
        sw,
        founder_alt,
        dosage,
        gp,
        n_threads,
        [logu, n_positions](int sample, int p, float* out36) {
            const float* row = logu + ((size_t)sample * n_positions + p) * S;
            for (int s = 0; s < S; ++s) out36[s] = row[s];
        }
    );
}


extern "C" int stitchcont_k8_unordered_reduce_checkpointed(
    const float* logu,
    const float* sw,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    int checkpoint_interval,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!logu || !sw || !founder_alt || !dosage || !gp) return -1;
    return reduce_with_emitter_checkpointed(
        n_samples,
        n_positions,
        sw,
        founder_alt,
        dosage,
        gp,
        n_threads,
        checkpoint_interval,
        [logu, n_positions](int sample, int p, float* out36) {
            const float* row = logu + ((size_t)sample * n_positions + p) * S;
            for (int s = 0; s < S; ++s) out36[s] = row[s];
        }
    );
}

extern "C" int stitchcont_k8_counts_reduce(
    const float* ref_obs,
    const float* alt_obs,
    const float* other_obs,
    const float* sw,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    float sequencing_error_rate,
    float min_emission_prob,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!ref_obs || !alt_obs || !sw || !founder_alt || !dosage || !gp) return -1;
    const StateMap sm;
    return reduce_with_emitter(
        n_samples,
        n_positions,
        sw,
        founder_alt,
        dosage,
        gp,
        n_threads,
        [ref_obs, alt_obs, other_obs, founder_alt, n_positions, sequencing_error_rate, min_emission_prob, &sm](int sample, int p, float* out36) {
            const size_t idx = (size_t)sample * n_positions + p;
            compute_logu_row_from_counts(
                ref_obs[idx],
                alt_obs[idx],
                other_obs ? other_obs[idx] : 0.0f,
                founder_alt,
                n_positions,
                p,
                sequencing_error_rate,
                min_emission_prob,
                sm,
                out36
            );
        }
    );
}


extern "C" int stitchcont_k8_counts_reduce_checkpointed(
    const float* ref_obs,
    const float* alt_obs,
    const float* other_obs,
    const float* sw,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    int checkpoint_interval,
    float sequencing_error_rate,
    float min_emission_prob,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!ref_obs || !alt_obs || !sw || !founder_alt || !dosage || !gp) return -1;
    const StateMap sm;
    return reduce_with_emitter_checkpointed(
        n_samples,
        n_positions,
        sw,
        founder_alt,
        dosage,
        gp,
        n_threads,
        checkpoint_interval,
        [ref_obs, alt_obs, other_obs, founder_alt, n_positions, sequencing_error_rate, min_emission_prob, &sm](int sample, int p, float* out36) {
            const size_t idx = (size_t)sample * n_positions + p;
            compute_logu_row_from_counts(
                ref_obs[idx],
                alt_obs[idx],
                other_obs ? other_obs[idx] : 0.0f,
                founder_alt,
                n_positions,
                p,
                sequencing_error_rate,
                min_emission_prob,
                sm,
                out36
            );
        }
    );
}

extern "C" int stitchcont_k8_apply_fragments_unordered(
    float* logu,
    const int64_t* fragment_sample_offsets,
    const int64_t* fragment_center_idx,
    const int64_t* fragment_obs_offsets,
    const int64_t* fragment_obs_pos_idx,
    const int8_t* fragment_obs_code,
    const float* fragment_obs_qual,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    int64_t n_fragments,
    int64_t n_observations,
    float sequencing_error_rate,
    float min_emission_prob,
    int mode_replace,
    int rescale,
    float max_emission_matrix_difference,
    int n_threads
) {
    if (!logu || !fragment_sample_offsets || !fragment_center_idx || !fragment_obs_offsets || !fragment_obs_pos_idx || !fragment_obs_code || !founder_alt) return -1;
    if (n_samples <= 0 || n_positions <= 0 || n_fragments < 0 || n_observations < 0) return -2;
    const StateMap sm;
    set_threads(n_threads);
    std::vector<unsigned char> touched((size_t)n_samples * n_positions, 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        const int64_t f_start = fragment_sample_offsets[sample];
        const int64_t f_stop = fragment_sample_offsets[sample + 1];
        float hap_log[K];
        for (int64_t f = f_start; f < f_stop; ++f) {
            if (f < 0 || f >= n_fragments) continue;
            const int64_t center = fragment_center_idx[f];
            if (center < 0 || center >= n_positions) continue;
            const int64_t obs_start = fragment_obs_offsets[f];
            const int64_t obs_stop = fragment_obs_offsets[f + 1];
            if (obs_start < 0 || obs_stop < obs_start || obs_stop > n_observations) continue;
            for (int k = 0; k < K; ++k) hap_log[k] = 0.0f;
            for (int64_t o = obs_start; o < obs_stop; ++o) {
                const int64_t pos = fragment_obs_pos_idx[o];
                if (pos < 0 || pos >= n_positions) continue;
                const int code = static_cast<int>(fragment_obs_code[o]);
                const float err = base_error_from_qual(fragment_obs_qual ? fragment_obs_qual[o] : sequencing_error_rate, sequencing_error_rate);
                for (int k = 0; k < K; ++k) {
                    const float a = founder_alt[(size_t)k * n_positions + pos];
                    hap_log[k] += obs_log_prob_for_founder(a, code, err, min_emission_prob);
                }
            }
            const size_t touched_idx = (size_t)sample * n_positions + static_cast<size_t>(center);
            float* row = logu + touched_idx * S;
            if (mode_replace && touched[touched_idx] == 0) {
                for (int s = 0; s < S; ++s) row[s] = 0.0f;
                touched[touched_idx] = 1;
            } else if (!mode_replace) {
                touched[touched_idx] = 1;
            }
            for (int s = 0; s < S; ++s) {
                const int i = sm.pi[s], j = sm.pj[s];
                const float maxh = std::max(hap_log[i], hap_log[j]);
                const float pair_log = maxh + std::log(0.5f * (std::exp(hap_log[i] - maxh) + std::exp(hap_log[j] - maxh)));
                row[s] += pair_log;
            }
        }
    }

    if (rescale) {
        const float min_state_log = -std::log(std::max(max_emission_matrix_difference, 1.000001f));
        // NOTE: the loop bound must be computed in 64-bit.  n_samples and
        // n_positions are int; at 50k samples x 100k variants their product is
        // 5e9, which overflows signed 32-bit (max 2.147e9) and yields UB / a
        // negative bound, silently skipping the rescale.  OpenMP requires a
        // signed loop counter, so use a 64-bit signed index.
        const long long n_cells = (long long)n_samples * (long long)n_positions;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long idx = 0; idx < n_cells; ++idx) {
            if (!touched[(size_t)idx]) continue;
            float* row = logu + (size_t)idx * S;
            float maxv = row[0];
            for (int s = 1; s < S; ++s) if (row[s] > maxv) maxv = row[s];
            for (int s = 0; s < S; ++s) row[s] = std::max(row[s] - maxv, min_state_log);
        }
    }
    return 0;
}

namespace {

inline void logkk_to_emit_row(const float* row, int kk, float* erow) {
    float maxv = row[0];
    float minv = row[0];
    for (int idx = 1; idx < kk; ++idx) {
        if (row[idx] > maxv) maxv = row[idx];
        if (row[idx] < minv) minv = row[idx];
    }
    if (maxv - minv < 1e-7f) {
        for (int idx = 0; idx < kk; ++idx) erow[idx] = 1.0f;
    } else {
        for (int idx = 0; idx < kk; ++idx) erow[idx] = std::exp(row[idx] - maxv);
    }
}

inline void normalize_dyn(float* v, int n) {
    float sum = sum_float_vector(v, n);
    if (!std::isfinite(sum) || sum <= 0.0f) {
        const float u = 1.0f / static_cast<float>(n);
        for (int i = 0; i < n; ++i) v[i] = u;
    } else {
        scale_float_vector(v, n, 1.0f / sum);
    }
}

inline void low_rank_forward_dyn(
    const float* m,
    float sw,
    const float* left,
    const float* right,
    int k,
    int rank,
    float* out
) {
    const float s = std::min(1.0f, std::max(0.0f, sw));
    const float a = 1.0f - s;
    const float b = s;
    const int kk = k * k;
    const float aa = a * a;
    for (int idx = 0; idx < kk; ++idx) out[idx] = aa * m[idx];
    if (b == 0.0f) return;
    std::vector<float> ut_m((size_t)rank * k, 0.0f);  // r,j = U.T M
    std::vector<float> m_u((size_t)k * rank, 0.0f);   // i,r = M U
    std::vector<float> small((size_t)rank * rank, 0.0f); // r,t = U.T M U

    for (int r = 0; r < rank; ++r) {
        for (int j = 0; j < k; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) acc += left[(size_t)i * rank + r] * m[(size_t)i * k + j];
            ut_m[(size_t)r * k + j] = acc;
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int r = 0; r < rank; ++r) {
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) acc += m[(size_t)i * k + j] * left[(size_t)j * rank + r];
            m_u[(size_t)i * rank + r] = acc;
        }
    }
    for (int r = 0; r < rank; ++r) {
        for (int t = 0; t < rank; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) acc += ut_m[(size_t)r * k + j] * left[(size_t)j * rank + t];
            small[(size_t)r * rank + t] = acc;
        }
    }
    const float ab = a * b;
    const float bb = b * b;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            float term1 = 0.0f;
            float term2 = 0.0f;
            for (int r = 0; r < rank; ++r) {
                term1 += right[(size_t)i * rank + r] * ut_m[(size_t)r * k + j];
                term2 += m_u[(size_t)i * rank + r] * right[(size_t)j * rank + r];
            }
            float term3 = 0.0f;
            for (int r = 0; r < rank; ++r) {
                const float ri = right[(size_t)i * rank + r];
                for (int t = 0; t < rank; ++t) {
                    term3 += ri * small[(size_t)r * rank + t] * right[(size_t)j * rank + t];
                }
            }
            float v = out[(size_t)i * k + j] + ab * (term1 + term2) + bb * term3;
            out[(size_t)i * k + j] = std::isfinite(v) && v > 0.0f ? v : 0.0f;
        }
    }
}

inline void low_rank_backward_dyn(
    const float* m,
    float sw,
    const float* left,
    const float* right,
    int k,
    int rank,
    float* out
) {
    const float s = std::min(1.0f, std::max(0.0f, sw));
    const float a = 1.0f - s;
    const float b = s;
    const int kk = k * k;
    const float aa = a * a;
    for (int idx = 0; idx < kk; ++idx) out[idx] = aa * m[idx];
    if (b == 0.0f) return;
    std::vector<float> vt_m((size_t)rank * k, 0.0f);  // r,j = V.T M
    std::vector<float> m_v((size_t)k * rank, 0.0f);   // i,r = M V
    std::vector<float> small((size_t)rank * rank, 0.0f); // r,t = V.T M V
    for (int r = 0; r < rank; ++r) {
        for (int j = 0; j < k; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < k; ++i) acc += right[(size_t)i * rank + r] * m[(size_t)i * k + j];
            vt_m[(size_t)r * k + j] = acc;
        }
    }
    for (int i = 0; i < k; ++i) {
        for (int r = 0; r < rank; ++r) {
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) acc += m[(size_t)i * k + j] * right[(size_t)j * rank + r];
            m_v[(size_t)i * rank + r] = acc;
        }
    }
    for (int r = 0; r < rank; ++r) {
        for (int t = 0; t < rank; ++t) {
            float acc = 0.0f;
            for (int j = 0; j < k; ++j) acc += vt_m[(size_t)r * k + j] * right[(size_t)j * rank + t];
            small[(size_t)r * rank + t] = acc;
        }
    }
    const float ab = a * b;
    const float bb = b * b;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            float term1 = 0.0f;
            float term2 = 0.0f;
            for (int r = 0; r < rank; ++r) {
                term1 += left[(size_t)i * rank + r] * vt_m[(size_t)r * k + j];
                term2 += m_v[(size_t)i * rank + r] * left[(size_t)j * rank + r];
            }
            float term3 = 0.0f;
            for (int r = 0; r < rank; ++r) {
                const float li = left[(size_t)i * rank + r];
                for (int t = 0; t < rank; ++t) {
                    term3 += li * small[(size_t)r * rank + t] * left[(size_t)j * rank + t];
                }
            }
            float v = out[(size_t)i * k + j] + ab * (term1 + term2) + bb * term3;
            out[(size_t)i * k + j] = std::isfinite(v) && v > 0.0f ? v : 0.0f;
        }
    }
}

inline void reduce_ordered_state_to_outputs_dyn(
    const float* st,
    const float* founder_alt,
    int n_positions,
    int p,
    int k,
    float* dosage_out,
    float* gp_out
) {
    float min_a = founder_alt[p];
    float max_a = founder_alt[p];
    for (int h = 1; h < k; ++h) {
        const float a = founder_alt[(size_t)h * n_positions + p];
        if (a < min_a) min_a = a;
        if (a > max_a) max_a = a;
    }
    float ds = 0.0f, gp0 = 0.0f, gp2 = 0.0f;
    if (max_a - min_a < 1e-7f) {
        const float a = 0.5f * (min_a + max_a);
        ds = 2.0f * a;
        gp0 = (1.0f - a) * (1.0f - a);
        gp2 = a * a;
    } else {
        for (int i = 0; i < k; ++i) {
            const float ai = founder_alt[(size_t)i * n_positions + p];
            for (int j = 0; j < k; ++j) {
                const float aj = founder_alt[(size_t)j * n_positions + p];
                const float pr = st[(size_t)i * k + j];
                ds += pr * (ai + aj);
                gp0 += pr * (1.0f - ai) * (1.0f - aj);
                gp2 += pr * ai * aj;
            }
        }
    }
    float gp1 = 1.0f - gp0 - gp2;
    if (gp1 < 0.0f) gp1 = 0.0f;
    float gsum = gp0 + gp1 + gp2;
    if (!std::isfinite(gsum) || gsum <= 0.0f) {
        gp0 = gp1 = gp2 = 1.0f / 3.0f;
        gsum = 1.0f;
    }
    *dosage_out = ds;
    gp_out[0] = gp0 / gsum;
    gp_out[1] = gp1 / gsum;
    gp_out[2] = gp2 / gsum;
}


inline void compute_logkk_row_from_counts_dyn(
    const float ref_obs,
    const float alt_obs,
    const float other_obs,
    const float* founder_alt,
    const int n_positions,
    const int p,
    const int k,
    const float eps,
    const float min_prob,
    float* outkk
) {
    const float p_err_other = eps / 3.0f;
    for (int i = 0; i < k; ++i) {
        const float ai = founder_alt[(size_t)i * n_positions + p];
        for (int j = 0; j < k; ++j) {
            const float aj = founder_alt[(size_t)j * n_positions + p];
            float hap_alt = 0.5f * (ai + aj);
            if (hap_alt < 0.0f) hap_alt = 0.0f;
            if (hap_alt > 1.0f) hap_alt = 1.0f;
            const float p_alt = clamp_prob(hap_alt * (1.0f - eps) + (1.0f - hap_alt) * p_err_other, min_prob);
            const float p_ref = clamp_prob((1.0f - hap_alt) * (1.0f - eps) + hap_alt * p_err_other, min_prob);
            const float p_oth = clamp_prob(p_err_other, min_prob);
            outkk[(size_t)i * k + j] = alt_obs * std::log(p_alt) + ref_obs * std::log(p_ref) + other_obs * std::log(p_oth);
        }
    }
}

inline void logkk_counts_to_emit_row_dyn(
    const float ref_obs,
    const float alt_obs,
    const float other_obs,
    const float* founder_alt,
    const int n_positions,
    const int p,
    const int k,
    const float eps,
    const float min_prob,
    float* erow,
    float* tmp_log
) {
    if (ref_obs == 0.0f && alt_obs == 0.0f && other_obs == 0.0f) {
        const int kk = k * k;
        for (int idx = 0; idx < kk; ++idx) erow[idx] = 1.0f;
        return;
    }
    compute_logkk_row_from_counts_dyn(ref_obs, alt_obs, other_obs, founder_alt, n_positions, p, k, eps, min_prob, tmp_log);
    logkk_to_emit_row(tmp_log, k * k, erow);
}

inline void build_sparse_topk(
    const float* offdiag,
    int k,
    int top_k,
    std::vector<int>& dest,
    std::vector<float>& weight
) {
    top_k = std::max(1, std::min(top_k, k));
    dest.assign((size_t)k * top_k, 0);
    weight.assign((size_t)k * top_k, 0.0f);
    std::vector<int> idx(k);
    for (int src = 0; src < k; ++src) {
        for (int j = 0; j < k; ++j) idx[j] = j;
        std::partial_sort(idx.begin(), idx.begin() + top_k, idx.end(), [offdiag, src, k](int a, int b) {
            return offdiag[(size_t)src * k + a] > offdiag[(size_t)src * k + b];
        });
        float sum = 0.0f;
        for (int m = 0; m < top_k; ++m) {
            const int d = idx[m];
            const float w = std::max(0.0f, offdiag[(size_t)src * k + d]);
            dest[(size_t)src * top_k + m] = d;
            weight[(size_t)src * top_k + m] = w;
            sum += w;
        }
        if (!std::isfinite(sum) || sum <= 0.0f) {
            for (int m = 0; m < top_k; ++m) {
                dest[(size_t)src * top_k + m] = (src + 1 + m) % k;
                weight[(size_t)src * top_k + m] = 1.0f / static_cast<float>(top_k);
            }
        } else {
            for (int m = 0; m < top_k; ++m) weight[(size_t)src * top_k + m] /= sum;
        }
    }
}

inline void sparse_forward_dyn(
    const float* m,
    float sw,
    const int* dest,
    const float* weight,
    int k,
    int top_k,
    float* out
) {
    const int kk = k * k;
    for (int idx = 0; idx < kk; ++idx) out[idx] = 0.0f;
    const float s = std::min(1.0f, std::max(0.0f, sw));
    const float a = 1.0f - s;
    const float b = s;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            const float mv = m[(size_t)i * k + j];
            if (mv == 0.0f) continue;
            out[(size_t)i * k + j] += mv * a * a;
            for (int m1 = 0; m1 < top_k; ++m1) {
                const int d1 = dest[(size_t)i * top_k + m1];
                const float p1 = b * weight[(size_t)i * top_k + m1];
                out[(size_t)d1 * k + j] += mv * p1 * a;
                for (int m2 = 0; m2 < top_k; ++m2) {
                    const int d2 = dest[(size_t)j * top_k + m2];
                    const float p2 = b * weight[(size_t)j * top_k + m2];
                    out[(size_t)d1 * k + d2] += mv * p1 * p2;
                }
            }
            for (int m2 = 0; m2 < top_k; ++m2) {
                const int d2 = dest[(size_t)j * top_k + m2];
                const float p2 = b * weight[(size_t)j * top_k + m2];
                out[(size_t)i * k + d2] += mv * a * p2;
            }
        }
    }
}

inline void sparse_backward_dyn(
    const float* m,
    float sw,
    const int* dest,
    const float* weight,
    int k,
    int top_k,
    float* out
) {
    const float s = std::min(1.0f, std::max(0.0f, sw));
    const float a = 1.0f - s;
    const float b = s;
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            float acc = a * a * m[(size_t)i * k + j];
            for (int m1 = 0; m1 < top_k; ++m1) {
                const int d1 = dest[(size_t)i * top_k + m1];
                const float p1 = b * weight[(size_t)i * top_k + m1];
                acc += p1 * a * m[(size_t)d1 * k + j];
                for (int m2 = 0; m2 < top_k; ++m2) {
                    const int d2 = dest[(size_t)j * top_k + m2];
                    const float p2 = b * weight[(size_t)j * top_k + m2];
                    acc += p1 * p2 * m[(size_t)d1 * k + d2];
                }
            }
            for (int m2 = 0; m2 < top_k; ++m2) {
                const int d2 = dest[(size_t)j * top_k + m2];
                const float p2 = b * weight[(size_t)j * top_k + m2];
                acc += a * p2 * m[(size_t)i * k + d2];
            }
            out[(size_t)i * k + j] = std::isfinite(acc) && acc > 0.0f ? acc : 0.0f;
        }
    }
}


} // namespace

extern "C" int stitchcont_diploid_low_rank_reduce(
    const float* log_emission,
    const float* sw,
    const float* founder_alt,
    const float* left_factor,
    const float* right_factor,
    int n_samples,
    int n_positions,
    int k,
    int rank,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!log_emission || !sw || !founder_alt || !left_factor || !right_factor || !dosage || !gp) return -1;
    if (n_samples <= 0 || n_positions <= 0 || k <= 0 || rank <= 0) return -2;
    const int kk = k * k;
    set_threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        std::vector<float> emit((size_t)n_positions * kk);
        for (int p = 0; p < n_positions; ++p) {
            const float* logrow = log_emission + (((size_t)sample * n_positions + p) * kk);
            logkk_to_emit_row(logrow, kk, emit.data() + (size_t)p * kk);
        }
        std::vector<float> alpha((size_t)n_positions * kk);
        std::vector<float> beta((size_t)n_positions * kk);
        float* a0 = alpha.data();
        float z = 0.0f;
        const float* e0 = emit.data();
        for (int idx = 0; idx < kk; ++idx) { a0[idx] = e0[idx]; z += a0[idx]; }
        if (!std::isfinite(z) || z <= 0.0f) z = 1.0f;
        for (int idx = 0; idx < kk; ++idx) a0[idx] /= z;
        std::vector<float> pred(kk);
        for (int p = 1; p < n_positions; ++p) {
            const float* prev = alpha.data() + (size_t)(p - 1) * kk;
            float* cur = alpha.data() + (size_t)p * kk;
            low_rank_forward_dyn(prev, sw[(size_t)sample * n_positions + p], left_factor, right_factor, k, rank, pred.data());
            const float* erow = emit.data() + (size_t)p * kk;
            for (int idx = 0; idx < kk; ++idx) cur[idx] = pred[idx] * erow[idx];
            normalize_dyn(cur, kk);
        }
        float* blast = beta.data() + (size_t)(n_positions - 1) * kk;
        const float u = 1.0f / static_cast<float>(kk);
        for (int idx = 0; idx < kk; ++idx) blast[idx] = u;
        std::vector<float> nextterm(kk);
        for (int p = n_positions - 2; p >= 0; --p) {
            const float* nextb = beta.data() + (size_t)(p + 1) * kk;
            const float* enext = emit.data() + (size_t)(p + 1) * kk;
            for (int idx = 0; idx < kk; ++idx) nextterm[idx] = nextb[idx] * enext[idx];
            float* cur = beta.data() + (size_t)p * kk;
            low_rank_backward_dyn(nextterm.data(), sw[(size_t)sample * n_positions + p + 1], left_factor, right_factor, k, rank, cur);
            normalize_dyn(cur, kk);
        }
        std::vector<float> st(kk);
        for (int p = 0; p < n_positions; ++p) {
            const float* ap = alpha.data() + (size_t)p * kk;
            const float* bp = beta.data() + (size_t)p * kk;
            float sum = 0.0f;
            for (int idx = 0; idx < kk; ++idx) { st[idx] = ap[idx] * bp[idx]; sum += st[idx]; }
            if (!std::isfinite(sum) || sum <= 0.0f) sum = 1.0f;
            for (int idx = 0; idx < kk; ++idx) st[idx] /= sum;
            reduce_ordered_state_to_outputs_dyn(
                st.data(), founder_alt, n_positions, p, k,
                dosage + (size_t)sample * n_positions + p,
                gp + ((size_t)sample * n_positions + p) * 3
            );
        }
    }
    return 0;
}


extern "C" int stitchcont_diploid_low_rank_counts_reduce(
    const float* ref_obs,
    const float* alt_obs,
    const float* other_obs,
    const float* sw,
    const float* founder_alt,
    const float* left_factor,
    const float* right_factor,
    int n_samples,
    int n_positions,
    int k,
    int rank,
    float sequencing_error_rate,
    float min_emission_prob,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!ref_obs || !alt_obs || !sw || !founder_alt || !left_factor || !right_factor || !dosage || !gp) return -1;
    if (n_samples <= 0 || n_positions <= 0 || k <= 0 || rank <= 0) return -2;
    const int kk = k * k;
    set_threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        std::vector<float> emit((size_t)n_positions * kk);
        std::vector<float> tmp_log(kk);
        for (int p = 0; p < n_positions; ++p) {
            const size_t idx = (size_t)sample * n_positions + p;
            logkk_counts_to_emit_row_dyn(
                ref_obs[idx], alt_obs[idx], other_obs ? other_obs[idx] : 0.0f,
                founder_alt, n_positions, p, k, sequencing_error_rate, min_emission_prob,
                emit.data() + (size_t)p * kk, tmp_log.data()
            );
        }
        std::vector<float> alpha((size_t)n_positions * kk), beta((size_t)n_positions * kk), pred(kk), nextterm(kk), st(kk);
        float* a0 = alpha.data();
        const float* e0 = emit.data();
        float z = 0.0f;
        const float u0 = 1.0f / static_cast<float>(kk);
        for (int idx = 0; idx < kk; ++idx) { a0[idx] = u0 * e0[idx]; z += a0[idx]; }
        if (!std::isfinite(z) || z <= 0.0f) z = 1.0f;
        for (int idx = 0; idx < kk; ++idx) a0[idx] /= z;
        for (int p = 1; p < n_positions; ++p) {
            low_rank_forward_dyn(alpha.data() + (size_t)(p - 1) * kk, sw[(size_t)sample * n_positions + p], left_factor, right_factor, k, rank, pred.data());
            float* cur = alpha.data() + (size_t)p * kk;
            const float* erow = emit.data() + (size_t)p * kk;
            for (int idx = 0; idx < kk; ++idx) cur[idx] = pred[idx] * erow[idx];
            normalize_dyn(cur, kk);
        }
        float* blast = beta.data() + (size_t)(n_positions - 1) * kk;
        for (int idx = 0; idx < kk; ++idx) blast[idx] = u0;
        for (int p = n_positions - 2; p >= 0; --p) {
            const float* nextb = beta.data() + (size_t)(p + 1) * kk;
            const float* enext = emit.data() + (size_t)(p + 1) * kk;
            for (int idx = 0; idx < kk; ++idx) nextterm[idx] = nextb[idx] * enext[idx];
            float* cur = beta.data() + (size_t)p * kk;
            low_rank_backward_dyn(nextterm.data(), sw[(size_t)sample * n_positions + p + 1], left_factor, right_factor, k, rank, cur);
            normalize_dyn(cur, kk);
        }
        for (int p = 0; p < n_positions; ++p) {
            const float* ap = alpha.data() + (size_t)p * kk;
            const float* bp = beta.data() + (size_t)p * kk;
            float sum = 0.0f;
            for (int idx = 0; idx < kk; ++idx) { st[idx] = ap[idx] * bp[idx]; sum += st[idx]; }
            if (!std::isfinite(sum) || sum <= 0.0f) sum = 1.0f;
            for (int idx = 0; idx < kk; ++idx) st[idx] /= sum;
            reduce_ordered_state_to_outputs_dyn(st.data(), founder_alt, n_positions, p, k, dosage + (size_t)sample * n_positions + p, gp + ((size_t)sample * n_positions + p) * 3);
        }
    }
    return 0;
}

extern "C" int stitchcont_diploid_sparse_topk_reduce(
    const float* log_emission,
    const float* sw,
    const float* founder_alt,
    const float* offdiag,
    int n_samples,
    int n_positions,
    int k,
    int top_k,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!log_emission || !sw || !founder_alt || !offdiag || !dosage || !gp) return -1;
    if (n_samples <= 0 || n_positions <= 0 || k <= 0 || top_k <= 0) return -2;
    top_k = std::max(1, std::min(top_k, k));
    const int kk = k * k;
    std::vector<int> dest;
    std::vector<float> weight;
    build_sparse_topk(offdiag, k, top_k, dest, weight);
    set_threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        std::vector<float> emit((size_t)n_positions * kk), alpha((size_t)n_positions * kk), beta((size_t)n_positions * kk);
        for (int p = 0; p < n_positions; ++p) logkk_to_emit_row(log_emission + (((size_t)sample * n_positions + p) * kk), kk, emit.data() + (size_t)p * kk);
        float* a0 = alpha.data();
        const float* e0 = emit.data();
        float z = 0.0f;
        const float u0 = 1.0f / static_cast<float>(kk);
        for (int idx = 0; idx < kk; ++idx) { a0[idx] = u0 * e0[idx]; z += a0[idx]; }
        if (!std::isfinite(z) || z <= 0.0f) z = 1.0f;
        for (int idx = 0; idx < kk; ++idx) a0[idx] /= z;
        std::vector<float> pred(kk), nextterm(kk), st(kk);
        for (int p = 1; p < n_positions; ++p) {
            sparse_forward_dyn(alpha.data() + (size_t)(p - 1) * kk, sw[(size_t)sample * n_positions + p], dest.data(), weight.data(), k, top_k, pred.data());
            float* cur = alpha.data() + (size_t)p * kk;
            const float* erow = emit.data() + (size_t)p * kk;
            for (int idx = 0; idx < kk; ++idx) cur[idx] = pred[idx] * erow[idx];
            normalize_dyn(cur, kk);
        }
        float* blast = beta.data() + (size_t)(n_positions - 1) * kk;
        for (int idx = 0; idx < kk; ++idx) blast[idx] = u0;
        for (int p = n_positions - 2; p >= 0; --p) {
            const float* nextb = beta.data() + (size_t)(p + 1) * kk;
            const float* enext = emit.data() + (size_t)(p + 1) * kk;
            for (int idx = 0; idx < kk; ++idx) nextterm[idx] = nextb[idx] * enext[idx];
            float* cur = beta.data() + (size_t)p * kk;
            sparse_backward_dyn(nextterm.data(), sw[(size_t)sample * n_positions + p + 1], dest.data(), weight.data(), k, top_k, cur);
            normalize_dyn(cur, kk);
        }
        for (int p = 0; p < n_positions; ++p) {
            const float* ap = alpha.data() + (size_t)p * kk;
            const float* bp = beta.data() + (size_t)p * kk;
            float sum = 0.0f;
            for (int idx = 0; idx < kk; ++idx) { st[idx] = ap[idx] * bp[idx]; sum += st[idx]; }
            if (!std::isfinite(sum) || sum <= 0.0f) sum = 1.0f;
            for (int idx = 0; idx < kk; ++idx) st[idx] /= sum;
            reduce_ordered_state_to_outputs_dyn(st.data(), founder_alt, n_positions, p, k, dosage + (size_t)sample * n_positions + p, gp + ((size_t)sample * n_positions + p) * 3);
        }
    }
    return 0;
}

extern "C" int stitchcont_diploid_sparse_topk_counts_reduce(
    const float* ref_obs,
    const float* alt_obs,
    const float* other_obs,
    const float* sw,
    const float* founder_alt,
    const float* offdiag,
    int n_samples,
    int n_positions,
    int k,
    int top_k,
    float sequencing_error_rate,
    float min_emission_prob,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!ref_obs || !alt_obs || !sw || !founder_alt || !offdiag || !dosage || !gp) return -1;
    if (n_samples <= 0 || n_positions <= 0 || k <= 0 || top_k <= 0) return -2;
    top_k = std::max(1, std::min(top_k, k));
    const int kk = k * k;
    std::vector<int> dest;
    std::vector<float> weight;
    build_sparse_topk(offdiag, k, top_k, dest, weight);
    set_threads(n_threads);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int sample = 0; sample < n_samples; ++sample) {
        // Counts-fused implementation: build only this sample's emission matrix,
        // not the full n_samples*n_positions*k*k log-emission tensor.
        std::vector<float> emit((size_t)n_positions * kk);
        std::vector<float> tmp_log(kk);
        for (int pidx = 0; pidx < n_positions; ++pidx) {
            const size_t obs_idx = (size_t)sample * n_positions + pidx;
            logkk_counts_to_emit_row_dyn(
                ref_obs[obs_idx], alt_obs[obs_idx], other_obs ? other_obs[obs_idx] : 0.0f,
                founder_alt, n_positions, pidx, k, sequencing_error_rate, min_emission_prob,
                emit.data() + (size_t)pidx * kk, tmp_log.data()
            );
        }
        std::vector<float> alpha((size_t)n_positions * kk), beta((size_t)n_positions * kk);
        float* a0 = alpha.data();
        const float* e0 = emit.data();
        float z = 0.0f;
        const float u0 = 1.0f / static_cast<float>(kk);
        for (int idx = 0; idx < kk; ++idx) { a0[idx] = u0 * e0[idx]; z += a0[idx]; }
        if (!std::isfinite(z) || z <= 0.0f) z = 1.0f;
        for (int idx = 0; idx < kk; ++idx) a0[idx] /= z;
        std::vector<float> pred(kk), nextterm(kk), st(kk);
        for (int pidx = 1; pidx < n_positions; ++pidx) {
            sparse_forward_dyn(alpha.data() + (size_t)(pidx - 1) * kk, sw[(size_t)sample * n_positions + pidx], dest.data(), weight.data(), k, top_k, pred.data());
            float* cur = alpha.data() + (size_t)pidx * kk;
            const float* erow = emit.data() + (size_t)pidx * kk;
            for (int idx = 0; idx < kk; ++idx) cur[idx] = pred[idx] * erow[idx];
            normalize_dyn(cur, kk);
        }
        float* blast = beta.data() + (size_t)(n_positions - 1) * kk;
        for (int idx = 0; idx < kk; ++idx) blast[idx] = u0;
        for (int pidx = n_positions - 2; pidx >= 0; --pidx) {
            const float* nextb = beta.data() + (size_t)(pidx + 1) * kk;
            const float* enext = emit.data() + (size_t)(pidx + 1) * kk;
            for (int idx = 0; idx < kk; ++idx) nextterm[idx] = nextb[idx] * enext[idx];
            float* cur = beta.data() + (size_t)pidx * kk;
            sparse_backward_dyn(nextterm.data(), sw[(size_t)sample * n_positions + pidx + 1], dest.data(), weight.data(), k, top_k, cur);
            normalize_dyn(cur, kk);
        }
        for (int pidx = 0; pidx < n_positions; ++pidx) {
            const float* ap = alpha.data() + (size_t)pidx * kk;
            const float* bp = beta.data() + (size_t)pidx * kk;
            float sum = 0.0f;
            for (int idx = 0; idx < kk; ++idx) { st[idx] = ap[idx] * bp[idx]; sum += st[idx]; }
            if (!std::isfinite(sum) || sum <= 0.0f) sum = 1.0f;
            for (int idx = 0; idx < kk; ++idx) st[idx] /= sum;
            reduce_ordered_state_to_outputs_dyn(st.data(), founder_alt, n_positions, pidx, k, dosage + (size_t)sample * n_positions + pidx, gp + ((size_t)sample * n_positions + pidx) * 3);
        }
    }
    return 0;
}

extern "C" int stitchcont_k8_counts_fragments_reduce(
    const float* ref_obs,
    const float* alt_obs,
    const float* other_obs,
    const float* sw,
    const float* founder_alt,
    const int64_t* fragment_sample_offsets,
    const int64_t* fragment_center_idx,
    const int64_t* fragment_obs_offsets,
    const int64_t* fragment_obs_pos_idx,
    const int8_t* fragment_obs_code,
    const float* fragment_obs_qual,
    int n_samples,
    int n_positions,
    int64_t n_fragments,
    int64_t n_observations,
    float sequencing_error_rate,
    float min_emission_prob,
    int mode_replace,
    int rescale,
    float max_emission_matrix_difference,
    float* dosage,
    float* gp,
    int n_threads
) {
    if (!ref_obs || !alt_obs || !sw || !founder_alt || !dosage || !gp) return -1;
    std::vector<float> logu((size_t)n_samples * n_positions * S);
    int rc = stitchcont_k8_counts_to_logu(ref_obs, alt_obs, other_obs, founder_alt, n_samples, n_positions, sequencing_error_rate, min_emission_prob, logu.data(), n_threads);
    if (rc != 0) return rc;
    if (fragment_sample_offsets && fragment_center_idx && fragment_obs_offsets && fragment_obs_pos_idx && fragment_obs_code && n_fragments > 0 && n_observations > 0) {
        rc = stitchcont_k8_apply_fragments_unordered(
            logu.data(), fragment_sample_offsets, fragment_center_idx, fragment_obs_offsets, fragment_obs_pos_idx, fragment_obs_code, fragment_obs_qual,
            founder_alt, n_samples, n_positions, n_fragments, n_observations, sequencing_error_rate, min_emission_prob,
            mode_replace, rescale, max_emission_matrix_difference, n_threads
        );
        if (rc != 0) return rc;
    }
    return stitchcont_k8_unordered_reduce(logu.data(), sw, founder_alt, n_samples, n_positions, dosage, gp, n_threads);
}
