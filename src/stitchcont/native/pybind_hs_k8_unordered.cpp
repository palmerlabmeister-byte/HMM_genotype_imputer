#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "hs_k8_unordered.hpp"

namespace py = pybind11;

namespace {

void require_1d(const py::buffer_info& b, const char* name) {
    if (b.ndim != 1) {
        throw std::runtime_error(std::string(name) + " must be a 1-D C-contiguous array");
    }
}

void require_2d(const py::buffer_info& b, const char* name) {
    if (b.ndim != 2) {
        throw std::runtime_error(std::string(name) + " must be a 2-D C-contiguous array");
    }
}

void require_3d(const py::buffer_info& b, const char* name) {
    if (b.ndim != 3) {
        throw std::runtime_error(std::string(name) + " must be a 3-D C-contiguous array");
    }
}

void require_4d(const py::buffer_info& b, const char* name) {
    if (b.ndim != 4) {
        throw std::runtime_error(std::string(name) + " must be a 4-D C-contiguous array");
    }
}

void require_founder_alt(const py::buffer_info& fa_b, int n_positions) {
    require_2d(fa_b, "founder_alt");
    if (fa_b.shape[0] != 8 || fa_b.shape[1] != n_positions) {
        throw std::runtime_error("founder_alt must have shape [8, n_positions]");
    }
}

}  // namespace

py::dict run_k8_unordered_reduce(
    py::array_t<float, py::array::c_style | py::array::forcecast> log_emission_u,
    py::array_t<float, py::array::c_style | py::array::forcecast> switch_prob,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    int n_threads,
    int checkpoint_interval
) {
    py::buffer_info log_b = log_emission_u.request();
    py::buffer_info sw_b = switch_prob.request();
    py::buffer_info fa_b = founder_alt.request();
    require_3d(log_b, "log_emission_u");
    require_2d(sw_b, "switch_prob");

    const int n_samples = static_cast<int>(log_b.shape[0]);
    const int n_positions = static_cast<int>(log_b.shape[1]);
    const int n_states = static_cast<int>(log_b.shape[2]);
    if (n_states != 36) {
        throw std::runtime_error("K8 unordered backend requires log_emission_u.shape[2] == 36");
    }
    if (sw_b.shape[0] != n_samples || sw_b.shape[1] != n_positions) {
        throw std::runtime_error("switch_prob must have shape [n_samples, n_positions]");
    }
    require_founder_alt(fa_b, n_positions);

    py::array_t<float> dosage({n_samples, n_positions});
    py::array_t<float> gp({n_samples, n_positions, 3});
    py::buffer_info d_b = dosage.request();
    py::buffer_info gp_b = gp.request();

    int rc = 0;
    if (checkpoint_interval > 0) {
        rc = stitchcont_k8_unordered_reduce_checkpointed(
            static_cast<const float*>(log_b.ptr),
            static_cast<const float*>(sw_b.ptr),
            static_cast<const float*>(fa_b.ptr),
            n_samples,
            n_positions,
            checkpoint_interval,
            static_cast<float*>(d_b.ptr),
            static_cast<float*>(gp_b.ptr),
            n_threads
        );
    } else {
        rc = stitchcont_k8_unordered_reduce(
            static_cast<const float*>(log_b.ptr),
            static_cast<const float*>(sw_b.ptr),
            static_cast<const float*>(fa_b.ptr),
            n_samples,
            n_positions,
            static_cast<float*>(d_b.ptr),
            static_cast<float*>(gp_b.ptr),
            n_threads
        );
    }
    if (rc != 0) {
        throw std::runtime_error("stitchcont_k8_unordered_reduce returned nonzero status " + std::to_string(rc));
    }

    py::dict out;
    out["dosage"] = dosage;
    out["genotype_posterior"] = gp;
    out["backend"] = py::str("pybind11_cpp_openmp_k8_unordered");
    return out;
}

py::array_t<float> build_k8_logu_from_counts(
    py::array_t<float, py::array::c_style | py::array::forcecast> ref_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> alt_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> other_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    float sequencing_error_rate,
    float min_emission_prob,
    int n_threads
) {
    py::buffer_info ref_b = ref_obs.request();
    py::buffer_info alt_b = alt_obs.request();
    py::buffer_info oth_b = other_obs.request();
    py::buffer_info fa_b = founder_alt.request();
    require_2d(ref_b, "ref_obs");
    require_2d(alt_b, "alt_obs");
    require_2d(oth_b, "other_obs");
    const int n_samples = static_cast<int>(ref_b.shape[0]);
    const int n_positions = static_cast<int>(ref_b.shape[1]);
    if (alt_b.shape[0] != n_samples || alt_b.shape[1] != n_positions || oth_b.shape[0] != n_samples || oth_b.shape[1] != n_positions) {
        throw std::runtime_error("ref_obs, alt_obs and other_obs must have the same shape");
    }
    require_founder_alt(fa_b, n_positions);
    py::array_t<float> logu({n_samples, n_positions, 36});
    py::buffer_info log_b = logu.request();
    int rc = stitchcont_k8_counts_to_logu(
        static_cast<const float*>(ref_b.ptr),
        static_cast<const float*>(alt_b.ptr),
        static_cast<const float*>(oth_b.ptr),
        static_cast<const float*>(fa_b.ptr),
        n_samples,
        n_positions,
        sequencing_error_rate,
        min_emission_prob,
        static_cast<float*>(log_b.ptr),
        n_threads
    );
    if (rc != 0) {
        throw std::runtime_error("stitchcont_k8_counts_to_logu returned nonzero status " + std::to_string(rc));
    }
    return logu;
}

py::dict run_k8_counts_reduce(
    py::array_t<float, py::array::c_style | py::array::forcecast> ref_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> alt_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> other_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> switch_prob,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    float sequencing_error_rate,
    float min_emission_prob,
    int n_threads,
    int checkpoint_interval
) {
    py::buffer_info ref_b = ref_obs.request();
    py::buffer_info alt_b = alt_obs.request();
    py::buffer_info oth_b = other_obs.request();
    py::buffer_info sw_b = switch_prob.request();
    py::buffer_info fa_b = founder_alt.request();
    require_2d(ref_b, "ref_obs");
    require_2d(alt_b, "alt_obs");
    require_2d(oth_b, "other_obs");
    require_2d(sw_b, "switch_prob");
    const int n_samples = static_cast<int>(ref_b.shape[0]);
    const int n_positions = static_cast<int>(ref_b.shape[1]);
    if (alt_b.shape[0] != n_samples || alt_b.shape[1] != n_positions || oth_b.shape[0] != n_samples || oth_b.shape[1] != n_positions || sw_b.shape[0] != n_samples || sw_b.shape[1] != n_positions) {
        throw std::runtime_error("ref_obs, alt_obs, other_obs and switch_prob must have the same shape");
    }
    require_founder_alt(fa_b, n_positions);
    py::array_t<float> dosage({n_samples, n_positions});
    py::array_t<float> gp({n_samples, n_positions, 3});
    py::buffer_info d_b = dosage.request();
    py::buffer_info gp_b = gp.request();
    int rc = 0;
    if (checkpoint_interval > 0) {
        rc = stitchcont_k8_counts_reduce_checkpointed(
            static_cast<const float*>(ref_b.ptr),
            static_cast<const float*>(alt_b.ptr),
            static_cast<const float*>(oth_b.ptr),
            static_cast<const float*>(sw_b.ptr),
            static_cast<const float*>(fa_b.ptr),
            n_samples,
            n_positions,
            checkpoint_interval,
            sequencing_error_rate,
            min_emission_prob,
            static_cast<float*>(d_b.ptr),
            static_cast<float*>(gp_b.ptr),
            n_threads
        );
    } else {
        rc = stitchcont_k8_counts_reduce(
            static_cast<const float*>(ref_b.ptr),
            static_cast<const float*>(alt_b.ptr),
            static_cast<const float*>(oth_b.ptr),
            static_cast<const float*>(sw_b.ptr),
            static_cast<const float*>(fa_b.ptr),
            n_samples,
            n_positions,
            sequencing_error_rate,
            min_emission_prob,
            static_cast<float*>(d_b.ptr),
            static_cast<float*>(gp_b.ptr),
            n_threads
        );
    }
    if (rc != 0) {
        throw std::runtime_error("stitchcont_k8_counts_reduce returned nonzero status " + std::to_string(rc));
    }
    py::dict out;
    out["dosage"] = dosage;
    out["genotype_posterior"] = gp;
    out["backend"] = py::str("pybind11_cpp_openmp_k8_counts_fused");
    return out;
}

py::dict apply_fragments_unordered_inplace(
    py::array_t<float, py::array::c_style | py::array::forcecast> log_emission_u,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_sample_offsets,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_center_idx,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_obs_offsets,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_obs_pos_idx,
    py::array_t<signed char, py::array::c_style | py::array::forcecast> fragment_obs_code,
    py::array_t<float, py::array::c_style | py::array::forcecast> fragment_obs_qual,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    float sequencing_error_rate,
    float min_emission_prob,
    bool mode_replace,
    bool rescale,
    float max_emission_matrix_difference,
    int n_threads
) {
    py::buffer_info log_b = log_emission_u.request();
    py::buffer_info fso_b = fragment_sample_offsets.request();
    py::buffer_info fci_b = fragment_center_idx.request();
    py::buffer_info foo_b = fragment_obs_offsets.request();
    py::buffer_info fop_b = fragment_obs_pos_idx.request();
    py::buffer_info foc_b = fragment_obs_code.request();
    py::buffer_info foq_b = fragment_obs_qual.request();
    py::buffer_info fa_b = founder_alt.request();
    require_3d(log_b, "log_emission_u");
    require_1d(fso_b, "fragment_sample_offsets");
    require_1d(fci_b, "fragment_center_idx");
    require_1d(foo_b, "fragment_obs_offsets");
    require_1d(fop_b, "fragment_obs_pos_idx");
    require_1d(foc_b, "fragment_obs_code");
    require_1d(foq_b, "fragment_obs_qual");
    const int n_samples = static_cast<int>(log_b.shape[0]);
    const int n_positions = static_cast<int>(log_b.shape[1]);
    const int n_states = static_cast<int>(log_b.shape[2]);
    if (n_states != 36) {
        throw std::runtime_error("K8 unordered fragment backend requires log_emission_u.shape[2] == 36");
    }
    if (fso_b.shape[0] != n_samples + 1) {
        throw std::runtime_error("fragment_sample_offsets must have length n_samples + 1");
    }
    // 64-bit: at low coverage over a full cohort the total observation count
    // can exceed 2.1e9; narrowing to int would wrap the value and bypass the
    // obs_stop > n_observations bounds guard in the kernel.
    const int64_t n_fragments = static_cast<int64_t>(fci_b.shape[0]);
    const int64_t n_observations = static_cast<int64_t>(fop_b.shape[0]);
    if (foo_b.shape[0] != n_fragments + 1 || foc_b.shape[0] != n_observations || foq_b.shape[0] != n_observations) {
        throw std::runtime_error("fragment arrays have inconsistent lengths");
    }
    require_founder_alt(fa_b, n_positions);
    int rc = stitchcont_k8_apply_fragments_unordered(
        static_cast<float*>(log_b.ptr),
        reinterpret_cast<const int64_t*>(fso_b.ptr),
        reinterpret_cast<const int64_t*>(fci_b.ptr),
        reinterpret_cast<const int64_t*>(foo_b.ptr),
        reinterpret_cast<const int64_t*>(fop_b.ptr),
        reinterpret_cast<const int8_t*>(foc_b.ptr),
        static_cast<const float*>(foq_b.ptr),
        static_cast<const float*>(fa_b.ptr),
        n_samples,
        n_positions,
        n_fragments,
        n_observations,
        sequencing_error_rate,
        min_emission_prob,
        mode_replace ? 1 : 0,
        rescale ? 1 : 0,
        max_emission_matrix_difference,
        n_threads
    );
    if (rc != 0) {
        throw std::runtime_error("stitchcont_k8_apply_fragments_unordered returned nonzero status " + std::to_string(rc));
    }
    py::dict out;
    out["log_emission_u"] = log_emission_u;
    out["backend"] = py::str("pybind11_cpp_openmp_k8_fragments_unordered");
    return out;
}

py::dict run_diploid_low_rank_reduce(
    py::array_t<float, py::array::c_style | py::array::forcecast> log_emission,
    py::array_t<float, py::array::c_style | py::array::forcecast> switch_prob,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    py::array_t<float, py::array::c_style | py::array::forcecast> left_factor,
    py::array_t<float, py::array::c_style | py::array::forcecast> right_factor,
    int n_threads
) {
    py::buffer_info log_b = log_emission.request();
    py::buffer_info sw_b = switch_prob.request();
    py::buffer_info fa_b = founder_alt.request();
    py::buffer_info left_b = left_factor.request();
    py::buffer_info right_b = right_factor.request();
    require_4d(log_b, "log_emission");
    require_2d(sw_b, "switch_prob");
    require_2d(fa_b, "founder_alt");
    require_2d(left_b, "left_factor");
    require_2d(right_b, "right_factor");
    const int n_samples = static_cast<int>(log_b.shape[0]);
    const int n_positions = static_cast<int>(log_b.shape[1]);
    const int k = static_cast<int>(log_b.shape[2]);
    if (log_b.shape[3] != k) {
        throw std::runtime_error("log_emission must have shape [n_samples, n_positions, K, K]");
    }
    if (sw_b.shape[0] != n_samples || sw_b.shape[1] != n_positions) {
        throw std::runtime_error("switch_prob must have shape [n_samples, n_positions]");
    }
    if (fa_b.shape[0] != k || fa_b.shape[1] != n_positions) {
        throw std::runtime_error("founder_alt must have shape [K, n_positions]");
    }
    const int rank = static_cast<int>(left_b.shape[1]);
    if (left_b.shape[0] != k || right_b.shape[0] != k || right_b.shape[1] != rank) {
        throw std::runtime_error("left_factor and right_factor must both have shape [K, rank]");
    }
    py::array_t<float> dosage({n_samples, n_positions});
    py::array_t<float> gp({n_samples, n_positions, 3});
    py::buffer_info d_b = dosage.request();
    py::buffer_info gp_b = gp.request();
    int rc = stitchcont_diploid_low_rank_reduce(
        static_cast<const float*>(log_b.ptr),
        static_cast<const float*>(sw_b.ptr),
        static_cast<const float*>(fa_b.ptr),
        static_cast<const float*>(left_b.ptr),
        static_cast<const float*>(right_b.ptr),
        n_samples,
        n_positions,
        k,
        rank,
        static_cast<float*>(d_b.ptr),
        static_cast<float*>(gp_b.ptr),
        n_threads
    );
    if (rc != 0) {
        throw std::runtime_error("stitchcont_diploid_low_rank_reduce returned nonzero status " + std::to_string(rc));
    }
    py::dict out;
    out["dosage"] = dosage;
    out["genotype_posterior"] = gp;
    out["backend"] = py::str("pybind11_cpp_openmp_diploid_low_rank");
    return out;
}


py::dict run_diploid_low_rank_counts_reduce(
    py::array_t<float, py::array::c_style | py::array::forcecast> ref_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> alt_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> other_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> switch_prob,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    py::array_t<float, py::array::c_style | py::array::forcecast> left_factor,
    py::array_t<float, py::array::c_style | py::array::forcecast> right_factor,
    float sequencing_error_rate,
    float min_emission_prob,
    int n_threads
) {
    py::buffer_info ref_b = ref_obs.request();
    py::buffer_info alt_b = alt_obs.request();
    py::buffer_info oth_b = other_obs.request();
    py::buffer_info sw_b = switch_prob.request();
    py::buffer_info fa_b = founder_alt.request();
    py::buffer_info left_b = left_factor.request();
    py::buffer_info right_b = right_factor.request();
    require_2d(ref_b, "ref_obs");
    require_2d(alt_b, "alt_obs");
    require_2d(oth_b, "other_obs");
    require_2d(sw_b, "switch_prob");
    require_2d(fa_b, "founder_alt");
    require_2d(left_b, "left_factor");
    require_2d(right_b, "right_factor");
    const int n_samples = static_cast<int>(ref_b.shape[0]);
    const int n_positions = static_cast<int>(ref_b.shape[1]);
    const int k = static_cast<int>(fa_b.shape[0]);
    const int rank = static_cast<int>(left_b.shape[1]);
    if (alt_b.shape[0] != n_samples || alt_b.shape[1] != n_positions || oth_b.shape[0] != n_samples || oth_b.shape[1] != n_positions || sw_b.shape[0] != n_samples || sw_b.shape[1] != n_positions) {
        throw std::runtime_error("ref_obs, alt_obs, other_obs and switch_prob must have shape [n_samples, n_positions]");
    }
    if (fa_b.shape[1] != n_positions || left_b.shape[0] != k || right_b.shape[0] != k || right_b.shape[1] != rank) {
        throw std::runtime_error("founder_alt must be [K, n_positions] and factors must be [K, rank]");
    }
    py::array_t<float> dosage({n_samples, n_positions});
    py::array_t<float> gp({n_samples, n_positions, 3});
    py::buffer_info d_b = dosage.request();
    py::buffer_info gp_b = gp.request();
    int rc = stitchcont_diploid_low_rank_counts_reduce(
        static_cast<const float*>(ref_b.ptr),
        static_cast<const float*>(alt_b.ptr),
        static_cast<const float*>(oth_b.ptr),
        static_cast<const float*>(sw_b.ptr),
        static_cast<const float*>(fa_b.ptr),
        static_cast<const float*>(left_b.ptr),
        static_cast<const float*>(right_b.ptr),
        n_samples, n_positions, k, rank,
        sequencing_error_rate, min_emission_prob,
        static_cast<float*>(d_b.ptr), static_cast<float*>(gp_b.ptr), n_threads
    );
    if (rc != 0) throw std::runtime_error("stitchcont_diploid_low_rank_counts_reduce returned nonzero status " + std::to_string(rc));
    py::dict out;
    out["dosage"] = dosage;
    out["genotype_posterior"] = gp;
    out["backend"] = py::str("pybind11_cpp_openmp_diploid_low_rank_counts_fused");
    return out;
}

py::dict run_diploid_sparse_topk_reduce(
    py::array_t<float, py::array::c_style | py::array::forcecast> log_emission,
    py::array_t<float, py::array::c_style | py::array::forcecast> switch_prob,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    py::array_t<float, py::array::c_style | py::array::forcecast> offdiag,
    int top_k,
    int n_threads
) {
    py::buffer_info log_b = log_emission.request();
    py::buffer_info sw_b = switch_prob.request();
    py::buffer_info fa_b = founder_alt.request();
    py::buffer_info off_b = offdiag.request();
    require_4d(log_b, "log_emission");
    require_2d(sw_b, "switch_prob");
    require_2d(fa_b, "founder_alt");
    require_2d(off_b, "offdiag");
    const int n_samples = static_cast<int>(log_b.shape[0]);
    const int n_positions = static_cast<int>(log_b.shape[1]);
    const int k = static_cast<int>(log_b.shape[2]);
    if (log_b.shape[3] != k || sw_b.shape[0] != n_samples || sw_b.shape[1] != n_positions || fa_b.shape[0] != k || fa_b.shape[1] != n_positions || off_b.shape[0] != k || off_b.shape[1] != k) {
        throw std::runtime_error("incompatible sparse_topk input shapes");
    }
    py::array_t<float> dosage({n_samples, n_positions});
    py::array_t<float> gp({n_samples, n_positions, 3});
    py::buffer_info d_b = dosage.request();
    py::buffer_info gp_b = gp.request();
    int rc = stitchcont_diploid_sparse_topk_reduce(
        static_cast<const float*>(log_b.ptr), static_cast<const float*>(sw_b.ptr), static_cast<const float*>(fa_b.ptr), static_cast<const float*>(off_b.ptr),
        n_samples, n_positions, k, top_k, static_cast<float*>(d_b.ptr), static_cast<float*>(gp_b.ptr), n_threads
    );
    if (rc != 0) throw std::runtime_error("stitchcont_diploid_sparse_topk_reduce returned nonzero status " + std::to_string(rc));
    py::dict out;
    out["dosage"] = dosage;
    out["genotype_posterior"] = gp;
    out["backend"] = py::str("pybind11_cpp_openmp_diploid_sparse_topk");
    return out;
}

py::dict run_diploid_sparse_topk_counts_reduce(
    py::array_t<float, py::array::c_style | py::array::forcecast> ref_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> alt_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> other_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> switch_prob,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    py::array_t<float, py::array::c_style | py::array::forcecast> offdiag,
    int top_k,
    float sequencing_error_rate,
    float min_emission_prob,
    int n_threads
) {
    py::buffer_info ref_b = ref_obs.request();
    py::buffer_info alt_b = alt_obs.request();
    py::buffer_info oth_b = other_obs.request();
    py::buffer_info sw_b = switch_prob.request();
    py::buffer_info fa_b = founder_alt.request();
    py::buffer_info off_b = offdiag.request();
    require_2d(ref_b, "ref_obs"); require_2d(alt_b, "alt_obs"); require_2d(oth_b, "other_obs"); require_2d(sw_b, "switch_prob"); require_2d(fa_b, "founder_alt"); require_2d(off_b, "offdiag");
    const int n_samples = static_cast<int>(ref_b.shape[0]);
    const int n_positions = static_cast<int>(ref_b.shape[1]);
    const int k = static_cast<int>(fa_b.shape[0]);
    if (alt_b.shape[0] != n_samples || alt_b.shape[1] != n_positions || oth_b.shape[0] != n_samples || oth_b.shape[1] != n_positions || sw_b.shape[0] != n_samples || sw_b.shape[1] != n_positions || fa_b.shape[1] != n_positions || off_b.shape[0] != k || off_b.shape[1] != k) {
        throw std::runtime_error("incompatible sparse_topk count input shapes");
    }
    py::array_t<float> dosage({n_samples, n_positions});
    py::array_t<float> gp({n_samples, n_positions, 3});
    py::buffer_info d_b = dosage.request();
    py::buffer_info gp_b = gp.request();
    int rc = stitchcont_diploid_sparse_topk_counts_reduce(
        static_cast<const float*>(ref_b.ptr), static_cast<const float*>(alt_b.ptr), static_cast<const float*>(oth_b.ptr), static_cast<const float*>(sw_b.ptr), static_cast<const float*>(fa_b.ptr), static_cast<const float*>(off_b.ptr),
        n_samples, n_positions, k, top_k, sequencing_error_rate, min_emission_prob, static_cast<float*>(d_b.ptr), static_cast<float*>(gp_b.ptr), n_threads
    );
    if (rc != 0) throw std::runtime_error("stitchcont_diploid_sparse_topk_counts_reduce returned nonzero status " + std::to_string(rc));
    py::dict out;
    out["dosage"] = dosage;
    out["genotype_posterior"] = gp;
    out["backend"] = py::str("pybind11_cpp_openmp_diploid_sparse_topk_counts_fused");
    return out;
}

py::dict run_k8_counts_fragments_reduce(
    py::array_t<float, py::array::c_style | py::array::forcecast> ref_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> alt_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> other_obs,
    py::array_t<float, py::array::c_style | py::array::forcecast> switch_prob,
    py::array_t<float, py::array::c_style | py::array::forcecast> founder_alt,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_sample_offsets,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_center_idx,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_obs_offsets,
    py::array_t<long long, py::array::c_style | py::array::forcecast> fragment_obs_pos_idx,
    py::array_t<signed char, py::array::c_style | py::array::forcecast> fragment_obs_code,
    py::array_t<float, py::array::c_style | py::array::forcecast> fragment_obs_qual,
    float sequencing_error_rate,
    float min_emission_prob,
    bool mode_replace,
    bool rescale,
    float max_emission_matrix_difference,
    int n_threads
) {
    py::buffer_info ref_b = ref_obs.request(); py::buffer_info alt_b = alt_obs.request(); py::buffer_info oth_b = other_obs.request(); py::buffer_info sw_b = switch_prob.request(); py::buffer_info fa_b = founder_alt.request();
    py::buffer_info fso_b = fragment_sample_offsets.request(); py::buffer_info fci_b = fragment_center_idx.request(); py::buffer_info foo_b = fragment_obs_offsets.request(); py::buffer_info fop_b = fragment_obs_pos_idx.request(); py::buffer_info foc_b = fragment_obs_code.request(); py::buffer_info foq_b = fragment_obs_qual.request();
    require_2d(ref_b, "ref_obs"); require_2d(alt_b, "alt_obs"); require_2d(oth_b, "other_obs"); require_2d(sw_b, "switch_prob");
    const int n_samples = static_cast<int>(ref_b.shape[0]);
    const int n_positions = static_cast<int>(ref_b.shape[1]);
    require_founder_alt(fa_b, n_positions);
    if (alt_b.shape[0] != n_samples || alt_b.shape[1] != n_positions || oth_b.shape[0] != n_samples || oth_b.shape[1] != n_positions || sw_b.shape[0] != n_samples || sw_b.shape[1] != n_positions || fso_b.shape[0] != n_samples + 1) {
        throw std::runtime_error("incompatible K8 count/fragment input shapes");
    }
    // 64-bit: at low coverage over a full cohort the total observation count
    // can exceed 2.1e9; narrowing to int would wrap the value and bypass the
    // obs_stop > n_observations bounds guard in the kernel.
    const int64_t n_fragments = static_cast<int64_t>(fci_b.shape[0]);
    const int64_t n_observations = static_cast<int64_t>(fop_b.shape[0]);
    if (foo_b.shape[0] != n_fragments + 1 || foc_b.shape[0] != n_observations || foq_b.shape[0] != n_observations) {
        throw std::runtime_error("fragment arrays have inconsistent lengths");
    }
    py::array_t<float> dosage({n_samples, n_positions});
    py::array_t<float> gp({n_samples, n_positions, 3});
    py::buffer_info d_b = dosage.request(); py::buffer_info gp_b = gp.request();
    int rc = stitchcont_k8_counts_fragments_reduce(
        static_cast<const float*>(ref_b.ptr), static_cast<const float*>(alt_b.ptr), static_cast<const float*>(oth_b.ptr), static_cast<const float*>(sw_b.ptr), static_cast<const float*>(fa_b.ptr),
        reinterpret_cast<const int64_t*>(fso_b.ptr), reinterpret_cast<const int64_t*>(fci_b.ptr), reinterpret_cast<const int64_t*>(foo_b.ptr), reinterpret_cast<const int64_t*>(fop_b.ptr), reinterpret_cast<const int8_t*>(foc_b.ptr), static_cast<const float*>(foq_b.ptr),
        n_samples, n_positions, n_fragments, n_observations, sequencing_error_rate, min_emission_prob, mode_replace ? 1 : 0, rescale ? 1 : 0, max_emission_matrix_difference, static_cast<float*>(d_b.ptr), static_cast<float*>(gp_b.ptr), n_threads
    );
    if (rc != 0) throw std::runtime_error("stitchcont_k8_counts_fragments_reduce returned nonzero status " + std::to_string(rc));
    py::dict out;
    out["dosage"] = dosage;
    out["genotype_posterior"] = gp;
    out["backend"] = py::str("pybind11_cpp_openmp_k8_counts_fragments_fused");
    return out;
}

PYBIND11_MODULE(_hs_k8_unordered, m) {
    m.doc() = "pybind11 wrapper for stitchcont native K8 unordered diploid CPU kernels";
    m.def(
        "run_k8_unordered_reduce",
        &run_k8_unordered_reduce,
        py::arg("log_emission_u"),
        py::arg("switch_prob"),
        py::arg("founder_alt"),
        py::arg("n_threads") = 0,
        py::arg("checkpoint_interval") = 0,
        "Run the K8 unordered diploid HMM kernel from precomputed unordered log emissions."
    );
    m.def(
        "build_k8_logu_from_counts",
        &build_k8_logu_from_counts,
        py::arg("ref_obs"),
        py::arg("alt_obs"),
        py::arg("other_obs"),
        py::arg("founder_alt"),
        py::arg("sequencing_error_rate"),
        py::arg("min_emission_prob"),
        py::arg("n_threads") = 0,
        "Build unordered K8 log-emission matrix directly from dense count/quality evidence."
    );
    m.def(
        "run_k8_counts_reduce",
        &run_k8_counts_reduce,
        py::arg("ref_obs"),
        py::arg("alt_obs"),
        py::arg("other_obs"),
        py::arg("switch_prob"),
        py::arg("founder_alt"),
        py::arg("sequencing_error_rate"),
        py::arg("min_emission_prob"),
        py::arg("n_threads") = 0,
        py::arg("checkpoint_interval") = 0,
        "Fused count-emission plus K8 unordered diploid HMM reduction."
    );
    m.def(
        "run_diploid_low_rank_reduce",
        &run_diploid_low_rank_reduce,
        py::arg("log_emission"),
        py::arg("switch_prob"),
        py::arg("founder_alt"),
        py::arg("left_factor"),
        py::arg("right_factor"),
        py::arg("n_threads") = 0,
        "Run native ordered diploid low-rank algebraic transition HMM and reduce directly to dosage/GP."
    );
    m.def(
        "apply_fragments_unordered_inplace",
        &apply_fragments_unordered_inplace,
        py::arg("log_emission_u"),
        py::arg("fragment_sample_offsets"),
        py::arg("fragment_center_idx"),
        py::arg("fragment_obs_offsets"),
        py::arg("fragment_obs_pos_idx"),
        py::arg("fragment_obs_code"),
        py::arg("fragment_obs_qual"),
        py::arg("founder_alt"),
        py::arg("sequencing_error_rate"),
        py::arg("min_emission_prob"),
        py::arg("mode_replace"),
        py::arg("rescale"),
        py::arg("max_emission_matrix_difference"),
        py::arg("n_threads") = 0,
        "Apply STITCH-style fragment likelihoods directly to unordered K8 log emissions."
    );
    m.def(
        "run_diploid_low_rank_counts_reduce",
        &run_diploid_low_rank_counts_reduce,
        py::arg("ref_obs"), py::arg("alt_obs"), py::arg("other_obs"), py::arg("switch_prob"), py::arg("founder_alt"), py::arg("left_factor"), py::arg("right_factor"),
        py::arg("sequencing_error_rate"), py::arg("min_emission_prob"), py::arg("n_threads") = 0,
        "Fused count-emission plus native ordered diploid low-rank HMM reduction."
    );
    m.def(
        "run_diploid_sparse_topk_reduce",
        &run_diploid_sparse_topk_reduce,
        py::arg("log_emission"), py::arg("switch_prob"), py::arg("founder_alt"), py::arg("offdiag"), py::arg("top_k"), py::arg("n_threads") = 0,
        "Run native ordered diploid sparse top-k transition HMM from precomputed log emissions."
    );
    m.def(
        "run_diploid_sparse_topk_counts_reduce",
        &run_diploid_sparse_topk_counts_reduce,
        py::arg("ref_obs"), py::arg("alt_obs"), py::arg("other_obs"), py::arg("switch_prob"), py::arg("founder_alt"), py::arg("offdiag"), py::arg("top_k"), py::arg("sequencing_error_rate"), py::arg("min_emission_prob"), py::arg("n_threads") = 0,
        "Fused count-emission plus native ordered diploid sparse top-k transition HMM."
    );
    m.def(
        "run_k8_counts_fragments_reduce",
        &run_k8_counts_fragments_reduce,
        py::arg("ref_obs"), py::arg("alt_obs"), py::arg("other_obs"), py::arg("switch_prob"), py::arg("founder_alt"),
        py::arg("fragment_sample_offsets"), py::arg("fragment_center_idx"), py::arg("fragment_obs_offsets"), py::arg("fragment_obs_pos_idx"), py::arg("fragment_obs_code"), py::arg("fragment_obs_qual"),
        py::arg("sequencing_error_rate"), py::arg("min_emission_prob"), py::arg("mode_replace"), py::arg("rescale"), py::arg("max_emission_matrix_difference"), py::arg("n_threads") = 0,
        "Fused K8 count evidence + fragment likelihood + unordered HMM reduction."
    );

}
