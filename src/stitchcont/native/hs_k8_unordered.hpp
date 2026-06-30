#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

int stitchcont_k8_unordered_reduce(
    const float* logu,
    const float* sw,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    float* dosage,
    float* gp,
    int n_threads
);

int stitchcont_k8_unordered_reduce_checkpointed(
    const float* logu,
    const float* sw,
    const float* founder_alt,
    int n_samples,
    int n_positions,
    int checkpoint_interval,
    float* dosage,
    float* gp,
    int n_threads
);

int stitchcont_k8_counts_reduce(
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
);

int stitchcont_k8_counts_reduce_checkpointed(
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
);

int stitchcont_k8_counts_to_logu(
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
);

int stitchcont_k8_apply_fragments_unordered(
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
);

int stitchcont_diploid_low_rank_reduce(
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
);

int stitchcont_diploid_low_rank_counts_reduce(
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
);

int stitchcont_diploid_sparse_topk_reduce(
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
);

int stitchcont_diploid_sparse_topk_counts_reduce(
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
);

int stitchcont_k8_counts_fragments_reduce(
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
);

#ifdef __cplusplus
}
#endif
