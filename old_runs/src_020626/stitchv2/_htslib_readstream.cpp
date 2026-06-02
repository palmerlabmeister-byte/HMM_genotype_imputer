#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <htslib/hts.h>
#include <htslib/faidx.h>
#include <htslib/sam.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct PendingFragment {
    std::vector<int32_t> pos;
    std::vector<int8_t> code;
    std::vector<uint8_t> qual;
};

struct ExtractedSample {
    int64_t n_reads = 0;
    std::vector<int32_t> fragment_centers;
    std::vector<int64_t> fragment_obs_offsets;
    std::vector<int32_t> fragment_obs_pos;
    std::vector<int8_t> fragment_obs_code;
    std::vector<uint8_t> fragment_obs_qual;
    std::string error_message;
};

struct MatchInterval {
    int64_t start_1based = 0;
    int64_t end_1based = 0;
    int32_t qpos_start = 0;
};

static constexpr uint8_t VARIANT_SNP = 0;
static constexpr uint8_t VARIANT_INSERTION = 1;
static constexpr uint8_t VARIANT_DELETION = 2;

static inline bool cigar_consumes_reference(int op) {
    return op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF || op == BAM_CDEL || op == BAM_CREF_SKIP;
}

static inline int64_t reference_span_from_cigar(const bam1_t* rec) {
    int64_t span = 0;
    const uint32_t* cigar = bam_get_cigar(rec);
    for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
        const int op = bam_cigar_op(cigar[i]);
        if (cigar_consumes_reference(op)) {
            span += static_cast<int64_t>(bam_cigar_oplen(cigar[i]));
        }
    }
    return span;
}

static inline uint8_t seq_base_ascii_upper(const uint8_t* seq, int qpos) {
    static constexpr char nt16_to_char[] = "=ACMGRSVTWYHKDBN";
    char base = nt16_to_char[bam_seqi(seq, qpos)];
    if (base >= 'a' && base <= 'z') {
        base = static_cast<char>(base - ('a' - 'A'));
    }
    return static_cast<uint8_t>(base);
}

static std::string aux_string_tag(const bam1_t* rec, const char* tag) {
    if (tag == nullptr || std::strlen(tag) < 2) {
        return std::string();
    }
    char tag2[3] = {tag[0], tag[1], '\0'};
    uint8_t* aux = bam_aux_get(rec, tag2);
    if (aux == nullptr) {
        return std::string();
    }
    const char* value = bam_aux2Z(aux);
    if (value == nullptr || value[0] == '\0') {
        return std::string();
    }
    return std::string(value);
}

static void append_pending_observation(
    PendingFragment& frag,
    const std::vector<int32_t>& read_pos,
    const std::vector<int8_t>& read_code,
    const std::vector<uint8_t>& read_qual
) {
    frag.pos.insert(frag.pos.end(), read_pos.begin(), read_pos.end());
    frag.code.insert(frag.code.end(), read_code.begin(), read_code.end());
    frag.qual.insert(frag.qual.end(), read_qual.begin(), read_qual.end());
}

static inline int base_index_from_ascii(uint8_t base) {
    if (base >= 'a' && base <= 'z') {
        base = static_cast<uint8_t>(base - ('a' - 'A'));
    }
    switch (base) {
        case 'A':
            return 0;
        case 'C':
            return 1;
        case 'G':
            return 2;
        case 'T':
            return 3;
        default:
            return -1;
    }
}

static inline uint8_t ascii_from_base_index(int idx) {
    static constexpr uint8_t bases[] = {'A', 'C', 'G', 'T'};
    if (idx < 0 || idx > 3) {
        return static_cast<uint8_t>('N');
    }
    return bases[idx];
}

static inline uint8_t capped_base_quality(
    const uint8_t* qual,
    int32_t qpos,
    uint8_t mapq,
    bool cap_base_quality_by_mapping_quality
) {
    uint8_t q = qual[qpos];
    if (q == 255U) {
        q = 30U;
    }
    if (cap_base_quality_by_mapping_quality) {
        q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(mapq)));
    }
    return q;
}

static inline uint8_t min_quality(uint8_t a, uint8_t b) {
    return static_cast<uint8_t>(std::min<int>(static_cast<int>(a), static_cast<int>(b)));
}

static inline float phred_weight(uint8_t q) {
    const float error_rate = std::pow(10.0f, -(static_cast<float>(q) / 10.0f));
    return 1.0f - error_rate;
}

static inline void saturating_increment_uint16(uint16_t& value) {
    if (value < std::numeric_limits<uint16_t>::max()) {
        ++value;
    }
}

static int32_t qpos_for_ref_pos(const std::vector<MatchInterval>& intervals, int64_t pos_1based) {
    for (const MatchInterval& interval : intervals) {
        if (pos_1based >= interval.start_1based && pos_1based <= interval.end_1based) {
            return interval.qpos_start + static_cast<int32_t>(pos_1based - interval.start_1based);
        }
    }
    return -1;
}

static bool min_quality_for_ref_span(
    const std::vector<MatchInterval>& intervals,
    const uint8_t* qual,
    uint8_t mapq,
    bool cap_base_quality_by_mapping_quality,
    int64_t start_1based,
    int64_t end_1based,
    uint8_t& out_min_quality
) {
    if (end_1based < start_1based) {
        return false;
    }
    bool seen = false;
    uint8_t best = 255U;
    for (int64_t pos = start_1based; pos <= end_1based; ++pos) {
        const int32_t qpos = qpos_for_ref_pos(intervals, pos);
        if (qpos < 0) {
            return false;
        }
        best = seen ? min_quality(best, capped_base_quality(qual, qpos, mapq, cap_base_quality_by_mapping_quality))
                    : capped_base_quality(qual, qpos, mapq, cap_base_quality_by_mapping_quality);
        seen = true;
    }
    if (!seen) {
        return false;
    }
    out_min_quality = best;
    return true;
}

static std::string bam_inserted_sequence(const uint8_t* seq, int32_t qpos, int32_t len) {
    std::string out;
    out.reserve(static_cast<size_t>(std::max(len, 0)));
    for (int32_t i = 0; i < len; ++i) {
        out.push_back(static_cast<char>(seq_base_ascii_upper(seq, qpos + i)));
    }
    return out;
}

static uint8_t min_insert_quality(
    const uint8_t* qual,
    uint8_t mapq,
    bool cap_base_quality_by_mapping_quality,
    int32_t qpos,
    int32_t len,
    uint8_t initial_quality
) {
    uint8_t best = initial_quality;
    for (int32_t i = 0; i < len; ++i) {
        best = min_quality(best, capped_base_quality(qual, qpos + i, mapq, cap_base_quality_by_mapping_quality));
    }
    return best;
}

static bool py_sequence_to_strings(PyObject* obj, npy_intp expected_len, const char* name, std::vector<std::string>& out) {
    PyObject* seq = PySequence_Fast(obj, name);
    if (seq == nullptr) {
        return false;
    }
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    if (n != static_cast<Py_ssize_t>(expected_len)) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must have length %zd.", name, static_cast<Py_ssize_t>(expected_len));
        return false;
    }
    out.clear();
    out.reserve(static_cast<size_t>(std::max<Py_ssize_t>(n, 0)));
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(seq, i);
        PyObject* item_str = PyObject_Str(item);
        if (item_str == nullptr) {
            Py_DECREF(seq);
            return false;
        }
        const char* text = PyUnicode_AsUTF8(item_str);
        if (text == nullptr) {
            Py_DECREF(item_str);
            Py_DECREF(seq);
            return false;
        }
        out.emplace_back(text);
        Py_DECREF(item_str);
    }
    Py_DECREF(seq);
    return true;
}

static void append_fragment_compacted(
    const std::vector<int32_t>& in_pos,
    const std::vector<int8_t>& in_code,
    const std::vector<uint8_t>& in_qual,
    const int64_t* physical_positions,
    std::vector<int32_t>& out_centers,
    std::vector<int64_t>& out_obs_offsets,
    std::vector<int32_t>& out_obs_pos,
    std::vector<int8_t>& out_obs_code,
    std::vector<uint8_t>& out_obs_qual
) {
    if (in_pos.empty()) {
        return;
    }
    const size_t n = in_pos.size();
    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), static_cast<size_t>(0));
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return in_pos[a] < in_pos[b];
    });

    std::vector<int32_t> compact_pos;
    std::vector<int8_t> compact_code;
    std::vector<uint8_t> compact_qual;
    compact_pos.reserve(n);
    compact_code.reserve(n);
    compact_qual.reserve(n);

    size_t i = 0;
    while (i < n) {
        size_t j = i + 1;
        size_t best = order[i];
        const int32_t p = in_pos[best];
        while (j < n && in_pos[order[j]] == p) {
            const size_t cand = order[j];
            if (in_qual[cand] > in_qual[best]) {
                best = cand;
            }
            ++j;
        }
        compact_pos.push_back(in_pos[best]);
        compact_code.push_back(in_code[best]);
        compact_qual.push_back(in_qual[best]);
        i = j;
    }

    if (compact_pos.empty()) {
        return;
    }
    int32_t center = compact_pos[compact_pos.size() / 2];
    if (physical_positions != nullptr) {
        double mean_pos = 0.0;
        for (int32_t p : compact_pos) {
            mean_pos += static_cast<double>(physical_positions[p]);
        }
        mean_pos /= static_cast<double>(compact_pos.size());
        double best_dist = std::numeric_limits<double>::infinity();
        for (int32_t p : compact_pos) {
            const double dist = std::abs(static_cast<double>(physical_positions[p]) - mean_pos);
            if (dist < best_dist) {
                best_dist = dist;
                center = p;
            }
        }
    }
    out_centers.push_back(center);
    out_obs_pos.insert(out_obs_pos.end(), compact_pos.begin(), compact_pos.end());
    out_obs_code.insert(out_obs_code.end(), compact_code.begin(), compact_code.end());
    out_obs_qual.insert(out_obs_qual.end(), compact_qual.begin(), compact_qual.end());
    out_obs_offsets.push_back(static_cast<int64_t>(out_obs_pos.size()));
}

static bool extract_one_sample_to_rows(
    const char* bam_path,
    const char* chromosome,
    long region_start,
    long region_stop,
    const int32_t* lookup,
    npy_intp lookup_len,
    const uint8_t* ref_codes,
    const uint8_t* alt_codes,
    const int64_t* physical_positions,
    const uint8_t* variant_types,
    const std::vector<std::string>* ref_alleles,
    const std::vector<std::string>* alt_alleles,
    npy_intp n_positions,
    int min_base_quality,
    int min_mapping_quality,
    bool do_merge,
    int hts_threads,
    bool snp_only_bamreader,
    bool stitch_style_bamreader,
    bool variant_aware_bamreader,
    int max_indel_len,
    int max_insert_size,
    bool cap_base_quality_by_mapping_quality,
    bool ref_alt_only,
    bool merge_unpaired_by_query,
    bool use_bx_tag,
    const char* bx_tag,
    int bx_tag_upper_limit,
    uint16_t* depth_out,
    uint16_t* ref_count_out,
    uint16_t* alt_count_out,
    uint16_t* other_count_out,
    float* ref_weight_out,
    float* alt_weight_out,
    float* other_weight_out,
    ExtractedSample& out
) {
    out.n_reads = 0;
    out.fragment_centers.clear();
    out.fragment_obs_offsets.clear();
    out.fragment_obs_pos.clear();
    out.fragment_obs_code.clear();
    out.fragment_obs_qual.clear();
    out.fragment_obs_offsets.push_back(0);
    out.error_message.clear();

    std::vector<uint32_t> depth(static_cast<size_t>(n_positions), 0U);
    std::vector<uint32_t> ref_count(static_cast<size_t>(n_positions), 0U);
    std::vector<uint32_t> alt_count(static_cast<size_t>(n_positions), 0U);
    std::vector<uint32_t> other_count(static_cast<size_t>(n_positions), 0U);
    std::vector<float> ref_weight(static_cast<size_t>(n_positions), 0.0f);
    std::vector<float> alt_weight(static_cast<size_t>(n_positions), 0.0f);
    std::vector<float> other_weight(static_cast<size_t>(n_positions), 0.0f);

    float qual_weight[256];
    for (int q = 0; q < 256; ++q) {
        const float error_rate = std::pow(10.0f, -(static_cast<float>(q) / 10.0f));
        qual_weight[q] = 1.0f - error_rate;
    }

    std::unordered_map<std::string, PendingFragment> pending;
    int error_code = 0;
    std::string error_message;

    samFile* fp = sam_open(bam_path, "r");
    if (fp == nullptr) {
        error_code = 1;
        error_message = std::string("Failed to open BAM/CRAM file: ") + bam_path;
    }
    bam_hdr_t* hdr = nullptr;
    hts_idx_t* idx = nullptr;
    hts_itr_t* itr = nullptr;
    bam1_t* rec = nullptr;
    const int threads = std::max(hts_threads, 1);
    if (error_code == 0) {
        hts_set_threads(fp, threads);
        hdr = sam_hdr_read(fp);
        if (hdr == nullptr) {
            error_code = 1;
            error_message = std::string("Failed to read header from: ") + bam_path;
        }
    }
    if (error_code == 0) {
        idx = sam_index_load(fp, bam_path);
        if (idx == nullptr) {
            error_code = 1;
            error_message = std::string("Missing/invalid index for: ") + bam_path;
        }
    }
    if (error_code == 0) {
        const std::string region_query =
            std::string(chromosome) + ":" + std::to_string(region_start + 1) + "-" + std::to_string(region_stop);
        itr = sam_itr_querys(idx, hdr, region_query.c_str());
        if (itr == nullptr) {
            error_code = 1;
            error_message = std::string("Failed to create iterator for region: ") + region_query;
        }
    }
    if (error_code == 0) {
        rec = bam_init1();
        if (rec == nullptr) {
            error_code = 1;
            error_message = "Failed to allocate bam record.";
        }
    }

    std::vector<int32_t> read_pos;
    std::vector<int8_t> read_code;
    std::vector<uint8_t> read_qual;
    read_pos.reserve(64);
    read_code.reserve(64);
    read_qual.reserve(64);
    npy_intp stitch_scan_start = 0;

    while (error_code == 0 && sam_itr_next(fp, itr, rec) >= 0) {
        const uint16_t flag = rec->core.flag;
        if ((flag & BAM_FUNMAP) != 0 || (flag & BAM_FSECONDARY) != 0 || (flag & BAM_FSUPPLEMENTARY) != 0 || (flag & BAM_FDUP) != 0) {
            continue;
        }
        if (rec->core.qual < min_mapping_quality) {
            continue;
        }
        if (max_insert_size > 0 && std::abs(static_cast<int>(rec->core.isize)) > max_insert_size) {
            continue;
        }
        const uint8_t* seq = bam_get_seq(rec);
        const uint8_t* qual = bam_get_qual(rec);
        if (seq == nullptr || qual == nullptr) {
            continue;
        }

        read_pos.clear();
        read_code.clear();
        read_qual.clear();

        int32_t ref_pos = rec->core.pos;
        int32_t qpos = 0;
        const uint32_t* cigar = bam_get_cigar(rec);
        npy_intp stitch_target_cursor = 0;
        npy_intp stitch_target_stop = 0;
        if (stitch_style_bamreader || variant_aware_bamreader) {
            const int64_t read_start_1 = static_cast<int64_t>(rec->core.pos) + 1;
            const int64_t ref_span = reference_span_from_cigar(rec);
            if (ref_span <= 0) {
                continue;
            }
            const int64_t read_end_1 = read_start_1 + ref_span - 1;
            while (stitch_scan_start < n_positions && physical_positions[stitch_scan_start] < read_start_1) {
                ++stitch_scan_start;
            }
            stitch_target_cursor = stitch_scan_start;
            const int64_t* begin = physical_positions + stitch_target_cursor;
            const int64_t* end = physical_positions + n_positions;
            stitch_target_stop = static_cast<npy_intp>(std::upper_bound(begin, end, read_end_1) - physical_positions);
            if (stitch_target_cursor >= stitch_target_stop) {
                continue;
            }
        }
        if (variant_aware_bamreader) {
            std::vector<MatchInterval> match_intervals;
            std::vector<int32_t> indel_alt_or_event_seen;
            match_intervals.reserve(rec->core.n_cigar);
            for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
                const int op = bam_cigar_op(cigar[i]);
                const int oplen = bam_cigar_oplen(cigar[i]);
                if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                    const int64_t block_start_1 = static_cast<int64_t>(ref_pos) + 1;
                    const int64_t block_end_1 = static_cast<int64_t>(ref_pos) + static_cast<int64_t>(oplen);
                    match_intervals.push_back(MatchInterval{block_start_1, block_end_1, qpos});
                    while (stitch_target_cursor < stitch_target_stop && physical_positions[stitch_target_cursor] < block_start_1) {
                        ++stitch_target_cursor;
                    }
                    while (stitch_target_cursor < stitch_target_stop && physical_positions[stitch_target_cursor] <= block_end_1) {
                        const int32_t target_idx = static_cast<int32_t>(stitch_target_cursor);
                        if (variant_types[target_idx] == VARIANT_SNP) {
                            const int32_t local_qpos = qpos + static_cast<int32_t>(physical_positions[stitch_target_cursor] - block_start_1);
                            const uint8_t q = capped_base_quality(
                                qual,
                                local_qpos,
                                rec->core.qual,
                                cap_base_quality_by_mapping_quality
                            );
                            if (q >= min_base_quality) {
                                const uint8_t obs_base = seq_base_ascii_upper(seq, local_qpos);
                                int8_t code = 2;
                                if (obs_base == ref_codes[target_idx]) {
                                    code = 0;
                                } else if (obs_base == alt_codes[target_idx]) {
                                    code = 1;
                                } else if (ref_alt_only) {
                                    ++stitch_target_cursor;
                                    continue;
                                }
                                read_pos.push_back(target_idx);
                                read_code.push_back(code);
                                read_qual.push_back(q);
                            }
                        }
                        ++stitch_target_cursor;
                    }
                    ref_pos += oplen;
                    qpos += oplen;
                } else if (op == BAM_CINS) {
                    if (oplen <= max_indel_len && qpos >= 1) {
                        const int64_t event_anchor_1 = static_cast<int64_t>(ref_pos);
                        const std::string inserted = bam_inserted_sequence(seq, qpos, oplen);
                        const int64_t lower_pos = event_anchor_1 - static_cast<int64_t>(std::max(max_indel_len, 1)) - 2;
                        const int64_t* begin = physical_positions;
                        const int64_t* end = physical_positions + n_positions;
                        const int64_t* it = std::lower_bound(begin, end, std::max<int64_t>(lower_pos, 1));
                        while (it != end && *it <= event_anchor_1) {
                            const int32_t target_idx = static_cast<int32_t>(it - begin);
                            if (variant_types[target_idx] == VARIANT_INSERTION) {
                                const std::string& ref_allele = (*ref_alleles)[static_cast<size_t>(target_idx)];
                                const std::string& alt_allele = (*alt_alleles)[static_cast<size_t>(target_idx)];
                                const int64_t allele_anchor = physical_positions[target_idx] + static_cast<int64_t>(ref_allele.size()) - 1;
                                if (allele_anchor == event_anchor_1) {
                                    indel_alt_or_event_seen.push_back(target_idx);
                                    if (alt_allele.size() > ref_allele.size()) {
                                        const std::string alt_insert = alt_allele.substr(ref_allele.size());
                                        if (alt_insert == inserted) {
                                            uint8_t q = capped_base_quality(
                                                qual,
                                                qpos - 1,
                                                rec->core.qual,
                                                cap_base_quality_by_mapping_quality
                                            );
                                            q = min_insert_quality(
                                                qual,
                                                rec->core.qual,
                                                cap_base_quality_by_mapping_quality,
                                                qpos,
                                                oplen,
                                                q
                                            );
                                            if (q >= min_base_quality) {
                                                read_pos.push_back(target_idx);
                                                read_code.push_back(1);
                                                read_qual.push_back(q);
                                            }
                                        } else if (!ref_alt_only) {
                                            uint8_t q = capped_base_quality(
                                                qual,
                                                qpos - 1,
                                                rec->core.qual,
                                                cap_base_quality_by_mapping_quality
                                            );
                                            q = min_insert_quality(
                                                qual,
                                                rec->core.qual,
                                                cap_base_quality_by_mapping_quality,
                                                qpos,
                                                oplen,
                                                q
                                            );
                                            if (q >= min_base_quality) {
                                                read_pos.push_back(target_idx);
                                                read_code.push_back(2);
                                                read_qual.push_back(q);
                                            }
                                        }
                                    }
                                }
                            }
                            ++it;
                        }
                    }
                    qpos += oplen;
                } else if (op == BAM_CDEL) {
                    if (oplen <= max_indel_len) {
                        const int64_t event_anchor_1 = static_cast<int64_t>(ref_pos);
                        const int64_t lower_pos = event_anchor_1 - static_cast<int64_t>(std::max(max_indel_len, 1)) - 2;
                        const int64_t* begin = physical_positions;
                        const int64_t* end = physical_positions + n_positions;
                        const int64_t* it = std::lower_bound(begin, end, std::max<int64_t>(lower_pos, 1));
                        while (it != end && *it <= event_anchor_1) {
                            const int32_t target_idx = static_cast<int32_t>(it - begin);
                            if (variant_types[target_idx] == VARIANT_DELETION) {
                                const std::string& ref_allele = (*ref_alleles)[static_cast<size_t>(target_idx)];
                                const std::string& alt_allele = (*alt_alleles)[static_cast<size_t>(target_idx)];
                                const int64_t allele_anchor = physical_positions[target_idx] + static_cast<int64_t>(alt_allele.size()) - 1;
                                const int deletion_len = static_cast<int>(ref_allele.size()) - static_cast<int>(alt_allele.size());
                                if (allele_anchor == event_anchor_1) {
                                    indel_alt_or_event_seen.push_back(target_idx);
                                    uint8_t q = 255U;
                                    bool have_q = false;
                                    if (qpos > 0) {
                                        q = capped_base_quality(qual, qpos - 1, rec->core.qual, cap_base_quality_by_mapping_quality);
                                        have_q = true;
                                    }
                                    if (qpos < rec->core.l_qseq) {
                                        const uint8_t q_next = capped_base_quality(qual, qpos, rec->core.qual, cap_base_quality_by_mapping_quality);
                                        q = have_q ? min_quality(q, q_next) : q_next;
                                        have_q = true;
                                    }
                                    if (!have_q) {
                                        q = static_cast<uint8_t>(std::min<int>(30, static_cast<int>(rec->core.qual)));
                                    }
                                    if (deletion_len == oplen) {
                                        if (q >= min_base_quality) {
                                            read_pos.push_back(target_idx);
                                            read_code.push_back(1);
                                            read_qual.push_back(q);
                                        }
                                    } else if (!ref_alt_only && q >= min_base_quality) {
                                        read_pos.push_back(target_idx);
                                        read_code.push_back(2);
                                        read_qual.push_back(q);
                                    }
                                }
                            }
                            ++it;
                        }
                    }
                    ref_pos += oplen;
                } else if (op == BAM_CREF_SKIP) {
                    ref_pos += oplen;
                } else if (op == BAM_CSOFT_CLIP) {
                    qpos += oplen;
                } else {
                    // BAM_CHARD_CLIP, BAM_CPAD, BAM_CBACK etc: consume neither or handled as no-op.
                }
            }

            const int64_t read_start_1 = static_cast<int64_t>(rec->core.pos) + 1;
            const int64_t ref_span = reference_span_from_cigar(rec);
            const int64_t read_end_1 = read_start_1 + ref_span - 1;
            if (!indel_alt_or_event_seen.empty()) {
                std::sort(indel_alt_or_event_seen.begin(), indel_alt_or_event_seen.end());
                indel_alt_or_event_seen.erase(
                    std::unique(indel_alt_or_event_seen.begin(), indel_alt_or_event_seen.end()),
                    indel_alt_or_event_seen.end()
                );
            }
            const int64_t* begin = physical_positions;
            const int64_t* end = physical_positions + n_positions;
            const int64_t* it = std::lower_bound(begin, end, read_start_1);
            while (it != end && *it <= read_end_1) {
                const int32_t target_idx = static_cast<int32_t>(it - begin);
                if (
                    variant_types[target_idx] != VARIANT_SNP &&
                    !std::binary_search(indel_alt_or_event_seen.begin(), indel_alt_or_event_seen.end(), target_idx)
                ) {
                    const std::string& ref_allele = (*ref_alleles)[static_cast<size_t>(target_idx)];
                    uint8_t q = 0U;
                    bool supported = false;
                    if (variant_types[target_idx] == VARIANT_INSERTION) {
                        const int64_t anchor = physical_positions[target_idx] + static_cast<int64_t>(ref_allele.size()) - 1;
                        supported = min_quality_for_ref_span(
                            match_intervals,
                            qual,
                            rec->core.qual,
                            cap_base_quality_by_mapping_quality,
                            anchor,
                            anchor + 1,
                            q
                        );
                    } else if (variant_types[target_idx] == VARIANT_DELETION) {
                        supported = min_quality_for_ref_span(
                            match_intervals,
                            qual,
                            rec->core.qual,
                            cap_base_quality_by_mapping_quality,
                            physical_positions[target_idx],
                            physical_positions[target_idx] + static_cast<int64_t>(ref_allele.size()) - 1,
                            q
                        );
                    }
                    if (supported && q >= min_base_quality) {
                        read_pos.push_back(target_idx);
                        read_code.push_back(0);
                        read_qual.push_back(q);
                    }
                }
                ++it;
            }
        } else {
        for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
            const int op = bam_cigar_op(cigar[i]);
            const int oplen = bam_cigar_oplen(cigar[i]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                if (stitch_style_bamreader) {
                    const int64_t block_start_1 = static_cast<int64_t>(ref_pos) + 1;
                    const int64_t block_end_1 = static_cast<int64_t>(ref_pos) + static_cast<int64_t>(oplen);
                    while (stitch_target_cursor < stitch_target_stop && physical_positions[stitch_target_cursor] < block_start_1) {
                        ++stitch_target_cursor;
                    }
                    while (stitch_target_cursor < stitch_target_stop && physical_positions[stitch_target_cursor] <= block_end_1) {
                        const int32_t target_idx = static_cast<int32_t>(stitch_target_cursor);
                        const int32_t local_qpos = qpos + static_cast<int32_t>(physical_positions[stitch_target_cursor] - block_start_1);
                        uint8_t q = qual[local_qpos];
                        if (cap_base_quality_by_mapping_quality) {
                            q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(rec->core.qual)));
                        }
                        if (q >= min_base_quality) {
                            const uint8_t obs_base = seq_base_ascii_upper(seq, local_qpos);
                            int8_t code = 2;
                            if (obs_base == ref_codes[target_idx]) {
                                code = 0;
                            } else if (obs_base == alt_codes[target_idx]) {
                                code = 1;
                            } else if (ref_alt_only) {
                                ++stitch_target_cursor;
                                continue;
                            }
                            read_pos.push_back(target_idx);
                            read_code.push_back(code);
                            read_qual.push_back(q);
                        }
                        ++stitch_target_cursor;
                    }
                    ref_pos += oplen;
                    qpos += oplen;
                } else if (snp_only_bamreader) {
                    const int64_t block_start_1 = static_cast<int64_t>(ref_pos) + 1;
                    const int64_t block_end_1 = static_cast<int64_t>(ref_pos) + static_cast<int64_t>(oplen);
                    const int64_t* begin = physical_positions;
                    const int64_t* end = physical_positions + n_positions;
                    const int64_t* it = std::lower_bound(begin, end, block_start_1);
                    while (it != end && *it <= block_end_1) {
                        const int32_t target_idx = static_cast<int32_t>(it - begin);
                        const int32_t local_qpos = qpos + static_cast<int32_t>(*it - block_start_1);
                        uint8_t q = qual[local_qpos];
                        if (cap_base_quality_by_mapping_quality) {
                            q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(rec->core.qual)));
                        }
                        if (q >= min_base_quality) {
                            const uint8_t obs_base = seq_base_ascii_upper(seq, local_qpos);
                            int8_t code = 2;
                            if (obs_base == ref_codes[target_idx]) {
                                code = 0;
                            } else if (obs_base == alt_codes[target_idx]) {
                                code = 1;
                            } else if (ref_alt_only) {
                                ++it;
                                continue;
                            }
                            read_pos.push_back(target_idx);
                            read_code.push_back(code);
                            read_qual.push_back(q);
                        }
                        ++it;
                    }
                    ref_pos += oplen;
                    qpos += oplen;
                } else {
                    for (int j = 0; j < oplen; ++j) {
                        const int64_t rel = static_cast<int64_t>(ref_pos) - static_cast<int64_t>(region_start);
                        if (rel >= 0 && rel < lookup_len) {
                            const int32_t target_idx = lookup[rel];
                            if (target_idx >= 0) {
                                uint8_t q = qual[qpos];
                                if (cap_base_quality_by_mapping_quality) {
                                    q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(rec->core.qual)));
                                }
                                if (q >= min_base_quality) {
                                    const uint8_t obs_base = seq_base_ascii_upper(seq, qpos);
                                    int8_t code = 2;
                                    if (obs_base == ref_codes[target_idx]) {
                                        code = 0;
                                    } else if (obs_base == alt_codes[target_idx]) {
                                        code = 1;
                                    } else if (ref_alt_only) {
                                        ++ref_pos;
                                        ++qpos;
                                        continue;
                                    }
                                    read_pos.push_back(target_idx);
                                    read_code.push_back(code);
                                    read_qual.push_back(q);
                                }
                            }
                        }
                        ++ref_pos;
                        ++qpos;
                    }
                }
            } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                qpos += oplen;
            } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                ref_pos += oplen;
            }
        }
        }

        if (read_pos.empty()) {
            continue;
        }
        ++out.n_reads;
        for (size_t i = 0; i < read_pos.size(); ++i) {
            const int32_t idx_pos = read_pos[i];
            if (idx_pos < 0 || idx_pos >= n_positions) {
                continue;
            }
            const size_t uidx = static_cast<size_t>(idx_pos);
            depth[uidx] += 1U;
            const float w = qual_weight[read_qual[i]];
            if (read_code[i] == 0) {
                ref_count[uidx] += 1U;
                ref_weight[uidx] += w;
            } else if (read_code[i] == 1) {
                alt_count[uidx] += 1U;
                alt_weight[uidx] += w;
            } else {
                other_count[uidx] += 1U;
                other_weight[uidx] += w;
            }
        }

        bool accumulate_to_end = false;
        std::string merge_key;
        if (do_merge) {
            if (use_bx_tag) {
                const std::string bx_value = aux_string_tag(rec, bx_tag);
                if (!bx_value.empty()) {
                    merge_key = std::string("BX:") + bx_value;
                    accumulate_to_end = true;
                }
            }
            if (merge_key.empty() && (flag & BAM_FPAIRED) != 0) {
                merge_key = std::string("Q:") + bam_get_qname(rec);
            } else if (merge_key.empty() && merge_unpaired_by_query) {
                merge_key = std::string("Q:") + bam_get_qname(rec);
                accumulate_to_end = true;
            }
        }

        if (!merge_key.empty() && accumulate_to_end) {
            auto it = pending.find(merge_key);
            if (it == pending.end()) {
                PendingFragment frag;
                frag.pos = read_pos;
                frag.code = read_code;
                frag.qual = read_qual;
                pending.emplace(merge_key, std::move(frag));
            } else {
                const size_t next_size = it->second.pos.size() + read_pos.size();
                if (next_size > static_cast<size_t>(std::max(bx_tag_upper_limit, 1))) {
                    append_fragment_compacted(
                        it->second.pos,
                        it->second.code,
                        it->second.qual,
                        physical_positions,
                        out.fragment_centers,
                        out.fragment_obs_offsets,
                        out.fragment_obs_pos,
                        out.fragment_obs_code,
                        out.fragment_obs_qual
                    );
                    it->second.pos = read_pos;
                    it->second.code = read_code;
                    it->second.qual = read_qual;
                } else {
                    append_pending_observation(it->second, read_pos, read_code, read_qual);
                }
            }
        } else if (!merge_key.empty()) {
            auto it = pending.find(merge_key);
            if (it == pending.end()) {
                PendingFragment frag;
                frag.pos = read_pos;
                frag.code = read_code;
                frag.qual = read_qual;
                pending.emplace(merge_key, std::move(frag));
            } else {
                append_pending_observation(it->second, read_pos, read_code, read_qual);
                append_fragment_compacted(
                    it->second.pos,
                    it->second.code,
                    it->second.qual,
                    physical_positions,
                    out.fragment_centers,
                    out.fragment_obs_offsets,
                    out.fragment_obs_pos,
                    out.fragment_obs_code,
                    out.fragment_obs_qual
                );
                pending.erase(it);
            }
        } else {
            append_fragment_compacted(
                read_pos,
                read_code,
                read_qual,
                physical_positions,
                out.fragment_centers,
                out.fragment_obs_offsets,
                out.fragment_obs_pos,
                out.fragment_obs_code,
                out.fragment_obs_qual
            );
        }
    }

    if (error_code == 0 && do_merge) {
        for (const auto& kv : pending) {
            append_fragment_compacted(
                kv.second.pos,
                kv.second.code,
                kv.second.qual,
                physical_positions,
                out.fragment_centers,
                out.fragment_obs_offsets,
                out.fragment_obs_pos,
                out.fragment_obs_code,
                out.fragment_obs_qual
            );
        }
    }

    if (rec != nullptr) {
        bam_destroy1(rec);
    }
    if (itr != nullptr) {
        hts_itr_destroy(itr);
    }
    if (idx != nullptr) {
        hts_idx_destroy(idx);
    }
    if (hdr != nullptr) {
        bam_hdr_destroy(hdr);
    }
    if (fp != nullptr) {
        sam_close(fp);
    }

    if (error_code != 0) {
        out.error_message = error_message;
        return false;
    }

    const uint32_t max_u16 = static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
    for (npy_intp i = 0; i < n_positions; ++i) {
        const size_t j = static_cast<size_t>(i);
        depth_out[j] = static_cast<uint16_t>(std::min(depth[j], max_u16));
        ref_count_out[j] = static_cast<uint16_t>(std::min(ref_count[j], max_u16));
        alt_count_out[j] = static_cast<uint16_t>(std::min(alt_count[j], max_u16));
        other_count_out[j] = static_cast<uint16_t>(std::min(other_count[j], max_u16));
        ref_weight_out[j] = ref_weight[j];
        alt_weight_out[j] = alt_weight[j];
        other_weight_out[j] = other_weight[j];
    }
    return true;
}

struct DiscoveryAccumulator {
    int64_t region_start = 0;
    int64_t region_stop = 0;
    size_t region_len = 0;
    std::vector<uint32_t> a_count;
    std::vector<uint32_t> c_count;
    std::vector<uint32_t> g_count;
    std::vector<uint32_t> t_count;
    std::vector<uint32_t> other_count;
    std::vector<uint32_t> a_forward;
    std::vector<uint32_t> c_forward;
    std::vector<uint32_t> g_forward;
    std::vector<uint32_t> t_forward;
    std::vector<uint32_t> a_reverse;
    std::vector<uint32_t> c_reverse;
    std::vector<uint32_t> g_reverse;
    std::vector<uint32_t> t_reverse;
    std::vector<uint32_t> sample_support;

    explicit DiscoveryAccumulator(int64_t start, int64_t stop)
        : region_start(start),
          region_stop(stop),
          region_len(static_cast<size_t>(std::max<int64_t>(stop - start, 0))),
          a_count(region_len, 0U),
          c_count(region_len, 0U),
          g_count(region_len, 0U),
          t_count(region_len, 0U),
          other_count(region_len, 0U),
          a_forward(region_len, 0U),
          c_forward(region_len, 0U),
          g_forward(region_len, 0U),
          t_forward(region_len, 0U),
          a_reverse(region_len, 0U),
          c_reverse(region_len, 0U),
          g_reverse(region_len, 0U),
          t_reverse(region_len, 0U),
          sample_support(region_len, 0U) {}

    inline void add_base(size_t offset, int base_idx, bool reverse) {
        if (base_idx == 0) {
            ++a_count[offset];
            reverse ? ++a_reverse[offset] : ++a_forward[offset];
        } else if (base_idx == 1) {
            ++c_count[offset];
            reverse ? ++c_reverse[offset] : ++c_forward[offset];
        } else if (base_idx == 2) {
            ++g_count[offset];
            reverse ? ++g_reverse[offset] : ++g_forward[offset];
        } else if (base_idx == 3) {
            ++t_count[offset];
            reverse ? ++t_reverse[offset] : ++t_forward[offset];
        } else {
            ++other_count[offset];
        }
    }

    inline uint32_t count_for_base(size_t offset, int base_idx) const {
        if (base_idx == 0) {
            return a_count[offset];
        }
        if (base_idx == 1) {
            return c_count[offset];
        }
        if (base_idx == 2) {
            return g_count[offset];
        }
        if (base_idx == 3) {
            return t_count[offset];
        }
        return 0U;
    }

    inline uint32_t forward_for_base(size_t offset, int base_idx) const {
        if (base_idx == 0) {
            return a_forward[offset];
        }
        if (base_idx == 1) {
            return c_forward[offset];
        }
        if (base_idx == 2) {
            return g_forward[offset];
        }
        if (base_idx == 3) {
            return t_forward[offset];
        }
        return 0U;
    }

    inline uint32_t reverse_for_base(size_t offset, int base_idx) const {
        if (base_idx == 0) {
            return a_reverse[offset];
        }
        if (base_idx == 1) {
            return c_reverse[offset];
        }
        if (base_idx == 2) {
            return g_reverse[offset];
        }
        if (base_idx == 3) {
            return t_reverse[offset];
        }
        return 0U;
    }
};

static bool discover_one_sample_snps(
    const char* bam_path,
    const char* chromosome,
    int64_t region_start,
    int64_t region_stop,
    const char* reference_seq,
    DiscoveryAccumulator& acc,
    int min_base_quality,
    int min_mapping_quality,
    int hts_threads,
    int max_insert_size,
    bool cap_base_quality_by_mapping_quality,
    int64_t& n_reads_seen,
    int64_t& n_bases_seen,
    std::string& error_message
) {
    std::vector<uint8_t> sample_alt_seen(acc.region_len, static_cast<uint8_t>(0));
    samFile* fp = sam_open(bam_path, "r");
    if (fp == nullptr) {
        error_message = std::string("Failed to open BAM/CRAM file: ") + bam_path;
        return false;
    }
    const int threads = std::max(hts_threads, 1);
    hts_set_threads(fp, threads);
    bam_hdr_t* hdr = sam_hdr_read(fp);
    if (hdr == nullptr) {
        sam_close(fp);
        error_message = std::string("Failed to read header from: ") + bam_path;
        return false;
    }
    hts_idx_t* idx = sam_index_load(fp, bam_path);
    if (idx == nullptr) {
        bam_hdr_destroy(hdr);
        sam_close(fp);
        error_message = std::string("Missing/invalid index for: ") + bam_path;
        return false;
    }
    const std::string region_query =
        std::string(chromosome) + ":" + std::to_string(region_start + 1) + "-" + std::to_string(region_stop);
    hts_itr_t* itr = sam_itr_querys(idx, hdr, region_query.c_str());
    if (itr == nullptr) {
        hts_idx_destroy(idx);
        bam_hdr_destroy(hdr);
        sam_close(fp);
        error_message = std::string("Failed to create iterator for region: ") + region_query;
        return false;
    }
    bam1_t* rec = bam_init1();
    if (rec == nullptr) {
        hts_itr_destroy(itr);
        hts_idx_destroy(idx);
        bam_hdr_destroy(hdr);
        sam_close(fp);
        error_message = "Failed to allocate bam record.";
        return false;
    }

    while (sam_itr_next(fp, itr, rec) >= 0) {
        const uint16_t flag = rec->core.flag;
        if ((flag & BAM_FUNMAP) != 0 || (flag & BAM_FSECONDARY) != 0 || (flag & BAM_FSUPPLEMENTARY) != 0 || (flag & BAM_FDUP) != 0) {
            continue;
        }
        if (rec->core.qual < min_mapping_quality) {
            continue;
        }
        if (max_insert_size > 0 && std::abs(static_cast<int>(rec->core.isize)) > max_insert_size) {
            continue;
        }
        const uint8_t* seq = bam_get_seq(rec);
        const uint8_t* qual = bam_get_qual(rec);
        if (seq == nullptr || qual == nullptr) {
            continue;
        }

        ++n_reads_seen;
        int64_t ref_pos = static_cast<int64_t>(rec->core.pos);
        int32_t qpos = 0;
        const bool reverse = (flag & BAM_FREVERSE) != 0;
        const uint32_t* cigar = bam_get_cigar(rec);
        for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
            const int op = bam_cigar_op(cigar[i]);
            const int oplen = bam_cigar_oplen(cigar[i]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                for (int j = 0; j < oplen; ++j) {
                    const int64_t cur_ref_pos = ref_pos + static_cast<int64_t>(j);
                    if (cur_ref_pos >= region_start && cur_ref_pos < region_stop) {
                        uint8_t q = qual[qpos + j];
                        if (q == 255U) {
                            q = 30U;
                        }
                        if (cap_base_quality_by_mapping_quality) {
                            q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(rec->core.qual)));
                        }
                        if (q >= min_base_quality) {
                            const size_t offset = static_cast<size_t>(cur_ref_pos - region_start);
                            const int ref_idx = base_index_from_ascii(static_cast<uint8_t>(reference_seq[offset]));
                            const int obs_idx = base_index_from_ascii(seq_base_ascii_upper(seq, qpos + j));
                            acc.add_base(offset, obs_idx, reverse);
                            if (ref_idx >= 0 && obs_idx >= 0 && obs_idx != ref_idx && sample_alt_seen[offset] == 0U) {
                                sample_alt_seen[offset] = 1U;
                                ++acc.sample_support[offset];
                            }
                            ++n_bases_seen;
                        }
                    }
                }
                ref_pos += static_cast<int64_t>(oplen);
                qpos += oplen;
            } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                qpos += oplen;
            } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                ref_pos += static_cast<int64_t>(oplen);
            }
        }
    }

    bam_destroy1(rec);
    hts_itr_destroy(itr);
    hts_idx_destroy(idx);
    bam_hdr_destroy(hdr);
    sam_close(fp);
    return true;
}

static PyObject* extract_sample_read_stream(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    const char* bam_path = nullptr;
    const char* chromosome = nullptr;
    long region_start = 0;  // 0-based inclusive
    long region_stop = 0;   // 0-based exclusive upper bound represented as 1-based terminal in python caller
    PyObject* lookup_obj = nullptr;
    PyObject* ref_obj = nullptr;
    PyObject* alt_obj = nullptr;
    PyObject* positions_obj = nullptr;
    int min_base_quality = 13;
    int min_mapping_quality = 20;
    int merge_fragments_by_query = 1;
    int hts_threads = 1;
    int snp_only_bamreader = 0;
    int stitch_style_bamreader = 0;
    int max_insert_size = 0;
    int cap_base_quality_by_mapping_quality = 0;
    int ref_alt_only = 0;
    int merge_unpaired_by_query = 0;
    int use_bx_tag = 0;
    const char* bx_tag = "BX";
    int bx_tag_upper_limit = 50000;

    static const char* kwlist[] = {
        "bam_path",
        "chromosome",
        "region_start",
        "region_stop",
        "lookup",
        "ref_codes",
        "alt_codes",
        "positions",
        "min_base_quality",
        "min_mapping_quality",
        "merge_fragments_by_query",
        "hts_threads",
        "snp_only_bamreader",
        "stitch_style_bamreader",
        "max_insert_size",
        "cap_base_quality_by_mapping_quality",
        "ref_alt_only",
        "merge_unpaired_by_query",
        "use_bx_tag",
        "bx_tag",
        "bx_tag_upper_limit",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "ssllOOOOii|piiiiiiiisi",
            const_cast<char**>(kwlist),
            &bam_path,
            &chromosome,
            &region_start,
            &region_stop,
            &lookup_obj,
            &ref_obj,
            &alt_obj,
            &positions_obj,
            &min_base_quality,
            &min_mapping_quality,
            &merge_fragments_by_query,
            &hts_threads,
            &snp_only_bamreader,
            &stitch_style_bamreader,
            &max_insert_size,
            &cap_base_quality_by_mapping_quality,
            &ref_alt_only,
            &merge_unpaired_by_query,
            &use_bx_tag,
            &bx_tag,
            &bx_tag_upper_limit)) {
        return nullptr;
    }

    PyArrayObject* lookup_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(lookup_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* ref_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(ref_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* alt_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(alt_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* positions_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(positions_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    if (lookup_arr == nullptr || ref_arr == nullptr || alt_arr == nullptr || positions_arr == nullptr) {
        Py_XDECREF(lookup_arr);
        Py_XDECREF(ref_arr);
        Py_XDECREF(alt_arr);
        Py_XDECREF(positions_arr);
        return nullptr;
    }
    if (PyArray_NDIM(lookup_arr) != 1 || PyArray_NDIM(ref_arr) != 1 || PyArray_NDIM(alt_arr) != 1 || PyArray_NDIM(positions_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "lookup/ref_codes/alt_codes/positions must be 1D arrays.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        return nullptr;
    }

    const npy_intp n_positions = PyArray_SIZE(ref_arr);
    if (PyArray_SIZE(alt_arr) != n_positions || PyArray_SIZE(positions_arr) != n_positions) {
        PyErr_SetString(PyExc_ValueError, "ref_codes, alt_codes, and positions must have same length.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        return nullptr;
    }

    const int32_t* lookup = reinterpret_cast<int32_t*>(PyArray_DATA(lookup_arr));
    const uint8_t* ref_codes = reinterpret_cast<uint8_t*>(PyArray_DATA(ref_arr));
    const uint8_t* alt_codes = reinterpret_cast<uint8_t*>(PyArray_DATA(alt_arr));
    const int64_t* physical_positions = reinterpret_cast<int64_t*>(PyArray_DATA(positions_arr));
    const npy_intp lookup_len = PyArray_SIZE(lookup_arr);
    const bool do_merge = merge_fragments_by_query != 0;
    const int threads = std::max(hts_threads, 1);

    std::vector<uint32_t> depth(static_cast<size_t>(n_positions), 0U);
    std::vector<uint32_t> ref_count(static_cast<size_t>(n_positions), 0U);
    std::vector<uint32_t> alt_count(static_cast<size_t>(n_positions), 0U);
    std::vector<uint32_t> other_count(static_cast<size_t>(n_positions), 0U);
    std::vector<float> ref_weight(static_cast<size_t>(n_positions), 0.0f);
    std::vector<float> alt_weight(static_cast<size_t>(n_positions), 0.0f);
    std::vector<float> other_weight(static_cast<size_t>(n_positions), 0.0f);

    float qual_weight[256];
    for (int q = 0; q < 256; ++q) {
        const float error_rate = std::pow(10.0f, -(static_cast<float>(q) / 10.0f));
        qual_weight[q] = 1.0f - error_rate;
    }

    int64_t n_reads = 0;
    std::vector<int32_t> fragment_centers;
    std::vector<int64_t> fragment_obs_offsets;
    std::vector<int32_t> fragment_obs_pos;
    std::vector<int8_t> fragment_obs_code;
    std::vector<uint8_t> fragment_obs_qual;
    fragment_obs_offsets.push_back(0);

    std::unordered_map<std::string, PendingFragment> pending;

    int error_code = 0;
    std::string error_message;

    Py_BEGIN_ALLOW_THREADS
    samFile* fp = sam_open(bam_path, "r");
    if (fp == nullptr) {
        error_code = 1;
        error_message = std::string("Failed to open BAM/CRAM file: ") + bam_path;
    }
    bam_hdr_t* hdr = nullptr;
    hts_idx_t* idx = nullptr;
    hts_itr_t* itr = nullptr;
    bam1_t* rec = nullptr;
    if (error_code == 0) {
        hts_set_threads(fp, threads);
        hdr = sam_hdr_read(fp);
        if (hdr == nullptr) {
            error_code = 1;
            error_message = std::string("Failed to read header from: ") + bam_path;
        }
    }
    if (error_code == 0) {
        idx = sam_index_load(fp, bam_path);
        if (idx == nullptr) {
            error_code = 1;
            error_message = std::string("Missing/invalid index for: ") + bam_path;
        }
    }
    if (error_code == 0) {
        const std::string region_query =
            std::string(chromosome) + ":" + std::to_string(region_start + 1) + "-" + std::to_string(region_stop);
        itr = sam_itr_querys(idx, hdr, region_query.c_str());
        if (itr == nullptr) {
            error_code = 1;
            error_message = std::string("Failed to create iterator for region: ") + region_query;
        }
    }
    if (error_code == 0) {
        rec = bam_init1();
        if (rec == nullptr) {
            error_code = 1;
            error_message = "Failed to allocate bam record.";
        }
    }

    std::vector<int32_t> read_pos;
    std::vector<int8_t> read_code;
    std::vector<uint8_t> read_qual;
    read_pos.reserve(64);
    read_code.reserve(64);
    read_qual.reserve(64);
    npy_intp stitch_scan_start = 0;

    while (error_code == 0 && sam_itr_next(fp, itr, rec) >= 0) {
        const uint16_t flag = rec->core.flag;
        if ((flag & BAM_FUNMAP) != 0 || (flag & BAM_FSECONDARY) != 0 || (flag & BAM_FSUPPLEMENTARY) != 0 || (flag & BAM_FDUP) != 0) {
            continue;
        }
        if (rec->core.qual < min_mapping_quality) {
            continue;
        }
        if (max_insert_size > 0 && std::abs(static_cast<int>(rec->core.isize)) > max_insert_size) {
            continue;
        }
        const uint8_t* seq = bam_get_seq(rec);
        const uint8_t* qual = bam_get_qual(rec);
        if (seq == nullptr || qual == nullptr) {
            continue;
        }

        read_pos.clear();
        read_code.clear();
        read_qual.clear();

        int32_t ref_pos = rec->core.pos;
        int32_t qpos = 0;
        const uint32_t* cigar = bam_get_cigar(rec);
        npy_intp stitch_target_cursor = 0;
        npy_intp stitch_target_stop = 0;
        if (stitch_style_bamreader != 0) {
            const int64_t read_start_1 = static_cast<int64_t>(rec->core.pos) + 1;
            const int64_t ref_span = reference_span_from_cigar(rec);
            if (ref_span <= 0) {
                continue;
            }
            const int64_t read_end_1 = read_start_1 + ref_span - 1;
            while (stitch_scan_start < n_positions && physical_positions[stitch_scan_start] < read_start_1) {
                ++stitch_scan_start;
            }
            stitch_target_cursor = stitch_scan_start;
            const int64_t* begin = physical_positions + stitch_target_cursor;
            const int64_t* end = physical_positions + n_positions;
            stitch_target_stop = static_cast<npy_intp>(std::upper_bound(begin, end, read_end_1) - physical_positions);
            if (stitch_target_cursor >= stitch_target_stop) {
                continue;
            }
        }
        for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
            const int op = bam_cigar_op(cigar[i]);
            const int oplen = bam_cigar_oplen(cigar[i]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                if (stitch_style_bamreader != 0) {
                    const int64_t block_start_1 = static_cast<int64_t>(ref_pos) + 1;
                    const int64_t block_end_1 = static_cast<int64_t>(ref_pos) + static_cast<int64_t>(oplen);
                    while (stitch_target_cursor < stitch_target_stop && physical_positions[stitch_target_cursor] < block_start_1) {
                        ++stitch_target_cursor;
                    }
                    while (stitch_target_cursor < stitch_target_stop && physical_positions[stitch_target_cursor] <= block_end_1) {
                        const int32_t target_idx = static_cast<int32_t>(stitch_target_cursor);
                        const int32_t local_qpos = qpos + static_cast<int32_t>(physical_positions[stitch_target_cursor] - block_start_1);
                        uint8_t q = qual[local_qpos];
                        if (cap_base_quality_by_mapping_quality != 0) {
                            q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(rec->core.qual)));
                        }
                        if (q >= min_base_quality) {
                            const uint8_t obs_base = seq_base_ascii_upper(seq, local_qpos);
                            int8_t code = 2;
                            if (obs_base == ref_codes[target_idx]) {
                                code = 0;
                            } else if (obs_base == alt_codes[target_idx]) {
                                code = 1;
                            } else if (ref_alt_only != 0) {
                                ++stitch_target_cursor;
                                continue;
                            }
                            read_pos.push_back(target_idx);
                            read_code.push_back(code);
                            read_qual.push_back(q);
                        }
                        ++stitch_target_cursor;
                    }
                    ref_pos += oplen;
                    qpos += oplen;
                } else if (snp_only_bamreader != 0) {
                    const int64_t block_start_1 = static_cast<int64_t>(ref_pos) + 1;
                    const int64_t block_end_1 = static_cast<int64_t>(ref_pos) + static_cast<int64_t>(oplen);
                    const int64_t* begin = physical_positions;
                    const int64_t* end = physical_positions + n_positions;
                    const int64_t* it = std::lower_bound(begin, end, block_start_1);
                    while (it != end && *it <= block_end_1) {
                        const int32_t target_idx = static_cast<int32_t>(it - begin);
                        const int32_t local_qpos = qpos + static_cast<int32_t>(*it - block_start_1);
                        uint8_t q = qual[local_qpos];
                        if (cap_base_quality_by_mapping_quality != 0) {
                            q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(rec->core.qual)));
                        }
                        if (q >= min_base_quality) {
                            const uint8_t obs_base = seq_base_ascii_upper(seq, local_qpos);
                            int8_t code = 2;
                            if (obs_base == ref_codes[target_idx]) {
                                code = 0;
                            } else if (obs_base == alt_codes[target_idx]) {
                                code = 1;
                            } else if (ref_alt_only != 0) {
                                ++it;
                                continue;
                            }
                            read_pos.push_back(target_idx);
                            read_code.push_back(code);
                            read_qual.push_back(q);
                        }
                        ++it;
                    }
                    ref_pos += oplen;
                    qpos += oplen;
                } else {
                    for (int j = 0; j < oplen; ++j) {
                        const int64_t rel = static_cast<int64_t>(ref_pos) - static_cast<int64_t>(region_start);
                        if (rel >= 0 && rel < lookup_len) {
                            const int32_t target_idx = lookup[rel];
                            if (target_idx >= 0) {
                                uint8_t q = qual[qpos];
                                if (cap_base_quality_by_mapping_quality != 0) {
                                    q = static_cast<uint8_t>(std::min<int>(static_cast<int>(q), static_cast<int>(rec->core.qual)));
                                }
                                if (q >= min_base_quality) {
                                    const uint8_t obs_base = seq_base_ascii_upper(seq, qpos);
                                    int8_t code = 2;
                                    if (obs_base == ref_codes[target_idx]) {
                                        code = 0;
                                    } else if (obs_base == alt_codes[target_idx]) {
                                        code = 1;
                                    } else if (ref_alt_only != 0) {
                                        ++ref_pos;
                                        ++qpos;
                                        continue;
                                    }
                                    read_pos.push_back(target_idx);
                                    read_code.push_back(code);
                                    read_qual.push_back(q);
                                }
                            }
                        }
                        ++ref_pos;
                        ++qpos;
                    }
                }
            } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) {
                qpos += oplen;
            } else if (op == BAM_CDEL || op == BAM_CREF_SKIP) {
                ref_pos += oplen;
            } else {
                // BAM_CHARD_CLIP, BAM_CPAD, BAM_CBACK etc: consume neither or handled as no-op.
            }
        }

        if (read_pos.empty()) {
            continue;
        }
        ++n_reads;
        for (size_t i = 0; i < read_pos.size(); ++i) {
            const int32_t idx_pos = read_pos[i];
            if (idx_pos < 0 || idx_pos >= n_positions) {
                continue;
            }
            const size_t uidx = static_cast<size_t>(idx_pos);
            depth[uidx] += 1U;
            const float w = qual_weight[read_qual[i]];
            if (read_code[i] == 0) {
                ref_count[uidx] += 1U;
                ref_weight[uidx] += w;
            } else if (read_code[i] == 1) {
                alt_count[uidx] += 1U;
                alt_weight[uidx] += w;
            } else {
                other_count[uidx] += 1U;
                other_weight[uidx] += w;
            }
        }

        bool accumulate_to_end = false;
        std::string merge_key;
        if (do_merge) {
            if (use_bx_tag != 0) {
                const std::string bx_value = aux_string_tag(rec, bx_tag);
                if (!bx_value.empty()) {
                    merge_key = std::string("BX:") + bx_value;
                    accumulate_to_end = true;
                }
            }
            if (merge_key.empty() && (flag & BAM_FPAIRED) != 0) {
                merge_key = std::string("Q:") + bam_get_qname(rec);
            } else if (merge_key.empty() && merge_unpaired_by_query != 0) {
                merge_key = std::string("Q:") + bam_get_qname(rec);
                accumulate_to_end = true;
            }
        }

        if (!merge_key.empty() && accumulate_to_end) {
            auto it = pending.find(merge_key);
            if (it == pending.end()) {
                PendingFragment frag;
                frag.pos = read_pos;
                frag.code = read_code;
                frag.qual = read_qual;
                pending.emplace(merge_key, std::move(frag));
            } else {
                const size_t next_size = it->second.pos.size() + read_pos.size();
                if (next_size > static_cast<size_t>(std::max(bx_tag_upper_limit, 1))) {
                    append_fragment_compacted(
                        it->second.pos,
                        it->second.code,
                        it->second.qual,
                        physical_positions,
                        fragment_centers,
                        fragment_obs_offsets,
                        fragment_obs_pos,
                        fragment_obs_code,
                        fragment_obs_qual
                    );
                    it->second.pos = read_pos;
                    it->second.code = read_code;
                    it->second.qual = read_qual;
                } else {
                    append_pending_observation(it->second, read_pos, read_code, read_qual);
                }
            }
        } else if (!merge_key.empty()) {
            auto it = pending.find(merge_key);
            if (it == pending.end()) {
                PendingFragment frag;
                frag.pos = read_pos;
                frag.code = read_code;
                frag.qual = read_qual;
                pending.emplace(merge_key, std::move(frag));
            } else {
                append_pending_observation(it->second, read_pos, read_code, read_qual);
                append_fragment_compacted(
                    it->second.pos,
                    it->second.code,
                    it->second.qual,
                    physical_positions,
                    fragment_centers,
                    fragment_obs_offsets,
                    fragment_obs_pos,
                    fragment_obs_code,
                    fragment_obs_qual
                );
                pending.erase(it);
            }
        } else {
            append_fragment_compacted(
                read_pos,
                read_code,
                read_qual,
                physical_positions,
                fragment_centers,
                fragment_obs_offsets,
                fragment_obs_pos,
                fragment_obs_code,
                fragment_obs_qual
            );
        }
    }

    if (error_code == 0 && do_merge) {
        for (const auto& kv : pending) {
            append_fragment_compacted(
                kv.second.pos,
                kv.second.code,
                kv.second.qual,
                physical_positions,
                fragment_centers,
                fragment_obs_offsets,
                fragment_obs_pos,
                fragment_obs_code,
                fragment_obs_qual
            );
        }
    }

    if (rec != nullptr) {
        bam_destroy1(rec);
    }
    if (itr != nullptr) {
        hts_itr_destroy(itr);
    }
    if (idx != nullptr) {
        hts_idx_destroy(idx);
    }
    if (hdr != nullptr) {
        bam_hdr_destroy(hdr);
    }
    if (fp != nullptr) {
        sam_close(fp);
    }
    Py_END_ALLOW_THREADS

    if (error_code != 0) {
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }

    const npy_intp one_dim[1] = {n_positions};
    PyObject* depth_arr = PyArray_SimpleNew(1, one_dim, NPY_UINT16);
    PyObject* ref_count_arr = PyArray_SimpleNew(1, one_dim, NPY_UINT16);
    PyObject* alt_count_arr = PyArray_SimpleNew(1, one_dim, NPY_UINT16);
    PyObject* other_count_arr = PyArray_SimpleNew(1, one_dim, NPY_UINT16);
    PyObject* ref_weight_arr = PyArray_SimpleNew(1, one_dim, NPY_FLOAT32);
    PyObject* alt_weight_arr = PyArray_SimpleNew(1, one_dim, NPY_FLOAT32);
    PyObject* other_weight_arr = PyArray_SimpleNew(1, one_dim, NPY_FLOAT32);
    if (depth_arr == nullptr || ref_count_arr == nullptr || alt_count_arr == nullptr || other_count_arr == nullptr ||
        ref_weight_arr == nullptr || alt_weight_arr == nullptr || other_weight_arr == nullptr) {
        Py_XDECREF(depth_arr);
        Py_XDECREF(ref_count_arr);
        Py_XDECREF(alt_count_arr);
        Py_XDECREF(other_count_arr);
        Py_XDECREF(ref_weight_arr);
        Py_XDECREF(alt_weight_arr);
        Py_XDECREF(other_weight_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        return nullptr;
    }

    auto* depth_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(depth_arr)));
    auto* ref_count_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ref_count_arr)));
    auto* alt_count_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(alt_count_arr)));
    auto* other_count_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(other_count_arr)));
    auto* ref_weight_out = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ref_weight_arr)));
    auto* alt_weight_out = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(alt_weight_arr)));
    auto* other_weight_out = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(other_weight_arr)));

    const uint32_t max_u16 = static_cast<uint32_t>(std::numeric_limits<uint16_t>::max());
    for (npy_intp i = 0; i < n_positions; ++i) {
        const size_t j = static_cast<size_t>(i);
        depth_out[j] = static_cast<uint16_t>(std::min(depth[j], max_u16));
        ref_count_out[j] = static_cast<uint16_t>(std::min(ref_count[j], max_u16));
        alt_count_out[j] = static_cast<uint16_t>(std::min(alt_count[j], max_u16));
        other_count_out[j] = static_cast<uint16_t>(std::min(other_count[j], max_u16));
        ref_weight_out[j] = ref_weight[j];
        alt_weight_out[j] = alt_weight[j];
        other_weight_out[j] = other_weight[j];
    }

    const npy_intp centers_dim[1] = {static_cast<npy_intp>(fragment_centers.size())};
    const npy_intp obs_offsets_dim[1] = {static_cast<npy_intp>(fragment_obs_offsets.size())};
    const npy_intp obs_dim[1] = {static_cast<npy_intp>(fragment_obs_pos.size())};
    PyObject* centers_arr = PyArray_SimpleNew(1, centers_dim, NPY_INT32);
    PyObject* obs_offsets_arr = PyArray_SimpleNew(1, obs_offsets_dim, NPY_INT64);
    PyObject* obs_pos_arr = PyArray_SimpleNew(1, obs_dim, NPY_INT32);
    PyObject* obs_code_arr = PyArray_SimpleNew(1, obs_dim, NPY_INT8);
    PyObject* obs_qual_arr = PyArray_SimpleNew(1, obs_dim, NPY_UINT8);
    if (centers_arr == nullptr || obs_offsets_arr == nullptr || obs_pos_arr == nullptr || obs_code_arr == nullptr || obs_qual_arr == nullptr) {
        Py_XDECREF(centers_arr);
        Py_XDECREF(obs_offsets_arr);
        Py_XDECREF(obs_pos_arr);
        Py_XDECREF(obs_code_arr);
        Py_XDECREF(obs_qual_arr);
        Py_DECREF(depth_arr);
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        return nullptr;
    }
    if (!fragment_centers.empty()) {
        std::copy(
            fragment_centers.begin(),
            fragment_centers.end(),
            reinterpret_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(centers_arr)))
        );
    }
    if (!fragment_obs_offsets.empty()) {
        std::copy(
            fragment_obs_offsets.begin(),
            fragment_obs_offsets.end(),
            reinterpret_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_offsets_arr)))
        );
    }
    if (!fragment_obs_pos.empty()) {
        std::copy(
            fragment_obs_pos.begin(),
            fragment_obs_pos.end(),
            reinterpret_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_pos_arr)))
        );
        std::copy(
            fragment_obs_code.begin(),
            fragment_obs_code.end(),
            reinterpret_cast<int8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_code_arr)))
        );
        std::copy(
            fragment_obs_qual.begin(),
            fragment_obs_qual.end(),
            reinterpret_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_qual_arr)))
        );
    }

    PyObject* n_reads_obj = PyLong_FromLongLong(n_reads);
    if (n_reads_obj == nullptr) {
        Py_DECREF(depth_arr);
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        Py_DECREF(centers_arr);
        Py_DECREF(obs_offsets_arr);
        Py_DECREF(obs_pos_arr);
        Py_DECREF(obs_code_arr);
        Py_DECREF(obs_qual_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        return nullptr;
    }

    PyObject* out = PyTuple_New(13);
    if (out == nullptr) {
        Py_DECREF(depth_arr);
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        Py_DECREF(n_reads_obj);
        Py_DECREF(centers_arr);
        Py_DECREF(obs_offsets_arr);
        Py_DECREF(obs_pos_arr);
        Py_DECREF(obs_code_arr);
        Py_DECREF(obs_qual_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        return nullptr;
    }
    PyTuple_SET_ITEM(out, 0, depth_arr);
    PyTuple_SET_ITEM(out, 1, ref_count_arr);
    PyTuple_SET_ITEM(out, 2, alt_count_arr);
    PyTuple_SET_ITEM(out, 3, other_count_arr);
    PyTuple_SET_ITEM(out, 4, ref_weight_arr);
    PyTuple_SET_ITEM(out, 5, alt_weight_arr);
    PyTuple_SET_ITEM(out, 6, other_weight_arr);
    PyTuple_SET_ITEM(out, 7, n_reads_obj);
    PyTuple_SET_ITEM(out, 8, centers_arr);
    PyTuple_SET_ITEM(out, 9, obs_offsets_arr);
    PyTuple_SET_ITEM(out, 10, obs_pos_arr);
    PyTuple_SET_ITEM(out, 11, obs_code_arr);
    PyTuple_SET_ITEM(out, 12, obs_qual_arr);

    Py_DECREF(lookup_arr);
    Py_DECREF(ref_arr);
    Py_DECREF(alt_arr);
    Py_DECREF(positions_arr);
    return out;
}

static PyObject* extract_samples_read_stream_batch(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    PyObject* bam_paths_obj = nullptr;
    const char* chromosome = nullptr;
    long region_start = 0;
    long region_stop = 0;
    PyObject* lookup_obj = nullptr;
    PyObject* ref_obj = nullptr;
    PyObject* alt_obj = nullptr;
    PyObject* positions_obj = nullptr;
    PyObject* ref_alleles_obj = nullptr;
    PyObject* alt_alleles_obj = nullptr;
    PyObject* variant_types_obj = nullptr;
    int min_base_quality = 13;
    int min_mapping_quality = 20;
    int merge_fragments_by_query = 1;
    int hts_threads = 1;
    int snp_only_bamreader = 0;
    int stitch_style_bamreader = 0;
    int max_insert_size = 0;
    int cap_base_quality_by_mapping_quality = 0;
    int ref_alt_only = 0;
    int merge_unpaired_by_query = 0;
    int use_bx_tag = 0;
    const char* bx_tag = "BX";
    int bx_tag_upper_limit = 50000;
    int variant_aware_bamreader = 0;
    int max_indel_len = 50;
    int io_workers = 1;

    static const char* kwlist[] = {
        "bam_paths",
        "chromosome",
        "region_start",
        "region_stop",
        "lookup",
        "ref_codes",
        "alt_codes",
        "positions",
        "min_base_quality",
        "min_mapping_quality",
        "merge_fragments_by_query",
        "hts_threads",
        "snp_only_bamreader",
        "stitch_style_bamreader",
        "max_insert_size",
        "cap_base_quality_by_mapping_quality",
        "ref_alt_only",
        "merge_unpaired_by_query",
        "use_bx_tag",
        "bx_tag",
        "bx_tag_upper_limit",
        "ref_alleles",
        "alt_alleles",
        "variant_types",
        "variant_aware_bamreader",
        "max_indel_len",
        "io_workers",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OsllOOOOii|piiiiiiiisiOOOiii",
            const_cast<char**>(kwlist),
            &bam_paths_obj,
            &chromosome,
            &region_start,
            &region_stop,
            &lookup_obj,
            &ref_obj,
            &alt_obj,
            &positions_obj,
            &min_base_quality,
            &min_mapping_quality,
            &merge_fragments_by_query,
            &hts_threads,
            &snp_only_bamreader,
            &stitch_style_bamreader,
            &max_insert_size,
            &cap_base_quality_by_mapping_quality,
            &ref_alt_only,
            &merge_unpaired_by_query,
            &use_bx_tag,
            &bx_tag,
            &bx_tag_upper_limit,
            &ref_alleles_obj,
            &alt_alleles_obj,
            &variant_types_obj,
            &variant_aware_bamreader,
            &max_indel_len,
            &io_workers)) {
        return nullptr;
    }

    PyObject* path_seq = PySequence_Fast(bam_paths_obj, "bam_paths must be a sequence of path strings.");
    if (path_seq == nullptr) {
        return nullptr;
    }
    const Py_ssize_t n_samples_py = PySequence_Fast_GET_SIZE(path_seq);
    std::vector<std::string> bam_paths;
    bam_paths.reserve(static_cast<size_t>(std::max<Py_ssize_t>(n_samples_py, 0)));
    for (Py_ssize_t i = 0; i < n_samples_py; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(path_seq, i);
        PyObject* item_str = PyObject_Str(item);
        if (item_str == nullptr) {
            Py_DECREF(path_seq);
            return nullptr;
        }
        const char* path_c = PyUnicode_AsUTF8(item_str);
        if (path_c == nullptr) {
            Py_DECREF(item_str);
            Py_DECREF(path_seq);
            return nullptr;
        }
        bam_paths.emplace_back(path_c);
        Py_DECREF(item_str);
    }

    PyArrayObject* lookup_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(lookup_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* ref_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(ref_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* alt_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(alt_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* positions_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(positions_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* variant_types_arr = nullptr;
    if (variant_types_obj != nullptr && variant_types_obj != Py_None) {
        variant_types_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(variant_types_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    }
    if (lookup_arr == nullptr || ref_arr == nullptr || alt_arr == nullptr || positions_arr == nullptr) {
        Py_XDECREF(lookup_arr);
        Py_XDECREF(ref_arr);
        Py_XDECREF(alt_arr);
        Py_XDECREF(positions_arr);
        Py_XDECREF(variant_types_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    if (variant_types_obj != nullptr && variant_types_obj != Py_None && variant_types_arr == nullptr) {
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_XDECREF(variant_types_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    if (PyArray_NDIM(lookup_arr) != 1 || PyArray_NDIM(ref_arr) != 1 || PyArray_NDIM(alt_arr) != 1 || PyArray_NDIM(positions_arr) != 1 ||
        (variant_types_arr != nullptr && PyArray_NDIM(variant_types_arr) != 1)) {
        PyErr_SetString(PyExc_ValueError, "lookup/ref_codes/alt_codes/positions must be 1D arrays.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_XDECREF(variant_types_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    const npy_intp n_positions = PyArray_SIZE(ref_arr);
    if (
        PyArray_SIZE(alt_arr) != n_positions ||
        PyArray_SIZE(positions_arr) != n_positions ||
        (variant_types_arr != nullptr && PyArray_SIZE(variant_types_arr) != n_positions)
    ) {
        PyErr_SetString(PyExc_ValueError, "ref_codes, alt_codes, variant_types, and positions must have same length.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_XDECREF(variant_types_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    std::vector<uint8_t> default_variant_types;
    const uint8_t* variant_types = nullptr;
    if (variant_types_arr != nullptr) {
        variant_types = reinterpret_cast<uint8_t*>(PyArray_DATA(variant_types_arr));
    } else {
        default_variant_types.assign(static_cast<size_t>(n_positions), VARIANT_SNP);
        variant_types = default_variant_types.data();
    }
    std::vector<std::string> ref_alleles;
    std::vector<std::string> alt_alleles;
    if (variant_aware_bamreader != 0) {
        if (ref_alleles_obj == nullptr || alt_alleles_obj == nullptr) {
            PyErr_SetString(PyExc_ValueError, "variant_aware_bamreader requires ref_alleles and alt_alleles.");
            Py_DECREF(lookup_arr);
            Py_DECREF(ref_arr);
            Py_DECREF(alt_arr);
            Py_DECREF(positions_arr);
            Py_XDECREF(variant_types_arr);
            Py_DECREF(path_seq);
            return nullptr;
        }
        if (!py_sequence_to_strings(ref_alleles_obj, n_positions, "ref_alleles", ref_alleles) ||
            !py_sequence_to_strings(alt_alleles_obj, n_positions, "alt_alleles", alt_alleles)) {
            Py_DECREF(lookup_arr);
            Py_DECREF(ref_arr);
            Py_DECREF(alt_arr);
            Py_DECREF(positions_arr);
            Py_XDECREF(variant_types_arr);
            Py_DECREF(path_seq);
            return nullptr;
        }
    }

    const npy_intp n_samples = static_cast<npy_intp>(bam_paths.size());
    const npy_intp matrix_dims[2] = {n_samples, n_positions};
    PyObject* depth_arr = PyArray_SimpleNew(2, matrix_dims, NPY_UINT16);
    PyObject* ref_count_arr = PyArray_SimpleNew(2, matrix_dims, NPY_UINT16);
    PyObject* alt_count_arr = PyArray_SimpleNew(2, matrix_dims, NPY_UINT16);
    PyObject* other_count_arr = PyArray_SimpleNew(2, matrix_dims, NPY_UINT16);
    PyObject* ref_weight_arr = PyArray_SimpleNew(2, matrix_dims, NPY_FLOAT32);
    PyObject* alt_weight_arr = PyArray_SimpleNew(2, matrix_dims, NPY_FLOAT32);
    PyObject* other_weight_arr = PyArray_SimpleNew(2, matrix_dims, NPY_FLOAT32);
    if (depth_arr == nullptr || ref_count_arr == nullptr || alt_count_arr == nullptr || other_count_arr == nullptr ||
        ref_weight_arr == nullptr || alt_weight_arr == nullptr || other_weight_arr == nullptr) {
        Py_XDECREF(depth_arr);
        Py_XDECREF(ref_count_arr);
        Py_XDECREF(alt_count_arr);
        Py_XDECREF(other_count_arr);
        Py_XDECREF(ref_weight_arr);
        Py_XDECREF(alt_weight_arr);
        Py_XDECREF(other_weight_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }

    auto* depth_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(depth_arr)));
    auto* ref_count_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ref_count_arr)));
    auto* alt_count_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(alt_count_arr)));
    auto* other_count_out = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(other_count_arr)));
    auto* ref_weight_out = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ref_weight_arr)));
    auto* alt_weight_out = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(alt_weight_arr)));
    auto* other_weight_out = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(other_weight_arr)));

    const int32_t* lookup = reinterpret_cast<int32_t*>(PyArray_DATA(lookup_arr));
    const uint8_t* ref_codes = reinterpret_cast<uint8_t*>(PyArray_DATA(ref_arr));
    const uint8_t* alt_codes = reinterpret_cast<uint8_t*>(PyArray_DATA(alt_arr));
    const int64_t* physical_positions = reinterpret_cast<int64_t*>(PyArray_DATA(positions_arr));
    const npy_intp lookup_len = PyArray_SIZE(lookup_arr);
    const bool do_merge = merge_fragments_by_query != 0;
    const int threads = std::max(hts_threads, 1);

    std::vector<int32_t> n_reads_vec(static_cast<size_t>(n_samples), 0);
    std::vector<int64_t> sample_offsets(static_cast<size_t>(n_samples) + 1, 0);
    std::vector<int32_t> all_centers;
    std::vector<int64_t> all_obs_offsets;
    std::vector<int32_t> all_obs_pos;
    std::vector<int8_t> all_obs_code;
    std::vector<uint8_t> all_obs_qual;
    all_obs_offsets.push_back(0);
    int error_code = 0;
    std::string error_message;

    std::vector<ExtractedSample> sample_results(static_cast<size_t>(n_samples));
    std::atomic<npy_intp> next_sample(0);
    std::mutex error_mutex;
    const int n_worker_threads = static_cast<int>(std::max<npy_intp>(
        1,
        std::min<npy_intp>(static_cast<npy_intp>(std::max(io_workers, 1)), n_samples)
    ));

    Py_BEGIN_ALLOW_THREADS
    auto worker = [&]() {
        while (true) {
            if (error_code != 0) {
                return;
            }
            const npy_intp sample_idx = next_sample.fetch_add(1);
            if (sample_idx >= n_samples) {
                return;
            }
            ExtractedSample& sample = sample_results[static_cast<size_t>(sample_idx)];
            const size_t row_offset = static_cast<size_t>(sample_idx * n_positions);
            const bool ok = extract_one_sample_to_rows(
                bam_paths[static_cast<size_t>(sample_idx)].c_str(),
                chromosome,
                region_start,
                region_stop,
                lookup,
                lookup_len,
                ref_codes,
                alt_codes,
                physical_positions,
                variant_types,
                (variant_aware_bamreader != 0 ? &ref_alleles : nullptr),
                (variant_aware_bamreader != 0 ? &alt_alleles : nullptr),
                n_positions,
                min_base_quality,
                min_mapping_quality,
                do_merge,
                threads,
                snp_only_bamreader != 0,
                stitch_style_bamreader != 0,
                variant_aware_bamreader != 0,
                max_indel_len,
                max_insert_size,
                cap_base_quality_by_mapping_quality != 0,
                ref_alt_only != 0,
                merge_unpaired_by_query != 0,
                use_bx_tag != 0,
                bx_tag,
                bx_tag_upper_limit,
                depth_out + row_offset,
                ref_count_out + row_offset,
                alt_count_out + row_offset,
                other_count_out + row_offset,
                ref_weight_out + row_offset,
                alt_weight_out + row_offset,
                other_weight_out + row_offset,
                sample
            );
            if (!ok) {
                std::lock_guard<std::mutex> lock(error_mutex);
                if (error_code == 0) {
                    error_code = 1;
                    error_message = "sample " + std::to_string(sample_idx) + ": " + sample.error_message;
                }
                return;
            }
        }
    };
    if (n_worker_threads <= 1) {
        worker();
    } else {
        std::vector<std::thread> worker_threads;
        worker_threads.reserve(static_cast<size_t>(n_worker_threads));
        for (int worker_idx = 0; worker_idx < n_worker_threads; ++worker_idx) {
            worker_threads.emplace_back(worker);
        }
        for (std::thread& thread : worker_threads) {
            thread.join();
        }
    }
    Py_END_ALLOW_THREADS

    for (npy_intp sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        const ExtractedSample& sample = sample_results[static_cast<size_t>(sample_idx)];
        n_reads_vec[static_cast<size_t>(sample_idx)] = static_cast<int32_t>(std::min<int64_t>(
            sample.n_reads,
            static_cast<int64_t>(std::numeric_limits<int32_t>::max())
        ));
        sample_offsets[static_cast<size_t>(sample_idx) + 1] =
            sample_offsets[static_cast<size_t>(sample_idx)] + static_cast<int64_t>(sample.fragment_centers.size());
        all_centers.insert(all_centers.end(), sample.fragment_centers.begin(), sample.fragment_centers.end());
        const int64_t base_offset = all_obs_offsets.back();
        for (size_t off_idx = 1; off_idx < sample.fragment_obs_offsets.size(); ++off_idx) {
            all_obs_offsets.push_back(base_offset + sample.fragment_obs_offsets[off_idx]);
        }
        all_obs_pos.insert(all_obs_pos.end(), sample.fragment_obs_pos.begin(), sample.fragment_obs_pos.end());
        all_obs_code.insert(all_obs_code.end(), sample.fragment_obs_code.begin(), sample.fragment_obs_code.end());
        all_obs_qual.insert(all_obs_qual.end(), sample.fragment_obs_qual.begin(), sample.fragment_obs_qual.end());
    }

    if (error_code != 0) {
        Py_DECREF(depth_arr);
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_XDECREF(variant_types_arr);
        Py_DECREF(path_seq);
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }

    const npy_intp n_reads_dim[1] = {n_samples};
    const npy_intp sample_offsets_dim[1] = {n_samples + 1};
    const npy_intp centers_dim[1] = {static_cast<npy_intp>(all_centers.size())};
    const npy_intp obs_offsets_dim[1] = {static_cast<npy_intp>(all_obs_offsets.size())};
    const npy_intp obs_dim[1] = {static_cast<npy_intp>(all_obs_pos.size())};
    PyObject* n_reads_arr = PyArray_SimpleNew(1, n_reads_dim, NPY_INT32);
    PyObject* sample_offsets_arr = PyArray_SimpleNew(1, sample_offsets_dim, NPY_INT64);
    PyObject* centers_arr = PyArray_SimpleNew(1, centers_dim, NPY_INT32);
    PyObject* obs_offsets_arr = PyArray_SimpleNew(1, obs_offsets_dim, NPY_INT64);
    PyObject* obs_pos_arr = PyArray_SimpleNew(1, obs_dim, NPY_INT32);
    PyObject* obs_code_arr = PyArray_SimpleNew(1, obs_dim, NPY_INT8);
    PyObject* obs_qual_arr = PyArray_SimpleNew(1, obs_dim, NPY_UINT8);
    if (n_reads_arr == nullptr || sample_offsets_arr == nullptr || centers_arr == nullptr || obs_offsets_arr == nullptr ||
        obs_pos_arr == nullptr || obs_code_arr == nullptr || obs_qual_arr == nullptr) {
        Py_XDECREF(n_reads_arr);
        Py_XDECREF(sample_offsets_arr);
        Py_XDECREF(centers_arr);
        Py_XDECREF(obs_offsets_arr);
        Py_XDECREF(obs_pos_arr);
        Py_XDECREF(obs_code_arr);
        Py_XDECREF(obs_qual_arr);
        Py_DECREF(depth_arr);
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_XDECREF(variant_types_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    if (!n_reads_vec.empty()) {
        std::copy(
            n_reads_vec.begin(),
            n_reads_vec.end(),
            reinterpret_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(n_reads_arr)))
        );
    }
    if (!sample_offsets.empty()) {
        std::copy(
            sample_offsets.begin(),
            sample_offsets.end(),
            reinterpret_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(sample_offsets_arr)))
        );
    }
    if (!all_centers.empty()) {
        std::copy(
            all_centers.begin(),
            all_centers.end(),
            reinterpret_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(centers_arr)))
        );
    }
    if (!all_obs_offsets.empty()) {
        std::copy(
            all_obs_offsets.begin(),
            all_obs_offsets.end(),
            reinterpret_cast<int64_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_offsets_arr)))
        );
    }
    if (!all_obs_pos.empty()) {
        std::copy(
            all_obs_pos.begin(),
            all_obs_pos.end(),
            reinterpret_cast<int32_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_pos_arr)))
        );
        std::copy(
            all_obs_code.begin(),
            all_obs_code.end(),
            reinterpret_cast<int8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_code_arr)))
        );
        std::copy(
            all_obs_qual.begin(),
            all_obs_qual.end(),
            reinterpret_cast<uint8_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(obs_qual_arr)))
        );
    }

    PyObject* out = PyTuple_New(14);
    if (out == nullptr) {
        Py_DECREF(depth_arr);
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        Py_DECREF(n_reads_arr);
        Py_DECREF(sample_offsets_arr);
        Py_DECREF(centers_arr);
        Py_DECREF(obs_offsets_arr);
        Py_DECREF(obs_pos_arr);
        Py_DECREF(obs_code_arr);
        Py_DECREF(obs_qual_arr);
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_XDECREF(variant_types_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    PyTuple_SET_ITEM(out, 0, depth_arr);
    PyTuple_SET_ITEM(out, 1, ref_count_arr);
    PyTuple_SET_ITEM(out, 2, alt_count_arr);
    PyTuple_SET_ITEM(out, 3, other_count_arr);
    PyTuple_SET_ITEM(out, 4, ref_weight_arr);
    PyTuple_SET_ITEM(out, 5, alt_weight_arr);
    PyTuple_SET_ITEM(out, 6, other_weight_arr);
    PyTuple_SET_ITEM(out, 7, n_reads_arr);
    PyTuple_SET_ITEM(out, 8, sample_offsets_arr);
    PyTuple_SET_ITEM(out, 9, centers_arr);
    PyTuple_SET_ITEM(out, 10, obs_offsets_arr);
    PyTuple_SET_ITEM(out, 11, obs_pos_arr);
    PyTuple_SET_ITEM(out, 12, obs_code_arr);
    PyTuple_SET_ITEM(out, 13, obs_qual_arr);

    Py_DECREF(lookup_arr);
    Py_DECREF(ref_arr);
    Py_DECREF(alt_arr);
    Py_DECREF(positions_arr);
    Py_XDECREF(variant_types_arr);
    Py_DECREF(path_seq);
    return out;
}

static PyObject* materialize_dense_from_fragments(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    long long n_samples_ll = 0;
    long long n_positions_ll = 0;
    PyObject* sample_offsets_obj = nullptr;
    PyObject* center_idx_obj = nullptr;
    PyObject* obs_offsets_obj = nullptr;
    PyObject* obs_pos_idx_obj = nullptr;
    PyObject* obs_code_obj = nullptr;
    PyObject* obs_qual_obj = nullptr;
    int support_only = 0;

    static const char* kwlist[] = {
        "n_samples",
        "n_positions",
        "fragment_sample_offsets",
        "fragment_center_idx",
        "fragment_obs_offsets",
        "fragment_obs_pos_idx",
        "fragment_obs_code",
        "fragment_obs_qual",
        "support_only",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "LLOOOOOO|p",
            const_cast<char**>(kwlist),
            &n_samples_ll,
            &n_positions_ll,
            &sample_offsets_obj,
            &center_idx_obj,
            &obs_offsets_obj,
            &obs_pos_idx_obj,
            &obs_code_obj,
            &obs_qual_obj,
            &support_only)) {
        return nullptr;
    }
    if (n_samples_ll < 0 || n_positions_ll < 0) {
        PyErr_SetString(PyExc_ValueError, "n_samples and n_positions must be non-negative.");
        return nullptr;
    }
    const npy_intp n_samples = static_cast<npy_intp>(n_samples_ll);
    const npy_intp n_positions = static_cast<npy_intp>(n_positions_ll);

    PyArrayObject* sample_offsets_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(sample_offsets_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* center_idx_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(center_idx_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* obs_offsets_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obs_offsets_obj, NPY_INT64, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* obs_pos_idx_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obs_pos_idx_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* obs_code_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obs_code_obj, NPY_INT8, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* obs_qual_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(obs_qual_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    if (
        sample_offsets_arr == nullptr ||
        center_idx_arr == nullptr ||
        obs_offsets_arr == nullptr ||
        obs_pos_idx_arr == nullptr ||
        obs_code_arr == nullptr ||
        obs_qual_arr == nullptr
    ) {
        Py_XDECREF(sample_offsets_arr);
        Py_XDECREF(center_idx_arr);
        Py_XDECREF(obs_offsets_arr);
        Py_XDECREF(obs_pos_idx_arr);
        Py_XDECREF(obs_code_arr);
        Py_XDECREF(obs_qual_arr);
        return nullptr;
    }
    if (
        PyArray_NDIM(sample_offsets_arr) != 1 ||
        PyArray_NDIM(center_idx_arr) != 1 ||
        PyArray_NDIM(obs_offsets_arr) != 1 ||
        PyArray_NDIM(obs_pos_idx_arr) != 1 ||
        PyArray_NDIM(obs_code_arr) != 1 ||
        PyArray_NDIM(obs_qual_arr) != 1
    ) {
        PyErr_SetString(PyExc_ValueError, "fragment arrays must be one-dimensional.");
        Py_DECREF(sample_offsets_arr);
        Py_DECREF(center_idx_arr);
        Py_DECREF(obs_offsets_arr);
        Py_DECREF(obs_pos_idx_arr);
        Py_DECREF(obs_code_arr);
        Py_DECREF(obs_qual_arr);
        return nullptr;
    }
    const npy_intp n_fragments = PyArray_SIZE(center_idx_arr);
    const npy_intp n_observations = PyArray_SIZE(obs_pos_idx_arr);
    if (
        PyArray_SIZE(sample_offsets_arr) != n_samples + 1 ||
        PyArray_SIZE(obs_offsets_arr) != n_fragments + 1 ||
        PyArray_SIZE(obs_code_arr) != n_observations ||
        PyArray_SIZE(obs_qual_arr) != n_observations
    ) {
        PyErr_SetString(PyExc_ValueError, "fragment array lengths are inconsistent.");
        Py_DECREF(sample_offsets_arr);
        Py_DECREF(center_idx_arr);
        Py_DECREF(obs_offsets_arr);
        Py_DECREF(obs_pos_idx_arr);
        Py_DECREF(obs_code_arr);
        Py_DECREF(obs_qual_arr);
        return nullptr;
    }

    const auto* sample_offsets = reinterpret_cast<int64_t*>(PyArray_DATA(sample_offsets_arr));
    const auto* obs_offsets = reinterpret_cast<int64_t*>(PyArray_DATA(obs_offsets_arr));
    const auto* obs_pos_idx = reinterpret_cast<int32_t*>(PyArray_DATA(obs_pos_idx_arr));
    const auto* obs_code = reinterpret_cast<int8_t*>(PyArray_DATA(obs_code_arr));
    const auto* obs_qual = reinterpret_cast<uint8_t*>(PyArray_DATA(obs_qual_arr));

    if (
        (n_samples >= 0 && sample_offsets[0] != 0) ||
        (n_fragments >= 0 && obs_offsets[0] != 0) ||
        sample_offsets[n_samples] != static_cast<int64_t>(n_fragments) ||
        obs_offsets[n_fragments] != static_cast<int64_t>(n_observations)
    ) {
        PyErr_SetString(PyExc_ValueError, "fragment offsets are inconsistent.");
        Py_DECREF(sample_offsets_arr);
        Py_DECREF(center_idx_arr);
        Py_DECREF(obs_offsets_arr);
        Py_DECREF(obs_pos_idx_arr);
        Py_DECREF(obs_code_arr);
        Py_DECREF(obs_qual_arr);
        return nullptr;
    }

    const npy_intp dims[2] = {n_samples, n_positions};
    PyObject* ref_count_arr = PyArray_ZEROS(2, dims, NPY_UINT16, 0);
    PyObject* alt_count_arr = PyArray_ZEROS(2, dims, NPY_UINT16, 0);
    PyObject* other_count_arr = PyArray_ZEROS(2, dims, NPY_UINT16, 0);
    PyObject* depth_arr = PyArray_ZEROS(2, dims, NPY_UINT16, 0);
    PyObject* ref_weight_arr = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    PyObject* alt_weight_arr = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    PyObject* other_weight_arr = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (
        ref_count_arr == nullptr ||
        alt_count_arr == nullptr ||
        other_count_arr == nullptr ||
        depth_arr == nullptr ||
        ref_weight_arr == nullptr ||
        alt_weight_arr == nullptr ||
        other_weight_arr == nullptr
    ) {
        Py_XDECREF(ref_count_arr);
        Py_XDECREF(alt_count_arr);
        Py_XDECREF(other_count_arr);
        Py_XDECREF(depth_arr);
        Py_XDECREF(ref_weight_arr);
        Py_XDECREF(alt_weight_arr);
        Py_XDECREF(other_weight_arr);
        Py_DECREF(sample_offsets_arr);
        Py_DECREF(center_idx_arr);
        Py_DECREF(obs_offsets_arr);
        Py_DECREF(obs_pos_idx_arr);
        Py_DECREF(obs_code_arr);
        Py_DECREF(obs_qual_arr);
        return nullptr;
    }

    auto* ref_count = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ref_count_arr)));
    auto* alt_count = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(alt_count_arr)));
    auto* other_count = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(other_count_arr)));
    auto* depth = reinterpret_cast<uint16_t*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(depth_arr)));
    auto* ref_weight = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(ref_weight_arr)));
    auto* alt_weight = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(alt_weight_arr)));
    auto* other_weight = reinterpret_cast<float*>(PyArray_DATA(reinterpret_cast<PyArrayObject*>(other_weight_arr)));

    int error_code = 0;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp sample_idx = 0; sample_idx < n_samples && error_code == 0; ++sample_idx) {
        const int64_t frag_start = sample_offsets[sample_idx];
        const int64_t frag_stop = sample_offsets[sample_idx + 1];
        if (frag_start < 0 || frag_stop < frag_start || frag_stop > static_cast<int64_t>(n_fragments)) {
            error_code = 1;
            break;
        }
        const npy_intp row_offset = sample_idx * n_positions;
        for (int64_t frag_idx = frag_start; frag_idx < frag_stop; ++frag_idx) {
            const int64_t obs_start = obs_offsets[frag_idx];
            const int64_t obs_stop = obs_offsets[frag_idx + 1];
            if (obs_start < 0 || obs_stop < obs_start || obs_stop > static_cast<int64_t>(n_observations)) {
                error_code = 1;
                break;
            }
            for (int64_t obs_idx = obs_start; obs_idx < obs_stop; ++obs_idx) {
                const int32_t p = obs_pos_idx[obs_idx];
                if (p < 0 || p >= n_positions) {
                    error_code = 2;
                    break;
                }
                const npy_intp cell = row_offset + static_cast<npy_intp>(p);
                if (support_only != 0) {
                    depth[cell] = 1;
                    continue;
                }
                saturating_increment_uint16(depth[cell]);
                const float weight = phred_weight(obs_qual[obs_idx]);
                if (obs_code[obs_idx] == 0) {
                    saturating_increment_uint16(ref_count[cell]);
                    ref_weight[cell] += weight;
                } else if (obs_code[obs_idx] == 1) {
                    saturating_increment_uint16(alt_count[cell]);
                    alt_weight[cell] += weight;
                } else {
                    saturating_increment_uint16(other_count[cell]);
                    other_weight[cell] += weight;
                }
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(sample_offsets_arr);
    Py_DECREF(center_idx_arr);
    Py_DECREF(obs_offsets_arr);
    Py_DECREF(obs_pos_idx_arr);
    Py_DECREF(obs_code_arr);
    Py_DECREF(obs_qual_arr);

    if (error_code != 0) {
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(depth_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        if (error_code == 2) {
            PyErr_SetString(PyExc_ValueError, "fragment observation position index is out of bounds.");
        } else {
            PyErr_SetString(PyExc_ValueError, "fragment offsets are inconsistent.");
        }
        return nullptr;
    }

    PyObject* out = PyTuple_New(7);
    if (out == nullptr) {
        Py_DECREF(ref_count_arr);
        Py_DECREF(alt_count_arr);
        Py_DECREF(other_count_arr);
        Py_DECREF(depth_arr);
        Py_DECREF(ref_weight_arr);
        Py_DECREF(alt_weight_arr);
        Py_DECREF(other_weight_arr);
        return nullptr;
    }
    PyTuple_SET_ITEM(out, 0, ref_count_arr);
    PyTuple_SET_ITEM(out, 1, alt_count_arr);
    PyTuple_SET_ITEM(out, 2, other_count_arr);
    PyTuple_SET_ITEM(out, 3, depth_arr);
    PyTuple_SET_ITEM(out, 4, ref_weight_arr);
    PyTuple_SET_ITEM(out, 5, alt_weight_arr);
    PyTuple_SET_ITEM(out, 6, other_weight_arr);
    return out;
}

static PyObject* discover_snp_candidates(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    PyObject* bam_paths_obj = nullptr;
    const char* reference_fasta = nullptr;
    const char* chromosome = nullptr;
    long long region_start_ll = 0;
    long long region_stop_ll = 0;
    int min_base_quality = 13;
    int min_mapping_quality = 20;
    int hts_threads = 1;
    int max_insert_size = 0;
    int cap_base_quality_by_mapping_quality = 0;
    int min_depth = 3;
    int min_alt_count = 2;
    int min_alt_samples = 1;
    double min_alt_fraction = 0.05;
    double max_other_fraction = 0.20;

    static const char* kwlist[] = {
        "bam_paths",
        "reference_fasta",
        "chromosome",
        "region_start",
        "region_stop",
        "min_base_quality",
        "min_mapping_quality",
        "hts_threads",
        "max_insert_size",
        "cap_base_quality_by_mapping_quality",
        "min_depth",
        "min_alt_count",
        "min_alt_samples",
        "min_alt_fraction",
        "max_other_fraction",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OssLL|iiiiiiiidd",
            const_cast<char**>(kwlist),
            &bam_paths_obj,
            &reference_fasta,
            &chromosome,
            &region_start_ll,
            &region_stop_ll,
            &min_base_quality,
            &min_mapping_quality,
            &hts_threads,
            &max_insert_size,
            &cap_base_quality_by_mapping_quality,
            &min_depth,
            &min_alt_count,
            &min_alt_samples,
            &min_alt_fraction,
            &max_other_fraction)) {
        return nullptr;
    }
    const int64_t region_start = static_cast<int64_t>(region_start_ll);
    const int64_t region_stop = static_cast<int64_t>(region_stop_ll);
    if (region_start < 0 || region_stop <= region_start) {
        PyErr_SetString(PyExc_ValueError, "region_start must be >= 0 and region_stop must be greater than region_start.");
        return nullptr;
    }
    if (region_start > static_cast<int64_t>(std::numeric_limits<int>::max()) ||
        (region_stop - 1) > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        PyErr_SetString(PyExc_ValueError, "discover_snp_candidates currently requires chunk coordinates within int32 FASTA limits.");
        return nullptr;
    }

    PyObject* path_seq = PySequence_Fast(bam_paths_obj, "bam_paths must be a sequence of path strings.");
    if (path_seq == nullptr) {
        return nullptr;
    }
    const Py_ssize_t n_paths_py = PySequence_Fast_GET_SIZE(path_seq);
    std::vector<std::string> bam_paths;
    bam_paths.reserve(static_cast<size_t>(std::max<Py_ssize_t>(n_paths_py, 0)));
    for (Py_ssize_t i = 0; i < n_paths_py; ++i) {
        PyObject* item = PySequence_Fast_GET_ITEM(path_seq, i);
        PyObject* item_str = PyUnicode_FromObject(item);
        if (item_str == nullptr) {
            Py_DECREF(path_seq);
            return nullptr;
        }
        const char* path_c = PyUnicode_AsUTF8(item_str);
        if (path_c == nullptr) {
            Py_DECREF(item_str);
            Py_DECREF(path_seq);
            return nullptr;
        }
        if (path_c[0] != '\0') {
            bam_paths.emplace_back(path_c);
        }
        Py_DECREF(item_str);
    }
    if (bam_paths.empty()) {
        Py_DECREF(path_seq);
        PyErr_SetString(PyExc_ValueError, "No BAM/CRAM paths were provided for discovery.");
        return nullptr;
    }

    faidx_t* fai = fai_load(reference_fasta);
    if (fai == nullptr) {
        Py_DECREF(path_seq);
        PyErr_SetString(PyExc_RuntimeError, "Failed to load FASTA index. Run samtools faidx/pysam.faidx first or check --reference-fasta.");
        return nullptr;
    }
    int ref_len = 0;
    char* ref_seq = faidx_fetch_seq(
        fai,
        chromosome,
        static_cast<int>(region_start),
        static_cast<int>(region_stop - 1),
        &ref_len
    );
    if (ref_seq == nullptr || ref_len != static_cast<int>(region_stop - region_start)) {
        if (ref_seq != nullptr) {
            std::free(ref_seq);
        }
        fai_destroy(fai);
        Py_DECREF(path_seq);
        PyErr_SetString(PyExc_RuntimeError, "Failed to fetch the requested reference FASTA interval.");
        return nullptr;
    }

    DiscoveryAccumulator acc(region_start, region_stop);
    int64_t n_reads_seen = 0;
    int64_t n_bases_seen = 0;
    std::string error_message;
    for (const std::string& bam_path : bam_paths) {
        if (!discover_one_sample_snps(
                bam_path.c_str(),
                chromosome,
                region_start,
                region_stop,
                ref_seq,
                acc,
                min_base_quality,
                min_mapping_quality,
                hts_threads,
                max_insert_size,
                cap_base_quality_by_mapping_quality != 0,
                n_reads_seen,
                n_bases_seen,
                error_message)) {
            std::free(ref_seq);
            fai_destroy(fai);
            Py_DECREF(path_seq);
            PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
            return nullptr;
        }
    }

    std::vector<int64_t> out_pos;
    std::vector<uint8_t> out_ref;
    std::vector<uint8_t> out_alt;
    std::vector<uint32_t> out_depth;
    std::vector<uint32_t> out_ref_count;
    std::vector<uint32_t> out_alt_count;
    std::vector<uint32_t> out_other_count;
    std::vector<uint32_t> out_a_count;
    std::vector<uint32_t> out_c_count;
    std::vector<uint32_t> out_g_count;
    std::vector<uint32_t> out_t_count;
    std::vector<uint32_t> out_sample_support;
    std::vector<uint32_t> out_alt_forward;
    std::vector<uint32_t> out_alt_reverse;

    for (size_t offset = 0; offset < acc.region_len; ++offset) {
        const int ref_idx = base_index_from_ascii(static_cast<uint8_t>(ref_seq[offset]));
        if (ref_idx < 0) {
            continue;
        }
        const uint32_t a = acc.a_count[offset];
        const uint32_t c = acc.c_count[offset];
        const uint32_t g = acc.g_count[offset];
        const uint32_t t = acc.t_count[offset];
        const uint32_t other = acc.other_count[offset];
        const uint32_t acgt_depth = a + c + g + t;
        const uint32_t total_depth = acgt_depth + other;
        if (total_depth < static_cast<uint32_t>(std::max(min_depth, 0))) {
            continue;
        }
        int alt_idx = -1;
        uint32_t best_alt_count = 0U;
        for (int base_idx = 0; base_idx < 4; ++base_idx) {
            if (base_idx == ref_idx) {
                continue;
            }
            const uint32_t count = acc.count_for_base(offset, base_idx);
            if (count > best_alt_count) {
                best_alt_count = count;
                alt_idx = base_idx;
            }
        }
        if (alt_idx < 0 || best_alt_count < static_cast<uint32_t>(std::max(min_alt_count, 0))) {
            continue;
        }
        if (acc.sample_support[offset] < static_cast<uint32_t>(std::max(min_alt_samples, 0))) {
            continue;
        }
        const double alt_fraction = static_cast<double>(best_alt_count) / std::max<double>(static_cast<double>(total_depth), 1.0);
        const double other_fraction = static_cast<double>(other) / std::max<double>(static_cast<double>(total_depth), 1.0);
        if (alt_fraction < min_alt_fraction || other_fraction > max_other_fraction) {
            continue;
        }
        out_pos.push_back(region_start + static_cast<int64_t>(offset) + 1);
        out_ref.push_back(ascii_from_base_index(ref_idx));
        out_alt.push_back(ascii_from_base_index(alt_idx));
        out_depth.push_back(total_depth);
        out_ref_count.push_back(acc.count_for_base(offset, ref_idx));
        out_alt_count.push_back(best_alt_count);
        out_other_count.push_back(other);
        out_a_count.push_back(a);
        out_c_count.push_back(c);
        out_g_count.push_back(g);
        out_t_count.push_back(t);
        out_sample_support.push_back(acc.sample_support[offset]);
        out_alt_forward.push_back(acc.forward_for_base(offset, alt_idx));
        out_alt_reverse.push_back(acc.reverse_for_base(offset, alt_idx));
    }

    std::free(ref_seq);
    fai_destroy(fai);
    Py_DECREF(path_seq);

    const npy_intp dims[1] = {static_cast<npy_intp>(out_pos.size())};
    auto make_array = [&](const auto& vec, int typenum) -> PyObject* {
        PyObject* arr = PyArray_SimpleNew(1, dims, typenum);
        if (arr == nullptr) {
            return nullptr;
        }
        if (!vec.empty()) {
            std::memcpy(PyArray_DATA(reinterpret_cast<PyArrayObject*>(arr)), vec.data(), vec.size() * sizeof(typename std::decay<decltype(vec)>::type::value_type));
        }
        return arr;
    };

    PyObject* out = PyDict_New();
    if (out == nullptr) {
        return nullptr;
    }
    auto set_item = [&](const char* key, PyObject* value) -> bool {
        if (value == nullptr) {
            return false;
        }
        const int rc = PyDict_SetItemString(out, key, value);
        Py_DECREF(value);
        return rc == 0;
    };
    bool ok = true;
    ok = ok && set_item("position", make_array(out_pos, NPY_INT64));
    ok = ok && set_item("ref_code", make_array(out_ref, NPY_UINT8));
    ok = ok && set_item("alt_code", make_array(out_alt, NPY_UINT8));
    ok = ok && set_item("depth", make_array(out_depth, NPY_UINT32));
    ok = ok && set_item("ref_count", make_array(out_ref_count, NPY_UINT32));
    ok = ok && set_item("alt_count", make_array(out_alt_count, NPY_UINT32));
    ok = ok && set_item("other_count", make_array(out_other_count, NPY_UINT32));
    ok = ok && set_item("a_count", make_array(out_a_count, NPY_UINT32));
    ok = ok && set_item("c_count", make_array(out_c_count, NPY_UINT32));
    ok = ok && set_item("g_count", make_array(out_g_count, NPY_UINT32));
    ok = ok && set_item("t_count", make_array(out_t_count, NPY_UINT32));
    ok = ok && set_item("sample_support", make_array(out_sample_support, NPY_UINT32));
    ok = ok && set_item("alt_forward_count", make_array(out_alt_forward, NPY_UINT32));
    ok = ok && set_item("alt_reverse_count", make_array(out_alt_reverse, NPY_UINT32));
    ok = ok && set_item("n_bams", PyLong_FromLong(static_cast<long>(bam_paths.size())));
    ok = ok && set_item("n_reads_seen", PyLong_FromLongLong(n_reads_seen));
    ok = ok && set_item("n_bases_seen", PyLong_FromLongLong(n_bases_seen));
    ok = ok && set_item("region_start", PyLong_FromLongLong(region_start));
    ok = ok && set_item("region_stop", PyLong_FromLongLong(region_stop));
    if (!ok) {
        Py_DECREF(out);
        return nullptr;
    }
    return out;
}

PyMethodDef module_methods[] = {
    {
        "extract_sample_read_stream",
        reinterpret_cast<PyCFunction>(extract_sample_read_stream),
        METH_VARARGS | METH_KEYWORDS,
        "HTSlib-backed sample-level read-stream extractor.",
    },
    {
        "extract_samples_read_stream_batch",
        reinterpret_cast<PyCFunction>(extract_samples_read_stream_batch),
        METH_VARARGS | METH_KEYWORDS,
        "HTSlib-backed multi-sample read-stream extractor.",
    },
    {
        "materialize_dense_from_fragments",
        reinterpret_cast<PyCFunction>(materialize_dense_from_fragments),
        METH_VARARGS | METH_KEYWORDS,
        "Materialize dense count and weight arrays from compact STITCHV2 fragment evidence.",
    },
    {
        "discover_snp_candidates",
        reinterpret_cast<PyCFunction>(discover_snp_candidates),
        METH_VARARGS | METH_KEYWORDS,
        "HTSlib-backed CIGAR plus FASTA SNP candidate discovery.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_htslib_readstream",
    "Compiled HTSlib-backed read-stream extractor for STITCHV2.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__htslib_readstream(void) {
    import_array();
    return PyModule_Create(&module_def);
}
