#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <htslib/hts.h>
#include <htslib/sam.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
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
    npy_intp n_positions,
    int min_base_quality,
    int min_mapping_quality,
    bool do_merge,
    int hts_threads,
    bool snp_only_bamreader,
    bool stitch_style_bamreader,
    int max_insert_size,
    bool cap_base_quality_by_mapping_quality,
    bool ref_alt_only,
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
        if (stitch_style_bamreader) {
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

        if (do_merge && (flag & BAM_FPAIRED) != 0) {
            const std::string qname(bam_get_qname(rec));
            auto it = pending.find(qname);
            if (it == pending.end()) {
                PendingFragment frag;
                frag.pos = read_pos;
                frag.code = read_code;
                frag.qual = read_qual;
                pending.emplace(qname, std::move(frag));
            } else {
                PendingFragment merged;
                merged.pos.reserve(it->second.pos.size() + read_pos.size());
                merged.code.reserve(it->second.code.size() + read_code.size());
                merged.qual.reserve(it->second.qual.size() + read_qual.size());
                merged.pos.insert(merged.pos.end(), it->second.pos.begin(), it->second.pos.end());
                merged.code.insert(merged.code.end(), it->second.code.begin(), it->second.code.end());
                merged.qual.insert(merged.qual.end(), it->second.qual.begin(), it->second.qual.end());
                merged.pos.insert(merged.pos.end(), read_pos.begin(), read_pos.end());
                merged.code.insert(merged.code.end(), read_code.begin(), read_code.end());
                merged.qual.insert(merged.qual.end(), read_qual.begin(), read_qual.end());
                append_fragment_compacted(
                    merged.pos,
                    merged.code,
                    merged.qual,
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
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "ssllOOOOii|piiiiii",
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
            &ref_alt_only)) {
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

        if (do_merge && (flag & BAM_FPAIRED) != 0) {
            const std::string qname(bam_get_qname(rec));
            auto it = pending.find(qname);
            if (it == pending.end()) {
                PendingFragment frag;
                frag.pos = read_pos;
                frag.code = read_code;
                frag.qual = read_qual;
                pending.emplace(qname, std::move(frag));
            } else {
                PendingFragment merged;
                merged.pos.reserve(it->second.pos.size() + read_pos.size());
                merged.code.reserve(it->second.code.size() + read_code.size());
                merged.qual.reserve(it->second.qual.size() + read_qual.size());
                merged.pos.insert(merged.pos.end(), it->second.pos.begin(), it->second.pos.end());
                merged.code.insert(merged.code.end(), it->second.code.begin(), it->second.code.end());
                merged.qual.insert(merged.qual.end(), it->second.qual.begin(), it->second.qual.end());
                merged.pos.insert(merged.pos.end(), read_pos.begin(), read_pos.end());
                merged.code.insert(merged.code.end(), read_code.begin(), read_code.end());
                merged.qual.insert(merged.qual.end(), read_qual.begin(), read_qual.end());
                append_fragment_compacted(
                    merged.pos,
                    merged.code,
                    merged.qual,
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
    int min_base_quality = 13;
    int min_mapping_quality = 20;
    int merge_fragments_by_query = 1;
    int hts_threads = 1;
    int snp_only_bamreader = 0;
    int stitch_style_bamreader = 0;
    int max_insert_size = 0;
    int cap_base_quality_by_mapping_quality = 0;
    int ref_alt_only = 0;

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
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OsllOOOOii|piiiiii",
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
            &ref_alt_only)) {
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
    if (lookup_arr == nullptr || ref_arr == nullptr || alt_arr == nullptr || positions_arr == nullptr) {
        Py_XDECREF(lookup_arr);
        Py_XDECREF(ref_arr);
        Py_XDECREF(alt_arr);
        Py_XDECREF(positions_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    if (PyArray_NDIM(lookup_arr) != 1 || PyArray_NDIM(ref_arr) != 1 || PyArray_NDIM(alt_arr) != 1 || PyArray_NDIM(positions_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "lookup/ref_codes/alt_codes/positions must be 1D arrays.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_DECREF(path_seq);
        return nullptr;
    }
    const npy_intp n_positions = PyArray_SIZE(ref_arr);
    if (PyArray_SIZE(alt_arr) != n_positions || PyArray_SIZE(positions_arr) != n_positions) {
        PyErr_SetString(PyExc_ValueError, "ref_codes, alt_codes, and positions must have same length.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        Py_DECREF(positions_arr);
        Py_DECREF(path_seq);
        return nullptr;
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

    Py_BEGIN_ALLOW_THREADS
    for (npy_intp sample_idx = 0; sample_idx < n_samples; ++sample_idx) {
        ExtractedSample sample;
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
            n_positions,
            min_base_quality,
            min_mapping_quality,
            do_merge,
            threads,
            snp_only_bamreader != 0,
            stitch_style_bamreader != 0,
            max_insert_size,
            cap_base_quality_by_mapping_quality != 0,
            ref_alt_only != 0,
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
            error_code = 1;
            error_message = "sample " + std::to_string(sample_idx) + ": " + sample.error_message;
            break;
        }
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
    Py_END_ALLOW_THREADS

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
    Py_DECREF(path_seq);
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
