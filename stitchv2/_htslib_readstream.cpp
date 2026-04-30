#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <htslib/hts.h>
#include <htslib/sam.h>

#include <algorithm>
#include <cmath>
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
    out_centers.push_back(compact_pos[compact_pos.size() / 2]);
    out_obs_pos.insert(out_obs_pos.end(), compact_pos.begin(), compact_pos.end());
    out_obs_code.insert(out_obs_code.end(), compact_code.begin(), compact_code.end());
    out_obs_qual.insert(out_obs_qual.end(), compact_qual.begin(), compact_qual.end());
    out_obs_offsets.push_back(static_cast<int64_t>(out_obs_pos.size()));
}

static PyObject* extract_sample_read_stream(PyObject* /*self*/, PyObject* args, PyObject* kwargs) {
    const char* bam_path = nullptr;
    const char* chromosome = nullptr;
    long region_start = 0;  // 0-based inclusive
    long region_stop = 0;   // 0-based exclusive upper bound represented as 1-based terminal in python caller
    PyObject* lookup_obj = nullptr;
    PyObject* ref_obj = nullptr;
    PyObject* alt_obj = nullptr;
    int min_base_quality = 13;
    int min_mapping_quality = 20;
    int merge_fragments_by_query = 1;
    int hts_threads = 1;

    static const char* kwlist[] = {
        "bam_path",
        "chromosome",
        "region_start",
        "region_stop",
        "lookup",
        "ref_codes",
        "alt_codes",
        "min_base_quality",
        "min_mapping_quality",
        "merge_fragments_by_query",
        "hts_threads",
        nullptr,
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "ssllOOOii|pi",
            const_cast<char**>(kwlist),
            &bam_path,
            &chromosome,
            &region_start,
            &region_stop,
            &lookup_obj,
            &ref_obj,
            &alt_obj,
            &min_base_quality,
            &min_mapping_quality,
            &merge_fragments_by_query,
            &hts_threads)) {
        return nullptr;
    }

    PyArrayObject* lookup_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(lookup_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* ref_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(ref_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    PyArrayObject* alt_arr = reinterpret_cast<PyArrayObject*>(PyArray_FROM_OTF(alt_obj, NPY_UINT8, NPY_ARRAY_IN_ARRAY));
    if (lookup_arr == nullptr || ref_arr == nullptr || alt_arr == nullptr) {
        Py_XDECREF(lookup_arr);
        Py_XDECREF(ref_arr);
        Py_XDECREF(alt_arr);
        return nullptr;
    }
    if (PyArray_NDIM(lookup_arr) != 1 || PyArray_NDIM(ref_arr) != 1 || PyArray_NDIM(alt_arr) != 1) {
        PyErr_SetString(PyExc_ValueError, "lookup/ref_codes/alt_codes must be 1D arrays.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        return nullptr;
    }

    const npy_intp n_positions = PyArray_SIZE(ref_arr);
    if (PyArray_SIZE(alt_arr) != n_positions) {
        PyErr_SetString(PyExc_ValueError, "ref_codes and alt_codes must have same length.");
        Py_DECREF(lookup_arr);
        Py_DECREF(ref_arr);
        Py_DECREF(alt_arr);
        return nullptr;
    }

    const int32_t* lookup = reinterpret_cast<int32_t*>(PyArray_DATA(lookup_arr));
    const uint8_t* ref_codes = reinterpret_cast<uint8_t*>(PyArray_DATA(ref_arr));
    const uint8_t* alt_codes = reinterpret_cast<uint8_t*>(PyArray_DATA(alt_arr));
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

    while (error_code == 0 && sam_itr_next(fp, itr, rec) >= 0) {
        const uint16_t flag = rec->core.flag;
        if ((flag & BAM_FUNMAP) != 0 || (flag & BAM_FSECONDARY) != 0 || (flag & BAM_FSUPPLEMENTARY) != 0 || (flag & BAM_FDUP) != 0) {
            continue;
        }
        if (rec->core.qual < min_mapping_quality) {
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
        for (uint32_t i = 0; i < rec->core.n_cigar; ++i) {
            const int op = bam_cigar_op(cigar[i]);
            const int oplen = bam_cigar_oplen(cigar[i]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                for (int j = 0; j < oplen; ++j) {
                    const int64_t rel = static_cast<int64_t>(ref_pos) - static_cast<int64_t>(region_start);
                    if (rel >= 0 && rel < lookup_len) {
                        const int32_t target_idx = lookup[rel];
                        if (target_idx >= 0) {
                            const uint8_t q = qual[qpos];
                            if (q >= min_base_quality) {
                                const uint8_t obs_base = seq_base_ascii_upper(seq, qpos);
                                int8_t code = 2;
                                if (obs_base == ref_codes[target_idx]) {
                                    code = 0;
                                } else if (obs_base == alt_codes[target_idx]) {
                                    code = 1;
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
    return out;
}

PyMethodDef module_methods[] = {
    {
        "extract_sample_read_stream",
        reinterpret_cast<PyCFunction>(extract_sample_read_stream),
        METH_VARARGS | METH_KEYWORDS,
        "HTSlib-backed sample-level read-stream extractor.",
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
