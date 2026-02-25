#include "IndexIVFJHQ.h"
#include "IndexIVFJHQScanner.h"

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

namespace faiss {

void IndexIVFJHQ::search(idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before search");
    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be positive");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of queries must be positive");

    const IVFJHQSearchParameters* jhq_params = dynamic_cast<const IVFJHQSearchParameters*>(params);

    size_t search_nprobe = this->nprobe;
    if (jhq_params && jhq_params->nprobe > 0) {
        search_nprobe = jhq_params->nprobe;
    }

    if (should_use_single_level_adapter()) {
        ensure_single_level_adapter_ready();
    }

    const size_t actual_nprobe = std::min(nlist, search_nprobe);
    FAISS_THROW_IF_NOT_MSG(actual_nprobe > 0, "nprobe must be positive");

    auto& workspace = get_search_workspace();
    workspace.ensure_capacity(n, actual_nprobe, d);

    idx_t* idx_buffer = workspace.list_indices.data();
    float* coarse_dis_buffer = workspace.coarse_distances.data();
    float* rotated_queries = workspace.rotated_queries.data();
    const size_t total_lists = static_cast<size_t>(n) * actual_nprobe;

    double t0 = getmillisecs();

    quantizer->search(n, x, actual_nprobe, coarse_dis_buffer, idx_buffer);

    double t1 = getmillisecs();

    if (jhq.use_jl_transform && jhq.is_rotation_trained) {
        jhq.apply_jl_rotation(n, x, rotated_queries);
    } else {
        std::memcpy(rotated_queries, x, sizeof(float) * n * d);
    }

    if (jhq.normalize_l2) {
        fvec_renorm_L2(static_cast<size_t>(d),
                       static_cast<size_t>(n),
                       rotated_queries);
    }

    invlists->prefetch_lists(idx_buffer, total_lists);

    const IVFSearchParameters* ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
    search_preassigned_with_rotated_queries(n,
        rotated_queries,
        k,
        idx_buffer,
        coarse_dis_buffer,
        distances,
        labels,
        false,
        ivf_params,
        &indexIVF_stats);

    double t2 = getmillisecs();

    indexIVF_stats.quantization_time += t1 - t0;
    indexIVF_stats.search_time += t2 - t0;
    indexIVFJHQ_stats.quantization_time += t1 - t0;
    indexIVFJHQ_stats.total_search_time += t2 - t0;
}

void IndexIVFJHQ::search_preassigned(idx_t n,
    const float* x,
    idx_t k,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    bool store_pairs,
    const IVFSearchParameters* params,
    IndexIVFStats* ivf_stats) const
{
    const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);
    bool is_max_heap = (metric_type == METRIC_INNER_PRODUCT);

    const IVFJHQSearchParameters* jhq_params =
        dynamic_cast<const IVFJHQSearchParameters*>(params);
    float oversampling = default_jhq_oversampling;
    bool use_early_termination_param = use_early_termination;
    bool compute_residuals_param = true;
    if (jhq_params) {
        if (jhq_params->jhq_oversampling_factor > 0) {
            oversampling = jhq_params->jhq_oversampling_factor;
        }
        use_early_termination_param = jhq_params->use_early_termination;
        compute_residuals_param = jhq_params->compute_residuals;
    }
    const float effective_oversampling = std::max(1.0f, oversampling);

    size_t total_ndis = 0;

    const bool adapter_active = should_use_single_level_adapter();
    const bool need_pre_transformed_queries = adapter_active &&
        ((jhq.use_jl_transform && jhq.is_rotation_trained) || jhq.normalize_l2);
    std::vector<float> transformed_queries_single_level;
    if (need_pre_transformed_queries) {
        transformed_queries_single_level.resize(static_cast<size_t>(n) * d);
        if (jhq.use_jl_transform && jhq.is_rotation_trained) {
            jhq.apply_jl_rotation(n, x, transformed_queries_single_level.data());
        } else {
            std::memcpy(transformed_queries_single_level.data(),
                        x,
                        sizeof(float) * static_cast<size_t>(n) * d);
        }
        if (jhq.normalize_l2) {
            fvec_renorm_L2(static_cast<size_t>(d),
                           static_cast<size_t>(n),
                           transformed_queries_single_level.data());
        }
    }
    const float* adapter_query_data = need_pre_transformed_queries
        ? transformed_queries_single_level.data()
        : x;

#pragma omp parallel if (n > 1) reduction(+ : total_ndis)
    {
        std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(store_pairs, nullptr, params));
        auto* jhq_scanner = dynamic_cast<IVFJHQScanner*>(scanner.get());

        std::vector<float> candidate_primary;
        std::vector<idx_t> candidate_locs;

#pragma omp for schedule(dynamic)
        for (idx_t i = 0; i < n; ++i) {
            float* simi = distances + i * k;
            idx_t* idxi = labels + i * k;

            if (is_max_heap) {
                heap_heapify<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_heapify<CMax<float, idx_t>>(k, simi, idxi);
            }

            if (jhq_scanner) {
                jhq_scanner->reset_for_reuse();
                jhq_scanner->set_query(x + i * d);
            } else {
                scanner->set_query(adapter_query_data + i * d);
            }

            size_t ndis_query = 0;

            const bool can_use_global_two_stage =
                jhq_scanner != nullptr &&
                use_early_termination_param &&
                compute_residuals_param &&
                (jhq.num_levels > 1) &&
                (effective_oversampling > 1.0f);

            size_t total_scanned = 0;
            size_t candidate_keep = 0;
            if (can_use_global_two_stage) {
                for (size_t ik = 0; ik < nprobe; ++ik) {
                    idx_t key = keys[i * nprobe + ik];
                    if (key < 0) {
                        continue;
                    }
                    total_scanned += invlists->list_size(key);
                }

                candidate_keep = std::max<size_t>(
                    static_cast<size_t>(k),
                    static_cast<size_t>(std::min<double>(
                        static_cast<double>(total_scanned),
                        std::ceil(static_cast<double>(k) * static_cast<double>(effective_oversampling)))));
            }

            const bool use_global_two_stage =
                can_use_global_two_stage && candidate_keep > 0 && candidate_keep < total_scanned;

            if (!use_global_two_stage) {
                for (size_t ik = 0; ik < nprobe; ik++) {
                    idx_t key = keys[i * nprobe + ik];
                    if (key < 0) {
                        continue;
                    }

                    scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                    size_t list_size = invlists->list_size(key);
                    if (list_size == 0) {
                        continue;
                    }

                    InvertedLists::ScopedCodes codes(invlists, key);
                    InvertedLists::ScopedIds ids(invlists, key);

                    scanner->scan_codes(
                        list_size, codes.get(), ids.get(), simi, idxi, k);
                    ndis_query += list_size;
                }
            } else {
                candidate_primary.resize(candidate_keep);
                candidate_locs.resize(candidate_keep);

                size_t heap_size = 0;
                bool heap_ready = false;
                const bool separated_storage_available =
                    jhq.has_pre_decoded_codes() &&
                    mapping_initialized.load(std::memory_order_acquire);

                for (size_t ik = 0; ik < nprobe; ++ik) {
                    idx_t key = keys[i * nprobe + ik];
                    if (key < 0) {
                        continue;
                    }

                    const size_t list_size = invlists->list_size(key);
                    if (list_size == 0) {
                        continue;
                    }

                    scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                    const float* primary = nullptr;
                    if (separated_storage_available) {
                        primary = jhq_scanner->compute_primary_distances_for_list(list_size, nullptr);
                    } else {
                        InvertedLists::ScopedCodes codes(invlists, key);
                        primary = jhq_scanner->compute_primary_distances_for_list(list_size, codes.get());
                    }

                    for (size_t j = 0; j < list_size; ++j) {
                        const float pd = primary[j];
                        const idx_t loc = lo_build(key, j);

                        if (!heap_ready) {
                            candidate_primary[heap_size] = pd;
                            candidate_locs[heap_size] = loc;
                            ++heap_size;
                            if (heap_size == candidate_keep) {
                                heap_heapify<CMax<float, idx_t>>(
                                    candidate_keep,
                                    candidate_primary.data(),
                                    candidate_locs.data());
                                heap_ready = true;
                            }
                            continue;
                        }

                        if (pd < candidate_primary[0]) {
                            heap_replace_top<CMax<float, idx_t>>(
                                candidate_keep,
                                candidate_primary.data(),
                                candidate_locs.data(),
                                pd,
                                loc);
                        }
                    }
                }

                ndis_query += total_scanned;

                const size_t candidate_count = heap_ready ? candidate_keep : heap_size;
                if (candidate_count > 0 && heap_ready) {
                    heap_reorder<CMax<float, idx_t>>(
                        candidate_count,
                        candidate_primary.data(),
                        candidate_locs.data());
                }

                for (size_t ci = 0; ci < candidate_count; ++ci) {
                    const idx_t loc = candidate_locs[ci];
                    const idx_t list_no = lo_listno(loc);
                    const idx_t offset = lo_offset(loc);
                    if (list_no < 0) {
                        continue;
                    }

                    const size_t list_size = invlists->list_size(list_no);
                    if (offset < 0 || static_cast<size_t>(offset) >= list_size) {
                        continue;
                    }

                    scanner->set_list(list_no,  0.0f);
                    InvertedLists::ScopedCodes codes(invlists, list_no);
                    InvertedLists::ScopedIds ids(invlists, list_no);

                    const uint8_t* packed_code =
                        codes.get() + static_cast<size_t>(offset) * code_size;
                    const float dis = jhq_scanner->refine_distance_from_primary_for_list(
                        static_cast<size_t>(offset),
                        packed_code,
                        candidate_primary[ci]);
                    const idx_t id = store_pairs ? loc : ids.get()[offset];

                    if (is_max_heap) {
                        if (CMin<float, idx_t>::cmp(simi[0], dis)) {
                            heap_replace_top<CMin<float, idx_t>>(
                                k, simi, idxi, dis, id);
                        }
                    } else {
                        if (CMax<float, idx_t>::cmp(simi[0], dis)) {
                            heap_replace_top<CMax<float, idx_t>>(
                                k, simi, idxi, dis, id);
                        }
                    }
                }
            }

            if (is_max_heap) {
                heap_reorder<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
            }

            total_ndis += ndis_query;
        }
    }

    if (ivf_stats) {
        ivf_stats->ndis += total_ndis;
    }

    indexIVFJHQ_stats.nq += n;
    indexIVFJHQ_stats.ndis += total_ndis;
}

void IndexIVFJHQ::search_preassigned_with_rotated_queries(
    idx_t n,
    const float* x_rotated,
    idx_t k,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    bool store_pairs,
    const IVFSearchParameters* params,
    IndexIVFStats* ivf_stats) const
{
    const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);
    bool is_max_heap = (metric_type == METRIC_INNER_PRODUCT);

    const IVFJHQSearchParameters* jhq_params =
        dynamic_cast<const IVFJHQSearchParameters*>(params);
    float oversampling = default_jhq_oversampling;
    bool use_early_termination_param = use_early_termination;
    bool compute_residuals_param = true;
    if (jhq_params) {
        if (jhq_params->jhq_oversampling_factor > 0) {
            oversampling = jhq_params->jhq_oversampling_factor;
        }
        use_early_termination_param = jhq_params->use_early_termination;
        compute_residuals_param = jhq_params->compute_residuals;
    }
    const float effective_oversampling = std::max(1.0f, oversampling);

    size_t total_ndis = 0;

#pragma omp parallel if (n > 1) reduction(+ : total_ndis)
    {
        std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(store_pairs, nullptr, params));
        auto* jhq_scanner = dynamic_cast<IVFJHQScanner*>(scanner.get());

        std::vector<float> candidate_primary;
        std::vector<idx_t> candidate_locs;

#pragma omp for schedule(dynamic)
        for (idx_t i = 0; i < n; ++i) {
            float* simi = distances + i * k;
            idx_t* idxi = labels + i * k;

            if (is_max_heap) {
                heap_heapify<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_heapify<CMax<float, idx_t>>(k, simi, idxi);
            }

            if (jhq_scanner) {
                jhq_scanner->reset_for_reuse();
                jhq_scanner->set_rotated_query(x_rotated + i * d);
            } else {
                scanner->set_query(x_rotated + i * d);
            }

            size_t ndis_query = 0;

            const bool can_use_global_two_stage =
                jhq_scanner != nullptr &&
                use_early_termination_param &&
                compute_residuals_param &&
                (jhq.num_levels > 1) &&
                (effective_oversampling > 1.0f);

            size_t total_scanned = 0;
            size_t candidate_keep = 0;
            if (can_use_global_two_stage) {
                for (size_t ik = 0; ik < nprobe; ++ik) {
                    idx_t key = keys[i * nprobe + ik];
                    if (key < 0) {
                        continue;
                    }
                    total_scanned += invlists->list_size(key);
                }

                candidate_keep = std::max<size_t>(
                    static_cast<size_t>(k),
                    static_cast<size_t>(std::min<double>(
                        static_cast<double>(total_scanned),
                        std::ceil(static_cast<double>(k) * static_cast<double>(effective_oversampling)))));
            }

            const bool use_global_two_stage =
                can_use_global_two_stage && candidate_keep > 0 && candidate_keep < total_scanned;

            if (!use_global_two_stage) {
                for (size_t ik = 0; ik < nprobe; ik++) {
                    idx_t key = keys[i * nprobe + ik];
                    if (key < 0) {
                        continue;
                    }

                    scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                    size_t list_size = invlists->list_size(key);
                    if (list_size == 0) {
                        continue;
                    }

                    InvertedLists::ScopedCodes codes(invlists, key);
                    InvertedLists::ScopedIds ids(invlists, key);

                    scanner->scan_codes(
                        list_size, codes.get(), ids.get(), simi, idxi, k);
                    ndis_query += list_size;
                }
            } else {
                candidate_primary.resize(candidate_keep);
                candidate_locs.resize(candidate_keep);

                size_t heap_size = 0;
                bool heap_ready = false;
                const bool separated_storage_available =
                    jhq.has_pre_decoded_codes() &&
                    mapping_initialized.load(std::memory_order_acquire);

                for (size_t ik = 0; ik < nprobe; ++ik) {
                    idx_t key = keys[i * nprobe + ik];
                    if (key < 0) {
                        continue;
                    }

                    const size_t list_size = invlists->list_size(key);
                    if (list_size == 0) {
                        continue;
                    }

                    scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                    const float* primary = nullptr;
                    if (separated_storage_available) {
                        primary = jhq_scanner->compute_primary_distances_for_list(list_size, nullptr);
                    } else {
                        InvertedLists::ScopedCodes codes(invlists, key);
                        primary = jhq_scanner->compute_primary_distances_for_list(list_size, codes.get());
                    }

                    for (size_t j = 0; j < list_size; ++j) {
                        const float pd = primary[j];
                        const idx_t loc = lo_build(key, j);

                        if (!heap_ready) {
                            candidate_primary[heap_size] = pd;
                            candidate_locs[heap_size] = loc;
                            ++heap_size;
                            if (heap_size == candidate_keep) {
                                heap_heapify<CMax<float, idx_t>>(
                                    candidate_keep,
                                    candidate_primary.data(),
                                    candidate_locs.data());
                                heap_ready = true;
                            }
                            continue;
                        }

                        if (pd < candidate_primary[0]) {
                            heap_replace_top<CMax<float, idx_t>>(
                                candidate_keep,
                                candidate_primary.data(),
                                candidate_locs.data(),
                                pd,
                                loc);
                        }
                    }
                }

                ndis_query += total_scanned;

                const size_t candidate_count = heap_ready ? candidate_keep : heap_size;
                if (candidate_count > 0 && heap_ready) {
                    heap_reorder<CMax<float, idx_t>>(
                        candidate_count,
                        candidate_primary.data(),
                        candidate_locs.data());
                }

                for (size_t ci = 0; ci < candidate_count; ++ci) {
                    const idx_t loc = candidate_locs[ci];
                    const idx_t list_no = lo_listno(loc);
                    const idx_t offset = lo_offset(loc);
                    if (list_no < 0) {
                        continue;
                    }

                    const size_t list_size = invlists->list_size(list_no);
                    if (offset < 0 || static_cast<size_t>(offset) >= list_size) {
                        continue;
                    }

                    scanner->set_list(list_no,  0.0f);
                    InvertedLists::ScopedCodes codes(invlists, list_no);
                    InvertedLists::ScopedIds ids(invlists, list_no);

                    const uint8_t* packed_code =
                        codes.get() + static_cast<size_t>(offset) * code_size;
                    const float dis = jhq_scanner->refine_distance_from_primary_for_list(
                        static_cast<size_t>(offset),
                        packed_code,
                        candidate_primary[ci]);
                    const idx_t id = store_pairs ? loc : ids.get()[offset];

                    if (is_max_heap) {
                        if (CMin<float, idx_t>::cmp(simi[0], dis)) {
                            heap_replace_top<CMin<float, idx_t>>(
                                k, simi, idxi, dis, id);
                        }
                    } else {
                        if (CMax<float, idx_t>::cmp(simi[0], dis)) {
                            heap_replace_top<CMax<float, idx_t>>(
                                k, simi, idxi, dis, id);
                        }
                    }
                }
            }

            if (is_max_heap) {
                heap_reorder<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
            }

            total_ndis += ndis_query;
        }
    }

    if (ivf_stats) {
        ivf_stats->ndis += total_ndis;
    }

    indexIVFJHQ_stats.nq += n;
    indexIVFJHQ_stats.ndis += total_ndis;
}

void IndexIVFJHQ::range_search_preassigned(idx_t nx,
    const float* x,
    float radius,
    const idx_t* keys,
    const float* coarse_dis,
    RangeSearchResult* result,
    bool store_pairs,
    const IVFSearchParameters* params,
    IndexIVFStats* stats) const
{
    FAISS_THROW_IF_NOT_MSG(
        is_trained, "Index must be trained before range search");

    const size_t search_nprobe = params && params->nprobe > 0 ? params->nprobe : this->nprobe;
    const size_t actual_nprobe = std::min(nlist, search_nprobe);

    std::unique_ptr<InvertedListScanner> scanner(
        get_InvertedListScanner(store_pairs, nullptr, params));

    result->nq = nx;
    if (!result->lims) {
        result->lims = new size_t[nx + 1];
    }

    for (idx_t i = 0; i <= nx; ++i) {
        result->lims[i] = 0;
    }

    size_t total_ndis = 0;
    double search_start_time = getmillisecs();

    bool use_parallel = nx > 10;
    std::vector<std::vector<std::pair<float, idx_t>>> all_results(nx);

    if (use_parallel) {
#pragma omp parallel for reduction(+ : total_ndis)
        for (idx_t i = 0; i < nx; ++i) {
            process_range_query(i,
                x,
                radius,
                keys,
                coarse_dis,
                actual_nprobe,
                store_pairs,
                params,
                all_results[i],
                total_ndis);
        }
    } else {
        for (idx_t i = 0; i < nx; ++i) {
            process_range_query(i,
                x,
                radius,
                keys,
                coarse_dis,
                actual_nprobe,
                store_pairs,
                params,
                all_results[i],
                total_ndis);
        }
    }

    for (idx_t i = 0; i < nx; ++i) {
        result->lims[i + 1] = result->lims[i] + all_results[i].size();
    }

    size_t total_results = result->lims[nx];
    if (total_results > 0) {
        result->distances = new float[total_results];
        result->labels = new idx_t[total_results];

        size_t offset = 0;
        for (idx_t i = 0; i < nx; ++i) {
            auto& query_results = all_results[i];
            std::sort(query_results.begin(), query_results.end());

            for (size_t j = 0; j < query_results.size(); ++j) {
                result->distances[offset + j] = query_results[j].first;
                result->labels[offset + j] = query_results[j].second;
            }
            offset += query_results.size();
        }
    } else {
        result->distances = nullptr;
        result->labels = nullptr;
    }

    if (stats) {
        stats->ndis += total_ndis;
        stats->search_time += getmillisecs() - search_start_time;
    }
}

void IndexIVFJHQ::process_range_query(
    idx_t query_idx,
    const float* x,
    float radius,
    const idx_t* keys,
    const float* coarse_dis,
    size_t nprobe,
    bool store_pairs,
    const IVFSearchParameters* params,
    std::vector<std::pair<float, idx_t>>& results,
    size_t& ndis_counter) const
{
    std::unique_ptr<InvertedListScanner> scanner(
        get_InvertedListScanner(store_pairs, nullptr, params));

    scanner->set_query(x + query_idx * d);

    for (size_t ik = 0; ik < nprobe; ik++) {
        idx_t key = keys[query_idx * nprobe + ik];
        if (key < 0)
            continue;

        scanner->set_list(key, coarse_dis[query_idx * nprobe + ik]);

        size_t list_size = invlists->list_size(key);
        if (list_size == 0)
            continue;

        InvertedLists::ScopedCodes codes(invlists, key);
        InvertedLists::ScopedIds ids(invlists, key);

        for (size_t j = 0; j < list_size; ++j) {
            float distance = scanner->distance_to_code(codes.get() + j * code_size);

            bool within_range = (metric_type == METRIC_L2)
                ? (distance <= radius)
                : (distance >= radius);

            if (within_range) {
                idx_t id = store_pairs ? lo_build(key, j) : ids.get()[j];
                results.emplace_back(distance, id);
            }
        }

        ndis_counter += list_size;
    }
}

InvertedListScanner* IndexIVFJHQ::get_InvertedListScanner(
    bool store_pairs,
    const IDSelector* sel,
    const IVFSearchParameters* params_in) const
{
    if (should_use_single_level_adapter()) {
        ensure_single_level_adapter_ready();
        return single_level_adapter_->get_InvertedListScanner(
            store_pairs, sel, params_in);
    }

    const IVFJHQSearchParameters* jhq_params = dynamic_cast<const IVFJHQSearchParameters*>(params_in);

    return new IVFJHQScanner(*this, store_pairs, sel, jhq_params);
}

void IndexIVFJHQ::benchmark_search(idx_t nq,
    const float* queries,
    idx_t k,
    size_t nprobe_val,
    int num_runs) const
{
    if (!is_trained) {
        return;
    }

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    size_t old_nprobe = nprobe;
    const_cast<IndexIVFJHQ*>(this)->nprobe = nprobe_val;
    if (nq > 0) {
        search(1, queries, k, distances.data(), labels.data());
    }

    for (int run = 0; run < num_runs; ++run) {
        search(nq, queries, k, distances.data(), labels.data());
    }

    const_cast<IndexIVFJHQ*>(this)->nprobe = old_nprobe;
}

} 
