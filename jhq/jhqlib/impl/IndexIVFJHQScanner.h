#pragma once

#include "IndexIVFJHQ.h"

#include <faiss/utils/AlignedTable.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace faiss {



struct IVFJHQScanner final : InvertedListScanner {
public:
    explicit IVFJHQScanner(
        const IndexIVFJHQ& idx,
        bool store_pairs = false,
        const IDSelector* sel = nullptr,
        const IVFJHQSearchParameters* search_params = nullptr);

    void set_query(const float* query_vector) override;
    void set_list(idx_t list_no, float coarse_dis) override;

    size_t scan_codes(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const override;

    void scan_codes_range(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float radius,
        RangeQueryResult& res) const override;

    
    
    
    const float* compute_primary_distances_for_list(
        size_t list_size,
        const uint8_t* codes) const;

    float refine_distance_from_primary_for_list(
        size_t offset_in_list,
        const uint8_t* packed_code,
        float primary_distance) const;

    void set_rotated_query(const float* query_vector_rotated);
    void reset_for_reuse();
    size_t get_workspace_memory_usage() const;
    void print_performance_stats() const;

private:
    struct BatchCandidate {
        float distance;
        idx_t id;
        size_t original_idx;
    };

    static constexpr size_t kHeapBatchSize = 64;

    const IndexIVFJHQ& index;
    const IVFJHQSearchParameters* params;
    const float* query = nullptr;
    idx_t list_no = -1;

    bool use_early_termination;
    bool compute_residuals;
    float oversampling_factor;

    mutable std::vector<float> query_rotated;
    mutable std::vector<float> jhq_primary_tables;
    mutable std::vector<float> jhq_residual_tables;
    mutable std::vector<size_t> jhq_residual_offsets;
    mutable bool tables_computed;

    mutable AlignedTable<float> workspace_primary_distances;
    mutable std::vector<size_t> workspace_candidate_indices;
    mutable std::vector<float> workspace_candidate_distances;
    mutable std::vector<uint16_t> workspace_primary_distances_quantized;
    mutable float primary_distance_min = 0.0f;
    mutable float primary_distance_scale = 1.0f;

    mutable size_t workspace_capacity_lists;
    mutable size_t workspace_capacity_primary_tables;
    mutable size_t workspace_capacity_residual_tables;
    mutable size_t workspace_capacity_candidates;

    mutable bool is_reusable;
    mutable std::chrono::steady_clock::time_point last_used;
    mutable size_t reuse_count;

    mutable size_t total_scans_performed;
    mutable size_t total_codes_processed;

    mutable const jhq_internal::PreDecodedCodes* current_list_pre_decoded = nullptr;

    
    
    mutable struct OptimizedWorkspace {
        bool initialized = false;
        size_t last_list_size = 0;

        void ensure_capacity(size_t list_size, size_t candidates)
        {
            (void)candidates;
            if (list_size > last_list_size * 1.5) {
                last_list_size = list_size;
            }
            initialized = true;
        }
    } optimized_workspace;

    
    mutable std::vector<BatchCandidate> heap_batch_buffer_;

    bool query_already_rotated = false;

    float distance_to_code(const uint8_t* code) const;
    float distance_to_code_with_bit_decoding(const uint8_t* code) const;
    float distance_to_code_separated_storage(size_t offset_in_list) const;
    float distance_to_code_pre_decoded(const uint8_t* code) const;
    float refine_distance_from_primary(
        size_t offset_in_list,
        const uint8_t* packed_code,
        float primary_distance) const;

    float compute_residual_distance_separated_storage(const uint8_t* residual_codes) const;
    float compute_residual_distance_pre_decoded(size_t vector_idx_in_list) const;

    void distance_four_codes(
        const uint8_t* code1,
        const uint8_t* code2,
        const uint8_t* code3,
        const uint8_t* code4,
        float& dist1,
        float& dist2,
        float& dist3,
        float& dist4) const;
    void distance_four_codes_pre_decoded(
        const uint8_t* code1,
        const uint8_t* code2,
        const uint8_t* code3,
        const uint8_t* code4,
        float& dist1,
        float& dist2,
        float& dist3,
        float& dist4) const;
    void distance_four_codes_with_bit_decoding(
        const uint8_t* code1,
        const uint8_t* code2,
        const uint8_t* code3,
        const uint8_t* code4,
        float& dist1,
        float& dist2,
        float& dist3,
        float& dist4) const;
    void distance_sixteen_codes(const uint8_t* codes[], float distances[]) const;

    void compute_primary_distances(size_t list_size, const uint8_t* codes) const;
    void compute_primary_distances_pre_decoded(size_t list_size, float* distances) const;
    void compute_primary_distances_with_bit_decoding(
        size_t list_size,
        const uint8_t* codes,
        float* distances) const;
    void compute_primary_distances_separated_storage(size_t list_size, float* distances) const;
    void compute_primary_distances_packed_codes(
        size_t list_size,
        const uint8_t* codes,
        float* distances) const;

    void ensure_workspace_capacity(size_t max_list_size, size_t max_candidates) const;
    void ensure_table_capacity(size_t primary_table_size, size_t residual_table_size) const;

    size_t scan_codes_exhaustive(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const;
    size_t scan_codes_exhaustive_l2_gated(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const;
    size_t scan_codes_early_termination(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const;
    size_t scan_codes_small_k_simd(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const;
    size_t scan_codes_k4_unrolled(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const;
    size_t scan_codes_k8_unrolled(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const;

    void select_top_candidates(
        const float* primary_distances,
        size_t list_size,
        size_t n_candidates) const;

    void update_heap_single_candidate(
        size_t j,
        float dis,
        bool is_max_heap,
        size_t k,
        float* simi,
        idx_t* idxi,
        const idx_t* ids,
        size_t& nup) const;
    void update_heap_four_candidates(
        const size_t saved_j[4],
        float dist0,
        float dist1,
        float dist2,
        float dist3,
        bool is_max_heap,
        size_t k,
        float* simi,
        idx_t* idxi,
        const idx_t* ids,
        size_t& nup) const;
    void update_heap_sixteen_candidates(
        const size_t saved_j[16],
        const float distances[16],
        bool is_max_heap,
        size_t k,
        float* simi,
        idx_t* idxi,
        const idx_t* ids,
        size_t& nup) const;
    void update_heap_sixteen_candidates_batched(
        const size_t saved_j[16],
        const float distances[16],
        bool is_max_heap,
        size_t k,
        float* simi,
        idx_t* idxi,
        const idx_t* ids,
        size_t& nup) const;
    void flush_heap_batch(
        bool is_max_heap,
        size_t k,
        float* simi,
        idx_t* idxi,
        size_t& nup) const;

    void resize_workspace(
        size_t required_size,
        std::vector<float>& workspace,
        size_t& current_capacity) const;
    void resize_workspace(
        size_t required_size,
        std::vector<size_t>& workspace,
        size_t& current_capacity) const;
    void resize_workspace(
        size_t required_size,
        AlignedTable<float>& workspace,
        size_t& current_capacity) const;

    void quantize_primary_distances(size_t list_size) const;
    float reconstruct_primary_distance(uint16_t qvalue) const;

    bool has_separated_storage_available() const;

    idx_t get_candidate_id(size_t j, const idx_t* ids) const;
    bool passes_selector(size_t j, const idx_t* ids) const;

    size_t get_residual_table_index(int m, int level, int d, uint32_t scalar_id) const;

    mutable std::vector<idx_t> workspace_candidate_indices_faiss;
    mutable std::vector<float> workspace_candidate_distances_faiss;
};

} 
