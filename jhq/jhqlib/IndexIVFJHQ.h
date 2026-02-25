#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/impl/platform_macros.h>

#include "IndexJHQ.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace faiss {

struct IndexIVFPQ;

struct IVFJHQSearchParameters : IVFSearchParameters {
    float jhq_oversampling_factor = -1.0f;
    bool use_early_termination = true;
    bool compute_residuals = true;
    bool use_precomputed_tables = true;

    IVFJHQSearchParameters() = default;
    ~IVFJHQSearchParameters() override = default;
};

struct IndexIVFJHQ : IndexIVF {

    IndexJHQ jhq;
    mutable std::unique_ptr<IndexIVFPQ> single_level_adapter_;

    float default_jhq_oversampling = 4.0f;
    bool use_early_termination = true;

    using IndexIVF::nprobe;

    int parallel_mode = 2;

    mutable std::vector<float> rotated_coarse_centroids;
    mutable std::mutex rotated_centroids_mutex_;
    mutable std::atomic<bool> rotated_centroids_computed{false};

    bool use_kmeans_refinement = false;
    int kmeans_niter = 25;
    int kmeans_nredo = 1;
    int kmeans_seed = 1234;

    mutable std::vector<jhq_internal::PreDecodedCodes> list_pre_decoded_codes;
    mutable std::mutex pre_decoded_codes_mutex_;
    mutable std::atomic<bool> pre_decoded_codes_initialized{false};
    mutable std::mutex single_level_adapter_mutex_;
    mutable std::atomic<bool> single_level_adapter_dirty_{true};

    bool use_pre_decoded_codes = true;
    size_t pre_decode_threshold = 1000;

    struct SearchWorkspace {
        std::vector<idx_t> list_indices;
        std::vector<float> coarse_distances;
        std::vector<float> rotated_queries;

        void ensure_capacity(idx_t nq, size_t nprobe, size_t dim)
        {
            const size_t total_lists = static_cast<size_t>(nq) * nprobe;
            const size_t rotated_size = static_cast<size_t>(nq) * dim;

            if (list_indices.capacity() < total_lists) {
                list_indices.reserve(total_lists);
            }
            if (coarse_distances.capacity() < total_lists) {
                coarse_distances.reserve(total_lists);
            }
            if (rotated_queries.capacity() < rotated_size) {
                rotated_queries.reserve(rotated_size);
            }

            list_indices.resize(total_lists);
            coarse_distances.resize(total_lists);
            rotated_queries.resize(rotated_size);
        }
    };

    SearchWorkspace& get_search_workspace() const;
    static thread_local SearchWorkspace search_workspace_;
    void ensure_single_level_adapter_ready() const;
    void mark_single_level_adapter_dirty();
    bool should_use_single_level_adapter() const
    {
        return jhq.num_levels == 1;
    }

    IndexIVFJHQ();

    ~IndexIVFJHQ() override;

    IndexIVFJHQ(const IndexIVFJHQ& other) = delete;
    IndexIVFJHQ& operator=(const IndexIVFJHQ& other) = delete;

    IndexIVFJHQ(IndexIVFJHQ&& other) noexcept = default;
    IndexIVFJHQ& operator=(IndexIVFJHQ&& other) noexcept = default;

    IndexIVFJHQ(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        const std::vector<int>& level_bits,
        bool use_jl_transform = true,
        float jhq_oversampling = 4.0f,
        MetricType metric = METRIC_L2,
        bool own_invlists = true);

    void train(idx_t n, const float* x) override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr) const override;
    void search_preassigned_with_rotated_queries(idx_t n, const float* x_rotated, idx_t k, const idx_t* keys, const float* coarse_dis, float* distances, idx_t* labels, bool store_pairs, const IVFSearchParameters* params, IndexIVFStats* ivf_stats) const;

    void search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params = nullptr,
        IndexIVFStats* stats = nullptr) const override;

    void range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs = false,
        const IVFSearchParameters* params = nullptr,
        IndexIVFStats* stats = nullptr) const override;

    void add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context = nullptr) override;

    void encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listno = false) const override;

    void decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* list_nos,
        float* x) const override;

    InvertedListScanner* get_InvertedListScanner(
        bool store_pairs = false,
        const IDSelector* sel = nullptr,
        const IVFSearchParameters* params = nullptr) const override;

    void reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const override;

    size_t sa_code_size() const override;
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    void reset() override;
    void check_compatible_for_merge(const Index& otherIndex) const override;
    void merge_from(Index& otherIndex, idx_t add_id) override;

    void set_clustering_parameters(bool use_kmeans, int niter, int seed)
    {
        set_clustering_parameters(use_kmeans, niter,  1, seed);
    }

    void set_clustering_parameters(bool use_kmeans, int niter, int nredo, int seed)
    {
        use_kmeans_refinement = use_kmeans;
        kmeans_niter = niter;
        kmeans_nredo = nredo;
        kmeans_seed = seed;

        jhq.set_clustering_parameters(use_kmeans, niter, nredo, seed);
    }

    void set_parallel_mode(int mode)
    {
        FAISS_THROW_IF_NOT_MSG(mode >= 0 && mode <= 3, "Invalid parallel mode");
        parallel_mode = mode;
    }

    void write(IOWriter* f) const;
    static IndexIVFJHQ* read(IOReader* f);

    void set_jhq_oversampling(float oversampling);
    void set_early_termination(bool enable);
    void optimize_for_search();

    size_t get_memory_usage() const;
    float get_compression_ratio() const;
    float get_search_accuracy_estimate(size_t nprobe_val) const;
    void print_stats() const;
    void benchmark_search(idx_t nq, const float* queries, idx_t k,
        size_t nprobe_val, int num_runs = 5) const;

    idx_t train_encoder_num_vectors() const override;

    void train_jhq_with_precomputed_residuals(idx_t n, const float* residuals);
    void add_with_precomputed_assignments(idx_t n, const float* x,
        const idx_t* xids = nullptr,
        const idx_t* coarse_idx = nullptr);

    void precompute_rotated_centroids() const;
    const float* get_rotated_centroid(idx_t list_no) const
    {
        if (!rotated_centroids_computed) {
            precompute_rotated_centroids();
        }
        return rotated_coarse_centroids.data() + list_no * d;
    }

    void train_jhq_on_originals(idx_t n, const float* x);
    void validate_parameters() const;

    bool has_fast_primary_codes() const
    {
        return jhq.has_optimized_layout() && !jhq.has_pre_decoded_codes();
    }

    void extract_primary_codes_from_jhq_code(
        const uint8_t* jhq_code,
        uint8_t* primary_codes) const;

    void initialize_pre_decoded_codes() const;
    void extract_list_codes(idx_t list_no) const;
    void extract_single_code_to_pre_decoded(const uint8_t* packed_code, jhq_internal::PreDecodedCodes& pre_decoded, size_t vector_idx) const;
    void invalidate_pre_decoded_codes();
    bool has_pre_decoded_codes_for_list(idx_t list_no) const;
    const jhq_internal::PreDecodedCodes* get_pre_decoded_codes_for_list(idx_t list_no) const;

    size_t get_pre_decoded_memory_usage() const;
    void set_pre_decode_threshold(size_t threshold);
    void enable_pre_decoded_codes(bool enable);

    mutable std::vector<std::vector<idx_t>> list_to_global_mapping;
    mutable std::mutex mapping_mutex_;
    mutable std::atomic<bool> mapping_initialized{false};

    void initialize_vector_mapping() const;
    idx_t get_global_vector_index(idx_t list_no, idx_t offset) const;

    void encode_to_separated_storage(idx_t n, const float* x);
    void initialize_vector_mapping_for_new_vectors(
        idx_t n,
        idx_t old_ntotal,
        const idx_t* coarse_idx,
        const idx_t* xids);
    void encode_vectors_fallback(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes) const;

protected:
    void initialize_optimized_layout();

    void process_range_query(
        idx_t query_idx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        size_t nprobe,
        bool store_pairs,
        const IVFSearchParameters* params,
        std::vector<std::pair<float, idx_t>>& results,
        size_t& ndis_counter) const;

    void cleanup();
};

struct IndexIVFJHQStats {
    size_t nq = 0;
    size_t nlist = 0;
    size_t ndis = 0;
    size_t n_early_termination = 0;
    size_t n_primary_only = 0;
    size_t n_full_residual = 0;

    double quantization_time = 0.0;
    double jhq_search_time = 0.0;
    double total_search_time = 0.0;

    void reset();
    void add(const IndexIVFJHQStats& other);
};

FAISS_API extern IndexIVFJHQStats indexIVFJHQ_stats;

size_t estimate_ivfjhq_memory(
    int d,
    size_t n,
    size_t nlist,
    int M,
    const std::vector<int>& level_bits,
    bool use_tables = false);

void write_index_ivf_jhq(const IndexIVFJHQ* idx, IOWriter* f);
IndexIVFJHQ* read_index_ivf_jhq(IOReader* f);
} 
