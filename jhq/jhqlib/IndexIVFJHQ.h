#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/io.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/index_io.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/partitioning.h>

#include "IndexJHQ.h"

#include <memory>
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

struct BatchCandidate {
    float distance;
    idx_t id;
    size_t original_idx;
};

struct IndexIVFJHQ : IndexIVF {

    IndexJHQ jhq;
    mutable std::unique_ptr<IndexIVFPQ> single_level_adapter_;

    float default_jhq_oversampling = 4.0f;
    bool use_early_termination = true;

    using IndexIVF::nprobe;

    int parallel_mode = 2;

    mutable std::vector<float> rotated_coarse_centroids;
    mutable bool rotated_centroids_computed = false;

    bool use_kmeans_refinement = false;
    int kmeans_niter = 25;
    int kmeans_seed = 1234;

    mutable std::vector<jhq_internal::PreDecodedCodes> list_pre_decoded_codes;
    mutable bool pre_decoded_codes_initialized = false;
    mutable bool single_level_adapter_dirty_ = true;

    bool use_pre_decoded_codes = true;
    size_t pre_decode_threshold = 1000;

    static constexpr size_t HEAP_BATCH_SIZE = 64;

    mutable std::vector<BatchCandidate> heap_batch_buffer;

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

    virtual ~IndexIVFJHQ() = default;

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
        use_kmeans_refinement = use_kmeans;
        kmeans_niter = niter;
        kmeans_seed = seed;

        jhq.set_clustering_parameters(use_kmeans, niter, seed);

        if (verbose) {
            printf("Clustering parameters set: use_kmeans=%s, niter=%d, seed=%d\n",
                use_kmeans ? "true" : "false", niter, seed);
        }
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
        uint8_t* primary_codes) const
    {
        faiss::BitstringReader bit_reader(jhq_code, jhq.code_size);

        for (int m = 0; m < jhq.M; ++m) {
            uint32_t centroid_id = bit_reader.read(jhq.level_bits[0]);
            primary_codes[m] = static_cast<uint8_t>(centroid_id);
            bit_reader.i += jhq.residual_bits_per_subspace;
        }
    }

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
    mutable bool mapping_initialized = false;

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

struct IVFJHQDistanceComputer : FlatCodesDistanceComputer {
    const IndexIVFJHQ& index;

    const float* query;
    idx_t current_list_no;

    std::unique_ptr<JHQDistanceComputer> jhq_computer;

    explicit IVFJHQDistanceComputer(const IndexIVFJHQ& idx);

    void set_query_and_list(const float* query_vec, idx_t list_no);
    void set_list(idx_t list_no);

    float distance_to_code(const uint8_t* code) override;

    void set_query(const float* x) override;
    float operator()(idx_t i) override;
    float symmetric_distance(idx_t i, idx_t j);

    void distances_batch_4(
        const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3,
        float& dis0, float& dis1, float& dis2, float& dis3) override;
};

struct IVFJHQScanner : InvertedListScanner {
    const IndexIVFJHQ& index;
    const IVFJHQSearchParameters* params;
    const float* query;
    idx_t list_no;

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
        alignas(64) float primary_distances_cache[65536];
        alignas(64) float candidate_distances_cache[8192];
        alignas(64) size_t candidate_indices_cache[8192];

        bool initialized = false;
        size_t last_list_size = 0;

        void ensure_capacity(size_t list_size, size_t candidates)
        {
            if (list_size > last_list_size * 1.5) {
                last_list_size = list_size;
            }
            initialized = true;
        }
    } optimized_workspace;

    explicit IVFJHQScanner(
        const IndexIVFJHQ& idx,
        bool store_pairs = false,
        const IDSelector* sel = nullptr,
        const IVFJHQSearchParameters* search_params = nullptr);

    virtual ~IVFJHQScanner() = default;

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
    void set_rotated_query(const float* query_vector_rotated);
    float distance_to_code(const uint8_t* code) const;
    float distance_to_code_with_bit_decoding(const uint8_t* code) const;
    void distance_four_codes_with_bit_decoding(const uint8_t* code1, const uint8_t* code2, const uint8_t* code3, const uint8_t* code4, float& dist1, float& dist2, float& dist3, float& dist4) const;
    void compute_primary_distances_with_bit_decoding(size_t list_size, const uint8_t* codes, float* distances) const;

    void compute_primary_distances_packed_codes(size_t list_size, const uint8_t* codes, float* distances) const;
    float distance_to_code_separated_storage(size_t offset_in_list) const;
    float compute_residual_distance_separated_storage(
        const uint8_t* residual_codes) const;

    void ensure_workspace_capacity(size_t max_list_size, size_t max_candidates) const;
    void ensure_table_capacity(size_t primary_table_size, size_t residual_table_size) const;

    void reset_for_reuse();

    size_t get_workspace_memory_usage() const;
    void print_performance_stats() const;

    std::string get_search_config() const
    {
        std::ostringstream oss;
        oss << "IVFJHQScanner{"
            << "early_term=" << use_early_termination
            << ", residuals=" << compute_residuals
            << ", oversampling=" << oversampling_factor
            << ", M=" << index.jhq.M
            << ", levels=" << index.jhq.num_levels
            << "}";
        return oss.str();
    }

    bool is_initialized() const
    {
        return tables_computed && query != nullptr;
    }

    size_t get_memory_usage() const
    {
        size_t total = 0;
        total += query_rotated.size() * sizeof(float);
        total += jhq_primary_tables.size() * sizeof(float);
        total += jhq_residual_tables.size() * sizeof(float);
        total += workspace_primary_distances.size() * sizeof(float);
        total += workspace_candidate_indices.size() * sizeof(size_t);
        total += workspace_candidate_distances.size() * sizeof(float);
        return total;
    }

    void compute_primary_distances_separated_storage(size_t list_size, float* distances) const;
    bool has_separated_storage_available() const;

private:
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
    size_t scan_codes_small_k_simd(size_t list_size, const uint8_t* codes, const idx_t* ids, float* simi, idx_t* idxi, size_t k) const;
    size_t scan_codes_k4_unrolled(size_t list_size, const uint8_t* codes, const idx_t* ids, float* simi, idx_t* idxi, size_t k) const;
    size_t scan_codes_k8_unrolled(size_t list_size, const uint8_t* codes, const idx_t* ids, float* simi, idx_t* idxi, size_t k) const;

    size_t scan_codes_early_termination(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const;

    void compute_primary_distances(
        size_t list_size,
        const uint8_t* codes) const;
    void compute_primary_distances_pre_decoded(size_t list_size, float* distances) const;

    void resize_workspace(size_t required_size, std::vector<float>& workspace, size_t& current_capacity) const;
    void resize_workspace(size_t required_size, std::vector<size_t>& workspace, size_t& current_capacity) const;
    void resize_workspace(size_t required_size, AlignedTable<float>& workspace, size_t& current_capacity) const;
    void quantize_primary_distances(size_t list_size) const;
    float reconstruct_primary_distance(uint16_t qvalue) const;

    float distance_to_code_pre_decoded(const uint8_t* code) const;
    float compute_residual_distance_pre_decoded(size_t vector_idx_in_list) const;
    void distance_four_codes(
        const uint8_t* code1, const uint8_t* code2,
        const uint8_t* code3, const uint8_t* code4,
        float& dist1, float& dist2, float& dist3, float& dist4) const;
    void distance_four_codes_pre_decoded(const uint8_t* code1, const uint8_t* code2, const uint8_t* code3, const uint8_t* code4, float& dist1, float& dist2, float& dist3, float& dist4) const;
    void distance_sixteen_codes(
        const uint8_t* codes[], float distances[]) const;

    void update_heap_sixteen_candidates(
        const size_t saved_j[16],
        const float distances[16],
        bool is_max_heap, size_t k,
        float* simi, idx_t* idxi, const idx_t* ids, size_t& nup) const;

    void update_heap_sixteen_candidates_batched(
        const size_t saved_j[16],
        const float distances[16],
        bool is_max_heap, size_t k,
        float* simi, idx_t* idxi, const idx_t* ids, size_t& nup) const;

    void update_heap_four_candidates(
        const size_t saved_j[4],
        float dist0, float dist1, float dist2, float dist3,
        bool is_max_heap, size_t k,
        float* simi, idx_t* idxi, const idx_t* ids, size_t& nup) const;
    void flush_heap_batch(bool is_max_heap, size_t k, float* simi, idx_t* idxi, size_t& nup) const;
    void update_heap_single_candidate(
        size_t j, float dis, bool is_max_heap, size_t k,
        float* simi, idx_t* idxi, const idx_t* ids, size_t& nup) const;

    size_t get_residual_table_index(int m, int level, int d, uint32_t scalar_id) const
    {
        return jhq_residual_offsets[level] + m * index.jhq.Ds * (1 << index.jhq.level_bits[level]) + d * (1 << index.jhq.level_bits[level]) + scalar_id;
    }

    bool query_already_rotated = false;

    void select_top_candidates(
        const float* primary_distances,
        size_t list_size,
        size_t n_candidates) const;

    mutable std::vector<idx_t> workspace_candidate_indices_faiss;
    mutable std::vector<float> workspace_candidate_distances_faiss;

    idx_t get_candidate_id(size_t j, const idx_t* ids) const;
    bool passes_selector(size_t j, const idx_t* ids) const;
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
} // namespace faiss
