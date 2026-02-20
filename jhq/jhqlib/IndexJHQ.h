#pragma once

#include <algorithm>
#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <immintrin.h>

#include <cstdint>
#include <faiss/utils/hamming.h>
#include <iostream>
#include <lapacke.h>
#include <memory>
#include <random>
#include <faiss/impl/ProductQuantizer.h>
#include <vector>

#ifdef __has_include
#if __has_include(<openblas/cblas.h>)
#include <openblas/cblas.h>
#elif __has_include(<cblas.h>)
#include <cblas.h>
#elif __has_include(<cblas-openblas.h>)
#include <cblas-openblas.h>
#else
#error "CBLAS header not found"
#endif
#else
#include <cblas.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#define FORCE_INLINE __attribute__((always_inline)) inline
#define RESTRICT __restrict__
#elif defined(_MSC_VER)
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define FORCE_INLINE __forceinline
#define RESTRICT __restrict
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#define FORCE_INLINE inline
#define RESTRICT
#endif

namespace jhq_internal {
#ifdef __AVX512F__
static constexpr bool has_avx512 = true;
static constexpr int simd_width = 16;
#elif defined(__AVX2__)
static constexpr bool has_avx512 = false;
static constexpr int simd_width = 8;
#else
static constexpr bool has_avx512 = false;
static constexpr int simd_width = 1;
#endif
}

namespace faiss {

namespace jhq_internal {

float erfinv_approx(float x);
float fvec_L2sqr_avx512(const float* x, const float* y, size_t d);
float fvec_L2sqr_avx2(const float* x, const float* y, size_t d);
float fvec_L2sqr_dispatch(const float* x, const float* y, size_t d);

static constexpr size_t CACHE_LINE_SIZE = 64;
static constexpr size_t AVX512_ALIGNMENT = 64;
static constexpr size_t AVX2_ALIGNMENT = 32;
static constexpr size_t MAX_ROTATE_SIZE_GB = 50;
static constexpr size_t MAX_TRAIN_SIZE_GB = 50;

template<size_t Alignment = AVX512_ALIGNMENT>
class AlignedAllocator {
public:
    static void* allocate(size_t size)
    {
        void* ptr = nullptr;
        size_t aligned_size = (size + Alignment - 1) & ~(Alignment - 1);
        int result = posix_memalign(&ptr, Alignment, aligned_size);
        if (result != 0 || ptr == nullptr) {
            throw std::bad_alloc();
        }
        return ptr;
    }

    static void deallocate(void* ptr)
    {
        if (ptr)
            std::free(ptr);
    }
};

template<typename T, size_t Alignment = AVX512_ALIGNMENT>
class AlignedBuffer {
public:
    AlignedBuffer()
        : data_(nullptr)
        , size_(0)
        , capacity_(0)
    {
    }

    explicit AlignedBuffer(size_t size)
        : size_(size)
        , capacity_(size)
    {
        if (size > 0) {
            data_ = static_cast<T*>(AlignedAllocator<Alignment>::allocate(capacity_ * sizeof(T)));
        } else {
            data_ = nullptr;
        }
    }

    ~AlignedBuffer()
    {
        AlignedAllocator<Alignment>::deallocate(data_);
    }

    AlignedBuffer(AlignedBuffer&& other) noexcept
        : data_(other.data_)
        , size_(other.size_)
        , capacity_(other.capacity_)
    {
        other.data_ = nullptr;
        other.size_ = 0;
        other.capacity_ = 0;
    }

    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept
    {
        if (this != &other) {
            AlignedAllocator<Alignment>::deallocate(data_);
            data_ = other.data_;
            size_ = other.size_;
            capacity_ = other.capacity_;
            other.data_ = nullptr;
            other.size_ = 0;
            other.capacity_ = 0;
        }
        return *this;
    }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void resize(size_t new_size)
    {
        if (new_size > capacity_) {
            AlignedAllocator<Alignment>::deallocate(data_);
            capacity_ = new_size;
            if (capacity_ > 0) {
                data_ = static_cast<T*>(AlignedAllocator<Alignment>::allocate(capacity_ * sizeof(T)));
            } else {
                data_ = nullptr;
            }
        }
        size_ = new_size;
    }

    void clear() { size_ = 0; }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    bool empty() const { return size_ == 0; }

    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }

private:
    T* data_;
    size_t size_;
    size_t capacity_;
};

struct PreDecodedCodes {
    AlignedBuffer<uint8_t> primary_codes;
    AlignedBuffer<uint8_t> residual_codes;

    size_t primary_stride;
    size_t residual_stride;
    size_t residual_level_stride;
    size_t residual_subspace_stride;

    int num_levels;
    int M;
    int Ds;
    bool is_initialized;

    PreDecodedCodes();
    void initialize(int M_val, int Ds_val, int num_levels_val, idx_t ntotal);
    void clear();
    bool empty() const;

    inline const uint8_t* get_primary_codes(idx_t vector_idx) const
    {
        FAISS_THROW_IF_NOT_MSG(is_initialized, "PreDecodedCodes not initialized");
        FAISS_THROW_IF_NOT_MSG(vector_idx * primary_stride < primary_codes.size(),
            "Primary codes index out of bounds");
        return primary_codes.data() + vector_idx * primary_stride;
    }

    inline const uint8_t* get_residual_codes(idx_t vector_idx) const
    {
        FAISS_THROW_IF_NOT_MSG(is_initialized && num_levels > 1,
            "Residual codes not available");
        FAISS_THROW_IF_NOT_MSG(vector_idx * residual_stride < residual_codes.size(),
            "Residual codes index out of bounds");
        return residual_codes.data() + vector_idx * residual_stride;
    }

    inline uint8_t get_residual_code(idx_t vector_idx, int m, int level, int d) const
    {
        const uint8_t* base = get_residual_codes(vector_idx);
        const size_t offset = m * residual_subspace_stride + (level - 1) * residual_level_stride + d;
        return base[offset];
    }

    size_t memory_usage() const;
};

}

struct JHQSearchParameters : SearchParameters {
    float oversampling_factor = -1.0f;
    bool use_early_termination = true;
    bool compute_residuals = true;
    ~JHQSearchParameters() override { }
};

struct IndexJHQ : IndexFlatCodes {
private:
    struct SearchWorkspace {
        jhq_internal::AlignedBuffer<float> query_rotated;
        jhq_internal::AlignedBuffer<float> primary_distance_table;
        jhq_internal::AlignedBuffer<float> residual_distance_tables;
        jhq_internal::AlignedBuffer<float> all_primary_distances;
        jhq_internal::AlignedBuffer<float> candidate_distances;
        jhq_internal::AlignedBuffer<idx_t> candidate_indices;
    };

public:
    int M;
    int Ds;
    int num_levels;
    std::vector<int> level_bits;

    bool use_jl_transform;
    bool use_analytical_init;
    float default_oversampling;
    bool verbose;

    std::vector<float> rotation_matrix;
    bool is_rotation_trained;

    std::vector<std::vector<std::vector<float>>> codewords;
    std::vector<std::vector<std::vector<float>>> scalar_codebooks;

    mutable bool residual_pq_dirty_ = true;
    bool use_kmeans_refinement;
    int kmeans_niter;
    int kmeans_seed;

    size_t residual_bits_per_subspace;

    mutable jhq_internal::PreDecodedCodes separated_codes_;

    mutable bool memory_layout_initialized_;

    size_t primary_codewords_stride_;
    size_t scalar_codebooks_stride_;
    size_t residual_codes_stride_;

    IndexJHQ();
    explicit IndexJHQ(
        int d,
        int M,
        const std::vector<int>& level_bits,
        bool use_jl_transform = true,
        float default_oversampling = 4.0f,
        bool use_analytical_init = true,
        bool verbose = false,
        MetricType metric = METRIC_L2);

    IndexJHQ(const IndexJHQ& other);
    IndexJHQ& operator=(const IndexJHQ& other);
    virtual ~IndexJHQ();

    void train(idx_t n, const float* x) override;
    bool is_trained_() const;
    void reset();

    void add(idx_t n, const float* x) override;
    void reset_data();

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params = nullptr) const override;

    void range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params = nullptr) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
    void reconstruct(idx_t key, float* recons) const override;
    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void set_default_oversampling(float oversampling);
    size_t get_memory_usage() const;

    void apply_jl_rotation(idx_t n, const float* x_in, float* x_out) const;

    void compute_primary_distance_tables_flat(
        const float* query_rotated,
        int K0,
        float* distance_table_flat) const;

    idx_t train_encoder_num_vectors() const
    {
        idx_t base_requirement = 1000;

        if (!level_bits.empty()) {
            int K0 = 1 << level_bits[0];
            base_requirement = std::max(base_requirement, static_cast<idx_t>(K0 * 50));
        }

        base_requirement *= M;

        return base_requirement;
    }

    void compute_residual_distance_tables(
        const float* query_rotated,
        std::vector<float>& flat_tables,
        std::vector<size_t>& level_offsets) const;

    mutable std::unique_ptr<ProductQuantizer> residual_pq_;

    const ProductQuantizer* get_residual_product_quantizer() const;
    void mark_residual_tables_dirty() { residual_pq_dirty_ = true; }

    void analytical_gaussian_init(const float* data, idx_t n, int dim, int k, float* centroids) const;
    void generate_qr_rotation_matrix(int random_seed = 1234);

    void set_clustering_parameters(bool use_kmeans, int niter, int seed);

    void write(IOWriter* f) const;
    static IndexJHQ* read(IOReader* f);
    void initialize_data_structures();

    void initialize_memory_layout();
    void invalidate_memory_layout();

    SearchWorkspace& get_search_workspace() const;

    bool has_optimized_layout() const
    {
        return memory_layout_initialized_ && has_pre_decoded_codes();
    }

    void compute_primary_distance_table(
        const float* query_rotated,
        float* distance_table) const;

    void extract_all_codes_after_add();
    bool has_pre_decoded_codes() const;
    size_t get_pre_decoded_memory_usage() const;

private:
    thread_local static SearchWorkspace workspace_;

    void train_subspace_quantizers(int subspace_idx, idx_t n, const float* subspace_data, int random_seed);
    void encode_single_vector(const float* x, uint8_t* code) const;
    void decode_single_code(const uint8_t* code, float* x) const;
    void validate_parameters() const;
    void compute_code_size();

    void train_primary_level(int subspace_idx, idx_t n, const float* data, int K, int random_seed);
    void train_residual_level(int subspace_idx, int level, idx_t n, const float* residuals, int K);
    void update_residuals_after_level(int subspace_idx, int level, idx_t n, float* residuals);

    void compute_primary_distances_flat(
        const float* distance_table_flat,
        int K0,
        float* distances) const;

    void search_single_query_early_termination(
        const float* query,
        idx_t k,
        float oversampling,
        float* distances,
        idx_t* labels) const;

    size_t search_single_query_exhaustive(
        const float* query,
        idx_t k,
        bool compute_residuals,
        float* distances,
        idx_t* labels) const;

    void compute_primary_distances(
        const float* distance_table_flat,
        int K0,
        float* distances) const;

    static void compute_subspace_distances_simd(
        const float* query_sub,
        const float* codewords,
        float* distances,
        int K,
        int Ds);

    friend struct JHQDistanceComputer;

    void extract_single_vector_all_codes(idx_t vector_idx) const;

    void encode_to_separated_storage(idx_t n, const float* x_rotated) const;
    void encode_single_vector_separated(const float* x, idx_t vector_idx) const;
    int find_best_centroid(const float* residual, const std::vector<float>& centroids, int K) const;
    void subtract_centroid(float* residual, const std::vector<float>& centroids, int best_k) const;
    void encode_residual_levels_separated(int m, const float* residual, uint8_t* residual_dest, size_t& offset) const;
    float compute_exact_distance_separated(idx_t vector_idx, const float* query_rotated) const;
};

struct JHQDistanceComputer : FlatCodesDistanceComputer {
public:
    const IndexJHQ& index;

    std::vector<float> query_rotated;
    std::vector<float> primary_distance_table_flat;
    std::vector<float> residual_distance_tables_flat;
    std::vector<size_t> residual_table_offsets;

    mutable std::vector<float> temp_workspace;

    bool tables_computed;
    bool has_residual_levels;

    bool use_quantized_tables;
    std::vector<uint16_t> quantized_primary_tables;
    float quantization_scale;
    float quantization_offset;

    const int M;
    const int Ds;
    const int num_levels;
    const int K0;
    const size_t primary_table_size;

    explicit JHQDistanceComputer(const IndexJHQ& idx);

    void set_query(const float* x) override;
    float distance_to_code(const uint8_t* code) final;
    float operator()(idx_t i) override;
    float symmetric_dis(idx_t i, idx_t j) override;
    void distances_batch_4(
        const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3,
        float& dis0, float& dis1, float& dis2, float& dis3) override;

private:
    void enable_quantization();
    void compute_and_quantize_tables();
    float precomputed_distance_to_code(const uint8_t* code) const;
    float distance_to_code_with_decoding(const uint8_t* code) const;
    void apply_rotation_to_query(const float* x);
    void compute_primary_distance_table();
    void compute_residual_distance_tables();
    void compute_residual_buffer_sizes();
    float compute_precomputed_residual_distance(size_t vector_idx) const;
};

template<int PRIMARY_BITS, int RESIDUAL_BITS, int NUM_LEVELS, int DS>
struct JHQTraits {
    static constexpr int primary_bits = PRIMARY_BITS;
    static constexpr int residual_bits = RESIDUAL_BITS;
    static constexpr int num_levels = NUM_LEVELS;
    static constexpr int Ds = DS;

    static constexpr int K0 = 1 << PRIMARY_BITS;
    static constexpr int K_RES = 1 << RESIDUAL_BITS;
    static constexpr int total_primary_bits = PRIMARY_BITS;
    static constexpr int total_residual_bits = RESIDUAL_BITS * DS * (NUM_LEVELS - 1);
    static constexpr int total_bits_per_subspace = total_primary_bits + total_residual_bits;

    static constexpr size_t primary_table_size_per_subspace = K0;
    static constexpr size_t residual_table_size_per_level_per_subspace = DS * K_RES;

    static constexpr bool is_simd_friendly = (DS % 8 == 0) && (K0 % 8 == 0);
    static constexpr bool supports_avx512 = (DS >= 16) && (K0 >= 16);
};

namespace jhq_internal {

float compute_cross_term_from_codes(
    const IndexJHQ& index,
    const uint8_t* primary_codes,
    const uint8_t* residual_codes,
    size_t residual_subspace_stride,
    size_t residual_level_stride);

} // namespace jhq_internal

void write_index_jhq(const IndexJHQ* idx, IOWriter* f);
IndexJHQ* read_index_jhq(IOReader* f);
} // namespace faiss
