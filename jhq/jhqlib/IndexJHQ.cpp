
#include "IndexJHQ.h"

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__SSE2__)
#include <immintrin.h>
#endif
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <faiss/impl/index_read_utils.h>
#include <faiss/impl/io_macros.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>

#include <faiss/utils/prefetch.h>

namespace faiss {

namespace jhq_internal {

#if defined(__x86_64__) || defined(__i386__)
inline bool cpu_supports_avx2_runtime()
{
#if defined(__AVX2__)
    static const bool has = __builtin_cpu_supports("avx2");
    return has;
#else
    return false;
#endif
}

inline bool cpu_supports_avx512_runtime()
{
#if defined(__AVX512F__)
    static const bool has = __builtin_cpu_supports("avx512f");
    return has;
#else
    return false;
#endif
}
#else
inline bool cpu_supports_avx2_runtime() { return false; }
inline bool cpu_supports_avx512_runtime() { return false; }
#endif

#if defined(__AVX2__)
inline float hsum256_ps(__m256 v)
{
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    const __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}
#endif

float erfinv_approx(float x)
{
    if (std::abs(x) >= 1.0f)
        return (x > 0) ? 10.0f : -10.0f;

    const float a = 0.147f;
    const float pi = 3.14159265358979323846f;

    float ln1mx2 = std::log(1.0f - x * x);
    float term1 = 2.0f / (pi * a) + ln1mx2 / 2.0f;
    float term2 = ln1mx2 / a;

    float result = std::sqrt(std::sqrt(term1 * term1 - term2) - term1);
    return (x > 0) ? result : -result;
}

float fvec_L2sqr_dispatch(const float* x, const float* y, size_t d)
{
    return faiss::fvec_L2sqr(x, y, d);
}

float compute_cross_term_from_codes(
    const IndexJHQ& index,
    const uint8_t* primary_codes,
    const uint8_t* residual_codes,
    size_t residual_subspace_stride,
    size_t residual_level_stride)
{
    if (!primary_codes || !residual_codes || index.num_levels <= 1) {
        return 0.0f;
    }

    float dot = 0.0f;
    const int K0 = index.primary_ksub();
    if (K0 <= 0) {
        return 0.0f;
    }
    const bool has_avx512 = cpu_supports_avx512_runtime();
    const bool has_avx2 = cpu_supports_avx2_runtime();
    const bool two_level_residual = (index.num_levels == 2);

    for (int m = 0; m < index.M; ++m) {
        const uint32_t centroid_id = std::min<uint32_t>(
            primary_codes[m], static_cast<uint32_t>(std::max(0, K0 - 1)));
        const float* centroid_ptr = index.get_primary_centroids_ptr(m) +
            static_cast<size_t>(centroid_id) * index.Ds;

        const uint8_t* subspace_residual = residual_codes + m * residual_subspace_stride;
        float subspace_dot = 0.0f;

        
        if (two_level_residual) {
            const float* codebook = index.get_scalar_codebook_ptr(m, 1);
            const int K_res = index.scalar_codebook_ksub(1);
            const uint8_t* level_codes = subspace_residual;
            const int max_idx = K_res - 1;
            int d = 0;

#if defined(__AVX512F__)
            if (has_avx512 && index.Ds >= 16) {
                const __m512i max_idx_v = _mm512_set1_epi32(max_idx);
                __m512 acc = _mm512_setzero_ps();
                for (; d + 15 < index.Ds; d += 16) {
                    const __m128i codes_u8 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(level_codes + d));
                    __m512i idx = _mm512_cvtepu8_epi32(codes_u8);
                    idx = _mm512_min_epi32(idx, max_idx_v);
                    const __m512 residual_vals =
                        _mm512_i32gather_ps(idx, codebook, 4);
                    const __m512 centroid_vals = _mm512_loadu_ps(centroid_ptr + d);
                    acc = _mm512_add_ps(acc, _mm512_mul_ps(centroid_vals, residual_vals));
                }
                subspace_dot += _mm512_reduce_add_ps(acc);
            }
#endif
#if defined(__AVX2__)
            if (has_avx2 && index.Ds - d >= 8) {
                const __m256i max_idx_v = _mm256_set1_epi32(max_idx);
                __m256 acc = _mm256_setzero_ps();
                for (; d + 7 < index.Ds; d += 8) {
                    const __m128i codes_u8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(level_codes + d));
                    __m256i idx = _mm256_cvtepu8_epi32(codes_u8);
                    idx = _mm256_min_epi32(idx, max_idx_v);
                    const __m256 residual_vals =
                        _mm256_i32gather_ps(codebook, idx, 4);
                    const __m256 centroid_vals = _mm256_loadu_ps(centroid_ptr + d);
                    acc = _mm256_add_ps(acc, _mm256_mul_ps(centroid_vals, residual_vals));
                }
                subspace_dot += hsum256_ps(acc);
            }
#endif
            for (; d < index.Ds; ++d) {
                const int safe_idx = std::min<int>(level_codes[d], max_idx);
                subspace_dot += centroid_ptr[d] * codebook[static_cast<size_t>(safe_idx)];
            }
            dot += subspace_dot;
            continue;
        }

        
        for (int d = 0; d < index.Ds; ++d) {
            float residual_acc = 0.0f;
            for (int level = 1; level < index.num_levels; ++level) {
                const float* codebook = index.get_scalar_codebook_ptr(m, level);
                const uint8_t* level_codes =
                    subspace_residual + static_cast<size_t>(level - 1) * residual_level_stride;
                const size_t max_idx = static_cast<size_t>(index.scalar_codebook_ksub(level) - 1);
                const size_t safe_idx = std::min(static_cast<size_t>(level_codes[d]), max_idx);
                residual_acc += codebook[safe_idx];
            }
            subspace_dot += centroid_ptr[d] * residual_acc;
        }
        dot += subspace_dot;
    }

    return 2.0f * dot;
}

float compute_residual_norm_sq_from_codes(
    const IndexJHQ& index,
    const uint8_t* residual_codes,
    size_t residual_subspace_stride,
    size_t residual_level_stride)
{
    if (!residual_codes || index.num_levels != 2) {
        return 0.0f;
    }
    const int K_res = index.scalar_codebook_ksub(1);
    if (K_res <= 0) {
        return 0.0f;
    }
    if (!index.scalar_codebooks_flat_valid_ ||
        index.scalar_codebooks_stride_ == 0 ||
        index.scalar_codebook_level_offsets_.size() < 2 ||
        index.scalar_codebooks_flat_.empty()) {
        return 0.0f;
    }

    const bool has_avx512 = cpu_supports_avx512_runtime();
    const bool has_avx2 = cpu_supports_avx2_runtime();

    const float* scalar_base = index.scalar_codebooks_flat_.data();
    const size_t scalar_stride = index.scalar_codebooks_stride_;
    const size_t level_off = index.scalar_codebook_level_offsets_[1];

    float sum = 0.0f;
    const int max_idx = K_res - 1;

    for (int m = 0; m < index.M; ++m) {
        const float* codebook =
            scalar_base + static_cast<size_t>(m) * scalar_stride + level_off;
        const uint8_t* codes_m =
            residual_codes + static_cast<size_t>(m) * residual_subspace_stride;

        int d = 0;
#if defined(__AVX512F__)
        if (has_avx512 && index.Ds >= 16) {
            const __m512i max_idx_v = _mm512_set1_epi32(max_idx);
            __m512 acc = _mm512_setzero_ps();
            for (; d + 15 < index.Ds; d += 16) {
                const __m128i codes_u8 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(codes_m + d));
                __m512i idx = _mm512_cvtepu8_epi32(codes_u8);
                idx = _mm512_min_epi32(idx, max_idx_v);
                const __m512 rv = _mm512_i32gather_ps(idx, codebook, 4);
                acc = _mm512_fmadd_ps(rv, rv, acc);
            }
            sum += _mm512_reduce_add_ps(acc);
        }
#endif
#if defined(__AVX2__)
        if (has_avx2 && index.Ds - d >= 8) {
            const __m256i max_idx_v = _mm256_set1_epi32(max_idx);
            __m256 acc = _mm256_setzero_ps();
            for (; d + 7 < index.Ds; d += 8) {
                const __m128i codes_u8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(codes_m + d));
                __m256i idx = _mm256_cvtepu8_epi32(codes_u8);
                idx = _mm256_min_epi32(idx, max_idx_v);
                const __m256 rv = _mm256_i32gather_ps(codebook, idx, 4);
                acc = _mm256_fmadd_ps(rv, rv, acc);
            }
            sum += hsum256_ps(acc);
        }
#endif
        for (; d < index.Ds; ++d) {
            const int safe_idx = std::min<int>(codes_m[d], max_idx);
            const float r = codebook[static_cast<size_t>(safe_idx)];
            sum += r * r;
        }
    }

    (void)residual_level_stride;
    return sum;
}

PreDecodedCodes::PreDecodedCodes()
    : primary_stride(0)
    , residual_stride(0)
    , residual_level_stride(0)
    , residual_subspace_stride(0)
    , residual_packed4_stride(0)
    , num_levels(0)
    , M(0)
    , Ds(0)
    , is_initialized(false)
{
}

void PreDecodedCodes::initialize(int M_val, int Ds_val, int num_levels_val, idx_t ntotal)
{
    
    
    

    const int new_M = M_val;
    const int new_Ds = Ds_val;
    const int new_levels = num_levels_val;

    const size_t new_primary_stride = static_cast<size_t>(std::max(0, new_M));
    const size_t new_residual_level_stride = static_cast<size_t>(std::max(0, new_Ds));
    const size_t new_residual_subspace_stride =
        static_cast<size_t>(std::max(0, new_Ds)) * static_cast<size_t>(std::max(0, new_levels - 1));
    const size_t new_residual_stride =
        static_cast<size_t>(std::max(0, new_M)) * new_residual_subspace_stride;

    const bool same_layout =
        is_initialized &&
        M == new_M &&
        Ds == new_Ds &&
        num_levels == new_levels &&
        primary_stride == new_primary_stride &&
        residual_level_stride == new_residual_level_stride &&
        residual_subspace_stride == new_residual_subspace_stride &&
        residual_stride == new_residual_stride;

    M = new_M;
    Ds = new_Ds;
    num_levels = new_levels;
    primary_stride = new_primary_stride;
    residual_level_stride = new_residual_level_stride;
    residual_subspace_stride = new_residual_subspace_stride;
    residual_stride = new_residual_stride;
    residual_packed4_stride = 0;

    const size_t new_total = (ntotal > 0) ? static_cast<size_t>(ntotal) : 0u;

    if (!same_layout) {
        primary_codes.clear();
        residual_codes.clear();
        residual_codes_packed4.clear();
        cross_terms.clear();
        residual_norms.clear();
    } else {
        
        residual_codes_packed4.clear();
        residual_packed4_stride = 0;
    }

    if (new_total > 0) {
        if (same_layout) {
            primary_codes.resize_preserve(new_total * primary_stride);
        } else {
            primary_codes.resize(new_total * primary_stride);
        }
        if (num_levels > 1) {
            if (same_layout) {
                residual_codes.resize_preserve(new_total * residual_stride);
            } else {
                residual_codes.resize(new_total * residual_stride);
            }
            cross_terms.resize(new_total, 0.0f);
            
            if (!residual_norms.empty()) {
                residual_norms.resize(new_total, 0.0f);
            }
        } else {
            
            residual_codes.clear();
            cross_terms.clear();
            residual_norms.clear();
        }
    } else {
        
        primary_codes.clear();
        residual_codes.clear();
        residual_codes_packed4.clear();
        cross_terms.clear();
        residual_norms.clear();
    }

    is_initialized = true;
}

void PreDecodedCodes::clear()
{
    primary_codes.clear();
    residual_codes.clear();
    residual_codes_packed4.clear();
    cross_terms.clear();
    residual_norms.clear();
    residual_packed4_stride = 0;
    is_initialized = false;
}

bool PreDecodedCodes::empty() const
{
    return !is_initialized || primary_codes.empty();
}

size_t PreDecodedCodes::memory_usage() const
{
    return primary_codes.size() + residual_codes.size()
        + residual_codes_packed4.size()
        + cross_terms.size() * sizeof(float)
        + residual_norms.size() * sizeof(float)
        + sizeof(*this);
}

}

thread_local IndexJHQ::SearchWorkspace IndexJHQ::workspace_;

void IndexJHQ::compress_rotation_to_bf16()
{
    if (rotation_matrix.empty()) {
        return;
    }
    const size_t n = rotation_matrix.size();
    rotation_matrix_bf16.resize(n);
    for (size_t i = 0; i < n; ++i) {
        rotation_matrix_bf16[i] = jhq_internal::float_to_bf16(rotation_matrix[i]);
    }
    rotation_matrix.clear();
    rotation_matrix.shrink_to_fit();
    use_bf16_rotation = true;
}

IndexJHQ::IndexJHQ()
    : IndexFlatCodes()
    , M(0)
    , Ds(0)
    , num_levels(0)
    , use_jl_transform(false)
    , use_analytical_init(false)
    , default_oversampling(4.0f)
    , verbose(false)
    , use_bf16_rotation(false)
    , is_rotation_trained(false)
    , use_kmeans_refinement(true)
    , kmeans_niter(5)
    , kmeans_nredo(1)
    , kmeans_seed(1234)
    , sample_primary(0)
    , sample_residual(20000)
    , random_sample_training(true)
    , residual_bits_per_subspace(0)
    , memory_layout_initialized_(false)
    , primary_codewords_stride_(0)
    , scalar_codebooks_stride_(0)
    , residual_codes_stride_(0)
{
    this->d = 0;
    this->metric_type = METRIC_L2;
    code_size = 0;
}

IndexJHQ::IndexJHQ(
    int d,
    int M,
    const std::vector<int>& level_bits,
    bool use_jl_transform,
    float default_oversampling,
    bool use_analytical_init,
    bool verbose,
    MetricType metric)
    : IndexFlatCodes()
    , M(M)
    , Ds(d / M)
    , num_levels(static_cast<int>(level_bits.size()))
    , level_bits(level_bits)
    , use_jl_transform(use_jl_transform)
    , use_analytical_init(use_analytical_init)
    , default_oversampling(default_oversampling)
    , verbose(verbose)
    , use_bf16_rotation(false)
    , is_rotation_trained(false)
    , residual_pq_dirty_(true)
    , use_kmeans_refinement(true)
    , kmeans_niter(5)
    , kmeans_nredo(1)
    , kmeans_seed(1234)
    , sample_primary(0)
    , sample_residual(20000)
    , random_sample_training(true)
    , residual_bits_per_subspace(0)
    , memory_layout_initialized_(false)
    , primary_codewords_stride_(0)
    , scalar_codebooks_stride_(0)
    , residual_codes_stride_(0)
{
    this->d = d;
    this->metric_type = metric;

    validate_parameters();
    initialize_data_structures();
    compute_code_size();
}

IndexJHQ::IndexJHQ(const IndexJHQ& other)
    : IndexFlatCodes(other)
    , M(other.M)
    , Ds(other.Ds)
    , num_levels(other.num_levels)
    , level_bits(other.level_bits)
    , use_jl_transform(other.use_jl_transform)
    , normalize_l2(other.normalize_l2)
    , use_analytical_init(other.use_analytical_init)
    , default_oversampling(other.default_oversampling)
    , verbose(other.verbose)
    , rotation_matrix(other.rotation_matrix)
    , rotation_matrix_bf16(other.rotation_matrix_bf16)
    , use_bf16_rotation(other.use_bf16_rotation)
    , is_rotation_trained(other.is_rotation_trained)
    , scalar_codebooks_flat_(other.scalar_codebooks_flat_)
    , scalar_codebook_level_offsets_(other.scalar_codebook_level_offsets_)
    , scalar_codebooks_flat_valid_(other.scalar_codebooks_flat_valid_)
    , residual_pq_dirty_(other.residual_pq_dirty_)
    , primary_pq_dirty_(other.primary_pq_dirty_)
    , use_kmeans_refinement(other.use_kmeans_refinement)
    , kmeans_niter(other.kmeans_niter)
    , kmeans_nredo(other.kmeans_nredo)
    , kmeans_seed(other.kmeans_seed)
    , sample_primary(other.sample_primary)
    , sample_residual(other.sample_residual)
    , random_sample_training(other.random_sample_training)
    , residual_bits_per_subspace(other.residual_bits_per_subspace)
    , memory_layout_initialized_(other.memory_layout_initialized_)
    , primary_codewords_stride_(other.primary_codewords_stride_)
    , scalar_codebooks_stride_(other.scalar_codebooks_stride_)
    , residual_codes_stride_(other.residual_codes_stride_)
{
    if (other.primary_pq_) {
        primary_pq_ = std::make_unique<ProductQuantizer>(*other.primary_pq_);
    }
    if (other.residual_pq_) {
        residual_pq_ = std::make_unique<ProductQuantizer>(*other.residual_pq_);
    }

    if (other.has_pre_decoded_codes()) {
        separated_codes_.initialize(other.separated_codes_.M,
                                    other.separated_codes_.Ds,
                                    other.separated_codes_.num_levels,
                                    other.ntotal);
        std::memcpy(separated_codes_.primary_codes.data(),
                    other.separated_codes_.primary_codes.data(),
                    other.separated_codes_.primary_codes.size());
        if (other.separated_codes_.num_levels > 1) {
            std::memcpy(separated_codes_.residual_codes.data(),
                        other.separated_codes_.residual_codes.data(),
                        other.separated_codes_.residual_codes.size());
        }
        if (other.separated_codes_.has_residual_codes_packed4()) {
            separated_codes_.residual_packed4_stride = other.separated_codes_.residual_packed4_stride;
            separated_codes_.residual_codes_packed4.resize(
                other.separated_codes_.residual_codes_packed4.size());
            std::memcpy(separated_codes_.residual_codes_packed4.data(),
                        other.separated_codes_.residual_codes_packed4.data(),
                        other.separated_codes_.residual_codes_packed4.size());
        }
        separated_codes_.cross_terms = other.separated_codes_.cross_terms;
        separated_codes_.residual_norms = other.separated_codes_.residual_norms;
    } else {
        separated_codes_.clear();
        memory_layout_initialized_ = false;
    }
}

IndexJHQ& IndexJHQ::operator=(const IndexJHQ& other)
{
    if (this != &other) {
        IndexFlatCodes::operator=(other);
        M = other.M;
        Ds = other.Ds;
        num_levels = other.num_levels;
        level_bits = other.level_bits;
        use_jl_transform = other.use_jl_transform;
        normalize_l2 = other.normalize_l2;
        use_analytical_init = other.use_analytical_init;
        default_oversampling = other.default_oversampling;
        verbose = other.verbose;
        rotation_matrix = other.rotation_matrix;
        rotation_matrix_bf16 = other.rotation_matrix_bf16;
        use_bf16_rotation = other.use_bf16_rotation;
        is_rotation_trained = other.is_rotation_trained;
        primary_pq_dirty_ = other.primary_pq_dirty_;
        residual_pq_dirty_ = other.residual_pq_dirty_;
        use_kmeans_refinement = other.use_kmeans_refinement;
        kmeans_niter = other.kmeans_niter;
        kmeans_nredo = other.kmeans_nredo;
        kmeans_seed = other.kmeans_seed;
        sample_primary = other.sample_primary;
        sample_residual = other.sample_residual;
        random_sample_training = other.random_sample_training;
        residual_bits_per_subspace = other.residual_bits_per_subspace;
        memory_layout_initialized_ = other.memory_layout_initialized_;
        primary_codewords_stride_ = other.primary_codewords_stride_;
        scalar_codebooks_stride_ = other.scalar_codebooks_stride_;
        residual_codes_stride_ = other.residual_codes_stride_;
        scalar_codebooks_flat_ = other.scalar_codebooks_flat_;
        scalar_codebook_level_offsets_ = other.scalar_codebook_level_offsets_;
        scalar_codebooks_flat_valid_ = other.scalar_codebooks_flat_valid_;
        if (other.primary_pq_) {
            primary_pq_ = std::make_unique<ProductQuantizer>(*other.primary_pq_);
        } else {
            primary_pq_.reset();
        }
        if (other.residual_pq_) {
            residual_pq_ = std::make_unique<ProductQuantizer>(*other.residual_pq_);
        } else {
            residual_pq_.reset();
        }

        if (other.has_pre_decoded_codes()) {
            separated_codes_.initialize(other.separated_codes_.M,
                                        other.separated_codes_.Ds,
                                        other.separated_codes_.num_levels,
                                        other.ntotal);
            std::memcpy(separated_codes_.primary_codes.data(),
                        other.separated_codes_.primary_codes.data(),
                        other.separated_codes_.primary_codes.size());
            if (other.separated_codes_.num_levels > 1) {
                std::memcpy(separated_codes_.residual_codes.data(),
                            other.separated_codes_.residual_codes.data(),
                            other.separated_codes_.residual_codes.size());
            }
            if (other.separated_codes_.has_residual_codes_packed4()) {
                separated_codes_.residual_packed4_stride = other.separated_codes_.residual_packed4_stride;
                separated_codes_.residual_codes_packed4.resize(
                    other.separated_codes_.residual_codes_packed4.size());
                std::memcpy(separated_codes_.residual_codes_packed4.data(),
                            other.separated_codes_.residual_codes_packed4.data(),
                            other.separated_codes_.residual_codes_packed4.size());
            }
            separated_codes_.cross_terms = other.separated_codes_.cross_terms;
            separated_codes_.residual_norms = other.separated_codes_.residual_norms;
        } else {
            separated_codes_.clear();
            memory_layout_initialized_ = false;
        }
    }
    return *this;
}

IndexJHQ::~IndexJHQ() = default;

void IndexJHQ::validate_parameters() const
{
    FAISS_THROW_IF_NOT_MSG(d > 0, "Vector dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(M > 0, "Number of subspaces must be positive");
    FAISS_THROW_IF_NOT_MSG(d % M == 0, "Vector dimension must be divisible by number of subspaces");
    FAISS_THROW_IF_NOT_MSG(num_levels > 0, "Must have at least one quantization level");
    FAISS_THROW_IF_NOT_MSG(default_oversampling >= 1.0f, "Oversampling factor must be >= 1.0");

    for (int i = 0; i < num_levels; ++i) {
        FAISS_THROW_IF_NOT_MSG(level_bits[i] > 0 && level_bits[i] <= 16,
            "Level bits must be in range [1, 16]");
    }

    FAISS_THROW_IF_NOT_MSG(metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT,
        "Only L2 and inner product metrics are supported");
}

void IndexJHQ::initialize_data_structures()
{
    if (use_jl_transform) {
        rotation_matrix.resize(static_cast<size_t>(d) * d);
    }

    if (num_levels > 0 && !level_bits.empty()) {
        primary_pq_ = std::make_unique<ProductQuantizer>(
            static_cast<size_t>(d),
            static_cast<size_t>(M),
            static_cast<size_t>(level_bits[0]));
    } else {
        primary_pq_.reset();
    }

    scalar_codebooks_flat_.clear();
    scalar_codebook_level_offsets_.clear();
    scalar_codebooks_flat_valid_ = false;

    residual_pq_dirty_ = true;
    primary_pq_dirty_ = false;
    residual_pq_.reset();
    scalar_codebook_level_offsets_.assign(static_cast<size_t>(num_levels), 0u);
    size_t stride = 0;
    for (int level = 1; level < num_levels; ++level) {
        scalar_codebook_level_offsets_[static_cast<size_t>(level)] = stride;
        stride += static_cast<size_t>(scalar_codebook_ksub(level));
    }
    scalar_codebooks_stride_ = stride;
    scalar_codebooks_flat_.assign(static_cast<size_t>(M) * stride, 0.0f);
    scalar_codebooks_flat_valid_ = true;
}

void IndexJHQ::compute_code_size()
{
    int total_bits = 0;
    for (int l = 0; l < num_levels; ++l) {
        if (l == 0) {
            total_bits += M * level_bits[l];
        } else {
            total_bits += d * level_bits[l];
        }
    }
    code_size = (total_bits + 7) / 8;
}

int IndexJHQ::scalar_codebook_ksub(int level) const
{
    FAISS_THROW_IF_NOT_MSG(level > 0 && level < num_levels, "scalar_codebook_ksub: level out of bounds");
    return 1 << level_bits[static_cast<size_t>(level)];
}

void IndexJHQ::rebuild_scalar_codebooks_flat()
{
    if (num_levels <= 1 || M <= 0) {
        scalar_codebooks_stride_ = 0;
        scalar_codebooks_flat_.clear();
        scalar_codebook_level_offsets_.clear();
        scalar_codebooks_flat_valid_ = true;
        return;
    }

    std::vector<size_t> new_offsets(static_cast<size_t>(num_levels), 0u);
    size_t stride = 0;
    for (int level = 1; level < num_levels; ++level) {
        new_offsets[static_cast<size_t>(level)] = stride;
        stride += static_cast<size_t>(1u << level_bits[static_cast<size_t>(level)]);
    }
    scalar_codebooks_stride_ = stride;
    scalar_codebook_level_offsets_ = std::move(new_offsets);

    if (stride == 0) {
        scalar_codebooks_flat_.clear();
        scalar_codebooks_flat_valid_ = true;
        return;
    }

    const size_t expected_size = static_cast<size_t>(M) * stride;
    if (scalar_codebooks_flat_valid_ &&
        scalar_codebooks_flat_.size() == expected_size) {
        return;
    }

    scalar_codebooks_flat_.assign(expected_size, 0.0f);
    scalar_codebooks_flat_valid_ = true;
}

const float* IndexJHQ::get_scalar_codebook_ptr(int subspace_idx, int level) const
{
    FAISS_THROW_IF_NOT_MSG(subspace_idx >= 0 && subspace_idx < M, "get_scalar_codebook_ptr: subspace_idx out of bounds");
    FAISS_THROW_IF_NOT_MSG(level > 0 && level < num_levels, "get_scalar_codebook_ptr: level out of bounds");

    if (scalar_codebooks_flat_valid_ &&
        scalar_codebooks_stride_ > 0 &&
        scalar_codebook_level_offsets_.size() >= static_cast<size_t>(num_levels) &&
        !scalar_codebooks_flat_.empty()) {
        return scalar_codebooks_flat_.data() +
            static_cast<size_t>(subspace_idx) * scalar_codebooks_stride_ +
            scalar_codebook_level_offsets_[static_cast<size_t>(level)];
    }

    FAISS_THROW_MSG("get_scalar_codebook_ptr: flat scalar codebooks unavailable");
}

float* IndexJHQ::get_scalar_codebook_ptr_mutable(int subspace_idx, int level)
{
    FAISS_THROW_IF_NOT_MSG(
        subspace_idx >= 0 && subspace_idx < M,
        "get_scalar_codebook_ptr_mutable: subspace_idx out of bounds");
    FAISS_THROW_IF_NOT_MSG(
        level > 0 && level < num_levels,
        "get_scalar_codebook_ptr_mutable: level out of bounds");

    const size_t expected_size = static_cast<size_t>(M) * scalar_codebooks_stride_;
    if (!scalar_codebooks_flat_valid_ ||
        scalar_codebooks_flat_.size() != expected_size) {
        scalar_codebooks_flat_.assign(expected_size, 0.0f);
        scalar_codebooks_flat_valid_ = true;
    }
    return scalar_codebooks_flat_.data() +
        static_cast<size_t>(subspace_idx) * scalar_codebooks_stride_ +
        scalar_codebook_level_offsets_[static_cast<size_t>(level)];
}

bool IndexJHQ::is_trained_() const
{
    return is_trained;
}

}  
