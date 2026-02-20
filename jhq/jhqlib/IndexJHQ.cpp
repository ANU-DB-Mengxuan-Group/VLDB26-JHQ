
#include <IndexJHQ.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <immintrin.h>

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

namespace faiss {

namespace jhq_internal {

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

float fvec_L2sqr_avx512(const float* x, const float* y, size_t d)
{
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 15 < d; i += 16) {
        __m512 vx = _mm512_loadu_ps(x + i);
        __m512 vy = _mm512_loadu_ps(y + i);
        __m512 vdiff = _mm512_sub_ps(vx, vy);
        acc = _mm512_fmadd_ps(vdiff, vdiff, acc);
    }
    float total_dist = _mm512_reduce_add_ps(acc);
    for (; i < d; ++i) {
        const float diff = x[i] - y[i];
        total_dist += diff * diff;
    }
    return total_dist;
}

float fvec_L2sqr_avx2(const float* x, const float* y, size_t d)
{
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 vdiff = _mm256_sub_ps(vx, vy);
        acc = _mm256_fmadd_ps(vdiff, vdiff, acc);
    }
    __m128 sum_128 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
    __m128 sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
    float total_dist = _mm_cvtss_f32(sum_32);
    for (; i < d; ++i) {
        const float diff = x[i] - y[i];
        total_dist += diff * diff;
    }
    return total_dist;
}

float fvec_L2sqr_dispatch(const float* x, const float* y, size_t d)
{
#if defined(HAVE_AVX512)
    return fvec_L2sqr_avx512(x, y, d);
#elif defined(HAVE_AVX2)
    return fvec_L2sqr_avx2(x, y, d);
#else
    return faiss::fvec_L2sqr(x, y, d);
#endif
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

    for (int m = 0; m < index.M; ++m) {
        const auto& centroids = index.codewords[m][0];
        if (centroids.empty()) {
            continue;
        }

        const int num_centroids = static_cast<int>(centroids.size() / index.Ds);
        const uint32_t centroid_id = std::min<uint32_t>(
            primary_codes[m], static_cast<uint32_t>(std::max(0, num_centroids - 1)));
        const float* centroid_ptr = centroids.data() + static_cast<size_t>(centroid_id) * index.Ds;

        const uint8_t* subspace_residual = residual_codes + m * residual_subspace_stride;

        for (int d = 0; d < index.Ds; ++d) {
            float residual_value = 0.0f;
            for (int level = 1; level < index.num_levels; ++level) {
                const auto& codebook = index.scalar_codebooks[m][level - 1];
                if (codebook.empty()) {
                    continue;
                }
                const uint8_t scalar_id = subspace_residual[(level - 1) * residual_level_stride + d];
                const size_t max_idx = codebook.size() - 1;
                const size_t safe_idx = std::min(static_cast<size_t>(scalar_id), max_idx);
                residual_value += codebook[safe_idx];
            }
            dot += centroid_ptr[d] * residual_value;
        }
    }

    return 2.0f * dot;
}

PreDecodedCodes::PreDecodedCodes()
    : primary_stride(0)
    , residual_stride(0)
    , residual_level_stride(0)
    , residual_subspace_stride(0)
    , num_levels(0)
    , M(0)
    , Ds(0)
    , is_initialized(false)
{
}

void PreDecodedCodes::initialize(int M_val, int Ds_val, int num_levels_val, idx_t ntotal)
{
    M = M_val;
    Ds = Ds_val;
    num_levels = num_levels_val;

    primary_stride = M;
    residual_level_stride = Ds;
    residual_subspace_stride = Ds * (num_levels - 1);
    residual_stride = M * residual_subspace_stride;

    if (ntotal > 0) {
        primary_codes.resize(ntotal * primary_stride);
        if (num_levels > 1) {
            residual_codes.resize(ntotal * residual_stride);
        }
    }

    is_initialized = true;
}

void PreDecodedCodes::clear()
{
    primary_codes.clear();
    residual_codes.clear();
    is_initialized = false;
}

bool PreDecodedCodes::empty() const
{
    return !is_initialized || primary_codes.empty();
}

size_t PreDecodedCodes::memory_usage() const
{
    return primary_codes.size() + residual_codes.size() + sizeof(*this);
}

}

thread_local IndexJHQ::SearchWorkspace IndexJHQ::workspace_;

IndexJHQ::IndexJHQ()
    : IndexFlatCodes()
    , M(0)
    , Ds(0)
    , num_levels(0)
    , use_jl_transform(false)
    , use_analytical_init(false)
    , default_oversampling(4.0f)
    , verbose(false)
    , is_rotation_trained(false)
    , residual_bits_per_subspace(0)
    , memory_layout_initialized_(false)
    , primary_codewords_stride_(0)
    , scalar_codebooks_stride_(0)
    , residual_codes_stride_(0)
    , use_kmeans_refinement(false)
    , kmeans_niter(25)
    , kmeans_seed(1234)
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
    , is_rotation_trained(false)
    , memory_layout_initialized_(false)
{
    this->d = d;
    this->metric_type = metric;

    if (verbose) {
        std::cout << "Initializing JHQ: d=" << d << ", M=" << M
                  << ", levels=" << num_levels << std::endl;
    }

    validate_parameters();
    initialize_data_structures();
    compute_code_size();

    if (verbose) {
        std::cout << "JHQ initialized: code_size=" << code_size
                  << " bytes, compression=" << (32.0f * d) / (8.0f * code_size)
                  << "x" << std::endl;
    }
}

IndexJHQ::IndexJHQ(const IndexJHQ& other)
    : IndexFlatCodes(other)
    , M(other.M)
    , Ds(other.Ds)
    , num_levels(other.num_levels)
    , level_bits(other.level_bits)
    , use_jl_transform(other.use_jl_transform)
    , use_analytical_init(other.use_analytical_init)
    , default_oversampling(other.default_oversampling)
    , verbose(other.verbose)
    , rotation_matrix(other.rotation_matrix)
    , is_rotation_trained(other.is_rotation_trained)
    , codewords(other.codewords)
    , scalar_codebooks(other.scalar_codebooks)
    , residual_pq_dirty_(true)
{
    if (other.residual_pq_) {
        residual_pq_ = std::make_unique<ProductQuantizer>(*other.residual_pq_);
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
        use_analytical_init = other.use_analytical_init;
        default_oversampling = other.default_oversampling;
        verbose = other.verbose;
        rotation_matrix = other.rotation_matrix;
        is_rotation_trained = other.is_rotation_trained;
        codewords = other.codewords;
        scalar_codebooks = other.scalar_codebooks;
        residual_pq_dirty_ = true;
        if (other.residual_pq_) {
            residual_pq_ = std::make_unique<ProductQuantizer>(*other.residual_pq_);
        } else {
            residual_pq_.reset();
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

    codewords.resize(M);
    scalar_codebooks.resize(M);

    for (int m = 0; m < M; ++m) {
        codewords[m].resize(num_levels);
        scalar_codebooks[m].resize(num_levels - 1);

        for (int l = 0; l < num_levels; ++l) {
            int K = 1 << level_bits[l];

            if (l == 0) {
                codewords[m][l].resize(static_cast<size_t>(K) * Ds);
            } else {
                scalar_codebooks[m][l - 1].resize(K);
                codewords[m][l].clear();
            }
        }
    }

    residual_pq_dirty_ = true;
    residual_pq_.reset();
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

    if (verbose) {
        std::cout << "Total bits per vector: " << total_bits
                  << " (" << code_size << " bytes)" << std::endl;
    }
}

bool IndexJHQ::is_trained_() const
{
    return is_trained;
}

void IndexJHQ::train(idx_t n, const float* x)
{
    if (verbose) {
        std::cout << "\n=== Training JHQ on " << n << " vectors ===" << std::endl;
    }

    FAISS_THROW_IF_NOT_MSG(n > 0, "Training set cannot be empty");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Training data cannot be null");

    if (use_jl_transform) {
        generate_qr_rotation_matrix(1234);
    }

    const size_t memory_per_vector = d * sizeof(float);
    const size_t max_memory_bytes = jhq_internal::MAX_TRAIN_SIZE_GB * 1024 * 1024 * 1024;
    const idx_t batch_size = std::min(
        n,
        std::max(static_cast<idx_t>(1000),
            static_cast<idx_t>(max_memory_bytes / (3 * memory_per_vector))));

    if (verbose) {
        std::cout << "Training with batch size: " << batch_size << " vectors" << std::endl;
    }

    std::vector<std::vector<float>> subspace_data(M);
    for (int m = 0; m < M; ++m) {
        subspace_data[m].reserve(n * Ds);
    }

    for (idx_t batch_start = 0; batch_start < n; batch_start += batch_size) {
        const idx_t batch_end = std::min(n, batch_start + batch_size);
        const idx_t current_batch_size = batch_end - batch_start;

        std::vector<float> x_batch_rotated(current_batch_size * d);

        apply_jl_rotation(current_batch_size,
            x + batch_start * d,
            x_batch_rotated.data());

        for (idx_t i = 0; i < current_batch_size; ++i) {
            for (int m = 0; m < M; ++m) {
                const float* subspace_start = x_batch_rotated.data() + i * d + m * Ds;
                subspace_data[m].insert(subspace_data[m].end(),
                    subspace_start,
                    subspace_start + Ds);
            }
        }

        if (verbose && batch_start + batch_size < n) {
            std::cout << "Processed batch " << (batch_start / batch_size + 1)
                      << "/" << ((n + batch_size - 1) / batch_size) << std::endl;
        }
    }

    if (verbose) {
        std::cout << "Training " << M << " subspace quantizers..." << std::endl;
    }

#pragma omp parallel for
    for (int m = 0; m < M; ++m) {
        train_subspace_quantizers(m, n, subspace_data[m].data(), 42 + m);
    }

    is_trained = true;
    initialize_memory_layout();
    mark_residual_tables_dirty();

    if (verbose) {
        std::cout << "=== Training Complete ===" << std::endl;
    }
}

void IndexJHQ::train_subspace_quantizers(
    int subspace_idx,
    idx_t n,
    const float* subspace_data,
    int random_seed)
{
    std::vector<float> current_residuals(subspace_data, subspace_data + n * Ds);

    for (int level = 0; level < num_levels; ++level) {
        int K = 1 << level_bits[level];

        if (level == 0) {
            train_primary_level(subspace_idx, n, current_residuals.data(), K, random_seed + level);
        } else {
            train_residual_level(subspace_idx, level, n, current_residuals.data(), K);
        }

        if (level < num_levels - 1) {
            update_residuals_after_level(subspace_idx, level, n, current_residuals.data());
        }
    }

    mark_residual_tables_dirty();
}

void IndexJHQ::train_primary_level(
    int subspace_idx,
    idx_t n,
    const float* data,
    int K,
    int random_seed)
{
    ClusteringParameters cp;

    if (use_kmeans_refinement) {
        cp.niter = kmeans_niter;
        cp.nredo = 1;
        cp.seed = kmeans_seed + random_seed;

        if (verbose && subspace_idx == 0) {
            std::cout << "  Using analytical initialization + k-means refinement ("
                      << cp.niter << " iterations)" << std::endl;
        }
    } else {
        cp.niter = 0;
        cp.nredo = 0;
        cp.seed = kmeans_seed + random_seed;

        if (verbose && subspace_idx == 0) {
            std::cout << "  Using analytical initialization only (no k-means)" << std::endl;
        }
    }

    Clustering clustering(Ds, K, cp);

    std::vector<float> init_centroids(K * Ds);
    analytical_gaussian_init(data, n, Ds, K, init_centroids.data());
    clustering.centroids.assign(init_centroids.begin(), init_centroids.end());

    IndexFlatL2 clustering_index(Ds);

    try {
        clustering.train(n, data, clustering_index);
    } catch (const FaissException& e) {
        if (verbose) {
            std::cerr << "K-means failed for subspace " << subspace_idx << std::endl;
        }

        std::mt19937 rng(random_seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (int i = 0; i < K * Ds; ++i) {
            clustering.centroids[i] = dist(rng);
        }
    }

    FAISS_THROW_IF_NOT(clustering.centroids.size() == K * Ds);
    codewords[subspace_idx][0] = clustering.centroids;
}

void IndexJHQ::train_residual_level(
    int subspace_idx,
    int level,
    idx_t n,
    const float* residuals,
    int K)
{
    std::vector<float>& scalar_codebook = scalar_codebooks[subspace_idx][level - 1];
    if (K <= 0) {
        scalar_codebook.clear();
        mark_residual_tables_dirty();
        return;
    }

    scalar_codebook.resize(K);

    const size_t total_values = static_cast<size_t>(n) * static_cast<size_t>(Ds);

    if (total_values == 0) {
        std::fill(scalar_codebook.begin(), scalar_codebook.end(), 0.0f);
    } else if (use_analytical_init && K > 2) {
        const size_t max_samples = std::min(total_values, static_cast<size_t>(10000));
        std::vector<float> samples;
        samples.reserve(max_samples);

        const size_t stride = std::max(static_cast<size_t>(1), total_values / max_samples);

        for (size_t i = 0; i < total_values; i += stride) {
            samples.push_back(residuals[i]);
            if (samples.size() >= max_samples)
                break;
        }

        std::sort(samples.begin(), samples.end());

        for (int k = 0; k < K; ++k) {
            float quantile = static_cast<float>(k) / (K - 1);
            size_t idx = static_cast<size_t>(quantile * (samples.size() - 1));
            scalar_codebook[k] = samples[idx];
        }
    } else {
#ifdef __AVX512F__
        __m512 vmin = _mm512_set1_ps(residuals[0]);
        __m512 vmax = _mm512_set1_ps(residuals[0]);

        size_t i = 0;
        for (; i + 15 < total_values; i += 16) {
            __m512 vals = _mm512_loadu_ps(&residuals[i]);
            vmin = _mm512_min_ps(vmin, vals);
            vmax = _mm512_max_ps(vmax, vals);
        }

        float min_val = _mm512_reduce_min_ps(vmin);
        float max_val = _mm512_reduce_max_ps(vmax);

        for (; i < total_values; ++i) {
            min_val = std::min(min_val, residuals[i]);
            max_val = std::max(max_val, residuals[i]);
        }

#else
        float min_val = residuals[0];
        float max_val = residuals[0];
        for (size_t i = 1; i < total_values; ++i) {
            min_val = std::min(min_val, residuals[i]);
            max_val = std::max(max_val, residuals[i]);
        }
#endif

        if (max_val - min_val < 1e-8f) {
            std::fill(scalar_codebook.begin(), scalar_codebook.end(), min_val);
        } else {
            for (int k = 0; k < K; ++k) {
                scalar_codebook[k] = min_val + (max_val - min_val) * k / (K - 1);
            }
        }
    }

    mark_residual_tables_dirty();
}

void IndexJHQ::update_residuals_after_level(
    int subspace_idx,
    int level,
    idx_t n,
    float* residuals)
{
    if (level == 0) {
        const std::vector<float>& centroids = codewords[subspace_idx][0];
        int K = static_cast<int>(centroids.size() / Ds);

        IndexFlatL2 quantizer(Ds);
        quantizer.add(K, centroids.data());

        std::vector<idx_t> labels(n);
        std::vector<float> dis(n);
        quantizer.search(n, residuals, 1, dis.data(), labels.data());

#pragma omp parallel for
        for (idx_t i = 0; i < n; ++i) {
            const float* best_centroid = centroids.data() + labels[i] * Ds;
            float* current_residual = residuals + i * Ds;
            for (int d = 0; d < Ds; ++d) {
                current_residual[d] -= best_centroid[d];
            }
        }
        return;
    }

    const std::vector<float>& scalar_codebook = scalar_codebooks[subspace_idx][level - 1];
    const int K = static_cast<int>(scalar_codebook.size());

#pragma omp parallel for schedule(static) if (n > 10000)
    for (idx_t i = 0; i < n; ++i) {
        float* current_residual = residuals + i * Ds;

        int d = 0;

#if defined(__AVX512F__)
        constexpr int VECTOR_BLOCK_SIZE = 64;
        const __m512 sign_mask = _mm512_set1_ps(-0.0f);

        alignas(64) float codebook_aligned[K];
        std::memcpy(codebook_aligned, scalar_codebook.data(), K * sizeof(float));

        for (; d + 15 < Ds; d += 16) {
            if (d + 32 < Ds) {
                _mm_prefetch(&current_residual[d + 16], _MM_HINT_T0);
            }

            __m512 vals = _mm512_loadu_ps(&current_residual[d]);
            __m512i best_indices = _mm512_setzero_si512();
            __m512 best_dists = _mm512_set1_ps(FLT_MAX);

            int k = 0;

            for (; k + 7 < K; k += 8) {
                if (k + 16 < K) {
                    _mm_prefetch(&codebook_aligned[k + 8], _MM_HINT_T0);
                }

#pragma unroll 8
                for (int kk = 0; kk < 8; ++kk) {
                    __m512 codebook_val = _mm512_set1_ps(codebook_aligned[k + kk]);
                    __m512 diffs = _mm512_sub_ps(vals, codebook_val);
                    __m512 abs_diffs = _mm512_abs_ps(diffs);

                    __mmask16 mask = _mm512_cmp_ps_mask(abs_diffs, best_dists, _CMP_LT_OQ);
                    best_dists = _mm512_mask_blend_ps(mask, best_dists, abs_diffs);
                    best_indices = _mm512_mask_blend_epi32(mask, best_indices, _mm512_set1_epi32(k + kk));
                }
            }

            for (; k < K; ++k) {
                __m512 codebook_val = _mm512_set1_ps(codebook_aligned[k]);
                __m512 diffs = _mm512_sub_ps(vals, codebook_val);
                __m512 abs_diffs = _mm512_abs_ps(diffs);

                __mmask16 mask = _mm512_cmp_ps_mask(abs_diffs, best_dists, _CMP_LT_OQ);
                best_dists = _mm512_mask_blend_ps(mask, best_dists, abs_diffs);
                best_indices = _mm512_mask_blend_epi32(mask, best_indices, _mm512_set1_epi32(k));
            }

            alignas(64) int indices[16];
            _mm512_store_si512((__m512i*)indices, best_indices);

            alignas(64) float quantized_vals[16];
            for (int j = 0; j < 16; ++j) {
                quantized_vals[j] = codebook_aligned[indices[j]];
            }

            __m512 quantized = _mm512_load_ps(quantized_vals);
            __m512 result = _mm512_sub_ps(vals, quantized);
            _mm512_storeu_ps(&current_residual[d], result);
        }
#elif defined(__AVX2__)
        const __m256 sign_mask = _mm256_set1_ps(-0.0f);

        for (; d + 7 < Ds; d += 8) {
            __m256 vals = _mm256_loadu_ps(&current_residual[d]);
            __m256i best_indices = _mm256_setzero_si256();
            __m256 best_dists = _mm256_set1_ps(FLT_MAX);

            for (int k = 0; k < K; ++k) {
                __m256 codebook_val = _mm256_broadcast_ss(&scalar_codebook[k]);
                __m256 diffs = _mm256_sub_ps(vals, codebook_val);
                __m256 abs_diffs = _mm256_andnot_ps(sign_mask, diffs);

                __m256 mask = _mm256_cmp_ps(abs_diffs, best_dists, _CMP_LT_OQ);
                best_dists = _mm256_blendv_ps(best_dists, abs_diffs, mask);
                best_indices = _mm256_blendv_epi8(best_indices,
                    _mm256_set1_epi32(k),
                    _mm256_castps_si256(mask));
            }

            alignas(32) int indices[8];
            _mm256_store_si256((__m256i*)indices, best_indices);

            for (int j = 0; j < 8; ++j) {
                current_residual[d + j] -= scalar_codebook[indices[j]];
            }
        }
#endif

        for (; d < Ds; ++d) {
            float val = current_residual[d];
            int best_k = 0;
            float best_dist = std::abs(val - scalar_codebook[0]);

            for (int k = 1; k < K; ++k) {
                float dist = std::abs(val - scalar_codebook[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }
            current_residual[d] -= scalar_codebook[best_k];
        }
    }
}

void IndexJHQ::generate_qr_rotation_matrix(int random_seed)
{
    if (!use_jl_transform) {
        is_rotation_trained = true;
        return;
    }

    if (verbose) {
        std::cout << "Generating QR-based Johnson-Lindenstrauss rotation matrix..." << std::endl;
    }

    std::mt19937 rng(random_seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    rotation_matrix.resize(static_cast<size_t>(d) * d);
    for (size_t i = 0; i < rotation_matrix.size(); ++i) {
        rotation_matrix[i] = dist(rng);
    }

    std::vector<float> tau(std::min(d, d));

    lapack_int info = LAPACKE_sgeqrf(
        LAPACK_ROW_MAJOR, d, d,
        rotation_matrix.data(), d, tau.data());

    if (info == 0) {
        info = LAPACKE_sorgqr(
            LAPACK_ROW_MAJOR, d, d, d,
            rotation_matrix.data(), d, tau.data());
    }

    if (info != 0 && verbose) {
        std::cerr << "QR decomposition failed with info = " << info << std::endl;
    }

    is_rotation_trained = true;

    if (verbose) {
        std::cout << "QR rotation matrix generated" << std::endl;
    }
}

void IndexJHQ::apply_jl_rotation(idx_t n, const float* x_in, float* x_out) const
{
    if (!use_jl_transform || !is_rotation_trained || !x_in || !x_out) {
        if (x_in != x_out && x_in && x_out) {
            std::memcpy(x_out, x_in, sizeof(float) * n * d);
        }
        return;
    }

    if (rotation_matrix.empty() || rotation_matrix.size() != static_cast<size_t>(d) * d) {
        if (verbose) {
            std::cerr << "ERROR: Invalid rotation matrix size: " << rotation_matrix.size()
                      << ", expected: " << (static_cast<size_t>(d) * d) << std::endl;
        }
        if (x_in != x_out) {
            std::memcpy(x_out, x_in, sizeof(float) * n * d);
        }
        return;
    }

    const size_t memory_per_vector_mb = (2 * d * sizeof(float)) / (1024 * 1024);
    const size_t max_memory_mb = jhq_internal::MAX_ROTATE_SIZE_GB * 1024;
    const size_t max_batch_size = memory_per_vector_mb > 0 ? max_memory_mb / memory_per_vector_mb : n;

    const size_t batch_size = std::min(static_cast<size_t>(n),
        std::max(max_batch_size, static_cast<size_t>(1)));

    if (batch_size >= static_cast<size_t>(n)) {
        if (n == 1) {
            const float* rot_matrix = rotation_matrix.data();
            for (int i = 0; i < d; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < d; ++j) {
                    sum += x_in[j] * rot_matrix[i * d + j];
                }
                x_out[i] = sum;
            }
        } else {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(n), d, d,
                1.0f, x_in, d,
                rotation_matrix.data(), d,
                0.0f, x_out, d);
        }
    } else {
        for (idx_t i = 0; i < n; i += batch_size) {
            const idx_t current_batch = std::min(batch_size, static_cast<size_t>(n - i));
            const float* batch_input = x_in + i * d;
            float* batch_output = x_out + i * d;

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                static_cast<int>(current_batch), d, d,
                1.0f, batch_input, d,
                rotation_matrix.data(), d,
                0.0f, batch_output, d);
        }
    }
}

void IndexJHQ::initialize_memory_layout()
{
    if (memory_layout_initialized_)
        return;

    const int K0 = 1 << level_bits[0];

    primary_codewords_stride_ = K0 * Ds;
    scalar_codebooks_stride_ = 0;
    for (int level = 1; level < num_levels; ++level) {
        scalar_codebooks_stride_ += (1 << level_bits[level]);
    }
    residual_codes_stride_ = M * (num_levels - 1) * Ds;

    residual_bits_per_subspace = 0;
    for (int level = 1; level < num_levels; ++level) {
        residual_bits_per_subspace += static_cast<size_t>(Ds) * level_bits[level];
    }

    memory_layout_initialized_ = true;
}

void IndexJHQ::invalidate_memory_layout()
{
    if (!memory_layout_initialized_) {
        return;
    }

    if (verbose) {
        std::cout << "Invalidating and clearing optimized JHQ memory layout." << std::endl;
    }

    separated_codes_.clear();

    memory_layout_initialized_ = false;
}

void IndexJHQ::sa_encode(idx_t n, const float* x, uint8_t* bytes) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before encoding");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of vectors must be positive");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Input vectors cannot be null");

    std::vector<float> x_rotated(n * d);
    apply_jl_rotation(n, x, x_rotated.data());

    if (bytes == nullptr) {
        encode_to_separated_storage(n, x_rotated.data());
        return;
    }

#pragma omp parallel for schedule(static)
    for (idx_t i = 0; i < n; ++i) {
        encode_single_vector(x_rotated.data() + i * d, bytes + i * code_size);
    }
}

void IndexJHQ::encode_single_vector(const float* x, uint8_t* code) const
{
    faiss::BitstringWriter bit_writer(code, code_size);
    std::memset(code, 0, code_size);

    for (int m = 0; m < M; ++m) {
        const float* subspace_vector = x + m * Ds;
        std::vector<float> current_residual(subspace_vector, subspace_vector + Ds);

        for (int level = 0; level < num_levels; ++level) {
            if (level == 0) {
                const std::vector<float>& centroids = codewords[m][0];
                int K = static_cast<int>(centroids.size() / Ds);

                int best_k = 0;
                float best_dist = std::numeric_limits<float>::max();

#ifdef __AVX512F__
                if (K >= 16 && Ds >= 8) {
                    __m512 best_dists = _mm512_set1_ps(std::numeric_limits<float>::max());
                    __m512i best_indices = _mm512_setzero_si512();

                    for (int k = 0; k + 15 < K; k += 16) {
                        __m512 dists = _mm512_setzero_ps();

                        for (int d = 0; d < Ds; ++d) {
                            __m512 query_val = _mm512_set1_ps(current_residual[d]);

                            const float* centroid_dim_base = centroids.data() + d;

                            __m512i indices = _mm512_set_epi32(
                                (k + 15) * Ds, (k + 14) * Ds, (k + 13) * Ds, (k + 12) * Ds,
                                (k + 11) * Ds, (k + 10) * Ds, (k + 9) * Ds, (k + 8) * Ds,
                                (k + 7) * Ds, (k + 6) * Ds, (k + 5) * Ds, (k + 4) * Ds,
                                (k + 3) * Ds, (k + 2) * Ds, (k + 1) * Ds, k * Ds);

                            __m512 cent_vals = _mm512_i32gather_ps(indices, centroid_dim_base, 4);

                            __m512 diff = _mm512_sub_ps(query_val, cent_vals);
                            dists = _mm512_fmadd_ps(diff, diff, dists);
                        }

                        __mmask16 mask = _mm512_cmp_ps_mask(dists, best_dists, _CMP_LT_OQ);
                        best_dists = _mm512_mask_blend_ps(mask, best_dists, dists);

                        __m512i current_indices = _mm512_set_epi32(
                            k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                            k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
                        best_indices = _mm512_mask_blend_epi32(mask, best_indices, current_indices);
                    }

                    alignas(64) float final_dists[16];
                    alignas(64) int final_indices[16];
                    _mm512_store_ps(final_dists, best_dists);
                    _mm512_store_si512((__m512i*)final_indices, best_indices);

                    for (int i = 0; i < 16; ++i) {
                        if (final_dists[i] < best_dist) {
                            best_dist = final_dists[i];
                            best_k = final_indices[i];
                        }
                    }

                    for (int k = (K / 16) * 16; k < K; ++k) {
                        float dist = fvec_L2sqr(
                            current_residual.data(),
                            centroids.data() + k * Ds,
                            Ds);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }
                } else
#endif
                {
                    for (int k = 0; k < K; ++k) {
                        float dist = fvec_L2sqr(
                            current_residual.data(),
                            centroids.data() + k * Ds,
                            Ds);

                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }
                }

                bit_writer.write(best_k, level_bits[0]);

                const float* best_centroid = centroids.data() + best_k * Ds;
#ifdef __AVX512F__
                if (Ds >= 16) {
                    int d = 0;
                    for (; d + 15 < Ds; d += 16) {
                        __m512 residual_vals = _mm512_loadu_ps(&current_residual[d]);
                        __m512 centroid_vals = _mm512_loadu_ps(&best_centroid[d]);
                        __m512 result = _mm512_sub_ps(residual_vals, centroid_vals);
                        _mm512_storeu_ps(&current_residual[d], result);
                    }
                    for (; d < Ds; ++d) {
                        current_residual[d] -= best_centroid[d];
                    }
                } else
#endif
                {
                    for (int d = 0; d < Ds; ++d) {
                        current_residual[d] -= best_centroid[d];
                    }
                }

            } else {
                const std::vector<float>& scalar_codebook = scalar_codebooks[m][level - 1];
                int K = static_cast<int>(scalar_codebook.size());

                for (int d = 0; d < Ds; ++d) {
                    int best_k = 0;
                    float best_dist = std::abs(current_residual[d] - scalar_codebook[0]);

                    for (int k = 1; k < K; ++k) {
                        float dist = std::abs(current_residual[d] - scalar_codebook[k]);
                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }

                    bit_writer.write(best_k, level_bits[level]);
                    current_residual[d] -= scalar_codebook[best_k];
                }
            }
        }
    }
}

void IndexJHQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before decoding");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of vectors must be positive");
    FAISS_THROW_IF_NOT_MSG(bytes != nullptr, "Input codes cannot be null");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Output vectors cannot be null");

    for (idx_t i = 0; i < n; ++i) {
        decode_single_code(
            bytes + i * code_size,
            x + i * d);
    }

    if (use_jl_transform && is_rotation_trained) {
        std::vector<float> x_temp(n * d);
        std::memcpy(x_temp.data(), x, sizeof(float) * n * d);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(n), d, d,
            1.0f, x_temp.data(), d,
            rotation_matrix.data(), d,
            0.0f, x, d);
    }
}

void IndexJHQ::decode_single_code(const uint8_t* code, float* x) const
{
    faiss::BitstringReader bit_reader(code, code_size);
    std::memset(x, 0, sizeof(float) * d);

    for (int m = 0; m < M; ++m) {
        float* subspace_vector = x + m * Ds;

        for (int level = 0; level < num_levels; ++level) {
            if (level == 0) {
                uint32_t centroid_id = bit_reader.read(level_bits[0]);
                const std::vector<float>& centroids = codewords[m][0];
                const float* centroid = centroids.data() + centroid_id * Ds;

#ifdef __AVX512F__
                if (Ds >= 16) {
                    int d = 0;
                    for (; d + 15 < Ds; d += 16) {
                        __m512 subspace_vals = _mm512_loadu_ps(&subspace_vector[d]);
                        __m512 centroid_vals = _mm512_loadu_ps(&centroid[d]);
                        __m512 result = _mm512_add_ps(subspace_vals, centroid_vals);
                        _mm512_storeu_ps(&subspace_vector[d], result);
                    }
                    for (; d < Ds; ++d) {
                        subspace_vector[d] += centroid[d];
                    }
                } else
#endif
                {
                    cblas_saxpy(Ds, 1.0f, centroid, 1, subspace_vector, 1);
                }
            } else {
                const std::vector<float>& scalar_codebook = scalar_codebooks[m][level - 1];

#ifdef __AVX512F__
                if (Ds >= 16) {
                    int d = 0;
                    for (; d + 15 < Ds; d += 16) {
                        alignas(64) uint32_t scalar_ids[16];
                        for (int i = 0; i < 16; ++i) {
                            uint32_t scalar_id = bit_reader.read(level_bits[level]);
                            scalar_ids[i] = std::min(scalar_id, static_cast<uint32_t>(scalar_codebook.size() - 1));
                        }

                        __m512i indices = _mm512_loadu_si512(scalar_ids);
                        __m512 scalar_vals = _mm512_i32gather_ps(indices, scalar_codebook.data(), 4);

                        __m512 subspace_vals = _mm512_loadu_ps(&subspace_vector[d]);
                        __m512 result = _mm512_add_ps(subspace_vals, scalar_vals);
                        _mm512_storeu_ps(&subspace_vector[d], result);
                    }

                    for (; d < Ds; ++d) {
                        uint32_t scalar_id = bit_reader.read(level_bits[level]);
                        if (scalar_id >= scalar_codebook.size())
                            scalar_id = scalar_codebook.size() - 1;
                        subspace_vector[d] += scalar_codebook[scalar_id];
                    }
                } else
#endif
                {
                    for (int d = 0; d < Ds; ++d) {
                        uint32_t scalar_id = bit_reader.read(level_bits[level]);
                        if (scalar_id >= scalar_codebook.size())
                            scalar_id = scalar_codebook.size() - 1;
                        subspace_vector[d] += scalar_codebook[scalar_id];
                    }
                }
            }
        }
    }
}

void IndexJHQ::reconstruct(idx_t key, float* recons) const
{
    FAISS_THROW_IF_NOT_MSG(key >= 0 && key < ntotal, "Invalid vector key");

    const uint8_t* code = codes.data() + key * code_size;
    decode_single_code(code, recons);

    if (use_jl_transform && is_rotation_trained) {
        std::vector<float> temp(d);
        std::memcpy(temp.data(), recons, sizeof(float) * d);

        for (int i = 0; i < d; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < d; ++j) {
                sum += temp[j] * rotation_matrix[i * d + j];
            }
            recons[i] = sum;
        }
    }
}

void IndexJHQ::reconstruct_n(idx_t i0, idx_t ni, float* recons) const
{
    FAISS_THROW_IF_NOT_MSG(i0 >= 0 && i0 + ni <= ntotal, "Invalid range for reconstruction");

    for (idx_t i = 0; i < ni; ++i) {
        reconstruct(i0 + i, recons + i * d);
    }
}

void IndexJHQ::add(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before adding vectors");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of vectors must be positive");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Input vectors cannot be null");

    sa_encode(n, x, nullptr);
    ntotal += n;

    codes.resize(0);

    if (verbose) {
        std::cout << "Added " << n << " vectors to separated storage, total: " << ntotal << std::endl;
        std::cout << "Primary codes memory: " << (separated_codes_.primary_codes.size() / 1024.0) << " KB" << std::endl;
        if (num_levels > 1) {
            std::cout << "Residual codes memory: " << (separated_codes_.residual_codes.size() / 1024.0) << " KB" << std::endl;
        }
    }
}

void IndexJHQ::search(
    idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before searching");
    FAISS_THROW_IF_NOT_MSG(ntotal > 0, "Index is empty");
    FAISS_THROW_IF_NOT_MSG(n > 0 && k > 0, "Invalid search parameters");
    FAISS_THROW_IF_NOT_MSG(k <= ntotal, "k cannot be larger than database size");

    const JHQSearchParameters* jhq_params = dynamic_cast<const JHQSearchParameters*>(params);

    float oversampling = default_oversampling;
    bool use_early_termination = true;
    bool compute_residuals = true;

    if (jhq_params) {
        if (jhq_params->oversampling_factor > 0) {
            oversampling = jhq_params->oversampling_factor;
        }
        use_early_termination = jhq_params->use_early_termination;
        compute_residuals = jhq_params->compute_residuals;
    }

    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const float* query = x + query_idx * d;
        float* query_distances = distances + query_idx * k;
        idx_t* query_labels = labels + query_idx * k;

        if (use_early_termination) {
            search_single_query_early_termination(
                query, k, oversampling,
                query_distances, query_labels);
        } else {
            search_single_query_exhaustive(
                query, k, compute_residuals,
                query_distances, query_labels);
        }
    }
}

void IndexJHQ::search_single_query_early_termination(
    const float* query,
    idx_t k,
    float oversampling,
    float* distances,
    idx_t* labels) const
{
    idx_t n_candidates = static_cast<idx_t>(std::min(
        static_cast<double>(ntotal),
        static_cast<double>(k * oversampling)));
    if (n_candidates < k) {
        n_candidates = std::min((idx_t)ntotal, k);
    }
    if (ntotal == 0 || n_candidates == 0) {
        for (idx_t i = 0; i < k; ++i) {
            distances[i] = -1.0f;
            labels[i] = -1;
        }
        return;
    }

    std::vector<float> query_rotated(d);
    std::vector<float> primary_distances(ntotal);

    apply_jl_rotation(1, query, query_rotated.data());

    const int K0 = 1 << level_bits[0];
    std::vector<float> primary_distance_table_flat(M * K0);

    compute_primary_distance_tables_flat(
        query_rotated.data(), K0, primary_distance_table_flat.data());

    compute_primary_distances(
        primary_distance_table_flat.data(), K0, primary_distances.data());

    if (num_levels == 1) {
        faiss::maxheap_heapify(k, distances, labels, primary_distances.data(), nullptr, k);
        for (idx_t i = k; i < ntotal; i++) {
            if (primary_distances[i] < distances[0]) {
                faiss::maxheap_replace_top(k, distances, labels, primary_distances[i], i);
            }
        }
        faiss::heap_reorder<faiss::CMax<float, idx_t>>(k, distances, labels);
        return;
    }

    std::vector<idx_t> candidate_indices;

    if (ntotal > 100000) {
        constexpr int SAMPLE_SIZE = 10000;
        std::vector<std::pair<float, idx_t>> samples;
        samples.reserve(SAMPLE_SIZE);

        for (int i = 0; i < SAMPLE_SIZE; ++i) {
            idx_t idx = (static_cast<size_t>(i) * ntotal) / SAMPLE_SIZE;
            samples.emplace_back(primary_distances[idx], idx);
        }

        std::nth_element(samples.begin(),
            samples.begin() + (n_candidates * SAMPLE_SIZE) / ntotal,
            samples.end());

        float threshold = samples[(n_candidates * SAMPLE_SIZE) / ntotal].first * 1.1f;

        candidate_indices.reserve(n_candidates * 2);
        for (idx_t i = 0; i < ntotal && candidate_indices.size() < n_candidates * 2; ++i) {
            if (primary_distances[i] <= threshold) {
                candidate_indices.push_back(i);
            }
        }

        std::partial_sort(candidate_indices.begin(),
            candidate_indices.begin() + std::min(n_candidates, static_cast<idx_t>(candidate_indices.size())),
            candidate_indices.end(),
            [&](idx_t a, idx_t b) {
                return primary_distances[a] < primary_distances[b];
            });

        candidate_indices.resize(std::min(n_candidates, static_cast<idx_t>(candidate_indices.size())));
    } else {
        candidate_indices.resize(ntotal);
        std::iota(candidate_indices.begin(), candidate_indices.end(), 0);

        std::partial_sort(candidate_indices.begin(),
            candidate_indices.begin() + n_candidates,
            candidate_indices.end(),
            [&](idx_t a, idx_t b) {
                return primary_distances[a] < primary_distances[b];
            });

        candidate_indices.resize(n_candidates);
    }

    std::vector<float> final_distances(candidate_indices.size());
    constexpr int PREFETCH_DISTANCE = 8;

    for (size_t i = 0; i < candidate_indices.size(); ++i) {
        idx_t db_idx = candidate_indices[i];

        if (i + PREFETCH_DISTANCE < candidate_indices.size()) {
            idx_t prefetch_idx = candidate_indices[i + PREFETCH_DISTANCE];
            if (num_levels > 1) {
                const uint8_t* residual_codes = separated_codes_.get_residual_codes(prefetch_idx);
                _mm_prefetch(residual_codes, _MM_HINT_T0);
            }
        }

        final_distances[i] = compute_exact_distance_separated(db_idx, query_rotated.data());
    }

    for (idx_t i = 0; i < k && i < static_cast<idx_t>(candidate_indices.size()); i++) {
        distances[i] = final_distances[i];
        labels[i] = candidate_indices[i];
    }
    faiss::heap_heapify<faiss::CMax<float, idx_t>>(k, distances, labels);

    for (size_t i = k; i < candidate_indices.size(); i++) {
        if (final_distances[i] < distances[0]) {
            faiss::heap_replace_top<faiss::CMax<float, idx_t>>(
                k, distances, labels, final_distances[i], candidate_indices[i]);
        }
    }

    faiss::heap_reorder<faiss::CMax<float, idx_t>>(k, distances, labels);
}

size_t IndexJHQ::search_single_query_exhaustive(
    const float* query,
    idx_t k,
    bool compute_residuals,
    float* distances,
    idx_t* labels) const
{
    std::vector<float> query_rotated(d);
    apply_jl_rotation(1, query, query_rotated.data());

    std::vector<float> all_distances(ntotal);
    std::vector<float> reconstructed_vector(d);

    for (idx_t i = 0; i < ntotal; ++i) {
        decode_single_code(codes.data() + i * code_size, reconstructed_vector.data());

        all_distances[i] = fvec_L2sqr(
            query_rotated.data(),
            reconstructed_vector.data(),
            d);
    }

    if (k == 1) {
        float best_dist = -1.0f;
        idx_t best_idx = -1;
        if (ntotal > 0) {
            best_dist = all_distances[0];
            best_idx = 0;
            for (idx_t i = 1; i < ntotal; ++i) {
                if (all_distances[i] < best_dist) {
                    best_dist = all_distances[i];
                    best_idx = i;
                }
            }
        }
        distances[0] = best_dist;
        labels[0] = best_idx;
    } else {
        std::vector<std::pair<float, idx_t>> dist_idx_pairs(ntotal);
        for (idx_t i = 0; i < ntotal; ++i) {
            dist_idx_pairs[i] = { all_distances[i], i };
        }

        std::partial_sort(
            dist_idx_pairs.begin(),
            dist_idx_pairs.begin() + k,
            dist_idx_pairs.end());

        for (idx_t i = 0; i < k; ++i) {
            distances[i] = dist_idx_pairs[i].first;
            labels[i] = dist_idx_pairs[i].second;
        }
    }

    return ntotal;
}

void IndexJHQ::range_search(
    idx_t n,
    const float* x,
    float radius,
    RangeSearchResult* result,
    const SearchParameters* params) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before range search");
    FAISS_THROW_IF_NOT_MSG(ntotal > 0, "Index is empty");

    result->nq = n;
    if (!result->lims) {
        result->lims = new size_t[n + 1];
    }
    result->lims[0] = 0;

    std::vector<std::vector<std::pair<float, idx_t>>> all_results(n);

    std::unique_ptr<JHQDistanceComputer> dis(
        (JHQDistanceComputer*)get_FlatCodesDistanceComputer());

    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        const float* query = x + query_idx * d;

        dis->set_query(query);

        for (idx_t db_idx = 0; db_idx < ntotal; ++db_idx) {
            const float total_distance = (*dis)(db_idx);

            if (total_distance <= radius) {
                all_results[query_idx].emplace_back(total_distance, db_idx);
            }
        }
        result->lims[query_idx + 1] = result->lims[query_idx] + all_results[query_idx].size();
    }

    size_t total_results = result->lims[n];
    if (total_results > 0) {
        result->distances = new float[total_results];
        result->labels = new idx_t[total_results];
    } else {
        result->distances = nullptr;
        result->labels = nullptr;
    }

    size_t offset = 0;
    for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
        std::sort(all_results[query_idx].begin(), all_results[query_idx].end());
        for (const auto& pair : all_results[query_idx]) {
            result->distances[offset] = pair.first;
            result->labels[offset] = pair.second;
            offset++;
        }
    }
}

void IndexJHQ::compute_primary_distance_tables_flat(
    const float* query_rotated,
    int K0,
    float* distance_table_flat) const {

    if (num_levels == 1) {
        for (int m = 0; m < M; ++m) {
            const float* query_subspace = query_rotated + m * Ds;
            const std::vector<float>& centroids = codewords[m][0];
            float* table_m_ptr = distance_table_flat + (size_t)m * K0;
            compute_subspace_distances_simd(
                query_subspace,
                centroids.data(),
                table_m_ptr,
                K0,
                Ds);
        }
        return;
    }

    constexpr int SUBSPACE_BLOCK = 8;  // Increased block size
    constexpr int CENTROID_BLOCK = 32; // Increased for better vectorization

#ifdef __AVX512F__
    // AVX512 optimized version
    for (int m_block = 0; m_block < M; m_block += SUBSPACE_BLOCK) {
        const int m_end = std::min(m_block + SUBSPACE_BLOCK, M);

        // Prefetch next block
        if (m_block + SUBSPACE_BLOCK < M) {
            const float* next_query = query_rotated + (m_block + SUBSPACE_BLOCK) * Ds;
            _mm_prefetch(reinterpret_cast<const char*>(next_query), _MM_HINT_T0);
        }

        for (int m = m_block; m < m_end; ++m) {
            const float* query_subspace = query_rotated + m * Ds;
            const std::vector<float>& centroids = codewords[m][0];
            float* table_m_ptr = distance_table_flat + (size_t)m * K0;

            int k = 0;

            // Process centroids in blocks of 32 for better cache usage
            for (; k + CENTROID_BLOCK <= K0; k += CENTROID_BLOCK) {
                // Prefetch next centroid block
                if (k + CENTROID_BLOCK < K0) {
                    const float* next_centroids = centroids.data() + (k + CENTROID_BLOCK) * Ds;
                    _mm_prefetch(reinterpret_cast<const char*>(next_centroids), _MM_HINT_T1);
                    _mm_prefetch(reinterpret_cast<const char*>(next_centroids + 16), _MM_HINT_T1);
                }

                // Process 16 centroids at once, twice per block
                for (int sub_block = 0; sub_block < 2; ++sub_block) {
                    const int centroid_offset = k + sub_block * 16;
                    __m512 distances = _mm512_setzero_ps();

                    if (Ds >= 16) {
                        // Vectorize across dimensions when Ds is large enough
                        int d = 0;
                        for (; d + 15 < Ds; d += 16) {
                            __m512 query_vec = _mm512_loadu_ps(&query_subspace[d]);
                            __m512 acc = _mm512_setzero_ps();

                            for (int i = 0; i < 16; ++i) {
                                __m512 centroid_vec = _mm512_loadu_ps(&centroids[(centroid_offset + i) * Ds + d]);
                                __m512 diff = _mm512_sub_ps(query_vec, centroid_vec);
                                __m512 sq_diff = _mm512_mul_ps(diff, diff);

                                // Horizontal add across dimensions for this centroid
                                float centroid_contribution = _mm512_reduce_add_ps(sq_diff);
                                distances = _mm512_mask_add_ps(distances, (1 << i), distances,
                                                             _mm512_set1_ps(centroid_contribution));
                            }
                        }

                        // Handle remaining dimensions
                        for (; d < Ds; ++d) {
                            float query_val = query_subspace[d];
                            for (int i = 0; i < 16; ++i) {
                                float centroid_val = centroids[(centroid_offset + i) * Ds + d];
                                float diff = query_val - centroid_val;
                                distances = _mm512_mask_add_ps(distances, (1 << i), distances,
                                                             _mm512_set1_ps(diff * diff));
                            }
                        }
                    } else {
                        // Original approach for smaller Ds
                        for (int d = 0; d < Ds; ++d) {
                            __m512 query_val = _mm512_set1_ps(query_subspace[d]);

                            __m512i centroid_indices = _mm512_set_epi32(
                                (centroid_offset + 15) * Ds + d, (centroid_offset + 14) * Ds + d,
                                (centroid_offset + 13) * Ds + d, (centroid_offset + 12) * Ds + d,
                                (centroid_offset + 11) * Ds + d, (centroid_offset + 10) * Ds + d,
                                (centroid_offset + 9) * Ds + d,  (centroid_offset + 8) * Ds + d,
                                (centroid_offset + 7) * Ds + d,  (centroid_offset + 6) * Ds + d,
                                (centroid_offset + 5) * Ds + d,  (centroid_offset + 4) * Ds + d,
                                (centroid_offset + 3) * Ds + d,  (centroid_offset + 2) * Ds + d,
                                (centroid_offset + 1) * Ds + d,  centroid_offset * Ds + d);

                            __m512 cent_vals = _mm512_i32gather_ps(centroid_indices, centroids.data(), 4);
                            __m512 diff = _mm512_sub_ps(query_val, cent_vals);
                            distances = _mm512_fmadd_ps(diff, diff, distances);
                        }
                    }

                    _mm512_storeu_ps(&table_m_ptr[centroid_offset], distances);
                }
            }

            // Handle remaining centroids
            for (; k < K0; ++k) {
                table_m_ptr[k] = jhq_internal::fvec_L2sqr_dispatch(
                    query_subspace, centroids.data() + k * Ds, Ds);
            }
        }
    }

#elif defined(__AVX2__)
    // Enhanced AVX2 version
    for (int m_block = 0; m_block < M; m_block += SUBSPACE_BLOCK) {
        const int m_end = std::min(m_block + SUBSPACE_BLOCK, M);

        for (int m = m_block; m < m_end; ++m) {
            const float* query_subspace = query_rotated + m * Ds;
            const std::vector<float>& centroids = codewords[m][0];
            float* table_m_ptr = distance_table_flat + (size_t)m * K0;

            int k = 0;

            // Process 16 centroids at once (2 AVX2 vectors)
            for (; k + 16 <= K0; k += 16) {
                // First 8 centroids
                __m256 distances1 = _mm256_setzero_ps();
                for (int d = 0; d < Ds; ++d) {
                    __m256 query_val = _mm256_set1_ps(query_subspace[d]);

                    alignas(32) float centroid_vals[8];
                    for (int i = 0; i < 8; ++i) {
                        centroid_vals[i] = centroids[(k + i) * Ds + d];
                    }

                    __m256 cent_vals = _mm256_load_ps(centroid_vals);
                    __m256 diff = _mm256_sub_ps(query_val, cent_vals);
                    distances1 = _mm256_fmadd_ps(diff, diff, distances1);
                }
                _mm256_storeu_ps(&table_m_ptr[k], distances1);

                // Second 8 centroids
                __m256 distances2 = _mm256_setzero_ps();
                for (int d = 0; d < Ds; ++d) {
                    __m256 query_val = _mm256_set1_ps(query_subspace[d]);

                    alignas(32) float centroid_vals[8];
                    for (int i = 0; i < 8; ++i) {
                        centroid_vals[i] = centroids[(k + 8 + i) * Ds + d];
                    }

                    __m256 cent_vals = _mm256_load_ps(centroid_vals);
                    __m256 diff = _mm256_sub_ps(query_val, cent_vals);
                    distances2 = _mm256_fmadd_ps(diff, diff, distances2);
                }
                _mm256_storeu_ps(&table_m_ptr[k + 8], distances2);
            }

            // Handle remaining centroids
            for (; k < K0; ++k) {
                table_m_ptr[k] = jhq_internal::fvec_L2sqr_dispatch(
                    query_subspace, centroids.data() + k * Ds, Ds);
            }
        }
    }
#else
    // Scalar version with better unrolling
    for (int m = 0; m < M; ++m) {
        const float* query_subspace = query_rotated + m * Ds;
        const std::vector<float>& centroids = codewords[m][0];
        float* table_m_ptr = distance_table_flat + (size_t)m * K0;

        int k = 0;
        // Unroll by 4 for better instruction-level parallelism
        for (; k + 3 < K0; k += 4) {
            table_m_ptr[k] = jhq_internal::fvec_L2sqr_dispatch(
                query_subspace, centroids.data() + k * Ds, Ds);
            table_m_ptr[k + 1] = jhq_internal::fvec_L2sqr_dispatch(
                query_subspace, centroids.data() + (k + 1) * Ds, Ds);
            table_m_ptr[k + 2] = jhq_internal::fvec_L2sqr_dispatch(
                query_subspace, centroids.data() + (k + 2) * Ds, Ds);
            table_m_ptr[k + 3] = jhq_internal::fvec_L2sqr_dispatch(
                query_subspace, centroids.data() + (k + 3) * Ds, Ds);
        }

        for (; k < K0; ++k) {
            table_m_ptr[k] = jhq_internal::fvec_L2sqr_dispatch(
                query_subspace, centroids.data() + k * Ds, Ds);
        }
    }
#endif
}

void IndexJHQ::compute_residual_distance_tables(
    const float* query_rotated,
    std::vector<float>& flat_tables,
    std::vector<size_t>& level_offsets) const
{
    if (num_levels <= 1) {
        flat_tables.clear();
        level_offsets.clear();
        return;
    }

    const ProductQuantizer* residual_pq = get_residual_product_quantizer();
    if (residual_pq) {
        const size_t total_subquantizers = residual_pq->M;
        const int K_res = 1 << level_bits[1];

        flat_tables.resize(total_subquantizers * K_res);
        level_offsets.assign(num_levels, 0);
        for (int level = 1; level < num_levels; ++level) {
            level_offsets[level] = static_cast<size_t>(level - 1) * M * Ds * K_res;
        }

        std::vector<float> query_for_residual(total_subquantizers);
        size_t offset = 0;
        for (int level = 1; level < num_levels; ++level) {
            for (int m = 0; m < M; ++m) {
                const float* query_subspace = query_rotated + m * Ds;
                std::memcpy(query_for_residual.data() + offset,
                    query_subspace,
                    Ds * sizeof(float));
                offset += Ds;
            }
        }

        residual_pq->compute_distance_table(
            query_for_residual.data(), flat_tables.data());
        return;
    }

    level_offsets.assign(num_levels, 0);
    size_t total_size = 0;
    for (int level = 1; level < num_levels; ++level) {
        level_offsets[level] = total_size;
        int K = 1 << level_bits[level];
        total_size += static_cast<size_t>(M) * Ds * K;
    }
    flat_tables.resize(total_size);

    for (int m = 0; m < M; ++m) {
        const float* query_subspace = query_rotated + m * Ds;

        for (int level = 1; level < num_levels; ++level) {
            const auto& scalar_codebook = scalar_codebooks[m][level - 1];
            int K = static_cast<int>(scalar_codebook.size());
            size_t current_level_offset = level_offsets[level];

            for (int d = 0; d < Ds; d += 4) {
                const int d_end = std::min(d + 4, Ds);

                if (d + 4 < Ds) {
                    for (int d_pf = d + 4; d_pf < std::min(d + 8, Ds); ++d_pf) {
                        size_t pf_idx = current_level_offset +
                            (size_t)m * Ds * K + (size_t)d_pf * K;
                        _mm_prefetch(&flat_tables[pf_idx], _MM_HINT_T0);
                    }
                }

                for (int d_inner = d; d_inner < d_end; ++d_inner) {
                    const float query_val = query_subspace[d_inner];
                    size_t table_start_idx = current_level_offset +
                        (size_t)m * Ds * K + (size_t)d_inner * K;
                    float* table_ptr = flat_tables.data() + table_start_idx;

#ifdef __AVX512F__
                    __m512 query_broadcast = _mm512_set1_ps(query_val);

                    int k = 0;
                    for (; k + 15 < K; k += 16) {
                        if (k + 32 < K) {
                            _mm_prefetch(&scalar_codebook[k + 16], _MM_HINT_T0);
                        }

                        __m512 codebook_vals = _mm512_loadu_ps(&scalar_codebook[k]);
                        __m512 diffs = _mm512_sub_ps(query_broadcast, codebook_vals);
                        __m512 squared_diffs = _mm512_mul_ps(diffs, diffs);
                        _mm512_storeu_ps(&table_ptr[k], squared_diffs);
                    }

                    for (; k < K; ++k) {
                        float diff = query_val - scalar_codebook[k];
                        table_ptr[k] = diff * diff;
                    }

#elif defined(__AVX2__)
                    __m256 query_broadcast = _mm256_set1_ps(query_val);

                    int k = 0;
                    for (; k + 7 < K; k += 8) {
                        __m256 codebook_vals = _mm256_loadu_ps(&scalar_codebook[k]);
                        __m256 diffs = _mm256_sub_ps(query_broadcast, codebook_vals);
                        __m256 squared_diffs = _mm256_mul_ps(diffs, diffs);
                        _mm256_storeu_ps(&table_ptr[k], squared_diffs);
                    }

                    for (; k < K; ++k) {
                        float diff = query_val - scalar_codebook[k];
                        table_ptr[k] = diff * diff;
                    }

#else
                    for (int k = 0; k < K; ++k) {
                        float diff = query_val - scalar_codebook[k];
                        table_ptr[k] = diff * diff;
                    }
#endif
                }
            }
        }
    }
}

const ProductQuantizer* IndexJHQ::get_residual_product_quantizer() const
{
    if (num_levels != 2) {
        return nullptr;
    }

    const int bits = level_bits[1];
    const size_t K = static_cast<size_t>(1) << bits;

    for (int m = 0; m < M; ++m) {
        if (scalar_codebooks[m][0].size() != K) {
            return nullptr;
        }
    }

    if (!residual_pq_ || residual_pq_dirty_) {
        const size_t total_subquantizers = static_cast<size_t>(M) * Ds;
        residual_pq_ = std::make_unique<ProductQuantizer>(
            total_subquantizers, total_subquantizers, bits);

        for (int m = 0; m < M; ++m) {
            const auto& codebook = scalar_codebooks[m][0];
            for (int d = 0; d < Ds; ++d) {
                const size_t sub_idx = static_cast<size_t>(m) * Ds + d;
                float* centroid_ptr = residual_pq_->get_centroids(sub_idx, 0);
                std::memcpy(centroid_ptr, codebook.data(), K * sizeof(float));
            }
        }

        residual_pq_dirty_ = false;
    }

    return residual_pq_.get();
}

void IndexJHQ::compute_primary_distances_flat(
    const float* distance_table_flat,
    int K0,
    float* distances) const
{
    compute_primary_distances(distance_table_flat, K0, distances);
}

void IndexJHQ::compute_primary_distances(
    const float* distance_table_flat,
    int K0,
    float* distances) const {

    if (!distance_table_flat || !distances || ntotal == 0) {
        return;
    }

    constexpr int VECTOR_BATCH_SIZE = 512; // Larger batch for better cache usage
    constexpr int PREFETCH_DISTANCE = 64;

#pragma omp parallel for schedule(static) if (ntotal > 5000) // Lower threshold
    for (idx_t batch_start = 0; batch_start < ntotal; batch_start += VECTOR_BATCH_SIZE) {
        const idx_t batch_end = std::min(ntotal, batch_start + VECTOR_BATCH_SIZE);

        // Enhanced prefetching strategy
        for (idx_t i = batch_start; i < batch_end; i += PREFETCH_DISTANCE) {
            const idx_t prefetch_end = std::min(batch_end, i + PREFETCH_DISTANCE * 2);
            for (idx_t j = i; j < prefetch_end; j += 8) {
                if (j < ntotal) {
                    _mm_prefetch(separated_codes_.get_primary_codes(j), _MM_HINT_T0);
                }
            }
        }

        for (idx_t i = batch_start; i < batch_end; ++i) {
            const uint8_t* primary_codes = separated_codes_.get_primary_codes(i);
            float total_distance = 0.0f;

#ifdef __AVX512F__
            if (M >= 64) {
                // Ultra-wide vectorization for very large M
                __m512 acc[4] = {
                    _mm512_setzero_ps(), _mm512_setzero_ps(),
                    _mm512_setzero_ps(), _mm512_setzero_ps()
                };

                int m = 0;
                for (; m + 63 < M; m += 64) {
                    for (int chunk = 0; chunk < 4; ++chunk) {
                        const int chunk_offset = m + chunk * 16;

                        __m128i codes_128 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(&primary_codes[chunk_offset]));
                        __m512i codes_512 = _mm512_cvtepu8_epi32(codes_128);

                        __m512i base_offsets = _mm512_set1_epi32(chunk_offset * K0);
                        __m512i element_offsets = _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
                        element_offsets = _mm512_mullo_epi32(element_offsets, _mm512_set1_epi32(K0));
                        __m512i subspace_offsets = _mm512_add_epi32(base_offsets, element_offsets);

                        __m512i indices = _mm512_add_epi32(codes_512, subspace_offsets);
                        __m512 dists = _mm512_i32gather_ps(indices, distance_table_flat, 4);
                        acc[chunk] = _mm512_add_ps(acc[chunk], dists);
                    }
                }

                // Combine all accumulators
                __m512 combined1 = _mm512_add_ps(acc[0], acc[1]);
                __m512 combined2 = _mm512_add_ps(acc[2], acc[3]);
                __m512 final_combined = _mm512_add_ps(combined1, combined2);
                total_distance += _mm512_reduce_add_ps(final_combined);

                // Handle remaining subspaces
                for (; m < M; ++m) {
                    uint32_t code_val = primary_codes[m];
                    code_val = std::min(code_val, static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[m * K0 + code_val];
                }

            } else if (M >= 32) {
                // Medium M optimization
                __m512 acc1 = _mm512_setzero_ps();
                __m512 acc2 = _mm512_setzero_ps();

                int m = 0;
                for (; m + 31 < M; m += 32) {
                    // First 16
                    __m128i codes1_128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&primary_codes[m]));
                    __m512i codes1_512 = _mm512_cvtepu8_epi32(codes1_128);

                    __m512i offsets1 = _mm512_set_epi32(
                        (m+15)*K0, (m+14)*K0, (m+13)*K0, (m+12)*K0,
                        (m+11)*K0, (m+10)*K0, (m+9)*K0, (m+8)*K0,
                        (m+7)*K0, (m+6)*K0, (m+5)*K0, (m+4)*K0,
                        (m+3)*K0, (m+2)*K0, (m+1)*K0, m*K0);

                    __m512i indices1 = _mm512_add_epi32(codes1_512, offsets1);
                    __m512 dists1 = _mm512_i32gather_ps(indices1, distance_table_flat, 4);
                    acc1 = _mm512_add_ps(acc1, dists1);

                    // Second 16
                    __m128i codes2_128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&primary_codes[m + 16]));
                    __m512i codes2_512 = _mm512_cvtepu8_epi32(codes2_128);

                    __m512i offsets2 = _mm512_set_epi32(
                        (m+31)*K0, (m+30)*K0, (m+29)*K0, (m+28)*K0,
                        (m+27)*K0, (m+26)*K0, (m+25)*K0, (m+24)*K0,
                        (m+23)*K0, (m+22)*K0, (m+21)*K0, (m+20)*K0,
                        (m+19)*K0, (m+18)*K0, (m+17)*K0, (m+16)*K0);

                    __m512i indices2 = _mm512_add_epi32(codes2_512, offsets2);
                    __m512 dists2 = _mm512_i32gather_ps(indices2, distance_table_flat, 4);
                    acc2 = _mm512_add_ps(acc2, dists2);
                }

                total_distance += _mm512_reduce_add_ps(acc1) + _mm512_reduce_add_ps(acc2);

                for (; m < M; ++m) {
                    uint32_t code_val = primary_codes[m];
                    code_val = std::min(code_val, static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[m * K0 + code_val];
                }

            } else if (M >= 16) {
                // Original AVX512 path
                __m512 acc = _mm512_setzero_ps();
                int m = 0;

                for (; m + 15 < M; m += 16) {
                    __m128i codes_128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&primary_codes[m]));
                    __m512i codes_512 = _mm512_cvtepu8_epi32(codes_128);

                    __m512i subspace_offsets = _mm512_set_epi32(
                        (m+15)*K0, (m+14)*K0, (m+13)*K0, (m+12)*K0,
                        (m+11)*K0, (m+10)*K0, (m+9)*K0, (m+8)*K0,
                        (m+7)*K0, (m+6)*K0, (m+5)*K0, (m+4)*K0,
                        (m+3)*K0, (m+2)*K0, (m+1)*K0, m*K0);

                    __m512i indices = _mm512_add_epi32(codes_512, subspace_offsets);
                    __m512 dists = _mm512_i32gather_ps(indices, distance_table_flat, 4);
                    acc = _mm512_add_ps(acc, dists);
                }

                total_distance += _mm512_reduce_add_ps(acc);

                for (; m < M; ++m) {
                    uint32_t code_val = primary_codes[m];
                    code_val = std::min(code_val, static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[m * K0 + code_val];
                }
            } else
#elif defined(__AVX2__)
            // Enhanced AVX2 path similar to AVX512 but with 8-wide vectors
            if (M >= 32) {
                __m256 acc[4] = {
                    _mm256_setzero_ps(), _mm256_setzero_ps(),
                    _mm256_setzero_ps(), _mm256_setzero_ps()
                };

                int m = 0;
                for (; m + 31 < M; m += 32) {
                    for (int chunk = 0; chunk < 4; ++chunk) {
                        const int chunk_offset = m + chunk * 8;

                        __m128i codes_64 = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(&primary_codes[chunk_offset]));
                        __m256i codes_256 = _mm256_cvtepu8_epi32(codes_64);

                        __m256i offsets = _mm256_set_epi32(
                            (chunk_offset+7)*K0, (chunk_offset+6)*K0,
                            (chunk_offset+5)*K0, (chunk_offset+4)*K0,
                            (chunk_offset+3)*K0, (chunk_offset+2)*K0,
                            (chunk_offset+1)*K0, chunk_offset*K0);

                        __m256i indices = _mm256_add_epi32(codes_256, offsets);
                        __m256 dists = _mm256_i32gather_ps(distance_table_flat, indices, 4);
                        acc[chunk] = _mm256_add_ps(acc[chunk], dists);
                    }
                }

                // Combine accumulators
                __m256 combined1 = _mm256_add_ps(acc[0], acc[1]);
                __m256 combined2 = _mm256_add_ps(acc[2], acc[3]);
                __m256 final_combined = _mm256_add_ps(combined1, combined2);

                __m128 sum_128 = _mm_add_ps(_mm256_castps256_ps128(final_combined),
                    _mm256_extractf128_ps(final_combined, 1));
                __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                __m128 sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
                total_distance += _mm_cvtss_f32(sum_32);

                for (; m < M; ++m) {
                    uint32_t code_val = primary_codes[m];
                    code_val = std::min(code_val, static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[m * K0 + code_val];
                }
            } else if (M >= 16) {
                // Original AVX2 16-wide processing
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                int m = 0;

                for (; m + 15 < M; m += 16) {
                    __m128i codes1_64 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(&primary_codes[m]));
                    __m256i codes1_256 = _mm256_cvtepu8_epi32(codes1_64);

                    __m256i offsets1 = _mm256_set_epi32(
                        (m+7)*K0, (m+6)*K0, (m+5)*K0, (m+4)*K0,
                        (m+3)*K0, (m+2)*K0, (m+1)*K0, m*K0);

                    __m256i indices1 = _mm256_add_epi32(codes1_256, offsets1);
                    __m256 dists1 = _mm256_i32gather_ps(distance_table_flat, indices1, 4);
                    acc1 = _mm256_add_ps(acc1, dists1);

                    __m128i codes2_64 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(&primary_codes[m + 8]));
                    __m256i codes2_256 = _mm256_cvtepu8_epi32(codes2_64);

                    __m256i offsets2 = _mm256_set_epi32(
                        (m+15)*K0, (m+14)*K0, (m+13)*K0, (m+12)*K0,
                        (m+11)*K0, (m+10)*K0, (m+9)*K0, (m+8)*K0);

                    __m256i indices2 = _mm256_add_epi32(codes2_256, offsets2);
                    __m256 dists2 = _mm256_i32gather_ps(distance_table_flat, indices2, 4);
                    acc2 = _mm256_add_ps(acc2, dists2);
                }

                __m256 combined = _mm256_add_ps(acc1, acc2);
                __m128 sum_128 = _mm_add_ps(_mm256_castps256_ps128(combined),
                    _mm256_extractf128_ps(combined, 1));
                __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                __m128 sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
                total_distance += _mm_cvtss_f32(sum_32);

                for (; m < M; ++m) {
                    uint32_t code_val = primary_codes[m];
                    code_val = std::min(code_val, static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[m * K0 + code_val];
                }
            } else
#endif
            {
                // Enhanced scalar version
                int m = 0;
                for (; m + 7 < M; m += 8) {
                    float sum =
                        distance_table_flat[m*K0 + std::min(static_cast<uint32_t>(primary_codes[m]), static_cast<uint32_t>(K0-1))] +
                        distance_table_flat[(m+1)*K0 + std::min(static_cast<uint32_t>(primary_codes[m+1]), static_cast<uint32_t>(K0-1))] +
                        distance_table_flat[(m+2)*K0 + std::min(static_cast<uint32_t>(primary_codes[m+2]), static_cast<uint32_t>(K0-1))] +
                        distance_table_flat[(m+3)*K0 + std::min(static_cast<uint32_t>(primary_codes[m+3]), static_cast<uint32_t>(K0-1))] +
                        distance_table_flat[(m+4)*K0 + std::min(static_cast<uint32_t>(primary_codes[m+4]), static_cast<uint32_t>(K0-1))] +
                        distance_table_flat[(m+5)*K0 + std::min(static_cast<uint32_t>(primary_codes[m+5]), static_cast<uint32_t>(K0-1))] +
                        distance_table_flat[(m+6)*K0 + std::min(static_cast<uint32_t>(primary_codes[m+6]), static_cast<uint32_t>(K0-1))] +
                        distance_table_flat[(m+7)*K0 + std::min(static_cast<uint32_t>(primary_codes[m+7]), static_cast<uint32_t>(K0-1))];
                    total_distance += sum;
                }
                for (; m < M; ++m) {
                    uint32_t code_val = std::min(static_cast<uint32_t>(primary_codes[m]),
                                               static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[m * K0 + code_val];
                }
            }

            distances[i] = total_distance;
        }
    }
}

FlatCodesDistanceComputer* IndexJHQ::get_FlatCodesDistanceComputer() const
{
    return new JHQDistanceComputer(*this);
}

void IndexJHQ::extract_all_codes_after_add()
{
    if (ntotal == 0) {
        separated_codes_.clear();
        return;
    }

    if (verbose) {
        std::cout << "Extracting all codes (primary + residual) for " << ntotal
                  << " vectors..." << std::endl;
    }

    separated_codes_.initialize(M, Ds, num_levels, ntotal);

    residual_bits_per_subspace = 0;
    for (int level = 1; level < num_levels; ++level) {
        residual_bits_per_subspace += static_cast<size_t>(Ds) * level_bits[level];
    }

    const double extract_start = getmillisecs();

#pragma omp parallel for schedule(static, 1024) if (ntotal > 10000)
    for (idx_t i = 0; i < ntotal; ++i) {
        extract_single_vector_all_codes(i);
    }

    const double extract_time = getmillisecs() - extract_start;

    if (verbose) {
        std::cout << "Code extraction completed in " << extract_time << " ms" << std::endl;
        std::cout << "  - Primary codes: "
                  << (separated_codes_.primary_codes.size() / (1024.0 * 1024.0))
                  << " MB" << std::endl;
        if (num_levels > 1) {
            std::cout << "  - Residual codes: "
                      << (separated_codes_.residual_codes.size() / (1024.0 * 1024.0))
                      << " MB" << std::endl;
        }
        std::cout << "  - Total pre-decoded memory: "
                  << (get_pre_decoded_memory_usage() / (1024.0 * 1024.0))
                  << " MB" << std::endl;
    }
}

void IndexJHQ::extract_single_vector_all_codes(idx_t vector_idx) const
{
    FAISS_THROW_IF_NOT_MSG(vector_idx < ntotal, "Vector index out of bounds");
    FAISS_THROW_IF_NOT_MSG(separated_codes_.is_initialized,
        "Separated codes not initialized");

    const uint8_t* packed_code = codes.data() + vector_idx * code_size;
    BitstringReader bit_reader(packed_code, code_size);

    uint8_t* primary_dest = const_cast<uint8_t*>(
        separated_codes_.get_primary_codes(vector_idx));

    uint8_t* residual_dest = nullptr;
    if (num_levels > 1) {
        residual_dest = const_cast<uint8_t*>(
            separated_codes_.get_residual_codes(vector_idx));
    }

    size_t residual_offset = 0;

    for (int m = 0; m < M; ++m) {
        const uint32_t primary_centroid_id = bit_reader.read(level_bits[0]);
        primary_dest[m] = static_cast<uint8_t>(primary_centroid_id);

        if (num_levels > 1 && residual_dest != nullptr) {
            for (int level = 1; level < num_levels; ++level) {
                for (int d = 0; d < Ds; ++d) {
                    const uint32_t scalar_id = bit_reader.read(level_bits[level]);
                    residual_dest[residual_offset++] = static_cast<uint8_t>(scalar_id);
                }
            }
        } else {
            bit_reader.i += residual_bits_per_subspace;
        }
    }
}

size_t IndexJHQ::get_memory_usage() const
{
    size_t total_bytes = 0;

    if (use_jl_transform) {
        total_bytes += rotation_matrix.size() * sizeof(float);
    }

    for (int m = 0; m < M; ++m) {
        if (!codewords[m].empty() && !codewords[m][0].empty()) {
            total_bytes += codewords[m][0].size() * sizeof(float);
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int l = 0; l < static_cast<int>(scalar_codebooks[m].size()); ++l) {
            total_bytes += scalar_codebooks[m][l].size() * sizeof(float);
        }
    }

    total_bytes += separated_codes_.memory_usage();
    total_bytes += 1024;

    return total_bytes;
}

void IndexJHQ::reset()
{
    is_trained = false;
    is_rotation_trained = false;

    rotation_matrix.clear();
    codewords.clear();
    scalar_codebooks.clear();

    residual_pq_.reset();
    residual_pq_dirty_ = true;

    codes.resize(0);
    separated_codes_.clear();
    ntotal = 0;
    residual_bits_per_subspace = 0;

    initialize_data_structures();
}

void IndexJHQ::reset_data()
{
    codes.clear();
    separated_codes_.clear();
    ntotal = 0;
    residual_pq_dirty_ = true;
}

void IndexJHQ::analytical_gaussian_init(
    const float* data,
    idx_t n,
    int dim,
    int k,
    float* centroids) const
{
    if (!use_analytical_init || k <= 1) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (int i = 0; i < k * dim; ++i) {
            centroids[i] = dist(rng);
        }
        return;
    }

    std::vector<float> mean(dim, 0.0f);
    std::vector<float> variance(dim, 0.0f);

    const float inv_n = 1.0f / n;
    for (idx_t i = 0; i < n; ++i) {
        const float* row = data + i * dim;
        for (int j = 0; j < dim; ++j) {
            mean[j] += row[j];
        }
    }

#ifdef __AVX512F__
    if (dim >= 16) {
        for (int j = 0; j + 15 < dim; j += 16) {
            __m512 mean_vec = _mm512_loadu_ps(&mean[j]);
            mean_vec = _mm512_mul_ps(mean_vec, _mm512_set1_ps(inv_n));
            _mm512_storeu_ps(&mean[j], mean_vec);
        }
        for (int j = (dim / 16) * 16; j < dim; ++j) {
            mean[j] *= inv_n;
        }
    } else
#endif
    {
        for (int j = 0; j < dim; ++j) {
            mean[j] *= inv_n;
        }
    }

    const float inv_n_minus_1 = 1.0f / (n - 1);
    for (idx_t i = 0; i < n; ++i) {
        const float* row = data + i * dim;

#ifdef __AVX512F__
        if (dim >= 16) {
            for (int j = 0; j + 15 < dim; j += 16) {
                __m512 data_vec = _mm512_loadu_ps(&row[j]);
                __m512 mean_vec = _mm512_loadu_ps(&mean[j]);
                __m512 diff = _mm512_sub_ps(data_vec, mean_vec);
                __m512 var_vec = _mm512_loadu_ps(&variance[j]);
                var_vec = _mm512_fmadd_ps(diff, diff, var_vec);
                _mm512_storeu_ps(&variance[j], var_vec);
            }
            for (int j = (dim / 16) * 16; j < dim; ++j) {
                float diff = row[j] - mean[j];
                variance[j] += diff * diff;
            }
        } else
#endif
        {
            for (int j = 0; j < dim; ++j) {
                float diff = row[j] - mean[j];
                variance[j] += diff * diff;
            }
        }
    }

    for (int j = 0; j < dim; ++j) {
        variance[j] *= inv_n_minus_1;
    }

    float robust_variance;
    if (dim <= 8) {
        std::vector<float> sorted_variance = variance;
        std::nth_element(sorted_variance.begin(),
            sorted_variance.begin() + dim / 2,
            sorted_variance.end());
        robust_variance = sorted_variance[dim / 2];
    } else {
        std::vector<float> sorted_variance = variance;
        int start_idx = dim / 4;
        int end_idx = 3 * dim / 4;

        std::partial_sort(sorted_variance.begin(),
            sorted_variance.begin() + end_idx,
            sorted_variance.end());

        float trimmed_mean = 0.0f;
        for (int i = start_idx; i < end_idx; ++i) {
            trimmed_mean += sorted_variance[i];
        }
        trimmed_mean /= (end_idx - start_idx);

        if (use_jl_transform && dim > 32) {
            float median_var = sorted_variance[dim / 2];
            robust_variance = 0.8f * median_var + 0.2f * trimmed_mean;
        } else {
            robust_variance = trimmed_mean;
        }
    }

    float std_scale = std::sqrt(std::max(robust_variance, 1e-10f));

    std::vector<float> gaussian_quantiles(k);

    if (k == 2) {
        gaussian_quantiles[0] = -0.6745f;
        gaussian_quantiles[1] = 0.6745f;
    } else if (k <= 8) {
        static const float precomputed_quantiles[][8] = {
            {},
            {},
            { -0.6745f, 0.6745f },
            { -1.0364f, 0.0f, 1.0364f },
            { -1.2816f, -0.5244f, 0.5244f, 1.2816f },
            { -1.4652f, -0.7416f, 0.0f, 0.7416f, 1.4652f },
            { -1.5982f, -0.9082f, -0.3584f, 0.3584f, 0.9082f, 1.5982f },
            { -1.7109f, -1.0364f, -0.5244f, 0.0f, 0.5244f, 1.0364f, 1.7109f },
            { -1.8119f, -1.1503f, -0.6745f, -0.2533f, 0.2533f, 0.6745f, 1.1503f, 1.8119f }
        };

        for (int i = 0; i < k; ++i) {
            gaussian_quantiles[i] = precomputed_quantiles[k][i];
        }

        if (use_jl_transform && dim > 64) {
            float adjustment = 1.0f + 0.03f * std::log(static_cast<float>(dim) / 64.0f);
            adjustment = std::min(adjustment, 1.15f);
            for (int i = 0; i < k; ++i) {
                gaussian_quantiles[i] *= adjustment;
            }
        }
    } else {
        const size_t sample_size = std::min(static_cast<size_t>(n), static_cast<size_t>(2000));
        const size_t stride = std::max(static_cast<size_t>(1), n / sample_size);

        std::vector<float> norms;
        norms.reserve(sample_size);

        for (size_t i = 0; i < sample_size; i += stride) {
            if (i >= n)
                break;
            const float* point = data + i * dim;
            float norm_sq = 0.0f;

#ifdef __AVX512F__
            if (dim >= 16) {
                __m512 acc = _mm512_setzero_ps();
                int d = 0;
                for (; d + 15 < dim; d += 16) {
                    __m512 data_vec = _mm512_loadu_ps(&point[d]);
                    __m512 mean_vec = _mm512_loadu_ps(&mean[d]);
                    __m512 diff = _mm512_sub_ps(data_vec, mean_vec);
                    acc = _mm512_fmadd_ps(diff, diff, acc);
                }
                norm_sq = _mm512_reduce_add_ps(acc);
                for (; d < dim; ++d) {
                    float diff = point[d] - mean[d];
                    norm_sq += diff * diff;
                }
            } else
#endif
            {
                for (int d = 0; d < dim; ++d) {
                    float diff = point[d] - mean[d];
                    norm_sq += diff * diff;
                }
            }
            norms.push_back(std::sqrt(norm_sq));
        }

        std::sort(norms.begin(), norms.end());

        for (int i = 0; i < k; ++i) {
            float quantile = static_cast<float>(i + 1) / (k + 1);
            size_t idx = static_cast<size_t>(quantile * (norms.size() - 1));
            float norm_quantile = norms[idx];

            if (norm_quantile > std_scale * 0.1f) {
                gaussian_quantiles[i] = norm_quantile / std_scale;
            } else {
                float erf_arg = 2.0f * quantile - 1.0f;
                gaussian_quantiles[i] = std::sqrt(2.0f) * jhq_internal::erfinv_approx(erf_arg);
            }
        }
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dir_dist(0.0f, 1.0f);
    std::normal_distribution<float> noise_dist(0.0f, std_scale * 0.01f);

    std::vector<float> directions_flat(k * dim);

    for (int i = 0; i < k; ++i) {
        float* dir_i = directions_flat.data() + i * dim;

        for (int d = 0; d < dim; ++d) {
            dir_i[d] = dir_dist(rng);
        }

        for (int j = 0; j < i; ++j) {
            const float* dir_j = directions_flat.data() + j * dim;

            float dot_product = 0.0f;
#ifdef __AVX512F__
            if (dim >= 16) {
                __m512 acc = _mm512_setzero_ps();
                int d = 0;
                for (; d + 15 < dim; d += 16) {
                    __m512 vi = _mm512_loadu_ps(&dir_i[d]);
                    __m512 vj = _mm512_loadu_ps(&dir_j[d]);
                    acc = _mm512_fmadd_ps(vi, vj, acc);
                }
                dot_product = _mm512_reduce_add_ps(acc);
                for (; d < dim; ++d) {
                    dot_product += dir_i[d] * dir_j[d];
                }
            } else
#endif
            {
                for (int d = 0; d < dim; ++d) {
                    dot_product += dir_i[d] * dir_j[d];
                }
            }

#ifdef __AVX512F__
            if (dim >= 16) {
                __m512 dot_vec = _mm512_set1_ps(dot_product);
                int d = 0;
                for (; d + 15 < dim; d += 16) {
                    __m512 vi = _mm512_loadu_ps(&dir_i[d]);
                    __m512 vj = _mm512_loadu_ps(&dir_j[d]);
                    vi = _mm512_fnmadd_ps(dot_vec, vj, vi);
                    _mm512_storeu_ps(&dir_i[d], vi);
                }
                for (; d < dim; ++d) {
                    dir_i[d] -= dot_product * dir_j[d];
                }
            } else
#endif
            {
                for (int d = 0; d < dim; ++d) {
                    dir_i[d] -= dot_product * dir_j[d];
                }
            }
        }

        float norm_sq = 0.0f;
#ifdef __AVX512F__
        if (dim >= 16) {
            __m512 acc = _mm512_setzero_ps();
            int d = 0;
            for (; d + 15 < dim; d += 16) {
                __m512 vi = _mm512_loadu_ps(&dir_i[d]);
                acc = _mm512_fmadd_ps(vi, vi, acc);
            }
            norm_sq = _mm512_reduce_add_ps(acc);
            for (; d < dim; ++d) {
                norm_sq += dir_i[d] * dir_i[d];
            }
        } else
#endif
        {
            for (int d = 0; d < dim; ++d) {
                norm_sq += dir_i[d] * dir_i[d];
            }
        }

        float norm = std::sqrt(norm_sq);
        if (norm > 1e-10f) {
            float inv_norm = 1.0f / norm;
#ifdef __AVX512F__
            if (dim >= 16) {
                __m512 inv_norm_vec = _mm512_set1_ps(inv_norm);
                int d = 0;
                for (; d + 15 < dim; d += 16) {
                    __m512 vi = _mm512_loadu_ps(&dir_i[d]);
                    vi = _mm512_mul_ps(vi, inv_norm_vec);
                    _mm512_storeu_ps(&dir_i[d], vi);
                }
                for (; d < dim; ++d) {
                    dir_i[d] *= inv_norm;
                }
            } else
#endif
            {
                for (int d = 0; d < dim; ++d) {
                    dir_i[d] *= inv_norm;
                }
            }
        }
    }

    for (int i = 0; i < k; ++i) {
        float radius = std::abs(gaussian_quantiles[i]) * std_scale;
        const float* dir_i = directions_flat.data() + i * dim;
        float* centroid_i = centroids + i * dim;

        for (int j = 0; j < dim; ++j) {
            float dim_adjustment = std::sqrt(variance[j] / robust_variance);
            dim_adjustment = std::clamp(dim_adjustment, 0.7f, 1.5f);

            centroid_i[j] = mean[j] + radius * dir_i[j] * dim_adjustment + noise_dist(rng);
        }
    }
}

void IndexJHQ::set_default_oversampling(float oversampling)
{
    FAISS_THROW_IF_NOT_MSG(oversampling >= 1.0f, "Oversampling factor must be >= 1.0");
    default_oversampling = oversampling;

    if (verbose) {
        std::cout << "Updated default oversampling to " << oversampling << std::endl;
    }
}

void IndexJHQ::write(IOWriter* f) const
{
    write_index_jhq(this, f);
}

IndexJHQ* IndexJHQ::read(IOReader* f)
{
    return read_index_jhq(f);
}

bool IndexJHQ::has_pre_decoded_codes() const
{
    return separated_codes_.is_initialized && !separated_codes_.empty() && ntotal > 0;
}

size_t IndexJHQ::get_pre_decoded_memory_usage() const
{
    return separated_codes_.memory_usage();
}

void IndexJHQ::set_clustering_parameters(bool use_kmeans, int niter, int seed)
{
    use_kmeans_refinement = use_kmeans;
    kmeans_niter = niter;
    kmeans_seed = seed;

    if (verbose) {
        std::cout << "JHQ clustering mode: " << (use_kmeans ? "Analytical + K-means" : "Analytical only") << std::endl;
    }
}

IndexJHQ::SearchWorkspace& IndexJHQ::get_search_workspace() const
{
    const int K0 = 1 << level_bits[0];
    const int K_res = num_levels > 1 ? (1 << level_bits[1]) : 0;

    workspace_.query_rotated.resize(M * Ds);
    workspace_.primary_distance_table.resize(M * K0);

    if (num_levels > 1) {
        size_t residual_table_size = M * (num_levels - 1) * Ds * K_res;
        workspace_.residual_distance_tables.resize(residual_table_size);
    }

    return workspace_;
}

void IndexJHQ::compute_primary_distance_table(
    const float* query_rotated,
    float* distance_table) const
{
    const int K0 = 1 << level_bits[0];
    constexpr int SUBSPACE_BLOCK_SIZE = 4;

    for (int m_block = 0; m_block < M; m_block += SUBSPACE_BLOCK_SIZE) {
        const int m_end = std::min(m_block + SUBSPACE_BLOCK_SIZE, M);

        for (int m = m_block; m < m_end; ++m) {
            const float* query_sub = query_rotated + m * Ds;
            const std::vector<float>& centroids = codewords[m][0];
            float* table_m = distance_table + m * K0;

            compute_subspace_distances_simd(query_sub, centroids.data(), table_m, K0, Ds);
        }
    }
}

void IndexJHQ::compute_subspace_distances_simd(
    const float* query_sub,
    const float* codewords,
    float* distances,
    int K,
    int Ds)
{
#ifdef __AVX512F__
    int k = 0;
    for (; k + 16 <= K; k += 16) {
        __m512 acc = _mm512_setzero_ps();

        for (int d = 0; d < Ds; ++d) {
            __m512 query_val = _mm512_set1_ps(query_sub[d]);
            __m512 centroid_vals = _mm512_loadu_ps(codewords + k * Ds + d);
            __m512 diff = _mm512_sub_ps(query_val, centroid_vals);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        _mm512_store_ps(distances + k, acc);
    }

    for (; k < K; ++k) {
        distances[k] = fvec_L2sqr(query_sub, codewords + k * Ds, Ds);
    }

#elif defined(__AVX2__)
    int k = 0;
    for (; k + 8 <= K; k += 8) {
        __m256 acc = _mm256_setzero_ps();

        for (int d = 0; d < Ds; ++d) {
            __m256 query_val = _mm256_set1_ps(query_sub[d]);
            __m256 centroid_vals = _mm256_loadu_ps(codewords + k * Ds + d);
            __m256 diff = _mm256_sub_ps(query_val, centroid_vals);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        _mm256_store_ps(distances + k, acc);
    }

    for (; k < K; ++k) {
        distances[k] = fvec_L2sqr(query_sub, codewords + k * Ds, Ds);
    }

#else
    for (int k = 0; k < K; ++k) {
        distances[k] = fvec_L2sqr(query_sub, codewords + k * Ds, Ds);
    }
#endif
}

JHQDistanceComputer::JHQDistanceComputer(const IndexJHQ& idx)
    : FlatCodesDistanceComputer(idx.codes.data(), idx.code_size)
    , index(idx)
    , M(idx.M)
    , Ds(idx.Ds)
    , num_levels(idx.num_levels)
    , K0(1 << idx.level_bits[0])
    , tables_computed(false)
    , primary_table_size(static_cast<size_t>(M) * K0)
    , has_residual_levels(num_levels > 1)
    , use_quantized_tables(true)
{
    query_rotated.resize(idx.d);
    primary_distance_table_flat.resize(primary_table_size);
    temp_workspace.resize(std::max(static_cast<size_t>(M * K0), static_cast<size_t>(1024)));

    if (has_residual_levels) {
        compute_residual_buffer_sizes();
    }

    if (primary_table_size * sizeof(float) > 1024 * 1024) {
        enable_quantization();
    }
}

void JHQDistanceComputer::set_query(const float* x)
{
    apply_rotation_to_query(x);

    if (use_quantized_tables) {
        compute_and_quantize_tables();
    } else {
        compute_primary_distance_table();
    }

    if (has_residual_levels) {
        compute_residual_distance_tables();
    }

    tables_computed = true;
}

float JHQDistanceComputer::distance_to_code(const uint8_t* code)
{
    FAISS_THROW_IF_NOT(tables_computed);

    if (index.has_pre_decoded_codes()) {
        return precomputed_distance_to_code(code);
    } else {
        return distance_to_code_with_decoding(code);
    }
}

float JHQDistanceComputer::operator()(idx_t i)
{
    const uint8_t* code = codes + i * code_size;
    return distance_to_code(code);
}

float JHQDistanceComputer::symmetric_dis(idx_t i, idx_t j)
{
    std::vector<float> xi(index.d), xj(index.d);
    index.reconstruct(i, xi.data());
    index.reconstruct(j, xj.data());
    return fvec_L2sqr(xi.data(), xj.data(), index.d);
}

void JHQDistanceComputer::distances_batch_4(
    const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3,
    float& dis0, float& dis1, float& dis2, float& dis3)
{
    dis0 = (*this)(idx0);
    dis1 = (*this)(idx1);
    dis2 = (*this)(idx2);
    dis3 = (*this)(idx3);
}

void JHQDistanceComputer::enable_quantization()
{
    use_quantized_tables = true;
    quantized_primary_tables.resize(primary_table_size);
}

void JHQDistanceComputer::compute_and_quantize_tables()
{
    compute_primary_distance_table();

    auto [min_it, max_it] = std::minmax_element(
        primary_distance_table_flat.begin(), primary_distance_table_flat.end());
    quantization_offset = *min_it;
    quantization_scale = (*max_it - *min_it) / 65535.0f;

    for (size_t i = 0; i < primary_table_size; ++i) {
        float normalized = (primary_distance_table_flat[i] - quantization_offset) / quantization_scale;
        quantized_primary_tables[i] = static_cast<uint16_t>(std::clamp(normalized, 0.0f, 65535.0f));
    }
}

float JHQDistanceComputer::precomputed_distance_to_code(const uint8_t* code) const
{
    const size_t vector_idx = (code - codes) / code_size;
    const uint8_t* primary_codes_ptr = index.separated_codes_.get_primary_codes(vector_idx);
    const uint8_t* residual_codes_ptr = has_residual_levels
        ? index.separated_codes_.get_residual_codes(vector_idx)
        : nullptr;

    float total_distance = 0.0f;

    int m = 0;
    for (; m + 3 < M; m += 4) {
        if (m + 7 < M) {
            _mm_prefetch(reinterpret_cast<const char*>(&primary_codes_ptr[m + 4]), _MM_HINT_T0);
        }

        if (use_quantized_tables) {
            for (int i = 0; i < 4; ++i) {
                const uint8_t centroid_id = primary_codes_ptr[m + i];
                const uint16_t quantized_dist = quantized_primary_tables[(m + i) * K0 + centroid_id];
                total_distance += quantization_offset + quantized_dist * quantization_scale;
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                const uint8_t centroid_id = primary_codes_ptr[m + i];
                total_distance += primary_distance_table_flat[(m + i) * K0 + centroid_id];
            }
        }
    }

    for (; m < M; ++m) {
        const uint8_t centroid_id = primary_codes_ptr[m];
        if (use_quantized_tables) {
            const uint16_t quantized_dist = quantized_primary_tables[m * K0 + centroid_id];
            total_distance += quantization_offset + quantized_dist * quantization_scale;
        } else {
            total_distance += primary_distance_table_flat[m * K0 + centroid_id];
        }
    }

    if (has_residual_levels) {
        total_distance += compute_precomputed_residual_distance(vector_idx);
        if (residual_codes_ptr) {
            total_distance += jhq_internal::compute_cross_term_from_codes(
                index,
                primary_codes_ptr,
                residual_codes_ptr,
                index.separated_codes_.residual_subspace_stride,
                index.separated_codes_.residual_level_stride);
        }
    }

    return total_distance;
}

float JHQDistanceComputer::distance_to_code_with_decoding(const uint8_t* code) const
{
    BitstringReader bit_reader(code, index.code_size);
    float total_distance = 0.0f;
    float cross_term = 0.0f;

    float* residual_buffer = nullptr;
    if (has_residual_levels) {
        if (temp_workspace.size() < static_cast<size_t>(Ds)) {
            temp_workspace.resize(Ds);
        }
        residual_buffer = temp_workspace.data();
    }

    const uint32_t primary_limit = static_cast<uint32_t>((1 << index.level_bits[0]) - 1);

    for (int m = 0; m < M; ++m) {
        uint32_t centroid_id = bit_reader.read(index.level_bits[0]);
        centroid_id = std::min(centroid_id, primary_limit);
        const size_t table_idx = static_cast<size_t>(m) * K0 + centroid_id;
        const float primary_contrib = use_quantized_tables
            ? quantization_offset + quantized_primary_tables[table_idx] * quantization_scale
            : primary_distance_table_flat[table_idx];
        total_distance += primary_contrib;

        if (has_residual_levels) {
            std::fill(residual_buffer, residual_buffer + Ds, 0.0f);

            const auto& centroids = index.codewords[m][0];
            const int num_centroids = centroids.empty() ? 0 : static_cast<int>(centroids.size() / Ds);
            const uint32_t safe_centroid = centroids.empty()
                ? 0
                : std::min<uint32_t>(centroid_id, static_cast<uint32_t>(std::max(0, num_centroids - 1)));
            const float* centroid_ptr = centroids.empty()
                ? nullptr
                : centroids.data() + static_cast<size_t>(safe_centroid) * Ds;

            for (int level = 1; level < num_levels; ++level) {
                const size_t level_offset = residual_table_offsets[level];
                const int K_res = 1 << index.level_bits[level];
                const uint32_t residual_limit = static_cast<uint32_t>(K_res - 1);
                const size_t table_base = level_offset + static_cast<size_t>(m) * Ds * K_res;
                const auto& scalar_codebook = index.scalar_codebooks[m][level - 1];
                const int codebook_size = static_cast<int>(scalar_codebook.size());

                for (int d = 0; d < Ds; ++d) {
                    uint32_t scalar_id = bit_reader.read(index.level_bits[level]);
                    scalar_id = std::min(scalar_id, residual_limit);
                    const size_t lut_idx = table_base + static_cast<size_t>(d) * K_res + scalar_id;
                    total_distance += residual_distance_tables_flat[lut_idx];

                    if (codebook_size > 0) {
                        const uint32_t safe_scalar = std::min<uint32_t>(
                            scalar_id,
                            static_cast<uint32_t>(codebook_size - 1));
                        residual_buffer[d] += scalar_codebook[safe_scalar];
                    }
                }
            }

            if (centroid_ptr) {
                for (int d = 0; d < Ds; ++d) {
                    cross_term += centroid_ptr[d] * residual_buffer[d];
                }
            }
        } else {
            bit_reader.i += index.residual_bits_per_subspace;
        }
    }

    if (has_residual_levels) {
        total_distance += 2.0f * cross_term;
    }

    return total_distance;
}

void JHQDistanceComputer::apply_rotation_to_query(const float* x)
{
    if (index.use_jl_transform && index.is_rotation_trained) {
        const float* rot_matrix = index.rotation_matrix.data();
        for (int i = 0; i < index.d; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < index.d; ++j) {
                sum += x[j] * rot_matrix[i * index.d + j];
            }
            query_rotated[i] = sum;
        }
    } else {
        std::memcpy(query_rotated.data(), x, sizeof(float) * index.d);
    }
}

void JHQDistanceComputer::compute_primary_distance_table()
{
    for (int m = 0; m < M; ++m) {
        const float* query_subspace = query_rotated.data() + m * Ds;
        const std::vector<float>& centroids = index.codewords[m][0];

        float* table_m_ptr = primary_distance_table_flat.data() + m * K0;

        for (int k = 0; k < K0; ++k) {
            const float* centroid = centroids.data() + k * Ds;
            table_m_ptr[k] = jhq_internal::fvec_L2sqr_dispatch(query_subspace, centroid, Ds);
        }
    }
}

void JHQDistanceComputer::compute_residual_distance_tables()
{
    size_t offset = 0;

    for (int level = 1; level < num_levels; ++level) {
        residual_table_offsets[level] = offset;
        const int K = 1 << index.level_bits[level];

        for (int m = 0; m < M; ++m) {
            const float* query_subspace = query_rotated.data() + m * Ds;
            const auto& scalar_codebook = index.scalar_codebooks[m][level - 1];

            for (int d = 0; d < Ds; ++d) {
                const float query_val = query_subspace[d];

                for (int k = 0; k < K; ++k) {
                    const float diff = query_val - scalar_codebook[k];
                    residual_distance_tables_flat[offset++] = diff * diff;
                }
            }
        }
    }
}

void JHQDistanceComputer::compute_residual_buffer_sizes()
{
    residual_table_offsets.resize(num_levels);

    size_t total_residual_size = 0;
    for (int level = 1; level < num_levels; ++level) {
        const int K = 1 << index.level_bits[level];
        total_residual_size += static_cast<size_t>(M) * Ds * K;
    }

    residual_distance_tables_flat.resize(total_residual_size);
}

float JHQDistanceComputer::compute_precomputed_residual_distance(size_t vector_idx) const
{
    if (!index.has_pre_decoded_codes() || index.num_levels <= 1) {
        return 0.0f;
    }

    const uint8_t* residual_codes = index.separated_codes_.get_residual_codes(vector_idx);
    float residual_distance = 0.0f;
    size_t residual_offset = 0;

    for (int m = 0; m < M; ++m) {
        for (int level = 1; level < num_levels; ++level) {
            const size_t level_table_offset = residual_table_offsets[level];
            const int K_res = 1 << index.level_bits[level];

            for (int d = 0; d < Ds; ++d) {
                const uint8_t scalar_id = residual_codes[residual_offset++];
                const size_t table_idx = level_table_offset + m * Ds * K_res + d * K_res + scalar_id;
                residual_distance += residual_distance_tables_flat[table_idx];
            }
        }
    }

    return residual_distance;
}

FlatCodesDistanceComputer* create_specialized_jhq_distance_computer(const IndexJHQ& index)
{
    return new JHQDistanceComputer(index);
}

void write_index_jhq(const IndexJHQ* idx, IOWriter* f)
{
    uint32_t magic = 0x4A525051;
    f->operator()(&magic, sizeof(magic), 1);

    f->operator()(&idx->d, sizeof(idx->d), 1);
    f->operator()(&idx->ntotal, sizeof(idx->ntotal), 1);
    f->operator()(&idx->is_trained, sizeof(idx->is_trained), 1);
    f->operator()(&idx->metric_type, sizeof(idx->metric_type), 1);
    f->operator()(&idx->code_size, sizeof(idx->code_size), 1);

    f->operator()(&idx->M, sizeof(idx->M), 1);
    f->operator()(&idx->Ds, sizeof(idx->Ds), 1);
    f->operator()(&idx->num_levels, sizeof(idx->num_levels), 1);

    size_t level_bits_size = idx->level_bits.size();
    f->operator()(&level_bits_size, sizeof(level_bits_size), 1);
    if (level_bits_size > 0) {
        f->operator()(idx->level_bits.data(), sizeof(int), level_bits_size);
    }

    f->operator()(&idx->use_jl_transform, sizeof(idx->use_jl_transform), 1);
    f->operator()(&idx->use_analytical_init, sizeof(idx->use_analytical_init), 1);
    f->operator()(&idx->default_oversampling, sizeof(idx->default_oversampling), 1);
    f->operator()(&idx->verbose, sizeof(idx->verbose), 1);

    f->operator()(&idx->use_kmeans_refinement, sizeof(idx->use_kmeans_refinement), 1);
    f->operator()(&idx->kmeans_niter, sizeof(idx->kmeans_niter), 1);
    f->operator()(&idx->kmeans_seed, sizeof(idx->kmeans_seed), 1);

    f->operator()(&idx->is_rotation_trained, sizeof(idx->is_rotation_trained), 1);

    if (idx->use_jl_transform && idx->is_rotation_trained) {
        size_t rot_size = idx->rotation_matrix.size();
        f->operator()(&rot_size, sizeof(rot_size), 1);
        if (rot_size > 0) {
            f->operator()(idx->rotation_matrix.data(), sizeof(float), rot_size);
        }
    }

    for (int m = 0; m < idx->M; ++m) {
        for (int level = 0; level < idx->num_levels; ++level) {
            size_t codeword_size = idx->codewords[m][level].size();
            f->operator()(&codeword_size, sizeof(codeword_size), 1);
            if (codeword_size > 0) {
                f->operator()(idx->codewords[m][level].data(), sizeof(float), codeword_size);
            }
        }
    }

    for (int m = 0; m < idx->M; ++m) {
        for (int level = 0; level < idx->num_levels - 1; ++level) {
            size_t codebook_size = idx->scalar_codebooks[m][level].size();
            f->operator()(&codebook_size, sizeof(codebook_size), 1);
            if (codebook_size > 0) {
                f->operator()(idx->scalar_codebooks[m][level].data(), sizeof(float), codebook_size);
            }
        }
    }

    size_t codes_size = idx->codes.size();
    f->operator()(&codes_size, sizeof(codes_size), 1);
    if (codes_size > 0) {
        f->operator()(idx->codes.data(), sizeof(uint8_t), codes_size);
    }

    f->operator()(&idx->residual_bits_per_subspace, sizeof(idx->residual_bits_per_subspace), 1);

    f->operator()(&idx->memory_layout_initialized_, sizeof(idx->memory_layout_initialized_), 1);
}

IndexJHQ* read_index_jhq(IOReader* f)
{
    uint32_t magic;
    f->operator()(&magic, sizeof(magic), 1);
    FAISS_THROW_IF_NOT_MSG(magic == 0x4A525051, "Invalid JHQ magic number");

    IndexJHQ* idx = new IndexJHQ();

    f->operator()(&idx->d, sizeof(idx->d), 1);
    f->operator()(&idx->ntotal, sizeof(idx->ntotal), 1);
    f->operator()(&idx->is_trained, sizeof(idx->is_trained), 1);
    f->operator()(&idx->metric_type, sizeof(idx->metric_type), 1);
    f->operator()(&idx->code_size, sizeof(idx->code_size), 1);

    f->operator()(&idx->M, sizeof(idx->M), 1);
    f->operator()(&idx->Ds, sizeof(idx->Ds), 1);
    f->operator()(&idx->num_levels, sizeof(idx->num_levels), 1);

    size_t level_bits_size;
    f->operator()(&level_bits_size, sizeof(level_bits_size), 1);
    idx->level_bits.resize(level_bits_size);
    if (level_bits_size > 0) {
        f->operator()(idx->level_bits.data(), sizeof(int), level_bits_size);
    }

    f->operator()(&idx->use_jl_transform, sizeof(idx->use_jl_transform), 1);
    f->operator()(&idx->use_analytical_init, sizeof(idx->use_analytical_init), 1);
    f->operator()(&idx->default_oversampling, sizeof(idx->default_oversampling), 1);
    f->operator()(&idx->verbose, sizeof(idx->verbose), 1);

    f->operator()(&idx->use_kmeans_refinement, sizeof(idx->use_kmeans_refinement), 1);
    f->operator()(&idx->kmeans_niter, sizeof(idx->kmeans_niter), 1);
    f->operator()(&idx->kmeans_seed, sizeof(idx->kmeans_seed), 1);

    f->operator()(&idx->is_rotation_trained, sizeof(idx->is_rotation_trained), 1);

    if (idx->use_jl_transform && idx->is_rotation_trained) {
        size_t rot_size;
        f->operator()(&rot_size, sizeof(rot_size), 1);
        idx->rotation_matrix.resize(rot_size);
        if (rot_size > 0) {
            f->operator()(idx->rotation_matrix.data(), sizeof(float), rot_size);
        }
    }

    idx->initialize_data_structures();

    for (int m = 0; m < idx->M; ++m) {
        for (int level = 0; level < idx->num_levels; ++level) {
            size_t codeword_size;
            f->operator()(&codeword_size, sizeof(codeword_size), 1);
            idx->codewords[m][level].resize(codeword_size);
            if (codeword_size > 0) {
                f->operator()(idx->codewords[m][level].data(), sizeof(float), codeword_size);
            }
        }
    }

    for (int m = 0; m < idx->M; ++m) {
        for (int level = 0; level < idx->num_levels - 1; ++level) {
            size_t codebook_size;
            f->operator()(&codebook_size, sizeof(codebook_size), 1);
            idx->scalar_codebooks[m][level].resize(codebook_size);
            if (codebook_size > 0) {
                f->operator()(idx->scalar_codebooks[m][level].data(), sizeof(float), codebook_size);
            }
        }
    }

    size_t codes_size;
    f->operator()(&codes_size, sizeof(codes_size), 1);
    idx->codes.resize(codes_size);
    if (codes_size > 0) {
        f->operator()(idx->codes.data(), sizeof(uint8_t), codes_size);
    }

    size_t saved_residual_bits_per_subspace;
    f->operator()(&saved_residual_bits_per_subspace, sizeof(saved_residual_bits_per_subspace), 1);

    bool saved_memory_layout_initialized;
    f->operator()(&saved_memory_layout_initialized, sizeof(saved_memory_layout_initialized), 1);

    if (idx->ntotal > 0 && idx->is_trained) {
        idx->residual_bits_per_subspace = 0;
        for (int level = 1; level < idx->num_levels; ++level) {
            idx->residual_bits_per_subspace += static_cast<size_t>(idx->Ds) * idx->level_bits[level];
        }

        if (!idx->is_rotation_trained && idx->use_jl_transform) {
            if (idx->verbose) {
                std::cout << "WARNING: JL transform enabled but rotation not trained, fixing..." << std::endl;
            }
            idx->generate_qr_rotation_matrix(42);
        }

        idx->memory_layout_initialized_ = false;
        idx->initialize_memory_layout();
        idx->extract_all_codes_after_add();

        size_t expected_codes_size = idx->ntotal * idx->code_size;
        if (idx->codes.size() != expected_codes_size) {
            if (idx->verbose) {
                std::cout << "WARNING: Codes size mismatch! Expected: " << expected_codes_size
                          << ", Got: " << idx->codes.size() << std::endl;
            }
        }

        if (idx->verbose) {
            std::cout << "Completed comprehensive IndexJHQ state restoration:" << std::endl;
            std::cout << "  - Vectors: " << idx->ntotal << std::endl;
            std::cout << "  - Code size: " << idx->code_size << " bytes" << std::endl;
            std::cout << "  - Residual bits per subspace: " << idx->residual_bits_per_subspace << std::endl;
            std::cout << "  - Memory layout: " << (idx->memory_layout_initialized_ ? "optimized" : "standard") << std::endl;
            std::cout << "  - JL transform: " << (idx->is_rotation_trained ? "ready" : "disabled") << std::endl;
        }
    }

    return idx;
}

void IndexJHQ::encode_to_separated_storage(idx_t n, const float* x_rotated) const
{
    const idx_t old_total = ntotal;

    const_cast<jhq_internal::PreDecodedCodes&>(separated_codes_).initialize(M, Ds, num_levels, old_total + n);

    if (old_total > 0 && codes.size() > 0) {
        if (verbose) {
            std::cout << "Migrating " << old_total << " existing vectors to separated storage..." << std::endl;
        }

#pragma omp parallel for schedule(static) if (old_total > 1000)
        for (idx_t i = 0; i < old_total; ++i) {
            extract_single_vector_all_codes(i);
        }
    }

    if (verbose) {
        std::cout << "Encoding " << n << " new vectors to separated storage..." << std::endl;
    }

#pragma omp parallel for schedule(static) if (n > 1000)
    for (idx_t i = 0; i < n; ++i) {
        encode_single_vector_separated(
            x_rotated + i * d,
            old_total + i);
    }
}

void IndexJHQ::encode_single_vector_separated(const float* x, idx_t vector_idx) const
{
    uint8_t* primary_dest = const_cast<uint8_t*>(separated_codes_.get_primary_codes(vector_idx));
    uint8_t* residual_dest = (num_levels > 1) ? const_cast<uint8_t*>(separated_codes_.get_residual_codes(vector_idx)) : nullptr;

    size_t residual_offset = 0;

    for (int m = 0; m < M; ++m) {
        const float* subspace_vector = x + m * Ds;
        std::vector<float> current_residual(subspace_vector, subspace_vector + Ds);

        const auto& centroids = codewords[m][0];
        int K = static_cast<int>(centroids.size() / Ds);
        int best_k = find_best_centroid(current_residual.data(), centroids, K);
        primary_dest[m] = static_cast<uint8_t>(best_k);

        subtract_centroid(current_residual.data(), centroids, best_k);

        if (residual_dest && num_levels > 1) {
            encode_residual_levels_separated(m, current_residual.data(), residual_dest, residual_offset);
        }
    }
}

int IndexJHQ::find_best_centroid(const float* residual, const std::vector<float>& centroids, int K) const
{
    int best_k = 0;
    float best_dist = std::numeric_limits<float>::max();

#ifdef __AVX512F__
    if (K >= 16 && Ds >= 8) {
        __m512 best_dists = _mm512_set1_ps(std::numeric_limits<float>::max());
        __m512i best_indices = _mm512_setzero_si512();

        for (int k = 0; k + 15 < K; k += 16) {
            __m512 dists = _mm512_setzero_ps();

            for (int d = 0; d < Ds; ++d) {
                __m512 query_val = _mm512_set1_ps(residual[d]);

                alignas(64) float centroid_vals[16];
                for (int i = 0; i < 16; ++i) {
                    centroid_vals[i] = centroids[(k + i) * Ds + d];
                }
                __m512 cent_vals = _mm512_load_ps(centroid_vals);

                __m512 diff = _mm512_sub_ps(query_val, cent_vals);
                dists = _mm512_fmadd_ps(diff, diff, dists);
            }

            __mmask16 mask = _mm512_cmp_ps_mask(dists, best_dists, _CMP_LT_OQ);
            best_dists = _mm512_mask_blend_ps(mask, best_dists, dists);

            __m512i current_indices = _mm512_set_epi32(
                k + 15, k + 14, k + 13, k + 12, k + 11, k + 10, k + 9, k + 8,
                k + 7, k + 6, k + 5, k + 4, k + 3, k + 2, k + 1, k);
            best_indices = _mm512_mask_blend_epi32(mask, best_indices, current_indices);
        }

        alignas(64) float final_dists[16];
        alignas(64) int final_indices[16];
        _mm512_store_ps(final_dists, best_dists);
        _mm512_store_si512((__m512i*)final_indices, best_indices);

        for (int i = 0; i < 16; ++i) {
            if (final_dists[i] < best_dist) {
                best_dist = final_dists[i];
                best_k = final_indices[i];
            }
        }

        for (int k = (K / 16) * 16; k < K; ++k) {
            float dist = jhq_internal::fvec_L2sqr_dispatch(
                residual, centroids.data() + k * Ds, Ds);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }
    } else
#endif
    {
        for (int k = 0; k < K; ++k) {
            float dist = jhq_internal::fvec_L2sqr_dispatch(
                residual, centroids.data() + k * Ds, Ds);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }
    }

    return best_k;
}

void IndexJHQ::subtract_centroid(float* residual, const std::vector<float>& centroids, int best_k) const
{
    const float* best_centroid = centroids.data() + best_k * Ds;

#ifdef __AVX512F__
    if (Ds >= 16) {
        int d = 0;
        for (; d + 15 < Ds; d += 16) {
            __m512 residual_vals = _mm512_loadu_ps(&residual[d]);
            __m512 centroid_vals = _mm512_loadu_ps(&best_centroid[d]);
            __m512 result = _mm512_sub_ps(residual_vals, centroid_vals);
            _mm512_storeu_ps(&residual[d], result);
        }
        for (; d < Ds; ++d) {
            residual[d] -= best_centroid[d];
        }
    } else
#endif
    {
        for (int d = 0; d < Ds; ++d) {
            residual[d] -= best_centroid[d];
        }
    }
}

void IndexJHQ::encode_residual_levels_separated(int m, const float* residual, uint8_t* residual_dest, size_t& offset) const
{
    for (int level = 1; level < num_levels; ++level) {
        const auto& scalar_codebook = scalar_codebooks[m][level - 1];
        int K = static_cast<int>(scalar_codebook.size());

        for (int d = 0; d < Ds; ++d) {
            int best_k = 0;
            float best_dist = std::abs(residual[d] - scalar_codebook[0]);

            for (int k = 1; k < K; ++k) {
                float dist = std::abs(residual[d] - scalar_codebook[k]);
                if (dist < best_dist) {
                    best_dist = dist;
                    best_k = k;
                }
            }

            residual_dest[offset++] = static_cast<uint8_t>(best_k);
        }
    }
}

float IndexJHQ::compute_exact_distance_separated(idx_t vector_idx, const float* query_rotated) const
{
    const uint8_t* primary_codes = separated_codes_.get_primary_codes(vector_idx);
    const uint8_t* residual_codes = (num_levels > 1) ? separated_codes_.get_residual_codes(vector_idx) : nullptr;

    float total_distance = 0.0f;
    size_t residual_offset = 0;

    for (int m = 0; m < M; ++m) {
        const float* query_sub = query_rotated + m * Ds;

        uint8_t centroid_id = primary_codes[m];
        const auto& centroids = codewords[m][0];
        const float* primary_centroid = centroids.data() + centroid_id * Ds;

        std::vector<float> query_residual(Ds);
        for (int d = 0; d < Ds; ++d) {
            query_residual[d] = query_sub[d] - primary_centroid[d];
        }

        std::vector<float> db_residual(Ds, 0.0f);
        if (residual_codes && num_levels > 1) {
            for (int level = 1; level < num_levels; ++level) {
                const auto& scalar_codebook = scalar_codebooks[m][level - 1];
                for (int d = 0; d < Ds; ++d) {
                    uint8_t scalar_id = residual_codes[residual_offset++];
                    db_residual[d] += scalar_codebook[scalar_id];
                }
            }
        }

        total_distance += jhq_internal::fvec_L2sqr_dispatch(
            query_residual.data(), db_residual.data(), Ds);
    }

    return total_distance;
}
} // namespace faiss
