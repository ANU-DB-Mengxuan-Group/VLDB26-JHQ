#include "IndexJHQ.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>

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

#if defined(QIG_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(QIG_USE_LAPACK)
#include <lapacke.h>
#endif

#include <faiss/utils/prefetch.h>

namespace faiss {

namespace {

std::vector<float> sample_subspace_rows(
    const float* data,
    idx_t n,
    int Ds,
    idx_t sample_n,
    uint64_t seed,
    bool random_sample) {
    FAISS_THROW_IF_NOT_MSG(data != nullptr, "sample_subspace_rows: data is null");
    FAISS_THROW_IF_NOT_MSG(n > 0, "sample_subspace_rows: n must be > 0");
    FAISS_THROW_IF_NOT_MSG(Ds > 0, "sample_subspace_rows: Ds must be > 0");
    FAISS_THROW_IF_NOT_MSG(sample_n > 0 && sample_n <= n,
        "sample_subspace_rows: invalid sample_n");

    std::vector<float> out(static_cast<size_t>(sample_n) * static_cast<size_t>(Ds));
    std::vector<idx_t> picked(static_cast<size_t>(sample_n));

    if (random_sample) {
        
        for (idx_t i = 0; i < sample_n; ++i) {
            picked[static_cast<size_t>(i)] = i;
        }
        std::mt19937_64 rng(seed);
        for (idx_t i = sample_n; i < n; ++i) {
            std::uniform_int_distribution<idx_t> dist(0, i);
            const idx_t j = dist(rng);
            if (j < sample_n) {
                picked[static_cast<size_t>(j)] = i;
            }
        }
        std::sort(picked.begin(), picked.end());
    } else {
        
        for (idx_t i = 0; i < sample_n; ++i) {
            picked[static_cast<size_t>(i)] =
                std::min<idx_t>(n - 1, (i * n) / sample_n);
        }
    }

    for (idx_t i = 0; i < sample_n; ++i) {
        const idx_t src = picked[static_cast<size_t>(i)];
        std::memcpy(
            out.data() + static_cast<size_t>(i) * static_cast<size_t>(Ds),
            data + static_cast<size_t>(src) * static_cast<size_t>(Ds),
            static_cast<size_t>(Ds) * sizeof(float));
    }
    return out;
}

std::vector<float> sample_scalar_values(
    const float* values,
    size_t total_values,
    size_t sample_n,
    uint64_t seed,
    bool random_sample) {
    FAISS_THROW_IF_NOT_MSG(values != nullptr, "sample_scalar_values: values is null");
    FAISS_THROW_IF_NOT_MSG(total_values > 0, "sample_scalar_values: total_values must be > 0");
    FAISS_THROW_IF_NOT_MSG(sample_n > 0 && sample_n <= total_values,
        "sample_scalar_values: invalid sample_n");

    std::vector<float> samples(sample_n);
    if (sample_n == total_values) {
        std::memcpy(samples.data(), values, sample_n * sizeof(float));
        return samples;
    }

    if (random_sample) {
        
        for (size_t i = 0; i < sample_n; ++i) {
            samples[i] = values[i];
        }
        std::mt19937_64 rng(seed);
        for (size_t i = sample_n; i < total_values; ++i) {
            std::uniform_int_distribution<size_t> dist(0, i);
            const size_t j = dist(rng);
            if (j < sample_n) {
                samples[j] = values[i];
            }
        }
    } else {
        
        for (size_t i = 0; i < sample_n; ++i) {
            const size_t idx = std::min(total_values - 1, (i * total_values) / sample_n);
            samples[i] = values[idx];
        }
    }

    return samples;
}

void refine_scalar_codebook_lloyd_1d(
    std::vector<float>& codebook,
    const std::vector<float>& sorted_samples,
    int niter) {
    if (codebook.empty() || sorted_samples.empty() || niter <= 0) {
        return;
    }

    const int K = static_cast<int>(codebook.size());
    if (K <= 1) {
        return;
    }

    std::vector<float> boundaries(static_cast<size_t>(K - 1), 0.0f);
    std::vector<double> sums(static_cast<size_t>(K), 0.0);
    std::vector<size_t> counts(static_cast<size_t>(K), 0u);

    for (int iter = 0; iter < niter; ++iter) {
        std::sort(codebook.begin(), codebook.end());
        for (int k = 0; k + 1 < K; ++k) {
            boundaries[static_cast<size_t>(k)] =
                0.5f * (codebook[static_cast<size_t>(k)] +
                        codebook[static_cast<size_t>(k + 1)]);
        }

        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0u);

        int cur = 0;
        for (const float v : sorted_samples) {
            while (cur + 1 < K &&
                   v > boundaries[static_cast<size_t>(cur)]) {
                ++cur;
            }
            sums[static_cast<size_t>(cur)] += static_cast<double>(v);
            counts[static_cast<size_t>(cur)]++;
        }

        bool changed = false;
        for (int k = 0; k < K; ++k) {
            const size_t kk = static_cast<size_t>(k);
            if (counts[kk] == 0u) {
                continue;
            }
            const float updated =
                static_cast<float>(sums[kk] / static_cast<double>(counts[kk]));
            changed = changed || (std::fabs(updated - codebook[kk]) > 1e-6f);
            codebook[kk] = updated;
        }

        if (!changed) {
            break;
        }
    }

    std::sort(codebook.begin(), codebook.end());
}

}  

namespace jhq_internal {

namespace {

size_t get_available_memory_bytes()
{
#if defined(__linux__)
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (meminfo) {
        char line[256];
        while (fgets(line, sizeof(line), meminfo)) {
            size_t mem_kb = 0;
            if (sscanf(line, "MemAvailable: %zu kB", &mem_kb) == 1) {
                fclose(meminfo);
                return mem_kb * 1024;
            }
        }
        fclose(meminfo);
    }
    return 0;
#elif defined(__APPLE__)
    int mib[2] = {CTL_HW, HW_MEMSIZE};
    size_t mem_size = 0;
    size_t len = sizeof(mem_size);
    if (sysctl(mib, 2, &mem_size, &len, nullptr, 0) == 0) {
        return mem_size / 2;
    }
    return 0;
#else
    return 0;
#endif
}

} 

size_t get_max_batch_memory_bytes(const char* env_var)
{
    const char* env_val = std::getenv(env_var);
    if (env_val) {
        const size_t mb = std::strtoull(env_val, nullptr, 10);
        if (mb > 0) {
            return mb * 1024 * 1024;
        }
    }

    const size_t available = get_available_memory_bytes();
    if (available > 0) {
        return std::min<size_t>(available / 2, 16ULL * 1024 * 1024 * 1024);
    }

    return DEFAULT_BATCH_MEMORY_BYTES;
}

} 

void IndexJHQ::train(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT_MSG(n > 0, "Training set cannot be empty");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Training data cannot be null");

    if (use_jl_transform && !is_rotation_trained) {
        generate_qr_rotation_matrix(1234);
    }

    
    const size_t bytes_per_vector = 3ULL * static_cast<size_t>(d) * sizeof(float);
    const size_t max_memory_bytes = jhq_internal::get_max_batch_memory_bytes("QIG_MAX_TRAIN_MEMORY_MB");
    const idx_t batch_size = std::min(
        n,
        std::max(static_cast<idx_t>(1000),
            static_cast<idx_t>(max_memory_bytes / bytes_per_vector)));

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

        if (normalize_l2) {
            fvec_renorm_L2(static_cast<size_t>(d),
                           static_cast<size_t>(current_batch_size),
                           x_batch_rotated.data());
        }

        for (idx_t i = 0; i < current_batch_size; ++i) {
            for (int m = 0; m < M; ++m) {
                const float* subspace_start = x_batch_rotated.data() + i * d + m * Ds;
                subspace_data[m].insert(subspace_data[m].end(),
                    subspace_start,
                    subspace_start + Ds);
            }
        }

    }

    
    struct OpenBlasThreadGuard {
        int prev = 0;
        explicit OpenBlasThreadGuard(int threads) {
            prev = openblas_get_num_threads();
            openblas_set_num_threads(threads);
        }
        ~OpenBlasThreadGuard() {
            openblas_set_num_threads(prev);
        }
        OpenBlasThreadGuard(const OpenBlasThreadGuard&) = delete;
        OpenBlasThreadGuard& operator=(const OpenBlasThreadGuard&) = delete;
    };

    {
        const OpenBlasThreadGuard ob_guard(1);
#pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            train_subspace_quantizers(m, n, std::move(subspace_data[m]), kmeans_seed + m);
        }
    }

    rebuild_scalar_codebooks_flat();
    is_trained = true;
    initialize_memory_layout();
    mark_residual_tables_dirty();
}

void IndexJHQ::train_subspace_quantizers(
    int subspace_idx,
    idx_t n,
    std::vector<float>&& subspace_data,
    int random_seed)
{
    std::vector<float> current_residuals = std::move(subspace_data);
    FAISS_THROW_IF_NOT_MSG(
        current_residuals.size() == static_cast<size_t>(n) * static_cast<size_t>(Ds),
        "train_subspace_quantizers: subspace data size mismatch");

    for (int level = 0; level < num_levels; ++level) {
        int K = 1 << level_bits[level];

        if (level == 0) {
            idx_t train_n = n;
            const float* train_data = current_residuals.data();
            std::vector<float> sampled_rows;
            if (sample_primary > 0 && sample_primary < n) {
                train_n = sample_primary;
                sampled_rows = sample_subspace_rows(
                    current_residuals.data(),
                    n,
                    Ds,
                    train_n,
                    static_cast<uint64_t>(kmeans_seed) +
                        static_cast<uint64_t>(subspace_idx) * 131ULL,
                    random_sample_training);
                train_data = sampled_rows.data();
            }
            train_primary_level(subspace_idx, train_n, train_data, K, random_seed + level);
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
        cp.nredo = std::max(1, kmeans_nredo);
        cp.seed = kmeans_seed + random_seed;
    } else {
        cp.niter = 0;
        cp.nredo = 0;
        cp.seed = kmeans_seed + random_seed;
    }

    Clustering clustering(Ds, K, cp);

    std::vector<float> init_centroids(K * Ds);
    analytical_gaussian_init(data, n, Ds, K, init_centroids.data());
    clustering.centroids.assign(init_centroids.begin(), init_centroids.end());

    IndexFlatL2 clustering_index(Ds);

    try {
        clustering.train(n, data, clustering_index);
    } catch (const FaissException& e) {
        std::mt19937 rng(random_seed);
        std::normal_distribution<float> dist(0.0f, 0.1f);
        for (int i = 0; i < K * Ds; ++i) {
            clustering.centroids[i] = dist(rng);
        }
    }

    FAISS_THROW_IF_NOT(clustering.centroids.size() == K * Ds);
    FAISS_THROW_IF_NOT_MSG(
        K == primary_ksub(),
        "train_primary_level: K mismatch with primary PQ ksub");
    float* dst = get_primary_centroids_ptr_mutable(subspace_idx);
    std::memcpy(
        dst,
        clustering.centroids.data(),
        static_cast<size_t>(K) * static_cast<size_t>(Ds) * sizeof(float));
    primary_pq_dirty_ = false;
}

void IndexJHQ::train_residual_level(
    int subspace_idx,
    int level,
    idx_t n,
    const float* residuals,
    int K)
{
    std::vector<float> scalar_codebook(static_cast<size_t>(std::max(0, K)), 0.0f);
    if (K <= 0) {
        float* dst = get_scalar_codebook_ptr_mutable(subspace_idx, level);
        (void)dst;
        mark_residual_tables_dirty();
        return;
    }

    const size_t total_values = static_cast<size_t>(n) * static_cast<size_t>(Ds);

    if (total_values == 0) {
        std::fill(scalar_codebook.begin(), scalar_codebook.end(), 0.0f);
    } else if (use_analytical_init && K > 2) {
        const size_t target_samples = (sample_residual > 0)
            ? std::min(total_values, static_cast<size_t>(sample_residual))
            : total_values;
        std::vector<float> samples = sample_scalar_values(
            residuals,
            total_values,
            target_samples,
            static_cast<uint64_t>(kmeans_seed) +
                static_cast<uint64_t>(subspace_idx) * 257ULL +
                static_cast<uint64_t>(level) * 17ULL,
            random_sample_training);

        std::sort(samples.begin(), samples.end());

        for (int k = 0; k < K; ++k) {
            float quantile = static_cast<float>(k) / (K - 1);
            size_t idx = static_cast<size_t>(quantile * (samples.size() - 1));
            scalar_codebook[k] = samples[idx];
        }

        if (use_kmeans_refinement && kmeans_niter > 0) {
            refine_scalar_codebook_lloyd_1d(
                scalar_codebook,
                samples,
                kmeans_niter);
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

    float* dst = get_scalar_codebook_ptr_mutable(subspace_idx, level);
    FAISS_THROW_IF_NOT_MSG(dst != nullptr, "train_residual_level: null scalar codebook destination");
    std::memcpy(dst, scalar_codebook.data(), static_cast<size_t>(K) * sizeof(float));

    mark_residual_tables_dirty();
}

void IndexJHQ::update_residuals_after_level(
    int subspace_idx,
    int level,
    idx_t n,
    float* residuals)
{
    if (level == 0) {
        const float* centroids = get_primary_centroids_ptr(subspace_idx);
        const int K = primary_ksub();

        IndexFlatL2 quantizer(Ds);
        quantizer.add(K, centroids);

        std::vector<idx_t> labels(n);
        std::vector<float> dis(n);
        quantizer.search(n, residuals, 1, dis.data(), labels.data());

#pragma omp parallel for schedule(static)
        for (idx_t i = 0; i < n; ++i) {
            const float* best_centroid = centroids + labels[i] * Ds;
            float* current_residual = residuals + i * Ds;
            for (int d = 0; d < Ds; ++d) {
                current_residual[d] -= best_centroid[d];
            }
        }
        return;
    }

    const float* scalar_codebook = get_scalar_codebook_ptr(subspace_idx, level);
    const int K = scalar_codebook_ksub(level);

#pragma omp parallel for schedule(static) if (n > 10000)
    for (idx_t i = 0; i < n; ++i) {
        float* current_residual = residuals + i * Ds;

        int d = 0;

#if defined(__AVX512F__)
        for (; d + 15 < Ds; d += 16) {
            if (d + 32 < Ds) {
                prefetch_L1(&current_residual[d + 16]);
            }

            __m512 vals = _mm512_loadu_ps(&current_residual[d]);
            __m512i best_indices = _mm512_setzero_si512();
            __m512 best_dists = _mm512_set1_ps(FLT_MAX);

            int k = 0;

            for (; k + 7 < K; k += 8) {
                if (k + 16 < K) {
                    prefetch_L1(&scalar_codebook[k + 8]);
                }

#pragma unroll 8
                for (int kk = 0; kk < 8; ++kk) {
                    __m512 codebook_val = _mm512_set1_ps(scalar_codebook[k + kk]);
                    __m512 diffs = _mm512_sub_ps(vals, codebook_val);
                    __m512 abs_diffs = _mm512_abs_ps(diffs);

                    __mmask16 mask = _mm512_cmp_ps_mask(abs_diffs, best_dists, _CMP_LT_OQ);
                    best_dists = _mm512_mask_blend_ps(mask, best_dists, abs_diffs);
                    best_indices = _mm512_mask_blend_epi32(mask, best_indices, _mm512_set1_epi32(k + kk));
                }
            }

            for (; k < K; ++k) {
                __m512 codebook_val = _mm512_set1_ps(scalar_codebook[k]);
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
                quantized_vals[j] = scalar_codebook[indices[j]];
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
            const int best_k = find_nearest_scalar_sorted(
                scalar_codebook, K, current_residual[d]);
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

    rotation_matrix.resize(static_cast<size_t>(d) * d);
    faiss::float_randn(rotation_matrix.data(), static_cast<size_t>(d) * d, random_seed);

#if defined(QIG_USE_ACCELERATE)
    
    
    {
        std::vector<float> col_major(static_cast<size_t>(d) * d);
        
        for (int i = 0; i < d; ++i) {
            for (int j = 0; j < d; ++j) {
                col_major[j * d + i] = rotation_matrix[i * d + j];
            }
        }

        std::vector<float> tau(static_cast<size_t>(d));
        __CLPK_integer n = d;
        __CLPK_integer lda = d;
        __CLPK_integer lwork = d * 64;  
        std::vector<float> work(lwork);
        __CLPK_integer info = 0;

        
        sgeqrf_(&n, &n, col_major.data(), &lda, tau.data(),
                work.data(), &lwork, &info);

        if (info == 0) {
            
            sorgqr_(&n, &n, &n, col_major.data(), &lda, tau.data(),
                    work.data(), &lwork, &info);
        }

        if (info == 0) {
            
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    rotation_matrix[i * d + j] = col_major[j * d + i];
                }
            }
            is_rotation_trained = true;
            return;
        }

        FAISS_THROW_FMT("Accelerate LAPACK QR decomposition failed with info = %d", info);
    }
#elif defined(QIG_USE_LAPACK)
    {
        std::vector<float> tau(static_cast<size_t>(d));
        lapack_int info = LAPACKE_sgeqrf(
            LAPACK_ROW_MAJOR, d, d,
            rotation_matrix.data(), d, tau.data());

        if (info == 0) {
            info = LAPACKE_sorgqr(
                LAPACK_ROW_MAJOR, d, d, d,
                rotation_matrix.data(), d, tau.data());
        }

        if (info != 0) {
            FAISS_THROW_FMT("LAPACK QR decomposition failed with info = %d", info);
        }

        is_rotation_trained = true;
    }
#else
    FAISS_THROW_MSG(
        "JL rotation requires LAPACK. Install LAPACKE (Linux) or use Apple Accelerate (macOS). "
        "Alternatively, disable JL rotation with use_jl_transform=false.");
#endif
}

void IndexJHQ::apply_jl_rotation(idx_t n, const float* x_in, float* x_out) const
{
    if (!use_jl_transform || !is_rotation_trained || !x_in || !x_out) {
        if (x_in != x_out && x_in && x_out) {
            std::memcpy(x_out, x_in, sizeof(float) * n * d);
        }
        return;
    }

    const size_t expected_rotation_size = static_cast<size_t>(d) * d;
    const bool have_bf16_rotation =
        use_bf16_rotation &&
        !rotation_matrix_bf16.empty() &&
        rotation_matrix_bf16.size() == expected_rotation_size;

    
    if (have_bf16_rotation && n == 1) {
        
        for (idx_t q = 0; q < n; ++q) {
            const float* x = x_in + q * d;
            float* y = x_out + q * d;

            for (int32_t i = 0; i < d; ++i) {
                const uint16_t* row_bf16 = rotation_matrix_bf16.data() + static_cast<size_t>(i) * d;
                int32_t j = 0;
                float sum = 0.0f;

#if defined(__AVX2__)
                __m256 acc = _mm256_setzero_ps();
                for (; j + 8 <= d; j += 8) {
                    __m128i bf = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row_bf16 + j));
                    __m256 rv = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bf), 16));
                    __m256 xv = _mm256_loadu_ps(x + j);
                    acc = _mm256_fmadd_ps(rv, xv, acc);
                }
                
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 s128 = _mm_add_ps(lo, hi);
                s128 = _mm_hadd_ps(s128, s128);
                s128 = _mm_hadd_ps(s128, s128);
                sum = _mm_cvtss_f32(s128);
#endif
                for (; j < d; ++j) {
                    sum += jhq_internal::bf16_to_float(row_bf16[j]) * x[j];
                }
                y[i] = sum;
            }
        }
        return;
    }

    
    
    const float* rotation_ptr = nullptr;
    std::vector<float> rotation_matrix_expanded;

    if (!rotation_matrix.empty()) {
        if (rotation_matrix.size() != expected_rotation_size) {
            if (x_in != x_out) {
                std::memcpy(x_out, x_in, sizeof(float) * n * d);
            }
            return;
        }
        rotation_ptr = rotation_matrix.data();
    } else if (have_bf16_rotation) {
        rotation_matrix_expanded.resize(expected_rotation_size);
        for (size_t i = 0; i < expected_rotation_size; ++i) {
            rotation_matrix_expanded[i] = jhq_internal::bf16_to_float(rotation_matrix_bf16[i]);
        }
        rotation_ptr = rotation_matrix_expanded.data();
    } else {
        if (x_in != x_out) {
            std::memcpy(x_out, x_in, sizeof(float) * n * d);
        }
        return;
    }

    
    if (n == 1) {
        cblas_sgemv(CblasRowMajor,
            CblasNoTrans,
            d,
            d,
            1.0f,
            rotation_ptr,
            d,
            x_in,
            1,
            0.0f,
            x_out,
            1);
        return;
    }

    
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
        static_cast<int>(n), d, d,
        1.0f, x_in, d,
        rotation_ptr, d,
        0.0f, x_out, d);
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

    if (normalize_l2) {
        fvec_renorm_L2(static_cast<size_t>(d),
                       static_cast<size_t>(n),
                       x_rotated.data());
    }

    if (bytes == nullptr) {
        encode_to_separated_storage(n, x_rotated.data());
        return;
    }

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual_scratch(static_cast<size_t>(Ds));
#pragma omp for schedule(static)
        for (idx_t i = 0; i < n; ++i) {
            encode_single_vector_with_scratch(
                x_rotated.data() + i * d,
                bytes + i * code_size,
                residual_scratch.data());
        }
    }
}

void IndexJHQ::encode_single_vector(const float* x, uint8_t* code) const
{
    std::vector<float> residual_scratch(static_cast<size_t>(Ds));
    encode_single_vector_with_scratch(x, code, residual_scratch.data());
}

void IndexJHQ::encode_single_vector_with_scratch(
    const float* x,
    uint8_t* code,
    float* current_residual) const
{
    faiss::BitstringWriter bit_writer(code, code_size);
    std::memset(code, 0, code_size);

    for (int m = 0; m < M; ++m) {
        const float* subspace_vector = x + m * Ds;
        std::memcpy(
            current_residual,
            subspace_vector,
            static_cast<size_t>(Ds) * sizeof(float));

        for (int level = 0; level < num_levels; ++level) {
            if (level == 0) {
                const float* centroids = get_primary_centroids_ptr(m);
                const int K = primary_ksub();

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

                            const float* centroid_dim_base = centroids + d;

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
                            current_residual,
                            centroids + k * Ds,
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
                            current_residual,
                            centroids + k * Ds,
                            Ds);

                        if (dist < best_dist) {
                            best_dist = dist;
                            best_k = k;
                        }
                    }
                }

                bit_writer.write(best_k, level_bits[0]);

                const float* best_centroid = centroids + best_k * Ds;
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
                const int K_res = scalar_codebook_ksub(level);
                const float* scalar_codebook = get_scalar_codebook_ptr(m, level);

                for (int d = 0; d < Ds; ++d) {
                    const int best_k = find_nearest_scalar_sorted(
                        scalar_codebook,
                        K_res,
                        current_residual[d]);

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
                const float* centroids = get_primary_centroids_ptr(m);
                const uint32_t max_id = static_cast<uint32_t>(std::max(0, primary_ksub() - 1));
                centroid_id = std::min(centroid_id, max_id);
                const float* centroid = centroids + centroid_id * Ds;

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
                const int K_res = scalar_codebook_ksub(level);
                const float* scalar_codebook = get_scalar_codebook_ptr(m, level);

#ifdef __AVX512F__
                if (Ds >= 16) {
                    int d = 0;
                    for (; d + 15 < Ds; d += 16) {
                        alignas(64) uint32_t scalar_ids[16];
                        for (int i = 0; i < 16; ++i) {
                            uint32_t scalar_id = bit_reader.read(level_bits[level]);
                            scalar_ids[i] = std::min(scalar_id, static_cast<uint32_t>(K_res - 1));
                        }

                        __m512i indices = _mm512_loadu_si512(scalar_ids);
                        __m512 scalar_vals = _mm512_i32gather_ps(indices, scalar_codebook, 4);

                        __m512 subspace_vals = _mm512_loadu_ps(&subspace_vector[d]);
                        __m512 result = _mm512_add_ps(subspace_vals, scalar_vals);
                        _mm512_storeu_ps(&subspace_vector[d], result);
                    }

                    for (; d < Ds; ++d) {
                        uint32_t scalar_id = bit_reader.read(level_bits[level]);
                        if (scalar_id >= static_cast<uint32_t>(K_res))
                            scalar_id = static_cast<uint32_t>(K_res - 1);
                        subspace_vector[d] += scalar_codebook[scalar_id];
                    }
                } else
#endif
                {
                    for (int d = 0; d < Ds; ++d) {
                        uint32_t scalar_id = bit_reader.read(level_bits[level]);
                        if (scalar_id >= static_cast<uint32_t>(K_res))
                            scalar_id = static_cast<uint32_t>(K_res - 1);
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

        
        cblas_sgemv(CblasRowMajor,
            CblasTrans,
            d,
            d,
            1.0f,
            rotation_matrix.data(),
            d,
            temp.data(),
            1,
            0.0f,
            recons,
            1);
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
}

void IndexJHQ::add_pretransformed(idx_t n, const float* x_pretransformed)
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before adding vectors");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of vectors must be positive");
    FAISS_THROW_IF_NOT_MSG(x_pretransformed != nullptr, "Input vectors cannot be null");

    encode_to_separated_storage(n, x_pretransformed);
    ntotal += n;

    codes.resize(0);
}

}  
