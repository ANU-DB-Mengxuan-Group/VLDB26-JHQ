#include "IndexJHQ.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <faiss/utils/prefetch.h>

namespace faiss {

namespace {

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

const bool kHasAVX2 = cpu_supports_avx2_runtime();
const bool kHasAVX512 = cpu_supports_avx512_runtime();

int parse_env_int(const char* name, int default_value)
{
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    if (end == v || (end && *end != '\0')) {
        return default_value;
    }
    if (parsed < static_cast<long>(std::numeric_limits<int>::min()) ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

int choose_primary_prefetch_lookahead(int M, int Ds)
{
    static const int env_override = parse_env_int("JHQ_PRIMARY_PREFETCH_LOOKAHEAD", -1);
    if (env_override >= 0) {
        return std::clamp(env_override, 0, 64);
    }

    
    int lookahead = 12;
    if (M >= 192) {
        lookahead = 8;
    } else if (M >= 128) {
        lookahead = 10;
    } else if (M >= 64) {
        lookahead = 12;
    } else if (M >= 32) {
        lookahead = 16;
    } else {
        lookahead = 20;
    }

    
    if (Ds >= 32) {
        lookahead += 4;
    } else if (Ds <= 8) {
        lookahead -= 2;
    }

    return std::clamp(lookahead, 4, 32);
}

#if defined(__AVX2__)
inline float hsum256_ps(__m256 v)
{
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

inline void compute_primary_ip_lut_ds8_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m256 q = _mm256_loadu_ps(query_subspace);
    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 8;
            const __m256 cv = _mm256_loadu_ps(c);
            const __m256 prod = _mm256_mul_ps(q, cv);
            table_m_ptr[k + u] = -hsum256_ps(prod);
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 8;
        const __m256 cv = _mm256_loadu_ps(c);
        const __m256 prod = _mm256_mul_ps(q, cv);
        table_m_ptr[k] = -hsum256_ps(prod);
    }
}

inline void compute_primary_ip_lut_ds4_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m128 q = _mm_loadu_ps(query_subspace);
    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 4;
            const __m128 cv = _mm_loadu_ps(c);
            const __m128 prod = _mm_mul_ps(q, cv);
            __m128 sum = _mm_hadd_ps(prod, prod);
            sum = _mm_hadd_ps(sum, sum);
            table_m_ptr[k + u] = -_mm_cvtss_f32(sum);
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 4;
        const __m128 cv = _mm_loadu_ps(c);
        const __m128 prod = _mm_mul_ps(q, cv);
        __m128 sum = _mm_hadd_ps(prod, prod);
        sum = _mm_hadd_ps(sum, sum);
        table_m_ptr[k] = -_mm_cvtss_f32(sum);
    }
}

inline void compute_primary_ip_lut_ds16_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m256 q0 = _mm256_loadu_ps(query_subspace);
    const __m256 q1 = _mm256_loadu_ps(query_subspace + 8);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 16;
            const __m256 c0 = _mm256_loadu_ps(c);
            const __m256 c1 = _mm256_loadu_ps(c + 8);
            const __m256 prod0 = _mm256_mul_ps(q0, c0);
            const __m256 prod1 = _mm256_mul_ps(q1, c1);
            table_m_ptr[k + u] = -hsum256_ps(_mm256_add_ps(prod0, prod1));
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 16;
        const __m256 c0 = _mm256_loadu_ps(c);
        const __m256 c1 = _mm256_loadu_ps(c + 8);
        const __m256 prod0 = _mm256_mul_ps(q0, c0);
        const __m256 prod1 = _mm256_mul_ps(q1, c1);
        table_m_ptr[k] = -hsum256_ps(_mm256_add_ps(prod0, prod1));
    }
}

inline void compute_primary_ip_lut_ds32_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m256 q0 = _mm256_loadu_ps(query_subspace);
    const __m256 q1 = _mm256_loadu_ps(query_subspace + 8);
    const __m256 q2 = _mm256_loadu_ps(query_subspace + 16);
    const __m256 q3 = _mm256_loadu_ps(query_subspace + 24);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 32;
            const __m256 c0 = _mm256_loadu_ps(c);
            const __m256 c1 = _mm256_loadu_ps(c + 8);
            const __m256 c2 = _mm256_loadu_ps(c + 16);
            const __m256 c3 = _mm256_loadu_ps(c + 24);
            const __m256 prod0 = _mm256_mul_ps(q0, c0);
            const __m256 prod1 = _mm256_mul_ps(q1, c1);
            const __m256 prod2 = _mm256_mul_ps(q2, c2);
            const __m256 prod3 = _mm256_mul_ps(q3, c3);
            const __m256 sum01 = _mm256_add_ps(prod0, prod1);
            const __m256 sum23 = _mm256_add_ps(prod2, prod3);
            table_m_ptr[k + u] = -hsum256_ps(_mm256_add_ps(sum01, sum23));
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 32;
        const __m256 c0 = _mm256_loadu_ps(c);
        const __m256 c1 = _mm256_loadu_ps(c + 8);
        const __m256 c2 = _mm256_loadu_ps(c + 16);
        const __m256 c3 = _mm256_loadu_ps(c + 24);
        const __m256 prod0 = _mm256_mul_ps(q0, c0);
        const __m256 prod1 = _mm256_mul_ps(q1, c1);
        const __m256 prod2 = _mm256_mul_ps(q2, c2);
        const __m256 prod3 = _mm256_mul_ps(q3, c3);
        const __m256 sum01 = _mm256_add_ps(prod0, prod1);
        const __m256 sum23 = _mm256_add_ps(prod2, prod3);
        table_m_ptr[k] = -hsum256_ps(_mm256_add_ps(sum01, sum23));
    }
}

inline void compute_primary_lut_ds8_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m256 q = _mm256_loadu_ps(query_subspace);
    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 8;
            const __m256 cv = _mm256_loadu_ps(c);
            const __m256 diff = _mm256_sub_ps(q, cv);
            table_m_ptr[k + u] = hsum256_ps(_mm256_mul_ps(diff, diff));
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 8;
        const __m256 cv = _mm256_loadu_ps(c);
        const __m256 diff = _mm256_sub_ps(q, cv);
        table_m_ptr[k] = hsum256_ps(_mm256_mul_ps(diff, diff));
    }
}

inline void compute_primary_lut_ds4_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m128 q = _mm_loadu_ps(query_subspace);
    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 4;
            const __m128 cv = _mm_loadu_ps(c);
            const __m128 diff = _mm_sub_ps(q, cv);
            const __m128 sq = _mm_mul_ps(diff, diff);
            __m128 sum = _mm_hadd_ps(sq, sq);
            sum = _mm_hadd_ps(sum, sum);
            table_m_ptr[k + u] = _mm_cvtss_f32(sum);
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 4;
        const __m128 cv = _mm_loadu_ps(c);
        const __m128 diff = _mm_sub_ps(q, cv);
        const __m128 sq = _mm_mul_ps(diff, diff);
        __m128 sum = _mm_hadd_ps(sq, sq);
        sum = _mm_hadd_ps(sum, sum);
        table_m_ptr[k] = _mm_cvtss_f32(sum);
    }
}

inline void compute_primary_lut_ds16_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m256 q0 = _mm256_loadu_ps(query_subspace);
    const __m256 q1 = _mm256_loadu_ps(query_subspace + 8);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 16;
            const __m256 c0 = _mm256_loadu_ps(c);
            const __m256 c1 = _mm256_loadu_ps(c + 8);
            const __m256 d0 = _mm256_sub_ps(q0, c0);
            const __m256 d1 = _mm256_sub_ps(q1, c1);
            const __m256 sq = _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1));
            table_m_ptr[k + u] = hsum256_ps(sq);
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 16;
        const __m256 c0 = _mm256_loadu_ps(c);
        const __m256 c1 = _mm256_loadu_ps(c + 8);
        const __m256 d0 = _mm256_sub_ps(q0, c0);
        const __m256 d1 = _mm256_sub_ps(q1, c1);
        const __m256 sq = _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1));
        table_m_ptr[k] = hsum256_ps(sq);
    }
}

inline void compute_primary_lut_ds32_avx2(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m256 q0 = _mm256_loadu_ps(query_subspace);
    const __m256 q1 = _mm256_loadu_ps(query_subspace + 8);
    const __m256 q2 = _mm256_loadu_ps(query_subspace + 16);
    const __m256 q3 = _mm256_loadu_ps(query_subspace + 24);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 32;
            const __m256 c0 = _mm256_loadu_ps(c);
            const __m256 c1 = _mm256_loadu_ps(c + 8);
            const __m256 c2 = _mm256_loadu_ps(c + 16);
            const __m256 c3 = _mm256_loadu_ps(c + 24);
            const __m256 d0 = _mm256_sub_ps(q0, c0);
            const __m256 d1 = _mm256_sub_ps(q1, c1);
            const __m256 d2 = _mm256_sub_ps(q2, c2);
            const __m256 d3 = _mm256_sub_ps(q3, c3);
            const __m256 sq01 = _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1));
            const __m256 sq23 = _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3));
            table_m_ptr[k + u] = hsum256_ps(_mm256_add_ps(sq01, sq23));
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 32;
        const __m256 c0 = _mm256_loadu_ps(c);
        const __m256 c1 = _mm256_loadu_ps(c + 8);
        const __m256 c2 = _mm256_loadu_ps(c + 16);
        const __m256 c3 = _mm256_loadu_ps(c + 24);
        const __m256 d0 = _mm256_sub_ps(q0, c0);
        const __m256 d1 = _mm256_sub_ps(q1, c1);
        const __m256 d2 = _mm256_sub_ps(q2, c2);
        const __m256 d3 = _mm256_sub_ps(q3, c3);
        const __m256 sq01 = _mm256_add_ps(_mm256_mul_ps(d0, d0), _mm256_mul_ps(d1, d1));
        const __m256 sq23 = _mm256_add_ps(_mm256_mul_ps(d2, d2), _mm256_mul_ps(d3, d3));
        table_m_ptr[k] = hsum256_ps(_mm256_add_ps(sq01, sq23));
    }
}
#endif

#if defined(__AVX512F__)
inline void compute_primary_ip_lut_ds16_avx512(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m512 q = _mm512_loadu_ps(query_subspace);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 16;
            const __m512 cv = _mm512_loadu_ps(c);
            table_m_ptr[k + u] = -_mm512_reduce_add_ps(_mm512_mul_ps(q, cv));
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 16;
        const __m512 cv = _mm512_loadu_ps(c);
        table_m_ptr[k] = -_mm512_reduce_add_ps(_mm512_mul_ps(q, cv));
    }
}

inline void compute_primary_ip_lut_ds32_avx512(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m512 q0 = _mm512_loadu_ps(query_subspace);
    const __m512 q1 = _mm512_loadu_ps(query_subspace + 16);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 32;
            const __m512 c0 = _mm512_loadu_ps(c);
            const __m512 c1 = _mm512_loadu_ps(c + 16);
            const __m512 prod = _mm512_add_ps(_mm512_mul_ps(q0, c0), _mm512_mul_ps(q1, c1));
            table_m_ptr[k + u] = -_mm512_reduce_add_ps(prod);
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 32;
        const __m512 c0 = _mm512_loadu_ps(c);
        const __m512 c1 = _mm512_loadu_ps(c + 16);
        const __m512 prod = _mm512_add_ps(_mm512_mul_ps(q0, c0), _mm512_mul_ps(q1, c1));
        table_m_ptr[k] = -_mm512_reduce_add_ps(prod);
    }
}

inline void compute_primary_lut_ds16_avx512(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m512 q = _mm512_loadu_ps(query_subspace);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 16;
            const __m512 cv = _mm512_loadu_ps(c);
            const __m512 diff = _mm512_sub_ps(q, cv);
            table_m_ptr[k + u] = _mm512_reduce_add_ps(_mm512_mul_ps(diff, diff));
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 16;
        const __m512 cv = _mm512_loadu_ps(c);
        const __m512 diff = _mm512_sub_ps(q, cv);
        table_m_ptr[k] = _mm512_reduce_add_ps(_mm512_mul_ps(diff, diff));
    }
}

inline void compute_primary_lut_ds32_avx512(
    const float* query_subspace,
    const float* centroids,
    int K0,
    float* table_m_ptr)
{
    const __m512 q0 = _mm512_loadu_ps(query_subspace);
    const __m512 q1 = _mm512_loadu_ps(query_subspace + 16);

    int k = 0;
    for (; k + 3 < K0; k += 4) {
        for (int u = 0; u < 4; ++u) {
            const float* c = centroids + static_cast<size_t>(k + u) * 32;
            const __m512 c0 = _mm512_loadu_ps(c);
            const __m512 c1 = _mm512_loadu_ps(c + 16);
            const __m512 d0 = _mm512_sub_ps(q0, c0);
            const __m512 d1 = _mm512_sub_ps(q1, c1);
            const __m512 sq = _mm512_add_ps(_mm512_mul_ps(d0, d0), _mm512_mul_ps(d1, d1));
            table_m_ptr[k + u] = _mm512_reduce_add_ps(sq);
        }
    }
    for (; k < K0; ++k) {
        const float* c = centroids + static_cast<size_t>(k) * 32;
        const __m512 c0 = _mm512_loadu_ps(c);
        const __m512 c1 = _mm512_loadu_ps(c + 16);
        const __m512 d0 = _mm512_sub_ps(q0, c0);
        const __m512 d1 = _mm512_sub_ps(q1, c1);
        const __m512 sq = _mm512_add_ps(_mm512_mul_ps(d0, d0), _mm512_mul_ps(d1, d1));
        table_m_ptr[k] = _mm512_reduce_add_ps(sq);
    }
}
#endif

#if defined(__AVX2__)
inline void compute_residual_scalar_lut_ip_avx2(
    float query_val,
    const float* codebook,
    int K,
    float* table_ptr)
{
    const __m256 q = _mm256_set1_ps(-query_val);
    int k = 0;
    for (; k + 7 < K; k += 8) {
        const __m256 c = _mm256_loadu_ps(codebook + k);
        _mm256_storeu_ps(table_ptr + k, _mm256_mul_ps(q, c));
    }
    for (; k < K; ++k) {
        table_ptr[k] = -query_val * codebook[k];
    }
}

inline void compute_residual_scalar_lut_avx2(
    float query_val,
    const float* codebook,
    int K,
    float* table_ptr)
{
    const __m256 q = _mm256_set1_ps(query_val);
    int k = 0;
    for (; k + 7 < K; k += 8) {
        const __m256 c = _mm256_loadu_ps(codebook + k);
        const __m256 d = _mm256_sub_ps(q, c);
        _mm256_storeu_ps(table_ptr + k, _mm256_mul_ps(d, d));
    }
    for (; k < K; ++k) {
        const float diff = query_val - codebook[k];
        table_ptr[k] = diff * diff;
    }
}
#endif

#if defined(__AVX512F__)
inline void compute_residual_scalar_lut_ip_avx512(
    float query_val,
    const float* codebook,
    int K,
    float* table_ptr)
{
    const __m512 q = _mm512_set1_ps(-query_val);
    int k = 0;
    for (; k + 15 < K; k += 16) {
        const __m512 c = _mm512_loadu_ps(codebook + k);
        _mm512_storeu_ps(table_ptr + k, _mm512_mul_ps(q, c));
    }
    for (; k < K; ++k) {
        table_ptr[k] = -query_val * codebook[k];
    }
}

inline void compute_residual_scalar_lut_avx512(
    float query_val,
    const float* codebook,
    int K,
    float* table_ptr)
{
    const __m512 q = _mm512_set1_ps(query_val);
    int k = 0;
    for (; k + 15 < K; k += 16) {
        const __m512 c = _mm512_loadu_ps(codebook + k);
        const __m512 d = _mm512_sub_ps(q, c);
        _mm512_storeu_ps(table_ptr + k, _mm512_mul_ps(d, d));
    }
    for (; k < K; ++k) {
        const float diff = query_val - codebook[k];
        table_ptr[k] = diff * diff;
    }
}
#endif

inline void compute_residual_scalar_lut(
    float query_val,
    const float* codebook,
    int K,
    float* table_ptr,
    bool prefer_avx512,
    bool has_avx2,
    bool has_avx512)
{
#if defined(__AVX512F__)
    if (prefer_avx512 && has_avx512) {
        compute_residual_scalar_lut_avx512(query_val, codebook, K, table_ptr);
        return;
    }
#endif
#if defined(__AVX2__)
    if (has_avx2) {
        compute_residual_scalar_lut_avx2(query_val, codebook, K, table_ptr);
        return;
    }
#endif
#if defined(__AVX512F__)
    if (has_avx512) {
        compute_residual_scalar_lut_avx512(query_val, codebook, K, table_ptr);
        return;
    }
#endif
    for (int k = 0; k < K; ++k) {
        const float diff = query_val - codebook[k];
        table_ptr[k] = diff * diff;
    }
}

inline void compute_residual_scalar_lut_ip(
    float query_val,
    const float* codebook,
    int K,
    float* table_ptr,
    bool prefer_avx512,
    bool has_avx2,
    bool has_avx512)
{
    (void)prefer_avx512;
#if defined(__AVX512F__)
    if (prefer_avx512 && has_avx512) {
        compute_residual_scalar_lut_ip_avx512(query_val, codebook, K, table_ptr);
        return;
    }
#endif
#if defined(__AVX2__)
    if (has_avx2) {
        compute_residual_scalar_lut_ip_avx2(query_val, codebook, K, table_ptr);
        return;
    }
#endif
#if defined(__AVX512F__)
    if (has_avx512) {
        compute_residual_scalar_lut_ip_avx512(query_val, codebook, K, table_ptr);
        return;
    }
#endif
    for (int k = 0; k < K; ++k) {
        table_ptr[k] = -query_val * codebook[k];
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

#pragma omp parallel for schedule(dynamic) if (n > 1)
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

    SearchWorkspace& workspace = get_search_workspace();
    float* query_rotated = workspace.query_rotated.data();
    float* primary_distances = workspace.all_primary_distances.data();

    apply_jl_rotation(1, query, query_rotated);
    if (normalize_l2) {
        fvec_renorm_L2(static_cast<size_t>(d), 1, query_rotated);
    }

    const int K0 = 1 << level_bits[0];
    float* primary_distance_table_flat = workspace.primary_distance_table.data();

    compute_primary_distance_tables_flat(
        query_rotated, K0, primary_distance_table_flat);

    compute_primary_distances(
        primary_distance_table_flat, K0, primary_distances);

    if (num_levels == 1) {
        faiss::maxheap_heapify(k, distances, labels, primary_distances, nullptr, k);
        for (idx_t i = k; i < ntotal; i++) {
            if (primary_distances[i] < distances[0]) {
                faiss::maxheap_replace_top(k, distances, labels, primary_distances[i], i);
            }
        }
        faiss::heap_reorder<faiss::CMax<float, idx_t>>(k, distances, labels);
        if (metric_type == METRIC_INNER_PRODUCT) {
            for (idx_t i = 0; i < k; ++i) {
                distances[i] = -distances[i];
            }
        }
        return;
    }

    const idx_t candidate_keep = std::min<idx_t>(n_candidates, ntotal);
    idx_t* candidate_indices = nullptr;
    size_t candidate_count = static_cast<size_t>(candidate_keep);

    workspace.candidate_indices.resize(candidate_count);
    workspace.candidate_distances.resize(candidate_count);
    candidate_indices = workspace.candidate_indices.data();
    float* top_primary = workspace.candidate_distances.data();

    if (candidate_keep > 0) {
        faiss::maxheap_heapify(
            candidate_keep,
            top_primary,
            candidate_indices,
            primary_distances,
            nullptr,
            candidate_keep);
        for (idx_t i = candidate_keep; i < ntotal; ++i) {
            if (primary_distances[i] < top_primary[0]) {
                faiss::maxheap_replace_top(
                    candidate_keep,
                    top_primary,
                    candidate_indices,
                    primary_distances[i],
                    i);
            }
        }
        faiss::heap_reorder<faiss::CMax<float, idx_t>>(
            candidate_keep,
            top_primary,
            candidate_indices);
    }

    float* final_distances = workspace.candidate_distances.data();
    if (candidate_count > 0 && has_pre_decoded_codes()) {
        FAISS_THROW_IF_NOT_MSG(
            candidate_count <= static_cast<size_t>(std::numeric_limits<int>::max()),
            "candidate_count exceeds JHQDistanceComputer batch limit");
        JHQDistanceComputer* dis = workspace.dc.get();
        FAISS_THROW_IF_NOT_MSG(dis != nullptr, "JHQ workspace distance computer not initialized");
        dis->set_query_rotated(query_rotated);
        dis->distances_batch(
            candidate_indices,
            static_cast<int>(candidate_count),
            final_distances,
            top_primary);
    } else {
        constexpr int PREFETCH_DISTANCE = 8;
        if (separated_codes_.is_initialized && !separated_codes_.empty()) {
            std::vector<float> query_residual(static_cast<size_t>(Ds));
            std::vector<float> db_residual(static_cast<size_t>(Ds));
            const uint8_t* primary_base = separated_codes_.primary_codes.data();
            const size_t primary_stride = separated_codes_.primary_stride;
            const bool has_residual =
                (num_levels > 1) &&
                !separated_codes_.residual_codes.empty() &&
                separated_codes_.residual_stride > 0;
            const uint8_t* residual_base = has_residual
                ? separated_codes_.residual_codes.data()
                : nullptr;
            const size_t residual_stride = separated_codes_.residual_stride;
            for (size_t i = 0; i < candidate_count; ++i) {
                const idx_t db_idx = candidate_indices[i];
                const uint8_t* primary_codes =
                    primary_base + static_cast<size_t>(db_idx) * primary_stride;
                const uint8_t* residual_codes = has_residual
                    ? (residual_base + static_cast<size_t>(db_idx) * residual_stride)
                    : nullptr;

                if (has_residual && i + PREFETCH_DISTANCE < candidate_count) {
                    const idx_t prefetch_idx = candidate_indices[i + PREFETCH_DISTANCE];
                    const uint8_t* residual_prefetch =
                        residual_base + static_cast<size_t>(prefetch_idx) * residual_stride;
                    prefetch_L1(residual_prefetch);
                }

                final_distances[i] = compute_exact_distance_separated_codes_scratch(
                    primary_codes,
                    residual_codes,
                    query_rotated,
                    query_residual.data(),
                    db_residual.data());
            }
        } else if (codes.size() > 0) {
            float* reconstructed_vector = workspace.reconstructed_vector.data();
            for (size_t i = 0; i < candidate_count; ++i) {
                const idx_t db_idx = candidate_indices[i];
                decode_single_code(codes.data() + static_cast<size_t>(db_idx) * code_size, reconstructed_vector);
                if (metric_type == METRIC_INNER_PRODUCT) {
                    final_distances[i] = -fvec_inner_product(query_rotated, reconstructed_vector, d);
                } else {
                    final_distances[i] = fvec_L2sqr(query_rotated, reconstructed_vector, d);
                }
            }
        } else {
            FAISS_THROW_MSG("No codes available for early-termination distance evaluation");
        }
    }

    for (idx_t i = 0; i < k && i < static_cast<idx_t>(candidate_count); i++) {
        distances[i] = final_distances[i];
        labels[i] = candidate_indices[i];
    }
    faiss::heap_heapify<faiss::CMax<float, idx_t>>(k, distances, labels);

    for (size_t i = k; i < candidate_count; i++) {
        if (final_distances[i] < distances[0]) {
            faiss::heap_replace_top<faiss::CMax<float, idx_t>>(
                k, distances, labels, final_distances[i], candidate_indices[i]);
        }
    }

    faiss::heap_reorder<faiss::CMax<float, idx_t>>(k, distances, labels);
    if (metric_type == METRIC_INNER_PRODUCT) {
        for (idx_t i = 0; i < k; ++i) {
            distances[i] = -distances[i];
        }
    }
}

size_t IndexJHQ::search_single_query_exhaustive(
    const float* query,
    idx_t k,
    bool compute_residuals,
    float* distances,
    idx_t* labels) const
{
    SearchWorkspace& workspace = get_search_workspace();
    float* query_rotated = workspace.query_rotated.data();
    float* all_distances = workspace.all_primary_distances.data();

    apply_jl_rotation(1, query, query_rotated);
    if (normalize_l2) {
        fvec_renorm_L2(static_cast<size_t>(d), 1, query_rotated);
    }

    
    if (separated_codes_.is_initialized && !separated_codes_.empty()) {
        JHQDistanceComputer* dis = workspace.dc.get();
        FAISS_THROW_IF_NOT_MSG(dis != nullptr, "JHQ workspace distance computer not initialized");
        dis->set_query_rotated(query_rotated);

        constexpr int BATCH = 4096;
        std::vector<idx_t> batch_ids(static_cast<size_t>(BATCH));
        for (idx_t i = 0; i < ntotal; i += BATCH) {
            const int cur = static_cast<int>(std::min<idx_t>(BATCH, ntotal - i));
            for (int j = 0; j < cur; ++j) {
                batch_ids[static_cast<size_t>(j)] = i + static_cast<idx_t>(j);
            }
            dis->distances_batch(
                batch_ids.data(),
                cur,
                all_distances + i);
        }
    } else if (codes.size() > 0) {
        
        float* reconstructed_vector = workspace.reconstructed_vector.data();
        for (idx_t i = 0; i < ntotal; ++i) {
            decode_single_code(codes.data() + i * code_size, reconstructed_vector);
            if (metric_type == METRIC_INNER_PRODUCT) {
                all_distances[i] = -fvec_inner_product(query_rotated, reconstructed_vector, d);
            } else {
                all_distances[i] = fvec_L2sqr(query_rotated, reconstructed_vector, d);
            }
        }
    } else {
        FAISS_THROW_MSG("No codes available for search - index may not have been populated");
    }

    faiss::maxheap_heapify(k, distances, labels, all_distances, nullptr, k);
    for (idx_t i = k; i < ntotal; ++i) {
        if (all_distances[i] < distances[0]) {
            faiss::maxheap_replace_top(k, distances, labels, all_distances[i], i);
        }
    }
    faiss::heap_reorder<faiss::CMax<float, idx_t>>(k, distances, labels);
    if (metric_type == METRIC_INNER_PRODUCT) {
        for (idx_t i = 0; i < k; ++i) {
            distances[i] = -distances[i];
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
    FAISS_THROW_IF_NOT_MSG(result != nullptr, "range_search result must not be null");
    (void)params;

    result->nq = static_cast<size_t>(n);

    if (metric_type == METRIC_INNER_PRODUCT) {
#pragma omp parallel if (n > 1)
        {
            RangeSearchPartialResult pres(result);
            std::unique_ptr<JHQDistanceComputer> dis(
                static_cast<JHQDistanceComputer*>(get_FlatCodesDistanceComputer()));
#pragma omp for schedule(static)
            for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
                RangeQueryResult& qres = pres.new_result(query_idx);
                const float* query = x + query_idx * d;
                dis->set_query(query);
                const float thr = -radius;
                for (idx_t db_idx = 0; db_idx < ntotal; ++db_idx) {
                    const float neg_ip = (*dis)(db_idx);
                    if (neg_ip <= thr) {
                        qres.add(-neg_ip, db_idx);
                    }
                }
            }
            pres.finalize();
        }
        return;
    }

#pragma omp parallel if (n > 1)
    {
        RangeSearchPartialResult pres(result);
        std::unique_ptr<JHQDistanceComputer> dis(
            static_cast<JHQDistanceComputer*>(get_FlatCodesDistanceComputer()));
        SearchWorkspace& workspace = get_search_workspace();

        float* query_rotated = workspace.query_rotated.data();
        float* primary_distance_table_flat = workspace.primary_distance_table.data();
        float* primary_distances = workspace.all_primary_distances.data();
        const int K0 = 1 << level_bits[0];
        const bool has_predecoded = has_pre_decoded_codes();
        const bool has_cross_terms = has_predecoded &&
            num_levels > 1 &&
            separated_codes_.cross_terms.size() == static_cast<size_t>(ntotal);

        if (has_predecoded && ntotal > 0) {
            workspace.candidate_indices.resize(static_cast<size_t>(ntotal));
            workspace.candidate_primary_distances.resize(static_cast<size_t>(ntotal));
        }

#pragma omp for schedule(static)
        for (idx_t query_idx = 0; query_idx < n; ++query_idx) {
            RangeQueryResult& qres = pres.new_result(query_idx);
            const float* query = x + query_idx * d;
            if (!has_predecoded) {
                dis->set_query(query);
                for (idx_t db_idx = 0; db_idx < ntotal; ++db_idx) {
                    const float total_distance = (*dis)(db_idx);
                    if (total_distance <= radius) {
                        qres.add(total_distance, db_idx);
                    }
                }
                continue;
            }

            apply_jl_rotation(1, query, query_rotated);
            if (normalize_l2) {
                fvec_renorm_L2(static_cast<size_t>(d), 1, query_rotated);
            }

            compute_primary_distance_tables_flat(
                query_rotated,
                K0,
                primary_distance_table_flat);
            compute_primary_distances(
                primary_distance_table_flat,
                K0,
                primary_distances);

            idx_t* candidate_ids = workspace.candidate_indices.data();
            float* candidate_primary = workspace.candidate_primary_distances.data();
            size_t candidate_count = 0;

            if (num_levels == 1) {
                for (idx_t db_idx = 0; db_idx < ntotal; ++db_idx) {
                    const float primary = primary_distances[db_idx];
                    if (primary <= radius) {
                        candidate_ids[candidate_count] = db_idx;
                        candidate_primary[candidate_count] = primary;
                        ++candidate_count;
                    }
                }
            } else if (has_cross_terms) {
                const float* cross_terms = separated_codes_.cross_terms.data();
                for (idx_t db_idx = 0; db_idx < ntotal; ++db_idx) {
                    const float primary = primary_distances[db_idx];
                    
                    
                    if (primary + cross_terms[db_idx] <= radius) {
                        candidate_ids[candidate_count] = db_idx;
                        candidate_primary[candidate_count] = primary;
                        ++candidate_count;
                    }
                }
            } else {
                
                for (idx_t db_idx = 0; db_idx < ntotal; ++db_idx) {
                    candidate_ids[candidate_count] = db_idx;
                    candidate_primary[candidate_count] = primary_distances[db_idx];
                    ++candidate_count;
                }
            }

            if (candidate_count == 0) {
                continue;
            }

            workspace.candidate_distances.resize(candidate_count);
            float* candidate_distances = workspace.candidate_distances.data();
            dis->set_query_rotated_with_lut(query_rotated, primary_distance_table_flat);
            FAISS_THROW_IF_NOT_MSG(
                candidate_count <= static_cast<size_t>(std::numeric_limits<int>::max()),
                "range_search candidate count exceeds JHQDistanceComputer batch limit");
            dis->distances_batch(
                candidate_ids,
                static_cast<int>(candidate_count),
                candidate_distances,
                candidate_primary);

            for (size_t i = 0; i < candidate_count; ++i) {
                if (candidate_distances[i] <= radius) {
                    qres.add(candidate_distances[i], candidate_ids[i]);
                }
            }
        }
        pres.finalize();
    }
}

void IndexJHQ::compute_primary_distance_tables_flat(
    const float* query_rotated,
    int K0,
    float* distance_table_flat) const {
    FAISS_THROW_IF_NOT_MSG(query_rotated != nullptr, "compute_primary_distance_tables_flat: query_rotated is null");
    FAISS_THROW_IF_NOT_MSG(distance_table_flat != nullptr, "compute_primary_distance_tables_flat: output table is null");
    FAISS_THROW_IF_NOT_MSG(K0 > 0, "compute_primary_distance_tables_flat: K0 must be > 0");

    if (metric_type == METRIC_INNER_PRODUCT) {
        const bool can_run_manual =
            (kHasAVX2 || kHasAVX512) &&
            (Ds == 4 || Ds == 8 || Ds == 16 || Ds == 32);

        if (can_run_manual) {
            for (int m = 0; m < M; ++m) {
                const float* query_subspace = query_rotated + static_cast<size_t>(m) * Ds;
                const float* centroids = get_primary_centroids_ptr(m);
                float* table_m_ptr = distance_table_flat + static_cast<size_t>(m) * K0;

                if (Ds == 4) {
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds4_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                } else if (Ds == 8) {
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds8_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                } else if (Ds == 16 && K0 <= 16) {
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds16_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
#if defined(__AVX512F__)
                    if (kHasAVX512) {
                        compute_primary_ip_lut_ds16_avx512(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                } else if (Ds == 16 && K0 >= 64) {
#if defined(__AVX512F__)
                    if (kHasAVX512) {
                        compute_primary_ip_lut_ds16_avx512(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds16_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                } else if (Ds == 16) {
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds16_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
#if defined(__AVX512F__)
                    if (kHasAVX512) {
                        compute_primary_ip_lut_ds16_avx512(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                } else if (Ds == 32 && K0 <= 16) {
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds32_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
#if defined(__AVX512F__)
                    if (kHasAVX512) {
                        compute_primary_ip_lut_ds32_avx512(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                } else if (Ds == 32 && K0 >= 64) {
#if defined(__AVX512F__)
                    if (kHasAVX512) {
                        compute_primary_ip_lut_ds32_avx512(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds32_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                } else if (Ds == 32) {
#if defined(__AVX2__)
                    if (kHasAVX2) {
                        compute_primary_ip_lut_ds32_avx2(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
#if defined(__AVX512F__)
                    if (kHasAVX512) {
                        compute_primary_ip_lut_ds32_avx512(query_subspace, centroids, K0, table_m_ptr);
                        continue;
                    }
#endif
                }

                for (int k = 0; k < K0; ++k) {
                    table_m_ptr[k] = -fvec_inner_product(query_subspace, centroids + static_cast<size_t>(k) * Ds, Ds);
                }
            }
            return;
        }

        for (int m = 0; m < M; ++m) {
            const float* query_subspace = query_rotated + static_cast<size_t>(m) * Ds;
            const float* centroids = get_primary_centroids_ptr(m);
            float* table_m_ptr = distance_table_flat + static_cast<size_t>(m) * K0;
            for (int k = 0; k < K0; ++k) {
                table_m_ptr[k] = -fvec_inner_product(query_subspace, centroids + static_cast<size_t>(k) * Ds, Ds);
            }
        }
        return;
    }

    const bool can_run_manual =
        (kHasAVX2 || kHasAVX512) &&
        (Ds == 4 || Ds == 8 || Ds == 16 || Ds == 32);

    if (can_run_manual) {
        for (int m = 0; m < M; ++m) {
            const float* query_subspace = query_rotated + static_cast<size_t>(m) * Ds;
            const float* centroids = get_primary_centroids_ptr(m);
            float* table_m_ptr = distance_table_flat + static_cast<size_t>(m) * K0;

            
            
            
            
            if (Ds == 4) {
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds4_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            } else if (Ds == 8) {
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds8_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            } else if (Ds == 16 && K0 <= 16) {
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds16_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
#if defined(__AVX512F__)
                if (kHasAVX512) {
                    compute_primary_lut_ds16_avx512(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            } else if (Ds == 16 && K0 >= 64) {
#if defined(__AVX512F__)
                if (kHasAVX512) {
                    compute_primary_lut_ds16_avx512(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds16_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            } else if (Ds == 16) {
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds16_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
#if defined(__AVX512F__)
                if (kHasAVX512) {
                    compute_primary_lut_ds16_avx512(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            } else if (Ds == 32 && K0 <= 16) {
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds32_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
#if defined(__AVX512F__)
                if (kHasAVX512) {
                    compute_primary_lut_ds32_avx512(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            } else if (Ds == 32 && K0 >= 64) {
#if defined(__AVX512F__)
                if (kHasAVX512) {
                    compute_primary_lut_ds32_avx512(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds32_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            } else if (Ds == 32) {
#if defined(__AVX2__)
                if (kHasAVX2) {
                    compute_primary_lut_ds32_avx2(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
#if defined(__AVX512F__)
                if (kHasAVX512) {
                    compute_primary_lut_ds32_avx512(query_subspace, centroids, K0, table_m_ptr);
                    continue;
                }
#endif
            }

            compute_subspace_distances_simd(
                query_subspace,
                centroids,
                table_m_ptr,
                K0,
                Ds);
        }
        return;
    }

    const ProductQuantizer* primary_pq = get_primary_product_quantizer();
    if (primary_pq != nullptr &&
        static_cast<int>(primary_pq->M) == M &&
        static_cast<int>(primary_pq->ksub) == K0 &&
        static_cast<int>(primary_pq->dsub) == Ds) {
        primary_pq->compute_distance_table(query_rotated, distance_table_flat);
        return;
    }

    for (int m = 0; m < M; ++m) {
        const float* query_subspace = query_rotated + static_cast<size_t>(m) * Ds;
        const float* centroids = get_primary_centroids_ptr(m);
        float* table_m_ptr = distance_table_flat + static_cast<size_t>(m) * K0;
        compute_subspace_distances_simd(
            query_subspace,
            centroids,
            table_m_ptr,
            K0,
            Ds);
    }
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

    level_offsets.assign(num_levels, 0);
    size_t total_size = 0;
    for (int level = 1; level < num_levels; ++level) {
        level_offsets[level] = total_size;
        int K = 1 << level_bits[level];
        total_size += static_cast<size_t>(M) * Ds * K;
    }
    flat_tables.resize(total_size);

    const bool has_manual_simd = kHasAVX2 || kHasAVX512;

    const bool metric_ip = (metric_type == METRIC_INNER_PRODUCT);

    if (num_levels == 2 && has_manual_simd) {
        const int K_res = 1 << level_bits[1];
        const bool prefer_avx512 = (K_res >= 64) && kHasAVX512;
        const bool force_avx2 = (K_res <= 16) && kHasAVX2;
        const bool choose_avx512 = prefer_avx512 && !force_avx2;

        for (int m = 0; m < M; ++m) {
            const float* query_subspace = query_rotated + static_cast<size_t>(m) * Ds;
            const float* scalar_codebook = get_scalar_codebook_ptr(m, 1);
            const size_t current_level_offset = level_offsets[1];

            for (int d = 0; d < Ds; ++d) {
                const float query_val = query_subspace[d];
                const size_t table_start_idx = current_level_offset +
                    static_cast<size_t>(m) * static_cast<size_t>(Ds) * static_cast<size_t>(K_res) +
                    static_cast<size_t>(d) * static_cast<size_t>(K_res);
                float* table_ptr = flat_tables.data() + table_start_idx;

                if (metric_ip) {
                    compute_residual_scalar_lut_ip(
                        query_val,
                        scalar_codebook,
                        K_res,
                        table_ptr,
                        choose_avx512,
                        kHasAVX2,
                        kHasAVX512);
                } else {
                    compute_residual_scalar_lut(
                        query_val,
                        scalar_codebook,
                        K_res,
                        table_ptr,
                        choose_avx512,
                        kHasAVX2,
                        kHasAVX512);
                }
            }
        }
        return;
    }

    const ProductQuantizer* residual_pq = get_residual_product_quantizer();
    if (!metric_ip && residual_pq != nullptr) {
        const size_t total_subquantizers = residual_pq->M;
        const int K_res = 1 << level_bits[1];
        flat_tables.resize(total_subquantizers * static_cast<size_t>(K_res));

        std::vector<float> query_for_residual(total_subquantizers);
        size_t offset = 0;
        for (int level = 1; level < num_levels; ++level) {
            for (int m = 0; m < M; ++m) {
                const float* query_subspace = query_rotated + static_cast<size_t>(m) * Ds;
                std::memcpy(
                    query_for_residual.data() + offset,
                    query_subspace,
                    static_cast<size_t>(Ds) * sizeof(float));
                offset += static_cast<size_t>(Ds);
            }
        }

        residual_pq->compute_distance_table(query_for_residual.data(), flat_tables.data());
        return;
    }

    for (int m = 0; m < M; ++m) {
        const float* query_subspace = query_rotated + static_cast<size_t>(m) * Ds;

        for (int level = 1; level < num_levels; ++level) {
            const int K = scalar_codebook_ksub(level);
            const float* scalar_codebook = get_scalar_codebook_ptr(m, level);
            size_t current_level_offset = level_offsets[level];

            const bool prefer_avx512 = (K >= 64) && kHasAVX512;
            const bool force_avx2 = (K <= 16) && kHasAVX2;
            const bool choose_avx512 = prefer_avx512 && !force_avx2;

            for (int d = 0; d < Ds; ++d) {
                const float query_val = query_subspace[d];
                const size_t table_start_idx = current_level_offset +
                    static_cast<size_t>(m) * static_cast<size_t>(Ds) * static_cast<size_t>(K) +
                    static_cast<size_t>(d) * static_cast<size_t>(K);
                float* table_ptr = flat_tables.data() + table_start_idx;

                if (metric_ip) {
                    compute_residual_scalar_lut_ip(
                        query_val,
                        scalar_codebook,
                        K,
                        table_ptr,
                        choose_avx512,
                        kHasAVX2,
                        kHasAVX512);
                } else {
                    compute_residual_scalar_lut(
                        query_val,
                        scalar_codebook,
                        K,
                        table_ptr,
                        choose_avx512,
                        kHasAVX2,
                        kHasAVX512);
                }
            }
        }
    }
}

const ProductQuantizer* IndexJHQ::get_primary_product_quantizer() const
{
    if (num_levels < 1 || level_bits.empty()) {
        return nullptr;
    }

    const int bits = level_bits[0];
    if (bits <= 0 || bits > 8) {
        return nullptr;
    }
    const size_t K0 = static_cast<size_t>(1) << bits;

    if (!primary_pq_) {
        primary_pq_ = std::make_unique<ProductQuantizer>(
            static_cast<size_t>(d),
            static_cast<size_t>(M),
            bits);
    }

    if (primary_pq_->d != static_cast<size_t>(d) ||
        primary_pq_->M != static_cast<size_t>(M) ||
        primary_pq_->dsub != static_cast<size_t>(Ds) ||
        primary_pq_->ksub != K0) {
        return nullptr;
    }

    return primary_pq_.get();
}

const float* IndexJHQ::get_primary_centroids_ptr(int subspace_idx) const
{
    FAISS_THROW_IF_NOT_MSG(subspace_idx >= 0 && subspace_idx < M, "subspace_idx out of bounds");
    const ProductQuantizer* pq = get_primary_product_quantizer();
    FAISS_THROW_IF_NOT_MSG(pq != nullptr, "Primary ProductQuantizer unavailable");
    return pq->get_centroids(static_cast<size_t>(subspace_idx), 0);
}

float* IndexJHQ::get_primary_centroids_ptr_mutable(int subspace_idx)
{
    FAISS_THROW_IF_NOT_MSG(subspace_idx >= 0 && subspace_idx < M, "subspace_idx out of bounds");
    const ProductQuantizer* pq_const = get_primary_product_quantizer();
    FAISS_THROW_IF_NOT_MSG(pq_const != nullptr, "Primary ProductQuantizer unavailable");
    ProductQuantizer* pq = const_cast<ProductQuantizer*>(pq_const);
    return pq->get_centroids(static_cast<size_t>(subspace_idx), 0);
}

int IndexJHQ::primary_ksub() const
{
    FAISS_THROW_IF_NOT_MSG(!level_bits.empty(), "level_bits is empty");
    return 1 << level_bits[0];
}

const ProductQuantizer* IndexJHQ::get_residual_product_quantizer() const
{
    if (num_levels != 2) {
        return nullptr;
    }

    const int bits = level_bits[1];
    const size_t K = static_cast<size_t>(1) << bits;

    if (!residual_pq_ || residual_pq_dirty_) {
        const size_t total_subquantizers = static_cast<size_t>(M) * Ds;
        residual_pq_ = std::make_unique<ProductQuantizer>(
            total_subquantizers, total_subquantizers, bits);

        for (int m = 0; m < M; ++m) {
            const float* codebook = get_scalar_codebook_ptr(m, 1);
            for (int d = 0; d < Ds; ++d) {
                const size_t sub_idx = static_cast<size_t>(m) * Ds + d;
                float* centroid_ptr = residual_pq_->get_centroids(sub_idx, 0);
                std::memcpy(centroid_ptr, codebook, K * sizeof(float));
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
    FAISS_THROW_IF_NOT_MSG(
        separated_codes_.is_initialized && !separated_codes_.primary_codes.empty(),
        "compute_primary_distances: primary codes are not initialized");
    FAISS_THROW_IF_NOT_MSG(
        separated_codes_.primary_stride >= static_cast<size_t>(M),
        "compute_primary_distances: invalid primary stride");

    const uint8_t* primary_codes_base = separated_codes_.primary_codes.data();
    const size_t primary_stride = separated_codes_.primary_stride;

    constexpr int VECTOR_BATCH_SIZE = 512;
    const int prefetch_lookahead = choose_primary_prefetch_lookahead(M, Ds);

    enum class PrimaryAccumMode {
        Scalar,
        AVX2,
        AVX512
    };
    PrimaryAccumMode mode = PrimaryAccumMode::Scalar;
#if defined(__AVX512F__)
    if (kHasAVX512 && M >= 16) {
        mode = PrimaryAccumMode::AVX512;
    } else
#endif
#if defined(__AVX2__)
    if (kHasAVX2 && M >= 8) {
        mode = PrimaryAccumMode::AVX2;
    }
#endif

#if defined(__AVX512F__)
    if (mode == PrimaryAccumMode::AVX512) {
        const __m512i k0_vec = _mm512_set1_epi32(K0);
        const __m512i lane_offsets = _mm512_set_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i stride_offsets = _mm512_mullo_epi32(lane_offsets, k0_vec);
        const __m512i max_code = _mm512_set1_epi32(K0 - 1);

#pragma omp parallel for schedule(static) if (ntotal > 5000)
        for (idx_t batch_start = 0; batch_start < ntotal; batch_start += VECTOR_BATCH_SIZE) {
            const idx_t batch_end = std::min(ntotal, batch_start + VECTOR_BATCH_SIZE);
            for (idx_t i = batch_start; i < batch_end; ++i) {
                const idx_t pf_i = i + static_cast<idx_t>(prefetch_lookahead);
                if (pf_i < batch_end) {
                    prefetch_L1(primary_codes_base + static_cast<size_t>(pf_i) * primary_stride);
                }

                const uint8_t* primary_codes =
                    primary_codes_base + static_cast<size_t>(i) * primary_stride;
                __m512 acc = _mm512_setzero_ps();
                int m = 0;
                for (; m + 15 < M; m += 16) {
                    __m128i c128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(primary_codes + m));
                    __m512i c512 = _mm512_cvtepu8_epi32(c128);
                    c512 = _mm512_min_epi32(c512, max_code);
                    __m512i base = _mm512_set1_epi32(m * K0);
                    __m512i idx = _mm512_add_epi32(_mm512_add_epi32(base, stride_offsets), c512);
                    acc = _mm512_add_ps(acc, _mm512_i32gather_ps(idx, distance_table_flat, 4));
                }

                float total_distance = _mm512_reduce_add_ps(acc);
                for (; m < M; ++m) {
                    const uint32_t code_val = std::min<uint32_t>(
                        primary_codes[m], static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[
                        static_cast<size_t>(m) * static_cast<size_t>(K0) + code_val];
                }
                distances[i] = total_distance;
            }
        }
        return;
    }
#endif

#if defined(__AVX2__)
    if (mode == PrimaryAccumMode::AVX2) {
        const __m256i k0_vec = _mm256_set1_epi32(K0);
        const __m256i lane_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
        const __m256i stride_offsets = _mm256_mullo_epi32(lane_offsets, k0_vec);
        const __m256i max_code = _mm256_set1_epi32(K0 - 1);

#pragma omp parallel for schedule(static) if (ntotal > 5000)
        for (idx_t batch_start = 0; batch_start < ntotal; batch_start += VECTOR_BATCH_SIZE) {
            const idx_t batch_end = std::min(ntotal, batch_start + VECTOR_BATCH_SIZE);
            for (idx_t i = batch_start; i < batch_end; ++i) {
                const idx_t pf_i = i + static_cast<idx_t>(prefetch_lookahead);
                if (pf_i < batch_end) {
                    prefetch_L1(primary_codes_base + static_cast<size_t>(pf_i) * primary_stride);
                }

                const uint8_t* primary_codes =
                    primary_codes_base + static_cast<size_t>(i) * primary_stride;
                __m256 acc = _mm256_setzero_ps();
                int m = 0;
                for (; m + 7 < M; m += 8) {
                    __m128i c64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(primary_codes + m));
                    __m256i c256 = _mm256_cvtepu8_epi32(c64);
                    c256 = _mm256_min_epi32(c256, max_code);
                    __m256i base = _mm256_set1_epi32(m * K0);
                    __m256i idx = _mm256_add_epi32(_mm256_add_epi32(base, stride_offsets), c256);
                    acc = _mm256_add_ps(acc, _mm256_i32gather_ps(distance_table_flat, idx, 4));
                }

                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 sum = _mm_add_ps(lo, hi);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                float total_distance = _mm_cvtss_f32(sum);
                for (; m < M; ++m) {
                    const uint32_t code_val = std::min<uint32_t>(
                        primary_codes[m], static_cast<uint32_t>(K0 - 1));
                    total_distance += distance_table_flat[
                        static_cast<size_t>(m) * static_cast<size_t>(K0) + code_val];
                }
                distances[i] = total_distance;
            }
        }
        return;
    }
#endif

#pragma omp parallel for schedule(static) if (ntotal > 5000)
    for (idx_t batch_start = 0; batch_start < ntotal; batch_start += VECTOR_BATCH_SIZE) {
        const idx_t batch_end = std::min(ntotal, batch_start + VECTOR_BATCH_SIZE);
        for (idx_t i = batch_start; i < batch_end; ++i) {
            const idx_t pf_i = i + static_cast<idx_t>(prefetch_lookahead);
            if (pf_i < batch_end) {
                prefetch_L1(primary_codes_base + static_cast<size_t>(pf_i) * primary_stride);
            }

            const uint8_t* primary_codes =
                primary_codes_base + static_cast<size_t>(i) * primary_stride;
            float total_distance = 0.0f;

            int m = 0;
            for (; m + 7 < M; m += 8) {
                total_distance +=
                    distance_table_flat[(m + 0) * K0 + std::min<uint32_t>(primary_codes[m + 0], static_cast<uint32_t>(K0 - 1))] +
                    distance_table_flat[(m + 1) * K0 + std::min<uint32_t>(primary_codes[m + 1], static_cast<uint32_t>(K0 - 1))] +
                    distance_table_flat[(m + 2) * K0 + std::min<uint32_t>(primary_codes[m + 2], static_cast<uint32_t>(K0 - 1))] +
                    distance_table_flat[(m + 3) * K0 + std::min<uint32_t>(primary_codes[m + 3], static_cast<uint32_t>(K0 - 1))] +
                    distance_table_flat[(m + 4) * K0 + std::min<uint32_t>(primary_codes[m + 4], static_cast<uint32_t>(K0 - 1))] +
                    distance_table_flat[(m + 5) * K0 + std::min<uint32_t>(primary_codes[m + 5], static_cast<uint32_t>(K0 - 1))] +
                    distance_table_flat[(m + 6) * K0 + std::min<uint32_t>(primary_codes[m + 6], static_cast<uint32_t>(K0 - 1))] +
                    distance_table_flat[(m + 7) * K0 + std::min<uint32_t>(primary_codes[m + 7], static_cast<uint32_t>(K0 - 1))];
            }
            for (; m < M; ++m) {
                const uint32_t code_val = std::min<uint32_t>(
                    primary_codes[m], static_cast<uint32_t>(K0 - 1));
                total_distance += distance_table_flat[
                    static_cast<size_t>(m) * static_cast<size_t>(K0) + code_val];
            }
            distances[i] = total_distance;
        }
    }
}

FlatCodesDistanceComputer* IndexJHQ::get_FlatCodesDistanceComputer() const
{
    return new JHQDistanceComputer(*this);
}

void IndexJHQ::rebuild_residual_codes_packed4() const
{
    auto& separated = const_cast<jhq_internal::PreDecodedCodes&>(separated_codes_);
    separated.residual_codes_packed4.clear();
    separated.residual_packed4_stride = 0;

    if (num_levels != 2 || static_cast<int>(level_bits.size()) < 2 || level_bits[1] != 4) {
        return;
    }
    if (separated.residual_stride == 0 || separated.residual_codes.empty()) {
        return;
    }
    if (separated.primary_stride == 0 || separated.primary_codes.empty()) {
        return;
    }

    const idx_t nvec = static_cast<idx_t>(separated.primary_codes.size() / separated.primary_stride);
    if (nvec <= 0) {
        return;
    }

    const size_t src_stride = separated.residual_stride;
    const size_t dst_stride = (src_stride + 1) / 2;
    separated.residual_codes_packed4.resize(static_cast<size_t>(nvec) * dst_stride);
    separated.residual_packed4_stride = dst_stride;

#pragma omp parallel for schedule(static, 1024) if (nvec > 10000)
    for (idx_t i = 0; i < nvec; ++i) {
        const uint8_t* src = separated.get_residual_codes(i);
        uint8_t* dst = separated.residual_codes_packed4.data() + static_cast<size_t>(i) * dst_stride;
        size_t s = 0;
        size_t d = 0;
        for (; s + 1 < src_stride; s += 2, ++d) {
            const uint8_t lo = static_cast<uint8_t>(src[s] & 0x0F);
            const uint8_t hi = static_cast<uint8_t>(src[s + 1] & 0x0F);
            dst[d] = static_cast<uint8_t>(lo | static_cast<uint8_t>(hi << 4));
        }
        if (s < src_stride) {
            dst[d] = static_cast<uint8_t>(src[s] & 0x0F);
        }
    }
}

void IndexJHQ::extract_all_codes_after_add(
    bool compute_cross_terms,
    bool compute_residual_norms)
{
    if (ntotal == 0) {
        separated_codes_.clear();
        return;
    }

    if (metric_type != METRIC_L2) {
        compute_cross_terms = false;
        compute_residual_norms = false;
    }

    separated_codes_.initialize(M, Ds, num_levels, ntotal);
    if (compute_residual_norms && num_levels == 2) {
        separated_codes_.residual_norms.resize(static_cast<size_t>(ntotal), 0.0f);
    }

    residual_bits_per_subspace = 0;
    for (int level = 1; level < num_levels; ++level) {
        residual_bits_per_subspace += static_cast<size_t>(Ds) * level_bits[level];
    }

#pragma omp parallel for schedule(static, 1024) if (ntotal > 10000)
    for (idx_t i = 0; i < ntotal; ++i) {
        extract_single_vector_all_codes(i);
    }

    if (num_levels > 1 && (compute_cross_terms || (compute_residual_norms && num_levels == 2))) {
#pragma omp parallel for schedule(static, 1024) if (ntotal > 10000)
        for (idx_t i = 0; i < ntotal; ++i) {
            if (compute_cross_terms) {
                separated_codes_.cross_terms[i] = jhq_internal::compute_cross_term_from_codes(
                    *this,
                    separated_codes_.get_primary_codes(i),
                    separated_codes_.get_residual_codes(i),
                    separated_codes_.residual_subspace_stride,
                    separated_codes_.residual_level_stride);
            }
            if (compute_residual_norms && num_levels == 2) {
                separated_codes_.residual_norms[i] =
                    jhq_internal::compute_residual_norm_sq_from_codes(
                        *this,
                        separated_codes_.get_residual_codes(i),
                        separated_codes_.residual_subspace_stride,
                        separated_codes_.residual_level_stride);
            }
        }
    }

    rebuild_residual_codes_packed4();
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
        total_bytes += rotation_matrix_bf16.size() * sizeof(uint16_t);
    }

    if (primary_pq_) {
        total_bytes += primary_pq_->centroids.size() * sizeof(float);
    }

    if (scalar_codebooks_flat_valid_ && !scalar_codebooks_flat_.empty()) {
        total_bytes += scalar_codebooks_flat_.size() * sizeof(float);
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
    rotation_matrix_bf16.clear();
    use_bf16_rotation = false;

    primary_pq_.reset();
    primary_pq_dirty_ = false;
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
    if (!use_analytical_init || k <= 1 || n <= 1 || dim <= 0) {
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
    if (!std::isfinite(std_scale) || std_scale <= 0.0f) {
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.01f);
        for (int i = 0; i < k * dim; ++i) {
            centroids[i] = dist(rng);
        }
        return;
    }

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
        const size_t n_size = static_cast<size_t>(n);
        const size_t sample_size = std::min(n_size, static_cast<size_t>(2000));
        const size_t stride = std::max<size_t>(1, n_size / sample_size);

        std::vector<float> norms;
        norms.reserve(sample_size);

        for (size_t i = 0; i < sample_size; i += stride) {
            if (i >= n_size)
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
    constexpr float kOrthoEps = 1e-8f;

    for (int i = 0; i < k; ++i) {
        float* dir_i = directions_flat.data() + i * dim;

        for (int d = 0; d < dim; ++d) {
            dir_i[d] = dir_dist(rng);
        }

        
        
        const int basis_count = std::min(i, dim);
        for (int pass = 0; pass < 2 && basis_count > 0; ++pass) {
            for (int j = 0; j < basis_count; ++j) {
                const float* dir_j = directions_flat.data() + j * dim;
                const float dot_product =
                    fvec_inner_product(dir_i, dir_j, static_cast<size_t>(dim));
                for (int d = 0; d < dim; ++d) {
                    dir_i[d] -= dot_product * dir_j[d];
                }
            }
        }

        float norm_sq = fvec_norm_L2sqr(dir_i, static_cast<size_t>(dim));
        if (!std::isfinite(norm_sq) || norm_sq < kOrthoEps) {
            
            for (int d = 0; d < dim; ++d) {
                dir_i[d] = dir_dist(rng);
            }
            norm_sq = fvec_norm_L2sqr(dir_i, static_cast<size_t>(dim));
        }

        const float norm = std::sqrt(std::max(norm_sq, kOrthoEps));
        const float inv_norm = 1.0f / norm;
        for (int d = 0; d < dim; ++d) {
            dir_i[d] *= inv_norm;
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
}

}  
