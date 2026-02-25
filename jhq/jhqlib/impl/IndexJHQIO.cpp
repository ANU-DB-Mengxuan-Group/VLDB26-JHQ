#include "IndexJHQ.h"

#include <faiss/impl/mapped_io.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <faiss/utils/distances.h>
#include <faiss/utils/hamming.h>
#include <iostream>
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

bool force_jhq_exact_distance()
{
    
    
    
    
    static const bool enabled = []() {
        const char* v = std::getenv("JHQ_FORCE_EXACT_DISTANCE");
        if (!v || v[0] == '\0') {
            return false;
        }
        if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F') {
            return false;
        }
        return true;
    }();
    return enabled;
}

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

inline uint32_t hsum256_epi32(__m256i v)
{
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i sum = _mm_add_epi32(lo, hi);
    __m128i tmp = _mm_shuffle_epi32(sum, _MM_SHUFFLE(1, 0, 3, 2));
    sum = _mm_add_epi32(sum, tmp);
    tmp = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1));
    sum = _mm_add_epi32(sum, tmp);
    return static_cast<uint32_t>(_mm_cvtsi128_si32(sum));
}
#endif

	bool parse_env_bool(const char* name, bool default_value)
	{
	    const char* v = std::getenv(name);
    if (!v) {
        return default_value;
    }
    if (v[0] == '\0') {
        return default_value;
    }
    if (v[0] == '0' || v[0] == 'n' || v[0] == 'N' || v[0] == 'f' || v[0] == 'F') {
        return false;
    }
	    return true;
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

void IndexJHQ::set_clustering_parameters(bool use_kmeans, int niter, int nredo, int seed)
{
    use_kmeans_refinement = use_kmeans;
    kmeans_niter = std::max(1, niter);
    kmeans_nredo = std::max(1, nredo);
    kmeans_seed = seed;
}

IndexJHQ::SearchWorkspace& IndexJHQ::get_search_workspace() const
{
    if (workspace_.owner != this) {
        workspace_ = SearchWorkspace{};
        workspace_.owner = this;
    }

    const int K0 = 1 << level_bits[0];

    workspace_.query_rotated.resize(M * Ds);
    workspace_.primary_distance_table.resize(M * K0);
    workspace_.all_primary_distances.resize(ntotal);
    workspace_.reconstructed_vector.resize(d);

    if (!workspace_.dc) {
        workspace_.dc = std::make_unique<JHQDistanceComputer>(*this);
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
            const float* centroids = get_primary_centroids_ptr(m);
            float* table_m = distance_table + m * K0;

            if (metric_type == METRIC_INNER_PRODUCT) {
                for (int k = 0; k < K0; ++k) {
                    table_m[k] = -fvec_inner_product(query_sub, centroids + static_cast<size_t>(k) * Ds, Ds);
                }
            } else {
                compute_subspace_distances_simd(query_sub, centroids, table_m, K0, Ds);
            }
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
    if (Ds >= 16) {
        for (; k < K; ++k) {
            const float* centroid_k = codewords + k * Ds;
            __m512 acc = _mm512_setzero_ps();

            int d = 0;
            for (; d + 16 <= Ds; d += 16) {
                __m512 q_vals = _mm512_loadu_ps(query_sub + d);
                __m512 c_vals = _mm512_loadu_ps(centroid_k + d);
                __m512 diff = _mm512_sub_ps(q_vals, c_vals);
                acc = _mm512_fmadd_ps(diff, diff, acc);
            }

            
            float dist = _mm512_reduce_add_ps(acc);

            
            for (; d < Ds; ++d) {
                float diff = query_sub[d] - centroid_k[d];
                dist += diff * diff;
            }

            distances[k] = dist;
        }
    } else if (Ds == 4) {
        const __m128 q_vals = _mm_loadu_ps(query_sub);
        for (; k < K; ++k) {
            const float* centroid_k = codewords + static_cast<size_t>(k) * 4;
            const __m128 c_vals = _mm_loadu_ps(centroid_k);
            const __m128 diff = _mm_sub_ps(q_vals, c_vals);
            const __m128 sq = _mm_mul_ps(diff, diff);
            __m128 sum = _mm_hadd_ps(sq, sq);
            sum = _mm_hadd_ps(sum, sum);
            distances[k] = _mm_cvtss_f32(sum);
        }
    } else {
        
        for (; k < K; ++k) {
            distances[k] = fvec_L2sqr(query_sub, codewords + k * Ds, Ds);
        }
    }

#elif defined(__AVX2__)
    
    int k = 0;
    if (Ds >= 8) {
        for (; k < K; ++k) {
            const float* centroid_k = codewords + k * Ds;
            __m256 acc = _mm256_setzero_ps();

            int d = 0;
            for (; d + 8 <= Ds; d += 8) {
                __m256 q_vals = _mm256_loadu_ps(query_sub + d);
                __m256 c_vals = _mm256_loadu_ps(centroid_k + d);
                __m256 diff = _mm256_sub_ps(q_vals, c_vals);
                acc = _mm256_fmadd_ps(diff, diff, acc);
            }

            
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum128 = _mm_add_ps(lo, hi);
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            float dist = _mm_cvtss_f32(sum128);

            
            for (; d < Ds; ++d) {
                float diff = query_sub[d] - centroid_k[d];
                dist += diff * diff;
            }

            distances[k] = dist;
        }
    } else if (Ds == 4) {
        const __m128 q_vals = _mm_loadu_ps(query_sub);
        for (; k < K; ++k) {
            const float* centroid_k = codewords + static_cast<size_t>(k) * 4;
            const __m128 c_vals = _mm_loadu_ps(centroid_k);
            const __m128 diff = _mm_sub_ps(q_vals, c_vals);
            const __m128 sq = _mm_mul_ps(diff, diff);
            __m128 sum = _mm_hadd_ps(sq, sq);
            sum = _mm_hadd_ps(sum, sum);
            distances[k] = _mm_cvtss_f32(sum);
        }
    } else {
        
        for (; k < K; ++k) {
            distances[k] = fvec_L2sqr(query_sub, codewords + k * Ds, Ds);
        }
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
	    , tables_computed(false)
	    , has_residual_levels(idx.num_levels > 1)
    , use_quantized_tables(false)
    , quantization_scale(1.0f)
    , quantization_offset(0.0f)
    , residual_quant_scale(1.0f)
    , residual_quant_offset(0.0f)
    , M(idx.M)
    , Ds(idx.Ds)
	    , num_levels(idx.num_levels)
	    , K0(1 << idx.level_bits[0])
	    , primary_table_size(static_cast<size_t>(M) * K0)
	{
	    query_rotated.resize(idx.d);
	    primary_distance_table_flat.resize(primary_table_size);
	    temp_workspace.resize(std::max(static_cast<size_t>(M * K0), static_cast<size_t>(1024)));

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
	        if (use_br8_direct_dot_()) {
	            const size_t expected = static_cast<size_t>(index.ntotal);
	            FAISS_THROW_IF_NOT_MSG(
	                index.separated_codes_.residual_norms.size() == expected,
	                "BR8 DirectDot requires residual_norms (rebuild index or load v4 index)");
	            query_norm_sq_ = fvec_norm_L2sqr(query_rotated.data(), index.d);
	            residual_distance_tables_flat.clear();
	            residual_table_offsets.clear();
	            quantized_residual_tables.clear();
	            residual_quant_scale = 1.0f;
	            residual_quant_offset = 0.0f;
	        } else {
	            query_norm_sq_ = 0.0f;
	            compute_residual_distance_tables();
	        }
	    }

    tables_computed = true;
}

void JHQDistanceComputer::set_query_rotated(const float* x_rotated)
{
    FAISS_THROW_IF_NOT_MSG(x_rotated != nullptr, "set_query_rotated: x_rotated is null");
    std::memcpy(query_rotated.data(), x_rotated, sizeof(float) * index.d);

    if (use_quantized_tables) {
        compute_and_quantize_tables();
    } else {
        compute_primary_distance_table();
    }

	    if (has_residual_levels) {
	        if (use_br8_direct_dot_()) {
	            const size_t expected = static_cast<size_t>(index.ntotal);
	            FAISS_THROW_IF_NOT_MSG(
	                index.separated_codes_.residual_norms.size() == expected,
	                "BR8 DirectDot requires residual_norms (rebuild index or load v4 index)");
	            query_norm_sq_ = fvec_norm_L2sqr(query_rotated.data(), index.d);
	            residual_distance_tables_flat.clear();
	            residual_table_offsets.clear();
	            quantized_residual_tables.clear();
	            residual_quant_scale = 1.0f;
	            residual_quant_offset = 0.0f;
	        } else {
	            query_norm_sq_ = 0.0f;
	            compute_residual_distance_tables();
	        }
	    }

    tables_computed = true;
}

void JHQDistanceComputer::set_query_rotated_with_lut(const float* x_rotated, const float* primary_lut)
{
    FAISS_THROW_IF_NOT_MSG(x_rotated != nullptr, "set_query_rotated_with_lut: x_rotated is null");
    FAISS_THROW_IF_NOT_MSG(primary_lut != nullptr, "set_query_rotated_with_lut: primary_lut is null");

    std::memcpy(query_rotated.data(), x_rotated, sizeof(float) * index.d);

    
    
    std::memcpy(primary_distance_table_flat.data(), primary_lut,
                primary_table_size * sizeof(float));

    if (use_quantized_tables) {
        auto [min_it, max_it] = std::minmax_element(
            primary_distance_table_flat.begin(), primary_distance_table_flat.end());
        quantization_offset = *min_it;
        const float range = *max_it - *min_it;
        quantization_scale = (range > 0.0f) ? (range / 65535.0f) : 1.0f;
        const float inv_scale = (range > 0.0f) ? (65535.0f / range) : 0.0f;
        for (size_t i = 0; i < primary_table_size; ++i) {
            float normalized = (primary_distance_table_flat[i] - quantization_offset) * inv_scale;
            quantized_primary_tables[i] = static_cast<uint16_t>(std::clamp(normalized, 0.0f, 65535.0f));
        }
    }

	    if (has_residual_levels) {
	        if (use_br8_direct_dot_()) {
	            const size_t expected = static_cast<size_t>(index.ntotal);
	            FAISS_THROW_IF_NOT_MSG(
	                index.separated_codes_.residual_norms.size() == expected,
	                "BR8 DirectDot requires residual_norms (rebuild index or load v4 index)");
	            query_norm_sq_ = fvec_norm_L2sqr(query_rotated.data(), index.d);
	            residual_distance_tables_flat.clear();
	            residual_table_offsets.clear();
	            quantized_residual_tables.clear();
	            residual_quant_scale = 1.0f;
	            residual_quant_offset = 0.0f;
	        } else {
	            query_norm_sq_ = 0.0f;
	            compute_residual_distance_tables();
	        }
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
    if (index.has_pre_decoded_codes()) {
        return distance_to_index(i);
    }
    const uint8_t* code = codes + i * code_size;
    return distance_to_code(code);
}

float JHQDistanceComputer::symmetric_dis(idx_t i, idx_t j)
{
    std::vector<float> xi(index.d), xj(index.d);
    index.reconstruct(i, xi.data());
    index.reconstruct(j, xj.data());
    if (index.metric_type == METRIC_INNER_PRODUCT) {
        return -fvec_inner_product(xi.data(), xj.data(), index.d);
    }
    return fvec_L2sqr(xi.data(), xj.data(), index.d);
}

void JHQDistanceComputer::distances_batch_4(
    const idx_t idx0, const idx_t idx1, const idx_t idx2, const idx_t idx3,
    float& dis0, float& dis1, float& dis2, float& dis3)
{
    if (!index.has_pre_decoded_codes()) {
        prefetch_L1(codes + static_cast<size_t>(idx0) * code_size);
        prefetch_L1(codes + static_cast<size_t>(idx1) * code_size);
        prefetch_L1(codes + static_cast<size_t>(idx2) * code_size);
        prefetch_L1(codes + static_cast<size_t>(idx3) * code_size);
        dis0 = (*this)(idx0);
        dis1 = (*this)(idx1);
        dis2 = (*this)(idx2);
        dis3 = (*this)(idx3);
        return;
    }

    const uint8_t* primary_base = index.separated_codes_.primary_codes.data();
    const size_t primary_stride = index.separated_codes_.primary_stride;
    prefetch_L1(primary_base + static_cast<size_t>(idx0) * primary_stride);
    prefetch_L1(primary_base + static_cast<size_t>(idx1) * primary_stride);
    prefetch_L1(primary_base + static_cast<size_t>(idx2) * primary_stride);
    prefetch_L1(primary_base + static_cast<size_t>(idx3) * primary_stride);

    dis0 = distance_to_index(idx0);
    dis1 = distance_to_index(idx1);
    dis2 = distance_to_index(idx2);
    dis3 = distance_to_index(idx3);
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
    const float range = *max_it - *min_it;
    quantization_scale = (range > 0.0f) ? (range / 65535.0f) : 1.0f;
    const float inv_scale = (range > 0.0f) ? (65535.0f / range) : 0.0f;

    for (size_t i = 0; i < primary_table_size; ++i) {
        float normalized = (primary_distance_table_flat[i] - quantization_offset) * inv_scale;
        quantized_primary_tables[i] = static_cast<uint16_t>(std::clamp(normalized, 0.0f, 65535.0f));
    }
}

float JHQDistanceComputer::precomputed_distance_to_code(const uint8_t* code) const
{
    const size_t vector_idx = (code - codes) / code_size;
    return distance_to_index(static_cast<idx_t>(vector_idx));
}

float JHQDistanceComputer::distance_to_index(idx_t vector_idx) const
{
    return distance_to_index_with_primary(vector_idx, 0.0f, false);
}

float JHQDistanceComputer::distance_to_index_with_primary(
    idx_t vector_idx,
    float primary_distance,
    bool has_primary_distance) const
{
    FAISS_THROW_IF_NOT(tables_computed);
    FAISS_THROW_IF_NOT(index.has_pre_decoded_codes());

    if (force_jhq_exact_distance()) {
        
        
        return index.compute_exact_distance_separated(vector_idx, query_rotated.data());
    }

    const bool metric_ip = (index.metric_type == METRIC_INNER_PRODUCT);

    const size_t vec = static_cast<size_t>(vector_idx);
    const uint8_t* primary_codes_ptr =
        index.separated_codes_.primary_codes.data() +
        vec * index.separated_codes_.primary_stride;
    const uint8_t* residual_codes_ptr =
        (has_residual_levels && index.separated_codes_.residual_stride > 0)
            ? (index.separated_codes_.residual_codes.data() +
               vec * index.separated_codes_.residual_stride)
            : nullptr;

    float total_distance = primary_distance;

    if (!has_primary_distance) {
        if (!use_quantized_tables) {
#if defined(__AVX512F__)
            if (kHasAVX512 && M >= 16) {
                const __m512i k_vec = _mm512_set1_epi32(K0);
                const __m512i lane_offsets = _mm512_set_epi32(
                    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                const __m512i stride_offsets = _mm512_mullo_epi32(lane_offsets, k_vec);
                const __m512i max_code = _mm512_set1_epi32(K0 - 1);
                __m512 acc = _mm512_setzero_ps();
                int m = 0;

                for (; m + 15 < M; m += 16) {
                    __m128i c128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(primary_codes_ptr + m));
                    __m512i c512 = _mm512_cvtepu8_epi32(c128);
                    c512 = _mm512_min_epi32(c512, max_code);
                    __m512i base = _mm512_set1_epi32(m * K0);
                    __m512i idx = _mm512_add_epi32(_mm512_add_epi32(base, stride_offsets), c512);
                    acc = _mm512_add_ps(acc, _mm512_i32gather_ps(idx, primary_distance_table_flat.data(), 4));

                    if (m + 31 < M) {
                        prefetch_L1(primary_codes_ptr + m + 16);
                    }
                }

                total_distance += _mm512_reduce_add_ps(acc);
                for (; m < M; ++m) {
                    const uint32_t centroid_id = std::min<uint32_t>(
                        primary_codes_ptr[m], static_cast<uint32_t>(K0 - 1));
                    total_distance += primary_distance_table_flat[
                        static_cast<size_t>(m) * static_cast<size_t>(K0) + centroid_id];
                }
            } else
#endif
#if defined(__AVX2__)
            if (kHasAVX2 && M >= 8) {
                const __m256i k_vec = _mm256_set1_epi32(K0);
                const __m256i lane_offsets = _mm256_set_epi32(7, 6, 5, 4, 3, 2, 1, 0);
                const __m256i stride_offsets = _mm256_mullo_epi32(lane_offsets, k_vec);
                const __m256i max_code = _mm256_set1_epi32(K0 - 1);
                __m256 acc = _mm256_setzero_ps();
                int m = 0;
                for (; m + 7 < M; m += 8) {
                    __m128i c64 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(primary_codes_ptr + m));
                    __m256i c256 = _mm256_cvtepu8_epi32(c64);
                    c256 = _mm256_min_epi32(c256, max_code);
                    __m256i base = _mm256_set1_epi32(m * K0);
                    __m256i idx = _mm256_add_epi32(
                        _mm256_add_epi32(base, stride_offsets),
                        c256);
                    acc = _mm256_add_ps(acc, _mm256_i32gather_ps(primary_distance_table_flat.data(), idx, 4));

                    if (m + 15 < M) {
                        prefetch_L1(primary_codes_ptr + m + 8);
                    }
                }

                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 lo = _mm256_castps256_ps128(acc);
                __m128 sum = _mm_add_ps(lo, hi);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                total_distance += _mm_cvtss_f32(sum);

                for (; m < M; ++m) {
                    const uint32_t centroid_id = std::min<uint32_t>(
                        primary_codes_ptr[m], static_cast<uint32_t>(K0 - 1));
                    total_distance += primary_distance_table_flat[
                        static_cast<size_t>(m) * static_cast<size_t>(K0) + centroid_id];
                }
            } else
#endif
            {
                int m = 0;
                for (; m + 3 < M; m += 4) {
                    if (m + 7 < M) {
                        prefetch_L1(reinterpret_cast<const char*>(&primary_codes_ptr[m + 4]));
                    }

                    for (int i = 0; i < 4; ++i) {
                        const uint32_t centroid_id = std::min<uint32_t>(
                            primary_codes_ptr[m + i], static_cast<uint32_t>(K0 - 1));
                        total_distance += primary_distance_table_flat[
                            static_cast<size_t>(m + i) * static_cast<size_t>(K0) + centroid_id];
                    }
                }

                for (; m < M; ++m) {
                    const uint32_t centroid_id = std::min<uint32_t>(
                        primary_codes_ptr[m], static_cast<uint32_t>(K0 - 1));
                    total_distance += primary_distance_table_flat[
                        static_cast<size_t>(m) * static_cast<size_t>(K0) + centroid_id];
                }
            }
        } else {
            int m = 0;
            for (; m + 3 < M; m += 4) {
                if (m + 7 < M) {
                    prefetch_L1(reinterpret_cast<const char*>(&primary_codes_ptr[m + 4]));
                }

                for (int i = 0; i < 4; ++i) {
                    const uint32_t centroid_id = std::min<uint32_t>(
                        primary_codes_ptr[m + i], static_cast<uint32_t>(K0 - 1));
                    const uint16_t quantized_dist = quantized_primary_tables[
                        static_cast<size_t>(m + i) * static_cast<size_t>(K0) + centroid_id];
                    total_distance += quantization_offset + quantized_dist * quantization_scale;
                }
            }

            for (; m < M; ++m) {
                const uint32_t centroid_id = std::min<uint32_t>(
                    primary_codes_ptr[m], static_cast<uint32_t>(K0 - 1));
                const uint16_t quantized_dist =
                    quantized_primary_tables[static_cast<size_t>(m) * static_cast<size_t>(K0) + centroid_id];
                total_distance += quantization_offset + quantized_dist * quantization_scale;
            }
        }
    }

    if (has_residual_levels) {
        total_distance += compute_precomputed_residual_distance(vector_idx);
        if (!metric_ip && residual_codes_ptr) {
            if (!index.separated_codes_.cross_terms.empty()) {
                total_distance += index.separated_codes_.cross_terms[vector_idx];
            } else {
                total_distance += jhq_internal::compute_cross_term_from_codes(
                    index,
                    primary_codes_ptr,
                    residual_codes_ptr,
                    index.separated_codes_.residual_subspace_stride,
                    index.separated_codes_.residual_level_stride);
            }
        }
    }

    return total_distance;
}

void JHQDistanceComputer::distances_batch(
    const idx_t* ids,
    int n,
    float* distances,
    const float* precomputed_primary)
{
    if (n <= 0) {
        return;
    }

    FAISS_THROW_IF_NOT(tables_computed);
    const int prefetch_lookahead = std::max(0, prefetch_lookahead_);

    if (!index.has_pre_decoded_codes()) {
        for (int i = 0; i < n; ++i) {
            const int pf_i = i + prefetch_lookahead;
            if (prefetch_lookahead > 0 && pf_i < n) {
                prefetch_L2(codes + static_cast<size_t>(ids[pf_i]) * code_size);
            }
            distances[i] = distance_to_code(codes + ids[i] * code_size);
        }
        return;
    }

    const uint8_t* primary_codes_base = index.separated_codes_.primary_codes.data();
    const size_t primary_stride = index.separated_codes_.primary_stride;
    const bool has_residual_codes = has_residual_levels && index.separated_codes_.residual_stride > 0;
    const bool has_residual_codes_packed4 = has_residual_codes
        && index.separated_codes_.has_residual_codes_packed4();
    const uint8_t* residual_codes_base = has_residual_codes
        ? index.separated_codes_.residual_codes.data()
        : nullptr;
    const size_t residual_stride = has_residual_codes ? index.separated_codes_.residual_stride : 0;
    const uint8_t* residual_codes_packed4_base = has_residual_codes_packed4
        ? index.separated_codes_.residual_codes_packed4.data()
        : nullptr;
    const size_t residual_packed4_stride = has_residual_codes_packed4
        ? index.separated_codes_.residual_packed4_stride
        : 0;
    const bool metric_ip = (index.metric_type == METRIC_INNER_PRODUCT);
    const bool has_cross_terms = !metric_ip && has_residual_codes
        && !index.separated_codes_.cross_terms.empty()
        && index.separated_codes_.cross_terms.size() == static_cast<size_t>(index.ntotal);
    const float* cross_terms_base = has_cross_terms
        ? index.separated_codes_.cross_terms.data()
        : nullptr;
    const bool has_primary = (precomputed_primary != nullptr);
    const bool need_primary_codes_for_cross = !metric_ip && has_residual_codes && !has_cross_terms;

    if (has_primary) {
        for (int i = 0; i < n; ++i) {
            const int pf_i = i + prefetch_lookahead;
            if (prefetch_lookahead > 0 && pf_i < n) {
                const size_t pf_vec = static_cast<size_t>(ids[pf_i]);
                if (need_primary_codes_for_cross) {
                    prefetch_L2(primary_codes_base + pf_vec * primary_stride);
                }
                if (has_residual_codes) {
                    if (has_residual_codes_packed4) {
                        prefetch_L2(residual_codes_packed4_base + pf_vec * residual_packed4_stride);
                        if (!has_cross_terms) {
                            prefetch_L2(residual_codes_base + pf_vec * residual_stride);
                        }
                    } else {
                        prefetch_L2(residual_codes_base + pf_vec * residual_stride);
                    }
                }
                if (has_cross_terms) {
                    prefetch_L1(cross_terms_base + pf_vec);
                }
            }

            const size_t vec = static_cast<size_t>(ids[i]);
            float total_distance = precomputed_primary[i];
            if (has_residual_levels) {
                total_distance += compute_precomputed_residual_distance(vec);
                if (!metric_ip && has_residual_codes) {
                    if (has_cross_terms) {
                        total_distance += cross_terms_base[vec];
                    } else {
                        const uint8_t* primary_codes_ptr = primary_codes_base + vec * primary_stride;
                        const uint8_t* residual_codes_ptr = residual_codes_base + vec * residual_stride;
                        total_distance += jhq_internal::compute_cross_term_from_codes(
                            index,
                            primary_codes_ptr,
                            residual_codes_ptr,
                            index.separated_codes_.residual_subspace_stride,
                            index.separated_codes_.residual_level_stride);
                    }
                }
            }
            distances[i] = total_distance;
        }
        return;
    }

    for (int i = 0; i < n; ++i) {
        const int pf_i = i + prefetch_lookahead;
        if (prefetch_lookahead > 0 && pf_i < n) {
            const size_t pf_vec = static_cast<size_t>(ids[pf_i]);
            prefetch_L2(primary_codes_base + pf_vec * primary_stride);
            if (has_residual_codes) {
                if (has_residual_codes_packed4) {
                    prefetch_L2(residual_codes_packed4_base + pf_vec * residual_packed4_stride);
                    if (!has_cross_terms) {
                        prefetch_L2(residual_codes_base + pf_vec * residual_stride);
                    }
                } else {
                    prefetch_L2(residual_codes_base + pf_vec * residual_stride);
                }
            }
            if (has_cross_terms) {
                prefetch_L1(cross_terms_base + pf_vec);
            }
        }
        distances[i] = distance_to_index_with_primary(ids[i], 0.0f, false);
    }
}

float JHQDistanceComputer::distance_to_code_with_decoding(const uint8_t* code) const
{
    const bool metric_ip = (index.metric_type == METRIC_INNER_PRODUCT);
    BitstringReader bit_reader(code, index.code_size);
    float total_distance = 0.0f;
    float cross_term = 0.0f;

    float* residual_buffer = nullptr;
    if (has_residual_levels && !metric_ip) {
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
            const float* centroid_ptr = nullptr;
            if (!metric_ip) {
                std::fill(residual_buffer, residual_buffer + Ds, 0.0f);
                const uint32_t max_centroid = static_cast<uint32_t>(std::max(0, index.primary_ksub() - 1));
                const uint32_t safe_centroid = std::min<uint32_t>(centroid_id, max_centroid);
                centroid_ptr = index.get_primary_centroids_ptr(m) + static_cast<size_t>(safe_centroid) * Ds;
            }

            for (int level = 1; level < num_levels; ++level) {
                const size_t level_offset = residual_table_offsets[level];
                const int K_res = 1 << index.level_bits[level];
                const uint32_t residual_limit = static_cast<uint32_t>(K_res - 1);
                const size_t table_base = level_offset + static_cast<size_t>(m) * Ds * K_res;
                const int codebook_size = metric_ip ? 0 : index.scalar_codebook_ksub(level);
                const float* scalar_codebook =
                    (codebook_size > 0) ? index.get_scalar_codebook_ptr(m, level) : nullptr;

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

            if (!metric_ip && centroid_ptr) {
                for (int d = 0; d < Ds; ++d) {
                    cross_term += centroid_ptr[d] * residual_buffer[d];
                }
            }
        } else {
            bit_reader.i += index.residual_bits_per_subspace;
        }
    }

    if (has_residual_levels && !metric_ip) {
        total_distance += 2.0f * cross_term;
    }

    return total_distance;
}

void JHQDistanceComputer::apply_rotation_to_query(const float* x)
{
    if (index.use_jl_transform && index.is_rotation_trained) {
        if (index.use_bf16_rotation && !index.rotation_matrix_bf16.empty()) {
            
            index.apply_jl_rotation(1, x, query_rotated.data());
        } else {
            cblas_sgemv(
                CblasRowMajor,
                CblasNoTrans,
                index.d,
                index.d,
                1.0f,
                index.rotation_matrix.data(),
                index.d,
                x,
                1,
                0.0f,
                query_rotated.data(),
                1);
        }
    } else {
        std::memcpy(query_rotated.data(), x, sizeof(float) * index.d);
    }
    if (index.normalize_l2) {
        fvec_renorm_L2(static_cast<size_t>(index.d), 1, query_rotated.data());
    }
}

void JHQDistanceComputer::compute_primary_distance_table()
{
    index.compute_primary_distance_tables_flat(
        query_rotated.data(),
        K0,
        primary_distance_table_flat.data());
}

void JHQDistanceComputer::compute_residual_distance_tables()
{
    index.compute_residual_distance_tables(
        query_rotated.data(),
        residual_distance_tables_flat,
        residual_table_offsets);

    
    
    const size_t n = residual_distance_tables_flat.size();
    if (n > 0) {
        const float* src = residual_distance_tables_flat.data();
        float vmin = src[0];
        float vmax = src[0];
        for (size_t i = 1; i < n; ++i) {
            if (src[i] < vmin) vmin = src[i];
            if (src[i] > vmax) vmax = src[i];
        }
        const float range = vmax - vmin;
        residual_quant_offset = vmin;
        residual_quant_scale = (range > 0.0f) ? (range / 65535.0f) : 1.0f;
        const float inv_scale = (range > 0.0f) ? (65535.0f / range) : 0.0f;
        
        
        
        quantized_residual_tables.resize(n + 1);
        uint16_t* dst = quantized_residual_tables.data();
        for (size_t i = 0; i < n; ++i) {
            float v = (src[i] - vmin) * inv_scale;
            dst[i] = static_cast<uint16_t>(std::min(v, 65535.0f));
        }
        dst[n] = 0;
    } else {
        quantized_residual_tables.clear();
        residual_quant_scale = 1.0f;
        residual_quant_offset = 0.0f;
    }
}

	bool JHQDistanceComputer::use_br8_direct_dot_() const
	{
        if (index.metric_type != METRIC_L2) {
            return false;
        }
	    if (!index.has_pre_decoded_codes()) {
	        return false;
	    }
	    if (num_levels != 2 || index.num_levels != 2) {
        return false;
    }
    if (index.level_bits.size() < 2 || index.level_bits[1] != 8) {
        return false;
    }
    if (index.ntotal <= 0) {
        return false;
    }
    if (!index.scalar_codebooks_flat_valid_ ||
        index.scalar_codebooks_stride_ == 0 ||
        index.scalar_codebook_level_offsets_.size() < 2 ||
        index.scalar_codebooks_flat_.empty()) {
        return false;
    }
    if (index.scalar_codebook_level_offsets_[1] + 256 > index.scalar_codebooks_stride_) {
        return false;
    }
    if (index.scalar_codebooks_flat_.size() <
        static_cast<size_t>(index.M) * index.scalar_codebooks_stride_) {
        return false;
    }
    if (index.separated_codes_.residual_stride == 0 ||
        index.separated_codes_.residual_codes.empty()) {
        return false;
    }
	    if (index.separated_codes_.residual_codes.size() <
	        static_cast<size_t>(index.ntotal) * index.separated_codes_.residual_stride) {
	        return false;
	    }
	    return true;
	}

float JHQDistanceComputer::compute_br8_residual_direct_dot_(size_t vector_idx) const
{
    const int K_res = 1 << index.level_bits[1];
    FAISS_THROW_IF_NOT_MSG(K_res == 256, "compute_br8_residual_direct_dot_: expected K_res=256");
    FAISS_THROW_IF_NOT_MSG(
        vector_idx < index.separated_codes_.residual_norms.size(),
        "compute_br8_residual_direct_dot_: residual_norms out of bounds");

    const float* RESTRICT query = query_rotated.data();
    const uint8_t* RESTRICT residual_codes =
        index.separated_codes_.residual_codes.data() +
        vector_idx * index.separated_codes_.residual_stride;

    const float* RESTRICT scalar_base = index.scalar_codebooks_flat_.data();
    const size_t scalar_stride = index.scalar_codebooks_stride_;
    const size_t level_off = index.scalar_codebook_level_offsets_[1];

    float dot = 0.0f;
    for (int m = 0; m < M; ++m) {
        const float* RESTRICT q = query + static_cast<size_t>(m) * Ds;
        const uint8_t* RESTRICT codes_m = residual_codes + static_cast<size_t>(m) * Ds;
        const float* RESTRICT codebook =
            scalar_base + static_cast<size_t>(m) * scalar_stride + level_off;

#if defined(__AVX512F__)
        if (kHasAVX512 && Ds == 16) {
            const __m512 qv = _mm512_loadu_ps(q);
            const __m128i c128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes_m));
            const __m512i idx = _mm512_cvtepu8_epi32(c128);
            const __m512 rv = _mm512_i32gather_ps(idx, codebook, 4);
            dot += _mm512_reduce_add_ps(_mm512_mul_ps(qv, rv));
            continue;
        }
#endif
#if defined(__AVX2__)
        if (kHasAVX2 && Ds == 8) {
            const __m256 qv = _mm256_loadu_ps(q);
            const __m128i c64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes_m));
            const __m256i idx = _mm256_cvtepu8_epi32(c64);
            const __m256 rv = _mm256_i32gather_ps(codebook, idx, 4);
            dot += hsum256_ps(_mm256_mul_ps(qv, rv));
            continue;
        }
#endif

        for (int d = 0; d < Ds; ++d) {
            dot += q[d] * codebook[codes_m[d]];
        }
    }

    const float r_norm_sq = index.separated_codes_.residual_norms[vector_idx];
    return r_norm_sq - 2.0f * dot + query_norm_sq_;
}

float JHQDistanceComputer::compute_precomputed_residual_distance(size_t vector_idx) const
{
    if (!index.has_pre_decoded_codes() || index.num_levels <= 1) {
        return 0.0f;
    }

    if (use_br8_direct_dot_()) {
        return compute_br8_residual_direct_dot_(vector_idx);
    }

    
    const bool use_quant = !quantized_residual_tables.empty();

    
    if (num_levels == 2) {
        const int K_res = 1 << index.level_bits[1];
        const size_t level_stride_m = static_cast<size_t>(Ds) * static_cast<size_t>(K_res);

        
        if (use_quant) {
            const uint16_t* q_tables = quantized_residual_tables.data() + residual_table_offsets[1];
            const int total_lookups = M * Ds;

            if (index.separated_codes_.has_residual_codes_packed4() && K_res == 16) {
                const uint8_t* residual_codes =
                    index.separated_codes_.residual_codes_packed4.data() +
                    vector_idx * index.separated_codes_.residual_packed4_stride;
                uint32_t q_acc = 0;
#if defined(__AVX512F__)
                if (kHasAVX512 && Ds == 16) {
                    const __m512i d_offsets = _mm512_set_epi32(
                        15 * 16, 14 * 16, 13 * 16, 12 * 16,
                        11 * 16, 10 * 16, 9 * 16, 8 * 16,
                        7 * 16, 6 * 16, 5 * 16, 4 * 16,
                        3 * 16, 2 * 16, 1 * 16, 0);
                    const __m512i nibble_mask = _mm512_set1_epi32(0x0F);
                    const __m512i u16_mask = _mm512_set1_epi32(0xFFFF);
                    __m512i acc = _mm512_setzero_si512();
                    for (int m = 0; m < M; ++m) {
                        const uint16_t* tbl = q_tables + static_cast<size_t>(m) * level_stride_m;
                        __m128i packed8 = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(residual_codes));
                        __m128i shifted = _mm_srli_epi16(packed8, 4);
                        __m128i interleaved = _mm_unpacklo_epi8(packed8, shifted);
                        __m512i codes32 = _mm512_cvtepu8_epi32(interleaved);
                        codes32 = _mm512_and_epi32(codes32, nibble_mask);
                        __m512i indices = _mm512_add_epi32(d_offsets, codes32);
                        __m512i gathered = _mm512_i32gather_epi32(indices, tbl, 2);
                        gathered = _mm512_and_epi32(gathered, u16_mask);
                        acc = _mm512_add_epi32(acc, gathered);
                        residual_codes += 8;
                        if (m + 1 < M) {
                            prefetch_L1(q_tables + static_cast<size_t>(m + 1) * level_stride_m);
                        }
                    }
                    q_acc = static_cast<uint32_t>(_mm512_reduce_add_epi32(acc));
                } else
#endif
                {
                    for (int m = 0; m < M; ++m) {
                        const uint16_t* tbl = q_tables + static_cast<size_t>(m) * level_stride_m;
                        int d = 0;
                        for (; d + 1 < Ds; d += 2) {
                            const uint8_t packed = *residual_codes++;
                            q_acc += tbl[packed & 0x0F];
                            tbl += K_res;
                            q_acc += tbl[(packed >> 4) & 0x0F];
                            tbl += K_res;
                        }
                        if (d < Ds) {
                            const uint8_t packed = *residual_codes++;
                            q_acc += tbl[packed & 0x0F];
                        }
                    }
                }
                return static_cast<float>(q_acc) * residual_quant_scale
                    + static_cast<float>(total_lookups) * residual_quant_offset;
            }

            
            const uint8_t* residual_codes =
                index.separated_codes_.residual_codes.data() +
                vector_idx * index.separated_codes_.residual_stride;
            uint32_t q_acc = 0;
#if defined(__AVX512F__)
            if (kHasAVX512 && Ds == 16) {
                const __m512i d_offsets = _mm512_set_epi32(
                    15 * K_res, 14 * K_res, 13 * K_res, 12 * K_res,
                    11 * K_res, 10 * K_res, 9 * K_res, 8 * K_res,
                    7 * K_res, 6 * K_res, 5 * K_res, 4 * K_res,
                    3 * K_res, 2 * K_res, 1 * K_res, 0);
                const __m512i u16_mask = _mm512_set1_epi32(0xFFFF);
                __m512i acc = _mm512_setzero_si512();
                for (int m = 0; m < M; ++m) {
                    const uint16_t* tbl = q_tables + static_cast<size_t>(m) * level_stride_m;
                    __m128i codes128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(residual_codes + m * Ds));
                    __m512i codes32 = _mm512_cvtepu8_epi32(codes128);
                    __m512i indices = _mm512_add_epi32(d_offsets, codes32);
                    __m512i gathered = _mm512_i32gather_epi32(indices, tbl, 2);
                    gathered = _mm512_and_epi32(gathered, u16_mask);
                    acc = _mm512_add_epi32(acc, gathered);
                    if (m + 1 < M) {
                        prefetch_L1(residual_codes + (m + 1) * Ds);
                        prefetch_L1(q_tables + static_cast<size_t>(m + 1) * level_stride_m);
                    }
                }
                q_acc = static_cast<uint32_t>(_mm512_reduce_add_epi32(acc));
            } else
#endif
#if defined(__AVX2__)
            if (kHasAVX2 && Ds == 8) {
                const __m256i d_offsets = _mm256_set_epi32(
                    7 * K_res, 6 * K_res, 5 * K_res, 4 * K_res,
                    3 * K_res, 2 * K_res, 1 * K_res, 0);
                const __m256i u16_mask = _mm256_set1_epi32(0xFFFF);
                __m256i acc = _mm256_setzero_si256();
                for (int m = 0; m < M; ++m) {
                    const uint16_t* tbl = q_tables + static_cast<size_t>(m) * level_stride_m;
                    __m128i codes64 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(residual_codes + m * Ds));
                    __m256i codes32 = _mm256_cvtepu8_epi32(codes64);
                    __m256i indices = _mm256_add_epi32(d_offsets, codes32);
                    __m256i gathered = _mm256_i32gather_epi32(
                        reinterpret_cast<const int*>(tbl), indices, 2);
                    gathered = _mm256_and_si256(gathered, u16_mask);
                    acc = _mm256_add_epi32(acc, gathered);
                    if (m + 1 < M) {
                        prefetch_L1(residual_codes + (m + 1) * Ds);
                        prefetch_L1(q_tables + static_cast<size_t>(m + 1) * level_stride_m);
                    }
                }
                q_acc = hsum256_epi32(acc);
            } else
#endif
            {
                for (int m = 0; m < M; ++m) {
                    const uint16_t* tbl = q_tables + static_cast<size_t>(m) * level_stride_m;
                    for (int d = 0; d < Ds; ++d) {
                        q_acc += tbl[*residual_codes++];
                        tbl += K_res;
                    }
                }
            }
            return static_cast<float>(q_acc) * residual_quant_scale
                + static_cast<float>(total_lookups) * residual_quant_offset;
        }

        
        float residual_distance = 0.0f;
        const float* level_tables = residual_distance_tables_flat.data() + residual_table_offsets[1];

        if (index.separated_codes_.has_residual_codes_packed4() && K_res == 16) {
            const uint8_t* residual_codes =
                index.separated_codes_.residual_codes_packed4.data() +
                vector_idx * index.separated_codes_.residual_packed4_stride;
#if defined(__AVX512F__)
            if (kHasAVX512 && Ds == 16) {
                
                
                
                const __m512i d_offsets = _mm512_set_epi32(
                    15 * 16, 14 * 16, 13 * 16, 12 * 16,
                    11 * 16, 10 * 16, 9 * 16, 8 * 16,
                    7 * 16, 6 * 16, 5 * 16, 4 * 16,
                    3 * 16, 2 * 16, 1 * 16, 0);
                const __m512i lo_mask = _mm512_set1_epi32(0x0F);
                __m512 acc = _mm512_setzero_ps();
                for (int m = 0; m < M; ++m) {
                    const float* table_md = level_tables + static_cast<size_t>(m) * level_stride_m;
                    
                    __m128i packed8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(residual_codes));
                    __m128i shifted = _mm_srli_epi16(packed8, 4);
                    __m128i interleaved = _mm_unpacklo_epi8(packed8, shifted);
                    __m512i codes32 = _mm512_cvtepu8_epi32(interleaved);
                    codes32 = _mm512_and_epi32(codes32, lo_mask);
                    __m512i indices = _mm512_add_epi32(d_offsets, codes32);
                    acc = _mm512_add_ps(acc, _mm512_i32gather_ps(indices, table_md, 4));
                    residual_codes += 8;
                    if (m + 1 < M) {
                        prefetch_L1(level_tables + static_cast<size_t>(m + 1) * level_stride_m);
                    }
                }
                residual_distance += _mm512_reduce_add_ps(acc);
                return residual_distance;
            } else
#endif
#if defined(__AVX2__)
            if (kHasAVX2 && Ds == 16) {
                
                
                const __m256i d_offsets_lo = _mm256_set_epi32(
                    7 * 16, 6 * 16, 5 * 16, 4 * 16,
                    3 * 16, 2 * 16, 1 * 16, 0);
                const __m256i d_offsets_hi = _mm256_set_epi32(
                    15 * 16, 14 * 16, 13 * 16, 12 * 16,
                    11 * 16, 10 * 16, 9 * 16, 8 * 16);
                const __m256i lo_mask = _mm256_set1_epi32(0x0F);
                __m256 acc = _mm256_setzero_ps();
                for (int m = 0; m < M; ++m) {
                    const float* table_md = level_tables + static_cast<size_t>(m) * level_stride_m;
                    __m128i packed8 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(residual_codes));
                    __m128i shifted = _mm_srli_epi16(packed8, 4);
                    __m128i interleaved = _mm_unpacklo_epi8(packed8, shifted);
                    __m256i codes_lo = _mm256_cvtepu8_epi32(interleaved);
                    codes_lo = _mm256_and_si256(codes_lo, lo_mask);
                    __m256i idx_lo = _mm256_add_epi32(d_offsets_lo, codes_lo);
                    acc = _mm256_add_ps(acc, _mm256_i32gather_ps(table_md, idx_lo, 4));
                    __m128i hi8 = _mm_srli_si128(interleaved, 8);
                    __m256i codes_hi = _mm256_cvtepu8_epi32(hi8);
                    codes_hi = _mm256_and_si256(codes_hi, lo_mask);
                    __m256i idx_hi = _mm256_add_epi32(d_offsets_hi, codes_hi);
                    acc = _mm256_add_ps(acc, _mm256_i32gather_ps(table_md, idx_hi, 4));
                    residual_codes += 8;
                    if (m + 1 < M) {
                        prefetch_L1(level_tables + static_cast<size_t>(m + 1) * level_stride_m);
                    }
                }
                __m128 hi = _mm256_extractf128_ps(acc, 1);
                __m128 lo_ps = _mm256_castps256_ps128(acc);
                __m128 sum = _mm_add_ps(lo_ps, hi);
                sum = _mm_hadd_ps(sum, sum);
                sum = _mm_hadd_ps(sum, sum);
                residual_distance += _mm_cvtss_f32(sum);
                return residual_distance;
            } else
#endif
            {
                for (int m = 0; m < M; ++m) {
                    const float* table_md = level_tables + static_cast<size_t>(m) * level_stride_m;
                    int d = 0;
                    for (; d + 1 < Ds; d += 2) {
                        const uint8_t packed = *residual_codes++;
                        residual_distance += table_md[packed & 0x0F];
                        table_md += K_res;
                        residual_distance += table_md[(packed >> 4) & 0x0F];
                        table_md += K_res;
                    }
                    if (d < Ds) {
                        const uint8_t packed = *residual_codes++;
                        residual_distance += table_md[packed & 0x0F];
                    }
                }
                return residual_distance;
            }
        }

        const uint8_t* residual_codes =
            index.separated_codes_.residual_codes.data() +
            vector_idx * index.separated_codes_.residual_stride;
#if defined(__AVX512F__)
        if (kHasAVX512 && Ds == 16) {
            
            
            const __m512i d_offsets = _mm512_set_epi32(
                15 * K_res, 14 * K_res, 13 * K_res, 12 * K_res,
                11 * K_res, 10 * K_res, 9 * K_res, 8 * K_res,
                7 * K_res, 6 * K_res, 5 * K_res, 4 * K_res,
                3 * K_res, 2 * K_res, 1 * K_res, 0);
            __m512 acc = _mm512_setzero_ps();
            for (int m = 0; m < M; ++m) {
                const float* table_md = level_tables + static_cast<size_t>(m) * level_stride_m;
                __m128i codes128 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(residual_codes + m * Ds));
                __m512i codes32 = _mm512_cvtepu8_epi32(codes128);
                __m512i indices = _mm512_add_epi32(d_offsets, codes32);
                acc = _mm512_add_ps(acc, _mm512_i32gather_ps(indices, table_md, 4));
                if (m + 1 < M) {
                    prefetch_L1(residual_codes + (m + 1) * Ds);
                    prefetch_L1(level_tables + static_cast<size_t>(m + 1) * level_stride_m);
                }
            }
            residual_distance += _mm512_reduce_add_ps(acc);
            return residual_distance;
        }
#endif
#if defined(__AVX2__)
        if (kHasAVX2 && Ds == 16) {
            
            
            const __m256i d_offsets_lo = _mm256_set_epi32(
                7 * K_res, 6 * K_res, 5 * K_res, 4 * K_res,
                3 * K_res, 2 * K_res, 1 * K_res, 0);
            const __m256i d_offsets_hi = _mm256_set_epi32(
                15 * K_res, 14 * K_res, 13 * K_res, 12 * K_res,
                11 * K_res, 10 * K_res, 9 * K_res, 8 * K_res);
            __m256 acc = _mm256_setzero_ps();
            for (int m = 0; m < M; ++m) {
                const float* table_md = level_tables + static_cast<size_t>(m) * level_stride_m;
                __m128i lo8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(residual_codes + m * Ds));
                __m256i codes_lo = _mm256_cvtepu8_epi32(lo8);
                __m256i idx_lo = _mm256_add_epi32(d_offsets_lo, codes_lo);
                acc = _mm256_add_ps(acc, _mm256_i32gather_ps(table_md, idx_lo, 4));
                __m128i hi8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(residual_codes + m * Ds + 8));
                __m256i codes_hi = _mm256_cvtepu8_epi32(hi8);
                __m256i idx_hi = _mm256_add_epi32(d_offsets_hi, codes_hi);
                acc = _mm256_add_ps(acc, _mm256_i32gather_ps(table_md, idx_hi, 4));
                if (m + 1 < M) {
                    prefetch_L1(residual_codes + (m + 1) * Ds);
                    prefetch_L1(level_tables + static_cast<size_t>(m + 1) * level_stride_m);
                }
            }
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 sum = _mm_add_ps(lo, hi);
            sum = _mm_hadd_ps(sum, sum);
            sum = _mm_hadd_ps(sum, sum);
            residual_distance += _mm_cvtss_f32(sum);
            return residual_distance;
        }
        if (kHasAVX2 && Ds == 8) {
            
            const __m256i d_offsets = _mm256_set_epi32(
                7 * K_res, 6 * K_res, 5 * K_res, 4 * K_res,
                3 * K_res, 2 * K_res, 1 * K_res, 0);
            __m256 acc = _mm256_setzero_ps();
            for (int m = 0; m < M; ++m) {
                const float* table_md = level_tables + static_cast<size_t>(m) * level_stride_m;
                __m128i codes64 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i*>(residual_codes + m * Ds));
                __m256i codes32 = _mm256_cvtepu8_epi32(codes64);
                __m256i indices = _mm256_add_epi32(d_offsets, codes32);
                acc = _mm256_add_ps(acc, _mm256_i32gather_ps(table_md, indices, 4));
                if (m + 1 < M) {
                    prefetch_L1(residual_codes + (m + 1) * Ds);
                    prefetch_L1(level_tables + static_cast<size_t>(m + 1) * level_stride_m);
                }
            }
            residual_distance += hsum256_ps(acc);
            return residual_distance;
        }
#endif
        {
            for (int m = 0; m < M; ++m) {
                const float* table_md = level_tables + static_cast<size_t>(m) * level_stride_m;
                for (int d = 0; d < Ds; ++d) {
                    const uint8_t scalar_id = *residual_codes++;
                    residual_distance += table_md[scalar_id];
                    table_md += K_res;
                }
            }
            return residual_distance;
        }
    }

    float residual_distance = 0.0f;
    const uint8_t* residual_codes =
        index.separated_codes_.residual_codes.data() +
        vector_idx * index.separated_codes_.residual_stride;
    for (int m = 0; m < M; ++m) {
        for (int level = 1; level < num_levels; ++level) {
            const int K_res = 1 << index.level_bits[level];
            const float* table_md = residual_distance_tables_flat.data()
                + residual_table_offsets[level]
                + static_cast<size_t>(m) * static_cast<size_t>(Ds) * static_cast<size_t>(K_res);

            for (int d = 0; d < Ds; ++d) {
                const uint8_t scalar_id = *residual_codes++;
                residual_distance += table_md[scalar_id];
                table_md += K_res;
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
    uint32_t version = 4;
    f->operator()(&version, sizeof(version), 1);

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
    f->operator()(&idx->kmeans_nredo, sizeof(idx->kmeans_nredo), 1);
    f->operator()(&idx->kmeans_seed, sizeof(idx->kmeans_seed), 1);

    f->operator()(&idx->is_rotation_trained, sizeof(idx->is_rotation_trained), 1);

    if (idx->use_jl_transform && idx->is_rotation_trained) {
        
        if (idx->use_bf16_rotation && !idx->rotation_matrix_bf16.empty()) {
            size_t rot_size = idx->rotation_matrix_bf16.size();
            f->operator()(&rot_size, sizeof(rot_size), 1);
            if (rot_size > 0) {
                
                constexpr size_t CHUNK = 4096;
                std::vector<float> buf(CHUNK);
                for (size_t off = 0; off < rot_size; off += CHUNK) {
                    size_t cnt = std::min(CHUNK, rot_size - off);
                    for (size_t i = 0; i < cnt; ++i) {
                        uint32_t bits = static_cast<uint32_t>(idx->rotation_matrix_bf16[off + i]) << 16;
                        std::memcpy(&buf[i], &bits, sizeof(float));
                    }
                    f->operator()(buf.data(), sizeof(float), cnt);
                }
            }
        } else {
            size_t rot_size = idx->rotation_matrix.size();
            f->operator()(&rot_size, sizeof(rot_size), 1);
            if (rot_size > 0) {
                f->operator()(idx->rotation_matrix.data(), sizeof(float), rot_size);
            }
        }
    }

    const size_t primary_codeword_size =
        static_cast<size_t>(idx->primary_ksub()) * static_cast<size_t>(idx->Ds);
    for (int m = 0; m < idx->M; ++m) {
        for (int level = 0; level < idx->num_levels; ++level) {
            if (level == 0) {
                f->operator()(&primary_codeword_size, sizeof(primary_codeword_size), 1);
                if (primary_codeword_size > 0) {
                    const float* primary_centroids = idx->get_primary_centroids_ptr(m);
                    f->operator()(primary_centroids, sizeof(float), primary_codeword_size);
                }
                continue;
            }
            const size_t codeword_size = 0;
            f->operator()(&codeword_size, sizeof(codeword_size), 1);
        }
    }

    size_t scalar_stride = 0;
    for (int level = 1; level < idx->num_levels; ++level) {
        scalar_stride += static_cast<size_t>(idx->scalar_codebook_ksub(level));
    }
    const size_t scalar_flat_size = static_cast<size_t>(idx->M) * scalar_stride;
    f->operator()(&scalar_flat_size, sizeof(scalar_flat_size), 1);
    if (scalar_flat_size > 0) {
        const bool can_write_flat =
            idx->scalar_codebooks_flat_valid_ &&
            idx->scalar_codebooks_flat_.size() >= scalar_flat_size;
        if (can_write_flat) {
            f->operator()(idx->scalar_codebooks_flat_.data(), sizeof(float), scalar_flat_size);
        } else {
            std::vector<float> scalar_flat_tmp(scalar_flat_size, 0.0f);
            for (int m = 0; m < idx->M; ++m) {
                float* dst = scalar_flat_tmp.data() + static_cast<size_t>(m) * scalar_stride;
                size_t off = 0;
                for (int level = 1; level < idx->num_levels; ++level) {
                    const size_t ksub = static_cast<size_t>(idx->scalar_codebook_ksub(level));
                    const float* scalar_codebook = idx->get_scalar_codebook_ptr(m, level);
                    FAISS_THROW_IF_NOT_MSG(
                        scalar_codebook != nullptr,
                        "write_index_jhq: null scalar codebook pointer");
                    std::memcpy(dst + off, scalar_codebook, ksub * sizeof(float));
                    off += ksub;
                }
            }
            f->operator()(scalar_flat_tmp.data(), sizeof(float), scalar_flat_size);
        }
    }

    const uint8_t* codes_ptr = idx->codes.data();
    size_t codes_size = idx->codes.size();
    std::vector<uint8_t> packed_codes;

    if (codes_size == 0 && idx->has_pre_decoded_codes() &&
        idx->code_size > 0 && idx->ntotal > 0) {
        packed_codes.resize(static_cast<size_t>(idx->ntotal) * idx->code_size);
        for (idx_t i = 0; i < idx->ntotal; ++i) {
            uint8_t* dest = packed_codes.data() + static_cast<size_t>(i) * idx->code_size;
            faiss::BitstringWriter bit_writer(dest, idx->code_size);
            std::memset(dest, 0, idx->code_size);

            const uint8_t* primary_codes = idx->get_primary_codes_ptr(i);
            const uint8_t* residual_codes = (idx->num_levels > 1)
                ? idx->get_residual_codes_ptr(i)
                : nullptr;
            size_t residual_offset = 0;

            for (int m = 0; m < idx->M; ++m) {
                bit_writer.write(primary_codes[m], idx->level_bits[0]);
                if (idx->num_levels > 1 && residual_codes != nullptr) {
                    for (int level = 1; level < idx->num_levels; ++level) {
                        for (int d = 0; d < idx->Ds; ++d) {
                            bit_writer.write(residual_codes[residual_offset++],
                                             idx->level_bits[level]);
                        }
                    }
                }
            }
        }
        codes_ptr = packed_codes.data();
        codes_size = packed_codes.size();
    }

    f->operator()(&codes_size, sizeof(codes_size), 1);
    if (codes_size > 0) {
        f->operator()(codes_ptr, sizeof(uint8_t), codes_size);
    }

    f->operator()(&idx->residual_bits_per_subspace, sizeof(idx->residual_bits_per_subspace), 1);

    f->operator()(&idx->memory_layout_initialized_, sizeof(idx->memory_layout_initialized_), 1);

    
    size_t cross_terms_size = idx->separated_codes_.cross_terms.size();
    FAISS_THROW_IF_NOT_MSG(
        cross_terms_size == 0 || cross_terms_size == static_cast<size_t>(idx->ntotal),
        "Invalid JHQ cross_terms payload size");
    f->operator()(&cross_terms_size, sizeof(cross_terms_size), 1);
    if (cross_terms_size > 0) {
        f->operator()(idx->separated_codes_.cross_terms.data(), sizeof(float), cross_terms_size);
    }

    
    size_t residual_norms_size = idx->separated_codes_.residual_norms.size();
    if (idx->num_levels != 2) {
        residual_norms_size = 0;
    }
    FAISS_THROW_IF_NOT_MSG(
        residual_norms_size == 0 || residual_norms_size == static_cast<size_t>(idx->ntotal),
        "Invalid JHQ residual_norms payload size");
    f->operator()(&residual_norms_size, sizeof(residual_norms_size), 1);
    if (residual_norms_size > 0) {
        f->operator()(idx->separated_codes_.residual_norms.data(), sizeof(float), residual_norms_size);
    }
}

IndexJHQ* read_index_jhq(IOReader* f)
{
    uint32_t magic;
    f->operator()(&magic, sizeof(magic), 1);
    FAISS_THROW_IF_NOT_MSG(magic == 0x4A525051, "Invalid JHQ magic number");
    uint32_t version = 0;
    f->operator()(&version, sizeof(version), 1);
    FAISS_THROW_IF_NOT_MSG(
        version == 3 || version == 4,
        "Unsupported JHQ version (expected 3 or 4)");

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
    f->operator()(&idx->kmeans_nredo, sizeof(idx->kmeans_nredo), 1);
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

    const size_t expected_primary_codeword_size =
        static_cast<size_t>(idx->primary_ksub()) * static_cast<size_t>(idx->Ds);
    for (int m = 0; m < idx->M; ++m) {
        for (int level = 0; level < idx->num_levels; ++level) {
            size_t codeword_size = 0;
            f->operator()(&codeword_size, sizeof(codeword_size), 1);
            if (level == 0) {
                FAISS_THROW_IF_NOT_MSG(
                    codeword_size == expected_primary_codeword_size,
                    "Invalid JHQ primary centroid payload size");
                if (codeword_size > 0) {
                    float* primary_centroids = idx->get_primary_centroids_ptr_mutable(m);
                    f->operator()(primary_centroids, sizeof(float), codeword_size);
                }
                continue;
            }
            FAISS_THROW_IF_NOT_MSG(
                codeword_size == 0,
                "Invalid JHQ residual vector-codebook payload (expected 0 in v3)");
        }
    }
    idx->primary_pq_dirty_ = false;

    size_t scalar_flat_size = 0;
    f->operator()(&scalar_flat_size, sizeof(scalar_flat_size), 1);
    size_t scalar_stride = 0;
    for (int level = 1; level < idx->num_levels; ++level) {
        scalar_stride += static_cast<size_t>(idx->scalar_codebook_ksub(level));
    }
    const size_t expected_scalar_flat_size = static_cast<size_t>(idx->M) * scalar_stride;
    FAISS_THROW_IF_NOT_MSG(
        scalar_flat_size == expected_scalar_flat_size,
        "Invalid JHQ scalar codebook flat payload size");
    idx->scalar_codebook_level_offsets_.assign(static_cast<size_t>(idx->num_levels), 0u);
    {
        size_t off = 0;
        for (int level = 1; level < idx->num_levels; ++level) {
            idx->scalar_codebook_level_offsets_[static_cast<size_t>(level)] = off;
            off += static_cast<size_t>(idx->scalar_codebook_ksub(level));
        }
    }
    idx->scalar_codebooks_stride_ = scalar_stride;
    idx->scalar_codebooks_flat_.resize(scalar_flat_size);
    if (scalar_flat_size > 0) {
        f->operator()(idx->scalar_codebooks_flat_.data(), sizeof(float), scalar_flat_size);
    }
    idx->scalar_codebooks_flat_valid_ = true;

    size_t codes_size = 0;
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
            idx->generate_qr_rotation_matrix(42);
        }

        const size_t expected_codes_size = idx->ntotal * idx->code_size;
        const bool has_packed_codes =
            idx->code_size > 0 && idx->codes.size() == expected_codes_size &&
            expected_codes_size > 0;

        idx->memory_layout_initialized_ = false;
        idx->initialize_memory_layout();
        if (has_packed_codes) {
            
            idx->extract_all_codes_after_add(false, false);
        } else {
            idx->separated_codes_.clear();
        }
    }

    
    size_t cross_terms_size = 0;
    f->operator()(&cross_terms_size, sizeof(cross_terms_size), 1);
    FAISS_THROW_IF_NOT_MSG(
        cross_terms_size == 0 || cross_terms_size == static_cast<size_t>(idx->ntotal),
        "Invalid JHQ cross_terms payload size");
    if (cross_terms_size > 0) {
        idx->separated_codes_.cross_terms.resize(cross_terms_size);
        f->operator()(idx->separated_codes_.cross_terms.data(), sizeof(float), cross_terms_size);
    } else if (idx->num_levels > 1 && idx->has_pre_decoded_codes()) {
        
#pragma omp parallel for schedule(static, 1024) if (idx->ntotal > 10000)
        for (idx_t i = 0; i < idx->ntotal; ++i) {
            idx->separated_codes_.cross_terms[i] = jhq_internal::compute_cross_term_from_codes(
                *idx,
                idx->separated_codes_.get_primary_codes(i),
                idx->separated_codes_.get_residual_codes(i),
                idx->separated_codes_.residual_subspace_stride,
                idx->separated_codes_.residual_level_stride);
        }
    }

    if (version >= 4) {
        size_t residual_norms_size = 0;
        f->operator()(&residual_norms_size, sizeof(residual_norms_size), 1);
        FAISS_THROW_IF_NOT_MSG(
            residual_norms_size == 0 || residual_norms_size == static_cast<size_t>(idx->ntotal),
            "Invalid JHQ residual_norms payload size");
        if (residual_norms_size > 0) {
            idx->separated_codes_.residual_norms.resize(residual_norms_size);
            f->operator()(idx->separated_codes_.residual_norms.data(), sizeof(float), residual_norms_size);
        } else if (idx->num_levels == 2 && idx->has_pre_decoded_codes()) {
            
            idx->separated_codes_.residual_norms.resize(static_cast<size_t>(idx->ntotal), 0.0f);
#pragma omp parallel for schedule(static, 1024) if (idx->ntotal > 10000)
            for (idx_t i = 0; i < idx->ntotal; ++i) {
                idx->separated_codes_.residual_norms[i] =
                    jhq_internal::compute_residual_norm_sq_from_codes(
                        *idx,
                        idx->separated_codes_.get_residual_codes(i),
                        idx->separated_codes_.residual_subspace_stride,
                        idx->separated_codes_.residual_level_stride);
            }
        }
    }

    return idx;
}

IndexJHQ* read_index_jhq(const char* fname)
{
    FAISS_THROW_IF_NOT_MSG(fname != nullptr && fname[0] != '\0',
                           "read_index_jhq: invalid file path");
    const bool prefer_mmap = parse_env_bool("JHQ_USE_MMAP_READ", true);
    if (prefer_mmap) {
        std::shared_ptr<MmappedFileMappingOwner> mmap_owner;
        try {
            mmap_owner = std::make_shared<MmappedFileMappingOwner>(fname);
        } catch (...) {
            mmap_owner.reset();
        }
        if (mmap_owner) {
            MappedFileIOReader reader(mmap_owner);
            return read_index_jhq(&reader);
        }
    }
    FileIOReader reader(fname);
    return read_index_jhq(&reader);
}

void IndexJHQ::encode_to_separated_storage(idx_t n, const float* x_rotated) const
{
    const idx_t old_total = ntotal;
    const bool have_flat_codes = (old_total > 0 && codes.size() > 0);

    separated_codes_.initialize(M, Ds, num_levels, old_total + n);

    if (old_total > 0 && codes.size() > 0) {
#pragma omp parallel for schedule(static) if (old_total > 1000)
        for (idx_t i = 0; i < old_total; ++i) {
            extract_single_vector_all_codes(i);
        }
    }

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual_scratch(static_cast<size_t>(Ds));
#pragma omp for schedule(static)
        for (idx_t i = 0; i < n; ++i) {
            encode_single_vector_separated_with_scratch(
                x_rotated + i * d,
                old_total + i,
                residual_scratch.data());
        }
    }

    if (num_levels > 1 && metric_type == METRIC_L2) {
        const bool compute_norms = (num_levels == 2);
        if (compute_norms) {
            separated_codes_.residual_norms.resize(static_cast<size_t>(old_total + n), 0.0f);
        }
        
        
        
        if (have_flat_codes) {
#pragma omp parallel for schedule(static) if (old_total > 1000)
            for (idx_t i = 0; i < old_total; ++i) {
                separated_codes_.cross_terms[i] =
                    jhq_internal::compute_cross_term_from_codes(
                        *this,
                        separated_codes_.get_primary_codes(i),
                        separated_codes_.get_residual_codes(i),
                        separated_codes_.residual_subspace_stride,
                        separated_codes_.residual_level_stride);
                if (compute_norms) {
                    separated_codes_.residual_norms[i] =
                        jhq_internal::compute_residual_norm_sq_from_codes(
                            *this,
                            separated_codes_.get_residual_codes(i),
                            separated_codes_.residual_subspace_stride,
                            separated_codes_.residual_level_stride);
                }
            }
        }
        
#pragma omp parallel for schedule(static) if (n > 1000)
        for (idx_t i = 0; i < n; ++i) {
            const idx_t vec_idx = old_total + i;
            separated_codes_.cross_terms[vec_idx] =
                jhq_internal::compute_cross_term_from_codes(
                    *this,
                    separated_codes_.get_primary_codes(vec_idx),
                    separated_codes_.get_residual_codes(vec_idx),
                    separated_codes_.residual_subspace_stride,
                    separated_codes_.residual_level_stride);
            if (compute_norms) {
                separated_codes_.residual_norms[vec_idx] =
                    jhq_internal::compute_residual_norm_sq_from_codes(
                        *this,
                        separated_codes_.get_residual_codes(vec_idx),
                        separated_codes_.residual_subspace_stride,
                        separated_codes_.residual_level_stride);
            }
        }
    }

    rebuild_residual_codes_packed4();
}

void IndexJHQ::encode_single_vector_separated(const float* x, idx_t vector_idx) const
{
    std::vector<float> residual_scratch(static_cast<size_t>(Ds));
    encode_single_vector_separated_with_scratch(
        x, vector_idx, residual_scratch.data());
}

void IndexJHQ::encode_single_vector_separated_with_scratch(
    const float* x,
    idx_t vector_idx,
    float* current_residual) const
{
    uint8_t* primary_dest = separated_codes_.get_primary_codes_mutable(vector_idx);
    uint8_t* residual_dest = (num_levels > 1)
        ? separated_codes_.get_residual_codes_mutable(vector_idx)
        : nullptr;

    size_t residual_offset = 0;

    for (int m = 0; m < M; ++m) {
        const float* subspace_vector = x + m * Ds;
        std::memcpy(
            current_residual,
            subspace_vector,
            static_cast<size_t>(Ds) * sizeof(float));

        const float* centroids = get_primary_centroids_ptr(m);
        const int K = primary_ksub();
        int best_k = find_best_centroid(current_residual, centroids, K);
        primary_dest[m] = static_cast<uint8_t>(best_k);

        subtract_centroid(current_residual, centroids, best_k);

        if (residual_dest && num_levels > 1) {
            encode_residual_levels_separated(
                m, current_residual, residual_dest, residual_offset);
        }
    }
}

int IndexJHQ::find_best_centroid(const float* residual, const float* centroids, int K) const
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
                residual, centroids + k * Ds, Ds);
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
                residual, centroids + k * Ds, Ds);
            if (dist < best_dist) {
                best_dist = dist;
                best_k = k;
            }
        }
    }

    return best_k;
}

int IndexJHQ::find_nearest_scalar_sorted(const float* codebook, int K, float value) const
{
    FAISS_THROW_IF_NOT_MSG(codebook != nullptr, "find_nearest_scalar_sorted: null codebook");
    FAISS_THROW_IF_NOT_MSG(K > 0, "find_nearest_scalar_sorted: invalid codebook size");

    const float* begin = codebook;
    const float* end = codebook + static_cast<size_t>(K);
    const float* it = std::lower_bound(begin, end, value);
    if (it == begin) {
        return 0;
    }
    if (it == end) {
        return K - 1;
    }

    const size_t hi = static_cast<size_t>(it - begin);
    const size_t lo = hi - 1;
    const float d_lo = std::abs(value - codebook[lo]);
    const float d_hi = std::abs(codebook[hi] - value);
    return static_cast<int>((d_lo <= d_hi) ? lo : hi);
}

void IndexJHQ::subtract_centroid(float* residual, const float* centroids, int best_k) const
{
    const float* best_centroid = centroids + best_k * Ds;

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
        const int K_res = scalar_codebook_ksub(level);
        const float* scalar_codebook = get_scalar_codebook_ptr(m, level);
        FAISS_THROW_IF_NOT_MSG(scalar_codebook != nullptr, "encode_residual_levels_separated: null scalar codebook");

        for (int d = 0; d < Ds; ++d) {
            const int best_k = find_nearest_scalar_sorted(
                scalar_codebook,
                K_res,
                residual[d]);

            residual_dest[offset++] = static_cast<uint8_t>(best_k);
        }
    }
}

float IndexJHQ::compute_exact_distance_separated_codes_scratch(
    const uint8_t* primary_codes,
    const uint8_t* residual_codes,
    const float* query_rotated,
    float* query_residual,
    float* db_residual) const
{
    if (metric_type == METRIC_INNER_PRODUCT) {
        float total_distance = 0.0f;
        size_t residual_offset = 0;
        for (int m = 0; m < M; ++m) {
            const float* query_sub = query_rotated + m * Ds;

            uint8_t centroid_id = primary_codes[m];
            centroid_id = static_cast<uint8_t>(
                std::min<uint32_t>(centroid_id, static_cast<uint32_t>(std::max(0, primary_ksub() - 1))));
            const float* centroids = get_primary_centroids_ptr(m);
            const float* primary_centroid = centroids + centroid_id * Ds;
            total_distance -= fvec_inner_product(query_sub, primary_centroid, Ds);

            if (residual_codes && num_levels > 1) {
                for (int level = 1; level < num_levels; ++level) {
                    const float* scalar_codebook = get_scalar_codebook_ptr(m, level);
                    const uint32_t max_scalar = static_cast<uint32_t>(
                        std::max(0, scalar_codebook_ksub(level) - 1));
                    for (int d = 0; d < Ds; ++d) {
                        uint8_t scalar_id = residual_codes[residual_offset++];
                        const uint32_t safe_scalar =
                            std::min<uint32_t>(static_cast<uint32_t>(scalar_id), max_scalar);
                        total_distance -= query_sub[d] * scalar_codebook[safe_scalar];
                    }
                }
            }
        }

        (void)query_residual;
        (void)db_residual;
        return total_distance;
    }

    float total_distance = 0.0f;
    size_t residual_offset = 0;

    for (int m = 0; m < M; ++m) {
        const float* query_sub = query_rotated + m * Ds;

        uint8_t centroid_id = primary_codes[m];
        centroid_id = static_cast<uint8_t>(
            std::min<uint32_t>(centroid_id, static_cast<uint32_t>(std::max(0, primary_ksub() - 1))));
        const float* centroids = get_primary_centroids_ptr(m);
        const float* primary_centroid = centroids + centroid_id * Ds;

        for (int d = 0; d < Ds; ++d) {
            query_residual[d] = query_sub[d] - primary_centroid[d];
        }

        std::fill_n(db_residual, static_cast<size_t>(Ds), 0.0f);
        if (residual_codes && num_levels > 1) {
            for (int level = 1; level < num_levels; ++level) {
                const float* scalar_codebook = get_scalar_codebook_ptr(m, level);
                for (int d = 0; d < Ds; ++d) {
                    uint8_t scalar_id = residual_codes[residual_offset++];
                    db_residual[d] += scalar_codebook[scalar_id];
                }
            }
        }

        total_distance += jhq_internal::fvec_L2sqr_dispatch(
            query_residual, db_residual, Ds);
    }

    return total_distance;
}

float IndexJHQ::compute_exact_distance_separated_scratch(
    idx_t vector_idx,
    const float* query_rotated,
    float* query_residual,
    float* db_residual) const
{
    FAISS_THROW_IF_NOT_MSG(
        separated_codes_.is_initialized && vector_idx >= 0 &&
            static_cast<size_t>(vector_idx) < static_cast<size_t>(ntotal),
        "compute_exact_distance_separated_scratch: vector_idx out of bounds");
    const uint8_t* primary_codes = separated_codes_.primary_codes.data() +
        static_cast<size_t>(vector_idx) * separated_codes_.primary_stride;
    const uint8_t* residual_codes =
        (num_levels > 1 && !separated_codes_.residual_codes.empty() &&
         separated_codes_.residual_stride > 0)
        ? (separated_codes_.residual_codes.data() +
           static_cast<size_t>(vector_idx) * separated_codes_.residual_stride)
        : nullptr;
    return compute_exact_distance_separated_codes_scratch(
        primary_codes,
        residual_codes,
        query_rotated,
        query_residual,
        db_residual);
}

float IndexJHQ::compute_exact_distance_separated(idx_t vector_idx, const float* query_rotated) const
{
    thread_local std::vector<float> query_residual_tls;
    thread_local std::vector<float> db_residual_tls;

    if (query_residual_tls.size() != static_cast<size_t>(Ds)) {
        query_residual_tls.resize(static_cast<size_t>(Ds));
    }
    if (db_residual_tls.size() != static_cast<size_t>(Ds)) {
        db_residual_tls.resize(static_cast<size_t>(Ds));
    }

    return compute_exact_distance_separated_scratch(
        vector_idx,
        query_rotated,
        query_residual_tls.data(),
        db_residual_tls.data());
}
} 
