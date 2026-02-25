#include "IndexIVFJHQScanner.h"

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__SSE2__)
#include <immintrin.h>
#endif

namespace faiss {

class JHQDecoder {
public:
    const uint8_t* code;
    mutable int current_bit_pos;

    JHQDecoder()
        : code(nullptr)
        , current_bit_pos(0)
    {
    }

    JHQDecoder(const uint8_t* code)
        : code(code)
        , current_bit_pos(0)
    {
    }

    inline void reset(const uint8_t* new_code)
    {
        code = new_code;
        current_bit_pos = 0;
    }

    inline uint32_t decode(int nbits) const
    {
        if (nbits == 8 && (current_bit_pos % 8) == 0) {
            uint32_t result = code[current_bit_pos / 8];
            current_bit_pos += 8;
            return result;
        }

        if (nbits <= 6) {
#ifdef __AVX512F__
            const int byte_offset = current_bit_pos / 8;
            const int bit_offset = current_bit_pos % 8;

            if (bit_offset + nbits <= 8) {
                const uint32_t byte_val = code[byte_offset];
                const uint32_t mask = (1U << nbits) - 1;
                const uint32_t result = (byte_val >> bit_offset) & mask;
                current_bit_pos += nbits;
                return result;
            } else {
                const uint64_t two_bytes = *reinterpret_cast<const uint16_t*>(&code[byte_offset]);
                const uint32_t mask = (1U << nbits) - 1;
                const uint32_t result = (two_bytes >> bit_offset) & mask;
                current_bit_pos += nbits;
                return result;
            }
#endif
        }

        uint32_t result = 0;
        int byte_offset = current_bit_pos / 8;
        int bit_offset = current_bit_pos % 8;

        if (bit_offset + nbits <= 8) {
            result = (code[byte_offset] >> bit_offset) & ((1U << nbits) - 1);
        } else {
            result = code[byte_offset] >> bit_offset;
            int remaining_bits = nbits - (8 - bit_offset);
            result |= (code[byte_offset + 1] & ((1U << remaining_bits) - 1))
                << (8 - bit_offset);
        }

        current_bit_pos += nbits;
        return result;
    }

    inline void decode_batch_same_bits(int nbits,
        int count,
        uint32_t* results) const
    {
#ifdef __AVX512F__
        if (nbits <= 8 && count >= 16 && (current_bit_pos % 8) == 0) {
            return decode_batch_byte_aligned_simd(nbits, count, results);
        }
#endif

        if (nbits == 8 && (current_bit_pos % 8) == 0) {
            const uint8_t* byte_ptr = code + (current_bit_pos / 8);
            for (int i = 0; i < count; ++i) {
                results[i] = byte_ptr[i];
            }
            current_bit_pos += count * 8;
        } else {
            for (int i = 0; i < count; ++i) {
                results[i] = decode(nbits);
            }
        }
    }

    inline void skip_bits(int nbits) const
    {
        current_bit_pos += nbits;
    }

private:
    void decode_batch_byte_aligned_simd(int nbits,
        int count,
        uint32_t* results) const
    {
#ifdef __AVX512F__
        const uint8_t* byte_ptr = code + (current_bit_pos / 8);
        const uint32_t mask = (1U << nbits) - 1;

        if (nbits == 8) {
            for (int i = 0; i + 15 < count; i += 16) {
                __m128i bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(&byte_ptr[i]));
                __m512i result = _mm512_cvtepu8_epi32(bytes);
                _mm512_storeu_si512(reinterpret_cast<__m512i*>(&results[i]),
                    result);
            }
            for (int i = (count / 16) * 16; i < count; ++i) {
                results[i] = byte_ptr[i];
            }
        } else if (nbits == 4) {
            for (int i = 0; i + 31 < count;
                i += 32) {
                __m128i bytes = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(&byte_ptr[i / 2]));
                __m512i expanded = _mm512_cvtepu8_epi32(bytes);
                __m512i low_nibbles = _mm512_and_si512(expanded, _mm512_set1_epi32(0x0F));
                __m512i high_nibbles = _mm512_and_si512(
                    _mm512_srli_epi32(expanded, 4), _mm512_set1_epi32(0x0F));
                for (int j = 0; j < 32 && i + j < count; ++j) {
                    uint8_t byte_val = byte_ptr[(i + j) / 2];
                    results[i + j] = (j % 2 == 0) ? (byte_val & 0x0F)
                                                  : ((byte_val >> 4) & 0x0F);
                }
            }
        }
        current_bit_pos += count * nbits;
#endif
    }
};

template<typename Traits>
inline void distance_four_codes_jhq(const int M,
    const int Ds,
    const float* primary_tables,
    const float* residual_tables,
    const std::vector<size_t>& residual_offsets,
    const uint8_t* code1,
    const uint8_t* code2,
    const uint8_t* code3,
    const uint8_t* code4,
    float& dist1,
    float& dist2,
    float& dist3,
    float& dist4)
{
    static constexpr int K0 = Traits::K0;
    static constexpr int NUM_LEVELS = Traits::num_levels;
    static constexpr int RESIDUAL_BITS = Traits::residual_bits;

    JHQDecoder decoders[4] = { { code1 }, { code2 }, { code3 }, { code4 } };
    dist1 = dist2 = dist3 = dist4 = 0.0f;

    for (int m = 0; m < M; ++m) {
        uint32_t centroid_ids[4];
        for (int i = 0; i < 4; ++i) {
            centroid_ids[i] = decoders[i].decode(Traits::primary_bits);
        }

        const float* table_base = primary_tables + m * K0;
        dist1 += table_base[centroid_ids[0]];
        dist2 += table_base[centroid_ids[1]];
        dist3 += table_base[centroid_ids[2]];
        dist4 += table_base[centroid_ids[3]];

        if constexpr (NUM_LEVELS > 1) {
            const int K_res = 1 << RESIDUAL_BITS;
            for (int level = 1; level < NUM_LEVELS; ++level) {
                const size_t level_offset = residual_offsets[level];

                for (int d = 0; d < Ds; ++d) {
                    uint32_t scalar_ids[4];
                    for (int i = 0; i < 4; ++i) {
                        scalar_ids[i] = decoders[i].decode(RESIDUAL_BITS);
                    }

                    size_t table_base_idx = level_offset + m * Ds * K_res + d * K_res;
                    const float* res_table = residual_tables + table_base_idx;

                    dist1 += res_table[scalar_ids[0]];
                    dist2 += res_table[scalar_ids[1]];
                    dist3 += res_table[scalar_ids[2]];
                    dist4 += res_table[scalar_ids[3]];
                }
            }
        } else {
            for (int i = 0; i < 4; ++i) {
                decoders[i].skip_bits(Traits::residual_bits_per_subspace);
            }
        }
    }
}

template<typename Traits>
inline void distance_sixteen_codes_jhq(
    const int M,
    const int Ds,
    const float* primary_tables,
    const float* residual_tables,
    const std::vector<size_t>& residual_offsets,
    const uint8_t* codes[16],
    float distances[16])
{
    static constexpr int K0 = Traits::K0;
    static constexpr int NUM_LEVELS = Traits::num_levels;
    static constexpr int RESIDUAL_BITS = Traits::residual_bits;

#ifdef __AVX512F__
    JHQDecoder decoders[16] = {
        JHQDecoder(codes[0]), JHQDecoder(codes[1]),
        JHQDecoder(codes[2]), JHQDecoder(codes[3]),
        JHQDecoder(codes[4]), JHQDecoder(codes[5]),
        JHQDecoder(codes[6]), JHQDecoder(codes[7]),
        JHQDecoder(codes[8]), JHQDecoder(codes[9]),
        JHQDecoder(codes[10]), JHQDecoder(codes[11]),
        JHQDecoder(codes[12]), JHQDecoder(codes[13]),
        JHQDecoder(codes[14]), JHQDecoder(codes[15])
    };

    __m512 total_distances = _mm512_setzero_ps();

    for (int m = 0; m < M; ++m) {
        alignas(64) uint32_t centroid_ids[16];
        for (int i = 0; i < 16; ++i) {
            centroid_ids[i] = decoders[i].decode(Traits::primary_bits);
        }

        __m512i indices = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(centroid_ids));
        __m512i table_base_indices = _mm512_add_epi32(_mm512_set1_epi32(m * K0), indices);

        __m512 primary_dists = _mm512_i32gather_ps(table_base_indices, primary_tables, 4);
        total_distances = _mm512_add_ps(total_distances, primary_dists);

        if constexpr (NUM_LEVELS > 1) {
            const int K_res = 1 << RESIDUAL_BITS;
            for (int level = 1; level < NUM_LEVELS; ++level) {
                const size_t level_offset = residual_offsets[level];

                for (int d = 0; d < Ds; ++d) {
                    alignas(64) uint32_t scalar_ids[16];
                    for (int i = 0; i < 16; ++i) {
                        scalar_ids[i] = decoders[i].decode(RESIDUAL_BITS);
                    }

                    __m512i scalar_indices = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(scalar_ids));
                    __m512i residual_base = _mm512_set1_epi32(
                        level_offset + m * Ds * K_res + d * K_res);
                    __m512i residual_indices = _mm512_add_epi32(residual_base, scalar_indices);

                    __m512 residual_dists = _mm512_i32gather_ps(
                        residual_indices, residual_tables, 4);
                    total_distances = _mm512_add_ps(total_distances, residual_dists);
                }
            }
        }
    }

    _mm512_storeu_ps(distances, total_distances);

#else
    for (int group = 0; group < 16; group += 4) {
        distance_four_codes_jhq<Traits>(
            M,
            Ds,
            primary_tables,
            residual_tables,
            residual_offsets,
            codes[group],
            codes[group + 1],
            codes[group + 2],
            codes[group + 3],
            distances[group],
            distances[group + 1],
            distances[group + 2],
            distances[group + 3]);
    }
#endif
}


IVFJHQScanner::IVFJHQScanner(
    const IndexIVFJHQ& idx,
    bool store_pairs,
    const IDSelector* sel,
    const IVFJHQSearchParameters* search_params)
    : InvertedListScanner(store_pairs, sel)
    , index(idx)
    , params(search_params)
    , query(nullptr)
    , use_early_termination(idx.use_early_termination)
    , compute_residuals(true)
    , oversampling_factor(idx.default_jhq_oversampling)
    , tables_computed(false)
    , workspace_capacity_lists(0)
    , workspace_capacity_primary_tables(0)
    , workspace_capacity_residual_tables(0)
    , is_reusable(false)
    , last_used(std::chrono::steady_clock::now())
    , reuse_count(0)
    , total_scans_performed(0)
    , total_codes_processed(0)
    , workspace_capacity_candidates(0)
{
    if (params) {
        if (params->jhq_oversampling_factor > 0) {
            oversampling_factor = params->jhq_oversampling_factor;
        }
        use_early_termination = params->use_early_termination;
        compute_residuals = params->compute_residuals;
    }

    code_size = idx.jhq.code_size;

    const size_t initial_list_capacity = 2048;
    const size_t initial_candidate_capacity = 1024;
    const size_t initial_primary_table_capacity = idx.jhq.M * 256;
    const size_t initial_residual_table_capacity = idx.jhq.M * idx.jhq.Ds * 64 * (idx.jhq.num_levels - 1);

    workspace_primary_distances.resize(initial_list_capacity);
    workspace_candidate_indices.reserve(initial_candidate_capacity);
    workspace_candidate_distances.reserve(initial_candidate_capacity);

    workspace_capacity_lists = initial_list_capacity;

    const int K0 = 1 << index.jhq.level_bits[0];
    jhq_primary_tables.reserve(initial_primary_table_capacity);
    jhq_residual_tables.reserve(initial_residual_table_capacity);
    workspace_capacity_primary_tables = initial_primary_table_capacity;
    workspace_capacity_residual_tables = initial_residual_table_capacity;

    workspace_candidate_distances_faiss.reserve(initial_candidate_capacity);
    workspace_candidate_indices_faiss.reserve(initial_candidate_capacity);
    workspace_capacity_candidates = initial_candidate_capacity;
}

void IVFJHQScanner::set_query(const float* query_vector)
{
    query = query_vector;
    tables_computed = false;
    query_already_rotated = false;

    if (!query_vector)
        return;

    const int K0 = 1 << index.jhq.level_bits[0];
    size_t primary_table_size = index.jhq.M * K0;
    size_t residual_table_size = 0;

    if (index.jhq.num_levels > 1) {
        for (int level = 1; level < index.jhq.num_levels; ++level) {
            residual_table_size += index.jhq.M * index.jhq.Ds * (1 << index.jhq.level_bits[level]);
        }
    }

    ensure_table_capacity(primary_table_size, residual_table_size);

    if (query_rotated.size() != index.d) {
        query_rotated.resize(index.d);
    }

    index.jhq.apply_jl_rotation(1, query, query_rotated.data());
    if (index.jhq.normalize_l2) {
        fvec_renorm_L2(static_cast<size_t>(index.d),
                       1,
                       query_rotated.data());
    }

    jhq_primary_tables.resize(primary_table_size);
    index.jhq.compute_primary_distance_tables_flat(
        query_rotated.data(), K0, jhq_primary_tables.data());

    if (index.jhq.num_levels > 1) {
        jhq_residual_tables.resize(residual_table_size);
        index.jhq.compute_residual_distance_tables(
            query_rotated.data(),
            jhq_residual_tables,
            jhq_residual_offsets);
    }

    tables_computed = true;
}

void IVFJHQScanner::set_rotated_query(const float* query_vector_rotated)
{
    query = query_vector_rotated;
    tables_computed = false;
    query_already_rotated = true;

    if (!query_vector_rotated)
        return;

    const int K0 = 1 << index.jhq.level_bits[0];
    size_t primary_table_size = index.jhq.M * K0;
    size_t residual_table_size = 0;

    if (index.jhq.num_levels > 1) {
        for (int level = 1; level < index.jhq.num_levels; ++level) {
            residual_table_size += index.jhq.M * index.jhq.Ds * (1 << index.jhq.level_bits[level]);
        }
    }

    ensure_table_capacity(primary_table_size, residual_table_size);

    if (query_rotated.size() != index.d) {
        query_rotated.resize(index.d);
    }
    std::memcpy(
        query_rotated.data(), query_vector_rotated, sizeof(float) * index.d);
    if (index.jhq.normalize_l2) {
        fvec_renorm_L2(static_cast<size_t>(index.d),
                       1,
                       query_rotated.data());
    }

    jhq_primary_tables.resize(primary_table_size);
    index.jhq.compute_primary_distance_tables_flat(
        query_rotated.data(), K0, jhq_primary_tables.data());

    if (index.jhq.num_levels > 1) {
        jhq_residual_tables.resize(residual_table_size);
        index.jhq.compute_residual_distance_tables(
            query_rotated.data(),
            jhq_residual_tables,
            jhq_residual_offsets);
    }

    tables_computed = true;
    query_already_rotated = false;
}

void IVFJHQScanner::set_list(idx_t list_no, float coarse_dis)
{
    (void)coarse_dis;
    this->list_no = list_no;

    if (has_separated_storage_available()) {
        current_list_pre_decoded = nullptr;
    } else {
        current_list_pre_decoded = index.get_pre_decoded_codes_for_list(list_no);
    }
}

const float* IVFJHQScanner::compute_primary_distances_for_list(
    size_t list_size,
    const uint8_t* codes) const
{
    ensure_workspace_capacity(list_size,  0);
    compute_primary_distances(list_size, codes);
    return workspace_primary_distances.data();
}

float IVFJHQScanner::refine_distance_from_primary_for_list(
    size_t offset_in_list,
    const uint8_t* packed_code,
    float primary_distance) const
{
    return refine_distance_from_primary(offset_in_list, packed_code, primary_distance);
}

float IVFJHQScanner::distance_to_code(const uint8_t* code) const
{
    if (!tables_computed) {
        FAISS_THROW_MSG("Tables not computed");
    }

    const size_t offset_in_list = (code - index.invlists->get_codes(list_no)) / index.jhq.code_size;

    if (has_separated_storage_available()) {
        return distance_to_code_separated_storage(offset_in_list);
    }

    if (current_list_pre_decoded) {
        return distance_to_code_pre_decoded(code);
    }

    return distance_to_code_with_bit_decoding(code);
}

bool IVFJHQScanner::has_separated_storage_available() const
{
    return index.jhq.has_pre_decoded_codes() && index.mapping_initialized.load(std::memory_order_acquire) && index.jhq.separated_codes_.is_initialized && !index.jhq.separated_codes_.empty();
}

float IVFJHQScanner::distance_to_code_separated_storage(size_t offset_in_list) const
{
    const idx_t global_vec_idx = index.get_global_vector_index(list_no, offset_in_list);

    if (global_vec_idx < 0) {
        
        
        const uint8_t* list_codes_base = index.invlists->get_codes(list_no);
        if (!list_codes_base) {
            return std::numeric_limits<float>::max();
        }
        const uint8_t* packed_code = list_codes_base + offset_in_list * code_size;
        return distance_to_code_with_bit_decoding(packed_code);
    }

    const uint8_t* primary_codes = index.jhq.separated_codes_.get_primary_codes(global_vec_idx);
    const int K0 = 1 << index.jhq.level_bits[0];

    float total_distance = 0.0f;

    for (int m = 0; m < index.jhq.M; ++m) {
        uint8_t centroid_id = primary_codes[m];
        total_distance += jhq_primary_tables[m * K0 + centroid_id];
    }

    if (compute_residuals && index.jhq.num_levels > 1) {
        const uint8_t* residual_codes = index.jhq.separated_codes_.get_residual_codes(global_vec_idx);
        total_distance += compute_residual_distance_separated_storage(residual_codes);
        if (!index.jhq.separated_codes_.cross_terms.empty() &&
            static_cast<size_t>(global_vec_idx) < index.jhq.separated_codes_.cross_terms.size()) {
            total_distance += index.jhq.separated_codes_.cross_terms[static_cast<size_t>(global_vec_idx)];
        } else {
            total_distance += jhq_internal::compute_cross_term_from_codes(
                index.jhq,
                primary_codes,
                residual_codes,
                index.jhq.separated_codes_.residual_subspace_stride,
                index.jhq.separated_codes_.residual_level_stride);
        }
    }

    return total_distance;
}

float IVFJHQScanner::refine_distance_from_primary(
    size_t offset_in_list,
    const uint8_t* packed_code,
    float primary_distance) const
{
    if (!compute_residuals || index.jhq.num_levels <= 1) {
        return primary_distance;
    }

    
    if (has_separated_storage_available()) {
        const idx_t global_vec_idx = index.get_global_vector_index(list_no, offset_in_list);
        if (global_vec_idx >= 0) {
            const uint8_t* residual_codes =
                index.jhq.separated_codes_.get_residual_codes(global_vec_idx);

            float refined = primary_distance +
                compute_residual_distance_separated_storage(residual_codes);

            if (!index.jhq.separated_codes_.cross_terms.empty() &&
                static_cast<size_t>(global_vec_idx) < index.jhq.separated_codes_.cross_terms.size()) {
                refined += index.jhq.separated_codes_.cross_terms[static_cast<size_t>(global_vec_idx)];
            } else {
                const uint8_t* primary_codes =
                    index.jhq.separated_codes_.get_primary_codes(global_vec_idx);
                refined += jhq_internal::compute_cross_term_from_codes(
                    index.jhq,
                    primary_codes,
                    residual_codes,
                    index.jhq.separated_codes_.residual_subspace_stride,
                    index.jhq.separated_codes_.residual_level_stride);
            }

            return refined;
        }
        
    }

    
    if (current_list_pre_decoded) {
        float refined = primary_distance +
            compute_residual_distance_pre_decoded(offset_in_list);
        const uint8_t* primary_codes =
            current_list_pre_decoded->get_primary_codes(offset_in_list);
        const uint8_t* residual_codes =
            current_list_pre_decoded->get_residual_codes(offset_in_list);
        if (!current_list_pre_decoded->cross_terms.empty() &&
            offset_in_list < current_list_pre_decoded->cross_terms.size()) {
            refined += current_list_pre_decoded->cross_terms[offset_in_list];
        } else {
            refined += jhq_internal::compute_cross_term_from_codes(
                index.jhq,
                primary_codes,
                residual_codes,
                current_list_pre_decoded->residual_subspace_stride,
                current_list_pre_decoded->residual_level_stride);
        }
        return refined;
    }

    
    return distance_to_code_with_bit_decoding(packed_code);
}

float IVFJHQScanner::compute_residual_distance_separated_storage(
    const uint8_t* residual_codes) const
{
    if (index.jhq.num_levels <= 1 || !residual_codes) {
        return 0.0f;
    }

    float residual_distance = 0.0f;
    size_t residual_offset = 0;

    for (int m = 0; m < index.jhq.M; ++m) {
        if (m + 1 < index.jhq.M) {
            const size_t next_offset = residual_offset + (index.jhq.num_levels - 1) * index.jhq.Ds;
            _mm_prefetch(reinterpret_cast<const char*>(&residual_codes[next_offset]), _MM_HINT_T0);
        }

        for (int level = 1; level < index.jhq.num_levels; ++level) {
            const size_t level_table_offset = jhq_residual_offsets[level];
            const int K_res = 1 << index.jhq.level_bits[level];
            const size_t table_base = level_table_offset + m * index.jhq.Ds * K_res;

#ifdef __AVX512F__
            if (index.jhq.Ds >= 32) {
                __m512 acc1 = _mm512_setzero_ps();
                __m512 acc2 = _mm512_setzero_ps();
                int d = 0;

                for (; d + 31 < index.jhq.Ds; d += 32) {
                    __m128i scalar_ids1_128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&residual_codes[residual_offset + d]));
                    __m512i scalar_ids1 = _mm512_cvtepu8_epi32(scalar_ids1_128);

                    __m512i table_indices1 = _mm512_set_epi32(
                        table_base + (d + 15) * K_res, table_base + (d + 14) * K_res,
                        table_base + (d + 13) * K_res, table_base + (d + 12) * K_res,
                        table_base + (d + 11) * K_res, table_base + (d + 10) * K_res,
                        table_base + (d + 9) * K_res, table_base + (d + 8) * K_res,
                        table_base + (d + 7) * K_res, table_base + (d + 6) * K_res,
                        table_base + (d + 5) * K_res, table_base + (d + 4) * K_res,
                        table_base + (d + 3) * K_res, table_base + (d + 2) * K_res,
                        table_base + (d + 1) * K_res, table_base + d * K_res);

                    table_indices1 = _mm512_add_epi32(table_indices1, scalar_ids1);
                    __m512 distances1 = _mm512_i32gather_ps(table_indices1,
                        jhq_residual_tables.data(), 4);
                    acc1 = _mm512_add_ps(acc1, distances1);

                    __m128i scalar_ids2_128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&residual_codes[residual_offset + d + 16]));
                    __m512i scalar_ids2 = _mm512_cvtepu8_epi32(scalar_ids2_128);

                    __m512i table_indices2 = _mm512_set_epi32(
                        table_base + (d + 31) * K_res, table_base + (d + 30) * K_res,
                        table_base + (d + 29) * K_res, table_base + (d + 28) * K_res,
                        table_base + (d + 27) * K_res, table_base + (d + 26) * K_res,
                        table_base + (d + 25) * K_res, table_base + (d + 24) * K_res,
                        table_base + (d + 23) * K_res, table_base + (d + 22) * K_res,
                        table_base + (d + 21) * K_res, table_base + (d + 20) * K_res,
                        table_base + (d + 19) * K_res, table_base + (d + 18) * K_res,
                        table_base + (d + 17) * K_res, table_base + (d + 16) * K_res);

                    table_indices2 = _mm512_add_epi32(table_indices2, scalar_ids2);
                    __m512 distances2 = _mm512_i32gather_ps(table_indices2,
                        jhq_residual_tables.data(), 4);
                    acc2 = _mm512_add_ps(acc2, distances2);
                }

                residual_distance += _mm512_reduce_add_ps(acc1) + _mm512_reduce_add_ps(acc2);

                for (; d < index.jhq.Ds; ++d) {
                    const uint8_t scalar_id = residual_codes[residual_offset + d];
                    const size_t table_idx = table_base + d * K_res + scalar_id;
                    residual_distance += jhq_residual_tables[table_idx];
                }
            } else if (index.jhq.Ds >= 16) {
                __m512 acc = _mm512_setzero_ps();
                int d = 0;

                for (; d + 15 < index.jhq.Ds; d += 16) {
                    __m128i scalar_ids_128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&residual_codes[residual_offset + d]));
                    __m512i scalar_ids = _mm512_cvtepu8_epi32(scalar_ids_128);

                    __m512i table_indices = _mm512_set_epi32(
                        table_base + (d + 15) * K_res, table_base + (d + 14) * K_res,
                        table_base + (d + 13) * K_res, table_base + (d + 12) * K_res,
                        table_base + (d + 11) * K_res, table_base + (d + 10) * K_res,
                        table_base + (d + 9) * K_res, table_base + (d + 8) * K_res,
                        table_base + (d + 7) * K_res, table_base + (d + 6) * K_res,
                        table_base + (d + 5) * K_res, table_base + (d + 4) * K_res,
                        table_base + (d + 3) * K_res, table_base + (d + 2) * K_res,
                        table_base + (d + 1) * K_res, table_base + d * K_res);

                    table_indices = _mm512_add_epi32(table_indices, scalar_ids);
                    __m512 distances = _mm512_i32gather_ps(table_indices,
                        jhq_residual_tables.data(), 4);
                    acc = _mm512_add_ps(acc, distances);
                }

                residual_distance += _mm512_reduce_add_ps(acc);

                for (; d < index.jhq.Ds; ++d) {
                    const uint8_t scalar_id = residual_codes[residual_offset + d];
                    const size_t table_idx = table_base + d * K_res + scalar_id;
                    residual_distance += jhq_residual_tables[table_idx];
                }
            } else
#elif defined(__AVX2__)
            if (index.jhq.Ds >= 16) {
                __m256 acc1 = _mm256_setzero_ps();
                __m256 acc2 = _mm256_setzero_ps();
                int d = 0;

                for (; d + 15 < index.jhq.Ds; d += 16) {
                    __m128i scalar_ids1_64 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(&residual_codes[residual_offset + d]));
                    __m256i scalar_ids1 = _mm256_cvtepu8_epi32(scalar_ids1_64);

                    __m256i table_indices1 = _mm256_set_epi32(
                        table_base + (d + 7) * K_res, table_base + (d + 6) * K_res,
                        table_base + (d + 5) * K_res, table_base + (d + 4) * K_res,
                        table_base + (d + 3) * K_res, table_base + (d + 2) * K_res,
                        table_base + (d + 1) * K_res, table_base + d * K_res);

                    table_indices1 = _mm256_add_epi32(table_indices1, scalar_ids1);
                    __m256 distances1 = _mm256_i32gather_ps(
                        jhq_residual_tables.data(), table_indices1, 4);
                    acc1 = _mm256_add_ps(acc1, distances1);

                    __m128i scalar_ids2_64 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(&residual_codes[residual_offset + d + 8]));
                    __m256i scalar_ids2 = _mm256_cvtepu8_epi32(scalar_ids2_64);

                    __m256i table_indices2 = _mm256_set_epi32(
                        table_base + (d + 15) * K_res, table_base + (d + 14) * K_res,
                        table_base + (d + 13) * K_res, table_base + (d + 12) * K_res,
                        table_base + (d + 11) * K_res, table_base + (d + 10) * K_res,
                        table_base + (d + 9) * K_res, table_base + (d + 8) * K_res);

                    table_indices2 = _mm256_add_epi32(table_indices2, scalar_ids2);
                    __m256 distances2 = _mm256_i32gather_ps(
                        jhq_residual_tables.data(), table_indices2, 4);
                    acc2 = _mm256_add_ps(acc2, distances2);
                }

                __m256 combined = _mm256_add_ps(acc1, acc2);
                __m128 sum_128 = _mm_add_ps(_mm256_castps256_ps128(combined),
                    _mm256_extractf128_ps(combined, 1));
                __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                __m128 sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
                residual_distance += _mm_cvtss_f32(sum_32);

                for (; d < index.jhq.Ds; ++d) {
                    const uint8_t scalar_id = residual_codes[residual_offset + d];
                    const size_t table_idx = table_base + d * K_res + scalar_id;
                    residual_distance += jhq_residual_tables[table_idx];
                }
            } else
#endif
            {
                int d = 0;
                for (; d + 3 < index.jhq.Ds; d += 4) {
                    float sum = jhq_residual_tables[table_base + d * K_res + residual_codes[residual_offset + d]] + jhq_residual_tables[table_base + (d + 1) * K_res + residual_codes[residual_offset + d + 1]] + jhq_residual_tables[table_base + (d + 2) * K_res + residual_codes[residual_offset + d + 2]] + jhq_residual_tables[table_base + (d + 3) * K_res + residual_codes[residual_offset + d + 3]];
                    residual_distance += sum;
                }
                for (; d < index.jhq.Ds; ++d) {
                    const uint8_t scalar_id = residual_codes[residual_offset + d];
                    const size_t table_idx = table_base + d * K_res + scalar_id;
                    residual_distance += jhq_residual_tables[table_idx];
                }
            }

            residual_offset += index.jhq.Ds;
        }
    }

    return residual_distance;
}

float IVFJHQScanner::distance_to_code_pre_decoded(
    const uint8_t* code) const
{
    const size_t codes_per_list = index.invlists->list_size(list_no);
    const size_t vector_idx_in_list = (code - index.invlists->get_codes(list_no)) / index.jhq.code_size;

    if (vector_idx_in_list >= codes_per_list) {
        return distance_to_code_with_bit_decoding(code);
    }

    const uint8_t* primary_codes = current_list_pre_decoded->get_primary_codes(vector_idx_in_list);

    float total_distance = 0.0f;
    const int K0 = 1 << index.jhq.level_bits[0];

    for (int m = 0; m < index.jhq.M; ++m) {
        const uint8_t centroid_id = primary_codes[m];
        total_distance += jhq_primary_tables[m * K0 + centroid_id];
    }

    if (compute_residuals && index.jhq.num_levels > 1) {
        total_distance += compute_residual_distance_pre_decoded(vector_idx_in_list);
        if (!current_list_pre_decoded->cross_terms.empty() &&
            vector_idx_in_list < current_list_pre_decoded->cross_terms.size()) {
            total_distance += current_list_pre_decoded->cross_terms[vector_idx_in_list];
        } else {
            const uint8_t* residual_codes =
                current_list_pre_decoded->get_residual_codes(vector_idx_in_list);
            total_distance += jhq_internal::compute_cross_term_from_codes(
                index.jhq,
                primary_codes,
                residual_codes,
                current_list_pre_decoded->residual_subspace_stride,
                current_list_pre_decoded->residual_level_stride);
        }
    }

    return total_distance;
}

float IVFJHQScanner::compute_residual_distance_pre_decoded(
    size_t vector_idx_in_list) const
{
    const uint8_t* residual_codes = current_list_pre_decoded->get_residual_codes(vector_idx_in_list);
    const ProductQuantizer* residual_pq = index.jhq.get_residual_product_quantizer();
    float residual_distance = 0.0f;
    size_t residual_offset = 0;

    if (residual_pq && index.jhq.num_levels == 2) {
        const int K_res = 1 << index.jhq.level_bits[1];
        const size_t level_offset = jhq_residual_offsets[1];

        for (int m = 0; m < index.jhq.M; ++m) {
            for (int d = 0; d < index.jhq.Ds; ++d) {
                const uint8_t scalar_id = residual_codes[residual_offset++];
                const size_t table_idx = level_offset +
                    (size_t)m * index.jhq.Ds * K_res + (size_t)d * K_res + scalar_id;
                residual_distance += jhq_residual_tables[table_idx];
            }
        }

        return residual_distance;
    }

    for (int m = 0; m < index.jhq.M; ++m) {
        if (m + 1 < index.jhq.M) {
            const size_t next_offset = residual_offset + (index.jhq.num_levels - 1) * index.jhq.Ds;
            if (next_offset < current_list_pre_decoded->residual_codes.size()) {
                _mm_prefetch(reinterpret_cast<const char*>(&residual_codes[next_offset]), _MM_HINT_T0);
            }
        }

        for (int level = 1; level < index.jhq.num_levels; ++level) {
            const size_t level_table_offset = jhq_residual_offsets[level];
            const int K_res = 1 << index.jhq.level_bits[level];
            const size_t table_base = level_table_offset + m * index.jhq.Ds * K_res;

#ifdef __AVX512F__
            if (index.jhq.Ds >= 16) {
                __m512 acc = _mm512_setzero_ps();
                int d = 0;

                for (; d + 15 < index.jhq.Ds; d += 16) {
                    __m128i scalar_ids_128 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(&residual_codes[residual_offset + d]));
                    __m512i scalar_ids = _mm512_cvtepu8_epi32(scalar_ids_128);

                    __m512i table_indices = _mm512_set_epi32(
                        table_base + (d + 15) * K_res, table_base + (d + 14) * K_res,
                        table_base + (d + 13) * K_res, table_base + (d + 12) * K_res,
                        table_base + (d + 11) * K_res, table_base + (d + 10) * K_res,
                        table_base + (d + 9) * K_res, table_base + (d + 8) * K_res,
                        table_base + (d + 7) * K_res, table_base + (d + 6) * K_res,
                        table_base + (d + 5) * K_res, table_base + (d + 4) * K_res,
                        table_base + (d + 3) * K_res, table_base + (d + 2) * K_res,
                        table_base + (d + 1) * K_res, table_base + d * K_res);

                    table_indices = _mm512_add_epi32(table_indices, scalar_ids);

                    __m512 distances = _mm512_i32gather_ps(
                        table_indices, jhq_residual_tables.data(), 4);
                    acc = _mm512_add_ps(acc, distances);
                }

                residual_distance += _mm512_reduce_add_ps(acc);

                for (; d < index.jhq.Ds; ++d) {
                    const uint8_t scalar_id = residual_codes[residual_offset + d];
                    const size_t table_idx = table_base + d * K_res + scalar_id;
                    residual_distance += jhq_residual_tables[table_idx];
                }
            } else
#elif defined(__AVX2__)
            if (index.jhq.Ds >= 8) {
                __m256 acc = _mm256_setzero_ps();
                int d = 0;

                for (; d + 7 < index.jhq.Ds; d += 8) {
                    __m128i scalar_ids_64 = _mm_loadl_epi64(
                        reinterpret_cast<const __m128i*>(&residual_codes[residual_offset + d]));
                    __m256i scalar_ids = _mm256_cvtepu8_epi32(scalar_ids_64);

                    __m256i table_indices = _mm256_set_epi32(
                        table_base + (d + 7) * K_res, table_base + (d + 6) * K_res,
                        table_base + (d + 5) * K_res, table_base + (d + 4) * K_res,
                        table_base + (d + 3) * K_res, table_base + (d + 2) * K_res,
                        table_base + (d + 1) * K_res, table_base + d * K_res);

                    table_indices = _mm256_add_epi32(table_indices, scalar_ids);

                    __m256 distances = _mm256_i32gather_ps(
                        jhq_residual_tables.data(), table_indices, 4);
                    acc = _mm256_add_ps(acc, distances);
                }

                __m128 sum_128 = _mm_add_ps(_mm256_castps256_ps128(acc),
                    _mm256_extractf128_ps(acc, 1));
                __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                __m128 sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
                residual_distance += _mm_cvtss_f32(sum_32);

                for (; d < index.jhq.Ds; ++d) {
                    const uint8_t scalar_id = residual_codes[residual_offset + d];
                    const size_t table_idx = table_base + d * K_res + scalar_id;
                    residual_distance += jhq_residual_tables[table_idx];
                }
            } else
#endif
            {
                for (int d = 0; d < index.jhq.Ds; ++d) {
                    const uint8_t scalar_id = residual_codes[residual_offset + d];
                    const size_t table_idx = table_base + d * K_res + scalar_id;
                    residual_distance += jhq_residual_tables[table_idx];
                }
            }

            residual_offset += index.jhq.Ds;
        }
    }

    return residual_distance;
}

float IVFJHQScanner::distance_to_code_with_bit_decoding(
    const uint8_t* code) const
{
    JHQDecoder decoder(code);
    float total_distance = 0.0f;
    const int K0 = 1 << index.jhq.level_bits[0];
    float cross_term = 0.0f;
    std::vector<float> residual_buffer(
        compute_residuals && index.jhq.num_levels > 1 ? index.jhq.Ds : 0);

    const uint32_t primary_limit = static_cast<uint32_t>(K0 - 1);

    for (int m = 0; m < index.jhq.M; ++m) {
        uint32_t centroid_id = decoder.decode(index.jhq.level_bits[0]);
        centroid_id = std::min(centroid_id, primary_limit);
        total_distance += jhq_primary_tables[m * K0 + centroid_id];

        if (compute_residuals && index.jhq.num_levels > 1) {
            std::fill(residual_buffer.begin(), residual_buffer.end(), 0.0f);

            const float* centroids = index.jhq.get_primary_centroids_ptr(m);
            const uint32_t safe_centroid = std::min<uint32_t>(
                centroid_id, static_cast<uint32_t>(K0 - 1));
            const float* centroid_ptr =
                centroids + static_cast<size_t>(safe_centroid) * index.jhq.Ds;

            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const uint32_t residual_limit = static_cast<uint32_t>(K_res - 1);
                const size_t level_offset = jhq_residual_offsets[level];
                const size_t table_base = level_offset + static_cast<size_t>(m) * index.jhq.Ds * K_res;
                const int codebook_size = index.jhq.scalar_codebook_ksub(level);
                const float* scalar_codebook = index.jhq.get_scalar_codebook_ptr(m, level);

                for (int d = 0; d < index.jhq.Ds; ++d) {
                    uint32_t scalar_id = decoder.decode(index.jhq.level_bits[level]);
                    scalar_id = std::min(scalar_id, residual_limit);
                    const size_t table_idx = table_base + static_cast<size_t>(d) * K_res + scalar_id;
                    total_distance += jhq_residual_tables[table_idx];

                    if (codebook_size > 0) {
                        const uint32_t safe_scalar = std::min<uint32_t>(
                            scalar_id,
                            static_cast<uint32_t>(codebook_size - 1));
                        residual_buffer[d] += scalar_codebook[safe_scalar];
                    }
                }
            }

            for (int d = 0; d < index.jhq.Ds; ++d) {
                cross_term += centroid_ptr[d] * residual_buffer[d];
            }
        } else {
            decoder.skip_bits(index.jhq.residual_bits_per_subspace);
        }
    }

    if (compute_residuals && index.jhq.num_levels > 1) {
        total_distance += 2.0f * cross_term;
    }

    return total_distance;
}

void IVFJHQScanner::distance_four_codes(const uint8_t* code1,
    const uint8_t* code2,
    const uint8_t* code3,
    const uint8_t* code4,
    float& dist1,
    float& dist2,
    float& dist3,
    float& dist4) const
{
    _mm_prefetch(
        reinterpret_cast<const char*>(code1 + code_size), _MM_HINT_T0);
    _mm_prefetch(
        reinterpret_cast<const char*>(code2 + code_size), _MM_HINT_T0);
    _mm_prefetch(
        reinterpret_cast<const char*>(code3 + code_size), _MM_HINT_T0);
    _mm_prefetch(
        reinterpret_cast<const char*>(code4 + code_size), _MM_HINT_T0);

    if (current_list_pre_decoded) {
        distance_four_codes_pre_decoded(
            code1, code2, code3, code4, dist1, dist2, dist3, dist4);
        return;
    }

    distance_four_codes_with_bit_decoding(
        code1, code2, code3, code4, dist1, dist2, dist3, dist4);
}

void IVFJHQScanner::distance_four_codes_pre_decoded(const uint8_t* code1,
    const uint8_t* code2,
    const uint8_t* code3,
    const uint8_t* code4,
    float& dist1,
    float& dist2,
    float& dist3,
    float& dist4) const
{
    const uint8_t* list_codes_base = index.invlists->get_codes(list_no);
    const size_t code_size = index.jhq.code_size;

    const size_t idx1 = (code1 - list_codes_base) / code_size;
    const size_t idx2 = (code2 - list_codes_base) / code_size;
    const size_t idx3 = (code3 - list_codes_base) / code_size;
    const size_t idx4 = (code4 - list_codes_base) / code_size;

    const uint8_t* primary1 = current_list_pre_decoded->get_primary_codes(idx1);
    const uint8_t* primary2 = current_list_pre_decoded->get_primary_codes(idx2);
    const uint8_t* primary3 = current_list_pre_decoded->get_primary_codes(idx3);
    const uint8_t* primary4 = current_list_pre_decoded->get_primary_codes(idx4);

    const int K0 = 1 << index.jhq.level_bits[0];
    dist1 = dist2 = dist3 = dist4 = 0.0f;

#ifdef __AVX512F__
    if (index.jhq.M >= 32) {
        __m128 distances = _mm_setzero_ps();

        for (int m = 0; m + 31 < index.jhq.M; m += 32) {
            if (m + 64 < index.jhq.M) {
                _mm_prefetch(&primary1[m + 32], _MM_HINT_T0);
                _mm_prefetch(&primary2[m + 32], _MM_HINT_T0);
                _mm_prefetch(&primary3[m + 32], _MM_HINT_T0);
                _mm_prefetch(&primary4[m + 32], _MM_HINT_T0);
            }

            for (int block = 0; block < 2; ++block) {
                int offset = m + block * 16;

                __m128i codes1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&primary1[offset]));
                __m128i codes2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&primary2[offset]));
                __m128i codes3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&primary3[offset]));
                __m128i codes4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&primary4[offset]));

                __m512i codes1_512 = _mm512_cvtepu8_epi32(codes1);
                __m512i codes2_512 = _mm512_cvtepu8_epi32(codes2);
                __m512i codes3_512 = _mm512_cvtepu8_epi32(codes3);
                __m512i codes4_512 = _mm512_cvtepu8_epi32(codes4);

                __m512i subspace_offsets = _mm512_set_epi32(
                    (offset + 15) * K0, (offset + 14) * K0, (offset + 13) * K0, (offset + 12) * K0,
                    (offset + 11) * K0, (offset + 10) * K0, (offset + 9) * K0, (offset + 8) * K0,
                    (offset + 7) * K0, (offset + 6) * K0, (offset + 5) * K0, (offset + 4) * K0,
                    (offset + 3) * K0, (offset + 2) * K0, (offset + 1) * K0, offset * K0);

                codes1_512 = _mm512_add_epi32(codes1_512, subspace_offsets);
                codes2_512 = _mm512_add_epi32(codes2_512, subspace_offsets);
                codes3_512 = _mm512_add_epi32(codes3_512, subspace_offsets);
                codes4_512 = _mm512_add_epi32(codes4_512, subspace_offsets);

                __m512 dists1 = _mm512_i32gather_ps(codes1_512, jhq_primary_tables.data(), 4);
                __m512 dists2 = _mm512_i32gather_ps(codes2_512, jhq_primary_tables.data(), 4);
                __m512 dists3 = _mm512_i32gather_ps(codes3_512, jhq_primary_tables.data(), 4);
                __m512 dists4 = _mm512_i32gather_ps(codes4_512, jhq_primary_tables.data(), 4);

                __m128 sum1 = _mm_set1_ps(_mm512_reduce_add_ps(dists1));
                __m128 sum2 = _mm_set1_ps(_mm512_reduce_add_ps(dists2));
                __m128 sum3 = _mm_set1_ps(_mm512_reduce_add_ps(dists3));
                __m128 sum4 = _mm_set1_ps(_mm512_reduce_add_ps(dists4));

                __m128 combined = _mm_unpacklo_ps(
                    _mm_unpacklo_ps(sum1, sum3),
                    _mm_unpacklo_ps(sum2, sum4));
                distances = _mm_add_ps(distances, combined);
            }
        }

        alignas(16) float result[4];
        _mm_store_ps(result, distances);
        dist1 += result[0];
        dist2 += result[1];
        dist3 += result[2];
        dist4 += result[3];

        for (int m = (index.jhq.M / 32) * 32; m < index.jhq.M; ++m) {
            const float* table_base = jhq_primary_tables.data() + m * K0;
            dist1 += table_base[primary1[m]];
            dist2 += table_base[primary2[m]];
            dist3 += table_base[primary3[m]];
            dist4 += table_base[primary4[m]];
        }
    } else
#endif
    {
        for (int m = 0; m < index.jhq.M; ++m) {
            const float* table_base = jhq_primary_tables.data() + m * K0;
            dist1 += table_base[primary1[m]];
            dist2 += table_base[primary2[m]];
            dist3 += table_base[primary3[m]];
            dist4 += table_base[primary4[m]];
        }
    }

    if (compute_residuals && index.jhq.num_levels > 1) {
        dist1 += compute_residual_distance_pre_decoded(idx1);
        dist2 += compute_residual_distance_pre_decoded(idx2);
        dist3 += compute_residual_distance_pre_decoded(idx3);
        dist4 += compute_residual_distance_pre_decoded(idx4);

        const uint8_t* residual1 = current_list_pre_decoded->get_residual_codes(idx1);
        const uint8_t* residual2 = current_list_pre_decoded->get_residual_codes(idx2);
        const uint8_t* residual3 = current_list_pre_decoded->get_residual_codes(idx3);
        const uint8_t* residual4 = current_list_pre_decoded->get_residual_codes(idx4);

        dist1 += jhq_internal::compute_cross_term_from_codes(
            index.jhq,
            primary1,
            residual1,
            current_list_pre_decoded->residual_subspace_stride,
            current_list_pre_decoded->residual_level_stride);
        dist2 += jhq_internal::compute_cross_term_from_codes(
            index.jhq,
            primary2,
            residual2,
            current_list_pre_decoded->residual_subspace_stride,
            current_list_pre_decoded->residual_level_stride);
        dist3 += jhq_internal::compute_cross_term_from_codes(
            index.jhq,
            primary3,
            residual3,
            current_list_pre_decoded->residual_subspace_stride,
            current_list_pre_decoded->residual_level_stride);
        dist4 += jhq_internal::compute_cross_term_from_codes(
            index.jhq,
            primary4,
            residual4,
            current_list_pre_decoded->residual_subspace_stride,
            current_list_pre_decoded->residual_level_stride);
    }
}

void IVFJHQScanner::distance_four_codes_with_bit_decoding(
    const uint8_t* code1,
    const uint8_t* code2,
    const uint8_t* code3,
    const uint8_t* code4,
    float& dist1,
    float& dist2,
    float& dist3,
    float& dist4) const
{
    JHQDecoder decoders[4] = { { code1 }, { code2 }, { code3 }, { code4 } };

    for (int m = 0; m < index.jhq.M; ++m) {
        if (m + 1 < index.jhq.M) {
            const int K0 = 1 << index.jhq.level_bits[0];
            const float* next_table_base = jhq_primary_tables.data() + (m + 1) * K0;
            _mm_prefetch(
                reinterpret_cast<const char*>(next_table_base),
                _MM_HINT_T1);
        }
    }

#ifdef __AVX2__
    __m128 distances = _mm_setzero_ps();
    float cross_terms[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const int K0 = 1 << index.jhq.level_bits[0];

    for (int m = 0; m < index.jhq.M; ++m) {
        const float* table_base = jhq_primary_tables.data() + m * K0;

        const uint32_t id0 = std::min(
            decoders[0].decode(index.jhq.level_bits[0]),
            static_cast<uint32_t>(K0 - 1));
        const uint32_t id1 = std::min(
            decoders[1].decode(index.jhq.level_bits[0]),
            static_cast<uint32_t>(K0 - 1));
        const uint32_t id2 = std::min(
            decoders[2].decode(index.jhq.level_bits[0]),
            static_cast<uint32_t>(K0 - 1));
        const uint32_t id3 = std::min(
            decoders[3].decode(index.jhq.level_bits[0]),
            static_cast<uint32_t>(K0 - 1));

        const float* centroids = index.jhq.get_primary_centroids_ptr(m);
        const size_t centroid_block = static_cast<size_t>(index.jhq.Ds);
        const float* centroid_ptr0 = centroids + static_cast<size_t>(id0) * centroid_block;
        const float* centroid_ptr1 = centroids + static_cast<size_t>(id1) * centroid_block;
        const float* centroid_ptr2 = centroids + static_cast<size_t>(id2) * centroid_block;
        const float* centroid_ptr3 = centroids + static_cast<size_t>(id3) * centroid_block;

        __m128i indices = _mm_set_epi32(id3, id2, id1, id0);
        __m128 primary_dists = _mm_i32gather_ps(table_base, indices, 4);

        distances = _mm_add_ps(distances, primary_dists);

        if (compute_residuals && index.jhq.num_levels > 1) {
            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const size_t level_offset = jhq_residual_offsets[level];
                const float* scalar_codebook = index.jhq.get_scalar_codebook_ptr(m, level);
                const int codebook_size = index.jhq.scalar_codebook_ksub(level);

                for (int d = 0; d < index.jhq.Ds; ++d) {
                    alignas(16) uint32_t scalar_ids[4];
                    for (int i = 0; i < 4; ++i) {
                        scalar_ids[i] = decoders[i].decode(index.jhq.level_bits[level]);
                    }

                    size_t table_base_idx = level_offset + m * index.jhq.Ds * K_res + d * K_res;
                    const float* res_table = jhq_residual_tables.data() + table_base_idx;

                    __m128i res_indices = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(scalar_ids));
                    __m128i max_res_idx = _mm_set1_epi32(K_res - 1);
                    res_indices = _mm_min_epi32(res_indices, max_res_idx);

                    __m128 residual_dists = _mm_i32gather_ps(res_table, res_indices, 4);
                    distances = _mm_add_ps(distances, residual_dists);

                    if (codebook_size > 0) {
                        const float centroid_vals[4] = {
                            centroid_ptr0[d],
                            centroid_ptr1[d],
                            centroid_ptr2[d],
                            centroid_ptr3[d]
                        };
                        const uint32_t safe_idx0 = std::min(
                            scalar_ids[0], static_cast<uint32_t>(codebook_size - 1));
                        const uint32_t safe_idx1 = std::min(
                            scalar_ids[1], static_cast<uint32_t>(codebook_size - 1));
                        const uint32_t safe_idx2 = std::min(
                            scalar_ids[2], static_cast<uint32_t>(codebook_size - 1));
                        const uint32_t safe_idx3 = std::min(
                            scalar_ids[3], static_cast<uint32_t>(codebook_size - 1));

                        cross_terms[0] += centroid_vals[0] * scalar_codebook[safe_idx0];
                        cross_terms[1] += centroid_vals[1] * scalar_codebook[safe_idx1];
                        cross_terms[2] += centroid_vals[2] * scalar_codebook[safe_idx2];
                        cross_terms[3] += centroid_vals[3] * scalar_codebook[safe_idx3];
                    }
                }
            }
        } else {
            int skip_bits = index.jhq.residual_bits_per_subspace;
            for (int i = 0; i < 4; ++i) {
                decoders[i].skip_bits(skip_bits);
            }
        }
    }

    alignas(16) float result[4];
    _mm_store_ps(result, distances);
    dist1 = result[0] + (compute_residuals && index.jhq.num_levels > 1 ? 2.0f * cross_terms[0] : 0.0f);
    dist2 = result[1] + (compute_residuals && index.jhq.num_levels > 1 ? 2.0f * cross_terms[1] : 0.0f);
    dist3 = result[2] + (compute_residuals && index.jhq.num_levels > 1 ? 2.0f * cross_terms[2] : 0.0f);
    dist4 = result[3] + (compute_residuals && index.jhq.num_levels > 1 ? 2.0f * cross_terms[3] : 0.0f);

#else
    dist1 = dist2 = dist3 = dist4 = 0.0f;
    float cross_terms[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const int K0 = 1 << index.jhq.level_bits[0];

    for (int m = 0; m < index.jhq.M; ++m) {
        uint32_t centroid_ids[4];
        for (int i = 0; i < 4; ++i) {
            centroid_ids[i] = decoders[i].decode(index.jhq.level_bits[0]);
        }

        const float* table_base = jhq_primary_tables.data() + m * K0;
        dist1 += table_base[std::min(
            centroid_ids[0], static_cast<uint32_t>(K0 - 1))];
        dist2 += table_base[std::min(
            centroid_ids[1], static_cast<uint32_t>(K0 - 1))];
        dist3 += table_base[std::min(
            centroid_ids[2], static_cast<uint32_t>(K0 - 1))];
        dist4 += table_base[std::min(
            centroid_ids[3], static_cast<uint32_t>(K0 - 1))];

        const float* centroids = index.jhq.get_primary_centroids_ptr(m);
        const size_t centroid_block = static_cast<size_t>(index.jhq.Ds);
        const float* centroid_ptrs[4];
        for (int i = 0; i < 4; ++i) {
            const uint32_t safe_centroid =
                std::min(centroid_ids[i], static_cast<uint32_t>(K0 - 1));
            centroid_ptrs[i] =
                centroids + static_cast<size_t>(safe_centroid) * centroid_block;
        }

        if (compute_residuals && index.jhq.num_levels > 1) {
            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const size_t level_offset = jhq_residual_offsets[level];
                const float* scalar_codebook = index.jhq.get_scalar_codebook_ptr(m, level);
                const int codebook_size = index.jhq.scalar_codebook_ksub(level);

                for (int d = 0; d < index.jhq.Ds; ++d) {
                    uint32_t scalar_ids[4];
                    for (int i = 0; i < 4; ++i) {
                        scalar_ids[i] = decoders[i].decode(index.jhq.level_bits[level]);
                    }

                    size_t table_base_idx = level_offset + m * index.jhq.Ds * K_res + d * K_res;
                    const float* res_table = jhq_residual_tables.data() + table_base_idx;

                    dist1 += res_table[std::min(
                        scalar_ids[0], static_cast<uint32_t>(K_res - 1))];
                    dist2 += res_table[std::min(
                        scalar_ids[1], static_cast<uint32_t>(K_res - 1))];
                    dist3 += res_table[std::min(
                        scalar_ids[2], static_cast<uint32_t>(K_res - 1))];
                    dist4 += res_table[std::min(
                        scalar_ids[3], static_cast<uint32_t>(K_res - 1))];

                    if (codebook_size > 0) {
                        for (int i = 0; i < 4; ++i) {
                            const uint32_t safe_idx = std::min(
                                scalar_ids[i], static_cast<uint32_t>(codebook_size - 1));
                            cross_terms[i] += centroid_ptrs[i][d] * scalar_codebook[safe_idx];
                        }
                    }
                }
            }
        } else {
            int skip_bits = index.jhq.residual_bits_per_subspace;
            for (int i = 0; i < 4; ++i) {
                decoders[i].skip_bits(skip_bits);
            }
        }
    }

    if (compute_residuals && index.jhq.num_levels > 1) {
        dist1 += 2.0f * cross_terms[0];
        dist2 += 2.0f * cross_terms[1];
        dist3 += 2.0f * cross_terms[2];
        dist4 += 2.0f * cross_terms[3];
    }
#endif
}

void IVFJHQScanner::distance_sixteen_codes(
    const uint8_t* codes[16],
    float distances[16]) const
{
    JHQDecoder decoders[16] = {
        JHQDecoder(codes[0]), JHQDecoder(codes[1]),
        JHQDecoder(codes[2]), JHQDecoder(codes[3]),
        JHQDecoder(codes[4]), JHQDecoder(codes[5]),
        JHQDecoder(codes[6]), JHQDecoder(codes[7]),
        JHQDecoder(codes[8]), JHQDecoder(codes[9]),
        JHQDecoder(codes[10]), JHQDecoder(codes[11]),
        JHQDecoder(codes[12]), JHQDecoder(codes[13]),
        JHQDecoder(codes[14]), JHQDecoder(codes[15])
    };

#ifdef __AVX512F__
    __m512 total_distances = _mm512_setzero_ps();
    float cross_terms[16] = { 0.0f };
    const int K0 = 1 << index.jhq.level_bits[0];

    for (int m = 0; m < index.jhq.M; ++m) {
        alignas(64) uint32_t centroid_ids[16];
        for (int i = 0; i < 16; ++i) {
            centroid_ids[i] = decoders[i].decode(index.jhq.level_bits[0]);
        }

        const float* table_base = jhq_primary_tables.data() + m * K0;
        const float* centroids = index.jhq.get_primary_centroids_ptr(m);
        const size_t centroid_block = static_cast<size_t>(index.jhq.Ds);
        const float* centroid_ptrs[16];
        for (int i = 0; i < 16; ++i) {
            centroid_ptrs[i] =
                centroids + static_cast<size_t>(centroid_ids[i]) * centroid_block;
        }
        __m512i indices = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(centroid_ids));
        __m512 primary_dists = _mm512_i32gather_ps(indices, table_base, 4);
        total_distances = _mm512_add_ps(total_distances, primary_dists);

        if (compute_residuals && index.jhq.num_levels > 1) {
            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const size_t level_offset = jhq_residual_offsets[level];
                const float* scalar_codebook = index.jhq.get_scalar_codebook_ptr(m, level);
                const int codebook_size = index.jhq.scalar_codebook_ksub(level);

                for (int d = 0; d < index.jhq.Ds; ++d) {
                    alignas(64) uint32_t scalar_ids[16];
                    for (int i = 0; i < 16; ++i) {
                        scalar_ids[i] = decoders[i].decode(index.jhq.level_bits[level]);
                    }

                    size_t table_base_idx = level_offset + m * index.jhq.Ds * K_res + d * K_res;
                    const float* res_table = jhq_residual_tables.data() + table_base_idx;

                    __m512i res_indices = _mm512_loadu_si512(
                        reinterpret_cast<const __m512i*>(scalar_ids));
                    __m512 residual_dists = _mm512_i32gather_ps(res_indices, res_table, 4);
                    total_distances = _mm512_add_ps(total_distances, residual_dists);

                    if (codebook_size > 0) {
                        for (int i = 0; i < 16; ++i) {
                            const uint32_t safe_idx = std::min(
                                scalar_ids[i], static_cast<uint32_t>(codebook_size - 1));
                            cross_terms[i] += centroid_ptrs[i][d] * scalar_codebook[safe_idx];
                        }
                    }
                }
            }
        } else {
            int skip_bits = index.jhq.residual_bits_per_subspace;
            for (int i = 0; i < 16; ++i) {
                decoders[i].skip_bits(skip_bits);
            }
        }
    }

    _mm512_storeu_ps(distances, total_distances);
    if (compute_residuals && index.jhq.num_levels > 1) {
        for (int i = 0; i < 16; ++i) {
            distances[i] += 2.0f * cross_terms[i];
        }
    }

#elif defined(__AVX2__)
    const int K0 = 1 << index.jhq.level_bits[0];
    float cross_terms[16] = { 0.0f };

    for (int i = 0; i < 16; ++i) {
        distances[i] = 0.0f;
    }

    for (int m = 0; m < index.jhq.M; ++m) {
        const float* centroids = index.jhq.get_primary_centroids_ptr(m);
        const size_t centroid_block = static_cast<size_t>(index.jhq.Ds);
        const float* centroid_ptrs[16];

        {
            alignas(32) uint32_t centroid_ids[8];
            for (int i = 0; i < 8; ++i) {
                centroid_ids[i] = decoders[i].decode(index.jhq.level_bits[0]);
                centroid_ptrs[i] = centroids + static_cast<size_t>(centroid_ids[i]) * centroid_block;
            }

            const float* table_base = jhq_primary_tables.data() + m * K0;
            __m256i indices = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(centroid_ids));
            __m256 primary_dists = _mm256_i32gather_ps(table_base, indices, 4);

            __m256 current_dists = _mm256_loadu_ps(&distances[0]);
            current_dists = _mm256_add_ps(current_dists, primary_dists);
            _mm256_storeu_ps(&distances[0], current_dists);
        }

        {
            alignas(32) uint32_t centroid_ids[8];
            for (int i = 0; i < 8; ++i) {
                centroid_ids[i] = decoders[i + 8].decode(index.jhq.level_bits[0]);
                centroid_ptrs[i + 8] = centroids + static_cast<size_t>(centroid_ids[i]) * centroid_block;
            }

            const float* table_base = jhq_primary_tables.data() + m * K0;
            __m256i indices = _mm256_loadu_si256(
                reinterpret_cast<const __m256i*>(centroid_ids));
            __m256 primary_dists = _mm256_i32gather_ps(table_base, indices, 4);

            __m256 current_dists = _mm256_loadu_ps(&distances[8]);
            current_dists = _mm256_add_ps(current_dists, primary_dists);
            _mm256_storeu_ps(&distances[8], current_dists);
        }

        if (compute_residuals && index.jhq.num_levels > 1) {
            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const size_t level_offset = jhq_residual_offsets[level];
                const float* scalar_codebook = index.jhq.get_scalar_codebook_ptr(m, level);
                const int codebook_size = index.jhq.scalar_codebook_ksub(level);

                for (int d = 0; d < index.jhq.Ds; ++d) {
                    size_t table_base_idx = level_offset + m * index.jhq.Ds * K_res + d * K_res;
                    const float* res_table = jhq_residual_tables.data() + table_base_idx;

                    {
                        alignas(32) uint32_t scalar_ids[8];
                        for (int i = 0; i < 8; ++i) {
                            scalar_ids[i] = decoders[i].decode(
                                index.jhq.level_bits[level]);
                        }

                        __m256i res_indices = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(scalar_ids));
                        __m256 residual_dists = _mm256_i32gather_ps(res_table, res_indices, 4);

                        __m256 current_dists = _mm256_loadu_ps(&distances[0]);
                        current_dists = _mm256_add_ps(current_dists, residual_dists);
                        _mm256_storeu_ps(&distances[0], current_dists);

                        if (codebook_size > 0) {
                            for (int i = 0; i < 8; ++i) {
                                const uint32_t safe_idx = std::min(
                                    scalar_ids[i], static_cast<uint32_t>(codebook_size - 1));
                                cross_terms[i] += centroid_ptrs[i][d] * scalar_codebook[safe_idx];
                            }
                        }
                    }

                    {
                        alignas(32) uint32_t scalar_ids[8];
                        for (int i = 0; i < 8; ++i) {
                            scalar_ids[i] = decoders[i + 8].decode(
                                index.jhq.level_bits[level]);
                        }

                        __m256i res_indices = _mm256_loadu_si256(
                            reinterpret_cast<const __m256i*>(scalar_ids));
                        __m256 residual_dists = _mm256_i32gather_ps(res_table, res_indices, 4);

                        __m256 current_dists = _mm256_loadu_ps(&distances[8]);
                        current_dists = _mm256_add_ps(current_dists, residual_dists);
                        _mm256_storeu_ps(&distances[8], current_dists);

                        if (codebook_size > 0) {
                            for (int i = 0; i < 8; ++i) {
                                const uint32_t safe_idx = std::min(
                                    scalar_ids[i], static_cast<uint32_t>(codebook_size - 1));
                                cross_terms[i + 8] += centroid_ptrs[i + 8][d] * scalar_codebook[safe_idx];
                            }
                        }
                    }
                }
            }
        } else {
            int skip_bits = index.jhq.residual_bits_per_subspace;
            for (int i = 0; i < 16; ++i) {
                decoders[i].skip_bits(skip_bits);
            }
        }
    }

    if (compute_residuals && index.jhq.num_levels > 1) {
        for (int i = 0; i < 16; ++i) {
            distances[i] += 2.0f * cross_terms[i];
        }
    }

#else
    distance_four_codes(
        codes[0],
        codes[1],
        codes[2],
        codes[3],
        distances[0],
        distances[1],
        distances[2],
        distances[3]);
    distance_four_codes(
        codes[4],
        codes[5],
        codes[6],
        codes[7],
        distances[4],
        distances[5],
        distances[6],
        distances[7]);
    distance_four_codes(
        codes[8],
        codes[9],
        codes[10],
        codes[11],
        distances[8],
        distances[9],
        distances[10],
        distances[11]);
    distance_four_codes(
        codes[12],
        codes[13],
        codes[14],
        codes[15],
        distances[12],
        distances[13],
        distances[14],
        distances[15]);
#endif
}

size_t IVFJHQScanner::scan_codes(size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float* simi,
    idx_t* idxi,
    size_t k) const
{
    FAISS_THROW_IF_NOT_MSG(query != nullptr, "Query must be set before scanning");
    FAISS_THROW_IF_NOT_MSG(list_no >= 0, "List must be set before scanning");

    total_scans_performed++;
    total_codes_processed += list_size;

    const float effective_oversampling = std::max(1.0f, oversampling_factor);
    const size_t target_candidates = static_cast<size_t>(std::min<double>(
        static_cast<double>(list_size),
        std::ceil(static_cast<double>(k) * static_cast<double>(effective_oversampling))));

    optimized_workspace.ensure_capacity(list_size, target_candidates);

    const bool multi_level_residual_scan =
        compute_residuals && (index.jhq.num_levels > 1);
    const bool should_use_two_stage_refine =
        use_early_termination &&
        multi_level_residual_scan &&
        (k < list_size) &&
        (target_candidates < list_size);

    if (should_use_two_stage_refine) {
        const size_t n_candidates = std::max(k, target_candidates);
        ensure_workspace_capacity(list_size, n_candidates);
        return scan_codes_early_termination(
            list_size, codes, ids, simi, idxi, k);
    }

    ensure_workspace_capacity(list_size, 0);
    return scan_codes_exhaustive(
        list_size, codes, ids, simi, idxi, k);
}

size_t IVFJHQScanner::scan_codes_exhaustive(
    size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float* simi,
    idx_t* idxi,
    size_t k) const
{
    size_t nup = 0;
    bool is_max_heap = (index.metric_type == METRIC_INNER_PRODUCT);

    heap_batch_buffer_.clear();

    if (!is_max_heap) {
        
        
        const bool primary_only_distance =
            (!compute_residuals) || (index.jhq.num_levels <= 1);
        if (primary_only_distance) {
            compute_primary_distances(list_size, codes);
            quantize_primary_distances(list_size);
            return scan_codes_exhaustive_l2_gated(
                list_size, codes, ids, simi, idxi, k);
        }
    }

    if (k <= 8) {
        return scan_codes_small_k_simd(list_size, codes, ids, simi, idxi, k);
    }

#ifdef __AVX512F__
    int counter = 0;
    size_t saved_j[16];

    for (size_t j = 0; j < list_size; j++) {
        saved_j[counter] = j;
        counter++;

        if (counter == 16) {
            const uint8_t* code_ptrs[16];
            for (int i = 0; i < 16; ++i) {
                code_ptrs[i] = codes + saved_j[i] * code_size;
            }

            float distances[16];
            distance_sixteen_codes(code_ptrs, distances);

            update_heap_sixteen_candidates(
                saved_j, distances, is_max_heap, k, simi, idxi, ids, nup);
            counter = 0;
        }
    }

    for (int i = 0; i < counter; i++) {
        float dis = distance_to_code(codes + saved_j[i] * code_size);
        update_heap_single_candidate(
            saved_j[i], dis, is_max_heap, k, simi, idxi, ids, nup);
    }

#else
    int counter = 0;
    size_t saved_j[4] = { 0, 0, 0, 0 };

   for (size_t j = 0; j < list_size; j++) {
        saved_j[counter] = j;
        counter++;

        if (counter == 4) {
            float distance_0, distance_1, distance_2, distance_3;

            distance_four_codes(codes + saved_j[0] * code_size,
                codes + saved_j[1] * code_size,
                codes + saved_j[2] * code_size,
                codes + saved_j[3] * code_size,
                distance_0,
                distance_1,
                distance_2,
                distance_3);

            update_heap_four_candidates(saved_j,
                distance_0,
                distance_1,
                distance_2,
                distance_3,
                is_max_heap,
                k,
                simi,
                idxi,
                ids,
                nup);
            counter = 0;
        }
    }

    for (int i = 0; i < counter; i++) {
        float dis = distance_to_code(codes + saved_j[i] * code_size);
        update_heap_single_candidate(
            saved_j[i], dis, is_max_heap, k, simi, idxi, ids, nup);
    }
#endif

    flush_heap_batch(is_max_heap, k, simi, idxi, nup);
    return nup;
}

size_t IVFJHQScanner::scan_codes_exhaustive_l2_gated(
    size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float* simi,
    idx_t* idxi,
    size_t k) const
{
    size_t nup = 0;
    const bool has_quantized = workspace_primary_distances_quantized.size() == list_size;
    const float* primary_distances = workspace_primary_distances.data();

    for (size_t j = 0; j < list_size; ++j) {
        if (!passes_selector(j, ids)) {
            continue;
        }

        float primary_dist = has_quantized
            ? reconstruct_primary_distance(workspace_primary_distances_quantized[j])
            : primary_distances[j];

        if (primary_dist >= simi[0]) {
            continue;
        }

        float dis = distance_to_code(codes + j * code_size);
        if (dis >= simi[0]) {
            continue;
        }

        idx_t id = get_candidate_id(j, ids);
        heap_replace_top<CMax<float, idx_t>>(k, simi, idxi, dis, id);
        nup++;
    }

    return nup;
}

size_t IVFJHQScanner::scan_codes_early_termination(
    size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float* simi,
    idx_t* idxi,
    size_t k) const
{
    const float effective_oversampling = std::max(1.0f, oversampling_factor);

    size_t n_candidates = static_cast<size_t>(std::min<double>(
        static_cast<double>(list_size),
        std::ceil(static_cast<double>(k) * static_cast<double>(effective_oversampling))));
    n_candidates = std::max(n_candidates, k);
    n_candidates = std::min(n_candidates, list_size);

    float* primary_distances_ptr = workspace_primary_distances.data();

    workspace_candidate_indices.clear();
    workspace_candidate_indices.reserve(n_candidates);

    compute_primary_distances(list_size, codes);

    select_top_candidates(primary_distances_ptr, list_size, n_candidates);

    size_t nup = 0;
    bool is_max_heap = (index.metric_type == METRIC_INNER_PRODUCT);

    for (size_t i = 0; i < n_candidates; ++i) {
        const size_t j = workspace_candidate_indices[i];
        if (!passes_selector(j, ids)) {
            continue;
        }

        const uint8_t* packed_code = codes + j * code_size;
        const float dis =
            refine_distance_from_primary(j, packed_code, primary_distances_ptr[j]);
        const idx_t id = get_candidate_id(j, ids);

        if (is_max_heap) {
            if (CMin<float, idx_t>::cmp(simi[0], dis)) {
                faiss::heap_replace_top<CMin<float, idx_t>>(
                    k, simi, idxi, dis, id);
                nup++;
            }
        } else {
            if (CMax<float, idx_t>::cmp(simi[0], dis)) {
                faiss::heap_replace_top<CMax<float, idx_t>>(
                    k, simi, idxi, dis, id);
                nup++;
            }
        }
    }

    return nup;
}

size_t IVFJHQScanner::scan_codes_small_k_simd(size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float* simi,
    idx_t* idxi,
    size_t k) const
{
    size_t nup = 0;
    bool is_max_heap = (index.metric_type == METRIC_INNER_PRODUCT);

    if (k <= 4) {
        return scan_codes_k4_unrolled(list_size, codes, ids, simi, idxi, k);
    } else {
        return scan_codes_k8_unrolled(list_size, codes, ids, simi, idxi, k);
    }
}

size_t IVFJHQScanner::scan_codes_k4_unrolled(size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float* simi,
    idx_t* idxi,
    size_t k) const
{
    size_t nup = 0;
    bool is_max_heap = (index.metric_type == METRIC_INNER_PRODUCT);

    for (size_t j = 0; j + 3 < list_size; j += 4) {
        float distances[4];
        distance_four_codes(codes + j * code_size,
            codes + (j + 1) * code_size,
            codes + (j + 2) * code_size,
            codes + (j + 3) * code_size,
            distances[0],
            distances[1],
            distances[2],
            distances[3]);

        for (int i = 0; i < 4; ++i) {
            size_t idx_offset = j + i;
            if (!passes_selector(idx_offset, ids)) {
                continue;
            }

            idx_t id = get_candidate_id(idx_offset, ids);

            if (is_max_heap) {
                if (CMin<float, idx_t>::cmp(simi[0], distances[i])) {
                    faiss::heap_replace_top<CMin<float, idx_t>>(
                        k, simi, idxi, distances[i], id);
                    nup++;
                }
            } else {
                if (CMax<float, idx_t>::cmp(simi[0], distances[i])) {
                    faiss::heap_replace_top<CMax<float, idx_t>>(
                        k, simi, idxi, distances[i], id);
                    nup++;
                }
            }
        }
    }

    return nup;
}

size_t IVFJHQScanner::scan_codes_k8_unrolled(size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float* simi,
    idx_t* idxi,
    size_t k) const
{
    size_t nup = 0;
    bool is_max_heap = (index.metric_type == METRIC_INNER_PRODUCT);

    for (size_t j = 0; j + 7 < list_size; j += 8) {
        float distances1[4];
        float distances2[4];

        distance_four_codes(codes + j * code_size,
            codes + (j + 1) * code_size,
            codes + (j + 2) * code_size,
            codes + (j + 3) * code_size,
            distances1[0],
            distances1[1],
            distances1[2],
            distances1[3]);

        distance_four_codes(codes + (j + 4) * code_size,
            codes + (j + 5) * code_size,
            codes + (j + 6) * code_size,
            codes + (j + 7) * code_size,
            distances2[0],
            distances2[1],
            distances2[2],
            distances2[3]);

        for (int i = 0; i < 4; ++i) {
            size_t idx_offset = j + i;
            if (!passes_selector(idx_offset, ids)) {
                continue;
            }

            idx_t id = get_candidate_id(idx_offset, ids);

            if (is_max_heap) {
                if (CMin<float, idx_t>::cmp(simi[0], distances1[i])) {
                    faiss::heap_replace_top<CMin<float, idx_t>>(
                        k, simi, idxi, distances1[i], id);
                    nup++;
                }
            } else {
                if (CMax<float, idx_t>::cmp(simi[0], distances1[i])) {
                    faiss::heap_replace_top<CMax<float, idx_t>>(
                        k, simi, idxi, distances1[i], id);
                    nup++;
                }
            }
        }

        for (int i = 0; i < 4; ++i) {
            size_t idx_offset = j + 4 + i;
            if (!passes_selector(idx_offset, ids)) {
                continue;
            }

            idx_t id = get_candidate_id(idx_offset, ids);

            if (is_max_heap) {
                if (CMin<float, idx_t>::cmp(simi[0], distances2[i])) {
                    faiss::heap_replace_top<CMin<float, idx_t>>(
                        k, simi, idxi, distances2[i], id);
                    nup++;
                }
            } else {
                if (CMax<float, idx_t>::cmp(simi[0], distances2[i])) {
                    faiss::heap_replace_top<CMax<float, idx_t>>(
                        k, simi, idxi, distances2[i], id);
                    nup++;
                }
            }
        }
    }

    for (size_t j = (list_size / 8) * 8; j < list_size; ++j) {
        if (!passes_selector(j, ids)) {
            continue;
        }

        float distance = distance_to_code(codes + j * code_size);
        idx_t id = get_candidate_id(j, ids);

        if (is_max_heap) {
            if (CMin<float, idx_t>::cmp(simi[0], distance)) {
                faiss::heap_replace_top<CMin<float, idx_t>>(
                    k, simi, idxi, distance, id);
                nup++;
            }
        } else {
            if (CMax<float, idx_t>::cmp(simi[0], distance)) {
                faiss::heap_replace_top<CMax<float, idx_t>>(
                    k, simi, idxi, distance, id);
                nup++;
            }
        }
    }

    return nup;
}

void IVFJHQScanner::scan_codes_range(size_t list_size,
    const uint8_t* codes,
    const idx_t* ids,
    float radius,
    RangeQueryResult& res) const
{
    for (size_t j = 0; j < list_size; j++) {
        float dis = distance_to_code(codes);

        bool within_range = (index.metric_type == METRIC_L2) ? (dis <= radius) : (dis >= radius);

        if (within_range) {
            idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            res.add(dis, id);
        }

        codes += code_size;
    }
}

void IVFJHQScanner::compute_primary_distances(
    size_t list_size,
    const uint8_t* codes) const
{

    const int K0 = 1 << index.jhq.level_bits[0];
    float* distances = workspace_primary_distances.data();

    
    
    
    
    
    
    
    
    const size_t level0_bits = static_cast<size_t>(index.jhq.level_bits[0]);
    const size_t bits_per_subspace = level0_bits + index.jhq.residual_bits_per_subspace;
    const bool separated_storage_available = has_separated_storage_available();
    const bool can_byte_stride_primary =
        codes != nullptr &&
        level0_bits == 8 &&
        bits_per_subspace % 8 == 0 &&
        bits_per_subspace / 8 * static_cast<size_t>(index.jhq.M) == index.jhq.code_size;
    if (can_byte_stride_primary) {
        const size_t bytes_per_subspace = bits_per_subspace / 8;
        if (!separated_storage_available || bytes_per_subspace == 1) {
        const size_t code_stride = index.jhq.code_size;

#pragma omp parallel for if (list_size > 10000)
        for (size_t j = 0; j < list_size; ++j) {
            const uint8_t* code = codes + j * code_stride;
            float total_distance = 0.0f;
            for (int m = 0; m < index.jhq.M; ++m) {
                const uint8_t centroid_id = code[static_cast<size_t>(m) * bytes_per_subspace];
                total_distance += jhq_primary_tables[static_cast<size_t>(m) * K0 + centroid_id];
            }
            distances[j] = total_distance;
        }
        return;
        }
    }

    if (separated_storage_available) {
        compute_primary_distances_separated_storage(list_size, distances);
    } else {
        if (current_list_pre_decoded) {
            compute_primary_distances_pre_decoded(list_size, distances);
        } else {
            compute_primary_distances_with_bit_decoding(list_size, codes, distances);
        }
    }
}

void IVFJHQScanner::compute_primary_distances_pre_decoded(
    size_t list_size,
    float* distances) const
{
    const int K0 = 1 << index.jhq.level_bits[0];

#pragma omp parallel for if (list_size > 10000)
    for (size_t i = 0; i < list_size; ++i) {
        const uint8_t* primary_codes = current_list_pre_decoded->get_primary_codes(i);

        float total_distance = 0.0f;

        for (int m = 0; m < index.jhq.M; ++m) {
            const uint8_t centroid_id = primary_codes[m];
            total_distance += jhq_primary_tables[m * K0 + centroid_id];
        }

        distances[i] = total_distance;
    }
}

void IVFJHQScanner::compute_primary_distances_with_bit_decoding(
    size_t list_size,
    const uint8_t* codes,
    float* distances) const
{
    const int K0 = 1 << index.jhq.level_bits[0];

    const bool use_separated_storage = index.jhq.has_pre_decoded_codes() && index.mapping_initialized.load(std::memory_order_acquire);

    if (use_separated_storage) {
        compute_primary_distances_separated_storage(list_size, distances);
    } else {
        compute_primary_distances_packed_codes(list_size, codes, distances);
    }
}

void IVFJHQScanner::compute_primary_distances_separated_storage(
    size_t list_size,
    float* distances) const {

    const int K0 = 1 << index.jhq.level_bits[0];
    constexpr size_t CACHE_BATCH_SIZE = 128; 

#pragma omp parallel for schedule(static) if (list_size > 2000) 
    for (size_t batch_start = 0; batch_start < list_size; batch_start += CACHE_BATCH_SIZE) {
        const size_t batch_end = std::min(list_size, batch_start + CACHE_BATCH_SIZE);

        
        for (size_t i = batch_start; i < batch_end; i += 16) {
            const size_t prefetch_end = std::min(batch_end, i + 64);
            for (size_t j = i; j < prefetch_end; j += 8) {
                if (j < list_size) {
                    const idx_t global_idx = index.get_global_vector_index(list_no, j);
                    if (global_idx >= 0) {
                        const uint8_t* primary_codes = index.jhq.separated_codes_.get_primary_codes(global_idx);
                        _mm_prefetch(primary_codes, _MM_HINT_T0);
                        _mm_prefetch(primary_codes + 32, _MM_HINT_T0); 
                    }
                }
            }
        }

        
        for (size_t offset_in_list = batch_start; offset_in_list < batch_end; ++offset_in_list) {
            const idx_t global_vec_idx = index.get_global_vector_index(list_no, offset_in_list);

            if (global_vec_idx >= 0) {
                const uint8_t* primary_codes = index.jhq.separated_codes_.get_primary_codes(global_vec_idx);
                float total_distance = 0.0f;

#ifdef __AVX512F__
                if (index.jhq.M >= 64) { 
                    __m512 acc1 = _mm512_setzero_ps();
                    __m512 acc2 = _mm512_setzero_ps();
                    __m512 acc3 = _mm512_setzero_ps();
                    __m512 acc4 = _mm512_setzero_ps();

                    int m = 0;
                    for (; m + 63 < index.jhq.M; m += 64) {
                        
                        if (m + 128 < index.jhq.M) {
                            _mm_prefetch(&primary_codes[m + 64], _MM_HINT_T0);
                            _mm_prefetch(&jhq_primary_tables[(m + 64) * K0], _MM_HINT_T1);
                        }

                        
                        for (int group = 0; group < 4; ++group) {
                            const int group_offset = m + group * 16;

                            __m128i codes_128 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(&primary_codes[group_offset]));
                            __m512i codes_512 = _mm512_cvtepu8_epi32(codes_128);

                            
                            __m512i offsets = _mm512_mullo_epi32(
                                _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
                                _mm512_set1_epi32(K0));
                            __m512i base_offset = _mm512_set1_epi32(group_offset * K0);
                            offsets = _mm512_add_epi32(offsets, base_offset);

                            __m512i indices = _mm512_add_epi32(codes_512, offsets);
                            __m512 dists = _mm512_i32gather_ps(indices, jhq_primary_tables.data(), 4);

                            
                            switch(group) {
                                case 0: acc1 = _mm512_add_ps(acc1, dists); break;
                                case 1: acc2 = _mm512_add_ps(acc2, dists); break;
                                case 2: acc3 = _mm512_add_ps(acc3, dists); break;
                                case 3: acc4 = _mm512_add_ps(acc4, dists); break;
                            }
                        }
                    }

                    
                    __m512 combined1 = _mm512_add_ps(acc1, acc2);
                    __m512 combined2 = _mm512_add_ps(acc3, acc4);
                    __m512 final_acc = _mm512_add_ps(combined1, combined2);
                    total_distance += _mm512_reduce_add_ps(final_acc);

                    
                    for (; m < index.jhq.M; ++m) {
                        uint32_t code_val = primary_codes[m];
                        total_distance += jhq_primary_tables[m * K0 + code_val];
                    }

                } else if (index.jhq.M >= 32) {
                    
                    __m512 acc1 = _mm512_setzero_ps();
                    __m512 acc2 = _mm512_setzero_ps();

                    int m = 0;
                    for (; m + 31 < index.jhq.M; m += 32) {
                        
                        __m128i codes1_128 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(&primary_codes[m]));
                        __m512i codes1_512 = _mm512_cvtepu8_epi32(codes1_128);

                        __m512i offsets1 = _mm512_set_epi32(
                            (m+15)*K0, (m+14)*K0, (m+13)*K0, (m+12)*K0,
                            (m+11)*K0, (m+10)*K0, (m+9)*K0, (m+8)*K0,
                            (m+7)*K0, (m+6)*K0, (m+5)*K0, (m+4)*K0,
                            (m+3)*K0, (m+2)*K0, (m+1)*K0, m*K0);

                        __m512i indices1 = _mm512_add_epi32(codes1_512, offsets1);
                        __m512 dists1 = _mm512_i32gather_ps(indices1, jhq_primary_tables.data(), 4);
                        acc1 = _mm512_add_ps(acc1, dists1);

                        
                        __m128i codes2_128 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(&primary_codes[m + 16]));
                        __m512i codes2_512 = _mm512_cvtepu8_epi32(codes2_128);

                        __m512i offsets2 = _mm512_set_epi32(
                            (m+31)*K0, (m+30)*K0, (m+29)*K0, (m+28)*K0,
                            (m+27)*K0, (m+26)*K0, (m+25)*K0, (m+24)*K0,
                            (m+23)*K0, (m+22)*K0, (m+21)*K0, (m+20)*K0,
                            (m+19)*K0, (m+18)*K0, (m+17)*K0, (m+16)*K0);

                        __m512i indices2 = _mm512_add_epi32(codes2_512, offsets2);
                        __m512 dists2 = _mm512_i32gather_ps(indices2, jhq_primary_tables.data(), 4);
                        acc2 = _mm512_add_ps(acc2, dists2);
                    }

                    total_distance += _mm512_reduce_add_ps(acc1) + _mm512_reduce_add_ps(acc2);

                    
                    for (; m < index.jhq.M; ++m) {
                        uint32_t code_val = primary_codes[m];
                        total_distance += jhq_primary_tables[m * K0 + code_val];
                    }

                } else if (index.jhq.M >= 16) {
                    
                    __m512 acc = _mm512_setzero_ps();
                    int m = 0;

                    for (; m + 15 < index.jhq.M; m += 16) {
                        __m128i codes_128 = _mm_loadu_si128(
                            reinterpret_cast<const __m128i*>(&primary_codes[m]));
                        __m512i codes_512 = _mm512_cvtepu8_epi32(codes_128);

                        __m512i subspace_offsets = _mm512_set_epi32(
                            (m+15)*K0, (m+14)*K0, (m+13)*K0, (m+12)*K0,
                            (m+11)*K0, (m+10)*K0, (m+9)*K0, (m+8)*K0,
                            (m+7)*K0, (m+6)*K0, (m+5)*K0, (m+4)*K0,
                            (m+3)*K0, (m+2)*K0, (m+1)*K0, m*K0);

                        __m512i indices = _mm512_add_epi32(codes_512, subspace_offsets);
                        __m512 dists = _mm512_i32gather_ps(indices, jhq_primary_tables.data(), 4);
                        acc = _mm512_add_ps(acc, dists);
                    }

                    total_distance += _mm512_reduce_add_ps(acc);

                    for (; m < index.jhq.M; ++m) {
                        uint32_t code_val = primary_codes[m];
                        total_distance += jhq_primary_tables[m * K0 + code_val];
                    }
                } else
#elif defined(__AVX2__)
                if (index.jhq.M >= 32) {
                    
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    __m256 acc4 = _mm256_setzero_ps();

                    int m = 0;
                    for (; m + 31 < index.jhq.M; m += 32) {
                        
                        for (int group = 0; group < 4; ++group) {
                            const int group_offset = m + group * 8;

                            __m128i codes_64 = _mm_loadl_epi64(
                                reinterpret_cast<const __m128i*>(&primary_codes[group_offset]));
                            __m256i codes_256 = _mm256_cvtepu8_epi32(codes_64);

                            __m256i offsets = _mm256_set_epi32(
                                (group_offset+7)*K0, (group_offset+6)*K0,
                                (group_offset+5)*K0, (group_offset+4)*K0,
                                (group_offset+3)*K0, (group_offset+2)*K0,
                                (group_offset+1)*K0, group_offset*K0);

                            __m256i indices = _mm256_add_epi32(codes_256, offsets);
                            __m256 dists = _mm256_i32gather_ps(jhq_primary_tables.data(), indices, 4);

                            switch(group) {
                                case 0: acc1 = _mm256_add_ps(acc1, dists); break;
                                case 1: acc2 = _mm256_add_ps(acc2, dists); break;
                                case 2: acc3 = _mm256_add_ps(acc3, dists); break;
                                case 3: acc4 = _mm256_add_ps(acc4, dists); break;
                            }
                        }
                    }

                    
                    __m256 combined1 = _mm256_add_ps(acc1, acc2);
                    __m256 combined2 = _mm256_add_ps(acc3, acc4);
                    __m256 final_combined = _mm256_add_ps(combined1, combined2);

                    __m128 sum_128 = _mm_add_ps(_mm256_castps256_ps128(final_combined),
                        _mm256_extractf128_ps(final_combined, 1));
                    __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                    __m128 sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
                    total_distance += _mm_cvtss_f32(sum_32);

                    for (; m < index.jhq.M; ++m) {
                        uint32_t code_val = primary_codes[m];
                        total_distance += jhq_primary_tables[m * K0 + code_val];
                    }

                } else if (index.jhq.M >= 16) {
                    
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    int m = 0;

                    for (; m + 15 < index.jhq.M; m += 16) {
                        
                        __m128i codes1_64 = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(&primary_codes[m]));
                        __m256i codes1_256 = _mm256_cvtepu8_epi32(codes1_64);

                        __m256i offsets1 = _mm256_set_epi32(
                            (m+7)*K0, (m+6)*K0, (m+5)*K0, (m+4)*K0,
                            (m+3)*K0, (m+2)*K0, (m+1)*K0, m*K0);

                        __m256i indices1 = _mm256_add_epi32(codes1_256, offsets1);
                        __m256 dists1 = _mm256_i32gather_ps(jhq_primary_tables.data(), indices1, 4);
                        acc1 = _mm256_add_ps(acc1, dists1);

                        
                        __m128i codes2_64 = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(&primary_codes[m + 8]));
                        __m256i codes2_256 = _mm256_cvtepu8_epi32(codes2_64);

                        __m256i offsets2 = _mm256_set_epi32(
                            (m+15)*K0, (m+14)*K0, (m+13)*K0, (m+12)*K0,
                            (m+11)*K0, (m+10)*K0, (m+9)*K0, (m+8)*K0);

                        __m256i indices2 = _mm256_add_epi32(codes2_256, offsets2);
                        __m256 dists2 = _mm256_i32gather_ps(jhq_primary_tables.data(), indices2, 4);
                        acc2 = _mm256_add_ps(acc2, dists2);
                    }

                    __m256 combined = _mm256_add_ps(acc1, acc2);
                    __m128 sum_128 = _mm_add_ps(_mm256_castps256_ps128(combined),
                        _mm256_extractf128_ps(combined, 1));
                    __m128 sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
                    __m128 sum_32 = _mm_add_ss(sum_64, _mm_movehdup_ps(sum_64));
                    total_distance += _mm_cvtss_f32(sum_32);

                    for (; m < index.jhq.M; ++m) {
                        uint32_t code_val = primary_codes[m];
                        total_distance += jhq_primary_tables[m * K0 + code_val];
                    }
                } else
#endif
                {
                    
                    int m = 0;
                    for (; m + 7 < index.jhq.M; m += 8) {
                        
                        float sum =
                            jhq_primary_tables[m*K0 + primary_codes[m]] +
                            jhq_primary_tables[(m+1)*K0 + primary_codes[m+1]] +
                            jhq_primary_tables[(m+2)*K0 + primary_codes[m+2]] +
                            jhq_primary_tables[(m+3)*K0 + primary_codes[m+3]] +
                            jhq_primary_tables[(m+4)*K0 + primary_codes[m+4]] +
                            jhq_primary_tables[(m+5)*K0 + primary_codes[m+5]] +
                            jhq_primary_tables[(m+6)*K0 + primary_codes[m+6]] +
                            jhq_primary_tables[(m+7)*K0 + primary_codes[m+7]];
                        total_distance += sum;
                    }
                    for (; m < index.jhq.M; ++m) {
                        total_distance += jhq_primary_tables[m * K0 + primary_codes[m]];
                    }
                }

                distances[offset_in_list] = total_distance;
            } else {
                
                const uint8_t* list_codes_base = index.invlists->get_codes(list_no);
                if (!list_codes_base) {
                    distances[offset_in_list] = std::numeric_limits<float>::max();
                    continue;
                }

                const uint8_t* packed_code =
                    list_codes_base + offset_in_list * index.jhq.code_size;
                JHQDecoder decoder(packed_code);
                float total_distance = 0.0f;
                const size_t skip_bits = index.jhq.residual_bits_per_subspace;

                for (int m = 0; m < index.jhq.M; ++m) {
                    uint32_t centroid_id = decoder.decode(index.jhq.level_bits[0]);
                    centroid_id = std::min<uint32_t>(
                        centroid_id,
                        static_cast<uint32_t>(std::max(0, K0 - 1)));
                    total_distance += jhq_primary_tables[m * K0 + centroid_id];
                    decoder.skip_bits(static_cast<int>(skip_bits));
                }

                distances[offset_in_list] = total_distance;
            }
        }
    }
}

void IVFJHQScanner::compute_primary_distances_packed_codes(
    size_t list_size,
    const uint8_t* codes,
    float* distances) const
{
    const int K0 = 1 << index.jhq.level_bits[0];
    const size_t skip_bits = index.jhq.residual_bits_per_subspace;

    for (size_t j = 0; j < list_size; ++j) {
        JHQDecoder decoder(codes + j * index.jhq.code_size);
        float total_distance = 0.0f;

        for (int m = 0; m < index.jhq.M; ++m) {
            uint32_t centroid_id = decoder.decode(index.jhq.level_bits[0]);
            total_distance += jhq_primary_tables[m * K0 + centroid_id];
            decoder.skip_bits(skip_bits);
        }

        distances[j] = total_distance;
    }
}

void IVFJHQScanner::select_top_candidates(
    const float* primary_distances,
    size_t list_size,
    size_t n_candidates) const
{
    n_candidates = std::min(n_candidates, list_size);

    if (n_candidates == 0) {
        workspace_candidate_indices.clear();
        return;
    }

    workspace_candidate_distances_faiss.resize(n_candidates);
    workspace_candidate_indices_faiss.resize(n_candidates);

    bool is_min_heap = (index.metric_type == METRIC_L2);

    if (is_min_heap) {
        for (size_t i = 0; i < n_candidates; ++i) {
            workspace_candidate_distances_faiss[i] = primary_distances[i];
            workspace_candidate_indices_faiss[i] = static_cast<idx_t>(i);
        }

        faiss::heap_heapify<faiss::CMax<float, idx_t>>(
            n_candidates,
            workspace_candidate_distances_faiss.data(),
            workspace_candidate_indices_faiss.data());

        for (size_t i = n_candidates; i < list_size; ++i) {
            if (primary_distances[i] < workspace_candidate_distances_faiss[0]) {
                faiss::heap_replace_top<faiss::CMax<float, idx_t>>(
                    n_candidates,
                    workspace_candidate_distances_faiss.data(),
                    workspace_candidate_indices_faiss.data(),
                    primary_distances[i],
                    static_cast<idx_t>(i));
            }
        }

        faiss::heap_reorder<faiss::CMax<float, idx_t>>(
            n_candidates,
            workspace_candidate_distances_faiss.data(),
            workspace_candidate_indices_faiss.data());

    } else {
        for (size_t i = 0; i < n_candidates; ++i) {
            workspace_candidate_distances_faiss[i] = primary_distances[i];
            workspace_candidate_indices_faiss[i] = static_cast<idx_t>(i);
        }

        faiss::heap_heapify<faiss::CMin<float, idx_t>>(
            n_candidates,
            workspace_candidate_distances_faiss.data(),
            workspace_candidate_indices_faiss.data());

        for (size_t i = n_candidates; i < list_size; ++i) {
            if (primary_distances[i] > workspace_candidate_distances_faiss[0]) {
                faiss::heap_replace_top<faiss::CMin<float, idx_t>>(
                    n_candidates,
                    workspace_candidate_distances_faiss.data(),
                    workspace_candidate_indices_faiss.data(),
                    primary_distances[i],
                    static_cast<idx_t>(i));
            }
        }

        faiss::heap_reorder<faiss::CMin<float, idx_t>>(
            n_candidates,
            workspace_candidate_distances_faiss.data(),
            workspace_candidate_indices_faiss.data());
    }

    workspace_candidate_indices.clear();
    workspace_candidate_indices.reserve(n_candidates);
    for (size_t i = 0; i < n_candidates; ++i) {
        workspace_candidate_indices.push_back(
            static_cast<size_t>(workspace_candidate_indices_faiss[i]));
    }
}

void IVFJHQScanner::update_heap_single_candidate(
    size_t j,
    float dis,
    bool is_max_heap,
    size_t k,
    float* simi,
    idx_t* idxi,
    const idx_t* ids,
    size_t& nup) const
{
    if (!passes_selector(j, ids)) {
        return;
    }

    if (is_max_heap) {
        if (CMin<float, idx_t>::cmp(simi[0], dis)) {
            idx_t id = get_candidate_id(j, ids);
            faiss::heap_replace_top<CMin<float, idx_t>>(k, simi, idxi, dis, id);
            nup++;
        }
    } else {
        if (CMax<float, idx_t>::cmp(simi[0], dis)) {
            idx_t id = get_candidate_id(j, ids);
            faiss::heap_replace_top<CMax<float, idx_t>>(k, simi, idxi, dis, id);
            nup++;
        }
    }
}

void IVFJHQScanner::update_heap_four_candidates(
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
    size_t& nup) const
{
    if (k > 32) {
        float distances[4] = { dist0, dist1, dist2, dist3 };

        for (int i = 0; i < 4; i++) {
            if (!passes_selector(saved_j[i], ids)) {
                continue;
            }

            idx_t id = get_candidate_id(saved_j[i], ids);

            if (is_max_heap) {
                if (!CMin<float, idx_t>::cmp(simi[0], distances[i])) {
                    continue;
                }
            } else {
                if (!CMax<float, idx_t>::cmp(simi[0], distances[i])) {
                    continue;
                }
            }

            heap_batch_buffer_.push_back(
                { distances[i], id, saved_j[i] });
        }

        if (heap_batch_buffer_.size() >= kHeapBatchSize) {
            flush_heap_batch(is_max_heap, k, simi, idxi, nup);
        }
    } else {
        float distances[4] = { dist0, dist1, dist2, dist3 };
        for (int i = 0; i < 4; i++) {
            if (!passes_selector(saved_j[i], ids)) {
                continue;
            }

            idx_t id = get_candidate_id(saved_j[i], ids);

            if (is_max_heap) {
                if (CMin<float, idx_t>::cmp(simi[0], distances[i])) {
                    faiss::heap_replace_top<CMin<float, idx_t>>(
                        k, simi, idxi, distances[i], id);
                    nup++;
                }
            } else {
                if (CMax<float, idx_t>::cmp(simi[0], distances[i])) {
                    faiss::heap_replace_top<CMax<float, idx_t>>(
                        k, simi, idxi, distances[i], id);
                    nup++;
                }
            }
        }
    }
}

void IVFJHQScanner::update_heap_sixteen_candidates(
    const size_t saved_j[16],
    const float distances[16],
    bool is_max_heap,
    size_t k,
    float* simi,
    idx_t* idxi,
    const idx_t* ids,
    size_t& nup) const
{
    for (int i = 0; i < 16; i++) {
        if (!passes_selector(saved_j[i], ids)) {
            continue;
        }

        idx_t id = get_candidate_id(saved_j[i], ids);

        if (is_max_heap) {
            if (CMin<float, idx_t>::cmp(simi[0], distances[i])) {
                faiss::heap_replace_top<CMin<float, idx_t>>(
                    k, simi, idxi, distances[i], id);
                nup++;
            }
        } else {
            if (CMax<float, idx_t>::cmp(simi[0], distances[i])) {
                faiss::heap_replace_top<CMax<float, idx_t>>(
                    k, simi, idxi, distances[i], id);
                nup++;
            }
        }
    }
}

void IVFJHQScanner::update_heap_sixteen_candidates_batched(
    const size_t saved_j[16],
    const float distances[16],
    bool is_max_heap,
    size_t k,
    float* simi,
    idx_t* idxi,
    const idx_t* ids,
    size_t& nup) const
{
    if (k > 32) {
        for (int i = 0; i < 16; i++) {
            if (!passes_selector(saved_j[i], ids)) {
                continue;
            }

            idx_t id = get_candidate_id(saved_j[i], ids);

            if (is_max_heap) {
                if (!CMin<float, idx_t>::cmp(simi[0], distances[i])) {
                    continue;
                }
            } else {
                if (!CMax<float, idx_t>::cmp(simi[0], distances[i])) {
                    continue;
                }
            }

            heap_batch_buffer_.push_back(
                { distances[i], id, saved_j[i] });
        }

        if (heap_batch_buffer_.size() >= kHeapBatchSize) {
            flush_heap_batch(is_max_heap, k, simi, idxi, nup);
        }
    } else {
        update_heap_sixteen_candidates(
            saved_j, distances, is_max_heap, k, simi, idxi, ids, nup);
    }
}

void IVFJHQScanner::flush_heap_batch(bool is_max_heap,
    size_t k,
    float* simi,
    idx_t* idxi,
    size_t& nup) const
{
    if (heap_batch_buffer_.empty())
        return;

    if (is_max_heap) {
        std::sort(heap_batch_buffer_.begin(),
            heap_batch_buffer_.end(),
            [](const BatchCandidate& a, const BatchCandidate& b) {
                return a.distance > b.distance;
            });
    } else {
        std::sort(heap_batch_buffer_.begin(),
            heap_batch_buffer_.end(),
            [](const BatchCandidate& a, const BatchCandidate& b) {
                return a.distance < b.distance;
            });
    }

    for (const auto& candidate : heap_batch_buffer_) {
        if (is_max_heap) {
            if (CMin<float, idx_t>::cmp(simi[0], candidate.distance)) {
                faiss::heap_replace_top<CMin<float, idx_t>>(
                    k, simi, idxi, candidate.distance, candidate.id);
                nup++;
            }
        } else {
            if (CMax<float, idx_t>::cmp(simi[0], candidate.distance)) {
                faiss::heap_replace_top<CMax<float, idx_t>>(
                    k, simi, idxi, candidate.distance, candidate.id);
                nup++;
            }
        }
    }

    heap_batch_buffer_.clear();
}

idx_t IVFJHQScanner::get_candidate_id(
    size_t j,
    const idx_t* ids) const
{
    if (store_pairs || ids == nullptr) {
        return lo_build(list_no, j);
    }
    return ids[j];
}

bool IVFJHQScanner::passes_selector(
    size_t j,
    const idx_t* ids) const
{
    if (!sel) {
        return true;
    }
    return sel->is_member(get_candidate_id(j, ids));
}

void IVFJHQScanner::ensure_workspace_capacity(
    size_t max_list_size,
    size_t max_candidates) const
{
    bool resized = false;

    if (workspace_capacity_lists < max_list_size) {
        size_t new_capacity = static_cast<size_t>(max_list_size * 1.3f);
        resize_workspace(
            new_capacity, workspace_primary_distances, workspace_capacity_lists);
        workspace_primary_distances_quantized.reserve(new_capacity);
        resized = true;
    }

    if (workspace_capacity_candidates < max_candidates) {
        size_t new_capacity = static_cast<size_t>(max_candidates * 1.3f);
        workspace_candidate_distances_faiss.reserve(new_capacity);
        workspace_candidate_indices_faiss.reserve(new_capacity);
        workspace_capacity_candidates = new_capacity;
        resized = true;
    }
}

void IVFJHQScanner::quantize_primary_distances(size_t list_size) const
{
    workspace_primary_distances_quantized.clear();
    if (list_size == 0) {
        primary_distance_min = 0.0f;
        primary_distance_scale = 1.0f;
        return;
    }

    const float* distances = workspace_primary_distances.data();
    float min_val = std::numeric_limits<float>::infinity();
    float max_val = -std::numeric_limits<float>::infinity();

    for (size_t i = 0; i < list_size; ++i) {
        float v = distances[i];
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
    }

    primary_distance_min = min_val;
    float range = max_val - min_val;
    if (range <= 1e-12f) {
        primary_distance_scale = 1.0f;
        workspace_primary_distances_quantized.assign(list_size, 0);
        return;
    }

    primary_distance_scale = range / 65535.0f;
    float inv_scale = 1.0f / primary_distance_scale;

    workspace_primary_distances_quantized.resize(list_size);
    for (size_t i = 0; i < list_size; ++i) {
        float normalized = (distances[i] - primary_distance_min) * inv_scale;
        uint32_t quantized = static_cast<uint32_t>(
            std::round(std::max(0.0f, std::min(65535.0f, normalized))));
        workspace_primary_distances_quantized[i] =
            static_cast<uint16_t>(quantized);
    }
}

float IVFJHQScanner::reconstruct_primary_distance(uint16_t qvalue) const
{
    return primary_distance_min +
        primary_distance_scale * static_cast<float>(qvalue);
}

void IVFJHQScanner::ensure_table_capacity(
    size_t primary_table_size,
    size_t residual_table_size) const
{
    if (workspace_capacity_primary_tables < primary_table_size) {
        size_t new_capacity = static_cast<size_t>(primary_table_size * 1.2f);
        jhq_primary_tables.reserve(new_capacity);
        workspace_capacity_primary_tables = new_capacity;
    }

    if (workspace_capacity_residual_tables < residual_table_size) {
        size_t new_capacity = static_cast<size_t>(residual_table_size * 1.2f);
        jhq_residual_tables.reserve(new_capacity);
        workspace_capacity_residual_tables = new_capacity;
    }
}

void IVFJHQScanner::resize_workspace(
    size_t required_size,
    std::vector<float>& workspace,
    size_t& current_capacity) const
{
    if (current_capacity < required_size) {
        workspace.reserve(required_size);
        current_capacity = required_size;
    }
}

void IVFJHQScanner::resize_workspace(
    size_t required_size,
    std::vector<size_t>& workspace,
    size_t& current_capacity) const
{
    if (current_capacity < required_size) {
        workspace.reserve(required_size);
        current_capacity = required_size;
    }
}

void IVFJHQScanner::resize_workspace(
    size_t required_size,
    AlignedTable<float>& workspace,
    size_t& current_capacity) const
{
    if (current_capacity < required_size) {
        workspace.resize(required_size);
        current_capacity = required_size;
    }
}

void IVFJHQScanner::reset_for_reuse()
{
    query = nullptr;
    list_no = -1;
    tables_computed = false;
    is_reusable = true;
    last_used = std::chrono::steady_clock::now();
    reuse_count++;

    workspace_candidate_indices.clear();
    workspace_candidate_distances.clear();

    query_rotated.clear();
    jhq_residual_offsets.clear();
}

size_t IVFJHQScanner::get_workspace_memory_usage() const
{
    size_t total_bytes = 0;
    total_bytes += workspace_primary_distances.size() * sizeof(float);
    total_bytes += workspace_candidate_indices.capacity() * sizeof(size_t);
    total_bytes += workspace_candidate_distances.capacity() * sizeof(float);
    total_bytes += workspace_primary_distances_quantized.capacity() * sizeof(uint16_t);
    total_bytes += query_rotated.capacity() * sizeof(float);
    total_bytes += jhq_primary_tables.capacity() * sizeof(float);
    total_bytes += jhq_residual_tables.capacity() * sizeof(float);
    total_bytes += jhq_residual_offsets.capacity() * sizeof(size_t);
    return total_bytes;
}

void IVFJHQScanner::print_performance_stats() const
{
    (void)this;
}
size_t IVFJHQScanner::get_residual_table_index(int m, int level, int d, uint32_t scalar_id) const
{
    return jhq_residual_offsets[level] +
        static_cast<size_t>(m) * static_cast<size_t>(index.jhq.Ds) * static_cast<size_t>(1 << index.jhq.level_bits[level]) +
        static_cast<size_t>(d) * static_cast<size_t>(1 << index.jhq.level_bits[level]) +
        static_cast<size_t>(scalar_id);
}

} 
