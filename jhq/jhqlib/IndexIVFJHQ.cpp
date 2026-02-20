#include "IndexIVFJHQ.h"

#include <faiss/Clustering.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/index_read_utils.h>
#include <faiss/impl/io_macros.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <limits>

namespace faiss {

thread_local IndexIVFJHQ::SearchWorkspace IndexIVFJHQ::search_workspace_;

IndexIVFJHQ::SearchWorkspace& IndexIVFJHQ::get_search_workspace() const
{
    return search_workspace_;
}

void IndexIVFJHQ::mark_single_level_adapter_dirty()
{
    single_level_adapter_dirty_ = true;
}

void IndexIVFJHQ::ensure_single_level_adapter_ready() const
{
    if (!should_use_single_level_adapter()) {
        return;
    }

    if (!single_level_adapter_) {
        single_level_adapter_ = std::make_unique<IndexIVFPQ>(
            quantizer,
            d,
            nlist,
            jhq.M,
            jhq.level_bits[0],
            metric_type,
            false);
        single_level_adapter_->own_invlists = false;
        single_level_adapter_->by_residual = false;
        single_level_adapter_->use_precomputed_table = 0;
        single_level_adapter_->scan_table_threshold = 0;
        single_level_adapter_dirty_ = true;
    }

    if (single_level_adapter_dirty_) {
        auto& adapter = *single_level_adapter_;
        adapter.quantizer = quantizer;
        adapter.nlist = nlist;
        adapter.invlists = invlists;
        adapter.ntotal = ntotal;
        adapter.metric_type = metric_type;
        adapter.by_residual = false;
        adapter.use_precomputed_table = 0;
        adapter.scan_table_threshold = 0;
        adapter.parallel_mode = parallel_mode;

        adapter.pq = ProductQuantizer(d, jhq.M, jhq.level_bits[0]);
        adapter.pq.set_derived_values();
        for (int m = 0; m < jhq.M; ++m) {
            const auto& centroids = jhq.codewords[m][0];
            FAISS_THROW_IF_NOT_MSG(
                centroids.size() == adapter.pq.ksub * adapter.pq.dsub,
                "Primary codebook size mismatch for single-level adapter");
            std::memcpy(
                adapter.pq.get_centroids(m, 0),
                centroids.data(),
                centroids.size() * sizeof(float));
        }

        adapter.code_size = adapter.pq.code_size;
        adapter.is_trained = is_trained;
        adapter.ntotal = ntotal;
        single_level_adapter_dirty_ = false;
    } else {
        single_level_adapter_->invlists = invlists;
        single_level_adapter_->ntotal = ntotal;
        single_level_adapter_->metric_type = metric_type;
        single_level_adapter_->parallel_mode = parallel_mode;
    }
}

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

IndexIVFJHQ::IndexIVFJHQ(Index* quantizer,
    size_t d,
    size_t nlist,
    size_t M,
    const std::vector<int>& level_bits,
    bool use_jl_transform,
    float jhq_oversampling,
    MetricType metric,
    bool own_invlists)
    : IndexIVF(quantizer, d, nlist, 0, metric, own_invlists)
    , jhq(d,
          M,
          level_bits,
          use_jl_transform,
          jhq_oversampling,
          true,
          false,
          metric)
    , default_jhq_oversampling(jhq_oversampling)
    , use_early_termination(true)
{
    if (verbose) {
        printf("Initializing IndexIVFJHQ: d=%zd, nlist=%zd, M=%zd, levels=%zd\n",
            d,
            nlist,
            M,
            level_bits.size());
    }

    validate_parameters();
    code_size = jhq.code_size;

    if (own_invlists && invlists) {
        invlists->code_size = code_size;
    }

    is_trained = false;

    if (verbose) {
        printf("IndexIVFJHQ initialized: code_size=%zd bytes\n", code_size);
        printf("JHQ configuration: M=%d, Ds=%d, levels=%d\n",
            jhq.M,
            jhq.Ds,
            jhq.num_levels);
    }
}

IndexIVFJHQ::IndexIVFJHQ()
    : IndexIVF()
    , jhq()
    , default_jhq_oversampling(4.0f)
    , use_early_termination(true)
{
}

void IndexIVFJHQ::validate_parameters() const
{
    FAISS_THROW_IF_NOT_MSG(d > 0, "Vector dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
        d == quantizer->d,
        "Quantizer dimension must match index dimension");
    FAISS_THROW_IF_NOT_MSG(
        d == jhq.d, "JHQ dimension must match index dimension");
    FAISS_THROW_IF_NOT_MSG(nlist > 0, "Number of lists must be positive");
    FAISS_THROW_IF_NOT_MSG(
        default_jhq_oversampling >= 1.0f,
        "JHQ oversampling must be >= 1.0");
    FAISS_THROW_IF_NOT_MSG(
        metric_type == jhq.metric_type,
        "Index and JHQ must use the same metric");
    FAISS_THROW_IF_NOT_MSG(
        jhq.M > 0 && jhq.Ds > 0,
        "Invalid JHQ subspace configuration");
    FAISS_THROW_IF_NOT_MSG(
        jhq.num_levels > 0,
        "JHQ must have at least one quantization level");
}

void IndexIVFJHQ::train(idx_t n, const float* x)
{
    if (verbose) {
        printf("\n=== Training IndexIVFJHQ (Optimized) ===\n");
        printf("Training vectors: %zd\n", n);
        printf("Dimension: %d\n", d);
        printf("Coarse clusters: %zd\n", nlist);
        printf("JHQ subspaces: %d\n", jhq.M);
    }

    FAISS_THROW_IF_NOT_MSG(n > 0, "Training set cannot be empty");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Training data cannot be null");
    FAISS_THROW_IF_NOT_MSG(
        n >= nlist,
        "Need at least as many training vectors as coarse clusters");

    double training_start_time = getmillisecs();

    if (verbose) {
        printf("Stage 1: Training coarse quantizer...\n");
    }
    double coarse_start_time = getmillisecs();
    train_q1(n, x, verbose, metric_type);
    double coarse_time = getmillisecs() - coarse_start_time;

    if (verbose) {
        printf("Stage 2: Training JHQ on original vectors...\n");
    }
    double jhq_start_time = getmillisecs();
    train_jhq_on_originals(n, x);
    double jhq_time = getmillisecs() - jhq_start_time;

    is_trained = true;

    initialize_optimized_layout();
    mark_single_level_adapter_dirty();

    double total_time = getmillisecs() - training_start_time;

    if (verbose) {
        printf("=== Training Complete ===\n");
        printf("Total training time: %.2f ms\n", total_time);
        printf("  - Coarse quantizer: %.1f%%\n",
            100.0 * coarse_time / total_time);
        printf("  - JHQ training: %.1f%%\n", 100.0 * jhq_time / total_time);
        printf("Memory usage: %.2f MB\n",
            get_memory_usage() / (1024.0 * 1024.0));
    }
}

void IndexIVFJHQ::train_jhq_on_originals(idx_t n, const float* x)
{
    if (verbose) {
        printf("Training JHQ on %zd original vectors...\n", n);
    }

    jhq.train(n, x);

    FAISS_THROW_IF_NOT_MSG(jhq.is_trained_(), "JHQ training failed");
    is_trained = true;

    if (verbose) {
        printf("JHQ training successful:\n");
        printf("  - Code size: %zd bytes per vector\n", jhq.code_size);
        printf("  - Compression ratio: %.1fx\n",
            (sizeof(float) * d) / (float)jhq.code_size);
        printf("  - JL transform: %s\n",
            jhq.use_jl_transform ? "enabled" : "disabled");
        printf("  - Levels: %d\n", jhq.num_levels);
    }
    mark_single_level_adapter_dirty();
}

void IndexIVFJHQ::train_jhq_with_precomputed_residuals(
    idx_t n,
    const float* residuals)
{
    if (verbose) {
        printf("Training JHQ with precomputed residuals (%zd vectors)...\n", n);
    }

    double jhq_start_time = getmillisecs();
    train_jhq_on_originals(n, residuals);
    double jhq_time = getmillisecs() - jhq_start_time;

    if (verbose) {
        printf("JHQ trained on residuals in %.2f ms\n", jhq_time);
    }

    initialize_optimized_layout();
    is_trained = true;
}

idx_t IndexIVFJHQ::train_encoder_num_vectors() const
{
    return jhq.train_encoder_num_vectors();
}

void IndexIVFJHQ::add_with_precomputed_assignments(
    idx_t n,
    const float* x,
    const idx_t* xids,
    const idx_t* coarse_idx)
{
    if (verbose) {
        printf("Adding %zd vectors with precomputed assignments...\n", n);
    }

    double start_time = getmillisecs();
    add_core(n, x, xids, coarse_idx);
    double add_time = getmillisecs() - start_time;
    mark_single_level_adapter_dirty();

    if (verbose) {
        printf("Added vectors in %.2f ms (%.2f vectors/ms)\n",
            add_time,
            n / add_time);
    }
}

void IndexIVFJHQ::add_core(idx_t n,
    const float* x,
    const idx_t* xids,
    const idx_t* coarse_idx,
    void* inverted_list_context)
{
    FAISS_THROW_IF_NOT_MSG(
        is_trained, "Index must be trained before adding vectors");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of vectors must be positive");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Input vectors cannot be null");
    FAISS_THROW_IF_NOT_MSG(
        coarse_idx != nullptr, "Coarse indices cannot be null");

    const idx_t batch_size = 65536;
    if (n > batch_size) {
        for (idx_t i0 = 0; i0 < n; i0 += batch_size) {
            idx_t i1 = std::min(n, i0 + batch_size);
            if (verbose) {
                printf("IndexIVFJHQ::add_core: processing batch %" PRId64
                       ":%" PRId64 " / %" PRId64 "\n",
                    i0, i1, n);
            }
            add_core(i1 - i0,
                x + i0 * d,
                xids ? xids + i0 : nullptr,
                coarse_idx + i0,
                inverted_list_context);
        }
        return;
    }

    double encode_start = getmillisecs();

    const idx_t old_ntotal = ntotal;
    encode_to_separated_storage(n, x);

    initialize_vector_mapping_for_new_vectors(n, old_ntotal, coarse_idx, xids);

    std::vector<uint8_t> codes(n * code_size);
    encode_vectors_fallback(n, x, coarse_idx, codes.data());

    double encode_time = getmillisecs() - encode_start;

    DirectMapAdd dm_adder(direct_map, n, xids);
    size_t n_added = 0, n_ignored = 0;

    double add_start = getmillisecs();

    for (idx_t i = 0; i < n; ++i) {
        idx_t list_no = coarse_idx[i];

        if (list_no < 0) {
            n_ignored++;
            if (xids) {
                dm_adder.add(i, -1, 0);
            }
            continue;
        }

        FAISS_THROW_IF_NOT_MSG(
            list_no < static_cast<idx_t>(nlist),
            "Invalid coarse cluster assignment");

        idx_t id = xids ? xids[i] : ntotal + i;
        size_t offset = invlists->add_entry(
            list_no, id, codes.data() + i * code_size, inverted_list_context);

        dm_adder.add(i, list_no, offset);
        n_added++;
    }

    double add_time = getmillisecs() - add_start;

    ntotal += n;

    if (verbose && (n_ignored > 0 || n > 1000)) {
        printf("IndexIVFJHQ::add_core: added %zd vectors, ignored %zd\n",
            n_added, n_ignored);
        printf("  - Separated storage encoding: %.2f ms\n", encode_time);
        printf("  - Inverted list updates: %.2f ms\n", add_time);
        printf("  - Using separated storage: %s\n",
            jhq.has_pre_decoded_codes() ? "YES" : "NO");
    }

    if (pre_decoded_codes_initialized) {
        invalidate_pre_decoded_codes();
        if (use_pre_decoded_codes && ntotal > pre_decode_threshold * nlist / 4) {
            initialize_pre_decoded_codes();
        }
    }

    mark_single_level_adapter_dirty();
}

void IndexIVFJHQ::search(idx_t n,
    const float* x,
    idx_t k,
    float* distances,
    idx_t* labels,
    const SearchParameters* params) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before search");
    FAISS_THROW_IF_NOT_MSG(k > 0, "k must be positive");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of queries must be positive");

    const IVFJHQSearchParameters* jhq_params = dynamic_cast<const IVFJHQSearchParameters*>(params);

    size_t search_nprobe = this->nprobe;
    if (jhq_params && jhq_params->nprobe > 0) {
        search_nprobe = jhq_params->nprobe;
    }

    if (should_use_single_level_adapter()) {
        ensure_single_level_adapter_ready();
    }

    const size_t actual_nprobe = std::min(nlist, search_nprobe);
    FAISS_THROW_IF_NOT_MSG(actual_nprobe > 0, "nprobe must be positive");

    auto& workspace = get_search_workspace();
    workspace.ensure_capacity(n, actual_nprobe, d);

    idx_t* idx_buffer = workspace.list_indices.data();
    float* coarse_dis_buffer = workspace.coarse_distances.data();
    float* rotated_queries = workspace.rotated_queries.data();
    const size_t total_lists = static_cast<size_t>(n) * actual_nprobe;

    double t0 = getmillisecs();

    quantizer->search(n, x, actual_nprobe, coarse_dis_buffer, idx_buffer);

    double t1 = getmillisecs();

    if (jhq.use_jl_transform && jhq.is_rotation_trained) {
        if (verbose) {
            printf("Batch rotating %zd queries for JHQ search...\n", n);
        }
        jhq.apply_jl_rotation(n, x, rotated_queries);
    } else {
        std::memcpy(rotated_queries, x, sizeof(float) * n * d);
    }

    invlists->prefetch_lists(idx_buffer, total_lists);

    const IVFSearchParameters* ivf_params = dynamic_cast<const IVFSearchParameters*>(params);
    search_preassigned_with_rotated_queries(n,
        rotated_queries,
        k,
        idx_buffer,
        coarse_dis_buffer,
        distances,
        labels,
        false,
        ivf_params,
        &indexIVF_stats);

    double t2 = getmillisecs();

    indexIVF_stats.quantization_time += t1 - t0;
    indexIVF_stats.search_time += t2 - t0;
    indexIVFJHQ_stats.quantization_time += t1 - t0;
    indexIVFJHQ_stats.total_search_time += t2 - t0;
}

void IndexIVFJHQ::search_preassigned(idx_t n,
    const float* x,
    idx_t k,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    bool store_pairs,
    const IVFSearchParameters* params,
    IndexIVFStats* ivf_stats) const
{
    const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);
    bool is_max_heap = (metric_type == METRIC_INNER_PRODUCT);

    size_t total_ndis = 0;

    std::vector<float> rotated_queries_single_level;
    const bool adapter_active = should_use_single_level_adapter();
    const bool need_pre_rotated_queries = adapter_active && jhq.use_jl_transform && jhq.is_rotation_trained;
    if (need_pre_rotated_queries) {
        rotated_queries_single_level.resize(static_cast<size_t>(n) * d);
        jhq.apply_jl_rotation(n, x, rotated_queries_single_level.data());
    }
    const float* adapter_query_data = need_pre_rotated_queries ? rotated_queries_single_level.data() : x;

#pragma omp parallel if (n > 1) reduction(+ : total_ndis)
    {
        std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(store_pairs, nullptr, params));
        auto* jhq_scanner = dynamic_cast<IVFJHQScanner*>(scanner.get());

#pragma omp for schedule(static)
        for (idx_t i = 0; i < n; ++i) {
            float* simi = distances + i * k;
            idx_t* idxi = labels + i * k;

            if (is_max_heap) {
                heap_heapify<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_heapify<CMax<float, idx_t>>(k, simi, idxi);
            }

            if (jhq_scanner) {
                jhq_scanner->reset_for_reuse();
                jhq_scanner->set_query(x + i * d);
            } else {
                scanner->set_query(adapter_query_data + i * d);
            }

            size_t ndis_query = 0;

            for (size_t ik = 0; ik < nprobe; ik++) {
                idx_t key = keys[i * nprobe + ik];
                if (key < 0) {
                    continue;
                }

                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                size_t list_size = invlists->list_size(key);
                if (list_size == 0) {
                    continue;
                }

                InvertedLists::ScopedCodes codes(invlists, key);
                InvertedLists::ScopedIds ids(invlists, key);

                scanner->scan_codes(
                    list_size, codes.get(), ids.get(), simi, idxi, k);
                ndis_query += list_size;
            }

            if (is_max_heap) {
                heap_reorder<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
            }

            total_ndis += ndis_query;
        }
    }

    if (ivf_stats) {
        ivf_stats->ndis += total_ndis;
    }

    indexIVFJHQ_stats.nq += n;
    indexIVFJHQ_stats.ndis += total_ndis;
}

void IndexIVFJHQ::search_preassigned_with_rotated_queries(
    idx_t n,
    const float* x_rotated,
    idx_t k,
    const idx_t* keys,
    const float* coarse_dis,
    float* distances,
    idx_t* labels,
    bool store_pairs,
    const IVFSearchParameters* params,
    IndexIVFStats* ivf_stats) const
{
    const size_t nprobe = std::min(nlist, params ? params->nprobe : this->nprobe);
    bool is_max_heap = (metric_type == METRIC_INNER_PRODUCT);

    size_t total_ndis = 0;

#pragma omp parallel if (n > 1) reduction(+ : total_ndis)
    {
        std::unique_ptr<InvertedListScanner> scanner(
            get_InvertedListScanner(store_pairs, nullptr, params));
        auto* jhq_scanner = dynamic_cast<IVFJHQScanner*>(scanner.get());

#pragma omp for schedule(static)
        for (idx_t i = 0; i < n; ++i) {
            float* simi = distances + i * k;
            idx_t* idxi = labels + i * k;

            if (is_max_heap) {
                heap_heapify<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_heapify<CMax<float, idx_t>>(k, simi, idxi);
            }

            if (jhq_scanner) {
                jhq_scanner->reset_for_reuse();
                jhq_scanner->set_rotated_query(x_rotated + i * d);
            } else {
                scanner->set_query(x_rotated + i * d);
            }

            size_t ndis_query = 0;

            for (size_t ik = 0; ik < nprobe; ik++) {
                idx_t key = keys[i * nprobe + ik];
                if (key < 0) {
                    continue;
                }

                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                size_t list_size = invlists->list_size(key);
                if (list_size == 0) {
                    continue;
                }

                InvertedLists::ScopedCodes codes(invlists, key);
                InvertedLists::ScopedIds ids(invlists, key);

                scanner->scan_codes(
                    list_size, codes.get(), ids.get(), simi, idxi, k);
                ndis_query += list_size;
            }

            if (is_max_heap) {
                heap_reorder<CMin<float, idx_t>>(k, simi, idxi);
            } else {
                heap_reorder<CMax<float, idx_t>>(k, simi, idxi);
            }

            total_ndis += ndis_query;
        }
    }

    if (ivf_stats) {
        ivf_stats->ndis += total_ndis;
    }

    indexIVFJHQ_stats.nq += n;
    indexIVFJHQ_stats.ndis += total_ndis;
}

void IndexIVFJHQ::range_search_preassigned(idx_t nx,
    const float* x,
    float radius,
    const idx_t* keys,
    const float* coarse_dis,
    RangeSearchResult* result,
    bool store_pairs,
    const IVFSearchParameters* params,
    IndexIVFStats* stats) const
{
    FAISS_THROW_IF_NOT_MSG(
        is_trained, "Index must be trained before range search");

    const IVFJHQSearchParameters* jhq_params = dynamic_cast<const IVFJHQSearchParameters*>(params);

    const size_t search_nprobe = params && params->nprobe > 0 ? params->nprobe : this->nprobe;
    const size_t actual_nprobe = std::min(nlist, search_nprobe);

    std::unique_ptr<InvertedListScanner> scanner(
        get_InvertedListScanner(store_pairs, nullptr, params));

    result->nq = nx;
    if (!result->lims) {
        result->lims = new size_t[nx + 1];
    }

    for (idx_t i = 0; i <= nx; ++i) {
        result->lims[i] = 0;
    }

    size_t total_ndis = 0;
    double search_start_time = getmillisecs();

    bool use_parallel = nx > 10;
    std::vector<std::vector<std::pair<float, idx_t>>> all_results(nx);

    if (use_parallel) {
#pragma omp parallel for reduction(+ : total_ndis)
        for (idx_t i = 0; i < nx; ++i) {
            process_range_query(i,
                x,
                radius,
                keys,
                coarse_dis,
                actual_nprobe,
                store_pairs,
                params,
                all_results[i],
                total_ndis);
        }
    } else {
        for (idx_t i = 0; i < nx; ++i) {
            process_range_query(i,
                x,
                radius,
                keys,
                coarse_dis,
                actual_nprobe,
                store_pairs,
                params,
                all_results[i],
                total_ndis);
        }
    }

    for (idx_t i = 0; i < nx; ++i) {
        result->lims[i + 1] = result->lims[i] + all_results[i].size();
    }

    size_t total_results = result->lims[nx];
    if (total_results > 0) {
        result->distances = new float[total_results];
        result->labels = new idx_t[total_results];

        size_t offset = 0;
        for (idx_t i = 0; i < nx; ++i) {
            auto& query_results = all_results[i];
            std::sort(query_results.begin(), query_results.end());

            for (size_t j = 0; j < query_results.size(); ++j) {
                result->distances[offset + j] = query_results[j].first;
                result->labels[offset + j] = query_results[j].second;
            }
            offset += query_results.size();
        }
    } else {
        result->distances = nullptr;
        result->labels = nullptr;
    }

    if (stats) {
        stats->ndis += total_ndis;
        stats->search_time += getmillisecs() - search_start_time;
    }
}

void IndexIVFJHQ::process_range_query(
    idx_t query_idx,
    const float* x,
    float radius,
    const idx_t* keys,
    const float* coarse_dis,
    size_t nprobe,
    bool store_pairs,
    const IVFSearchParameters* params,
    std::vector<std::pair<float, idx_t>>& results,
    size_t& ndis_counter) const
{
    std::unique_ptr<InvertedListScanner> scanner(
        get_InvertedListScanner(store_pairs, nullptr, params));

    scanner->set_query(x + query_idx * d);

    for (size_t ik = 0; ik < nprobe; ik++) {
        idx_t key = keys[query_idx * nprobe + ik];
        if (key < 0)
            continue;

        scanner->set_list(key, coarse_dis[query_idx * nprobe + ik]);

        size_t list_size = invlists->list_size(key);
        if (list_size == 0)
            continue;

        InvertedLists::ScopedCodes codes(invlists, key);
        InvertedLists::ScopedIds ids(invlists, key);

        for (size_t j = 0; j < list_size; ++j) {
            float distance = scanner->distance_to_code(codes.get() + j * code_size);

            bool within_range = (metric_type == METRIC_L2)
                ? (distance <= radius)
                : (distance >= radius);

            if (within_range) {
                idx_t id = store_pairs ? lo_build(key, j) : ids.get()[j];
                results.emplace_back(distance, id);
            }
        }

        ndis_counter += list_size;
    }
}

void IndexIVFJHQ::reconstruct_from_offset(int64_t list_no,
    int64_t offset,
    float* recons) const
{
    FAISS_THROW_IF_NOT_MSG(
        is_trained, "Index must be trained before reconstruction");
    FAISS_THROW_IF_NOT_MSG(
        list_no >= 0 && list_no < static_cast<int64_t>(nlist),
        "Invalid list number");

    const uint8_t* code = invlists->get_single_code(list_no, offset);
    jhq.sa_decode(1, code, recons);
}

void IndexIVFJHQ::encode_vectors(idx_t n,
    const float* x,
    const idx_t* list_nos,
    uint8_t* codes,
    bool include_listno) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before encoding");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of vectors must be positive");
    FAISS_THROW_IF_NOT_MSG(
        x != nullptr && list_nos != nullptr && codes != nullptr,
        "Input pointers cannot be null");

    if (verbose && n > 10000) {
        printf("Encoding %zd vectors with optimized IVFJHQ...\n", n);
    }

    if (include_listno) {
        size_t coarse_size = coarse_code_size();
        std::vector<uint8_t> jhq_codes(n * jhq.code_size);

        jhq.sa_encode(n, x, jhq_codes.data());

#pragma omp parallel for if (n > 1000)
        for (idx_t i = 0; i < n; ++i) {
            uint8_t* code_i = codes + i * (coarse_size + jhq.code_size);
            encode_listno(list_nos[i], code_i);
            std::memcpy(
                code_i + coarse_size,
                jhq_codes.data() + i * jhq.code_size,
                jhq.code_size);
        }
    } else {
        jhq.sa_encode(n, x, codes);
    }

    if (verbose && n > 10000) {
        printf("Encoded %zd vectors successfully\n", n);
    }
}

void IndexIVFJHQ::decode_vectors(idx_t n,
    const uint8_t* codes,
    const idx_t* list_nos,
    float* x) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before decoding");
    FAISS_THROW_IF_NOT_MSG(n > 0, "Number of vectors must be positive");
    FAISS_THROW_IF_NOT_MSG(
        codes != nullptr && list_nos != nullptr && x != nullptr,
        "Input pointers cannot be null");

    jhq.sa_decode(n, codes, x);
}

size_t IndexIVFJHQ::sa_code_size() const
{
    return coarse_code_size() + jhq.code_size;
}

void IndexIVFJHQ::sa_encode(idx_t n, const float* x, uint8_t* bytes) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before encoding");

    std::vector<idx_t> assign(n);
    std::vector<float> coarse_distances(n);
    quantizer->search(n, x, 1, coarse_distances.data(), assign.data());

    encode_vectors(n, x, assign.data(), bytes, true);
}

void IndexIVFJHQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const
{
    FAISS_THROW_IF_NOT_MSG(is_trained, "Index must be trained before decoding");

    size_t coarse_size = coarse_code_size();
    std::vector<uint8_t> jhq_codes(n * jhq.code_size);

    for (idx_t i = 0; i < n; ++i) {
        const uint8_t* code = bytes + i * sa_code_size();
        std::memcpy(
            jhq_codes.data() + i * jhq.code_size,
            code + coarse_size,
            jhq.code_size);
    }

    jhq.sa_decode(n, jhq_codes.data(), x);
}

void IndexIVFJHQ::reset()
{
    IndexIVF::reset();
    jhq.reset();
    rotated_coarse_centroids.clear();
    rotated_centroids_computed = false;
    is_trained = false;
    single_level_adapter_.reset();
    single_level_adapter_dirty_ = true;

    if (verbose) {
        printf("IndexIVFJHQ reset completed\n");
    }
}

void IndexIVFJHQ::merge_from(Index& otherIndex, idx_t add_id)
{
    check_compatible_for_merge(otherIndex);

    IndexIVFJHQ* other = static_cast<IndexIVFJHQ*>(&otherIndex);

    if (verbose) {
        printf("Merging IndexIVFJHQ: this.ntotal=%" PRId64
               ", other.ntotal=%" PRId64 "\n",
            ntotal,
            other->ntotal);
    }

    invlists->merge_from(other->invlists, add_id);

    ntotal += other->ntotal;
    other->ntotal = 0;

    if (jhq.has_optimized_layout()) {
        jhq.invalidate_memory_layout();
    }

    invalidate_pre_decoded_codes();
    mark_single_level_adapter_dirty();

    if (verbose) {
        printf("IndexIVFJHQ merge completed: ntotal = %" PRId64 "\n", ntotal);
    }
}

void IndexIVFJHQ::check_compatible_for_merge(
    const Index& otherIndex) const
{
    const IndexIVFJHQ* other = dynamic_cast<const IndexIVFJHQ*>(&otherIndex);
    FAISS_THROW_IF_NOT_MSG(other != nullptr, "Other index is not IndexIVFJHQ");

    FAISS_THROW_IF_NOT_MSG(other->d == d, "Vector dimensions must match");
    FAISS_THROW_IF_NOT_MSG(other->nlist == nlist, "Number of lists must match");
    FAISS_THROW_IF_NOT_MSG(
        other->metric_type == metric_type, "Metrics must match");

    FAISS_THROW_IF_NOT_MSG(other->jhq.M == jhq.M, "JHQ M must match");
    FAISS_THROW_IF_NOT_MSG(other->jhq.Ds == jhq.Ds, "JHQ Ds must match");
    FAISS_THROW_IF_NOT_MSG(
        other->jhq.num_levels == jhq.num_levels, "JHQ levels must match");

    for (int i = 0; i < jhq.num_levels; ++i) {
        FAISS_THROW_IF_NOT_MSG(
            other->jhq.level_bits[i] == jhq.level_bits[i],
            "JHQ level bits must match");
    }

    FAISS_THROW_IF_NOT_MSG(
        other->jhq.use_jl_transform == jhq.use_jl_transform,
        "JL transform usage must match");

    quantizer->check_compatible_for_merge(*other->quantizer);
}

void IndexIVFJHQ::set_jhq_oversampling(float oversampling)
{
    FAISS_THROW_IF_NOT_MSG(
        oversampling >= 1.0f, "Oversampling factor must be >= 1.0");

    default_jhq_oversampling = oversampling;
    jhq.set_default_oversampling(oversampling);

    if (verbose) {
        printf("Updated JHQ oversampling to %.2f\n", oversampling);
    }
}

void IndexIVFJHQ::set_early_termination(bool enable)
{
    use_early_termination = enable;

    if (verbose) {
        printf("Early termination %s\n", enable ? "enabled" : "disabled");
    }
}

void IndexIVFJHQ::set_pre_decode_threshold(size_t threshold)
{
    if (threshold != pre_decode_threshold) {
        pre_decode_threshold = threshold;
        invalidate_pre_decoded_codes();

        if (verbose) {
            printf("Set pre-decode threshold to %zu\n", threshold);
        }
    }
}

void IndexIVFJHQ::enable_pre_decoded_codes(bool enable)
{
    if (enable != use_pre_decoded_codes) {
        use_pre_decoded_codes = enable;
        if (!enable) {
            invalidate_pre_decoded_codes();
        }

        if (verbose) {
            printf("Pre-decoded codes %s\n", enable ? "enabled" : "disabled");
        }
    }
}

size_t IndexIVFJHQ::get_memory_usage() const
{
    size_t total_bytes = 0;

    total_bytes += sizeof(*this);

    if (quantizer) {
        total_bytes += nlist * d * sizeof(float);
        total_bytes += 1024;
    }

    total_bytes += jhq.get_memory_usage();

    if (invlists) {
        for (size_t i = 0; i < nlist; ++i) {
            size_t list_size = invlists->list_size(i);
            total_bytes += list_size * (code_size + sizeof(idx_t));
        }
    }

    total_bytes += rotated_coarse_centroids.size() * sizeof(float);
    total_bytes += 4096;

    return total_bytes;
}

float IndexIVFJHQ::get_compression_ratio() const
{
    if (!is_trained) {
        return 0.0f;
    }

    size_t original_size = d * sizeof(float);
    size_t compressed_size = code_size;

    size_t coarse_bits = 0;
    size_t nl = nlist - 1;
    while (nl > 0) {
        coarse_bits++;
        nl >>= 1;
    }
    compressed_size += (coarse_bits + 7) / 8;

    return static_cast<float>(original_size) / compressed_size;
}

float IndexIVFJHQ::get_search_accuracy_estimate(size_t nprobe_val) const
{
    if (!is_trained || ntotal == 0) {
        return 0.0f;
    }

    float probe_fraction = static_cast<float>(nprobe_val) / nlist;
    float base_accuracy = 1.0f - std::exp(-probe_fraction * 3.0f);
    float jhq_accuracy_factor = 0.9f + 0.05f * jhq.num_levels;
    jhq_accuracy_factor = std::min(jhq_accuracy_factor, 1.0f);
    float jl_factor = jhq.use_jl_transform ? 1.02f : 1.0f;
    float et_factor = use_early_termination
        ? (0.95f + 0.05f * std::min(default_jhq_oversampling, 8.0f) / 8.0f)
        : 1.0f;

    return base_accuracy * jhq_accuracy_factor * jl_factor * et_factor;
}

void IndexIVFJHQ::initialize_optimized_layout()
{
    jhq.initialize_memory_layout();
    precompute_rotated_centroids();
}

void IndexIVFJHQ::precompute_rotated_centroids() const
{
    if (rotated_centroids_computed)
        return;

    if (verbose) {
        printf("Precomputing rotated coarse centroids...\n");
    }

    rotated_coarse_centroids.resize(nlist * d);

    std::vector<float> centroids_flat(nlist * d);
    for (size_t i = 0; i < nlist; ++i) {
        quantizer->reconstruct(i, centroids_flat.data() + i * d);
    }

    jhq.apply_jl_rotation(
        nlist, centroids_flat.data(), rotated_coarse_centroids.data());
    rotated_centroids_computed = true;

    if (verbose) {
        printf("Rotated coarse centroids precomputed\n");
    }
}

void IndexIVFJHQ::optimize_for_search()
{
    if (!is_trained) {
        if (verbose) {
            printf("Cannot optimize: index not trained\n");
        }
        return;
    }

    if (verbose) {
        printf("Optimizing IndexIVFJHQ for search performance...\n");
    }

    double opt_start = getmillisecs();

    if (!jhq.is_trained_()) {
        if (verbose) {
            printf("WARNING: JHQ not marked as trained, fixing...\n");
        }
        jhq.is_trained = true;
    }

    if (jhq.residual_bits_per_subspace == 0 && jhq.num_levels > 1) {
        for (int level = 1; level < jhq.num_levels; ++level) {
            jhq.residual_bits_per_subspace += static_cast<size_t>(jhq.Ds) * jhq.level_bits[level];
        }
        if (verbose) {
            printf("Recalculated residual bits per subspace: %zu\n",
                jhq.residual_bits_per_subspace);
        }
    }

    precompute_rotated_centroids();

    if (!jhq.memory_layout_initialized_) {
        jhq.initialize_memory_layout();
    }

    if (use_early_termination) {
        float optimal_oversampling = 4.0f;
        if (nlist > 10000)
            optimal_oversampling = 3.0f;
        if (jhq.num_levels > 2)
            optimal_oversampling = 5.0f;
        set_jhq_oversampling(optimal_oversampling);
    }

    double opt_time = getmillisecs() - opt_start;

    if (should_use_single_level_adapter()) {
        if (use_pre_decoded_codes) {
            enable_pre_decoded_codes(false);
        }
        ensure_single_level_adapter_ready();
    } else if (use_pre_decoded_codes && ntotal > 0) {
        if (verbose) {
            printf("Initializing pre-decoded codes for search optimization...\n");
        }
        initialize_pre_decoded_codes();

        if (verbose) {
            size_t optimized_lists = 0;
            for (size_t i = 0; i < nlist; ++i) {
                if (has_pre_decoded_codes_for_list(i)) {
                    optimized_lists++;
                }
            }
            printf("Pre-decoded codes initialized for %zu/%zu lists\n", optimized_lists, nlist);
        }
    }

    if (verbose) {
        printf("Search optimization completed in %.2f ms\n", opt_time);
        printf("Final configuration:\n");
        printf("  - JHQ trained: %s\n", jhq.is_trained_() ? "YES" : "NO");
        printf("  - Residual bits per subspace: %zu\n",
            jhq.residual_bits_per_subspace);
        printf("  - Early termination: %s\n",
            use_early_termination ? "enabled" : "disabled");
        printf("  - Oversampling factor: %.1f\n", default_jhq_oversampling);
        printf("  - Memory layout: %s\n",
            jhq.memory_layout_initialized_ ? "optimized" : "standard");
        printf("  - Estimated memory: %.2f MB\n",
            get_memory_usage() / (1024.0 * 1024.0));
    }
}

void IndexIVFJHQ::print_stats() const
{
    printf("IndexIVFJHQ Statistics:\n");
    printf("  Dimension: %d\n", d);
    printf("  Number of lists: %zd\n", nlist);
    printf("  Total vectors: %" PRId64 "\n", ntotal);
    printf("  Code size: %zd bytes\n", code_size);
    printf("  Compression ratio: %.1fx\n", get_compression_ratio());
    printf("  Memory usage: %.2f MB\n",
        get_memory_usage() / (1024.0 * 1024.0));

    printf("\nJHQ Configuration:\n");
    printf("  Subspaces (M): %d\n", jhq.M);
    printf("  Subspace dimension (Ds): %d\n", jhq.Ds);
    printf("  Number of levels: %d\n", jhq.num_levels);
    printf("  Level bits: ");
    for (int i = 0; i < jhq.num_levels; ++i) {
        printf("%d ", jhq.level_bits[i]);
    }
    printf("\n");
    printf("  JL transform: %s\n",
        jhq.use_jl_transform ? "enabled" : "disabled");

    printf("\nOptimization Status:\n");
    printf("\n");
    printf("  Early termination: %s\n",
        use_early_termination ? "enabled" : "disabled");
    printf("  Oversampling factor: %.1f\n", default_jhq_oversampling);
    printf("  Optimized layout: %s\n",
        jhq.has_optimized_layout() ? "enabled" : "disabled");
    printf("  Rotated centroids: %s\n",
        rotated_centroids_computed ? "precomputed" : "computed on-demand");

    if (invlists && ntotal > 0) {
        std::vector<size_t> list_sizes(nlist);
        size_t min_size = SIZE_MAX, max_size = 0;
        double avg_size = 0.0;

        for (size_t i = 0; i < nlist; ++i) {
            list_sizes[i] = invlists->list_size(i);
            min_size = std::min(min_size, list_sizes[i]);
            max_size = std::max(max_size, list_sizes[i]);
            avg_size += list_sizes[i];
        }
        avg_size /= nlist;

        printf("\nList Distribution:\n");
        printf("  Average list size: %.1f\n", avg_size);
        printf("  Min/Max list size: %zd / %zd\n", min_size, max_size);

        double variance = 0.0;
        for (size_t i = 0; i < nlist; ++i) {
            double diff = list_sizes[i] - avg_size;
            variance += diff * diff;
        }
        variance /= nlist;
        printf("  Standard deviation: %.1f\n", std::sqrt(variance));
    }
}

void IndexIVFJHQ::benchmark_search(idx_t nq,
    const float* queries,
    idx_t k,
    size_t nprobe_val,
    int num_runs) const
{
    if (!is_trained) {
        printf("Index not trained - cannot benchmark\n");
        return;
    }

    printf("Benchmarking IndexIVFJHQ search performance:\n");
    printf("  Queries: %zd, k: %zd, nprobe: %zd\n", nq, k, nprobe_val);
    printf("  Runs: %d\n", num_runs);

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    size_t old_nprobe = nprobe;
    const_cast<IndexIVFJHQ*>(this)->nprobe = nprobe_val;
    search(1, queries, k, distances.data(), labels.data());

    double total_time = 0.0;
    double min_time = std::numeric_limits<double>::max();
    double max_time = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        double start_time = getmillisecs();
        search(nq, queries, k, distances.data(), labels.data());
        double run_time = getmillisecs() - start_time;

        total_time += run_time;
        min_time = std::min(min_time, run_time);
        max_time = std::max(max_time, run_time);

        printf("  Run %d: %.2f ms\n", run + 1, run_time);
    }

    const_cast<IndexIVFJHQ*>(this)->nprobe = old_nprobe;

    double avg_time = total_time / num_runs;
    printf("\nBenchmark Results:\n");
    printf("  Average time: %.2f ms\n", avg_time);
    printf("  Min time: %.2f ms\n", min_time);
    printf("  Max time: %.2f ms\n", max_time);
    printf("  Queries per second: %.1f\n", nq * 1000.0 / avg_time);
    printf("  Microseconds per query: %.1f\n", avg_time * 1000.0 / nq);
}

void IndexIVFJHQ::initialize_pre_decoded_codes() const
{
    if (pre_decoded_codes_initialized || !use_pre_decoded_codes || !is_trained) {
        return;
    }

    if (verbose) {
        printf("Initializing pre-decoded codes for lists with >%zu vectors...\n",
            pre_decode_threshold);
    }

    list_pre_decoded_codes.resize(nlist);

    size_t eligible_lists = 0;
    size_t total_eligible_vectors = 0;

    for (size_t i = 0; i < nlist; ++i) {
        size_t list_size = invlists->list_size(i);
        if (list_size >= pre_decode_threshold) {
            eligible_lists++;
            total_eligible_vectors += list_size;
        }
    }

    if (verbose) {
        printf("Pre-decoding %zu lists (%zu vectors total)...\n",
            eligible_lists,
            total_eligible_vectors);
    }

    double start_time = getmillisecs();

#pragma omp parallel for if (eligible_lists > 4)
    for (size_t i = 0; i < nlist; ++i) {
        size_t list_size = invlists->list_size(i);
        if (list_size >= pre_decode_threshold) {
            extract_list_codes(i);
        }
    }

    double extract_time = getmillisecs() - start_time;
    pre_decoded_codes_initialized = true;

    if (verbose) {
        printf("Pre-decoding completed in %.2f ms\n", extract_time);
        printf("Memory usage: %.2f MB\n",
            get_pre_decoded_memory_usage() / (1024.0 * 1024.0));
    }
}

void IndexIVFJHQ::extract_list_codes(idx_t list_no) const
{
    size_t list_size = invlists->list_size(list_no);
    if (list_size == 0) {
        return;
    }

    auto& pre_decoded = list_pre_decoded_codes[list_no];
    pre_decoded.initialize(jhq.M, jhq.Ds, jhq.num_levels, list_size);

    InvertedLists::ScopedCodes codes(invlists, list_no);
    const uint8_t* raw_codes = codes.get();

    for (size_t i = 0; i < list_size; ++i) {
        const uint8_t* packed_code = raw_codes + i * jhq.code_size;
        extract_single_code_to_pre_decoded(packed_code, pre_decoded, i);
    }
}

void IndexIVFJHQ::extract_single_code_to_pre_decoded(
    const uint8_t* packed_code,
    jhq_internal::PreDecodedCodes& pre_decoded,
    size_t vector_idx) const
{
    BitstringReader bit_reader(packed_code, jhq.code_size);

    uint8_t* primary_dest = const_cast<uint8_t*>(pre_decoded.get_primary_codes(vector_idx));

    uint8_t* residual_dest = nullptr;
    if (jhq.num_levels > 1) {
        residual_dest = const_cast<uint8_t*>(
            pre_decoded.get_residual_codes(vector_idx));
    }

    size_t residual_offset = 0;

    for (int m = 0; m < jhq.M; ++m) {
        const uint32_t primary_centroid_id = bit_reader.read(jhq.level_bits[0]);
        primary_dest[m] = static_cast<uint8_t>(primary_centroid_id);

        if (jhq.num_levels > 1 && residual_dest != nullptr) {
            for (int level = 1; level < jhq.num_levels; ++level) {
                for (int d = 0; d < jhq.Ds; ++d) {
                    const uint32_t scalar_id = bit_reader.read(jhq.level_bits[level]);
                    residual_dest[residual_offset++] = static_cast<uint8_t>(scalar_id);
                }
            }
        } else {
            bit_reader.i += jhq.residual_bits_per_subspace;
        }
    }
}

bool IndexIVFJHQ::has_pre_decoded_codes_for_list(idx_t list_no) const
{
    if (!pre_decoded_codes_initialized || list_no >= nlist) {
        return false;
    }

    return !list_pre_decoded_codes[list_no].empty() && invlists->list_size(list_no) >= pre_decode_threshold;
}

const jhq_internal::PreDecodedCodes*
IndexIVFJHQ::get_pre_decoded_codes_for_list(idx_t list_no) const
{
    if (has_pre_decoded_codes_for_list(list_no)) {
        return &list_pre_decoded_codes[list_no];
    }
    return nullptr;
}

void IndexIVFJHQ::invalidate_pre_decoded_codes()
{
    if (verbose && pre_decoded_codes_initialized) {
        printf("Invalidating pre-decoded codes\n");
    }

    list_pre_decoded_codes.clear();
    pre_decoded_codes_initialized = false;
}

size_t IndexIVFJHQ::get_pre_decoded_memory_usage() const
{
    size_t total = 0;
    for (const auto& pre_decoded : list_pre_decoded_codes) {
        total += pre_decoded.memory_usage();
    }
    return total;
}

InvertedListScanner* IndexIVFJHQ::get_InvertedListScanner(
    bool store_pairs,
    const IDSelector* sel,
    const IVFSearchParameters* params_in) const
{
    if (should_use_single_level_adapter()) {
        ensure_single_level_adapter_ready();
        return single_level_adapter_->get_InvertedListScanner(
            store_pairs, sel, params_in);
    }

    const IVFJHQSearchParameters* jhq_params = dynamic_cast<const IVFJHQSearchParameters*>(params_in);

    return new IVFJHQScanner(*this, store_pairs, sel, jhq_params);
}

void IndexIVFJHQ::write(IOWriter* f) const
{
    write_index_ivf_jhq(this, f);
}

IndexIVFJHQ* IndexIVFJHQ::read(IOReader* f)
{
    return read_index_ivf_jhq(f);
}

void IndexIVFJHQ::cleanup() { }

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
    this->list_no = list_no;

    if (has_separated_storage_available()) {
        current_list_pre_decoded = nullptr;
        if (index.verbose) {
            static std::atomic<int> call_count { 0 };
            if (++call_count % 1000 == 0) {
                printf("Using separated storage for list %ld (bandwidth optimized)\n", list_no);
            }
        }
    } else {
        current_list_pre_decoded = index.get_pre_decoded_codes_for_list(list_no);
        if (index.verbose && current_list_pre_decoded) {
            static std::atomic<int> call_count { 0 };
            if (++call_count % 1000 == 0) {
                printf("Using pre-decoded codes for list %ld (size: %zu)\n",
                    list_no, index.invlists->list_size(list_no));
            }
        }
    }
}

float IVFJHQScanner::distance_to_code(const uint8_t* code) const
{
    if (!tables_computed) {
        FAISS_THROW_MSG("Tables not computed");
    }

    const size_t offset_in_list = (code - index.invlists->get_codes(list_no)) / index.jhq.code_size;

    if (has_separated_storage_available()) {
        return distance_to_code_separated_storage(offset_in_list);
    } else {
        return distance_to_code_with_bit_decoding(code);
    }
}

bool IVFJHQScanner::has_separated_storage_available() const
{
    return index.jhq.has_pre_decoded_codes() && index.mapping_initialized && index.jhq.separated_codes_.is_initialized && !index.jhq.separated_codes_.empty();
}

float IVFJHQScanner::distance_to_code_separated_storage(size_t offset_in_list) const
{
    const idx_t global_vec_idx = index.get_global_vector_index(list_no, offset_in_list);

    if (global_vec_idx < 0) {
        return std::numeric_limits<float>::max();
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
        total_distance += jhq_internal::compute_cross_term_from_codes(
            index.jhq,
            primary_codes,
            residual_codes,
            index.jhq.separated_codes_.residual_subspace_stride,
            index.jhq.separated_codes_.residual_level_stride);
    }

    return total_distance;
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
        const uint8_t* residual_codes = current_list_pre_decoded->get_residual_codes(vector_idx_in_list);
        total_distance += jhq_internal::compute_cross_term_from_codes(
            index.jhq,
            primary_codes,
            residual_codes,
            current_list_pre_decoded->residual_subspace_stride,
            current_list_pre_decoded->residual_level_stride);
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

            const auto& centroids = index.jhq.codewords[m][0];
            const int num_centroids = centroids.empty() ? 0 : static_cast<int>(centroids.size() / index.jhq.Ds);
            const uint32_t safe_centroid = centroids.empty()
                ? 0
                : std::min<uint32_t>(centroid_id, static_cast<uint32_t>(std::max(0, num_centroids - 1)));
            const float* centroid_ptr = centroids.empty()
                ? nullptr
                : centroids.data() + static_cast<size_t>(safe_centroid) * index.jhq.Ds;

            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const uint32_t residual_limit = static_cast<uint32_t>(K_res - 1);
                const size_t level_offset = jhq_residual_offsets[level];
                const size_t table_base = level_offset + static_cast<size_t>(m) * index.jhq.Ds * K_res;
                const auto& scalar_codebook = index.jhq.scalar_codebooks[m][level - 1];
                const int codebook_size = static_cast<int>(scalar_codebook.size());

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

            if (centroid_ptr) {
                for (int d = 0; d < index.jhq.Ds; ++d) {
                    cross_term += centroid_ptr[d] * residual_buffer[d];
                }
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

        const auto& centroids = index.jhq.codewords[m][0];
        const size_t centroid_block = index.jhq.Ds;
        const float* centroid_ptr0 = (centroids.empty() || centroid_block == 0)
            ? nullptr
            : centroids.data() + static_cast<size_t>(id0) * centroid_block;
        const float* centroid_ptr1 = (centroids.empty() || centroid_block == 0)
            ? nullptr
            : centroids.data() + static_cast<size_t>(id1) * centroid_block;
        const float* centroid_ptr2 = (centroids.empty() || centroid_block == 0)
            ? nullptr
            : centroids.data() + static_cast<size_t>(id2) * centroid_block;
        const float* centroid_ptr3 = (centroids.empty() || centroid_block == 0)
            ? nullptr
            : centroids.data() + static_cast<size_t>(id3) * centroid_block;

        __m128i indices = _mm_set_epi32(id3, id2, id1, id0);
        __m128 primary_dists = _mm_i32gather_ps(table_base, indices, 4);

        distances = _mm_add_ps(distances, primary_dists);

        if (compute_residuals && index.jhq.num_levels > 1) {
            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const size_t level_offset = jhq_residual_offsets[level];
                const auto& scalar_codebook = index.jhq.scalar_codebooks[m][level - 1];
                const int codebook_size = static_cast<int>(scalar_codebook.size());

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
                            centroid_ptr0 ? centroid_ptr0[d] : 0.0f,
                            centroid_ptr1 ? centroid_ptr1[d] : 0.0f,
                            centroid_ptr2 ? centroid_ptr2[d] : 0.0f,
                            centroid_ptr3 ? centroid_ptr3[d] : 0.0f
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

        const auto& centroids = index.jhq.codewords[m][0];
        const size_t centroid_block = index.jhq.Ds;
        const float* centroid_ptrs[4];
        for (int i = 0; i < 4; ++i) {
            if (centroids.empty() || centroid_block == 0) {
                centroid_ptrs[i] = nullptr;
            } else {
                centroid_ptrs[i] = centroids.data() + static_cast<size_t>(centroid_ids[i]) * centroid_block;
            }
        }

        if (compute_residuals && index.jhq.num_levels > 1) {
            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const size_t level_offset = jhq_residual_offsets[level];
                const auto& scalar_codebook = index.jhq.scalar_codebooks[m][level - 1];
                const int codebook_size = static_cast<int>(scalar_codebook.size());

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
                            if (!centroid_ptrs[i]) {
                                continue;
                            }
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
        const auto& centroids = index.jhq.codewords[m][0];
        const size_t centroid_block = index.jhq.Ds;
        const float* centroid_ptrs[16];
        for (int i = 0; i < 16; ++i) {
            if (centroids.empty() || centroid_block == 0) {
                centroid_ptrs[i] = nullptr;
            } else {
                centroid_ptrs[i] = centroids.data() + static_cast<size_t>(centroid_ids[i]) * centroid_block;
            }
        }
        __m512i indices = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(centroid_ids));
        __m512 primary_dists = _mm512_i32gather_ps(indices, table_base, 4);
        total_distances = _mm512_add_ps(total_distances, primary_dists);

        if (compute_residuals && index.jhq.num_levels > 1) {
            for (int level = 1; level < index.jhq.num_levels; ++level) {
                const int K_res = 1 << index.jhq.level_bits[level];
                const size_t level_offset = jhq_residual_offsets[level];
                const auto& scalar_codebook = index.jhq.scalar_codebooks[m][level - 1];
                const int codebook_size = static_cast<int>(scalar_codebook.size());

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
                            if (!centroid_ptrs[i]) {
                                continue;
                            }
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
        const auto& centroids = index.jhq.codewords[m][0];
        const size_t centroid_block = index.jhq.Ds;
        const float* centroid_ptrs[16];

        {
            alignas(32) uint32_t centroid_ids[8];
            for (int i = 0; i < 8; ++i) {
                centroid_ids[i] = decoders[i].decode(index.jhq.level_bits[0]);
                if (centroids.empty() || centroid_block == 0) {
                    centroid_ptrs[i] = nullptr;
                } else {
                    centroid_ptrs[i] = centroids.data() + static_cast<size_t>(centroid_ids[i]) * centroid_block;
                }
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
                if (centroids.empty() || centroid_block == 0) {
                    centroid_ptrs[i + 8] = nullptr;
                } else {
                    centroid_ptrs[i + 8] = centroids.data() + static_cast<size_t>(centroid_ids[i]) * centroid_block;
                }
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
                                if (!centroid_ptrs[i]) {
                                    continue;
                                }
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
                                if (!centroid_ptrs[i + 8]) {
                                    continue;
                                }
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

    optimized_workspace.ensure_capacity(list_size, k * oversampling_factor);

    bool should_use_early_termination = use_early_termination && index.jhq.num_levels > 1 && k < list_size && list_size > k * oversampling_factor * 8 && oversampling_factor >= 3.0f;

    if (should_use_early_termination) {
        size_t n_candidates = static_cast<size_t>(std::min(
            static_cast<double>(list_size),
            static_cast<double>(k * oversampling_factor)));
        n_candidates = std::max(n_candidates, k);
        n_candidates = std::min(n_candidates, list_size);

        ensure_workspace_capacity(list_size, n_candidates);
        return scan_codes_early_termination(
            list_size, codes, ids, simi, idxi, k);
    } else {
        ensure_workspace_capacity(list_size, 0);
        return scan_codes_exhaustive(
            list_size, codes, ids, simi, idxi, k);
    }
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

    index.heap_batch_buffer.clear();

    if (!is_max_heap) {
        compute_primary_distances(list_size, codes);
        quantize_primary_distances(list_size);
        return scan_codes_exhaustive_l2_gated(
            list_size, codes, ids, simi, idxi, k);
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
    float effective_oversampling = std::min(oversampling_factor, 6.0f);

    size_t n_candidates = static_cast<size_t>(std::min(
        static_cast<double>(list_size),
        static_cast<double>(k * effective_oversampling)));
    n_candidates = std::max(n_candidates, k);
    n_candidates = std::min(n_candidates, list_size);

    float* primary_distances_ptr = workspace_primary_distances.data();

    workspace_candidate_indices.clear();
    workspace_candidate_indices.reserve(n_candidates);

    compute_primary_distances(list_size, codes);

    workspace_candidate_indices.resize(list_size);
    std::iota(
        workspace_candidate_indices.begin(),
        workspace_candidate_indices.end(),
        0);

    select_top_candidates(primary_distances_ptr, list_size, n_candidates);

    workspace_candidate_indices.resize(n_candidates);
    quantize_primary_distances(list_size);

    size_t nup = 0;
    bool is_max_heap = (index.metric_type == METRIC_INNER_PRODUCT);

    size_t i = 0;
    for (; i + 3 < n_candidates; i += 4) {
        size_t j0 = workspace_candidate_indices[i];
        size_t j1 = workspace_candidate_indices[i + 1];
        size_t j2 = workspace_candidate_indices[i + 2];
        size_t j3 = workspace_candidate_indices[i + 3];

        bool valid0 = passes_selector(j0, ids);
        bool valid1 = passes_selector(j1, ids);
        bool valid2 = passes_selector(j2, ids);
        bool valid3 = passes_selector(j3, ids);

        if (!valid0 && !valid1 && !valid2 && !valid3) {
            continue;
        }

        if (valid0) {
            idx_t id = get_candidate_id(j0, ids);
            float primary_dist = reconstruct_primary_distance(
                workspace_primary_distances_quantized[j0]);
            bool promising = is_max_heap || CMin<float, idx_t>::cmp(primary_dist, simi[0]);
            if (promising) {
                float dist0 = distance_to_code(codes + j0 * code_size);
                if (is_max_heap) {
                    if (CMin<float, idx_t>::cmp(simi[0], dist0)) {
                        faiss::heap_replace_top<CMin<float, idx_t>>(
                            k, simi, idxi, dist0, id);
                        nup++;
                    }
                } else {
                    if (CMax<float, idx_t>::cmp(simi[0], dist0)) {
                        faiss::heap_replace_top<CMax<float, idx_t>>(
                            k, simi, idxi, dist0, id);
                        nup++;
                    }
                }
            }
        }

        if (valid1) {
            idx_t id = get_candidate_id(j1, ids);
            float primary_dist = reconstruct_primary_distance(
                workspace_primary_distances_quantized[j1]);
            bool promising = is_max_heap || CMin<float, idx_t>::cmp(primary_dist, simi[0]);
            if (promising) {
                float dist1 = distance_to_code(codes + j1 * code_size);
                if (is_max_heap) {
                    if (CMin<float, idx_t>::cmp(simi[0], dist1)) {
                        faiss::heap_replace_top<CMin<float, idx_t>>(
                            k, simi, idxi, dist1, id);
                        nup++;
                    }
                } else {
                    if (CMax<float, idx_t>::cmp(simi[0], dist1)) {
                        faiss::heap_replace_top<CMax<float, idx_t>>(
                            k, simi, idxi, dist1, id);
                        nup++;
                    }
                }
            }
        }

        if (valid2) {
            idx_t id = get_candidate_id(j2, ids);
            float primary_dist = reconstruct_primary_distance(
                workspace_primary_distances_quantized[j2]);
            bool promising = is_max_heap || CMin<float, idx_t>::cmp(primary_dist, simi[0]);
            if (promising) {
                float dist2 = distance_to_code(codes + j2 * code_size);
                if (is_max_heap) {
                    if (CMin<float, idx_t>::cmp(simi[0], dist2)) {
                        faiss::heap_replace_top<CMin<float, idx_t>>(
                            k, simi, idxi, dist2, id);
                        nup++;
                    }
                } else {
                    if (CMax<float, idx_t>::cmp(simi[0], dist2)) {
                        faiss::heap_replace_top<CMax<float, idx_t>>(
                            k, simi, idxi, dist2, id);
                        nup++;
                    }
                }
            }
        }

        if (valid3) {
            idx_t id = get_candidate_id(j3, ids);
            float primary_dist = reconstruct_primary_distance(
                workspace_primary_distances_quantized[j3]);
            bool promising = is_max_heap || CMin<float, idx_t>::cmp(primary_dist, simi[0]);
            if (promising) {
                float dist3 = distance_to_code(codes + j3 * code_size);
                if (is_max_heap) {
                    if (CMin<float, idx_t>::cmp(simi[0], dist3)) {
                        faiss::heap_replace_top<CMin<float, idx_t>>(
                            k, simi, idxi, dist3, id);
                        nup++;
                    }
                } else {
                    if (CMax<float, idx_t>::cmp(simi[0], dist3)) {
                        faiss::heap_replace_top<CMax<float, idx_t>>(
                            k, simi, idxi, dist3, id);
                        nup++;
                    }
                }
            }
        }
    }

    for (; i < n_candidates; ++i) {
        size_t j = workspace_candidate_indices[i];

        if (!passes_selector(j, ids)) {
            continue;
        }

        idx_t id = get_candidate_id(j, ids);
        float primary_dist = reconstruct_primary_distance(
            workspace_primary_distances_quantized[j]);
        bool promising = is_max_heap || CMin<float, idx_t>::cmp(primary_dist, simi[0]);
        if (!promising) {
            continue;
        }

        const uint8_t* code = codes + j * code_size;
        float dis = distance_to_code(code);

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

    if (has_separated_storage_available()) {
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

    const bool use_separated_storage = index.jhq.has_pre_decoded_codes() && index.mapping_initialized;

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
    constexpr size_t CACHE_BATCH_SIZE = 128; // Increased for better cache utilization

#pragma omp parallel for schedule(static) if (list_size > 2000) // Lower threshold
    for (size_t batch_start = 0; batch_start < list_size; batch_start += CACHE_BATCH_SIZE) {
        const size_t batch_end = std::min(list_size, batch_start + CACHE_BATCH_SIZE);

        // Enhanced prefetching with multiple streams
        for (size_t i = batch_start; i < batch_end; i += 16) {
            const size_t prefetch_end = std::min(batch_end, i + 64);
            for (size_t j = i; j < prefetch_end; j += 8) {
                if (j < list_size) {
                    const idx_t global_idx = index.get_global_vector_index(list_no, j);
                    if (global_idx >= 0) {
                        const uint8_t* primary_codes = index.jhq.separated_codes_.get_primary_codes(global_idx);
                        _mm_prefetch(primary_codes, _MM_HINT_T0);
                        _mm_prefetch(primary_codes + 32, _MM_HINT_T0); // Next cache line
                    }
                }
            }
        }

        // Process batch with enhanced SIMD
        for (size_t offset_in_list = batch_start; offset_in_list < batch_end; ++offset_in_list) {
            const idx_t global_vec_idx = index.get_global_vector_index(list_no, offset_in_list);

            if (global_vec_idx >= 0) {
                const uint8_t* primary_codes = index.jhq.separated_codes_.get_primary_codes(global_vec_idx);
                float total_distance = 0.0f;

#ifdef __AVX512F__
                if (index.jhq.M >= 64) { // Process very large M more efficiently
                    __m512 acc1 = _mm512_setzero_ps();
                    __m512 acc2 = _mm512_setzero_ps();
                    __m512 acc3 = _mm512_setzero_ps();
                    __m512 acc4 = _mm512_setzero_ps();

                    int m = 0;
                    for (; m + 63 < index.jhq.M; m += 64) {
                        // Prefetch next iteration
                        if (m + 128 < index.jhq.M) {
                            _mm_prefetch(&primary_codes[m + 64], _MM_HINT_T0);
                            _mm_prefetch(&jhq_primary_tables[(m + 64) * K0], _MM_HINT_T1);
                        }

                        // Process 4 groups of 16 codes simultaneously
                        for (int group = 0; group < 4; ++group) {
                            const int group_offset = m + group * 16;

                            __m128i codes_128 = _mm_loadu_si128(
                                reinterpret_cast<const __m128i*>(&primary_codes[group_offset]));
                            __m512i codes_512 = _mm512_cvtepu8_epi32(codes_128);

                            // Create subspace offsets more efficiently
                            __m512i offsets = _mm512_mullo_epi32(
                                _mm512_set_epi32(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0),
                                _mm512_set1_epi32(K0));
                            __m512i base_offset = _mm512_set1_epi32(group_offset * K0);
                            offsets = _mm512_add_epi32(offsets, base_offset);

                            __m512i indices = _mm512_add_epi32(codes_512, offsets);
                            __m512 dists = _mm512_i32gather_ps(indices, jhq_primary_tables.data(), 4);

                            // Accumulate in different registers to avoid dependency chains
                            switch(group) {
                                case 0: acc1 = _mm512_add_ps(acc1, dists); break;
                                case 1: acc2 = _mm512_add_ps(acc2, dists); break;
                                case 2: acc3 = _mm512_add_ps(acc3, dists); break;
                                case 3: acc4 = _mm512_add_ps(acc4, dists); break;
                            }
                        }
                    }

                    // Combine accumulators
                    __m512 combined1 = _mm512_add_ps(acc1, acc2);
                    __m512 combined2 = _mm512_add_ps(acc3, acc4);
                    __m512 final_acc = _mm512_add_ps(combined1, combined2);
                    total_distance += _mm512_reduce_add_ps(final_acc);

                    // Handle remaining subspaces
                    for (; m < index.jhq.M; ++m) {
                        uint32_t code_val = primary_codes[m];
                        total_distance += jhq_primary_tables[m * K0 + code_val];
                    }

                } else if (index.jhq.M >= 32) {
                    // Optimized version for medium M values
                    __m512 acc1 = _mm512_setzero_ps();
                    __m512 acc2 = _mm512_setzero_ps();

                    int m = 0;
                    for (; m + 31 < index.jhq.M; m += 32) {
                        // Process first 16
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

                        // Process second 16
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

                    // Handle remaining
                    for (; m < index.jhq.M; ++m) {
                        uint32_t code_val = primary_codes[m];
                        total_distance += jhq_primary_tables[m * K0 + code_val];
                    }

                } else if (index.jhq.M >= 16) {
                    // Original AVX512 path for smaller M
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
                    // Enhanced AVX2 version for larger M
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    __m256 acc3 = _mm256_setzero_ps();
                    __m256 acc4 = _mm256_setzero_ps();

                    int m = 0;
                    for (; m + 31 < index.jhq.M; m += 32) {
                        // Process 4 groups of 8 codes
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

                    // Combine and reduce
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
                    // Original AVX2 path
                    __m256 acc1 = _mm256_setzero_ps();
                    __m256 acc2 = _mm256_setzero_ps();
                    int m = 0;

                    for (; m + 15 < index.jhq.M; m += 16) {
                        // First 8
                        __m128i codes1_64 = _mm_loadl_epi64(
                            reinterpret_cast<const __m128i*>(&primary_codes[m]));
                        __m256i codes1_256 = _mm256_cvtepu8_epi32(codes1_64);

                        __m256i offsets1 = _mm256_set_epi32(
                            (m+7)*K0, (m+6)*K0, (m+5)*K0, (m+4)*K0,
                            (m+3)*K0, (m+2)*K0, (m+1)*K0, m*K0);

                        __m256i indices1 = _mm256_add_epi32(codes1_256, offsets1);
                        __m256 dists1 = _mm256_i32gather_ps(jhq_primary_tables.data(), indices1, 4);
                        acc1 = _mm256_add_ps(acc1, dists1);

                        // Second 8
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
                    // Scalar fallback with unrolling
                    int m = 0;
                    for (; m + 7 < index.jhq.M; m += 8) {
                        // Unroll loop for better instruction-level parallelism
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
                distances[offset_in_list] = std::numeric_limits<float>::max();
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

            index.heap_batch_buffer.push_back(
                { distances[i], id, saved_j[i] });
        }

        if (index.heap_batch_buffer.size() >= IndexIVFJHQ::HEAP_BATCH_SIZE) {
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

            index.heap_batch_buffer.push_back(
                { distances[i], id, saved_j[i] });
        }

        if (index.heap_batch_buffer.size() >= IndexIVFJHQ::HEAP_BATCH_SIZE) {
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
    if (index.heap_batch_buffer.empty())
        return;

    if (is_max_heap) {
        std::sort(index.heap_batch_buffer.begin(),
            index.heap_batch_buffer.end(),
            [](const BatchCandidate& a, const BatchCandidate& b) {
                return a.distance > b.distance;
            });
    } else {
        std::sort(index.heap_batch_buffer.begin(),
            index.heap_batch_buffer.end(),
            [](const BatchCandidate& a, const BatchCandidate& b) {
                return a.distance < b.distance;
            });
    }

    for (const auto& candidate : index.heap_batch_buffer) {
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

    index.heap_batch_buffer.clear();
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
    if (total_scans_performed > 0) {
        printf("IVFJHQScanner Performance Stats:\n");
        printf("  Total scans: %zu\n", total_scans_performed);
        printf("  Total codes processed: %zu\n", total_codes_processed);
        printf("  Average codes per scan: %.1f\n",
            static_cast<double>(total_codes_processed) / total_scans_performed);
        printf("  Reuse count: %zu\n", reuse_count);
        printf("  Memory usage: %.2f KB\n",
            get_workspace_memory_usage() / 1024.0);
    }
}

IVFJHQDistanceComputer::IVFJHQDistanceComputer(const IndexIVFJHQ& idx)
    : FlatCodesDistanceComputer(nullptr, idx.jhq.code_size)
    , index(idx)
    , query(nullptr)
    , current_list_no(-1)
{
    jhq_computer.reset(static_cast<JHQDistanceComputer*>(
        idx.jhq.get_FlatCodesDistanceComputer()));
}

float IVFJHQDistanceComputer::distance_to_code(const uint8_t* code)
{
    FAISS_THROW_IF_NOT_MSG(
        jhq_computer != nullptr, "JHQ computer not initialized");
    return jhq_computer->distance_to_code(code);
}

void IVFJHQDistanceComputer::set_query(const float* x)
{
    query = x;
    if (jhq_computer) {
        jhq_computer->set_query(x);
    }
}

void IVFJHQDistanceComputer::set_list(idx_t list_no)
{
    if (list_no != current_list_no && query != nullptr) {
        set_query_and_list(query, list_no);
    }
}

void IVFJHQDistanceComputer::set_query_and_list(const float* query_vec,
    idx_t list_no)
{
    query = query_vec;
    current_list_no = list_no;
    jhq_computer->set_query(query);
}

float IVFJHQDistanceComputer::operator()(idx_t i)
{
    if (index.direct_map.type != DirectMap::NoMap) {
        idx_t lo = index.direct_map.get(i);
        idx_t list_no = lo_listno(lo);
        idx_t offset = lo_offset(lo);

        set_list(list_no);

        const uint8_t* code = index.invlists->get_single_code(list_no, offset);
        return distance_to_code(code);
    } else {
        std::vector<float> db_vector(index.d);
        index.reconstruct(i, db_vector.data());

        if (query != nullptr) {
            return fvec_L2sqr(query, db_vector.data(), index.d);
        } else {
            FAISS_THROW_MSG("No query set and no direct map available");
        }
    }
}

float IVFJHQDistanceComputer::symmetric_distance(idx_t i, idx_t j)
{
    std::vector<float> xi(index.d), xj(index.d);
    index.reconstruct(i, xi.data());
    index.reconstruct(j, xj.data());
    return fvec_L2sqr(xi.data(), xj.data(), index.d);
}

void IVFJHQDistanceComputer::distances_batch_4(const idx_t idx0,
    const idx_t idx1,
    const idx_t idx2,
    const idx_t idx3,
    float& dis0,
    float& dis1,
    float& dis2,
    float& dis3)
{
    dis0 = (*this)(idx0);
    dis1 = (*this)(idx1);
    dis2 = (*this)(idx2);
    dis3 = (*this)(idx3);
}

IndexIVFJHQStats indexIVFJHQ_stats;

void IndexIVFJHQStats::reset()
{
    std::memset(this, 0, sizeof(*this));
}

void IndexIVFJHQStats::add(const IndexIVFJHQStats& other)
{
    nq += other.nq;
    nlist += other.nlist;
    ndis += other.ndis;
    n_early_termination += other.n_early_termination;
    n_primary_only += other.n_primary_only;
    n_full_residual += other.n_full_residual;
    quantization_time += other.quantization_time;
    jhq_search_time += other.jhq_search_time;
    total_search_time += other.total_search_time;
}

size_t estimate_ivfjhq_memory(int d,
    size_t n,
    size_t nlist,
    int M,
    const std::vector<int>& level_bits,
    bool use_tables)
{
    size_t total_bytes = 0;

    total_bytes += nlist * d * sizeof(float);

    for (int level = 0; level < static_cast<int>(level_bits.size()); ++level) {
        int K = 1 << level_bits[level];
        if (level == 0) {
            total_bytes += M * K * (d / M) * sizeof(float);
        } else {
            total_bytes += M * K * sizeof(float);
        }
    }

    total_bytes += d * d * sizeof(float);

    int jhq_bits = 0;
    for (int level = 0; level < static_cast<int>(level_bits.size()); ++level) {
        if (level == 0) {
            jhq_bits += M * level_bits[level];
        } else {
            jhq_bits += d * level_bits[level];
        }
    }
    size_t jhq_code_size = (jhq_bits + 7) / 8;
    total_bytes += n * jhq_code_size;

    total_bytes += n * sizeof(idx_t);

    if (use_tables) {
        size_t table_size_per_centroid = M * (1 << level_bits[0]);
        for (int level = 1; level < static_cast<int>(level_bits.size());
            ++level) {
            table_size_per_centroid += M * (d / M) * (1 << level_bits[level]);
        }
        total_bytes += nlist * table_size_per_centroid * sizeof(float);
    }

    total_bytes += 1024 * 1024;

    return total_bytes;
}

void write_index_ivf_jhq(const IndexIVFJHQ* idx, IOWriter* f)
{
    uint32_t magic = 0x4956464A;
    f->operator()(&magic, sizeof(magic), 1);

    int32_t d = static_cast<int32_t>(idx->d);
    int64_t ntotal = static_cast<int64_t>(idx->ntotal);
    int32_t nlist = static_cast<int32_t>(idx->nlist);
    int32_t nprobe = static_cast<int32_t>(idx->nprobe);
    uint32_t code_size = static_cast<uint32_t>(idx->code_size);

    f->operator()(&d, sizeof(d), 1);
    f->operator()(&ntotal, sizeof(ntotal), 1);
    f->operator()(&idx->is_trained, sizeof(idx->is_trained), 1);
    f->operator()(&idx->metric_type, sizeof(idx->metric_type), 1);
    f->operator()(&nlist, sizeof(nlist), 1);
    f->operator()(&nprobe, sizeof(nprobe), 1);
    f->operator()(&code_size, sizeof(code_size), 1);

    const faiss::IndexFlat* flat_quantizer = dynamic_cast<const faiss::IndexFlat*>(idx->quantizer);
    if (flat_quantizer && flat_quantizer->ntotal > 0) {
        uint64_t centroids_size_bytes = static_cast<uint64_t>(flat_quantizer->codes.size());
        uint64_t centroids_size_floats = centroids_size_bytes / sizeof(float);

        f->operator()(&centroids_size_floats, sizeof(centroids_size_floats), 1);

        if (centroids_size_floats > 0) {
            f->operator()(
                flat_quantizer->codes.data(),
                sizeof(uint8_t),
                centroids_size_bytes);
        }
    } else {
        uint64_t centroids_size_floats = 0;
        f->operator()(&centroids_size_floats, sizeof(centroids_size_floats), 1);
    }

    int32_t jhq_M = static_cast<int32_t>(idx->jhq.M);
    int32_t jhq_Ds = static_cast<int32_t>(idx->jhq.Ds);
    int32_t jhq_num_levels = static_cast<int32_t>(idx->jhq.num_levels);

    f->operator()(&jhq_M, sizeof(jhq_M), 1);
    f->operator()(&jhq_Ds, sizeof(jhq_Ds), 1);
    f->operator()(&jhq_num_levels, sizeof(jhq_num_levels), 1);

    uint32_t level_bits_size = static_cast<uint32_t>(idx->jhq.level_bits.size());
    f->operator()(&level_bits_size, sizeof(level_bits_size), 1);
    if (level_bits_size > 0) {
        f->operator()(
            idx->jhq.level_bits.data(), sizeof(int32_t), level_bits_size);
    }

    f->operator()(
        &idx->jhq.use_jl_transform, sizeof(idx->jhq.use_jl_transform), 1);
    f->operator()(
        &idx->jhq.use_analytical_init,
        sizeof(idx->jhq.use_analytical_init),
        1);
    f->operator()(
        &idx->jhq.default_oversampling,
        sizeof(idx->jhq.default_oversampling),
        1);
    f->operator()(
        &idx->jhq.is_rotation_trained,
        sizeof(idx->jhq.is_rotation_trained),
        1);

    if (idx->jhq.use_jl_transform && idx->jhq.is_rotation_trained) {
        uint64_t rot_size = static_cast<uint64_t>(idx->jhq.rotation_matrix.size());
        f->operator()(&rot_size, sizeof(rot_size), 1);
        if (rot_size > 0) {
            f->operator()(
                idx->jhq.rotation_matrix.data(), sizeof(float), rot_size);
        }
    } else {
        uint64_t rot_size = 0;
        f->operator()(&rot_size, sizeof(rot_size), 1);
    }

    for (int m = 0; m < idx->jhq.M; ++m) {
        for (int level = 0; level < idx->jhq.num_levels; ++level) {
            uint32_t codeword_size = static_cast<uint32_t>(idx->jhq.codewords[m][level].size());
            f->operator()(&codeword_size, sizeof(codeword_size), 1);
            if (codeword_size > 0) {
                f->operator()(
                    idx->jhq.codewords[m][level].data(),
                    sizeof(float),
                    codeword_size);
            }
        }
    }

    for (int m = 0; m < idx->jhq.M; ++m) {
        for (int level = 0; level < idx->jhq.num_levels - 1; ++level) {
            uint32_t codebook_size = static_cast<uint32_t>(
                idx->jhq.scalar_codebooks[m][level].size());
            f->operator()(&codebook_size, sizeof(codebook_size), 1);
            if (codebook_size > 0) {
                f->operator()(
                    idx->jhq.scalar_codebooks[m][level].data(),
                    sizeof(float),
                    codebook_size);
            }
        }
    }

    write_InvertedLists(idx->invlists, f);

    f->operator()(
        &idx->default_jhq_oversampling,
        sizeof(idx->default_jhq_oversampling),
        1);
    f->operator()(
        &idx->use_early_termination, sizeof(idx->use_early_termination), 1);
}

IndexIVFJHQ* read_index_ivf_jhq(IOReader* f)
{
    uint32_t magic;
    f->operator()(&magic, sizeof(magic), 1);

    if (magic != 0x4956464A) {
        FAISS_THROW_MSG("Invalid IVFJHQ magic number");
    }

    int32_t d, nlist, nprobe;
    uint32_t code_size;
    int64_t ntotal;
    bool is_trained;
    faiss::MetricType metric_type;

    f->operator()(&d, sizeof(d), 1);
    f->operator()(&ntotal, sizeof(ntotal), 1);
    f->operator()(&is_trained, sizeof(is_trained), 1);
    f->operator()(&metric_type, sizeof(metric_type), 1);
    f->operator()(&nlist, sizeof(nlist), 1);
    f->operator()(&nprobe, sizeof(nprobe), 1);
    f->operator()(&code_size, sizeof(code_size), 1);

    if (d <= 0 || d > 100000) {
        FAISS_THROW_MSG("Invalid dimension read from file: " + std::to_string(d));
    }
    if (nlist <= 0 || nlist > 1000000) {
        FAISS_THROW_MSG(
            "Invalid nlist read from file: " + std::to_string(nlist));
    }

    auto quantizer = std::make_unique<faiss::IndexFlatL2>(d);
    uint64_t centroids_size_floats;
    f->operator()(&centroids_size_floats, sizeof(centroids_size_floats), 1);

    if (centroids_size_floats > 0) {
        uint64_t expected_centroids_size = static_cast<uint64_t>(nlist) * d;

        if (centroids_size_floats != expected_centroids_size) {
            FAISS_THROW_MSG(
                "Centroids size mismatch! Read: " + std::to_string(centroids_size_floats) + ", Expected: " + std::to_string(expected_centroids_size));
        }

        uint64_t centroids_size_bytes = centroids_size_floats * sizeof(float);
        quantizer->codes.resize(centroids_size_bytes);
        f->operator()(
            quantizer->codes.data(), sizeof(uint8_t), centroids_size_bytes);

        quantizer->ntotal = nlist;
    }
    quantizer->is_trained = true;

    int32_t jhq_M, jhq_Ds, jhq_num_levels;
    f->operator()(&jhq_M, sizeof(jhq_M), 1);
    f->operator()(&jhq_Ds, sizeof(jhq_Ds), 1);
    f->operator()(&jhq_num_levels, sizeof(jhq_num_levels), 1);

    uint32_t level_bits_size;
    f->operator()(&level_bits_size, sizeof(level_bits_size), 1);

    std::vector<int> level_bits(level_bits_size);
    if (level_bits_size > 0) {
        f->operator()(level_bits.data(), sizeof(int32_t), level_bits_size);
    }

    bool use_jl_transform, use_analytical_init, is_rotation_trained;
    float default_oversampling;
    f->operator()(&use_jl_transform, sizeof(use_jl_transform), 1);
    f->operator()(&use_analytical_init, sizeof(use_analytical_init), 1);
    f->operator()(&default_oversampling, sizeof(default_oversampling), 1);
    f->operator()(&is_rotation_trained, sizeof(is_rotation_trained), 1);

    auto idx = std::make_unique<faiss::IndexIVFJHQ>(quantizer.release(),
        d,
        nlist,
        jhq_M,
        level_bits,
        use_jl_transform,
        default_oversampling,
        metric_type,
        true);

    idx->ntotal = ntotal;
    idx->is_trained = is_trained;
    idx->nprobe = nprobe;
    idx->code_size = code_size;

    uint64_t rot_size;
    f->operator()(&rot_size, sizeof(rot_size), 1);

    if (rot_size > 0) {
        idx->jhq.rotation_matrix.resize(rot_size);
        f->operator()(
            idx->jhq.rotation_matrix.data(), sizeof(float), rot_size);
    }
    idx->jhq.is_rotation_trained = is_rotation_trained;

    for (int m = 0; m < jhq_M; ++m) {
        for (int level = 0; level < jhq_num_levels; ++level) {
            uint32_t codeword_size;
            f->operator()(&codeword_size, sizeof(codeword_size), 1);

            idx->jhq.codewords[m][level].resize(codeword_size);
            if (codeword_size > 0) {
                f->operator()(
                    idx->jhq.codewords[m][level].data(),
                    sizeof(float),
                    codeword_size);
            }
        }
    }

    for (int m = 0; m < jhq_M; ++m) {
        for (int level = 0; level < jhq_num_levels - 1; ++level) {
            uint32_t codebook_size;
            f->operator()(&codebook_size, sizeof(codebook_size), 1);

            idx->jhq.scalar_codebooks[m][level].resize(codebook_size);
            if (codebook_size > 0) {
                f->operator()(
                    idx->jhq.scalar_codebooks[m][level].data(),
                    sizeof(float),
                    codebook_size);
            }
        }
    }

    delete idx->invlists;
    idx->invlists = read_InvertedLists(f);

    f->operator()(
        &idx->default_jhq_oversampling,
        sizeof(idx->default_jhq_oversampling),
        1);
    f->operator()(
        &idx->use_early_termination, sizeof(idx->use_early_termination), 1);

    if (idx->ntotal > 0 && idx->is_trained) {
        idx->jhq.is_trained = true;
        idx->jhq.ntotal = 0;

        idx->jhq.residual_bits_per_subspace = 0;
        for (int level = 1; level < idx->jhq.num_levels; ++level) {
            idx->jhq.residual_bits_per_subspace += static_cast<size_t>(idx->jhq.Ds) * idx->jhq.level_bits[level];
        }

        idx->rotated_centroids_computed = false;

        idx->jhq.memory_layout_initialized_ = false;
        idx->jhq.initialize_memory_layout();

        idx->optimize_for_search();

        if (idx->verbose) {
            std::cout
                << "Completed comprehensive state restoration after loading"
                << std::endl;
            std::cout << "  - JHQ residual bits per subspace: "
                      << idx->jhq.residual_bits_per_subspace << std::endl;
            std::cout << "  - Rotated centroids: "
                      << (idx->rotated_centroids_computed ? "ready"
                                                          : "on-demand")
                      << std::endl;
        }
    }

    return idx.release();
}

void IndexIVFJHQ::initialize_vector_mapping() const
{
    if (mapping_initialized)
        return;

    if (verbose) {
        printf("Initializing vector mapping for separated storage access...\n");
    }

    list_to_global_mapping.resize(nlist);
    idx_t global_idx = 0;

    for (idx_t global_vec_idx = 0; global_vec_idx < ntotal; ++global_vec_idx) {
        if (direct_map.type != DirectMap::NoMap) {
            idx_t list_offset = direct_map.get(global_vec_idx);
            idx_t list_no = lo_listno(list_offset);
            idx_t offset_in_list = lo_offset(list_offset);

            if (list_no >= 0 && list_no < nlist) {
                if (list_to_global_mapping[list_no].size() <= offset_in_list) {
                    list_to_global_mapping[list_no].resize(offset_in_list + 1, -1);
                }
                list_to_global_mapping[list_no][offset_in_list] = global_vec_idx;
            }
        }
    }

    mapping_initialized = true;

    if (verbose) {
        printf("Vector mapping initialized for %ld vectors\n", ntotal);
    }
}

idx_t IndexIVFJHQ::get_global_vector_index(idx_t list_no, idx_t offset) const
{
    if (!mapping_initialized) {
        initialize_vector_mapping();
    }

    if (list_no >= 0 && list_no < list_to_global_mapping.size() && offset >= 0 && offset < list_to_global_mapping[list_no].size()) {
        return list_to_global_mapping[list_no][offset];
    }
    return -1;
}

void IndexIVFJHQ::encode_to_separated_storage(idx_t n, const float* x)
{
    if (verbose && n > 1000) {
        printf("Encoding %zd vectors to separated storage...\n", n);
    }

    jhq.sa_encode(n, x, nullptr);

    if (verbose && n > 1000) {
        printf("Separated storage populated. Memory usage: %.2f MB\n",
            jhq.get_pre_decoded_memory_usage() / (1024.0 * 1024.0));
    }
}

void IndexIVFJHQ::initialize_vector_mapping_for_new_vectors(
    idx_t n,
    idx_t old_ntotal,
    const idx_t* coarse_idx,
    const idx_t* xids)
{

    if (!mapping_initialized) {
        list_to_global_mapping.resize(nlist);
        mapping_initialized = true;
    }

    for (idx_t i = 0; i < n; ++i) {
        idx_t list_no = coarse_idx[i];
        if (list_no >= 0 && list_no < nlist) {
            idx_t global_vec_idx = old_ntotal + i;

            size_t current_list_size = invlists->list_size(list_no);
            if (list_to_global_mapping[list_no].size() <= current_list_size) {
                list_to_global_mapping[list_no].resize(current_list_size + 1, -1);
            }

            list_to_global_mapping[list_no][current_list_size] = global_vec_idx;
        }
    }

    if (verbose && n > 1000) {
        printf("Vector mapping updated for %zd new vectors\n", n);
    }
}

void IndexIVFJHQ::encode_vectors_fallback(
    idx_t n,
    const float* x,
    const idx_t* list_nos,
    uint8_t* codes) const
{

    jhq.sa_encode(n, x, codes);
}
} // namespace faiss
