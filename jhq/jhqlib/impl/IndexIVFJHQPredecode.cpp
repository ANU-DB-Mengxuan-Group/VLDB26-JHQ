#include "IndexIVFJHQ.h"

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <atomic>
#include <mutex>

namespace faiss {

void IndexIVFJHQ::set_pre_decode_threshold(size_t threshold)
{
    if (threshold != pre_decode_threshold) {
        pre_decode_threshold = threshold;
        invalidate_pre_decoded_codes();
    }
}

void IndexIVFJHQ::enable_pre_decoded_codes(bool enable)
{
    if (enable != use_pre_decoded_codes) {
        use_pre_decoded_codes = enable;
        if (!enable) {
            invalidate_pre_decoded_codes();
        }
    }
}

void IndexIVFJHQ::initialize_pre_decoded_codes() const
{
    if (pre_decoded_codes_initialized.load(std::memory_order_acquire) || !use_pre_decoded_codes || !is_trained) {
        return;
    }

    std::lock_guard<std::mutex> lock(pre_decoded_codes_mutex_);
    if (pre_decoded_codes_initialized.load(std::memory_order_relaxed) || !use_pre_decoded_codes || !is_trained) {
        return;
    }

    list_pre_decoded_codes.resize(nlist);

    size_t eligible_lists = 0;

    for (size_t i = 0; i < nlist; ++i) {
        size_t list_size = invlists->list_size(i);
        if (list_size >= pre_decode_threshold) {
            eligible_lists++;
        }
    }

#pragma omp parallel for if (eligible_lists > 4)
    for (size_t i = 0; i < nlist; ++i) {
        size_t list_size = invlists->list_size(i);
        if (list_size >= pre_decode_threshold) {
            extract_list_codes(i);
        }
    }
    pre_decoded_codes_initialized.store(true, std::memory_order_release);
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

    const idx_t vec_idx = static_cast<idx_t>(vector_idx);
    uint8_t* primary_dest = pre_decoded.get_primary_codes_mutable(vec_idx);

    uint8_t* residual_dest = nullptr;
    if (jhq.num_levels > 1) {
        residual_dest = pre_decoded.get_residual_codes_mutable(vec_idx);
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

    if (jhq.num_levels > 1 && metric_type == METRIC_L2 && residual_dest != nullptr &&
        !pre_decoded.cross_terms.empty() && vector_idx < pre_decoded.cross_terms.size()) {
        pre_decoded.cross_terms[vector_idx] = jhq_internal::compute_cross_term_from_codes(
            jhq,
            primary_dest,
            residual_dest,
            pre_decoded.residual_subspace_stride,
            pre_decoded.residual_level_stride);
    }
}

bool IndexIVFJHQ::has_pre_decoded_codes_for_list(idx_t list_no) const
{
    if (!pre_decoded_codes_initialized.load(std::memory_order_acquire) || list_no >= nlist) {
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
    list_pre_decoded_codes.clear();
    pre_decoded_codes_initialized.store(false, std::memory_order_release);
}

size_t IndexIVFJHQ::get_pre_decoded_memory_usage() const
{
    size_t total = 0;
    for (const auto& pre_decoded : list_pre_decoded_codes) {
        total += pre_decoded.memory_usage();
    }
    return total;
}

} 
