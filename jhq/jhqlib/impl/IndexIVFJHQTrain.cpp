#include "IndexIVFJHQ.h"

#include <faiss/Clustering.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <mutex>
#include <vector>

namespace faiss {

void IndexIVFJHQ::train(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT_MSG(n > 0, "Training set cannot be empty");
    FAISS_THROW_IF_NOT_MSG(x != nullptr, "Training data cannot be null");
    FAISS_THROW_IF_NOT_MSG(
        n >= nlist,
        "Need at least as many training vectors as coarse clusters");

    train_q1(n, x, verbose, metric_type);
    train_jhq_on_originals(n, x);

    is_trained = true;

    initialize_optimized_layout();
    mark_single_level_adapter_dirty();
}

void IndexIVFJHQ::train_jhq_on_originals(idx_t n, const float* x)
{
    jhq.train(n, x);

    FAISS_THROW_IF_NOT_MSG(jhq.is_trained_(), "JHQ training failed");
    is_trained = true;
    mark_single_level_adapter_dirty();
}

void IndexIVFJHQ::train_jhq_with_precomputed_residuals(
    idx_t n,
    const float* residuals)
{
    train_jhq_on_originals(n, residuals);

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
    add_core(n, x, xids, coarse_idx);
    mark_single_level_adapter_dirty();
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
            add_core(i1 - i0,
                x + i0 * d,
                xids ? xids + i0 : nullptr,
                coarse_idx + i0,
                inverted_list_context);
        }
        return;
    }

    const idx_t old_ntotal = ntotal;
    encode_to_separated_storage(n, x);

    initialize_vector_mapping_for_new_vectors(n, old_ntotal, coarse_idx, xids);

    std::vector<uint8_t> codes(n * code_size);
    encode_vectors_fallback(n, x, coarse_idx, codes.data());

    DirectMapAdd dm_adder(direct_map, n, xids);

    for (idx_t i = 0; i < n; ++i) {
        idx_t list_no = coarse_idx[i];

        if (list_no < 0) {
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
    }

    ntotal += n;

    if (pre_decoded_codes_initialized.load(std::memory_order_acquire)) {
        invalidate_pre_decoded_codes();
        if (use_pre_decoded_codes && ntotal > pre_decode_threshold * nlist / 4) {
            initialize_pre_decoded_codes();
        }
    }

    mark_single_level_adapter_dirty();
}

void IndexIVFJHQ::encode_to_separated_storage(idx_t n, const float* x)
{
    const idx_t old_total = ntotal;
    jhq.ntotal = old_total;
    jhq.sa_encode(n, x, nullptr);
    jhq.ntotal = old_total + n;
}

void IndexIVFJHQ::initialize_vector_mapping_for_new_vectors(
    idx_t n,
    idx_t old_ntotal,
    const idx_t* coarse_idx,
    const idx_t* xids)
{
    (void)xids;
    if (!invlists) {
        return;
    }

    std::lock_guard<std::mutex> lock(mapping_mutex_);

    if (!mapping_initialized.load(std::memory_order_relaxed)) {
        list_to_global_mapping.resize(nlist);
        mapping_initialized.store(true, std::memory_order_release);
    }

    std::vector<size_t> next_offsets(nlist);
    for (size_t list_no = 0; list_no < nlist; ++list_no) {
        next_offsets[list_no] = invlists->list_size(list_no);
    }

    for (idx_t i = 0; i < n; ++i) {
        idx_t list_no = coarse_idx[i];
        if (list_no >= 0 && list_no < static_cast<idx_t>(nlist)) {
            const idx_t global_vec_idx = old_ntotal + i;
            const size_t list = static_cast<size_t>(list_no);
            const size_t offset = next_offsets[list]++;

            if (list_to_global_mapping[list].size() <= offset) {
                list_to_global_mapping[list].resize(offset + 1, -1);
            }
            list_to_global_mapping[list][offset] = global_vec_idx;
        }
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

} 
