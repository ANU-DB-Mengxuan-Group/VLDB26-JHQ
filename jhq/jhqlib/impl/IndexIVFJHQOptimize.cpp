#include "IndexIVFJHQ.h"

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <vector>

namespace faiss {

void IndexIVFJHQ::reset()
{
    IndexIVF::reset();
    jhq.reset();
    rotated_coarse_centroids.clear();
    rotated_centroids_computed.store(false, std::memory_order_release);
    is_trained = false;
    single_level_adapter_.reset();
    single_level_adapter_dirty_.store(true, std::memory_order_release);
}

void IndexIVFJHQ::merge_from(Index& otherIndex, idx_t add_id)
{
    check_compatible_for_merge(otherIndex);

    IndexIVFJHQ* other = static_cast<IndexIVFJHQ*>(&otherIndex);

    invlists->merge_from(other->invlists, add_id);

    ntotal += other->ntotal;
    other->ntotal = 0;

    if (jhq.has_optimized_layout()) {
        jhq.invalidate_memory_layout();
    }

    invalidate_pre_decoded_codes();
    mark_single_level_adapter_dirty();
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
}

void IndexIVFJHQ::set_early_termination(bool enable)
{
    use_early_termination = enable;
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
    if (rotated_centroids_computed.load(std::memory_order_acquire))
        return;

    std::lock_guard<std::mutex> lock(rotated_centroids_mutex_);
    if (rotated_centroids_computed.load(std::memory_order_relaxed))
        return;

    rotated_coarse_centroids.resize(nlist * d);

    std::vector<float> centroids_flat(nlist * d);
    for (size_t i = 0; i < nlist; ++i) {
        quantizer->reconstruct(i, centroids_flat.data() + i * d);
    }

    jhq.apply_jl_rotation(
        nlist, centroids_flat.data(), rotated_coarse_centroids.data());
    rotated_centroids_computed.store(true, std::memory_order_release);
}

void IndexIVFJHQ::optimize_for_search()
{
    if (!is_trained) {
        return;
    }

    if (!jhq.is_trained_()) {
        jhq.is_trained = true;
    }

    if (jhq.residual_bits_per_subspace == 0 && jhq.num_levels > 1) {
        for (int level = 1; level < jhq.num_levels; ++level) {
            jhq.residual_bits_per_subspace += static_cast<size_t>(jhq.Ds) * jhq.level_bits[level];
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

    if (should_use_single_level_adapter()) {
        if (use_pre_decoded_codes) {
            enable_pre_decoded_codes(false);
        }
        ensure_single_level_adapter_ready();
    } else if (use_pre_decoded_codes && ntotal > 0) {
        initialize_pre_decoded_codes();
    }
}

void IndexIVFJHQ::print_stats() const
{
    (void)this;
}

} 
