#include "IndexIVFJHQ.h"

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <atomic>
#include <cstring>
#include <memory>
#include <mutex>

namespace faiss {

IndexIVFJHQ::~IndexIVFJHQ() = default;

void IndexIVFJHQ::extract_primary_codes_from_jhq_code(
    const uint8_t* jhq_code,
    uint8_t* primary_codes) const
{
    FAISS_THROW_IF_NOT_MSG(jhq_code != nullptr, "jhq_code is null");
    FAISS_THROW_IF_NOT_MSG(primary_codes != nullptr, "primary_codes is null");

    faiss::BitstringReader bit_reader(jhq_code, jhq.code_size);
    for (int m = 0; m < jhq.M; ++m) {
        uint32_t centroid_id = bit_reader.read(jhq.level_bits[0]);
        primary_codes[m] = static_cast<uint8_t>(centroid_id);
        bit_reader.i += jhq.residual_bits_per_subspace;
    }
}

thread_local IndexIVFJHQ::SearchWorkspace IndexIVFJHQ::search_workspace_;

IndexIVFJHQ::SearchWorkspace& IndexIVFJHQ::get_search_workspace() const
{
    return search_workspace_;
}

void IndexIVFJHQ::mark_single_level_adapter_dirty()
{
    single_level_adapter_dirty_.store(true, std::memory_order_release);
}

void IndexIVFJHQ::ensure_single_level_adapter_ready() const
{
    if (!should_use_single_level_adapter()) {
        return;
    }

    std::lock_guard<std::mutex> lock(single_level_adapter_mutex_);

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
        single_level_adapter_dirty_.store(true, std::memory_order_relaxed);
    }

    if (single_level_adapter_dirty_.load(std::memory_order_acquire)) {
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
        FAISS_THROW_IF_NOT_MSG(
            adapter.pq.ksub == jhq.primary_ksub(),
            "Primary codebook ksub mismatch for single-level adapter");
        FAISS_THROW_IF_NOT_MSG(
            adapter.pq.dsub == jhq.Ds,
            "Primary codebook dsub mismatch for single-level adapter");
        for (int m = 0; m < jhq.M; ++m) {
            const float* centroids = jhq.get_primary_centroids_ptr(m);
            const size_t centroid_size = static_cast<size_t>(adapter.pq.ksub) *
                static_cast<size_t>(adapter.pq.dsub);
            std::memcpy(
                adapter.pq.get_centroids(m, 0),
                centroids,
                centroid_size * sizeof(float));
        }

        adapter.code_size = adapter.pq.code_size;
        adapter.is_trained = is_trained;
        adapter.ntotal = ntotal;
        single_level_adapter_dirty_.store(false, std::memory_order_release);
    } else {
        single_level_adapter_->invlists = invlists;
        single_level_adapter_->ntotal = ntotal;
        single_level_adapter_->metric_type = metric_type;
        single_level_adapter_->parallel_mode = parallel_mode;
    }
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
    validate_parameters();
    code_size = jhq.code_size;

    if (own_invlists && invlists) {
        invlists->code_size = code_size;
    }

    is_trained = false;
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

} 
