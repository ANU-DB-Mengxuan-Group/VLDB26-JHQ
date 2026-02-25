#include "IndexIVFJHQ.h"

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/index_read_utils.h>
#include <faiss/impl/io_macros.h>
#include <faiss/invlists/DirectMap.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace faiss {

void IndexIVFJHQ::write(IOWriter* f) const
{
    write_index_ivf_jhq(this, f);
}

IndexIVFJHQ* IndexIVFJHQ::read(IOReader* f)
{
    return read_index_ivf_jhq(f);
}

void IndexIVFJHQ::cleanup() { }

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

namespace {

IndexJHQ make_jhq_model_only_for_ivf(const IndexJHQ& src)
{
    IndexJHQ model(
        src.d,
        src.M,
        src.level_bits,
        src.use_jl_transform,
        src.default_oversampling,
        src.use_analytical_init,
        src.verbose,
        src.metric_type);

    model.normalize_l2 = src.normalize_l2;
    model.use_kmeans_refinement = src.use_kmeans_refinement;
    model.kmeans_niter = src.kmeans_niter;
    model.kmeans_nredo = src.kmeans_nredo;
    model.kmeans_seed = src.kmeans_seed;
    model.sample_primary = src.sample_primary;
    model.sample_residual = src.sample_residual;
    model.random_sample_training = src.random_sample_training;

    model.is_trained = src.is_trained;
    model.is_rotation_trained = src.is_rotation_trained;
    model.use_bf16_rotation = src.use_bf16_rotation;
    model.rotation_matrix = src.rotation_matrix;
    model.rotation_matrix_bf16 = src.rotation_matrix_bf16;

    if (model.use_bf16_rotation && !model.rotation_matrix_bf16.empty()) {
        model.rotation_matrix.clear();
        model.rotation_matrix.shrink_to_fit();
    }

    const size_t primary_codeword_size =
        static_cast<size_t>(src.primary_ksub()) *
        static_cast<size_t>(src.Ds);
    for (int m = 0; m < src.M; ++m) {
        const float* centroids = src.get_primary_centroids_ptr(m);
        float* centroids_dst = model.get_primary_centroids_ptr_mutable(m);
        std::memcpy(
            centroids_dst,
            centroids,
            primary_codeword_size * sizeof(float));
    }

    if (src.num_levels > 1) {
        for (int m = 0; m < src.M; ++m) {
            for (int level = 1; level < src.num_levels; ++level) {
                const float* scalar_codebook =
                    src.get_scalar_codebook_ptr(m, level);
                float* scalar_codebook_dst =
                    model.get_scalar_codebook_ptr_mutable(m, level);
                const size_t ksub =
                    static_cast<size_t>(src.scalar_codebook_ksub(level));
                std::memcpy(
                    scalar_codebook_dst,
                    scalar_codebook,
                    ksub * sizeof(float));
            }
        }
    }

    
    model.ntotal = 0;
    model.codes.clear();
    model.separated_codes_.clear();
    model.memory_layout_initialized_ = false;
    model.initialize_memory_layout();

    return model;
}

} 

void write_index_ivf_jhq(const IndexIVFJHQ* idx, IOWriter* f)
{
    uint32_t magic = 0x4956464A;
    f->operator()(&magic, sizeof(magic), 1);
    uint32_t version = 1;
    f->operator()(&version, sizeof(version), 1);

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

    const IndexJHQ jhq_model = make_jhq_model_only_for_ivf(idx->jhq);
    write_index_jhq(&jhq_model, f);

    write_InvertedLists(idx->invlists, f);

    f->operator()(
        &idx->default_jhq_oversampling,
        sizeof(idx->default_jhq_oversampling),
        1);
    f->operator()(
        &idx->use_early_termination, sizeof(idx->use_early_termination), 1);
    int32_t parallel_mode = static_cast<int32_t>(idx->parallel_mode);
    f->operator()(&parallel_mode, sizeof(parallel_mode), 1);
    f->operator()(&idx->use_pre_decoded_codes, sizeof(idx->use_pre_decoded_codes), 1);
    uint64_t pre_decode_threshold = static_cast<uint64_t>(idx->pre_decode_threshold);
    f->operator()(&pre_decode_threshold, sizeof(pre_decode_threshold), 1);
}

IndexIVFJHQ* read_index_ivf_jhq(IOReader* f)
{
    uint32_t magic;
    f->operator()(&magic, sizeof(magic), 1);

    if (magic != 0x4956464A) {
        FAISS_THROW_MSG("Invalid IVFJHQ magic number");
    }
    uint32_t version = 0;
    f->operator()(&version, sizeof(version), 1);
    FAISS_THROW_IF_NOT_MSG(
        version == 1,
        "Unsupported IVFJHQ version (expected 1)");

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

    std::unique_ptr<IndexJHQ> jhq_model(read_index_jhq(f));
    FAISS_THROW_IF_NOT_MSG(jhq_model != nullptr, "Failed to read embedded JHQ");
    FAISS_THROW_IF_NOT_MSG(
        jhq_model->d == d,
        "Embedded JHQ dimension does not match IVFJHQ dimension");
    FAISS_THROW_IF_NOT_MSG(
        jhq_model->metric_type == metric_type,
        "Embedded JHQ metric does not match IVFJHQ metric");

    auto idx = std::make_unique<faiss::IndexIVFJHQ>(
        quantizer.release(),
        d,
        nlist,
        jhq_model->M,
        jhq_model->level_bits,
        jhq_model->use_jl_transform,
        jhq_model->default_oversampling,
        metric_type,
        true);

    idx->jhq = *jhq_model;
    idx->jhq.ntotal = 0;
    idx->jhq.codes.clear();
    idx->jhq.separated_codes_.clear();
    idx->jhq.memory_layout_initialized_ = false;
    idx->jhq.initialize_memory_layout();

    idx->ntotal = ntotal;
    idx->is_trained = is_trained;
    idx->nprobe = nprobe;
    FAISS_THROW_IF_NOT_MSG(
        code_size == static_cast<uint32_t>(idx->jhq.code_size),
        "IVFJHQ code_size mismatch with embedded JHQ");
    idx->code_size = idx->jhq.code_size;

    delete idx->invlists;
    idx->invlists = read_InvertedLists(f);
    if (idx->invlists) {
        idx->invlists->code_size = idx->code_size;
    }

    f->operator()(
        &idx->default_jhq_oversampling,
        sizeof(idx->default_jhq_oversampling),
        1);
    f->operator()(
        &idx->use_early_termination, sizeof(idx->use_early_termination), 1);
    int32_t parallel_mode = 0;
    f->operator()(&parallel_mode, sizeof(parallel_mode), 1);
    idx->parallel_mode = parallel_mode;
    f->operator()(&idx->use_pre_decoded_codes, sizeof(idx->use_pre_decoded_codes), 1);
    uint64_t pre_decode_threshold = 0;
    f->operator()(&pre_decode_threshold, sizeof(pre_decode_threshold), 1);
    idx->pre_decode_threshold = static_cast<size_t>(pre_decode_threshold);

    if (idx->ntotal > 0 && idx->is_trained) {
        idx->rotated_centroids_computed.store(false, std::memory_order_release);
        idx->mark_single_level_adapter_dirty();
        idx->optimize_for_search();
    }

    return idx.release();
}

void IndexIVFJHQ::initialize_vector_mapping() const
{
    if (mapping_initialized.load(std::memory_order_acquire))
        return;

    std::lock_guard<std::mutex> lock(mapping_mutex_);
    if (mapping_initialized.load(std::memory_order_relaxed))
        return;

    list_to_global_mapping.resize(nlist);

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

    mapping_initialized.store(true, std::memory_order_release);
}

idx_t IndexIVFJHQ::get_global_vector_index(idx_t list_no, idx_t offset) const
{
    if (!mapping_initialized.load(std::memory_order_acquire)) {
        initialize_vector_mapping();
    }

    if (list_no >= 0 && list_no < list_to_global_mapping.size() && offset >= 0 && offset < list_to_global_mapping[list_no].size()) {
        return list_to_global_mapping[list_no][offset];
    }
    return -1;
}

} 
