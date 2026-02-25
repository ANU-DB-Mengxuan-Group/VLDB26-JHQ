#include "IndexIVFJHQ.h"

#include <faiss/impl/FaissAssert.h>

#include <cstring>
#include <vector>

namespace faiss {

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

} 
