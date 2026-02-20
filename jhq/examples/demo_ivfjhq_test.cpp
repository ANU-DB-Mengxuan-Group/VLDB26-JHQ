#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <thread>
#include <vector>

#include <omp.h>

#include "config.h"
#include "IndexIVFJHQ.h"
#include <faiss/IndexFlat.h>
#include <faiss/impl/io.h>

// Dataset to test
const std::vector<std::string> DATASETS = {
    "dbpedia-openai3-text-embedding-3-large-1024-100K",
    "dbpedia-openai3-text-embedding-3-large-3072-100K",
};

const size_t K = 10; // Number of nearest neighbors to retrieve

struct IVFJHQQueryParam {
    size_t nprobe;
    float jhq_oversampling_factor;

    IVFJHQQueryParam(size_t np = 32, float os = 4.0f)
        : nprobe(np)
        , jhq_oversampling_factor(os)
    {
    }
};

struct IVFJHQParams {
    std::string name;
    size_t nlist;
    int M;
    std::vector<int> level_bits;
    bool use_jl_transform;
    float jhq_oversampling;
    bool use_early_termination;
    bool compute_residuals;

    // Clustering control parameters
    bool use_kmeans_refinement;
    int kmeans_niter;
    int kmeans_seed;

    std::vector<IVFJHQQueryParam> query_params;

    IVFJHQParams(const std::string& n, size_t nl, int m, const std::vector<int>& bits,
        bool jl = true, float os = 4.0f,
        bool early_term = true, bool residuals = true,
        bool use_kmeans = false, int niter = 25, int seed = 1234)
        : name(n)
        , nlist(nl)
        , M(m)
        , level_bits(bits)
        , use_jl_transform(jl)
        , jhq_oversampling(os)
        , use_early_termination(early_term)
        , compute_residuals(residuals)
        , use_kmeans_refinement(use_kmeans)
        , kmeans_niter(niter)
        , kmeans_seed(seed)
    {
    }

    // Generate a unique filename for this configuration
    std::string getIndexFilename() const
    {
        std::stringstream ss;
        ss << "ivfjhq_nlist" << nlist << "_M" << M;

        // Add level bits
        ss << "_bits";
        for (int bits : level_bits) {
            ss << "_" << bits;
        }

        // Add clustering method
        if (use_kmeans_refinement) {
            ss << "_kmeans" << kmeans_niter << "_seed" << kmeans_seed;
        } else {
            ss << "_analytical";
        }

        // Add JL transform info
        if (use_jl_transform) {
            ss << "_jl";
        }

        ss << ".index";
        return ss.str();
    }

    // Get clustering method string for folder names
    std::string getClusteringMethodName() const
    {
        return use_kmeans_refinement ? "analytical_kmeans" : "analytical_only";
    }
};

// Memory usage tracking
size_t getCurrentMemoryUsage()
{
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string label, value, unit;
            iss >> label >> value >> unit;
            return std::stoull(value) * 1024; // Convert KB to bytes
        }
    }
    return 0;
}

void printMemoryUsage(const std::string& stage)
{
    size_t memory_bytes = getCurrentMemoryUsage();
    double memory_gb = static_cast<double>(memory_bytes) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "[" << stage << "] --> Current Memory Usage: "
              << std::fixed << std::setprecision(2) << memory_gb << " GB" << std::endl;
}

// Function to read float vectors from a .fvecs file
std::vector<std::vector<float>> readFvecs(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<float>> data;
    int32_t dim;
    size_t vector_count = 0;

    while (file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid dimension (" + std::to_string(dim) + ") in file: " + filename);
        }
        std::vector<float> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float))) {
            throw std::runtime_error("Error reading vector data from file: " + filename);
        }
        data.push_back(std::move(vec));
        vector_count++;

        if (vector_count % 10000 == 0) {
            std::cout << "Reading vectors: " << vector_count << "\r" << std::flush;
        }
    }

    std::cout << "readFvecs: Read " << vector_count << " vectors from " << filename << std::endl;
    return data;
}

// Function to read integer vectors from a .ivecs file
std::vector<std::vector<int>> readIvecs(const std::string& filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Error opening file: " + filename);
    }

    std::vector<std::vector<int>> data;
    int32_t dim;
    size_t vector_count = 0;

    while (file.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        if (dim <= 0) {
            throw std::runtime_error("Invalid dimension (" + std::to_string(dim) + ") in file: " + filename);
        }
        std::vector<int> vec(dim);
        if (!file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int))) {
            throw std::runtime_error("Error reading int vector data from file: " + filename);
        }
        data.push_back(std::move(vec));
        vector_count++;
    }

    std::cout << "readIvecs: Read " << vector_count << " vectors from " << filename << std::endl;
    return data;
}

// Function to calculate recall@K
double calculateRecall(const std::vector<std::vector<int>>& ground_truth,
    const std::vector<std::vector<faiss::idx_t>>& results)
{
    if (ground_truth.size() != results.size()) {
        std::cerr << "Warning: ground truth size (" << ground_truth.size()
                  << ") doesn't match results size (" << results.size() << ")" << std::endl;
        return 0.0;
    }

    double total_recall = 0.0;
    for (size_t i = 0; i < ground_truth.size(); ++i) {
        if (!ground_truth[i].empty() && !results[i].empty()) {
            size_t found = 0;
            size_t check_size = std::min(K, results[i].size());

            for (size_t j = 0; j < check_size; ++j) {
                if (std::find(ground_truth[i].begin(), ground_truth[i].end(),
                        static_cast<int>(results[i][j]))
                    != ground_truth[i].end()) {
                    found++;
                }
            }
            total_recall += static_cast<double>(found) / std::min(K, ground_truth[i].size());
        }
    }
    return total_recall / ground_truth.size();
}

namespace IndexSaveLoad {

bool saveIndex(const faiss::IndexIVFJHQ* index, const std::string& filename)
{
    try {
        std::cout << "Saving index to: " << filename << std::endl;

        // Ensure directory exists
        std::filesystem::path filepath(filename);
        std::filesystem::create_directories(filepath.parent_path());

        FILE* file = fopen(filename.c_str(), "wb");
        if (!file) {
            std::cerr << "Failed to open file for writing: " << filename << std::endl;
            return false;
        }

        faiss::FileIOWriter writer(file);

        write_index_ivf_jhq(index, &writer);

        // Force write to disk
        fflush(file);
        fsync(fileno(file));
        fclose(file);

        // Verify the file was actually written
        std::ifstream check_file(filename, std::ios::binary | std::ios::ate);
        if (check_file.is_open()) {
            auto file_size = check_file.tellg();
            check_file.close();

            if (file_size < 1000) { // Should be much larger than 1KB
                std::cerr << "ERROR: File seems too small!" << std::endl;
                return false;
            }
        }

        std::cout << "Index saved successfully!" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving index: " << e.what() << std::endl;
        return false;
    }
}

std::unique_ptr<faiss::IndexJHQ> loadIndex(const std::string& filename)
{
    try {
        std::cout << "Loading index from: " << filename << std::endl;

        FILE* file = fopen(filename.c_str(), "rb");
        if (!file) {
            std::cout << "Index file not found: " << filename << std::endl;
            return nullptr;
        }

        faiss::FileIOReader reader(file);
        auto index = std::unique_ptr<faiss::IndexJHQ>(
            faiss::read_index_ivf_jhq(&reader));
        fclose(file);

        std::cout << "Index loaded successfully!" << std::endl;
        std::cout << "  - Vectors: " << index->ntotal << std::endl;
        std::cout << "  - Dimension: " << index->d << std::endl;
        std::cout << "  - Lists: " << index->nlist << std::endl;

        return index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        return nullptr;
    }
}
}

std::unique_ptr<faiss::IndexJHQ> loadIndex(const std::string& filename)
{
    try {
        std::cout << "Loading index from: " << filename << std::endl;

        FILE* file = fopen(filename.c_str(), "rb");
        if (!file) {
            std::cout << "Index file not found: " << filename << std::endl;
            return nullptr;
        }

        faiss::FileIOReader reader(file);
        // Use the specific IndexJHQ read method directly
        auto index = std::unique_ptr<faiss::IndexJHQ>(
            faiss::read_index_ivf_jhq(&reader));
        fclose(file);

        std::cout << "Index loaded successfully!" << std::endl;
        std::cout << "  - Vectors: " << index->ntotal << std::endl;
        std::cout << "  - Dimension: " << index->d << std::endl;
        std::cout << "  - Lists: " << index->nlist << std::endl;

        return index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        return nullptr;
    }
}

// CSV file managers for separate outputs
struct CSVFileManager {
    std::ofstream analytical_recall_stream;
    std::ofstream analytical_metrics_stream;
    std::ofstream kmeans_recall_stream;
    std::ofstream kmeans_metrics_stream;

    bool initialized = false;

    bool initialize(const std::filesystem::path& result_path, const std::string& dataset_name)
    {
        // Create method-specific directories
        std::filesystem::path analytical_path = result_path / "analytical_only";
        std::filesystem::path kmeans_path = result_path / "analytical_kmeans";

        std::filesystem::create_directories(analytical_path);
        std::filesystem::create_directories(kmeans_path);

        // Open analytical files
        analytical_recall_stream.open(analytical_path / (dataset_name + "_analytical_only_recall_results.csv"));
        analytical_metrics_stream.open(analytical_path / (dataset_name + "_analytical_only_detailed_metrics.csv"));

        // Open k-means files
        kmeans_recall_stream.open(kmeans_path / (dataset_name + "_analytical_kmeans_recall_results.csv"));
        kmeans_metrics_stream.open(kmeans_path / (dataset_name + "_analytical_kmeans_detailed_metrics.csv"));

        if (!analytical_recall_stream || !analytical_metrics_stream || !kmeans_recall_stream || !kmeans_metrics_stream) {
            return false;
        }

        // Write headers for all files
        std::string recall_header = "Config_Name,nlist,nprobe,M,JHQ_Level_Bits,JHQ_Levels,JHQ_Oversampling,"
                                    "Recall@"
            + std::to_string(K) + ",QPS,Train_Time_s,Encode_Time_s,Total_Time_s,"
                                  "Use_Early_Term,Compute_Residuals,Search_Time_s,"
                                  "Clustering_Mode,KMeans_Iters,KMeans_Seed\n";

        std::string metrics_header = "Config_Name,nlist,M,JHQ_Subspace_Dim,Total_JHQ_Centroids,"
                                     "JHQ_Bits,Total_Bits,Memory_MB,Compression_Ratio,"
                                     "Clustering_Mode,KMeans_Iters\n";

        analytical_recall_stream << recall_header;
        analytical_metrics_stream << metrics_header;
        kmeans_recall_stream << recall_header;
        kmeans_metrics_stream << metrics_header;

        initialized = true;
        return true;
    }

    std::ofstream& getRecallStream(bool use_kmeans)
    {
        return use_kmeans ? kmeans_recall_stream : analytical_recall_stream;
    }

    std::ofstream& getMetricsStream(bool use_kmeans)
    {
        return use_kmeans ? kmeans_metrics_stream : analytical_metrics_stream;
    }

    void close()
    {
        if (analytical_recall_stream.is_open())
            analytical_recall_stream.close();
        if (analytical_metrics_stream.is_open())
            analytical_metrics_stream.close();
        if (kmeans_recall_stream.is_open())
            kmeans_recall_stream.close();
        if (kmeans_metrics_stream.is_open())
            kmeans_metrics_stream.close();
    }

    void printSavedPaths(const std::filesystem::path& result_path, const std::string& dataset_name)
    {
        std::cout << "Results saved to:" << std::endl;
        std::cout << "  Analytical-only method:" << std::endl;
        std::cout << "    - Recall: " << (result_path / "analytical_only" / (dataset_name + "_analytical_only_recall_results.csv")) << std::endl;
        std::cout << "    - Metrics: " << (result_path / "analytical_only" / (dataset_name + "_analytical_only_detailed_metrics.csv")) << std::endl;
        std::cout << "  K-means method:" << std::endl;
        std::cout << "    - Recall: " << (result_path / "analytical_kmeans" / (dataset_name + "_analytical_kmeans_recall_results.csv")) << std::endl;
        std::cout << "    - Metrics: " << (result_path / "analytical_kmeans" / (dataset_name + "_analytical_kmeans_detailed_metrics.csv")) << std::endl;
    }
};

class IVFJHQ {
public:
    IVFJHQ(const IVFJHQParams& params, const std::string& dataset_name)
        : params_(params)
        , dataset_name_(dataset_name)
    {
        std::cout << "IVFJHQ Configuration:" << std::endl;
        std::cout << "  Name: " << params_.name << std::endl;
        std::cout << "  nlist (coarse clusters): " << params_.nlist << std::endl;
        std::cout << "  M (JHQ subspaces): " << params_.M << std::endl;
        std::cout << "  JHQ level bits: [";
        for (size_t i = 0; i < params_.level_bits.size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << params_.level_bits[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  JL Transform: " << (params_.use_jl_transform ? "enabled" : "disabled") << std::endl;
        std::cout << "  JHQ Oversampling: " << params_.jhq_oversampling << std::endl;
        std::cout << "  Use Early Termination: " << (params_.use_early_termination ? "enabled" : "disabled") << std::endl;
        std::cout << "  Compute Residuals: " << (params_.compute_residuals ? "enabled" : "disabled") << std::endl;
        std::cout << "  Clustering Mode: " << (params_.use_kmeans_refinement ? "Analytical + K-means" : "Analytical Only") << std::endl;
        if (params_.use_kmeans_refinement) {
            std::cout << "  K-means iterations: " << params_.kmeans_niter << std::endl;
            std::cout << "  K-means seed: " << params_.kmeans_seed << std::endl;
        }
    }

private:
    // Generate time file path based on index file path
    std::string getTimeFilePath(const std::filesystem::path& index_file) const
    {
        std::string time_file = index_file.string();
        // Replace .index extension with .time
        size_t pos = time_file.find_last_of('.');
        if (pos != std::string::npos) {
            time_file = time_file.substr(0, pos) + ".time";
        } else {
            time_file += ".time";
        }
        return time_file;
    }

    // Save indexing time to file
    void saveIndexingTime(const std::filesystem::path& index_file) const
    {
        std::string time_file = getTimeFilePath(index_file);
        std::ofstream file(time_file);
        if (!file) {
            std::cerr << "Warning: Could not save indexing time to: " << time_file << std::endl;
            return;
        }

        // Save both train and encode times
        file << train_time_ << std::endl;
        file << encode_time_ << std::endl;

        std::cout << "Saved indexing times to: " << std::filesystem::path(time_file).filename() << std::endl;
        std::cout << "  Train time: " << train_time_ << "s" << std::endl;
        std::cout << "  Encode time: " << encode_time_ << "s" << std::endl;
    }

    // Load indexing time from file
    void loadIndexingTime(const std::filesystem::path& index_file)
    {
        std::string time_file = getTimeFilePath(index_file);
        std::ifstream file(time_file);
        if (!file) {
            std::cout << "Warning: Could not load indexing time from: " << std::filesystem::path(time_file).filename()
                      << ". Setting times to 0." << std::endl;
            train_time_ = 0.0;
            encode_time_ = 0.0;
            return;
        }

        file >> train_time_;
        file >> encode_time_;

        std::cout << "Loaded indexing times from: " << std::filesystem::path(time_file).filename() << std::endl;
        std::cout << "  Train time: " << train_time_ << "s" << std::endl;
        std::cout << "  Encode time: " << encode_time_ << "s" << std::endl;
    }

public:
    // Try to load existing index, otherwise train new one
    bool tryLoadOrTrain(const std::vector<std::vector<float>>& X,
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<int>>& cluster_ids,
        const std::filesystem::path& index_cache_dir)
    {
        // Try to load existing index first
        std::filesystem::path index_file = index_cache_dir / params_.getIndexFilename();

        index_ = IndexSaveLoad::loadIndex(index_file.string());

        if (index_) {
            std::cout << "Using pre-trained index!" << std::endl;

            dim_ = index_->d;

            // Load the associated indexing times
            loadIndexingTime(index_file);
            return true;
        }

        // Train new index if loading failed
        std::cout << "Training new index..." << std::endl;
        bool success = fit(X, centroids, cluster_ids);

        if (success) {
            // Save the trained index
            std::filesystem::create_directories(index_cache_dir);
            if (IndexSaveLoad::saveIndex(index_.get(), index_file.string())) {
                // Save the indexing times
                saveIndexingTime(index_file);
            } else {
                std::cout << "Warning: Failed to save index to cache" << std::endl;
            }
        }

        return success;
    }

    bool fit(const std::vector<std::vector<float>>& X,
        const std::vector<std::vector<float>>& centroids,
        const std::vector<std::vector<int>>& cluster_ids)
    {
        if (X.empty()) {
            throw std::runtime_error("Input data is empty");
        }
        if (centroids.empty()) {
            throw std::runtime_error("Precomputed centroids data is empty");
        }
        if (cluster_ids.empty()) {
            throw std::runtime_error("Precomputed cluster IDs data is empty");
        }
        if (params_.nlist != centroids.size()) {
            throw std::runtime_error("Mismatch between params.nlist (" + std::to_string(params_.nlist)
                + ") and number of loaded centroids (" + std::to_string(centroids.size()) + ")");
        }

        dim_ = static_cast<int>(X[0].size());
        size_t n_data = X.size();

        std::cout << "IVFJHQ Training (JHQ on original vectors):" << std::endl;
        std::cout << "  Dataset: " << dataset_name_ << std::endl;
        std::cout << "  Vectors: " << n_data << std::endl;
        std::cout << "  Dimension: " << dim_ << std::endl;

        if (dim_ <= 0) {
            throw std::runtime_error("Invalid dimension: " + std::to_string(dim_));
        }

        if (dim_ % params_.M != 0) {
            throw std::runtime_error("Dimension " + std::to_string(dim_) + " must be divisible by M=" + std::to_string(params_.M));
        }

        std::cout << "IVFJHQ: Creating index with precomputed coarse quantizer..." << std::endl;

        // Flatten base data
        std::vector<float> data_flat(n_data * dim_);
        for (size_t i = 0; i < n_data; ++i) {
            std::memcpy(&data_flat[i * dim_], X[i].data(), dim_ * sizeof(float));
        }

        auto quantizer = std::make_unique<faiss::IndexFlatL2>(dim_);

        if (centroids.size() != params_.nlist) {
            throw std::runtime_error("Centroids size (" + std::to_string(centroids.size()) + ") doesn't match nlist (" + std::to_string(params_.nlist) + ")");
        }

        for (size_t i = 0; i < centroids.size(); ++i) {
            if (centroids[i].size() != static_cast<size_t>(dim_)) {
                throw std::runtime_error("Centroid " + std::to_string(i) + " has wrong dimension: " + std::to_string(centroids[i].size()) + " (expected: " + std::to_string(dim_) + ")");
            }
        }

        std::vector<float> centroids_flat;
        centroids_flat.reserve(centroids.size() * dim_);

        for (size_t i = 0; i < centroids.size(); ++i) {
            centroids_flat.insert(centroids_flat.end(), centroids[i].begin(), centroids[i].end());
        }

        if (centroids_flat.size() != centroids.size() * dim_) {
            throw std::runtime_error("Flattened centroids size mismatch: " + std::to_string(centroids_flat.size()) + " vs expected " + std::to_string(centroids.size() * dim_));
        }

        size_t bytes_size = centroids_flat.size() * sizeof(float);
        quantizer->codes.resize(bytes_size);

        std::memcpy(quantizer->codes.data(), centroids_flat.data(), bytes_size);

        quantizer->ntotal = centroids.size();
        quantizer->is_trained = true;

        if (quantizer->codes.size() != quantizer->ntotal * quantizer->d * sizeof(float)) {
            throw std::runtime_error("Quantizer codes size is still wrong after byte assignment!");
        }

        // Step 2: Create IVFJHQ index with the pre-trained quantizer
        index_ = std::make_unique<faiss::IndexJHQ>(
            quantizer.release(), dim_, params_.nlist, params_.M, params_.level_bits,
            params_.use_jl_transform, params_.jhq_oversampling,
            faiss::METRIC_L2, true);

        index_->set_clustering_parameters(
            params_.use_kmeans_refinement,
            params_.kmeans_niter,
            params_.kmeans_seed);

        auto train_start = std::chrono::high_resolution_clock::now();

        index_->train_jhq_on_originals(n_data, data_flat.data());

        auto train_end = std::chrono::high_resolution_clock::now();
        train_time_ = std::chrono::duration<double>(train_end - train_start).count();

        if (!index_->is_trained) {
            throw std::runtime_error("IVFJHQ training failed");
        }
        std::cout << "JHQ training completed in " << train_time_ << "s" << std::endl;

        std::cout << "Optimizing for search..." << std::endl;
        auto opt_start = std::chrono::high_resolution_clock::now();

        index_->optimize_for_search();

        auto opt_end = std::chrono::high_resolution_clock::now();
        auto opt_time = std::chrono::duration<double>(opt_end - opt_start).count();
        std::cout << "Search optimization completed in " << opt_time << "s" << std::endl;

        std::vector<faiss::idx_t> assign(n_data);
        for (size_t i = 0; i < n_data; ++i) {
            assign[i] = static_cast<faiss::idx_t>(cluster_ids[i][0]);
        }

        auto add_start = std::chrono::high_resolution_clock::now();
        index_->add_with_precomputed_assignments(n_data, data_flat.data(), nullptr, assign.data());
        auto add_end = std::chrono::high_resolution_clock::now();
        encode_time_ = std::chrono::duration<double>(add_end - add_start).count();

        std::cout << "Added " << index_->ntotal << " vectors in " << encode_time_ << "s" << std::endl;

        return true;
    }

    void setQueryArguments(const IVFJHQQueryParam& param)
    {
        current_query_param_ = param;
        std::cout << "IVFJHQ Query Parameters:" << std::endl;
        std::cout << "  nprobe: " << param.nprobe << std::endl;
        std::cout << "  JHQ Oversampling: " << param.jhq_oversampling_factor << std::endl;
    }

    std::vector<std::vector<faiss::idx_t>> queryBatch(
        const std::vector<std::vector<float>>& query_vecs,
        size_t k)
    {
        if (query_vecs.empty()) {
            return {};
        }

        size_t n_queries = query_vecs.size();

        const int correct_dim = index_ ? index_->d : 0;

        for (size_t i = 0; i < std::min(n_queries, static_cast<size_t>(3)); ++i) {
            if (query_vecs[i].size() != static_cast<size_t>(correct_dim)) {
                std::cout << "ERROR: Query vector " << i << " has size " << query_vecs[i].size()
                          << " but expected " << correct_dim << std::endl;
                throw std::runtime_error("Query vector has incorrect dimensionality");
            }
        }

        for (const auto& query_vec : query_vecs) {
            if (query_vec.size() != static_cast<size_t>(correct_dim)) {
                std::cout << "ERROR: Found query vector with size " << query_vec.size()
                          << " (expected " << correct_dim << ")" << std::endl;
                throw std::runtime_error("Query vector has incorrect dimensionality");
            }
        }

        std::vector<float> queries_flat(n_queries * dim_);
        for (size_t i = 0; i < n_queries; ++i) {
            std::memcpy(&queries_flat[i * dim_], query_vecs[i].data(), dim_ * sizeof(float));
        }

        std::vector<float> distances(n_queries * k);
        std::vector<faiss::idx_t> labels(n_queries * k);

        faiss::IVFJHQSearchParameters search_params;
        search_params.nprobe = current_query_param_.nprobe;
        search_params.jhq_oversampling_factor = current_query_param_.jhq_oversampling_factor;
        search_params.use_early_termination = params_.use_early_termination;
        search_params.compute_residuals = params_.compute_residuals;

        index_->search(n_queries, queries_flat.data(), k, distances.data(), labels.data(), &search_params);

        std::vector<std::vector<faiss::idx_t>> results(n_queries);
        for (size_t i = 0; i < n_queries; ++i) {
            results[i].assign(
                labels.begin() + i * k,
                labels.begin() + (i + 1) * k);
        }

        return results;
    }

    struct AlgorithmInfo {
        std::string algorithm_name = "IVFJHQ";
        int dimension;
        size_t nlist;
        int num_jhq_subspaces;
        int jhq_subspace_dimension;
        std::vector<int> jhq_bits_per_level;
        int jhq_total_bits_per_vector;
        int total_bits_per_vector;
        bool uses_jl_transform;
        float jhq_oversampling_factor;
        size_t num_database_vectors;
        size_t total_jhq_centroids;
        double memory_usage_mb;
        float compression_ratio;
    };

    AlgorithmInfo getAlgorithmInfo() const
    {
        AlgorithmInfo info;
        info.dimension = dim_;
        info.nlist = params_.nlist;
        info.num_jhq_subspaces = params_.M;
        info.jhq_subspace_dimension = dim_ / params_.M;
        info.jhq_bits_per_level = params_.level_bits;

        info.jhq_total_bits_per_vector = 0;
        for (int l = 0; l < static_cast<int>(params_.level_bits.size()); ++l) {
            if (l == 0) {
                info.jhq_total_bits_per_vector += params_.M * params_.level_bits[l];
            } else {
                info.jhq_total_bits_per_vector += dim_ * params_.level_bits[l];
            }
        }

        int coarse_bits = 0;
        size_t nl = params_.nlist - 1;
        while (nl > 0) {
            coarse_bits++;
            nl >>= 1;
        }
        info.total_bits_per_vector = info.jhq_total_bits_per_vector + coarse_bits;

        info.uses_jl_transform = params_.use_jl_transform;
        info.jhq_oversampling_factor = params_.jhq_oversampling;

        if (index_) {
            info.num_database_vectors = static_cast<size_t>(index_->ntotal);
            info.memory_usage_mb = index_->get_memory_usage() / (1024.0 * 1024.0);
            info.compression_ratio = index_->get_compression_ratio();
        }

        info.total_jhq_centroids = 0;
        for (int l = 0; l < static_cast<int>(params_.level_bits.size()); ++l) {
            info.total_jhq_centroids += static_cast<size_t>(params_.M) * (1 << params_.level_bits[l]);
        }

        return info;
    }

    double getIndexingTime() const
    {
        return train_time_ + encode_time_;
    }

    double getTrainTime() const { return train_time_; }
    double getEncodeTime() const { return encode_time_; }

private:
    std::unique_ptr<faiss::IndexJHQ> index_;
    IVFJHQParams params_;
    std::string dataset_name_;
    int dim_;
    IVFJHQQueryParam current_query_param_;
    double train_time_ = 0.0;
    double encode_time_ = 0.0;
};

std::vector<std::pair<size_t, std::string>> detectAvailableCentroids(const std::filesystem::path& base_path, const std::string& dataset_name)
{
    std::vector<std::pair<size_t, std::string>> available_centroids;

    std::vector<size_t> common_counts = {
        // 64,
        128,
        // 256,
        // 512,
        // 1024,
        // 2048,
        // 4096
    };

    for (size_t count : common_counts) {
        std::string centroid_file = (base_path / (dataset_name + "_centroid_" + std::to_string(count) + ".fvecs")).string();
        std::string cluster_file = (base_path / (dataset_name + "_cluster_id_" + std::to_string(count) + ".ivecs")).string();

        if (std::filesystem::exists(centroid_file) && std::filesystem::exists(cluster_file)) {
            available_centroids.emplace_back(count, std::to_string(count));
            std::cout << "Found centroids: " << count << " clusters" << std::endl;
        }
    }

    if (available_centroids.empty()) {
        std::cerr << "Warning: No centroid files found for dataset " << dataset_name << std::endl;
    }

    return available_centroids;
}

std::vector<IVFJHQParams> getIVFJHQParamSets(const std::vector<std::pair<size_t, std::string>>& available_centroids)
{
    std::vector<IVFJHQParams> params;

    struct JHQConfig {
        std::string name_suffix;
        int M;
        std::vector<int> level_bits;
        float oversampling;
        bool use_early_termination;
        bool compute_residuals;
        bool use_kmeans_refinement;
        int kmeans_niter;
        int kmeans_seed;
    };

    std::vector<JHQConfig> jhq_configs = {
        { "M128_8bits_1level_analytical", 128, { 8 }, 4.0f, true, false, false, 0, 1234 },
        { "M128_4-8bits_analytical", 128, { 4, 8 }, 4.0f, true, true, false, 0, 1234 },
        { "M128_8-4bits_analytical", 128, { 8, 4 }, 4.0f, true, true, false, 0, 1234 },
    };

    for (const auto& [nlist, suffix] : available_centroids) {
        for (const auto& config : jhq_configs) {
            std::string name = "IVFJHQ_nlist" + suffix + "_" + config.name_suffix;

            params.emplace_back(name, nlist, config.M, config.level_bits,
                true, config.oversampling,
                config.use_early_termination, config.compute_residuals,
                config.use_kmeans_refinement, config.kmeans_niter, config.kmeans_seed);
        }
    }

    std::vector<IVFJHQQueryParam> query_params = {
        IVFJHQQueryParam(1, 4.0f),
        IVFJHQQueryParam(4, 4.0f),
        IVFJHQQueryParam(8, 4.0f),
        IVFJHQQueryParam(16, 4.0f),

    };

    for (auto& param : params) {
        param.query_params = query_params;
    }

    return params;
}

void processDataset(const std::string& dataset_name)
{
    try {
        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "Processing dataset: " << dataset_name << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        std::filesystem::path base_path = std::filesystem::path(data_dir) / dataset_name;
        std::filesystem::path result_path = std::filesystem::path(result_dir) / "jhq-test" / dataset_name;
        std::filesystem::path index_cache_dir = std::filesystem::path(index_dir) / "jhq-index" / dataset_name;

        std::filesystem::create_directories(result_path);
        std::filesystem::create_directories(index_cache_dir);

        auto available_centroids = detectAvailableCentroids(base_path, dataset_name);
        if (available_centroids.empty()) {
            std::cerr << "No centroid files found for dataset " << dataset_name << std::endl;
            return;
        }

        std::cout << "\n--- Reading Data ---" << std::endl;
        printMemoryUsage("Before Loading");

        auto base_data = readFvecs((base_path / (dataset_name + "_base.fvecs")).string());
        auto query_data = readFvecs((base_path / (dataset_name + "_query.fvecs")).string());
        auto ground_truth = readIvecs((base_path / (dataset_name + "_groundtruth.ivecs")).string());

        const size_t num_queries_to_use = 1000;
        if (query_data.size() > num_queries_to_use) {
            std::cout << "Limiting queries from " << query_data.size() << " to " << num_queries_to_use << std::endl;
            query_data.resize(num_queries_to_use);
            if (ground_truth.size() > num_queries_to_use) {
                ground_truth.resize(num_queries_to_use);
            }
        }

        auto param_sets = getIVFJHQParamSets(available_centroids);

        int analytical_count = 0, kmeans_count = 0;
        for (const auto& params : param_sets) {
            if (params.use_kmeans_refinement) {
                kmeans_count++;
            } else {
                analytical_count++;
            }
        }

        std::cout << "Testing " << param_sets.size() << " manually configured parameter sets:" << std::endl;
        std::cout << "  - Analytical-only: " << analytical_count << " configurations" << std::endl;
        std::cout << "  - Analytical + K-means: " << kmeans_count << " configurations" << std::endl;

        CSVFileManager csv_manager;
        if (!csv_manager.initialize(result_path, dataset_name)) {
            throw std::runtime_error("Failed to initialize CSV files");
        }

        for (size_t param_idx = 0; param_idx < param_sets.size(); ++param_idx) {
            const auto& params = param_sets[param_idx];

            std::cout << "\n[Test " << (param_idx + 1) << "/" << param_sets.size()
                      << "] Testing: " << params.name << std::endl;
            std::cout << "  Clustering: " << (params.use_kmeans_refinement ? ("K-means(" + std::to_string(params.kmeans_niter) + " iter, seed=" + std::to_string(params.kmeans_seed) + ")") : "Analytical-only") << std::endl;

            try {
                std::string centroid_file = (base_path / (dataset_name + "_centroid_" + std::to_string(params.nlist) + ".fvecs")).string();
                std::string cluster_file = (base_path / (dataset_name + "_cluster_id_" + std::to_string(params.nlist) + ".ivecs")).string();

                auto centroids = readFvecs(centroid_file);
                auto cluster_ids = readIvecs(cluster_file);

                if (params.nlist != centroids.size()) {
                    std::cout << "Warning: Expected " << params.nlist << " centroids but loaded "
                              << centroids.size() << ". Skipping." << std::endl;
                    continue;
                }

                if (base_data.size() != cluster_ids.size()) {
                    throw std::runtime_error("Base data size does not match cluster ID data size");
                }

                std::cout << "Loaded " << centroids.size() << " centroids and "
                          << cluster_ids.size() << " cluster assignments" << std::endl;

                IVFJHQ ivfjhq(params, dataset_name);

                std::cout << "Training/Loading IVFJHQ with " << params.nlist << " centroids..." << std::endl;
                printMemoryUsage("Before Training/Loading");

                bool success = ivfjhq.tryLoadOrTrain(base_data, centroids, cluster_ids, index_cache_dir);

                if (!success) {
                    std::cout << "Failed to train/load index for config: " << params.name << std::endl;
                    continue;
                }

                printMemoryUsage("After Training/Loading");

                auto algo_info = ivfjhq.getAlgorithmInfo();

                std::cout << "Index ready!" << std::endl;
                std::cout << "  Train Time: " << ivfjhq.getTrainTime() << "s" << std::endl;
                std::cout << "  Encode Time: " << ivfjhq.getEncodeTime() << "s" << std::endl;
                std::cout << "  Total Time: " << ivfjhq.getIndexingTime() << "s" << std::endl;
                std::cout << "  Memory Usage: " << algo_info.memory_usage_mb << " MB" << std::endl;
                std::cout << "  Compression Ratio: " << algo_info.compression_ratio << "x" << std::endl;

                for (const auto& qparam : params.query_params) {
                    ivfjhq.setQueryArguments(qparam);

                    std::cout << "Running batched search (nprobe=" << qparam.nprobe
                              << ", OS=" << qparam.jhq_oversampling_factor << ")..." << std::endl;

                    auto search_start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<faiss::idx_t>> all_results = ivfjhq.queryBatch(query_data, K);
                    auto search_end = std::chrono::high_resolution_clock::now();

                    auto search_duration = std::chrono::duration<double>(search_end - search_start);
                    double qps = query_data.size() / search_duration.count();
                    double search_time_s = search_duration.count();

                    double recall = calculateRecall(ground_truth, all_results);

                    std::cout << "Results:" << std::endl;
                    std::cout << "  Recall@" << K << ": " << std::fixed << std::setprecision(2) << recall << std::endl;
                    std::cout << "  QPS: " << std::setprecision(2) << qps << std::endl;
                    std::cout << "  Search Time: " << std::setprecision(2) << search_time_s << "s" << std::endl;

                    std::string level_bits_str = "\"[";
                    for (size_t i = 0; i < params.level_bits.size(); ++i) {
                        if (i > 0)
                            level_bits_str += ",";
                        level_bits_str += std::to_string(params.level_bits[i]);
                    }
                    level_bits_str += "]\"";

                    auto& recall_stream = csv_manager.getRecallStream(params.use_kmeans_refinement);
                    recall_stream << params.name << ","
                                  << params.nlist << ","
                                  << qparam.nprobe << ","
                                  << params.M << ","
                                  << level_bits_str << ","
                                  << params.level_bits.size() << ","
                                  << qparam.jhq_oversampling_factor << ","
                                  << std::fixed << std::setprecision(2) << recall << ","
                                  << std::setprecision(2) << qps << ","
                                  << ivfjhq.getTrainTime() << ","
                                  << ivfjhq.getEncodeTime() << ","
                                  << ivfjhq.getIndexingTime() << ","
                                  << (params.use_early_termination ? "true" : "false") << ","
                                  << (params.compute_residuals ? "true" : "false") << ","
                                  << std::setprecision(6) << search_time_s << ","
                                  << (params.use_kmeans_refinement ? "analytical_kmeans" : "analytical_only") << ","
                                  << params.kmeans_niter << ","
                                  << params.kmeans_seed << std::endl;
                }

                auto& metrics_stream = csv_manager.getMetricsStream(params.use_kmeans_refinement);
                metrics_stream << params.name << ","
                               << algo_info.nlist << ","
                               << algo_info.num_jhq_subspaces << ","
                               << algo_info.jhq_subspace_dimension << ","
                               << algo_info.total_jhq_centroids << ","
                               << algo_info.jhq_total_bits_per_vector << ","
                               << algo_info.total_bits_per_vector << ","
                               << std::fixed << std::setprecision(2) << algo_info.memory_usage_mb << ","
                               << std::setprecision(2) << algo_info.compression_ratio << ","
                               << (params.use_kmeans_refinement ? "analytical_kmeans" : "analytical_only") << ","
                               << params.kmeans_niter << std::endl;

            } catch (const std::exception& e) {
                std::cerr << "ERROR during test with config '" << params.name << "': " << e.what() << std::endl;
                continue;
            }
        }

        csv_manager.close();

        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "Dataset processing completed!" << std::endl;
        csv_manager.printSavedPaths(result_path, dataset_name);
        std::cout << "Index cache saved in: " << index_cache_dir << std::endl;
        std::cout << std::string(60, '=') << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error processing dataset " << dataset_name << ": " << e.what() << std::endl;
    }
}

int main(int argc, char** argv)
{
    std::vector<std::string> datasets_to_process;

    if (argc > 1) {
        for (int i = 1; i < argc; i++) {
            datasets_to_process.push_back(argv[i]);
        }
    } else {
        datasets_to_process = DATASETS;
    }

    std::cout << std::string(60, '=') << std::endl;

    for (const auto& dataset : datasets_to_process) {
        processDataset(dataset);
        std::this_thread::sleep_for(std::chrono::seconds(static_cast<long>(2)));
    }

    return 0;
}