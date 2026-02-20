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
#include "IndexJHQ.h"
#include <faiss/impl/io.h>

const std::vector<std::string> DATASETS = {
    "dbpedia-openai3-text-embedding-3-large-1024-100K",
    "dbpedia-openai3-text-embedding-3-large-3072-100K",
};

const size_t K = 10; // Number of nearest neighbors to retrieve

struct JHQQueryParam {
    float oversampling_factor;
    bool use_early_termination;
    bool compute_residuals;

    JHQQueryParam(float os = 4.0f, bool early_term = true, bool residuals = true)
        : oversampling_factor(os)
        , use_early_termination(early_term)
        , compute_residuals(residuals)
    {
    }
};

struct JHQParams {
    std::string name;
    int M;
    std::vector<int> level_bits;
    bool use_jl_transform;
    bool use_analytical_init;
    float default_oversampling;

    bool use_kmeans_refinement;
    int kmeans_niter;
    int kmeans_seed;

    std::vector<JHQQueryParam> query_params;

    JHQParams(const std::string& n, int m, const std::vector<int>& bits,
        bool jl = true, bool analytical = true, float def_os = 4.0f,
        bool use_kmeans = false, int niter = 25, int seed = 1234)
        : name(n)
        , M(m)
        , level_bits(bits)
        , use_jl_transform(jl)
        , use_analytical_init(analytical)
        , default_oversampling(def_os)
        , use_kmeans_refinement(use_kmeans)
        , kmeans_niter(niter)
        , kmeans_seed(seed)
    {
    }

    std::string getIndexFilename() const
    {
        std::stringstream ss;
        ss << "jhq_M" << M;

        ss << "_bits";
        for (int bits : level_bits) {
            ss << "_" << bits;
        }

        if (use_kmeans_refinement) {
            ss << "_kmeans" << kmeans_niter << "_seed" << kmeans_seed;
        } else {
            ss << "_analytical";
        }

        if (use_jl_transform) {
            ss << "_jl";
        }

        ss << ".index";
        return ss.str();
    }

    std::string getClusteringMethodName() const
    {
        return use_kmeans_refinement ? "analytical_kmeans" : "analytical_only";
    }
};

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

bool saveIndex(const faiss::IndexJHQ* index, const std::string& filename)
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
        index->write(&writer);

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
            faiss::IndexJHQ::read(&reader));
        fclose(file);

        std::cout << "Index loaded successfully!" << std::endl;
        std::cout << "  - Vectors: " << index->ntotal << std::endl;
        std::cout << "  - Dimension: " << index->d << std::endl;
        std::cout << "  - M: " << index->M << std::endl;

        return index;
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        return nullptr;
    }
}

} // namespace IndexSaveLoad

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
        std::string recall_header = "Config_Name,M,Level_Bits,Num_Levels,Oversampling,"
                                    "Recall@"
            + std::to_string(K) + ",QPS,Train_Time_s,Encode_Time_s,Total_Time_s,"
                                  "Use_Early_Term,Compute_Residuals,Search_Time_s,"
                                  "Clustering_Mode,KMeans_Iters,KMeans_Seed\n";

        std::string metrics_header = "Config_Name,M,Subspace_Dim,Total_Centroids,"
                                     "Total_Bits,Memory_MB,Compression_Ratio,"
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
        std::cout << "    - Recall: " << (result_path / "analytical_only" / (dataset_name + "_analytical_only_recall_results.csv")) << std::endl;
        std::cout << "    - Metrics: " << (result_path / "analytical_only" / (dataset_name + "_analytical_only_detailed_metrics.csv")) << std::endl;
        std::cout << "    - Recall: " << (result_path / "analytical_kmeans" / (dataset_name + "_analytical_kmeans_recall_results.csv")) << std::endl;
        std::cout << "    - Metrics: " << (result_path / "analytical_kmeans" / (dataset_name + "_analytical_kmeans_detailed_metrics.csv")) << std::endl;
    }
};

class JHQ {
public:
    JHQ(const JHQParams& params, const std::string& dataset_name)
        : params_(params)
        , dataset_name_(dataset_name)
    {
        std::cout << "JHQ Configuration:" << std::endl;
        std::cout << "  Name: " << params_.name << std::endl;
        std::cout << "  M: " << params_.M << std::endl;
        std::cout << "  Level bits: [";
        for (size_t i = 0; i < params_.level_bits.size(); ++i) {
            if (i > 0)
                std::cout << ", ";
            std::cout << params_.level_bits[i];
        }
        std::cout << "]" << std::endl;
        std::cout << "  JL Transform: " << (params_.use_jl_transform ? "enabled" : "disabled") << std::endl;
        std::cout << "  Analytical Init: " << (params_.use_analytical_init ? "enabled" : "disabled") << std::endl;
        std::cout << "  Default Oversampling: " << params_.default_oversampling << std::endl;
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
        const std::filesystem::path& index_cache_dir)
    {
        // Try to load existing index first
        std::filesystem::path index_file = index_cache_dir / params_.getIndexFilename();

        index_ = IndexSaveLoad::loadIndex(index_file.string());

        if (index_) {
            std::cout << "Using pre-trained index!" << std::endl;

            // Set dim_ from loaded index
            dim_ = index_->d;

            // Load the associated indexing times
            loadIndexingTime(index_file);
            return true;
        }

        // Train new index if loading failed
        std::cout << "Training new index..." << std::endl;
        bool success = fit(X);

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

    bool fit(const std::vector<std::vector<float>>& X)
    {
        if (X.empty()) {
            throw std::runtime_error("Input data is empty");
        }

        // Extract dimension from first vector
        dim_ = static_cast<int>(X[0].size());
        size_t n_data = X.size();

        std::cout << "JHQ Training:" << std::endl;
        std::cout << "  Dataset: " << dataset_name_ << std::endl;
        std::cout << "  Vectors: " << n_data << std::endl;
        std::cout << "  Dimension: " << dim_ << std::endl;

        // Validate dimension before creating index
        if (dim_ <= 0) {
            throw std::runtime_error("Invalid dimension: " + std::to_string(dim_));
        }

        // Validate M parameter
        if (dim_ % params_.M != 0) {
            throw std::runtime_error("Dimension " + std::to_string(dim_) + " must be divisible by M=" + std::to_string(params_.M));
        }

        return createAndTrainIndex(X);
    }

    void setQueryArguments(const JHQQueryParam& param)
    {
        current_query_param_ = param;
        std::cout << "JHQ Query Parameters:" << std::endl;
        std::cout << "  Oversampling: " << param.oversampling_factor << std::endl;
        std::cout << "  Early Termination: " << (param.use_early_termination ? "enabled" : "disabled") << std::endl;
        std::cout << "  Compute Residuals: " << (param.compute_residuals ? "enabled" : "disabled") << std::endl;
    }

    std::vector<std::vector<faiss::idx_t>> queryBatch(
        const std::vector<std::vector<float>>& query_vecs,
        size_t k)
    {
        if (query_vecs.empty()) {
            return {};
        }

        size_t n_queries = query_vecs.size();
        if (query_vecs[0].size() != static_cast<size_t>(dim_)) {
            throw std::runtime_error("Query vector has incorrect dimensionality");
        }

        // Flatten query data
        std::vector<float> queries_flat(n_queries * dim_);
        for (size_t i = 0; i < n_queries; ++i) {
            std::memcpy(&queries_flat[i * dim_], query_vecs[i].data(), dim_ * sizeof(float));
        }

        // Allocate result arrays
        std::vector<float> distances(n_queries * k);
        std::vector<faiss::idx_t> labels(n_queries * k);

        // Create search parameters
        faiss::JHQSearchParameters search_params;
        search_params.oversampling_factor = current_query_param_.oversampling_factor;
        search_params.use_early_termination = current_query_param_.use_early_termination;
        search_params.compute_residuals = current_query_param_.compute_residuals;

        // Perform batched search
        index_->search(n_queries, queries_flat.data(), k, distances.data(), labels.data(), &search_params);

        // Convert results to vector of vectors
        std::vector<std::vector<faiss::idx_t>> results(n_queries);
        for (size_t i = 0; i < n_queries; ++i) {
            results[i].assign(
                labels.begin() + i * k,
                labels.begin() + (i + 1) * k);
        }

        return results;
    }

    // Get algorithm info
    struct AlgorithmInfo {
        std::string algorithm_name = "JHQ";
        int dimension;
        int num_subspaces;
        int subspace_dimension;
        std::vector<int> bits_per_level;
        int total_bits_per_vector;
        bool uses_jl_transform;
        bool uses_analytical_init;
        float default_oversampling_factor;
        size_t num_database_vectors;
        size_t total_centroids;
        double memory_usage_mb;
        float compression_ratio;
    };

    AlgorithmInfo getAlgorithmInfo() const
    {
        AlgorithmInfo info;
        info.dimension = dim_;
        info.num_subspaces = params_.M;
        info.subspace_dimension = dim_ / params_.M;
        info.bits_per_level = params_.level_bits;

        // Calculate total bits per vector
        info.total_bits_per_vector = 0;
        for (int l = 0; l < static_cast<int>(params_.level_bits.size()); ++l) {
            if (l == 0) {
                info.total_bits_per_vector += params_.M * params_.level_bits[l];
            } else {
                info.total_bits_per_vector += dim_ * params_.level_bits[l];
            }
        }

        info.uses_jl_transform = params_.use_jl_transform;
        info.uses_analytical_init = params_.use_analytical_init;
        info.default_oversampling_factor = params_.default_oversampling;

        if (index_) {
            info.num_database_vectors = static_cast<size_t>(index_->ntotal);
            info.memory_usage_mb = index_->get_memory_usage() / (1024.0 * 1024.0);
            info.compression_ratio = (sizeof(float) * dim_) / static_cast<float>(index_->code_size);
        }

        // Calculate total centroids
        info.total_centroids = 0;
        for (int l = 0; l < static_cast<int>(params_.level_bits.size()); ++l) {
            info.total_centroids += static_cast<size_t>(params_.M) * (1 << params_.level_bits[l]);
        }

        return info;
    }

    double getIndexingTime() const { return train_time_ + encode_time_; }
    double getTrainTime() const { return train_time_; }
    double getEncodeTime() const { return encode_time_; }

private:
    bool createAndTrainIndex(const std::vector<std::vector<float>>& X)
    {
        try {
            // Flatten data for FAISS
            std::vector<float> data_flat(X.size() * dim_);
            for (size_t i = 0; i < X.size(); ++i) {
                std::memcpy(&data_flat[i * dim_], X[i].data(), dim_ * sizeof(float));
            }

            // Create JHQ index
            index_ = std::make_unique<faiss::IndexJHQ>(
                dim_,
                params_.M,
                params_.level_bits,
                params_.use_jl_transform,
                params_.default_oversampling,
                params_.use_analytical_init,
                true, // verbose
                faiss::METRIC_L2);

            // Set clustering parameters
            index_->set_clustering_parameters(
                params_.use_kmeans_refinement,
                params_.kmeans_niter,
                params_.kmeans_seed);

            // Train the index
            std::cout << "JHQ: Training on " << X.size() << " vectors..." << std::endl;
            auto train_start = std::chrono::high_resolution_clock::now();

            index_->train(X.size(), data_flat.data());

            auto train_end = std::chrono::high_resolution_clock::now();
            auto train_duration = std::chrono::duration<double>(train_end - train_start);
            train_time_ = train_duration.count();

            if (!index_->is_trained_()) {
                throw std::runtime_error("JHQ index failed to train");
            }

            std::cout << "JHQ: Training completed in " << train_time_ << "s" << std::endl;

            // Add vectors to index
            std::cout << "JHQ: Adding vectors to index..." << std::endl;
            auto add_start = std::chrono::high_resolution_clock::now();

            index_->add(X.size(), data_flat.data());

            auto add_end = std::chrono::high_resolution_clock::now();
            auto add_duration = std::chrono::duration<double>(add_end - add_start);
            encode_time_ = add_duration.count();

            std::cout << "JHQ: Added " << index_->ntotal << " vectors in " << encode_time_ << "s" << std::endl;

            return true;

        } catch (const std::exception& e) {
            std::cerr << "Error in createAndTrainIndex: " << e.what() << std::endl;
            return false;
        }
    }

private:
    std::unique_ptr<faiss::IndexJHQ> index_;
    JHQParams params_;
    std::string dataset_name_;
    int dim_;
    JHQQueryParam current_query_param_;
    double train_time_ = 0.0;
    double encode_time_ = 0.0;
};

// Manual parameter configuration with clustering control
std::vector<JHQParams> getJHQParamSets()
{
    std::vector<JHQParams> params;

    struct JHQConfig {
        std::string name_suffix;
        int M;
        std::vector<int> level_bits;
        float oversampling;
        // Manual clustering control
        bool use_kmeans_refinement;
        int kmeans_niter;
        int kmeans_seed;
    };

    std::vector<JHQConfig> jhq_configs = {
        { "M128_8bits_1level_analytical", 128, { 8 }, 4.0f, false, 0, 1234 },
        { "M128_4-8bits_analytical", 128, { 4, 8 }, 4.0f, false, 0, 1234 },
        { "M128_8-4bits_analytical", 128, { 8, 4 }, 4.0f, false, 0, 1234 },
    };

    // Generate parameters for each configuration
    for (const auto& config : jhq_configs) {
        std::string name = "JHQ_" + config.name_suffix;

        params.emplace_back(name, config.M, config.level_bits,
            true, true, config.oversampling,
            config.use_kmeans_refinement, config.kmeans_niter, config.kmeans_seed);
    }

    // Set query parameters
    std::vector<JHQQueryParam> query_params = {
        JHQQueryParam(4.0f, true, true), // Default
    };

    for (auto& param : params) {
        param.query_params = query_params;
    }

    return params;
}

// Main dataset processing function
void processDataset(const std::string& dataset_name)
{
    try {
        std::cout << "\n"
                  << std::string(60, '=') << std::endl;
        std::cout << "Processing dataset: " << dataset_name << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Setup paths
        std::filesystem::path base_path = std::filesystem::path(data_dir) / dataset_name;
        std::filesystem::path result_path = std::filesystem::path(result_dir) / "jhq-test" / dataset_name;
        std::filesystem::path index_cache_dir = std::filesystem::path(index_dir) / "jhq-index" / dataset_name;

        std::filesystem::create_directories(result_path);
        std::filesystem::create_directories(index_cache_dir);

        // Load base dataset files
        std::cout << "\n--- Reading Data ---" << std::endl;
        printMemoryUsage("Before Loading");

        auto base_data = readFvecs((base_path / (dataset_name + "_base.fvecs")).string());
        auto query_data = readFvecs((base_path / (dataset_name + "_query.fvecs")).string());
        auto ground_truth = readIvecs((base_path / (dataset_name + "_groundtruth.ivecs")).string());

        // Limit queries for faster testing
        const size_t num_queries_to_use = 1000;
        if (query_data.size() > num_queries_to_use) {
            std::cout << "Limiting queries from " << query_data.size() << " to " << num_queries_to_use << std::endl;
            query_data.resize(num_queries_to_use);
            if (ground_truth.size() > num_queries_to_use) {
                ground_truth.resize(num_queries_to_use);
            }
        }

        // Generate parameter sets with manual clustering control
        auto param_sets = getJHQParamSets();

        // Count configurations by method
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

        // Initialize CSV file manager
        CSVFileManager csv_manager;
        if (!csv_manager.initialize(result_path, dataset_name)) {
            throw std::runtime_error("Failed to initialize CSV files");
        }

        // Process each parameter configuration
        for (size_t param_idx = 0; param_idx < param_sets.size(); ++param_idx) {
            const auto& params = param_sets[param_idx];

            std::cout << "\n[Test " << (param_idx + 1) << "/" << param_sets.size()
                      << "] Testing: " << params.name << std::endl;
            std::cout << "  Clustering: " << (params.use_kmeans_refinement ? ("K-means(" + std::to_string(params.kmeans_niter) + " iter, seed=" + std::to_string(params.kmeans_seed) + ")") : "Analytical-only") << std::endl;

            try {
                JHQ jhq(params, dataset_name);

                std::cout << "Training/Loading JHQ..." << std::endl;
                printMemoryUsage("Before Training/Loading");

                // Try to load existing index or train new one
                bool success = jhq.tryLoadOrTrain(base_data, index_cache_dir);

                if (!success) {
                    std::cout << "Failed to train/load index for config: " << params.name << std::endl;
                    continue;
                }

                printMemoryUsage("After Training/Loading");

                // Get algorithm info
                auto algo_info = jhq.getAlgorithmInfo();

                std::cout << "Index ready!" << std::endl;
                std::cout << "  Train Time: " << jhq.getTrainTime() << "s" << std::endl;
                std::cout << "  Encode Time: " << jhq.getEncodeTime() << "s" << std::endl;
                std::cout << "  Total Time: " << jhq.getIndexingTime() << "s" << std::endl;
                std::cout << "  Memory Usage: " << algo_info.memory_usage_mb << " MB" << std::endl;
                std::cout << "  Compression Ratio: " << algo_info.compression_ratio << "x" << std::endl;

                // Test each query parameter configuration
                for (const auto& qparam : params.query_params) {
                    jhq.setQueryArguments(qparam);

                    std::cout << "Running batched search (OS=" << qparam.oversampling_factor << ")..." << std::endl;

                    auto search_start = std::chrono::high_resolution_clock::now();
                    std::vector<std::vector<faiss::idx_t>> all_results = jhq.queryBatch(query_data, K);
                    auto search_end = std::chrono::high_resolution_clock::now();

                    auto search_duration = std::chrono::duration<double>(search_end - search_start);
                    double qps = query_data.size() / search_duration.count();
                    double search_time_s = search_duration.count();

                    // Calculate recall
                    double recall = calculateRecall(ground_truth, all_results);

                    std::cout << "Results:" << std::endl;
                    std::cout << "  Recall@" << K << ": " << std::fixed << std::setprecision(2) << recall << std::endl;
                    std::cout << "  QPS: " << std::setprecision(2) << qps << std::endl;
                    std::cout << "  Search Time: " << std::setprecision(2) << search_time_s << "s" << std::endl;

                    // Prepare CSV data
                    std::string level_bits_str = "\"[";
                    for (size_t i = 0; i < params.level_bits.size(); ++i) {
                        if (i > 0)
                            level_bits_str += ",";
                        level_bits_str += std::to_string(params.level_bits[i]);
                    }
                    level_bits_str += "]\"";

                    // Write to appropriate recall CSV based on clustering method
                    auto& recall_stream = csv_manager.getRecallStream(params.use_kmeans_refinement);
                    recall_stream << params.name << ","
                                  << params.M << ","
                                  << level_bits_str << ","
                                  << params.level_bits.size() << ","
                                  << qparam.oversampling_factor << ","
                                  << std::fixed << std::setprecision(2) << recall << ","
                                  << std::setprecision(2) << qps << ","
                                  << jhq.getTrainTime() << ","
                                  << jhq.getEncodeTime() << ","
                                  << jhq.getIndexingTime() << ","
                                  << (qparam.use_early_termination ? "true" : "false") << ","
                                  << (qparam.compute_residuals ? "true" : "false") << ","
                                  << std::setprecision(6) << search_time_s << ","
                                  << (params.use_kmeans_refinement ? "analytical_kmeans" : "analytical_only") << ","
                                  << params.kmeans_niter << ","
                                  << params.kmeans_seed << std::endl;
                }

                auto& metrics_stream = csv_manager.getMetricsStream(params.use_kmeans_refinement);
                metrics_stream << params.name << ","
                               << algo_info.num_subspaces << ","
                               << algo_info.subspace_dimension << ","
                               << algo_info.total_centroids << ","
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