# JHQ: Johnson-Lindenstrauss Enhanced Hierarchical Quantization

Implementation for **VLDB 2026** submission: *JHQ: Johnson-Lindenstrauss Enhanced Hierarchical Quantization for High-Dimensional Approximate Nearest Neighbor Search*

## Overview

JHQ introduces two novel vector quantization algorithms that leverages the orthogonal **Johnson-Lindenstrauss (JL) transformation** to address critical bottlenecks in high-dimensional approximate nearest neighbor (ANN) search. Main quantitative benefits include:

- **3-100× query speedup** over state-of-the-art methods at ≥95% recall
- **Up to 3,600× faster index construction** (compared with Additive-based quantizers) through training-free codebook generation
- **Superior performance** on datasets with up to 3,072 dimensions
- **Provable error bounds** with theoretical guarantees


## Organization of Supplementary Materials

- Appendix (Proofs of Lemmas and Theorems, and additional experiments further supporting the observations reported in the main paper): `appendix.pdf`
- Source code: `/jhq`


## Quick Start

```bash
git clone --recurse-submodules https://github.com/jiabhan/JHQ.git
cd JHQ
mkdir build && cd build

cmake .. \
  -DDATA_DIR=/path/to/your/datasets \
  -DINDEX_DIR=/path/to/your/indices \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
```

### Prerequisites

- FAISS (included as submodule)
- OpenBLAS
- OpenMP
- LAPACK
- Python packages (for preprocessing): `datasets`, `numpy`, `scikit-learn`, `faiss-cpu/faiss-gpu`
- C++ Compiler: GCC 14+ with C++23 support
- CMake: Version 3.27 or higher

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install build-essential cmake libopenblas-dev liblapack-dev libomp-dev
```

**macOS:**
```bash
brew install cmake openblas lapack libomp
```


### Build Options

The build system supports several configuration options:

```bash
# Development options
-DENABLE_WARNINGS=ON              # Enable compiler warnings (default: ON)
-DENABLE_WARNINGS_AS_ERRORS=OFF   # Treat warnings as errors (default: OFF)
-DENABLE_TESTING=ON               # Enable unit tests (default: ON)
-DENABLE_CLANG_TIDY=ON            # Enable clang-tidy analysis (default: ON)
-DENABLE_CLANG_FORMAT=ON          # Enable clang-format (default: ON)

# FAISS options
-DFAISS_ENABLE_GPU=OFF            # Enable GPU support (default: OFF)
-DFAISS_ENABLE_PYTHON=OFF         # Enable Python bindings (default: OFF)
```

### Troubleshooting

**C++ support issues:**
```bash
# Check compiler version
g++ --version      # Need GCC 14+

# Ubuntu: Install newer GCC
sudo apt install gcc-14 g++-14
export CC=gcc-11 CXX=g++-14
```

**LAPACK/BLAS not found:**
```bash
# Ubuntu/Debian
sudo apt install liblapack-dev libopenblas-dev
```

**FAISS submodule issues:**
```bash
# Initialize and update submodules
git submodule update --init --recursive

# If FAISS fails to build, try:
cd external/faiss
git checkout main
git pull origin main
cd ../..
```

## File Structure

```
JHQ/
├── jhqlib/                  # Core implementation
├── examples/                # Example programs
├── external/                # Third-party submodule
└── CMakeLists.txt           # Build configuration
```

## Datasets

### Quick Test Datasets (100K vectors)

| Dataset           | Dimensions | Size | Source |
|-------------------|------------|------|--------|
| OpenAI3-1024-100K | 1024       | 100K | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1024-100K) |
| OpenAI3-3072-100K | 3072       | 100K | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-100K) |

### Full Evaluation Datasets

Evaluation datasets used in the paper:

| Dataset             | Dimensions | Size | Source |
|---------------------|------------|------|--------|
| OpenAI3-1536        | 1536       | 1M   | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1536-1M) |
| OpenAI3-3072        | 3072       | 1M   | [HuggingFace](https://huggingface.co/datasets/Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M) |
| Vogue-768           | 768        | 933K | [HuggingFace](https://huggingface.co/datasets/tonyassi/vogue933k-embeddings) |
| Arxiv-Abstracts-768 | 768        | 2.3M | [HuggingFace](https://huggingface.co/datasets/macrocosm/arxiv_abstracts) |
| BGE-M3-1024         | 1024       | 10M  | [HuggingFace](https://huggingface.co/datasets/Upstash/wikipedia-2024-06-bge-m3) |
| Stella-TREC24       | 1024       | 17M  | [HuggingFace](https://huggingface.co/datasets/ielabgroup/stella_trec24_biogen_embedding) |

### Datasets Preprocessing

#### HuggingFace Dataset Processing

```python
import ast
from datasets import load_dataset
from sklearn.preprocessing import normalize

def convert_embeddings(example):
    """Convert string embeddings to float arrays."""
    if isinstance(example['embedding'], str):
        try:
            embedding_list = ast.literal_eval(example['embedding'])
            example['embedding'] = [float(x) for x in embedding_list]
        except (ValueError, SyntaxError):
            clean_str = example['embedding'].strip('[]')
            string_values = [x.strip().strip('"').strip("'")
                           for x in clean_str.split(',')]
            example['embedding'] = [float(x) for x in string_values if x]
    elif isinstance(example['embedding'], list):
        example['embedding'] = [float(x) if isinstance(x, str) else x
                               for x in example['embedding']]
    return example

def process_huggingface_dataset(dataset_name, output_dir, split="corpus"):
    """Download and convert HuggingFace dataset to .fvecs format."""

    # Load dataset
    dataset_dict = load_dataset(dataset_name)
    dataset = dataset_dict.get(split, dataset_dict.get("train",
                               list(dataset_dict.values())[0]))

    # Convert embeddings
    dataset = dataset.map(convert_embeddings, num_proc=4)

    # Extract embeddings and normalize
    embeddings = np.array(dataset['embedding'], dtype=np.float32)
    embeddings = normalize(embeddings, axis=1, norm='l2')

    # Save as .fvecs
    output_file = f"{output_dir}/{dataset_name.replace('/', '_')}_base.fvecs"
    write_fvecs(output_file, embeddings)

    print(f"Processed {len(embeddings):,} vectors to {output_file}")
    return embeddings
```

#### Train/Query Split

```python
def create_train_query_split(embeddings, query_size=1000, seed=42):
    """Split embeddings into base and query sets."""
    np.random.seed(seed)
    n_total = len(embeddings)

    if query_size >= n_total:
        raise ValueError(f"Query size ({query_size}) must be < total ({n_total})")

    # Random query indices
    query_indices = np.random.choice(n_total, query_size, replace=False)
    base_mask = np.ones(n_total, dtype=bool)
    base_mask[query_indices] = False

    query_vectors = embeddings[query_indices]
    base_vectors = embeddings[base_mask]

    return base_vectors, query_vectors
```
#### Ground Truth Computation

```python
import faiss

def compute_ground_truth(base_vectors, query_vectors, k=20, use_gpu=True):
    """Compute exact k-NN ground truth using FAISS."""
    dimension = base_vectors.shape[1]

    # Create index
    if use_gpu and faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, dimension)
    else:
        index = faiss.IndexFlatL2(dimension)

    # Add base vectors and search
    index.add(base_vectors.astype(np.float32))
    distances, labels = index.search(query_vectors.astype(np.float32), k)

    return labels.astype(np.int32)
```

#### Complete Processing Pipeline

```python
def preprocess_dataset(dataset_name, output_dir, query_size=1000, k=20):
    """Complete preprocessing pipeline."""

    # Step 1: Download and process
    embeddings = process_huggingface_dataset(dataset_name, output_dir)

    # Step 2: Create train/query split
    base_vectors, query_vectors = create_train_query_split(embeddings, query_size)

    # Step 3: Save splits
    dataset_clean = dataset_name.replace('/', '_')
    write_fvecs(f"{output_dir}/{dataset_clean}_base.fvecs", base_vectors)
    write_fvecs(f"{output_dir}/{dataset_clean}_query.fvecs", query_vectors)

    # Step 4: Compute and save ground truth
    ground_truth = compute_ground_truth(base_vectors, query_vectors, k)
    write_ivecs(f"{output_dir}/{dataset_clean}_groundtruth.ivecs", ground_truth)

    print(f"Preprocessing complete:")
    print(f"  Base vectors: {len(base_vectors):,}")
    print(f"  Query vectors: {len(query_vectors):,}")
    print(f"  Ground truth: {ground_truth.shape}")

# Usage example
preprocess_dataset(
    "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-1024-100K",
    output_dir="datasets",
    query_size=1000,
    k=20
)
```

#### File I/O Utilities

```python
import numpy as np
import struct

def write_fvecs(filename, vectors):
    """Write float vectors in .fvecs format."""
    vectors = vectors.astype(np.float32)
    with open(filename, 'wb') as f:
        for vector in vectors:
            d = np.int32(len(vector))
            f.write(d.tobytes())
            f.write(vector.tobytes())

def read_fvecs(filename):
    """Read float vectors from .fvecs format."""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            d_bytes = f.read(4)
            if len(d_bytes) < 4:
                break
            d = np.frombuffer(d_bytes, dtype=np.int32)[0]
            vector = np.frombuffer(f.read(4 * d), dtype=np.float32)
            vectors.append(vector)
    return np.array(vectors)

def write_ivecs(filename, vectors):
    """Write integer vectors in .ivecs format."""
    vectors = vectors.astype(np.int32)
    with open(filename, 'wb') as f:
        for vector in vectors:
            d = np.int32(len(vector))
            f.write(d.tobytes())
            f.write(vector.tobytes())
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
