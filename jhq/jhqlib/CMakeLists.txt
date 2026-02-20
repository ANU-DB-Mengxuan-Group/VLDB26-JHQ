add_library(${LIBRARY_NAME} STATIC
        IndexJHQ.cpp
        IndexJHQ.h
        IndexIVFJHQ.h
        IndexIVFJHQ.cpp
)

target_include_directories(${LIBRARY_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}/jhqlib
        ${PROJECT_SOURCE_DIR}/external/faiss
)

# Link FAISS and all its dependencies
target_link_libraries(${LIBRARY_NAME} PUBLIC
        faiss
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
        OpenMP::OpenMP_CXX
        lapacke
)

# Additional system libraries that FAISS might need
if(UNIX)
    target_link_libraries(${LIBRARY_NAME} PUBLIC
            pthread
            m
            dl
    )
endif()

# Ensure we can find FAISS headers
target_compile_definitions(${LIBRARY_NAME} PUBLIC
        FINTEGER=int
)

# =================================================================
#  CPU Feature Detection and Compiler Flag Configuration
# =================================================================

# Include the CMake module for checking if source code compiles
include(CheckSourceCompiles)

# Set common compiler flags for warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${LIBRARY_NAME} PRIVATE
            -Wall
            -Wextra
            -Wno-unused-parameter
            -Wno-sign-compare
    )
endif()

# --- 1. Manually check for AVX-512 support ---
set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
set(CMAKE_REQUIRED_FLAGS "-mavx512f -mavx512vl -mavx512bw -mavx512dq")
include(CheckCXXSourceCompiles)
check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m512 a = _mm512_setzero_ps();
    __m512 b = _mm512_setzero_ps();
    __m512 c = _mm512_add_ps(a, b);
    (void)c;
    return 0;
}" HAVE_AVX512)
set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS}) # Restore flags

# --- 2. Manually check for AVX2 support (if AVX-512 is not found) ---
if(NOT HAVE_AVX512)
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "-mavx2 -mfma")
    check_cxx_source_compiles("
#include <immintrin.h>
int main() {
    __m256 a = _mm256_setzero_ps();
    __m256 b = _mm256_setzero_ps();
    __m256 c = _mm256_fmadd_ps(a, b, b);
    (void)c;
    return 0;
}" HAVE_AVX2)
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS}) # Restore flags
endif()


# --- 3. Add flags to the target based on detection results ---
if(HAVE_AVX512)
    message(STATUS "Host CPU supports AVX-512. Compiling with AVX-512 flags.")
    target_compile_options(${LIBRARY_NAME} PRIVATE
            -mavx512f
            -mavx512vl
            -mavx512bw
            -mavx512dq
            -mfma
    )
elseif(HAVE_AVX2)
    message(STATUS "Host CPU supports AVX2. Compiling with AVX2 and FMA flags.")
    target_compile_options(${LIBRARY_NAME} PRIVATE
            -mavx2
            -mfma
    )
else()
    message(STATUS "No AVX2 or AVX-512 support detected. Compiling with default instruction set.")
endif()

# =================================================================