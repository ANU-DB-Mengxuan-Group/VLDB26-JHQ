cmake_minimum_required(VERSION 3.27)

project(
        "JHQ"
        VERSION 1.0.0
        LANGUAGES C CXX)

# Global CMake variables are set here
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Modify this to your desired dataset dir
set(DATA_DIR /datasets/)
# Modify this to your desired index dir
set(INDEX_DIR /index/)

set(RESULT_DIR ${PROJECT_SOURCE_DIR}/result/)

# Options
option(ENABLE_WARNINGS "Enable to add warnings to a target." ON)
option(ENABLE_WARNINGS_AS_ERRORS "Enable to treat warnings as errors." OFF)

option(ENABLE_TESTING "Enable a Unit Testing build." ON)

option(ENABLE_CLANG_TIDY "Enable to add clang tidy." ON)

option(ENABLE_CLANG_FORMAT "Enable to add clang-format." ON)
option(ENABLE_CMAKE_FORMAT "Enable to add cmake-format." ON)

# Project/Library Names
set(LIBRARY_NAME "jhqlib")
set(UNIT_TEST_NAME "unit_tests")
set(EXECUTABLE_NAME "main")

# CMAKE MODULES
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

set(CMAKE_XCODE_BINARY_DIR ${PROJECT_SOURCE_DIR}/build-xcode)

if(ENABLE_WARNINGS)
    include(Warnings)
endif()


# EXTERNAL LIBRARIES
message(STATUS "Using CPM")
include(CPM)

# Find dependencies for FAISS
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(OpenMP REQUIRED)

# Add FAISS subdirectory
option(FAISS_ENABLE_GPU "Enable support for GPU indexes." OFF)
set(FAISS_ENABLE_PYTHON OFF)
# === Disable FAISS perf tests and other test targets ===
set(FAISS_ENABLE_PERF_TESTS OFF)
set(BUILD_TESTING OFF)
add_subdirectory(external/faiss EXCLUDE_FROM_ALL)

# SUB DIRECTORIES

add_subdirectory(configured)
add_subdirectory(external)
add_subdirectory(jhqlib)
add_subdirectory(examples)

set(FAISS_ENABLE_GPU OFF CACHE BOOL "Disable GPU support in FAISS")
set(FAISS_ENABLE_PYTHON OFF CACHE BOOL "Disable Python bindings in FAISS")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/jhqlib)
include_directories(${PROJECT_SOURCE_DIR}/external/faiss)