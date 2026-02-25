set(JHQLIB_NAME jhqlib)

add_library(${JHQLIB_NAME} STATIC
        IndexJHQ.cpp
        IndexJHQ.h
        impl/IndexJHQTrain.cpp
        impl/IndexJHQSearch.cpp
        impl/IndexJHQIO.cpp
        IndexIVFJHQ.h
        IndexIVFJHQ.cpp
        impl/IndexIVFJHQTrain.cpp
        impl/IndexIVFJHQSearch.cpp
        impl/IndexIVFJHQCodec.cpp
        impl/IndexIVFJHQOptimize.cpp
        impl/IndexIVFJHQPredecode.cpp
        impl/IndexIVFJHQIO.cpp
        impl/IndexIVFJHQScanner.h
        impl/IndexIVFJHQScanner.cpp
)

target_include_directories(${JHQLIB_NAME} PUBLIC
        ${PROJECT_SOURCE_DIR}/jhqlib
        ${PROJECT_SOURCE_DIR}/external/faiss
)

target_link_libraries(${JHQLIB_NAME} PUBLIC
        faiss
        ${BLAS_LIBRARIES}
        OpenMP::OpenMP_CXX
)
