# Add executable for the demo
add_executable(demo_jrpq_openai_1024_100k_test demo_jhq_test.cpp)
include_directories("${PROJECT_BINARY_DIR}/configured_files/include")
target_link_libraries(demo_jrpq_openai_1024_100k_test PRIVATE ${LIBRARY_NAME})
target_include_directories(demo_jrpq_openai_1024_100k_test PRIVATE
        ${CMAKE_SOURCE_DIR}
)

# Add executable for the demo
add_executable(demo_ivfjrpq_openai_1024_100k_test demo_ivfjhq_test.cpp)
include_directories("${PROJECT_BINARY_DIR}/configured_files/include")
target_link_libraries(demo_ivfjrpq_openai_1024_100k_test PRIVATE ${LIBRARY_NAME})
target_include_directories(demo_ivfjrpq_openai_1024_100k_test PRIVATE
        ${CMAKE_SOURCE_DIR}
)
