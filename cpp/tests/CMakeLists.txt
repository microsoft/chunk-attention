add_executable(test_chunk_attn)
if (USE_CUDA)
    target_sources(test_chunk_attn
            PRIVATE
            test_gpu_kernel.cpp
    )
endif ()
if (USE_MKL)
    target_sources(test_chunk_attn
            PRIVATE
            test_cpu_kernel.cpp
    )
endif ()
target_link_libraries(test_chunk_attn
        PRIVATE
        gtest
        gtest_main
        chunk_attn_lib)

add_test(
        NAME test_pyis_cpp
        COMMAND $<TARGET_FILE:test_chunk_attn>
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)