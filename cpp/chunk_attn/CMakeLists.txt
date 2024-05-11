# pybind11 use the variable PYTHON_EXECUTABLE(case sensitive) to detect python
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

add_library(chunk_attn_lib STATIC)
target_sources(chunk_attn_lib PRIVATE
        "chunk_info.h"
        "chunk.h"
        "chunk.cpp"
        "chunk_allocator.h"
        "chunk_allocator.cpp"
        "kv_cache.h"
        "kv_cache.cpp"
        "attention.h"
        "attention.cpp"
        "task_executor.h"
        "task_executor.cpp"
        "logging.h"
        "logging.cpp"
        "str_utils.h"
        "str_utils.cpp"
        "small_vector.h")

target_include_directories(chunk_attn_lib PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${TORCH_INCLUDE_DIRS}
        ${SPDLOG_INCLUDE_DIR})
target_link_libraries(chunk_attn_lib PUBLIC
        ${TORCH_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY})
message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")
target_link_directories(chunk_attn_lib PUBLIC
        ${TORCH_INSTALL_PREFIX}/lib)

if (USE_MKL)
    find_package(MKL CONFIG REQUIRED)
    find_package(OpenMP)
    message(STATUS "${MKL_IMPORTED_TARGETS}")
    message("MKL_H=${MKL_H}")
    target_compile_options(chunk_attn_lib PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
    target_sources(chunk_attn_lib PRIVATE
            "kernel_cpu_mkl.h"
            "kernel_cpu_mkl.cpp"
            "kernel_cpu_tls.h"
            "spin_lock.h")
    target_include_directories(chunk_attn_lib PUBLIC
            $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
    # add this because torch will link mkl automatically if you install MKL
    target_link_directories(chunk_attn_lib PUBLIC
            ${MKL_H}/../lib/intel64)
    target_link_libraries(chunk_attn_lib PUBLIC
            $<LINK_ONLY:MKL::MKL>
            OpenMP::OpenMP_CXX
    )
    target_compile_definitions(chunk_attn_lib PUBLIC USE_MKL=1)
endif ()

# https://cliutils.gitlab.io/modern-cmake/chapters/packages/CUDA.html
if (USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    target_sources(chunk_attn_lib PRIVATE
        kernel_cuda.cu
        kernel_cuda.h
        cuda_compat.h
        reduction_utils.cuh       
        pos_encoding_kernels.h
        pos_encoding_kernels.cu
        layernorm_kernels.h
        layernorm_kernels.cu     
    )
    target_link_libraries(chunk_attn_lib PUBLIC
                nvtx3-cpp)
    target_compile_definitions(chunk_attn_lib PUBLIC USE_CUDA=1)
    set_target_properties(chunk_attn_lib PROPERTIES CUDA_ARCHITECTURES "86")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        target_compile_options(chunk_attn_lib PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:-G>)
        target_compile_options(chunk_attn_lib PUBLIC $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Release>>:-O3>)
    endif()
endif ()

if (TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
endif ()

if(${Python3_FOUND} AND ${Python3_Interpreter_FOUND})
    pybind11_add_module(chunk_attn_c)
    target_sources(chunk_attn_c PRIVATE
            "python_exports.cpp"
    )
    if(USE_CUDA)
        target_sources(chunk_attn_c PRIVATE
                "python_exports_ops.cpp"
        )
    endif()
    target_link_libraries(chunk_attn_c PUBLIC chunk_attn_lib)
endif()