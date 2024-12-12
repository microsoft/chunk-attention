# cmake -S . -B build -G "Visual Studio 16 2019" -A x64
# cmake --build build --config Release

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(chunk_attn VERSION 0.1)

include(CMakeDependentOption)
include(cmake/policy.cmake)

# c/c++ common compilation options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
endif()
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_compile_definitions(NOMINMAX)
endif()
add_compile_definitions(NVTX_DISABLE)

set(PYTHON
        ""
        CACHE STRING "set which version of python to use, default to auto detect")
include(cmake/detect_python.cmake)

set(TORCH
        ""
        CACHE STRING "set PyTorch installation path")
if(TORCH STREQUAL "")
    message(FATAL_ERROR
            "Please set PyTorch installation path by \"-DTORCH=</path/to/python>/site-packages/torch\"\n"
            "Find your PyTorch installation path by running:\n"
            "python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)'")
else()
    find_package(Torch REQUIRED PATHS ${TORCH})
    message("TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}")
    message("TORCH_LIBRARIES=${TORCH_LIBRARIES}")
    message("TORCH_INCLUDE_DIRS=${TORCH_INCLUDE_DIRS}")
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
    message("TORCH_PYTHON_LIBRARY=${TORCH_PYTHON_LIBRARY}")
endif()

option(USE_MKL "compile CPU kernel with MKL BLAS" ON)
option(USE_CUDA "compile GPU kernel with CUDA" OFF)

if(USE_MKL)
    add_compile_definitions(NVTX_DISABLE)
endif()

add_subdirectory(third_party)
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(DIST_PLATFORM "--plat-name=win_amd64")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(DIST_PLATFORM "--plat-name=manylinux1_x86_64")
endif()

add_subdirectory(cpp)

if(${Python3_FOUND} AND ${Python3_Interpreter_FOUND})
    add_custom_target(
        chunk_attn_whl ALL
        DEPENDS chunk_attn_c
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/src/chunk_attn/lib/
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:chunk_attn_c>
        ${CMAKE_SOURCE_DIR}/src/chunk_attn/lib/
        #    COMMAND "${Python3_EXECUTABLE}" setup.py bdist_wheel ${DIST_PLATFORM}
        COMMAND "${Python3_EXECUTABLE}" -m pip install -e .
        COMMAND "${Python3_EXECUTABLE}" -c \"import mypy.stubgen\; mypy.stubgen.main(['-p', 'chunk_attn.lib.chunk_attn_c', '-o', './src'])\"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
endif()
