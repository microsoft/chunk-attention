# libtorch

if(TORCH_INSTALL_DIR STREQUAL "")

    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(PYTORCH_URL
            https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-${TORCH}%2Bcpu.zip
        )
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        set(PYTORCH_URL
            https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${TORCH}%2Bcpu.zip)
        set(BINARY_EXTENSION "so")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(PYTORCH_URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-${TORCH}.zip)
        set(BINARY_EXTENSION "dylib")
    endif()

    FetchContent_Declare(libtorch URL ${PYTORCH_URL})
    FetchContent_MakeAvailable(libtorch)
    set(TORCH_INSTALL_DIR
        ${libtorch_SOURCE_DIR}
        CACHE PATH "" FORCE)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        file(GLOB TORCH_DLL_FILES ${TORCH_INSTALL_DIR}/lib/*.dll)
        set(TORCH_DLLS
            ${TORCH_DLL_FILES}
            CACHE PATH "" FORCE)
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        file(GLOB TORCH_DLL_FILES ${TORCH_INSTALL_DIR}/lib/*.${BINARY_EXTENSION})
        set(TORCH_DLLS
            ${TORCH_DLL_FILES}
            CACHE PATH "" FORCE)
    endif()

endif()

message("TORCH_INSTALL_DIR=${TORCH_INSTALL_DIR}")
message("TORCH_DLLS=${TORCH_DLLS}")