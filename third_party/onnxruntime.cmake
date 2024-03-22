if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(ONNXRUNTIME_URL
        https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip
    )
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(ONNXRUNTIME_URL
        https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz
    )
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(ONNXRUNTIME_URL
        https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-osx-x64-${ORT_VERSION}.tgz
    )
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    set(ONNXRUNTIME_URL
        https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar
    )
endif()

FetchContent_Declare(onnxruntime_prebuilt 
                     URL ${ONNXRUNTIME_URL}
                     DOWNLOAD_EXTRACT_TIMESTAMP FALSE)
FetchContent_MakeAvailable(onnxruntime_prebuilt)
message("onnxruntime_prebuilt_SOURCE_DIR=${onnxruntime_prebuilt_SOURCE_DIR}")

if(${CMAKE_SYSTEM_NAME} MATCHES "Android")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION 
        ${onnxruntime_prebuilt_SOURCE_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so)
    target_include_directories(onnxruntime INTERFACE 
        ${onnxruntime_prebuilt_SOURCE_DIR}/headers)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_IMPLIB ${onnxruntime_prebuilt_SOURCE_DIR}/lib/onnxruntime.lib)
    set_target_properties(onnxruntime PROPERTIES BINARY_DIRECTORY ${onnxruntime_prebuilt_SOURCE_DIR}/lib)
    target_include_directories(onnxruntime INTERFACE ${onnxruntime_prebuilt_SOURCE_DIR}/include)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION 
        ${onnxruntime_prebuilt_SOURCE_DIR}/lib/onnxruntime.so)
    target_include_directories(onnxruntime INTERFACE 
        ${onnxruntime_prebuilt_SOURCE_DIR}/headers)
else()
    message(FATAL_ERROR "${CMAKE_SYSTEM_NAME} is not supported yet")
endif()

