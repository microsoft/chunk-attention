
#spdlog
FetchContent_Declare(
    spdlog
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL https://github.com/gabime/spdlog/archive/refs/tags/v1.13.0.zip
)
FetchContent_MakeAvailable(spdlog)

set(SPDLOG_INCLUDE_DIR
    ${spdlog_SOURCE_DIR}/include
    CACHE PATH "" FORCE)