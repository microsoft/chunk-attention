# pybind11
FetchContent_Declare(
    pybind11
    DOWNLOAD_EXTRACT_TIMESTAMP FALSE
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.11.1.zip
)
FetchContent_MakeAvailable(pybind11)