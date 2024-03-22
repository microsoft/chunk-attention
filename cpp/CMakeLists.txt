# pybind11 use the variable PYTHON_EXECUTABLE(case sensitive) to detect python
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

add_subdirectory(chunk_attn)
add_subdirectory(tests)
