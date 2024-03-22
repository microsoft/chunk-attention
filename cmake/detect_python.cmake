# https://www.scivision.dev/cmake-find-python-conda-deactivate/
# CMake Find modules are by their nature a little aggressive about finding libraries and executables. This becomes a factor on Windows in particular when Anaconda Python is not active in the current Terminal. CMake find_package(Python) by default prefers Anaconda Python over system Python unless overridden as below. Anaconda Python won’t operate correctly without conda activate, which presumably the user has either forgotten to do or doesn’t desire at the moment. To decrease the aggressiveness and find Windows Store Python etc. when conda isn’t activated on Windows, add to the project CMakeLists.txt before find_package(Python):

# this avoids non-active conda from getting picked anyway on Windows
set(Python_FIND_REGISTRY LAST)
# Use environment variable PATH to decide preference for Python
set(Python_FIND_VIRTUALENV STANDARD)

# detect python executable
if(PYTHON STREQUAL "")
    find_package(Python3 COMPONENTS Interpreter Development)
else()
    find_package(Python3 ${PYTHON} EXACT COMPONENTS Interpreter Development)
endif()
if(NOT ${Python3_FOUND} OR NOT ${Python3_Interpreter_FOUND})
    message(WARNING "Python_FOUND=${Python3_FOUND} Python_Interpreter_FOUND=${Python3_Interpreter_FOUND}")
endif()
set(Python3_EXECUTABLE
    ${Python3_EXECUTABLE}
    CACHE FILEPATH "" FORCE)
set(Python3_LIBRARIES
        ${Python3_LIBRARIES}
        CACHE FILEPATH "" FORCE)
set(Python3_LIBRARY_DIRS
        ${Python3_LIBRARY_DIRS}
        CACHE FILEPATH "" FORCE)
set(Python3_INCLUDE_DIRS
        ${Python3_INCLUDE_DIRS}
        CACHE FILEPATH "" FORCE)
message(PYTHON_EXECUTABLE=${Python3_EXECUTABLE})