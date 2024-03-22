# gtest

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0)
set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
set(BUILD_GMOCK
    OFF
    CACHE BOOL "" FORCE)
set(INSTALL_GTEST
    OFF
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)