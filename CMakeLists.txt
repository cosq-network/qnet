cmake_minimum_required(VERSION 3.15)
project(qnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(DEFINED ENV{VCPKG_ROOT} AND NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenBLAS REQUIRED)

add_library(qnet SHARED
    src/tensor.cpp
    src/blas.cpp
    src/ops.cpp
    src/graph.cpp
    src/safetensors.cpp
    src/serializer.cpp
    src/thread_pool.cpp
)

target_include_directories(qnet PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_include_directories(qnet PRIVATE
    ${OpenBLAS_INCLUDE_DIRS}
)

target_link_libraries(qnet PRIVATE
    Eigen3::Eigen
    ${OpenBLAS_LIBRARIES}
)

option(BUILD_PYTHON_BINDINGS "Build Python Extensions via pybind11" OFF)
if(BUILD_PYTHON_BINDINGS)
    find_package(pybind11 REQUIRED)
    pybind11_add_module(pyqnet src/bindings/python.cpp)
    target_link_libraries(pyqnet PRIVATE qnet)
endif()

option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
