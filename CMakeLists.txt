cmake_minimum_required(VERSION 3.15)
project(qnet VERSION 0.1.0 LANGUAGES CXX)

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

target_link_libraries(qnet PRIVATE
    Eigen3::Eigen
    OpenBLAS::OpenBLAS
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

include(GNUInstallDirs)

# ── Install rules ────────────────────────────────────────────────────
install(TARGETS qnet
    EXPORT qnetTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/qnet
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

# ── CMake package config (find_package) ──────────────────────────────
include(CMakePackageConfigHelpers)

set(qnet_CMAKE_CONFIG_DEST "${CMAKE_INSTALL_LIBDIR}/cmake/qnet")

install(EXPORT qnetTargets
    FILE qnetTargets.cmake
    NAMESPACE qnet::
    DESTINATION "${qnet_CMAKE_CONFIG_DEST}"
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/qnetConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/qnetConfig.cmake"
    INSTALL_DESTINATION "${qnet_CMAKE_CONFIG_DEST}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/qnetConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/qnetConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/qnetConfigVersion.cmake"
    DESTINATION "${qnet_CMAKE_CONFIG_DEST}"
)

export(EXPORT qnetTargets
    NAMESPACE qnet::
    FILE "${CMAKE_CURRENT_BINARY_DIR}/qnetTargets.cmake"
)

# ── CPack packaging ──────────────────────────────────────────────────
set(CPACK_PACKAGE_NAME "qnet")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Lightweight C++ Neural Network Graph Framework")
set(CPACK_PACKAGE_VENDOR "COSQ Network Private Limited")
set(CPACK_PACKAGE_CONTACT "info@cosq.net")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/cosq/qnet")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(CPACK_COMPONENTS_ALL Runtime Development)

# ── Platform-specific generators ─────────────────────────────────────
if(WIN32)
    set(CPACK_GENERATOR "NSIS;ZIP")
    set(CPACK_NSIS_DISPLAY_NAME "qnet ${CPACK_PACKAGE_VERSION}")
    set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
    set(CPACK_NSIS_URL_INFO_ABOUT "${CPACK_PACKAGE_HOMEPAGE_URL}")
    set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES64")
    set(CPACK_NSIS_MODIFY_PATH ON)

elseif(APPLE)
    set(CPACK_GENERATOR "DragNDrop;TGZ")
    set(CPACK_DMG_VOLUME_NAME "qnet ${CPACK_PACKAGE_VERSION}")
    set(CPACK_DMG_FORMAT "UDBZ")

else()
    set(CPACK_GENERATOR "DEB;RPM;TGZ")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "${CPACK_PACKAGE_CONTACT}")
    set(CPACK_DEBIAN_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.deb")
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libopenblas-dev")
    set(CPACK_RPM_PACKAGE_REQUIRES "openblas-devel")
    set(CPACK_RPM_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.rpm")
endif()

include(CPack)
