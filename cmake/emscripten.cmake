# Emscripten toolchain helper for qnet WASM builds
# Usage:
#   emcmake cmake -B build_wasm -S . -DCMAKE_TOOLCHAIN_FILE=$EMSDK/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake
#
# Then build with: emmake make -C build_wasm

message(STATUS "Configuring qnet for Emscripten/WASM target")

add_library(qnet_wasm STATIC
    src/tensor.cpp
    src/ops.cpp
    src/graph.cpp
    src/safetensors.cpp
)

target_include_directories(qnet_wasm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(qnet_wasm PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# WASM-specific compile flags
target_compile_options(qnet_wasm PRIVATE
    -s WASM=1
    -s ALLOW_MEMORY_GROWTH=1
    -O3
    -flto
)
