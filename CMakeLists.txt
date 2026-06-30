cmake_minimum_required(VERSION 3.22)

project(higgs-audio-ggml-cpp
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build HiggsAudio.cpp dependencies as static libraries" FORCE)

set(HIGGS_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vendor/ggml" CACHE PATH "Path to vendored ggml")
option(HIGGS_ALLOW_WORKBENCH_GGML_FALLBACK "Allow local ggbond workbench ggml fallback when vendor/ggml is missing" OFF)

if(NOT EXISTS "${HIGGS_GGML_DIR}/CMakeLists.txt")
    set(HIGGS_WORKBENCH_GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/ggml")
    if(HIGGS_ALLOW_WORKBENCH_GGML_FALLBACK AND EXISTS "${HIGGS_WORKBENCH_GGML_DIR}/CMakeLists.txt")
        message(STATUS "vendor/ggml is missing; using workbench ggml at ${HIGGS_WORKBENCH_GGML_DIR}")
        set(HIGGS_GGML_DIR "${HIGGS_WORKBENCH_GGML_DIR}")
    else()
        message(FATAL_ERROR "vendor/ggml is missing; run git submodule update --init --recursive, pass -DHIGGS_GGML_DIR=/path/to/ggml, or set -DHIGGS_ALLOW_WORKBENCH_GGML_FALLBACK=ON for local ggbond development.")
    endif()
endif()

add_subdirectory("${HIGGS_GGML_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/ggml")

set(TOKENIZERS_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_CPP_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_CPP_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(TOKENIZERS_CPP_FETCH_DEPS OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/tokenizers.cpp)

add_library(shine STATIC
    third_party/shine/src/lib/bitstream.c
    third_party/shine/src/lib/huffman.c
    third_party/shine/src/lib/l3bitstream.c
    third_party/shine/src/lib/l3loop.c
    third_party/shine/src/lib/l3mdct.c
    third_party/shine/src/lib/l3subband.c
    third_party/shine/src/lib/layer3.c
    third_party/shine/src/lib/reservoir.c
    third_party/shine/src/lib/tables.c
)
target_include_directories(shine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/third_party/shine/src/lib")

add_library(higgs_audio_ggml_cpp STATIC
    src/higgs_pipeline.cpp
)

target_include_directories(higgs_audio_ggml_cpp
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/miniaudio"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/shine/src/lib"
        "${HIGGS_GGML_DIR}/include"
        "${HIGGS_GGML_DIR}/src"
)

target_link_libraries(higgs_audio_ggml_cpp PUBLIC ggml ggml-cpu tokenizers_cpp::tokenizers_cpp shine)
if(TARGET ggml-cuda)
    target_link_libraries(higgs_audio_ggml_cpp PUBLIC ggml-cuda)
endif()

add_executable(higgs-audio-cli src/higgs_audio_cli.cpp)
target_link_libraries(higgs-audio-cli PRIVATE higgs_audio_ggml_cpp)

add_executable(higgs-audio-server src/higgs_audio_server.cpp)
target_link_libraries(higgs-audio-server PRIVATE higgs_audio_ggml_cpp)

install(TARGETS higgs_audio_ggml_cpp higgs-audio-cli higgs-audio-server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

install(DIRECTORY docs/
    DESTINATION share/higgs-audio-ggml-cpp/docs
)
