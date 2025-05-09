cmake_minimum_required(VERSION 3.10)
project(qwen3)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/lib_pcie)
    message("PCIE mode, starting......")
endif()


add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(qwen3 demo.cpp)
target_link_libraries(qwen3 PUBLIC bmrt bmlib tokenizers_cpp tokenizers_c sentencepiece pthread dl)
target_compile_options(qwen3 PRIVATE -Wno-error=attributes)