cmake_minimum_required(VERSION 3.10)
project(qwen)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 
set(TARGET_ARCH pcie)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/sophon/libsophon-current/include)

if (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    # link_directories(${PROJECT_SOURCE_DIR}/../support/lib_pcie)
    link_directories(/opt/sophon/libsophon-current/lib/)
    message("Pcie mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)

add_executable(qwen_parallel demo_parallel.cpp tokenizer.cpp)
target_link_libraries(qwen_parallel bmrt bmlib re2)
