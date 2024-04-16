cmake_minimum_required(VERSION 3.10)
project(llama2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 

set(TARGET_ARCH pcie)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/sophon/libsophon-current/include)

if (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/lib/)
    link_directories(/opt/sophon/libsophon-current/lib)
	message("Pcie mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(llama2_parallel demo_parallel.cpp)
target_link_libraries(llama2_parallel bmlib bmrt sentencepiece)
