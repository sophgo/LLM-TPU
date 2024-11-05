cmake_minimum_required(VERSION 3.10)
project(qwen2)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/../support/include)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../support/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../support/lib_pcie)
    message("PCIE mode, starting......")
endif()

if (CMAKE_TYPE STREQUAL "DUMP")
    add_definitions(-DDUMP_TENSOR)
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -lcnpy)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/cnpy)
    add_subdirectory(third_party/cnpy)
else()
    add_definitions(-DDEBUG --std=c++17 -fPIC -Wall)
endif()

set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)


pybind11_add_module(chat ${CPP_FILES})
if (CMAKE_TYPE STREQUAL "DUMP")
    target_link_libraries(chat PUBLIC bmrt bmlib cnpy)
else()
    target_link_libraries(chat PUBLIC bmrt bmlib)
endif()
install(TARGETS chat DESTINATION python)