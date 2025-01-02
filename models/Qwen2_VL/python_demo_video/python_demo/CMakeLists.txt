cmake_minimum_required(VERSION 3.10)
project(Qwen2-VL)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(SUPPORT_DIR ${PROJECT_SOURCE_DIR}/../../support)

include_directories(${SUPPORT_DIR}/include)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${SUPPORT_DIR}/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${SUPPORT_DIR}/lib_pcie)
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

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp ${SUPPORT_DIR}/src/*.cpp)

find_package(ZLIB REQUIRED)

pybind11_add_module(chat ${CPP_FILES})
if (CMAKE_TYPE STREQUAL "DUMP")
    target_link_libraries(chat PUBLIC bmrt bmlib cnpy ZLIB::ZLIB)
else()
    target_link_libraries(chat PUBLIC bmrt bmlib ZLIB::ZLIB)
endif()
install(TARGETS chat DESTINATION python)