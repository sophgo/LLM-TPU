cmake_minimum_required(VERSION 3.10)
project(janus)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(${PROJECT_SOURCE_DIR}/../../../support/include)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../../../support/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../../../support/lib_pcie)
    message("PCIE mode, starting......")
endif()

add_definitions(-DDEBUG -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

pybind11_add_module(chat ${CPP_FILES})
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)

