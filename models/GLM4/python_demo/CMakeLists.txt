cmake_minimum_required(VERSION 3.10)
project(chatglm)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/../../../support/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/cnpy)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
	add_definitions(-DSOC_TARGET)
	link_directories(${PROJECT_SOURCE_DIR}/../../../support/lib_soc)
	message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../../../support/lib_pcie)
	message("PCIE mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)
pybind11_add_module(chat chat.cpp)

target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)