cmake_minimum_required(VERSION 3.10)
project(model)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

if (NOT DEFINED TYPE)
    set(TYPE "default")
endif()

include_directories(${PROJECT_SOURCE_DIR}/../../support/include)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../../support/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../../support/lib_pcie)
    message("PCIE mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -lcryptopp)

set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

# file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

# pybind11_add_module(chat ${CPP_FILES})
pybind11_add_module(chat chat.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib ${OPENCV_LIBS})
install(TARGETS chat DESTINATION python)

add_library(cipher SHARED crypto.cpp)
target_link_libraries(cipher cryptopp)

add_library(cipher_varlen SHARED crypto_varlen.cpp)
target_link_libraries(cipher_varlen cryptopp)
