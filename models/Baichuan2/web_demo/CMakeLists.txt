cmake_minimum_required(VERSION 2.8)
project(baichuan2)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(CMAKE_INSTALL_PREFIX install)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
	add_definitions(-DSOC_TARGET)
	link_directories(${PROJECT_SOURCE_DIR}/../src/lib_soc)
	message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../src/lib_pcie)
	message("Pcie mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "soc")
    add_definitions(-DSOC_TARGET)
    set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
    link_directories(${PROJECT_SOURCE_DIR}/lib_soc)
	message("SoC mode, starting......")
endif()




include_directories(${PROJECT_SOURCE_DIR}/../src/include)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_library(tpuchat SHARED chat.cpp)
target_link_libraries(tpuchat bmrt bmlib sentencepiece)
