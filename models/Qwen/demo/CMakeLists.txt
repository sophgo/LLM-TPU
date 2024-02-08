cmake_minimum_required(VERSION 3.10)
project(qwen)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/../support/include)
# include_directories(${PROJECT_SOURCE_DIR}/../support/third_party/abseil-cpp) 
# include_directories(${PROJECT_SOURCE_DIR}/../support/third_party/re2)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
	add_definitions(-DSOC_TARGET)
	link_directories(${PROJECT_SOURCE_DIR}/../support/lib_soc)
	message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/../support/lib_pcie)
	message("Pcie mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "soc")
    add_definitions(-DSOC_TARGET)
    set(CMAKE_C_COMPILER /opt/aarch64-linux-gnu-7.5.0/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_ASM_COMPILER /opt/aarch64-linux-gnu-7.5.0/bin/aarch64-linux-gnu-gcc)
    set(CMAKE_CXX_COMPILER /opt/aarch64-linux-gnu-7.5.0/bin/aarch64-linux-gnu-g++)
    link_directories(${PROJECT_SOURCE_DIR}/../support/lib_soc)
	message("SoC mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

set(ABSL_ENABLE_INSTALL ON)
set(ABSL_PROPAGATE_CXX_STD ON)
# find_package(re2 REQUIRED)
add_subdirectory(third_party/abseil-cpp)
add_subdirectory(third_party/re2)
# link_directories(${PROJECT_SOURCE_DIR}/../support/third_party/abseil-cpp)
# link_directories(${PROJECT_SOURCE_DIR}/../support/third_party/re2)

# add_executable(tokenizer tokenizer.cpp)
# target_link_libraries(tokenizer re2)

add_executable(qwen demo.cpp tokenizer.cpp)
target_link_libraries(qwen bmrt bmlib re2)

