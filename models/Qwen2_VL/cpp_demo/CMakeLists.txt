cmake_minimum_required(VERSION 3.10)
project(qwen2_vl)
set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/third_party/include)
include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/third_party/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/third_party/lib_pcie)
    message("PCIE mode, starting......")
endif()

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_videoio)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(qwen2_vl demo.cpp)
target_link_libraries(qwen2_vl PUBLIC bmlib bmrt tokenizers_cpp  -lpthread -ltokenizers_c ${OPENCV_LIBS} dl)
