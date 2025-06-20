cmake_minimum_required(VERSION 3.10)
project(qwen2_5_vl)
set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/sophon/libsophon-current/include)

# 如果自带有/opt/sophon/sophon-opencv-latest，则添加
# include_directories(/opt/sophon/sophon-opencv-latest/include/opencv4)
# include_directories(/opt/sophon/sophon-ffmpeg-latest/include)
# link_directories(/opt/sophon/sophon-opencv-latest/lib)

link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(/opt/sophon/libsophon-current/lib)


if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    add_definitions(-DSOC_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/lib_soc)
    message("SoC mode, starting......")
elseif (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    link_directories(${PROJECT_SOURCE_DIR}/lib_pcie)
    message("PCIE mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

set(OPENCV_LIBS opencv_core opencv_imgcodecs)
add_executable(pipeline pipeline.cpp chat.cpp)
target_link_libraries(pipeline PUBLIC bmrt bmlib tokenizers_cpp tokenizers_c sentencepiece ${OPENCV_LIBS} pthread dl)
target_compile_options(pipeline PRIVATE -Wno-error=attributes)