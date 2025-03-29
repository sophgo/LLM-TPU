cmake_minimum_required(VERSION 3.10)
project(qwen2_vl)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/opencv4)
include_directories(/opt/sophon/libsophon-current/include)
include_directories(/opt/sophon/sophon-ffmpeg-latest/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(/opt/sophon/libsophon-current/lib)
link_directories(/opt/sophon/sophon-opencv-latest/lib)

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

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_videoio opencv_imgcodecs)

add_executable(qwen2_vl demo.cpp)
target_link_libraries(qwen2_vl PUBLIC bmrt bmlib tokenizers_cpp tokenizers_c sentencepiece ${OPENCV_LIBS} pthread dl)
target_compile_options(qwen2_vl PRIVATE -Wno-error=attributes)