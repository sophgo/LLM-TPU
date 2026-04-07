cmake_minimum_required(VERSION 3.10)
project(qwen3vl_distributed)
set(CMAKE_CXX_STANDARD 17)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

# OpenCV
if(SOPHON_OPENCV)
    include_directories(/opt/sophon/sophon-opencv-latest/include/opencv4)
    include_directories(/opt/sophon/sophon-ffmpeg-latest/include)
    link_directories(/opt/sophon/sophon-opencv-latest/lib)
else()
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()

# Include paths
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR})
include_directories(/opt/sophon/libsophon-current/include)
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

set(OPENCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

# Step0: Embedding + VIT + User Interface
add_executable(step0_embed_vit step0_main.cpp)
target_link_libraries(step0_embed_vit PUBLIC
    bmrt bmlib tokenizers_cpp tokenizers_c sentencepiece
    ${OPENCV_LIBS} pthread dl)
target_compile_options(step0_embed_vit PRIVATE -Wno-error=attributes)

# Step1: Block / Block_Cache
add_executable(step1_block step1_main.cpp)
target_link_libraries(step1_block PUBLIC bmrt bmlib pthread dl)
target_compile_options(step1_block PRIVATE -Wno-error=attributes)

# Step2: LMHead
add_executable(step2_lmhead step2_main.cpp)
target_link_libraries(step2_lmhead PUBLIC bmrt bmlib pthread dl)
target_compile_options(step2_lmhead PRIVATE -Wno-error=attributes)
