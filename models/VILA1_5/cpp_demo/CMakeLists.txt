cmake_minimum_required(VERSION 3.10)
project(vila_cpp)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/opencv4)
include_directories(/opt/sophon/libsophon-current/include)
include_directories(/opt/sophon//sophon-ffmpeg-latest/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)
link_directories(/opt/sophon/libsophon-current/lib)
link_directories(/opt/sophon/sophon-opencv-latest/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_library(vila_cpp vlm_interface.cpp PillowResize.cpp)
target_link_libraries(vila_cpp PUBLIC bmrt bmlib bmcv opencv_core opencv_imgproc opencv_imgcodecs sentencepiece)
target_compile_options(vila_cpp PRIVATE -Wno-error=attributes)
install(TARGETS vila_cpp DESTINATION lib)

add_executable(vlm_test vlm_test.cpp)
target_link_libraries(vlm_test PUBLIC vila_cpp opencv_videoio)
target_compile_options(vlm_test PRIVATE -Wno-error=attributes)
install(TARGETS vlm_test DESTINATION bin)


