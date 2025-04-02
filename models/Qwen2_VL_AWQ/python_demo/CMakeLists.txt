cmake_minimum_required(VERSION 3.10)
project(Qwen2-VL)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

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

# OpenCV configuration
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

foreach(CPP_FILE ${CPP_FILES})
    get_filename_component(MODULE_NAME ${CPP_FILE} NAME_WE)
    pybind11_add_module(${MODULE_NAME} ${CPP_FILE})
    target_link_libraries(${MODULE_NAME} PUBLIC bmrt bmlib opencv_core opencv_imgcodecs opencv_videoio)
    install(TARGETS ${MODULE_NAME} DESTINATION python)
endforeach()

