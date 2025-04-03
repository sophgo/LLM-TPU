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

# OpenCV configuration
if (${TYPE} STREQUAL "media")
    add_definitions(-DENABLE_MEDIA)
    find_package(OpenCV REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})
    link_directories(${OpenCV_LIBRARY_DIRS})
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
    set(OPENCV_LIBS opencv_core opencv_imgcodecs opencv_videoio)
else()
    set(OPENCV_LIBS "")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall)

set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

pybind11_add_module(chat ${CPP_FILES})
target_link_libraries(chat PUBLIC bmrt bmlib ${OPENCV_LIBS})
install(TARGETS chat DESTINATION python)
