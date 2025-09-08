cmake_minimum_required(VERSION 3.20)
project(qwen3)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(/opt/tpuv7/tpuv7-current/include)
link_directories(/opt/tpuv7/tpuv7-current/lib)

add_definitions(-DDEBUG -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

pybind11_add_module(chat ${CPP_FILES})
target_link_libraries(chat PUBLIC tpuv7_modelrt tpuv7_rt)
install(TARGETS chat DESTINATION python)

