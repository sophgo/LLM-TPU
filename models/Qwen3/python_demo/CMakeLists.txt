cmake_minimum_required(VERSION 3.10)
project(qwen3)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(-DDEBUG -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

pybind11_add_module(chat ${CPP_FILES})
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)

