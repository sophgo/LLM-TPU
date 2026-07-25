cmake_minimum_required(VERSION 3.10)
project(qwen)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 
set(TARGET_ARCH pcie)

include_directories(/opt/sophon/libsophon-current/include)

if (${TARGET_ARCH} STREQUAL "pcie")
    add_definitions(-DPCIE_TARGET)
    # link_directories(${PROJECT_SOURCE_DIR}/../support/lib_pcie)
    link_directories(/opt/sophon/libsophon-current/lib/)
    # link_directories(/workspace3/libsophon/install/libsophon-0.5.1/lib/)
    message("Pcie mode, starting......")
endif()

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(chat chat.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)
