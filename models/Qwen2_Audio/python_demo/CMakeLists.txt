cmake_minimum_required(VERSION 3.10)
project(Qwen2-Audio)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH soc)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
set(pybind11_DIR "$ENV{HOME}/.local/lib/python3.10/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(chat chat.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)

