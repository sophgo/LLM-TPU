cmake_minimum_required(VERSION 2.8)
project(lwm)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)


add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(lwm demo.cpp)
target_link_libraries(lwm bmlib bmrt sentencepiece)