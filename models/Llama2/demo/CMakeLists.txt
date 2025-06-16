cmake_minimum_required(VERSION 3.10)
project(llama2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "") 

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_executable(llama2 demo.cpp)
target_link_libraries(llama2 bmlib bmrt sentencepiece)
