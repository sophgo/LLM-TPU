cmake_minimum_required(VERSION 2.8)
project(baichuan2)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(CMAKE_INSTALL_PREFIX install)

include_directories(${PROJECT_SOURCE_DIR}/include)
link_directories(${PROJECT_SOURCE_DIR}/lib)

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_executable(baichuan2 demo.cpp)
target_link_libraries(baichuan2 bmrt bmlib sentencepiece)

