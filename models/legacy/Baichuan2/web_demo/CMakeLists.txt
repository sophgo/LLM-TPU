cmake_minimum_required(VERSION 2.8)
project(baichuan2)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(CMAKE_INSTALL_PREFIX install)

include_directories(${PROJECT_SOURCE_DIR}/../demo/include)
link_directories(${PROJECT_SOURCE_DIR}/../demo/lib)

include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(-DDEBUG --std=c++17 -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

add_library(tpuchat SHARED chat.cpp)
target_link_libraries(tpuchat bmrt bmlib sentencepiece)
