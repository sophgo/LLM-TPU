cmake_minimum_required(VERSION 3.10)
project(MiniCPMV4_7_1B)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

set(CMAKE_CXX_STANDARD 17)

# Prefer the on-board SDK; fall back to the headers/libs shipped in
# LLM-TPU/support for containers without /opt/sophon.
if (EXISTS /opt/sophon/libsophon-current/include)
    include_directories(/opt/sophon/libsophon-current/include)
    link_directories(/opt/sophon/libsophon-current/lib)
else()
    include_directories(${PROJECT_SOURCE_DIR}/../../../support/include)
    if (TARGET_ARCH STREQUAL "soc")
        link_directories(${PROJECT_SOURCE_DIR}/../../../support/lib_soc)
    else()
        link_directories(${PROJECT_SOURCE_DIR}/../../../support/lib_pcie)
    endif()
endif()

add_definitions(-DDEBUG -fPIC -Wall -Werror)
set(CMAKE_BUILD_TYPE "Debug")

find_package(pybind11 REQUIRED CONFIG)

file(GLOB CPP_FILES ${PROJECT_SOURCE_DIR}/*.cpp)

pybind11_add_module(chat ${CPP_FILES})
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)
