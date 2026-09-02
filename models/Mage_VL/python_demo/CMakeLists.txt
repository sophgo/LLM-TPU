cmake_minimum_required(VERSION 3.10)
project(Mage_VL)

if (NOT DEFINED TARGET_ARCH)
    set(TARGET_ARCH pcie)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/opt/sophon/libsophon-current/include)
link_directories(/opt/sophon/libsophon-current/lib)

add_definitions(--std=c++17 -fPIC -Wall -Werror)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development)
execute_process(COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED CONFIG)

pybind11_add_module(chat chat.cpp)
target_link_libraries(chat PUBLIC bmrt bmlib)
install(TARGETS chat DESTINATION python)
