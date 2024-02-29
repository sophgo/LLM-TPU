cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "DEBUG")
project(wizardocder-tpu)

message(STATUS "Host arch is: ${CMAKE_HOST_SYSTEM_PROCESSOR}")


if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  set(LIBSOPHON_DIR ${CMAKE_SOURCE_DIR}/lib_pcie CACHE STRING "The directory of libsophon")
elseif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(LIBSOPHON_DIR ${CMAKE_SOURCE_DIR}/lib_soc CACHE STRING "The directory of libsophon")
else()
  message(FATAL_ERROR "Detected architecture is neither x86_64 nor aarch64, Exit")
endif()

message(STATUS "LIBSOPHON_DIR is set to: ${LIBSOPHON_DIR}")

set(VOCAB_DIR ${CMAKE_SOURCE_DIR}/../vocab)




set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")

include(FetchContent)

set(THIRD_PARTY_DIR ${CMAKE_SOURCE_DIR}/third_party)


FetchContent_Declare(
  ctre
  SOURCE_DIR ${THIRD_PARTY_DIR}/compile-time-regular-expressions
  GIT_TAG        v3.8.1
)



FetchContent_Declare(
  simdjson
  SOURCE_DIR ${THIRD_PARTY_DIR}/simdjson
  GIT_TAG        v3.6.3
)


FetchContent_Declare(
  cnpy
  SOURCE_DIR ${THIRD_PARTY_DIR}/cnpy
)


FetchContent_MakeAvailable(simdjson cnpy ctre)


add_executable(wizardcoder demo.cpp gpt2tokenizer.cc)
target_include_directories(wizardcoder PRIVATE ${LIBSOPHON_DIR}/include)
target_link_directories(wizardcoder PRIVATE ${LIBSOPHON_DIR}/lib)
target_link_libraries(wizardcoder PRIVATE bmrt bmlib simdjson::simdjson ctre cnpy)
target_compile_definitions(wizardcoder PRIVATE "VOCAB_DIR=\"${VOCAB_DIR}/vocab.json\"")
