cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(chatglm2)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LIBSOPHON /opt/sophon/libsophon-current)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Release")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -Wall -pthread -fPIC -flto")

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT error)


include(FetchContent)
FetchContent_Declare(
    sentencepiece
    GIT_REPOSITORY https://github.com/google/sentencepiece.git
    GIT_TAG v0.1.99
)
FetchContent_MakeAvailable(sentencepiece)
message(${sentencepiece_SOURCE_DIR})


FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.12.0
)
FetchContent_MakeAvailable(spdlog)


include_directories(${LIBSOPHON}/include)
include_directories(${sentencepiece_SOURCE_DIR}/src)
link_directories(${LIBSOPHON}/lib)
link_directories(${PROJECT_SOURCE_DIR}/lib)


add_library(chatglm2 src/chat.cc src/chat_inner.cc)
target_include_directories(chatglm2 PRIVATE ${LIBSOPHON}/include ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(chatglm2 PRIVATE bmrt bmlib sentencepiece-static spdlog::spdlog)



if(ipo_supported)
    message(STATUS "target libchatglm2 IPO / LTO enabled")
    set_property(TARGET chatglm2 PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()


add_subdirectory(c)
add_subdirectory(python)