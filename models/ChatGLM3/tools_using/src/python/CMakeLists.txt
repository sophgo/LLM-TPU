project(pybmglm2)

find_package(Python REQUIRED COMPONENTS Interpreter Development)
find_package(SWIG REQUIRED COMPONENTS python)
include(UseSWIG)
set(UseSWIG_TARGET_NAME_PREFERENCE STANDARD)

macro(configure_pyglm2 source)
    set_source_files_properties(${source} PROPERTIES
        CPLUSPLUS ON
        USE_TARGET_INCLUDE_DIRECTORIES TRUE
    )
endmacro()

swig_add_library(pyglm2
    TYPE SHARED
    LANGUAGE python
    SOURCES pyglm2.i
)

set_property(TARGET pyglm2 PROPERTY SWIG_USE_TARGET_INCLUDE_DIRECTORIES TRUE)
set_source_files_properties(pyglm2.i PROPERTIES
    CPLUSPLUS ON
    INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/../c/
)

target_include_directories(pyglm2 PRIVATE  ${PROJECT_SOURCE_DIR}/../c/)

configure_pyglm2(pyglm2.i)

target_link_libraries(pyglm2 PRIVATE
    bmglm2
    Python::Module
)