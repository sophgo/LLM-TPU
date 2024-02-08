

add_library(bmglm2 chatglm_c.cc)
set_property(TARGET bmglm2 PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(bmglm2 PUBLIC chatglm2)
target_include_directories(bmglm2 PRIVATE ${PROJECT_SOURCE_DIR}/include)

