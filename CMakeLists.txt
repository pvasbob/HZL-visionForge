cmake_minimum_required(VERSION 3.20)

project(
    HZLVisionForge
    VERSION 0.1.0
    DESCRIPTION "Interactive GPU-accelerated image and video processing"
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(HZLWarnings)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hzl_project_options INTERFACE)
target_compile_features(hzl_project_options INTERFACE cxx_std_17)
hzl_enable_warnings(hzl_project_options)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/hzl/version.hpp"
    @ONLY
)

add_executable(hzl-visionforge src/app/main.cpp)
target_link_libraries(hzl-visionforge PRIVATE hzl_project_options)
target_include_directories(
    hzl-visionforge
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

if(BUILD_TESTING)
    add_test(NAME application.version COMMAND hzl-visionforge --version)
    set_tests_properties(
        application.version
        PROPERTIES PASS_REGULAR_EXPRESSION "HZL-VisionForge ${PROJECT_VERSION}"
    )
endif()
