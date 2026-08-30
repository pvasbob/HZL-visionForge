cmake_minimum_required(VERSION 3.20)

project(
    HZLVisionForge
    VERSION 0.1.0
    DESCRIPTION "Interactive GPU-accelerated image and video processing"
    LANGUAGES CXX
)

include(CheckLanguage)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found. Install the CUDA toolkit or set CUDACXX.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(CTest)
include(HZLWarnings)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to compile")
endif()

add_library(hzl_project_options INTERFACE)
target_compile_features(hzl_project_options INTERFACE cxx_std_17)
hzl_enable_warnings(hzl_project_options)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/hzl/version.hpp"
    @ONLY
)

add_executable(
    hzl-visionforge
    src/app/main.cpp
    src/media/opencv_environment.cpp
    src/platform/cuda_environment.cu
)
target_link_libraries(
    hzl-visionforge
    PRIVATE
        hzl_project_options
        CUDA::cudart
        ${OpenCV_LIBS}
)
target_include_directories(
    hzl-visionforge
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        ${OpenCV_INCLUDE_DIRS}
)

if(BUILD_TESTING)
    add_test(NAME application.version COMMAND hzl-visionforge --version)
    set_tests_properties(
        application.version
        PROPERTIES PASS_REGULAR_EXPRESSION "HZL-VisionForge ${PROJECT_VERSION}"
    )

    add_test(NAME environment.cuda COMMAND hzl-visionforge --gpu-info)
    set_tests_properties(
        environment.cuda
        PROPERTIES
            PASS_REGULAR_EXPRESSION "CUDA toolkit:"
            SKIP_RETURN_CODE 2
    )

    add_test(NAME environment.opencv COMMAND hzl-visionforge --opencv-info)
    set_tests_properties(
        environment.opencv
        PROPERTIES PASS_REGULAR_EXPRESSION "OpenCV smoke test: passed"
    )
endif()
