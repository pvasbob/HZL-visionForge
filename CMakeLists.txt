cmake_minimum_required(VERSION 3.20)

project(
    HZLVisionForge
    VERSION 0.1.0
    DESCRIPTION "Interactive GPU-accelerated image and video processing"
    LANGUAGES C CXX
)

include(CheckLanguage)
include(FetchContent)
check_language(CUDA)
if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "CUDA compiler not found. Install the CUDA toolkit or set CUDACXX.")
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio)
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.8
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(glad)
set(GLAD_SOURCES_DIR "${glad_SOURCE_DIR}")
include("${glad_SOURCE_DIR}/cmake/GladConfig.cmake")
glad_add_library(glad_gl_core_45 REPRODUCIBLE API gl:core=4.5)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.9b-docking
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(
    imgui_backend STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_demo.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)
target_include_directories(
    imgui_backend
    PUBLIC
        "${imgui_SOURCE_DIR}"
        "${imgui_SOURCE_DIR}/backends"
)
target_link_libraries(imgui_backend PUBLIC glfw OpenGL::GL)
target_compile_features(imgui_backend PUBLIC cxx_std_17)
set_target_properties(
    imgui_backend
    PROPERTIES
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
)

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

add_library(
    hzl_media STATIC
    src/media/image_export.cpp
    src/media/image_loader.cpp
    src/media/opencv_environment.cpp
)
target_link_libraries(
    hzl_media
    PUBLIC
        hzl_project_options
        ${OpenCV_LIBS}
)
target_include_directories(
    hzl_media
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${OpenCV_INCLUDE_DIRS}
)

add_executable(
    hzl-visionforge
    src/app/main.cpp
    src/platform/cuda_environment.cu
    src/rendering/graphics_environment.cpp
    src/rendering/image_texture.cpp
    src/ui/imgui_layer.cpp
)
target_link_libraries(
    hzl-visionforge
    PRIVATE
        hzl_project_options
        hzl_media
        CUDA::cudart
        OpenGL::GL
        glad_gl_core_45
        glfw
        imgui_backend
)
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

    add_test(NAME environment.graphics COMMAND hzl-visionforge --graphics-info)
    set_tests_properties(
        environment.graphics
        PROPERTIES
            PASS_REGULAR_EXPRESSION "OpenGL context: 4\\.[5-9]"
            SKIP_RETURN_CODE 5
    )

    add_test(NAME environment.imgui COMMAND hzl-visionforge --imgui-info)
    set_tests_properties(
        environment.imgui
        PROPERTIES PASS_REGULAR_EXPRESSION "Dear ImGui smoke test: passed"
    )

    add_executable(image_loader_test tests/image_loader_test.cpp)
    target_link_libraries(image_loader_test PRIVATE hzl_media)
    add_test(NAME media.image_loader COMMAND image_loader_test)
    set_tests_properties(
        media.image_loader
        PROPERTIES PASS_REGULAR_EXPRESSION "Image loader tests: passed"
    )

    add_executable(image_export_test tests/image_export_test.cpp)
    target_link_libraries(image_export_test PRIVATE hzl_media)
    add_test(NAME media.image_export COMMAND image_export_test)
    set_tests_properties(
        media.image_export
        PROPERTIES PASS_REGULAR_EXPRESSION "Image export tests: passed"
    )
endif()
