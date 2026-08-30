#include "hzl/processing/cuda_memory.hpp"
#include "hzl/rendering/cuda_gl_interop.hpp"

#include <glad/gl.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool check(const bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
    }
    return condition;
}

std::vector<unsigned char> read_texture(const unsigned int texture,
                                        const std::size_t width,
                                        const std::size_t height) {
    std::vector<unsigned char> pixels(width * height * 4U);
    glGetTextureImage(texture,
                      0,
                      GL_RGBA,
                      GL_UNSIGNED_BYTE,
                      static_cast<GLsizei>(pixels.size()),
                      pixels.data());
    return pixels;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "CUDA-OpenGL interop tests skipped: CUDA device unavailable\n";
        return 77;
    }
    if (glfwInit() != GLFW_TRUE) {
        std::cout << "CUDA-OpenGL interop tests skipped: display unavailable\n";
        return 77;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow(64, 64, "HZL interop test", nullptr, nullptr);
    if (window == nullptr) {
        glfwTerminate();
        std::cout << "CUDA-OpenGL interop tests skipped: OpenGL 4.5 unavailable\n";
        return 77;
    }
    glfwMakeContextCurrent(window);
    if (gladLoadGL(glfwGetProcAddress) == 0) {
        glfwDestroyWindow(window);
        glfwTerminate();
        std::cerr << "FAIL: GLAD initialization failed\n";
        return 1;
    }

    constexpr std::size_t width = 17;
    constexpr std::size_t height = 13;
    constexpr std::size_t row_bytes = width * 4U;
    bool passed = true;
    {
        std::vector<unsigned char> first_pixels(row_bytes * height);
        std::vector<unsigned char> second_pixels(row_bytes * height);
        for (std::size_t index = 0; index < first_pixels.size(); ++index) {
            first_pixels[index] = static_cast<unsigned char>((index * 17U) % 251U);
            second_pixels[index] = static_cast<unsigned char>((index * 29U + 7U) % 253U);
        }

        hzl::processing::cuda::ImageBuffer cuda_image{width, height};
        hzl::rendering::CudaGlDoubleBuffer presentation;
        presentation.resize(width, height);
        const unsigned int original_front = presentation.front_texture_id();
        const unsigned int original_back = presentation.back_texture_id();
        passed = check(original_front != 0U && original_back != 0U &&
                           original_front != original_back,
                       "double buffer did not create two distinct textures") &&
                 passed;

        cuda_image.upload(first_pixels.data(), row_bytes);
        presentation.upload_back(cuda_image);
        passed = check(presentation.back_ready(), "back buffer was not marked ready") &&
                 passed;
        presentation.swap();
        passed = check(presentation.front_texture_id() == original_back,
                       "first swap did not present the original back texture") &&
                 passed;
        passed = check(read_texture(presentation.front_texture_id(), width, height) ==
                           first_pixels,
                       "first CUDA-to-OpenGL copy changed pixels") &&
                 passed;

        cuda_image.upload(second_pixels.data(), row_bytes);
        presentation.upload_back(cuda_image);
        presentation.swap();
        passed = check(presentation.front_texture_id() == original_front,
                       "second swap did not alternate textures") &&
                 passed;
        passed = check(read_texture(presentation.front_texture_id(), width, height) ==
                           second_pixels,
                       "second CUDA-to-OpenGL copy changed pixels") &&
                 passed;

        presentation.resize(width, height);
        passed = check(presentation.front_texture_id() == original_front &&
                           presentation.back_texture_id() == original_back,
                       "same-size resize replaced reusable interop textures") &&
                 passed;
        presentation.clear();
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    if (passed) {
        std::cout << "CUDA-OpenGL interop and double-buffer tests: passed\n";
        return 0;
    }
    return 1;
}
