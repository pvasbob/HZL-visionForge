#include "hzl/processing/cuda_basic_filters.hpp"

#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace hzl::processing::cuda {
namespace {

constexpr unsigned int block_width = 16;
constexpr unsigned int block_height = 16;

__device__ unsigned char saturated_round(const float value) {
    return static_cast<unsigned char>(
        max(0, min(255, __float2int_rn(value))));
}

__global__ void grayscale_kernel(const unsigned char* input,
                                 const std::size_t input_pitch,
                                 unsigned char* output,
                                 const std::size_t output_pitch,
                                 const unsigned int width,
                                 const unsigned int height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* source = input + y * input_pitch + x * 4U;
    unsigned char* destination = output + y * output_pitch + x * 4U;
    const unsigned char gray = saturated_round(
        0.299F * static_cast<float>(source[0]) +
        0.587F * static_cast<float>(source[1]) +
        0.114F * static_cast<float>(source[2]));
    destination[0] = gray;
    destination[1] = gray;
    destination[2] = gray;
    destination[3] = source[3];
}

__global__ void invert_kernel(const unsigned char* input,
                              const std::size_t input_pitch,
                              unsigned char* output,
                              const std::size_t output_pitch,
                              const unsigned int width,
                              const unsigned int height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* source = input + y * input_pitch + x * 4U;
    unsigned char* destination = output + y * output_pitch + x * 4U;
    destination[0] = static_cast<unsigned char>(255U - source[0]);
    destination[1] = static_cast<unsigned char>(255U - source[1]);
    destination[2] = static_cast<unsigned char>(255U - source[2]);
    destination[3] = source[3];
}

__global__ void brightness_contrast_kernel(const unsigned char* input,
                                           const std::size_t input_pitch,
                                           unsigned char* output,
                                           const std::size_t output_pitch,
                                           const unsigned int width,
                                           const unsigned int height,
                                           const float brightness,
                                           const float contrast) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* source = input + y * input_pitch + x * 4U;
    unsigned char* destination = output + y * output_pitch + x * 4U;
    destination[0] = saturated_round(contrast * source[0] + brightness);
    destination[1] = saturated_round(contrast * source[1] + brightness);
    destination[2] = saturated_round(contrast * source[2] + brightness);
    destination[3] = source[3];
}

__global__ void gamma_kernel(const unsigned char* input,
                             const std::size_t input_pitch,
                             unsigned char* output,
                             const std::size_t output_pitch,
                             const unsigned int width,
                             const unsigned int height,
                             const float inverse_gamma) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const unsigned char* source = input + y * input_pitch + x * 4U;
    unsigned char* destination = output + y * output_pitch + x * 4U;
    for (unsigned int channel = 0; channel < 3; ++channel) {
        const float normalized = static_cast<float>(source[channel]) / 255.0F;
        destination[channel] =
            saturated_round(powf(normalized, inverse_gamma) * 255.0F);
    }
    destination[3] = source[3];
}

struct LaunchConfiguration {
    dim3 grid;
    dim3 block;
    unsigned int width;
    unsigned int height;
};

LaunchConfiguration prepare_launch(const ImageBuffer& input,
                                   ImageBuffer& output) {
    if (input.empty() || input.data() == nullptr) {
        throw std::invalid_argument{"CUDA filter input cannot be empty."};
    }
    if (input.width() > std::numeric_limits<unsigned int>::max() ||
        input.height() > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error{"CUDA filter dimensions exceed kernel limits."};
    }

    output.resize(input.width(), input.height());
    const auto width = static_cast<unsigned int>(input.width());
    const auto height = static_cast<unsigned int>(input.height());
    return {dim3{(width - 1U) / block_width + 1U,
                 (height - 1U) / block_height + 1U,
                 1U},
            dim3{block_width, block_height, 1U},
            width,
            height};
}

void finish_launch(const cudaStream_t stream, const char* operation) {
    check(cudaGetLastError(), operation);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

}  // namespace

void grayscale(const ImageBuffer& input,
               ImageBuffer& output,
               const cudaStream_t stream) {
    const LaunchConfiguration launch = prepare_launch(input, output);
    grayscale_kernel<<<launch.grid, launch.block, 0, stream>>>(
        static_cast<const unsigned char*>(input.data()),
        input.stride_bytes(),
        static_cast<unsigned char*>(output.data()),
        output.stride_bytes(),
        launch.width,
        launch.height);
    finish_launch(stream, "grayscale kernel launch");
}

void invert(const ImageBuffer& input,
            ImageBuffer& output,
            const cudaStream_t stream) {
    const LaunchConfiguration launch = prepare_launch(input, output);
    invert_kernel<<<launch.grid, launch.block, 0, stream>>>(
        static_cast<const unsigned char*>(input.data()),
        input.stride_bytes(),
        static_cast<unsigned char*>(output.data()),
        output.stride_bytes(),
        launch.width,
        launch.height);
    finish_launch(stream, "invert kernel launch");
}

void brightness_contrast(const ImageBuffer& input,
                         ImageBuffer& output,
                         const float brightness,
                         const float contrast,
                         const cudaStream_t stream) {
    if (!std::isfinite(brightness) || !std::isfinite(contrast) || contrast < 0.0F) {
        throw std::invalid_argument{
            "Brightness and contrast must be finite; contrast cannot be negative."};
    }
    const LaunchConfiguration launch = prepare_launch(input, output);
    brightness_contrast_kernel<<<launch.grid, launch.block, 0, stream>>>(
        static_cast<const unsigned char*>(input.data()),
        input.stride_bytes(),
        static_cast<unsigned char*>(output.data()),
        output.stride_bytes(),
        launch.width,
        launch.height,
        brightness,
        contrast);
    finish_launch(stream, "brightness/contrast kernel launch");
}

void gamma_correction(const ImageBuffer& input,
                      ImageBuffer& output,
                      const float gamma,
                      const cudaStream_t stream) {
    if (!std::isfinite(gamma) || gamma <= 0.0F) {
        throw std::invalid_argument{"Gamma must be finite and greater than zero."};
    }
    const LaunchConfiguration launch = prepare_launch(input, output);
    gamma_kernel<<<launch.grid, launch.block, 0, stream>>>(
        static_cast<const unsigned char*>(input.data()),
        input.stride_bytes(),
        static_cast<unsigned char*>(output.data()),
        output.stride_bytes(),
        launch.width,
        launch.height,
        1.0F / gamma);
    finish_launch(stream, "gamma kernel launch");
}

}  // namespace hzl::processing::cuda
