#include "hzl/processing/cuda_color_filters.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace hzl::processing::cuda {
namespace {

constexpr unsigned int block_size = 256;

__device__ unsigned char rounded_byte(const float value) {
    return static_cast<unsigned char>(max(0, min(255, __float2int_rn(value))));
}

__global__ void luminance_histogram_kernel(const unsigned char* input,
                                           const std::size_t pitch,
                                           const unsigned int width,
                                           const unsigned int height,
                                           unsigned int* histogram) {
    __shared__ unsigned int local[256];
    local[threadIdx.x] = 0;
    __syncthreads();
    const unsigned int count = width * height;
    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += blockDim.x * gridDim.x) {
        const unsigned int x = index % width;
        const unsigned int y = index / width;
        const unsigned char* pixel = input + y * pitch + x * 4U;
        const unsigned int luminance =
            (77U * pixel[0] + 150U * pixel[1] + 29U * pixel[2] + 128U) >> 8U;
        atomicAdd(&local[luminance], 1U);
    }
    __syncthreads();
    atomicAdd(&histogram[threadIdx.x], local[threadIdx.x]);
}

__global__ void rgb_histogram_kernel(const unsigned char* input,
                                     const std::size_t pitch,
                                     const unsigned int width,
                                     const unsigned int height,
                                     unsigned int* histograms) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int count = width * height;
    if (index >= count) return;
    const unsigned int x = index % width;
    const unsigned int y = index / width;
    const unsigned char* pixel = input + y * pitch + x * 4U;
    atomicAdd(&histograms[pixel[0]], 1U);
    atomicAdd(&histograms[256U + pixel[1]], 1U);
    atomicAdd(&histograms[512U + pixel[2]], 1U);
}

__global__ void build_equalization_lookup(unsigned int* values,
                                          const unsigned int pixel_count) {
    const unsigned int channel = threadIdx.x;
    if (channel >= 3U) return;
    unsigned int cumulative = 0;
    unsigned int minimum = 0;
    bool found = false;
    unsigned int* lookup = values + channel * 256U;
    for (unsigned int value = 0; value < 256U; ++value) {
        cumulative += lookup[value];
        if (!found && cumulative != 0U) {
            minimum = cumulative;
            found = true;
        }
        lookup[value] = pixel_count == minimum
            ? value
            : static_cast<unsigned int>(__float2int_rn(
                  static_cast<float>(cumulative - minimum) * 255.0F /
                  static_cast<float>(pixel_count - minimum)));
    }
}

__global__ void apply_equalization_kernel(const unsigned char* input,
                                          const std::size_t input_pitch,
                                          unsigned char* output,
                                          const std::size_t output_pitch,
                                          const unsigned int width,
                                          const unsigned int height,
                                          const unsigned int* lookup) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    const unsigned int x = index % width;
    const unsigned int y = index / width;
    const unsigned char* source = input + y * input_pitch + x * 4U;
    unsigned char* destination = output + y * output_pitch + x * 4U;
    destination[0] = static_cast<unsigned char>(lookup[source[0]]);
    destination[1] = static_cast<unsigned char>(lookup[256U + source[1]]);
    destination[2] = static_cast<unsigned char>(lookup[512U + source[2]]);
    destination[3] = source[3];
}

__global__ void tone_map_kernel(const unsigned char* input,
                                const std::size_t input_pitch,
                                unsigned char* output,
                                const std::size_t output_pitch,
                                const unsigned int width,
                                const unsigned int height,
                                const float scale) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    const unsigned int x = index % width;
    const unsigned int y = index / width;
    const unsigned char* source = input + y * input_pitch + x * 4U;
    unsigned char* destination = output + y * output_pitch + x * 4U;
    for (unsigned int channel = 0; channel < 3U; ++channel) {
        const float linear = static_cast<float>(source[channel]) / 255.0F * scale;
        destination[channel] = rounded_byte(linear / (1.0F + linear) * 255.0F);
    }
    destination[3] = source[3];
}

__global__ void color_grade_kernel(const unsigned char* input,
                                   const std::size_t input_pitch,
                                   unsigned char* output,
                                   const std::size_t output_pitch,
                                   const unsigned int width,
                                   const unsigned int height,
                                   const float saturation,
                                   const float temperature,
                                   const float tint,
                                   const float red_gain,
                                   const float green_gain,
                                   const float blue_gain) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= width * height) return;
    const unsigned int x = index % width;
    const unsigned int y = index / width;
    const unsigned char* source = input + y * input_pitch + x * 4U;
    unsigned char* destination = output + y * output_pitch + x * 4U;
    const float luminance = 0.299F * source[0] + 0.587F * source[1] +
                            0.114F * source[2];
    const float shifts[3]{temperature * 32.0F, tint * 16.0F,
                          -temperature * 32.0F};
    const float gains[3]{red_gain, green_gain, blue_gain};
    for (unsigned int channel = 0; channel < 3U; ++channel) {
        destination[channel] = rounded_byte(
            (luminance + (source[channel] - luminance) * saturation +
             shifts[channel]) * gains[channel]);
    }
    destination[3] = source[3];
}

struct Dimensions { unsigned int width; unsigned int height; unsigned int count; };

Dimensions validate(const ImageBuffer& input) {
    if (input.empty() || input.data() == nullptr) {
        throw std::invalid_argument{"CUDA color filter input cannot be empty."};
    }
    const std::size_t count = input.width() * input.height();
    if (input.width() > std::numeric_limits<unsigned int>::max() ||
        input.height() > std::numeric_limits<unsigned int>::max() ||
        count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error{"CUDA color filter dimensions exceed limits."};
    }
    return {static_cast<unsigned int>(input.width()),
            static_cast<unsigned int>(input.height()),
            static_cast<unsigned int>(count)};
}

unsigned int blocks_for(const unsigned int count) {
    return (count + block_size - 1U) / block_size;
}

void finish(const cudaStream_t stream, const char* operation) {
    check(cudaGetLastError(), operation);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

}  // namespace

Histogram luminance_histogram(const ImageBuffer& input, const cudaStream_t stream) {
    const Dimensions dimensions = validate(input);
    unsigned int* device_histogram = nullptr;
    check(cudaMalloc(&device_histogram, 256U * sizeof(unsigned int)), "cudaMalloc histogram");
    try {
        check(cudaMemsetAsync(device_histogram, 0, 256U * sizeof(unsigned int), stream),
              "cudaMemsetAsync histogram");
        luminance_histogram_kernel<<<blocks_for(dimensions.count), block_size, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()), input.stride_bytes(),
            dimensions.width, dimensions.height, device_histogram);
        check(cudaGetLastError(), "luminance histogram kernel launch");
        Histogram result{};
        check(cudaMemcpyAsync(result.data(), device_histogram,
                              result.size() * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync histogram");
        check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        check(cudaFree(device_histogram), "cudaFree histogram");
        return result;
    } catch (...) {
        static_cast<void>(cudaFree(device_histogram));
        throw;
    }
}

void histogram_equalization(const ImageBuffer& input, ImageBuffer& output,
                            const cudaStream_t stream) {
    const Dimensions dimensions = validate(input);
    output.resize(input.width(), input.height());
    unsigned int* lookup = nullptr;
    check(cudaMalloc(&lookup, 3U * 256U * sizeof(unsigned int)), "cudaMalloc equalization");
    try {
        check(cudaMemsetAsync(lookup, 0, 3U * 256U * sizeof(unsigned int), stream),
              "cudaMemsetAsync equalization");
        rgb_histogram_kernel<<<blocks_for(dimensions.count), block_size, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()), input.stride_bytes(),
            dimensions.width, dimensions.height, lookup);
        build_equalization_lookup<<<1, 3, 0, stream>>>(lookup, dimensions.count);
        apply_equalization_kernel<<<blocks_for(dimensions.count), block_size, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()), input.stride_bytes(),
            static_cast<unsigned char*>(output.data()), output.stride_bytes(),
            dimensions.width, dimensions.height, lookup);
        finish(stream, "histogram equalization kernel launch");
        check(cudaFree(lookup), "cudaFree equalization");
    } catch (...) {
        static_cast<void>(cudaFree(lookup));
        throw;
    }
}

void tone_map(const ImageBuffer& input, ImageBuffer& output, const float exposure,
              const cudaStream_t stream) {
    if (!std::isfinite(exposure)) throw std::invalid_argument{"Exposure must be finite."};
    const Dimensions dimensions = validate(input);
    output.resize(input.width(), input.height());
    tone_map_kernel<<<blocks_for(dimensions.count), block_size, 0, stream>>>(
        static_cast<const unsigned char*>(input.data()), input.stride_bytes(),
        static_cast<unsigned char*>(output.data()), output.stride_bytes(),
        dimensions.width, dimensions.height, exp2f(exposure));
    finish(stream, "tone mapping kernel launch");
}

void color_grade(const ImageBuffer& input, ImageBuffer& output,
                 const float saturation, const float temperature, const float tint,
                 const float red_gain, const float green_gain, const float blue_gain,
                 const cudaStream_t stream) {
    if (!std::isfinite(saturation) || !std::isfinite(temperature) ||
        !std::isfinite(tint) || !std::isfinite(red_gain) ||
        !std::isfinite(green_gain) || !std::isfinite(blue_gain) ||
        saturation < 0.0F || red_gain < 0.0F || green_gain < 0.0F || blue_gain < 0.0F) {
        throw std::invalid_argument{"Color grading parameters are invalid."};
    }
    const Dimensions dimensions = validate(input);
    output.resize(input.width(), input.height());
    color_grade_kernel<<<blocks_for(dimensions.count), block_size, 0, stream>>>(
        static_cast<const unsigned char*>(input.data()), input.stride_bytes(),
        static_cast<unsigned char*>(output.data()), output.stride_bytes(),
        dimensions.width, dimensions.height, saturation, temperature, tint,
        red_gain, green_gain, blue_gain);
    finish(stream, "color grading kernel launch");
}

}  // namespace hzl::processing::cuda
