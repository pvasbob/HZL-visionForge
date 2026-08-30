#include "hzl/processing/cuda_convolution_filters.hpp"

#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace hzl::processing::cuda {
namespace {

constexpr unsigned int block_width = 16;
constexpr unsigned int block_height = 16;
constexpr int maximum_kernel_size = 31;
constexpr int maximum_weight_count =
    maximum_kernel_size * maximum_kernel_size;

__constant__ float gaussian_weights[maximum_weight_count];

__device__ int reflect_101(int coordinate, const int length) {
    if (length <= 1) {
        return 0;
    }
    while (coordinate < 0 || coordinate >= length) {
        coordinate = coordinate < 0 ? -coordinate
                                    : 2 * length - coordinate - 2;
    }
    return coordinate;
}

__device__ unsigned char saturated_round(const float value) {
    return static_cast<unsigned char>(max(0, min(255, __float2int_rn(value))));
}

__device__ const unsigned char* pixel_at(const unsigned char* image,
                                         const std::size_t pitch,
                                         const int x,
                                         const int y,
                                         const int width,
                                         const int height) {
    const int reflected_x = reflect_101(x, width);
    const int reflected_y = reflect_101(y, height);
    return image + static_cast<std::size_t>(reflected_y) * pitch +
           static_cast<std::size_t>(reflected_x) * 4U;
}

__device__ void load_shared_tile(const unsigned char* input,
                                 const std::size_t input_pitch,
                                 unsigned char* tile,
                                 const int tile_width,
                                 const int tile_height,
                                 const int radius,
                                 const int image_width,
                                 const int image_height) {
    const int thread_index = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int thread_count = static_cast<int>(blockDim.x * blockDim.y);
    const int tile_pixel_count = tile_width * tile_height;
    for (int index = thread_index; index < tile_pixel_count; index += thread_count) {
        const int tile_x = index % tile_width;
        const int tile_y = index / tile_width;
        const int image_x = static_cast<int>(blockIdx.x * blockDim.x) + tile_x - radius;
        const int image_y = static_cast<int>(blockIdx.y * blockDim.y) + tile_y - radius;
        const unsigned char* source = pixel_at(input,
                                               input_pitch,
                                               image_x,
                                               image_y,
                                               image_width,
                                               image_height);
        unsigned char* destination = tile + static_cast<std::size_t>(index) * 4U;
        destination[0] = source[0];
        destination[1] = source[1];
        destination[2] = source[2];
        destination[3] = source[3];
    }
}

__global__ void box_blur_global_kernel(const unsigned char* input,
                                       const std::size_t input_pitch,
                                       unsigned char* output,
                                       const std::size_t output_pitch,
                                       const int width,
                                       const int height,
                                       const int radius) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }

    int sums[3]{0, 0, 0};
    for (int offset_y = -radius; offset_y <= radius; ++offset_y) {
        for (int offset_x = -radius; offset_x <= radius; ++offset_x) {
            const unsigned char* pixel = pixel_at(input,
                                                  input_pitch,
                                                  x + offset_x,
                                                  y + offset_y,
                                                  width,
                                                  height);
            sums[0] += pixel[0];
            sums[1] += pixel[1];
            sums[2] += pixel[2];
        }
    }
    const int diameter = radius * 2 + 1;
    const float inverse_count = 1.0F / static_cast<float>(diameter * diameter);
    unsigned char* destination = output + static_cast<std::size_t>(y) * output_pitch +
                                 static_cast<std::size_t>(x) * 4U;
    destination[0] = saturated_round(sums[0] * inverse_count);
    destination[1] = saturated_round(sums[1] * inverse_count);
    destination[2] = saturated_round(sums[2] * inverse_count);
    destination[3] = input[static_cast<std::size_t>(y) * input_pitch +
                           static_cast<std::size_t>(x) * 4U + 3U];
}

__global__ void box_blur_shared_kernel(const unsigned char* input,
                                       const std::size_t input_pitch,
                                       unsigned char* output,
                                       const std::size_t output_pitch,
                                       const int width,
                                       const int height,
                                       const int radius) {
    extern __shared__ unsigned char tile[];
    const int tile_width = static_cast<int>(blockDim.x) + radius * 2;
    const int tile_height = static_cast<int>(blockDim.y) + radius * 2;
    load_shared_tile(input,
                     input_pitch,
                     tile,
                     tile_width,
                     tile_height,
                     radius,
                     width,
                     height);
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    int sums[3]{0, 0, 0};
    for (int offset_y = 0; offset_y <= radius * 2; ++offset_y) {
        for (int offset_x = 0; offset_x <= radius * 2; ++offset_x) {
            const unsigned char* pixel =
                tile + (static_cast<std::size_t>(threadIdx.y + offset_y) *
                            static_cast<std::size_t>(tile_width) +
                        threadIdx.x + offset_x) *
                           4U;
            sums[0] += pixel[0];
            sums[1] += pixel[1];
            sums[2] += pixel[2];
        }
    }
    const int diameter = radius * 2 + 1;
    const float inverse_count = 1.0F / static_cast<float>(diameter * diameter);
    unsigned char* destination = output + static_cast<std::size_t>(y) * output_pitch +
                                 static_cast<std::size_t>(x) * 4U;
    destination[0] = saturated_round(sums[0] * inverse_count);
    destination[1] = saturated_round(sums[1] * inverse_count);
    destination[2] = saturated_round(sums[2] * inverse_count);
    destination[3] = tile[(static_cast<std::size_t>(threadIdx.y + radius) *
                               static_cast<std::size_t>(tile_width) +
                           threadIdx.x + radius) *
                              4U +
                          3U];
}

template <bool Sharpen>
__global__ void gaussian_global_kernel(const unsigned char* input,
                                       const std::size_t input_pitch,
                                       unsigned char* output,
                                       const std::size_t output_pitch,
                                       const int width,
                                       const int height,
                                       const int radius,
                                       const float amount) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    float sums[3]{0.0F, 0.0F, 0.0F};
    const int diameter = radius * 2 + 1;
    for (int offset_y = -radius; offset_y <= radius; ++offset_y) {
        for (int offset_x = -radius; offset_x <= radius; ++offset_x) {
            const unsigned char* pixel = pixel_at(input,
                                                  input_pitch,
                                                  x + offset_x,
                                                  y + offset_y,
                                                  width,
                                                  height);
            const int weight_index =
                (offset_y + radius) * diameter + offset_x + radius;
            const float weight = gaussian_weights[weight_index];
            sums[0] += pixel[0] * weight;
            sums[1] += pixel[1] * weight;
            sums[2] += pixel[2] * weight;
        }
    }
    const unsigned char* center = input + static_cast<std::size_t>(y) * input_pitch +
                                  static_cast<std::size_t>(x) * 4U;
    unsigned char* destination = output + static_cast<std::size_t>(y) * output_pitch +
                                 static_cast<std::size_t>(x) * 4U;
    for (int channel = 0; channel < 3; ++channel) {
        const float value = Sharpen
                                ? center[channel] * (1.0F + amount) -
                                      sums[channel] * amount
                                : sums[channel];
        destination[channel] = saturated_round(value);
    }
    destination[3] = center[3];
}

template <bool Sharpen>
__global__ void gaussian_shared_kernel(const unsigned char* input,
                                       const std::size_t input_pitch,
                                       unsigned char* output,
                                       const std::size_t output_pitch,
                                       const int width,
                                       const int height,
                                       const int radius,
                                       const float amount) {
    extern __shared__ unsigned char tile[];
    const int tile_width = static_cast<int>(blockDim.x) + radius * 2;
    const int tile_height = static_cast<int>(blockDim.y) + radius * 2;
    load_shared_tile(input,
                     input_pitch,
                     tile,
                     tile_width,
                     tile_height,
                     radius,
                     width,
                     height);
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    float sums[3]{0.0F, 0.0F, 0.0F};
    const int diameter = radius * 2 + 1;
    for (int offset_y = 0; offset_y < diameter; ++offset_y) {
        for (int offset_x = 0; offset_x < diameter; ++offset_x) {
            const unsigned char* pixel =
                tile + (static_cast<std::size_t>(threadIdx.y + offset_y) *
                            static_cast<std::size_t>(tile_width) +
                        threadIdx.x + offset_x) *
                           4U;
            const float weight = gaussian_weights[offset_y * diameter + offset_x];
            sums[0] += pixel[0] * weight;
            sums[1] += pixel[1] * weight;
            sums[2] += pixel[2] * weight;
        }
    }
    const unsigned char* center =
        tile + (static_cast<std::size_t>(threadIdx.y + radius) *
                    static_cast<std::size_t>(tile_width) +
                threadIdx.x + radius) *
                   4U;
    unsigned char* destination = output + static_cast<std::size_t>(y) * output_pitch +
                                 static_cast<std::size_t>(x) * 4U;
    for (int channel = 0; channel < 3; ++channel) {
        const float value = Sharpen
                                ? center[channel] * (1.0F + amount) -
                                      sums[channel] * amount
                                : sums[channel];
        destination[channel] = saturated_round(value);
    }
    destination[3] = center[3];
}

__device__ constexpr float emboss_kernel[9]{-2.0F, -1.0F, 0.0F,
                                            -1.0F,  0.0F, 1.0F,
                                             0.0F,  1.0F, 2.0F};

__global__ void emboss_global_kernel(const unsigned char* input,
                                     const std::size_t input_pitch,
                                     unsigned char* output,
                                     const std::size_t output_pitch,
                                     const int width,
                                     const int height,
                                     const float strength) {
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    float sums[3]{0.0F, 0.0F, 0.0F};
    for (int offset_y = -1; offset_y <= 1; ++offset_y) {
        for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            const unsigned char* pixel = pixel_at(input,
                                                  input_pitch,
                                                  x + offset_x,
                                                  y + offset_y,
                                                  width,
                                                  height);
            const float weight = emboss_kernel[(offset_y + 1) * 3 + offset_x + 1];
            sums[0] += pixel[0] * weight;
            sums[1] += pixel[1] * weight;
            sums[2] += pixel[2] * weight;
        }
    }
    const unsigned char* center = input + static_cast<std::size_t>(y) * input_pitch +
                                  static_cast<std::size_t>(x) * 4U;
    unsigned char* destination = output + static_cast<std::size_t>(y) * output_pitch +
                                 static_cast<std::size_t>(x) * 4U;
    destination[0] = saturated_round(sums[0] * strength + 128.0F);
    destination[1] = saturated_round(sums[1] * strength + 128.0F);
    destination[2] = saturated_round(sums[2] * strength + 128.0F);
    destination[3] = center[3];
}

__global__ void emboss_shared_kernel(const unsigned char* input,
                                     const std::size_t input_pitch,
                                     unsigned char* output,
                                     const std::size_t output_pitch,
                                     const int width,
                                     const int height,
                                     const float strength) {
    extern __shared__ unsigned char tile[];
    constexpr int radius = 1;
    const int tile_width = static_cast<int>(blockDim.x) + 2;
    const int tile_height = static_cast<int>(blockDim.y) + 2;
    load_shared_tile(input,
                     input_pitch,
                     tile,
                     tile_width,
                     tile_height,
                     radius,
                     width,
                     height);
    __syncthreads();
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    float sums[3]{0.0F, 0.0F, 0.0F};
    for (int offset_y = 0; offset_y < 3; ++offset_y) {
        for (int offset_x = 0; offset_x < 3; ++offset_x) {
            const unsigned char* pixel =
                tile + (static_cast<std::size_t>(threadIdx.y + offset_y) *
                            static_cast<std::size_t>(tile_width) +
                        threadIdx.x + offset_x) *
                           4U;
            const float weight = emboss_kernel[offset_y * 3 + offset_x];
            sums[0] += pixel[0] * weight;
            sums[1] += pixel[1] * weight;
            sums[2] += pixel[2] * weight;
        }
    }
    const unsigned char* center =
        tile + (static_cast<std::size_t>(threadIdx.y + 1) *
                    static_cast<std::size_t>(tile_width) +
                threadIdx.x + 1) *
                   4U;
    unsigned char* destination = output + static_cast<std::size_t>(y) * output_pitch +
                                 static_cast<std::size_t>(x) * 4U;
    destination[0] = saturated_round(sums[0] * strength + 128.0F);
    destination[1] = saturated_round(sums[1] * strength + 128.0F);
    destination[2] = saturated_round(sums[2] * strength + 128.0F);
    destination[3] = center[3];
}

struct LaunchConfiguration {
    dim3 grid;
    dim3 block;
    int width;
    int height;
};

void validate_kernel_size(const int kernel_size) {
    if (kernel_size < 1 || kernel_size > maximum_kernel_size ||
        kernel_size % 2 == 0) {
        throw std::invalid_argument{
            "CUDA convolution kernel size must be odd and between 1 and 31."};
    }
}

LaunchConfiguration prepare_launch(const ImageBuffer& input,
                                   ImageBuffer& output) {
    if (&input == &output) {
        throw std::invalid_argument{"CUDA convolution filters cannot run in place."};
    }
    if (input.empty() || input.data() == nullptr) {
        throw std::invalid_argument{"CUDA convolution input cannot be empty."};
    }
    if (input.width() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        input.height() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error{"CUDA convolution dimensions exceed kernel limits."};
    }
    output.resize(input.width(), input.height());
    const int width = static_cast<int>(input.width());
    const int height = static_cast<int>(input.height());
    return {dim3{(static_cast<unsigned int>(width) - 1U) / block_width + 1U,
                 (static_cast<unsigned int>(height) - 1U) / block_height + 1U,
                 1U},
            dim3{block_width, block_height, 1U},
            width,
            height};
}

std::size_t shared_memory_bytes(const int radius) {
    return static_cast<std::size_t>(block_width + 2U * radius) *
           static_cast<std::size_t>(block_height + 2U * radius) * 4U;
}

void finish_launch(const cudaStream_t stream, const char* operation) {
    check(cudaGetLastError(), operation);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

void upload_gaussian_weights(const int kernel_size,
                             const float sigma,
                             const cudaStream_t stream) {
    const int radius = kernel_size / 2;
    const float effective_sigma = sigma == 0.0F
                                      ? 0.3F * ((kernel_size - 1) * 0.5F - 1.0F) +
                                            0.8F
                                      : sigma;
    std::vector<float> weights(static_cast<std::size_t>(kernel_size * kernel_size));
    float sum = 0.0F;
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            const float exponent =
                -static_cast<float>(x * x + y * y) /
                (2.0F * effective_sigma * effective_sigma);
            const float weight = std::exp(exponent);
            weights[static_cast<std::size_t>((y + radius) * kernel_size +
                                             x + radius)] = weight;
            sum += weight;
        }
    }
    for (float& weight : weights) {
        weight /= sum;
    }
    check(cudaMemcpyToSymbolAsync(gaussian_weights,
                                  weights.data(),
                                  weights.size() * sizeof(float),
                                  0,
                                  cudaMemcpyHostToDevice,
                                  stream),
          "cudaMemcpyToSymbolAsync Gaussian weights");
}

}  // namespace


void box_blur(const ImageBuffer& input,
              ImageBuffer& output,
              const int kernel_size,
              const KernelImplementation implementation,
              const cudaStream_t stream) {
    validate_kernel_size(kernel_size);
    const int radius = kernel_size / 2;
    const LaunchConfiguration launch = prepare_launch(input, output);
    if (implementation == KernelImplementation::shared_memory) {
        box_blur_shared_kernel<<<launch.grid,
                                 launch.block,
                                 shared_memory_bytes(radius),
                                 stream>>>(static_cast<const unsigned char*>(input.data()),
                                           input.stride_bytes(),
                                           static_cast<unsigned char*>(output.data()),
                                           output.stride_bytes(),
                                           launch.width,
                                           launch.height,
                                           radius);
    } else {
        box_blur_global_kernel<<<launch.grid, launch.block, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            radius);
    }
    finish_launch(stream, "box blur kernel launch");
}

void gaussian_blur(const ImageBuffer& input,
                   ImageBuffer& output,
                   const int kernel_size,
                   const float sigma,
                   const KernelImplementation implementation,
                   const cudaStream_t stream) {
    validate_kernel_size(kernel_size);
    if (!std::isfinite(sigma) || sigma < 0.0F) {
        throw std::invalid_argument{"Gaussian sigma must be finite and non-negative."};
    }
    upload_gaussian_weights(kernel_size, sigma, stream);
    const int radius = kernel_size / 2;
    const LaunchConfiguration launch = prepare_launch(input, output);
    if (implementation == KernelImplementation::shared_memory) {
        gaussian_shared_kernel<false><<<launch.grid,
                                        launch.block,
                                        shared_memory_bytes(radius),
                                        stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            radius,
            0.0F);
    } else {
        gaussian_global_kernel<false><<<launch.grid, launch.block, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            radius,
            0.0F);
    }
    finish_launch(stream, "Gaussian blur kernel launch");
}

void sharpen(const ImageBuffer& input,
             ImageBuffer& output,
             const float amount,
             const KernelImplementation implementation,
             const cudaStream_t stream) {
    if (!std::isfinite(amount) || amount < 0.0F) {
        throw std::invalid_argument{"Sharpen amount must be finite and non-negative."};
    }
    constexpr int kernel_size = 7;
    constexpr float sigma = 1.0F;
    upload_gaussian_weights(kernel_size, sigma, stream);
    constexpr int radius = kernel_size / 2;
    const LaunchConfiguration launch = prepare_launch(input, output);
    if (implementation == KernelImplementation::shared_memory) {
        gaussian_shared_kernel<true><<<launch.grid,
                                       launch.block,
                                       shared_memory_bytes(radius),
                                       stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            radius,
            amount);
    } else {
        gaussian_global_kernel<true><<<launch.grid, launch.block, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            radius,
            amount);
    }
    finish_launch(stream, "sharpen kernel launch");
}

void emboss(const ImageBuffer& input,
            ImageBuffer& output,
            const float strength,
            const KernelImplementation implementation,
            const cudaStream_t stream) {
    if (!std::isfinite(strength) || strength < 0.0F) {
        throw std::invalid_argument{"Emboss strength must be finite and non-negative."};
    }
    const LaunchConfiguration launch = prepare_launch(input, output);
    if (implementation == KernelImplementation::shared_memory) {
        emboss_shared_kernel<<<launch.grid,
                               launch.block,
                               shared_memory_bytes(1),
                               stream>>>(static_cast<const unsigned char*>(input.data()),
                                         input.stride_bytes(),
                                         static_cast<unsigned char*>(output.data()),
                                         output.stride_bytes(),
                                         launch.width,
                                         launch.height,
                                         strength);
    } else {
        emboss_global_kernel<<<launch.grid, launch.block, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            strength);
    }
    finish_launch(stream, "emboss kernel launch");
}

}  // namespace hzl::processing::cuda
