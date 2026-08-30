#include "hzl/processing/cuda_edge_filters.hpp"

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
constexpr int radius = 1;

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

__device__ unsigned char rgba_to_gray(const unsigned char* pixel) {
    return saturated_round(0.299F * pixel[0] + 0.587F * pixel[1] +
                           0.114F * pixel[2]);
}

__device__ unsigned char gray_at(const unsigned char* input,
                                 const std::size_t pitch,
                                 int x,
                                 int y,
                                 const int width,
                                 const int height) {
    x = reflect_101(x, width);
    y = reflect_101(y, height);
    return rgba_to_gray(input + static_cast<std::size_t>(y) * pitch +
                        static_cast<std::size_t>(x) * 4U);
}

__device__ void store_edge(unsigned char* output,
                           const std::size_t output_pitch,
                           const unsigned char* input,
                           const std::size_t input_pitch,
                           const int x,
                           const int y,
                           const unsigned char edge) {
    unsigned char* destination = output + static_cast<std::size_t>(y) * output_pitch +
                                 static_cast<std::size_t>(x) * 4U;
    destination[0] = edge;
    destination[1] = edge;
    destination[2] = edge;
    destination[3] = input[static_cast<std::size_t>(y) * input_pitch +
                           static_cast<std::size_t>(x) * 4U + 3U];
}

__device__ void load_gray_tile(const unsigned char* input,
                               const std::size_t input_pitch,
                               unsigned char* tile,
                               const int tile_width,
                               const int tile_height,
                               const int width,
                               const int height) {
    const int thread_index = static_cast<int>(threadIdx.y * blockDim.x + threadIdx.x);
    const int thread_count = static_cast<int>(blockDim.x * blockDim.y);
    for (int index = thread_index; index < tile_width * tile_height;
         index += thread_count) {
        const int tile_x = index % tile_width;
        const int tile_y = index / tile_width;
        const int image_x = static_cast<int>(blockIdx.x * blockDim.x) + tile_x - 1;
        const int image_y = static_cast<int>(blockIdx.y * blockDim.y) + tile_y - 1;
        tile[index] = gray_at(input,
                              input_pitch,
                              image_x,
                              image_y,
                              width,
                              height);
    }
}

__device__ float sobel_magnitude(const unsigned char top_left,
                                 const unsigned char top,
                                 const unsigned char top_right,
                                 const unsigned char left,
                                 const unsigned char right,
                                 const unsigned char bottom_left,
                                 const unsigned char bottom,
                                 const unsigned char bottom_right) {
    const float horizontal = -top_left - 2.0F * left - bottom_left +
                             top_right + 2.0F * right + bottom_right;
    const float vertical = -top_left - 2.0F * top - top_right +
                           bottom_left + 2.0F * bottom + bottom_right;
    return hypotf(horizontal, vertical);
}

__global__ void sobel_global_kernel(const unsigned char* input,
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
    const float magnitude = sobel_magnitude(
        gray_at(input, input_pitch, x - 1, y - 1, width, height),
        gray_at(input, input_pitch, x, y - 1, width, height),
        gray_at(input, input_pitch, x + 1, y - 1, width, height),
        gray_at(input, input_pitch, x - 1, y, width, height),
        gray_at(input, input_pitch, x + 1, y, width, height),
        gray_at(input, input_pitch, x - 1, y + 1, width, height),
        gray_at(input, input_pitch, x, y + 1, width, height),
        gray_at(input, input_pitch, x + 1, y + 1, width, height));
    store_edge(output,
               output_pitch,
               input,
               input_pitch,
               x,
               y,
               saturated_round(magnitude * strength));
}

__global__ void sobel_shared_kernel(const unsigned char* input,
                                    const std::size_t input_pitch,
                                    unsigned char* output,
                                    const std::size_t output_pitch,
                                    const int width,
                                    const int height,
                                    const float strength) {
    extern __shared__ unsigned char tile[];
    const int tile_width = static_cast<int>(blockDim.x) + 2;
    const int tile_height = static_cast<int>(blockDim.y) + 2;
    load_gray_tile(input,
                   input_pitch,
                   tile,
                   tile_width,
                   tile_height,
                   width,
                   height);
    __syncthreads();
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int center = (static_cast<int>(threadIdx.y) + 1) * tile_width +
                       static_cast<int>(threadIdx.x) + 1;
    const float magnitude = sobel_magnitude(tile[center - tile_width - 1],
                                            tile[center - tile_width],
                                            tile[center - tile_width + 1],
                                            tile[center - 1],
                                            tile[center + 1],
                                            tile[center + tile_width - 1],
                                            tile[center + tile_width],
                                            tile[center + tile_width + 1]);
    store_edge(output,
               output_pitch,
               input,
               input_pitch,
               x,
               y,
               saturated_round(magnitude * strength));
}

__global__ void laplacian_global_kernel(const unsigned char* input,
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
    const int value = gray_at(input, input_pitch, x, y - 1, width, height) +
                      gray_at(input, input_pitch, x - 1, y, width, height) -
                      4 * gray_at(input, input_pitch, x, y, width, height) +
                      gray_at(input, input_pitch, x + 1, y, width, height) +
                      gray_at(input, input_pitch, x, y + 1, width, height);
    store_edge(output,
               output_pitch,
               input,
               input_pitch,
               x,
               y,
               saturated_round(fabsf(static_cast<float>(value)) * strength));
}

__global__ void laplacian_shared_kernel(const unsigned char* input,
                                        const std::size_t input_pitch,
                                        unsigned char* output,
                                        const std::size_t output_pitch,
                                        const int width,
                                        const int height,
                                        const float strength) {
    extern __shared__ unsigned char tile[];
    const int tile_width = static_cast<int>(blockDim.x) + 2;
    const int tile_height = static_cast<int>(blockDim.y) + 2;
    load_gray_tile(input,
                   input_pitch,
                   tile,
                   tile_width,
                   tile_height,
                   width,
                   height);
    __syncthreads();
    const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
    if (x >= width || y >= height) {
        return;
    }
    const int center = (static_cast<int>(threadIdx.y) + 1) * tile_width +
                       static_cast<int>(threadIdx.x) + 1;
    const int value = tile[center - tile_width] + tile[center - 1] -
                      4 * tile[center] + tile[center + 1] +
                      tile[center + tile_width];
    store_edge(output,
               output_pitch,
               input,
               input_pitch,
               x,
               y,
               saturated_round(fabsf(static_cast<float>(value)) * strength));
}

struct LaunchConfiguration {
    dim3 grid;
    dim3 block;
    int width;
    int height;
};

LaunchConfiguration prepare_launch(const ImageBuffer& input,
                                   ImageBuffer& output) {
    if (&input == &output) {
        throw std::invalid_argument{"CUDA edge filters cannot run in place."};
    }
    if (input.empty() || input.data() == nullptr) {
        throw std::invalid_argument{"CUDA edge input cannot be empty."};
    }
    if (input.width() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        input.height() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error{"CUDA edge dimensions exceed kernel limits."};
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

void validate_strength(const float strength) {
    if (!std::isfinite(strength) || strength < 0.0F) {
        throw std::invalid_argument{"Edge strength must be finite and non-negative."};
    }
}

void finish_launch(const cudaStream_t stream, const char* operation) {
    check(cudaGetLastError(), operation);
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

constexpr std::size_t shared_memory_bytes =
    (block_width + radius * 2U) * (block_height + radius * 2U);

}  // namespace

void sobel_edges(const ImageBuffer& input,
                 ImageBuffer& output,
                 const float strength,
                 const KernelImplementation implementation,
                 const cudaStream_t stream) {
    validate_strength(strength);
    const LaunchConfiguration launch = prepare_launch(input, output);
    if (implementation == KernelImplementation::shared_memory) {
        sobel_shared_kernel<<<launch.grid,
                              launch.block,
                              shared_memory_bytes,
                              stream>>>(static_cast<const unsigned char*>(input.data()),
                                        input.stride_bytes(),
                                        static_cast<unsigned char*>(output.data()),
                                        output.stride_bytes(),
                                        launch.width,
                                        launch.height,
                                        strength);
    } else {
        sobel_global_kernel<<<launch.grid, launch.block, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            strength);
    }
    finish_launch(stream, "Sobel kernel launch");
}

void laplacian_edges(const ImageBuffer& input,
                     ImageBuffer& output,
                     const float strength,
                     const KernelImplementation implementation,
                     const cudaStream_t stream) {
    validate_strength(strength);
    const LaunchConfiguration launch = prepare_launch(input, output);
    if (implementation == KernelImplementation::shared_memory) {
        laplacian_shared_kernel<<<launch.grid,
                                  launch.block,
                                  shared_memory_bytes,
                                  stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            strength);
    } else {
        laplacian_global_kernel<<<launch.grid, launch.block, 0, stream>>>(
            static_cast<const unsigned char*>(input.data()),
            input.stride_bytes(),
            static_cast<unsigned char*>(output.data()),
            output.stride_bytes(),
            launch.width,
            launch.height,
            strength);
    }
    finish_launch(stream, "Laplacian kernel launch");
}

}  // namespace hzl::processing::cuda
