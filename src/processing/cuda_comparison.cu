#include "hzl/processing/cuda_comparison.hpp"

#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace hzl::processing::cuda {
namespace {

__global__ void difference_kernel(const unsigned char* original,
                                  const std::size_t original_pitch,
                                  const unsigned char* processed,
                                  const std::size_t processed_pitch,
                                  unsigned char* output,
                                  const std::size_t output_pitch,
                                  const unsigned int width,
                                  const unsigned int height) {
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    const unsigned char* source_original =
        original + static_cast<std::size_t>(y) * original_pitch + x * 4U;
    const unsigned char* source_processed =
        processed + static_cast<std::size_t>(y) * processed_pitch + x * 4U;
    unsigned char* destination =
        output + static_cast<std::size_t>(y) * output_pitch + x * 4U;
    for (unsigned int channel = 0; channel < 3; ++channel) {
        const int difference = static_cast<int>(source_original[channel]) -
                               static_cast<int>(source_processed[channel]);
        destination[channel] =
            static_cast<unsigned char>(difference < 0 ? -difference : difference);
    }
    destination[3] = 255U;
}

}  // namespace

void absolute_difference(const ImageBuffer& original,
                         const ImageBuffer& processed,
                         ImageBuffer& output,
                         const cudaStream_t stream) {
    if (&original == &output || &processed == &output) {
        throw std::invalid_argument{"CUDA difference output cannot alias an input."};
    }
    if (original.empty() || processed.empty() ||
        original.width() != processed.width() ||
        original.height() != processed.height()) {
        throw std::invalid_argument{
            "CUDA difference inputs must be non-empty and equally sized."};
    }
    if (original.width() > std::numeric_limits<unsigned int>::max() ||
        original.height() > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error{"CUDA difference dimensions exceed kernel limits."};
    }
    output.resize(original.width(), original.height());
    const auto width = static_cast<unsigned int>(original.width());
    const auto height = static_cast<unsigned int>(original.height());
    constexpr dim3 block{16, 16, 1};
    const dim3 grid{(width - 1U) / block.x + 1U,
                    (height - 1U) / block.y + 1U,
                    1U};
    difference_kernel<<<grid, block, 0, stream>>>(
        static_cast<const unsigned char*>(original.data()),
        original.stride_bytes(),
        static_cast<const unsigned char*>(processed.data()),
        processed.stride_bytes(),
        static_cast<unsigned char*>(output.data()),
        output.stride_bytes(),
        width,
        height);
    check(cudaGetLastError(), "absolute difference kernel launch");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
}

}  // namespace hzl::processing::cuda
