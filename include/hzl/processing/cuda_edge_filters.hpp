#pragma once

#include "hzl/processing/cuda_filter_common.hpp"
#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

namespace hzl::processing::cuda {

void sobel_edges(const ImageBuffer& input,
                 ImageBuffer& output,
                 float strength = 1.0F,
                 KernelImplementation implementation =
                     KernelImplementation::shared_memory,
                 cudaStream_t stream = nullptr);
void laplacian_edges(const ImageBuffer& input,
                     ImageBuffer& output,
                     float strength = 1.0F,
                     KernelImplementation implementation =
                         KernelImplementation::shared_memory,
                     cudaStream_t stream = nullptr);

}  // namespace hzl::processing::cuda
