#pragma once

#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

namespace hzl::processing::cuda {

void absolute_difference(const ImageBuffer& original,
                         const ImageBuffer& processed,
                         ImageBuffer& output,
                         cudaStream_t stream = nullptr);

}  // namespace hzl::processing::cuda
