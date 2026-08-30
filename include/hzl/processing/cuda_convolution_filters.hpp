#pragma once

#include "hzl/processing/cuda_filter_common.hpp"
#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

namespace hzl::processing::cuda {

void box_blur(const ImageBuffer& input,
              ImageBuffer& output,
              int kernel_size,
              KernelImplementation implementation =
                  KernelImplementation::shared_memory,
              cudaStream_t stream = nullptr);
void gaussian_blur(const ImageBuffer& input,
                   ImageBuffer& output,
                   int kernel_size,
                   float sigma,
                   KernelImplementation implementation =
                       KernelImplementation::shared_memory,
                   cudaStream_t stream = nullptr);
void sharpen(const ImageBuffer& input,
             ImageBuffer& output,
             float amount,
             KernelImplementation implementation =
                 KernelImplementation::shared_memory,
             cudaStream_t stream = nullptr);
void emboss(const ImageBuffer& input,
            ImageBuffer& output,
            float strength,
            KernelImplementation implementation =
                KernelImplementation::shared_memory,
            cudaStream_t stream = nullptr);

}  // namespace hzl::processing::cuda
