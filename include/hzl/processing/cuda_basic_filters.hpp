#pragma once

#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

namespace hzl::processing::cuda {

// These reference launchers resize/reuse the output buffer, launch on the given
// stream, check launch errors, and synchronize before returning.
void grayscale(const ImageBuffer& input,
               ImageBuffer& output,
               cudaStream_t stream = nullptr);
void invert(const ImageBuffer& input,
            ImageBuffer& output,
            cudaStream_t stream = nullptr);
void brightness_contrast(const ImageBuffer& input,
                         ImageBuffer& output,
                         float brightness,
                         float contrast,
                         cudaStream_t stream = nullptr);
void gamma_correction(const ImageBuffer& input,
                      ImageBuffer& output,
                      float gamma,
                      cudaStream_t stream = nullptr);

}  // namespace hzl::processing::cuda
