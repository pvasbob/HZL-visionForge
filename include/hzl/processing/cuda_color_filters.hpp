#pragma once

#include "hzl/processing/cuda_memory.hpp"

#include <array>
#include <cstdint>

namespace hzl::processing::cuda {

using Histogram = std::array<std::uint32_t, 256>;

[[nodiscard]] Histogram luminance_histogram(
    const ImageBuffer& input,
    cudaStream_t stream = nullptr);
void histogram_equalization(const ImageBuffer& input,
                            ImageBuffer& output,
                            cudaStream_t stream = nullptr);
void tone_map(const ImageBuffer& input,
              ImageBuffer& output,
              float exposure,
              cudaStream_t stream = nullptr);
void color_grade(const ImageBuffer& input,
                 ImageBuffer& output,
                 float saturation,
                 float temperature,
                 float tint,
                 float red_gain,
                 float green_gain,
                 float blue_gain,
                 cudaStream_t stream = nullptr);

}  // namespace hzl::processing::cuda
