#pragma once

#include "hzl/processing/cuda_memory.hpp"

#include <cstddef>
#include <cstdint>
#include <variant>
#include <vector>

namespace hzl::processing {

enum class FilterType {
    grayscale,
    invert,
    brightness_contrast,
    gamma,
    box_blur,
    gaussian_blur,
    sharpen,
    emboss,
    sobel,
    laplacian,
};

struct BrightnessContrastParameters {
    float brightness{0.0F};
    float contrast{1.0F};
};
struct GammaParameters { float gamma{1.0F}; };
struct BoxBlurParameters { int kernel_size{3}; };
struct GaussianBlurParameters { int kernel_size{5}; float sigma{1.2F}; };
struct SharpenParameters { float amount{1.0F}; };
struct EmbossParameters { float strength{1.0F}; };
struct EdgeParameters { float strength{1.0F}; };

using FilterParameters = std::variant<std::monostate,
                                      BrightnessContrastParameters,
                                      GammaParameters,
                                      BoxBlurParameters,
                                      GaussianBlurParameters,
                                      SharpenParameters,
                                      EmbossParameters,
                                      EdgeParameters>;

struct PipelineOperation {
    std::uint64_t id{0};
    FilterType type{FilterType::grayscale};
    bool enabled{true};
    FilterParameters parameters;
};

[[nodiscard]] const char* filter_name(FilterType type) noexcept;

class GpuPipeline {
public:
    [[nodiscard]] std::uint64_t add(FilterType type);
    [[nodiscard]] bool remove(std::uint64_t id);
    [[nodiscard]] bool move_up(std::uint64_t id);
    [[nodiscard]] bool move_down(std::uint64_t id);
    void clear() noexcept;

    [[nodiscard]] std::vector<PipelineOperation>& operations() noexcept {
        return operations_;
    }
    [[nodiscard]] const std::vector<PipelineOperation>& operations() const noexcept {
        return operations_;
    }

    [[nodiscard]] const cuda::ImageBuffer& process(
        const cuda::ImageBuffer& input,
        cudaStream_t stream = nullptr);

private:
    std::vector<PipelineOperation> operations_;
    cuda::ImageBuffer buffers_[2];
    std::uint64_t next_id_{1};
};

}  // namespace hzl::processing
