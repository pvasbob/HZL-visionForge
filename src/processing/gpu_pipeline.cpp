#include "hzl/processing/gpu_pipeline.hpp"

#include "hzl/processing/cuda_basic_filters.hpp"
#include "hzl/processing/cuda_convolution_filters.hpp"
#include "hzl/processing/cuda_edge_filters.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace hzl::processing {
namespace {

FilterParameters default_parameters(const FilterType type) {
    switch (type) {
        case FilterType::grayscale:
        case FilterType::invert:
            return std::monostate{};
        case FilterType::brightness_contrast:
            return BrightnessContrastParameters{};
        case FilterType::gamma:
            return GammaParameters{};
        case FilterType::box_blur:
            return BoxBlurParameters{};
        case FilterType::gaussian_blur:
            return GaussianBlurParameters{};
        case FilterType::sharpen:
            return SharpenParameters{};
        case FilterType::emboss:
            return EmbossParameters{};
        case FilterType::sobel:
        case FilterType::laplacian:
            return EdgeParameters{};
    }
    throw std::logic_error{"Unknown pipeline filter type."};
}

void launch(const PipelineOperation& operation,
            const cuda::ImageBuffer& input,
            cuda::ImageBuffer& output,
            const cudaStream_t stream) {
    switch (operation.type) {
        case FilterType::grayscale:
            cuda::grayscale(input, output, stream);
            break;
        case FilterType::invert:
            cuda::invert(input, output, stream);
            break;
        case FilterType::brightness_contrast: {
            const auto parameters =
                std::get<BrightnessContrastParameters>(operation.parameters);
            cuda::brightness_contrast(input,
                                      output,
                                      parameters.brightness,
                                      parameters.contrast,
                                      stream);
            break;
        }
        case FilterType::gamma:
            cuda::gamma_correction(
                input,
                output,
                std::get<GammaParameters>(operation.parameters).gamma,
                stream);
            break;
        case FilterType::box_blur:
            cuda::box_blur(
                input,
                output,
                std::get<BoxBlurParameters>(operation.parameters).kernel_size,
                cuda::KernelImplementation::shared_memory,
                stream);
            break;
        case FilterType::gaussian_blur: {
            const auto parameters =
                std::get<GaussianBlurParameters>(operation.parameters);
            cuda::gaussian_blur(input,
                                output,
                                parameters.kernel_size,
                                parameters.sigma,
                                cuda::KernelImplementation::shared_memory,
                                stream);
            break;
        }
        case FilterType::sharpen:
            cuda::sharpen(
                input,
                output,
                std::get<SharpenParameters>(operation.parameters).amount,
                cuda::KernelImplementation::shared_memory,
                stream);
            break;
        case FilterType::emboss:
            cuda::emboss(
                input,
                output,
                std::get<EmbossParameters>(operation.parameters).strength,
                cuda::KernelImplementation::shared_memory,
                stream);
            break;
        case FilterType::sobel:
            cuda::sobel_edges(
                input,
                output,
                std::get<EdgeParameters>(operation.parameters).strength,
                cuda::KernelImplementation::shared_memory,
                stream);
            break;
        case FilterType::laplacian:
            cuda::laplacian_edges(
                input,
                output,
                std::get<EdgeParameters>(operation.parameters).strength,
                cuda::KernelImplementation::shared_memory,
                stream);
            break;
    }
}

}  // namespace

const char* filter_name(const FilterType type) noexcept {
    switch (type) {
        case FilterType::grayscale: return "Grayscale";
        case FilterType::invert: return "Invert";
        case FilterType::brightness_contrast: return "Brightness / Contrast";
        case FilterType::gamma: return "Gamma";
        case FilterType::box_blur: return "Box Blur";
        case FilterType::gaussian_blur: return "Gaussian Blur";
        case FilterType::sharpen: return "Sharpen";
        case FilterType::emboss: return "Emboss";
        case FilterType::sobel: return "Sobel";
        case FilterType::laplacian: return "Laplacian";
    }
    return "Unknown";
}

std::uint64_t GpuPipeline::add(const FilterType type) {
    const std::uint64_t id = next_id_++;
    operations_.push_back({id, type, true, default_parameters(type)});
    return id;
}

bool GpuPipeline::remove(const std::uint64_t id) {
    const auto operation = std::find_if(
        operations_.begin(), operations_.end(),
        [id](const PipelineOperation& candidate) { return candidate.id == id; });
    if (operation == operations_.end()) {
        return false;
    }
    operations_.erase(operation);
    return true;
}

bool GpuPipeline::move_up(const std::uint64_t id) {
    const auto operation = std::find_if(
        operations_.begin(), operations_.end(),
        [id](const PipelineOperation& candidate) { return candidate.id == id; });
    if (operation == operations_.end() || operation == operations_.begin()) {
        return false;
    }
    std::iter_swap(operation, operation - 1);
    return true;
}

bool GpuPipeline::move_down(const std::uint64_t id) {
    const auto operation = std::find_if(
        operations_.begin(), operations_.end(),
        [id](const PipelineOperation& candidate) { return candidate.id == id; });
    if (operation == operations_.end() || operation + 1 == operations_.end()) {
        return false;
    }
    std::iter_swap(operation, operation + 1);
    return true;
}

void GpuPipeline::clear() noexcept {
    operations_.clear();
}

const cuda::ImageBuffer& GpuPipeline::process(const cuda::ImageBuffer& input,
                                              const cudaStream_t stream) {
    const cuda::ImageBuffer* current = &input;
    std::size_t output_index = 0;
    for (const PipelineOperation& operation : operations_) {
        if (!operation.enabled) {
            continue;
        }
        cuda::ImageBuffer& output = buffers_[output_index];
        launch(operation, *current, output, stream);
        current = &output;
        output_index = 1U - output_index;
    }
    return *current;
}

}  // namespace hzl::processing
