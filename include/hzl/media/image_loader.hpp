#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <optional>
#include <string>

namespace hzl::media {

enum class ImageLoadError {
    none,
    unsupported_format,
    file_not_found,
    decode_failed,
    unsupported_pixel_format,
};

struct ImageDocument {
    std::filesystem::path source_path;
    cv::Mat rgba_pixels;
    int original_channels{0};
    int original_depth{0};
};

struct ImageLoadResult {
    std::optional<ImageDocument> document;
    ImageLoadError error{ImageLoadError::none};
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return document.has_value();
    }
};

[[nodiscard]] ImageLoadResult load_image(const std::filesystem::path& path);
[[nodiscard]] const char* pixel_depth_name(int depth) noexcept;

}  // namespace hzl::media
