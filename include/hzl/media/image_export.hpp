#pragma once

#include <opencv2/core.hpp>

#include <filesystem>
#include <string>

namespace hzl::media {

enum class ImageExportError {
    none,
    invalid_image,
    unsupported_format,
    invalid_options,
    destination_exists,
    invalid_destination,
    encode_failed,
};

struct ImageExportOptions {
    int jpeg_quality{95};
    int png_compression{3};
    bool overwrite{false};
};

struct ImageExportResult {
    ImageExportError error{ImageExportError::none};
    std::string message;

    [[nodiscard]] explicit operator bool() const noexcept {
        return error == ImageExportError::none;
    }
};

[[nodiscard]] ImageExportResult export_image(
    const cv::Mat& rgba_pixels,
    const std::filesystem::path& destination,
    const ImageExportOptions& options = {});

}  // namespace hzl::media
