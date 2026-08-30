#include "hzl/media/image_export.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace hzl::media {
namespace {

enum class OutputFormat { png, jpeg, unsupported };

OutputFormat output_format(const std::filesystem::path& destination) {
    std::string extension = destination.extension().string();
    std::transform(extension.begin(),
                   extension.end(),
                   extension.begin(),
                   [](const unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    if (extension == ".png") {
        return OutputFormat::png;
    }
    if (extension == ".jpg" || extension == ".jpeg") {
        return OutputFormat::jpeg;
    }
    return OutputFormat::unsupported;
}

ImageExportResult failure(const ImageExportError error, std::string message) {
    return {error, std::move(message)};
}

}  // namespace

ImageExportResult export_image(const cv::Mat& rgba_pixels,
                               const std::filesystem::path& destination,
                               const ImageExportOptions& options) {
    if (rgba_pixels.empty() || rgba_pixels.type() != CV_8UC4) {
        return failure(ImageExportError::invalid_image,
                       "Export requires a non-empty RGBA8 image.");
    }
    if (destination.empty()) {
        return failure(ImageExportError::invalid_destination,
                       "No export destination was provided.");
    }

    const OutputFormat format = output_format(destination);
    if (format == OutputFormat::unsupported) {
        return failure(ImageExportError::unsupported_format,
                       "Unsupported export extension. Use PNG, JPG, or JPEG.");
    }
    if (options.jpeg_quality < 0 || options.jpeg_quality > 100 ||
        options.png_compression < 0 || options.png_compression > 9) {
        return failure(ImageExportError::invalid_options,
                       "JPEG quality must be 0-100 and PNG compression 0-9.");
    }

    std::error_code filesystem_error;
    const bool destination_exists =
        std::filesystem::exists(destination, filesystem_error);
    if (filesystem_error) {
        return failure(ImageExportError::invalid_destination,
                       "Could not inspect the export destination: " +
                           filesystem_error.message());
    }
    if (destination_exists && !options.overwrite) {
        return failure(ImageExportError::destination_exists,
                       "Destination already exists. Enable overwrite to replace it: " +
                           destination.string());
    }

    const std::filesystem::path parent = destination.parent_path();
    if (!parent.empty() &&
        !std::filesystem::is_directory(parent, filesystem_error)) {
        return failure(ImageExportError::invalid_destination,
                       "Export directory does not exist: " + parent.string());
    }
    if (filesystem_error) {
        return failure(ImageExportError::invalid_destination,
                       "Could not inspect the export directory: " +
                           filesystem_error.message());
    }

    try {
        cv::Mat encoded_pixels;
        std::vector<int> parameters;
        if (format == OutputFormat::png) {
            cv::cvtColor(rgba_pixels, encoded_pixels, cv::COLOR_RGBA2BGRA);
            parameters = {cv::IMWRITE_PNG_COMPRESSION, options.png_compression};
        } else {
            cv::cvtColor(rgba_pixels, encoded_pixels, cv::COLOR_RGBA2BGR);
            parameters = {cv::IMWRITE_JPEG_QUALITY, options.jpeg_quality};
        }

        if (!cv::imwrite(destination.string(), encoded_pixels, parameters)) {
            return failure(ImageExportError::encode_failed,
                           "OpenCV could not encode the exported image.");
        }
    } catch (const cv::Exception& exception) {
        return failure(ImageExportError::encode_failed,
                       "OpenCV export failed: " +
                           std::string{exception.what()});
    } catch (const std::exception& exception) {
        return failure(ImageExportError::encode_failed,
                       "Image export failed: " +
                           std::string{exception.what()});
    }

    return {};
}

}  // namespace hzl::media
