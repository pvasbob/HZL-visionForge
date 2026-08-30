#include "hzl/media/image_loader.hpp"

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

bool is_supported_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(),
                   extension.end(),
                   extension.begin(),
                   [](const unsigned char character) {
                       return static_cast<char>(std::tolower(character));
                   });
    return extension == ".png" || extension == ".jpg" ||
           extension == ".jpeg" || extension == ".bmp" ||
           extension == ".tif" || extension == ".tiff";
}

ImageLoadResult failure(const ImageLoadError error, std::string message) {
    return {std::nullopt, error, std::move(message)};
}

cv::Mat convert_depth_to_u8(const cv::Mat& source) {
    if (source.depth() == CV_8U) {
        return source;
    }

    cv::Mat converted;
    source.convertTo(converted, CV_MAKETYPE(CV_8U, source.channels()), 1.0 / 257.0);
    return converted;
}

cv::Mat convert_to_rgba(const cv::Mat& source) {
    cv::Mat rgba;
    switch (source.channels()) {
        case 1:
            cv::cvtColor(source, rgba, cv::COLOR_GRAY2RGBA);
            break;
        case 2: {
            std::vector<cv::Mat> gray_alpha;
            cv::split(source, gray_alpha);
            std::vector<cv::Mat> rgba_channels{
                gray_alpha[0], gray_alpha[0], gray_alpha[0], gray_alpha[1]};
            cv::merge(rgba_channels, rgba);
            break;
        }
        case 3:
            cv::cvtColor(source, rgba, cv::COLOR_BGR2RGBA);
            break;
        case 4:
            cv::cvtColor(source, rgba, cv::COLOR_BGRA2RGBA);
            break;
        default:
            break;
    }
    return rgba;
}

}  // namespace

ImageLoadResult load_image(const std::filesystem::path& path) {
    if (path.empty()) {
        return failure(ImageLoadError::file_not_found,
                       "No image path was provided.");
    }
    if (!is_supported_extension(path)) {
        return failure(ImageLoadError::unsupported_format,
                       "Unsupported image extension. Use PNG, JPEG, BMP, or TIFF.");
    }

    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(path, filesystem_error)) {
        return failure(ImageLoadError::file_not_found,
                       "Image file does not exist or is not a regular file: " +
                           path.string());
    }

    try {
        cv::Mat decoded = cv::imread(path.string(), cv::IMREAD_UNCHANGED);
        if (decoded.empty()) {
            return failure(ImageLoadError::decode_failed,
                           "OpenCV could not decode the image: " + path.string());
        }
        if (decoded.depth() != CV_8U && decoded.depth() != CV_16U) {
            return failure(
                ImageLoadError::unsupported_pixel_format,
                "Unsupported pixel depth " +
                    std::string{pixel_depth_name(decoded.depth())} +
                    ". Only 8-bit and 16-bit integer images are supported.");
        }
        if (decoded.channels() < 1 || decoded.channels() > 4) {
            return failure(ImageLoadError::unsupported_pixel_format,
                           "Unsupported channel count: " +
                               std::to_string(decoded.channels()));
        }

        const int original_channels = decoded.channels();
        const int original_depth = decoded.depth();
        cv::Mat rgba = convert_to_rgba(convert_depth_to_u8(decoded));
        if (rgba.empty() || rgba.type() != CV_8UC4) {
            return failure(ImageLoadError::unsupported_pixel_format,
                           "Could not normalize the image to RGBA8.");
        }
        if (!rgba.isContinuous()) {
            rgba = rgba.clone();
        }

        return {ImageDocument{path,
                              std::move(rgba),
                              original_channels,
                              original_depth},
                ImageLoadError::none,
                {}};
    } catch (const cv::Exception& exception) {
        return failure(ImageLoadError::decode_failed,
                       "OpenCV failed to load the image: " +
                           std::string{exception.what()});
    } catch (const std::exception& exception) {
        return failure(ImageLoadError::decode_failed,
                       "Failed to load the image: " +
                           std::string{exception.what()});
    }
}

const char* pixel_depth_name(const int depth) noexcept {
    switch (depth) {
        case CV_8U:
            return "8-bit unsigned";
        case CV_8S:
            return "8-bit signed";
        case CV_16U:
            return "16-bit unsigned";
        case CV_16S:
            return "16-bit signed";
        case CV_32S:
            return "32-bit signed";
        case CV_32F:
            return "32-bit float";
        case CV_64F:
            return "64-bit float";
        default:
            return "unknown";
    }
}

}  // namespace hzl::media
