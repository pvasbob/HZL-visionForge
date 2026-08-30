#include "hzl/media/image_export.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

namespace {

class TemporaryDirectory {
public:
    TemporaryDirectory()
        : path_{std::filesystem::temp_directory_path() /
                ("hzl-visionforge-exports-" +
                 std::to_string(std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count()))} {
        std::filesystem::create_directories(path_);
    }

    ~TemporaryDirectory() {
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

bool check(const bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
    }
    return condition;
}

}  // namespace

int main() {
    const TemporaryDirectory temporary_directory;
    cv::Mat rgba{9, 13, CV_8UC4, cv::Scalar{20, 80, 190, 127}};
    bool passed = true;

    const std::filesystem::path png_path =
        temporary_directory.path() / "export.png";
    const hzl::media::ImageExportResult png_result =
        hzl::media::export_image(rgba, png_path);
    passed = check(static_cast<bool>(png_result),
                   "PNG export failed: " + png_result.message) &&
             passed;
    const cv::Mat png_round_trip =
        cv::imread(png_path.string(), cv::IMREAD_UNCHANGED);
    passed = check(png_round_trip.rows == 9 && png_round_trip.cols == 13 &&
                       png_round_trip.type() == CV_8UC4,
                   "PNG dimensions or alpha channel were not preserved") &&
             passed;
    if (!png_round_trip.empty()) {
        const cv::Vec4b pixel = png_round_trip.at<cv::Vec4b>(0, 0);
        passed = check(pixel[0] == 190U && pixel[1] == 80U &&
                           pixel[2] == 20U && pixel[3] == 127U,
                       "PNG RGBA-to-BGRA conversion is incorrect") &&
                 passed;
    }

    const hzl::media::ImageExportResult protected_result =
        hzl::media::export_image(rgba, png_path);
    passed = check(protected_result.error ==
                       hzl::media::ImageExportError::destination_exists,
                   "existing destination was overwritten without permission") &&
             passed;

    hzl::media::ImageExportOptions overwrite_options;
    overwrite_options.overwrite = true;
    passed = check(static_cast<bool>(
                       hzl::media::export_image(rgba, png_path, overwrite_options)),
                   "explicit PNG overwrite failed") &&
             passed;

    const std::filesystem::path jpeg_path =
        temporary_directory.path() / "export.jpg";
    hzl::media::ImageExportOptions jpeg_options;
    jpeg_options.jpeg_quality = 87;
    const hzl::media::ImageExportResult jpeg_result =
        hzl::media::export_image(rgba, jpeg_path, jpeg_options);
    passed = check(static_cast<bool>(jpeg_result),
                   "JPEG export failed: " + jpeg_result.message) &&
             passed;
    const cv::Mat jpeg_round_trip = cv::imread(jpeg_path.string());
    passed = check(jpeg_round_trip.rows == 9 && jpeg_round_trip.cols == 13 &&
                       jpeg_round_trip.type() == CV_8UC3,
                   "JPEG dimensions or channel count are incorrect") &&
             passed;

    const hzl::media::ImageExportResult unsupported = hzl::media::export_image(
        rgba, temporary_directory.path() / "export.tiff");
    passed = check(unsupported.error ==
                       hzl::media::ImageExportError::unsupported_format,
                   "unsupported export extension produced the wrong error") &&
             passed;

    const hzl::media::ImageExportResult invalid =
        hzl::media::export_image(cv::Mat{},
                                 temporary_directory.path() / "invalid.png");
    passed = check(invalid.error == hzl::media::ImageExportError::invalid_image,
                   "invalid image produced the wrong error") &&
             passed;

    if (passed) {
        std::cout << "Image export tests: passed\n";
        return 0;
    }
    return 1;
}
