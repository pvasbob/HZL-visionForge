#include "hzl/media/image_loader.hpp"

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
                ("hzl-visionforge-images-" +
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
    const cv::Mat color_image{7, 11, CV_8UC3, cv::Scalar{15, 90, 210}};
    bool passed = true;

    for (const char* extension_value : {".png", ".jpg", ".bmp", ".tiff"}) {
        const std::string extension{extension_value};
        const std::filesystem::path path =
            temporary_directory.path() / ("fixture" + extension);
        passed = check(cv::imwrite(path.string(), color_image),
                       "could not generate " + extension + " fixture") &&
                 passed;

        const hzl::media::ImageLoadResult result = hzl::media::load_image(path);
        passed = check(static_cast<bool>(result),
                       "could not load " + extension + ": " + result.message) &&
                 passed;
        if (result) {
            const hzl::media::ImageDocument& document = result.document.value();
            passed = check(document.rgba_pixels.rows == 7 &&
                               document.rgba_pixels.cols == 11,
                           extension + " dimensions changed") &&
                     passed;
            passed = check(document.rgba_pixels.type() == CV_8UC4,
                           extension + " was not normalized to RGBA8") &&
                     passed;
            passed = check(document.rgba_pixels.isContinuous(),
                           extension + " pixels are not contiguous") &&
                     passed;
        }
    }

    const std::filesystem::path grayscale_path =
        temporary_directory.path() / "grayscale16.tiff";
    const cv::Mat grayscale_image{3, 5, CV_16UC1, cv::Scalar{32768}};
    passed = check(cv::imwrite(grayscale_path.string(), grayscale_image),
                   "could not generate 16-bit TIFF fixture") &&
             passed;
    const hzl::media::ImageLoadResult grayscale_result =
        hzl::media::load_image(grayscale_path);
    passed = check(static_cast<bool>(grayscale_result),
                   "could not load 16-bit grayscale TIFF") &&
             passed;
    if (grayscale_result) {
        passed = check(grayscale_result.document->original_depth == CV_16U &&
                           grayscale_result.document->original_channels == 1 &&
                           grayscale_result.document->rgba_pixels.type() == CV_8UC4,
                       "16-bit grayscale metadata or normalization is incorrect") &&
                 passed;
    }

    const hzl::media::ImageLoadResult unsupported =
        hzl::media::load_image(temporary_directory.path() / "fixture.gif");
    passed = check(unsupported.error ==
                       hzl::media::ImageLoadError::unsupported_format,
                   "unsupported extension did not produce the expected error") &&
             passed;

    const hzl::media::ImageLoadResult missing =
        hzl::media::load_image(temporary_directory.path() / "missing.png");
    passed = check(missing.error == hzl::media::ImageLoadError::file_not_found,
                   "missing file did not produce the expected error") &&
             passed;

    if (passed) {
        std::cout << "Image loader tests: passed\n";
        return 0;
    }
    return 1;
}
