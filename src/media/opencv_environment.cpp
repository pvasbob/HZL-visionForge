#include "hzl/media/opencv_environment.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdint>
#include <exception>
#include <ostream>

namespace hzl::media {

OpenCvProbeResult report_opencv_environment(std::ostream& output,
                                            std::ostream& errors) {
    output << "OpenCV version: " << CV_VERSION << '\n'
           << "OpenCV components: core, imgproc, imgcodecs, videoio\n";

    try {
        cv::Mat color_image{2, 2, CV_8UC3, cv::Scalar{10, 20, 30}};
        cv::Mat grayscale_image;
        cv::cvtColor(color_image, grayscale_image, cv::COLOR_BGR2GRAY);

        if (grayscale_image.rows != 2 || grayscale_image.cols != 2 ||
            grayscale_image.type() != CV_8UC1 ||
            grayscale_image.at<std::uint8_t>(0, 0) != 22U) {
            errors << "OpenCV smoke test failed: unexpected grayscale output.\n";
            return OpenCvProbeResult::smoke_test_failed;
        }
    } catch (const cv::Exception& exception) {
        errors << "OpenCV smoke test failed: " << exception.what() << '\n';
        return OpenCvProbeResult::smoke_test_failed;
    } catch (const std::exception& exception) {
        errors << "OpenCV smoke test failed: " << exception.what() << '\n';
        return OpenCvProbeResult::smoke_test_failed;
    }

    output << "OpenCV smoke test: passed\n";
    return OpenCvProbeResult::available;
}

}  // namespace hzl::media
