#include "hzl/media/video_source.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <exception>
#include <filesystem>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace hzl::media {
namespace {

cv::Mat normalize_frame(const cv::Mat& decoded) {
    cv::Mat eight_bit;
    if (decoded.depth() == CV_8U) {
        eight_bit = decoded;
    } else if (decoded.depth() == CV_16U) {
        decoded.convertTo(eight_bit,
                          CV_MAKETYPE(CV_8U, decoded.channels()),
                          1.0 / 257.0);
    } else {
        throw std::runtime_error{"Unsupported video frame pixel depth."};
    }

    cv::Mat rgba;
    switch (eight_bit.channels()) {
        case 1:
            cv::cvtColor(eight_bit, rgba, cv::COLOR_GRAY2RGBA);
            break;
        case 2: {
            std::vector<cv::Mat> gray_alpha;
            cv::split(eight_bit, gray_alpha);
            std::vector<cv::Mat> channels{
                gray_alpha[0], gray_alpha[0], gray_alpha[0], gray_alpha[1]};
            cv::merge(channels, rgba);
            break;
        }
        case 3:
            cv::cvtColor(eight_bit, rgba, cv::COLOR_BGR2RGBA);
            break;
        case 4:
            cv::cvtColor(eight_bit, rgba, cv::COLOR_BGRA2RGBA);
            break;
        default:
            throw std::runtime_error{"Unsupported video frame channel count."};
    }
    return rgba.isContinuous() ? rgba : rgba.clone();
}

}  // namespace

MediaOpenResult VideoSource::open_file(const std::filesystem::path& path) {
    close();
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(path, filesystem_error)) {
        return {false, "Video file does not exist: " + path.string()};
    }
    try {
        if (!capture_.open(path.string(), cv::CAP_ANY)) {
            return {false, "OpenCV could not open the video: " + path.string()};
        }
        camera_ = false;
        label_ = path.string();
        fps_ = capture_.get(cv::CAP_PROP_FPS);
        width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
        return {true, {}};
    } catch (const cv::Exception& exception) {
        close();
        return {false, "OpenCV video open failed: " +
                           std::string{exception.what()}};
    }
}

MediaOpenResult VideoSource::open_camera(const int camera_index) {
    close();
    if (camera_index < 0) {
        return {false, "Camera index cannot be negative."};
    }
    try {
        if (!capture_.open(camera_index, cv::CAP_ANY)) {
            return {false,
                    "OpenCV could not open camera " +
                        std::to_string(camera_index) + '.'};
        }
        camera_ = true;
        label_ = "Camera " + std::to_string(camera_index);
        fps_ = capture_.get(cv::CAP_PROP_FPS);
        width_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_HEIGHT));
        return {true, {}};
    } catch (const cv::Exception& exception) {
        close();
        return {false, "OpenCV camera open failed: " +
                           std::string{exception.what()}};
    }
}

FrameReadResult VideoSource::read() {
    if (!capture_.isOpened()) {
        return {FrameReadStatus::error, {}, frame_number_,
                "No video or camera source is open."};
    }
    try {
        cv::Mat decoded;
        if (!capture_.read(decoded) || decoded.empty()) {
            return {camera_ ? FrameReadStatus::error
                            : FrameReadStatus::end_of_stream,
                    {},
                    frame_number_,
                    camera_ ? "Camera frame capture failed."
                            : "Video reached end of stream."};
        }
        cv::Mat rgba = normalize_frame(decoded);
        ++frame_number_;
        width_ = rgba.cols;
        height_ = rgba.rows;
        return {FrameReadStatus::frame,
                std::move(rgba),
                frame_number_,
                {}};
    } catch (const cv::Exception& exception) {
        return {FrameReadStatus::error, {}, frame_number_,
                "OpenCV frame decode failed: " +
                    std::string{exception.what()}};
    } catch (const std::exception& exception) {
        return {FrameReadStatus::error, {}, frame_number_, exception.what()};
    }
}

void VideoSource::close() noexcept {
    capture_.release();
    camera_ = false;
    fps_ = 0.0;
    width_ = 0;
    height_ = 0;
    frame_number_ = 0;
    label_.clear();
}

}  // namespace hzl::media
