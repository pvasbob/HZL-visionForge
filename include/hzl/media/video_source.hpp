#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <cstdint>
#include <filesystem>
#include <string>

namespace hzl::media {

struct MediaOpenResult {
    bool success{false};
    std::string message;
    [[nodiscard]] explicit operator bool() const noexcept { return success; }
};

enum class FrameReadStatus { frame, end_of_stream, error };

struct FrameReadResult {
    FrameReadStatus status{FrameReadStatus::error};
    cv::Mat rgba_pixels;
    std::uint64_t frame_number{0};
    std::string message;
    [[nodiscard]] explicit operator bool() const noexcept {
        return status == FrameReadStatus::frame;
    }
};

class VideoSource {
public:
    [[nodiscard]] MediaOpenResult open_file(
        const std::filesystem::path& path);
    [[nodiscard]] MediaOpenResult open_camera(int camera_index);
    [[nodiscard]] FrameReadResult read();
    void close() noexcept;

    [[nodiscard]] bool is_open() const noexcept { return capture_.isOpened(); }
    [[nodiscard]] bool is_camera() const noexcept { return camera_; }
    [[nodiscard]] double fps() const noexcept { return fps_; }
    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] const std::string& label() const noexcept { return label_; }

private:
    cv::VideoCapture capture_;
    bool camera_{false};
    double fps_{0.0};
    int width_{0};
    int height_{0};
    std::uint64_t frame_number_{0};
    std::string label_;
};

}  // namespace hzl::media
