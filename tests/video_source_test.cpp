#include "hzl/media/video_source.hpp"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>

namespace {

class TemporaryVideo {
public:
    TemporaryVideo()
        : path_{std::filesystem::temp_directory_path() /
                ("hzl-visionforge-video-" +
                 std::to_string(std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count()) +
                 ".avi")} {}
    ~TemporaryVideo() {
        std::error_code ignored;
        std::filesystem::remove(path_, ignored);
    }
    [[nodiscard]] const std::filesystem::path& path() const { return path_; }

private:
    std::filesystem::path path_;
};

}  // namespace

int main() {
    const TemporaryVideo temporary;
    constexpr int width = 32;
    constexpr int height = 24;
    constexpr int frame_count = 4;
    cv::VideoWriter writer(temporary.path().string(),
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           20.0,
                           cv::Size{width, height});
    if (!writer.isOpened()) {
        std::cout << "Video source tests skipped: MJPG encoder unavailable\n";
        return 77;
    }
    for (int frame = 0; frame < frame_count; ++frame) {
        writer.write(cv::Mat(height,
                             width,
                             CV_8UC3,
                             cv::Scalar{static_cast<double>(frame * 20), 80.0, 170.0}));
    }
    writer.release();

    hzl::media::VideoSource source;
    const hzl::media::MediaOpenResult open = source.open_file(temporary.path());
    if (!open) {
        std::cerr << "FAIL: " << open.message << '\n';
        return 1;
    }
    bool passed = source.width() == width && source.height() == height &&
                  source.fps() > 0.0;
    int decoded_frames = 0;
    while (true) {
        hzl::media::FrameReadResult frame = source.read();
        if (frame.status == hzl::media::FrameReadStatus::end_of_stream) {
            break;
        }
        if (!frame) {
            std::cerr << "FAIL: " << frame.message << '\n';
            return 1;
        }
        passed = frame.rgba_pixels.rows == height &&
                     frame.rgba_pixels.cols == width &&
                     frame.rgba_pixels.type() == CV_8UC4 &&
                     frame.rgba_pixels.isContinuous() &&
                     frame.frame_number ==
                         static_cast<std::uint64_t>(decoded_frames + 1) &&
                 passed;
        ++decoded_frames;
    }
    passed = decoded_frames == frame_count && passed;
    source.close();
    passed = !source.is_open() && passed;

    if (passed) {
        std::cout << "Video source tests: passed\n";
        return 0;
    }
    std::cerr << "FAIL: video metadata or decoded frames differ\n";
    return 1;
}
