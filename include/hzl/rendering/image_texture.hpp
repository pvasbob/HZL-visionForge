#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace hzl::rendering {

class ImageTexture {
public:
    ImageTexture() = default;
    ~ImageTexture();

    ImageTexture(const ImageTexture&) = delete;
    ImageTexture& operator=(const ImageTexture&) = delete;

    [[nodiscard]] bool upload_rgba8(const cv::Mat& pixels,
                                    std::string& error_message);
    void reset() noexcept;

    [[nodiscard]] unsigned int id() const noexcept { return id_; }
    [[nodiscard]] int width() const noexcept { return width_; }
    [[nodiscard]] int height() const noexcept { return height_; }
    [[nodiscard]] bool valid() const noexcept { return id_ != 0U; }

private:
    unsigned int id_{0};
    int width_{0};
    int height_{0};
};

}  // namespace hzl::rendering
