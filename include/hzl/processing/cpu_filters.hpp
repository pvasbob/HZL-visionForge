#pragma once

#include <opencv2/core.hpp>

#include <array>
#include <cstdint>

namespace hzl::processing::cpu {

// All filters accept and return contiguous CV_8UC4 RGBA images. Color filters
// preserve the source alpha channel and throw std::invalid_argument for invalid
// images or parameters.
[[nodiscard]] cv::Mat grayscale(const cv::Mat& rgba);
[[nodiscard]] cv::Mat invert(const cv::Mat& rgba);
[[nodiscard]] cv::Mat brightness_contrast(const cv::Mat& rgba,
                                          double brightness,
                                          double contrast);
[[nodiscard]] cv::Mat gamma_correction(const cv::Mat& rgba, double gamma);
[[nodiscard]] cv::Mat box_blur(const cv::Mat& rgba, int kernel_size);
[[nodiscard]] cv::Mat gaussian_blur(const cv::Mat& rgba,
                                    int kernel_size,
                                    double sigma);
[[nodiscard]] cv::Mat sharpen(const cv::Mat& rgba, double amount);
[[nodiscard]] cv::Mat emboss(const cv::Mat& rgba, double strength);
[[nodiscard]] cv::Mat sobel_edges(const cv::Mat& rgba, double strength = 1.0);
[[nodiscard]] cv::Mat laplacian_edges(const cv::Mat& rgba,
                                      double strength = 1.0);
using Histogram = std::array<std::uint32_t, 256>;
[[nodiscard]] Histogram luminance_histogram(const cv::Mat& rgba);
[[nodiscard]] cv::Mat histogram_equalization(const cv::Mat& rgba);
[[nodiscard]] cv::Mat tone_map(const cv::Mat& rgba, double exposure);
[[nodiscard]] cv::Mat color_grade(const cv::Mat& rgba,
                                  double saturation,
                                  double temperature,
                                  double tint,
                                  double red_gain,
                                  double green_gain,
                                  double blue_gain);

}  // namespace hzl::processing::cpu
