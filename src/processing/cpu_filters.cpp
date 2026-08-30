#include "hzl/processing/cpu_filters.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace hzl::processing::cpu {
namespace {

void validate_image(const cv::Mat& rgba) {
    if (rgba.empty() || rgba.type() != CV_8UC4) {
        throw std::invalid_argument{"CPU filters require a non-empty RGBA8 image."};
    }
}

void validate_kernel_size(const int kernel_size) {
    if (kernel_size < 1 || kernel_size % 2 == 0) {
        throw std::invalid_argument{"Kernel size must be a positive odd number."};
    }
}

std::array<cv::Mat, 4> split_rgba(const cv::Mat& rgba) {
    std::vector<cv::Mat> channels;
    cv::split(rgba, channels);
    return {channels[0], channels[1], channels[2], channels[3]};
}

cv::Mat merge_rgb_alpha(const cv::Mat& rgb, const cv::Mat& alpha) {
    std::vector<cv::Mat> rgb_channels;
    cv::split(rgb, rgb_channels);
    rgb_channels.push_back(alpha);

    cv::Mat output;
    cv::merge(rgb_channels, output);
    return output.isContinuous() ? output : output.clone();
}

cv::Mat rgb_from_rgba(const cv::Mat& rgba) {
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
    return rgb;
}

cv::Mat gray_with_alpha(const cv::Mat& gray, const cv::Mat& alpha) {
    std::vector<cv::Mat> channels{gray, gray, gray, alpha};
    cv::Mat output;
    cv::merge(channels, output);
    return output.isContinuous() ? output : output.clone();
}

cv::Mat source_alpha(const cv::Mat& rgba) {
    return split_rgba(rgba)[3];
}

}  // namespace

cv::Mat grayscale(const cv::Mat& rgba) {
    validate_image(rgba);
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    return gray_with_alpha(gray, source_alpha(rgba));
}

cv::Mat invert(const cv::Mat& rgba) {
    validate_image(rgba);
    cv::Mat rgb = rgb_from_rgba(rgba);
    cv::bitwise_not(rgb, rgb);
    return merge_rgb_alpha(rgb, source_alpha(rgba));
}

cv::Mat brightness_contrast(const cv::Mat& rgba,
                            const double brightness,
                            const double contrast) {
    validate_image(rgba);
    if (!std::isfinite(brightness) || !std::isfinite(contrast) || contrast < 0.0) {
        throw std::invalid_argument{
            "Brightness and contrast must be finite; contrast cannot be negative."};
    }

    cv::Mat adjusted;
    rgb_from_rgba(rgba).convertTo(adjusted, CV_8UC3, contrast, brightness);
    return merge_rgb_alpha(adjusted, source_alpha(rgba));
}

cv::Mat gamma_correction(const cv::Mat& rgba, const double gamma) {
    validate_image(rgba);
    if (!std::isfinite(gamma) || gamma <= 0.0) {
        throw std::invalid_argument{"Gamma must be finite and greater than zero."};
    }

    cv::Mat lookup_table(1, 256, CV_8UC1);
    for (int value = 0; value < 256; ++value) {
        const double normalized = static_cast<double>(value) / 255.0;
        lookup_table.at<unsigned char>(0, value) = static_cast<unsigned char>(
            std::lround(std::pow(normalized, 1.0 / gamma) * 255.0));
    }

    cv::Mat corrected;
    cv::LUT(rgb_from_rgba(rgba), lookup_table, corrected);
    return merge_rgb_alpha(corrected, source_alpha(rgba));
}

cv::Mat box_blur(const cv::Mat& rgba, const int kernel_size) {
    validate_image(rgba);
    validate_kernel_size(kernel_size);

    cv::Mat blurred;
    cv::blur(rgb_from_rgba(rgba),
             blurred,
             cv::Size{kernel_size, kernel_size},
             cv::Point{-1, -1},
             cv::BORDER_REFLECT_101);
    return merge_rgb_alpha(blurred, source_alpha(rgba));
}

cv::Mat gaussian_blur(const cv::Mat& rgba,
                      const int kernel_size,
                      const double sigma) {
    validate_image(rgba);
    validate_kernel_size(kernel_size);
    if (!std::isfinite(sigma) || sigma < 0.0) {
        throw std::invalid_argument{"Gaussian sigma must be finite and non-negative."};
    }

    cv::Mat blurred;
    cv::GaussianBlur(rgb_from_rgba(rgba),
                     blurred,
                     cv::Size{kernel_size, kernel_size},
                     sigma,
                     sigma,
                     cv::BORDER_REFLECT_101);
    return merge_rgb_alpha(blurred, source_alpha(rgba));
}

cv::Mat sharpen(const cv::Mat& rgba, const double amount) {
    validate_image(rgba);
    if (!std::isfinite(amount) || amount < 0.0) {
        throw std::invalid_argument{"Sharpen amount must be finite and non-negative."};
    }

    const cv::Mat rgb = rgb_from_rgba(rgba);
    cv::Mat blurred;
    cv::GaussianBlur(rgb, blurred, cv::Size{0, 0}, 1.0);
    cv::Mat sharpened;
    cv::addWeighted(rgb, 1.0 + amount, blurred, -amount, 0.0, sharpened);
    return merge_rgb_alpha(sharpened, source_alpha(rgba));
}

cv::Mat emboss(const cv::Mat& rgba, const double strength) {
    validate_image(rgba);
    if (!std::isfinite(strength) || strength < 0.0) {
        throw std::invalid_argument{"Emboss strength must be finite and non-negative."};
    }

    const cv::Mat kernel =
        (cv::Mat_<float>(3, 3) << -2.0F, -1.0F, 0.0F,
                                   -1.0F, 0.0F, 1.0F,
                                    0.0F, 1.0F, 2.0F);
    cv::Mat filtered_float;
    cv::filter2D(rgb_from_rgba(rgba),
                 filtered_float,
                 CV_32FC3,
                 kernel,
                 cv::Point{-1, -1},
                 0.0,
                 cv::BORDER_REFLECT_101);
    cv::Mat embossed;
    filtered_float.convertTo(embossed, CV_8UC3, strength, 128.0);
    return merge_rgb_alpha(embossed, source_alpha(rgba));
}

cv::Mat sobel_edges(const cv::Mat& rgba, const double strength) {
    validate_image(rgba);
    if (!std::isfinite(strength) || strength < 0.0) {
        throw std::invalid_argument{"Edge strength must be finite and non-negative."};
    }
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    cv::Mat horizontal;
    cv::Mat vertical;
    cv::Sobel(gray, horizontal, CV_32F, 1, 0, 3, 1.0, 0.0,
              cv::BORDER_REFLECT_101);
    cv::Sobel(gray, vertical, CV_32F, 0, 1, 3, 1.0, 0.0,
              cv::BORDER_REFLECT_101);
    cv::Mat magnitude;
    cv::magnitude(horizontal, vertical, magnitude);
    cv::Mat edges;
    magnitude.convertTo(edges, CV_8UC1, strength);
    return gray_with_alpha(edges, source_alpha(rgba));
}

cv::Mat laplacian_edges(const cv::Mat& rgba, const double strength) {
    validate_image(rgba);
    if (!std::isfinite(strength) || strength < 0.0) {
        throw std::invalid_argument{"Edge strength must be finite and non-negative."};
    }
    cv::Mat gray;
    cv::cvtColor(rgba, gray, cv::COLOR_RGBA2GRAY);
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_16S, 1, 1.0, 0.0,
                  cv::BORDER_REFLECT_101);
    cv::Mat edges;
    cv::convertScaleAbs(laplacian, edges, strength);
    return gray_with_alpha(edges, source_alpha(rgba));
}

}  // namespace hzl::processing::cpu
