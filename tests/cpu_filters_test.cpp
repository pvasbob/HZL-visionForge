#include "hzl/processing/cpu_filters.hpp"

#include <opencv2/core.hpp>

#include <array>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool check(const bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
    }
    return condition;
}

bool valid_output(const cv::Mat& source,
                  const cv::Mat& output,
                  const std::string& filter_name) {
    bool passed = check(output.size() == source.size(),
                        filter_name + " changed image dimensions");
    passed = check(output.type() == CV_8UC4,
                   filter_name + " did not return RGBA8") &&
             passed;
    passed = check(output.isContinuous(),
                   filter_name + " output is not contiguous") &&
             passed;
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            passed = check(output.at<cv::Vec4b>(row, column)[3] ==
                               source.at<cv::Vec4b>(row, column)[3],
                           filter_name + " changed alpha") &&
                     passed;
        }
    }
    return passed;
}

template <typename Operation>
bool throws_invalid_argument(Operation operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return true;
    } catch (const std::exception&) {
        return false;
    }
    return false;
}

}  // namespace

int main() {
    cv::Mat source(5, 7, CV_8UC4);
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            source.at<cv::Vec4b>(row, column) = cv::Vec4b{
                static_cast<unsigned char>(row * 31 + column * 7),
                static_cast<unsigned char>(row * 13 + column * 19),
                static_cast<unsigned char>(row * 23 + column * 11),
                static_cast<unsigned char>(80 + row * 17 + column)};
        }
    }

    using NamedOutput = std::pair<const char*, cv::Mat>;
    const std::array<NamedOutput, 13> outputs{{
        {"grayscale", hzl::processing::cpu::grayscale(source)},
        {"invert", hzl::processing::cpu::invert(source)},
        {"brightness/contrast",
         hzl::processing::cpu::brightness_contrast(source, 12.0, 1.1)},
        {"gamma", hzl::processing::cpu::gamma_correction(source, 1.8)},
        {"box blur", hzl::processing::cpu::box_blur(source, 3)},
        {"Gaussian blur",
         hzl::processing::cpu::gaussian_blur(source, 3, 1.2)},
        {"sharpen", hzl::processing::cpu::sharpen(source, 1.0)},
        {"emboss", hzl::processing::cpu::emboss(source, 1.0)},
        {"Sobel", hzl::processing::cpu::sobel_edges(source)},
        {"Laplacian", hzl::processing::cpu::laplacian_edges(source)},
        {"histogram equalization",
         hzl::processing::cpu::histogram_equalization(source)},
        {"tone mapping", hzl::processing::cpu::tone_map(source, 1.0)},
        {"color grading", hzl::processing::cpu::color_grade(
                              source, 1.2, 0.15, -0.1, 1.05, 0.95, 1.1)},
    }};

    bool passed = true;
    for (const NamedOutput& output : outputs) {
        passed = valid_output(source, output.second, output.first) && passed;
    }

    const cv::Vec4b source_pixel = source.at<cv::Vec4b>(2, 3);
    const cv::Vec4b inverted_pixel = outputs[1].second.at<cv::Vec4b>(2, 3);
    passed = check(inverted_pixel[0] == 255U - source_pixel[0] &&
                       inverted_pixel[1] == 255U - source_pixel[1] &&
                       inverted_pixel[2] == 255U - source_pixel[2],
                   "invert produced incorrect RGB values") &&
             passed;

    passed = check(cv::norm(hzl::processing::cpu::gamma_correction(source, 1.0),
                            source,
                            cv::NORM_INF) == 0.0,
                   "gamma 1.0 is not an identity") &&
             passed;
    passed = check(cv::norm(hzl::processing::cpu::brightness_contrast(
                                source, 0.0, 1.0),
                            source,
                            cv::NORM_INF) == 0.0,
                   "brightness 0 and contrast 1 are not an identity") &&
             passed;
    passed = check(cv::norm(hzl::processing::cpu::sharpen(source, 0.0),
                            source,
                            cv::NORM_INF) == 0.0,
                   "sharpen amount 0 is not an identity") &&
             passed;
    const hzl::processing::cpu::Histogram histogram =
        hzl::processing::cpu::luminance_histogram(source);
    std::uint64_t histogram_total = 0;
    for (const std::uint32_t count : histogram) histogram_total += count;
    passed = check(histogram_total == static_cast<std::uint64_t>(source.total()),
                   "histogram bin total differs from pixel count") && passed;

    cv::Mat known(1, 4, CV_8UC4);
    known.at<cv::Vec4b>(0, 0) = cv::Vec4b{0, 0, 0, 1};
    known.at<cv::Vec4b>(0, 1) = cv::Vec4b{255, 255, 255, 2};
    known.at<cv::Vec4b>(0, 2) = cv::Vec4b{255, 0, 0, 3};
    known.at<cv::Vec4b>(0, 3) = cv::Vec4b{0, 255, 0, 4};
    const auto known_histogram =
        hzl::processing::cpu::luminance_histogram(known);
    passed = check(known_histogram[0] == 1 && known_histogram[255] == 1 &&
                       known_histogram[77] == 1 && known_histogram[149] == 1,
                   "luminance histogram produced incorrect bins") && passed;

    cv::Mat constant(3, 3, CV_8UC4, cv::Scalar{42, 81, 123, 177});
    passed = check(cv::norm(hzl::processing::cpu::histogram_equalization(constant),
                            constant, cv::NORM_INF) == 0.0,
                   "equalization changed a constant image") && passed;
    passed = check(cv::norm(hzl::processing::cpu::color_grade(
                                source, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0),
                            source, cv::NORM_INF) == 0.0,
                   "neutral color grading is not an identity") && passed;

    passed = check(throws_invalid_argument([&] {
                       static_cast<void>(hzl::processing::cpu::box_blur(source, 2));
                   }),
                   "even box kernel was accepted") &&
             passed;
    passed = check(throws_invalid_argument([&] {
                       static_cast<void>(
                           hzl::processing::cpu::gamma_correction(source, 0.0));
                   }),
                   "zero gamma was accepted") &&
             passed;
    passed = check(throws_invalid_argument([] {
                       static_cast<void>(
                           hzl::processing::cpu::invert(cv::Mat{}));
                   }),
                   "empty input was accepted") &&
             passed;
    passed = check(throws_invalid_argument([&] {
                       static_cast<void>(hzl::processing::cpu::color_grade(
                           source, -1.0, 0.0, 0.0, 1.0, 1.0, 1.0));
                   }),
                   "negative saturation was accepted") && passed;

    if (passed) {
        std::cout << "CPU reference filter tests: passed\n";
        return 0;
    }
    return 1;
}
