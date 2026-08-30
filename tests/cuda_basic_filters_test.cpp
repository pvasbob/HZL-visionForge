#include "hzl/processing/cpu_filters.hpp"
#include "hzl/processing/cuda_basic_filters.hpp"
#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool compare_images(const cv::Mat& expected,
                    const cv::Mat& actual,
                    const int color_tolerance,
                    const std::string& filter_name) {
    if (expected.size() != actual.size() || expected.type() != actual.type()) {
        std::cerr << "FAIL: " << filter_name << " output shape/type differs\n";
        return false;
    }

    int maximum_color_difference = 0;
    for (int row = 0; row < expected.rows; ++row) {
        for (int column = 0; column < expected.cols; ++column) {
            const cv::Vec4b expected_pixel = expected.at<cv::Vec4b>(row, column);
            const cv::Vec4b actual_pixel = actual.at<cv::Vec4b>(row, column);
            for (int channel = 0; channel < 3; ++channel) {
                maximum_color_difference = std::max(
                    maximum_color_difference,
                    std::abs(static_cast<int>(expected_pixel[channel]) -
                             static_cast<int>(actual_pixel[channel])));
            }
            if (expected_pixel[3] != actual_pixel[3]) {
                std::cerr << "FAIL: " << filter_name << " changed alpha at "
                          << column << ',' << row << '\n';
                return false;
            }
        }
    }
    if (maximum_color_difference > color_tolerance) {
        std::cerr << "FAIL: " << filter_name << " maximum color difference "
                  << maximum_color_difference << " exceeds tolerance "
                  << color_tolerance << '\n';
        return false;
    }
    return true;
}

cv::Mat download(const hzl::processing::cuda::ImageBuffer& buffer) {
    cv::Mat result(static_cast<int>(buffer.height()),
                   static_cast<int>(buffer.width()),
                   CV_8UC4);
    buffer.download(result.data, result.step);
    return result;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "CUDA basic filter tests skipped: "
                  << (device_status == cudaSuccess
                          ? "no CUDA device"
                          : cudaGetErrorString(device_status))
                  << '\n';
        return 77;
    }

    // Deliberately not divisible by the 16x16 CUDA block dimensions.
    cv::Mat source(23, 37, CV_8UC4);
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            source.at<cv::Vec4b>(row, column) = cv::Vec4b{
                static_cast<unsigned char>((row * 29 + column * 7) % 256),
                static_cast<unsigned char>((row * 11 + column * 23) % 256),
                static_cast<unsigned char>((row * 17 + column * 13) % 256),
                static_cast<unsigned char>((row * 5 + column * 3) % 256)};
        }
    }

    hzl::processing::cuda::ImageBuffer input{
        static_cast<std::size_t>(source.cols),
        static_cast<std::size_t>(source.rows)};
    input.upload(source.data, source.step);
    hzl::processing::cuda::ImageBuffer output;
    bool passed = true;

    hzl::processing::cuda::grayscale(input, output);
    void* reusable_output_pointer = output.data();
    passed = compare_images(hzl::processing::cpu::grayscale(source),
                            download(output),
                            1,
                            "grayscale") &&
             passed;

    hzl::processing::cuda::invert(input, output);
    passed = compare_images(hzl::processing::cpu::invert(source),
                            download(output),
                            0,
                            "invert") &&
             passed;

    constexpr float brightness = 17.0F;
    constexpr float contrast = 1.15F;
    hzl::processing::cuda::brightness_contrast(
        input, output, brightness, contrast);
    passed = compare_images(
                 hzl::processing::cpu::brightness_contrast(
                     source, brightness, contrast),
                 download(output),
                 1,
                 "brightness/contrast") &&
             passed;

    constexpr float gamma = 1.8F;
    hzl::processing::cuda::gamma_correction(input, output, gamma);
    passed = compare_images(
                 hzl::processing::cpu::gamma_correction(source, gamma),
                 download(output),
                 1,
                 "gamma") &&
             passed;

    passed = (output.data() == reusable_output_pointer) && passed;
    if (output.data() != reusable_output_pointer) {
        std::cerr << "FAIL: CUDA filters did not reuse the output allocation\n";
    }

    try {
        hzl::processing::cuda::gamma_correction(input, output, 0.0F);
        std::cerr << "FAIL: zero gamma was accepted\n";
        passed = false;
    } catch (const std::invalid_argument&) {
    }

    if (passed) {
        std::cout << "CUDA basic filter tests: passed\n";
        return 0;
    }
    return 1;
}
