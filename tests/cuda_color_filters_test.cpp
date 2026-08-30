#include "hzl/processing/cpu_filters.hpp"
#include "hzl/processing/cuda_color_filters.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

namespace {

int maximum_difference(const cv::Mat& first, const cv::Mat& second) {
    int maximum = 0;
    for (int row = 0; row < first.rows; ++row) {
        for (int column = 0; column < first.cols; ++column) {
            const cv::Vec4b a = first.at<cv::Vec4b>(row, column);
            const cv::Vec4b b = second.at<cv::Vec4b>(row, column);
            for (int channel = 0; channel < 4; ++channel) {
                maximum = std::max(maximum, std::abs(
                    static_cast<int>(a[channel]) - static_cast<int>(b[channel])));
            }
        }
    }
    return maximum;
}

template <typename CpuOperation, typename GpuOperation>
bool compare(const cv::Mat& source, CpuOperation cpu_operation,
             GpuOperation gpu_operation, const int tolerance) {
    hzl::processing::cuda::ImageBuffer input{
        static_cast<std::size_t>(source.cols), static_cast<std::size_t>(source.rows)};
    input.upload(source.data, source.step);
    hzl::processing::cuda::ImageBuffer output;
    gpu_operation(input, output);
    cv::Mat actual(source.size(), CV_8UC4);
    output.download(actual.data, actual.step);
    return maximum_difference(actual, cpu_operation(source)) <= tolerance;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "CUDA color filter tests skipped: CUDA device unavailable\n";
        return 77;
    }

    cv::Mat source(37, 53, CV_8UC4);
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            source.at<cv::Vec4b>(row, column) = cv::Vec4b{
                static_cast<unsigned char>((row * 17 + column * 7) % 256),
                static_cast<unsigned char>((row * 3 + column * 19) % 256),
                static_cast<unsigned char>((row * 23 + column * 11) % 256),
                static_cast<unsigned char>((row + column * 5) % 256)};
        }
    }

    hzl::processing::cuda::ImageBuffer input{53, 37};
    input.upload(source.data, source.step);
    bool passed = hzl::processing::cuda::luminance_histogram(input) ==
                  hzl::processing::cpu::luminance_histogram(source);
    passed = compare(source, hzl::processing::cpu::histogram_equalization,
                     [](const auto& in, auto& out) {
                         hzl::processing::cuda::histogram_equalization(in, out);
                     }, 1) && passed;
    passed = compare(source,
                     [](const cv::Mat& image) {
                         return hzl::processing::cpu::tone_map(image, 1.25);
                     },
                     [](const auto& in, auto& out) {
                         hzl::processing::cuda::tone_map(in, out, 1.25F);
                     }, 1) && passed;
    passed = compare(source,
                     [](const cv::Mat& image) {
                         return hzl::processing::cpu::color_grade(
                             image, 1.3, 0.2, -0.15, 1.1, 0.9, 1.05);
                     },
                     [](const auto& in, auto& out) {
                         hzl::processing::cuda::color_grade(
                             in, out, 1.3F, 0.2F, -0.15F, 1.1F, 0.9F, 1.05F);
                     }, 1) && passed;

    if (passed) {
        std::cout << "CUDA color filter correctness tests: passed\n";
        return 0;
    }
    std::cerr << "FAIL: CUDA color output differs from CPU reference\n";
    return 1;
}
