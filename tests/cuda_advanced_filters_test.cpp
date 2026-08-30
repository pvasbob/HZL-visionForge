#include "hzl/processing/cpu_filters.hpp"
#include "hzl/processing/cuda_convolution_filters.hpp"
#include "hzl/processing/cuda_edge_filters.hpp"
#include "hzl/processing/cuda_filter_common.hpp"
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

cv::Mat download(const hzl::processing::cuda::ImageBuffer& buffer) {
    cv::Mat result(static_cast<int>(buffer.height()),
                   static_cast<int>(buffer.width()),
                   CV_8UC4);
    buffer.download(result.data, result.step);
    return result;
}

bool compare_images(const cv::Mat& expected,
                    const cv::Mat& actual,
                    const int tolerance,
                    const std::string& name) {
    if (expected.size() != actual.size() || expected.type() != actual.type()) {
        std::cerr << "FAIL: " << name << " output shape/type differs\n";
        return false;
    }
    int maximum_difference = 0;
    for (int row = 0; row < expected.rows; ++row) {
        for (int column = 0; column < expected.cols; ++column) {
            const cv::Vec4b expected_pixel = expected.at<cv::Vec4b>(row, column);
            const cv::Vec4b actual_pixel = actual.at<cv::Vec4b>(row, column);
            for (int channel = 0; channel < 3; ++channel) {
                maximum_difference = std::max(
                    maximum_difference,
                    std::abs(static_cast<int>(expected_pixel[channel]) -
                             static_cast<int>(actual_pixel[channel])));
            }
            if (expected_pixel[3] != actual_pixel[3]) {
                std::cerr << "FAIL: " << name << " changed alpha at " << column
                          << ',' << row << '\n';
                return false;
            }
        }
    }
    if (maximum_difference > tolerance) {
        std::cerr << "FAIL: " << name << " maximum difference "
                  << maximum_difference << " exceeds " << tolerance << '\n';
        return false;
    }
    return true;
}

template <typename Launch>
bool validate_implementations(const cv::Mat& expected,
                              hzl::processing::cuda::ImageBuffer& output,
                              const int tolerance,
                              const std::string& name,
                              Launch launch) {
    launch(hzl::processing::cuda::KernelImplementation::global_memory);
    const cv::Mat global_result = download(output);
    bool passed = compare_images(expected,
                                 global_result,
                                 tolerance,
                                 name + " global-memory");
    void* allocation = output.data();
    launch(hzl::processing::cuda::KernelImplementation::shared_memory);
    passed = compare_images(expected,
                            download(output),
                            tolerance,
                            name + " shared-memory") &&
             passed;
    if (output.data() != allocation) {
        std::cerr << "FAIL: " << name << " replaced a reusable output buffer\n";
        passed = false;
    }
    return passed;
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "CUDA advanced filter tests skipped: "
                  << (device_status == cudaSuccess
                          ? "no CUDA device"
                          : cudaGetErrorString(device_status))
                  << '\n';
        return 77;
    }

    cv::Mat source(29, 43, CV_8UC4);
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            source.at<cv::Vec4b>(row, column) = cv::Vec4b{
                static_cast<unsigned char>((row * 31 + column * 7) % 256),
                static_cast<unsigned char>((row * 13 + column * 19) % 256),
                static_cast<unsigned char>((row * 23 + column * 11) % 256),
                static_cast<unsigned char>((row * 17 + column * 5) % 256)};
        }
    }
    hzl::processing::cuda::ImageBuffer input{
        static_cast<std::size_t>(source.cols),
        static_cast<std::size_t>(source.rows)};
    input.upload(source.data, source.step);
    hzl::processing::cuda::ImageBuffer output;
    bool passed = true;

    passed = validate_implementations(
                 hzl::processing::cpu::box_blur(source, 5),
                 output,
                 1,
                 "box blur",
                 [&](const hzl::processing::cuda::KernelImplementation implementation) {
                     hzl::processing::cuda::box_blur(
                         input, output, 5, implementation);
                 }) &&
             passed;
    passed = validate_implementations(
                 hzl::processing::cpu::gaussian_blur(source, 7, 1.4),
                 output,
                 2,
                 "Gaussian blur",
                 [&](const hzl::processing::cuda::KernelImplementation implementation) {
                     hzl::processing::cuda::gaussian_blur(
                         input, output, 7, 1.4F, implementation);
                 }) &&
             passed;
    passed = validate_implementations(
                 hzl::processing::cpu::sharpen(source, 1.1),
                 output,
                 3,
                 "sharpen",
                 [&](const hzl::processing::cuda::KernelImplementation implementation) {
                     hzl::processing::cuda::sharpen(
                         input, output, 1.1F, implementation);
                 }) &&
             passed;
    passed = validate_implementations(
                 hzl::processing::cpu::emboss(source, 0.8),
                 output,
                 1,
                 "emboss",
                 [&](const hzl::processing::cuda::KernelImplementation implementation) {
                     hzl::processing::cuda::emboss(
                         input, output, 0.8F, implementation);
                 }) &&
             passed;
    passed = validate_implementations(
                 hzl::processing::cpu::sobel_edges(source, 0.75),
                 output,
                 2,
                 "Sobel",
                 [&](const hzl::processing::cuda::KernelImplementation implementation) {
                     hzl::processing::cuda::sobel_edges(
                         input, output, 0.75F, implementation);
                 }) &&
             passed;
    passed = validate_implementations(
                 hzl::processing::cpu::laplacian_edges(source, 1.25),
                 output,
                 1,
                 "Laplacian",
                 [&](const hzl::processing::cuda::KernelImplementation implementation) {
                     hzl::processing::cuda::laplacian_edges(
                         input, output, 1.25F, implementation);
                 }) &&
             passed;

    try {
        hzl::processing::cuda::box_blur(input, output, 4);
        std::cerr << "FAIL: even convolution kernel size was accepted\n";
        passed = false;
    } catch (const std::invalid_argument&) {
    }
    try {
        hzl::processing::cuda::sobel_edges(input, output, -1.0F);
        std::cerr << "FAIL: negative edge strength was accepted\n";
        passed = false;
    } catch (const std::invalid_argument&) {
    }

    if (passed) {
        std::cout << "CUDA advanced filter tests: passed\n";
        return 0;
    }
    return 1;
}
