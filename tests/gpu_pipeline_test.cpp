#include "hzl/processing/cpu_filters.hpp"
#include "hzl/processing/cuda_comparison.hpp"
#include "hzl/processing/gpu_pipeline.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

int main() {
    hzl::processing::GpuPipeline pipeline;
    const std::uint64_t grayscale =
        pipeline.add(hzl::processing::FilterType::grayscale);
    const std::uint64_t invert = pipeline.add(hzl::processing::FilterType::invert);
    const std::uint64_t adjustment =
        pipeline.add(hzl::processing::FilterType::brightness_contrast);
    bool passed = pipeline.operations().size() == 3;
    passed = pipeline.move_up(adjustment) && passed;
    passed = pipeline.operations()[1].id == adjustment && passed;
    passed = pipeline.move_down(adjustment) && passed;
    passed = pipeline.operations()[2].id == adjustment && passed;
    passed = !pipeline.move_up(grayscale) && passed;

    auto& parameters = std::get<hzl::processing::BrightnessContrastParameters>(
        pipeline.operations()[2].parameters);
    parameters.brightness = 9.0F;
    parameters.contrast = 1.1F;
    pipeline.operations()[1].enabled = false;

    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "GPU pipeline tests skipped: CUDA device unavailable\n";
        return passed ? 77 : 1;
    }

    cv::Mat source(19, 31, CV_8UC4);
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            source.at<cv::Vec4b>(row, column) = cv::Vec4b{
                static_cast<unsigned char>((row * 17 + column * 3) % 256),
                static_cast<unsigned char>((row * 7 + column * 13) % 256),
                static_cast<unsigned char>((row * 11 + column * 19) % 256),
                static_cast<unsigned char>((row + column) % 256)};
        }
    }
    hzl::processing::cuda::ImageBuffer input{31, 19};
    input.upload(source.data, source.step);
    const hzl::processing::cuda::ImageBuffer& output = pipeline.process(input);
    cv::Mat actual(19, 31, CV_8UC4);
    output.download(actual.data, actual.step);
    const cv::Mat expected = hzl::processing::cpu::brightness_contrast(
        hzl::processing::cpu::grayscale(source), 9.0, 1.1);

    int maximum_difference = 0;
    for (int row = 0; row < actual.rows; ++row) {
        for (int column = 0; column < actual.cols; ++column) {
            const cv::Vec4b a = actual.at<cv::Vec4b>(row, column);
            const cv::Vec4b e = expected.at<cv::Vec4b>(row, column);
            for (int channel = 0; channel < 4; ++channel) {
                maximum_difference = std::max(
                    maximum_difference,
                    std::abs(static_cast<int>(a[channel]) -
                             static_cast<int>(e[channel])));
            }
        }
    }
    passed = maximum_difference <= 1 && passed;

    hzl::processing::cuda::ImageBuffer difference;
    hzl::processing::cuda::absolute_difference(input, output, difference);
    cv::Mat difference_pixels(19, 31, CV_8UC4);
    difference.download(difference_pixels.data, difference_pixels.step);
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            const cv::Vec4b original = source.at<cv::Vec4b>(row, column);
            const cv::Vec4b processed = actual.at<cv::Vec4b>(row, column);
            const cv::Vec4b actual_difference =
                difference_pixels.at<cv::Vec4b>(row, column);
            for (int channel = 0; channel < 3; ++channel) {
                passed = actual_difference[channel] == static_cast<unsigned char>(
                    std::abs(static_cast<int>(original[channel]) -
                             static_cast<int>(processed[channel]))) &&
                         passed;
            }
            passed = actual_difference[3] == 255U && passed;
        }
    }
    passed = pipeline.remove(invert) && passed;
    pipeline.clear();
    passed = pipeline.operations().empty() && passed;
    passed = &pipeline.process(input) == &input && passed;

    if (passed) {
        std::cout << "GPU pipeline tests: passed\n";
        return 0;
    }
    std::cerr << "FAIL: GPU pipeline behavior or output differs\n";
    return 1;
}
