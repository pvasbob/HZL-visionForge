#include "hzl/processing/cpu_filters.hpp"
#include "hzl/processing/cuda_basic_filters.hpp"
#include "hzl/processing/cuda_convolution_filters.hpp"
#include "hzl/processing/cuda_filter_common.hpp"
#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename Operation>
double wall_time_ms(const int iterations, Operation operation) {
    operation();
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        operation();
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() /
           static_cast<double>(iterations);
}

template <typename Operation>
float cuda_time_ms(const int iterations, Operation operation) {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    hzl::processing::cuda::check(cudaEventCreate(&start), "cudaEventCreate start");
    try {
        hzl::processing::cuda::check(cudaEventCreate(&stop), "cudaEventCreate stop");
        operation();
        hzl::processing::cuda::check(cudaEventRecord(start), "cudaEventRecord start");
        for (int iteration = 0; iteration < iterations; ++iteration) {
            operation();
        }
        hzl::processing::cuda::check(cudaEventRecord(stop), "cudaEventRecord stop");
        hzl::processing::cuda::check(cudaEventSynchronize(stop),
                                     "cudaEventSynchronize stop");
        float elapsed = 0.0F;
        hzl::processing::cuda::check(cudaEventElapsedTime(&elapsed, start, stop),
                                     "cudaEventElapsedTime");
        static_cast<void>(cudaEventDestroy(stop));
        static_cast<void>(cudaEventDestroy(start));
        return elapsed / static_cast<float>(iterations);
    } catch (...) {
        if (stop != nullptr) {
            static_cast<void>(cudaEventDestroy(stop));
        }
        static_cast<void>(cudaEventDestroy(start));
        throw;
    }
}

void print_filter(const std::string& name,
                  const double cpu_ms,
                  const float gpu_ms,
                  const double megapixels) {
    std::cout << std::left << std::setw(14) << name << std::right << std::fixed
              << std::setprecision(3) << std::setw(11) << cpu_ms << std::setw(11)
              << gpu_ms << std::setw(11) << cpu_ms / gpu_ms << std::setw(14)
              << megapixels / (static_cast<double>(gpu_ms) / 1000.0) << '\n';
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "GPU benchmark skipped: "
                  << (status == cudaSuccess ? "no CUDA device"
                                            : cudaGetErrorString(status))
                  << '\n';
        return 77;
    }

    cudaDeviceProp properties{};
    hzl::processing::cuda::check(cudaGetDeviceProperties(&properties, 0),
                                 "cudaGetDeviceProperties");
    constexpr int width = 1920;
    constexpr int height = 1080;
    constexpr int iterations = 5;
    constexpr double megapixels =
        static_cast<double>(width) * static_cast<double>(height) / 1'000'000.0;

    cv::Mat source(height, width, CV_8UC4);
    for (int row = 0; row < source.rows; ++row) {
        for (int column = 0; column < source.cols; ++column) {
            source.at<cv::Vec4b>(row, column) = cv::Vec4b{
                static_cast<unsigned char>((row * 17 + column * 7) % 256),
                static_cast<unsigned char>((row * 11 + column * 13) % 256),
                static_cast<unsigned char>((row * 5 + column * 19) % 256),
                255U};
        }
    }

    hzl::processing::cuda::ImageBuffer input{width, height};
    hzl::processing::cuda::ImageBuffer output{width, height};
    std::vector<unsigned char> download_buffer(
        static_cast<std::size_t>(width) * height * 4U);
    input.upload(source.data, source.step);

    cv::Mat cpu_result;
    const double grayscale_cpu = wall_time_ms(iterations, [&] {
        cpu_result = hzl::processing::cpu::grayscale(source);
    });
    const float grayscale_gpu = cuda_time_ms(iterations, [&] {
        hzl::processing::cuda::grayscale(input, output);
    });
    const double gaussian_cpu = wall_time_ms(iterations, [&] {
        cpu_result = hzl::processing::cpu::gaussian_blur(source, 9, 2.0);
    });
    const float gaussian_gpu = cuda_time_ms(iterations, [&] {
        hzl::processing::cuda::gaussian_blur(
            input,
            output,
            9,
            2.0F,
            hzl::processing::cuda::KernelImplementation::shared_memory);
    });
    const double upload_ms = wall_time_ms(iterations, [&] {
        input.upload(source.data, source.step);
    });
    const double download_ms = wall_time_ms(iterations, [&] {
        output.download(download_buffer.data(),
                        static_cast<std::size_t>(width) * 4U);
    });

    std::cout << "GPU benchmark metadata\n"
              << "  Device: " << properties.name << '\n'
              << "  Compute capability: " << properties.major << '.'
              << properties.minor << '\n'
              << "  Input: " << width << 'x' << height << " RGBA8\n"
              << "  Build: "
#ifdef NDEBUG
              << "Release\n"
#else
              << "Debug\n"
#endif
              << "  Iterations: " << iterations << " after one warm-up\n\n"
              << std::left << std::setw(14) << "Filter" << std::right
              << std::setw(11) << "CPU ms" << std::setw(11) << "GPU ms"
              << std::setw(11) << "Speedup" << std::setw(14) << "GPU MPix/s"
              << '\n';
    print_filter("Grayscale", grayscale_cpu, grayscale_gpu, megapixels);
    print_filter("Gaussian 9x9", gaussian_cpu, gaussian_gpu, megapixels);
    std::cout << "\nTransfers (pageable host memory)\n"
              << "  Host to device: " << upload_ms << " ms\n"
              << "  Device to host: " << download_ms << " ms\n"
              << "GPU benchmark: completed\n";
    return 0;
}
