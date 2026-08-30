#include "hzl/processing/cuda_convolution_filters.hpp"
#include "hzl/processing/cuda_edge_filters.hpp"
#include "hzl/processing/cuda_filter_common.hpp"
#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

template <typename Operation>
float benchmark(const int iterations, Operation operation) {
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
        float elapsed_ms = 0.0F;
        hzl::processing::cuda::check(
            cudaEventElapsedTime(&elapsed_ms, start, stop),
            "cudaEventElapsedTime");
        static_cast<void>(cudaEventDestroy(stop));
        static_cast<void>(cudaEventDestroy(start));
        return elapsed_ms / static_cast<float>(iterations);
    } catch (...) {
        if (stop != nullptr) {
            static_cast<void>(cudaEventDestroy(stop));
        }
        static_cast<void>(cudaEventDestroy(start));
        throw;
    }
}

void print_result(const std::string& name,
                  const float baseline_ms,
                  const float optimized_ms) {
    std::cout << std::left << std::setw(16) << name << std::right << std::fixed
              << std::setprecision(3) << " global " << baseline_ms
              << " ms | shared " << optimized_ms << " ms | ratio "
              << baseline_ms / optimized_ms << "x\n";
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "CUDA optimization benchmark skipped: "
                  << (status == cudaSuccess ? "no CUDA device"
                                            : cudaGetErrorString(status))
                  << '\n';
        return 77;
    }

    constexpr std::size_t width = 1280;
    constexpr std::size_t height = 720;
    constexpr int iterations = 8;
    std::vector<unsigned char> host_pixels(width * height * 4U);
    for (std::size_t index = 0; index < host_pixels.size(); ++index) {
        host_pixels[index] = static_cast<unsigned char>((index * 37U) % 251U);
    }

    hzl::processing::cuda::ImageBuffer input{width, height};
    hzl::processing::cuda::ImageBuffer output;
    input.upload(host_pixels.data(), width * 4U);

    const float gaussian_global = benchmark(iterations, [&] {
        hzl::processing::cuda::gaussian_blur(
            input,
            output,
            9,
            2.0F,
            hzl::processing::cuda::KernelImplementation::global_memory);
    });
    const float gaussian_shared = benchmark(iterations, [&] {
        hzl::processing::cuda::gaussian_blur(
            input,
            output,
            9,
            2.0F,
            hzl::processing::cuda::KernelImplementation::shared_memory);
    });
    const float sobel_global = benchmark(iterations, [&] {
        hzl::processing::cuda::sobel_edges(
            input,
            output,
            1.0F,
            hzl::processing::cuda::KernelImplementation::global_memory);
    });
    const float sobel_shared = benchmark(iterations, [&] {
        hzl::processing::cuda::sobel_edges(
            input,
            output,
            1.0F,
            hzl::processing::cuda::KernelImplementation::shared_memory);
    });

    std::cout << "CUDA optimization benchmark: 1280x720 RGBA8, " << iterations
              << " timed iterations\n";
    print_result("Gaussian 9x9", gaussian_global, gaussian_shared);
    print_result("Sobel 3x3", sobel_global, sobel_shared);
    std::cout << "CUDA optimization benchmark: completed\n";
    return 0;
}
