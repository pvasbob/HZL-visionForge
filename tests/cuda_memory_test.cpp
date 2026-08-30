#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

bool check(const bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
    }
    return condition;
}

template <typename Exception, typename Operation>
bool throws(Operation operation) {
    try {
        operation();
    } catch (const Exception&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

}  // namespace

int main() {
    using hzl::processing::cuda::DeviceBuffer;
    using hzl::processing::cuda::ImageBuffer;

    static_assert(!std::is_copy_constructible_v<DeviceBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<DeviceBuffer>);
    static_assert(!std::is_copy_constructible_v<ImageBuffer>);
    static_assert(std::is_nothrow_move_constructible_v<ImageBuffer>);

    bool passed = throws<std::overflow_error>([] {
        static_cast<void>(hzl::processing::cuda::checked_rgba8_row_bytes(
            std::numeric_limits<std::size_t>::max()));
    });
    passed = check(passed, "RGBA8 row overflow was not rejected") && passed;
    passed = check(throws<std::overflow_error>([] {
                       static_cast<void>(
                           hzl::processing::cuda::checked_image_bytes(
                               std::numeric_limits<std::size_t>::max(), 2));
                   }),
                   "image size overflow was not rejected") &&
             passed;

    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    if (device_status != cudaSuccess || device_count == 0) {
        static_cast<void>(cudaGetLastError());
        std::cout << "CUDA memory tests skipped: "
                  << (device_status == cudaSuccess
                          ? "no CUDA device"
                          : cudaGetErrorString(device_status))
                  << '\n';
        return passed ? 77 : 1;
    }

    std::vector<unsigned char> source(4096);
    for (std::size_t index = 0; index < source.size(); ++index) {
        source[index] = static_cast<unsigned char>(index % 251U);
    }
    std::vector<unsigned char> destination(source.size(), 0);

    DeviceBuffer bytes;
    bytes.upload(source.data(), source.size());
    void* original_pointer = bytes.data();
    bytes.download(destination.data(), destination.size());
    passed = check(destination == source, "linear buffer round trip changed data") &&
             passed;
    bytes.resize(source.size() / 2U);
    passed = check(bytes.data() == original_pointer,
                   "linear buffer did not reuse sufficient capacity") &&
             passed;
    DeviceBuffer moved_bytes{std::move(bytes)};
    passed = check(bytes.data() == nullptr && moved_bytes.data() == original_pointer,
                   "linear buffer move did not transfer ownership") &&
             passed;

    constexpr std::size_t width = 37;
    constexpr std::size_t height = 19;
    constexpr std::size_t row_bytes = width * 4U;
    constexpr std::size_t host_stride = row_bytes + 12U;
    std::vector<unsigned char> image_source(host_stride * height, 0);
    for (std::size_t row = 0; row < height; ++row) {
        for (std::size_t column = 0; column < row_bytes; ++column) {
            image_source[row * host_stride + column] =
                static_cast<unsigned char>((row * 17U + column) % 253U);
        }
    }
    std::vector<unsigned char> image_destination(host_stride * height, 0);

    ImageBuffer image{width, height};
    image.upload(image_source.data(), host_stride);
    void* original_image_pointer = image.data();
    image.download(image_destination.data(), host_stride);
    for (std::size_t row = 0; row < height; ++row) {
        passed = check(std::equal(image_source.begin() +
                                      static_cast<std::ptrdiff_t>(row * host_stride),
                                  image_source.begin() +
                                      static_cast<std::ptrdiff_t>(row * host_stride +
                                                                  row_bytes),
                                  image_destination.begin() +
                                      static_cast<std::ptrdiff_t>(row * host_stride)),
                       "pitched image round trip changed row " +
                           std::to_string(row)) &&
                 passed;
    }
    image.resize(width / 2U, height / 2U);
    passed = check(image.data() == original_image_pointer,
                   "pitched image did not reuse sufficient capacity") &&
             passed;
    passed = check(image.stride_bytes() >= image.row_bytes(),
                   "CUDA image pitch is smaller than its logical row") &&
             passed;

    if (passed) {
        std::cout << "CUDA memory tests: passed\n";
        return 0;
    }
    return 1;
}
