#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace hzl::processing::cuda {
namespace {

std::string error_message(const cudaError_t error,
                          const std::string_view operation) {
    return std::string{operation} + " failed: " + cudaGetErrorString(error);
}

void validate_transfer(const void* pointer,
                       const std::size_t transfer_size,
                       const std::size_t available_size,
                       const char* role) {
    if (transfer_size > available_size) {
        throw std::out_of_range{std::string{role} +
                                " transfer exceeds the logical buffer size."};
    }
    if (transfer_size > 0 && pointer == nullptr) {
        throw std::invalid_argument{std::string{role} +
                                    " transfer pointer cannot be null."};
    }
}

}  // namespace

CudaException::CudaException(const cudaError_t error,
                             const std::string_view operation)
    : std::runtime_error{error_message(error, operation)}, error_{error} {}

void check(const cudaError_t result, const std::string_view operation) {
    if (result != cudaSuccess) {
        throw CudaException{result, operation};
    }
}

std::size_t checked_rgba8_row_bytes(const std::size_t width) {
    constexpr std::size_t rgba8_pixel_bytes = 4;
    if (width > std::numeric_limits<std::size_t>::max() / rgba8_pixel_bytes) {
        throw std::overflow_error{"RGBA8 row byte count overflowed size_t."};
    }
    return width * rgba8_pixel_bytes;
}

std::size_t checked_image_bytes(const std::size_t row_bytes,
                                const std::size_t height) {
    if (height > 0 && row_bytes > std::numeric_limits<std::size_t>::max() / height) {
        throw std::overflow_error{"Image byte count overflowed size_t."};
    }
    return row_bytes * height;
}

DeviceBuffer::DeviceBuffer(const std::size_t size_bytes) {
    resize(size_bytes);
}

DeviceBuffer::~DeviceBuffer() {
    clear();
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : data_{std::exchange(other.data_, nullptr)},
      size_bytes_{std::exchange(other.size_bytes_, 0)},
      capacity_bytes_{std::exchange(other.capacity_bytes_, 0)} {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        clear();
        data_ = std::exchange(other.data_, nullptr);
        size_bytes_ = std::exchange(other.size_bytes_, 0);
        capacity_bytes_ = std::exchange(other.capacity_bytes_, 0);
    }
    return *this;
}

void DeviceBuffer::resize(const std::size_t size_bytes) {
    if (size_bytes <= capacity_bytes_) {
        size_bytes_ = size_bytes;
        return;
    }

    void* new_data = nullptr;
    check(cudaMalloc(&new_data, size_bytes), "cudaMalloc");
    clear();
    data_ = new_data;
    size_bytes_ = size_bytes;
    capacity_bytes_ = size_bytes;
}

void DeviceBuffer::upload(const void* source, const std::size_t size_bytes) {
    validate_transfer(source, size_bytes, size_bytes, "Host-to-device");
    resize(size_bytes);
    if (size_bytes > 0) {
        check(cudaMemcpy(data_, source, size_bytes, cudaMemcpyHostToDevice),
              "cudaMemcpy host-to-device");
    }
}

void DeviceBuffer::download(void* destination,
                            const std::size_t size_bytes) const {
    validate_transfer(destination, size_bytes, size_bytes_, "Device-to-host");
    if (size_bytes > 0) {
        check(cudaMemcpy(destination, data_, size_bytes, cudaMemcpyDeviceToHost),
              "cudaMemcpy device-to-host");
    }
}

void DeviceBuffer::clear() noexcept {
    if (data_ != nullptr) {
        static_cast<void>(cudaFree(data_));
    }
    data_ = nullptr;
    size_bytes_ = 0;
    capacity_bytes_ = 0;
}

ImageBuffer::ImageBuffer(const std::size_t width, const std::size_t height) {
    resize(width, height);
}

ImageBuffer::~ImageBuffer() {
    clear();
}

ImageBuffer::ImageBuffer(ImageBuffer&& other) noexcept
    : data_{std::exchange(other.data_, nullptr)},
      width_{std::exchange(other.width_, 0)},
      height_{std::exchange(other.height_, 0)},
      row_bytes_{std::exchange(other.row_bytes_, 0)},
      pitch_bytes_{std::exchange(other.pitch_bytes_, 0)},
      capacity_height_{std::exchange(other.capacity_height_, 0)} {}

ImageBuffer& ImageBuffer::operator=(ImageBuffer&& other) noexcept {
    if (this != &other) {
        clear();
        data_ = std::exchange(other.data_, nullptr);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
        row_bytes_ = std::exchange(other.row_bytes_, 0);
        pitch_bytes_ = std::exchange(other.pitch_bytes_, 0);
        capacity_height_ = std::exchange(other.capacity_height_, 0);
    }
    return *this;
}

void ImageBuffer::resize(const std::size_t width, const std::size_t height) {
    if (width == 0 || height == 0) {
        width_ = 0;
        height_ = 0;
        row_bytes_ = 0;
        return;
    }

    const std::size_t new_row_bytes = checked_rgba8_row_bytes(width);
    static_cast<void>(checked_image_bytes(new_row_bytes, height));
    if (new_row_bytes <= pitch_bytes_ && height <= capacity_height_) {
        width_ = width;
        height_ = height;
        row_bytes_ = new_row_bytes;
        return;
    }

    void* new_data = nullptr;
    std::size_t new_pitch = 0;
    check(cudaMallocPitch(&new_data, &new_pitch, new_row_bytes, height),
          "cudaMallocPitch");
    clear();
    data_ = new_data;
    width_ = width;
    height_ = height;
    row_bytes_ = new_row_bytes;
    pitch_bytes_ = new_pitch;
    capacity_height_ = height;
}

void ImageBuffer::upload(const void* source,
                         const std::size_t source_stride_bytes) {
    if (empty()) {
        throw std::logic_error{"Cannot upload into an empty CUDA image buffer."};
    }
    if (source == nullptr) {
        throw std::invalid_argument{"Image upload source cannot be null."};
    }
    if (source_stride_bytes < row_bytes_) {
        throw std::invalid_argument{"Image upload stride is smaller than one row."};
    }
    check(cudaMemcpy2D(data_,
                       pitch_bytes_,
                       source,
                       source_stride_bytes,
                       row_bytes_,
                       height_,
                       cudaMemcpyHostToDevice),
          "cudaMemcpy2D host-to-device");
}

void ImageBuffer::download(void* destination,
                           const std::size_t destination_stride_bytes) const {
    if (empty()) {
        throw std::logic_error{"Cannot download from an empty CUDA image buffer."};
    }
    if (destination == nullptr) {
        throw std::invalid_argument{"Image download destination cannot be null."};
    }
    if (destination_stride_bytes < row_bytes_) {
        throw std::invalid_argument{
            "Image download stride is smaller than one row."};
    }
    check(cudaMemcpy2D(destination,
                       destination_stride_bytes,
                       data_,
                       pitch_bytes_,
                       row_bytes_,
                       height_,
                       cudaMemcpyDeviceToHost),
          "cudaMemcpy2D device-to-host");
}

void ImageBuffer::clear() noexcept {
    if (data_ != nullptr) {
        static_cast<void>(cudaFree(data_));
    }
    data_ = nullptr;
    width_ = 0;
    height_ = 0;
    row_bytes_ = 0;
    pitch_bytes_ = 0;
    capacity_height_ = 0;
}

}  // namespace hzl::processing::cuda
