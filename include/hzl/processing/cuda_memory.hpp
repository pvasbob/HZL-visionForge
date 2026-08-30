#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace hzl::processing::cuda {

class CudaException : public std::runtime_error {
public:
    CudaException(cudaError_t error, std::string_view operation);

    [[nodiscard]] cudaError_t error() const noexcept { return error_; }

private:
    cudaError_t error_;
};

void check(cudaError_t result, std::string_view operation);

[[nodiscard]] std::size_t checked_rgba8_row_bytes(std::size_t width);
[[nodiscard]] std::size_t checked_image_bytes(std::size_t row_bytes,
                                              std::size_t height);

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t size_bytes);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void resize(std::size_t size_bytes);
    void upload(const void* source, std::size_t size_bytes);
    void download(void* destination, std::size_t size_bytes) const;
    void clear() noexcept;

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size_bytes() const noexcept { return size_bytes_; }
    [[nodiscard]] std::size_t capacity_bytes() const noexcept {
        return capacity_bytes_;
    }
    [[nodiscard]] bool empty() const noexcept { return size_bytes_ == 0; }

private:
    void* data_{nullptr};
    std::size_t size_bytes_{0};
    std::size_t capacity_bytes_{0};
};

class ImageBuffer {
public:
    ImageBuffer() = default;
    ImageBuffer(std::size_t width, std::size_t height);
    ~ImageBuffer();

    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator=(const ImageBuffer&) = delete;
    ImageBuffer(ImageBuffer&& other) noexcept;
    ImageBuffer& operator=(ImageBuffer&& other) noexcept;

    void resize(std::size_t width, std::size_t height);
    void upload(const void* source, std::size_t source_stride_bytes);
    void download(void* destination, std::size_t destination_stride_bytes) const;
    void clear() noexcept;

    [[nodiscard]] void* data() noexcept { return data_; }
    [[nodiscard]] const void* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t width() const noexcept { return width_; }
    [[nodiscard]] std::size_t height() const noexcept { return height_; }
    [[nodiscard]] std::size_t row_bytes() const noexcept { return row_bytes_; }
    [[nodiscard]] std::size_t stride_bytes() const noexcept { return pitch_bytes_; }
    [[nodiscard]] std::size_t capacity_height() const noexcept {
        return capacity_height_;
    }
    [[nodiscard]] bool empty() const noexcept {
        return width_ == 0 || height_ == 0;
    }

private:
    void* data_{nullptr};
    std::size_t width_{0};
    std::size_t height_{0};
    std::size_t row_bytes_{0};
    std::size_t pitch_bytes_{0};
    std::size_t capacity_height_{0};
};

}  // namespace hzl::processing::cuda
