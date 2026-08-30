#pragma once

#include "hzl/processing/cuda_memory.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>

struct cudaGraphicsResource;

namespace hzl::rendering {

class CudaGlTexture {
public:
    CudaGlTexture() = default;
    ~CudaGlTexture();

    CudaGlTexture(const CudaGlTexture&) = delete;
    CudaGlTexture& operator=(const CudaGlTexture&) = delete;
    CudaGlTexture(CudaGlTexture&& other) noexcept;
    CudaGlTexture& operator=(CudaGlTexture&& other) noexcept;

    void resize(std::size_t width, std::size_t height);
    void copy_from(const processing::cuda::ImageBuffer& source,
                   cudaStream_t stream = nullptr);
    void clear() noexcept;

    [[nodiscard]] unsigned int texture_id() const noexcept { return texture_id_; }
    [[nodiscard]] std::size_t width() const noexcept { return width_; }
    [[nodiscard]] std::size_t height() const noexcept { return height_; }
    [[nodiscard]] bool valid() const noexcept {
        return texture_id_ != 0U && cuda_resource_ != nullptr;
    }

private:
    unsigned int texture_id_{0};
    cudaGraphicsResource* cuda_resource_{nullptr};
    std::size_t width_{0};
    std::size_t height_{0};
};

class CudaGlDoubleBuffer {
public:
    void resize(std::size_t width, std::size_t height);
    void upload_back(const processing::cuda::ImageBuffer& source,
                     cudaStream_t stream = nullptr);
    void swap();
    void clear() noexcept;

    [[nodiscard]] unsigned int front_texture_id() const noexcept;
    [[nodiscard]] unsigned int back_texture_id() const noexcept;
    [[nodiscard]] std::size_t width() const noexcept { return width_; }
    [[nodiscard]] std::size_t height() const noexcept { return height_; }
    [[nodiscard]] bool back_ready() const noexcept { return back_ready_; }

private:
    CudaGlTexture textures_[2];
    unsigned int front_index_{0};
    std::size_t width_{0};
    std::size_t height_{0};
    bool back_ready_{false};
};

}  // namespace hzl::rendering
