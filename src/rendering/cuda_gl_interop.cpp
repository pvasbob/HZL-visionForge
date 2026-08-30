#include "hzl/rendering/cuda_gl_interop.hpp"

#include "hzl/processing/cuda_memory.hpp"

#include <glad/gl.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <limits>
#include <stdexcept>
#include <utility>

namespace hzl::rendering {

CudaGlTexture::~CudaGlTexture() {
    clear();
}

CudaGlTexture::CudaGlTexture(CudaGlTexture&& other) noexcept
    : texture_id_{std::exchange(other.texture_id_, 0)},
      cuda_resource_{std::exchange(other.cuda_resource_, nullptr)},
      width_{std::exchange(other.width_, 0)},
      height_{std::exchange(other.height_, 0)} {}

CudaGlTexture& CudaGlTexture::operator=(CudaGlTexture&& other) noexcept {
    if (this != &other) {
        clear();
        texture_id_ = std::exchange(other.texture_id_, 0);
        cuda_resource_ = std::exchange(other.cuda_resource_, nullptr);
        width_ = std::exchange(other.width_, 0);
        height_ = std::exchange(other.height_, 0);
    }
    return *this;
}

void CudaGlTexture::resize(const std::size_t width, const std::size_t height) {
    if (width == 0 || height == 0) {
        throw std::invalid_argument{
            "CUDA-OpenGL texture dimensions must be greater than zero."};
    }
    if (width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        height > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::overflow_error{"CUDA-OpenGL texture dimensions exceed OpenGL limits."};
    }
    if (valid() && width == width_ && height == height_) {
        return;
    }

    while (glGetError() != GL_NO_ERROR) {
    }

    GLuint new_texture = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &new_texture);
    glTextureParameteri(new_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(new_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(new_texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(new_texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureStorage2D(new_texture,
                       1,
                       GL_RGBA8,
                       static_cast<GLsizei>(width),
                       static_cast<GLsizei>(height));
    const GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR) {
        glDeleteTextures(1, &new_texture);
        throw std::runtime_error{"OpenGL texture allocation failed for CUDA interop."};
    }

    cudaGraphicsResource* new_resource = nullptr;
    const cudaError_t register_result = cudaGraphicsGLRegisterImage(
        &new_resource,
        new_texture,
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard);
    if (register_result != cudaSuccess) {
        glDeleteTextures(1, &new_texture);
        processing::cuda::check(register_result,
                                "cudaGraphicsGLRegisterImage");
    }

    clear();
    texture_id_ = new_texture;
    cuda_resource_ = new_resource;
    width_ = width;
    height_ = height;
}

void CudaGlTexture::copy_from(
    const processing::cuda::ImageBuffer& source,
    const cudaStream_t stream) {
    if (!valid()) {
        throw std::logic_error{"CUDA-OpenGL texture is not initialized."};
    }
    if (source.empty() || source.width() != width_ || source.height() != height_) {
        throw std::invalid_argument{
            "CUDA source dimensions do not match the shared OpenGL texture."};
    }

    processing::cuda::check(
        cudaGraphicsMapResources(1, &cuda_resource_, stream),
        "cudaGraphicsMapResources");

    cudaArray_t mapped_array = nullptr;
    cudaError_t result = cudaGraphicsSubResourceGetMappedArray(
        &mapped_array, cuda_resource_, 0, 0);
    if (result == cudaSuccess) {
        result = cudaMemcpy2DToArrayAsync(mapped_array,
                                          0,
                                          0,
                                          source.data(),
                                          source.stride_bytes(),
                                          source.row_bytes(),
                                          source.height(),
                                          cudaMemcpyDeviceToDevice,
                                          stream);
    }
    const cudaError_t unmap_result =
        cudaGraphicsUnmapResources(1, &cuda_resource_, stream);
    processing::cuda::check(result, "CUDA copy to shared OpenGL texture");
    processing::cuda::check(unmap_result, "cudaGraphicsUnmapResources");
    processing::cuda::check(cudaStreamSynchronize(stream),
                            "CUDA-OpenGL presentation synchronization");
}

void CudaGlTexture::clear() noexcept {
    if (cuda_resource_ != nullptr) {
        static_cast<void>(cudaGraphicsUnregisterResource(cuda_resource_));
        cuda_resource_ = nullptr;
    }
    if (texture_id_ != 0U) {
        glDeleteTextures(1, &texture_id_);
        texture_id_ = 0;
    }
    width_ = 0;
    height_ = 0;
}

void CudaGlDoubleBuffer::resize(const std::size_t width,
                                const std::size_t height) {
    if (width == width_ && height == height_ && textures_[0].valid() &&
        textures_[1].valid()) {
        return;
    }

    CudaGlTexture first;
    CudaGlTexture second;
    first.resize(width, height);
    second.resize(width, height);
    textures_[0] = std::move(first);
    textures_[1] = std::move(second);
    front_index_ = 0;
    width_ = width;
    height_ = height;
    back_ready_ = false;
}

void CudaGlDoubleBuffer::upload_back(
    const processing::cuda::ImageBuffer& source,
    const cudaStream_t stream) {
    if (!textures_[0].valid() || !textures_[1].valid()) {
        throw std::logic_error{"CUDA-OpenGL double buffer is not initialized."};
    }
    textures_[1U - front_index_].copy_from(source, stream);
    back_ready_ = true;
}

void CudaGlDoubleBuffer::swap() {
    if (!back_ready_) {
        throw std::logic_error{"No completed CUDA-OpenGL back buffer is ready."};
    }
    front_index_ = 1U - front_index_;
    back_ready_ = false;
}

void CudaGlDoubleBuffer::clear() noexcept {
    textures_[0].clear();
    textures_[1].clear();
    front_index_ = 0;
    width_ = 0;
    height_ = 0;
    back_ready_ = false;
}

unsigned int CudaGlDoubleBuffer::front_texture_id() const noexcept {
    return textures_[front_index_].texture_id();
}

unsigned int CudaGlDoubleBuffer::back_texture_id() const noexcept {
    return textures_[1U - front_index_].texture_id();
}

}  // namespace hzl::rendering
