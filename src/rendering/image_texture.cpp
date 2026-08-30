#include "hzl/rendering/image_texture.hpp"

#include <glad/gl.h>

#include <string>

namespace hzl::rendering {

ImageTexture::~ImageTexture() {
    reset();
}

bool ImageTexture::upload_rgba8(const cv::Mat& pixels,
                                std::string& error_message) {
    if (pixels.empty() || pixels.type() != CV_8UC4 || !pixels.isContinuous()) {
        error_message =
            "OpenGL upload requires a non-empty, contiguous RGBA8 image.";
        return false;
    }

    while (glGetError() != GL_NO_ERROR) {
    }

    if (id_ != 0U && width_ == pixels.cols && height_ == pixels.rows) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTextureSubImage2D(id_,
                            0,
                            0,
                            0,
                            pixels.cols,
                            pixels.rows,
                            GL_RGBA,
                            GL_UNSIGNED_BYTE,
                            pixels.data);
        const GLenum update_error = glGetError();
        if (update_error != GL_NO_ERROR) {
            error_message = "OpenGL texture update failed with error code " +
                            std::to_string(update_error) + '.';
            return false;
        }
        error_message.clear();
        return true;
    }

    GLuint new_texture = 0;
    glCreateTextures(GL_TEXTURE_2D, 1, &new_texture);
    glTextureParameteri(new_texture, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTextureParameteri(new_texture, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTextureParameteri(new_texture, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTextureParameteri(new_texture, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTextureStorage2D(new_texture, 1, GL_RGBA8, pixels.cols, pixels.rows);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTextureSubImage2D(new_texture,
                        0,
                        0,
                        0,
                        pixels.cols,
                        pixels.rows,
                        GL_RGBA,
                        GL_UNSIGNED_BYTE,
                        pixels.data);

    const GLenum upload_error = glGetError();
    if (upload_error != GL_NO_ERROR) {
        glDeleteTextures(1, &new_texture);
        error_message = "OpenGL texture upload failed with error code " +
                        std::to_string(upload_error) + '.';
        return false;
    }

    reset();
    id_ = new_texture;
    width_ = pixels.cols;
    height_ = pixels.rows;
    error_message.clear();
    return true;
}

void ImageTexture::reset() noexcept {
    if (id_ != 0U) {
        glDeleteTextures(1, &id_);
        id_ = 0;
    }
    width_ = 0;
    height_ = 0;
}

}  // namespace hzl::rendering
