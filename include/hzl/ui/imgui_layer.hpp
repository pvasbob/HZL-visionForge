#pragma once

#include "hzl/media/image_loader.hpp"
#include "hzl/media/image_export.hpp"
#include "hzl/rendering/image_texture.hpp"
#include "hzl/processing/cuda_memory.hpp"
#include "hzl/rendering/cuda_gl_interop.hpp"

#include <array>
#include <iosfwd>
#include <optional>
#include <string>

struct GLFWwindow;

namespace hzl::ui {

class ImGuiLayer {
public:
    ImGuiLayer() = default;
    ~ImGuiLayer();

    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    [[nodiscard]] bool initialize(GLFWwindow* window, std::ostream& errors);
    void begin_frame();
    void draw_application_shell();
    void render();

private:
    GLFWwindow* window_{nullptr};
    bool glfw_backend_initialized_{false};
    bool opengl_backend_initialized_{false};
    bool context_created_{false};
    bool open_image_dialog_{false};
    bool export_image_dialog_{false};
    std::array<char, 1024> image_path_{};
    std::array<char, 1024> export_path_{};
    media::ImageExportOptions export_options_;
    std::optional<media::ImageDocument> image_;
    rendering::ImageTexture image_texture_;
    processing::cuda::ImageBuffer cuda_image_;
    rendering::CudaGlDoubleBuffer cuda_presentation_;
    bool using_cuda_presentation_{false};
    float zoom_{1.0F};
    float pan_x_{0.0F};
    float pan_y_{0.0F};
    bool fit_to_window_{true};
    std::string status_message_{"Ready | No media loaded"};
};

[[nodiscard]] const char* imgui_version() noexcept;
[[nodiscard]] bool report_imgui_environment(std::ostream& output,
                                            std::ostream& errors);

}  // namespace hzl::ui
