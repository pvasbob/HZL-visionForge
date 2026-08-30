#pragma once

#include "hzl/media/image_loader.hpp"
#include "hzl/media/image_export.hpp"
#include "hzl/media/video_source.hpp"
#include "hzl/processing/cuda_comparison.hpp"
#include "hzl/rendering/image_texture.hpp"
#include "hzl/processing/cuda_memory.hpp"
#include "hzl/processing/gpu_pipeline.hpp"
#include "hzl/rendering/cuda_gl_interop.hpp"

#include <array>
#include <chrono>
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
    void accept_frame(cv::Mat rgba_pixels, const std::string& label);
    void update_gpu_outputs(bool update_original);

    GLFWwindow* window_{nullptr};
    bool glfw_backend_initialized_{false};
    bool opengl_backend_initialized_{false};
    bool context_created_{false};
    bool open_image_dialog_{false};
    bool export_image_dialog_{false};
    bool open_video_dialog_{false};
    bool open_camera_dialog_{false};
    std::array<char, 1024> image_path_{};
    std::array<char, 1024> export_path_{};
    std::array<char, 1024> video_path_{};
    int camera_index_{0};
    media::ImageExportOptions export_options_;
    std::optional<media::ImageDocument> image_;
    media::VideoSource video_source_;
    std::chrono::steady_clock::time_point next_frame_time_{};
    rendering::ImageTexture image_texture_;
    processing::cuda::ImageBuffer cuda_image_;
    processing::cuda::ImageBuffer difference_image_;
    processing::GpuPipeline pipeline_;
    const processing::cuda::ImageBuffer* processed_image_{nullptr};
    rendering::CudaGlDoubleBuffer original_presentation_;
    rendering::CudaGlDoubleBuffer processed_presentation_;
    rendering::CudaGlDoubleBuffer difference_presentation_;
    bool using_cuda_presentation_{false};
    bool pipeline_dirty_{false};
    int comparison_mode_{1};
    float comparison_split_{0.5F};
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
