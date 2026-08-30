#pragma once

#include <iosfwd>

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
};

[[nodiscard]] const char* imgui_version() noexcept;
[[nodiscard]] bool report_imgui_environment(std::ostream& output,
                                            std::ostream& errors);

}  // namespace hzl::ui
