#include "hzl/rendering/graphics_environment.hpp"
#include "hzl/ui/imgui_layer.hpp"

#include <glad/gl.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <iostream>
#include <memory>
#include <ostream>
#include <string_view>

namespace hzl::rendering {
namespace {

constexpr int window_width = 1280;
constexpr int window_height = 720;
constexpr std::string_view window_title{"HZL-VisionForge"};

using WindowOwner = std::unique_ptr<GLFWwindow, void (*)(GLFWwindow*)>;

void glfw_error_callback(int code, const char* description);

class GlfwSession {
public:
    explicit GlfwSession(std::ostream& errors) {
        glfwSetErrorCallback(glfw_error_callback);
        initialized_ = glfwInit() == GLFW_TRUE;
        if (!initialized_) {
            errors << "Could not initialize GLFW. Check that a graphical display "
                      "is available.\n";
        }
    }

    ~GlfwSession() {
        if (initialized_) {
            glfwTerminate();
        }
    }

    GlfwSession(const GlfwSession&) = delete;
    GlfwSession& operator=(const GlfwSession&) = delete;

    [[nodiscard]] explicit operator bool() const { return initialized_; }

private:
    bool initialized_{false};
};

void glfw_error_callback(const int code, const char* description) {
    std::cerr << "GLFW error " << code << ": " << description << '\n';
}

WindowOwner create_window(const bool visible, std::ostream& errors) {
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);

    GLFWwindow* window = glfwCreateWindow(window_width,
                                          window_height,
                                          window_title.data(),
                                          nullptr,
                                          nullptr);
    if (window == nullptr) {
        errors << "Could not create an OpenGL 4.5 core context. Verify the "
                  "graphics driver and display configuration.\n";
    }
    return {window, glfwDestroyWindow};
}

bool load_opengl(std::ostream& errors) {
    const int version = gladLoadGL(glfwGetProcAddress);
    if (version == 0) {
        errors << "GLAD could not load OpenGL functions.\n";
        return false;
    }
    return true;
}

const char* gl_string(const GLenum name) {
    const GLubyte* value = glGetString(name);
    return value == nullptr ? "unknown" : reinterpret_cast<const char*>(value);
}

void framebuffer_size_callback(GLFWwindow*, const int width, const int height) {
    glViewport(0, 0, width, height);
}

}  // namespace

GraphicsResult report_graphics_environment(std::ostream& output,
                                           std::ostream& errors) {
    const GlfwSession glfw{errors};
    if (!glfw) {
        return GraphicsResult::environment_unavailable;
    }

    const WindowOwner window = create_window(false, errors);
    if (!window) {
        return GraphicsResult::environment_unavailable;
    }
    glfwMakeContextCurrent(window.get());

    if (!load_opengl(errors)) {
        return GraphicsResult::loader_failed;
    }

    GLint context_major = 0;
    GLint context_minor = 0;
    glGetIntegerv(GL_MAJOR_VERSION, &context_major);
    glGetIntegerv(GL_MINOR_VERSION, &context_minor);

    output << "GLFW version: " << glfwGetVersionString() << '\n'
           << "OpenGL context: " << context_major << '.' << context_minor
           << '\n'
           << "OpenGL vendor: " << gl_string(GL_VENDOR) << '\n'
           << "OpenGL renderer: " << gl_string(GL_RENDERER) << '\n'
           << "OpenGL version: " << gl_string(GL_VERSION) << '\n'
           << "GLSL version: " << gl_string(GL_SHADING_LANGUAGE_VERSION) << '\n'
           << "GLAD loader: initialized\n"
           << "Dear ImGui version: " << ui::imgui_version() << '\n';
    return GraphicsResult::success;
}

GraphicsResult run_graphics_application(std::ostream& errors) {
    const GlfwSession glfw{errors};
    if (!glfw) {
        return GraphicsResult::environment_unavailable;
    }

    const WindowOwner window = create_window(true, errors);
    if (!window) {
        return GraphicsResult::environment_unavailable;
    }
    glfwMakeContextCurrent(window.get());

    if (!load_opengl(errors)) {
        return GraphicsResult::loader_failed;
    }

    ui::ImGuiLayer imgui;
    if (!imgui.initialize(window.get(), errors)) {
        return GraphicsResult::loader_failed;
    }

    glfwSetFramebufferSizeCallback(window.get(), framebuffer_size_callback);
    glfwSwapInterval(1);

    while (glfwWindowShouldClose(window.get()) == GLFW_FALSE) {
        if (glfwGetKey(window.get(), GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window.get(), GLFW_TRUE);
        }

        glClearColor(0.055F, 0.071F, 0.102F, 1.0F);
        glClear(GL_COLOR_BUFFER_BIT);
        imgui.begin_frame();
        imgui.draw_application_shell();
        imgui.render();
        glfwSwapBuffers(window.get());
        glfwPollEvents();
    }

    return GraphicsResult::success;
}

}  // namespace hzl::rendering
