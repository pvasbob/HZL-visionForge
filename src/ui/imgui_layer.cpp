#include "hzl/ui/imgui_layer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <ostream>

namespace hzl::ui {
namespace {

void configure_io() {
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

}  // namespace

ImGuiLayer::~ImGuiLayer() {
    if (opengl_backend_initialized_) {
        ImGui_ImplOpenGL3_Shutdown();
    }
    if (glfw_backend_initialized_) {
        ImGui_ImplGlfw_Shutdown();
    }
    if (context_created_) {
        ImGui::DestroyContext();
    }
}

bool ImGuiLayer::initialize(GLFWwindow* window, std::ostream& errors) {
    if (window == nullptr) {
        errors << "Dear ImGui initialization requires a valid GLFW window.\n";
        return false;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    context_created_ = true;
    configure_io();
    ImGui::StyleColorsDark();

    if (!ImGui_ImplGlfw_InitForOpenGL(window, true)) {
        errors << "Dear ImGui GLFW backend initialization failed.\n";
        return false;
    }
    glfw_backend_initialized_ = true;

    if (!ImGui_ImplOpenGL3_Init("#version 450 core")) {
        errors << "Dear ImGui OpenGL backend initialization failed.\n";
        return false;
    }
    opengl_backend_initialized_ = true;
    window_ = window;
    return true;
}

void ImGuiLayer::begin_frame() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::draw_application_shell() {
    ImGui::DockSpaceOverViewport(0,
                                nullptr,
                                ImGuiDockNodeFlags_PassthruCentralNode);

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Exit", "Esc")) {
                glfwSetWindowShouldClose(window_, GLFW_TRUE);
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("View")) {
            ImGui::MenuItem("Pipeline", nullptr, true, false);
            ImGui::MenuItem("Profiler", nullptr, true, false);
            ImGui::EndMenu();
        }
        ImGui::EndMainMenuBar();
    }

    ImGui::Begin("Image Viewport");
    ImGui::TextDisabled("Open an image to begin processing.");
    ImGui::End();

    ImGui::Begin("Processing Pipeline");
    ImGui::TextDisabled("No processing operations configured.");
    ImGui::End();

    ImGui::Begin("Profiler");
    ImGui::Text("Frame time: --");
    ImGui::Text("GPU memory: --");
    ImGui::End();

    ImGui::Begin("Status");
    ImGui::TextUnformatted("Ready");
    ImGui::SameLine();
    ImGui::TextDisabled("| No media loaded");
    ImGui::End();
}

void ImGuiLayer::render() {
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

const char* imgui_version() noexcept {
    return ImGui::GetVersion();
}

bool report_imgui_environment(std::ostream& output, std::ostream& errors) {
    IMGUI_CHECKVERSION();
    ImGuiContext* context = ImGui::CreateContext();
    if (context == nullptr) {
        errors << "Dear ImGui context creation failed.\n";
        return false;
    }

    configure_io();
    const ImGuiIO& io = ImGui::GetIO();
    const bool keyboard_navigation =
        (io.ConfigFlags & ImGuiConfigFlags_NavEnableKeyboard) != 0;
    const bool docking = (io.ConfigFlags & ImGuiConfigFlags_DockingEnable) != 0;

    output << "Dear ImGui version: " << ImGui::GetVersion() << '\n'
           << "Keyboard navigation: "
           << (keyboard_navigation ? "enabled" : "disabled") << '\n'
           << "Docking: " << (docking ? "enabled" : "disabled") << '\n'
           << "Dear ImGui smoke test: "
           << (keyboard_navigation && docking ? "passed" : "failed") << '\n';

    ImGui::DestroyContext(context);
    return keyboard_navigation && docking;
}

}  // namespace hzl::ui
