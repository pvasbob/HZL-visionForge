#include "hzl/ui/imgui_layer.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>

namespace hzl::ui {
namespace {

void configure_io() {
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
}

float clamped_zoom(const float zoom) {
    return std::clamp(zoom, 0.02F, 64.0F);
}

void draw_checkerboard(ImDrawList& draw_list,
                       const ImVec2 image_min,
                       const ImVec2 image_max,
                       const ImVec2 canvas_min,
                       const ImVec2 canvas_max) {
    const ImVec2 visible_min{std::max(image_min.x, canvas_min.x),
                             std::max(image_min.y, canvas_min.y)};
    const ImVec2 visible_max{std::min(image_max.x, canvas_max.x),
                             std::min(image_max.y, canvas_max.y)};
    if (visible_min.x >= visible_max.x || visible_min.y >= visible_max.y) {
        return;
    }

    constexpr float tile_size = 12.0F;
    const int first_column =
        static_cast<int>(std::floor((visible_min.x - image_min.x) / tile_size));
    const int first_row =
        static_cast<int>(std::floor((visible_min.y - image_min.y) / tile_size));
    const float start_x = image_min.x + static_cast<float>(first_column) * tile_size;
    const float start_y = image_min.y + static_cast<float>(first_row) * tile_size;
    const ImU32 colors[]{IM_COL32(78, 82, 91, 255),
                         IM_COL32(112, 117, 128, 255)};

    int row = first_row;
    for (float y = start_y; y < visible_max.y; y += tile_size, ++row) {
        int column = first_column;
        for (float x = start_x; x < visible_max.x;
             x += tile_size, ++column) {
            draw_list.AddRectFilled(
                ImVec2{std::max(x, visible_min.x), std::max(y, visible_min.y)},
                ImVec2{std::min(x + tile_size, visible_max.x),
                       std::min(y + tile_size, visible_max.y)},
                colors[(row + column) & 1]);
        }
    }
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

    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O)) {
        open_image_dialog_ = true;
    }
    if (image_ && ImGui::GetIO().KeyCtrl && ImGui::GetIO().KeyShift &&
        ImGui::IsKeyPressed(ImGuiKey_S)) {
        export_image_dialog_ = true;
    }

    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Image...", "Ctrl+O")) {
                open_image_dialog_ = true;
            }
            if (ImGui::MenuItem("Export Image...",
                                "Ctrl+Shift+S",
                                false,
                                image_.has_value())) {
                export_image_dialog_ = true;
            }
            ImGui::Separator();
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

    if (open_image_dialog_) {
        ImGui::OpenPopup("Open Image");
        open_image_dialog_ = false;
    }
    if (ImGui::BeginPopupModal("Open Image", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("PNG, JPEG, BMP, or TIFF path:");
        ImGui::SetNextItemWidth(560.0F);
        const bool submitted = ImGui::InputText(
            "##image-path",
            image_path_.data(),
            image_path_.size(),
            ImGuiInputTextFlags_EnterReturnsTrue);
        const bool load_clicked = ImGui::Button("Load");
        ImGui::SameLine();
        const bool cancel_clicked = ImGui::Button("Cancel");
        if (submitted || load_clicked) {
            media::ImageLoadResult result = media::load_image(image_path_.data());
            if (result) {
                media::ImageDocument candidate =
                    std::move(result.document.value());
                std::string upload_error;
                if (image_texture_.upload_rgba8(candidate.rgba_pixels,
                                                upload_error)) {
                    image_ = std::move(candidate);
                    zoom_ = 1.0F;
                    pan_x_ = 0.0F;
                    pan_y_ = 0.0F;
                    fit_to_window_ = true;
                    export_path_.front() = '\0';
                    status_message_ = "Loaded " + image_->source_path.string();
                    ImGui::CloseCurrentPopup();
                } else {
                    status_message_ = upload_error;
                }
            } else {
                status_message_ = result.message;
            }
        }
        if (cancel_clicked) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (export_image_dialog_ && image_) {
        if (export_path_.front() == '\0') {
            const std::filesystem::path suggested =
                image_->source_path.parent_path() /
                (image_->source_path.stem().string() + "_export.png");
            std::snprintf(export_path_.data(),
                          export_path_.size(),
                          "%s",
                          suggested.string().c_str());
        }
        ImGui::OpenPopup("Export Image");
        export_image_dialog_ = false;
    }
    if (ImGui::BeginPopupModal("Export Image", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("PNG, JPG, or JPEG destination:");
        ImGui::SetNextItemWidth(560.0F);
        const bool submitted = ImGui::InputText(
            "##export-path",
            export_path_.data(),
            export_path_.size(),
            ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SliderInt("JPEG quality", &export_options_.jpeg_quality, 0, 100);
        ImGui::SliderInt("PNG compression",
                         &export_options_.png_compression,
                         0,
                         9);
        ImGui::Checkbox("Overwrite existing file", &export_options_.overwrite);
        const bool export_clicked = ImGui::Button("Export");
        ImGui::SameLine();
        const bool cancel_clicked = ImGui::Button("Cancel");
        if ((submitted || export_clicked) && image_) {
            const media::ImageExportResult result = media::export_image(
                image_->rgba_pixels, export_path_.data(), export_options_);
            if (result) {
                status_message_ = "Exported " + std::string{export_path_.data()};
                ImGui::CloseCurrentPopup();
            } else {
                status_message_ = result.message;
            }
        }
        if (cancel_clicked) {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    ImGui::Begin("Image Viewport");
    if (image_ && image_texture_.valid()) {
        ImGui::TextUnformatted(image_->source_path.filename().string().c_str());
        ImGui::SameLine();
        if (ImGui::Button("Fit")) {
            fit_to_window_ = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("100%")) {
            zoom_ = 1.0F;
            pan_x_ = 0.0F;
            pan_y_ = 0.0F;
            fit_to_window_ = false;
        }
        ImGui::SameLine();
        ImGui::Text("%.0f%%", zoom_ * 100.0F);

        ImVec2 canvas_size = ImGui::GetContentRegionAvail();
        canvas_size.x = std::max(canvas_size.x, 1.0F);
        canvas_size.y = std::max(canvas_size.y, 1.0F);
        const ImVec2 canvas_min = ImGui::GetCursorScreenPos();
        const ImVec2 canvas_max{canvas_min.x + canvas_size.x,
                                canvas_min.y + canvas_size.y};
        ImGui::InvisibleButton("##image-canvas",
                               canvas_size,
                               ImGuiButtonFlags_MouseButtonLeft);

        if (ImGui::IsItemHovered() &&
            ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
            fit_to_window_ = true;
        }

        if (fit_to_window_) {
            const float horizontal_zoom =
                canvas_size.x / static_cast<float>(image_texture_.width());
            const float vertical_zoom =
                canvas_size.y / static_cast<float>(image_texture_.height());
            zoom_ = clamped_zoom(std::min(horizontal_zoom, vertical_zoom));
            pan_x_ = 0.0F;
            pan_y_ = 0.0F;
        }

        const ImVec2 canvas_center{(canvas_min.x + canvas_max.x) * 0.5F,
                                   (canvas_min.y + canvas_max.y) * 0.5F};
        auto image_bounds = [&]() {
            const ImVec2 display_size{
                static_cast<float>(image_texture_.width()) * zoom_,
                static_cast<float>(image_texture_.height()) * zoom_};
            const ImVec2 minimum{canvas_center.x + pan_x_ - display_size.x * 0.5F,
                                 canvas_center.y + pan_y_ - display_size.y * 0.5F};
            return std::pair<ImVec2, ImVec2>{
                minimum,
                ImVec2{minimum.x + display_size.x,
                       minimum.y + display_size.y}};
        };

        auto bounds = image_bounds();
        const ImGuiIO& io = ImGui::GetIO();
        if (ImGui::IsItemHovered() && io.MouseWheel != 0.0F) {
            const float old_zoom = zoom_;
            const float zoom_factor = std::pow(1.15F, io.MouseWheel);
            const float new_zoom = clamped_zoom(old_zoom * zoom_factor);
            const ImVec2 image_coordinate{
                (io.MousePos.x - bounds.first.x) / old_zoom,
                (io.MousePos.y - bounds.first.y) / old_zoom};
            const ImVec2 new_display_size{
                static_cast<float>(image_texture_.width()) * new_zoom,
                static_cast<float>(image_texture_.height()) * new_zoom};
            pan_x_ = io.MousePos.x - image_coordinate.x * new_zoom -
                     canvas_center.x + new_display_size.x * 0.5F;
            pan_y_ = io.MousePos.y - image_coordinate.y * new_zoom -
                     canvas_center.y + new_display_size.y * 0.5F;
            zoom_ = new_zoom;
            fit_to_window_ = false;
            bounds = image_bounds();
        }
        if (ImGui::IsItemActive() &&
            ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
            pan_x_ += io.MouseDelta.x;
            pan_y_ += io.MouseDelta.y;
            fit_to_window_ = false;
            bounds = image_bounds();
        }

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        draw_list->AddRectFilled(canvas_min,
                                 canvas_max,
                                 IM_COL32(24, 27, 34, 255));
        draw_list->PushClipRect(canvas_min, canvas_max, true);
        draw_checkerboard(*draw_list,
                          bounds.first,
                          bounds.second,
                          canvas_min,
                          canvas_max);
        draw_list->AddImage(
            ImTextureRef{static_cast<ImTextureID>(image_texture_.id())},
            bounds.first,
            bounds.second);
        draw_list->PopClipRect();
        draw_list->AddRect(canvas_min,
                           canvas_max,
                           IM_COL32(70, 75, 86, 255));
    } else {
        ImGui::TextDisabled("Open an image to begin processing.");
    }
    ImGui::End();

    ImGui::Begin("Processing Pipeline");
    ImGui::TextDisabled("No processing operations configured.");
    ImGui::End();

    ImGui::Begin("Profiler");
    ImGui::Text("Frame time: --");
    ImGui::Text("GPU memory: --");
    ImGui::End();

    ImGui::Begin("Status");
    ImGui::TextWrapped("%s", status_message_.c_str());
    if (image_) {
        ImGui::Separator();
        ImGui::Text("%d x %d | RGBA8 | Zoom %.0f%%",
                    image_texture_.width(),
                    image_texture_.height(),
                    zoom_ * 100.0F);
    }
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
