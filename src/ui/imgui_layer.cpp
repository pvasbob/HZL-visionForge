#include "hzl/ui/imgui_layer.hpp"
#include "hzl/processing/cpu_filters.hpp"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdint>
#include <exception>
#include <iterator>
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

bool draw_filter_parameters(processing::PipelineOperation& operation) {
    bool changed = false;
    switch (operation.type) {
        case processing::FilterType::brightness_contrast: {
            auto& parameters = std::get<processing::BrightnessContrastParameters>(
                operation.parameters);
            changed = ImGui::SliderFloat(
                          "Brightness", &parameters.brightness, -255.0F, 255.0F) ||
                      changed;
            changed = ImGui::SliderFloat(
                          "Contrast", &parameters.contrast, 0.0F, 3.0F) ||
                      changed;
            break;
        }
        case processing::FilterType::gamma: {
            auto& parameters =
                std::get<processing::GammaParameters>(operation.parameters);
            changed = ImGui::SliderFloat("Gamma", &parameters.gamma, 0.1F, 5.0F);
            break;
        }
        case processing::FilterType::box_blur: {
            auto& parameters =
                std::get<processing::BoxBlurParameters>(operation.parameters);
            changed = ImGui::SliderInt(
                "Kernel", &parameters.kernel_size, 1, 31);
            if (parameters.kernel_size % 2 == 0) {
                parameters.kernel_size =
                    std::min(parameters.kernel_size + 1, 31);
            }
            break;
        }
        case processing::FilterType::gaussian_blur: {
            auto& parameters =
                std::get<processing::GaussianBlurParameters>(operation.parameters);
            changed = ImGui::SliderInt(
                          "Kernel", &parameters.kernel_size, 1, 31) ||
                      changed;
            if (parameters.kernel_size % 2 == 0) {
                parameters.kernel_size =
                    std::min(parameters.kernel_size + 1, 31);
            }
            changed = ImGui::SliderFloat(
                          "Sigma", &parameters.sigma, 0.0F, 10.0F) ||
                      changed;
            break;
        }
        case processing::FilterType::sharpen: {
            auto& parameters =
                std::get<processing::SharpenParameters>(operation.parameters);
            changed = ImGui::SliderFloat(
                "Amount", &parameters.amount, 0.0F, 5.0F);
            break;
        }
        case processing::FilterType::emboss: {
            auto& parameters =
                std::get<processing::EmbossParameters>(operation.parameters);
            changed = ImGui::SliderFloat(
                "Strength", &parameters.strength, 0.0F, 3.0F);
            break;
        }
        case processing::FilterType::sobel:
        case processing::FilterType::laplacian: {
            auto& parameters =
                std::get<processing::EdgeParameters>(operation.parameters);
            changed = ImGui::SliderFloat(
                "Strength", &parameters.strength, 0.0F, 5.0F);
            break;
        }
        case processing::FilterType::tone_mapping: {
            auto& parameters =
                std::get<processing::ToneMappingParameters>(operation.parameters);
            changed = ImGui::SliderFloat("Exposure (EV)", &parameters.exposure,
                                         -5.0F, 5.0F);
            break;
        }
        case processing::FilterType::color_grading: {
            auto& parameters =
                std::get<processing::ColorGradingParameters>(operation.parameters);
            changed = ImGui::SliderFloat("Saturation", &parameters.saturation,
                                         0.0F, 2.0F) || changed;
            changed = ImGui::SliderFloat("Temperature", &parameters.temperature,
                                         -1.0F, 1.0F) || changed;
            changed = ImGui::SliderFloat("Tint", &parameters.tint,
                                         -1.0F, 1.0F) || changed;
            changed = ImGui::SliderFloat("Red gain", &parameters.red_gain,
                                         0.0F, 2.0F) || changed;
            changed = ImGui::SliderFloat("Green gain", &parameters.green_gain,
                                         0.0F, 2.0F) || changed;
            changed = ImGui::SliderFloat("Blue gain", &parameters.blue_gain,
                                         0.0F, 2.0F) || changed;
            break;
        }
        case processing::FilterType::grayscale:
        case processing::FilterType::invert:
        case processing::FilterType::histogram_equalization:
            break;
    }
    return changed;
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

void ImGuiLayer::accept_frame(cv::Mat rgba_pixels, const std::string& label) {
    const bool dimensions_changed =
        !image_ || image_->rgba_pixels.cols != rgba_pixels.cols ||
        image_->rgba_pixels.rows != rgba_pixels.rows;
    std::string upload_error;
    if (!image_texture_.upload_rgba8(rgba_pixels, upload_error)) {
        status_message_ = upload_error;
        return;
    }

    image_ = media::ImageDocument{label,
                                  std::move(rgba_pixels),
                                  4,
                                  CV_8U};
    update_gpu_outputs(true);
    if (dimensions_changed) {
        zoom_ = 1.0F;
        pan_x_ = 0.0F;
        pan_y_ = 0.0F;
        fit_to_window_ = true;
    }
    export_path_.front() = '\0';
    status_message_ = "Loaded " + label +
                      (using_cuda_presentation_
                           ? " | CUDA pipeline and interop presentation"
                           : " | OpenGL fallback");
}

void ImGuiLayer::update_gpu_outputs(const bool update_original) {
    if (!image_) {
        return;
    }
    using_cuda_presentation_ = false;
    processed_image_ = nullptr;
    try {
        if (update_original) {
            cuda_image_.resize(
                static_cast<std::size_t>(image_->rgba_pixels.cols),
                static_cast<std::size_t>(image_->rgba_pixels.rows));
            cuda_image_.upload(image_->rgba_pixels.data, image_->rgba_pixels.step);
            original_presentation_.resize(cuda_image_.width(), cuda_image_.height());
            original_presentation_.upload_back(cuda_image_);
            original_presentation_.swap();
        }

        processed_image_ = &pipeline_.process(cuda_image_);
        processed_presentation_.resize(processed_image_->width(),
                                       processed_image_->height());
        processed_presentation_.upload_back(*processed_image_);
        processed_presentation_.swap();

        processing::cuda::absolute_difference(
            cuda_image_, *processed_image_, difference_image_);
        difference_presentation_.resize(difference_image_.width(),
                                        difference_image_.height());
        difference_presentation_.upload_back(difference_image_);
        difference_presentation_.swap();
        using_cuda_presentation_ = true;
        pipeline_dirty_ = false;
    } catch (const std::exception& exception) {
        status_message_ = std::string{"CUDA processing unavailable: "} +
                          exception.what();
        pipeline_dirty_ = false;
    }
}

void ImGuiLayer::draw_application_shell() {
    ImGui::DockSpaceOverViewport(0,
                                nullptr,
                                ImGuiDockNodeFlags_PassthruCentralNode);

    if (video_source_.is_open() &&
        std::chrono::steady_clock::now() >= next_frame_time_) {
        media::FrameReadResult frame = video_source_.read();
        if (frame) {
            accept_frame(std::move(frame.rgba_pixels), video_source_.label());
            const double fps = video_source_.fps() > 0.0 ? video_source_.fps() : 60.0;
            next_frame_time_ = std::chrono::steady_clock::now() +
                               std::chrono::duration_cast<
                                   std::chrono::steady_clock::duration>(
                                   std::chrono::duration<double>{1.0 / fps});
        } else {
            status_message_ = frame.message;
            video_source_.close();
        }
    } else if (pipeline_dirty_ && image_) {
        update_gpu_outputs(false);
    }

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
            if (ImGui::MenuItem("Open Video...")) {
                open_video_dialog_ = true;
            }
            if (ImGui::MenuItem("Open Camera...")) {
                open_camera_dialog_ = true;
            }
            if (ImGui::MenuItem("Stop Video / Camera",
                                nullptr,
                                false,
                                video_source_.is_open())) {
                video_source_.close();
                status_message_ = "Video/camera input stopped.";
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
                video_source_.close();
                accept_frame(std::move(candidate.rgba_pixels),
                             candidate.source_path.string());
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

    if (open_video_dialog_) {
        ImGui::OpenPopup("Open Video");
        open_video_dialog_ = false;
    }
    if (ImGui::BeginPopupModal("Open Video", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::TextUnformatted("Video file path:");
        ImGui::SetNextItemWidth(560.0F);
        const bool submitted = ImGui::InputText(
            "##video-path",
            video_path_.data(),
            video_path_.size(),
            ImGuiInputTextFlags_EnterReturnsTrue);
        const bool open_clicked = ImGui::Button("Open");
        ImGui::SameLine();
        const bool cancel_clicked = ImGui::Button("Cancel");
        if (submitted || open_clicked) {
            const media::MediaOpenResult result =
                video_source_.open_file(video_path_.data());
            if (result) {
                next_frame_time_ = std::chrono::steady_clock::now();
                status_message_ = "Opened video " + video_source_.label();
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

    if (open_camera_dialog_) {
        ImGui::OpenPopup("Open Camera");
        open_camera_dialog_ = false;
    }
    if (ImGui::BeginPopupModal("Open Camera", nullptr,
                               ImGuiWindowFlags_AlwaysAutoResize)) {
        ImGui::InputInt("Camera index", &camera_index_);
        const bool open_clicked = ImGui::Button("Open");
        ImGui::SameLine();
        const bool cancel_clicked = ImGui::Button("Cancel");
        if (open_clicked) {
            const media::MediaOpenResult result =
                video_source_.open_camera(camera_index_);
            if (result) {
                next_frame_time_ = std::chrono::steady_clock::now();
                status_message_ = "Opened " + video_source_.label();
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
            try {
                cv::Mat export_pixels = image_->rgba_pixels;
                if (using_cuda_presentation_ && processed_image_ != nullptr) {
                    export_pixels.create(image_->rgba_pixels.rows,
                                         image_->rgba_pixels.cols,
                                         CV_8UC4);
                    processed_image_->download(export_pixels.data,
                                               export_pixels.step);
                }
                const media::ImageExportResult result = media::export_image(
                    export_pixels, export_path_.data(), export_options_);
                if (result) {
                    status_message_ =
                        "Exported " + std::string{export_path_.data()};
                    ImGui::CloseCurrentPopup();
                } else {
                    status_message_ = result.message;
                }
            } catch (const std::exception& exception) {
                status_message_ = exception.what();
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
        constexpr const char* comparison_names[]{
            "Original", "Processed", "Split", "Side by side", "Difference"};
        ImGui::SameLine();
        ImGui::SetNextItemWidth(130.0F);
        ImGui::Combo("##comparison-mode",
                     &comparison_mode_,
                     comparison_names,
                     static_cast<int>(std::size(comparison_names)));
        if (comparison_mode_ == 2) {
            ImGui::SameLine();
            ImGui::SetNextItemWidth(120.0F);
            ImGui::SliderFloat("##split", &comparison_split_, 0.0F, 1.0F, "%.0f%%");
        }

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
        const unsigned int original_texture =
            using_cuda_presentation_ ? original_presentation_.front_texture_id()
                                     : image_texture_.id();
        const unsigned int processed_texture =
            using_cuda_presentation_ ? processed_presentation_.front_texture_id()
                                     : image_texture_.id();
        const unsigned int difference_texture =
            using_cuda_presentation_ ? difference_presentation_.front_texture_id()
                                     : image_texture_.id();

        if (comparison_mode_ == 3) {
            const float half_width = canvas_size.x * 0.5F;
            const float half_zoom = clamped_zoom(std::min(
                half_width / static_cast<float>(image_texture_.width()),
                canvas_size.y / static_cast<float>(image_texture_.height())));
            const ImVec2 side_size{
                static_cast<float>(image_texture_.width()) * half_zoom,
                static_cast<float>(image_texture_.height()) * half_zoom};
            const ImVec2 left_min{
                canvas_min.x + (half_width - side_size.x) * 0.5F,
                canvas_min.y + (canvas_size.y - side_size.y) * 0.5F};
            const ImVec2 right_min{
                canvas_min.x + half_width + (half_width - side_size.x) * 0.5F,
                left_min.y};
            draw_checkerboard(*draw_list,
                              left_min,
                              ImVec2{left_min.x + side_size.x,
                                     left_min.y + side_size.y},
                              canvas_min,
                              canvas_max);
            draw_checkerboard(*draw_list,
                              right_min,
                              ImVec2{right_min.x + side_size.x,
                                     right_min.y + side_size.y},
                              canvas_min,
                              canvas_max);
            draw_list->AddImage(
                ImTextureRef{static_cast<ImTextureID>(original_texture)},
                left_min,
                ImVec2{left_min.x + side_size.x, left_min.y + side_size.y});
            draw_list->AddImage(
                ImTextureRef{static_cast<ImTextureID>(processed_texture)},
                right_min,
                ImVec2{right_min.x + side_size.x, right_min.y + side_size.y});
            draw_list->AddLine(ImVec2{canvas_min.x + half_width, canvas_min.y},
                               ImVec2{canvas_min.x + half_width, canvas_max.y},
                               IM_COL32(220, 220, 220, 180));
        } else {
            draw_checkerboard(*draw_list,
                              bounds.first,
                              bounds.second,
                              canvas_min,
                              canvas_max);
            unsigned int displayed_texture = processed_texture;
            if (comparison_mode_ == 0) {
                displayed_texture = original_texture;
            } else if (comparison_mode_ == 4) {
                displayed_texture = difference_texture;
            }
            draw_list->AddImage(
                ImTextureRef{static_cast<ImTextureID>(displayed_texture)},
                bounds.first,
                bounds.second);
            if (comparison_mode_ == 2) {
                const float split_x = canvas_min.x +
                                      canvas_size.x * comparison_split_;
                draw_list->PushClipRect(canvas_min,
                                        ImVec2{split_x, canvas_max.y},
                                        true);
                draw_list->AddImage(
                    ImTextureRef{static_cast<ImTextureID>(original_texture)},
                    bounds.first,
                    bounds.second);
                draw_list->PopClipRect();
                draw_list->AddLine(ImVec2{split_x, canvas_min.y},
                                   ImVec2{split_x, canvas_max.y},
                                   IM_COL32(255, 255, 255, 220),
                                   2.0F);
            }
        }
        draw_list->PopClipRect();
        draw_list->AddRect(canvas_min,
                           canvas_max,
                           IM_COL32(70, 75, 86, 255));
    } else {
        ImGui::TextDisabled("Open an image to begin processing.");
    }
    ImGui::End();

    ImGui::Begin("Processing Pipeline");
    constexpr processing::FilterType filter_types[]{
        processing::FilterType::grayscale,
        processing::FilterType::invert,
        processing::FilterType::brightness_contrast,
        processing::FilterType::gamma,
        processing::FilterType::box_blur,
        processing::FilterType::gaussian_blur,
        processing::FilterType::sharpen,
        processing::FilterType::emboss,
        processing::FilterType::sobel,
        processing::FilterType::laplacian,
        processing::FilterType::histogram_equalization,
        processing::FilterType::tone_mapping,
        processing::FilterType::color_grading,
    };
    static int selected_filter = 0;
    ImGui::SetNextItemWidth(-90.0F);
    if (ImGui::BeginCombo("##new-filter",
                          processing::filter_name(filter_types[selected_filter]))) {
        for (int index = 0;
             index < static_cast<int>(std::size(filter_types));
             ++index) {
            if (ImGui::Selectable(processing::filter_name(filter_types[index]),
                                  selected_filter == index)) {
                selected_filter = index;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    if (ImGui::Button("Add")) {
        static_cast<void>(pipeline_.add(filter_types[selected_filter]));
        pipeline_dirty_ = true;
    }
    ImGui::SameLine();
    if (ImGui::Button("Clear") && !pipeline_.operations().empty()) {
        pipeline_.clear();
        pipeline_dirty_ = true;
    }
    ImGui::Separator();

    std::uint64_t remove_id = 0;
    std::uint64_t move_up_id = 0;
    std::uint64_t move_down_id = 0;
    for (std::size_t index = 0; index < pipeline_.operations().size(); ++index) {
        processing::PipelineOperation& operation = pipeline_.operations()[index];
        ImGui::PushID(static_cast<int>(operation.id));
        if (ImGui::Checkbox("##enabled", &operation.enabled)) {
            pipeline_dirty_ = true;
        }
        ImGui::SameLine();
        const bool open = ImGui::CollapsingHeader(
            processing::filter_name(operation.type),
            ImGuiTreeNodeFlags_DefaultOpen);
        if (open) {
            pipeline_dirty_ = draw_filter_parameters(operation) || pipeline_dirty_;
            if (ImGui::Button("Up") && index > 0) {
                move_up_id = operation.id;
            }
            ImGui::SameLine();
            if (ImGui::Button("Down") &&
                index + 1 < pipeline_.operations().size()) {
                move_down_id = operation.id;
            }
            ImGui::SameLine();
            if (ImGui::Button("Remove")) {
                remove_id = operation.id;
            }
        }
        ImGui::PopID();
        if (remove_id != 0 || move_up_id != 0 || move_down_id != 0) {
            break;
        }
    }
    if (remove_id != 0) {
        static_cast<void>(pipeline_.remove(remove_id));
        pipeline_dirty_ = true;
    } else if (move_up_id != 0) {
        static_cast<void>(pipeline_.move_up(move_up_id));
        pipeline_dirty_ = true;
    } else if (move_down_id != 0) {
        static_cast<void>(pipeline_.move_down(move_down_id));
        pipeline_dirty_ = true;
    }
    ImGui::End();

    ImGui::Begin("Profiler");
    ImGui::Text("Frame time: --");
    ImGui::Text("GPU memory: --");
    if (image_) {
        const processing::cpu::Histogram histogram =
            processing::cpu::luminance_histogram(image_->rgba_pixels);
        std::array<float, 256> plot{};
        std::transform(histogram.begin(), histogram.end(), plot.begin(),
                       [](const std::uint32_t count) {
                           return static_cast<float>(count);
                       });
        ImGui::PlotHistogram("Luminance", plot.data(),
                             static_cast<int>(plot.size()), 0, nullptr,
                             0.0F, FLT_MAX, ImVec2{0.0F, 100.0F});
    }
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
