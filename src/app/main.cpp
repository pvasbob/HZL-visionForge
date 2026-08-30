#include "hzl/media/image_export.hpp"
#include "hzl/media/image_loader.hpp"
#include "hzl/media/opencv_environment.hpp"
#include "hzl/media/video_source.hpp"
#include "hzl/platform/cuda_environment.hpp"
#include "hzl/rendering/graphics_environment.hpp"
#include "hzl/ui/imgui_layer.hpp"
#include "hzl/version.hpp"

#include <iostream>
#include <string_view>

namespace {

constexpr std::string_view application_name{"HZL-VisionForge"};

}  // namespace

int main(const int argc, const char* const argv[]) {
    if (argc == 2 && std::string_view{argv[1]} == "--version") {
        std::cout << application_name << ' ' << HZL_VISIONFORGE_VERSION << '\n';
        return 0;
    }
    if (argc == 2 && std::string_view{argv[1]} == "--gpu-info") {
        return static_cast<int>(
            hzl::platform::report_cuda_environment(std::cout, std::cerr));
    }
    if (argc == 2 && std::string_view{argv[1]} == "--opencv-info") {
        return static_cast<int>(
            hzl::media::report_opencv_environment(std::cout, std::cerr));
    }
    if (argc == 2 && std::string_view{argv[1]} == "--graphics-info") {
        return static_cast<int>(
            hzl::rendering::report_graphics_environment(std::cout, std::cerr));
    }
    if (argc == 2 && std::string_view{argv[1]} == "--imgui-info") {
        return hzl::ui::report_imgui_environment(std::cout, std::cerr) ? 0 : 7;
    }
    if (argc == 3 && std::string_view{argv[1]} == "--image-info") {
        hzl::media::ImageLoadResult result = hzl::media::load_image(argv[2]);
        if (!result) {
            std::cerr << "Image load failed: " << result.message << '\n';
            return 8;
        }
        const hzl::media::ImageDocument& image = result.document.value();
        std::cout << "Image: " << image.source_path.string() << '\n'
                  << "Dimensions: " << image.rgba_pixels.cols << " x "
                  << image.rgba_pixels.rows << '\n'
                  << "Source depth: "
                  << hzl::media::pixel_depth_name(image.original_depth) << '\n'
                  << "Source channels: " << image.original_channels << '\n'
                  << "Normalized format: RGBA8\n";
        return 0;
    }
    if (argc == 4 && std::string_view{argv[1]} == "--export-image") {
        const hzl::media::ImageLoadResult load_result =
            hzl::media::load_image(argv[2]);
        if (!load_result) {
            std::cerr << "Image load failed: " << load_result.message << '\n';
            return 8;
        }
        const hzl::media::ImageExportResult export_result =
            hzl::media::export_image(load_result.document->rgba_pixels, argv[3]);
        if (!export_result) {
            std::cerr << "Image export failed: " << export_result.message << '\n';
            return 9;
        }
        std::cout << "Exported image: " << argv[3] << '\n';
        return 0;
    }
    if (argc == 3 && std::string_view{argv[1]} == "--video-info") {
        hzl::media::VideoSource source;
        const hzl::media::MediaOpenResult open = source.open_file(argv[2]);
        if (!open) {
            std::cerr << "Video open failed: " << open.message << '\n';
            return 10;
        }
        const hzl::media::FrameReadResult frame = source.read();
        if (!frame) {
            std::cerr << "Video decode failed: " << frame.message << '\n';
            return 10;
        }
        std::cout << "Video: " << source.label() << '\n'
                  << "Dimensions: " << source.width() << " x "
                  << source.height() << '\n'
                  << "FPS: " << source.fps() << '\n'
                  << "First frame: RGBA8\n";
        return 0;
    }

    return static_cast<int>(
        hzl::rendering::run_graphics_application(std::cerr));
}
