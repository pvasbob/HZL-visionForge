#include "hzl/media/opencv_environment.hpp"
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

    return static_cast<int>(
        hzl::rendering::run_graphics_application(std::cerr));
}
