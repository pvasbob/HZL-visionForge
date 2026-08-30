#include "hzl/media/opencv_environment.hpp"
#include "hzl/platform/cuda_environment.hpp"
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

    std::cout << application_name
              << " application skeleton\nUse --gpu-info to inspect CUDA support "
                 "or --opencv-info to inspect OpenCV support.\n";
    return 0;
}
