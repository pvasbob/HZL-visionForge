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

    std::cout << application_name << " application skeleton\n";
    return 0;
}
