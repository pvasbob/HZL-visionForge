#pragma once

#include <iosfwd>

namespace hzl::rendering {

enum class GraphicsResult : int {
    success = 0,
    environment_unavailable = 5,
    loader_failed = 6,
};

[[nodiscard]] GraphicsResult report_graphics_environment(std::ostream& output,
                                                         std::ostream& errors);
[[nodiscard]] GraphicsResult run_graphics_application(std::ostream& errors);

}  // namespace hzl::rendering
