#pragma once

#include <iosfwd>

namespace hzl::media {

enum class OpenCvProbeResult : int {
    available = 0,
    smoke_test_failed = 4,
};

[[nodiscard]] OpenCvProbeResult report_opencv_environment(
    std::ostream& output,
    std::ostream& errors);

}  // namespace hzl::media
