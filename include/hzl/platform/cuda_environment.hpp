#pragma once

#include <iosfwd>

namespace hzl::platform {

enum class CudaProbeResult : int {
    compatible_device_found = 0,
    runtime_unavailable = 2,
    target_device_not_found = 3,
};

[[nodiscard]] CudaProbeResult report_cuda_environment(std::ostream& output,
                                                      std::ostream& errors);

}  // namespace hzl::platform
