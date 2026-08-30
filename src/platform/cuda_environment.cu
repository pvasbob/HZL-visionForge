#include "hzl/platform/cuda_environment.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <iomanip>
#include <ostream>
#include <string_view>

namespace hzl::platform {
namespace {

constexpr int target_compute_major = 8;
constexpr int target_compute_minor = 6;
constexpr std::string_view target_name{"RTX 3080"};

void print_cuda_version(std::ostream& output, const int version) {
    output << version / 1000 << '.' << (version % 1000) / 10;
}

}  // namespace

CudaProbeResult report_cuda_environment(std::ostream& output,
                                        std::ostream& errors) {
    output << "CUDA toolkit: ";
    print_cuda_version(output, CUDART_VERSION);
    output << '\n';

    int driver_version = 0;
    int runtime_version = 0;
    const cudaError_t driver_status = cudaDriverGetVersion(&driver_version);
    const cudaError_t runtime_status = cudaRuntimeGetVersion(&runtime_version);
    if (driver_status != cudaSuccess || runtime_status != cudaSuccess) {
        const cudaError_t failure =
            driver_status != cudaSuccess ? driver_status : runtime_status;
        errors << "CUDA runtime unavailable: " << cudaGetErrorString(failure)
               << "\nCheck that the NVIDIA driver is installed, loaded, and "
                  "accessible.\n";
        return CudaProbeResult::runtime_unavailable;
    }

    output << "CUDA driver API: ";
    print_cuda_version(output, driver_version);
    output << "\nCUDA runtime: ";
    print_cuda_version(output, runtime_version);
    output << '\n';

    int device_count = 0;
    const cudaError_t count_status = cudaGetDeviceCount(&device_count);
    if (count_status != cudaSuccess) {
        errors << "CUDA device detection failed: "
               << cudaGetErrorString(count_status)
               << "\nCheck that the NVIDIA driver is loaded and the process has "
                  "GPU access.\n";
        return CudaProbeResult::runtime_unavailable;
    }
    if (device_count == 0) {
        errors << "No CUDA-capable GPU was detected.\n";
        return CudaProbeResult::target_device_not_found;
    }

    bool compatible_device_found = false;
    for (int index = 0; index < device_count; ++index) {
        cudaDeviceProp properties{};
        const cudaError_t property_status =
            cudaGetDeviceProperties(&properties, index);
        if (property_status != cudaSuccess) {
            errors << "Could not inspect CUDA device " << index << ": "
                   << cudaGetErrorString(property_status) << '\n';
            continue;
        }

        const double memory_gib =
            static_cast<double>(properties.totalGlobalMem) /
            static_cast<double>(std::size_t{1024} * 1024U * 1024U);
        const bool is_target =
            std::string_view{properties.name}.find(target_name) !=
                std::string_view::npos &&
            properties.major == target_compute_major &&
            properties.minor == target_compute_minor;

        output << "Device " << index << ": " << properties.name << '\n'
               << "  Compute capability: " << properties.major << '.'
               << properties.minor << '\n'
               << "  Global memory: " << std::fixed << std::setprecision(2)
               << memory_gib << " GiB\n"
               << "  RTX 3080 sm_86 target: " << (is_target ? "yes" : "no")
               << '\n';
        compatible_device_found = compatible_device_found || is_target;
    }

    if (!compatible_device_found) {
        errors << "No NVIDIA GeForce RTX 3080 with compute capability 8.6 was "
                  "found.\n";
        return CudaProbeResult::target_device_not_found;
    }

    return CudaProbeResult::compatible_device_found;
}

}  // namespace hzl::platform
