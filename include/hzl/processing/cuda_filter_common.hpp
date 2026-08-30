#pragma once

namespace hzl::processing::cuda {

enum class KernelImplementation {
    global_memory,
    shared_memory,
};

}  // namespace hzl::processing::cuda
