# HZL-VisionForge

HZL-VisionForge is a native C++17 and CUDA desktop application for interactive,
GPU-accelerated image and video processing. The project is currently being built
incrementally; see the [requirements](docs/requirements.md),
[roadmap](docs/roadmap.md), and [repository layout](docs/architecture.md).

## Build

A C++17 compiler and CMake 3.20 or newer are required. Configure and build in a
separate directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the current application skeleton with `./build/hzl-visionforge`.
