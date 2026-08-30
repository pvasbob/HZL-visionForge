# HZL-VisionForge

HZL-VisionForge is a native C++17 and CUDA desktop application for interactive,
GPU-accelerated image and video processing. The project is currently being built
incrementally; see the [requirements](docs/requirements.md),
[roadmap](docs/roadmap.md), and [repository layout](docs/architecture.md).

## Build

A C++17 compiler, CMake 3.20 or newer, a CUDA toolkit, and OpenCV 4 with `core`,
`imgproc`, `imgcodecs`, and `videoio` are required. Configure and build in a
separate directory (the RTX 3080 target defaults to `sm_86`):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Run the current application skeleton with `./build/hzl-visionforge`.
Inspect the CUDA runtime and verify the target GPU with:

```sh
./build/hzl-visionforge --gpu-info
```

The command exits with status 2 when the CUDA runtime or NVIDIA driver is
unavailable, and status 3 when no RTX 3080 `sm_86` device is found.

Inspect the linked OpenCV version and run its in-memory smoke test with:

```sh
./build/hzl-visionforge --opencv-info
```
