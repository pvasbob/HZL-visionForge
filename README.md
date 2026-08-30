# HZL-VisionForge

HZL-VisionForge is a native C++17 and CUDA desktop application for interactive,
GPU-accelerated image and video processing. The project is currently being built
incrementally; see the [requirements](docs/requirements.md),
[roadmap](docs/roadmap.md), and [repository layout](docs/architecture.md).

## Build

A C++17 compiler, CMake 3.20 or newer, a CUDA toolkit, OpenCV 4 with `core`,
`imgproc`, `imgcodecs`, and `videoio`, OpenGL, and GLFW 3.3 are required. CMake
fetches pinned GLAD 2.0.8 and Dear ImGui 1.92.9b docking sources. Configure and
build in a separate directory (the RTX 3080 target defaults to `sm_86`):

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

Inspect the GLFW, OpenGL, and GLAD environment without opening a visible window:

```sh
./build/hzl-visionforge --graphics-info
```

Verify the Dear ImGui context, keyboard navigation, and docking configuration:

```sh
./build/hzl-visionforge --imgui-info
```

Load and inspect a PNG, JPEG, BMP, or TIFF image from the command line:

```sh
./build/hzl-visionforge --image-info path/to/image.png
```

Export a loaded image to PNG or JPEG from the command line:

```sh
./build/hzl-visionforge --export-image input.tiff output.png
```

Running `./build/hzl-visionforge` opens the dockable application shell with
viewport, processing, profiling, and status panels. Use **File > Open Image** to
load supported media and **File > Export Image** to save PNG or JPEG output. In
the viewport, use the mouse wheel to zoom around the
cursor, drag with the left mouse button to pan, double-click to fit, or use the
**Fit** and **100%** controls. Press Escape or use **File > Exit** to close it.

Run the CUDA baseline-versus-shared-memory comparison on a CUDA-capable host:

```sh
./build/cuda_optimization_benchmark
```
