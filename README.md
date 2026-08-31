# HZL-VisionForge

This is a C++17/CUDA project I built to experiment with real-time image and
video processing on an RTX 3080.

It can open images, videos, and webcams, run them through a configurable CUDA
filter pipeline, and display the result with OpenGL and Dear ImGui. There are a
few comparison views too, including split view and image difference.

The filters currently include basic color adjustments, blur, sharpen, emboss,
edge detection, histogram equalization, tone mapping, and color grading.

## Building it

You will need CMake 3.20+, a C++17 compiler, CUDA, OpenCV 4, OpenGL, and GLFW.
The build downloads GLAD and Dear ImGui automatically.

The CUDA target defaults to `sm_86`, since this project is mainly intended for
my RTX 3080.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Then run:

```sh
./build/hzl-visionforge
```

If the window does not open, these commands are useful for checking the setup:

```sh
./build/hzl-visionforge --gpu-info
./build/hzl-visionforge --graphics-info
./build/hzl-visionforge --opencv-info
```

## Using it

Open an image, video, or camera from the File menu. Filters can be added,
reordered, disabled, and adjusted in the pipeline panel.

Use the mouse wheel to zoom and drag the image to pan. The viewport also has
original, processed, split, side-by-side, and difference modes.

Processed images can be exported as PNG or JPEG.

There are also a few command-line helpers:

```sh
./build/hzl-visionforge --image-info image.png
./build/hzl-visionforge --video-info video.mp4
./build/hzl-visionforge --export-image input.png output.jpg
./build/hzl-visionforge --help
```

## Benchmarks

The main benchmark uses a 4K pipeline with color adjustment, Gaussian blur,
sharpening, and Sobel edge detection.

```sh
./build/gpu_benchmark --iterations 10
./build/gpu_benchmark --iterations 10 --csv > visionforge-4k.csv
```

There is also a smaller benchmark comparing the original and shared-memory CUDA
kernels:

```sh
./build/cuda_optimization_benchmark
```

Benchmark numbers depend quite a bit on the build type, driver, GPU load, and
power settings, so Release builds are the useful ones.

## Packaging

```sh
cmake --install build --prefix "$PWD/install"
cpack --config build/CPackConfig.cmake
```

More detailed notes are in the [user guide](docs/user-guide.md),
[architecture](docs/architecture.md), [performance notes](docs/performance.md),
and [roadmap](docs/roadmap.md).

This first version is Linux-only and expects an NVIDIA CUDA-capable GPU. It is a
learning/project application rather than a replacement for a full photo editor.
