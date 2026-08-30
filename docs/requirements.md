# HZL-VisionForge Requirements

## Product vision

HZL-VisionForge is a native desktop application for interactive, GPU-accelerated
image and video processing. OpenCV handles media input and output, custom CUDA
kernels perform the main processing, and OpenGL presents results through direct
CUDA-OpenGL interoperability.

The project should demonstrate production-minded C++ design, CUDA programming,
image-processing fundamentals, real-time graphics, and measurable optimization.

## Initial release scope

The first complete release shall:

- Run on 64-bit Linux with an NVIDIA GeForce RTX 3080.
- Use C++17, CMake, CUDA, OpenCV, OpenGL, GLFW, GLAD, and Dear ImGui.
- Load PNG, JPEG, BMP, and TIFF images.
- Save processed PNG and JPEG images.
- Read common video files and webcam streams through OpenCV.
- Provide an interactive viewport with zoom, pan, and fit-to-window controls.
- Apply an ordered, configurable pipeline of GPU image-processing operations.
- Compare original and processed output using split, side-by-side, and difference
  views.
- Display FPS, frame latency, kernel timings, transfer timings, and GPU memory use.
- Report unsupported input and GPU failures with actionable error messages.

## Core processing operations

The initial release shall provide custom CUDA implementations of:

- Grayscale and color inversion
- Brightness, contrast, and gamma correction
- Box blur and Gaussian blur
- Sharpening and embossing
- Sobel and Laplacian edge detection
- Histogram calculation and histogram equalization
- Basic tone mapping and color grading

Every operation shall be independently enabled and configurable. Operations shall
be reorderable without restarting the application.

## GPU and rendering requirements

- The RTX 3080 is the primary target, using compute capability 8.6 (`sm_86`).
- After input upload, production processing shall remain on the GPU.
- CUDA-OpenGL shared resources shall present processed frames without a per-frame
  GPU-to-CPU-to-GPU copy.
- CUDA calls and kernel launches shall use centralized error checking.
- CUDA and OpenGL resources shall have deterministic, RAII-based ownership.
- The application shall provide CPU/OpenCV reference implementations for testing
  and benchmarking, not as its primary real-time path.
- The render loop shall not allocate GPU memory during steady-state processing.

## Performance targets

On an RTX 3080 in a release build, excluding file decoding unless stated:

- Target 60 FPS at 3840 x 2160 for a representative pipeline containing color
  adjustment, Gaussian blur, sharpening, and edge detection.
- Keep visible response to control changes below 100 ms.
- Keep steady-state GPU memory below 2 GiB for the representative 4K pipeline.
- Publish reproducible CPU-versus-GPU benchmarks for individual filters and the
  representative pipeline.

Benchmark reports shall identify the input, filter settings, build type, GPU,
driver, and CUDA toolkit. Performance targets guide optimization but are not
treated as guarantees for arbitrary pipelines or neural-network models.

## User-interface requirements

The main window shall contain:

- A central image or video viewport
- Media-open and export actions
- A processing-pipeline panel
- Controls for the selected operation
- Original, processed, split, side-by-side, and difference modes
- Status information for dimensions, pixel format, zoom, FPS, and active GPU
- An optional profiling panel with timings and memory statistics

Long-running or failed operations shall show visible status and shall not silently
freeze or terminate the application.

## Engineering and quality requirements

- Separate platform, UI, media, processing, rendering, and profiling concerns.
- Use RAII for CPU, CUDA, and OpenGL resources.
- Enable strict compiler warnings in development builds.
- Test CUDA filters against CPU reference results with documented tolerances.
- Unit-test non-graphical logic and deterministic image operations.
- Use generated or legally redistributable test images.
- Document setup, controls, architecture, limitations, and benchmark methodology.
- Keep the repository buildable at every implementation milestone after the build
  system is introduced.

## Supported environment

Primary support is:

- 64-bit Linux
- NVIDIA GeForce RTX 3080 with 10 or 12 GiB of VRAM
- An NVIDIA driver compatible with the chosen CUDA toolkit
- A CUDA toolkit supporting Ampere (`sm_86`)
- OpenGL 4.5 or newer
- A C++17 compiler and CMake

Windows is a later portability goal rather than an initial-release requirement.

## Out of scope for the initial release

- Photo-catalog or digital-asset management
- Professional RAW development
- Cloud or multi-user collaboration
- Browser and mobile versions
- Multi-GPU execution
- Training neural networks inside the application
- Real-time guarantees for arbitrary neural models
- A general-purpose scripting or plugin language

## Definition of done

The initial release is complete when:

1. A new developer can build and launch it from the documented instructions.
2. Images, videos, and webcam frames pass through the configurable pipeline.
3. All core CUDA operations pass CPU-reference correctness tests.
4. CUDA-OpenGL presentation works without per-frame CPU readback.
5. Invalid inputs and unavailable GPU resources produce useful errors.
6. Automated tests pass in the supported environment.
7. Reproducible performance results are included in the repository.
8. The README includes screenshots or a demo video and explains the architecture.

## Future enhancements

Potential later work includes HDR merging, bilateral and non-local-means denoising,
optical flow, depth-aware effects, TensorRT segmentation or super-resolution, RAW
input, Windows support, and a reusable effect-plugin API.
