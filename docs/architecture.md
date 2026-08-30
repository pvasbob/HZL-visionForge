# Repository layout

The source tree separates application orchestration from platform, UI, media,
processing, rendering, and profiling concerns:

```text
assets/             Runtime assets and legally redistributable sample media
benchmarks/         Reproducible CPU/GPU benchmark programs and inputs
cmake/              Project-specific CMake modules
docs/               Requirements, roadmap, architecture, and user documentation
include/hzl/        Public and cross-module C++ headers
src/app/            Application startup and orchestration
src/media/          Image, video, webcam, and export adapters
src/platform/       Windowing, GPU discovery, and platform integration
src/processing/     CPU references, CUDA kernels, and processing pipelines
src/profiling/      Timing and GPU-memory instrumentation
src/rendering/      OpenGL resources and CUDA-OpenGL interoperability
src/ui/             Dear ImGui views and interactive controls
tests/              Unit, integration, and image-correctness tests
third_party/        Vendored dependency metadata or source when required
```

Production headers use the `hzl` namespace and mirror the module names below
`include/hzl`. Implementations remain in the corresponding directory below
`src`. Tests and benchmarks should follow the same module boundaries so that
graphical, GPU, and deterministic CPU logic can be exercised independently.

Generated build output belongs outside the source tree, conventionally in
`build/`, and must not be committed.
