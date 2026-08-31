# Architecture

HZL-VisionForge keeps media decoding on the CPU, processing on CUDA, and final
presentation in OpenGL. After the initial upload, processed pixels remain on the
GPU and CUDA writes directly into double-buffered OpenGL textures.

```text
image/video/webcam -> OpenCV RGBA8 -> CUDA ImageBuffer -> GpuPipeline
                                                        -> CUDA/OpenGL buffers
                                                        -> viewport + comparisons
```

## Modules

- `app` parses diagnostic commands and starts the graphical runtime.
- `media` normalizes images and decoded frames to contiguous RGBA8 and exports
  PNG/JPEG output.
- `platform` reports CUDA devices and validates the RTX 3080 `sm_86` target.
- `processing` owns CPU references, pitched CUDA memory, custom kernels, filter
  parameters, pipeline ordering, and comparison output.
- `rendering` owns the OpenGL context, texture fallback, CUDA registration, and
  front/back presentation resources.
- `ui` owns the Dear ImGui shell, media dialogs, viewport interaction, pipeline
  editor, histogram, telemetry, and diagnostic history.

CUDA and OpenGL resources use move-only RAII types. The processing pipeline
reuses two intermediate buffers; presentation uses front/back textures so a
completed frame remains visible while the next is produced. CPU filters provide
deterministic test and benchmark references, not the interactive render path.

## Repository layout

```text
benchmarks/         Reproducible CPU/GPU benchmark programs and inputs
cmake/              Project-specific CMake modules
docs/               Requirements, roadmap, architecture, and user documentation
include/hzl/        Public and cross-module C++ headers
src/app/            Application startup and orchestration
src/media/          Image, video, webcam, and export adapters
src/platform/       GPU discovery and platform diagnostics
src/processing/     CPU references, CUDA kernels, and processing pipelines
src/rendering/      OpenGL resources and CUDA-OpenGL interoperability
src/ui/             Dear ImGui views and interactive controls
tests/              Unit, integration, and image-correctness tests
packaging/          Desktop and distribution metadata
```

Production headers use the `hzl` namespace and mirror the module names below
`include/hzl`. Implementations remain in the corresponding directory below
`src`. Tests and benchmarks should follow the same module boundaries so that
graphical, GPU, and deterministic CPU logic can be exercised independently.

Generated build output belongs outside the source tree, conventionally in
`build/`, and must not be committed.

## Processing and failure flow

Each enabled pipeline operation reads one GPU buffer and writes the other. CUDA
launchers validate dimensions and parameters, resize only when capacity is
insufficient, check launches centrally, and preserve alpha. Exceptions cross the
processing boundary into the UI, where they become persistent diagnostics with
recovery guidance instead of terminating the render loop. A final application
boundary catches otherwise unhandled failures and points users to environment
diagnostic commands.
