# Performance and benchmark methodology

The `gpu_benchmark` executable measures the release representative pipeline at
3840 x 2160 by default. Its deterministic RGBA8 input and fixed settings are:

1. brightness/contrast `(8, 1.08)`
2. shared-memory Gaussian blur `(9 x 9, sigma 2.0)`
3. shared-memory sharpen `(0.75)`
4. shared-memory Sobel edge detection `(1.0)`

The pipeline reuses two pitched CUDA image buffers after warm-up. This bounds
steady-state allocation and uses the shared-memory convolution and edge kernels
selected during optimization. CPU and GPU measurements each have one untimed
warm-up. Transfer measurements are reported separately from processing.

Build and capture a reproducible report:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target gpu_benchmark -j
./build-release/gpu_benchmark --iterations 10
./build-release/gpu_benchmark --iterations 10 --csv > visionforge-4k.csv
```

The report records device, compute capability, driver and CUDA runtime versions,
build type, dimensions, settings, iteration count, latency, throughput, speedup,
transfers, and process-wide VRAM use. Width, height, and iteration count can be
overridden for investigation or CI. Published results should use an otherwise
idle machine, fixed performance power mode, and at least ten iterations.

The automated environment cannot access the workstation RTX 3080, so it does not
claim workstation measurements. Run the commands above on the supported host;
the output explicitly marks the 60 FPS and 2 GiB targets as `PASS` or `MISS`.
