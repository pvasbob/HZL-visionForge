# Release validation

Create and validate a release build on the supported RTX 3080 workstation:

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
ctest --test-dir build-release --output-on-failure
cpack --config build-release/CPackConfig.cmake
./build-release/gpu_benchmark --iterations 10 --csv > visionforge-4k.csv
```

CTest includes a staged-install smoke test. CPack creates a versioned `.tar.gz`
containing the executable, desktop entry, license, root README, and documentation.

Before tagging, confirm all tests pass on the RTX 3080, the 4K benchmark report is
saved, image/video/camera input and PNG/JPEG export work, comparison modes render
correctly, and a package extracted into a clean prefix starts successfully.

Known initial-release constraints are Linux-only support, an NVIDIA CUDA runtime,
OpenGL 4.5, no RAW workflow, and performance dependent on pipeline complexity.
