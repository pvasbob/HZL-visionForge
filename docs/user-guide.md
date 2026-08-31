# User guide

## Start and verify

Run `hzl-visionforge --gpu-info`, `--graphics-info`, and `--opencv-info` after
installation. They verify CUDA, the RTX 3080 target, OpenGL 4.5, GLAD, GLFW, and
OpenCV without requiring a media file. Run `hzl-visionforge` to open the desktop
application.

## Open and process media

Use **File > Open Image**, **Open Video**, or **Open Camera**. Images are
normalized to RGBA8. Add filters from **Processing Pipeline**, expand a filter to
edit its values, and use **Up**, **Down**, **Remove**, or the enabled checkbox.
Changes are applied in displayed order.

The available operations cover grayscale, inversion, brightness/contrast,
gamma, blur, sharpen, emboss, Sobel/Laplacian edges, histogram equalization,
tone mapping, and color grading. Alpha is preserved.

## Inspect and export

Choose Original, Processed, Split, Side by side, or Difference above the
viewport. Scroll to zoom around the cursor, drag to pan, double-click or choose
**Fit** to fit the media, and choose **100%** for native pixels.

Use **File > Export Image** to write PNG or JPEG. Export settings control PNG
compression, JPEG quality, and overwrite permission. The currently processed
result is exported when CUDA presentation is active.

The **Profiler** panel reports processing latency, rate, GPU memory, and the
luminance histogram. The **Status** panel retains recent information and errors.

## Troubleshooting

- No window: verify `DISPLAY`/Wayland-X11 access and run `--graphics-info`.
- CUDA unavailable: run `--gpu-info`, verify `nvidia-smi`, driver compatibility,
  and that the RTX 3080 is visible.
- Allocation or processing failure: reduce image dimensions or pipeline length,
  close other GPU applications, and retry.
- Media fails to open: verify the path, permissions, and OpenCV codec support.
- Export fails: select PNG/JPEG, enable overwrite if appropriate, and verify the
  destination directory is writable.

Command-line inspection and export syntax is listed by `hzl-visionforge --help`.
