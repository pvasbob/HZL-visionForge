#include "hzl/processing/cpu_filters.hpp"
#include "hzl/processing/gpu_pipeline.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
struct Options { int width{3840}; int height{2160}; int iterations{3}; bool csv{false}; };

int positive(const char* text, const std::string_view option) {
    const int value = std::atoi(text);
    if (value <= 0) throw std::invalid_argument{std::string{option} + " must be positive"};
    return value;
}

Options options_from(const int argc, const char* const argv[]) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument{argv[index]};
        if (argument == "--csv") options.csv = true;
        else if ((argument == "--width" || argument == "--height" ||
                  argument == "--iterations") && index + 1 < argc) {
            const int value = positive(argv[++index], argument);
            if (argument == "--width") options.width = value;
            else if (argument == "--height") options.height = value;
            else options.iterations = value;
        } else throw std::invalid_argument{"Usage: gpu_benchmark [--csv] [--width N] [--height N] [--iterations N]"};
    }
    return options;
}

template <typename Operation>
double time_ms(const int iterations, Operation operation) {
    operation();
    const auto start = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) operation();
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - start).count() / iterations;
}

cv::Mat cpu_pipeline(const cv::Mat& source) {
    cv::Mat result = hzl::processing::cpu::brightness_contrast(source, 8.0, 1.08);
    result = hzl::processing::cpu::gaussian_blur(result, 9, 2.0);
    result = hzl::processing::cpu::sharpen(result, 0.75);
    return hzl::processing::cpu::sobel_edges(result, 1.0);
}

void configure(hzl::processing::GpuPipeline& pipeline) {
    static_cast<void>(pipeline.add(hzl::processing::FilterType::brightness_contrast));
    static_cast<void>(pipeline.add(hzl::processing::FilterType::gaussian_blur));
    static_cast<void>(pipeline.add(hzl::processing::FilterType::sharpen));
    static_cast<void>(pipeline.add(hzl::processing::FilterType::sobel));
    auto& adjustment = std::get<hzl::processing::BrightnessContrastParameters>(pipeline.operations()[0].parameters);
    adjustment.brightness = 8.0F; adjustment.contrast = 1.08F;
    auto& gaussian = std::get<hzl::processing::GaussianBlurParameters>(pipeline.operations()[1].parameters);
    gaussian.kernel_size = 9; gaussian.sigma = 2.0F;
    std::get<hzl::processing::SharpenParameters>(pipeline.operations()[2].parameters).amount = 0.75F;
}

struct Results { double cpu{}; double gpu{}; double upload{}; double download{}; std::size_t vram{}; };

Results measure(const Options& options, const cv::Mat& source) {
    hzl::processing::cuda::ImageBuffer input{static_cast<std::size_t>(options.width),
                                             static_cast<std::size_t>(options.height)};
    input.upload(source.data, source.step);
    hzl::processing::GpuPipeline pipeline;
    configure(pipeline);
    cv::Mat cpu_result;
    Results result;
    result.cpu = time_ms(options.iterations, [&] { cpu_result = cpu_pipeline(source); });
    result.gpu = time_ms(options.iterations, [&] { static_cast<void>(pipeline.process(input)); });
    result.upload = time_ms(options.iterations, [&] { input.upload(source.data, source.step); });
    cv::Mat downloaded(options.height, options.width, CV_8UC4);
    const auto& output = pipeline.process(input);
    result.download = time_ms(options.iterations, [&] { output.download(downloaded.data, downloaded.step); });
    std::size_t free = 0;
    std::size_t total = 0;
    hzl::processing::cuda::check(cudaMemGetInfo(&free, &total), "cudaMemGetInfo");
    result.vram = total - free;
    return result;
}

void report(const Options& options, const cudaDeviceProp& device, const int driver,
            const int runtime, const Results& result) {
    const double megapixels = static_cast<double>(options.width) * options.height / 1'000'000.0;
    const double speedup = result.cpu / result.gpu;
    const double fps = 1000.0 / result.gpu;
    const double vram_mib = static_cast<double>(result.vram) / (1024.0 * 1024.0);
    if (options.csv) {
        std::cout << "device,driver,cuda_runtime,build,width,height,iterations,pipeline,cpu_ms,gpu_ms,speedup,gpu_fps,gpu_mpix_s,upload_ms,download_ms,process_vram_mib\n"
                  << '"' << device.name << "\"," << driver << ',' << runtime << ',';
#ifdef NDEBUG
        std::cout << "Release,";
#else
        std::cout << "Debug,";
#endif
        std::cout << options.width << ',' << options.height << ',' << options.iterations
                  << ",\"brightness+gaussian9x9+sharpen+sobel\"," << std::fixed
                  << std::setprecision(3) << result.cpu << ',' << result.gpu << ','
                  << speedup << ',' << fps << ',' << megapixels / (result.gpu / 1000.0)
                  << ',' << result.upload << ',' << result.download << ',' << vram_mib << '\n';
        return;
    }
    std::cout << "HZL-VisionForge reproducible benchmark\nDevice: " << device.name
              << " | compute " << device.major << '.' << device.minor
              << "\nDriver/runtime: " << driver << '/' << runtime << "\nBuild: ";
#ifdef NDEBUG
    std::cout << "Release\n";
#else
    std::cout << "Debug (use Release for published results)\n";
#endif
    std::cout << "Input: " << options.width << 'x' << options.height
              << " RGBA8 deterministic generated image\nPipeline: brightness(8,1.08), Gaussian(9,2), sharpen(0.75), Sobel(1)\n"
              << "Protocol: one warm-up, " << options.iterations << " timed iterations\n"
              << std::fixed << std::setprecision(3) << "CPU: " << result.cpu
              << " ms | GPU: " << result.gpu << " ms | speedup: " << speedup
              << "x\nGPU rate: " << fps << " FPS | "
              << megapixels / (result.gpu / 1000.0) << " MPix/s\nTransfers: upload "
              << result.upload << " ms | download " << result.download
              << " ms\nProcess VRAM in use: " << vram_mib << " MiB\nTargets: 60 FPS "
              << (fps >= 60.0 ? "PASS" : "MISS") << " | under 2 GiB "
              << (result.vram < 2ULL * 1024ULL * 1024ULL * 1024ULL ? "PASS" : "MISS")
              << "\nGPU benchmark: completed\n";
}
}  // namespace

int main(const int argc, const char* const argv[]) {
    try {
        const Options options = options_from(argc, argv);
        int count = 0;
        const cudaError_t status = cudaGetDeviceCount(&count);
        if (status != cudaSuccess || count == 0) {
            static_cast<void>(cudaGetLastError());
            std::cout << "GPU benchmark skipped: "
                      << (status == cudaSuccess ? "no CUDA device" : cudaGetErrorString(status)) << '\n';
            return 77;
        }
        cudaDeviceProp device{};
        hzl::processing::cuda::check(cudaGetDeviceProperties(&device, 0), "cudaGetDeviceProperties");
        int driver = 0;
        int runtime = 0;
        hzl::processing::cuda::check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
        hzl::processing::cuda::check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");
        cv::Mat source(options.height, options.width, CV_8UC4);
        for (int row = 0; row < source.rows; ++row)
            for (int column = 0; column < source.cols; ++column)
                source.at<cv::Vec4b>(row, column) = cv::Vec4b{
                    static_cast<unsigned char>((row * 17 + column * 7) % 256),
                    static_cast<unsigned char>((row * 11 + column * 13) % 256),
                    static_cast<unsigned char>((row * 5 + column * 19) % 256), 255U};
        report(options, device, driver, runtime, measure(options, source));
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "Benchmark failed: " << exception.what() << '\n';
        return 64;
    }
}
