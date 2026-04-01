#include <cuda_runtime.h>
#include <npp.h>
//#include <nppi_color_conversion.h>
//#include <nppi_filtering_functions.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string_view>

#include "plate/core/cuda_check.hpp"
#include "plate/filter/npp_filters.hpp"

namespace plate::filter {
namespace {

template <typename T>
class DeviceImage {
 public:
  DeviceImage(const int width, const int height)
      : width_(width), height_(height) {
    if (width_ <= 0 || height_ <= 0) {
      throw std::invalid_argument("DeviceImage requires positive dimensions.");
    }

    PLATE_CUDA_CHECK(
        cudaMallocPitch(reinterpret_cast<void **>(&data_), &pitch_,
                        static_cast<std::size_t>(width_) * sizeof(T), height_));
  }

  ~DeviceImage() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }

  DeviceImage(const DeviceImage &) = delete;
  DeviceImage &operator=(const DeviceImage &) = delete;
  DeviceImage(DeviceImage &&) = delete;
  DeviceImage &operator=(DeviceImage &&) = delete;

  [[nodiscard]] T *data() noexcept { return data_; }

  [[nodiscard]] const T *data() const noexcept { return data_; }

  [[nodiscard]] int step() const {
    if (pitch_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::overflow_error("Pitch does not fit in NPP step type.");
    }

    return static_cast<int>(pitch_);
  }

  [[nodiscard]] std::size_t pitch() const noexcept { return pitch_; }

  void copy_from_host(const std::uint8_t *source, const int source_step) const {
    PLATE_CUDA_CHECK(cudaMemcpy2D(data_, pitch_, source, source_step,
                                  static_cast<std::size_t>(width_) * sizeof(T),
                                  height_, cudaMemcpyHostToDevice));
  }

  void copy_to_host(std::uint8_t *destination,
                    const int destination_step) const {
    PLATE_CUDA_CHECK(cudaMemcpy2D(destination, destination_step, data_, pitch_,
                                  static_cast<std::size_t>(width_) * sizeof(T),
                                  height_, cudaMemcpyDeviceToHost));
  }

 private:
  T *data_{};
  std::size_t pitch_{};
  int width_{};
  int height_{};
};

__global__ void gradient_threshold_kernel(
    const Npp16s *gradient_x, const std::size_t gradient_x_pitch,
    const Npp16s *gradient_y, const std::size_t gradient_y_pitch,
    Npp8u *destination, const std::size_t destination_pitch, const int width,
    const int height, const int threshold) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto *row_x = reinterpret_cast<const Npp16s *>(
      reinterpret_cast<const char *>(gradient_x) + y * gradient_x_pitch);
  const auto *row_y = reinterpret_cast<const Npp16s *>(
      reinterpret_cast<const char *>(gradient_y) + y * gradient_y_pitch);
  auto *row_destination = reinterpret_cast<Npp8u *>(
      reinterpret_cast<char *>(destination) + y * destination_pitch);

  const int magnitude =
      abs(static_cast<int>(row_x[x])) + abs(static_cast<int>(row_y[x]));
  row_destination[x] = magnitude >= threshold ? 255 : 0;
}

__global__ void gradient_visualization_kernel(
    const Npp16s *gradient_x, const std::size_t gradient_x_pitch,
    const Npp16s *gradient_y, const std::size_t gradient_y_pitch,
    Npp8u *destination, const std::size_t destination_pitch, const int width,
    const int height, const int scale_divisor) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  const auto *row_x = reinterpret_cast<const Npp16s *>(
      reinterpret_cast<const char *>(gradient_x) + y * gradient_x_pitch);
  const auto *row_y = reinterpret_cast<const Npp16s *>(
      reinterpret_cast<const char *>(gradient_y) + y * gradient_y_pitch);
  auto *row_destination = reinterpret_cast<Npp8u *>(
      reinterpret_cast<char *>(destination) + y * destination_pitch);

  const int magnitude =
      abs(static_cast<int>(row_x[x])) + abs(static_cast<int>(row_y[x]));
  const int scaled_value = magnitude / max(1, scale_divisor);
  row_destination[x] = static_cast<Npp8u>(min(255, scaled_value));
}

__global__ void dilate_rect_kernel(const Npp8u *source,
                                   const std::size_t source_pitch,
                                   Npp8u *destination,
                                   const std::size_t destination_pitch,
                                   const int width, const int height,
                                   const int radius_x, const int radius_y) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int value = 0;
  for (int kernel_y = max(0, y - radius_y);
       kernel_y <= min(height - 1, y + radius_y); ++kernel_y) {
    const auto *row = reinterpret_cast<const Npp8u *>(
        reinterpret_cast<const char *>(source) + kernel_y * source_pitch);
    for (int kernel_x = max(0, x - radius_x);
         kernel_x <= min(width - 1, x + radius_x); ++kernel_x) {
      if (row[kernel_x] != 0) {
        value = 255;
        break;
      }
    }

    if (value != 0) {
      break;
    }
  }

  auto *row_destination = reinterpret_cast<Npp8u *>(
      reinterpret_cast<char *>(destination) + y * destination_pitch);
  row_destination[x] = static_cast<Npp8u>(value);
}

__global__ void erode_rect_kernel(const Npp8u *source,
                                  const std::size_t source_pitch,
                                  Npp8u *destination,
                                  const std::size_t destination_pitch,
                                  const int width, const int height,
                                  const int radius_x, const int radius_y) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height) {
    return;
  }

  int value = 255;
  for (int kernel_y = max(0, y - radius_y);
       kernel_y <= min(height - 1, y + radius_y); ++kernel_y) {
    const auto *row = reinterpret_cast<const Npp8u *>(
        reinterpret_cast<const char *>(source) + kernel_y * source_pitch);
    for (int kernel_x = max(0, x - radius_x);
         kernel_x <= min(width - 1, x + radius_x); ++kernel_x) {
      if (row[kernel_x] == 0) {
        value = 0;
        break;
      }
    }

    if (value == 0) {
      break;
    }
  }

  auto *row_destination = reinterpret_cast<Npp8u *>(
      reinterpret_cast<char *>(destination) + y * destination_pitch);
  row_destination[x] = static_cast<Npp8u>(value);
}

void launch_rect_kernel(dim3 grid, dim3 block, const bool is_dilation,
                        const Npp8u *source, const std::size_t source_pitch,
                        Npp8u *destination, const std::size_t destination_pitch,
                        const int width, const int height, const int radius_x,
                        const int radius_y) {
  if (is_dilation) {
    dilate_rect_kernel<<<grid, block>>>(source, source_pitch, destination,
                                        destination_pitch, width, height,
                                        radius_x, radius_y);
  } else {
    erode_rect_kernel<<<grid, block>>>(source, source_pitch, destination,
                                       destination_pitch, width, height,
                                       radius_x, radius_y);
  }
}

NppStreamContext get_npp_stream_context() {
  int cuda_device_id = 0;
  cudaDeviceProp device_properties{};
  NppStreamContext stream_context{};
  PLATE_CUDA_CHECK(cudaGetDevice(&cuda_device_id));
  PLATE_CUDA_CHECK(cudaGetDeviceProperties(&device_properties, cuda_device_id));

  stream_context.hStream = cudaStream_t{};
  stream_context.nCudaDeviceId = cuda_device_id;
  stream_context.nMultiProcessorCount = device_properties.multiProcessorCount;
  stream_context.nMaxThreadsPerMultiProcessor =
      device_properties.maxThreadsPerMultiProcessor;
  stream_context.nMaxThreadsPerBlock = device_properties.maxThreadsPerBlock;
  stream_context.nSharedMemPerBlock = device_properties.sharedMemPerBlock;
  stream_context.nCudaDevAttrComputeCapabilityMajor = device_properties.major;
  stream_context.nCudaDevAttrComputeCapabilityMinor = device_properties.minor;
  stream_context.nStreamFlags = 0;
  stream_context.nReserved0 = 0;
  return stream_context;
}

image::ImageBuffer copy_device_image_to_host(const DeviceImage<Npp8u> &device,
                                             const int width,
                                             const int height) {
  image::ImageBuffer host_image;
  host_image.width = width;
  host_image.height = height;
  host_image.channels = 1;
  host_image.row_stride = width;
  host_image.pixels.resize(static_cast<std::size_t>(host_image.row_stride) *
                           host_image.height);
  device.copy_to_host(host_image.pixels.data(), host_image.row_stride);
  return host_image;
}

void emit_stage_if_requested(const DeviceImage<Npp8u> &device_image,
                             const int width, const int height,
                             const std::string_view stage_name,
                             const FilterStageCallback &stage_callback) {
  if (!stage_callback) {
    return;
  }

  stage_callback(stage_name,
                 copy_device_image_to_host(device_image, width, height));
}

}  // namespace

std::tuple<int, int> calculate_down_scale(int srcW, int srcH, int maxW = 1080, int maxH = 600) {
    // If image is already smaller than target, we don't need to downscale
    if (srcW <= maxW && srcH <= maxH) {
        return {srcW, srcH};
    }

    // Calculate scaling factors
    float scaleW = static_cast<float>(maxW) / srcW;
    float scaleH = static_cast<float>(maxH) / srcH;

    // Use the smaller scale to ensure it fits within BOTH bounds
    float scale = std::min(scaleW, scaleH);

    printf("Scale: %.02f\n", scale);

    return {
        static_cast<int>(srcW * scale),
        static_cast<int>(srcH * scale)
    };
}

image::ImageBuffer resize3ChannelNPP(const image::ImageBuffer& src, int newWidth, int newHeight) {
    image::ImageBuffer dst;
    dst.width = newWidth;
    dst.height = newHeight;
    dst.channels = 3; // Ensure this is set to 3
    dst.row_stride = newWidth * 3; // 3 bytes per pixel
    dst.pixels.resize(dst.row_stride * dst.height);

    NppiSize srcSize = { src.width, src.height };
    NppiRect srcRoi  = { 0, 0, src.width, src.height };
    
    NppiSize dstSize = { dst.width, dst.height };
    NppiRect dstRoi  = { 0, 0, dst.width, dst.height };

    uint8_t *d_src, *d_dst;
    size_t srcStep, dstStep;
    
    // IMPORTANT: cudaMallocPitch width is in BYTES. 
    // So we multiply width by 3 for C3.
    cudaMallocPitch((void**)&d_src, &srcStep, src.width * 3, src.height);
    cudaMallocPitch((void**)&d_dst, &dstStep, dst.width * 3, dst.height);

    // Copy source to device
    cudaMemcpy2D(d_src, srcStep, src.pixels.data(), src.row_stride, 
                 src.width * 3, src.height, cudaMemcpyHostToDevice);

    // Execute 3-Channel Resize
    // Note: C3R instead of C1R
    PLATE_NPP_CHECK(nppiResize_8u_C3R(
        d_src, (int)srcStep, srcSize, srcRoi,
        d_dst, (int)dstStep, dstSize, dstRoi,
        NPPI_INTER_LINEAR
    ));

    // Copy result back to host
    cudaMemcpy2D(dst.pixels.data(), dst.row_stride, d_dst, dstStep, 
                 dst.width * 3, dst.height, cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);

    return dst;
}

__host__ image::ImageBuffer resize(const image::ImageBuffer& src, int newWidth, int newHeight) {
    image::ImageBuffer dst;
    dst.width = newWidth;
    dst.height = newHeight;
    dst.channels = src.channels;
    dst.row_stride = newWidth * src.channels; // Simple stride
    dst.pixels.resize(dst.row_stride * dst.height);

    // 1. Setup NPP Sizes and Rects
    NppiSize srcSize = { src.width, src.height };
    NppiRect srcRoi  = { 0, 0, src.width, src.height };
    
    NppiSize dstSize = { dst.width, dst.height };
    NppiRect dstRoi  = { 0, 0, dst.width, dst.height };

    // 2. Allocate Device Memory
    uint8_t *d_src, *d_dst;
    size_t srcStep, dstStep;
    
    // Using cudaMallocPitch is recommended for NPP to ensure memory alignment
    cudaMallocPitch((void**)&d_src, &srcStep, src.width, src.height);
    cudaMallocPitch((void**)&d_dst, &dstStep, dst.width, dst.height);

    // 3. Copy source from Host to Device
    // We use cudaMemcpy2D because the device might have added padding (pitch)
    cudaMemcpy2D(d_src, srcStep, src.pixels.data(), src.row_stride, 
                 src.width, src.height, cudaMemcpyHostToDevice);

    // 4. Execute NPP Resize
    // _8u (unsigned 8-bit), _C1 (1 channel), _R (ROI-based)
    PLATE_NPP_CHECK(nppiResize_8u_C3R(
        d_src, srcStep, srcSize, srcRoi,
        d_dst, dstStep, dstSize, dstRoi,
        NPPI_INTER_LINEAR
    ));

    // 5. Copy back to Host
    cudaMemcpy2D(dst.pixels.data(), dst.row_stride, d_dst, dstStep, 
                 dst.width, dst.height, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_src);
    cudaFree(d_dst);

    return dst;
}

image::ImageBuffer build_plate_candidate_mask(
    const image::ImageBuffer &image, const PlateMaskConfig &config,
    const FilterStageCallback &stage_callback) {
  if (image.channels != 1 && image.channels != 3) {
    throw std::invalid_argument(
        "Plate mask generation expects a 1-channel or 3-channel image.");
  }

  if (image.row_stride < image.width * image.channels) {
    throw std::invalid_argument("Invalid source image row stride.");
  }


  auto [w,h]=calculate_down_scale(image.width, image.height);
  image::ImageBuffer source_image = resize3ChannelNPP(image, w, h);

  constexpr Npp32f kBgrToGrayCoefficients[3] = {0.114F, 0.587F, 0.299F};
  constexpr int kGradientVisualizationScale = 4;
  const NppStreamContext stream_context = get_npp_stream_context();

  DeviceImage<Npp8u> grayscale(source_image.width, source_image.height);
  DeviceImage<Npp8u> blurred(source_image.width, source_image.height);
  DeviceImage<Npp16s> gradient_x(source_image.width, source_image.height);
  DeviceImage<Npp16s> gradient_y(source_image.width, source_image.height);
  DeviceImage<Npp8u> sobel_edge(source_image.width, source_image.height);
  DeviceImage<Npp8u> binary(source_image.width, source_image.height);
  DeviceImage<Npp8u> closed(source_image.width, source_image.height);
  DeviceImage<Npp8u> temporary(source_image.width, source_image.height);

  const NppiSize image_size{source_image.width, source_image.height};
  constexpr NppiPoint image_offset{0, 0};

  if (source_image.channels == 3) {
    DeviceImage<Npp8u> source_color(source_image.width * source_image.channels,
                                    source_image.height);
    source_color.copy_from_host(source_image.pixels.data(),
                                source_image.row_stride);
    PLATE_NPP_CHECK(nppiColorToGray_8u_C3C1R_Ctx(
        source_color.data(), source_color.step(), grayscale.data(),
        grayscale.step(), image_size, kBgrToGrayCoefficients, stream_context));
  } else {
    DeviceImage<Npp8u> source_grayscale(source_image.width,
                                        source_image.height);
    source_grayscale.copy_from_host(source_image.pixels.data(),
                                    source_image.row_stride);
    PLATE_CUDA_CHECK(cudaMemcpy2D(
        grayscale.data(), grayscale.pitch(), source_grayscale.data(),
        source_grayscale.pitch(), static_cast<std::size_t>(source_image.width),
        source_image.height, cudaMemcpyDeviceToDevice));
  }
  emit_stage_if_requested(grayscale, source_image.width, source_image.height,
                          "grayscale", stage_callback);

  PLATE_NPP_CHECK(nppiFilterGaussBorder_8u_C1R_Ctx(
      grayscale.data(), grayscale.step(), image_size, image_offset,
      blurred.data(), blurred.step(), image_size, NPP_MASK_SIZE_5_X_5,
      NPP_BORDER_REPLICATE, stream_context));
  emit_stage_if_requested(blurred, source_image.width, source_image.height,
                          "gauss-blur", stage_callback);

  PLATE_NPP_CHECK(nppiFilterSobelHorizBorder_8u16s_C1R_Ctx(
      blurred.data(), blurred.step(), image_size, image_offset,
      gradient_x.data(), gradient_x.step(), image_size, NPP_MASK_SIZE_3_X_3,
      NPP_BORDER_REPLICATE, stream_context));

  PLATE_NPP_CHECK(nppiFilterSobelVertBorder_8u16s_C1R_Ctx(
      blurred.data(), blurred.step(), image_size, image_offset,
      gradient_y.data(), gradient_y.step(), image_size, NPP_MASK_SIZE_3_X_3,
      NPP_BORDER_REPLICATE, stream_context));

  constexpr dim3 block_size{16, 16, 1};
  const dim3 grid_size{
      ((source_image.width + block_size.x - 1) / block_size.x),
      ((source_image.height + block_size.y - 1) / block_size.y), 1};

  gradient_visualization_kernel<<<grid_size, block_size>>>(
      gradient_x.data(), gradient_x.pitch(), gradient_y.data(),
      gradient_y.pitch(), sobel_edge.data(), sobel_edge.pitch(),
      source_image.width, source_image.height, kGradientVisualizationScale);
  PLATE_CUDA_CHECK(cudaGetLastError());
  emit_stage_if_requested(sobel_edge, source_image.width, source_image.height,
                          "sobel-edge", stage_callback);

  gradient_threshold_kernel<<<grid_size, block_size>>>(
      gradient_x.data(), gradient_x.pitch(), gradient_y.data(),
      gradient_y.pitch(), binary.data(), binary.pitch(), source_image.width,
      source_image.height, config.gradient_threshold);
  PLATE_CUDA_CHECK(cudaGetLastError());
  emit_stage_if_requested(binary, source_image.width, source_image.height,
                          "binary-mask", stage_callback);

  launch_rect_kernel(grid_size, block_size, true, binary.data(), binary.pitch(),
                     temporary.data(), temporary.pitch(), source_image.width,
                     source_image.height, config.closing_radius_x,
                     config.closing_radius_y);
  PLATE_CUDA_CHECK(cudaGetLastError());

  launch_rect_kernel(grid_size, block_size, false, temporary.data(),
                     temporary.pitch(), closed.data(), closed.pitch(),
                     source_image.width, source_image.height,
                     config.closing_radius_x, config.closing_radius_y);
  PLATE_CUDA_CHECK(cudaGetLastError());
  PLATE_CUDA_CHECK(cudaDeviceSynchronize());
  emit_stage_if_requested(closed, source_image.width, source_image.height,
                          "closed-mask", stage_callback);

  return copy_device_image_to_host(closed, source_image.width,
                                   source_image.height);
}

}  // namespace plate::filter
