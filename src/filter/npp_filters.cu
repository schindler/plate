#include <cuda_runtime.h>
#include <npp.h>

#include <algorithm>
#include <cmath>
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

struct ResizeDimensions {
  int width{};
  int height{};
};

ResizeDimensions calculate_resize_dimensions(const image::ImageBuffer &source,
                                             const PlateMaskConfig &config) {
  if (config.scale.has_value()) {
    if (*config.scale <= 0.0F) {
      throw std::invalid_argument("Scale must be a positive number.");
    }

    return {std::max(
                1, static_cast<int>(std::lround(source.width * *config.scale))),
            std::max(1, static_cast<int>(
                            std::lround(source.height * *config.scale)))};
  }

  if (source.width <= config.target_max_width &&
      source.height <= config.target_max_height) {
    return {source.width, source.height};
  }

  const float scale_width =
      static_cast<float>(config.target_max_width) / source.width;
  const float scale_height =
      static_cast<float>(config.target_max_height) / source.height;
  const float scale = std::min(scale_width, scale_height);

  return {std::max(1, static_cast<int>(std::lround(source.width * scale))),
          std::max(1, static_cast<int>(std::lround(source.height * scale)))};
}

image::ImageBuffer resize_with_npp(const image::ImageBuffer &source,
                                   const int new_width, const int new_height) {
  if (source.channels != 1 && source.channels != 3) {
    throw std::invalid_argument(
        "NPP resize expects a 1-channel or 3-channel image.");
  }
  if (new_width <= 0 || new_height <= 0) {
    throw std::invalid_argument("NPP resize expects positive dimensions.");
  }

  if (source.width == new_width && source.height == new_height) {
    return source;
  }

  image::ImageBuffer destination;
  destination.width = new_width;
  destination.height = new_height;
  destination.channels = source.channels;
  destination.row_stride = new_width * source.channels;
  destination.pixels.resize(static_cast<std::size_t>(destination.row_stride) *
                            destination.height);

  const NppiSize source_size = {source.width, source.height};
  const NppiRect source_roi = {0, 0, source.width, source.height};
  const NppiSize destination_size = {destination.width, destination.height};
  const NppiRect destination_roi = {0, 0, destination.width,
                                    destination.height};
  const NppStreamContext stream_context = get_npp_stream_context();

  DeviceImage<Npp8u> source_device(source.width * source.channels,
                                   source.height);
  DeviceImage<Npp8u> destination_device(
      destination.width * destination.channels, destination.height);
  source_device.copy_from_host(source.pixels.data(), source.row_stride);

  switch (source.channels) {
    case 1:
      PLATE_NPP_CHECK(nppiResize_8u_C1R_Ctx(
          source_device.data(), source_device.step(), source_size, source_roi,
          destination_device.data(), destination_device.step(),
          destination_size, destination_roi, NPPI_INTER_LINEAR,
          stream_context));
      break;
    case 3:
      PLATE_NPP_CHECK(nppiResize_8u_C3R_Ctx(
          source_device.data(), source_device.step(), source_size, source_roi,
          destination_device.data(), destination_device.step(),
          destination_size, destination_roi, NPPI_INTER_LINEAR,
          stream_context));
      break;
    default:
      throw std::invalid_argument(
          "NPP resize expects a 1-channel or 3-channel image.");
  }

  destination_device.copy_to_host(destination.pixels.data(),
                                  destination.row_stride);
  return destination;
}

}  // namespace

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

  const auto resize_dimensions = calculate_resize_dimensions(image, config);
  image::ImageBuffer source_image =
      resize_with_npp(image, resize_dimensions.width, resize_dimensions.height);

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
