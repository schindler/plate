#include <cuda_runtime.h>
#include <npp.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

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
      abs(row_x[x]) + abs(row_y[x]);
  row_destination[x] = magnitude >= threshold ? 255 : 0;
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

}  // namespace

image::ImageBuffer build_plate_candidate_mask(
    const image::ImageBuffer &grayscale, const PlateMaskConfig &config) {
  if (grayscale.channels != 1) {
    throw std::invalid_argument(
        "Plate mask generation expects a grayscale image.");
  }

  if (grayscale.row_stride < grayscale.width) {
    throw std::invalid_argument("Invalid grayscale row stride.");
  }

  DeviceImage<Npp8u> source(grayscale.width, grayscale.height);
  DeviceImage<Npp8u> blurred(grayscale.width, grayscale.height);
  DeviceImage<Npp16s> gradient_x(grayscale.width, grayscale.height);
  DeviceImage<Npp16s> gradient_y(grayscale.width, grayscale.height);
  DeviceImage<Npp8u> binary(grayscale.width, grayscale.height);
  DeviceImage<Npp8u> closed(grayscale.width, grayscale.height);
  DeviceImage<Npp8u> temporary(grayscale.width, grayscale.height);

  source.copy_from_host(grayscale.pixels.data(), grayscale.row_stride);

  const NppiSize image_size{grayscale.width, grayscale.height};
  constexpr NppiPoint image_offset{0, 0};

  PLATE_NPP_CHECK(nppiFilterGaussBorder_8u_C1R(
      source.data(), source.step(), image_size, image_offset, blurred.data(),
      blurred.step(), image_size, NPP_MASK_SIZE_5_X_5, NPP_BORDER_REPLICATE));

  PLATE_NPP_CHECK(nppiFilterSobelHorizBorder_8u16s_C1R(
      blurred.data(), blurred.step(), image_size, image_offset,
      gradient_x.data(), gradient_x.step(), image_size, NPP_MASK_SIZE_3_X_3,
      NPP_BORDER_REPLICATE));

  PLATE_NPP_CHECK(nppiFilterSobelVertBorder_8u16s_C1R(
      blurred.data(), blurred.step(), image_size, image_offset,
      gradient_y.data(), gradient_y.step(), image_size, NPP_MASK_SIZE_3_X_3,
      NPP_BORDER_REPLICATE));

  constexpr dim3 block_size{16, 16, 1};
  const dim3 grid_size{
      ((grayscale.width + block_size.x - 1) / block_size.x),
      ((grayscale.height + block_size.y - 1) / block_size.y),
      1};

  gradient_threshold_kernel<<<grid_size, block_size>>>(
      gradient_x.data(), gradient_x.pitch(), gradient_y.data(),
      gradient_y.pitch(), binary.data(), binary.pitch(), grayscale.width,
      grayscale.height, config.gradient_threshold);
  PLATE_CUDA_CHECK(cudaGetLastError());

  launch_rect_kernel(grid_size, block_size, true, binary.data(), binary.pitch(),
                     temporary.data(), temporary.pitch(), grayscale.width,
                     grayscale.height, config.closing_radius_x,
                     config.closing_radius_y);
  PLATE_CUDA_CHECK(cudaGetLastError());

  launch_rect_kernel(grid_size, block_size, false, temporary.data(),
                     temporary.pitch(), closed.data(), closed.pitch(),
                     grayscale.width, grayscale.height, config.closing_radius_x,
                     config.closing_radius_y);
  PLATE_CUDA_CHECK(cudaGetLastError());
  PLATE_CUDA_CHECK(cudaDeviceSynchronize());

  image::ImageBuffer mask;
  mask.width = grayscale.width;
  mask.height = grayscale.height;
  mask.channels = 1;
  mask.row_stride = grayscale.width;
  mask.pixels.resize(static_cast<std::size_t>(mask.row_stride) * mask.height);
  closed.copy_to_host(mask.pixels.data(), mask.row_stride);

  return mask;
}

}  // namespace plate::filter
