#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace plate::image {

struct ImageBuffer {
  int width{};
  int height{};
  int channels{};
  int row_stride{};
  std::vector<std::uint8_t> pixels;

  [[nodiscard]] bool empty() const noexcept {
    return width <= 0 || height <= 0 || channels <= 0 || pixels.empty();
  }

  [[nodiscard]] std::size_t size_bytes() const noexcept {
    return pixels.size();
  }
};

enum class ImageReadMode { kColor, kGrayscale };

}  // namespace plate::image
