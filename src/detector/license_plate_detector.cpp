#include "plate/detector/license_plate_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "plate/core/bounding_box.hpp"
#include "plate/filter/npp_filters.hpp"

namespace plate::detector {
namespace {

struct Component {
  int min_x{};
  int min_y{};
  int max_x{};
  int max_y{};
  int pixels{};
};

[[nodiscard]] int linear_index(const int x, const int y,
                               const int width) noexcept {
  return y * width + x;
}

[[nodiscard]] double score_component(const Component &component,
                                     const int image_width,
                                     const int image_height) {
  const int box_width = component.max_x - component.min_x + 1;
  const int box_height = component.max_y - component.min_y + 1;

  if (box_width <= 0 || box_height <= 0) {
    return 0.0;
  }

  const double aspect_ratio =
      static_cast<double>(box_width) / static_cast<double>(box_height);
  const double bounding_area =
      static_cast<double>(box_width) * static_cast<double>(box_height);
  const double image_area =
      static_cast<double>(image_width) * static_cast<double>(image_height);
  const double area_ratio = bounding_area / image_area;
  const double fill_ratio =
      static_cast<double>(component.pixels) / bounding_area;

  if (box_width < image_width / 8 ||
      box_height < std::max(12, image_height / 40)) {
    return 0.0;
  }

  if (box_height > image_height / 3) {
    return 0.0;
  }

  if (aspect_ratio < 2.0 || aspect_ratio > 6.5) {
    return 0.0;
  }

  if (area_ratio < 0.004 || area_ratio > 0.35) {
    return 0.0;
  }

  if (fill_ratio < 0.18 || fill_ratio > 0.95) {
    return 0.0;
  }

  const double center_x =
      static_cast<double>(component.min_x + component.max_x + 1) / 2.0 /
      image_width;
  const double center_y =
      static_cast<double>(component.min_y + component.max_y + 1) / 2.0 /
      image_height;

  const double aspect_score =
      1.0 - std::min(std::abs(aspect_ratio - 4.0) / 3.0, 1.0);
  const double fill_score =
      1.0 - std::min(std::abs(fill_ratio - 0.5) / 0.5, 1.0);
  const double area_score = std::min(area_ratio / 0.15, 1.0);
  const double center_score =
      1.0 -
      std::min((std::abs(center_x - 0.5) + std::abs(center_y - 0.6)) / 1.1,
               1.0);

  return 0.4 * aspect_score + 0.2 * fill_score + 0.25 * area_score +
         0.15 * center_score;
}

[[nodiscard]] core::BoundingBox expand_component(const Component &component,
                                                 const int image_width,
                                                 const int image_height) {
  const int box_width = component.max_x - component.min_x + 1;
  const int box_height = component.max_y - component.min_y + 1;
  const int pad_x = std::max(2, box_width / 15);
  const int pad_y = std::max(2, box_height / 5);

  return core::clamp_to_image({component.min_x - pad_x, component.min_y - pad_y,
                               box_width + 2 * pad_x, box_height + 2 * pad_y},
                              image_width, image_height);
}

}  // namespace

DetectionResult detect_license_plate(
    const image::ImageBuffer &source_image,
    const filter::FilterStageCallback &stage_callback) {
  if (source_image.channels != 1 && source_image.channels != 3) {
    throw std::invalid_argument(
        "License plate detection expects a 1-channel or 3-channel image.");
  }

  const auto mask =
      filter::build_plate_candidate_mask(source_image, {}, stage_callback);
  const int width = mask.width;
  const int height = mask.height;

  std::vector<std::uint8_t> visited(static_cast<std::size_t>(width) * height,
                                    0);
  std::vector<int> queue;
  queue.reserve(static_cast<std::size_t>(width) * height / 8);

  DetectionResult best{};

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (mask.pixels[static_cast<std::size_t>(y) * mask.row_stride + x] == 0) {
        continue;
      }

      const int seed_index = linear_index(x, y, width);
      if (visited[seed_index] != 0) {
        continue;
      }

      visited[seed_index] = 1;
      queue.clear();
      queue.push_back(seed_index);

      Component component{x, y, x, y, 0};

      for (std::size_t head = 0; head < queue.size(); ++head) {
        const int current_index = queue[head];
        const int current_y = current_index / width;
        const int current_x = current_index % width;

        ++component.pixels;
        component.min_x = std::min(component.min_x, current_x);
        component.min_y = std::min(component.min_y, current_y);
        component.max_x = std::max(component.max_x, current_x);
        component.max_y = std::max(component.max_y, current_y);

        for (int offset_y = -1; offset_y <= 1; ++offset_y) {
          for (int offset_x = -1; offset_x <= 1; ++offset_x) {
            if (offset_x == 0 && offset_y == 0) {
              continue;
            }

            const int next_x = current_x + offset_x;
            const int next_y = current_y + offset_y;

            if (next_x < 0 || next_x >= width || next_y < 0 ||
                next_y >= height) {
              continue;
            }

            const int next_index = linear_index(next_x, next_y, width);
            if (visited[next_index] != 0) {
              continue;
            }

            if (mask.pixels[static_cast<std::size_t>(next_y) * mask.row_stride +
                            next_x] == 0) {
              continue;
            }

            visited[next_index] = 1;
            queue.push_back(next_index);
          }
        }
      }

      const double score = score_component(component, width, height);
      if (score <= best.score) {
        continue;
      }

      best.box = expand_component(component, width, height);
      best.score = score;
    }
  }

  return best;
}

}  // namespace plate::detector
