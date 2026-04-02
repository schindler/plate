
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "plate/core/bounding_box.hpp"
#include "plate/core/cuda_check.hpp"
#include "plate/detector/license_plate_detector.hpp"
#include "plate/image/image_buffer.hpp"

namespace plate::detector {
namespace {

// --- CUDA Kernels ---

// 1. Assign each foreground pixel its own 1D index. Background gets -1.
__global__ void init_labels(const uint8_t* img, int* labels, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    int idx = y * w + x;
    // if (img[idx]) printf("%d\n",img[idx]);
    labels[idx] = (img[idx] < 10) ? idx : -1;
  }
}

// 2. Look at 8-way neighbors. Take the smallest label.
__global__ void propagate_labels(int* labels, int w, int h, bool* changed) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    int idx = y * w + x;
    int my_label = labels[idx];

    if (my_label == -1) return;

    int min_label = my_label;

    // 8-way connectivity check
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dx = -1; dx <= 1; ++dx) {
        int nx = x + dx;
        int ny = y + dy;
        if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
          int n_idx = ny * w + nx;
          int neighbor_label = labels[n_idx];
          if (neighbor_label != -1 && neighbor_label < min_label) {
            min_label = neighbor_label;
          }
        }
      }
    }

    // If we found a smaller label, update via atomicMin
    if (min_label < my_label) {
      atomicMin(&labels[my_label], min_label);
      *changed = true;
    }
  }
}

// 3. Flatten the label tree (Path Compression)
__global__ void compress_paths(int* labels, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < w && y < h) {
    int idx = y * w + x;
    int label = labels[idx];
    if (label != -1) {
      int root = label;
      while (root != labels[root]) {
        root = labels[root];
      }
      labels[idx] = root;
    }
  }
}

// 4. Calculate Bounding Boxes and pixel density areas
__global__ void computeBBoxes(const int* labels, int* minX, int* minY,
                              int* maxX, int* maxY, int* counts, int w, int h) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h) {
    int idx = y * w + x;
    int label = labels[idx];

    if (label != -1) {
      atomicMin(&minX[label], x);
      atomicMax(&maxX[label], x);
      atomicMin(&minY[label], y);
      atomicMax(&maxY[label], y);
      atomicAdd(&counts[label], 1);
    }
  }
}

}  // namespace

__host__ std::vector<core::BoundingBox> find_candidates(
    const image::ImageBuffer& img, const float scale,
    const DetectorConfig& config) {
  if (img.empty() || img.channels != 1) {
    return {};
  }

  const int w = img.width;
  const int h = img.height;
  const int num_pixels = w * h;
  const size_t mem_size = static_cast<size_t>(num_pixels) * sizeof(int);

  // 1. Allocate Device Memory
  uint8_t* d_img = nullptr;
  int *d_labels = nullptr, *d_minX = nullptr, *d_minY = nullptr,
      *d_maxX = nullptr, *d_maxY = nullptr, *d_counts = nullptr;
  bool* d_changed = nullptr;

  PLATE_CUDA_CHECK(cudaMalloc(&d_img, img.size_bytes()));
  PLATE_CUDA_CHECK(cudaMalloc(&d_labels, mem_size));
  PLATE_CUDA_CHECK(cudaMalloc(&d_minX, mem_size));
  PLATE_CUDA_CHECK(cudaMalloc(&d_minY, mem_size));
  PLATE_CUDA_CHECK(cudaMalloc(&d_maxX, mem_size));
  PLATE_CUDA_CHECK(cudaMalloc(&d_maxY, mem_size));
  PLATE_CUDA_CHECK(cudaMalloc(&d_counts, mem_size));
  PLATE_CUDA_CHECK(cudaMalloc(&d_changed, sizeof(bool)));

  PLATE_CUDA_CHECK(cudaMemcpy(d_img, img.pixels.data(), img.size_bytes(),
                              cudaMemcpyHostToDevice));

  // Initialize tracking arrays. Huge numbers for Mins, 0 for Maxes and counts.
  std::vector<int> initMin(num_pixels, 999999);
  std::vector<int> initZero(num_pixels, 0);
  PLATE_CUDA_CHECK(
      cudaMemcpy(d_minX, initMin.data(), mem_size, cudaMemcpyHostToDevice));
  PLATE_CUDA_CHECK(
      cudaMemcpy(d_minY, initMin.data(), mem_size, cudaMemcpyHostToDevice));
  PLATE_CUDA_CHECK(
      cudaMemcpy(d_maxX, initZero.data(), mem_size, cudaMemcpyHostToDevice));
  PLATE_CUDA_CHECK(
      cudaMemcpy(d_maxY, initZero.data(), mem_size, cudaMemcpyHostToDevice));
  PLATE_CUDA_CHECK(
      cudaMemcpy(d_counts, initZero.data(), mem_size, cudaMemcpyHostToDevice));

  // 2. Setup Execution Configuration
  dim3 blockSize(16, 16);
  dim3 gridSize((w + blockSize.x - 1) / blockSize.x,
                (h + blockSize.y - 1) / blockSize.y);

  // 3. Run Kernels
  init_labels<<<gridSize, blockSize>>>(d_img, d_labels, w, h);
  PLATE_CUDA_CHECK(cudaGetLastError());

  bool h_changed = true;
  while (h_changed) {
    h_changed = false;
    PLATE_CUDA_CHECK(cudaMemcpy(d_changed, &h_changed, sizeof(bool),
                                cudaMemcpyHostToDevice));

    propagate_labels<<<gridSize, blockSize>>>(d_labels, w, h, d_changed);
    PLATE_CUDA_CHECK(cudaGetLastError());
    compress_paths<<<gridSize, blockSize>>>(d_labels, w, h);
    PLATE_CUDA_CHECK(cudaGetLastError());

    PLATE_CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(bool),
                                cudaMemcpyDeviceToHost));
  }

  computeBBoxes<<<gridSize, blockSize>>>(d_labels, d_minX, d_minY, d_maxX,
                                         d_maxY, d_counts, w, h);
  PLATE_CUDA_CHECK(cudaGetLastError());

  // 4. Copy Bounding Boxes back to Host
  std::vector<int> minX(num_pixels), minY(num_pixels), maxX(num_pixels),
      maxY(num_pixels), counts(num_pixels);
  PLATE_CUDA_CHECK(
      cudaMemcpy(minX.data(), d_minX, mem_size, cudaMemcpyDeviceToHost));
  PLATE_CUDA_CHECK(
      cudaMemcpy(minY.data(), d_minY, mem_size, cudaMemcpyDeviceToHost));
  PLATE_CUDA_CHECK(
      cudaMemcpy(maxX.data(), d_maxX, mem_size, cudaMemcpyDeviceToHost));
  PLATE_CUDA_CHECK(
      cudaMemcpy(maxY.data(), d_maxY, mem_size, cudaMemcpyDeviceToHost));
  PLATE_CUDA_CHECK(
      cudaMemcpy(counts.data(), d_counts, mem_size, cudaMemcpyDeviceToHost));

  // Cleanup Device Memory
  cudaFree(d_img);
  cudaFree(d_labels);
  cudaFree(d_minX);
  cudaFree(d_minY);
  cudaFree(d_maxX);
  cudaFree(d_maxY);
  cudaFree(d_counts);
  cudaFree(d_changed);

  // 5. Filter and Score Candidates on CPU
  std::vector<core::BoundingBox> candidates;

  constexpr float kMinRatioMultiplier = 0.75F;
  constexpr float kMaxRatioMultiplier = 1.625F;
  const float min_ratio = config.expected_plate_ratio * kMinRatioMultiplier;
  const float max_ratio = config.expected_plate_ratio * kMaxRatioMultiplier;

  for (int i = 0; i < num_pixels; ++i) {
    if (counts[i] > 0) {
      const int bw = maxX[i] - minX[i] + 1;
      const int bh = maxY[i] - minY[i] + 1;
      const int b_area = bw * bh;
      const float ratio = static_cast<float>(bw) / static_cast<float>(bh);

      if (b_area >= config.min_area && b_area <= config.max_area) {
        if (ratio >= min_ratio && ratio <= max_ratio) {
          const float density = static_cast<float>(counts[i]) / b_area;
          if (density >= config.min_density) {
            const float ideal_ratio_diff =
                std::abs(ratio - config.expected_plate_ratio);
            const float score = density - (ideal_ratio_diff * 0.1F);
            candidates.push_back({static_cast<int>(minX[i] / scale),
                                  static_cast<int>(minY[i] / scale),
                                  static_cast<int>(bw / scale),
                                  static_cast<int>(bh / scale), score});
          }
        }
      }
    }
  }

  // 6. Sort by Score (Descending) and take top 10
  std::sort(candidates.begin(), candidates.end(),
            [](const auto& a, const auto& b) { return a.score > b.score; });

  if (candidates.size() > 10) {
    candidates.resize(10);
  }

  return candidates;
}

}  // namespace plate::detector
