#pragma once

#include <cuda_runtime.h>
#include <nppdefs.h>

#include <sstream>
#include <stdexcept>
#include <string>

namespace plate::core {

inline void cuda_check(const cudaError_t status, const char *expression,
                       const char *file, const int line) {
  if (status == cudaSuccess) {
    return;
  }

  std::ostringstream stream;
  stream << "CUDA failure for " << expression << " at " << file << ":" << line
         << " -> " << cudaGetErrorString(status);
  throw std::runtime_error(stream.str());
}

inline void npp_check(const NppStatus status, const char *expression,
                      const char *file, const int line) {
  if (status == NPP_SUCCESS) {
    return;
  }

  std::ostringstream stream;
  stream << "NPP failure for " << expression << " at " << file << ":" << line
         << " -> status " << static_cast<int>(status);
  throw std::runtime_error(stream.str());
}

}  // namespace plate::core

#define PLATE_CUDA_CHECK(expr) \
  ::plate::core::cuda_check((expr), #expr, __FILE__, __LINE__)
#define PLATE_NPP_CHECK(expr) \
  ::plate::core::npp_check((expr), #expr, __FILE__, __LINE__)
