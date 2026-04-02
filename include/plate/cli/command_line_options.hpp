#pragma once

#include <filesystem>
#include <ostream>
#include <vector>

#include "plate/detector/license_plate_detector.hpp"

namespace plate::cli {

struct CommandLineOptions {
  bool debug_enabled{false};
  detector::DetectorConfig detector_config;
  std::vector<std::filesystem::path> input_paths;
};

void print_usage(const char *program_name, std::ostream &output);

CommandLineOptions parse_command_line(int argc, char **argv);

}  // namespace plate::cli
