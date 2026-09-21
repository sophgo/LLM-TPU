//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//
//
// InternVL image / video preprocessing, ported from
// models/InternVL3/python_demo/pipeline.py (dynamic_preprocess +
// build_transform). Produces plain [num_tiles, 3, image_size, image_size]
// float tiles, ImageNet-normalized, with no patchify -- this is what the
// InternViT "vit" net consumes (one 448x448 tile per launch).
//
//===----------------------------------------------------------------------===//
#ifndef CV_UTILS_H_
#define CV_UTILS_H_

#include "PillowResize.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <set>
#include <string>
#include <utility>
#include <vector>

struct PreConfig {
  int image_size = 448;   // tile size (InternViT input side)
  int max_num = 4;        // max tiles for a still image
  int num_segments = 8;   // sampled frames for a video
  std::vector<float> mean = {0.485f, 0.456f, 0.406f}; // IMAGENET_MEAN
  std::vector<float> std = {0.229f, 0.224f, 0.225f};  // IMAGENET_STD
};

// Convert an OpenCV image (BGR / BGRA / gray) to a CV_8UC3 RGB image.
// cv::imread's default (3-channel BGR) mirrors PIL's Image.convert('RGB'),
// which drops any alpha channel.
inline cv::Mat convert_to_rgb(const cv::Mat &input_image) {
  CV_Assert(input_image.depth() == CV_8U);
  cv::Mat output_image;
  switch (input_image.channels()) {
  case 4: {
    // alpha blend over white: out = bgr * alpha + 255 * (1 - alpha)
    std::vector<cv::Mat> bgra_channels;
    cv::split(input_image, bgra_channels);
    cv::Mat alpha, inv_alpha;
    bgra_channels[3].convertTo(alpha, CV_32FC1, 1.0 / 255.0);
    cv::subtract(cv::Scalar(1.0), alpha, inv_alpha);
    std::vector<cv::Mat> blended_channels(3);
    for (int i = 0; i < 3; ++i) {
      cv::Mat channel;
      bgra_channels[i].convertTo(channel, CV_32FC1);
      blended_channels[i] = channel.mul(alpha) + inv_alpha * 255.0;
    }
    cv::merge(blended_channels, output_image);
    output_image.convertTo(output_image, CV_8UC3);
    cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
    break;
  }
  case 1:
    cv::cvtColor(input_image, output_image, cv::COLOR_GRAY2RGB);
    break;
  case 3:
    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2RGB);
    break;
  default:
    CV_Error(cv::Error::StsBadArg, "Unsupported channel number");
  }
  return output_image;
}

// refs: pipeline.py find_closest_aspect_ratio
inline std::pair<int, int>
find_closest_aspect_ratio(double aspect_ratio,
                          const std::vector<std::pair<int, int>> &target_ratios,
                          int width, int height, int image_size) {
  double best_ratio_diff = std::numeric_limits<double>::infinity();
  std::pair<int, int> best_ratio = {1, 1};
  double area = static_cast<double>(width) * height;
  for (const auto &ratio : target_ratios) {
    double target = static_cast<double>(ratio.first) / ratio.second;
    double ratio_diff = std::fabs(aspect_ratio - target);
    if (ratio_diff < best_ratio_diff) {
      best_ratio_diff = ratio_diff;
      best_ratio = ratio;
    } else if (ratio_diff == best_ratio_diff) {
      if (area > 0.5 * image_size * image_size * ratio.first * ratio.second) {
        best_ratio = ratio;
      }
    }
  }
  return best_ratio;
}

// refs: pipeline.py dynamic_preprocess. Splits `rgb` (CV_8UC3, RGB) into
// image_size x image_size tiles following the closest aspect ratio, optionally
// appending a thumbnail of the whole image. Tiles come back as CV_8UC3 RGB.
inline void dynamic_preprocess(const cv::Mat &rgb, int min_num, int max_num,
                               int image_size, bool use_thumbnail,
                               std::vector<cv::Mat> &out) {
  const int orig_width = rgb.cols;
  const int orig_height = rgb.rows;
  const double aspect_ratio =
      static_cast<double>(orig_width) / orig_height;

  // target_ratios = sorted unique (i, j) with min_num <= i*j <= max_num
  std::set<std::pair<int, int>> ratio_set; // lexicographically ordered
  for (int n = min_num; n <= max_num; ++n) {
    for (int i = 1; i <= n; ++i) {
      for (int j = 1; j <= n; ++j) {
        if (i * j <= max_num && i * j >= min_num) {
          ratio_set.insert({i, j});
        }
      }
    }
  }
  std::vector<std::pair<int, int>> target_ratios(ratio_set.begin(),
                                                 ratio_set.end());
  // stable so that ties on i*j keep the lexicographic (set) order
  std::stable_sort(target_ratios.begin(), target_ratios.end(),
                   [](const std::pair<int, int> &a,
                      const std::pair<int, int> &b) {
                     return a.first * a.second < b.first * b.second;
                   });

  auto target_aspect_ratio = find_closest_aspect_ratio(
      aspect_ratio, target_ratios, orig_width, orig_height, image_size);

  const int target_width = image_size * target_aspect_ratio.first;
  const int target_height = image_size * target_aspect_ratio.second;
  const int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

  cv::Mat resized_img =
      PillowResize::resize(rgb, cv::Size(target_width, target_height),
                           PillowResize::INTERPOLATION_BICUBIC);

  const int cols = target_width / image_size;
  out.clear();
  out.reserve(blocks + 1);
  for (int i = 0; i < blocks; ++i) {
    const int x0 = (i % cols) * image_size;
    const int y0 = (i / cols) * image_size;
    out.push_back(
        resized_img(cv::Rect(x0, y0, image_size, image_size)).clone());
  }
  assert(static_cast<int>(out.size()) == blocks);
  if (use_thumbnail && blocks != 1) {
    cv::Mat thumbnail =
        PillowResize::resize(rgb, cv::Size(image_size, image_size),
                             PillowResize::INTERPOLATION_BICUBIC);
    out.push_back(thumbnail);
  }
}

// refs: pipeline.py build_transform (ToTensor + Normalize). Rescales a CV_8UC3
// RGB tile to [0,1], applies ImageNet (x-mean)/std per channel, and writes it
// out in CHW order at `out` (3 * image_size * image_size floats).
inline void normalize_tile(const cv::Mat &tile_rgb_u8, float *out,
                           const PreConfig &cfg) {
  cv::Mat tile_f;
  tile_rgb_u8.convertTo(tile_f, CV_32FC3, 1.0 / 255.0);
  std::vector<cv::Mat> chw(3);
  cv::split(tile_f, chw);
  const size_t plane = static_cast<size_t>(tile_rgb_u8.rows) * tile_rgb_u8.cols;
  for (int c = 0; c < 3; ++c) {
    chw[c].convertTo(chw[c], CV_32FC1, 1.0 / cfg.std[c],
                     -cfg.mean[c] / cfg.std[c]);
    std::memcpy(out + c * plane, chw[c].ptr<float>(), plane * sizeof(float));
  }
}

// refs: pipeline.py process_image. Returns the number of tiles (num_patches,
// thumbnail included) and fills `pixel_values` with a flat
// [num_tiles, 3, image_size, image_size] float buffer. Returns -1 on error.
inline int process_image(std::vector<float> &pixel_values,
                         const std::string &path, const PreConfig &cfg) {
  cv::Mat raw = cv::imread(path);
  if (raw.empty()) {
    std::cerr << "Error: Unable to open image file: " << path << std::endl;
    return -1;
  }
  cv::Mat rgb = convert_to_rgb(raw);
  std::vector<cv::Mat> tiles;
  dynamic_preprocess(rgb, 1, cfg.max_num, cfg.image_size, true, tiles);

  const int num_tiles = static_cast<int>(tiles.size());
  const size_t plane = static_cast<size_t>(cfg.image_size) * cfg.image_size;
  const size_t tile_stride = 3 * plane;
  pixel_values.resize(static_cast<size_t>(num_tiles) * tile_stride);
  for (int t = 0; t < num_tiles; ++t) {
    normalize_tile(tiles[t], pixel_values.data() + t * tile_stride, cfg);
  }
  return num_tiles;
}

// refs: pipeline.py get_index + process_video. Samples num_segments frames
// evenly across the clip, turns each into a single image_size tile (max_num=1,
// no thumbnail) and concatenates them. Fills `num_patches_list` (one entry per
// sampled frame, always 1). Returns the number of frames, or -1 on error.
inline int process_video(std::vector<float> &pixel_values,
                         std::vector<int> &num_patches_list,
                         const std::string &path, const PreConfig &cfg) {
  cv::VideoCapture cap(path);
  if (!cap.isOpened()) {
    std::cerr << "Error: Unable to open video file: " << path << std::endl;
    return -1;
  }
  const double total_frames = cap.get(cv::CAP_PROP_FRAME_COUNT);
  const int max_frame = std::max(0, static_cast<int>(total_frames) - 1);
  const int num_segments = cfg.num_segments;
  const double seg_size = static_cast<double>(max_frame) / num_segments;

  std::vector<int> frame_indices;
  frame_indices.reserve(num_segments);
  for (int idx = 0; idx < num_segments; ++idx) {
    // int(start_idx + seg_size/2 + round(seg_size*idx)), start_idx = 0
    double v = seg_size / 2.0 + static_cast<double>(std::llround(seg_size * idx));
    frame_indices.push_back(static_cast<int>(v));
  }

  const size_t plane = static_cast<size_t>(cfg.image_size) * cfg.image_size;
  const size_t tile_stride = 3 * plane;
  pixel_values.clear();
  num_patches_list.clear();
  cv::Mat frame;
  for (int idx : frame_indices) {
    cap.set(cv::CAP_PROP_POS_FRAMES, idx);
    if (!cap.read(frame)) {
      continue;
    }
    cv::Mat rgb = convert_to_rgb(frame);
    std::vector<cv::Mat> tiles;
    dynamic_preprocess(rgb, 1, 1, cfg.image_size, true, tiles);
    for (const auto &tile : tiles) {
      const size_t base = pixel_values.size();
      pixel_values.resize(base + tile_stride);
      normalize_tile(tile, pixel_values.data() + base, cfg);
    }
    num_patches_list.push_back(static_cast<int>(tiles.size()));
  }
  cap.release();
  return static_cast<int>(num_patches_list.size());
}

#endif // CV_UTILS_H_
