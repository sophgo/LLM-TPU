/*****************************************************************************
 *
 *    Copyright (c) 2016-2026 by Sophgo Technologies Inc. All rights reserved.
 *
 *    The material in this file is confidential and contains trade secrets
 *    of Sophgo Technologies Inc. This is proprietary information owned by
 *    Sophgo Technologies Inc. No part of this work may be disclosed,
 *    reproduced, copied, transmitted, or used in any way for any purpose,
 *    without the express written permission of Sophgo Technologies Inc.
 *
 *****************************************************************************/
#ifndef CV_UTILS_H_
#define CV_UTILS_H_

#include <iostream>
#include <string>
#include <vector>

//===------------------------------------------------------------===//
// Resize
//===------------------------------------------------------------===//
const int IMAGE_FACTOR = 28;
const int MIN_PIXELS = 4 * 28 * 28;
const int MAX_PIXELS = 16384 * 28 * 28;
const int MAX_RATIO = 200;

int round_by_factor(int number, int factor) {
  return static_cast<int>(std::round(static_cast<double>(number) / factor)) *
         factor;
}

int ceil_by_factor(double number, int factor) {
  return static_cast<int>(std::ceil(number / factor)) * factor;
}

int floor_by_factor(double number, int factor) {
  return static_cast<int>(std::floor(number / factor)) * factor;
}

std::pair<int, int> _smart_resize(int height, int width,
                                 int factor = IMAGE_FACTOR,
                                 int min_pixels = MIN_PIXELS,
                                 int max_pixels = MAX_PIXELS) {
  // Check aspect ratio
  double aspect_ratio =
      static_cast<double>(std::max(height, width)) / std::min(height, width);
  if (aspect_ratio > MAX_RATIO) {
    throw std::invalid_argument("Absolute aspect ratio must be smaller than " +
                                std::to_string(MAX_RATIO) + ", got " +
                                std::to_string(aspect_ratio));
  }

  // Initial rounding
  int h_bar = std::max(factor, round_by_factor(height, factor));
  int w_bar = std::max(factor, round_by_factor(width, factor));

  // Adjust if total pixels exceed max_pixels
  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt(static_cast<double>(height * width) / max_pixels);
    h_bar = std::max(
        factor, floor_by_factor(static_cast<double>(height) / beta, factor));
    w_bar = std::max(
        factor, floor_by_factor(static_cast<double>(width) / beta, factor));
  }
  // Adjust if total pixels are below min_pixels
  else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
    h_bar = std::max(
        factor, ceil_by_factor(static_cast<double>(height) * beta, factor));
    w_bar = std::max(factor,
                     ceil_by_factor(static_cast<double>(width) * beta, factor));
  }

  return {h_bar, w_bar};
}

void tile(const std::vector<float> &x, std::vector<float> &y, int n) {
  for (int i = 0; i < n; ++i) {
    std::copy(x.begin(), x.end(), y.begin() + i * x.size());
  }
}

void flatten(const std::vector<std::vector<float>> &x, std::vector<float> &y) {
  for (size_t i = 0; i < x.size(); ++i) {
    std::copy(x[i].begin(), x[i].end(), y.begin() + x[i].size());
  }
}

std::vector<int> calc_grid_thw(const std::vector<std::vector<float>> &patches,
                               int resized_height, int resized_width,
                               const Config &config) {
  int grid_t = std::max((int)patches.size() / config.temporal_patch_size, 1);
  int grid_h = resized_height / config.patch_size;
  int grid_w = resized_width / config.patch_size;
  return {grid_t, grid_h, grid_w};
}

// refs:transformers/models/qwen2_vl/image_processing_qwen2_vl.py
std::vector<float>
rearrange_patches(const std::vector<std::vector<float>> &patches,
                  int resized_height, int resized_width, const Config &config) {
  int grid_t = config.grid_thw[0];
  int grid_h = config.grid_thw[1];
  int grid_w = config.grid_thw[2];
  int channel = 3;

  int grid_prod = grid_t * grid_h * grid_w;
  int conv_dim = channel * config.temporal_patch_size * config.patch_size * config.patch_size;
  int total_elements = grid_prod * conv_dim;
  if (grid_prod > config.MAX_PIXELS) {
    throw std::runtime_error("the resized image exceeds MAX_PIXELS, please use --resized_width/--resized_height in pipeline.py.");
  }

  std::vector<float> in(total_elements, 0);
  if (patches.size() == 1) {
    tile(patches[0], in, config.temporal_patch_size);
  } else {
    flatten(patches, in); // multi image
  }
  int merge_h = grid_h / config.spatial_merge_size; // grid_h=12 --> merge_h=6
  int merge_w = grid_w / config.spatial_merge_size; // grid_w=12 --> merge_w=6

  std::vector<float> out(total_elements, 0);
  for (size_t i = 0; i < in.size(); ++i) {
    // (t, s, c, gh, mh, ph, gw, mw, pw)
    int idx = i;
    int pw = idx % config.patch_size;
    idx /= config.patch_size;
    int mw = idx % config.spatial_merge_size;
    idx /= config.spatial_merge_size;
    int gw = idx % merge_w;
    idx /= merge_w;
    int ph = idx % config.patch_size;
    idx /= config.patch_size;
    int mh = idx % config.spatial_merge_size;
    idx /= config.spatial_merge_size;
    int gh = idx % merge_h;
    idx /= merge_h;
    int c = idx % channel;
    idx /= channel;
    int s = idx % config.temporal_patch_size;
    idx /= config.temporal_patch_size;
    int t = idx;

    int new_idx = t;
    new_idx = new_idx * merge_h + gh;
    new_idx = new_idx * merge_w + gw;
    new_idx = new_idx * config.spatial_merge_size + mh;
    new_idx = new_idx * config.spatial_merge_size + mw;
    new_idx = new_idx * channel + c;
    new_idx = new_idx * config.temporal_patch_size + s;
    new_idx = new_idx * config.patch_size + ph;
    new_idx = new_idx * config.patch_size + pw;

    out[new_idx] = in[i];
  }

  return out;
}

void process_video(const std::string &media_path) { return; }

void process_audio(const std::string &media_path) { return; }

void get_media_info(const std::vector<int> &tokens,
                    std::vector<int> &media_offset,
                    std::vector<int> &media_size, int media_token) {
  media_offset.clear();
  media_size.clear();

  size_t size = tokens.size();

  bool in_sequence = false;
  int current_start = 0;
  int current_length = 0;

  for (size_t i = 0; i < size; ++i) {
    if (tokens[i] == media_token) {
      if (!in_sequence) {
        current_start = i;
        current_length = 1;
        in_sequence = true;
      } else {
        current_length++;
      }
    } else {
      if (in_sequence) {
        media_offset.push_back(current_start);
        media_size.push_back(current_length);
        in_sequence = false;
      }
    }
  }

  if (in_sequence) {
    media_offset.push_back(current_start);
    media_size.push_back(current_length);
  }
}

#endif // CV_UTILS_H_