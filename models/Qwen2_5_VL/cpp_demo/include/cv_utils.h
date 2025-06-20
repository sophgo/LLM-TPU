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

#include "PillowResize.h"
#include <iostream>
#include <numeric>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

struct Config {
  std::string model_type;
  int SEQLEN;
  int MAX_PREFILL_LENGTH;
  int total_length;
  uint16_t mask_value;

  // vit config
  int max_pos;
  int MAX_PATCHES;
  int MAX_PIXELS;
  int MIN_PIXELS;
  std::vector<int> grid_thw;
  int media_offset;
  int media_size;
  int spatial_merge_size;
  int patch_size;
  int temporal_patch_size;
  int image_token_id;
  int video_token_id;
  int resized_height = 0;
  int resized_width = 0;
};

class Maker {
public:
  explicit Maker(Config &config) : config_(config) {}

  std::vector<int> insert_tokens(const std::vector<int> &raw_tokens,
                                 int media_token_id) {
    if (config_.model_type == "qwen2_vl" ||
        config_.model_type == "qwen2_5_vl") {
      return insert_qwen2vl_tokens(raw_tokens, media_token_id);
    } else {
      throw std::runtime_error("Not support now");
    }
  }

  // ViT
  std::vector<float> make_vit_attention_mask() {
    if (config_.model_type == "qwen2_vl" ||
        config_.model_type == "qwen2_5_vl") {
      return make_qwen2vl_vit_attention_mask();
    } else {
      throw std::runtime_error("Not support now");
    }
  }

  std::vector<int> make_vit_position_id() {
    if (config_.model_type == "qwen2_vl" ||
        config_.model_type == "qwen2_5_vl") {
      return make_qwen2vl_vit_position_id();
    } else {
      throw std::runtime_error("Not support now");
    }
  }

  // Prefill
  std::vector<uint16_t> make_attention_mask() {
    return make_default_attention_mask();
  }

  std::vector<int> make_position_id() {
    if ((config_.model_type == "qwen2_vl" ||
         config_.model_type == "qwen2_5_vl") &&
        config_.grid_thw.size() != 0) {
      return make_qwen2vl_position_id();
    } else {
      return make_default_position_id();
    }
  }

  // Decode
  std::vector<uint16_t> make_next_attention_mask() {
    return make_default_next_attention_mask();
  }

  std::vector<int> make_next_position_id() {
    if ((config_.model_type == "qwen2_vl" ||
         config_.model_type == "qwen2_5_vl") &&
        config_.grid_thw.size() != 0) {
      return make_qwen2vl_next_position_id();
    } else {
      return make_default_next_position_id();
    }
  }

private:
  Config &config_;

  // token processing
  std::vector<int> insert_qwen2vl_tokens(const std::vector<int> &raw_tokens,
                                         int media_token_id) {
    int merge_length = config_.spatial_merge_size * config_.spatial_merge_size;
    const int repeat_num =
        std::accumulate(config_.grid_thw.begin(), config_.grid_thw.end(), 1,
                        std::multiplies<int>()) /
        merge_length;

    std::vector<int> result;
    result.reserve((int)raw_tokens.size() + repeat_num);
    for (int token : raw_tokens) {
      if (token == media_token_id) {
        result.insert(result.end(), repeat_num, media_token_id);
      } else {
        result.push_back(token);
      }
    }
    return result;
  }

  // ViT position utilities
  std::vector<int> make_qwen2vl_vit_position_id() {
    std::vector<int> pos_ids;

    int t = config_.grid_thw[0];
    int h = config_.grid_thw[1];
    int w = config_.grid_thw[2];

    // generate hpos_ids
    std::vector<int> hpos_ids;
    for (int n = 0; n < h; n += config_.spatial_merge_size) {
      for (int _ = 0; _ < w / config_.spatial_merge_size; ++_) {
        hpos_ids.push_back(n);
        hpos_ids.push_back(n);
        hpos_ids.push_back(n + 1);
        hpos_ids.push_back(n + 1);
      }
    }

    // generate wpos_ids
    std::vector<int> wpos_ids;
    for (int _ = 0; _ < h / config_.spatial_merge_size; ++_) {
      for (int e = 0; e < w; e += config_.spatial_merge_size) {
        wpos_ids.push_back(e);
        wpos_ids.push_back(e + 1);
        wpos_ids.push_back(e);
        wpos_ids.push_back(e + 1);
      }
    }

    int valid_vit_pixels = h * w;
    pos_ids.resize(config_.MAX_PATCHES * 2, 0);
    for (int i = 0; i < t; ++i) {
      for (int j = 0; j < valid_vit_pixels; ++j) {
        pos_ids[i * valid_vit_pixels + 2 * j] = hpos_ids[j];
        pos_ids[i * valid_vit_pixels + 2 * j + 1] = wpos_ids[j];
      }
    }

    return pos_ids;
  }

  std::vector<float> make_qwen2vl_vit_attention_mask() {
    std::vector<float> attention_mask;
    int t = config_.grid_thw[0];
    int h = config_.grid_thw[1];
    int w = config_.grid_thw[2];

    // Compute cu_seqlens
    std::vector<int> cu_seqlens(t + 1, 0);
    for (int i = 0; i <= t; ++i) {
      cu_seqlens[i] = h * w * i;
    }

    // Initialize attention_mask with -10000
    attention_mask.resize(config_.MAX_PATCHES * config_.MAX_PATCHES, -10000.);

    // Update attention_mask based on cu_seqlens
    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
      int start = cu_seqlens[i - 1];
      int end = cu_seqlens[i];
      for (int row = start; row < end; ++row) {
        for (int col = start; col < end; ++col) {
          size_t index = row * config_.MAX_PATCHES + col;
          if (index < attention_mask.size()) {
            attention_mask[index] = 0;
          }
        }
      }
    }

    return attention_mask;
  }

  // LLM position utilities (Prefill)
  std::vector<int> make_qwen2vl_position_id() {
    std::vector<int> position_id;
    int text_len = config_.media_offset;

    int llm_grid_t = config_.grid_thw[0];
    int llm_grid_h = config_.grid_thw[1] / config_.spatial_merge_size;
    int llm_grid_w = config_.grid_thw[2] / config_.spatial_merge_size;

    std::vector<int> t_position_id;
    std::vector<int> h_position_id;
    std::vector<int> w_position_id;

    // Populate t_position_id
    for (int i = text_len; i < llm_grid_t + text_len; ++i) {
      for (int j = 0; j < llm_grid_h * llm_grid_w; ++j) {
        t_position_id.push_back(i);
      }
    }

    // Populate h_position_id
    for (int _ = 0; _ < llm_grid_t; ++_) {
      for (int i = 0; i < llm_grid_h; ++i) {
        for (int j = 0; j < llm_grid_w; ++j) {
          h_position_id.push_back(i + text_len);
        }
      }
    }

    // Populate w_position_id
    for (int _ = 0; _ < llm_grid_t; ++_) {
      for (int i = 0; i < llm_grid_h; ++i) {
        for (int j = text_len; j < llm_grid_w + text_len; ++j) {
          w_position_id.push_back(j);
        }
      }
    }

    // Calculate starting index for tail text length
    int st_idx = w_position_id.back() + 1;
    int tail_text_len = config_.total_length - config_.media_size - text_len;

    // Prepare final position ids
    position_id.reserve(config_.SEQLEN * 3);

    // Prepare head position ids
    std::vector<int> head_position_id;
    for (int i = 0; i < text_len; ++i) {
      head_position_id.push_back(i);
    }

    // Prepare tail position ids
    std::vector<int> tail_position_id;
    for (int i = st_idx; i < st_idx + tail_text_len; ++i) {
      tail_position_id.push_back(i);
    }

    // Fill position_id for t
    position_id.insert(
        position_id.end(), head_position_id.begin(),
        head_position_id.end()); // Fill with 0 for range text_len
    position_id.insert(position_id.end(), t_position_id.begin(),
                       t_position_id.end());
    position_id.insert(position_id.end(), tail_position_id.begin(),
                       tail_position_id.end());
    position_id.insert(position_id.end(), config_.SEQLEN - config_.total_length,
                       1); // Fill with 1

    // Fill position_id for h
    position_id.insert(
        position_id.end(), head_position_id.begin(),
        head_position_id.end()); // Fill with 0 for range text_len
    position_id.insert(position_id.end(), h_position_id.begin(),
                       h_position_id.end());
    position_id.insert(position_id.end(), tail_position_id.begin(),
                       tail_position_id.end());
    position_id.insert(position_id.end(), config_.SEQLEN - config_.total_length,
                       1); // Fill with 1

    // Fill position_id for w
    position_id.insert(
        position_id.end(), head_position_id.begin(),
        head_position_id.end()); // Fill with 0 for range text_len
    position_id.insert(position_id.end(), w_position_id.begin(),
                       w_position_id.end());
    position_id.insert(position_id.end(), tail_position_id.begin(),
                       tail_position_id.end());
    position_id.insert(position_id.end(), config_.SEQLEN - config_.total_length,
                       1); // Fill with 1

    config_.max_pos = st_idx + tail_text_len - 1;

    return position_id;
  }

  std::vector<int> make_default_position_id() {
    std::vector<int> position_id(config_.MAX_PREFILL_LENGTH, 0);
    for (int i = 0; i < config_.total_length; i++) {
      position_id[i] = i;
    }
    return position_id;
  }

  std::vector<uint16_t> make_default_attention_mask() {
    std::vector<uint16_t> attention_mask(config_.MAX_PREFILL_LENGTH *
                                             config_.MAX_PREFILL_LENGTH,
                                         config_.mask_value);
    for (int i = 0; i < config_.total_length; i++) {
      for (int j = 0; j < config_.total_length; j++) {
        if (j <= i) {
          attention_mask[i * config_.MAX_PREFILL_LENGTH + j] = 0;
        }
      }
    }

    return attention_mask;
  }

  // LLM position utilities (Decode)
  std::vector<uint16_t> make_default_next_attention_mask() {
    std::vector<uint16_t> attention_mask(config_.SEQLEN + 1, 0);
    for (int i = config_.total_length - 1; i < config_.SEQLEN; i++) {
      attention_mask[i] = config_.mask_value;
    }
    return attention_mask;
  }

  std::vector<int> make_qwen2vl_next_position_id() {
    config_.max_pos += 1;
    return {config_.max_pos, config_.max_pos, config_.max_pos};
  }

  std::vector<int> make_default_next_position_id() {
    return {config_.total_length - 1};
  }
};

//===------------------------------------------------------------===//
// Resize
//===------------------------------------------------------------===//
const int IMAGE_FACTOR = 28;
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

std::pair<int, int> smart_resize(int height, int width, int min_pixels,
                                 int max_pixels, int factor = IMAGE_FACTOR) {
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

std::vector<int> calc_grid_thw(int resized_height, int resized_width,
                               const Config &config) {
  int grid_t = 1; // Default for single image
  int grid_h = resized_height / config.patch_size;
  int grid_w = resized_width / config.patch_size;
  return {grid_t, grid_h, grid_w};
}

// refs:transformers/models/qwen2_vl/image_processing_qwen2_vl.py
void rearrange_patches(const std::vector<float> &image, std::vector<float> &out,
                       int resized_height, int resized_width,
                       const Config &config) {
  int grid_t = config.grid_thw[0];
  int grid_h = config.grid_thw[1];
  int grid_w = config.grid_thw[2];
  int channel = 3;

  int grid_prod = grid_t * grid_h * grid_w;
  int conv_dim = channel * config.temporal_patch_size * config.patch_size *
                 config.patch_size;
  int total_elements = grid_prod * conv_dim;
  assert(grid_prod <= config.MAX_PATCHES);

  std::vector<float> in(total_elements, 0);
  tile(image, in, config.temporal_patch_size);
  int merge_h = grid_h / config.spatial_merge_size; // grid_h=12 --> merge_h=6
  int merge_w = grid_w / config.spatial_merge_size; // grid_w=12 --> merge_w=6
  out.assign(total_elements, 0);
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
}

cv::Mat convert_to_rgb(const cv::Mat &input_image) {
  CV_Assert(input_image.depth() == CV_8U);

  cv::Mat output_image;

  switch (input_image.channels()) {
  case 4: {
    std::vector<cv::Mat> bgra_channels;
    cv::split(input_image, bgra_channels);

    cv::Mat alpha;
    bgra_channels[3].convertTo(alpha, CV_32FC1, 1.0 / 255.0);

    cv::Mat white_bg(input_image.size(), CV_32FC3,
                     cv::Scalar(1.0f, 1.0f, 1.0f));

    std::vector<cv::Mat> blended_channels;
    for (int i = 0; i < 3; ++i) {
      cv::Mat channel;
      bgra_channels[i].convertTo(channel, CV_32FC1, 1.0 / 255.0);
      cv::Mat blended = channel.mul(alpha) + white_bg.col(i).mul(1.0 - alpha);
      blended_channels.push_back(blended * 255.0);
    }

    cv::merge(blended_channels, output_image);
    output_image.convertTo(output_image, CV_8UC3);

    // BGR -> RGB
    cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
    break;
  }

  case 1: { // Gray
    cv::cvtColor(input_image, output_image, cv::COLOR_GRAY2RGB);
    break;
  }

  case 3: { // BGR
    cv::cvtColor(input_image, output_image, cv::COLOR_BGR2RGB);
    break;
  }

  default:
    CV_Error(cv::Error::StsBadArg, "Unsupported channel number");
  }

  return output_image;
}

void bicubic_resize(const cv::Mat &image, std::vector<float> &image_new,
                    int resized_height, int resized_width,
                    const std::vector<float> &image_mean,
                    const std::vector<float> &image_std) {
  auto rgb_image = convert_to_rgb(image);
  auto resized_image =
      PillowResize::resize(rgb_image, cv::Size(resized_width, resized_height),
                           PillowResize::INTERPOLATION_BICUBIC);
  // rescale
  resized_image.convertTo(resized_image, CV_32FC3, 0.00392156862745098, 0);

  // split channel
  std::vector<cv::Mat> rgbChannels(3);
  cv::split(resized_image, rgbChannels);

  // normaliza
  for (int c = 0; c < 3; c++) {
    rgbChannels[c] = (rgbChannels[c] - image_mean[c]) / image_std[c];
  }

  // combine channel
  cv::Mat normalized_image;
  cv::merge(rgbChannels, normalized_image);

  // convert to 1D
  image_new.reserve(resized_height * resized_width * 3);
  std::vector<cv::Mat> chw(3);
  cv::split(normalized_image, chw);
  for (int c = 0; c < 3; c++) {
    image_new.insert(image_new.end(), (float *)chw[c].datastart,
                     (float *)chw[c].dataend);
  }
}

void process_image(std::vector<float> &data, const std::string &media_path,
                   Config &config) {
  cv::Mat image = cv::imread(media_path);
  if (image.empty()) {
    std::cerr << "Error: Unable to open image file: " << media_path
              << std::endl;
    exit(1);
  }

  int width = image.cols;
  int height = image.rows;
  std::vector<float> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
  std::vector<float> image_std = {0.26862954f, 0.26130258f, 0.27577711f};

  auto resized =
      smart_resize(height, width, config.MIN_PIXELS, config.MAX_PIXELS);
  auto resized_height = resized.first;
  auto resized_width = resized.second;
  std::vector<float> image_new;
  bicubic_resize(image, image_new, resized_height, resized_width, image_mean,
                 image_std);

  config.grid_thw = calc_grid_thw(resized_height, resized_width, config);
  rearrange_patches(image_new, data, resized_height, resized_width, config);
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
