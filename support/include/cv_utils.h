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
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

//===------------------------------------------------------------===//
// Open image & video (Official OpenCV implementation)
//===------------------------------------------------------------===//
void opencv_extract_frames(std::vector<cv::Mat> &images, std::string video_file,
                           int num_frames) {
  // Open video
  cv::VideoCapture vidcap(video_file);
  if (!vidcap.isOpened()) {
    std::cerr << "Error: Unable to open video file: " << video_file
              << std::endl;
    exit(1);
  }

  // Get total frame count
  int frame_count = static_cast<int>(vidcap.get(cv::CAP_PROP_FRAME_COUNT));
  if (frame_count <= 0) {
    std::cerr
        << "Error: Video file has no frames or failed to retrieve frame count."
        << std::endl;
    exit(1);
  }

  // Calculate frame indices to extract
  std::vector<int> frame_indices;
  frame_indices.push_back(0); // Always include the first frame
  for (int i = 1; i < num_frames - 1; ++i) {
    frame_indices.push_back(
        static_cast<int>((frame_count - 1.0) / (num_frames - 1.0) * i));
  }
  if (num_frames > 1) {
    frame_indices.push_back(frame_count - 1); // Include the last frame
  }

  // Extract frames
  int count = 0;
  while (true) {
    cv::Mat frame;
    if (!vidcap.read(frame)) {
      break; // End of video
    }

    // Check if the current frame is one of the desired frames
    if (std::find(frame_indices.begin(), frame_indices.end(), count) !=
        frame_indices.end()) {
      cv::Mat rgb_frame;
      cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB); // Convert to RGB
      images.push_back(rgb_frame);
      if (images.size() >= static_cast<size_t>(num_frames)) {
        break;
      }
    }
    ++count;
  }

  vidcap.release();
}

void opencv_read_image(std::vector<cv::Mat> &images, std::string image_path) {
  // Read image
  cv::Mat image = cv::imread(image_path);
  if (image.empty()) {
    std::cerr << "Error: Unable to open image file: " << image_path
              << std::endl;
    exit(1);
  }

  images.push_back(image);
}

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

std::pair<int, int> smart_resize(int height, int width,
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

  int total_elements = grid_t * grid_h * grid_w * channel *
                       config.temporal_patch_size * config.patch_size *
                       config.patch_size;
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

std::vector<float> bicubic_resize(const cv::Mat &image, int resized_height,
                                  int resized_width,
                                  const std::vector<float> &image_mean,
                                  const std::vector<float> &image_std) {
  auto rgb_image = convert_to_rgb(image);
  auto resized_image =
      PillowResize::resize(rgb_image, cv::Size(resized_height, resized_width),
                           PillowResize::INTERPOLATION_BICUBIC);

  // rescale
  resized_image.convertTo(resized_image, CV_32FC1, 0.00392156862745098, 0);

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
  std::vector<float> processed_image;
  processed_image.reserve(resized_height * resized_width * 3);
  std::vector<cv::Mat> chw(3);
  cv::split(normalized_image, chw);
  for (int c = 0; c < 3; c++) {
    processed_image.insert(processed_image.end(), (float *)chw[c].datastart,
                           (float *)chw[c].dataend);
  }
  return processed_image;
}

std::vector<float> process_image(const std::string &media_path,
                                 Config &config) {
  std::vector<cv::Mat> images;
  opencv_read_image(images, media_path);

  std::vector<std::vector<float>> patches;
  int resized_height;
  int resized_width;
  for (size_t i = 0; i < images.size(); i++) {
    auto image = images[i];
    int width = image.cols;
    int height = image.rows;
    if (config.model_type == "qwen2_vl" || config.model_type == "qwen2_5_vl") {
      std::vector<float> image_mean = {0.48145466f, 0.4578275f, 0.40821073f};
      std::vector<float> image_std = {0.26862954f, 0.26130258f, 0.27577711f};

      if (config.resized_height == 0 || config.resized_width == 0) {
        auto resized = smart_resize(height, width);
        resized_height = config.resized_height == 0 ? resized.first : config.resized_height;
        resized_width = config.resized_width == 0 ? resized.second : config.resized_width;
      } else {
        resized_height = config.resized_height;
        resized_width = config.resized_width;
      }
      auto resized_image = bicubic_resize(image, resized_height, resized_width,
                                          image_mean, image_std);
      patches.push_back(resized_image);
    }
  }

  if (patches.size() == 0) {
    throw std::runtime_error("patches are empty!");
  }

  config.grid_thw =
      calc_grid_thw(patches, resized_height, resized_width, config);
  std::vector<float> flatten_patches =
      rearrange_patches(patches, resized_height, resized_width, config);
  return flatten_patches;
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