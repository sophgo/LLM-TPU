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
#pragma once
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "PillowResize.h"

//===------------------------------------------------------------===//
// Open image & video (Official OpenCV implementation)
//===------------------------------------------------------------===//
void opencv_extract_frames(std::vector<cv::Mat> &images, std::string video_file,
                           int num_frames) {
  // Open video
  cv::VideoCapture vidcap(video_file);
  if (!vidcap.isOpened()) {
    std::cerr << "Error: Unable to open video file: " << video_file << std::endl;
    exit(1);
  }

  // Get total frame count
  int frame_count = static_cast<int>(vidcap.get(cv::CAP_PROP_FRAME_COUNT));
  if (frame_count <= 0) {
    std::cerr << "Error: Video file has no frames or failed to retrieve frame count." << std::endl;
    exit(1);
  }

  // Calculate frame indices to extract
  std::vector<int> frame_indices;
  frame_indices.push_back(0); // Always include the first frame
  for (int i = 1; i < num_frames - 1; ++i) {
    frame_indices.push_back(static_cast<int>((frame_count - 1.0) / (num_frames - 1.0) * i));
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
    if (std::find(frame_indices.begin(), frame_indices.end(), count) != frame_indices.end()) {
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
    std::cerr << "Error: Unable to open image file: " << image_path << std::endl;
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
  return static_cast<int>(std::round(static_cast<double>(number) / factor)) * factor;
}

int ceil_by_factor(double number, int factor) {
  return static_cast<int>(std::ceil(number / factor)) * factor;
}

int floor_by_factor(double number, int factor) {
  return static_cast<int>(std::floor(number / factor)) * factor;
}

std::pair<int, int> smart_resize(int height, int width, int factor = IMAGE_FACTOR, 
                                 int min_pixels = MIN_PIXELS, int max_pixels = MAX_PIXELS) {
  // Check aspect ratio
  double aspect_ratio = static_cast<double>(std::max(height, width)) / std::min(height, width);
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
      h_bar = std::max(factor, floor_by_factor(static_cast<double>(height) / beta, factor));
      w_bar = std::max(factor, floor_by_factor(static_cast<double>(width) / beta, factor));
  }
  // Adjust if total pixels are below min_pixels
  else if (h_bar * w_bar < min_pixels) {
      double beta = std::sqrt(static_cast<double>(min_pixels) / (height * width));
      h_bar = std::max(factor, ceil_by_factor(static_cast<double>(height) * beta, factor));
      w_bar = std::max(factor, ceil_by_factor(static_cast<double>(width) * beta, factor));
  }

  return {h_bar, w_bar};
}

cv::Mat convert_to_rgb(const cv::Mat& input_image) {
    // 确保输入是 8-bit 图像
    CV_Assert(input_image.depth() == CV_8U);

    cv::Mat output_image;

    // 处理不同通道数的输入
    switch (input_image.channels()) {
    case 4: {  // BGRA 格式 (常见于 PNG)
        // 分离 BGRA 通道
        std::vector<cv::Mat> bgra_channels;
        cv::split(input_image, bgra_channels);

        // 提取并归一化 Alpha 通道
        cv::Mat alpha;
        bgra_channels[3].convertTo(alpha, CV_32FC1, 1.0/255.0);

        // 创建白色背景 (预乘用)
        cv::Mat white_bg(input_image.size(), CV_32FC3, cv::Scalar(1.0f, 1.0f, 1.0f));

        // 混合每个颜色通道
        std::vector<cv::Mat> blended_channels;
        for (int i = 0; i < 3; ++i) {
            cv::Mat channel;
            bgra_channels[i].convertTo(channel, CV_32FC1, 1.0/255.0);
            cv::Mat blended = channel.mul(alpha) + white_bg.col(i).mul(1.0 - alpha);
            blended_channels.push_back(blended * 255.0);
        }

        // 合并通道并转换类型
        cv::merge(blended_channels, output_image);
        output_image.convertTo(output_image, CV_8UC3);

        // BGR → RGB 转换
        cv::cvtColor(output_image, output_image, cv::COLOR_BGR2RGB);
        break;
    }

    case 1: {  // 灰度图 (直接转为 RGB)
        cv::cvtColor(input_image, output_image, cv::COLOR_GRAY2RGB);
        break;
    }

    case 3: {  // BGR 格式 (OpenCV 默认读取格式)
        cv::cvtColor(input_image, output_image, cv::COLOR_BGR2RGB);
        break;
    }

    default:
        CV_Error(cv::Error::StsBadArg, "Unsupported channel number");
    }

    return output_image;
}

std::vector<float> bicubic_resize(const cv::Mat &image, int resized_height, int resized_width, 
                                  const std::vector<float> &image_mean, const std::vector<float> &image_std) {
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
        processed_image.insert(processed_image.end(), (float*)chw[c].datastart, (float*)chw[c].dataend);
    }

    float sum = 0.f;
    for (float value : processed_image) {
        sum += value;
    }
    // 这里你可以根据需求对 sum 进行进一步处理，比如打印或者返回等
    std::cout << "processed_image 的元素和为: " << sum << std::endl;

    return processed_image;
}