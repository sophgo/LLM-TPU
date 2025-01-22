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
  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Error: Unable to open image file: " << image_path << std::endl;
    exit(1);
  }

  // Convert to RGB
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  images.push_back(image);
}


std::vector<float> bicubic_resize(const cv::Mat &image, int resized_height, int resized_width) {
  const int image_area = resized_height * resized_width;
  size_t single_chn_size = image_area * sizeof(float);
  std::vector<float> resized_image(image_area * 3);
  
  // preprocess
  auto pillow_image =
      PillowResize::resize(image, cv::Size(resized_height, resized_width),
                            PillowResize::INTERPOLATION_BICUBIC);
  pillow_image.convertTo(pillow_image, CV_32FC1, 0.00392156862745098, 0);
  std::vector<cv::Mat> rgbChannels(3);
  cv::split(pillow_image, rgbChannels);
  for (int c = 0; c < 3; c++) {
    rgbChannels[c].convertTo(rgbChannels[c], CV_32FC1, 1.0 / 0.5,
                              (0.0 - 0.5) / 0.5);
  }

  // convert to array
  memcpy((void *)resized_image.data(), (float *)rgbChannels[0].data,
          single_chn_size);
  memcpy((void *)(resized_image.data() + image_area),
          (float *)rgbChannels[1].data, single_chn_size);
  memcpy((void *)(resized_image.data() + image_area * 2),
          (float *)rgbChannels[2].data, single_chn_size);
  return resized_image;
}