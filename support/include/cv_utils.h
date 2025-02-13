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
#include <string>
#include <vector>
#include "PillowResize.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//===------------------------------------------------------------===//
// Open image & video
//===------------------------------------------------------------===//
std::vector<uint8_t> read_image(const char* filename, int& width, int& height, int& channels) {
    // 读取图像数据（自动解压JPEG）
    unsigned char* data = stbi_load(filename, &width, &height, &channels, STBI_rgb);
    if (!data) {
        std::cerr << "Error: Failed to load image " << filename << std::endl;
        return {};
    }

    // 将数据拷贝到vector中
    size_t data_size = width * height * channels;
    std::vector<uint8_t> image_data(data, data + data_size);

    // 释放stb_image分配的内存
    stbi_image_free(data);

    return image_data;
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

std::vector<float> bicubic_resize(const std::vector<uint8_t>& image, int channels, int height, int width,
                                  int resized_height, int resized_width,
                                  const std::vector<float>& image_mean, const std::vector<float>& image_std) {
    ResizeMat imagein_mat = {
        .width = static_cast<unsigned short>(width),
        .height = static_cast<unsigned short>(height),
        .channels = 3,
        .data = reinterpret_cast<unsigned long>(image.data()),
        .pixel_type = UINT8
    };

    std::vector<uint8_t> resized_image(resized_height * resized_width * channels);
    ResizeMat imagemid_mat = {
        .width = static_cast<unsigned short>(resized_width),
        .height = static_cast<unsigned short>(resized_height),
        .channels = 3,
        .data = reinterpret_cast<unsigned long>(resized_image.data()),
        .pixel_type = UINT8
    };

    PillowResize::resize(imagein_mat, imagemid_mat, PillowResize::INTERPOLATION_BICUBIC);

    std::vector<float> processed_image;
    processed_image.reserve(resized_height * resized_width * channels);
    for (size_t i = 0; i < resized_image.size(); ++i) {
        int c = i % channels;
        float value = static_cast<float>(resized_image[i]);
        // rescale
        value *= 0.00392156862745098f;
        // normalize
        value = (value - image_mean[c]) / image_std[c];
        processed_image.push_back(value);
    }

    return processed_image;
}