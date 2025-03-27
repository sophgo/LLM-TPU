#include "vlm_interface.hpp"
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

void opencv_extract_frames(std::vector<cv::Mat> &images, std::string video_file,
                           int num_frames, int dev_id = 0) {
  // open video
  auto vidcap = cv::VideoCapture(video_file, cv::CAP_FFMPEG, dev_id);
  if (!vidcap.isOpened()) {
    std::cerr << "Error: open video src failed in channel " << std::endl;
    exit(1);
  }

  // get frames
  int frame_count = (int)vidcap.get(cv::CAP_PROP_FRAME_COUNT);
  std::vector<int> frame_indices;
  frame_indices.push_back(0);
  for (int i = 1; i < num_frames - 1; i++) {
    frame_indices.push_back(
        (int)((float)(frame_count - 1.0) / (num_frames - 1.0) * i));
  }
  if (num_frames - 1 > 0)
    frame_indices.push_back(frame_count - 1);

  int count = 0;
  while (true) {
    cv::Mat image;
    if (frame_count >= num_frames) {
      vidcap.read(image);
      auto it = std::find(frame_indices.begin(), frame_indices.end(), count);
      if (it != frame_indices.end()) {
        cv::Mat rgb_image;
        cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
        images.push_back(rgb_image);
        if ((int)images.size() >= num_frames)
          break;
      }
      count += 1;
    } else {
      vidcap.read(image);
      if (image.empty()) {
        break;
      }
      cv::Mat rgb_image;
      cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
      images.push_back(rgb_image);
      count += 1;
    }
  }
  vidcap.release();
}

void opencv_image(std::vector<cv::Mat> &images, std::string image_path, int dev_id = 0) {
  auto image = cv::imread(image_path, cv::IMREAD_COLOR);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  images.push_back(image);
}

const int NUM_FRAMES = 1;
int main() {
  std::string llm_bmodel_path = "../BM1684X_F16_test/llama.bmodel";
  std::string vision_bmodel_path =
      "../BM1684X_F16_test/vision_embedding.bmodel";
  std::string tokenizer_path = "./llm_token/tokenizer.model";
  std::string image_path = "./test.png";
  std::vector<cv::Mat> images;
  HalfMatrix features;
  initVLMModel(llm_bmodel_path, vision_bmodel_path, tokenizer_path);
  opencv_image(images, image_path);
  extractVLMVisualFeature(images, features);
  std::string images_prompt;
  for (int i = 0; i < NUM_FRAMES; i++) {
    images_prompt += "<image>\n";
  }
  std::string prompt =
      "A chat between a curious user and an artificial intelligence "
      "assistant. The assistant gives helpful, detailed, and polite "
      "answers to the user's questions. USER: ";
  prompt += images_prompt;
  prompt += "<video>\\n tell me about alone musk ASSISTANT:";
  VLMResult result;
  inferVLMModel(prompt, features, result);
  std::cout << prompt << std::endl;
  std::cout << "result:" << result.output_text << std::endl;
  std::cout << "confidence:" << result.confidence << std::endl;
  destroyVLMModel();
  return 0;
}