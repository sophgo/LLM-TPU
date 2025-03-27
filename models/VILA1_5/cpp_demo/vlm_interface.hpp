#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

typedef uint16_t fp16;
using HalfMatrix = std::vector<std::vector<fp16>>;

typedef struct {
  std::string output_text = ""; // First output text
  float confidence = 0.0;       // Confidence of the first output text
} VLMResult;

/**
 * Initializes the VLM model with the specified configuration file.
 *
 * @param llm_bmodel The path to the llm bmodel file.
 * @param vit_bmodel The path to the vit bmodel file.
 * @param tokenizer_path The path to the tokenizer model.
 */
void initVLMModel(const std::string &llm_bmodel, const std::string &vit_bmodel,
                  const std::string &tokenizer_path);

/**
 * @brief Extracts VLM visual features from a vector of images.
 *
 * This function takes a vector of images and extracts VLM visual features from
 * them. The extracted features are stored in the output_features HalfMatrix.
 *
 * @param images The vector of images from which to extract the visual features.
 * @param output_features The HalfMatrix to store the extracted visual features.
 */
void extractVLMVisualFeature(const std::vector<cv::Mat> &images,
                             HalfMatrix &output_features);

/**
 * @brief Infers the VLM model based on the given prompt and features.
 *
 * This function takes a prompt string and a HalfMatrix of features as input and
 * performs inference to generate the VLM model. The result is stored in the
 * provided VLMResult object.
 *
 * @param prompt The prompt string for VLM model inference.
 * @param features The HalfMatrix of features used for inference.
 * @param result The VLMResult object to store the inference result.
 */
void inferVLMModel(const std::string &prompt, const HalfMatrix &features,
                   VLMResult &result);

/**
 * @brief Infers the VLM model using the given prompt and images.
 *
 * This function takes a prompt string and a vector of images as input and
 * performs inference on the VLM model. The result of the inference is stored in
 * the provided VLMResult object.
 *
 * @param prompt The prompt string for the VLM model.
 * @param images The vector of images to be used for inference.
 * @param result The VLMResult object to store the result of the inference.
 */
void inferVLMModel(const std::string &prompt,
                   const std::vector<cv::Mat> &images, VLMResult &result);

/**
 * @brief Destroys the VLM model.
 *
 * This function is responsible for destroying the VLM model and freeing up any
 * allocated resources. It should be called when the VLM model is no longer
 * needed.
 */
void destroyVLMModel();