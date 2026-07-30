//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chat.hpp"
#include "cv_utils.h"
#include "tokenizers-cpp/tokenizers_cpp.h"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

using tokenizers::Tokenizer;

// Load byte data from a file
static inline std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open [ " << path << " ]" << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

class ChatPipe {
public:
  Config config;

  ChatPipe(int devid, float video_ratio, float video_fps,
           const std::string &model_path, const std::string &config_path,
           bool do_sample = false, int repetition_window = 64);
  ~ChatPipe() { model.deinit(); }
  // Main chat loop
  void chat();
  // Single inference: process one input and return; with prefill_only, only prefill without generating an answer
  void run_once(const std::string &input_str, const std::string &media_path,
                bool prefill_only = false);
  // Shared-prompt prefill: only generate the kv cache and states and save a snapshot,
  // then each question runs inference independently based on that snapshot
  void share_prompt_prefill(const std::string &input_str,
                            const std::string &media_path);

private:
  Qwen3_5 model;
  int ID_IM_END, ID_VISION_START, ID_VISION_END;
  int IMAGE_PAD_TOKEN;
  int VIDEO_PAD_TOKEN;
  int tokens_per_second;
  int spatial_merge_size;
  int num_grid_per_side;
  int spatial_merge_unit;
  bool support_history;
  // Tokenizer and processor
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Maker> maker;

  // Compute rotary position embeddings (flat: [h0, w0, h1, w1, ...])
  std::vector<int> rot_pos(const std::vector<std::vector<int>> &grid_thw);

  // Get the media type
  typedef enum { IMAGE, VIDEO, TEXT, UNKNOWN } MediaType;
  MediaType get_media_type(const std::vector<std::string> &file_path);

  // Build the prompt
  std::string build_text_prompt(const std::string &input_str,
                                bool gen_header = true);
  std::string build_image_prompt(const std::string &input_str,
                                 const std::vector<std::vector<int>> &grid_thw,
                                 bool gen_header = true);
  std::string build_video_prompt(const std::string &input_str,
                                 const std::vector<int> &grid_thw,
                                 const std::vector<double> &timestamps,
                                 bool gen_header = true);

  // Get the RoPE index
  std::vector<std::vector<int>>
  get_rope_index(const std::vector<int> &input_ids,
                 const std::vector<std::vector<int>> &grid_thw, int pad_id);

  void fast_pos_embed_interpolate(const std::vector<int> &grid_thw,
                                  std::vector<int> &idx_out,
                                  std::vector<float> &weight_out);

  // Find token offsets
  std::vector<int> find_token_offset(const std::vector<int> &input_ids,
                                     int pad_id);

  // Get position embeddings
  std::vector<int> get_position_ids(int token_len);

  // Process image
  void vit_process_image(std::vector<float> &pixel_values, int vit_offset);

  // Process video
  void vit_process_video(std::vector<float> &pixel_values,
                         std::vector<int> &vit_offset);

  // Encode input
  std::vector<int> encode_input(const std::string &sentence_input);

  // Print chat instructions
  void print_chat_instructions();

  // Inference
  int forward_prefill(std::vector<int> &position_ids_1d, int &max_posid,
                      int &history_max_posid);

  // Persistent state: shares multi-turn history between chat() and run_once()
  int history_max_posid_state = 0;
  int share_max_posid = 0; // history_max_posid at the end of the shared prompt
};

// Get the media type
ChatPipe::MediaType
ChatPipe::get_media_type(const std::vector<std::string> &medias) {
  if (medias.empty() || medias[0].empty()) {
    return TEXT;
  }
  auto type = UNKNOWN;
  for (auto &m : medias) {
    std::string ext = m.substr(m.find_last_of('.') + 1);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
        ext == "webp") {
      if (type == UNKNOWN) {
        type = IMAGE;
      } else if (type != IMAGE) {
        printf("Error:Mixed media types detected.\n");
        return UNKNOWN;
      }
    } else if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
               ext == "flv" || ext == "wmv") {
      if (type == UNKNOWN) {
        type = VIDEO;
      } else if (type != VIDEO) {
        printf("Error:Mixed media types detected.\n");
        return UNKNOWN;
      }
    }
  }
  return type;
}

std::vector<int> ChatPipe::get_position_ids(int token_len) {
  std::vector<int> position_ids(token_len * 3);
  std::iota(position_ids.begin(), position_ids.begin() + token_len, 0);
  std::copy_n(position_ids.begin(), token_len, position_ids.begin() + token_len);
  std::copy_n(position_ids.begin(), token_len,
              position_ids.begin() + 2 * token_len);
  return position_ids;
}

// ChatPipe class constructor
ChatPipe::ChatPipe(int devid, float video_ratio, float video_fps,
                   const std::string &model_path,
                   const std::string &config_path, bool do_sample,
                   int repetition_window) {
  model.init(devid, model_path, config_path, do_sample, repetition_window);
  spatial_merge_size = 2;
  spatial_merge_unit = spatial_merge_size * spatial_merge_size;
  tokens_per_second = 2;
  num_grid_per_side = 48;
  support_history = model.support_history;

  std::cout << "Processor [" << config_path.c_str() << "] loading .... ";
  auto blob = LoadBytesFromFile((config_path + "/tokenizer.json").c_str());
  tok = Tokenizer::FromBlobJSON(blob);
  ID_IM_END = tok->TokenToId("<|im_end|>");
  ID_VISION_START = tok->TokenToId("<|vision_start|>");
  ID_VISION_END = tok->TokenToId("<|vision_end|>");
  IMAGE_PAD_TOKEN = tok->TokenToId("<|image_pad|>");
  VIDEO_PAD_TOKEN = tok->TokenToId("<|video_pad|>");

  std::cout << "Done!" << std::endl;

  config.temporal_patch_size = 2;
  config.spatial_merge_size = 2;
  config.patch_size = 16;
  config.SEQLEN = model.SEQLEN;
  config.MAX_INPUT_LENGTH = model.MAX_INPUT_LENGTH;
  config.video_ratio = video_ratio;
  config.MAX_PIXELS = model.MAX_PIXELS;
  config.MAX_PATCHES = model.MAX_PATCHES;
  config.MIN_PIXELS = 64 * 32 * 32;
  config.video_fps = video_fps;
  maker = std::make_unique<Maker>(config);
}

// Main function: returns two vectors
// idx_out: int32 equivalent (stored as int), length 4 * t * h * w
// weight_out: float32, length 4 * t * h * w
void ChatPipe::fast_pos_embed_interpolate(const std::vector<int> &grid_thw,
                                          std::vector<int> &idx_out,
                                          std::vector<float> &weight_out) {
  if (grid_thw.empty()) {
    throw std::invalid_argument("grid_thw must contain at least one element");
  }
  const int t = 1;
  const int h = grid_thw[1];
  const int w = grid_thw[2];

  if (h <= 0 || w <= 0 || t <= 0) {
    throw std::invalid_argument("t, h, w must be positive");
  }

  // linspace(0, n-1, h/w) mapped to floor/ceil grid indices + fractions
  const int n = num_grid_per_side;
  const float step_h = h > 1 ? float(n - 1) / float(h - 1) : 0.0f;
  const float step_w = w > 1 ? float(n - 1) / float(w - 1) : 0.0f;

  std::vector<int> base_h(h), base_h_ceil(h);
  std::vector<int> w_floor(w), w_ceil(w);
  std::vector<float> dh(h), dw(w);

  for (int i = 0; i < h; ++i) {
    float x = step_h * i;
    int f = static_cast<int>(x);
    base_h[i] = f * n;
    base_h_ceil[i] = std::min(f + 1, n - 1) * n;
    dh[i] = x - float(f);
  }
  for (int j = 0; j < w; ++j) {
    float x = step_w * j;
    int f = static_cast<int>(x);
    w_floor[j] = f;
    w_ceil[j] = std::min(f + 1, n - 1);
    dw[j] = x - float(f);
  }

  // Write directly in block-reordered (spatial merge) order: no intermediate
  // per-corner vectors and no out_order indirection.
  const int msize = spatial_merge_size;
  idx_out.resize(t * h * w * 4);
  weight_out.resize(t * h * w * 4);
  size_t k = 0;
  for (int i_blk = 0; i_blk < h / msize; ++i_blk) {
    for (int j_blk = 0; j_blk < w / msize; ++j_blk) {
      for (int i2 = 0; i2 < msize; ++i2) {
        const int i = i_blk * msize + i2;
        const float dh_i = dh[i];
        const float one_dh_i = 1.0f - dh_i;
        const int base_i = base_h[i];
        const int base_i_ceil = base_h_ceil[i];
        for (int j2 = 0; j2 < msize; ++j2) {
          const int j = j_blk * msize + j2;
          const float dw_j = dw[j];

          idx_out[k + 0] = base_i + w_floor[j];
          idx_out[k + 1] = base_i + w_ceil[j];
          idx_out[k + 2] = base_i_ceil + w_floor[j];
          idx_out[k + 3] = base_i_ceil + w_ceil[j];

          weight_out[k + 0] = one_dh_i * (1.0f - dw_j);
          weight_out[k + 1] = one_dh_i * dw_j;
          weight_out[k + 2] = dh_i * (1.0f - dw_j);
          weight_out[k + 3] = dh_i * dw_j;
          k += 4;
        }
      }
    }
  }
}

// Compute rotary position embeddings
// Returns a flat [h0, w0, h1, w1, ...] array in spatial-merge block order,
// with the per-frame block repeated t times.
std::vector<int>
ChatPipe::rot_pos(const std::vector<std::vector<int>> &grid_thw) {
  size_t total = 0;
  for (const auto &thw : grid_thw) {
    total += (size_t)thw[0] * thw[1] * thw[2] * 2;
  }

  std::vector<int> pos_ids;
  pos_ids.reserve(total);
  const int msize = spatial_merge_size;

  for (const auto &thw : grid_thw) {
    const int t = thw[0];
    const int h = thw[1];
    const int w = thw[2];

    // Build one frame: (h_idx, w_idx) pairs in block-reordered order
    std::vector<int> frame;
    frame.reserve((size_t)h * w * 2);
    for (int i_blk = 0; i_blk < h / msize; ++i_blk) {
      for (int j_blk = 0; j_blk < w / msize; ++j_blk) {
        for (int k = 0; k < msize; ++k) {
          for (int l = 0; l < msize; ++l) {
            frame.push_back(i_blk * msize + k);
            frame.push_back(j_blk * msize + l);
          }
        }
      }
    }

    // Repeat t times
    for (int i = 0; i < t; ++i) {
      pos_ids.insert(pos_ids.end(), frame.begin(), frame.end());
    }
  }

  return pos_ids;
}

// Return position_ids with shape [3][seq_len]
std::vector<std::vector<int>>
ChatPipe::get_rope_index(const std::vector<int> &input_ids,
                         const std::vector<std::vector<int>> &grid_thw,
                         int pad_id) {
  const size_t seq_length = input_ids.size();

  // Build the three rows directly instead of assembling per-segment vectors
  // and concatenating them afterwards.
  std::vector<std::vector<int>> position_ids(3);
  for (auto &row : position_ids) {
    row.reserve(seq_length);
  }

  const int image_nums =
      std::count(input_ids.begin(), input_ids.end(), ID_VISION_START);
  const int second_per_grid_t = pad_id == VIDEO_PAD_TOKEN ? 1 : 0;

  int st_idx = 0; // running max(position_ids) + 1
  size_t st = 0;

  // Append a text segment: 0..text_len-1 offset by st_idx, on all three rows
  auto append_text = [&](size_t text_len) {
    for (auto &row : position_ids) {
      const size_t off = row.size();
      row.resize(off + text_len);
      std::iota(row.begin() + off, row.end(), st_idx);
    }
    st_idx += (int)text_len;
  };

  for (int img_idx = 0; img_idx < image_nums; ++img_idx) {
    size_t ed_image = input_ids.size();
    auto it = std::find(input_ids.begin() + st, input_ids.end(), pad_id);
    if (it != input_ids.end()) {
      ed_image = it - input_ids.begin();
    }
    int t, h, w;
    if (pad_id == IMAGE_PAD_TOKEN) {
      t = grid_thw[img_idx][0];
      h = grid_thw[img_idx][1];
      w = grid_thw[img_idx][2];
    } else {
      t = 1;
      h = grid_thw[0][1];
      w = grid_thw[0][2];
    }
    size_t ed = ed_image;

    const int llm_grid_t = t;
    const int llm_grid_h = h / spatial_merge_size;
    const int llm_grid_w = w / spatial_merge_size;
    const size_t frame = (size_t)llm_grid_h * llm_grid_w;
    const size_t text_len = ed - st;

    append_text(text_len);

    // Append the media segment (grid indices offset past the text)
    const int offset = st_idx;
    auto &row_t = position_ids[0];
    auto &row_h = position_ids[1];
    auto &row_w = position_ids[2];
    for (int i = 0; i < llm_grid_t; i++) {
      row_t.insert(row_t.end(), frame,
                   offset + i * second_per_grid_t * tokens_per_second);
    }
    for (int n = 0; n < llm_grid_t; ++n) {
      for (int p = 0; p < llm_grid_h; ++p) {
        row_h.insert(row_h.end(), llm_grid_w, offset + p);
        for (int q = 0; q < llm_grid_w; ++q) {
          row_w.push_back(offset + q);
        }
      }
    }

    // Advance past the media segment max (closed form, no re-scan)
    const int grid_max =
        std::max({(llm_grid_t - 1) * second_per_grid_t * tokens_per_second,
                  llm_grid_h - 1, llm_grid_w - 1});
    st_idx = offset + grid_max + 1;
    st = ed + llm_grid_t * frame;
  }
  if (st < input_ids.size()) {
    append_text(input_ids.size() - st);
  }

  return position_ids;
}

std::string strip(const std::string &s) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  // Find the position of the first non-whitespace character
  size_t start = s.find_first_not_of(WHITESPACE);
  if (start == std::string::npos) {
    // All whitespace
    return "";
  }
  // Find the position of the last non-whitespace character
  size_t end = s.find_last_not_of(WHITESPACE);
  // substr(pos, len)，len = end-start+1
  return s.substr(start, end - start + 1);
}

int ChatPipe::forward_prefill(std::vector<int> &position_ids_1d, int &max_posid,
                              int &history_max_posid) {
  if (model.history_length == 0 || support_history == false) {
    history_max_posid = 0;
    return model.forward_first(position_ids_1d);
  }

  if (model.history_length + model.token_length + 128 > model.SEQLEN ||
      model.history_length > model.PREFILL_KV_LENGTH) {
    std::cerr << "Warning: History is full and clear it to continue."
              << std::endl;
    model.clear_history();
    history_max_posid = 0;
    return model.forward_first(position_ids_1d);
  }
  // all id should increase by history_max_posid
  for (auto &x : position_ids_1d) {
    x += history_max_posid;
  }
  max_posid += history_max_posid;
  return model.forward_first(position_ids_1d);
}

// Compute timestamps
static std::vector<double> calculate_timestamps(const std::vector<int> &indices,
                                                double video_fps,
                                                int merge_size = 2) {
  // Convert frame indices to timestamps (seconds)
  std::vector<double> timestamps(indices.size());
  for (size_t i = 0; i < indices.size(); ++i) {
    timestamps[i] = static_cast<double>(indices[i]) / video_fps;
  }

  // Take the average of the first and last values of each merged block
  std::vector<double> merged;
  merged.reserve(timestamps.size() / merge_size);
  for (size_t i = 0; i < timestamps.size(); i += merge_size) {
    size_t j = i + merge_size - 1;
    double avg = (timestamps[i] + timestamps[j]) / 2.0;
    merged.push_back(avg);
  }

  return merged;
}

static std::vector<std::string> splitString(const std::string &s) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    result.push_back(strip(item));
  }
  return result;
}

// Read a @-referenced .txt/.md file and return its contents.
static std::string readPromptFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open prompt file [ " << path << " ]" << std::endl;
    exit(EXIT_FAILURE);
  }
  std::ostringstream oss;
  oss << fs.rdbuf();
  std::string content = oss.str();
  // Trim trailing newlines so the file behaves like a typed prompt.
  while (!content.empty() &&
         (content.back() == '\n' || content.back() == '\r')) {
    content.pop_back();
  }
  return content;
}

// Whether a @-referenced path points to a prompt text file (.txt/.md).
static bool isPromptFilePath(const std::string &path) {
  std::string lower = path;
  std::transform(lower.begin(), lower.end(), lower.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  auto endsWith = [&lower](const std::string &suffix) {
    return lower.size() >= suffix.size() &&
           lower.compare(lower.size() - suffix.size(), suffix.size(),
                         suffix) == 0;
  };
  return endsWith(".txt") || endsWith(".md");
}

// Extract "@path" tokens from the question. A token whose path ends in
// .txt/.md is read as prompt text and replaces the token inline; any other
// token is treated as a media attachment. Returns the media paths joined
// with ',' and removes the media tokens from input_str.
static std::string extractMedia(std::string &input_str) {
  std::vector<std::string> medias;
  std::stringstream ss(input_str);
  std::string token, question, media_path;
  while (ss >> token) {
    if (token.size() > 1 && token[0] == '@') {
      std::string path = token.substr(1);
      if (isPromptFilePath(path)) {
        token = readPromptFile(path);
      } else {
        medias.push_back(path);
        continue;
      }
    }
    if (!question.empty()) {
      question += " ";
    }
    question += token;
  }
  input_str = question;
  for (size_t i = 0; i < medias.size(); i++) {
    if (i > 0) {
      media_path += ",";
    }
    media_path += medias[i];
  }
  return media_path;
}

// Main chat loop
void ChatPipe::chat() {
  print_chat_instructions();
  history_max_posid_state = 0;
  while (true) {
    std::string input_str;
    std::cout << "\nQuestion: ";
    std::getline(std::cin, input_str);
    input_str = strip(input_str);
    if (input_str == "/exit" || input_str == "/q" || input_str == "/quit") {
      break;
    }
    if (input_str == "/clear" || input_str == "/c" || input_str == "/new") {
      model.clear_history();
      history_max_posid_state = 0;
      std::cout << "Chat history cleared." << std::endl;
      continue;
    }

    std::string media_path = extractMedia(input_str);
    run_once(input_str, media_path);
  }
}

// Flatten [3][seq_len] position ids to 1D; also yields max of row 0
static std::vector<int>
flatten_position_ids(const std::vector<std::vector<int>> &position_ids,
                     int &max_posid) {
  std::vector<int> out;
  out.reserve(position_ids[0].size() * 3);
  for (const auto &row : position_ids) {
    out.insert(out.end(), row.begin(), row.end());
  }
  max_posid = *std::max_element(position_ids[0].begin(), position_ids[0].end());
  return out;
}

// Single inference
void ChatPipe::run_once(const std::string &input_str_in,
                        const std::string &media_path, bool prefill_only) {
  using clock = std::chrono::steady_clock;
  std::string input_str = strip(input_str_in);
  int token = 0;
  int max_posid = 0;
  int &history_max_posid = history_max_posid_state;
  // Before each inference, restore the shared prompt's kv cache and states so that each question
  // is handled independently based on the shared prompt (during shared prefill the snapshot does not exist yet, so restore is a no-op
  // and share_max_posid is at its initial value 0)
  model.restore_share_prompt();
  history_max_posid = share_max_posid;

  auto medias = splitString(media_path);
  auto media_type = get_media_type(medias);
  if (media_type == ChatPipe::UNKNOWN) {
    std::cout
        << "Unsupported media type. Please provide a valid image or video."
        << std::endl;
    return;
  }
  if (media_type != ChatPipe::TEXT) {
    // check file exists
    for (auto &m : medias) {
      if (!std::filesystem::exists(m)) {
        std::cerr << "File does not exist: " << m << std::endl;
        return;
      }
    }
  }

  if (!prefill_only) {
    std::cout << "\nAnswer:\n";
  }
  int64_t duration_prefill = 0, duration_vit = 0, duration_decode = 0;
  int input_token_num = 0;
  const int max_input_tokens =
      model.support_history ? model.SEQLEN : model.MAX_INPUT_LENGTH;
  clock::time_point clock_start;
  switch (media_type) {
  case ChatPipe::IMAGE: {
    int num_medias = medias.size();
    std::vector<std::vector<float>> pixel_values(num_medias);
    std::vector<std::vector<int>> grid_thws;
    grid_thws.reserve(num_medias);
    for (int i = 0; i < num_medias; ++i) {
      auto ret = process_image(pixel_values[i], medias[i], config);
      if (ret == false) {
        std::cerr << "Error processing image: " << medias[i] << std::endl;
        return;
      }
      grid_thws.push_back(config.grid_thw);
    }
    std::string sentence_input =
        build_image_prompt(input_str, grid_thws, !prefill_only);
    std::vector<int> tokens = encode_input(sentence_input);
    if ((int)(tokens.size()) > max_input_tokens) {
      std::cerr << "Input tokens exceed maximum length: " << max_input_tokens
                << std::endl;
      return;
    }
    input_token_num = tokens.size();
    auto vit_offset = find_token_offset(tokens, ID_VISION_START);
    clock_start = clock::now();
    model.forward_embed(tokens);
    auto clock_vit_start = clock::now();
    for (int i = 0; i < num_medias; ++i) {
      vit_process_image(pixel_values[i], vit_offset[i] + 1);
    }
    auto clock_vit_end = clock::now();
    duration_vit = std::chrono::duration_cast<std::chrono::milliseconds>(
                       clock_vit_end - clock_vit_start)
                       .count();
    auto position_ids = get_rope_index(tokens, grid_thws, IMAGE_PAD_TOKEN);
    std::vector<int> position_ids_1d =
        flatten_position_ids(position_ids, max_posid);
    token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
  } break;
  case VIDEO: {
    // Video only deal with first video path
    std::vector<float> pixel_values;
    std::vector<int> frame_indices;
    double fps;
    auto ret =
        process_video(pixel_values, frame_indices, medias[0], config, fps);
    if (ret == false) {
      std::cerr << "Error processing video: " << medias[0] << std::endl;
      return;
    }
    auto timestamps =
        calculate_timestamps(frame_indices, fps, config.spatial_merge_size);
    std::string sentence_input = build_video_prompt(input_str, config.grid_thw,
                                                    timestamps, !prefill_only);
    std::vector<int> tokens = encode_input(sentence_input);
    if ((int)(tokens.size()) > max_input_tokens) {
      std::cerr << "Input tokens exceed maximum length: " << max_input_tokens
                << std::endl;
      return;
    }
    input_token_num = tokens.size();
    auto vit_offset = find_token_offset(tokens, ID_VISION_START);
    clock_start = clock::now();
    model.forward_embed(tokens);
    auto clock_vit_start = clock::now();
    vit_process_video(pixel_values, vit_offset);
    auto clock_vit_end = clock::now();
    duration_vit = std::chrono::duration_cast<std::chrono::milliseconds>(
                       clock_vit_end - clock_vit_start)
                       .count();
    auto position_ids =
        get_rope_index(tokens, {config.grid_thw}, VIDEO_PAD_TOKEN);
    std::vector<int> position_ids_1d =
        flatten_position_ids(position_ids, max_posid);
    token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
  } break;
  case TEXT: {
    std::string sentence_input = build_text_prompt(input_str, !prefill_only);
    std::vector<int> tokens = encode_input(sentence_input);
    if ((int)(tokens.size()) > max_input_tokens) {
      std::cerr << "Input tokens exceed maximum length: " << max_input_tokens
                << std::endl;
      return;
    }
    input_token_num = tokens.size();
    clock_start = clock::now();
    model.forward_embed(tokens);
    auto position_ids_1d = get_position_ids(tokens.size());
    max_posid = tokens.size() - 1;
    token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
  } break;
  default:
    std::cerr << "Unsupported media type." << std::endl;
    return;
  }
  auto clock_prefill = clock::now();
  duration_prefill = std::chrono::duration_cast<std::chrono::milliseconds>(
                         clock_prefill - clock_start)
                         .count();
  if (prefill_only) {
    // Shared-prompt mode: only generate the kv cache and states, no answer is generated.
    // After prefill, history contains only the prompt tokens, and the next position is max_posid + 1
    history_max_posid = max_posid + 1;
    std::cout << "\nFTL: " << duration_prefill / 1000.0f << " s" << std::endl;
    if (duration_vit > 0) {
      std::cout << "Vision [" << config.grid_thw[0] << ", "
                << config.grid_thw[1] << ", " << config.grid_thw[2]
                << "]: " << duration_vit / 1000.0f << " s" << std::endl;
    }
    std::cout << "Shared Prompt Tokens: " << input_token_num << std::endl;
    return;
  }
  // Subsequent tokenization
  std::vector<int> full_word_tokens;
  std::string text;
  int output_token_num = 0;
  std::vector<int> following_position_ids(3);
  while (token != ID_IM_END && model.history_length < model.SEQLEN) {
    full_word_tokens.push_back(token);
    std::string word = tok->Decode(full_word_tokens);
    if (word.find("�") == std::string::npos) {
      if (full_word_tokens.size() == 1) {
        std::string pre_word = word;
        std::vector<int> double_token = {token, token};
        word = tok->Decode(double_token).substr(pre_word.length());
      }
      text += word;
      std::cout << word << std::flush;
      if (model.do_sample) {
        if (model.check_stop(text)) {
          break;
        }
      }
      full_word_tokens.clear();
    }
    max_posid++;
    following_position_ids[0] = following_position_ids[1] =
        following_position_ids[2] = max_posid;
    token = model.forward_next(following_position_ids);
    output_token_num++;
  }
  std::cout << std::endl;
  auto clock_end = clock::now();
  duration_decode = std::chrono::duration_cast<std::chrono::milliseconds>(
                        clock_end - clock_prefill)
                        .count();
  std::cout << "FTL: " << duration_prefill / 1000.0f << " s" << std::endl;
  if (output_token_num > 0 && duration_decode > 0) {
    std::cout << "TPS: " << output_token_num * 1000.0f / duration_decode
              << " tokens/s" << std::endl;
  }
  if (duration_vit > 0) {
    std::cout << "Vision [" << config.grid_thw[0] << ", " << config.grid_thw[1]
              << ", " << config.grid_thw[2] << "]: " << duration_vit / 1000.0f
              << " s" << std::endl;
  }
  std::cout << "Input Tokens: " << input_token_num
            << ", Output Tokens: " << output_token_num + 1 << std::endl;
  if (model.support_history) {
    std::cout << "Total Tokens: " << model.history_length << std::endl;
  }
}

// Shared-prompt prefill: only generate the kv cache and states; every subsequent question is based on them
void ChatPipe::share_prompt_prefill(const std::string &input_str,
                                    const std::string &media_path) {
  if (!support_history) {
    std::cerr << "\nError: this demo requires a bmodel compiled with "
                 "history support.\n";
    exit(EXIT_FAILURE);
  }
  std::cout << "\nPrefilling the shared prompt ..." << std::endl;
  run_once(input_str, media_path, true);
  if (model.history_length == 0) {
    std::cerr << "Error: failed to prefill the shared prompt." << std::endl;
    exit(EXIT_FAILURE);
  }
  model.save_share_prompt();
  share_max_posid = history_max_posid_state;
  std::cout << "Shared prompt saved. Every question will be based on it."
            << std::endl;
}

static std::string format_seconds(double curr_time) {
  std::ostringstream oss;
  oss << "<" << std::fixed << std::setprecision(1) << curr_time << " seconds>";
  return oss.str();
}

// Append n copies of a token (capacity reserved upfront, no reallocations)
static void append_repeated(std::string &out, const std::string &token,
                            size_t n) {
  out.reserve(out.size() + token.size() * n);
  for (size_t i = 0; i < n; ++i) {
    out += token;
  }
}

// Build the prompt
std::string ChatPipe::build_text_prompt(const std::string &input_str,
                                        bool gen_header) {
  std::string prompt = "<|im_start|>user\n";
  prompt += input_str + "<|im_end|>\n";
  if (gen_header) {
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  }
  return prompt;
}

std::string
ChatPipe::build_image_prompt(const std::string &input_str,
                             const std::vector<std::vector<int>> &grid_thw,
                             bool gen_header) {
  std::string prompt = "<|im_start|>user\n";
  int num_images = grid_thw.size();
  size_t total_pads = 0;
  for (const auto &thw : grid_thw) {
    total_pads += (size_t)thw[1] * thw[2] / 4;
  }
  prompt.reserve(prompt.size() + total_pads * 13 +
                 num_images * 30 + input_str.size() + 64);
  for (int i = 0; i < num_images; i++) {
    int pad_len = grid_thw[i][1] * grid_thw[i][2] / 4;
    prompt += "<|vision_start|>";
    append_repeated(prompt, "<|image_pad|>", pad_len);
    prompt += "<|vision_end|>";
  }
  prompt += input_str + "<|im_end|>\n";
  if (gen_header) {
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  }
  return prompt;
}

std::string ChatPipe::build_video_prompt(const std::string &input_str,
                                         const std::vector<int> &thw,
                                         const std::vector<double> &timestamps,
                                         bool gen_header) {
  std::string prompt = "<|im_start|>user\n";
  int t = thw[0];
  int h = thw[1];
  int w = thw[2];
  int pad_len = h * w / 4;
  prompt.reserve(prompt.size() + (size_t)t * (pad_len * 13 + 50) +
                 input_str.size() + 64);
  for (int i = 0; i < t; i++) {
    prompt += format_seconds(timestamps[i]);
    prompt += "<|vision_start|>";
    append_repeated(prompt, "<|video_pad|>", pad_len);
    prompt += "<|vision_end|>";
  }
  prompt += input_str + "<|im_end|>\n";
  if (gen_header) {
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
  }
  return prompt;
}

// Find token offsets
std::vector<int> ChatPipe::find_token_offset(const std::vector<int> &input_ids,
                                             int pad_id) {
  std::vector<int> offsets;
  int num = input_ids.size();
  for (int i = 0; i < num; ++i) {
    if (input_ids[i] == pad_id) {
      offsets.push_back(i);
    }
  }
  return offsets;
}

// Process image
void ChatPipe::vit_process_image(std::vector<float> &pixel_values,
                                 int vit_offset) {
  std::vector<int> position_ids = rot_pos({config.grid_thw});
  std::vector<int> pos_ids;
  std::vector<float> pos_weight;
  fast_pos_embed_interpolate(config.grid_thw, pos_ids, pos_weight);

  model.forward_vit(pixel_values.data(), position_ids, pos_ids, pos_weight,
                    config.grid_thw, vit_offset);
}

void ChatPipe::vit_process_video(std::vector<float> &pixel_values,
                                 std::vector<int> &vit_offset) {
  // hidden_states has the same length as pixel_values
  int t = config.grid_thw[0];
  int h = config.grid_thw[1];
  int w = config.grid_thw[2];
  assert(t == (int)(vit_offset.size()));
  std::vector<int> pos_ids;
  std::vector<float> pos_weight;
  fast_pos_embed_interpolate(config.grid_thw, pos_ids, pos_weight);
  // Call rot_pos to generate position_ids (same block for every frame)
  std::vector<std::vector<int>> grid_thw = {{1, h, w}};
  std::vector<int> position_ids = rot_pos(grid_thw);
  for (int i = 0; i < t; i++) {
    model.forward_vit(pixel_values.data() + i * h * w * model.VIT_DIMS,
                      position_ids, pos_ids, pos_weight, grid_thw[0],
                      vit_offset[i] + 1);
  }
}

// Encode input
std::vector<int> ChatPipe::encode_input(const std::string &sentence_input) {
  return tok->Encode(sentence_input);
}

void ChatPipe::print_chat_instructions() {
  std::cout
      << "\n================================================================="
         "\n"
      << "1. If you want to quit, please enter one of [/q, /quit, /exit]\n"
      << "2. To create a new chat session, please enter one of [/clear, /new]\n"
      << "3. To ask about an image or video, include @<path> in your question\n"
      << "4. To use the contents of a .txt or .md file as your question, "
         "include @<path>\n"
      << "================================================================="
         "\n";
}

void Usage() {
  printf(
      "Usage:\n"
      "  -h, --help        : Show help info \n"
      "  -m, --model       : Set model path \n"
      "  -c, --config      : Set config path \n"
      "  -r, --video_ratio : Set video ratio, default is 0.25\n"
      "  -f, --video_fps   : Set video fps, default is 1.0\n"
      "  -s, --do_sample   : Enable sampling during generation\n"
      "  -d, --devid       : Set devices to run for model, default is '0'\n"
      "  -p, --prompt      : Shared prompt text; only prefilled to generate\n"
      "                      the kv cache and states that every question is\n"
      "                      based on. Include @<path> to attach image/video\n"
      "                      (repeat for multiple images), or to read prompt\n"
      "                      text from a .txt/.md file\n"
      "  -w, --rep_window  : Sliding window size for repetition penalty; only\n"
      "                      the last N tokens are penalized. 64 (default);\n"
      "                      0 penalizes the full context (only with -s)\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &config_path, std::string &image_path,
                      int &device, float &video_ratio, float &video_fps,
                      bool &do_sample, std::string &prompt, bool &has_prompt,
                      int &rep_window) {
  struct option longOptions[] = {
      {"model", required_argument, nullptr, 'm'},
      {"config", required_argument, nullptr, 'c'},
      {"devid", required_argument, nullptr, 'd'},
      {"video_ratio", required_argument, nullptr, 'r'},
      {"video_fps", required_argument, nullptr, 'f'},
      {"do_sample", no_argument, nullptr, 's'},
      {"prompt", required_argument, nullptr, 'p'},
      {"rep_window", required_argument, nullptr, 'w'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;
  while ((option = getopt_long(argc, argv, "m:c:d:r:f:p:w:sh", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'd':
      device = atoi(optarg);
      break;
    case 'r':
      video_ratio = atof(optarg);
      break;
    case 'f':
      video_fps = atof(optarg);
      break;
    case 's':
      do_sample = true;
      break;
    case 'p':
      prompt = optarg;
      has_prompt = true;
      break;
    case 'w':
      rep_window = atoi(optarg);
      break;
    case 'h':
      Usage();
      exit(EXIT_SUCCESS);
    case '?':
      Usage();
      exit(EXIT_FAILURE);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char *argv[]) {
  std::string model_path;
  std::string config_path;
  std::string image_path;
  int dev_id = 0;
  float video_ratio = 0.25f; // Default video ratio is 0.25
  float video_fps = 1.0f;    // Sample 1 frame per second by default
  bool do_sample = false;
  std::string prompt;
  bool has_prompt = false;
  int rep_window = 64;

  processArguments(argc, argv, model_path, config_path, image_path, dev_id,
                   video_ratio, video_fps, do_sample, prompt, has_prompt,
                   rep_window);
  if (model_path.empty() || config_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }
  assert(video_fps > 0);
  ChatPipe pipeline(dev_id, video_ratio, video_fps, model_path, config_path,
                    do_sample, rep_window);
  // Shared-prompt mode: --prompt is only used to generate the kv cache and
  // states, then enter interactive chat where each question is based on the
  // shared prompt
  if (!has_prompt) {
    std::cerr << "Error: --prompt is required as the shared prompt (include "
                 "@<path> to attach image/video, or to read prompt text "
                 "from a .txt/.md file)"
              << std::endl;
    Usage();
    exit(EXIT_FAILURE);
  }
  std::string media_path = extractMedia(prompt);
  pipeline.share_prompt_prefill(prompt, media_path);
  pipeline.chat();
  return 0;
}