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

static const int VISION_PAD_TOKEN = 151654;
static const int IMAGE_PAD_TOKEN = 151655;
static const int VIDEO_PAD_TOKEN = 151656;

// 从文件加载字节数据
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
std::vector<float> convertIntToFloat(const std::vector<int> &position_ids) {
  std::vector<float> floatPositionIds;
  for (int val : position_ids) {
    floatPositionIds.push_back(static_cast<float>(val));
  }
  return floatPositionIds;
}

std::vector<int> convertFloatToInt(const std::vector<float> &position_ids) {
  std::vector<int> intPositionIds;
  for (int val : position_ids) {
    intPositionIds.push_back(static_cast<int>(val));
  }
  return intPositionIds;
}

class ChatPipe {
public:
  std::string sys_config;
  Config config;

  ChatPipe(int devid, float video_ratio, const std::string &model_path,
           const std::string &vocab_path, const std::string &system_prompt);
  // 聊天主循环
  void chat(std::string input_str = "", std::string media_path = "");

private:
  Qwen2_5VL model;
  int ID_IM_END, ID_VISION_START;
  int tokens_per_second = 2;
  int spatial_merge_size;
  int spatial_merge_unit;
  float video_ratio;
  bool support_history;
  int history_max_posid = 0;
  // 分词器和处理器
  std::unique_ptr<Tokenizer> tok;
  std::unique_ptr<Maker> maker;

  // 获取窗口索引和累积窗口序列长度
  std::pair<std::vector<int>, std::vector<int>>
  get_window_index(const std::vector<std::vector<int>> &grid_thw);

  // 获取注意力掩码
  std::vector<float> get_attn_mask(int seq_length,
                                   const std::vector<int> &cu_seqlens);

  // 计算旋转位置编码
  std::vector<std::vector<int>>
  rot_pos(const std::vector<std::vector<int>> &grid_thw);

  // 获取媒体类型
  typedef enum { IMAGE, VIDEO, TEXT, UNKNOWN } MediaType;
  MediaType get_media_type(const std::string &file_path);

  // 构建提示
  std::string build_prompt(std::string input_str, MediaType media_type);

  // 重新整理hidden_states
  std::vector<float>
  reorder_hidden_states(const std::vector<float> &hidden_states, int seq_len,
                        const std::vector<int> &window_index);

  // 获取rope索引
  std::vector<std::vector<std::vector<int>>>
  get_rope_index(const std::vector<std::vector<int>> &input_ids,
                 const std::vector<std::vector<int>> &grid_thw, int pad_id);

  // 查找分词偏移量
  int find_token_offset(const std::vector<int> &input_ids, int pad_id);

  // 获取位置编码
  std::vector<int> get_position_ids(int token_len);

  // 处理图像
  void vit_process_image(std::vector<float> &pixel_values, int vit_offset);

  // 处理视频
  void vit_process_video(std::vector<float> &pixel_values, int &vit_offset);

  // 编码输入
  std::vector<int> encode_input(const std::string &sentence_input);

  // 打印聊天说明
  void print_chat_instructions();

  // 推理
  int forward_prefill(std::vector<int> &position_ids_1d, int &max_posid,
                      int &history_max_posid);
};

// 获取媒体类型
ChatPipe::MediaType ChatPipe::get_media_type(const std::string &file_path) {
  if (file_path.empty()) {
    return TEXT;
  }
  std::string ext = file_path.substr(file_path.find_last_of('.') + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "bmp" ||
      ext == "webp") {
    return IMAGE;
  }
  if (ext == "mp4" || ext == "avi" || ext == "mov" || ext == "mkv" ||
      ext == "flv" || ext == "wmv") {
    return VIDEO;
  }
  return UNKNOWN;
}

std::vector<int> ChatPipe::get_position_ids(int token_len) {
  std::vector<int> position_ids(token_len * 3);
  for (int j = 0; j < 3; j++) {
    for (int i = 0; i < token_len; ++i) {
      position_ids[j * token_len + i] = i;
    }
  }
  return position_ids;
}

// ChatPipe 类构造函数
ChatPipe::ChatPipe(int devid, float vratio, const std::string &model_path,
                   const std::string &vocab_path,
                   const std::string &system_prompt) {
  model.init(devid, model_path);
  spatial_merge_size = 2;
  spatial_merge_unit = spatial_merge_size * spatial_merge_size;
  tokens_per_second = 2;
  video_ratio = vratio;
  support_history = model.support_history;
  sys_config = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";

  std::cout << "Processor [" << vocab_path.c_str() << "] loading .... ";
  auto blob = LoadBytesFromFile((vocab_path + "/tokenizer.json").c_str());
  tok = Tokenizer::FromBlobJSON(blob);
  ID_IM_END = tok->TokenToId("<|im_end|>");
  ID_VISION_START = tok->TokenToId("<|vision_start|>");
  std::cout << "Done!" << std::endl;

  config.model_type = "qwen2_vl";
  config.temporal_patch_size = 2;
  config.spatial_merge_size = 2;
  config.patch_size = 14;
  config.SEQLEN = model.SEQLEN;
  config.video_ratio = video_ratio;
  config.MAX_PIXELS = model.MAX_PIXELS;
  config.MAX_PATCHES = model.MAX_PATCHES;
  config.MIN_PIXELS = 64 * 28 * 28;
  maker = std::make_unique<Maker>(config);
}

// 获取窗口索引和累积窗口序列长度
std::pair<std::vector<int>, std::vector<int>>
ChatPipe::get_window_index(const std::vector<std::vector<int>> &grid_thw) {
  std::vector<int> window_index;
  std::vector<int> cu_window_seqlens = {0};
  int window_index_id = 0;
  int vit_merger_window_size = 4;
  int spatial_merge_unit = spatial_merge_size * spatial_merge_size;

  for (const auto &thw : grid_thw) {
    int grid_t = thw[0], grid_h = thw[1], grid_w = thw[2];
    int llm_grid_h = grid_h / 2;
    int llm_grid_w = grid_w / 2;
    int total = grid_t * llm_grid_h * llm_grid_w;
    std::vector<int> index(total);
    for (int i = 0; i < total; ++i)
      index[i] = i;

    int pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size;
    if (pad_h == vit_merger_window_size)
      pad_h = 0;
    int pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size;
    if (pad_w == vit_merger_window_size)
      pad_w = 0;
    int num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
    int num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

    // 填充索引
    std::vector<int> index_padded(
        (llm_grid_h + pad_h) * (llm_grid_w + pad_w) * grid_t, -100);
    for (int t = 0; t < grid_t; ++t) {
      for (int h = 0; h < llm_grid_h; ++h) {
        for (int w = 0; w < llm_grid_w; ++w) {
          int idx = t * llm_grid_h * llm_grid_w + h * llm_grid_w + w;
          int pad_idx = t * (llm_grid_h + pad_h) * (llm_grid_w + pad_w) +
                        h * (llm_grid_w + pad_w) + w;
          index_padded[pad_idx] = idx;
        }
      }
    }

    // 分窗口
    std::vector<int> index_new;
    std::vector<int> seqlens;
    for (int t = 0; t < grid_t; ++t) {
      for (int wh = 0; wh < num_windows_h; ++wh) {
        for (int ww = 0; ww < num_windows_w; ++ww) {
          std::vector<int> window;
          for (int i = 0; i < vit_merger_window_size; ++i) {
            for (int j = 0; j < vit_merger_window_size; ++j) {
              int h = wh * vit_merger_window_size + i;
              int w = ww * vit_merger_window_size + j;
              int pad_idx = t * (llm_grid_h + pad_h) * (llm_grid_w + pad_w) +
                            h * (llm_grid_w + pad_w) + w;
              int val = index_padded[pad_idx];
              if (val != -100)
                window.push_back(val);
            }
          }
          seqlens.push_back(window.size());
          for (int v : window)
            index_new.push_back(v + window_index_id);
        }
      }
    }
    window_index.insert(window_index.end(), index_new.begin(), index_new.end());
    int last = cu_window_seqlens.back();
    for (size_t i = 0; i < seqlens.size(); ++i) {
      last += seqlens[i] * spatial_merge_unit;
      cu_window_seqlens.push_back(last);
    }
    window_index_id += total;
  }
  return {window_index, cu_window_seqlens};
}

// 获取注意力掩码
std::vector<float> ChatPipe::get_attn_mask(int seq_length,
                                           const std::vector<int> &cu_seqlens) {
  std::vector<float> attention_mask(seq_length * seq_length, -10000.0f);
  for (size_t i = 1; i < cu_seqlens.size(); ++i) {
    int start = cu_seqlens[i - 1];
    int end = cu_seqlens[i];
    // 添加边界检查，避免越界访问
    if (start >= seq_length)
      continue;
    if (end > seq_length)
      end = seq_length;
    for (int row = start; row < end; ++row) {
      for (int col = start; col < end; ++col) {
        attention_mask[row * seq_length + col] = 0.0f;
      }
    }
  }
  return attention_mask;
}

// 计算旋转位置编码
std::vector<std::vector<int>>
ChatPipe::rot_pos(const std::vector<std::vector<int>> &grid_thw) {
  std::vector<std::vector<int>> pos_ids;

  for (const auto &thw : grid_thw) {
    int t = thw[0];
    int h = thw[1];
    int w = thw[2];

    // 生成 hpos_ids
    std::vector<int> hpos_ids;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        hpos_ids.push_back(i);
      }
    }

    // 重塑 hpos_ids
    std::vector<int> reshaped_hpos_ids;
    int h_merged = h / spatial_merge_size;
    int w_merged = w / spatial_merge_size;
    for (int i = 0; i < h_merged; ++i) {
      for (int j = 0; j < w_merged; ++j) {
        for (int k = 0; k < spatial_merge_size; ++k) {
          for (int l = 0; l < spatial_merge_size; ++l) {
            int src_idx = ((i * spatial_merge_size + k) * w) +
                          (j * spatial_merge_size + l);
            reshaped_hpos_ids.push_back(hpos_ids[src_idx]);
          }
        }
      }
    }

    // 生成 wpos_ids
    std::vector<int> wpos_ids;
    for (int i = 0; i < h; ++i) {
      for (int j = 0; j < w; ++j) {
        wpos_ids.push_back(j);
      }
    }

    // 重塑 wpos_ids
    std::vector<int> reshaped_wpos_ids;
    for (int i = 0; i < h_merged; ++i) {
      for (int j = 0; j < w_merged; ++j) {
        for (int k = 0; k < spatial_merge_size; ++k) {
          for (int l = 0; l < spatial_merge_size; ++l) {
            int src_idx = ((i * spatial_merge_size + k) * w) +
                          (j * spatial_merge_size + l);
            reshaped_wpos_ids.push_back(wpos_ids[src_idx]);
          }
        }
      }
    }

    // 合并 hpos_ids 和 wpos_ids
    std::vector<std::vector<int>> merged;
    for (size_t i = 0; i < reshaped_hpos_ids.size(); ++i) {
      merged.push_back({reshaped_hpos_ids[i], reshaped_wpos_ids[i]});
    }

    // 重复 t 次
    std::vector<std::vector<int>> repeated;
    for (int i = 0; i < t; ++i) {
      repeated.insert(repeated.end(), merged.begin(), merged.end());
    }

    pos_ids.insert(pos_ids.end(), repeated.begin(), repeated.end());
  }

  return pos_ids;
}

// 查找元素在向量中的所有索引
std::vector<size_t> argwhere(const std::vector<int> &vec, int value) {
  std::vector<size_t> indices;
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] == value) {
      indices.push_back(i);
    }
  }
  return indices;
}

// 生成从 start 到 end-1 的整数序列
std::vector<int> arange(int start, int end) {
  std::vector<int> result;
  for (int i = start; i < end; ++i) {
    result.push_back(i);
  }
  return result;
}

// 获取向量中的最大值
int max(const std::vector<int> &vec) {
  int max_val = vec[0];
  for (int val : vec) {
    if (val > max_val) {
      max_val = val;
    }
  }
  return max_val;
}

// 将二维向量在指定维度上拼接
std::vector<std::vector<int>>
cat(const std::vector<std::vector<std::vector<int>>> &vecs, int dim) {
  if (dim == 1) {
    std::vector<std::vector<int>> result(3);
    for (const auto &vec : vecs) {
      for (int i = 0; i < 3; ++i) {
        result[i].insert(result[i].end(), vec[i].begin(), vec[i].end());
      }
    }
    return result;
  }
  return {};
}
// 返回形状为 [3][batch_size][seq_len] 的position_ids
std::vector<std::vector<std::vector<int>>>
ChatPipe::get_rope_index(const std::vector<std::vector<int>> &input_ids,
                         const std::vector<std::vector<int>> &grid_thw,
                         int pad_id) {

  size_t batch_size = input_ids.size();
  size_t seq_length = input_ids[0].size();

  // 初始化 attention_mask 和 position_ids
  std::vector<std::vector<int>> attention_mask(batch_size,
                                               std::vector<int>(seq_length, 1));
  std::vector<std::vector<std::vector<int>>> position_ids(
      3, std::vector<std::vector<int>>(batch_size,
                                       std::vector<int>(seq_length, 1)));

  int image_index = 0;

  for (size_t i = 0; i < batch_size; ++i) {
    // 获取有效输入
    std::vector<int> valid_input_ids;
    for (size_t j = 0; j < seq_length; ++j) {
      if (attention_mask[i][j] == 1) {
        valid_input_ids.push_back(input_ids[i][j]);
      }
    }

    // 计算图像数量
    int image_nums = 0;
    std::vector<size_t> vision_start_indices =
        argwhere(valid_input_ids, ID_VISION_START);
    for (size_t idx : vision_start_indices) {
      if (idx + 1 < valid_input_ids.size() &&
          valid_input_ids[idx + 1] == pad_id) {
        ++image_nums;
      }
    }

    std::vector<int> input_tokens = valid_input_ids;
    std::vector<std::vector<std::vector<int>>> llm_pos_ids_list;
    size_t st = 0;
    int remain_images = image_nums;
    int second_per_grid_t = pad_id == VIDEO_PAD_TOKEN ? 1 : 0;
    for (int img_idx = 0; img_idx < image_nums; ++img_idx) {
      size_t ed_image = input_tokens.size();
      if (remain_images > 0) {
        auto it =
            std::find(input_tokens.begin() + st, input_tokens.end(), pad_id);
        if (it != input_tokens.end()) {
          ed_image = it - input_tokens.begin();
        }
      }

      int t = grid_thw[image_index][0];
      int h = grid_thw[image_index][1];
      int w = grid_thw[image_index][2];

      ++image_index;
      --remain_images;
      size_t ed = ed_image;

      int llm_grid_t = t;
      int llm_grid_h = h / spatial_merge_size;
      int llm_grid_w = w / spatial_merge_size;
      size_t text_len = ed - st;

      int st_idx = 0;
      if (!llm_pos_ids_list.empty()) {
        int max_val = 0;
        for (const auto &row : llm_pos_ids_list.back()) {
          int row_max = max(row);
          if (row_max > max_val) {
            max_val = row_max;
          }
        }
        st_idx = max_val + 1;
      }

      // 处理文本部分的位置索引
      std::vector<std::vector<int>> text_pos(3);
      std::vector<int> text_range = arange(0, text_len);
      for (int j = 0; j < 3; ++j) {
        std::vector<int> temp(text_range);
        for (int &val : temp) {
          val += st_idx;
        }
        text_pos[j] = temp;
      }
      llm_pos_ids_list.push_back(text_pos);

      // 处理图像部分的位置索引

      std::vector<int> t_index;
      for (int i = 0; i < llm_grid_t; i++) {
        auto time_val = i * second_per_grid_t * tokens_per_second;
        t_index.insert(t_index.end(), llm_grid_h * llm_grid_w, time_val);
      }

      std::vector<int> h_index;
      for (int n = 0; n < llm_grid_t; ++n) {
        for (int p = 0; p < llm_grid_h; ++p) {
          for (int q = 0; q < llm_grid_w; ++q) {
            h_index.push_back(p);
          }
        }
      }

      std::vector<int> w_index;
      for (int n = 0; n < llm_grid_t; ++n) {
        for (int p = 0; p < llm_grid_h; ++p) {
          for (int q = 0; q < llm_grid_w; ++q) {
            w_index.push_back(q);
          }
        }
      }

      std::vector<std::vector<int>> grid_pos = {t_index, h_index, w_index};
      for (auto &row : grid_pos) {
        for (int &val : row) {
          val += text_len + st_idx;
        }
      }
      llm_pos_ids_list.push_back(grid_pos);

      st = ed + llm_grid_t * llm_grid_h * llm_grid_w;
    }
    if (st < input_tokens.size()) {
      int st_idx = 0;
      if (!llm_pos_ids_list.empty()) {
        int max_val = 0;
        for (const auto &row : llm_pos_ids_list.back()) {
          int row_max = max(row);
          if (row_max > max_val) {
            max_val = row_max;
          }
        }
        st_idx = max_val + 1;
      }
      size_t text_len = input_tokens.size() - st;
      std::vector<std::vector<int>> text_pos(3);
      std::vector<int> text_range = arange(0, text_len);
      for (int j = 0; j < 3; ++j) {
        std::vector<int> temp(text_range);
        for (int &val : temp) {
          val += st_idx;
        }
        text_pos[j] = temp;
      }
      llm_pos_ids_list.push_back(text_pos);
    }

    std::vector<std::vector<int>> llm_positions = cat(llm_pos_ids_list, 1);

    size_t valid_index = 0;
    for (size_t j = 0; j < seq_length; ++j) {
      if (attention_mask[i][j] == 1) {
        for (int k = 0; k < 3; ++k) {
          position_ids[k][i][j] = llm_positions[k][valid_index];
        }
        ++valid_index;
      }
    }
  }
  return position_ids;
}

std::string strip(const std::string &s) {
  const std::string WHITESPACE = " \n\r\t\f\v";
  // 找到第一个非空白字符位置
  size_t start = s.find_first_not_of(WHITESPACE);
  if (start == std::string::npos) {
    // 全是空白
    return "";
  }
  // 找到最后一个非空白字符位置
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

// 聊天主循环
void ChatPipe::chat(std::string input_str, std::string media_path) {

  int token = 0;
  int tok_num = 0;
  int max_posid = 0;
  media_path = strip(media_path);
  auto media_type = get_media_type(media_path);
  if (media_type == ChatPipe::UNKNOWN) {
    std::cout
        << "Unsupported media type. Please provide a valid image or video."
        << std::endl;
    return;
  }

  std::cout << "\nAnswer:\n";
  std::string sentence_input = build_prompt(input_str, media_type);
  // std::cout << "Prompt: " << sentence_input << std::endl;

  std::vector<int> raw_tokens = encode_input(sentence_input);
  switch (media_type) {
  case ChatPipe::IMAGE: {
    std::vector<float> pixel_values;
    auto ret = process_image(pixel_values, media_path, config);
    if (ret == false) {
      std::cerr << "Error processing image: " << media_path << std::endl;
      return;
    }
    std::vector<int> tokens = maker->insert_tokens(raw_tokens, IMAGE_PAD_TOKEN);
    if ((int)(tokens.size()) > model.MAX_INPUT_LENGTH) {
      std::cerr << "Input tokens exceed maximum length: "
                << model.MAX_INPUT_LENGTH << std::endl;
      return;
    }

    int vit_offset = 0;
    vit_offset = find_token_offset(tokens, IMAGE_PAD_TOKEN);
    model.forward_embed(tokens);
    vit_process_image(pixel_values, vit_offset);
    std::vector<std::vector<std::vector<int>>> position_ids =
        get_rope_index({tokens}, {config.grid_thw}, IMAGE_PAD_TOKEN);

    // 找到三维数组position_ids中的最大值
    for (const auto &sub_tensor : position_ids[0]) {
      for (int val : sub_tensor) {
        if (val > max_posid) {
          max_posid = val;
        }
      }
    }
    // 将三维数组position_ids转换维1维
    std::vector<int> position_ids_1d;
    for (const auto &two_dim_tensor : position_ids) {
      for (const auto &one_dim_tensor : two_dim_tensor) {
        position_ids_1d.insert(position_ids_1d.end(), one_dim_tensor.begin(),
                               one_dim_tensor.end());
      }
    }
    token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
  } break;
  case VIDEO: {
    std::vector<float> pixel_values;
    auto ret = process_video(pixel_values, media_path, config);
    if (ret == false) {
      std::cerr << "Error processing video: " << media_path << std::endl;
      return;
    }
    std::vector<int> tokens = maker->insert_tokens(raw_tokens, VIDEO_PAD_TOKEN);
    if ((int)(tokens.size()) > model.MAX_INPUT_LENGTH) {
      std::cerr << "Input tokens exceed maximum length: "
                << model.MAX_INPUT_LENGTH << std::endl;
      return;
    }
    auto vit_offset = find_token_offset(tokens, VIDEO_PAD_TOKEN);
    model.forward_embed(tokens);
    vit_process_video(pixel_values, vit_offset);
    std::vector<std::vector<std::vector<int>>> position_ids =
        get_rope_index({tokens}, {config.grid_thw}, VIDEO_PAD_TOKEN);

    // 找到三维数组position_ids中的最大值
    for (const auto &sub_tensor : position_ids[0]) {
      for (int val : sub_tensor) {
        if (val > max_posid) {
          max_posid = val;
        }
      }
    }
    // 将三维数组position_ids转换维1维
    std::vector<int> position_ids_1d;
    for (const auto &two_dim_tensor : position_ids) {
      for (const auto &one_dim_tensor : two_dim_tensor) {
        position_ids_1d.insert(position_ids_1d.end(), one_dim_tensor.begin(),
                               one_dim_tensor.end());
      }
    }
    token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
  } break;
  case TEXT: {
    if ((int)(raw_tokens.size()) > model.MAX_INPUT_LENGTH) {
      std::cerr << "Input tokens exceed maximum length: "
                << model.MAX_INPUT_LENGTH << std::endl;
      return;
    }
    model.forward_embed(raw_tokens);
    auto position_ids_1d = get_position_ids(raw_tokens.size());
    max_posid = raw_tokens.size() - 1;
    token = forward_prefill(position_ids_1d, max_posid, history_max_posid);
  } break;
  default:
    std::cerr << "Unsupported media type." << std::endl;
    return;
  }
  // 后续分词
  std::vector<int> full_word_tokens;
  std::string text;
  while (token != ID_IM_END && model.history_length < model.SEQLEN) {
    // std::cout << "\nfull_word_tokens: " << token << "  " << std::endl;
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
      full_word_tokens.clear();
    }
    max_posid++;
    std::vector<int> following_position_ids = {max_posid, max_posid, max_posid};
    token = model.forward_next(following_position_ids);
    tok_num++;
  }
  history_max_posid = max_posid + 2;
  std::cout << std::endl;
}

// 构建提示
std::string ChatPipe::build_prompt(std::string input_str,
                                   MediaType media_type) {
  std::string prompt = sys_config;
  prompt += "<|im_start|>user\n";
  switch (media_type) {
  case IMAGE:
    prompt += "<|vision_start|><|image_pad|><|vision_end|>";
    break;
  case VIDEO:
    prompt += "<|vision_start|><|video_pad|><|vision_end|>";
    break;
  default:
    break;
  }
  prompt += input_str + "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

// 查找分词偏移量
int ChatPipe::find_token_offset(const std::vector<int> &input_ids, int pad_id) {
  for (size_t i = 0; i < input_ids.size(); ++i)
    if (input_ids[i] == pad_id)
      return static_cast<int>(i);
  return 0;
}

std::vector<float>
ChatPipe::reorder_hidden_states(const std::vector<float> &hidden_states,
                                int seq_len,
                                const std::vector<int> &window_index) {
  int w = seq_len;
  int h = hidden_states.size() / seq_len;
  // 转换为 w×h 大小的矩阵
  std::vector<std::vector<float>> matrix(w, std::vector<float>(h));
  for (int i = 0; i < w; ++i) {
    for (int j = 0; j < h; ++j) {
      matrix[i][j] = hidden_states[i * h + j];
    }
  }

  // 执行第一步重塑操作
  int first_dim = seq_len / spatial_merge_unit;
  std::vector<std::vector<std::vector<float>>> reshaped_1(
      first_dim, std::vector<std::vector<float>>(spatial_merge_unit,
                                                 std::vector<float>(h)));
  for (int i = 0; i < first_dim; ++i) {
    for (int j = 0; j < spatial_merge_unit; ++j) {
      for (int k = 0; k < h; ++k) {
        reshaped_1[i][j][k] = matrix[i * spatial_merge_unit + j][k];
      }
    }
  }

  // 执行索引操作
  std::vector<std::vector<std::vector<float>>> indexed(
      window_index.size(), std::vector<std::vector<float>>(
                               spatial_merge_unit, std::vector<float>(h)));
  for (size_t i = 0; i < window_index.size(); ++i) {
    int idx = window_index[i];
    for (int j = 0; j < spatial_merge_unit; ++j) {
      for (int k = 0; k < h; ++k) {
        indexed[i][j][k] = reshaped_1[idx][j][k];
      }
    }
  }

  // 执行第二步重塑操作
  std::vector<float> final_result(seq_len * h);
  for (int i = 0; i < seq_len; ++i) {
    for (int j = 0; j < h; ++j) {
      int outer_idx = i / spatial_merge_unit;
      int inner_idx = i % spatial_merge_unit;
      final_result[i * h + j] = indexed[outer_idx][inner_idx][j];
    }
  }

  return final_result;
}

std::vector<int>
calculate_cu_seqlens(const std::vector<std::vector<int>> &grid_thw) {
  std::vector<int> intermediate;
  // 对应 torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:,
  // 0])
  for (size_t i = 0; i < grid_thw.size(); ++i) {
    int product = grid_thw[i][1] * grid_thw[i][2];
    for (int j = 0; j < grid_thw[i][0]; ++j) {
      intermediate.push_back(product);
    }
  }

  // 对应 cumsum(dim=0, dtype=torch.int32)
  std::vector<int> cu_seqlens;
  int sum = 0;
  for (int val : intermediate) {
    sum += val;
    cu_seqlens.push_back(sum);
  }

  // 对应 F.pad(cu_seqlens, (1, 0), value=0)
  cu_seqlens.insert(cu_seqlens.begin(), 0);

  return cu_seqlens;
}

// 处理图像
void ChatPipe::vit_process_image(std::vector<float> &pixel_values,
                                 int vit_offset) {

  // hidden_states 在长度上等于pixel_values
  int t = config.grid_thw[0];
  int h = config.grid_thw[1];
  int w = config.grid_thw[2];
  std::vector<std::vector<int>> grid_thw = {config.grid_thw};

  // 调用 rot_pos 生成 position_ids
  std::vector<std::vector<int>> pos_ids_vec = rot_pos(grid_thw);

  // std::cout << "pos_ids_vec size: " << pos_ids_vec.size() << "  " <<
  // pos_ids_vec[0].size() << std::endl;
  std::vector<int> position_ids;
  for (const auto &v : pos_ids_vec) {
    position_ids.insert(position_ids.end(), v.begin(), v.end());
  }

  // 调用 get_window_index
  auto [window_index, cu_window_seqlens] = get_window_index(grid_thw);

  int seq_len = t * h * w;

  std::vector<float> processed_hidden_states =
      reorder_hidden_states(pixel_values, seq_len, window_index);
  std::vector<float> float_position_ids = convertIntToFloat(position_ids);
  std::vector<float> processed_position_ids =
      reorder_hidden_states(float_position_ids, seq_len, window_index);
  std::vector<int> int_processed_position_ids =
      convertFloatToInt(processed_position_ids);

  std::vector<int> cu_seqlens = calculate_cu_seqlens(grid_thw);
  // 生成掩码
  std::vector<float> full_attn_mask = get_attn_mask(seq_len, cu_seqlens);
  std::vector<float> window_attn_mask =
      get_attn_mask(seq_len, cu_window_seqlens);

  // reverse_indices
  std::vector<int> reverse_indices(window_index.size());
  for (size_t i = 0; i < window_index.size(); ++i)
    reverse_indices[window_index[i]] = i;

  model.forward_vit(processed_hidden_states, int_processed_position_ids,
                    full_attn_mask, window_attn_mask, config.grid_thw,
                    reverse_indices, vit_offset);
}

void ChatPipe::vit_process_video(std::vector<float> &pixel_values,
                                 int &vit_offset) {
  // hidden_states 在长度上等于pixel_values
  int t = config.grid_thw[0];
  int h = config.grid_thw[1];
  int w = config.grid_thw[2];
  int per_t = config.MAX_PATCHES / (h * w);
  std::vector<int> t_list;
  if (per_t >= t) {
    t_list.push_back(t);
  } else {
    for (int i = 0; i < t; i += per_t) {
      int unit = std::min(per_t, t - i);
      t_list.push_back(unit);
    }
  }
  int t_offset = 0;
  int v_offset = vit_offset;
  for (auto t_i : t_list) {
    // 调用 rot_pos 生成 position_ids
    std::vector<std::vector<int>> grid_thw = {{t_i, h, w}};
    std::vector<std::vector<int>> pos_ids_vec = rot_pos(grid_thw);

    std::vector<int> position_ids;
    for (const auto &v : pos_ids_vec) {
      position_ids.insert(position_ids.end(), v.begin(), v.end());
    }

    // 调用 get_window_index
    auto [window_index, cu_window_seqlens] = get_window_index(grid_thw);

    int seq_len = t_i * h * w;
    std::vector<float> pixel_current(t_i * h * w * model.VIT_DIMS);
    std::copy(pixel_values.begin() + t_offset * h * w * model.VIT_DIMS,
              pixel_values.begin() + (t_offset + t_i) * h * w * model.VIT_DIMS,
              pixel_current.begin());
    std::vector<float> processed_hidden_states =
        reorder_hidden_states(pixel_current, seq_len, window_index);
    std::vector<float> float_position_ids = convertIntToFloat(position_ids);
    std::vector<float> processed_position_ids =
        reorder_hidden_states(float_position_ids, seq_len, window_index);
    std::vector<int> int_processed_position_ids =
        convertFloatToInt(processed_position_ids);

    std::vector<int> cu_seqlens = calculate_cu_seqlens(grid_thw);
    // 生成掩码
    std::vector<float> full_attn_mask = get_attn_mask(seq_len, cu_seqlens);
    std::vector<float> window_attn_mask =
        get_attn_mask(seq_len, cu_window_seqlens);

    // reverse_indices
    std::vector<int> reverse_indices(window_index.size());
    for (size_t i = 0; i < window_index.size(); ++i) {
      reverse_indices[window_index[i]] = i;
    }

    model.forward_vit(processed_hidden_states, int_processed_position_ids,
                      full_attn_mask, window_attn_mask, config.grid_thw,
                      reverse_indices, v_offset);
    t_offset += t_i;
    v_offset += (t_i * h * w / 4);
  }
}

// 编码输入
std::vector<int> ChatPipe::encode_input(const std::string &sentence_input) {
  return tok->Encode(sentence_input);
}

void ChatPipe::print_chat_instructions() {
  std::cout
      << "\n=================================================================\n"
      << "1. If you want to quit, please enter one of [q, quit, exit]\n"
      << "2. To create a new chat session, please enter one of [clear, new]\n"
      << "=================================================================\n";
}

void Usage() {
  printf("Usage:\n"
         "  -h, --help      : Show help info \n"
         "  -m, --model     : Set model path \n"
         "  -c, --config    : Set config path \n"
         "  -v, --video_ratio : Set video ratio, default is 0.25\n"
         "  -d, --devid     : Set devices to run for model, default is '0'\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &config_path, std::string &image_path,
                      int &device, float &video_ratio) {
  struct option longOptions[] = {
      {"model", required_argument, nullptr, 'm'},
      {"config", required_argument, nullptr, 'c'},
      {"devid", required_argument, nullptr, 'd'},
      {"video_ratio", required_argument, nullptr, 'v'},
      {"help", no_argument, nullptr, 'h'},
      {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;
  while ((option = getopt_long(argc, argv, "m:c:d:v:h", longOptions,
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
    case 'v':
      video_ratio = atof(optarg);
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
  float video_ratio = 0.25f; // 默认视频比例为0.25

  processArguments(argc, argv, model_path, config_path, image_path, dev_id,
                   video_ratio);
  if (model_path.empty() || config_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  std::string system_prompt = "You are a helpful assistant.";
  ChatPipe pipeline0(dev_id, video_ratio, model_path, config_path,
                     system_prompt);
  ChatPipe pipeline1(dev_id, video_ratio, model_path, config_path,
                     system_prompt);
  printf("Chat0 =========================》\n");
  pipeline0.chat("What is in the image?", "./test.jpg");
  printf("Chat1 =========================》\n");
  pipeline1.chat("Who are you ?", "");
  return 0;
}