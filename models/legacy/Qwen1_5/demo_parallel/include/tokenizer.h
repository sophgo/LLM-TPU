#pragma once

#include "tiktoken.h"
#include "base64.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

class QwenTokenizer {
public:
  QwenTokenizer(const std::string &tiktoken_path);

  auto encode(const std::string &text, int max_length) const
      -> std::vector<int>;

  auto decode(const std::vector<int> &ids) const -> std::string;

  auto encode_history(const std::vector<std::string> &history,
                      int max_length, std::string input_mode = "prompted") const -> std::vector<int>;

  auto build_prompt(const std::vector<std::string> &history, const std::string &input_mode) const
      -> std::string;

  auto is_special_id(int id) const -> bool;

  tiktoken::tiktoken tokenizer;
  const int eod_id = 151643;
  const int im_start_id = 151644;
  const int im_end_id = 151645;
};