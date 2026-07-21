//===----------------------------------------------------------------------===//
//
// Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
//
// TPU-MLIR is licensed under the 2-Clause BSD License except for the
// third-party components.
//
//===----------------------------------------------------------------------===//

#include "chat.hpp"
#include "tokenizers-cpp/tokenizers_cpp.h"
#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using tokenizers::Tokenizer;

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
  ChatPipe(const std::string &model_path, const std::string &config_path,
           const std::string &system_prompt, bool save_history, bool do_sample,
           const std::vector<int> &devices, int rep_window);
  ~ChatPipe() { model.deinit(); }
  void chat();
  void answer(const std::string &input_str);

private:
  std::string build_prompt(const std::string &input_str);

  Qwen model;
  std::unique_ptr<Tokenizer> tok;
  std::vector<int> EOS;
  std::string sys_config;
  bool enable_history;
  std::vector<std::pair<std::string, std::string>> history_vector;
};

ChatPipe::ChatPipe(const std::string &model_path,
                   const std::string &config_path,
                   const std::string &system_prompt, bool save_history,
                   bool do_sample, const std::vector<int> &devices,
                   int rep_window) {
  model.init(model_path, config_path, do_sample, devices, rep_window);
  sys_config = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
  enable_history = save_history || model.support_history;

  // load tokenizer
  std::string tokenizer_path = config_path + "/tokenizer.json";
  std::cout << "Processor [" << tokenizer_path.c_str() << "] loading .... ";
  auto blob = LoadBytesFromFile(tokenizer_path);
  tok = Tokenizer::FromBlobJSON(blob);
  EOS.push_back(tok->TokenToId("<|endoftext|>"));
  EOS.push_back(tok->TokenToId("<|im_end|>"));
  for (auto id : model.eos_token_id) {
    EOS.push_back(id);
  }
  std::cout << "Done!" << std::endl;
}

std::string ChatPipe::build_prompt(const std::string &input_str) {
  std::string prompt;
  if (model.history_length == 0) {
    prompt = sys_config;
  }
  prompt += "<|im_start|>user\n";
  if (enable_history && !model.support_history) {
    for (const auto &item : history_vector) {
      prompt += item.first + "<|im_end|>\n" + "<|im_start|>assistant\n" +
                item.second + "<|im_end|>\n<|im_start|>user\n";
    }
  }
  prompt += input_str + "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

void ChatPipe::answer(const std::string &input_str) {
  std::string sentence_input = build_prompt(input_str);
  std::vector<int> tokens = tok->Encode(sentence_input);
  if (model.support_history) {
    // long inputs are prefilled in chunks of MAX_INPUT_LENGTH through
    // block_/block_kv_, so the bound is SEQLEN rather than MAX_INPUT_LENGTH
    if (model.history_length > 0 &&
        (model.history_length + (int)tokens.size() >= model.SEQLEN ||
         model.history_length > model.PREFILL_KV_LENGTH)) {
      std::cerr << "Warning: History is full, clear it to continue."
                << std::endl;
      model.clear_kv();
      // rebuild the prompt with the system config for the fresh session
      sentence_input = build_prompt(input_str);
      tokens = tok->Encode(sentence_input);
    }
    if ((int)tokens.size() >= model.SEQLEN) {
      std::cerr << "Error: Input length exceeds maximum sequence length of "
                << model.SEQLEN << std::endl;
      return;
    }
  } else if ((int)tokens.size() > model.MAX_INPUT_LENGTH) {
    std::cerr << "Error: Input length exceeds maximum input length of "
              << model.MAX_INPUT_LENGTH << std::endl;
    return;
  }
  int pre_token = 0;
  int tok_num = 0;
  auto t0 = std::chrono::system_clock::now();
  int token = model.forward_first(tokens);
  auto t1 = std::chrono::system_clock::now();

  std::string result;
  std::vector<int> pre_ids = {pre_token};
  std::string pre_word = tok->Decode(pre_ids);
  std::vector<int> full_word_token = {pre_token};
  while (std::find(EOS.begin(), EOS.end(), token) == EOS.end() &&
         model.history_length < model.SEQLEN) {
    full_word_token.push_back(token);
    std::string word = tok->Decode(full_word_token);
    std::string diff = word.substr(pre_word.size());
    if (diff.find("�") == std::string::npos) {
      full_word_token.clear();
      full_word_token.push_back(pre_token);
      result += diff;
      if (model.check_stop(result)) {
        break;
      }
      std::cout << diff << std::flush;
    }
    tok_num++;
    token = model.forward_next();
  }
  auto t2 = std::chrono::system_clock::now();
  auto use0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto use1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << std::endl;
  std::cout << "FTL: " << (use0.count() * 1e-6) << " s" << std::endl;
  std::cout << "TPS: " << tok_num / (use1.count() * 1e-6) << " token/s"
            << std::endl;
  if (model.support_history) {
    if (model.history_length >= model.SEQLEN) {
      model.clear_kv();
    }
  } else if (enable_history) {
    if (model.token_length >= model.SEQLEN) {
      history_vector.push_back({input_str, result});
      size_t half_size = history_vector.size() / 2;
      history_vector.erase(history_vector.begin(),
                           history_vector.begin() + half_size);
      std::cout << "history length exceed max sequence length, erase half"
                << std::endl;
    } else {
      history_vector.push_back({input_str, result});
    }
  } else {
    model.history_length = 0;
  }
  result.clear();
}

void ChatPipe::chat() {
  std::cout
      << "================================================================="
      << std::endl
      << "1. If you want to quit, please enter one of [q, quit, exit]"
      << std::endl
      << "2. To create a new chat session, please enter one of [clear, new]"
      << std::endl
      << "================================================================="
      << std::endl;
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str == "exit" || input_str == "q" || input_str == "quit") {
      break;
    }
    if (input_str == "clear" || input_str == "new") {
      history_vector = {};
      model.history_length = 0;
      model.clear_kv();
      std::cout << "New chat session created." << std::endl;
      continue;
    }
    std::cout << "\nAnswer: " << std::flush;
    answer(input_str);
    std::cout << std::endl;
  }
}

void Usage() {
  printf("Usage:\n"
         "  -h, --help      : Show help info.\n"
         "  -m, --model     : Set model path \n"
         "  -c, --config    : Set processor config path \n"
         "  -e, --enable_history : if set, enable history memory\n"
         "  -s, --do_sample : if set, sample by generation config\n"
         "  -d, --devid     : Set devices to run for model, default is '0'\n"
         "  -p, --prompt    : Programmatic mode prompt; if set, run a single\n"
         "                    inference and exit (non-interactive)\n"
         "  -t, --prompt_file : Path to a text file whose contents are used as the\n"
         "                    programmatic mode prompt. If --prompt is also set,\n"
         "                    the file contents come first, followed by the\n"
         "                    --prompt value (combined with a newline)\n"
         "  -w, --rep_window: Sliding window size for repetition penalty; only\n"
         "                    the last N tokens are penalized. 64 (default);\n"
         "                    0 penalizes the full context. Only used with -s\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &config_path, std::vector<int> &devices,
                      bool &enable_history, bool &do_sample,
                      std::string &prompt, bool &has_prompt,
                      std::string &prompt_file, int &rep_window) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"config", required_argument, nullptr, 'c'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"enable_history", no_argument, nullptr, 'e'},
                                 {"do_sample", no_argument, nullptr, 's'},
                                 {"prompt", required_argument, nullptr, 'p'},
                                 {"prompt_file", required_argument, nullptr, 't'},
                                 {"rep_window", required_argument, nullptr, 'w'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:c:d:p:t:w:esh", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'd':
      devices = {atoi(optarg)};
      break;
    case 'e':
      enable_history = true;
      break;
    case 's':
      do_sample = true;
      break;
    case 'p':
      prompt = optarg;
      has_prompt = true;
      break;
    case 't':
      prompt_file = optarg;
      has_prompt = true;
      break;
    case 'w':
      rep_window = atoi(optarg);
      break;
    case 'h':
    case '?':
      Usage();
      exit(EXIT_SUCCESS);
    default:
      exit(EXIT_FAILURE);
    }
  }
}

int main(int argc, char **argv) {
  std::string model_path;
  std::string config_path = "../../config";
  std::vector<int> devices = {0};
  bool enable_history = false;
  bool do_sample = false;
  std::string prompt;
  std::string prompt_file;
  bool has_prompt = false;
  int rep_window = 64;

  processArguments(argc, argv, model_path, config_path, devices, enable_history,
                   do_sample, prompt, has_prompt, prompt_file, rep_window);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  // If --prompt_file was provided, load the prompt text from that file. If
  // --prompt is also given, the file contents come first and the --prompt
  // value is appended afterwards so that the two can be combined.
  if (has_prompt && !prompt_file.empty()) {
    std::ifstream fs(prompt_file, std::ios::in | std::ios::binary);
    if (fs.fail()) {
      std::cerr << "Cannot open prompt file [ " << prompt_file << " ]"
                << std::endl;
      exit(EXIT_FAILURE);
    }
    std::ostringstream oss;
    oss << fs.rdbuf();
    std::string file_prompt = oss.str();
    // Trim a trailing newline so the file behaves like a CLI-supplied prompt.
    while (!file_prompt.empty() &&
           (file_prompt.back() == '\n' || file_prompt.back() == '\r')) {
      file_prompt.pop_back();
    }
    if (prompt.empty()) {
      prompt = file_prompt;
    } else {
      prompt = file_prompt + "\n" + prompt;
    }
  }

  std::string system_prompt = "You are a helpful assistant.";

  std::cout << "Init Environment ..." << std::endl;
  ChatPipe pipeline(model_path, config_path, system_prompt, enable_history,
                    do_sample, devices, rep_window);
  if (has_prompt) {
    // Programmatic (non-interactive) mode: run a single inference and exit.
    // std::cout << "\nQuestion: " << prompt << std::endl;
    std::cout << "\nAnswer: " << std::flush;
    pipeline.answer(prompt);
    std::cout << std::endl;
  } else {
    pipeline.chat();
  }
  return 0;
}
