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
#include "json.hpp"
#include "tokenizers-cpp/tokenizers_cpp.h"
#include <filesystem>
#include <fstream>
#include <sstream>

using tokenizers::Tokenizer;
using json = nlohmann::json;

// <IMG_CONTEXT>, the placeholder each ViT tile writes its embeddings over.
static const int MEDIA_TOKEN_ID = 151667;

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

typedef enum { TEXT, IMAGE, VIDEO, UNKNOWN } MediaType;

// Decide media type from the path extension; an empty path means pure text.
static MediaType get_media_type(const std::string &media_path) {
  if (media_path.empty()) {
    return TEXT;
  }
  auto pos = media_path.find_last_of('.');
  std::string ext =
      (pos == std::string::npos) ? "" : media_path.substr(pos + 1);
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  if (ext == "mp4" || ext == "mov" || ext == "avi" || ext == "mkv" ||
      ext == "wmv" || ext == "flv" || ext == "mpeg" || ext == "mpg") {
    return VIDEO;
  }
  return IMAGE; // any other existing file is treated as an image
}

// Extract "@path" media tokens from the question; strips them from input_str
// and returns the first media path (InternVL3 handles one media per turn).
static std::string extractMedia(std::string &input_str) {
  std::vector<std::string> medias;
  std::stringstream ss(input_str);
  std::string token, question;
  while (ss >> token) {
    if (token.size() > 1 && token[0] == '@') {
      medias.push_back(token.substr(1));
    } else {
      if (!question.empty()) {
        question += " ";
      }
      question += token;
    }
  }
  input_str = question;
  if (medias.size() > 1) {
    std::cout << "Only one media file is supported, using: " << medias[0]
              << std::endl;
  }
  return medias.empty() ? "" : medias[0];
}

// Replace the first occurrence of `from` in `s` with `to`.
static bool replace_first(std::string &s, const std::string &from,
                          const std::string &to) {
  size_t pos = s.find(from);
  if (pos == std::string::npos) {
    return false;
  }
  s.replace(pos, from.length(), to);
  return true;
}

static bool ends_with(const std::string &s, const std::string &suffix) {
  return suffix.size() <= s.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Parse comma-separated device ids ("0,1") into a vector, defaulting to {0}.
static std::vector<int> parse_devices(const std::string &s) {
  std::vector<int> devices;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      devices.push_back(std::stoi(item));
    }
  }
  if (devices.empty()) {
    devices.push_back(0);
  }
  return devices;
}

class ChatPipe {
public:
  void init(const std::vector<int> &devices, const std::string &model_path,
            const std::string &config_path, bool do_sample);
  void deinit();
  void chat();

private:
  void init_params(bool do_sample, const std::string &config_path);
  std::string build_prompt(const std::string &question,
                           const std::vector<int> &num_patches_list);
  void answer(const std::string &input_str, const std::string &media_path);
  void stream_answer(const std::vector<int> &tokens,
                     std::vector<float> &pixel_values);
  bool is_eos(int token) const;

  InternVL3 model;
  std::unique_ptr<Tokenizer> tok;
  PreConfig pre;
  std::string system_prompt;
  int ID_IMG_CONTEXT;
  std::vector<int> EOS;
  std::vector<std::string> stop_strings;
  bool support_history;
};

void ChatPipe::init(const std::vector<int> &devices,
                    const std::string &model_path,
                    const std::string &config_path, bool do_sample) {
  system_prompt =
      "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多"
      "家合作单位联合开发的多模态大语言模型。";

  // load tokenizer
  std::cout << "Config [" << config_path.c_str() << "] loading .... ";
  auto blob = LoadBytesFromFile((config_path + "/tokenizer.json").c_str());
  tok = Tokenizer::FromBlobJSON(blob);
  ID_IMG_CONTEXT = tok->TokenToId("<IMG_CONTEXT>");
  EOS = {tok->TokenToId("<|im_end|>")};
  std::cout << "Done!" << std::endl;

  // load model
  std::cout << "Init Environment ..." << std::endl;
  model.init(devices, model_path);
  support_history = model.support_history;

  init_params(do_sample, config_path);
}

void ChatPipe::deinit() { model.deinit(); }

// Mirror pipeline.py init_params: greedy by default; --do_sample reads the
// sampling parameters from generation_config.json (HF GenerationConfig
// defaults fill in any missing field).
void ChatPipe::init_params(bool do_sample, const std::string &config_path) {
  model.generation_mode = "greedy";
  stop_strings.clear();
  if (!do_sample) {
    return;
  }
  std::string gen_config_file = config_path + "/generation_config.json";
  std::ifstream f(gen_config_file);
  if (!f.is_open()) {
    std::cerr << "'" << gen_config_file << "' not found. '--do_sample' requires "
              << "generation_config.json to provide sampling parameters."
              << std::endl;
    exit(1);
  }
  json gc;
  f >> gc;
  model.generation_mode = "sample";
  model.temperature = gc.value("temperature", 1.0f);
  model.top_p = gc.value("top_p", 1.0f);
  model.top_k = gc.value("top_k", 50);
  model.penalty = gc.value("repetition_penalty", 1.0f);
  if (gc.contains("eos_token_id") && !gc["eos_token_id"].is_null()) {
    if (gc["eos_token_id"].is_number_integer()) {
      EOS.push_back(gc["eos_token_id"].get<int>());
    } else if (gc["eos_token_id"].is_array()) {
      for (auto &e : gc["eos_token_id"]) {
        EOS.push_back(e.get<int>());
      }
    }
  }
  if (gc.contains("stop_strings") && !gc["stop_strings"].is_null()) {
    if (gc["stop_strings"].is_string()) {
      stop_strings.push_back(gc["stop_strings"].get<std::string>());
    } else if (gc["stop_strings"].is_array()) {
      for (auto &s : gc["stop_strings"]) {
        stop_strings.push_back(s.get<std::string>());
      }
    }
  }
}

bool ChatPipe::is_eos(int token) const {
  for (int e : EOS) {
    if (token == e) {
      return true;
    }
  }
  return false;
}

// Mirror pipeline.py process_input's prompt assembly: emit one
// "Frame{i}: <image>\n" tag per media segment, then expand each <image> into
// <img> + <IMG_CONTEXT> * NUM_IMAGE_TOKEN * num_patches + </img>.
std::string ChatPipe::build_prompt(const std::string &input_str,
                                   const std::vector<int> &num_patches_list) {
  std::string question;
  for (size_t i = 0; i < num_patches_list.size(); ++i) {
    question += "Frame" + std::to_string(i + 1) + ": <image>\n";
  }
  question += input_str;

  const std::string IMG_START_TOKEN = "<img>";
  const std::string IMG_END_TOKEN = "</img>";
  const std::string IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>";
  for (int num_patches : num_patches_list) {
    int repeat = model.NUM_IMAGE_TOKEN * num_patches;
    std::string image_tokens = IMG_START_TOKEN;
    image_tokens.reserve(IMG_START_TOKEN.size() +
                         IMG_CONTEXT_TOKEN.size() * repeat +
                         IMG_END_TOKEN.size());
    for (int k = 0; k < repeat; ++k) {
      image_tokens += IMG_CONTEXT_TOKEN;
    }
    image_tokens += IMG_END_TOKEN;
    replace_first(question, "<image>", image_tokens);
  }

  std::string prompt;
  if (!support_history || model.history_length == 0) {
    prompt = "<|im_start|>system\n" + system_prompt + "<|im_end|>\n";
  }
  prompt += "<|im_start|>user\n" + question +
            "<|im_end|>\n<|im_start|>assistant\n";
  return prompt;
}

void ChatPipe::answer(const std::string &input_str,
                      const std::string &media_path) {
  MediaType media_type = get_media_type(media_path);

  std::vector<float> pixel_values;
  std::vector<int> num_patches_list;
  if (media_type == IMAGE) {
    int num = process_image(pixel_values, media_path, pre);
    if (num < 0) {
      return;
    }
    num_patches_list.push_back(num);
  } else if (media_type == VIDEO) {
    int frames = process_video(pixel_values, num_patches_list, media_path, pre);
    if (frames < 0) {
      return;
    }
  }

  std::string prompt = build_prompt(input_str, num_patches_list);
  std::vector<int> tokens = tok->Encode(prompt);
  int token_len = tokens.size();
  if (token_len > model.MAX_INPUT_LENGTH) {
    std::cout << "The maximum question length should be shorter than "
              << model.MAX_INPUT_LENGTH << " but we get " << token_len
              << " instead." << std::endl;
    return;
  }
  if (support_history) {
    if ((token_len + model.history_length > model.SEQLEN - 128) ||
        (model.history_length > model.PREFILL_KV_LENGTH)) {
      std::cout << "Warning: History is full and clear it to continue."
                << std::endl;
      model.clear_history();
    }
  }

  std::cout << "\nAnswer: " << std::flush;
  stream_answer(tokens, pixel_values);
}

void ChatPipe::stream_answer(const std::vector<int> &tokens,
                             std::vector<float> &pixel_values) {
  int tok_num = 0;
  auto t0 = std::chrono::system_clock::now();

  model.forward_embed(tokens);
  if (!pixel_values.empty()) {
    int vit_offset = -1;
    for (size_t i = 0; i < tokens.size(); ++i) {
      if (tokens[i] == ID_IMG_CONTEXT) {
        vit_offset = static_cast<int>(i);
        break;
      }
    }
    if (vit_offset < 0) {
      std::cerr << "Error: <IMG_CONTEXT> token not found in prompt."
                << std::endl;
      return;
    }
    model.forward_vit(pixel_values, vit_offset);
  }

  int token = model.forward_first();
  auto t1 = std::chrono::system_clock::now();

  std::vector<int> full_word_tokens;
  std::string text;
  while (!is_eos(token) && model.history_length < model.SEQLEN) {
    full_word_tokens.push_back(token);
    std::string word = tok->Decode(full_word_tokens);
    if (word.find("�") == std::string::npos) {
      // A lone token can lose its leading space when decoded in isolation;
      // decoding the pair and trimming recovers the in-context form.
      if (full_word_tokens.size() == 1) {
        std::string pre_word = word;
        std::vector<int> double_token = {token, token};
        word = tok->Decode(double_token).substr(pre_word.length());
      }
      text += word;
      bool stop = false;
      for (const auto &s : stop_strings) {
        if (!s.empty() && ends_with(text, s)) {
          stop = true;
          break;
        }
      }
      if (stop) {
        break;
      }
      std::cout << word << std::flush;
      full_word_tokens.clear();
    }
    token = model.forward_next();
    tok_num++;
  }
  auto t2 = std::chrono::system_clock::now();

  auto ftl = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
  auto decode = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
  std::cout << std::endl;
  std::cout << "FTL: " << (ftl.count() * 1e-6) << " s" << std::endl;
  if (tok_num > 0 && decode.count() > 0) {
    std::cout << "TPS: " << tok_num / (decode.count() * 1e-6) << " token/s"
              << std::endl;
  }
}

void ChatPipe::chat() {
  std::cout
      << "\n================================================================="
      << std::endl
      << "1. If you want to quit, please enter one of [/q, /quit, /exit]"
      << std::endl
      << "2. To create a new chat session, please enter one of [/clear, /new]"
      << std::endl
      << "3. To ask about an image or video, include @<path> in your question"
      << std::endl
      << "================================================================="
      << std::endl;
  while (true) {
    std::cout << "\nQuestion: ";
    std::string input_str;
    std::getline(std::cin, input_str);
    if (input_str == "/exit" || input_str == "/q" || input_str == "/quit") {
      break;
    }
    if (input_str == "/clear" || input_str == "/new" || input_str == "/c") {
      model.clear_history();
      std::cout << "New chat session created." << std::endl;
      continue;
    }

    // Media files are attached with @path in the question
    std::string media_path = extractMedia(input_str);
    if (!media_path.empty() && !std::filesystem::exists(media_path)) {
      std::cout << "Media file not found: " << media_path << std::endl;
      continue;
    }
    if (input_str.empty() && media_path.empty()) {
      std::cout << "Sorry: your question is empty!!" << std::endl;
      continue;
    }
    answer(input_str, media_path);
    std::cout << std::endl;
  }
}

void Usage() {
  printf("Usage:\n"
         "  -h, --help      : Show help info.\n"
         "  -m, --model     : Set model path \n"
         "  -c, --config    : Set config path, default is '../config'\n"
         "  -d, --devid     : Set devices to run for model, default is '0'\n"
         "                    (comma separated for multi-device, e.g. '0,1')\n"
         "  -s, --do_sample : if set, generate tokens by sample parameters\n");
}

void processArguments(int argc, char *argv[], std::string &model_path,
                      std::string &config_path, std::string &devid,
                      bool &do_sample) {
  struct option longOptions[] = {{"model", required_argument, nullptr, 'm'},
                                 {"config", required_argument, nullptr, 'c'},
                                 {"devid", required_argument, nullptr, 'd'},
                                 {"do_sample", no_argument, nullptr, 's'},
                                 {"help", no_argument, nullptr, 'h'},
                                 {nullptr, 0, nullptr, 0}};

  int optionIndex = 0;
  int option;

  while ((option = getopt_long(argc, argv, "m:c:d:sh", longOptions,
                               &optionIndex)) != -1) {
    switch (option) {
    case 'm':
      model_path = optarg;
      break;
    case 'c':
      config_path = optarg;
      break;
    case 'd':
      devid = optarg;
      break;
    case 's':
      do_sample = true;
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

int main(int argc, char **argv) {
  std::string model_path;
  std::string config_path = "../config";
  std::string devid = "0";
  bool do_sample = false;

  processArguments(argc, argv, model_path, config_path, devid, do_sample);
  if (model_path.empty()) {
    Usage();
    exit(EXIT_FAILURE);
  }

  std::vector<int> devices = parse_devices(devid);

  ChatPipe pipe;
  pipe.init(devices, model_path, config_path, do_sample);
  pipe.chat();
  pipe.deinit();
  return 0;
}
