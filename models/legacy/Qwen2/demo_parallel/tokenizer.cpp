#include "tokenizer.h"

static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

static std::pair<std::string, int> _parse(const std::string &line) {
  auto pos = line.find(" ");
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid encoder line: " + line);
  }

  auto token = base64::decode({line.data(), pos});
  int rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid encoder rank: " + line);
  }

  return {std::move(token), rank};
}

QwenTokenizer::QwenTokenizer(const std::string &tiktoken_path) {
  std::ifstream file(tiktoken_path);
  if (!file) {
    throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = _parse(line);

    if (!encoder.emplace(std::move(token), rank).second) {
      throw std::runtime_error("duplicate item: " + line);
    }
  }

  std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>",
                                            "<|im_end|>"};
  char buffer[14];
  for (size_t i = 0; i < 205; i++) {
    snprintf(buffer, 14, "<|extra_%zu|>", i);
    special_tokens_s.push_back(buffer);
  }
  size_t encoder_size = encoder.size();
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  special_tokens.reserve(special_tokens_s.size());
  for (size_t i = 0; i < special_tokens_s.size(); i++) {
    special_tokens[special_tokens_s[i]] = encoder_size + i;
  }

  tokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
}

auto QwenTokenizer::build_prompt(const std::vector<std::string> &history, const std::string &input_mode) const
    -> std::string {
  if (history.size() % 2 != 1) {
    std::cout << "invalid history size " << history.size();
    exit(-1);
  }

  std::ostringstream oss_prompt;

  if (input_mode == "prompted") {
    oss_prompt << "<|im_start|>system\nYou are a helpful assistant.<|im_end|>";
    for (size_t i = 0; i < history.size() - 1; i += 2) {
      oss_prompt << "\n<|im_start|>user\n"
                 << history[i] << "<|im_end|>\n<|im_start|>assistant\n"
                 << history[i + 1] << "<|im_end|>";
    }
    oss_prompt << "\n<|im_start|>user\n"
               << history.back() << "<|im_end|>\n<|im_start|>assistant\n";
  } else if (input_mode == "unprompted") {
    for (size_t i = 0; i < history.size(); i += 1) {
      oss_prompt << history[i];
    }
  }
  return oss_prompt.str();
}

auto QwenTokenizer::encode(const std::string &text, int max_length) const
    -> std::vector<int> {
  auto ids = tokenizer.encode(text);
  if ((int)ids.size() > max_length) {
    ids.erase(ids.begin(), ids.end() - max_length);
  }
  return ids;
}

auto QwenTokenizer::decode(const std::vector<int> &ids) const -> std::string {
  std::vector<int> normal_ids(ids);
  normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(),
                                  [this](int id) { return is_special_id(id); }),
                   normal_ids.end());
  auto text = tokenizer.decode(normal_ids);
  return text;
}

auto QwenTokenizer::encode_history(const std::vector<std::string> &history,
                                   int max_length,
                                   std::string input_mode) const
    -> std::vector<int> {
  std::string prompt = build_prompt(history, input_mode);
  std::vector<int> input_ids = encode(prompt, max_length);
  return input_ids;
}

auto QwenTokenizer::is_special_id(int id) const -> bool {
  return id == eod_id || id == im_start_id || id == im_end_id;
}