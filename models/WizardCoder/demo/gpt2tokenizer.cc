#include <bits/stdc++.h>
#include <simdjson.h>
#include "ctre-unicode.hpp"
#include "include/tokenizer.h"

namespace fs = std::filesystem;

std::unordered_map<char, std::string> bytes_to_unicode() {
    static const std::unordered_map<char, std::string> code_map = {
            {33, "!"},  {34, "\""}, {35, "#"},  {36, "$"},  {37, "%"},
            {38, "&"},  {39, "\'"}, {40, "("},  {41, ")"},  {42, "*"},
            {43, "+"},  {44, ","},  {45, "-"},  {46, "."},  {47, "/"},
            {48, "0"},  {49, "1"},  {50, "2"},  {51, "3"},  {52, "4"},
            {53, "5"},  {54, "6"},  {55, "7"},  {56, "8"},  {57, "9"},
            {58, ":"},  {59, ";"},  {60, "<"},  {61, "="},  {62, ">"},
            {63, "?"},  {64, "@"},  {65, "A"},  {66, "B"},  {67, "C"},
            {68, "D"},  {69, "E"},  {70, "F"},  {71, "G"},  {72, "H"},
            {73, "I"},  {74, "J"},  {75, "K"},  {76, "L"},  {77, "M"},
            {78, "N"},  {79, "O"},  {80, "P"},  {81, "Q"},  {82, "R"},
            {83, "S"},  {84, "T"},  {85, "U"},  {86, "V"},  {87, "W"},
            {88, "X"},  {89, "Y"},  {90, "Z"},  {91, "["},  {92, "\\"},
            {93, "]"},  {94, "^"},  {95, "_"},  {96, "`"},  {97, "a"},
            {98, "b"},  {99, "c"},  {100, "d"}, {101, "e"}, {102, "f"},
            {103, "g"}, {104, "h"}, {105, "i"}, {106, "j"}, {107, "k"},
            {108, "l"}, {109, "m"}, {110, "n"}, {111, "o"}, {112, "p"},
            {113, "q"}, {114, "r"}, {115, "s"}, {116, "t"}, {117, "u"},
            {118, "v"}, {119, "w"}, {120, "x"}, {121, "y"}, {122, "z"},
            {123, "{"}, {124, "|"}, {125, "}"}, {126, "~"}, {161, "¡"},
            {162, "¢"}, {163, "£"}, {164, "¤"}, {165, "¥"}, {166, "¦"},
            {167, "§"}, {168, "¨"}, {169, "©"}, {170, "ª"}, {171, "«"},
            {172, "¬"}, {174, "®"}, {175, "¯"}, {176, "°"}, {177, "±"},
            {178, "²"}, {179, "³"}, {180, "´"}, {181, "µ"}, {182, "¶"},
            {183, "·"}, {184, "¸"}, {185, "¹"}, {186, "º"}, {187, "»"},
            {188, "¼"}, {189, "½"}, {190, "¾"}, {191, "¿"}, {192, "À"},
            {193, "Á"}, {194, "Â"}, {195, "Ã"}, {196, "Ä"}, {197, "Å"},
            {198, "Æ"}, {199, "Ç"}, {200, "È"}, {201, "É"}, {202, "Ê"},
            {203, "Ë"}, {204, "Ì"}, {205, "Í"}, {206, "Î"}, {207, "Ï"},
            {208, "Ð"}, {209, "Ñ"}, {210, "Ò"}, {211, "Ó"}, {212, "Ô"},
            {213, "Õ"}, {214, "Ö"}, {215, "×"}, {216, "Ø"}, {217, "Ù"},
            {218, "Ú"}, {219, "Û"}, {220, "Ü"}, {221, "Ý"}, {222, "Þ"},
            {223, "ß"}, {224, "à"}, {225, "á"}, {226, "â"}, {227, "ã"},
            {228, "ä"}, {229, "å"}, {230, "æ"}, {231, "ç"}, {232, "è"},
            {233, "é"}, {234, "ê"}, {235, "ë"}, {236, "ì"}, {237, "í"},
            {238, "î"}, {239, "ï"}, {240, "ð"}, {241, "ñ"}, {242, "ò"},
            {243, "ó"}, {244, "ô"}, {245, "õ"}, {246, "ö"}, {247, "÷"},
            {248, "ø"}, {249, "ù"}, {250, "ú"}, {251, "û"}, {252, "ü"},
            {253, "ý"}, {254, "þ"}, {255, "ÿ"}, {0, "Ā"},   {1, "ā"},
            {2, "Ă"},   {3, "ă"},   {4, "Ą"},   {5, "ą"},   {6, "Ć"},
            {7, "ć"},   {8, "Ĉ"},   {9, "ĉ"},   {10, "Ċ"},  {11, "ċ"},
            {12, "Č"},  {13, "č"},  {14, "Ď"},  {15, "ď"},  {16, "Đ"},
            {17, "đ"},  {18, "Ē"},  {19, "ē"},  {20, "Ĕ"},  {21, "ĕ"},
            {22, "Ė"},  {23, "ė"},  {24, "Ę"},  {25, "ę"},  {26, "Ě"},
            {27, "ě"},  {28, "Ĝ"},  {29, "ĝ"},  {30, "Ğ"},  {31, "ğ"},
            {32, "Ġ"},  {127, "ġ"}, {128, "Ģ"}, {129, "ģ"}, {130, "Ĥ"},
            {131, "ĥ"}, {132, "Ħ"}, {133, "ħ"}, {134, "Ĩ"}, {135, "ĩ"},
            {136, "Ī"}, {137, "ī"}, {138, "Ĭ"}, {139, "ĭ"}, {140, "Į"},
            {141, "į"}, {142, "İ"}, {143, "ı"}, {144, "Ĳ"}, {145, "ĳ"},
            {146, "Ĵ"}, {147, "ĵ"}, {148, "Ķ"}, {149, "ķ"}, {150, "ĸ"},
            {151, "Ĺ"}, {152, "ĺ"}, {153, "Ļ"}, {154, "ļ"}, {155, "Ľ"},
            {156, "ľ"}, {157, "Ŀ"}, {158, "ŀ"}, {159, "Ł"}, {160, "ł"},
            {173, "Ń"}};
    return code_map;
}

static constexpr ctll::fixed_string pattern{
        R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)"};

std::unordered_map<std::string, char> unicode_to_bytes() {
    static std::unordered_map<std::string, char> code_map;
    auto                                         map = bytes_to_unicode();
    for (auto&& [k, v] : map)
        code_map[v] = k;
    return code_map;
}

std::optional<GPT2Tokenizer> GPT2Tokenizer::from_pretrained(
        std::string_view model_path) {
    fs::path path(model_path);

    auto merges_path = path.parent_path() / "merges.txt";
    auto vocab_path = path.parent_path() / "vocab.json";

    std::ifstream merges_ifs(merges_path);

    if (!merges_ifs.good()) {
        std::cerr << "open merges.txt error\n";
        return std::nullopt;
    }

    auto        result = GPT2Tokenizer();
    BPERanks    bpe_ranks;
    std::string merges_version;
    std::getline(merges_ifs, merges_version);

    for (struct {
             std::string line;
             size_t      i{0};
         } it;
         std::getline(merges_ifs, it.line);
         ++it.i) {
        const size_t                        split_point = it.line.find(' ');
        std::pair<std::string, std::string> p{
                {it.line.begin(), it.line.begin() + split_point},
                {it.line.begin() + split_point + 1, it.line.end()}};
        bpe_ranks.emplace(std::move(p), it.i);
    }

    result.m_bpe_ranks = std::move(bpe_ranks);

    simdjson::dom::parser  parser;
    simdjson::dom::object  object;
    simdjson::dom::element doc = parser.load(vocab_path);

    auto error = doc.get(object);
    if (error) return std::nullopt;

    Encoder encoder;
    Decoder decoder;

    for (auto&& [k, v] : object) {
        auto value = v.get_int64().value();
        encoder.emplace(k, value);
        decoder.emplace(value, k);
    }

    result.m_encoder = std::move(encoder);
    result.m_decoder = std::move(decoder);

    result.m_byte_encoder = bytes_to_unicode();
    result.m_byte_decoder = unicode_to_bytes();

    result.add_special_token("</s>");
    result.add_special_token("<pad>");

    return result;
}

std::vector<std::string> GPT2Tokenizer::tokenize(std::string_view text) {
    std::vector<std::string> bpe_tokens;
    for (auto m : ctre::range<pattern>(text)) {
        auto        token = m.to_string();
        std::string byte_token;
        for (auto&& e : token)
            byte_token += m_byte_encoder[e];
        auto result = bpe(byte_token);
        bpe_tokens.insert(bpe_tokens.end(), result.begin(), result.end());
    }

    return bpe_tokens;
}

inline int get_charsize(char c) {
    if ((c & 0xe0) == 0xc0)
        return 2;
    else if ((c & 0xf0) == 0xe0)
        return 3;
    else if ((c & 0xf8) == 0xf0)
        return 4;
    return 1;
}

std::string GPT2Tokenizer::decode(
        const std::vector<int>& token_ids,
        bool                    skip_special_token) {
    std::string decoded_string;
    for (auto&& id : token_ids) {
        auto decoded_token = m_decoder[id];
        for (int i = 0; i < decoded_token.size();) {
            int len = get_charsize(decoded_token[i]);
            decoded_string += m_byte_decoder[decoded_token.substr(i, len)];
            i += len;
        }
    }
    if (skip_special_token && m_special_tokens.contains(decoded_string))
        decoded_string = " ";
    return decoded_string;
}

std::string GPT2Tokenizer::decode_id(int id, bool skip_special_token) {
    return decode(std::vector<int>{id}, skip_special_token);
}

std::vector<int> GPT2Tokenizer::encode(std::string_view input_text) {
    std::vector<int> res;
    auto             tokens = tokenize(input_text);
    for (auto&& e : tokens)
        res.push_back(m_encoder[e]);
    return res;
}

std::vector<std::string> GPT2Tokenizer::bpe(const std::string& token) {
    std::vector<BPERanks::const_iterator> ranks;
    std::vector<std::string>              word;
    ranks.reserve(token.size() - 1);
    word.reserve(token.size());

    {
        size_t i = 0;
        while (1) {
            int len = get_charsize(token[i]);
            int next_len = get_charsize(token[i + len]);
            ranks.push_back(m_bpe_ranks.find(
                    {token.substr(i, len), token.substr(i + len, next_len)}));
            word.push_back(token.substr(i, len));
            i += len;
            if (i >= token.size()) break;
            if (i + next_len >= token.size()) {
                word.emplace_back(token.substr(i, next_len));
                break;
            }
        }
    }

    while (1) {
        const auto bigram = std::min_element(
                ranks.begin(),
                ranks.end(),
                [this](const auto& lhs, const auto& rhs) -> bool {
                    if (lhs == m_bpe_ranks.end() && lhs == m_bpe_ranks.end())
                        return false;
                    else if (
                            lhs == m_bpe_ranks.end() ||
                            rhs == m_bpe_ranks.end()) {
                        return (lhs != m_bpe_ranks.end());
                    } else {
                        return lhs->second < rhs->second;
                    }
                });

        if (*bigram == m_bpe_ranks.end()) break;

        const auto& [first, second] = (*bigram)->first;
        std::vector<std::string> new_word;

        int i = 0;
        while (i < word.size()) {
            const auto j = std::find(word.begin() + i, word.end(), first);
            if (j == word.end()) {
                new_word.insert(new_word.end(), word.begin() + i, word.end());
                break;
            }

            new_word.insert(new_word.end(), word.begin() + i, j);
            i = j - word.begin();
            if (word[i] == first && i < word.size() - 1 &&
                word[i + 1] == second)
                new_word.push_back(first + second), i += 2;
            else
                new_word.push_back(word[i++]);
        }

        word = std::move(new_word);
        if (word.size() == 1)
            break;
        else {
            ranks.resize(word.size() - 1);
            for (size_t i = 0; i < word.size() - 1; ++i)
                ranks[i] = m_bpe_ranks.find({word[i], word[i + 1]});
        }
    }
    m_cache[token] = word;
    return word;
}
