#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <bits/stdc++.h>
template <class T>
inline void hash_combine(std::size_t& seed, const T& v) {
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct GPT2Tokenizer {
    struct PairHash {
        size_t operator()(
                const std::pair<std::string, std::string>& p) const noexcept {
            size_t seed = 0;
            hash_combine(seed, p.first);
            hash_combine(seed, p.second);
            return seed;
        }
    };

    using BPE = std::pair<std::string, std::string>;
    using BPERanks = std::unordered_map<BPE, size_t, PairHash>;
    using Encoder = std::unordered_map<std::string, int>;
    using Decoder = std::unordered_map<int, std::string>;
    using Cache = std::unordered_map<std::string, std::vector<std::string>>;

    BPERanks                              m_bpe_ranks;
    Encoder                               m_encoder;
    Decoder                               m_decoder;
    std::unordered_map<char, std::string> m_byte_encoder;
    std::unordered_map<std::string, char> m_byte_decoder;
    Cache                                 m_cache;
    std::unordered_set<std::string>       m_special_tokens;

    std::vector<int>         encode(std::string_view);
    std::string              decode(const std::vector<int>&, bool);
    std::string              decode_id(int id, bool);
    std::vector<std::string> bpe(const std::string&);
    std::vector<std::string> tokenize(std::string_view);

    size_t add_special_token(const std::string& token) {
        auto new_id = vocab_size() + m_special_tokens.size();
        if (!m_special_tokens.count(token)) m_special_tokens.insert(token);
        return new_id;
    }

    size_t vocab_size() const noexcept {
        return m_encoder.size();
    }

    static std::optional<GPT2Tokenizer> from_pretrained(std::string_view);
};

#endif