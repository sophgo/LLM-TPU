#include <chat.h>
#include <spdlog/spdlog.h>
#include <atomic>
#include <cstdio>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

void ChatGLM2Inner::complete_stream_tokens(
        std::vector<int> tokens,
        int              length_limit) {
    spdlog::info("Initiang to complete using tokens");

    auto start_time = NOW_TIME;
    flag.store(true, std::memory_order_release);
    std::string gen_str;

    int pre_token = 0;
    int token = forward_first(tokens);

    auto first_token = NOW_TIME;

    int tok_num = 0;
    while (token != EOS && token_length < MAX_LEN &&
           flag.load(std::memory_order_acquire)) {
        std::string      pre_word;
        std::string      word;
        std::vector<int> pre_ids = {pre_token};
        std::vector<int> ids = {pre_token, token};
        sentencepiece.Decode(pre_ids, &pre_word);
        sentencepiece.Decode(ids, &word);

        std::string diff = word.substr(pre_word.size());
        history += diff;

        gen_str = diff;

        {
            std::unique_lock<std::mutex> lock(mu);
            stream_datas.push(gen_str);
            cond.notify_one();
        }

        if (token_length < MAX_LEN) {
            token_length++;
        }
        tok_num++;

        if (tok_num >= length_limit) {
            {
                spdlog::warn(
                        "Token length reaches the limit: {}", length_limit);
                std::unique_lock<std::mutex> lock(mu);
                stream_datas.push("##LENGTH");

                cond.notify_one();
            }
            break;
        }
        token = forward_next();
    }

    auto end_time = NOW_TIME;

    auto milliseconds = DUARTION_MILISEC(start_time, end_time);
    auto first_dura = DUARTION_MILISEC(start_time, first_token);
    spdlog::info(
            "The inference wraps up in {} seconds, utilizing {} tokens at a rate of {} tokens per second. The initial token incurs a time cost of {} milliseconds.",
            milliseconds * 1.0 / 1000,
            tok_num,
            tok_num * 1000.0 / milliseconds,
            first_dura);

    if (token_length >= MAX_LEN) {
        round = 0;
        history = history.substr(history.size() / 2);
    } else {
        history += "\n\n";
        round++;
    }

    {
        std::unique_lock<std::mutex> lock(mu);
        stream_datas.push("##STOP");
        cond.notify_one();
    }
}

void ChatGLM2Inner::complete_stream(const char* input, int length_limit) {
    flag.store(true, std::memory_order_release);
    std::string generated;
    history = std::string(input);
    std::vector<int> tokens;

    sentencepiece.Encode(history, &tokens);

    if (tokens.empty()) {
        history = "";
        round = 0;
        generated = "Sorry: your question is too wierd!!\n";

        {
            std::unique_lock<std::mutex> lock(mu);
            stream_datas.push(generated);

            stream_datas.push("##ERROR");

            cond.notify_one();
        }

        return;
    }
    if (tokens.size() > MAX_LEN - 10) {
        round = 0;
        history = "";
        generated = "Error: your question is too large!\n";

        {
            std::unique_lock<std::mutex> lock(mu);
            stream_datas.push(generated);

            stream_datas.push("##LENGTH");

            cond.notify_one();
        }

        return;
    }
    int pre_token = 0;
    int token = forward_first(tokens);

    int tok_num = 0;
    while (token != EOS && token_length < MAX_LEN &&
           flag.load(std::memory_order_acquire)) {
        std::string      pre_word;
        std::string      word;
        std::vector<int> pre_ids = {pre_token};
        std::vector<int> ids = {pre_token, token};
        sentencepiece.Decode(pre_ids, &pre_word);
        sentencepiece.Decode(ids, &word);

        std::string diff = word.substr(pre_word.size());
        history += diff;

        generated = diff;

        {
            std::unique_lock<std::mutex> lock(mu);
            stream_datas.push(generated);
            cond.notify_one();
        }

        if (token_length < MAX_LEN) {
            token_length++;
        }
        tok_num++;

        if (tok_num >= length_limit) {
            {
                std::unique_lock<std::mutex> lock(mu);
                stream_datas.push(generated);

                stream_datas.push("##LENGTH");

                cond.notify_one();
            }
            break;
        }
        token = forward_next();
    }

    if (token_length >= MAX_LEN) {
        round = 0;
        history = history.substr(history.size() / 2);
    } else {
        history += "\n\n";
        round++;
    }

    {
        std::unique_lock<std::mutex> lock(mu);
        stream_datas.push("##STOP");
        cond.notify_one();
    }
}

std::string ChatGLM2Inner::generate() {
    std::unique_lock<std::mutex> lock(mu);
    cond.wait(lock, [this] { return !stream_datas.empty(); });
    auto res = stream_datas.front();
    stream_datas.pop();
    return res;
}

void ChatGLM2Inner::run_stream(const char* input, int length_limit) {
    std::unique_lock<std::mutex> lk(mu);
    while (!stream_datas.empty())
        stream_datas.pop();
    std::thread t1{&ChatGLM2Inner::complete_stream, this, input, length_limit};
    t1.detach();
}

void ChatGLM2Inner::run_tokens_stream(
        std::vector<int>& tokens,
        int               length_limit) {
    std::unique_lock<std::mutex> lk(mu);
    while (!stream_datas.empty())
        stream_datas.pop();

    std::thread t1{
            &ChatGLM2Inner::complete_stream_tokens,
            this,
            std::vector<int>(tokens),
            length_limit};

    t1.detach();
}

std::vector<int> ChatGLM2Inner::complete_tokens(
        std::vector<int>& tokens,
        std::vector<int>& eos_ids,
        int               max_length) {
    std::vector<int> res;
    flag.store(true, std::memory_order_release);

    std::set<int> is_eos(eos_ids.begin(), eos_ids.end());

    if (tokens.empty()) {
        spdlog::warn("No tokens");
        return res;
    }

    if (tokens.size() > max_length - 10) {
        spdlog::warn("Too much tokens");
        return res;
    }
    auto start_time = NOW_TIME;
    int  token = forward_first(tokens);

    auto first_token = NOW_TIME;
    res.push_back(token);
    int tok_num = 1;

    while (!is_eos.count(token) && tok_num < max_length &&
           flag.load(std::memory_order_acquire)) {
        spdlog::info("Token Gen: {}", token);
        ++token_length;
        token = forward_next();
        ++tok_num;
        res.push_back(token);
    }
    auto end_time = NOW_TIME;
    auto milliseconds = DUARTION_MILISEC(start_time, end_time);
    auto first_dura = DUARTION_MILISEC(start_time, first_token);
    spdlog::info(
            "The inference wraps up in {} seconds, utilizing {} tokens at a rate of {} tokens per second. The initial token incurs a time cost of {} milliseconds.",
            milliseconds * 1.0 / 1000,
            tok_num,
            tok_num * 1000.0 / milliseconds,
            first_dura);
    std::string result;
    sentencepiece.Decode(res, &result);
    spdlog::debug("Decoding Result: {}", result);
    return res;
}
