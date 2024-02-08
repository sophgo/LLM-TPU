#ifndef __CHAT_H__
#define __CHAT_H__

#include <bits/stdc++.h>
#include <sentencepiece_processor.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <ostream>
#include <queue>
#include <string>
#include <string_view>
#include <vector>
#include <bmruntime_interface.h>

static const int NUM_LAYERS = 28;
static const int MAX_LEN = 512;
static const int HIDDEN_SIZE = 4096;

class ChatGLM2 {
   public:
    void init(int devid, const char* model_dir);
    void chat();

    void deinit();
    void load_sentencepiece();

    void debug();

   protected:
    void answer(const std::string& input_str);
    void tokenizer_encode(
            const std::string& input_str,
            std::vector<int>&  tokens);
    int  forward_first(std::vector<int>& tokens);
    int  forward_next();
    void move2end(const bm_tensor_t& kv);

   protected:
    bm_handle_t                           bm_handle;
    void*                                 p_bmrt;
    sentencepiece::SentencePieceProcessor sentencepiece;
    const bm_net_info_t*                  net_blocks[NUM_LAYERS];
    const bm_net_info_t*                  net_blocks_cache[NUM_LAYERS];
    const bm_net_info_t*                  net_embed;
    const bm_net_info_t*                  net_lm;
    bm_tensor_t                           inputs_embed_512, outputs_embed_512;
    bm_tensor_t                           inputs_lm, outputs_lm;
    bm_tensor_t inputs_pid, next_pid, inputs_attention, next_attention;
    bm_tensor_t past_key[NUM_LAYERS], past_value[NUM_LAYERS];
    std::string name_embed;
    std::string name_lm;
    std::string name_blocks[NUM_LAYERS];
    std::string name_blocks_cache[NUM_LAYERS];
    std::string history = "";
    int         round = 0;
    int         token_length;
    int         EOS;
};

class ChatGLM2Inner : public ChatGLM2 {
   public:
    ChatGLM2Inner() = default;
    std::string get_histoty() const {
        return history;
    }

    std::string complete(const char* input, int length_limit);

    void complete_stream(const char* input, int length_limit);

    void complete_stream_tokens(std::vector<int> tokens, int length_limit);

    void run_stream(const char* input, int length_limit);

    void run_tokens_stream(std::vector<int>& tokens, int length_limit);

    void stop_inference() {
        flag.store(false, std::memory_order_release);
    }

    std::string token_2_piece(int token);
    int         token_2_piece(int token, char* buf, int length);

    std::vector<int> complete_tokens(
            std::vector<int>& tokens,
            std::vector<int>& eos_ids,
            int                     max_length);

    std::string generate();
    std::string rdm();

    std::queue<std::string> stream_datas;

    std::mutex mu;

    std::mutex qmu;

    std::condition_variable cond;

    std::atomic<bool> flag;

    // int length_limit;
};

#define TRACE(format, ...) \
    printf("%s::%s(%d)\n" format, __FILE__, __FUNCTION__, __LINE__, __VA_ARGS__)

#define NOW_TIME (std::chrono::high_resolution_clock::now())

#define DUARTION_MILISEC(STRAT, END)                                       \
    std::chrono::duration_cast<std::chrono::milliseconds>((END) - (STRAT)) \
            .count()

#endif