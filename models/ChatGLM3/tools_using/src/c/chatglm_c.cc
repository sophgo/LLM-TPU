#include "chatglm_c.h"
#include <chat.h>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct BmGLM2 {
    ChatGLM2Inner* glm;
};

std::string res;
std::string tmp;

extern "C" BmGLM2* bmglm2_create() {
    BmGLM2* bmglm = new BmGLM2;
    bmglm->glm = new ChatGLM2Inner();
    return bmglm;
}

extern "C" void bmglm2_distroy(BmGLM2* instance) {
    delete instance->glm;
    delete instance;
}

extern "C" void bmglm2_init(
        BmGLM2*     instance,
        int         devid,
        const char* model_dir) {
    instance->glm->init(devid, model_dir);
}

extern "C" void bmglm2_deinit(BmGLM2* instance) {
    instance->glm->deinit();
}

extern "C" void bmglm2_init_stream(
        BmGLM2*     instance,
        const char* input,
        int         length_limit) {
    tmp = std::string(input);
    instance->glm->run_stream(tmp.c_str(), length_limit);
}

extern "C" const char* bmglm2_get_word(BmGLM2* instance) {
    res = instance->glm->generate();
    return res.c_str();
}

extern "C" void bmglm2_stop_inference(BmGLM2* instance) {
    instance->glm->stop_inference();
}

extern "C" void bmglm2_init_tokens(
        BmGLM2* instance,
        int*    input,
        int     length,
        int     length_limit) {
    auto vec = std::vector<int>(input, input + length);
    instance->glm->run_tokens_stream(vec, length_limit);
}

extern "C" void bmglm2_complete_tokens(
        BmGLM2* instance,
        int*    input_tokens,
        int     input_tokens_length,
        int*    eos_ids,
        int     eos_ids_num,
        int     max_token_length,
        int**   result_tokens,
        int*    result_length) {
    auto tokens =
            std::vector<int>(input_tokens, input_tokens + input_tokens_length);
    auto eos = std::vector<int>(eos_ids, eos_ids + eos_ids_num);
    auto res = instance->glm->complete_tokens(tokens, eos, max_token_length);

    *result_length = res.size();
    *result_tokens = (int*)malloc(res.size() * sizeof(int));
    memcpy(*result_tokens, res.data(), res.size() * sizeof(int));
}