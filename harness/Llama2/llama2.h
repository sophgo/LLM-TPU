#ifndef LLAMA2_H
#define LLAMA2_H

extern "C" {

class LLama2;

LLama2* Llama2_with_devid_and_model(
        int         devid,
        const char* bmodel_path,
        const char* tokenizer_path);

void Llama2_delete(LLama2* chat);

void Llama2_deinit(LLama2* chat);

const char* get_history(LLama2* chat);

const char* set_history(LLama2* chat, const char* history);

const char* Llama2_predict_first_token(LLama2* chat, const char* input_str);

const char* Llama2_predict_next_token(LLama2* chat);

const int get_eos(LLama2* chat);

void Llama2_chat_with_llama2_tpu(LLama2* chat);

const char* Llama2_complete(LLama2* model, const char* input_str);
}

#endif