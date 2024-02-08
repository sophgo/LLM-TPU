#ifndef __CHATGLM_C_H__
#define __CHATGLM_C_H__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BmGLM2 BmGLM2;

BmGLM2* bmglm2_create();

void bmglm2_distroy(BmGLM2* instance);

void bmglm2_init(BmGLM2* instance, int devid, const char* model_dir);

void bmglm2_deinit(BmGLM2* instance);

void bmglm2_init_stream(BmGLM2* instance, const char* input, int length_limit);

void bmglm2_stop_inference(BmGLM2* instance);

const char* bmglm2_get_word(BmGLM2* instance);

void bmglm2_init_tokens(
        BmGLM2* instance,
        int*    input,
        int     length,
        int     length_limit);

void bmglm2_complete_tokens(
        BmGLM2* instance,
        int*    input_tokens,
        int     input_tokens_length,
        int*    eos_ids,
        int     eos_ids_num,
        int     max_token_length,
        int**   result_tokens,
        int*    result_length);

#ifdef __cplusplus
}
#endif

#endif