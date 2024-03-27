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

using ushort = unsigned short;
using uint = unsigned int;
using half = unsigned short;

inline uint as_uint(const float x) {
    return *(uint*)&x;
}
inline float as_float(const uint x) {
    return *(float*)&x;
}

inline float half_to_float(
        const ushort x) { // IEEE-754 16-bit floating-point format (without
                          // infinity): 1-5-10, exp-15, +-131008.0,
                          // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint e = (x & 0x7C00) >> 10; // exponent
    const uint m = (x & 0x03FF) << 13; // mantissa
    const uint v =
            as_uint((float)m) >> 23; // evil log2 bit hack to count leading
                                     // zeros in denormalized format
    return as_float(
            (x & 0x8000) << 16 | (e != 0) * ((e + 112) << 23 | m) |
            ((e == 0) & (m != 0)) *
                    ((v - 37) << 23 |
                     ((m << (150 - v)) &
                      0x007FE000))); // sign : normalized : denormalized
}
inline half float_to_half(
        const float x) { // IEEE-754 16-bit floating-point format (without
                         // infinity): 1-5-10, exp-15, +-131008.0,
                         // +-6.1035156E-5, +-5.9604645E-8, 3.311 digits
    const uint b = as_uint(x) + 0x00001000; // round-to-nearest-even: add last
                                            // bit after truncated mantissa
    const uint e = (b & 0x7F800000) >> 23;  // exponent
    const uint m = b & 0x007FFFFF; // mantissa; in line below: 0x007FF000 =
                                   // 0x00800000-0x00001000 = decimal indicator
                                   // flag - initial rounding
    return (b & 0x80000000) >> 16 |
            (e > 112) * ((((e - 112) << 10) & 0x7C00) | m >> 13) |
            ((e < 113) & (e > 101)) *
            ((((0x007FF000 + m) >> (125 - e)) + 1) >> 1) |
            (e > 143) * 0x7FFF; // sign : normalized : denormalized : saturate
}

#endif