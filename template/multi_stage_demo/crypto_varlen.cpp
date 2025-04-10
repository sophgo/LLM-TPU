#include <cstdlib>
#include <cstddef>
#include <cstdint>

extern "C" uint8_t* encrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
extern "C" uint8_t* decrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);

uint8_t* encrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes) {
    uint64_t varlen = 1000;
    uint8_t* processed_data = (uint8_t*)malloc(input_bytes * sizeof(uint8_t) + varlen);
    if (processed_data == NULL) {
        *output_bytes = 0;
        return NULL;
    }

    for (uint64_t i = 0; i < input_bytes; ++i) {
        processed_data[i] = ~input[i];
    }

    // Append 10 zero bytes at the end
    for (uint64_t i = input_bytes; i < input_bytes + varlen; ++i) {
        processed_data[i] = 0x12;
    }

    *output_bytes = input_bytes + varlen;
    return processed_data;
}

uint8_t* decrypt(const uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes) {
    uint64_t varlen = 1000;
    uint8_t* processed_data = (uint8_t*)malloc(input_bytes * sizeof(uint8_t) - varlen);
    if (processed_data == NULL) {
        *output_bytes = 0;
        return NULL;
    }

    for (uint64_t i = 0; i < input_bytes - varlen; ++i) {
        processed_data[i] = ~input[i];
    }

    *output_bytes = input_bytes - varlen;
    return processed_data;
}
