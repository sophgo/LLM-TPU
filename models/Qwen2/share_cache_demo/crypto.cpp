

//#include <vector>
//#include <iostream>
//#include <fstream>
//#include <cryptopp/osrng.h>
//#include <cryptopp/aes.h>
//#include <cryptopp/modes.h>
//#include <cryptopp/filters.h>
#include <cstdlib>
#include <cstddef>
#include <cstdint>

// class IAESOFBCipher {
// public:
//     virtual ~IAESOFBCipher() {}
//     virtual void Encrypt(const std::vector<uint8_t>& data, std::vector<uint8_t>& processed_data) = 0;
//     virtual void Decrypt(const std::vector<uint8_t>& data, std::vector<uint8_t>& processed_data) = 0;
// };

// class AESOFBCipher : public IAESOFBCipher {
// private:
//     CryptoPP::SecByteBlock key;
//     CryptoPP::SecByteBlock iv;

// public:
//     AESOFBCipher() 
//         : key(CryptoPP::AES::DEFAULT_KEYLENGTH), iv(CryptoPP::AES::BLOCKSIZE) {
//         CryptoPP::AutoSeededRandomPool prng;
//         prng.GenerateBlock(key, key.size());
//         prng.GenerateBlock(iv, iv.size());
//     }

//     // // Encrypt
//     // void encrypt(const std::vector<uint8_t>& data, std::vector<uint8_t>& processed_data) {
//     //     processed_data.clear();
//     //     CryptoPP::OFB_Mode<CryptoPP::AES>::Encryption encryption;
//     //     encryption.SetKeyWithIV(key, key.size(), iv);

//     //     std::string cipher;
//     //     CryptoPP::ArraySource(data.data(), data.size(), true,
//     //         new CryptoPP::StreamTransformationFilter(encryption,
//     //             new CryptoPP::StringSink(cipher)
//     //         )
//     //     );
//     //     processed_data.assign(cipher.begin(), cipher.end());
//     // }

//     // // Decrypt
//     // void decrypt(const std::vector<uint8_t>& data, std::vector<uint8_t>& processed_data) {
//     //     processed_data.clear();
//     //     CryptoPP::OFB_Mode<CryptoPP::AES>::Decryption decryption;
//     //     decryption.SetKeyWithIV(key, key.size(), iv);

//     //     std::string plain;
//     //     CryptoPP::ArraySource(data.data(), data.size(), true,
//     //         new CryptoPP::StreamTransformationFilter(decryption,
//     //             new CryptoPP::StringSink(plain)
//     //         )
//     //     );
//     //     processed_data.assign(plain.begin(), plain.end());
//     // }

//     void Encrypt(const std::vector<uint8_t>& data, std::vector<uint8_t>& processed_data) {
//         processed_data.clear();
//         for(auto byte : data) {
//             processed_data.push_back(~byte);
//         }
//     }

//     void Decrypt(const std::vector<uint8_t>& data, std::vector<uint8_t>& processed_data) {
//         processed_data.clear();
//         for(auto byte : data) {
//             processed_data.push_back(~byte);
//         }
//     }
// };

// extern "C" IAESOFBCipher* createCipherInstance() {
//     return new AESOFBCipher();
// }

extern "C" uint8_t* encrypt(uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);
extern "C" uint8_t* decrypt(uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes);

uint8_t* encrypt(uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes) {
    uint8_t* processed_data = (uint8_t*)malloc(input_bytes * sizeof(uint8_t));
    if (processed_data == NULL) {
        *output_bytes = 0;
        return NULL;
    }

    for (uint64_t i = 0; i < input_bytes; ++i) {
        processed_data[i] = ~input[i];
    }

    *output_bytes = input_bytes;
    return processed_data;
}

uint8_t* decrypt(uint8_t* input, uint64_t input_bytes, uint64_t* output_bytes) {
    uint8_t* processed_data = (uint8_t*)malloc(input_bytes * sizeof(uint8_t));
    if (processed_data == NULL) {
        *output_bytes = 0;
        return NULL;
    }

    for (uint64_t i = 0; i < input_bytes; ++i) {
        processed_data[i] = ~input[i];
    }

    *output_bytes = input_bytes;
    return processed_data;
}
