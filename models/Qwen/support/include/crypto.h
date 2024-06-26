

#include <vector>
#include <iostream>
#include <fstream>
#include <cryptopp/osrng.h>
#include <cryptopp/aes.h>
#include <cryptopp/modes.h>
#include <cryptopp/filters.h>

class AESOFBCipher {
private:
    CryptoPP::SecByteBlock key;
    CryptoPP::SecByteBlock iv;

public:
    AESOFBCipher() 
        : key(CryptoPP::AES::DEFAULT_KEYLENGTH), iv(CryptoPP::AES::BLOCKSIZE) {
        CryptoPP::AutoSeededRandomPool prng;
        prng.GenerateBlock(key, key.size());
        prng.GenerateBlock(iv, iv.size());
    }

    // Encrypt
    void encrypt(const std::vector<uint8_t>& plaintext, std::vector<uint8_t>& ciphertext) {
        ciphertext.clear();
        CryptoPP::OFB_Mode<CryptoPP::AES>::Encryption encryption;
        encryption.SetKeyWithIV(key, key.size(), iv);

        std::string cipher;
        CryptoPP::ArraySource(plaintext.data(), plaintext.size(), true,
            new CryptoPP::StreamTransformationFilter(encryption,
                new CryptoPP::StringSink(cipher)
            )
        );
        ciphertext.assign(cipher.begin(), cipher.end());
    }

    // Decrypt
    void decrypt(const std::vector<uint8_t>& ciphertext, std::vector<uint8_t>& plaintext) {
        plaintext.clear();
        CryptoPP::OFB_Mode<CryptoPP::AES>::Decryption decryption;
        decryption.SetKeyWithIV(key, key.size(), iv);

        std::string plain;
        CryptoPP::ArraySource(ciphertext.data(), ciphertext.size(), true,
            new CryptoPP::StreamTransformationFilter(decryption,
                new CryptoPP::StringSink(plain)
            )
        );
        plaintext.assign(plain.begin(), plain.end());
    }
};