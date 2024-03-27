from typing import Any
from dotenv import load_dotenv
import ctypes
import os
import numpy as np
import sentencepiece as spm
from scipy.special import log_softmax
load_dotenv()

output_length = 32000


class TpuLLama2:
    class LLama2(ctypes.Structure):
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)

    def __init__(self) -> None:
        self.bmodel_path = os.getenv('LLAMA2_BMODEL_PATH')
        self.logits_path = os.getenv('LLAMA2_BMODEL_LOGITS_PATH')
        self.tokenizer_path = os.getenv('LLAMA2_TOKENIZER_PATH')
        self.lib_path = os.getenv('LLAMA2_LIB_PATH')
        self.logits_lib_path = os.getenv('LLAMA2_LOGITS_LIB_PATH')
        self.tokenizer = spm.SentencePieceProcessor(
            model_file=self.tokenizer_path)
        self.logits = []
        self.tokens = []

        self.lib = ctypes.CDLL(self.lib_path)
        self.lib.Llama2_with_devid_and_model.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.Llama2_with_devid_and_model.restype = ctypes.POINTER(
            self.LLama2)
        self.lib.Llama2_complete.argtypes = [
            ctypes.POINTER(self.LLama2), ctypes.c_char_p]
        self.lib.Llama2_complete.restype = ctypes.c_char_p

        self.logits_lib = ctypes.CDLL(self.logits_lib_path)
        self.logits_lib.Llama2_with_devid_and_model.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        self.logits_lib.Llama2_with_devid_and_model.restype = ctypes.POINTER(
            self.LLama2)

        self.logits_lib.Llama2_predict_first_token_logits.restype = ctypes.POINTER(
            ctypes.c_uint16)
        self.logits_lib.Llama2_predict_first_token_logits.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p]

        self.logits_lib.Llama2_predict_next_token_logits.restype = ctypes.POINTER(
            ctypes.c_uint16)
        self.logits_lib.Llama2_predict_next_token_logits.argtypes = [
            ctypes.c_void_p]

    def Llama2_complete(self, model, input_str):
        return self.lib.Llama2_complete(model, input_str.encode())

    def Llama2_with_devid_and_model(self, devid):
        return self.lib.Llama2_with_devid_and_model(devid, self.bmodel_path.encode(), self.tokenizer_path.encode())

    def Llama2_with_devid_and_model_logits(self, devid):
        return self.logits_lib.Llama2_with_devid_and_model(devid, self.logits_path.encode(), self.tokenizer_path.encode())

    def Llama2_predict_first_token_logits(self, model, input_str):
        return self.logits_lib.Llama2_predict_first_token_logits(model, input_str.encode())

    def Llama2_predict_next_token_logits(self, model):
        return self.logits_lib.Llama2_predict_next_token_logits(model)

    def Llama2_get_logits(self, data):
        return np.ctypeslib.as_array(
            (ctypes.c_ushort * output_length).from_address(ctypes.addressof(data.contents))).view(np.float16)

    def Llama2_decode_logits(self, logits):
        idx = np.argmax(logits).tolist()
        # token = np.argmax(logits)
        self.tokens.append(idx)
        print(self.tokenizer.Decode(idx), end=" ")
        self.logits.append(np.copy(logits))
        return idx

    def Llama2_complete_logits(self, model, input_str):
        data = self.Llama2_get_logits(
            self.Llama2_predict_first_token_logits(model, input_str))
        cnt = 1
        while True:
            if self.Llama2_decode_logits(data) == 2 or cnt > 500:
                break
            data = self.Llama2_get_logits(
                self.Llama2_predict_next_token_logits(model))
            cnt += 1
        return cnt


prompt =   """
Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.

Q: What happens to you if you eat watermelon seeds?
A: Nothing happens.
        
        """
        

prompt_1 =   """
Q: What is human life expectancy in the United States?
A: 
"""

if __name__ == "__main__":
    model = TpuLLama2()
    ss = 'Q: Who are you?\nA:'
    ctx = model.Llama2_with_devid_and_model_logits(0)

    # print(np.array(model.tokenizer.Encode(str)))
    
    data = model.Llama2_complete_logits(ctx, prompt_1)
    print(data)
    
    print(f'Prompt: {prompt_1}')
    
    print(f'Answer: {model.tokenizer.DecodeIds(model.tokens)}')
    # logits = np.stack(model.logits, axis=0)

    # tokens = np.argmax(logits, axis=1)
    # print(tokens)

    # print(data[0].shape)
    # tokens = np.argmax(data, axis=1)
    # print(tokens)
    # print(model.tokenizer.DecodeIds(tokens))

    # data = model.Llama2_predict_first_token_logits(ctx, str)
    # logits = model.Llama2_get_logits(data)
    # print(log_softmax(logits))

    # # model.Llama2_decode_logits()
    # cnt = 0
    # while True:
    #     data = model.Llama2_predict_next_token_logits(ctx)
    #     model.Llama2_decode_logits(model.Llama2_get_logits(data))
