from typing import Any
from dotenv import load_dotenv
import ctypes
import os

load_dotenv()


class TpuLLama2:
    class LLama2(ctypes.Structure):
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)

    def __init__(self) -> None:
        self.bmodel_path = os.getenv('LLAMA2_BMODEL_PATH')
        self.tokenizer_path = os.getenv('LLAMA2_TOKENIZER_PATH')
        self.lib_path = os.getenv('LLAMA2_LIB_PATH')

        self.lib = ctypes.CDLL(self.lib_path)
        self.lib.Llama2_with_devid_and_model.argtypes = [
            ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.Llama2_with_devid_and_model.restype = ctypes.POINTER(
            self.LLama2)

        self.lib.Llama2_complete.argtypes = [
            ctypes.POINTER(self.LLama2), ctypes.c_char_p]
        self.lib.Llama2_complete.restype = ctypes.c_char_p

    def Llama2_complete(self, model, input_str):
        return self.lib.Llama2_complete(model, input_str.encode())

    def Llama2_with_devid_and_model(self, devid):
        return self.lib.Llama2_with_devid_and_model(devid, self.bmodel_path.encode(), self.tokenizer_path.encode())


if __name__ == "__main__":
    model = TpuLLama2()
    data = 'Hello! who are you?'
    ctx = model.Llama2_with_devid_and_model(0)
    print(model.Llama2_complete(ctx, data))
