# coding=utf-8

import ctypes
import os

def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

class TokenWord(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_int),
        ("word", ctypes.c_char * 2048)  # 假设最大长度为 100，你可以根据实际情况调整
    ]

class TPUChatglm:
    def __init__(self, 
                device_id = 0,
                bmodel_path = "../compile/chatglm3-6b.bmodel",
                token_path = "../src/tokenizer.model",
                lib_path = "./build/libtpuchat.so"):

        check_file_exists(bmodel_path)
        check_file_exists(token_path)
        check_file_exists(lib_path) 

        self.lib = ctypes.cdll.LoadLibrary(lib_path)
        self.device_id = device_id
        self.bmodel_path = bmodel_path
        self.token_path = token_path
        self.libset()
        self.init()

    def libset(self):
        self.lib.ChatGLM_with_devid_and_model.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.ChatGLM_with_devid_and_model.restype = ctypes.c_void_p

        self.lib.ChatGLM_delete.argtypes = [ctypes.c_void_p]

        # deinit
        self.lib.ChatGLM_deinit.argtypes = [ctypes.c_void_p]

        # ChatGLM_predict_first_token
        self.lib.ChatGLM_predict_first_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ChatGLM_predict_first_token.restype = ctypes.c_char_p

        # ChatGLM_predict_next_token
        self.lib.ChatGLM_predict_next_token.argtypes = [ctypes.c_void_p]
        self.lib.ChatGLM_predict_next_token.restype = ctypes.c_char_p

        # get_eos
        self.lib.get_eos.argtypes = [ctypes.c_void_p]
        self.lib.get_eos.restype = ctypes.c_int
        # get_history
        self.lib.get_history.argtypes = [ctypes.c_void_p]
        self.lib.get_history.restype = ctypes.c_char_p
        # set history
        self.lib.set_history.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    def init(self):
        self.obj = self.lib.ChatGLM_with_devid_and_model(self.device_id, self.bmodel_path.encode('utf-8'),
                                                          self.token_path.encode('utf-8'))

    def predict_first_token(self, context):
        return self.lib.ChatGLM_predict_first_token(self.obj, context.encode('utf-8')).decode('utf-8')

    def predict_next_token(self):
        return self.lib.ChatGLM_predict_next_token(self.obj).decode('utf-8')

    def predict(self, context):

        first_token = self.predict_first_token(context)
        # print(first_token, end='')
        res = ''
        while True:
            next_token = self.predict_next_token()
            if next_token == '_GETMAX_' or next_token == '_GETEOS_':
                # print(next_token)
                break
            # print(next_token, end='')
            res += next_token
        return res

    def stream_predict(self, query, history):
        history.append((query, ''))

        res = ''
        first_token = self.predict_first_token(query)
        res += first_token

        while True:
            next_token = self.predict_next_token()
            if next_token == '_GETMAX_' or next_token == '_GETEOS_':
                break
            res += next_token
            history[-1] = (query, res)
            yield res, history

    def get_config(self):
        pass
