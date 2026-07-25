# coding=utf-8

import ctypes


class TokenWord(ctypes.Structure):
    _fields_ = [
        ("token", ctypes.c_int),
        ("word", ctypes.c_char * 2048)  # 假设最大长度为 100，你可以根据实际情况调整
    ]


class TPUChatglm:
    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary('./build/libtpuchat.so')
        device_id = 3
        bmodel_path = "../model/baichuan2-7b-test_int8.bmodel"
        token_path = "../model/tokenizer.model"
        self.device_id = device_id
        self.bmodel_path = bmodel_path
        self.token_path = token_path
        self.libset()
        self.init()

    def libset(self):
        self.lib.Baichuan2_with_devid_and_model.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p]
        self.lib.Baichuan2_with_devid_and_model.restype = ctypes.c_void_p

        self.lib.Baichuan2_delete.argtypes = [ctypes.c_void_p]

        # deinit
        self.lib.Baichuan2_deinit.argtypes = [ctypes.c_void_p]

        # Baichuan2_predict_first_token
        self.lib.Baichuan2_predict_first_token.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.Baichuan2_predict_first_token.restype = ctypes.c_char_p

        # Baichuan2_predict_next_token
        self.lib.Baichuan2_predict_next_token.argtypes = [ctypes.c_void_p]
        self.lib.Baichuan2_predict_next_token.restype = ctypes.c_char_p

        # get_eos
        self.lib.get_eos.argtypes = [ctypes.c_void_p]
        self.lib.get_eos.restype = ctypes.c_int
        # get_history
        self.lib.get_history.argtypes = [ctypes.c_void_p]
        self.lib.get_history.restype = ctypes.c_char_p
        # set history
        self.lib.set_history.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    def init(self):
        self.obj = self.lib.Baichuan2_with_devid_and_model(self.device_id, self.bmodel_path.encode('utf-8'),
                                                          self.token_path.encode('utf-8'))

    def predict_first_token(self, context):
        return self.lib.Baichuan2_predict_first_token(self.obj, context.encode('utf-8')).decode('utf-8')

    def predict_next_token(self):
        return self.lib.Baichuan2_predict_next_token(self.obj).decode('utf-8')

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

        prompt = ''
        # for i, (old_query, response) in enumerate(history):
        #     prompt += "[Round {}]\n\n问：{}\n\n答：{}\n\n".format(i + 1, old_query, response)
        # prompt += "[Round {}]\n\n问：{}\n\n答：".format(len(history) + 1, query)
        prompt = "<reserved_106>" + query + "<reserved_107>"
        
        res = ''
        first_token = self.predict_first_token(prompt)
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