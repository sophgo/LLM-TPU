import chat

devices = [1]
model_path = "../compile/chatglm3-6b_int8_1dev.bmodel"
tokenizer_path = "../support/tokenizer.model"
engine = chat.ChatGLM()
engine.init(devices, model_path, tokenizer_path)
engine.answer("你好")
engine.deinit()
