import chat

chip_number = 16
devices = [10]
model_path = "chatglm3-6b_int4_1dev.bmodel"
tokenizer_path = "../support/tokenizer.model"
engine = chat.ChatGLM()
engine.init(devices, model_path, tokenizer_path)
engine.answer("你好")
engine.deinit()
