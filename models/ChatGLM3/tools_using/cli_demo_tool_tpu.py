import os
import platform
from tokenization_chatglm import ChatGLMTokenizer
from bmglm2 import BmGLM2
import sys

tokenizer = ChatGLMTokenizer.from_pretrained(
    "THUDM/chatglm3-6b", trust_remote_code=True)


os_name = platform.system()
clear_command = 'cls' if os_name == 'Windows' else 'clear'
stop_stream = False


def build_prompt(history):
    prompt = "欢迎使用 ChatGLM3-6B-TPU 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B-TPU：{response}"
    return prompt


tools = [{'name': 'track', 'description': '追踪指定股票的实时价格', 'parameters': {
    'type': 'object', 'properties': {'symbol': {'description': '需要追踪的股票代码'}}, 'required': []}}]
system_item = {"role": "system",
               "content": "Answer the following questions as best as you can. You have access to the following tools:",
               "tools": tools}


def main(model_dir):
    model = BmGLM2(model_dir)
    past_key_values, history = None, [system_item]
    role = "user"
    global stop_stream
    print("欢迎使用 ChatGLM3-6B-TPU 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
    while True:
        query = input("\n用户：") if role == "user" else input("\n结果：")
        if query.strip() == "stop":
            break
        if query.strip() == "clear":
            past_key_values, history = None,  [system_item]
            role = "user"
            os.system(clear_command)
            print("欢迎使用 ChatGLM3-6B-TPU 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序")
            continue
        print("\nChatGLM：", end="")
        response, history = model.chat(
            tokenizer, query, history=history, role=role)
        print(response, end="", flush=True)
        print("")
        if isinstance(response, dict):
            role = "observation"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f'the model directory is required')
        sys.exit(1)
    model_dir = sys.argv[1]
    main(model_dir)
