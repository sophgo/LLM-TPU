import os
import json
import time
import argparse
from transformers import AutoTokenizer, AutoProcessor
import torch
import chat

class Model:
    def __init__(self, args):
        # test
        self.test_input = args.test_input
        self.test_media = args.test_media

        # preprocess parameters, such as prompt & tokenizer
        self.devices = [int(d) for d in args.devid.split(",")]
        config_path = os.path.join(args.dir_path, "config.json")
        self.tokenizer_path = os.path.join(args.dir_path, "tokenizer")
        self.embedding_path = os.path.join(args.dir_path, "embedding.bin")

        # config
        with open(config_path, 'r') as file:
            self.config = json.load(file)

        # Initialize model
        self.model_type = args.model_type if args.model_type else self.config['model_type']
        self.model = chat.Model()
        self.init_params(args)

        # Initialize model-specific mapper dynamically
        self.map(args)

        # warm up
        self.tokenizer.decode([0])
        self.init_history()

        # load model
        self.load_model(args, read_bmodel=True)

    def map(self, args):
        """Abstract model-specific mapper into a dictionary."""
        if self.model_type == "qwen2":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.EOS = [self.tokenizer.eos_token_id]
            self.append_user = lambda history, input_str: history.append(
                {"role": "user", "content": input_str}
            )
            self.append_assistant = lambda history, answer_str: history.append(
                {"role": "assistant", "content": answer_str}
            )
            self.apply_chat_template = lambda history: self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            self.system_prompt = {"role": "system", "content": "You are a helpful assistant."}
        elif self.model_type == "qwen":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.EOS = [self.tokenizer.im_end_id]
            self.append_user = lambda history, input_str: history.append(
                "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(answer_str)
            self.apply_chat_template = lambda history: "".join(history)
            self.system_prompt = "<|im_start|>system\nYou are a helpful assistant."
        elif self.model_type == "llama":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True, use_fast=False)
            self.system_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. " \
                                 "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. " \
                                 "Please ensure that your responses are socially unbiased and positive in nature. " \
                                 "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. " \
                                 "If you don't know the answer to a question, please don't share false information.\n<</SYS>>\n\n"
            self.EOS = [self.tokenizer.eos_token_id]
            self.append_user = lambda history, input_str: history.append(
                "{} [/INST] ".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(
                "{} </s><s>[INST] ".format(answer_str)
            )
            self.apply_chat_template = lambda history: "".join(history)
            self.tokenizer.add_prefix_space = False
        elif self.model_type == "llama3":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.system_prompt = {"role": "system", "content": "You are a helpful assistant."}
            self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            self.append_user = lambda history, input_str: history.append(
                {"role": "user", "content": input_str}
            )
            self.append_assistant = lambda history, answer_str: history.append(
                {"role": "assistant", "content": answer_str}
            )
            self.apply_chat_template = lambda history: self.tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
        elif self.model_type == "lwm":
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
            self.system_prompt = "You are a helpful assistant."
            self.EOS = [self.tokenizer.eos_token_id]
            self.append_user = lambda history, input_str: history.append(
                "USER: {} ASSISTANT: ".format(input_str)
            )
            self.append_assistant = lambda history, answer_str: history.append(
                "{} ".format(answer_str)
            )
            self.apply_chat_template = lambda history: "".join(history)
            self.tokenizer.add_prefix_space = False

        # VISION
        else:
            self.enable_vision = True
            processor_path = os.path.join(args.dir_path, "processor")
            if self.model_type == "qwen2_vl" or self.model_type == "qwen2_5_vl":
                # for qwen2_5_vl
                # pip install qwen-vl-utils[decord]==0.0.8
                # pip install git+https://github.com/huggingface/transformers accelerate
                self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer
                self.EOS = [self.tokenizer.convert_tokens_to_ids("<|end|>"), self.tokenizer.convert_tokens_to_ids("<|im_end|>")]

                if args.resized_width % self.config["vision_config"]["patch_size"] != 0 and \
                    args.resized_height % self.config["vision_config"]["patch_size"] != 0:
                    raise ValueError(
                        f"The resized width ({args.resized_width}) and height ({args.resized_height}) must be multiples of the patch size ({self.config['vision_config']['patch_size']})."
                    )

                self.model.config.resized_width = args.resized_width
                self.model.config.resized_height = args.resized_height
                self.model.config.image_token_id = self.config["image_token_id"]
                self.model.config.video_token_id = self.config["video_token_id"]
                self.model.config.spatial_merge_size = self.config["vision_config"]["spatial_merge_size"]
                self.model.config.patch_size = self.config["vision_config"]["patch_size"]
                self.model.config.temporal_patch_size = self.config["vision_config"]["temporal_patch_size"]
                self.append_user = lambda history, input_str: history.append(
                    {"role": "user", "content": input_str}
                )
                self.append_assistant = lambda history, answer_str: history.append(
                    {"role": "assistant", "content": answer_str}
                )
                self.apply_chat_template = lambda history: self.processor.apply_chat_template(
                    history, tokenize=False, add_generation_prompt=True
                )
                self.system_prompt = {"role": "system", "content": "You are a helpful assistant."}
            else:
                raise NotImplementedError(f"{self.model_type} not support now")
        return

    def load_model(self, args, read_bmodel):
        if not args.model_path:
            bmodel_files = [f for f in os.listdir(args.dir_path) if f.endswith('.bmodel')]
            if len(bmodel_files) > 1:
                raise RuntimeError(f"Found multiple bmodel files in {args.dir_path}, please specify one with --model_path")
            elif not bmodel_files:
                raise FileNotFoundError(f"No bmodel files found in {args.dir_path}")
            model_path = os.path.join(args.dir_path, bmodel_files[0])
        else:
            model_path = args.model_path

        load_start = time.time()
        self.model.init(self.devices, model_path, read_bmodel) # when read_bmodel = false, not to load weight, reuse weight
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.embedding_path = self.embedding_path if os.path.exists(self.embedding_path) else ""
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]
        self.model.config.model_type = self.model_type

        self.enable_history = args.enable_history

    def init_history(self):
        self.history = [self.system_prompt]

    def update_history(self):
        if self.model.total_length >= self.model.SEQLEN:
            print("... (reach the maximal length)", flush=True, end="")
            self.init_history()
        else:
            self.append_assistant(self.history, self.answer_cur)

    def clean_invalid_unicode(self, text):
        return text.encode('utf-8', 'ignore').decode('utf-8', 'ignore')

    def image_message(self, text, path):
        messages = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": path
                },
                {"type": "text", "text": text},
            ],
        }
        return messages

    def video_message(self, text, path):
        messages = {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": path,
                    "fps": 1.0,
                },
                {"type": "text", "text": text},
            ],
        }
        return messages

    def encode_tokens(self, text, media_path, media_type):
        if media_type == "image":
            messages = self.image_message(text, media_path)
            self.history.append(messages)
        elif media_type == "video":
            messages = self.video_message(text, media_path)
            self.history.append(messages)
        else:
            self.append_user(self.history, text)
        formatted_text = self.apply_chat_template(self.history)

        tokens = self.tokenizer(formatted_text).input_ids
        return tokens
    
    def process_media_input(self, media_path, media_type):
        if self.model_type in ["qwen2_vl", "qwen2_5_vl"]:
            inputs = self.model.process_media(media_path, media_type) # preprocess and vit launch
        else:
            raise NotImplementedError(f"Not support {self.model_type} now")
        return inputs
    
    def prefill_phase(self, text, media_path, media_type):
        """handle image or video"""
        print("\n回答: ", end="")
        first_start = time.time()

        tokens = self.encode_tokens(text, media_path, media_type)
        self.model.init_forward(tokens)

        if media_path:
            self.process_media_input(media_path, media_type)

        token = self.model.forward_first()

        first_end = time.time()
        self.first_duration = first_end - first_start
        return token

    def decode_phase(self, token):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        next_start = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS and self.model.total_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
            self.answer_token += full_word_tokens
            print(word, flush=True, end="")
            tok_num += 1
            full_word_tokens = []
            token = self.model.forward_next()

        # counting time
        next_end = time.time()
        next_duration = next_end - next_start
        tps = tok_num / next_duration

        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.init_history()

        print()
        print(f"FTL: {self.first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. 如果您想退出，请输入 [q, quit, exit] 之一
2. 要创建一个新的聊天会话，请输入 [clear, new] 之一
3. 如果是多模态模型，请输入图像或视频的路径
================================================================="""
        )
        # Stop Chatting with "exit" input
        while True:
            if self.test_input:
                input_str = self.test_input.strip()
                print(f"\n问题: {input_str}")
            else:
                input_str = self.clean_invalid_unicode(input("\n请输入问题: ").strip())

            # Quit
            if input_str in ["exit", "q", "quit"]:
                print("退出聊天会话。")
                break
            # New Chat
            elif input_str in ["clear", "new"]:
                self.init_history()
                print("聊天记录已清除，开始新的会话。")

            # VISION
            media_type = "text"
            media_path = ""
            if hasattr(self, "enable_vision") and self.enable_vision:
                if self.test_media:
                    media_path = self.test_media.strip()
                    print(f"路径: {media_path}") if media_path else None
                else:
                    media_path = input("\n路径: ").strip()
                
                if media_path:
                    if not os.path.exists(media_path):
                        print(f"路径不存在: {media_path}")
                    else:
                        ext = os.path.splitext(media_path)[1].lower()
                        if ext in [".jpg", ".jpeg", ".png"]:
                            media_type = "image"
                        elif ext == ".mp4":
                            media_type = "video"
                        else:
                            print(f"不支持的格式: {ext}")
                            media_path = ""

            token = self.prefill_phase(input_str, media_path, media_type)
            self.decode_phase(token)

            if self.test_input or self.test_media:
                break


def main(args):
    model = Model(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dir_path", type=str, default="./tmp",
                        help="dir path to the config/embedding/tokenizer")
    parser.add_argument('-b', '--model_path', type=str, default="",
                        help='path to the bmodel file')
    parser.add_argument('-d', '--devid', type=str, default='0',
                        help='device ID to use')
    parser.add_argument('--test_input', type=str,
                        help='the text for test')
    parser.add_argument('--test_media', type=str,
                        help='the media(image/video) path for test')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2,
                        help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32,
                        help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, default="greedy",
                        choices=["greedy", "penalty_sample"],
                        help='mode for generating next token')
    parser.add_argument('--resized_height', type=int, default=0,
                        help='use resized_height for vlm when resized_height != 0')
    parser.add_argument('--resized_width', type=int, default=0,
                        help='use resized_width for vlm when resized_width != 0')
    parser.add_argument('--enable_history', action='store_true',
                        help="if set, enables storing of history memory")
    parser.add_argument('--model_type', type=str, help="model type")
    args = parser.parse_args()
    main(args)
