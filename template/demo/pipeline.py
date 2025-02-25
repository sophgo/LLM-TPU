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
            if self.model_type == "qwen2_vl":
                self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
                self.tokenizer = self.processor.tokenizer
                self.EOS = [self.tokenizer.convert_tokens_to_ids("<|end|>"), self.tokenizer.convert_tokens_to_ids("<|im_end|>")]
                self.model.spatial_merge_size = self.config["vision_config"]["spatial_merge_size"]
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
            else:
                raise NotImplementedError(f"{self.model_type} not support now")
        return

    def load_model(self, args, read_bmodel):
        if not args.model_path:
            bmodel_files = [f for f in os.listdir(args.dir_path) if f.endswith('.bmodel')]
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
        self.model.model_type = self.model_type

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

    def encode_tokens(self):
        self.append_user(self.history, self.input_str)
        text = self.apply_chat_template(self.history)
        tokens = self.tokenizer(text).input_ids
        return tokens

    def image_message(self, path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": path,
                    },
                    {"type": "text", "text": self.input_str},
                ],
            }
        ]
        return messages

    def video_message(self, path):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": path,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": self.input_str},
                ],
            }
        ]
        return messages
    
    def process_media_input(self, path, media_type):
        """处理图像或视频输入"""
        if not os.path.exists(path):
            print(f"无法找到 {media_type} 路径: {path}")
            return None

        if self.model_type == "qwen2_vl":
            from qwen_vl_utils import process_vision_info
            self.append_user(self.history, self.input_str)

            messages = self.image_message(path) if media_type == "image" else self.video_message(path)
            image_inputs, video_inputs = process_vision_info(messages)
            text = self.apply_chat_template(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            if self.model.MAX_PIXELS < inputs.pixel_values.shape[0]:
                raise RuntimeError("your image is too large ")
        elif media_type == "image":
            # inputs = self.model.process_image(path)
            pass
        elif media_type == "video":
            # inputs = self.model.process_video(path)
            pass
        else:
            print(f"未知的媒体类型: {media_type}")
            return None
        return inputs
    
    def prefill_phase(self, inputs, media_type):
        """handle image or video"""
        print("\n回答: ", end="")
        first_start = time.time()

        if media_type == "image":
            vit_token_list = torch.where(inputs.input_ids == self.config["image_token_id"])[1].tolist()
            vit_offset = vit_token_list[0]
            valid_vit_length = len(vit_token_list)
            token = self.model.forward_first(
                inputs.input_ids.squeeze(0).tolist(),
                inputs.pixel_values.flatten().tolist(),
                inputs.image_grid_thw.squeeze(0).tolist(),
                vit_offset,
                valid_vit_length
            )
        elif media_type == "video":
            vit_token_list = torch.where(inputs.input_ids == self.config["video_token_id"])[1].tolist()
            vit_offset = vit_token_list[0]
            valid_vit_length = len(vit_token_list)
            token = self.model.forward_first(
                inputs.input_ids.squeeze(0).tolist(),
                inputs.pixel_values_videos.flatten().tolist(),
                inputs.video_grid_thw.squeeze(0).tolist(),
                vit_offset,
                valid_vit_length
            )
        elif media_type == "text":
            tokens = self.encode_tokens()
            token = self.model.forward_first(tokens)
        else:
            raise NotImplementedError("Not Support Now")

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
                self.input_str = self.test_input.strip()
                print(f"\n问题: {self.input_str}")
            else:
                self.input_str = self.clean_invalid_unicode(input("\n请输入问题: ").strip())

            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                print("退出聊天会话。")
                break
            # New Chat
            elif self.input_str in ["clear", "new"]:
                self.init_history()
                print("聊天记录已清除，开始新的会话。")

            # VISION
            if hasattr(self, "enable_vision") and self.enable_vision:
                if self.test_media:
                    media_path = self.test_media.strip()
                    print(f"路径: {media_path}")
                else:
                    media_path = input("\n请输入媒体路径: ").strip()
                _, ext = os.path.splitext(media_path)

                if ext in [".jpg", ".jpeg", ".png"]:
                    media_type = "image"
                    inputs = self.process_media_input(media_path, media_type)
                elif ext in [".mp4"]:
                    media_type = "video"
                    inputs = self.process_media_input(media_path, media_type)
                else:
                    print("输入的路径无效，请重新输入。")
                    continue

            # TEXT
            else:
                media_type = "text"
                inputs = self.input_str


            token = self.prefill_phase(inputs, media_type)
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
    parser.add_argument('--enable_history', action='store_true',
                        help="if set, enables storing of history memory")
    parser.add_argument('--model_type', type=str, help="model type")
    args = parser.parse_args()
    main(args)
