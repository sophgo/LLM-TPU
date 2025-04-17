import os
import json
import time
import argparse
from transformers import AutoTokenizer, AutoProcessor

import base64
from io import BytesIO
from PIL import Image

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
        self.load_model(args)

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

    def load_model(self, args):
        self.model.init_decrypt()

        load_start = time.time()
        self.model.init(self.devices, args.model_path) # when read_bmodel = false, not to load weight, reuse weight
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.lib_path = args.lib_path
        self.model.embedding_path = self.embedding_path if os.path.exists(self.embedding_path) else ""
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]
        self.model.config.model_type = self.model_type

        self.max_new_tokens = args.max_new_tokens
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

    def encode_tokens(self, text, media_path="", media_type="text"):
        if media_type == "image":
            self.messages = self.image_message(text, media_path)
            self.history.append(self.messages)
        elif media_type == "video":
            self.messages = self.video_message(text, media_path)
            self.history.append(self.messages)
        else:
            self.append_user(self.history, text)
        self.formatted_text = self.apply_chat_template(self.history)
        tokens = self.tokenizer(self.formatted_text).input_ids
        return tokens

    def process_image(self, image):
        image_obj = None
        if isinstance(image, Image.Image):
            image_obj = image
        elif image.startswith("data:image"):
            if "base64," in image:
                _, base64_data = image.split("base64,", 1)
                data = base64.b64decode(base64_data)
                image_obj = Image.open(BytesIO(data))
        else:
            image_obj = Image.open(image)
        if image_obj is None:
            raise ValueError(f"Unrecognized image input, support local path, http url, base64 and PIL.Image, got {image}")
        image = image_obj.convert("RGB")

        if self.model.config.resized_width and self.model.config.resized_height:
            resized_width = self.model.config.resized_width
            resized_height = self.model.config.resized_height
        else:
            height, width = image.height, image.width
            resized_height, resized_width = self.model.smart_resize(height, width)
            self.model.config.resized_width, self.model.config.resized_height = resized_width, resized_height

        image = image.resize((resized_width, resized_height))
        shape = list(image.size)
        self.model.process_image(image.tobytes(), shape)
        return


    def process_media_input(self, media_path, media_type):
        if self.model_type in ["qwen2_vl", "qwen2_5_vl"]:
            self.process_image(media_path) # preprocess and vit launch
        else:
            raise NotImplementedError(f"Not support {self.model_type} now")
        return
    
    def prefill_phase(self, text, media_path, media_type):
        print("\n回答: ", end="")
        first_start = time.time()

        tokens = self.encode_tokens(text, media_path, media_type)
        self.model.init_forward(tokens)

        if media_path:
            self.process_media_input(media_path, media_type)

        token = self.model.forward_first()

        first_end = time.time()
        self.ftl = first_end - first_start
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
        self.tps = tok_num / next_duration

        if self.enable_history:
            self.answer_cur = self.tokenizer.decode(self.answer_token)
            self.update_history()
        else:
            self.init_history()

        print()
        print(f"FTL: {self.ftl:.3f} s")
        print(f"TPS: {self.tps:.3f} token/s")

    def generate(self, text):
        tokens = self.encode_tokens(text)
        self.model.init_forward(tokens)

        first_start = time.time()
        token = self.model.forward_first()
        first_end = time.time()

        tok_num = 0

        next_start = time.time()
        result_tokens = []
        while token not in self.EOS and self.model.total_length < self.model.SEQLEN and len(result_tokens) < self.max_new_tokens:
            tok_num += 1
            result_tokens.append(token)
            token = self.model.forward_next()
        next_end = time.time()
        next_duration = next_end - next_start

        self.ftl = first_end - first_start
        self.tps = tok_num / next_duration

        self.init_history()
        return result_tokens

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
        self.model.deinit_decrypt()

    def test_sample(self, args, input_s):
            """
            Start a chat session.
            """
            self.load_model(args)
            input_str = input_s
            # input_str = self.test_input.strip()
            media_type = "image"
            media_path = self.test_media.strip()

            token = self.prefill_phase(input_str, media_path, media_type)
            self.decode_phase(token)

            #if self.test_input or self.test_media:
            #    break
            self.model.deinit_decrypt()
            self.model.deinit()


def main(args):
    model = Model(args)
    # model.chat()
    input_text = ["描述一下图片中的内容",
                "在一个阳光灿烂的早晨，当城市的喧嚣尚未完全苏醒，街道上偶尔传来几声鸟鸣，伴随着微风轻拂树叶的沙沙声，年轻的艺术家走出她的小公寓，怀揣着对新一天的期待，决定在附近的公园里寻找灵感，尽管她知道，自己面临着即将到来的展览的压力，那是她职业生涯中最重要的一次展示，然而她相信，只有在大自然的怀抱中，才能真正激发她的创造力，捕捉到那些潜藏在心底的灵感，最终将它们化为色彩斑斓的画布作品，传递出她内心深处对生活的热爱与思考。在她走向公园的路上，阳光透过树梢洒下斑驳的光影，仿佛在为她的创作之旅铺设一条光明的道路，而她的脑海中闪现出各种各样的画面，有的是她在旅行中遇到的奇妙风景，有的是她与朋友们欢聚时的欢声笑语，还有那些在夜晚独自思考时的沉静瞬间，这些记忆交织在一起，形成了一个个生动的场景，激励着她不断前行，尽管她内心深处也有些许的不安，因为她知道，艺术创作的过程往往充满了不确定性，有时灵感如潮水般涌来，而有时却又如同干涸的河流，让人感到无比沮丧。然而，她并不打算让这些负面情绪影响到自己，反而在心中默默告诉自己，要学会接受这种不完美，因为正是这种起伏的过程才让艺术变得更加真实和动人，她的目标不仅仅是完成一幅画作，而是通过每一笔每一划，表达出她对生活的独特理解和感悟，这种追求让她在绘画的过程中感到无比充实和快乐，仿佛每一次的创作都是一次心灵的洗礼，让她更加贴近自己的内心世界。当她终于走到公园的中央，那里绿树成荫，花香四溢，孩子们在草地上嬉戏玩耍，老人们则悠闲地坐在长椅上，享受着温暖的阳光，整个场景显得那么和谐而宁静，这一刻，她感受到了一种难以言喻的幸福，仿佛这一切都是为了她而存在。这段话概括一下，150字左右",
                "在不久的将来，城市的面貌将发生翻天覆地的变化。随着科技的飞速发展，我们将看到智能交通系统的普及，自动驾驶汽车将在城市的街道上自由穿行。这不仅能够减少交通拥堵，还能显著降低交通事故的发生率。想象一下，当你坐在自动驾驶的车辆中，轻松地浏览电子邮件或享受一杯咖啡，而车辆则安全地将你送到目的地。与此同时，城市的建筑将更加注重可持续性和生态友好。高楼大厦的外墙将覆盖着绿色植物，屋顶将安装太阳能电池板，为建筑提供清洁的能源。城市中的公园和绿地将被精心设计，成为人们休闲和社交的场所。这样的城市不仅美观，还能有效改善空气质量，为居民提供一个更健康的生活环境。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。而车辆则安全地将你送到目的地。与此同时，城市的建筑将更加注重可持续性和生态友好。高楼大厦的外墙将覆盖着绿色植物，屋顶将安装太阳能电池板，为建筑提供清洁的能源。城市中的公园和绿地将被精心设计，成为人们休闲和社交的场所。这样的城市不仅美观，还能有效改善空气质量，为居民提供一个更健康的生活环境。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。而车辆则安全地将你送到目的地。与此同时，城市的建筑将更加注重可持续性和生态友好。高楼大厦的外墙将覆盖着绿色植物，屋顶将安装太阳能电池板，为建筑提供清洁的能源。城市中的公园和绿地将被精心设计，成为人们休闲和社交的场所。这样的城市不仅美观，还能有效改善空气质量，为居民提供一个更健康的生活环境。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。而车辆则安全地将你送到目的地。与此同时，城市的建筑将更加注重可持续性和生态友好。高楼大厦的外墙将覆盖着绿色植物，屋顶将安装太阳能电池板，为建筑提供清洁的能源。城市中的公园和绿地将被精心设计，成为人们休闲和社交的场所。这样的城市不仅美观，还能有效改善空气质量，为居民提供一个更健康的生活环境。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。然而，未来城市的愿景并非没有挑战。随着技术的普及，隐私和安全问题将成为人们关注的焦点。如何在享受便利的同时保护个人信息，将是政府和科技公司必须共同面对的难题。此外，社会的数字鸿沟也可能加剧，如何确保每个人都能平等地享受科技带来的红利，将是实现未来城市愿景的关键。结合图片总结一下这段话的含义，300字左右。"]
    for i in range(0, 1000000):
        try:
            # model = Model(args)
            model.test_sample(args, input_text[i % 3])
        except Exception as e:
                print(f"{type(e).__name__} : {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="dir path to the bmodel/")
    parser.add_argument("-p", "--dir_path", type=str, default="./tmp",
                        help="dir path to the bmodel/config/tokenizer")
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
    parser.add_argument('--lib_path', type=str, default='', help='lib path by user')
    parser.add_argument('--model_type', type=str, help="model type")
    args = parser.parse_args()
    main(args)
