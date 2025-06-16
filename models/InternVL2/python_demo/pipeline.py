import time
import argparse
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from torchvision.transforms.functional import InterpolationMode
import chat
import os

# Preprocess the images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


class InternVL2():

    def __init__(self, args):
        # devid
        self.device = args.devid

        # load tokenizer
        print("Load " + args.tokenizer + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                                       trust_remote_code=True,
                                                       use_fast=False) # 这里使用use_fase和不用use_fast得到的结果不同，要特别注意
        self.tokenizer.decode([0])  # warm up

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = '<|system|>\n你是由上海人工智能实验室联合商汤科技开发的书生多模态大模型，英文名叫InternVL, 是一个有用无害的人工智能助手。<|end|><|user|>\n'
        self.system_ids = self.tokenizer.encode(self.system_prompt + "<img>")
        self.system_offset = len(self.system_ids)
        self.image_transform = build_transform(448)

        # load model
        self.model = chat.InternVL2()
        self.model.init(self.device, args.model_path)

        # parameters
        self.SEQLEN = self.model.SEQLEN
        self.ID_EOS = self.tokenizer.eos_token_id
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.ID_IM_CONTEXT = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.num_image_token = 256
        self.image_ids = [self.ID_IM_CONTEXT] * self.num_image_token

    def load_image(self, image_file):
        image = Image.open(image_file).convert('RGB')
        pixel_values = self.image_transform(image)
        return pixel_values

    def encode(self):
        if len(self.image_str_list) == 0:
            prompt = self.system_prompt + self.input_str + "<|end|><|assistant|>\n"
            self.input_ids = self.tokenizer.encode(prompt)
            self.image_offset = 0
            self.pixel_values = []
            return
        self.pixel_values = []
        self.image_offset = [self.system_offset + i * self.num_image_token for i in range(len(self.image_str_list))]
        self.image_offset = []
        image_num = 0
        if self.image_str_list and any(self.image_str_list):
            for image_str in self.image_str_list:
                if os.path.exists(image_str):
                    self.pixel_values = self.pixel_values + self.load_image(image_str).flatten().tolist()
                    self.image_offset.append(self.system_offset + image_num * self.num_image_token)
                    image_num+=1
                else:
                    continue
        
            self.system_prefix = self.system_prompt + "<img>" + "<IMG_CONTEXT>" * self.num_image_token * image_num
            prompt = self.system_prefix + "</img>\n{}<|end|><|assistant|>\n".format(self.input_str)
        else:
            self.system_prefix = self.system_prompt
            prompt = self.system_prefix + "\n{}<|end|><|assistant|>\n".format(self.input_str)
        self.input_ids = self.tokenizer.encode(prompt)

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            self.image_str = input("\nImage Path: ")
            self.image_str_list = [img for img in self.image_str.split(",")]
            print("\nAnswer:")
            image_num = 0
            for image_str in self.image_str_list:
                if not os.path.exists(image_str):
                    print("Can't find image: {}".format(image_str))
                    continue
                else:
                    image_num+=1
            self.encode()
            # Chat
            first_start = time.time()
            token = self.model.forward_first(self.input_ids, self.pixel_values,
                                             self.image_offset)
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            max_token_length = image_num * 256
            inputs_token_num = self.model.token_length
            while token not in [self.ID_EOS, self.ID_IM_END, self.ID_END
                                ] and self.model.token_length < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens,
                                             skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode(
                            [token, token],
                            skip_special_tokens=True)[len(pre_word):]
                    text += word
                    print(word, flush=True, end="")
                    full_word_tokens = []
                token = self.model.forward_next()
                tok_num += 1
                if tok_num >= max(max_token_length, 85):
                    break
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nself.SEQLEN: {self.SEQLEN}")
            print(f"inputs_token_length: {inputs_token_num}")
            print(f"self.model.token_length: {self.model.token_length}")
            print(f"max_token_length: {max_token_length}")
            print(f"FTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")
            print(f"TOKENS: {tok_num} token")
            print(f"Total time: {first_duration + next_duration} s")
    
def main(args):
    model = InternVL2(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='path to the bmodel file')
    parser.add_argument('-t',
                        '--tokenizer',
                        type=str,
                        default="../token_config",
                        help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=int,
                        default=0, help='device ID to use')
    args = parser.parse_args()
    main(args)
