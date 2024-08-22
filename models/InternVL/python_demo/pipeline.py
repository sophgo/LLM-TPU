import time
import json
import torch
import numpy as np
import argparse
from PIL import Image
import torchvision.transforms as T
from transformers import AutoTokenizer
from torchvision.transforms.functional import InterpolationMode

# Preprocess the images
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InternVL():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path, trust_remote_code=True
        )

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = 'You are a helpful digital assistant. Please provide safe, ethical and accurate information to the user.'
        self.EOS = [32000, 32007] 
        self.system = {"role":"system","content":self.system_prompt}
        self.history = [self.system]
        self.enable_history = args.enable_history

        # load model
        self.load_model(args)


    def load_model(self, args):
        if args.decode_mode == "basic":
            import chat
            self.model = chat.Phi3()
            self.model.init(self.devices, args.model_path)
            self.model.temperature = args.temperature
            self.model.top_p = args.top_p
            self.model.repeat_penalty = args.repeat_penalty
            self.model.repeat_last_n = args.repeat_last_n
            self.model.max_new_tokens = args.max_new_tokens
            self.model.generation_mode = args.generation_mode
            self.model.prompt_mode = args.prompt_mode
        else:
            raise ValueError("decode mode: {} is illegal!".format(args.decode_mode))
        
        self.SEQLEN = self.model.SEQLEN


    def clear(self):
        self.history = [self.system]


    def update_history(self):
        if self.model.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end='')
            self.history = [self.system]
        else:
            self.history.append({"role":"assistant","content":self.answer_cur})


    def encode_tokens(self):
        self.history.append({"role":"user","content":self.input_str})
        return self.tokenizer.apply_chat_template(self.history, tokenize=True, add_generation_prompt=True)


    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print(
            """\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
================================================================="""
        )
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break
            # New Chat
            elif self.input_str in ["clear", "new"]:
                self.clear()
            # Chat
            else:
                tokens = self.encode_tokens()
                # breakpoint()
                # check tokens
                if not tokens:
                    print("Sorry: your question is empty!!")
                    return
                if len(tokens) > self.SEQLEN:
                    print(
                        "The maximum question length should be shorter than {} but we get {} instead.".format(
                            self.SEQLEN, len(tokens)
                        )
                    )
                    return

                print("\nAnswer: ", end="")
                self.stream_answer(tokens)


    def stream_answer(self, tokens):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        # Image information
        pixel_values = load_image('./supports/image1.jpg', max_num=12)
        pixel_list = pixel_values.flatten().tolist()
        IMAGE_SIZE = 448
        DOWNSAMPLE_RATIO = 0.5
        PATCH_SIZE = 14
        num_patches = pixel_values.shape[0]
        num_image_token = int((IMAGE_SIZE // PATCH_SIZE) ** 2 * (DOWNSAMPLE_RATIO ** 2))
        begin = 70
        offset = num_patches * num_image_token

        # First token
        first_start = time.time()
        token = self.model.forward_first(tokens, pixel_list, begin, offset)
        first_end = time.time()

        # Following tokens
        while token not in self.EOS and self.model.token_length < self.SEQLEN:
            pre_word = self.tokenizer.decode([token], skip_special_tokens=True)
            word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.model.forward_next()

        self.answer_cur = self.tokenizer.decode(self.answer_token)
        
        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        if self.enable_history:
            self.update_history()
        else:
            self.clear()

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

    # InternVL Web Demo
    def stream_predict(self, prompt, image, history):
        """
        Stream the answer for the given tokens.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []
        self.input_str = prompt
        tokens = self.encode_tokens()

        image = image.convert("RGB")
        transform = build_transform(input_size=448)
        images = dynamic_preprocess(image, image_size=448, use_thumbnail=True, max_num=12)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)   
        pixel_list = pixel_values.flatten().tolist()    
        
        print(history)

        IMAGE_SIZE = 448
        DOWNSAMPLE_RATIO = 0.5
        PATCH_SIZE = 14
        num_patches = pixel_values.shape[0]
        num_image_token = int((IMAGE_SIZE // PATCH_SIZE) ** 2 * (DOWNSAMPLE_RATIO ** 2))
        begin = 70
        offset = num_patches * num_image_token

        first_start = time.time()
        token = self.model.forward_first(tokens, pixel_list, begin, offset)
        first_end = time.time()
        # Following tokens

        while token not in self.EOS and self.model.token_length < self.SEQLEN:
            pre_word = self.tokenizer.decode([token], skip_special_tokens=True)
            word = self.tokenizer.decode([token, token], skip_special_tokens=True)[len(pre_word):]
            yield word
            self.answer_token += [token]
            print(word, flush=True, end="")
            tok_num += 1
            token = self.model.forward_next()

        self.answer_cur = self.tokenizer.decode(self.answer_token)

        if self.model.token_length >= self.SEQLEN:
            yield '##TOKEN_LENGTH_MAX'
            return
        
        next_end = time.time()
        first_duration = first_end-first_start
        next_duration = next_end-first_end
        tps = tok_num / next_duration

        if self.enable_history:
            self.update_history()
        else:
            self.clear()

        print('\n\n')
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")  

def main(args):
    model = InternVL(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../support/token_config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    parser.add_argument('--decode_mode', type=str, default="basic", choices=["basic", "jacobi"], help='mode for decoding')
    parser.add_argument('--enable_history', action='store_true', default=True, help="if set, enables storing of history memory.")
    args = parser.parse_args()
    main(args)
