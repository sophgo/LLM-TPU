# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import os, sys, time
import torch
from transformers import AutoTokenizer, GenerationConfig
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

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


def dynamic_preprocess(image, min_num=1, max_num=4, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set((i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1)
                        for j in range(1, n + 1) if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def process_image(image_file, input_size=448, max_num=4):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


class InternVL3():

    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]
        self.system_prompt = '你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位联合开发的多模态大语言模型。'

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_path)

        # warm up
        self.tokenizer.decode([0])

        # load image
        self.EOS = [self.tokenizer.convert_tokens_to_ids('<|im_end|>')]

        # load model
        self.load_model(args)

    def load_model(self, args):
        import chat
        self.model = chat.InternVL3()
        self.model.init(self.devices, args.model_path)
        self.SEQLEN = self.model.SEQLEN
        self.init_params(args)

    def init_params(self, args):
        self.model.generation_mode = "greedy"
        self.stop_strings = []
        if args.do_sample:
            gen_config = GenerationConfig.from_pretrained(args.config_path)
            self.model.generation_mode = "sample"
            self.model.temperature = gen_config.temperature
            self.model.top_p = gen_config.top_p
            self.model.top_k = gen_config.top_k
            self.model.penalty = gen_config.repetition_penalty
            if gen_config.eos_token_id is not None:
                if isinstance(gen_config.eos_token_id, int):
                    self.EOS.append(gen_config.eos_token_id)
                if isinstance(gen_config.eos_token_id, list):
                    self.EOS.extend(gen_config.eos_token_id)
            if gen_config.stop_strings is not None:
                self.stop_strings = gen_config.stop_strings

    def process_input(self, media_path):
        if media_path == "":
            media_tokens = ""
            pixel_values = torch.tensor([])
        elif os.path.exists(media_path):
            media_tokens = "<image>\n"
            pixel_values = process_image(media_path)
            num_patches = pixel_values.shape[0]
            IMG_START_TOKEN = '<img>'
            IMG_END_TOKEN = '</img>'
            IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
            visual_length = self.model.NUM_IMAGE_TOKEN * num_patches
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * visual_length + IMG_END_TOKEN
            media_tokens = media_tokens.replace('<image>', image_tokens, 1)
        else:
            raise FileNotFoundError(f"Media file not found: {media_path}")

        prompt = f'<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{media_tokens}{self.input_str}<|im_end|>\n<|im_start|>assistant\n'
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        return input_ids.flatten().numpy().astype(np.int32), pixel_values.flatten().numpy().astype(
            np.float32)

    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
1. If you want to quit, please enter one of [q, quit, exit]
2. To create a new chat session, please enter one of [clear, new]
=================================================================""")
        # Stop Chatting with "exit" input
        while True:
            self.input_str = input("\nQuestion: ")
            # Quit
            if self.input_str in ["exit", "q", "quit"]:
                break

            media_path = input("\nImage Path: ")
            media_path = media_path.strip()
            inputs = self.process_input(media_path)
            token_len = len(inputs[0])

            # check tokens
            if not self.input_str:
                print("Sorry: your question is empty!!")
                return
            if token_len > self.SEQLEN - 128:
                print(
                    "The maximum question length should be shorter than {} but we get {} instead.".
                    format(self.SEQLEN, token_len))
                continue

            print("\nAnswer: ", end="")
            self.stream_answer(inputs)

    def stream_answer(self, inputs):
        """
        Stream the answer for the given inputs.
        """
        tok_num = 0
        self.answer_cur = ""

        # First token
        first_start = time.time()
        token = self.model.forward_first(inputs[0], inputs[1])
        first_end = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS and self.model.token_length < self.model.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
            self.answer_cur += word
            if any(self.answer_cur.endswith(stop) for stop in self.stop_strings):
                break
            print(word, flush=True, end="")
            token = self.model.forward_next()
            tok_num += 1
            full_word_tokens = []

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = tok_num / next_duration

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = InternVL3(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default="../support/processor",
                        help='path to the processor file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--do_sample',
                        action='store_true',
                        help="if set, generate tokens by sample parameters")
    args = parser.parse_args()
    main(args)
