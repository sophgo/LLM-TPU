import time
import argparse
from PIL import Image
from transformers import AutoTokenizer
import chat
import json
import os
import sys
import copy
import torch

from functools import partial, reduce
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_transforms import (
    convert_to_rgb,
    normalize,
    rescale,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    to_numpy_array,
)

current_dir = os.path.dirname(os.path.abspath(__file__))
compile_dir = os.path.abspath(os.path.join(current_dir, "../compile"))
sys.path.insert(0, compile_dir)
from llava.conversation import conv_templates

IMAGE_TOKEN_INDEX=-200
size=(384,384)
resample=PILImageResampling.BICUBIC
data_format=ChannelDimension.FIRST
rescale_factor=0.00392156862745098
image_mean=(0.5, 0.5, 0.5)
image_std=(0.5, 0.5, 0.5)
def preprocess(images, return_tensors="pt"):
    if isinstance(images, Image.Image):
        images = [images]
    else:
        # to adapt video data
        images = [to_numpy_array(image) for image in images]
        assert isinstance(images, list)

    transforms = [
        convert_to_rgb,
        to_numpy_array,
        partial(resize, size=size, resample=resample, data_format=data_format),
        partial(rescale, scale=rescale_factor, data_format=data_format),
        partial(normalize, mean=image_mean, std=image_std, data_format=data_format),
        partial(to_channel_dimension_format, channel_dim=data_format, input_channel_dim=data_format),
    ]

    images = reduce(lambda x, f: [*map(f, x)], transforms, images)
    return images

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

class Model():

    def __init__(self, args):
        # devid
        self.device = args.devid
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path,
                                                       trust_remote_code=True)

        # load model
        self.model = chat.Model()
        self.model.NUM_LAYERS = args.num_layers
        self.model.init(self.device, args.model_path)
        self.model.generation_mode = args.generation_mode

        self.EOS = self.tokenizer.eos_token_id

    def process(self, path, type):
        images = [Image.open(path).convert("RGB")]
        image_tensors = preprocess(images)
        image_tensors = [torch.tensor(image, dtype=torch.float32) for image in image_tensors]


        conv_template = "llava_llama_3"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], "<image>"*730+"\n" + self.input_str)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        
        return input_ids, image_tensors

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

            image_or_video_str = input("\nImage or Video Path: ")
            if not os.path.exists(image_or_video_str):
                print("Can't find image or video: {}".format(image_or_video_str))
                continue

            image_or_video_type = input("\nImage or Video Type: ")
            if image_or_video_type not in ["image", "video"]:
                print("The type you input is: {}, not image or video".format(image_or_video_type))
                continue
            input_ids, image_tensors = self.process(image_or_video_str, image_or_video_type)
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            if image_or_video_type == "image":
                vit_token_list = torch.where(input_ids==IMAGE_TOKEN_INDEX)[1].tolist()
                vit_offset = vit_token_list[0]
                valid_vit_length = len(vit_token_list)
                token = self.model.forward_first(input_ids.squeeze(0).tolist(), image_tensors[0].flatten().tolist(), vit_offset, valid_vit_length)
            elif image_or_video_type == "video":
                raise NotImplementedError
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.EOS] and self.model.token_length < self.model.SEQLEN:
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
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Model(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-p', '--tokenizer_path', type=str, default="../compile/exported_tokenizer/", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('-n', '--num_layers', type=int, default=32, help='number of layers to use')
    parser.add_argument('-g', '--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    args = parser.parse_args()
    main(args)
