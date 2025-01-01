import time
import argparse
from PIL import Image
import torchvision.transforms as T
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, Qwen2VLConfig
from qwen_vl_utils import process_vision_info
import chat
import json
import os
import torch


class Qwen2VL():

    def __init__(self, args):
        # devid
        self.device = args.devid
        self.processor = AutoProcessor.from_pretrained(args.processor_path,
                                                       trust_remote_code=True)
        
        self.tokenizer = self.processor.tokenizer
        with open(args.config, 'r') as f:
            self.config = json.load(f)
        self.resized_height = args.resized_height
        self.resized_width = args.resized_width

        # load model
        self.model = chat.Qwen2VL()
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]
        self.model.spatial_merge_size = self.config["vision_config"]["spatial_merge_size"]
        self.model.init(self.device, args.model_path)
        self.model.generation_mode = args.generation_mode
        self.SEQLEN = self.model.SEQLEN
        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def image_message(self, path):
        print("\033[31m如果输入为图片时，注意resized_height与resized_width与export_onnx.py时的保持一致\033[0m")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": path,
                        "resized_height": self.resized_height,
                        "resized_width": self.resized_width,
                    },
                    {"type": "text", "text": "Describe this image and tell a story."},
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
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Describe this video."},
                ],
            }
        ]
        return messages

    def process(self, path, type):
        messages = self.image_message(path) if type == "image" else self.video_message(path)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return inputs

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
            inputs = self.process(image_or_video_str, image_or_video_type)
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            if image_or_video_type == "image":
                vit_token_list = torch.where(inputs.input_ids==self.config["image_token_id"])[1].tolist()
                vit_offset = vit_token_list[0]
                valid_vit_length = len(vit_token_list)
                token = self.model.forward_first(inputs.input_ids.squeeze(0).tolist(), inputs.pixel_values.flatten().tolist(),
                                                inputs.image_grid_thw.squeeze(0).tolist(), vit_offset, valid_vit_length)
            elif image_or_video_type == "video":
                vit_token_list = torch.where(inputs.input_ids==self.config["video_token_id"])[1].tolist()
                vit_offset = vit_token_list[0]
                valid_vit_length = len(vit_token_list)
                token = self.model.forward_first(inputs.input_ids.squeeze(0).tolist(), inputs.pixel_values_videos.flatten().tolist(),
                                                inputs.video_grid_thw.squeeze(0).tolist(), vit_offset, valid_vit_length)
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
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
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = Qwen2VL(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-p', '--processor_path', type=str, default="../support/processor_config", help='path to the processor file')
    parser.add_argument('-c', '--config', type=str, default="../compile/files/Qwen2-VL-2B-Instruct/config.json", help='path to the model config file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('-g', '--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--resized_height', type=int, default=280, help='resized height')
    parser.add_argument('--resized_width', type=int, default=420, help='resized width')
    args = parser.parse_args()
    main(args)
