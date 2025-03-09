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
        self.processor = AutoProcessor.from_pretrained(args.config_path, trust_remote_code=True)
        config_file = args.config_path + "/config.json"
        self.tokenizer = self.processor.tokenizer
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        self.resized_height = args.resize[0]
        self.resized_width = args.resize[1]

        # load model
        self.model = chat.Qwen2VL()
        self.model.NUM_LAYERS = self.config["num_hidden_layers"]
        self.model.spatial_merge_size = self.config["vision_config"]["spatial_merge_size"]
        self.model.init(self.device, args.model_path)

        self.ID_END = self.tokenizer.convert_tokens_to_ids("<|end|>")
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

    def text_message(self):
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": self.input_str
                },
            ],
        }]
        return messages

    def image_message(self, path):
        if self.resized_height != 0 and self.resized_width != 0:
            messages = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": path,
                        "resized_height": self.resized_height,
                        "resized_width": self.resized_width,
                    },
                    {
                        "type": "text",
                        "text": self.input_str
                    },
                ],
            }]
        else:
            messages = [{
                "role":
                "user",
                "content": [
                    {
                        "type": "image",
                        "image": path,
                    },
                    {
                        "type": "text",
                        "text": self.input_str
                    },
                ],
            }]
        return messages

    def video_message(self, path):
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "video",
                    "video": path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {
                    "type": "text",
                    "text": self.input_str
                },
            ],
        }]
        return messages

    def get_media_type(self, file_path):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext in image_exts:
            return "image"
        if ext in video_exts:
            return "video"
        raise RuntimeError(f"Unsupported media type: {ext}")

    def process(self, messages):
        text = self.processor.apply_chat_template(messages,
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if "pixel_values" in inputs and inputs.pixel_values.shape[0] > self.model.MAX_PIXELS:
            raise ValueError(
                f"The video or image that you input is {inputs.pixel_values.shape[0]}, exceed to {self.model.MAX_PIXELS}"
            )
        if "pixel_values_videos" in inputs and inputs.pixel_values_videos.shape[
                0] > self.model.MAX_PIXELS:
            raise ValueError(
                f"The video or image that you input is {inputs.pixel_values_videos.shape[0]}, exceed to {self.model.MAX_PIXELS}"
            )

        return inputs

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

            media_path = input("\nImage or Video Path: ")
            if not os.path.exists(media_path):
                print("Can't find image or video: {}".format(media_path))
                continue
            media_type = self.get_media_type(media_path)
            if media_type == "image":
                messages = self.image_message(media_path)
            else:
                messages = self.video_message(media_path)
            inputs = self.process(messages)
            print("\nAnswer:")

            # Chat
            first_start = time.time()
            if media_type == "image":
                vit_token_list = torch.where(
                    inputs.input_ids == self.config["image_token_id"])[1].tolist()
                vit_offset = vit_token_list[0]
                valid_vit_length = len(vit_token_list)
                token = self.model.forward_first(
                    inputs.input_ids.squeeze(0).tolist(),
                    inputs.pixel_values.flatten().tolist(),
                    inputs.image_grid_thw.squeeze(0).tolist(), vit_offset, valid_vit_length)
            elif media_type == "video":
                vit_token_list = torch.where(
                    inputs.input_ids == self.config["video_token_id"])[1].tolist()
                vit_offset = vit_token_list[0]
                valid_vit_length = len(vit_token_list)
                token = self.model.forward_first(
                    inputs.input_ids.squeeze(0).tolist(),
                    inputs.pixel_values_videos.flatten().tolist(),
                    inputs.video_grid_thw.squeeze(0).tolist(), vit_offset, valid_vit_length)
            else:
                empty = []
                token = self.model.forward_first(
                    inputs.input_ids.squeeze(0).tolist(), empty, [], 0, 0)
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in [self.ID_IM_END, self.ID_END
                                ] and self.model.token_length < self.model.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token],
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
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default="config",
                        help='path to the processor file')
    parser.add_argument('-d', '--devid', type=int, default=0, help='device ID to use')
    parser.add_argument('-r',
                        '--resize',
                        nargs=2,
                        type=int,
                        default=[280, 420],
                        metavar=('HEIGHT', 'WIDTH'),
                        help='resized height and width, for example: --resize 280 420')
    args = parser.parse_args()
    main(args)
