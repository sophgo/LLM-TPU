import os
import time
import numpy as np
import argparse
from PIL import Image
from datasets import load_dataset
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


def parse_max_pixels(value):
    """
    If the input is a single number, convert it to an integer.
    If it contains a comma, parse it as a tuple (or list) of two integers, e.g., "128,124".
    """
    if ',' in value:
        parts = value.split(',')
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                "The input must be two integers separated by a comma, e.g., 128,124"
            )
        try:
            width = int(parts[0].strip())
            height = int(parts[1].strip())
        except ValueError:
            raise argparse.ArgumentTypeError(
                "The input values must be integers, e.g., 128,124")
        return int(width * height)
    else:
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "The input must be an integer or two integers separated by a comma, e.g., 128,124"
            )


def get_media_type(file):
    if isinstance(file, str):
        image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        _, ext = os.path.splitext(file)
        ext = ext.lower()
        if ext in image_exts:
            return "image"
        if ext in video_exts:
            return "video"
    elif isinstance(file, Image.Image):
        return "image"
    raise RuntimeError(f"Unsupported media type: {file}")


class Qwen3VL_Origin():

    def __init__(self, args):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_path, dtype="auto", device_map="auto")
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.MAX_PIXELS = args.max_pixels
        self.video_ratio = args.video_ratio

    def image_message(self, img_path):
        messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "image",
                    "image": img_path,
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": self.MAX_PIXELS
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
                    "fps": 1.0,
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": int(self.MAX_PIXELS * self.video_ratio),
                    "total_pixels": self.total_pixels
                },
                {
                    "type": "text",
                    "text": self.input_str
                },
            ],
        }]
        return messages

    def eval(self, input_str, media_path):
        self.input_str = input_str.strip()
        media_type = get_media_type(media_path)
        if media_type == "image":
            messages = self.image_message(media_path)
        elif media_type == "video":
            messages = self.video_message(media_path)
        inputs = self.processor.apply_chat_template(messages,
                                                    tokenize=True,
                                                    add_generation_prompt=True,
                                                    return_dict=True,
                                                    return_tensors="pt")
        inputs = inputs.to(self.model.device)
        outputs = self.model.generate(**inputs,
                                      max_new_tokens=1,
                                      do_sample=False)
        return self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:])


def load_model(args):
    if os.path.isdir(args.model_path):
        model = Qwen3VL_Origin(args)
        eval_type = "origin"
    elif args.model_path.endswith(".bmodel"):
        from pipeline import Qwen3_VL

        class Qwen3VL_Eval(Qwen3_VL):

            def __init__(self, args):
                super().__init__(args)

            def eval(self, input_str, media_path):
                """
                Start a eval session.
                """
                self.input_str = input_str.strip()

                media_type = get_media_type(media_path)
                if media_type == "image":
                    messages = self.image_message(media_path)
                elif media_type == "video":
                    messages = self.video_message(media_path)
                else:
                    print("Unsupported media type: {}".format(media_path))
                    return None

                inputs = self.process(messages, media_type)
                token_len = inputs.input_ids.numel()
                if token_len > self.model.MAX_INPUT_LENGTH:
                    if media_type in ["image", "video"]:
                        print("grid_thw:{}".format(
                            inputs.image_grid_thw if media_type ==
                            "image" else inputs.video_grid_thw))
                    print(
                        "Error: The maximum question length should be shorter than {} but we get {} instead."
                        .format(self.model.MAX_INPUT_LENGTH, token_len))
                    return None
                if self.support_history:
                    self.model.clear_history()
                    self.history_max_posid = 0

                # Chat
                self.model.forward_embed(inputs.input_ids.numpy())
                if media_type == "image":
                    self.vit_process_image(inputs)
                    position_ids = self.get_rope_index(inputs.input_ids,
                                                       inputs.image_grid_thw,
                                                       self.ID_IMAGE_PAD)
                    self.max_posid = int(position_ids.max())
                    token = self.forward_prefill(position_ids.numpy())
                elif media_type == "video":
                    self.vit_process_video(inputs)
                    position_ids = self.get_rope_index(inputs.input_ids,
                                                       inputs.video_grid_thw,
                                                       self.ID_VIDEO_PAD)
                    self.max_posid = int(position_ids.max())
                    token = self.forward_prefill(position_ids.numpy())
                else:
                    position_ids = 3 * [i for i in range(token_len)]
                    self.max_posid = token_len - 1
                    token = self.forward_prefill(
                        np.array(position_ids, dtype=np.int32))
                return self.tokenizer.decode(token)

        model = Qwen3VL_Eval(args)
        eval_type = "eval"
    return model, eval_type


def args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--model_path',
                        type=str,
                        required=True,
                        help='path to the bmodel file or the origin model dir')
    parser.add_argument('--datasets',
                        type=str,
                        default="HuggingFaceM4/A-OKVQA",
                        help='path to the datasets')
    parser.add_argument('--max_pixels',
                        type=parse_max_pixels,
                        default="768,768",
                        help='max pixels for input image')
    parser.add_argument('-c',
                        '--config_path',
                        type=str,
                        default="../models/Qwen3_VL/config",
                        help='path to the processor file')
    parser.add_argument('--video_ratio',
                        type=float,
                        default=0.25,
                        help='Set video ratio, default is 0.25')
    parser.add_argument('-d',
                        '--devid',
                        type=int,
                        default=0,
                        help='device ID to use')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args()
    model, eval_type = load_model(args)

    ds = load_dataset(args.datasets, split="train")
    total = len(ds)
    print(f"Total samples: {total}")
    correct = 0

    for idx, item in enumerate(ds):
        # print(item["image"])
        # print(item["question"])
        # print(item["choices"])
        text = item["question"] + " \n" + " ".join(
            [f"{i+1}: {choice}. " for i, choice in enumerate(item["choices"])])
        text += "\nPlease answer the question, only give the number of the correct choice."
        # print(text)
        answer = model.eval(text, item["image"])
        correct_answer = item['correct_choice_idx'] + 1
        # print(f"Predicted answer: {answer}; Correct answer: {correct_answer}")
        if answer is not None and answer.strip() == str(
                item["correct_choice_idx"] + 1):
            correct += 1
        if (idx + 1) % 100 == 0:
            print(f"{idx+1}/{total}  Current accuracy: {correct/(idx+1):.2%}")

    print(f"Final accuracy: {correct/total:.2%}")
    output_file = "result_{YYMMDDHHMMSS}.txt".format(
        YYMMDDHHMMSS=time.strftime("%Y%m%d%H%M%S"))
    with open(output_file, "w") as f:
        f.write(f"Eval type: {eval_type}\n")
        f.write(f"Eval model: {args.model_path}\n")
        f.write(f"Final accuracy: {correct/total:.2%}")
