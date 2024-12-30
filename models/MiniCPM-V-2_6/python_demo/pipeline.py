import time
import torch
import argparse
from PIL import Image
# import torchvision.transforms as T
from transformers import AutoTokenizer, AutoProcessor
# from torchvision.transforms.functional import InterpolationMode
import chat
import os


class MiniCPMV():
    def __init__(self, args):
        # devid
        self.device = args.devid

        # load tokenizer
        print("Load " + args.processor_path + " ...")
        self.processor = AutoProcessor.from_pretrained(
            args.processor_path, trust_remote_code=True
        )

        self.tokenizer = self.processor.tokenizer
        self.tokenizer.decode([0])  # warm up

        # load model
        self.model = chat.MiniCPMV()
        self.model.init(self.device, args.model_path)
        self.SEQLEN = self.model.SEQLEN
        self.ID_EOS = self.tokenizer.eos_token_id
        self.ID_IM_END = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        # parameters
        self.MAX_SLICE_NUMS = self.processor.image_processor.max_slice_nums

    def encode_with_image(self):
        print("\033[31m请注意，目前不支持图片size可变，因此图片会进行resize。目标size为export_onnx时的图片size\033[0m")
        print("\033[31m请注意，如果你export_onnx.py时使用的是其他图片size，请修改下面这行代码: single_imsize = (448, 448)\033[0m")
        single_imsize = (448, 448)
        inserted_image_str = "(<image>./</image>)\n"
        images = []
        contents = []
        for i in range(self.patch_num):
            images.append(Image.open(self.image_str[i]).convert('RGB').resize(single_imsize, Image.LANCZOS))
            contents.append(inserted_image_str)
        contents.append(self.input_str)

        msgs = [{'role': 'user', 'content': ''.join(contents)}]
        prompts_lists = self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            prompts_lists,
            [images],
            max_slice_nums=self.MAX_SLICE_NUMS,
            use_image_id=None,
            return_tensors="pt",
            max_length=8192
        )
        self.input_ids = inputs.input_ids[0]
        self.pixel_values = torch.cat(inputs["pixel_values"][0], dim=0).flatten().tolist()
        self.image_offsets = torch.where(self.input_ids==128244)[0].tolist()
        self.input_ids = self.input_ids.tolist()

    def encode(self):
        msgs = [{'role': 'user', 'content': '{}'.format(self.input_str)}]
        prompts_lists = self.processor.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            prompts_lists,
            [[]],
            return_tensors="pt",
            max_length=8192
        )
        self.image_offsets = []
        self.pixel_values = []
        self.input_ids = inputs.input_ids[0].tolist()

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
            try:
                self.patch_num = int(input("\nImage Num: "))
            except:
                self.patch_num = 0
            self.image_str = [input(f"\nImage Path {i}: ") for i in range(self.patch_num)] if self.patch_num >= 1 else []

            if self.image_str:
                missing_images = [x for x in self.image_str if not os.path.exists(x)]
                if missing_images:
                    print("\nMissing images: {}".format(", ".join(missing_images)))
                    continue
                else:
                    self.encode_with_image()
            else:
                self.encode()

            print("\nAnswer:")
            # Chat
            first_start = time.time()
            token = self.model.forward_first(
                self.input_ids, self.pixel_values, self.image_offsets, self.patch_num)
            first_end = time.time()
            tok_num = 1
            # Following tokens
            full_word_tokens = []
            while token not in [self.ID_EOS, self.ID_IM_END] and self.model.token_length < self.SEQLEN:
                full_word_tokens.append(token)
                word = self.tokenizer.decode(
                    full_word_tokens, skip_special_tokens=True)
                if "�" not in word:
                    if len(full_word_tokens) == 1:
                        pre_word = word
                        word = self.tokenizer.decode([token, token], skip_special_tokens=True)[
                            len(pre_word):]
                    print(word, flush=True, end="")
                    full_word_tokens = []
                tok_num += 1
                token = self.model.forward_next()
            next_end = time.time()
            first_duration = first_end - first_start
            next_duration = next_end - first_end
            tps = tok_num / next_duration
            print(f"\nFTL: {first_duration:.3f} s")
            print(f"TPS: {tps:.3f} token/s")


def main(args):
    model = MiniCPMV(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str,
                        required=True, help='path to the bmodel file')
    parser.add_argument('-p', '--processor_path', type=str,
                        default="../support/processor_config", help='path to the processor file')
    parser.add_argument('-d', '--devid', type=int,
                        default=0, help='device ID to use')
    args = parser.parse_args()
    main(args)
