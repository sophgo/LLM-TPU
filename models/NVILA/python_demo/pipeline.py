import argparse
import os, sys, time
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, SiglipImageProcessor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from support.preprocess import process_image
from einops import rearrange


class NVILA():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]
        self.system_prompt = 'You are a helpful assistant'

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(args.config_path)
        self.tokenizer.stop_tokens = ['<|im_end|>']
        self.tokenizer.stop_token_ids = [151645]
        self.tokenizer.add_tokens(["<vila/sentinel>"], special_tokens=True)
        self.tokenizer.sentinel_token = "<vila/sentinel>"
        self.tokenizer.sentinel_token_id = [151648]
        self.tokenizer.media_tokens = {"image": "<image>","video": "<vila/video>",}
        self.tokenizer.media_token_ids = {}
        for name, token in self.tokenizer.media_tokens.items():
            self.tokenizer.add_tokens([token], special_tokens=True)
            self.tokenizer.media_token_ids[name] = self.tokenizer.convert_tokens_to_ids(token)
        self.image_processor = SiglipImageProcessor.from_pretrained(args.config_path)

        # warm up
        self.tokenizer.decode([0])

        # load image
        self.EOS = [self.tokenizer.eos_token_id]

        # load model
        self.load_model(args)


    def load_model(self, args):
        import chat
        self.model = chat.NVILA()
        self.model.init(self.devices, args.model_path)
        self.SEQLEN = self.model.SEQLEN


    def process_input(self, media_path):
        if media_path == "" or not os.path.exists(media_path):
            print("Can't find image or video: {}".format(media_path))
            media_tokens = ""
            pixel_values, block_size = None, None
        else:
            media_tokens = "<image>\n"
            image = Image.open(media_path).convert('RGB')
            pixel_values, block_size = process_image(image, self.image_processor)

        prompt = f'<|im_start|>system\n{self.system_prompt}<|im_end|>\n<|im_start|>user\n{media_tokens}{self.input_str}<|im_end|>\n<|im_start|>assistant\n'
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
        return input_ids, pixel_values, block_size


    def merge_chessboard(self, x, num_split_h, num_split_w, N):
        x_merge = rearrange(x, "(nh nw) (h w) c-> 1 c (nh h) (nw w)", nh=num_split_h, nw=num_split_w, h=N, w=N)
        return x_merge


    def split_chessboard(self, x, num_split_h, num_split_w):
        x_split = rearrange(x, "1 c (nh h) (nw w) -> (nh nw) c h w", nh=num_split_h, nw=num_split_w)
        return x_split


    def vit_process_image(self, inputs):
        pixel_values = inputs[1]
        block_size = inputs[2]
        if pixel_values is None or block_size is None:
            return torch.tensor([])
        num_blocks = pixel_values.shape[0]
        image_features = []
        for i in range(num_blocks):
            image_features.append(
                torch.tensor(self.model.forward_vit(
                    pixel_values[i].flatten().tolist()), dtype=torch.int16)
                    .view(torch.bfloat16).reshape(1, 1024, 1152))
        image_feature = torch.cat(image_features, dim=0)
        scale0_feature = self.merge_chessboard(
            image_feature[0:1], num_split_h=1, num_split_w=1, N=32)
        scale1_feature = self.merge_chessboard(
            image_feature[1:5], num_split_h=2, num_split_w=2, N=32)
        scale2_feature = self.merge_chessboard(
            image_feature[5:], block_size[0], block_size[1], N=32)
        output_size = torch.Size([32*block_size[0], 32*block_size[1]])
        image_feature = torch.cat(
            [
                torch.nn.functional.interpolate(
                    scale0_feature, size=(32*block_size[0], 32*block_size[1])),
                torch.nn.functional.interpolate(
                    scale1_feature, size=(32*block_size[0], 32*block_size[1])),
                scale2_feature
            ],
            dim=1,
        )
        image_feature = self.split_chessboard(image_feature, block_size[0], block_size[1])
        full_image_feature = torch.zeros([20, 3456, 32, 32], dtype=torch.bfloat16)
        full_image_feature[:block_size[0] * block_size[1]] = image_feature
        full_image_feature = full_image_feature.view(torch.int16).flatten().tolist()
        image_feature = torch.tensor(self.model.forward_projector(full_image_feature), dtype=torch.int16)
        image_feature = image_feature.reshape(-1, 256, 3584)[:block_size[0] * block_size[1]].view(torch.bfloat16)
        image_feature = self.merge_chessboard(
            image_feature, block_size[0], block_size[1], N=16)
        image_feature = rearrange(image_feature, "1 c h w -> (h w) c")
        import numpy
        numpy.savez('res.npz',image_feature.to(torch.float32).numpy())
        return image_feature


    def chat(self):
        """
        Start a chat session.
        """
        # Instruct
        print("""\n=================================================================
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

            media_path = input("\nImage Path: ")
            media_path = media_path.strip()
            inputs = self.process_input(media_path)
            token_len = inputs[0].shape[0]

            # check tokens
            if not self.input_str:
                print("Sorry: your question is empty!!")
                return
            if token_len > self.SEQLEN - 128:
                print(
                    "The maximum question length should be shorter than {} but we get {} instead.".format(
                        self.SEQLEN, token_len
                    )
                )
                continue

            print("\nAnswer: ", end="")
            self.stream_answer(inputs)


    def stream_answer(self, inputs):
        """
        Stream the answer for the given inputs.
        """
        tok_num = 0
        self.answer_token = []

        # First token
        first_start = time.time()
        media_embeds = self.vit_process_image(inputs).view(torch.int16)
        token = self.model.forward_first(inputs[0].flatten().tolist(),
                                         media_embeds.flatten().tolist())
        first_end = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS:
            self.answer_token.append(token)
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue
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

        print(self.answer_token)

def main(args):
    model = NVILA(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../support/processor", help='path to the processor file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    args = parser.parse_args()
    main(args)
