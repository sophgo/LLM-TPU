import argparse

import chat
import torch
import time
import os
from transformers import AutoProcessor, GenerationConfig, AutoConfig


class Gemma3():

    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.config_path + " ...")
        self.processor = AutoProcessor.from_pretrained(args.config_path, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(args.config_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        self.system_prompt = {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant."
            }]
        }
        self.ID_IMAGE_PAD = self.config.image_token_index
        self.EOS = self.config.eos_token_id

        self.model = chat.Gemma3()
        self.init_params(args)
        self.load_model(args.model_path)

    def load_model(self, model_path):
        load_start = time.time()
        self.model.init(self.devices, model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

    def init_params(self, args):
        self.model.generation_mode = "greedy"
        if args.do_sample:
            gen_config = GenerationConfig.from_pretrained(args.config_path)
            self.model.generation_mode = "sample"
            self.model.temperature = gen_config.temperature
            self.model.top_p = gen_config.top_p
            self.model.top_k = gen_config.top_k
            self.model.penalty = gen_config.repetition_penalty
            for i in gen_config.eos_token_id:
                self.EOS.append(i)

    def text_message(self):
        # yapf: disable
        messages = [
            self.system_prompt,
            {
                "role": "user",
                "content": [{"type": "text", "text": self.input_str}],
            }]
        # yapf: enable
        return messages

    def image_message(self, path):
        # yapf: disable
        messages = [
            self.system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": path},
                    {"type": "text", "text": self.input_str}],
            }]
        # yapf: enable
        return messages

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
            if media_path == "":
                messages = self.text_message()
                media_type = "text"
            elif not os.path.exists(media_path):
                print("Can't find image: {}".format(media_path))
                continue
            else:
                messages = self.image_message(media_path)
                media_type = "image"

            inputs = self.processor.apply_chat_template(messages,
                                                        tokenize=True,
                                                        return_dict=True,
                                                        return_tensors="pt",
                                                        add_generation_prompt=True)
            token_len = inputs.input_ids.numel()
            if token_len >= self.model.SEQLEN - 128:
                print(
                    "The maximum question length should be shorter than {} but we get {} instead.".
                    format(self.model.SEQLEN, token_len))
                continue
            print("\nAnswer:")
            # Chat
            first_start = time.time()
            self.model.forward_embed(inputs.input_ids.flatten().tolist())
            if media_type == "image":
                vit_token_list = torch.where(inputs.input_ids == self.ID_IMAGE_PAD)[1].tolist()
                vit_offset = vit_token_list[0]
                self.model.forward_vit(inputs.pixel_values.numpy(), vit_offset)

            token = self.model.forward_first()
            first_end = time.time()
            tok_num = 0
            # Following tokens
            full_word_tokens = []
            text = ""
            while token not in self.EOS and self.model.token_length < self.model.SEQLEN:
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
    model = Gemma3(args)
    model.chat()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config", help='path to the tokenizer file')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--do_sample', action='store_true', help="if set, generate tokens by sample parameters")
    # yapf: enable
    args = parser.parse_args()
    main(args)
