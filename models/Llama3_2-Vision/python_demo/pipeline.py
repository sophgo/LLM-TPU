import argparse

import time
import torch
from PIL import Image
from transformers import AutoProcessor

class Llama3_2_Vision():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]

        # load tokenizer
        print("Load " + args.tokenizer_path + " ...")
        self.processor = AutoProcessor.from_pretrained(args.tokenizer_path)
        self.tokenizer = self.processor.tokenizer

        # warm up
        self.tokenizer.decode([0])

        # preprocess parameters, such as prompt & tokenizer
        # Prompting with images is incompatible with system messages
        self.EOS = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>"), 128001, 128008, 128009]
        self.image = Image.open(args.image_path)
        self.system = {"role": "user", "content": [{"type": "image"}]}
        self.history = [self.system]
        self.enable_history = args.enable_history

        # load model
        self.load_model(args)


    def load_model(self, args):
        import chat
        self.model = chat.Llama3_2()
        self.model.init(self.devices, args.model_path)
        self.model.temperature = args.temperature
        self.model.top_p = args.top_p
        self.model.repeat_penalty = args.repeat_penalty
        self.model.repeat_last_n = args.repeat_last_n
        self.model.max_new_tokens = args.max_new_tokens
        self.model.generation_mode = args.generation_mode
        self.model.prompt_mode = args.prompt_mode
        self.SEQLEN = self.model.SEQLEN


    def clear(self):
        self.history = [self.system]


    def update_history(self):
        if self.model.token_length >= self.SEQLEN:
            print("... (reach the maximal length)", flush=True, end='')
            self.history = [self.system]
        else:
            self.history.append({"role":"assistant","content":self.answer_cur})


    def process_input(self):
        self.history.append({"role":"user","content":[{"type": "text", "text": self.input_str}]})
        input_text = self.processor.apply_chat_template(self.history, add_generation_prompt=True)
        inputs = self.processor(self.image, input_text, return_tensors="pt")
        for ins in inputs.keys():
            if inputs[ins].dtype == torch.int64:
                inputs[ins] = inputs[ins].to(torch.int32)
            inputs[ins] = inputs[ins].flatten().tolist()
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
                inputs = self.process_input()
                tokens = inputs['input_ids']

                # check tokens
                if not self.input_str:
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
                self.stream_answer(inputs)


    def stream_answer(self, inputs):
        """
        Stream the answer for the given inputs.
        """
        tok_num = 0
        self.answer_cur = ""
        self.answer_token = []

        # First token
        first_start = time.time()
        token = self.model.forward_first(inputs['input_ids'],
                                         inputs['pixel_values'],
                                         inputs['aspect_ratio_ids'],
                                         inputs['aspect_ratio_mask'],
                                         inputs['cross_attention_mask'])
        first_end = time.time()

        # Following tokens
        full_word_tokens = []
        while token not in self.EOS and self.model.token_length < self.SEQLEN:
            full_word_tokens.append(token)
            word = self.tokenizer.decode(full_word_tokens, skip_special_tokens=True)
            if "�" in word:
                token = self.model.forward_next()
                tok_num += 1
                continue

            self.answer_token += full_word_tokens
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

        self.answer_cur = self.tokenizer.decode(self.answer_token)

        if self.enable_history:
            self.update_history()
        else:
            self.clear()

def main(args):
    model = Llama3_2_Vision(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='path to the bmodel file')
    parser.add_argument('-t', '--tokenizer_path', type=str, default="../token_config", help='path to the tokenizer file')
    parser.add_argument('-i', '--image_path', type=str, default="./test.jpg", help='path to image')
    parser.add_argument('-d', '--devid', type=str, default='0', help='device ID to use')
    parser.add_argument('--temperature', type=float, default=0.6, help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=0.9, help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.0, help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32, help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, choices=["greedy", "penalty_sample"], default="greedy", help='mode for generating next token')
    parser.add_argument('--prompt_mode', type=str, choices=["prompted", "unprompted"], default="prompted", help='use prompt format or original input')
    parser.add_argument('--enable_history', action='store_true', default=True, help="if set, enables storing of history memory.")
    args = parser.parse_args()
    main(args)
