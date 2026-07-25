import os, sys, re, time
import argparse
import chat
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from tokenizer.rwkv_tokenizer import TRIE_TOKENIZER
   

class RWKV7():
    def __init__(self, args):
        # devid
        self.devices = [int(d) for d in args.devid.split(",")]
        self.tokenizer = TRIE_TOKENIZER('../tokenizer/rwkv_vocab_v20230424.txt')
        self.state = torch.load(args.state_path) if args.state_path is not None else None     
        self.prompt_mode = args.prompt_mode

        self.model = chat.RWKV7()
        print(f"Load Model: {args.model_path} ...")
        load_start = time.time()
        self.model.init(self.devices, args.model_path)
        load_end = time.time()
        print(f"\nLoad Time: {(load_end - load_start):.3f} s")

        # generation config
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.presence_penalty = args.presence_penalty
        self.frequency_penalty = args.frequency_penalty
        self.penalty_decay = args.penalty_decay

        # if loading a state, change sample strategy
        if self.state != None:
            self.top_p = 0.2
            self.presence_penalty = 0.3
            self.frequency_penalty = 0.3
            self.state_list = []
            for i in range(self.model.NUM_LAYERS):
                self.state_list.append(
                    self.state[f'blocks.{i}.att.time_state']
                    .transpose(1,2).flatten().tolist())
            self.model.load_state(self.state_list)
        else:
            out = self.model.clear_state()

    def chat(self):
        print(
            "\n=================================================================\n"
            "1. If you want to quit, please enter one of [q, quit, exit]\n"
            "2. To create a new chat session, please enter one of [clear, new]\n"
            "================================================================="
            )
        while True:
            input_str = input("\nUser: ")
            if input_str in ["exit", "q", "quit"]:
                break
            elif input_str in ["clear", "new"]:
                self.model.clear_state()
                continue
            elif not input_str:
                print("Sorry: your question is empty!!")
                continue
            else:
                msg = re.sub(r"\n+", "\n", input_str.strip())
                ctx = "User: " + msg + "\n\nAssistant:"
                tokens = self.tokenizer.encode(ctx)
                print("\nAssistant:", end="")
                self.answer(tokens)

    def answer(self, tokens):
        # First token
        first_start = time.time()
        while(len(tokens) > 0):
            prefill_tokens = tokens[:self.model.CHUNK_LEN]
            if len(prefill_tokens) != self.model.CHUNK_LEN:
                prefill_tokens = [33 for _ in range(self.model.CHUNK_LEN)]
                prefill_tokens[:len(tokens)-4] = tokens[:-4]
                prefill_tokens[-4:] = tokens[-4:]
            out = self.model.forward_seq(prefill_tokens)
            tokens = tokens[self.model.CHUNK_LEN:]
        first_end = time.time()

        occurrence = {}
        out_tokens = []
        out_last = 0

        # Following tokens
        for i in range(99999):

            for n in occurrence:
                # repetition penalty
                out[n] -= self.presence_penalty + occurrence[n] * self.frequency_penalty 

            # disable END_OF_TEXT
            out[0] -= 1e10

            token = self.tokenizer.sample_logits(torch.tensor(out), temperature=self.temperature, top_p=self.top_p)

            for xxx in occurrence:
                occurrence[xxx] *= self.penalty_decay
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)

            out = self.model.forward_one(token)
            out_tokens += [token]

            tmp = self.tokenizer.decode(out_tokens[out_last:])   

            # only print & update out_last when it's a valid utf-8 string and not ending with \n
            if ("\ufffd" not in tmp) and (not tmp.endswith("\n")):
                print(tmp, end="", flush=True)
                out_last = i + 1

            if "\n\n" in tmp:
                print(tmp, end="", flush=True)
                break

        # counting time
        next_end = time.time()
        first_duration = first_end - first_start
        next_duration = next_end - first_end
        tps = (len(out_tokens) - 1)  / next_duration

        print()
        print(f"FTL: {first_duration:.3f} s")
        print(f"TPS: {tps:.3f} token/s")

def main(args):
    model = RWKV7(args)
    model.chat()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-s', '--state_path', type=str, default=None,
                        help='path to the state file')
    parser.add_argument('-d', '--devid', type=str, default='0',
                        help='device ID to use')
    parser.add_argument('--prompt_mode', type=str, choices=["chat", "instruction"],
                        default="chat", help='prompt format')

    ######### generation config #########
    parser.add_argument('--temperature', type=float, default=0.9,
                        help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=0.3,
                        help='cumulative probability to consider as a set of candidates')
    parser.add_argument('--presence_penalty', type=float, default=0.5, 
                        help='penalty for all appeared tokens')
    parser.add_argument('--frequency_penalty', type=int, default=0.5,
                        help='penalty by token appears times')
    parser.add_argument('--penalty_decay', type=int, default=0.996,
                        help='penalty decay rate')
    #####################################
    args = parser.parse_args()
    main(args)
