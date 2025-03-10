import os
import sys
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor

import datasets
import pandas as pd
from transformers import AutoTokenizer, AutoProcessor

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
template_dir = os.path.abspath(os.path.join(current_dir, "../../template/demo"))
sys.path.insert(0, parent_dir)
sys.path.insert(0, template_dir)


import chat
from pipeline import Model
from tools.indicators import Indicators


class Task():
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = self.init_model()
        self.indicator = Indicators()

    def init_model(self):
        if self.args.device == "tpu":
            return Model(self.args)

    def post_process(self, response):
        response = self.model.tokenizer.decode(response)

        # deepseek
        if "</think>" in response:
            res = response.split("</think>")[1]
        else:
            res = response

        return res

    def process_subset(self, subset):
        results = []
        for item in subset:
            result = self.process_item(item)
            results.append(result)
        return results

    def process_item(self, item):
        context = item['context']
        question = item['question']
        ref = item['answers']

        model_input = f"{context} {question}"
        result_tokens = self.model.generate(model_input)
        tokens = self.model.encode_tokens(model_input)
        answer = self.post_process(result_tokens)

        return {
            "context": context,
            "question": question,
            "reference": ", ".join(ref['text']),
            "answer": answer,
            "f1": self.indicator.calc_f1(answer, ref['text'])['f1'],
            "tps": self.model.tps,
            "ftl": self.model.ftl
        }


class ParallelEvaluator:
    def __init__(self, args):
        self.args = args
        self.devices = args.devid
        self.dataset = datasets.load_dataset("SQuAD", split="validation")
        self.results = []

    def analyze(self):
        task_args = self.prepare_worker_args(0)
        task = Task(task_args)

        lengths = []
        for item in tqdm(self.dataset):
            text = f"{item['context']} {item['question']}".strip()
            encoded = task.model.encode_tokens(text)
            lengths.append(len(encoded))
            task.model.init_history()

        print(f"Max Length: {max(lengths)}")
        print(f"Min Length: {min(lengths)}")
        print(f"Median Length: {int(np.median(lengths))}")

    def evaluate(self):
        subsets = self.split_dataset()
        
        with ProcessPoolExecutor(max_workers=len(self.devices)) as executor:
            futures = [
                executor.submit(
                    self.worker,
                    self.prepare_worker_args(dev_id),
                    subset
                )
                for dev_id, subset in enumerate(subsets)
            ]

            for future in tqdm(futures, desc="Processing Progress"):
                self.results.extend(future.result())

    def split_dataset(self):
        return [self.dataset.shard(len(self.devices), i) for i in range(len(self.devices))]

    def prepare_worker_args(self, dev_id):
        worker_args = deepcopy(self.args)
        worker_args.devid = self.devices[dev_id]
        return worker_args

    @staticmethod
    def worker(args, subset):
        try:
            task = Task(args)
            return task.process_subset(subset)
        except Exception as e:
            print(f"Device {args.devid} processing failed: {str(e)}")
            return []

    def save_results(self, output_dir="results", filename="squad_validation.csv"):
        os.makedirs(output_dir, exist_ok=True)
        full_path = os.path.join(output_dir, filename)
        
        df = pd.DataFrame(self.results)
        df.to_csv(full_path, index=False)
        print(df)
        stats = {
            'avg_f1': df['f1'].mean(),
            'avg_ftl': df['ftl'].mean(),
            'avg_tps': df['tps'].mean()
        }

        print("\nEvaluation Statistics:")
        for k, v in stats.items():
            print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
            
        return df

def main(args):
    evaluator = ParallelEvaluator(args)
    # evaluator.analyze()
    evaluator.evaluate()
    evaluator.save_results()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--dir_path", type=str, default="./tmp",
                        help="dir path to the config/embedding/tokenizer")
    parser.add_argument('-b', '--model_path', type=str, default="",
                        help='path to the bmodel file')
    parser.add_argument('-d', '--devid', type=lambda s: s.split(','), required=True,
                        help='such as 0,1,2,3')
    parser.add_argument('--test_input', type=str,
                        help='the text for test')
    parser.add_argument('--test_media', type=str,
                        help='the media(image/video) path for test')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature scaling factor for the likelihood distribution')
    parser.add_argument('--top_p', type=float, default=1.0,
                        help='cumulative probability of token words to consider as a set of candidates')
    parser.add_argument('--repeat_penalty', type=float, default=1.2,
                        help='penalty for repeated tokens')
    parser.add_argument('--repeat_last_n', type=int, default=32,
                        help='repeat penalty for recent n tokens')
    parser.add_argument('--max_new_tokens', type=int, default=1024,
                        help='max new token length to generate')
    parser.add_argument('--generation_mode', type=str, default="greedy",
                        choices=["greedy", "penalty_sample"],
                        help='mode for generating next token')
    parser.add_argument('--resized_height', type=int, default=0,
                        help='use resized_height for vlm when resized_height != 0')
    parser.add_argument('--resized_width', type=int, default=0,
                        help='use resized_width for vlm when resized_width != 0')
    parser.add_argument('--enable_history', action='store_true',
                        help="if set, enables storing of history memory")
    parser.add_argument('--model_type', type=str, help="model type")
    parser.add_argument('--device', type=str, default='tpu', help="run bmodel when device=tpu, run torch when device=gpu",
                        choices=["tpu", "gpu"])
    parser.add_argument('--verbose', type=str, default=0, help="run bmodel when device=tpu, run torch when device=gpu",
                        choices=[0, 1])
    args = parser.parse_args()
    main(args)




    