# ==============================================================================
# Copyright (C) 2025 Sophgo Technologies Inc.  All rights reserved.
#
# TPU-MLIR is licensed under the 2-Clause BSD License except for the
# third-party components.
#
# ==============================================================================

import argparse
import numpy as np
import time
import math

import chat
from transformers import AutoTokenizer


class BgeReranker():

    def __init__(self, args):
        self.device = [int(d) for d in args.devid.split(",")]

        print("Load " + args.config_path + " ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.config_path, trust_remote_code=True
        )
        # Warm up tokenizer
        self.tokenizer.decode([0])

        self.model = chat.BgeReranker()
        load_start = time.time()
        self.model.init(self.device, args.model_path)
        load_end = time.time()
        print(f"Load Time: {(load_end - load_start):.3f} s")
        print(f"SEQLEN: {self.model.SEQLEN}, Hidden: {self.model.HIDDEN_SIZE}, "
              f"Layers: {self.model.NUM_LAYERS}")

    def __del__(self):
        self.model.deinit()

    def compute_score(self, pairs, normalize=False):
        """Compute relevance scores for query-document pairs.

        Args:
            pairs: list of [query, document] pairs
            normalize: if True, apply sigmoid to map scores to [0, 1]

        Returns:
            list of float scores
        """
        if isinstance(pairs[0], str):
            pairs = [pairs]

        scores = []
        for pair in pairs:
            query, document = pair[0], pair[1]

            # Tokenize the pair as a single concatenated sequence (right padding to SEQLEN)
            # [[query, doc]] produces: <s> query </s></s> document </s> <pad>...
            inputs = self.tokenizer(
                [[query, document]],
                padding="max_length",
                truncation=True,
                max_length=self.model.SEQLEN,
                return_tensors="np",
            )
            input_ids = inputs["input_ids"][0].astype(np.int32)
            attention_mask = inputs["attention_mask"][0].astype(np.int32)
            token_length = int(attention_mask.sum())

            # Compute position_ids for XLM-RoBERTa:
            # padding positions (input_id == pad_token_id) → position 0
            # non-padding positions → cumulative from pad_token_id + 1
            pad_id = self.tokenizer.pad_token_id
            position_ids = np.zeros(self.model.SEQLEN, dtype=np.int32)
            pos = pad_id + 1  # XLM-RoBERTa offset: positions start at pad_token_id + 1
            for i in range(self.model.SEQLEN):
                if input_ids[i] != pad_id:
                    position_ids[i] = pos
                    pos += 1
                else:
                    position_ids[i] = pad_id  # padding → position padding_idx (embedding is zero)

            fwd_start = time.time()
            logit = self.model.forward(
                input_ids.tolist(),
                position_ids.tolist(),
                attention_mask.tolist(),
            )
            fwd_end = time.time()

            score = float(logit)
            if normalize:
                score = 1.0 / (1.0 + math.exp(-score))

            scores.append(score)
            print(f"  [{token_length} tokens, score={score:.4f}, "
                  f"{(fwd_end - fwd_start) * 1000:.1f} ms]")

        return scores


def main(args):
    model = BgeReranker(args)

    if args.query and args.documents:
        pairs = [[args.query, doc] for doc in args.documents]
        print(f"\nReranking {len(pairs)} pairs:")
        print(f"  Query: {args.query}")
        scores = model.compute_score(pairs, normalize=args.normalize)

        # Sort by score descending
        ranked = sorted(zip(scores, args.documents), reverse=True)
        print(f"\nResults ({'normalized' if args.normalize else 'raw logits'}):")
        for score, doc in ranked:
            print(f"  [{score:.4f}] {doc[:80]}")
    else:
        # interactive mode
        print("\n=== bge-reranker-v2-m3 Interactive ===")
        print("Commands:")
        print("  /q or /exit    — quit")
        print("  /norm          — toggle score normalization (current: {})".format(
            "on" if args.normalize else "off"))
        print("  <query> ||| <document>  — compute relevance score")
        while True:
            text = input("\nInput (query ||| document): ").strip()
            if not text:
                continue
            if text in ("/q", "/exit", "/quit"):
                break
            if text.startswith("/norm"):
                args.normalize = not args.normalize
                print(f"Normalization: {'on' if args.normalize else 'off'}")
                continue
            if "|||" not in text:
                print("Format: <query> ||| <document>")
                continue
            parts = text.split("|||", 1)
            query, document = parts[0].strip(), parts[1].strip()
            if not query or not document:
                print("Both query and document must be non-empty")
                continue
            scores = model.compute_score(
                [[query, document]], normalize=args.normalize
            )
            print(f"Score: {scores[0]:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # yapf: disable
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='path to the bmodel file')
    parser.add_argument('-c', '--config_path', type=str, default="../config",
                        help='path to the tokenizer config directory')
    parser.add_argument('-d', '--devid', type=str, default='0',
                        help='device ID to use')
    parser.add_argument('--normalize', action='store_true',
                        help='apply sigmoid to map scores to [0, 1]')
    parser.add_argument('--query', type=str, default=None,
                        help='query text for reranking demo')
    parser.add_argument('--documents', type=str, nargs='+', default=None,
                        help='document texts for reranking demo')
    # yapf: enable
    args = parser.parse_args()
    main(args)
